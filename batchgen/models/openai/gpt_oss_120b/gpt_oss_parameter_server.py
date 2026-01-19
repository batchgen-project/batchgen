# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""GPT-OSS-120B Parameter Server for BatchGen.

Handles MXFP4 weight loading with:
- Expert tensor slicing (128 experts per layer stored as single tensor)
- mlp1 splitting into gate_proj + up_proj (SwiGLU combined weights)
- Name mapping from checkpoint to BatchGen format

Checkpoint format (OpenAI):
    block.{L}.mlp.mlp1_weight.blocks: [128, 5760, K//2] packed uint8
    block.{L}.mlp.mlp1_weight.scales: [128, 5760, K//32] uint8
    block.{L}.mlp.mlp1_bias: [128, 5760] BF16
    block.{L}.mlp.mlp2_weight.blocks: [128, 2880, K//2] packed uint8
    block.{L}.mlp.mlp2_weight.scales: [128, 2880, K//32] uint8
    block.{L}.mlp.mlp2_bias: [128, 2880] BF16

BatchGen format (per expert):
    routed_expert_{L}_{E}/gate_proj.weight: [2880, K//2] packed
    routed_expert_{L}_{E}/gate_proj.weight_scales: [2880, K//32]
    routed_expert_{L}_{E}/up_proj.weight: [2880, K//2] packed
    routed_expert_{L}_{E}/up_proj.weight_scales: [2880, K//32]
    routed_expert_{L}_{E}/down_proj.weight: [2880, K//2] packed
    routed_expert_{L}_{E}/down_proj.weight_scales: [2880, K//32]
"""

import gc
import logging
import os
import shutil
import uuid

import torch
from safetensors import safe_open
from tqdm import tqdm, trange

from .model import GptOssForCausalLM
from .configuration_gpt_oss import GptOssConfig
from batchgen.config.model_registry import load_config

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server


class GptOss_Parameter_Server:
    """Parameter server for GPT-OSS-120B with MXFP4 weight handling.

    Handles:
    - Loading checkpoint with MXFP4 packed weights (uint8)
    - Slicing stacked expert tensors per layer
    - Splitting mlp1 into gate_proj + up_proj
    - Mapping checkpoint names to BatchGen format
    """

    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir, enable_hugetlbfs):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs

        # Use BatchGen's unified config system
        self.model_config = load_config(huggingface_ckpt_name)

        # Create HuggingFace-style config for model instantiation
        self.hf_config = GptOssConfig()
        self.hf_config._name_or_path = huggingface_ckpt_name

        # GPT-OSS model parameters
        self.num_layers = 36
        self.num_experts = 128
        self.intermediate_size = 2880
        self.hidden_size = 2880

        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"Python PM instantiation: GPU 0 free memory: {gpu0_memory:.2f} GB / {total_memory:.2f} GB")

    def Init(self):
        """Initialize parameter server and load weights."""
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"GPU 0 free mem pm start Init: {gpu0_memory:.2f} GB / {total_memory:.2f} GB")

        self._parse_state_dict()

        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"GPU 0 free mem before cpp pm instantiate: {gpu0_memory:.2f} GB / {total_memory:.2f} GB")

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs)

        # GPT-OSS-120B: ~65GB total (61GB MXFP4 experts + 4GB BF16 attn/embed)
        byte_size = 70 * 1024 * 1024 * 1024  # 70GB with buffer

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free/1024/1024/1024:.2f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size is not enough. Required: {byte_size}, Available: {free}. "
                "Please clear /dev/shm or increase the size by running "
                "'sudo mount -o remount,size=<size>G /dev/shm'"
            )

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(f"Tensor meta shared memory name: {self.tensor_meta_shm_name}")
        logging.info(f"Byte size: {byte_size}")

        # Convert checkpoint files to BatchGen format
        self._convert_checkpoint()

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            str(self.converted_ckpt_dir),  # Convert Path to string for C++ binding
            self.state_dict_name_map,
        )

        # Expose weights_storage for core engine
        self._weights_storage = self.parameter_server

        return self.shm_name, self.tensor_meta_shm_name

    @property
    def weights_storage(self):
        """Return the C++ parameter server as weights storage for core engine."""
        return self._weights_storage

    def _parse_state_dict(self):
        """Parse model state dict to build name mapping.

        Maps checkpoint tensor names to BatchGen module_key + tensor_key format.
        Handles:
        - Attention weights (BF16)
        - Expert weights (MXFP4 packed + scales)
        - Gate/router weights
        """
        # Create model to parse weight names
        self.hf_config._attn_implementation = "eager"
        model = GptOssForCausalLM._from_config(self.hf_config).to('cpu')
        model.eval()

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []

        for layer_idx in trange(self.num_layers, desc="Parsing state_dict"):
            # Attention weights - explicit list including sinks
            attn_tensor_names = [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "sinks",  # Attention sinks for GPT-OSS
            ]
            for name in attn_tensor_names:
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # Layer norms
            self.state_dict_name_map[f"model.layers.{layer_idx}.input_layernorm.weight"] = {
                "module_key": f"layer_{layer_idx}",
                "tensor_key": "input_layernorm.weight",
            }
            self.state_dict_name_map[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = {
                "module_key": f"layer_{layer_idx}",
                "tensor_key": "post_attention_layernorm.weight",
            }

            # Router/gate weights
            self.state_dict_name_map[f"model.layers.{layer_idx}.moe.router.weight"] = {
                "module_key": f"moe_router_{layer_idx}",
                "tensor_key": "weight",
            }

            # Expert weights - register per-expert
            # Include MXFP4 scale tensors which aren't in named_parameters()
            expert_tensor_names = [
                "gate_proj.weight",
                "gate_proj.weight_scales",  # MXFP4 scales
                "gate_proj.bias",
                "up_proj.weight",
                "up_proj.weight_scales",    # MXFP4 scales
                "up_proj.bias",
                "down_proj.weight",
                "down_proj.weight_scales",  # MXFP4 scales
                "down_proj.bias",
            ]
            for expert_idx in range(self.num_experts):
                for name in expert_tensor_names:
                    tensor_full_name = f"model.layers.{layer_idx}.moe.experts.{expert_idx}.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"routed_expert_{layer_idx}_{expert_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["routed_expert"].append(
                    f"routed_expert_{layer_idx}_{expert_idx}"
                )

        # Non-layer tensors (embeddings, final norm, lm_head)
        self.state_dict_name_map["model.embed_tokens.weight"] = {
            "module_key": "embed",
            "tensor_key": "weight",
        }
        self.state_dict_name_map["model.norm.weight"] = {
            "module_key": "final_norm",
            "tensor_key": "weight",
        }
        self.state_dict_name_map["lm_head.weight"] = {
            "module_key": "lm_head",
            "tensor_key": "weight",
        }

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def _convert_checkpoint(self):
        """Convert OpenAI checkpoint to BatchGen format with MXFP4 handling.

        Key operations:
        1. Load MXFP4 weights as uint8 (keep quantized)
        2. Slice stacked expert tensors [128, ...] -> 128 individual tensors
        3. Split mlp1 into gate_proj + up_proj
        4. Save with BatchGen naming convention
        """
        os.makedirs(self.converted_ckpt_dir, exist_ok=True)

        # Check if already converted
        converted_marker = os.path.join(self.converted_ckpt_dir, ".gpt_oss_converted")
        if os.path.exists(converted_marker):
            logging.info(f"Checkpoint already converted at {self.converted_ckpt_dir}")
            return

        logging.info(f"Converting checkpoint from {self.cache_dir} to {self.converted_ckpt_dir}")

        # Find all safetensor files
        safetensor_files = [
            os.path.join(self.cache_dir, f)
            for f in os.listdir(self.cache_dir)
            if f.endswith(".safetensors")
        ]

        if not safetensor_files:
            raise ValueError(f"No safetensor files found in {self.cache_dir}")

        # Build tensor name to file mapping
        tensor_to_file = {}
        for sf_file in safetensor_files:
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_to_file[key] = sf_file

        logging.info(f"Found {len(tensor_to_file)} tensors in checkpoint")

        # Process each layer
        for layer_idx in trange(self.num_layers, desc="Converting layers"):
            self._convert_layer(layer_idx, tensor_to_file)

        # Convert non-layer tensors (embeddings, final norm, lm_head)
        self._convert_non_layer_tensors(tensor_to_file)

        # Mark as converted
        with open(converted_marker, "w") as f:
            f.write("converted")

        logging.info("Checkpoint conversion complete")

    def _load_tensor(self, name: str, tensor_to_file: dict) -> torch.Tensor:
        """Load a single tensor from checkpoint."""
        if name not in tensor_to_file:
            raise KeyError(f"Tensor {name} not found in checkpoint")

        with safe_open(tensor_to_file[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def _convert_layer(self, layer_idx: int, tensor_to_file: dict):
        """Convert a single layer's weights.

        Handles:
        - Attention weights (copy as-is)
        - MLP expert weights (slice + split + rename)
        """
        output_file = os.path.join(self.converted_ckpt_dir, f"layer_{layer_idx}.bin")
        if os.path.exists(output_file):
            return

        layer_tensors = {}

        # Attention weights - copy directly
        attn_tensor_names = [
            f"block.{layer_idx}.attn.q_proj.weight",
            f"block.{layer_idx}.attn.k_proj.weight",
            f"block.{layer_idx}.attn.v_proj.weight",
            f"block.{layer_idx}.attn.o_proj.weight",
            f"block.{layer_idx}.attn.sinks",  # Attention sinks
        ]

        for name in attn_tensor_names:
            if name in tensor_to_file:
                tensor = self._load_tensor(name, tensor_to_file)
                # Map to BatchGen naming
                batchgen_name = name.replace(f"block.{layer_idx}.attn.", f"model.layers.{layer_idx}.self_attn.")
                layer_tensors[batchgen_name] = tensor

        # Layer norms
        norm_names = [
            (f"block.{layer_idx}.input_layernorm.weight", f"model.layers.{layer_idx}.input_layernorm.weight"),
            (f"block.{layer_idx}.post_attention_layernorm.weight", f"model.layers.{layer_idx}.post_attention_layernorm.weight"),
        ]
        for ckpt_name, bg_name in norm_names:
            if ckpt_name in tensor_to_file:
                layer_tensors[bg_name] = self._load_tensor(ckpt_name, tensor_to_file)

        # Router/gate weights
        gate_name = f"block.{layer_idx}.mlp.gate.weight"
        if gate_name in tensor_to_file:
            layer_tensors[f"model.layers.{layer_idx}.moe.router.weight"] = self._load_tensor(gate_name, tensor_to_file)

        # MLP weights - MXFP4 packed
        # mlp1 contains gate_proj + up_proj combined
        mlp1_blocks_name = f"block.{layer_idx}.mlp.mlp1_weight.blocks"
        mlp1_scales_name = f"block.{layer_idx}.mlp.mlp1_weight.scales"
        mlp1_bias_name = f"block.{layer_idx}.mlp.mlp1_bias"

        mlp2_blocks_name = f"block.{layer_idx}.mlp.mlp2_weight.blocks"
        mlp2_scales_name = f"block.{layer_idx}.mlp.mlp2_weight.scales"
        mlp2_bias_name = f"block.{layer_idx}.mlp.mlp2_bias"

        if mlp1_blocks_name in tensor_to_file:
            # Load stacked tensors [128, out_dim, in_dim//2]
            mlp1_blocks = self._load_tensor(mlp1_blocks_name, tensor_to_file)  # [128, 5760, K//2]
            mlp1_scales = self._load_tensor(mlp1_scales_name, tensor_to_file)  # [128, 5760, K//32]
            mlp1_bias = self._load_tensor(mlp1_bias_name, tensor_to_file)      # [128, 5760]

            mlp2_blocks = self._load_tensor(mlp2_blocks_name, tensor_to_file)  # [128, 2880, K//2]
            mlp2_scales = self._load_tensor(mlp2_scales_name, tensor_to_file)  # [128, 2880, K//32]
            mlp2_bias = self._load_tensor(mlp2_bias_name, tensor_to_file)      # [128, 2880]

            logging.debug(f"Layer {layer_idx} mlp1_blocks shape: {mlp1_blocks.shape}")
            logging.debug(f"Layer {layer_idx} mlp2_blocks shape: {mlp2_blocks.shape}")

            # Slice per expert and split mlp1
            for expert_idx in range(self.num_experts):
                # mlp1: split into gate_proj (first half) and up_proj (second half)
                expert_mlp1_blocks = mlp1_blocks[expert_idx]  # [5760, K//2]
                expert_mlp1_scales = mlp1_scales[expert_idx]  # [5760, K//32]
                expert_mlp1_bias = mlp1_bias[expert_idx]      # [5760]

                # Split at intermediate_size (2880)
                gate_blocks = expert_mlp1_blocks[:self.intermediate_size]  # [2880, K//2]
                gate_scales = expert_mlp1_scales[:self.intermediate_size]  # [2880, K//32]
                gate_bias = expert_mlp1_bias[:self.intermediate_size]      # [2880]

                up_blocks = expert_mlp1_blocks[self.intermediate_size:]    # [2880, K//2]
                up_scales = expert_mlp1_scales[self.intermediate_size:]    # [2880, K//32]
                up_bias = expert_mlp1_bias[self.intermediate_size:]        # [2880]

                # mlp2 -> down_proj
                down_blocks = mlp2_blocks[expert_idx]  # [2880, K//2]
                down_scales = mlp2_scales[expert_idx]  # [2880, K//32]
                down_bias = mlp2_bias[expert_idx]      # [2880]

                # Store with BatchGen naming
                prefix = f"model.layers.{layer_idx}.moe.experts.{expert_idx}"
                layer_tensors[f"{prefix}.gate_proj.weight"] = gate_blocks
                layer_tensors[f"{prefix}.gate_proj.weight_scales"] = gate_scales
                layer_tensors[f"{prefix}.gate_proj.bias"] = gate_bias

                layer_tensors[f"{prefix}.up_proj.weight"] = up_blocks
                layer_tensors[f"{prefix}.up_proj.weight_scales"] = up_scales
                layer_tensors[f"{prefix}.up_proj.bias"] = up_bias

                layer_tensors[f"{prefix}.down_proj.weight"] = down_blocks
                layer_tensors[f"{prefix}.down_proj.weight_scales"] = down_scales
                layer_tensors[f"{prefix}.down_proj.bias"] = down_bias

        # Save layer tensors
        self._save_tensors(layer_tensors, output_file)

        # Cleanup
        del layer_tensors
        gc.collect()

    def _convert_non_layer_tensors(self, tensor_to_file: dict):
        """Convert non-layer tensors (embeddings, final norm, lm_head)."""
        output_file = os.path.join(self.converted_ckpt_dir, "non_layer.bin")
        if os.path.exists(output_file):
            return

        tensors = {}

        # Embedding
        embed_name = "embed.weight"
        if embed_name in tensor_to_file:
            tensors["model.embed_tokens.weight"] = self._load_tensor(embed_name, tensor_to_file)

        # Final layer norm
        final_norm_name = "final_layernorm.weight"
        if final_norm_name in tensor_to_file:
            tensors["model.norm.weight"] = self._load_tensor(final_norm_name, tensor_to_file)

        # LM head
        lm_head_name = "lm_head.weight"
        if lm_head_name in tensor_to_file:
            tensors["lm_head.weight"] = self._load_tensor(lm_head_name, tensor_to_file)

        self._save_tensors(tensors, output_file)

    def _save_tensors(self, tensors: dict, output_file: str):
        """Save tensors in BatchGen binary format.

        Format: contiguous binary with JSON metadata.
        """
        import json

        metadata = {
            "file_name": os.path.basename(output_file),
            "state_dict": {},
            "total_byte_size": 0,
        }

        bin_file = output_file
        json_file = output_file.replace(".bin", ".json")

        with open(bin_file, "wb") as f:
            for name, tensor in tensors.items():
                tensor = tensor.contiguous()

                # Determine dtype string
                if tensor.dtype == torch.float32:
                    dtype_str = "float32"
                elif tensor.dtype == torch.float16:
                    dtype_str = "float16"
                elif tensor.dtype == torch.bfloat16:
                    dtype_str = "bfloat16"
                elif tensor.dtype == torch.uint8:
                    dtype_str = "uint8"
                elif tensor.dtype == torch.int64:
                    dtype_str = "int64"
                elif tensor.dtype == torch.int32:
                    dtype_str = "int32"
                else:
                    raise ValueError(f"Unsupported dtype: {tensor.dtype}")

                offset = metadata["total_byte_size"]
                byte_size = tensor.element_size() * tensor.numel()

                metadata["state_dict"][name] = {
                    "dtype": dtype_str,
                    "shape": list(tensor.shape),
                    "offset": offset,
                    "byte_size": byte_size,
                }
                metadata["total_byte_size"] += byte_size

                # Write tensor bytes
                # Note: bfloat16 is not supported by numpy, view as uint16 instead
                tensor = tensor.contiguous()
                if tensor.dtype == torch.bfloat16:
                    # View bf16 as uint16 for numpy compatibility (same byte layout)
                    tensor_bytes = tensor.view(torch.uint16).numpy().tobytes()
                else:
                    tensor_bytes = tensor.numpy().tobytes()
                f.write(tensor_bytes)

        with open(json_file, "w") as f:
            json.dump(metadata, f, indent=2)
