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

from .model import GptOss
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
        self.num_attention_heads = 64
        self.num_key_value_heads = 8
        self.head_dim = 64
        # QKV split dimensions: Q=64*64=4096, K=8*64=512, V=8*64=512
        self.q_dim = self.num_attention_heads * self.head_dim  # 4096
        self.kv_dim = self.num_key_value_heads * self.head_dim  # 512

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

        # Debug: Log what skeleton_state_dict the C++ parameter server returns
        try:
            skeleton_dict = self.parameter_server.get_skeleton_state_dict()
            logging.info(f"C++ Parameter_Server.get_skeleton_state_dict() returned {len(skeleton_dict)} keys")
            if skeleton_dict:
                sample_keys = list(skeleton_dict.keys())[:20]
                logging.info(f"  Sample skeleton keys: {sample_keys}")
            else:
                logging.warning("  WARNING: skeleton_state_dict is EMPTY!")
                # Check if the converted_ckpt_dir is correct
                logging.info(f"  converted_ckpt_dir: {self.converted_ckpt_dir}")
                logging.info(f"  converted_ckpt_dir exists: {os.path.exists(self.converted_ckpt_dir)}")
                if os.path.exists(self.converted_ckpt_dir):
                    files = os.listdir(self.converted_ckpt_dir)
                    logging.info(f"  Files in converted_ckpt_dir: {files}")
        except Exception as e:
            logging.error(f"Failed to get skeleton_state_dict: {e}")

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
        model = GptOss(self.hf_config).to('cpu')
        model.eval()

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []

        for layer_idx in trange(self.num_layers, desc="Parsing state_dict"):
            # Attention weights - these go to module_weights_storage_ for dynamic loading
            # Note: We add explicit list to match the converted checkpoint tensor names
            attn_tensor_names = [
                "q_proj.weight",
                "q_proj.bias",
                "k_proj.weight",
                "k_proj.bias",
                "v_proj.weight",
                "v_proj.bias",
                "o_proj.weight",
                "o_proj.bias",
                "sinks",  # Attention sinks for GPT-OSS
            ]
            for name in attn_tensor_names:
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # NOTE: Layer norms, router weights, embeddings, final norm, and lm_head
            # are NOT added to state_dict_name_map - they will go to skeleton_state_dict_
            # and be loaded directly into the model by Parallel_Strategy_Manager

            # Expert weights - register per-expert for dynamic loading
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
                    tensor_full_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"routed_expert_{layer_idx}_{expert_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["routed_expert"].append(
                    f"routed_expert_{layer_idx}_{expert_idx}"
                )

        # NOTE: Non-layer tensors (embeddings, final norm, lm_head) are NOT added to
        # state_dict_name_map - they will go to skeleton_state_dict_ automatically
        # because the C++ Parameter_Server puts any tensor not in state_dict_name_map
        # into skeleton_state_dict_

        # Log comprehensive summary of state_dict_name_map
        logging.info("=" * 60)
        logging.info("STATE_DICT_NAME_MAP SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total entries in state_dict_name_map: {len(self.state_dict_name_map)}")

        # Count by module type
        attn_count = sum(1 for k in self.state_dict_name_map if "self_attn" in k)
        expert_count = sum(1 for k in self.state_dict_name_map if "mlp.experts" in k)
        logging.info(f"  Attention tensors: {attn_count}")
        logging.info(f"  Expert tensors: {expert_count}")

        # Sample entries
        sample_entries = list(self.state_dict_name_map.items())[:5]
        logging.info(f"Sample state_dict_name_map entries:")
        for tensor_name, mapping in sample_entries:
            logging.info(f"  '{tensor_name}' -> module_key='{mapping['module_key']}', tensor_key='{mapping['tensor_key']}'")

        # Log what model parameters should go to skeleton (NOT in state_dict_name_map)
        skeleton_params = []
        for name, _ in model.named_parameters():
            if name not in self.state_dict_name_map:
                skeleton_params.append(name)

        logging.info(f"\nModel parameters NOT in state_dict_name_map (should go to skeleton): {len(skeleton_params)}")
        for param_name in skeleton_params[:20]:  # First 20
            logging.info(f"  SKELETON: {param_name}")
        if len(skeleton_params) > 20:
            logging.info(f"  ... and {len(skeleton_params) - 20} more")

        # Categorize skeleton params
        embed_params = [p for p in skeleton_params if "embed" in p]
        norm_params = [p for p in skeleton_params if "norm" in p and "layernorm" not in p.lower()]
        layernorm_params = [p for p in skeleton_params if "layernorm" in p.lower()]
        router_params = [p for p in skeleton_params if "router" in p]
        lm_head_params = [p for p in skeleton_params if "lm_head" in p]

        logging.info(f"\nSkeleton param categories:")
        logging.info(f"  Embedding: {len(embed_params)} - {embed_params}")
        logging.info(f"  Final norm: {len(norm_params)} - {norm_params}")
        logging.info(f"  Layer norms: {len(layernorm_params)}")
        logging.info(f"  Routers: {len(router_params)}")
        logging.info(f"  LM head: {len(lm_head_params)} - {lm_head_params}")
        logging.info("=" * 60)

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
            # Log what converted files exist
            converted_files = [f for f in os.listdir(self.converted_ckpt_dir) if f.endswith(('.bin', '.json'))]
            logging.info(f"Converted files: {sorted(converted_files)}")

            # Log tensor names from JSON metadata files - helps debug skeleton loading
            import json
            skeleton_candidates = []  # Tensors that should go to skeleton (not in state_dict_name_map)
            for json_file in sorted([f for f in converted_files if f.endswith('.json')]):
                json_path = os.path.join(self.converted_ckpt_dir, json_file)
                try:
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                        tensor_names = list(meta.get('state_dict', {}).keys())
                        logging.info(f"  {json_file}: {len(tensor_names)} tensors")
                        if tensor_names:
                            logging.info(f"    Sample tensors: {tensor_names[:5]}")
                        # Check which tensors are NOT in state_dict_name_map (these go to skeleton)
                        for tname in tensor_names:
                            if tname not in self.state_dict_name_map:
                                skeleton_candidates.append(tname)
                except Exception as e:
                    logging.warning(f"  {json_file}: Failed to read - {e}")

            # Log expected skeleton tensors
            if skeleton_candidates:
                logging.info(f"Tensors NOT in state_dict_name_map (should go to skeleton): {len(skeleton_candidates)}")
                logging.info(f"  Sample skeleton candidates: {skeleton_candidates[:10]}")
            else:
                logging.warning("WARNING: No skeleton candidates found! This may cause 'Missing skeleton weight' errors.")
                logging.warning("  Try deleting the converted checkpoint directory to force re-conversion:")
                logging.warning(f"    rm -rf {self.converted_ckpt_dir}")
            return

        logging.info(f"Converting checkpoint from {self.cache_dir} to {self.converted_ckpt_dir}")

        # Log the name mapping convention
        logging.info(f"\n{'='*60}")
        logging.info("CHECKPOINT -> BATCHGEN NAME MAPPING CONVENTION")
        logging.info(f"{'='*60}")
        logging.info("Attention (per layer):")
        logging.info("  block.{L}.attn.qkv.weight -> model.layers.{L}.self_attn.{q,k,v}_proj.weight (SPLIT)")
        logging.info("  block.{L}.attn.qkv.bias   -> model.layers.{L}.self_attn.{q,k,v}_proj.bias (SPLIT)")
        logging.info("  block.{L}.attn.out.weight -> model.layers.{L}.self_attn.o_proj.weight")
        logging.info("  block.{L}.attn.out.bias   -> model.layers.{L}.self_attn.o_proj.bias")
        logging.info("  block.{L}.attn.sinks      -> model.layers.{L}.self_attn.sinks")
        logging.info("")
        logging.info("Layer norms (SKELETON - NOT in state_dict_name_map):")
        logging.info("  block.{L}.attn.norm.weight -> model.layers.{L}.input_layernorm.weight")
        logging.info("  block.{L}.mlp.norm.weight  -> model.layers.{L}.post_attention_layernorm.weight")
        logging.info("")
        logging.info("Router (SKELETON - NOT in state_dict_name_map):")
        logging.info("  block.{L}.mlp.gate.weight -> model.layers.{L}.mlp.router.weight")
        logging.info("  block.{L}.mlp.gate.bias   -> model.layers.{L}.mlp.router.bias")
        logging.info("")
        logging.info("Expert MLP (MXFP4, per expert E=0..127):")
        logging.info("  block.{L}.mlp.mlp1_weight.blocks[E,:2880,:] -> model.layers.{L}.mlp.experts.{E}.gate_proj.weight")
        logging.info("  block.{L}.mlp.mlp1_weight.blocks[E,2880:,:] -> model.layers.{L}.mlp.experts.{E}.up_proj.weight")
        logging.info("  block.{L}.mlp.mlp2_weight.blocks[E]         -> model.layers.{L}.mlp.experts.{E}.down_proj.weight")
        logging.info("  (same pattern for scales and biases)")
        logging.info("")
        logging.info("Non-layer tensors (SKELETON - NOT in state_dict_name_map):")
        logging.info("  embedding.weight   -> model.embed_tokens.weight")
        logging.info("  norm.weight        -> model.norm.weight")
        logging.info("  unembedding.weight -> lm_head.weight")
        logging.info(f"{'='*60}\n")

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

        # Debug: Print first few tensor names from each category to verify naming convention
        sample_tensors = list(tensor_to_file.keys())[:30]
        logging.info(f"Sample checkpoint tensor names (first 30): {sample_tensors}")

        # Print layer 0 tensor names for verification
        layer0_tensors = [k for k in tensor_to_file.keys() if "block.0." in k][:20]
        logging.info(f"Layer 0 checkpoint tensors (for verification): {layer0_tensors}")

        # Print non-block tensors (embedding, norm, unembedding)
        non_block_tensors = [k for k in tensor_to_file.keys() if not k.startswith("block.")]
        logging.info(f"Non-block checkpoint tensors: {non_block_tensors}")

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
        - Attention weights (split fused QKV)
        - MLP expert weights (slice + split + rename)

        OpenAI checkpoint format:
        - block.{n}.attn.qkv.weight/bias - FUSED Q/K/V, need to split
        - block.{n}.attn.out.weight/bias - Output projection
        - block.{n}.attn.sinks - Attention sinks
        - block.{n}.attn.norm.scale - Pre-attention layer norm (input_layernorm) NOTE: .scale not .weight
        - block.{n}.mlp.norm.scale - Post-attention layer norm (post_attention_layernorm) NOTE: .scale not .weight
        - block.{n}.mlp.gate.weight/bias - Router
        """
        output_file = os.path.join(self.converted_ckpt_dir, f"layer_{layer_idx}.bin")
        if os.path.exists(output_file):
            logging.debug(f"Layer {layer_idx} already converted, skipping")
            return

        layer_tensors = {}
        skeleton_tensors = []  # Track which tensors should go to skeleton
        module_tensors = []    # Track which tensors should go to module_weights

        # Handle FUSED QKV - split into separate Q, K, V projections
        # QKV shape: [q_dim + 2*kv_dim, hidden_size] = [5120, 2880]
        qkv_weight_name = f"block.{layer_idx}.attn.qkv.weight"
        qkv_bias_name = f"block.{layer_idx}.attn.qkv.bias"

        if qkv_weight_name in tensor_to_file:
            qkv_weight = self._load_tensor(qkv_weight_name, tensor_to_file)  # [5120, 2880]
            # Split: Q=[4096, 2880], K=[512, 2880], V=[512, 2880]
            q_weight = qkv_weight[:self.q_dim]  # [4096, 2880]
            k_weight = qkv_weight[self.q_dim:self.q_dim + self.kv_dim]  # [512, 2880]
            v_weight = qkv_weight[self.q_dim + self.kv_dim:]  # [512, 2880]

            layer_tensors[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_weight
            layer_tensors[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_weight
            layer_tensors[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_weight
            module_tensors.extend([
                f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            ])
            logging.debug(f"Layer {layer_idx} QKV weight split: Q={q_weight.shape}, K={k_weight.shape}, V={v_weight.shape}")

        if qkv_bias_name in tensor_to_file:
            qkv_bias = self._load_tensor(qkv_bias_name, tensor_to_file)  # [5120]
            q_bias = qkv_bias[:self.q_dim]
            k_bias = qkv_bias[self.q_dim:self.q_dim + self.kv_dim]
            v_bias = qkv_bias[self.q_dim + self.kv_dim:]

            layer_tensors[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = q_bias
            layer_tensors[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = k_bias
            layer_tensors[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = v_bias
            module_tensors.extend([
                f"model.layers.{layer_idx}.self_attn.q_proj.bias",
                f"model.layers.{layer_idx}.self_attn.k_proj.bias",
                f"model.layers.{layer_idx}.self_attn.v_proj.bias",
            ])

        # Output projection
        out_weight_name = f"block.{layer_idx}.attn.out.weight"
        out_bias_name = f"block.{layer_idx}.attn.out.bias"

        if out_weight_name in tensor_to_file:
            layer_tensors[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = self._load_tensor(out_weight_name, tensor_to_file)
            module_tensors.append(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
        if out_bias_name in tensor_to_file:
            layer_tensors[f"model.layers.{layer_idx}.self_attn.o_proj.bias"] = self._load_tensor(out_bias_name, tensor_to_file)
            module_tensors.append(f"model.layers.{layer_idx}.self_attn.o_proj.bias")

        # Attention sinks
        sinks_name = f"block.{layer_idx}.attn.sinks"
        if sinks_name in tensor_to_file:
            layer_tensors[f"model.layers.{layer_idx}.self_attn.sinks"] = self._load_tensor(sinks_name, tensor_to_file)
            module_tensors.append(f"model.layers.{layer_idx}.self_attn.sinks")

        # Layer norms - OpenAI uses .scale instead of .weight
        # block.{n}.attn.norm.scale -> input_layernorm (pre-attention)
        # block.{n}.mlp.norm.scale -> post_attention_layernorm
        attn_norm_name = f"block.{layer_idx}.attn.norm.scale"
        mlp_norm_name = f"block.{layer_idx}.mlp.norm.scale"

        if attn_norm_name in tensor_to_file:
            layer_tensors[f"model.layers.{layer_idx}.input_layernorm.weight"] = self._load_tensor(attn_norm_name, tensor_to_file)
            skeleton_tensors.append(f"model.layers.{layer_idx}.input_layernorm.weight")
        if mlp_norm_name in tensor_to_file:
            layer_tensors[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = self._load_tensor(mlp_norm_name, tensor_to_file)
            skeleton_tensors.append(f"model.layers.{layer_idx}.post_attention_layernorm.weight")

        # Router/gate weights (including bias)
        gate_weight_name = f"block.{layer_idx}.mlp.gate.weight"
        gate_bias_name = f"block.{layer_idx}.mlp.gate.bias"

        if gate_weight_name in tensor_to_file:
            # Note: OpenAI gate weight is [hidden_size, num_experts], may need transpose
            gate_weight = self._load_tensor(gate_weight_name, tensor_to_file)
            # Check if transpose needed (BatchGen expects [num_experts, hidden_size])
            if gate_weight.shape[0] == self.hidden_size and gate_weight.shape[1] == self.num_experts:
                gate_weight = gate_weight.T.contiguous()
            layer_tensors[f"model.layers.{layer_idx}.mlp.router.weight"] = gate_weight
            skeleton_tensors.append(f"model.layers.{layer_idx}.mlp.router.weight")
        if gate_bias_name in tensor_to_file:
            layer_tensors[f"model.layers.{layer_idx}.mlp.router.bias"] = self._load_tensor(gate_bias_name, tensor_to_file)
            skeleton_tensors.append(f"model.layers.{layer_idx}.mlp.router.bias")

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
                prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                layer_tensors[f"{prefix}.gate_proj.weight"] = gate_blocks
                layer_tensors[f"{prefix}.gate_proj.weight_scales"] = gate_scales
                layer_tensors[f"{prefix}.gate_proj.bias"] = gate_bias

                layer_tensors[f"{prefix}.up_proj.weight"] = up_blocks
                layer_tensors[f"{prefix}.up_proj.weight_scales"] = up_scales
                layer_tensors[f"{prefix}.up_proj.bias"] = up_bias

                layer_tensors[f"{prefix}.down_proj.weight"] = down_blocks
                layer_tensors[f"{prefix}.down_proj.weight_scales"] = down_scales
                layer_tensors[f"{prefix}.down_proj.bias"] = down_bias

        # Log layer conversion summary
        if layer_idx == 0:  # Only log details for first layer to avoid spam
            logging.info(f"\n{'='*60}")
            logging.info(f"LAYER {layer_idx} CONVERSION SUMMARY (detailed for layer 0 only)")
            logging.info(f"{'='*60}")
            logging.info(f"Total tensors in layer: {len(layer_tensors)}")
            logging.info(f"\nSkeleton tensors (NOT in state_dict_name_map, will go to skeleton_state_dict_):")
            for t in skeleton_tensors:
                in_map = t in self.state_dict_name_map
                logging.info(f"  {t} -> in_state_dict_name_map={in_map}")
            logging.info(f"\nModule tensors (IN state_dict_name_map, will go to module_weights_storage_):")
            for t in module_tensors[:10]:  # First 10 only
                if t in self.state_dict_name_map:
                    mapping = self.state_dict_name_map[t]
                    logging.info(f"  {t} -> module_key='{mapping['module_key']}', tensor_key='{mapping['tensor_key']}'")
            if len(module_tensors) > 10:
                logging.info(f"  ... and {len(module_tensors) - 10} more module tensors")
            logging.info(f"{'='*60}\n")

        # Save layer tensors
        self._save_tensors(layer_tensors, output_file)

        # Cleanup
        del layer_tensors
        gc.collect()

    def _convert_non_layer_tensors(self, tensor_to_file: dict):
        """Convert non-layer tensors (embeddings, final norm, lm_head).

        OpenAI checkpoint format:
        - embedding.weight -> model.embed_tokens.weight
        - norm.scale -> model.norm.weight (final layer norm, note: .scale not .weight)
        - unembedding.weight -> lm_head.weight
        """
        output_file = os.path.join(self.converted_ckpt_dir, "non_layer.bin")
        if os.path.exists(output_file):
            return

        tensors = {}

        # Embedding - OpenAI uses "embedding.weight"
        embed_name = "embedding.weight"
        if embed_name in tensor_to_file:
            tensors["model.embed_tokens.weight"] = self._load_tensor(embed_name, tensor_to_file)
            logging.info(f"Loaded embedding: {tensors['model.embed_tokens.weight'].shape}")

        # Final layer norm - OpenAI uses "norm.scale" (not .weight)
        final_norm_name = "norm.scale"
        if final_norm_name in tensor_to_file:
            tensors["model.norm.weight"] = self._load_tensor(final_norm_name, tensor_to_file)
            logging.info(f"Loaded final norm: {tensors['model.norm.weight'].shape}")

        # LM head - OpenAI uses "unembedding.weight"
        lm_head_name = "unembedding.weight"
        if lm_head_name in tensor_to_file:
            tensors["lm_head.weight"] = self._load_tensor(lm_head_name, tensor_to_file)
            logging.info(f"Loaded lm_head: {tensors['lm_head.weight'].shape}")

        if not tensors:
            logging.warning("No non-layer tensors found! Check checkpoint tensor names.")
            # Debug: print available tensor names
            non_block_tensors = [k for k in tensor_to_file.keys() if not k.startswith("block.")]
            logging.warning(f"Available non-block tensors: {non_block_tensors}")
        else:
            # Log non-layer tensor mapping
            logging.info(f"\n{'='*60}")
            logging.info("NON-LAYER TENSORS CONVERSION")
            logging.info(f"{'='*60}")
            for batchgen_name in tensors.keys():
                in_map = batchgen_name in self.state_dict_name_map
                logging.info(f"  {batchgen_name}")
                logging.info(f"    -> in_state_dict_name_map: {in_map}")
                logging.info(f"    -> Will go to: {'module_weights_storage_' if in_map else 'skeleton_state_dict_'}")
            logging.info(f"{'='*60}\n")

        self._save_tensors(tensors, output_file)

    def _save_tensors(self, tensors: dict, output_file: str):
        """Save tensors in BatchGen binary format.

        Format: contiguous binary with JSON metadata.
        """
        import json

        logging.info(f"Saving {len(tensors)} tensors to {output_file}")
        if tensors:
            sample_keys = list(tensors.keys())[:10]
            logging.debug(f"  Sample tensor names: {sample_keys}")

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

        logging.info(f"  Saved {len(metadata['state_dict'])} tensors, total {metadata['total_byte_size']/1024/1024:.2f} MB")
