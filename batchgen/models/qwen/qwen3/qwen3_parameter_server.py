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

"""Qwen3 Parameter Server for BatchGen.

Handles BF16 weight loading from HuggingFace safetensors.
No quantization conversion needed — all weights are BF16.

HuggingFace Qwen3 weight naming:
    model.embed_tokens.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.self_attn.{q,k}_norm.weight    (QK-norm)
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.norm.weight
    lm_head.weight
"""

import gc
import logging
import os
import shutil
import uuid

import torch
from safetensors import safe_open
from tqdm import trange

from .model import Qwen3ForCausalLM
from .config import Qwen3Config
from batchgen.config.model_registry import load_config

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server


class Qwen3ParameterServer:
    """Parameter server for Qwen3 with BF16 weight handling.

    Handles:
    - Loading BF16 safetensors directly (no conversion needed)
    - Name mapping from HF format to BatchGen format
    - Skeleton vs module weight separation
    """

    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir,
                 enable_hugetlbfs, enable_memfd=False):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        self.enable_memfd = enable_memfd

        self.model_config = load_config(huggingface_ckpt_name)

        self.hf_config = Qwen3Config()
        self.hf_config._name_or_path = huggingface_ckpt_name

        # Qwen3 model parameters
        self.num_layers = self.model_config.num_hidden_layers
        self.hidden_size = self.model_config.hidden_size
        self.intermediate_size = self.model_config.intermediate_size
        self.num_attention_heads = self.model_config.num_attention_heads
        self.num_kv_heads = self.model_config.num_key_value_heads
        self.head_dim = self.model_config.head_dim

    def Init(self):
        """Initialize parameter server and load weights."""
        self._parse_state_dict()

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs, self.enable_memfd)

        # Qwen3-8B: ~16GB BF16 weights
        byte_size = 20 * 1024 * 1024 * 1024  # 20GB with buffer

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free/1024/1024/1024:.2f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size is not enough. Required: {byte_size}, Available: {free}. "
                "Please clear /dev/shm or increase the size."
            )

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(f"Byte size: {byte_size}")

        # Convert/prepare checkpoint
        self._convert_checkpoint()

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            str(self.converted_ckpt_dir),
            self.state_dict_name_map,
        )

        self._weights_storage = self.parameter_server
        return self.shm_name, self.tensor_meta_shm_name

    @property
    def weights_storage(self):
        return self._weights_storage

    def _parse_state_dict(self):
        """Parse model state dict to build name mapping.

        Attention weights go to module_weights_storage_ (dynamic loading via core engine).
        MLP weights, embeddings, layer norms, lm_head go to skeleton_state_dict_ (persistent).

        For dense 8B model on 96GB GPU, MLP weights are persistent in skeleton
        since the entire model fits in GPU memory.
        """
        self.weight_copy_task["attn"] = []

        for layer_idx in range(self.num_layers):
            # Attention weights (separate Q, K, V, O projections + QK-norm)
            attn_tensor_names = [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "q_norm.weight",
                "k_norm.weight",
            ]
            for name in attn_tensor_names:
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # MLP weights are NOT in state_dict_name_map — they go to skeleton
            # (persistent on GPU, no dynamic loading needed for 8B model)

        logging.info(f"State dict name map: {len(self.state_dict_name_map)} entries")
        logging.info(f"  Attention: {len(self.weight_copy_task['attn'])} layers")
        logging.info(f"  MLP: persistent in skeleton (not dynamically loaded)")

    def _convert_checkpoint(self):
        """Convert HF safetensors to BatchGen format.

        For Qwen3, this is straightforward — just copy BF16 tensors with
        BatchGen naming convention. No quantization conversion needed.
        """
        os.makedirs(self.converted_ckpt_dir, exist_ok=True)

        converted_marker = os.path.join(self.converted_ckpt_dir, ".qwen3_converted")
        if os.path.exists(converted_marker):
            logging.info(f"Checkpoint already converted at {self.converted_ckpt_dir}")
            return

        logging.info(f"Converting Qwen3 checkpoint from {self.cache_dir} to {self.converted_ckpt_dir}")

        # Find safetensor files
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

        # Convert non-layer tensors
        self._convert_non_layer_tensors(tensor_to_file)

        with open(converted_marker, "w") as f:
            f.write("converted")
        logging.info("Checkpoint conversion complete")

    def _load_tensor(self, name: str, tensor_to_file: dict) -> torch.Tensor:
        if name not in tensor_to_file:
            raise KeyError(f"Tensor {name} not found in checkpoint")
        with safe_open(tensor_to_file[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def _convert_layer(self, layer_idx: int, tensor_to_file: dict):
        """Convert a single layer's weights (BF16, no quantization)."""
        output_file = os.path.join(self.converted_ckpt_dir, f"layer_{layer_idx}.bin")
        if os.path.exists(output_file):
            return

        layer_tensors = {}

        # Qwen3 HF names map directly to BatchGen names
        tensor_names = {
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            f"model.layers.{layer_idx}.self_attn.q_norm.weight": f"model.layers.{layer_idx}.self_attn.q_norm.weight",
            f"model.layers.{layer_idx}.self_attn.k_norm.weight": f"model.layers.{layer_idx}.self_attn.k_norm.weight",
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"model.layers.{layer_idx}.mlp.up_proj.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
            f"model.layers.{layer_idx}.mlp.down_proj.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
            # Layer norms -> skeleton
            f"model.layers.{layer_idx}.input_layernorm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        }

        for hf_name, bg_name in tensor_names.items():
            if hf_name in tensor_to_file:
                layer_tensors[bg_name] = self._load_tensor(hf_name, tensor_to_file)
            else:
                logging.warning(f"Layer {layer_idx}: tensor {hf_name} not found in checkpoint")

        self._save_tensors(layer_tensors, output_file)
        del layer_tensors
        gc.collect()

    def _convert_non_layer_tensors(self, tensor_to_file: dict):
        """Convert non-layer tensors (embeddings, final norm, lm_head)."""
        output_file = os.path.join(self.converted_ckpt_dir, "non_layer.bin")
        if os.path.exists(output_file):
            return

        tensors = {}

        name_map = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        for hf_name, bg_name in name_map.items():
            if hf_name in tensor_to_file:
                tensors[bg_name] = self._load_tensor(hf_name, tensor_to_file)
                logging.info(f"Loaded {bg_name}: {tensors[bg_name].shape}")

        self._save_tensors(tensors, output_file)

    def _save_tensors(self, tensors: dict, output_file: str):
        """Save tensors in BatchGen binary format."""
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

                if tensor.dtype == torch.float32:
                    dtype_str = "float32"
                elif tensor.dtype == torch.float16:
                    dtype_str = "float16"
                elif tensor.dtype == torch.bfloat16:
                    dtype_str = "bfloat16"
                elif tensor.dtype == torch.uint8:
                    dtype_str = "uint8"
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

                tensor = tensor.contiguous()
                if tensor.dtype == torch.bfloat16:
                    tensor_bytes = tensor.view(torch.uint16).numpy().tobytes()
                else:
                    tensor_bytes = tensor.numpy().tobytes()
                f.write(tensor_bytes)

        with open(json_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Saved {len(metadata['state_dict'])} tensors, total {metadata['total_byte_size']/1024/1024:.2f} MB")
