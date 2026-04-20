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

"""Kimi K2.5 Parameter Server for BatchGen.

Handles weight loading for Kimi K2.5 (1.04T MoE, DeepSeek-V3 variant) with:
- INT4 W4A16 quantized routed expert weights (compressed-tensors format)
- BF16 attention weights (MLA, no FP8)
- 384 routed experts per MoE layer
- 1 shared expert per MoE layer (BF16, not quantized)

Checkpoint format (HuggingFace compressed-tensors):
    model.layers.{L}.self_attn.{name}                                → BF16 (MLA attention)
    model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.weight   → BF16 (shared expert)
    model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight_packed → uint8 (INT4 packed)
    model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight_scale  → bf16 (INT4 scale)

BatchGen format (state_dict_name_map):
    attn_{L}                  / {tensor_key}           → attention weights
    shared_expert_{L}         / {tensor_key}           → shared expert weights
    routed_expert_{L}_{E}     / {tensor_key}           → routed expert INT4 weights

Skeleton (NOT in state_dict_name_map, loaded directly by PSM):
    model.layers.{L}.input_layernorm.weight
    model.layers.{L}.post_attention_layernorm.weight
    model.layers.{L}.mlp.gate.weight  (router)
    model.embed_tokens.weight
    model.norm.weight
    lm_head.weight
"""

import gc
import logging
import os
import shutil
import uuid

import torch
from tqdm import trange

from batchgen.config.model_registry import load_config
from batchgen.ckpt_converter.ckpt_converter import ckpt_converter

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server


# MLA attention parameter names (same as DeepSeek-V3 with q_lora_rank)
_MLA_ATTN_TENSOR_NAMES = [
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
]

# Shared expert parameter names (BF16, no quantization)
_SHARED_EXPERT_TENSOR_NAMES = [
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
]

# Routed expert parameter names (INT4 compressed-tensors format)
# weight_packed: uint8 (2 INT4 values per byte)
# weight_scale: bf16 (one per group of 32 elements)
_INT4_EXPERT_TENSOR_NAMES = [
    "gate_proj.weight_packed",
    "gate_proj.weight_scale",
    "up_proj.weight_packed",
    "up_proj.weight_scale",
    "down_proj.weight_packed",
    "down_proj.weight_scale",
]


class KimiK25_Parameter_Server:
    """Parameter server for Kimi K2.5 with INT4 W4A16 weight handling.

    Handles:
    - Loading HuggingFace checkpoint with compressed-tensors INT4 packed weights
    - Mapping checkpoint names to BatchGen module_key + tensor_key format
    - Using ckpt_converter for standard safetensors → BatchGen binary conversion
    """

    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir, enable_hugetlbfs, enable_memfd=False):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        self.enable_memfd = enable_memfd

        # Use BatchGen's unified config system
        self.model_config = load_config(huggingface_ckpt_name)

        # K2.5 model parameters (from config)
        self.num_layers = self.model_config.num_hidden_layers        # 61
        self.num_experts = self.model_config.num_local_experts       # 384
        self.first_k_dense_replace = self.model_config.first_k_dense_replace  # 3

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

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs, self.enable_memfd)

        # Kimi K2.5: ~580GB INT4 experts + ~20GB BF16 (attn/shared/embed) ≈ 600GB
        # 650GB with buffer
        byte_size = 650 * 1024 * 1024 * 1024

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

        # Convert checkpoint files to BatchGen format using standard ckpt_converter
        # K2.5 uses HuggingFace safetensors format — no custom conversion needed
        if self.converted_ckpt_dir is None:
            converter = ckpt_converter()
            self.converted_ckpt_dir = converter.convert_model_directory(
                self.cache_dir, marlin=True, model_identifier=self.huggingface_ckpt_name)  # Marlin layout is default for K2.5 decode
        else:
            logging.info(f"Using pre-converted checkpoint: {self.converted_ckpt_dir}")

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            str(self.converted_ckpt_dir),
            self.state_dict_name_map,
        )
        return self.shm_name, self.tensor_meta_shm_name

    def _parse_state_dict(self):
        """Parse model structure to build name mapping.

        Maps checkpoint tensor names to BatchGen module_key + tensor_key format.

        NOTE: We do NOT instantiate the full model (1.04T params would exceed memory).
        Instead, we explicitly enumerate tensor names based on the known K2.5 architecture.

        Tensors NOT in state_dict_name_map (norms, router, embeddings) go to
        skeleton_state_dict_ automatically via the C++ Parameter_Server.
        """
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        for layer_idx in trange(self.num_layers, desc="Parsing state_dict"):
            # --- Attention weights (BF16 MLA) ---
            for name in _MLA_ATTN_TENSOR_NAMES:
                tensor_full_name = f"language_model.model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # MoE layers start after first_k_dense_replace dense layers
            if layer_idx >= self.first_k_dense_replace:
                # --- Shared expert weights (BF16, not quantized) ---
                for name in _SHARED_EXPERT_TENSOR_NAMES:
                    tensor_full_name = f"language_model.model.layers.{layer_idx}.mlp.shared_experts.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    f"shared_expert_{layer_idx}"
                )

                # --- Routed expert weights (INT4 compressed-tensors) ---
                for expert_idx in range(self.num_experts):
                    # INT4 raw weights (always present)
                    for name in _INT4_EXPERT_TENSOR_NAMES:
                        tensor_full_name = (
                            f"language_model.model.layers.{layer_idx}.mlp.experts."
                            f"{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": f"routed_expert_{layer_idx}_{expert_idx}",
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(
                        f"routed_expert_{layer_idx}_{expert_idx}"
                    )

        # Log summary
        attn_count = sum(1 for k in self.state_dict_name_map if "self_attn" in k)
        expert_count = sum(1 for k in self.state_dict_name_map if "mlp.experts" in k)
        shared_count = sum(1 for k in self.state_dict_name_map if "shared_experts" in k)
        logging.info(
            f"state_dict_name_map: {len(self.state_dict_name_map)} entries "
            f"(attn={attn_count}, routed_expert={expert_count}, shared_expert={shared_count})"
        )

        # Debug: Show first 10 keys to verify naming pattern
        sample_keys = list(self.state_dict_name_map.keys())[:10]
        logging.info(f"state_dict_name_map sample keys (first 10):")
        for key in sample_keys:
            logging.info(f"  {key}")

        gc.collect()
        torch.cuda.empty_cache()
