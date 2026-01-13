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

"""Parameter server for GPT-OSS-120B model.

Handles model weight loading and shared memory allocation for GPT-OSS.
Model specs:
- 36 layers, 128 experts, 117B total params (5.1B active)
- MXFP4 quantized expert weights stored as uint8 packed + uint8 scales

Key difference from DeepSeek:
- GPT-OSS stores all experts in ONE stacked tensor per layer: [num_experts, ...]
- We slice these into individual expert tensors during checkpoint conversion
- This allows reusing DeepSeek-style expert handling

OpenAI checkpoint tensor naming:
- block.{N}.mlp.mlp1_weight.blocks: [128, intermediate*2, hidden//2] - MXFP4 packed
- block.{N}.mlp.mlp1_weight.scales: [128, intermediate*2, hidden//32] - scales
- block.{N}.mlp.mlp2_weight.blocks: [128, hidden, intermediate//2] - MXFP4 packed
- block.{N}.mlp.mlp2_weight.scales: [128, hidden, intermediate//32] - scales

After slicing:
- block.{N}.mlp.experts.{E}.mlp1.packed: [intermediate*2, hidden//2]
- block.{N}.mlp.experts.{E}.mlp1.scales: [intermediate*2, hidden//32]
- etc.
"""

import gc
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm, trange

from batchgen.ckpt_converter.ckpt_converter import ckpt_converter

# Path to local GPT-OSS config directory (same as this file's directory)
_GPT_OSS_CONFIG_DIR = Path(__file__).parent

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server


@dataclass
class ModelConfig:
    """GPT-OSS model configuration (OpenAI style)."""
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class GptOss_Parameter_Server:
    """Parameter server for GPT-OSS-120B model weights.

    This class handles:
    1. Loading GPT-OSS checkpoint (stacked expert format)
    2. Slicing stacked experts into individual expert tensors
    3. Converting to BatchGen format with proper state_dict_name_map
    4. Allocating shared memory for efficient multi-GPU inference

    MXFP4 weights are stored as-is (blocks + scales) for W4A16 GEMM.
    """

    def __init__(
        self,
        huggingface_ckpt_name: str,
        cache_dir: str,
        pt_ckpt_dir: str,
        enable_hugetlbfs: bool,
    ):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.pt_ckpt_dir = pt_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs

        # Load model configuration from local config directory (OpenAI style)
        config_path = _GPT_OSS_CONFIG_DIR / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"GPT-OSS config not found at {config_path}. "
                f"Expected config directory: {_GPT_OSS_CONFIG_DIR}"
            )

        logging.info(f"Loading GPT-OSS config from: {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        self.model_config = ModelConfig(**config_dict)

        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(
            f"GPT-OSS Parameter Server: GPU 0 free memory: {gpu0_memory:.2f} GB / {total_memory:.2f} GB"
        )

    def Init(self):
        """Initialize the parameter server and allocate shared memory."""
        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory = total_memory / 1024 / 1024 / 1024
        logging.info(f"GPU 0 free mem at Init start: {gpu0_memory:.2f} GB / {total_memory:.2f} GB")

        # Step 1: Convert and slice checkpoint if needed
        self.converted_ckpt_dir = self._convert_and_slice_checkpoint()

        # Step 2: Build state_dict_name_map for sliced tensors
        self._parse_state_dict()

        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        logging.info(f"GPU 0 free mem before C++ PM init: {gpu0_memory:.2f} GB / {total_memory:.2f} GB")

        # Step 3: Initialize C++ parameter server
        self.parameter_server = Parameter_Server(self.enable_hugetlbfs)

        # GPT-OSS-120B memory estimate:
        # - MXFP4 experts: ~55 GB (128 experts × 36 layers × ~12MB each)
        # - BF16 attention: ~3 GB
        # - Embeddings: ~1 GB
        # Total: ~60 GB
        byte_size = 65 * 1024 * 1024 * 1024  # 65 GB with buffer

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free / 1024 / 1024 / 1024:.2f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory insufficient. Required: {byte_size / 1024**3:.2f} GB, "
                f"Available: {free / 1024**3:.2f} GB. "
                f"Run: sudo mount -o remount,size=<size>G /dev/shm"
            )

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters SHM: {self.shm_name}")
        logging.info(f"Tensor meta SHM: {self.tensor_meta_shm_name}")
        logging.info(f"Byte size: {byte_size / 1024**3:.2f} GB")

        # Step 4: Initialize parameter server with sliced checkpoint
        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            self.converted_ckpt_dir,
            self.state_dict_name_map,
        )
        return self.shm_name, self.tensor_meta_shm_name

    def _convert_and_slice_checkpoint(self) -> str:
        """Convert GPT-OSS checkpoint: slice stacked experts into individual tensors.

        GPT-OSS stores all 128 experts in one tensor per layer.
        We slice them into individual expert tensors for BatchGen compatibility.

        Returns:
            Path to converted checkpoint directory
        """
        output_dir = os.path.join(self.cache_dir, "batchgen_converted")
        marker_file = os.path.join(output_dir, ".conversion_complete")

        # Check if already converted
        # Note: ckpt_converter creates a 'converted_ckpt' subdirectory
        converted_ckpt_dir = os.path.join(output_dir, "converted_ckpt")
        if os.path.exists(marker_file):
            logging.info(f"Using existing converted checkpoint at {converted_ckpt_dir}")
            return converted_ckpt_dir

        logging.info(f"Converting GPT-OSS checkpoint with expert slicing...")
        os.makedirs(output_dir, exist_ok=True)

        # Find all checkpoint files
        ckpt_files = [
            f for f in os.listdir(self.cache_dir)
            if f.endswith(".safetensors")
        ]

        if not ckpt_files:
            raise FileNotFoundError(f"No .safetensors files in {self.cache_dir}")

        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_experts

        # Process each checkpoint file
        dtype_logged = set()  # Track which dtypes we've logged
        for ckpt_file in tqdm(ckpt_files, desc="Processing checkpoint files"):
            ckpt_path = os.path.join(self.cache_dir, ckpt_file)
            tensors = load_file(ckpt_path)

            sliced_tensors = {}

            for name, tensor in tensors.items():
                # Log dtype info once per tensor type
                tensor_type = None
                if "mlp1_weight.blocks" in name:
                    tensor_type = "mlp1_weight.blocks"
                elif "mlp1_weight.scales" in name:
                    tensor_type = "mlp1_weight.scales"
                elif "mlp1_bias" in name:
                    tensor_type = "mlp1_bias"
                elif "mlp2_weight.blocks" in name:
                    tensor_type = "mlp2_weight.blocks"
                elif "mlp2_weight.scales" in name:
                    tensor_type = "mlp2_weight.scales"
                elif "mlp2_bias" in name:
                    tensor_type = "mlp2_bias"

                if tensor_type and tensor_type not in dtype_logged:
                    logging.info(f"Original checkpoint tensor dtype: {tensor_type} -> {tensor.dtype}, shape={list(tensor.shape)}")
                    dtype_logged.add(tensor_type)

                # Check if this is a stacked expert tensor
                if self._is_stacked_expert_tensor(name):
                    # Slice and rename
                    layer_idx = self._extract_layer_idx(name)
                    sliced = self._slice_expert_tensor(name, tensor, layer_idx, num_experts)
                    sliced_tensors.update(sliced)
                else:
                    # Keep as-is (attention, embeddings, norms, router)
                    sliced_tensors[name] = tensor

            # Save sliced checkpoint
            output_path = os.path.join(output_dir, ckpt_file)
            save_file(sliced_tensors, output_path)
            logging.info(f"Saved sliced checkpoint: {output_path} ({len(sliced_tensors)} tensors)")

        # Run standard BatchGen conversion on sliced checkpoint
        converter = ckpt_converter()
        final_dir = converter.convert_model_directory(output_dir)

        # Mark conversion complete
        with open(marker_file, "w") as f:
            f.write("complete")

        logging.info(f"Checkpoint conversion complete: {final_dir}")
        return final_dir

    def _is_stacked_expert_tensor(self, name: str) -> bool:
        """Check if tensor is a stacked expert tensor (mlp1/mlp2 weights)."""
        stacked_patterns = [
            ".mlp.mlp1_weight.blocks",
            ".mlp.mlp1_weight.scales",
            ".mlp.mlp1_bias",
            ".mlp.mlp2_weight.blocks",
            ".mlp.mlp2_weight.scales",
            ".mlp.mlp2_bias",
        ]
        return any(p in name for p in stacked_patterns)

    def _extract_layer_idx(self, name: str) -> int:
        """Extract layer index from tensor name like 'block.5.mlp...'"""
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "block" and i + 1 < len(parts):
                return int(parts[i + 1])
        raise ValueError(f"Cannot extract layer index from {name}")

    def _slice_expert_tensor(
        self, name: str, tensor: torch.Tensor, layer_idx: int, num_experts: int
    ) -> Dict[str, torch.Tensor]:
        """Slice a stacked expert tensor into individual expert tensors.

        GPT-OSS MXFP4 checkpoint format (actual shapes):
        - mlp1_weight.blocks: [128, 5760, 90, 16] - 4D tensor, need to reshape to 2D
        - mlp1_weight.scales: [128, 5760, 90] - 3D tensor, already 2D after slicing
        - mlp1_bias: [128, 5760] - BF16, already 1D after slicing
        - mlp2_weight.blocks: [128, 2880, 90, 16]
        - mlp2_weight.scales: [128, 2880, 90]
        - mlp2_bias: [128, 2880]

        The packed format uses groups of 32 values:
        - blocks shape: [num_experts, out_dim, K//32, 16] where 16 bytes = 32 FP4 values
        - We reshape [out_dim, K//32, 16] -> [out_dim, K//2] for 2D packed tensor
        """
        sliced = {}

        # Determine tensor type and whether to reshape
        reshape_packed = False
        if "mlp1_weight.blocks" in name:
            tensor_suffix = "mlp1.packed"
            reshape_packed = True
        elif "mlp1_weight.scales" in name:
            tensor_suffix = "mlp1.scales"
        elif "mlp1_bias" in name:
            tensor_suffix = "mlp1.bias"
        elif "mlp2_weight.blocks" in name:
            tensor_suffix = "mlp2.packed"
            reshape_packed = True
        elif "mlp2_weight.scales" in name:
            tensor_suffix = "mlp2.scales"
        elif "mlp2_bias" in name:
            tensor_suffix = "mlp2.bias"
        else:
            raise ValueError(f"Unknown stacked tensor type: {name}")

        # Slice along first dimension (num_experts)
        assert tensor.shape[0] == num_experts, \
            f"Expected {num_experts} experts, got {tensor.shape[0]} in {name}"

        for expert_idx in range(num_experts):
            expert_tensor = tensor[expert_idx]

            # Reshape 3D packed tensor to 2D: [out_dim, K//32, 16] -> [out_dim, K//2]
            if reshape_packed and expert_tensor.dim() == 3:
                out_dim = expert_tensor.shape[0]
                # Flatten last two dimensions: [K//32, 16] -> [K//2]
                expert_tensor = expert_tensor.reshape(out_dim, -1)

            new_name = f"block.{layer_idx}.mlp.experts.{expert_idx}.{tensor_suffix}"
            sliced[new_name] = expert_tensor.contiguous()

        return sliced

    def _parse_state_dict(self):
        """Build state_dict_name_map for sliced expert tensors.

        Maps checkpoint tensor names to (module_key, tensor_key) pairs.
        Uses DeepSeek-style naming: routed_expert_{layer}_{expert}

        NOTE: GPT-OSS attention weights are NOT in state_dict_name_map.
        They are loaded as skeleton weights (once at init) because:
        1. BF16 attention weights are not quantized (no special handling needed)
        2. They don't change between requests
        3. No custom attention wrapper exists for GPT-OSS (uses vanilla model.py attention)

        Only expert MLP weights (MXFP4) are dynamically loaded via Expert_Wrapper.
        """
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []

        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_experts

        for layer_idx in trange(num_layers, desc="Building state_dict_name_map"):
            # =================================================================
            # Attention weights (BF16) - SKELETON LOADING
            # =================================================================
            # NOTE: Attention weights are NOT added to state_dict_name_map.
            # They will be loaded as skeleton weights via get_skeleton_state_dict()
            # and then loaded into the model by Parallel_Strategy_Manager._load_model_skeleton()
            #
            # Weights loaded as skeleton:
            # - block.{N}.attn.qkv.weight, block.{N}.attn.qkv.bias
            # - block.{N}.attn.out.weight, block.{N}.attn.out.bias
            # - block.{N}.attn.norm.scale, block.{N}.attn.sinks

            # =================================================================
            # Expert MLP weights (MXFP4) - DYNAMIC LOADING
            # Each expert is now a separate module (after slicing)
            # =================================================================
            for expert_idx in range(num_experts):
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"

                # mlp1 (gate+up projection) - MXFP4
                self.state_dict_name_map[f"block.{layer_idx}.mlp.experts.{expert_idx}.mlp1.packed"] = {
                    "module_key": module_key,
                    "tensor_key": "mlp1.packed",
                }
                self.state_dict_name_map[f"block.{layer_idx}.mlp.experts.{expert_idx}.mlp1.scales"] = {
                    "module_key": module_key,
                    "tensor_key": "mlp1.scales",
                }
                self.state_dict_name_map[f"block.{layer_idx}.mlp.experts.{expert_idx}.mlp1.bias"] = {
                    "module_key": module_key,
                    "tensor_key": "mlp1.bias",
                }

                # mlp2 (down projection) - MXFP4
                self.state_dict_name_map[f"block.{layer_idx}.mlp.experts.{expert_idx}.mlp2.packed"] = {
                    "module_key": module_key,
                    "tensor_key": "mlp2.packed",
                }
                self.state_dict_name_map[f"block.{layer_idx}.mlp.experts.{expert_idx}.mlp2.scales"] = {
                    "module_key": module_key,
                    "tensor_key": "mlp2.scales",
                }
                self.state_dict_name_map[f"block.{layer_idx}.mlp.experts.{expert_idx}.mlp2.bias"] = {
                    "module_key": module_key,
                    "tensor_key": "mlp2.bias",
                }

                self.weight_copy_task["routed_expert"].append(module_key)

        # Log summary
        num_attn = len(self.weight_copy_task["attn"])
        num_expert = len(self.weight_copy_task["routed_expert"])
        num_tensors = len(self.state_dict_name_map)
        logging.info(f"State dict map: {num_attn} attn modules, {num_expert} expert modules, {num_tensors} tensors")

        gc.collect()
        torch.cuda.empty_cache()
