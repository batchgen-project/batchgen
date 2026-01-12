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

"""GPT-OSS-120B initializer for BatchGen.

Handles model initialization, weight loading, and MXFP4 dequantization.
"""

import logging
import os
from typing import Tuple

import torch
import torch.nn as nn

from ....config.config import EngineConfig, ModelConfig
from .configuration_gpt_oss import GptOssConfig
from .planner import GptOssPlanner

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    from batchgen.models.engine_loader import core_engine as loader_module
    core_engine = loader_module.batchgen


def mxfp4_dequantize_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP4 packed weights.

    Args:
        packed: Packed FP4 values as uint8 [out_features, in_features//2]
        scales: Scale factors as uint8 [out_features, in_features//32]
        dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized weight tensor [out_features, in_features]
    """
    # Import the MXFP4 dequantization function
    from batchgen.quantization.mxfp4 import mxfp4_dequantize
    return mxfp4_dequantize(packed, scales, dtype)


class GptOssInitializer:
    """Initialize GPT-OSS-120B model for BatchGen inference.

    Handles:
    - Model configuration parsing
    - Engine configuration
    - Weight loading with MXFP4 dequantization
    - KV cache configuration for single H20 GPU
    """

    def __init__(self, input_arguments):
        self.hf_model_config = GptOssConfig()
        self.hf_model_config._name_or_path = input_arguments.huggingface_ckpt_name
        self.hf_model_config.architectures = ["GptOssForCausalLM"]

        self.host_kv_cache_size = input_arguments.host_kv_cache_size
        self.host_kv_cache_byte_size = input_arguments.host_kv_cache_size * (1024**3)
        self.global_kv_cache_size_gb = input_arguments.global_host_kv_cache_size_gb

        self.local_rank = input_arguments.local_rank
        self.global_rank = input_arguments.global_rank
        self.world_size = input_arguments.world_size
        self.enable_hugetlbfs = os.environ.get("BATCHGEN_ENABLE_HUGETLBFS", "0") == "1"
        logging.info(f"Enable hugetlbfs: {self.enable_hugetlbfs}")

        self.model_config = self._parse_model_config()

        self.engine_config = EngineConfig()
        logging.info(f"device: {input_arguments.device}")
        self.engine_config = self._set_basic_config(self.engine_config, input_arguments)
        self._default_engine_config()
        self.planner = GptOssPlanner()
        self.engine_config = self.planner.generate_config(self.engine_config)
        if self.global_rank == 0:
            logging.info(f"Engine config after planning: {self.engine_config}")

        self.shm_name = input_arguments.shm_name
        self.tensor_meta_shm_name = input_arguments.tensor_meta_shm_name

    def _set_basic_config(self, engine_config: EngineConfig, args) -> EngineConfig:
        """Set basic engine configuration from arguments."""
        engine_config.Basic_Config.device = args.device
        engine_config.Basic_Config.device_torch = torch.device(args.device)
        engine_config.Basic_Config.padding_length = args.padding_length
        engine_config.Basic_Config.max_decoding_length = args.max_decoding_length
        engine_config.Basic_Config.world_size = args.world_size
        engine_config.Basic_Config.local_rank = args.local_rank
        engine_config.Basic_Config.global_rank = args.global_rank

        # GPT-OSS uses BF16 for attention (not quantized)
        engine_config.Basic_Config.kv_dtype = "bfloat16"
        engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16

        return engine_config

    def _default_engine_config(self):
        """Configure default engine settings for GPT-OSS on H20."""
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")

        # KV cache configuration for GQA (8 KV heads, head_dim=64)
        # KV dim per layer = 2 * num_kv_heads * head_dim = 2 * 8 * 64 = 1024
        kv_dim = 2 * self.model_config.num_key_value_heads * self.model_config.head_dim

        self.engine_config.KV_Storage_Config.reserved_length = (
            self.engine_config.Basic_Config.padding_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * kv_dim
            * torch.finfo(self.engine_config.Basic_Config.kv_dtype_torch).bits // 8
        )
        self.engine_config.KV_Storage_Config.num_host_slots = (
            self.host_kv_cache_byte_size
            // self.engine_config.KV_Storage_Config.slot_byte_size
            // self.model_config.num_hidden_layers
        )
        self.engine_config.KV_Storage_Config.storage_byte_size = (
            self.host_kv_cache_byte_size
        )
        logging.info(
            f"Number of host kv slots: {self.engine_config.KV_Storage_Config.num_host_slots}"
        )

        # GPU buffer configuration
        self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
            * (
                self.engine_config.Basic_Config.max_decoding_length
                + self.engine_config.Basic_Config.padding_length
            )
        )

        # Module shapes for GPT-OSS (attention in BF16, experts in MXFP4)
        self.engine_config.GPU_Buffer_Config.module_shapes = {
            "attn": {
                # GQA: 64 query heads, 8 KV heads, head_dim=64
                "q_proj.weight": [4096, 2880],  # [64*64, hidden]
                "k_proj.weight": [512, 2880],   # [8*64, hidden]
                "v_proj.weight": [512, 2880],   # [8*64, hidden]
                "o_proj.weight": [2880, 4096],  # [hidden, 64*64]
            },
            "routed_expert": {
                # MXFP4 quantized: store as packed + scales
                "gate_proj.weight": [2880, 2880],  # Will be MXFP4
                "up_proj.weight": [2880, 2880],
                "down_proj.weight": [2880, 2880],
            },
        }

    def _parse_model_config(self) -> ModelConfig:
        """Parse GPT-OSS model configuration."""
        model_config = ModelConfig()

        model_config.model_type = "gpt_oss"
        model_config.num_hidden_layers = 36
        model_config.num_local_experts = 128
        model_config.num_attention_heads = 64
        model_config.num_key_value_heads = 8
        model_config.head_dim = 64
        model_config.hidden_size = 2880
        model_config.intermediate_size = 2880
        model_config.num_experts_per_tok = 4
        model_config.sliding_window = 128

        return model_config

    def Init(self, weights_storage) -> Tuple:
        """Initialize the core engine and load weights.

        Args:
            weights_storage: Storage for model weights

        Returns:
            Tuple of (core_engine, engine_config, model_config, hf_model_config)
        """
        try:
            torch.cuda.set_device(self.local_rank)
            if self.global_rank == 0:
                logging.info(f"Engine config: {self.engine_config}")

            self.core_engine = core_engine(
                self.engine_config, self.model_config, weights_storage
            )

            logging.info("Core engine created")
            logging.info(f"_name_or_path: {self.hf_model_config._name_or_path}")

            # GPT-OSS-120B: ~55GB MXFP4 weights
            if "gpt-oss" in self.hf_model_config._name_or_path.lower():
                param_byte_size = 55 * 1024 * 1024 * 1024  # ~55GB for MXFP4
            else:
                raise ValueError("Unknown huggingface model card for GPT-OSS")

            self.core_engine.Init()
            logging.info("Core engine initialized")

        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise e

        return (
            self.core_engine,
            self.engine_config,
            self.model_config,
            self.hf_model_config,
        )
