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

"""Kimi K2.5 initializer for BatchGen.

Handles model initialization, weight loading, and INT4 W4A16 dequantization.

Key differences from DeepSeek-V3 initializer:
- Weight dtype: BF16 (not FP8) — K2.5 doesn't use FP8 for anything
- Routed experts: INT4 packed (uint8) + scale (bf16), not FP8
- KV dtype: BF16 only (no FP8 option for attention)
- Module shapes reflect INT4 packing dimensions
- 384 routed experts (vs 256)
"""

import logging
import os
from typing import Tuple

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.config.model_registry import load_config
from batchgen.models.deepseek.deepseekv3.configuration_deepseek_v3 import DeepseekV3Config
from .planner import KimiK25Planner

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    from batchgen.models.engine_loader import core_engine as loader_module
    core_engine = loader_module.batchgen


class KimiK25Initializer:
    """Initialize Kimi K2.5 model for BatchGen inference.

    Handles:
    - Model configuration parsing (MLA + 384 MoE experts)
    - Engine configuration with INT4 W4A16 module shapes
    - KV cache configuration (MLA compressed KV)
    """

    def __init__(self, input_arguments):
        # Create HuggingFace config for model instantiation.
        # K2.5 uses DeepseekV3ForCausalLM, so use DeepseekV3Config with K2.5 overrides.
        self.loaded_model_config = DeepseekV3Config(
            n_routed_experts=384,
            n_group=1,
            topk_group=1,
            rope_theta=50000.0,
            first_k_dense_replace=1,  # K2.5 has 1 dense layer (layer 0), not 3
        )
        self.loaded_model_config._name_or_path = input_arguments.huggingface_ckpt_name
        self.loaded_model_config.architectures = ["DeepseekV3ForCausalLM"]

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
        self.planner = KimiK25Planner()
        self.engine_config = self.planner.generate_config(self.engine_config)
        if self.global_rank == 0:
            logging.info(f"Engine config after planning: {self.engine_config}")

        self.shm_name = input_arguments.shm_name
        self.tensor_meta_shm_name = input_arguments.tensor_meta_shm_name

    def _set_basic_config(self, engine_config: EngineConfig, args) -> EngineConfig:
        """Set basic engine configuration for K2.5.

        K2.5 differences from DeepSeek-V3:
        - weight_dtype: bfloat16 (not float8_e4m3fn) — INT4 packed is uint8 but BF16 is default
        - kv_dtype: bfloat16 only (no FP8 attention)
        - module_types: same (attn, routed_expert, shared_expert)
        """
        engine_config.Basic_Config.device = args.device
        engine_config.Basic_Config.device_torch = torch.device(f"cuda:{args.device}")

        # K2.5 uses BF16 for attention and shared experts
        # Routed expert packed weights are uint8, overridden in weight_dtypes
        engine_config.Basic_Config.weight_dtype = "bfloat16"
        engine_config.Basic_Config.weight_dtype_torch = torch.bfloat16

        # KV cache: BF16 (K2.5 attention is BF16, no FP8 option)
        kv_dtype = getattr(args, 'kv_dtype', None)
        if kv_dtype and kv_dtype.lower() in ['fp8', 'float8', 'float8_e4m3fn']:
            logging.warning("K2.5 attention is BF16 — ignoring FP8 kv_dtype, using BF16")
        engine_config.Basic_Config.kv_dtype = "bfloat16"
        engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16

        # Activation dtype
        engine_config.Basic_Config.activation_dtype = "bfloat16"
        engine_config.Basic_Config.activation_dtype_torch = torch.bfloat16

        # Module types
        engine_config.Basic_Config.module_types = ["attn", "routed_expert", "shared_expert"]

        # Standard planner inputs
        engine_config.Basic_Config.padding_length = args.padding_length
        engine_config.Basic_Config.max_decoding_length = args.max_decoding_length
        engine_config.Basic_Config.world_size = args.world_size
        engine_config.Basic_Config.rank = args.rank
        engine_config.Basic_Config.num_queries = getattr(args, 'num_queries', 1)
        engine_config.Basic_Config.num_threads = 0

        # GPU arch
        gpu_arch = getattr(args, 'gpu_arch', 'hopper')
        if gpu_arch and gpu_arch.lower() not in ['hopper', 'ampere']:
            raise ValueError("Currently gpu_arch must be 'hopper' or 'ampere'")
        engine_config.Basic_Config.gpu_arch = gpu_arch.lower() if gpu_arch else 'hopper'

        # EP offloading
        if getattr(args, 'enable_ep_with_offloading', False):
            engine_config.EP_Config.enable_offloading = True
            engine_config.EP_Config.offloading_ratio = getattr(args, 'ep_offloading_ratio', 0.0)
            logging.info(
                f"EP offloading config set: enable_offloading=True, "
                f"offloading_ratio={engine_config.EP_Config.offloading_ratio}"
            )

        return engine_config

    def _default_engine_config(self):
        """Configure default engine settings for K2.5.

        INT4 W4A16 module shapes:
        - Packed: 2 INT4 values per uint8 byte → in_features // 2
        - Scale: 1 scale per 32 elements → in_features // 32
        """
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")

        # KV cache: MLA compressed KV (same as DeepSeek-V3)
        # compressed_kv_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
        self.engine_config.KV_Storage_Config.reserved_length = (
            self.engine_config.Basic_Config.padding_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * self.model_config.compressed_kv_dim
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

        self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
            * (
                self.engine_config.Basic_Config.max_decoding_length
                + self.engine_config.Basic_Config.padding_length
            )
        )

        # INT4 packing dimensions
        hidden_size = 7168
        moe_intermediate = 2048
        packed_hidden = hidden_size // 2       # 3584 (INT4 packed)
        scale_hidden = hidden_size // 32       # 224  (INT4 scale groups)
        packed_intermediate = moe_intermediate // 2   # 1024
        scale_intermediate = moe_intermediate // 32   # 64

        # Module shapes
        self.engine_config.GPU_Buffer_Config.module_shapes = {
            # MLA attention — same as DeepSeek-V3 (BF16)
            "attn": {
                "q_a_proj.weight": [1536, 7168],
                "q_a_layernorm.weight": [1536],
                "q_b_proj.weight": [24576, 1536],
                "kv_a_proj_with_mqa.weight": [576, 7168],
                "kv_a_layernorm.weight": [512],
                "kv_b_proj.weight": [32768, 512],
                "o_proj.weight": [7168, 16384],
            },
            # Routed experts — INT4 packed (uint8) + scale (bf16)
            "routed_expert": {
                "gate_proj.weight_packed": [moe_intermediate, packed_hidden],
                "gate_proj.weight_scale": [moe_intermediate, scale_hidden],
                "up_proj.weight_packed": [moe_intermediate, packed_hidden],
                "up_proj.weight_scale": [moe_intermediate, scale_hidden],
                "down_proj.weight_packed": [hidden_size, packed_intermediate],
                "down_proj.weight_scale": [hidden_size, scale_intermediate],
            },
            # Shared expert — BF16 (not quantized)
            "shared_expert": {
                "gate_proj.weight": [moe_intermediate, hidden_size],
                "up_proj.weight": [moe_intermediate, hidden_size],
                "down_proj.weight": [hidden_size, moe_intermediate],
            },
        }

        # Per-module default weight dtypes
        self.engine_config.GPU_Buffer_Config.weight_dtypes = {
            "attn": torch.bfloat16,
            "routed_expert": torch.uint8,      # Default for INT4 packed weights
            "shared_expert": torch.bfloat16,
        }

        # Per-tensor dtype overrides for mixed-dtype modules
        self.engine_config.GPU_Buffer_Config.tensor_dtypes = {
            "attn": {
                # Layernorm weights must be BF16
                "q_a_layernorm.weight": torch.bfloat16,
                "kv_a_layernorm.weight": torch.bfloat16,
            },
            "routed_expert": {
                # INT4 packed tensors are int32 (8 INT4 values per word)
                "gate_proj.weight_packed": torch.int32,
                "up_proj.weight_packed": torch.int32,
                "down_proj.weight_packed": torch.int32,
                # Scale tensors are BF16
                "gate_proj.weight_scale": torch.bfloat16,
                "up_proj.weight_scale": torch.bfloat16,
                "down_proj.weight_scale": torch.bfloat16,
            },
        }

    def _parse_model_config(self) -> ModelConfig:
        """Parse K2.5 model configuration."""
        model_config = ModelConfig()

        model_config.model_type = "kimi_k25"
        model_config.num_hidden_layers = 61
        model_config.num_local_experts = 384
        model_config.num_attention_heads = 64
        model_config.num_key_value_heads = 64
        model_config.head_dim = 192  # qk_nope_head_dim + qk_rope_head_dim
        model_config.compressed_kv_dim = 576  # kv_lora_rank + qk_rope_head_dim
        return model_config

    def Init(self, weights_storage) -> Tuple:
        """Initialize the core engine and load weights.

        Args:
            weights_storage: C++ Parameter_Server providing weight storage.

        Returns:
            Tuple of (core_engine, engine_config, model_config, loaded_model_config)
        """
        try:
            torch.cuda.set_device(self.local_rank)
            if self.global_rank == 0:
                logging.info(f"Engine config: {self.engine_config}")

            self.core_engine = core_engine(
                self.engine_config, self.model_config, weights_storage
            )

            logging.info("Core engine created")
            logging.info(f"_name_or_path: {self.loaded_model_config._name_or_path}")

            self.core_engine.Init()
            logging.info("Core engine initialized")

        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise e

        return (
            self.core_engine,
            self.engine_config,
            self.model_config,
            self.loaded_model_config,
        )
