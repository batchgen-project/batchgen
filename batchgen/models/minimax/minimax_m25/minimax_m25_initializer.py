# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""MiniMax-M2.5 initializer for BatchGen.

Handles model initialization, weight loading, and engine configuration.

Key differences from Kimi K2.5 initializer:
- GQA attention (not MLA) — standard Q/K/V/O projections + QK norm
- FP8 expert weights (not INT4) — float8_e4m3fn, block [128,128]
- KV cache: standard GQA (8 KV heads × head_dim=128), not MLA compressed KV
- No shared experts
"""

import logging
import os
from typing import Tuple

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.config.model_registry import load_config
from .config import MiniMaxM25Config
from .planner import MiniMaxM25Planner

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    from batchgen.models.engine_loader import core_engine as loader_module
    core_engine = loader_module.batchgen


class MiniMaxM25Initializer:
    def __init__(self, input_arguments):
        self.batchgen_config = load_config(input_arguments.huggingface_ckpt_name)

        self.loaded_model_config = MiniMaxM25Config()
        self.loaded_model_config._name_or_path = input_arguments.huggingface_ckpt_name

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
        self.planner = MiniMaxM25Planner()
        self.engine_config = self.planner.generate_config(self.engine_config)
        if self.global_rank == 0:
            logging.info(f"Engine config after planning: {self.engine_config}")

        self.shm_name = input_arguments.shm_name
        self.tensor_meta_shm_name = input_arguments.tensor_meta_shm_name

    def _set_basic_config(self, engine_config: EngineConfig, args) -> EngineConfig:
        """Set basic engine configuration for M2.5.

        M2.5 differences from Kimi K2.5:
        - weight_dtype: float8_e4m3fn (FP8 experts, not INT4)
        - kv_dtype: bfloat16 (standard GQA KV cache)
        - module_types: attn + routed_expert only (no shared_expert)
        """
        engine_config.Basic_Config.device = args.device
        engine_config.Basic_Config.device_torch = torch.device(f"cuda:{args.device}")

        # FP8 expert weights
        engine_config.Basic_Config.weight_dtype = "float8_e4m3fn"
        engine_config.Basic_Config.weight_dtype_torch = torch.float8_e4m3fn

        # KV cache: BF16
        engine_config.Basic_Config.kv_dtype = "bfloat16"
        engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16

        # Activation dtype
        engine_config.Basic_Config.activation_dtype = "bfloat16"
        engine_config.Basic_Config.activation_dtype_torch = torch.bfloat16

        # Module types: no shared experts
        engine_config.Basic_Config.module_types = ["attn", "routed_expert"]

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
        """Configure default engine settings for M2.5.

        GQA KV cache: 8 KV heads × head_dim=128 × 2 (K+V) per layer per token.
        FP8 expert module shapes: [128,128] block-quantized.
        """
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")

        # KV cache: standard GQA (not MLA compressed)
        # Per-token KV size = num_kv_heads × head_dim × 2 (K+V) × dtype_bytes
        cfg = self.batchgen_config
        kv_dim_per_token = cfg.num_key_value_heads * cfg.head_dim * 2  # K + V
        kv_dtype_bytes = torch.finfo(self.engine_config.Basic_Config.kv_dtype_torch).bits // 8

        self.engine_config.KV_Storage_Config.reserved_length = (
            self.engine_config.Basic_Config.padding_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * kv_dim_per_token
            * kv_dtype_bytes
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

        # Module shapes
        hidden_size = cfg.hidden_size       # 3072
        intermediate = cfg.moe_intermediate_size  # 1536
        num_heads = cfg.num_attention_heads  # 48
        num_kv_heads = cfg.num_key_value_heads  # 8
        head_dim = cfg.head_dim              # 128

        self.engine_config.GPU_Buffer_Config.module_shapes = {
            # GQA attention (BF16)
            "attn": {
                "q_proj.weight": [num_heads * head_dim, hidden_size],
                "k_proj.weight": [num_kv_heads * head_dim, hidden_size],
                "v_proj.weight": [num_kv_heads * head_dim, hidden_size],
                "o_proj.weight": [hidden_size, num_heads * head_dim],
                "q_norm.weight": [num_heads * head_dim],
                "k_norm.weight": [num_kv_heads * head_dim],
            },
            # Routed experts — FP8 (float8_e4m3fn)
            "routed_expert": {
                "w1.weight": [intermediate, hidden_size],       # gate_proj
                "w1.weight_scale_inv": [intermediate // 128, hidden_size // 128],
                "w2.weight": [hidden_size, intermediate],       # down_proj
                "w2.weight_scale_inv": [hidden_size // 128, intermediate // 128],
                "w3.weight": [intermediate, hidden_size],       # up_proj
                "w3.weight_scale_inv": [intermediate // 128, hidden_size // 128],
            },
        }

        # Per-module default weight dtypes
        self.engine_config.GPU_Buffer_Config.weight_dtypes = {
            "attn": torch.bfloat16,
            "routed_expert": torch.float8_e4m3fn,
        }

        # Per-tensor dtype overrides
        self.engine_config.GPU_Buffer_Config.tensor_dtypes = {
            "attn": {
                "q_norm.weight": torch.bfloat16,
                "k_norm.weight": torch.bfloat16,
            },
            "routed_expert": {
                "w1.weight_scale_inv": torch.bfloat16,
                "w2.weight_scale_inv": torch.bfloat16,
                "w3.weight_scale_inv": torch.bfloat16,
            },
        }

    def _parse_model_config(self) -> ModelConfig:
        """Parse M2.5 model configuration."""
        cfg = self.batchgen_config
        model_config = ModelConfig()

        model_config.model_type = cfg.model_type
        model_config.num_hidden_layers = cfg.num_hidden_layers
        model_config.num_local_experts = cfg.num_local_experts
        model_config.num_attention_heads = cfg.num_attention_heads
        model_config.num_key_value_heads = cfg.num_key_value_heads
        model_config.head_dim = cfg.head_dim
        return model_config

    def Init(self, weights_storage) -> Tuple:
        """Initialize the core engine and load weights."""
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
