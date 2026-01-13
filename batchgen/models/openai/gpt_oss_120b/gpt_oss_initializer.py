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

Handles model initialization and weight loading using W4A16 fused GEMM.

GPT-OSS uses MXFP4 quantization (4-bit weights) with W4A16 inference:
- Weights: MXFP4 (uint8 packed + uint8 scales) - stay in 4-bit
- Activations: BF16
- GEMM: Fused dequant-GEMM (dequantize on-the-fly during matrix multiply)
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


class GptOssInitializer:
    """Initialize GPT-OSS-120B model for BatchGen inference.

    Handles:
    - Model configuration parsing (OpenAI-style config)
    - Engine configuration for W4A16 MXFP4 inference
    - Weight loading (MXFP4 packed + scales, no pre-dequantization)
    - KV cache configuration for single H20 GPU

    GPT-OSS-120B architecture:
    - 36 layers, 128 experts (Top-4 routing)
    - GQA: 64 attention heads, 8 KV heads
    - Combined QKV projection (not separate Q/K/V)
    - MXFP4 quantized expert weights (~55GB total)
    - BF16 attention weights (~3GB)
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
        # Device
        engine_config.Basic_Config.device = args.device
        engine_config.Basic_Config.device_torch = torch.device(f"cuda:{args.device}")

        # Distributed
        engine_config.Basic_Config.world_size = args.world_size
        engine_config.Basic_Config.rank = args.rank

        # GPU architecture
        if args.gpu_arch:
            engine_config.Basic_Config.gpu_arch = args.gpu_arch.lower()
        else:
            # Auto-detect GPU architecture
            props = torch.cuda.get_device_properties(args.device)
            if props.major >= 9:
                engine_config.Basic_Config.gpu_arch = "hopper"
            else:
                engine_config.Basic_Config.gpu_arch = "ampere"

        # Lengths
        engine_config.Basic_Config.padding_length = args.padding_length
        engine_config.Basic_Config.max_decoding_length = args.max_decoding_length

        # Weight dtype: GPT-OSS uses MXFP4 (W4A16 - keep weights as 4-bit uint8)
        # Expert weights stored as packed uint8 + scales uint8
        # Dequantization happens on-the-fly during fused GEMM
        engine_config.Basic_Config.weight_dtype = "uint8"
        engine_config.Basic_Config.weight_dtype_torch = torch.uint8

        # KV dtype: GPT-OSS uses BF16 for attention (not quantized)
        engine_config.Basic_Config.kv_dtype = "bfloat16"
        engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16

        # Attention dtype
        engine_config.Basic_Config.attention_dtype = "bfloat16"

        # Activation dtype
        engine_config.Basic_Config.activation_dtype = "bfloat16"
        engine_config.Basic_Config.activation_dtype_torch = torch.bfloat16

        # Module types - only routed_expert for GPT-OSS
        # NOTE: Attention weights are SKELETON (loaded once via _load_model_skeleton),
        # not dynamically loaded via HtoD. Including "attn" here would cause the
        # HtoD worker to block forever waiting on an empty queue.
        engine_config.Basic_Config.module_types = ["routed_expert"]

        # Misc
        engine_config.Basic_Config.num_threads = 0
        engine_config.Basic_Config.log_level = "info"

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
        # GPT-OSS uses combined QKV projection (OpenAI style), not separate Q/K/V
        hidden_size = 2880
        num_q_heads = 64
        num_kv_heads = 8
        head_dim = 64
        intermediate_size = 2880

        # Combined QKV dim: Q (64*64) + K (8*64) + V (8*64) = 4096 + 512 + 512 = 5120
        qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim  # 5120
        out_dim = num_q_heads * head_dim  # 4096

        # NOTE: Only routed_expert needs GPU buffer allocation.
        # Attention weights are SKELETON (loaded once into model parameters).
        self.engine_config.GPU_Buffer_Config.module_shapes = {
            "routed_expert": {
                # MXFP4 quantized expert weights
                # mlp1 = gate_proj || up_proj (concatenated)
                # mlp2 = down_proj
                # Stored as packed uint8 + scales uint8 + bias BF16
                "mlp1.packed": [intermediate_size * 2, hidden_size // 2],  # [5760, 1440]
                "mlp1.scales": [intermediate_size * 2, hidden_size // 32],  # [5760, 90]
                "mlp1.bias": [intermediate_size * 2],  # [5760]
                "mlp2.packed": [hidden_size, intermediate_size // 2],  # [2880, 1440]
                "mlp2.scales": [hidden_size, intermediate_size // 32],  # [2880, 90]
                "mlp2.bias": [hidden_size],  # [2880]
            },
        }

    def _parse_model_config(self) -> ModelConfig:
        """Parse GPT-OSS model configuration (OpenAI style)."""
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
        model_config.swiglu_limit = 7.0
        model_config.vocab_size = 201088
        model_config.rope_theta = 150000.0

        return model_config

    def Init(self, weights_storage) -> Tuple:
        """Initialize the core engine and load weights.

        GPT-OSS uses W4A16 inference (MXFP4 weights, BF16 activations).
        Weights are loaded as packed uint8 + scales uint8, no pre-dequantization.
        Dequantization happens on-the-fly in fused MXFP4 GEMM kernels.

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

            # GPT-OSS-120B memory estimate (W4A16):
            # - MXFP4 expert weights: ~55GB (128 experts × 36 layers, packed + scales)
            # - BF16 attention weights: ~3GB (not quantized)
            # - Embeddings + LM head: ~2GB
            # Total: ~60GB (fits on single H20 96GB GPU)
            if "gpt-oss" in self.hf_model_config._name_or_path.lower() or \
               "gpt_oss" in self.hf_model_config._name_or_path.lower():
                param_byte_size = 60 * 1024 * 1024 * 1024  # ~60GB for MXFP4 + BF16
            else:
                raise ValueError("Unknown model card for GPT-OSS")

            self.core_engine.Init()
            logging.info("Core engine initialized (W4A16 MXFP4 mode)")

        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise e

        return (
            self.core_engine,
            self.engine_config,
            self.model_config,
            self.hf_model_config,
        )
