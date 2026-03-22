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

"""Qwen3 initializer for BatchGen.

Handles model initialization and BF16 weight loading.
Target: Qwen3Guard-Gen-8B on single Ada 6000 / A6000 GPU.
"""

import logging
import os
from typing import Tuple, Optional

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from .config import Qwen3Config
from .qwen3_parameter_server import Qwen3ParameterServer

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    from batchgen.models.engine_loader import core_engine as loader_module
    core_engine = loader_module.batchgen


class Qwen3Initializer:
    """Initialize Qwen3 model for BatchGen inference.

    Handles:
    - Model configuration parsing
    - Engine configuration for single device
    - BF16 weight loading (no quantization)
    - KV cache configuration for Ada 6000 / A6000 (48GB)
    """

    def __init__(self, input_arguments):
        self.loaded_model_config = Qwen3Config()
        self.loaded_model_config._name_or_path = input_arguments.huggingface_ckpt_name
        self.loaded_model_config.architectures = ["Qwen3ForCausalLM"]

        self.host_kv_cache_size = input_arguments.host_kv_cache_size
        self.host_kv_cache_byte_size = input_arguments.host_kv_cache_size * (1024**3)
        self.global_kv_cache_size_gb = input_arguments.global_host_kv_cache_size_gb

        self.local_rank = input_arguments.local_rank
        self.global_rank = input_arguments.global_rank
        self.world_size = input_arguments.world_size

        self.cache_dir = getattr(input_arguments, 'cache_dir', None)
        self.converted_ckpt_dir = getattr(input_arguments, 'converted_ckpt_dir', None)

        self.enable_hugetlbfs = os.environ.get("BATCHGEN_ENABLE_HUGETLBFS", "0") == "1"

        self.model_config = self._parse_model_config()

        self.engine_config = EngineConfig()
        self.engine_config = self._set_basic_config(self.engine_config, input_arguments)
        self._default_engine_config()

        self.shm_name = input_arguments.shm_name
        self.tensor_meta_shm_name = input_arguments.tensor_meta_shm_name

        self.parameter_server: Optional[Qwen3ParameterServer] = None

    def _set_basic_config(self, engine_config: EngineConfig, args) -> EngineConfig:
        """Set basic engine configuration."""
        engine_config.Basic_Config.device = args.device
        engine_config.Basic_Config.device_torch = torch.device(args.device)
        engine_config.Basic_Config.padding_length = args.padding_length
        engine_config.Basic_Config.max_decoding_length = args.max_decoding_length

        # Qwen3: BF16 everything
        engine_config.Basic_Config.kv_dtype = "bfloat16"
        engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16
        engine_config.Basic_Config.weight_dtype = "bfloat16"
        engine_config.Basic_Config.weight_dtype_torch = torch.bfloat16
        engine_config.Basic_Config.activation_dtype = "bfloat16"
        engine_config.Basic_Config.activation_dtype_torch = torch.bfloat16

        engine_config.Basic_Config.world_size = args.world_size
        engine_config.Basic_Config.rank = args.rank
        engine_config.Basic_Config.num_queries = getattr(args, 'num_queries', 1)
        # Dense 8B model fits entirely in GPU memory — use only attn module type
        # MLP weights are persistent in skeleton (no dynamic loading needed)
        engine_config.Basic_Config.module_types = ["attn"]
        engine_config.Basic_Config.num_threads = 0
        engine_config.Basic_Config.gpu_arch = getattr(args, 'gpu_arch', 'ampere')

        return engine_config

    def _default_engine_config(self):
        """Configure default engine settings for Qwen3 on Ada/Ampere."""
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory:.2f} GB")

        # KV cache for GQA (8 KV heads, head_dim=128)
        # KV dim per layer = 2 * num_kv_heads * head_dim = 2 * 8 * 128 = 2048
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

        # Module shapes for Qwen3 (all BF16)
        hidden_size = self.model_config.hidden_size         # 4096
        intermediate_size = self.model_config.intermediate_size  # 12288
        num_heads = self.model_config.num_attention_heads    # 32
        num_kv_heads = self.model_config.num_key_value_heads  # 8
        head_dim = self.model_config.head_dim               # 128

        # Micro-batch sizes for dense model (no planner needed)
        self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 128
        self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size = 128
        self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 1
        self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 1
        self.engine_config.Module_Batching_Config.global_batch_size = 128

        # Module buffer counts for C++ core engine (must be Dict[str, int])
        self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
            "attn": 1,
        }
        self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
            "attn": 0,  # Persistent during decode
        }

        # Only attn uses dynamic weight loading via core engine
        # MLP weights are persistent in model skeleton (fits in GPU memory)
        self.engine_config.GPU_Buffer_Config.module_shapes = {
            "attn": {
                "q_proj.weight": [num_heads * head_dim, hidden_size],
                "k_proj.weight": [num_kv_heads * head_dim, hidden_size],
                "v_proj.weight": [num_kv_heads * head_dim, hidden_size],
                "o_proj.weight": [hidden_size, num_heads * head_dim],
                "q_norm.weight": [head_dim],
                "k_norm.weight": [head_dim],
            },
        }

        self.engine_config.GPU_Buffer_Config.weight_dtypes = {
            "attn": torch.bfloat16,
        }

    def _parse_model_config(self) -> ModelConfig:
        """Parse Qwen3 model configuration."""
        model_config = ModelConfig()

        model_config.model_type = "qwen3"
        model_config.num_hidden_layers = 36
        model_config.num_local_experts = 0  # Dense model (no MoE)
        model_config.num_attention_heads = 32
        model_config.num_key_value_heads = 8
        model_config.head_dim = 128
        model_config.hidden_size = 4096
        model_config.intermediate_size = 12288

        return model_config

    def _create_parameter_server(self) -> Qwen3ParameterServer:
        """Create and initialize the parameter server for BF16 weight loading."""
        logging.info("Creating Qwen3 parameter server")

        parameter_server = Qwen3ParameterServer(
            huggingface_ckpt_name=self.loaded_model_config._name_or_path,
            cache_dir=self.cache_dir,
            converted_ckpt_dir=self.converted_ckpt_dir,
            enable_hugetlbfs=self.enable_hugetlbfs,
        )
        parameter_server.Init()

        logging.info("Parameter server initialized")
        return parameter_server

    def Init(self, weights_storage=None) -> Tuple:
        """Initialize the core engine and load weights.

        Returns:
            Tuple of (core_engine, engine_config, model_config, loaded_model_config)
        """
        try:
            torch.cuda.set_device(self.local_rank)
            if self.global_rank == 0:
                logging.info(f"Engine config: {self.engine_config}")

            if weights_storage is None:
                self.parameter_server = self._create_parameter_server()
                weights_storage = self.parameter_server.weights_storage

            self.core_engine = core_engine(
                self.engine_config, self.model_config, weights_storage
            )

            logging.info("Core engine created")

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
