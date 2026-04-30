# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 initializer for BatchGen.

Standalone — no cross-model imports.

Key differences from DeepSeek V3 initializer:
- 78 layers, hidden_size=6144
- q_lora_rank=2048, qk_nope_head_dim=192, v_head_dim=256
- q_b_proj shape: [64*256, 2048] = [16384, 2048]
- kv_b_proj shape: [64*(192+256), 512] = [28672, 512]
- o_proj shape: [6144, 64*256] = [6144, 16384]
- compressed_kv_dim=576 (same as DeepSeek)
"""

import logging
import os

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from .configuration_glm5 import Glm5Config
from .set_basic_config import set_basic_config
from .planner import GLM5Planner
from batchgen.kv_cache.host_kv_mananger_config import build_host_kv_config

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    from batchgen.models.engine_loader import core_engine as loader_module
    core_engine = loader_module.batchgen


class GLM5Initializer:
    def __init__(self, input_arguments):
        self.loaded_model_config = Glm5Config()
        self.loaded_model_config._name_or_path = input_arguments.huggingface_ckpt_name
        self.loaded_model_config.architectures = ["GlmMoeDsaForCausalLM"]

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
        self.engine_config = set_basic_config(self.engine_config, input_arguments)
        self._default_engine_config()
        self.planner = GLM5Planner(model_name=input_arguments.huggingface_ckpt_name)
        self.engine_config = self.planner.generate_config(self.engine_config)
        if self.global_rank == 0:
            logging.info(f"Engine config after planning: {self.engine_config}")

        self.shm_name = input_arguments.shm_name
        self.tensor_meta_shm_name = input_arguments.tensor_meta_shm_name

    def _default_engine_config(self):
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")

        # KV storage config
        max_prompt_length = self.engine_config.Basic_Config.get_max_prompt_length()
        self.engine_config.KV_Storage_Config.reserved_length = (
            max_prompt_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * self.model_config.compressed_kv_dim  # 576
            * torch.finfo(self.engine_config.Basic_Config.kv_dtype_torch).bits // 8
        )
        self.engine_config.KV_Storage_Config.num_host_slots = (
            self.host_kv_cache_byte_size
            // self.engine_config.KV_Storage_Config.slot_byte_size
            // self.model_config.num_hidden_layers  # 78
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
                + max_prompt_length
            )
        )

        # GLM-5 module shapes
        # q_b_proj: [n_heads * (qk_nope + qk_rope), q_lora_rank] = [64*256, 2048] = [16384, 2048]
        # kv_b_proj: [n_heads * (qk_nope + v_head), kv_lora_rank] = [64*(192+256), 512] = [28672, 512]
        # o_proj: [hidden_size, n_heads * v_head_dim] = [6144, 64*256] = [6144, 16384]
        # q_a_layernorm and kv_a_layernorm are now loaded via the skeleton path
        # at model init (see glm5_parameter_server.py) — tiny BF16 RMSNorm
        # weights that don't benefit from the async copy-task. Listing them
        # here caused the copy-task bundle to allocate space but the actual
        # data never got written, leaving the live nn.Module at its ones_()
        # init and every attention Q/K RMSNorm silently wrong.
        self.engine_config.GPU_Buffer_Config.module_shapes = {
            "attn": {
                "q_a_proj.weight": [2048, 6144],
                "q_b_proj.weight": [16384, 2048],
                "kv_a_proj_with_mqa.weight": [576, 6144],
                "kv_b_proj.weight": [28672, 512],
                "o_proj.weight": [6144, 16384],
            },
            "routed_expert": {
                "gate_proj.weight": [2048, 6144],
                "up_proj.weight": [2048, 6144],
                "down_proj.weight": [6144, 2048],
            },
            "shared_expert": {
                "gate_proj.weight": [2048, 6144],
                "up_proj.weight": [2048, 6144],
                "down_proj.weight": [6144, 2048],
            },
        }

        # No per-tensor dtype overrides needed — attn bundle is all FP8 now
        # (norms moved to skeleton).
        self.engine_config.GPU_Buffer_Config.tensor_dtypes = {
            "attn": {},
        }

    def _parse_model_config(self):
        model_config = ModelConfig()
        model_config.model_type = "glm_moe_dsa"
        model_config.num_hidden_layers = 78
        model_config.num_local_experts = 256
        model_config.num_attention_heads = 64
        model_config.num_key_value_heads = 64
        model_config.head_dim = 256  # qk_nope + qk_rope = 192 + 64
        model_config.compressed_kv_dim = 576  # kv_lora_rank + qk_rope_head_dim
        model_config.first_k_dense_replace = 3
        return model_config

    def Init(self, weights_storage):
        try:
            torch.cuda.set_device(self.local_rank)
            if self.global_rank == 0:
                logging.info(f"Engine config: {self.engine_config}")
            self.core_engine = core_engine(
                self.engine_config, self.model_config, weights_storage
            )
            logging.info("Core engine created")

            self.core_engine.Init()
            logging.info("Core engine initialized")
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        return (
            self.core_engine,
            self.engine_config,
            self.model_config,
            self.loaded_model_config,
        )
