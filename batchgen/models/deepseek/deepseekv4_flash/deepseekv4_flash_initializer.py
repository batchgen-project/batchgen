# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash initializer.

The native V4 runtime is not a DeepSeek-V3 alias. This initializer carries the
V4 model/engine metadata so routing is correct while the dedicated V4 model
wrapper is being completed.
"""

from __future__ import annotations

import logging
import os

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.config.engine_config_parser import parse_config_from_json
from batchgen.kv_cache.host_kv_mananger_config import build_host_kv_config

from .set_basic_config import set_basic_config

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    from batchgen.models.engine_loader import core_engine as loader_module

    core_engine = loader_module.batchgen


class DeepSeekV4FlashInitializer:
    def __init__(self, input_arguments):
        self.loaded_model_config = None
        self.host_kv_cache_size = input_arguments.host_kv_cache_size
        self.host_kv_cache_byte_size = input_arguments.host_kv_cache_size * (1024**3)
        self.global_kv_cache_size_gb = input_arguments.global_host_kv_cache_size_gb
        self.local_rank = input_arguments.local_rank
        self.global_rank = input_arguments.global_rank
        self.world_size = input_arguments.world_size
        self.enable_hugetlbfs = os.environ.get("BATCHGEN_ENABLE_HUGETLBFS", "0") == "1"

        self.model_config = self._parse_model_config()
        self.engine_config = EngineConfig()
        self.engine_config = set_basic_config(self.engine_config, input_arguments)
        self._default_engine_config(input_arguments)

        self.shm_name = input_arguments.shm_name
        self.tensor_meta_shm_name = input_arguments.tensor_meta_shm_name

    def _parse_model_config(self):
        model_config = ModelConfig()
        model_config.model_type = "deepseek_v4_flash"
        model_config.num_hidden_layers = 43
        model_config.num_local_experts = 256
        model_config.num_attention_heads = 64
        model_config.num_key_value_heads = 1
        model_config.head_dim = 512
        model_config.compressed_kv_dim = 512
        return model_config

    def _default_engine_config(self, input_arguments):
        self.engine_config.KV_Storage_Config.reserved_length = (
            self.engine_config.Basic_Config.padding_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * self.model_config.compressed_kv_dim
            * torch.finfo(self.engine_config.Basic_Config.kv_dtype_torch).bits
            // 8
        )
        self.engine_config.KV_Storage_Config.num_host_slots = (
            self.host_kv_cache_byte_size
            // self.engine_config.KV_Storage_Config.slot_byte_size
            // self.model_config.num_hidden_layers
        )
        self.engine_config.KV_Storage_Config.storage_byte_size = (
            self.host_kv_cache_byte_size
        )
        self.engine_config.KV_Storage_Config.host_kv_cache_config = build_host_kv_config(
            input_arguments.huggingface_ckpt_name,
            self.host_kv_cache_byte_size,
        )
        self._set_batching_and_buffer_config()
        self.engine_config.GPU_Buffer_Config.module_shapes = {
            "attn": {
                "attn_sink": [64],
                "wq_a.weight": [1024, 4096],
                "wq_a.scale": [8, 32],
                "q_norm.weight": [1024],
                "wq_b.weight": [32768, 1024],
                "wq_b.scale": [256, 8],
                "wkv.weight": [512, 4096],
                "wkv.scale": [4, 32],
                "kv_norm.weight": [512],
                "wo_a.weight": [8192, 4096],
                "wo_a.scale": [64, 32],
                "wo_b.weight": [4096, 8192],
                "wo_b.scale": [32, 64],
            },
            "routed_expert": {
                "w1.weight": [2048, 2048],
                "w1.scale": [2048, 128],
                "w2.weight": [4096, 1024],
                "w2.scale": [4096, 64],
                "w3.weight": [2048, 2048],
                "w3.scale": [2048, 128],
            },
            "shared_expert": {
                "w1.weight": [2048, 4096],
                "w1.scale": [16, 32],
                "w2.weight": [4096, 2048],
                "w2.scale": [32, 16],
                "w3.weight": [2048, 4096],
                "w3.scale": [16, 32],
            },
        }
        logging.info(
            "DeepSeek-V4-Flash engine metadata initialized: host_slots=%s",
            self.engine_config.KV_Storage_Config.num_host_slots,
        )

    def _set_batching_and_buffer_config(self):
        reserved_length = self.engine_config.KV_Storage_Config.reserved_length
        world_size = max(1, int(self.world_size))
        experts_per_rank = self.model_config.num_local_experts // world_size

        self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 8
        self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 8
        self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 4096
        self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 128
        self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size = 128
        self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 2048

        self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
            "attn": 1,
            "routed_expert": experts_per_rank,
            "shared_expert": 1,
        }
        self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
            "attn": 1,
            "routed_expert": max(experts_per_rank, 1),
            "shared_expert": 1,
        }
        self.engine_config.GPU_Buffer_Config.num_k_buffer = 6
        self.engine_config.GPU_Buffer_Config.num_v_buffer = 0
        self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
            * reserved_length
        )
        self.engine_config.EP_Config.enable = True
        self.engine_config.EP_Config.num_local_expert_per_layer = experts_per_rank

    def Init(self, weights_storage):
        torch.cuda.set_device(self.local_rank)
        self.core_engine = core_engine(
            self.engine_config,
            self.model_config,
            weights_storage,
        )

    def get_configs(self):
        return self.loaded_model_config, self.engine_config, self.model_config

    def parse_json_config(self, json_file_path):
        parse_config_from_json(json_file_path, self.engine_config, self.model_config)
