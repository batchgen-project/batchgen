from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from batchgen.ckpt_converter.metadata_loader import (
    build_module_metadata,
    diff_shapes,
    load_checkpoint_metadata,
    resolve_torch_dtype,
)
from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.config.engine_config_parser import parse_config_from_json
from batchgen.config.model_registry import load_config
from batchgen.kv_cache.host_kv_mananger_config import build_host_kv_config

from .set_basic_config import set_basic_config
from .tensor_contract import build_v4_weight_contract, get_v4_attn_module_types

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    from batchgen.models.engine_loader import core_engine as loader_module

    core_engine = loader_module.batchgen


class DeepSeekV4FlashInitializer:
    def __init__(self, input_arguments):
        self.loaded_model_config = load_config(
            input_arguments.huggingface_ckpt_name
        )
        self.host_kv_cache_size = input_arguments.host_kv_cache_size
        self.host_kv_cache_byte_size = input_arguments.host_kv_cache_size * (
            1024**3
        )
        self.global_kv_cache_size_gb = (
            input_arguments.global_host_kv_cache_size_gb
        )
        self.local_rank = input_arguments.local_rank
        self.global_rank = input_arguments.global_rank
        self.world_size = input_arguments.world_size
        self.enable_hugetlbfs = (
            os.environ.get("BATCHGEN_ENABLE_HUGETLBFS", "0") == "1"
        )
        self.converted_ckpt_dir = getattr(
            input_arguments, "converted_ckpt_dir", None
        )

        self.model_config = self._parse_model_config()
        # Expose the converted-ckpt dir so the attention wrapper can reconstruct
        # full (de-sharded) attention weights for DP-only prefill.
        if self.converted_ckpt_dir is not None:
            self.model_config.converted_ckpt_dir = str(self.converted_ckpt_dir)
        self.module_metadata = self._load_module_metadata()

        self.engine_config = EngineConfig()
        self.engine_config = set_basic_config(
            self.engine_config, input_arguments
        )
        attn_types = get_v4_attn_module_types(self.model_config)
        self.engine_config.Basic_Config.module_types = attn_types + [
            "routed_expert",
            "shared_expert",
        ]
        self._default_engine_config(input_arguments)

        self.shm_name = input_arguments.shm_name
        self.tensor_meta_shm_name = input_arguments.tensor_meta_shm_name

    def _parse_model_config(self):
        loaded = self.loaded_model_config
        model_config = ModelConfig()
        model_config.model_type = "deepseek_v4_flash"
        model_config.num_hidden_layers = int(
            getattr(loaded, "num_hidden_layers", 43)
        )
        model_config.num_local_experts = int(
            getattr(loaded, "n_routed_experts", 256)
        )
        model_config.num_attention_heads = int(
            getattr(loaded, "num_attention_heads", 64)
        )
        model_config.num_key_value_heads = 1
        model_config.head_dim = int(getattr(loaded, "head_dim", 512))
        model_config.compressed_kv_dim = 512
        # The weight contract enumerates per-layer compressor/indexer attention
        # tensors based on compress_ratios; without this the GPU buffer omits
        # those slots and ratio-4/128 layers fail to load their weights.
        model_config.compress_ratios = list(
            getattr(loaded, "compress_ratios", [])
        )
        model_config.n_routed_experts = model_config.num_local_experts
        model_config.num_nextn_predict_layers = int(
            getattr(loaded, "num_nextn_predict_layers", 1)
        )
        return model_config

    def _load_module_metadata(self) -> Dict[str, Dict[str, Dict[str, object]]]:
        if self.converted_ckpt_dir is None:
            raise ValueError(
                "DeepSeek-V4-Flash initializer requires converted_ckpt_dir to be set; "
                "cannot auto-discover module shapes/dtypes from checkpoint metadata."
            )
        ckpt_dir = Path(self.converted_ckpt_dir)
        state_dict_name_map, _ = build_v4_weight_contract(self.model_config)
        tensor_metadata = load_checkpoint_metadata(
            ckpt_dir,
            rank=self.local_rank,
            world_size=self.world_size,
        )
        module_meta = build_module_metadata(
            tensor_metadata, state_dict_name_map
        )
        if self.global_rank == 0:
            for module_type in sorted(module_meta):
                logging.info(
                    "V4-Flash %s metadata: %d unique tensor keys discovered",
                    module_type,
                    len(module_meta[module_type]),
                )
        return module_meta

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
        self.engine_config.KV_Storage_Config.host_kv_cache_config = (
            build_host_kv_config(
                input_arguments.huggingface_ckpt_name,
                self.host_kv_cache_byte_size,
                kv_dtype_override=self.engine_config.Basic_Config.kv_dtype,
            )
        )
        self._set_batching_and_buffer_config()

        module_shapes, tensor_dtypes = self._build_buffer_metadata()
        self.engine_config.GPU_Buffer_Config.module_shapes = module_shapes
        self.engine_config.GPU_Buffer_Config.tensor_dtypes = tensor_dtypes

        if self.global_rank == 0:
            shape_summary = ", ".join(
                f"{mt}:{len(tensors)}"
                for mt, tensors in sorted(module_shapes.items())
            )
            logging.info(
                "DeepSeek-V4-Flash engine metadata initialized: host_slots=%s, "
                "module_shapes={%s}",
                self.engine_config.KV_Storage_Config.num_host_slots,
                shape_summary,
            )

    def _build_buffer_metadata(
        self,
    ) -> Tuple[
        Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, torch.dtype]]
    ]:
        module_shapes: Dict[str, Dict[str, List[int]]] = {}
        tensor_dtypes: Dict[str, Dict[str, torch.dtype]] = {}
        for module_type, tensors in self.module_metadata.items():
            if not tensors:
                raise ValueError(
                    f"V4-Flash: no tensors discovered for module_type={module_type!r}; "
                    f"checkpoint metadata may be incomplete or rank shards mismatched"
                )
            module_shapes[module_type] = {}
            tensor_dtypes[module_type] = {}
            for tensor_key, meta in sorted(tensors.items()):
                module_shapes[module_type][tensor_key] = list(meta["shape"])
                tensor_dtypes[module_type][tensor_key] = resolve_torch_dtype(
                    str(meta["dtype"])
                )
        return module_shapes, tensor_dtypes

    def _set_batching_and_buffer_config(self):
        reserved_length = self.engine_config.KV_Storage_Config.reserved_length
        world_size = max(1, int(self.world_size))
        experts_per_rank = self.model_config.num_local_experts // world_size
        offloading_ratio = float(self.engine_config.EP_Config.offloading_ratio)
        offloading_enabled = (
            self.engine_config.EP_Config.enable_offloading and offloading_ratio > 0.0
        )
        if offloading_enabled:
            num_local_expert_per_layer = int(experts_per_rank * (1.0 - offloading_ratio))
            num_local_expert_per_layer = max(
                0, min(experts_per_rank, num_local_expert_per_layer)
            )
            decode_routed_expert_buffers = (
                experts_per_rank - num_local_expert_per_layer + 2
            )
            logging.info(
                "DeepSeek-V4-Flash EP offloading enabled: %s persistent, %s offloaded, %s decode buffers",
                num_local_expert_per_layer,
                experts_per_rank - num_local_expert_per_layer,
                decode_routed_expert_buffers,
            )
        else:
            num_local_expert_per_layer = experts_per_rank
            decode_routed_expert_buffers = max(experts_per_rank, 1)

        self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 8
        self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 8
        self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 4096
        self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 128
        self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size = 128
        self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 2048

        prefill_buf = {"routed_expert": experts_per_rank, "shared_expert": 1}
        decode_buf = {
            "routed_expert": decode_routed_expert_buffers,
            "shared_expert": 1,
        }
        for mt in self.module_metadata:
            if mt.startswith("attn"):
                prefill_buf[mt] = 1
                decode_buf[mt] = 1
        self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = (
            prefill_buf
        )
        self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = (
            decode_buf
        )
        self.engine_config.GPU_Buffer_Config.num_k_buffer = 6
        self.engine_config.GPU_Buffer_Config.num_v_buffer = 0
        self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
            * reserved_length
        )
        self.engine_config.EP_Config.enable = True
        self.engine_config.EP_Config.num_local_expert_per_layer = num_local_expert_per_layer

    def Init(self, weights_storage):
        try:
            torch.cuda.set_device(self.local_rank)
            if self.global_rank == 0:
                logging.info("Engine config: %s", self.engine_config)
            self.core_engine = core_engine(
                self.engine_config,
                self.model_config,
                weights_storage,
            )
            logging.info("Core engine created")
            self.core_engine.Init()
            logging.info("Core engine initialized")
            self._verify_buffer_contract(weights_storage)
        except Exception:
            logging.exception(
                "Failed to initialize DeepSeek-V4-Flash core engine"
            )
            raise
        return (
            self.core_engine,
            self.engine_config,
            self.model_config,
            self.loaded_model_config,
        )

    def _verify_buffer_contract(self, weights_storage) -> None:
        declared = {
            module_type: {
                tensor_key: tuple(shape)
                for tensor_key, shape in tensors.items()
            }
            for module_type, tensors in self.engine_config.GPU_Buffer_Config.module_shapes.items()
        }
        diffs = diff_shapes(self.module_metadata, declared)
        if diffs:
            preview = "\n".join(
                f"  {mt}.{tk}: {msg}" for mt, tk, msg in diffs[:20]
            )
            raise ValueError(
                f"V4-Flash buffer contract mismatch ({len(diffs)} entries):\n{preview}"
            )

        for module_type, tensors in self.module_metadata.items():
            for tensor_key, meta in tensors.items():
                declared_dtype = (
                    self.engine_config.GPU_Buffer_Config.tensor_dtypes.get(
                        module_type, {}
                    ).get(tensor_key)
                )
                expected_dtype = resolve_torch_dtype(str(meta["dtype"]))
                if declared_dtype is None:
                    raise ValueError(
                        f"V4-Flash buffer contract: missing tensor_dtype for "
                        f"{module_type}.{tensor_key} (expected {expected_dtype})"
                    )
                if declared_dtype != expected_dtype:
                    raise ValueError(
                        f"V4-Flash buffer contract: dtype mismatch for "
                        f"{module_type}.{tensor_key}: declared={declared_dtype} "
                        f"actual_ckpt={expected_dtype}"
                    )

        if self.global_rank == 0:
            logging.info(
                "V4-Flash buffer contract verified: all module_shapes + tensor_dtypes "
                "match checkpoint metadata"
            )

    def get_configs(self):
        return self.loaded_model_config, self.engine_config, self.model_config

    def parse_json_config(self, json_file_path):
        parse_config_from_json(
            json_file_path, self.engine_config, self.model_config
        )
