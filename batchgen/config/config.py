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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
import logging


# Basic configuration section
@dataclass
class BasicConfig:
    log_level: str = "info"
    weight_dtype: Optional[str] = None
    weight_dtype_torch: Optional[Any] = None
    kv_dtype: Optional[str] = None
    kv_dtype_torch: Optional[Any] = None
    attention_dtype: Optional[str] = None
    activation_dtype: Optional[str] = None
    activation_dtype_torch: Optional[Any] = None
    device: Optional[str] = None
    device_torch: Optional[Any] = None
    attn_mode: int = 1
    module_types: Optional[List[str]] = None
    num_threads: Optional[int] = 0
    padding_length: Optional[int] = None
    max_decoding_length: Optional[int] = None
    num_queries: Optional[int] = None
    rank: Optional[int] = None
    world_size: Optional[int] = None
    gpu_arch: Optional[str] = None

    @staticmethod
    def _str_to_torch_dtype(dtype_str: str) -> Any:
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
        }
        return dtype_map.get(dtype_str, None)
    
    def __str__(self):
        return (
            f"BasicConfig:\n"
            f"  log_level: {self.log_level}\n"
            f"  weight_dtype: {self.weight_dtype}\n"
            f"  kv_dtype: {self.kv_dtype}\n"
            f"  activation_dtype: {self.activation_dtype}\n"
            f"  device: {self.device}\n"
            f"  attn_mode: {self.attn_mode}\n"
            f"  module_types: {self.module_types}\n"
            f"  num_threads: {self.num_threads}\n"
            f"  padding_length: {self.padding_length}\n"
            f"  max_decoding_length: {self.max_decoding_length}\n"
            f"  num_queries: {self.num_queries}\n"
            f"  rank: {self.rank}\n"
            f"  world_size: {self.world_size}\n"
            f"  gpu_arch: {self.gpu_arch}"
        )


# Module batching configuration section
@dataclass
class ModuleBatchingConfig:
    global_batch_size: Optional[int] = 0
    # Token-based prefill config (for prepack mode, always recommended)
    prefill_micro_batch_token_cap: int = 120_000  # Max tokens per prefill micro-batch
    prepack_row_capacity: Optional[int] = None  # Token budget per packed row (None = no limit)
    # Sequence-count based prefill config (for non-prepack mode)
    attn_prefill_micro_batch_size: Optional[int] = 0
    MoE_prefill_micro_batch_size: Optional[int] = 0
    expert_prefill_batch_size_upper_bound: Optional[int] = 0
    attn_decoding_micro_batch_size: Optional[int] = 0
    MoE_decoding_micro_batch_size: Optional[int] = 0
    expert_decoding_batch_size_upper_bound: Optional[int] = 0

    def __str__(self):
        return (
            f"ModuleBatchingConfig:\n"
            f"  global_batch_size: {self.global_batch_size}\n"
            f"  prefill_micro_batch_token_cap: {self.prefill_micro_batch_token_cap}\n"
            f"  prepack_row_capacity: {self.prepack_row_capacity}\n"
            f"  attn_prefill_micro_batch_size: {self.attn_prefill_micro_batch_size}\n"
            f"  MoE_prefill_micro_batch_size: {self.MoE_prefill_micro_batch_size}\n"
            f"  expert_prefill_batch_size_upper_bound: {self.expert_prefill_batch_size_upper_bound}\n"
            f"  attn_decoding_micro_batch_size: {self.attn_decoding_micro_batch_size}\n"
            f"  MoE_decoding_micro_batch_size: {self.MoE_decoding_micro_batch_size}\n"
            f"  expert_decoding_batch_size_upper_bound: {self.expert_decoding_batch_size_upper_bound}"
        )


# GPU buffer configuration section
@dataclass
class GPUBufferConfig:
    num_prefill_module_buffer: Union[int, Dict[str, int]] = 0
    num_decoding_module_buffer: Union[int, Dict[str, int]] = 0
    num_k_buffer: int = 0
    num_v_buffer: int = 0
    kv_buffer_num_tokens: int = 0
    module_shapes: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return (
            f"GPUBufferConfig:\n"
            f"  num_prefill_module_buffer: {self.num_prefill_module_buffer}\n"
            f"  num_decoding_module_buffer: {self.num_decoding_module_buffer}\n"
            f"  num_k_buffer: {self.num_k_buffer}\n"
            f"  num_v_buffer: {self.num_v_buffer}\n"
            f"  kv_buffer_num_tokens: {self.kv_buffer_num_tokens}\n"
            f"  module_shapes: {self.module_shapes}"
        )


# KV storage configuration section
@dataclass
class KVStorageConfig:
    num_host_slots: int = 0
    reserved_length: int = 0
    slot_byte_size: int = 0
    storage_byte_size: int = 0


# Expert parallelism configuration section
@dataclass
class EPConfig:
    enable: bool = False
    num_local_expert_per_layer: int = 0

    def __str__(self):
        return (
            f"EPConfig:\n"
            f"  enable: {self.enable}\n"
            f"  num_local_expert_per_layer: {self.num_local_expert_per_layer}"
        )

@dataclass
class HostPagedKVConfig:
    shm_name: str = ""  # Name of the shared memory segment.
    total_byte_size: int = 0 # Total byte size of the shared memory need to be allocated by the host paged kv manager instance.
    num_layers: int = 0
    num_pages_per_layer: int = 0
    page_size: int = 64  # Number of tokens per page. 
    num_k_heads: int = 0
    k_head_dim: int = 0
    num_v_heads: int = 0 # Zero for MLA.
    v_head_dim: int = 0
    kv_dtype: str = "bfloat16" # "bfloat16 or float8_e4m3fn"

@dataclass
class DevicePagedKVConfig:
    num_layers: int = 0
    num_pages_per_layer: int = 0
    page_size: int = 64  # Number of tokens per page. 
    num_k_heads: int = 0
    k_head_dim: int = 0
    num_v_heads: int = 0 # Zero for MLA.
    v_head_dim: int = 0
    kv_dtype: str = "bfloat16" # "bfloat16 or float8_e4m3fn"


# Main engine configuration class
@dataclass
class EngineConfig:
    Basic_Config: BasicConfig = field(default_factory=BasicConfig)
    Module_Batching_Config: ModuleBatchingConfig = field(default_factory=ModuleBatchingConfig)
    GPU_Buffer_Config: GPUBufferConfig = field(default_factory=GPUBufferConfig)
    KV_Storage_Config: KVStorageConfig = field(default_factory=KVStorageConfig)
    EP_Config: EPConfig = field(default_factory=EPConfig)
    Host_Paged_KV_Config: HostPagedKVConfig = field(default_factory=HostPagedKVConfig)
    Device_Paged_KV_Config: DevicePagedKVConfig = field(default_factory=DevicePagedKVConfig)

    def __str__(self) -> str:
        sections = [
            f"Module_Batching_Config:\n{self.Module_Batching_Config}",
            f"Basic_Config:\n{self.Basic_Config}",
            f"GPU_Buffer_Config:\n{self.GPU_Buffer_Config}",
            # f"KV_Storage_Config:\n{self.KV_Storage_Config}",
            f"EP_Config:\n{self.EP_Config}"
        ]
        return "EngineConfig:\n  " + "\n  ".join(sections)


class ModelConfig:
    def __init__(self):
        self.model_type = None
        self.num_hidden_layers = None
        self.num_local_experts = None
        self.num_attention_heads = None
        self.num_key_value_heads = None
        self.hidden_size = None
        self.intermediate_size = None
        self.head_dim = None

    def __str__(self):
        return f"""ModelConfig:
			model_type: {self.model_type}
			num_hidden_layers: {self.num_hidden_layers}
			num_local_experts: {self.num_local_experts}
			num_attention_heads: {self.num_attention_heads}
			num_key_value_heads: {self.num_key_value_heads}
			hidden_size: {self.hidden_size}
			intermediate_size: {self.intermediate_size}
			head_dim: {self.head_dim}"""
