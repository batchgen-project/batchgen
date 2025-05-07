# ---------------------------------------------------------------------------- #
#  MoE-Gen                                                                      #
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


# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self

#     def __setitem__(self, key, value):
#         super(AttrDict, self).__setitem__(key, value)
#         self.__dict__[key] = value

#     def __setattr__(self, key, value):
#         super(AttrDict, self).__setitem__(key, value)
#         super().__setattr__(key, value)

#     def __str__(self):
#         return "\n".join(
#             f"    {k}: {v}"
#             for k, v in self.__dict__.items()
#             if not k.startswith("_")
#         )


# class EngineConfig:
#     def __init__(self):
#         self.Basic_Config = AttrDict(
#             {
#                 "log_level": "info",
#                 "device": None,
#                 "torch_dtype": None,
#                 "dtype_str": None,
#                 "device_torch": None,
#                 "attn_mode": 1,
#                 "module_types": None,
#                 "num_threads": None,
#                 "padding_length": None,
#                 "max_decoding_length": None,
#                 "num_queries": None,
#                 "kv_dtype": None,
#             }
#         )

#         self.Module_Batching_Config = AttrDict(
#             {
#                 # "prefill_micro_batch_size": None,
#                 "global_batch_size": None,
#                 "attn_prefill_micro_batch_size": None,
#                 "MoE_prefill_micro_batch_size": None,
#                 "expert_prefill_batch_size_upper_bound": None,
#                 "attn_decoding_micro_batch_size": None,
#                 "MoE_decoding_micro_batch_size": None,
#                 "expert_decoding_batch_size_upper_bound": None,
#             }
#         )

#         self.GPU_Buffer_Config = AttrDict(
#             {
#                 "num_prefill_module_buffer": None,
#                 "num_decoding_module_buffer": None,
#                 "num_k_buffer": 0,
#                 "num_v_buffer": 0,
#                 "kv_buffer_num_tokens": 64 * 576,
#                 "module_shapes": {},
#             }
#         )

#         self.KV_Storage_Config = AttrDict(
#             {
#                 "num_host_slots": 200,
#                 "reserved_length": 576,
#                 "slot_byte_size": 576 * 1024 * 2,
#                 "storage_byte_size": 200 * 576 * 1024 * 2 * 32,
#             }
#         )

#     def __str__(self):
#         return (
#             "EngineConfig:\n"
#             f"  Module_Batching_Config:\n{self.Module_Batching_Config}\n"
#             f"  Basic_Config:\n{self.Basic_Config}\n"
#             f"  GPU_Buffer_Config:\n{self.GPU_Buffer_Config}\n"
#             f"  KV_Storage_Config:\n{self.KV_Storage_Config}"
#         )

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import json
import yaml
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
    activation_dtype: Optional[str] = None
    activation_dtype_torch: Optional[Any] = None
    # torch_dtype: Optional[Any] = None
    # dtype_str: Optional[str] = None
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


# Module batching configuration section
@dataclass
class ModuleBatchingConfig:
    global_batch_size: Optional[int] = 0
    attn_prefill_micro_batch_size: Optional[int] = 0
    MoE_prefill_micro_batch_size: Optional[int] = 0
    expert_prefill_batch_size_upper_bound: Optional[int] = 0
    attn_decoding_micro_batch_size: Optional[int] = 0
    MoE_decoding_micro_batch_size: Optional[int] = 0
    expert_decoding_batch_size_upper_bound: Optional[int] = 0


# GPU buffer configuration section
@dataclass
class GPUBufferConfig:
    num_prefill_module_buffer: Union[int, Dict[str, int]] = 0
    num_decoding_module_buffer: Union[int, Dict[str, int]] = 0
    num_k_buffer: int = 0
    num_v_buffer: int = 0
    kv_buffer_num_tokens: int = 0
    module_shapes: Dict[str, Any] = field(default_factory=dict)


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


# Main engine configuration class
@dataclass
class EngineConfig:
    Basic_Config: BasicConfig = field(default_factory=BasicConfig)
    Module_Batching_Config: ModuleBatchingConfig = field(default_factory=ModuleBatchingConfig)
    GPU_Buffer_Config: GPUBufferConfig = field(default_factory=GPUBufferConfig)
    KV_Storage_Config: KVStorageConfig = field(default_factory=KVStorageConfig)
    EP_Config: EPConfig = field(default_factory=EPConfig)

#     @classmethod
#     def from_dict(cls, config_dict: Dict[str, Any]) -> 'EngineConfig':
#         """Create EngineConfig from a dictionary"""
#         return cls(
#             Basic_Config=BasicConfig(**config_dict.get('Basic_Config', {})),
#             Module_Batching_Config=ModuleBatchingConfig(**config_dict.get('Module_Batching_Config', {})),
#             GPU_Buffer_Config=GPUBufferConfig(**config_dict.get('GPU_Buffer_Config', {})),
#             KV_Storage_Config=KVStorageConfig(**config_dict.get('KV_Storage_Config', {})),
#             EP_Config=EPConfig(**config_dict.get('EP_Config', {}))
#         )

#     @classmethod
#     def from_json(cls, json_path: Union[str, Path]) -> 'EngineConfig':
#         """Load configuration from JSON file"""
#         with open(json_path, 'r') as f:
#             config_dict = json.load(f)
#         return cls.from_dict(config_dict)

#     @classmethod
#     def from_yaml(cls, yaml_path: Union[str, Path]) -> 'EngineConfig':
#         """Load configuration from YAML file"""
#         with open(yaml_path, 'r') as f:
#             config_dict = yaml.safe_load(f)
#         return cls.from_dict(config_dict)

#     def to_dict(self) -> Dict[str, Any]:
#         """Convert config to dictionary for serialization"""
#         # Use asdict but handle torch dtypes which aren't serializable
#         config_dict = asdict(self)
        
#         # Handle torch dtypes
#         if self.Basic_Config.torch_dtype is not None:
#             if isinstance(self.Basic_Config.torch_dtype, torch.dtype):
#                 for torch_type_name in ["float16", "float32", "bfloat16", "float8_e4m3fn", "float8_e5m2"]:
#                     if getattr(torch, torch_type_name, None) == self.Basic_Config.torch_dtype:
#                         config_dict["Basic_Config"]["torch_dtype"] = torch_type_name
#                         break
#                 else:
#                     config_dict["Basic_Config"]["torch_dtype"] = str(self.Basic_Config.torch_dtype)
        
#         return config_dict

#     def to_json(self, json_path: Union[str, Path], indent: int = 2) -> None:
#         """Save configuration to JSON file"""
#         with open(json_path, 'w') as f:
#             json.dump(self.to_dict(), f, indent=indent)
#         logging.info(f"Configuration saved to {json_path}")

#     def to_yaml(self, yaml_path: Union[str, Path]) -> None:
#         """Save configuration to YAML file"""
#         with open(yaml_path, 'w') as f:
#             yaml.dump(self.to_dict(), f, default_flow_style=False)
#         logging.info(f"Configuration saved to {yaml_path}")

#     def merge_with(self, other: 'EngineConfig') -> 'EngineConfig':
#         """Merge with another config, with other taking precedence"""
#         result = deepcopy(self)
#         other_dict = other.to_dict()
        
#         # Merge each section
#         for section in ['Basic_Config', 'Module_Batching_Config', 'GPU_Buffer_Config', 
#                        'KV_Storage_Config', 'EP_Config']:
#             if section in other_dict:
#                 for key, value in other_dict[section].items():
#                     if value is not None:  # Only overwrite if value is not None
#                         getattr(result, section).__dict__[key] = value
        
#         return result

#     def __str__(self) -> str:
#         sections = [
#             f"Module_Batching_Config:\n{self.Module_Batching_Config}",
#             f"Basic_Config:\n{self.Basic_Config}",
#             f"GPU_Buffer_Config:\n{self.GPU_Buffer_Config}",
#             f"KV_Storage_Config:\n{self.KV_Storage_Config}",
#             f"EP_Config:\n{self.EP_Config}"
#         ]
#         return "EngineConfig:\n  " + "\n  ".join(sections)


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
