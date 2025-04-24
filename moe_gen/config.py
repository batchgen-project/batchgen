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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __setitem__(self, key, value):
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value

    def __setattr__(self, key, value):
        super(AttrDict, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __str__(self):
        return "\n".join(
            f"    {k}: {v}"
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        )


class EngineConfig:
    def __init__(self):
        self.Basic_Config = AttrDict(
            {
                "log_level": "info",
                "device": None,
                "torch_dtype": None,
                "dtype_str": None,
                "device_torch": None,
                "attn_mode": 1,
                "module_types": None,
                "num_threads": None,
                "padding_length": None,
                "max_decoding_length": None,
                "num_queries": None,
                "kv_dtype": None,
            }
        )

        self.Module_Batching_Config = AttrDict(
            {
                # "prefill_micro_batch_size": None,
                "global_batch_size": None,
                "attn_prefill_micro_batch_size": None,
                "MoE_prefill_micro_batch_size": None,
                "expert_prefill_batch_size_upper_bound": None,
                "attn_decoding_micro_batch_size": None,
                "MoE_decoding_micro_batch_size": None,
                "expert_decoding_batch_size_upper_bound": None,
            }
        )

        self.GPU_Buffer_Config = AttrDict(
            {
                "num_prefill_module_buffer": None,
                "num_decoding_module_buffer": None,
                "num_k_buffer": 0,
                "num_v_buffer": 0,
                "kv_buffer_num_tokens": 64 * 576,
                "module_shapes": {},
            }
        )

        self.KV_Storage_Config = AttrDict(
            {
                "num_host_slots": 200,
                "reserved_length": 576,
                "slot_byte_size": 576 * 1024 * 2,
                "storage_byte_size": 200 * 576 * 1024 * 2 * 32,
            }
        )

    def __str__(self):
        return (
            "EngineConfig:\n"
            f"  Module_Batching_Config:\n{self.Module_Batching_Config}\n"
            f"  Basic_Config:\n{self.Basic_Config}\n"
            f"  GPU_Buffer_Config:\n{self.GPU_Buffer_Config}\n"
            f"  KV_Storage_Config:\n{self.KV_Storage_Config}"
        )


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
