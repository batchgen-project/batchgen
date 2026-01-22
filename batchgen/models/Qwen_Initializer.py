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

import logging
import math
import time
import types
from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen2MoeForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    _flash_supports_window_size,
    apply_rotary_pos_emb,
    repeat_kv,
)

from batchgen.core_engine import batchgen as core_engine
from batchgen.engine import EngineConfig, ModelConfig
# Use new wrapper system - Attn_Wrapper/Expert_Wrapper are aliases for backward compatibility
from batchgen.models.wrappers import AttnWrapperBase, ExpertWrapperBase
Attn_Wrapper = AttnWrapperBase
Expert_Wrapper = ExpertWrapperBase


def _QKV_Proj(self, hidden_states, position_ids, kv_seq_len):
    """
    Project the hidden states to query, key, value states.
    """
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    # logging.debug(f"Entering rotary embedding")
    # print(f"Python Query states shape: {query_states.shape}", flush=True)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # logging.debug(f"Exiting rotary embedding")

    return query_states, key_states, value_states


def _attn_QKT_softmax(self, query_states, key_states, attention_mask):
    """
    - Compute the attention weights.
    - Apply the causal mask.
    - Apply the softmax.
    """
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    return attn_weights


def _attn_AV(self, attn_weights, value_states):
    """
    - A @ V
    """
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output


def _O_Proj(self, attn_output):
    """
    - Output projection.
    """
    bsz, _, q_len, _ = attn_output.size()
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output


def update_casual_mask(attention_mask):
    dtype = torch.bfloat16
    min_dtype = torch.finfo(dtype).min
    device = attention_mask.device
    target_length = attention_mask.size(-1)
    sequence_length = 1
    cache_position = torch.arange(
        target_length - 1, target_length, device=device
    )
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(
        target_length, device=device
    ) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(
        attention_mask.shape[0], 1, -1, -1
    )
    if attention_mask is not None:
        causal_mask = (
            causal_mask.clone()
        )  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length]
            + attention_mask[:, None, None, :]
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)
    return causal_mask


def vanilla_attn(
    self,
    hidden_states,
    past_key_states,
    past_value_states,
    attention_mask,
    position_ids,
):
    # if self.layer_idx == 0:
    # 	torch.save(hidden_states, f"test_decoding_input_0.pt")
    # 	torch.save(attention_mask, f"test_decoding_attention_mask_0.pt")
    # 	torch.save(position_ids, f"test_decoding_position_ids_0.pt")
    # if self.layer_idx == 0:
    # 	torch.save(past_key_states, f"test_decoding_past_key_states_0.pt")
    # 	torch.save(past_value_states, f"test_decoding_past_value_states_0.pt")

    # attention_mask = update_casual_mask(attention_mask)
    # logging.info(f"attention mask shape: {attention_mask.shape}")
    # logging.info(f"example attention_mask: {attention_mask[0]}")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    # if self.layer_idx == 0:
    # 	torch.save(query_states, f"test_decoding_query_states_0.pt")
    # 	torch.save(new_key_states, f"test_decoding_new_key_states_0.pt")
    # 	torch.save(new_value_states, f"test_decoding_new_value_states_0.pt")

    kv_seq_len = attention_mask.shape[-1]
    # logging.debug(f"Entering rotary embedding, kv_seq_len: {kv_seq_len}")
    rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    new_key_states = key_states.clone()
    new_value_states = value_states.clone()

    # if self.layer_idx == 0:
    # 	torch.save(query_states, f"test_decoding_query_states_rotary_0.pt")
    # 	torch.save(new_key_states, f"test_decoding_new_key_states_rotary_0.pt")

    key_states = torch.cat([past_key_states, key_states], dim=2)
    value_states = torch.cat([past_value_states, value_states], dim=2)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # if self.layer_idx == 0:
    # 	torch.save(key_states, f"test_decoding_key_states_repeated_0.pt")
    # 	torch.save(value_states, f"test_decoding_value_states_repeated_0.pt")
    # 	exit()

    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    # if self.layer_idx == 0:
    # 	torch.save(attn_weights, f"test_decoding_attn_weights_0.pt")

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # if self.layer_idx == 0:
    # 	torch.save(attn_weights, f"test_decoding_attn_weights_masked_0.pt")

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    # if self.layer_idx == 0:
    # 	torch.save(attn_output, f"test_decoding_attn_output_0.pt")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    # if self.layer_idx == 0:
    # 	torch.save(attn_output, f"test_decoding_attn_output_proj_0.pt")
    # 	exit()

    return attn_output, new_key_states, new_value_states


def decoding_attn(
    self,
    hidden_states: torch.Tensor,
    past_key_states: torch.Tensor,
    past_value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = attention_mask.shape[-1] - 1
    # Because the input can be padded, the absolute sequence length depends on the max position id.
    rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    new_key_states = key_states.clone()
    new_value_states = value_states.clone()

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, "sliding_window", None) is not None
        and kv_seq_len > self.config.sliding_window
        and self.config.use_sliding_window
    )

    key_states = torch.cat([past_key_states, key_states], dim=2)
    value_states = torch.cat([past_value_states, value_states], dim=2)

    # if not _flash_supports_window_size:
    # 	logger.warning_once(
    # 		"The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
    # 		" make sure to upgrade flash-attn library."
    # 	)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        # logger.warning_once(
        # 	f"The input hidden states seems to be silently casted in float32, this might be related to"
        # 	f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
        # 	f" {target_dtype}."
        # )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        use_sliding_windows=use_sliding_windows,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    # return attn_output, attn_weights, past_key_value
    return attn_output, new_key_states, new_value_states


class Qwen_Initializer:
    def __init__(self, model: Qwen2MoeForCausalLM, engine_config: EngineConfig):
        self.model = model
        self.engine_config = engine_config
        # TODO:
        self._test_engine_config()
        assert isinstance(
            self.model, Qwen2MoeForCausalLM
        ), "Model must be of type Qwen2MoeForCausalLM"
        self.model_config = self._parse_model_config()
        self.attn_name_to_tensor_mapping = []
        self.expert_name_to_tensor_mapping = []
        self.shared_expert_name_to_tensor_mapping = []

    def _test_engine_config(self):
        self.engine_config.Module_Batching_Config.prefill_micro_batch_size = 200
        self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 64
        self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 10000
        self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 50000

        self.engine_config.Basic_Config.log_level = "info"
        self.engine_config.Basic_Config.device = 7
        self.engine_config.Basic_Config.torch_dtype = torch.bfloat16
        self.engine_config.Basic_Config.dtype_str = "bfloat16"
        self.engine_config.Basic_Config.device_torch = torch.device("cuda:7")
        self.engine_config.Basic_Config.attn_mode = 1
        self.engine_config.Basic_Config.module_types = [
            "attn",
            "routed_expert",
            "shared_expert",
        ]

        self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
            "attn": 1,
            "routed_expert": 32,
            "shared_expert": 1,
        }
        self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
            "attn": 1,
            "routed_expert": 32,
            "shared_expert": 1,
        }
        self.engine_config.GPU_Buffer_Config.num_kv_buffer = 3
        self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
            * 544
        )
        self.engine_config.GPU_Buffer_Config.module_shapes = {}

        self.engine_config.KV_Storage_Config.num_host_slots = 100
        self.engine_config.KV_Storage_Config.reserved_length = 544
        self.engine_config.KV_Storage_Config.slot_byte_size = 544 * 512 * 2 * 2
        self.engine_config.KV_Storage_Config.storage_byte_size = (
            100 * 544 * 512 * 2 * 28 * 2
        )

    def _parse_model_config(self):
        model_config = ModelConfig()
        if "Qwen/Qwen2-57B-A14B" in self.model.config._name_or_path:
            model_config.model_type = "Qwen2"
            model_config.num_hidden_layers = 28
            model_config.num_local_experts = 64
            model_config.num_attention_heads = 28
            model_config.num_key_value_heads = 4
            model_config.hidden_size = 3584
            model_config.intermediate_size = 18944
            model_config.head_dim = 128

        else:
            raise ValueError("Model not supported")

        return model_config

    def Init(self):
        self._config_pinned_model_weights()
        self.core_engine = core_engine(self.engine_config, self.model_config)
        self.core_engine.Init(self.model_weights, self.weight_copy_task)
        self._prepare_model_skeleton()  # PyTorch model skeleton with weights removed.
        self._config_attn_module()
        self._config_expert_module()
        self._config_lm_head_hook()

        self.model.to(self.engine_config.Basic_Config.device_torch)
        self.model.eval()
        print(self.model, flush=True)

        return (
            self.core_engine,
            self.model,
            self.engine_config,
            self.model_config,
        )

    def _lm_head_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self):
        self.model.lm_head.register_forward_pre_hook(
            self._lm_head_forward_pre_hook
        )

    def _config_pinned_model_weights(self):
        """
        pin the tensor and pass to core_engine.
        """
        logging.info("Start configure pinned memory for model weights in host")
        logging.info("This process may take a few minutes.")
        start_time = time.time()
        for layer_idx in range(len(self.model.model.layers)):
            layer = self.model.model.layers[layer_idx]
            pinned_attn = {
                k: v for k, v in layer.self_attn.state_dict().items()
            }
            # pinned_attn = {k:v.clone() for k, v in layer.self_attn.state_dict().items()}
            self.attn_name_to_tensor_mapping.append(pinned_attn)
            if layer_idx == 0:
                self.attn_weights_shape = {
                    k: v.shape for k, v in pinned_attn.items()
                }

            shared_expert = {
                k: v for k, v in layer.mlp.shared_expert.state_dict().items()
            }
            self.shared_expert_name_to_tensor_mapping.append(shared_expert)
            if layer_idx == 0:
                self.shared_expert_weights_shape = {
                    k: v.shape for k, v in shared_expert.items()
                }

            layer_expert = []
            for expert_idx in range(len(layer.mlp.experts)):
                pinned_expert = {
                    k: v
                    for k, v in layer.mlp.experts[expert_idx]
                    .state_dict()
                    .items()
                }
                layer_expert.append(pinned_expert)
                if layer_idx == 0 and expert_idx == 0:
                    self.expert_weights_shape = {
                        k: v.shape for k, v in pinned_expert.items()
                    }
            self.expert_name_to_tensor_mapping.append(layer_expert)
        logging.info(
            "Finished configure pinned memory for model weights in host. Spend time: %.3f",
            time.time() - start_time,
        )
        self.engine_config.GPU_Buffer_Config.module_shapes["attn"] = (
            self.attn_weights_shape
        )
        self.engine_config.GPU_Buffer_Config.module_shapes["routed_expert"] = (
            self.expert_weights_shape
        )
        self.engine_config.GPU_Buffer_Config.module_shapes["shared_expert"] = (
            self.shared_expert_weights_shape
        )
        # model_weights = {
        # 	"attn": {"attn_" + str(i): self.attn_name_to_tensor_mapping[i] for i in range(len(self.attn_name_to_tensor_mapping))},
        # 	"expert": {"expert_" + str(i) + "_" + str(j): self.expert_name_to_tensor_mapping[i][j] for i in range(len(self.expert_name_to_tensor_mapping)) for j in range(len(self.expert_name_to_tensor_mapping[i]))}
        # }
        self.model_weights = {}
        for i in range(len(self.attn_name_to_tensor_mapping)):
            self.model_weights["attn_" + str(i)] = (
                self.attn_name_to_tensor_mapping[i]
            )
        for i in range(len(self.shared_expert_name_to_tensor_mapping)):
            self.model_weights["shared_expert_" + str(i)] = (
                self.shared_expert_name_to_tensor_mapping[i]
            )
        for i in range(len(self.expert_name_to_tensor_mapping)):
            for j in range(len(self.expert_name_to_tensor_mapping[i])):
                self.model_weights["routed_expert_" + str(i) + "_" + str(j)] = (
                    self.expert_name_to_tensor_mapping[i][j]
                )

        self.weight_copy_task = {}
        self.weight_copy_task["attn"] = [
            k for k in self.model_weights.keys() if "attn" in k
        ]
        self.weight_copy_task["routed_expert"] = [
            k for k in self.model_weights.keys() if "routed_expert" in k
        ]
        self.weight_copy_task["shared_expert"] = [
            k for k in self.model_weights.keys() if "shared_expert" in k
        ]

    def _prepare_model_skeleton(self):
        # for name, param in self.model.named_parameters():
        # 	if (("self_attn" in name or "expert" in name) and ("shared_expert_gate" not in name)):
        # 		# logging.debug(f"Zeroing out {name}")
        # 		param.data = torch.tensor(0.0,dtype=param.dtype)
        for layer_idx in range(self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
            for name, param in layer.self_attn.named_parameters():
                param.data = torch.tensor(
                    0.0, dtype=param.data.dtype, device=param.data.device
                )
            for name, param in layer.mlp.shared_expert.named_parameters():
                param.data = torch.tensor(
                    0.0, dtype=param.data.dtype, device=param.data.device
                )

            for expert_idx in range(self.model_config.num_local_experts):
                for name, param in layer.mlp.experts[
                    expert_idx
                ].named_parameters():
                    param.data = torch.tensor(
                        0.0, dtype=param.data.dtype, device=param.data.device
                    )

        model_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) // (1024**3)
        logging.info(f"Model Skeleton Size: {model_size} GB")

    def _config_attn_module(self):
        """
        - Configure the wrapper.
        """
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            setattr(
                attn_module,
                "QKV_Proj",
                types.MethodType(_QKV_Proj, attn_module),
            )
            setattr(
                attn_module,
                "attn_QKT_softmax",
                types.MethodType(_attn_QKT_softmax, attn_module),
            )
            setattr(
                attn_module, "attn_AV", types.MethodType(_attn_AV, attn_module)
            )
            setattr(
                attn_module, "O_Proj", types.MethodType(_O_Proj, attn_module)
            )
            # ssetattr(attn_module, "vanilla_attn", types.MethodType(vanilla_attn, attn_module))
            setattr(
                attn_module,
                "decoding_attn",
                types.MethodType(decoding_attn, attn_module),
            )

            attn_wrapper_instance = Attn_Wrapper(
                attn_module,
                layer_idx,
                self.core_engine,
                self.engine_config,
                self.model_config,
            )
            self.model.model.layers[layer_idx].self_attn = attn_wrapper_instance

    def _config_expert_module(self):
        """
        Replace expert module with the wrapper.
        """
        for layer_idx in range(len(self.model.model.layers)):
            layer = self.model.model.layers[layer_idx]
            layer.mlp.shared_expert = Expert_Wrapper(
                layer.mlp.shared_expert,
                layer_idx,
                -1,
                self.core_engine,
                self.engine_config,
                self.model_config,
            )
            for expert_idx in range(len(layer.mlp.experts)):
                layer.mlp.experts[expert_idx] = Expert_Wrapper(
                    layer.mlp.experts[expert_idx],
                    layer_idx,
                    expert_idx,
                    self.core_engine,
                    self.engine_config,
                    self.model_config,
                )
