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
import os
import time
import types
from multiprocessing import Process
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

from batchgen.config.model_registry import load_config

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    # jit compile
    from core_engine import batchgen as core_engine
from safetensors.torch import load_file
from tqdm import trange

from batchgen.config import EngineConfig, ModelConfig
from batchgen.models.Wrapper import Attn_Wrapper, Expert_Wrapper


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


def decoding_attn(
    self,
    hidden_states,
    past_key_states,
    past_value_states,
    attention_mask,
    position_ids,
):
    # attention_mask = update_casual_mask(attention_mask)
    # Vanilla attn
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    new_key_states = self.k_proj(hidden_states)
    new_value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    new_key_states = new_key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    new_value_states = new_value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = attention_mask.shape[-1]
    # logging.debug(f"Entering rotary embedding, kv_seq_len: {kv_seq_len}")
    cos, sin = self.rotary_emb(new_value_states, seq_len=kv_seq_len)
    query_states, new_key_states = apply_rotary_pos_emb(
        query_states, new_key_states, cos, sin, position_ids
    )

    key_states = torch.cat([past_key_states, new_key_states], dim=2)
    value_states = torch.cat([past_value_states, new_value_states], dim=2)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

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

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, new_key_states, new_value_states


class Mixtral_Initializer:
    def __init__(
        self,
        huggingface_ckpt_name: str,
        hf_cache_dir: str,
        cache_dir: Optional[str],
        engine_config: EngineConfig,
        skeleton_state_dict: dict = None,
        shm_name: str = None,
        tensor_meta_shm_name: str = None,
        pt_ckpt_dir: str = None,
        host_kv_cache_size: int = 0,
    ):
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.hf_cache_dir = hf_cache_dir
        self.cache_dir = cache_dir
        self.engine_config = engine_config
        self.skeleton_state_dict = skeleton_state_dict
        self.shm_name = shm_name
        self.tensor_meta_shm_name = tensor_meta_shm_name
        self.host_kv_cache_size = host_kv_cache_size
        self.host_kv_cache_byte_size = host_kv_cache_size * 1024 * 1024 * 1024

        self.model = None
        # Use BatchGen's unified config system instead of HuggingFace AutoConfig
        self.loaded_model_config = load_config(huggingface_ckpt_name)

        self.model_config = self._parse_model_config()
        self._default_engine_config()
        self.state_dict_name_map = {}
        self.weight_copy_task = {}

        # Use provided pt_ckpt_dir or create default one
        if pt_ckpt_dir:
            self.pt_ckpt_dir = pt_ckpt_dir
        else:
            self.pt_ckpt_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.huggingface_ckpt_name,
            )

        if not os.path.exists(self.pt_ckpt_dir):
            os.makedirs(self.pt_ckpt_dir)

    def _default_engine_config(self):
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")

        if "8x7B" in self.loaded_model_config._name_or_path:
            self.engine_config.Basic_Config.log_level = "info"
            self.engine_config.Basic_Config.torch_dtype = torch.bfloat16
            self.engine_config.Basic_Config.dtype_str = "bfloat16"
            # Get mode from env var
            mode = os.getenv("ATTN_MODE", "1")
            mode = int(mode)
            self.engine_config.Basic_Config.attn_mode = mode
            self.engine_config.Basic_Config.module_types = [
                "attn",
                "routed_expert",
            ]
            self.engine_config.Basic_Config.num_threads = 16

            # Determine the number of host kv slots.
            self.engine_config.KV_Storage_Config.reserved_length = (
                self.engine_config.Basic_Config.padding_length
                + self.engine_config.Basic_Config.max_decoding_length
            )
            self.engine_config.KV_Storage_Config.slot_byte_size = (
                self.engine_config.KV_Storage_Config.reserved_length
                * self.model_config.head_dim
                * self.model_config.num_key_value_heads
                * 2
            )
            self.engine_config.KV_Storage_Config.num_host_slots = (
                self.host_kv_cache_byte_size
                // self.engine_config.KV_Storage_Config.slot_byte_size
                // self.model_config.num_hidden_layers
            )
            self.engine_config.KV_Storage_Config.storage_byte_size = (
                self.host_kv_cache_byte_size
            )
            # logging.info(
            #     f"KV storage byte size: {self.engine_config.KV_Storage_Config.storage_byte_size}"
            # )

            self.engine_config.Module_Batching_Config.global_batch_size = min(
                self.engine_config.KV_Storage_Config.num_host_slots,
                self.engine_config.Basic_Config.num_queries,
            )
            context_length = (
                self.engine_config.Basic_Config.max_decoding_length
                + self.engine_config.Basic_Config.padding_length
            )
            if context_length > 544:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = math.floor(
                    200 * 544 / context_length
                )
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = math.floor(
                    400 * 544 / context_length
                )
            else:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 200
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 400
            # logging.info(
            #     f"attn_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size}"
            # )
            # logging.info(
            #     f"MoE_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size}"
            # )
            self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 2048

            if context_length > 544:
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = math.floor(
                    400 * 544 / context_length
                )
            else:
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 400
            # logging.info(
            #     f"attn_decoding_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size}"
            # )
            self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size = self.engine_config.Module_Batching_Config.global_batch_size
            self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 2048

            # L40
            if total_memory >= 47 and total_memory < 49:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size *= 2
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size *= 2
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size *= 2
            if total_memory >= 78:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size *= 3
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size *= 3
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size *= 3

            self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
                "attn": 1,
                "routed_expert": 8,
            }
            self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
                "attn": 1,
                "routed_expert": 8,
            }

            self.engine_config.GPU_Buffer_Config.num_k_buffer = 3
            self.engine_config.GPU_Buffer_Config.num_v_buffer = 3
            self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
                * (
                    self.engine_config.Basic_Config.max_decoding_length
                    + self.engine_config.Basic_Config.padding_length
                )
            )
            self.engine_config.GPU_Buffer_Config.module_shapes = {
                "attn": {
                    "q_proj.weight": [4096, 4096],
                    "k_proj.weight": [1024, 4096],
                    "v_proj.weight": [1024, 4096],
                    "o_proj.weight": [4096, 4096],
                },
                "routed_expert": {
                    "w1.weight": [14336, 4096],
                    "w2.weight": [4096, 14336],
                    "w3.weight": [14336, 4096],
                },
            }

        elif "8x22B" in self.loaded_model_config._name_or_path:
            self.engine_config.Basic_Config.log_level = "info"
            self.engine_config.Basic_Config.torch_dtype = torch.bfloat16
            self.engine_config.Basic_Config.dtype_str = "bfloat16"
            mode = os.getenv("ATTN_MODE", "1")
            mode = int(mode)
            self.engine_config.Basic_Config.attn_mode = mode
            self.engine_config.Basic_Config.module_types = [
                "attn",
                "routed_expert",
            ]
            self.engine_config.Basic_Config.num_threads = 16
            # Determine the number of host kv slots.
            self.engine_config.KV_Storage_Config.reserved_length = (
                self.engine_config.Basic_Config.padding_length
                + self.engine_config.Basic_Config.max_decoding_length
            )
            self.engine_config.KV_Storage_Config.slot_byte_size = (
                self.engine_config.KV_Storage_Config.reserved_length
                * self.model_config.head_dim
                * self.model_config.num_key_value_heads
                * 2
            )
            self.engine_config.KV_Storage_Config.num_host_slots = (
                self.host_kv_cache_byte_size
                // self.engine_config.KV_Storage_Config.slot_byte_size
                // self.model_config.num_hidden_layers
            )
            self.engine_config.KV_Storage_Config.storage_byte_size = (
                self.host_kv_cache_byte_size
            )
            # logging.info(
            #     f"KV storage byte size: {self.engine_config.KV_Storage_Config.storage_byte_size}"
            # )

            self.engine_config.Module_Batching_Config.global_batch_size = min(
                self.engine_config.KV_Storage_Config.num_host_slots,
                self.engine_config.Basic_Config.num_queries,
            )
            context_length = (
                self.engine_config.Basic_Config.max_decoding_length
                + self.engine_config.Basic_Config.padding_length
            )
            if context_length > 544:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = math.floor(
                    20 * 544 / context_length
                )
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = math.floor(
                    60 * 544 / context_length
                )
            else:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 20
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 60
            # logging.info(
            #     f"attn_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size}"
            # )
            # logging.info(
            #     f"MoE_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size}"
            # )
            self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 2048

            if context_length > 544:
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = math.floor(
                    100 * 544 / context_length
                )
            else:
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 100
            # logging.info(
            #     f"attn_decoding_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size}"
            # )
            self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size = self.engine_config.Module_Batching_Config.global_batch_size
            self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 2048

            # L40
            if total_memory >= 47 and total_memory < 49:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size *= 2
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size *= 2
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size *= 2
            if total_memory >= 78:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size *= 3
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size *= 3
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size *= 3

            self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
                "attn": 1,
                "routed_expert": 8,
            }
            self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
                "attn": 1,
                "routed_expert": 8,
            }

            self.engine_config.GPU_Buffer_Config.num_k_buffer = 3
            self.engine_config.GPU_Buffer_Config.num_v_buffer = 3
            self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
                * (
                    self.engine_config.Basic_Config.max_decoding_length
                    + self.engine_config.Basic_Config.padding_length
                )
            )

            self.engine_config.GPU_Buffer_Config.module_shapes = {
                "attn": {
                    "q_proj.weight": [6144, 6144],
                    "k_proj.weight": [1024, 6144],
                    "v_proj.weight": [1024, 6144],
                    "o_proj.weight": [6144, 6144],
                },
                "routed_expert": {
                    "w1.weight": [16384, 6144],
                    "w2.weight": [6144, 16384],
                    "w3.weight": [16384, 6144],
                },
            }

        else:
            raise ValueError(
                f"Model {self.loaded_model_config._name_or_path} not supported yet"
            )

    def _parse_model_config(self):
        model_config = ModelConfig()
        if "8x7B" in self.loaded_model_config._name_or_path:
            model_config.model_type = "mixtral"
            model_config.num_hidden_layers = 32
            model_config.num_local_experts = 8
            model_config.num_attention_heads = 32
            model_config.num_key_value_heads = 8
            model_config.hidden_size = 4096
            model_config.intermediate_size = 14336
            model_config.head_dim = 128

        elif "8x22B" in self.loaded_model_config._name_or_path:
            model_config.model_type = "mixtral"
            model_config.num_hidden_layers = 56
            model_config.num_local_experts = 8
            model_config.num_attention_heads = 48
            model_config.num_key_value_heads = 8
            model_config.hidden_size = 6144
            model_config.intermediate_size = 16384
            model_config.head_dim = 128

        else:
            raise ValueError("Model not supported")

        return model_config

    def save_and_load(self, file_path, save_dir):
        temp_state_dict = {}
        filename = os.path.basename(file_path)
        save_path = os.path.join(
            save_dir, filename.replace(".safetensors", ".pt")
        )

        if os.path.exists(save_path):
            logging.info(f"File {save_path} already exists, skipping")
            return

        temp_state_dict = load_file(file_path)
        torch.save(temp_state_dict, save_path)
        logging.info(f"Saved {file_path} to {save_path}")

    def _save_safetensors_to_pt(self):
        ckpt_files = os.listdir(self.cache_dir)
        ckpt_files = [
            os.path.join(self.cache_dir, ckpt)
            for ckpt in ckpt_files
            if ckpt.endswith(".safetensors")
        ]

        processes = []
        for ckpt in tqdm(
            ckpt_files, desc="Loading checkpoint files", smoothing=0
        ):
            p = Process(
                target=self.save_and_load, args=(ckpt, self.pt_ckpt_dir)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        logging.info("All safetensor loader processes joined")

    # def _load_model_skeleton(self):
    #     for key, param in self.skeleton_state_dict.items():
    #         if '.' in key:
    #             module_path, param_name = key.rsplit('.', 1)
    #             module = self.model
    #             for part in module_path.split('.'):
    #                 if part.isdigit():
    #                     module = module[int(part)]
    #                 else:
    #                     module = getattr(module, part)
    #             setattr(module, param_name, param)
    #         else:
    #             setattr(self.model, key, param)

    #     byte_size = (
    #         sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    #         * 2
    #         / (1024**3)
    #     )
    #     logging.info(f"Model size: {byte_size:.2f} GB")
    def _load_model_skeleton(self):
        for key, param in self.model.named_parameters():
            if key in self.skeleton_state_dict:
                param.data = self.skeleton_state_dict[key]

        model_skeletion_byte_size = (
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            * 2
            / (1024**3)
        )
        logging.info(f"Model skeleton size: {model_skeletion_byte_size:.2f} GB")

    def Init(self):
        self._parse_state_dict()
        self.core_engine = core_engine(self.engine_config, self.model_config)
        if "8x7B" in self.huggingface_ckpt_name:
            param_byte_size = 48 * 1024 * 1024 * 1024 * 2
        elif "8x22B" in self.huggingface_ckpt_name:
            param_byte_size = 143 * 1024 * 1024 * 1024 * 2
        else:
            raise ValueError("Model not supported yet")
        self.core_engine.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            param_byte_size,
            self.weight_copy_task,
        )
        self._load_model_skeleton()
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

    def _parse_state_dict(self):
        model_init_start_time = time.perf_counter()
        self.loaded_model_config._attn_implementation = "flash_attention_2"
        self.model = MixtralForCausalLM._from_config(self.loaded_model_config).to(
            self.engine_config.Basic_Config.device_torch
        )
        self.model.eval()
        logging.info(
            f"torch module init time: {time.perf_counter() - model_init_start_time} s"
        )

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        # self.weight_copy_task["shared_expert"] = []

        for layer_idx in trange(
            len(self.model.model.layers), desc="Parsing state_dict"
        ):
            for name, _ in self.model.model.layers[
                layer_idx
            ].self_attn.named_parameters():
                tensor_full_name = (
                    "model.layers." + str(layer_idx) + ".self_attn." + name
                )
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": "attn_" + str(layer_idx),
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append("attn_" + str(layer_idx))

            for expert_idx in range(
                len(self.model.model.layers[layer_idx].block_sparse_moe.experts)
            ):
                for name, _ in (
                    self.model.model.layers[layer_idx]
                    .block_sparse_moe.experts[expert_idx]
                    .named_parameters()
                ):
                    tensor_full_name = (
                        "model.layers."
                        + str(layer_idx)
                        + ".block_sparse_moe.experts."
                        + str(expert_idx)
                        + "."
                        + name
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": "routed_expert_"
                        + str(layer_idx)
                        + "_"
                        + str(expert_idx),
                        "tensor_key": name,
                    }
                self.weight_copy_task["routed_expert"].append(
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                )

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
            setattr(
                attn_module,
                "decoding_attn",
                types.MethodType(decoding_attn, attn_module),
            )
            # setattr(attn_module, "decoding_flash", types.MethodType(decoding_flash, attn_module))
            # setattr(attn_module, "prefill_attn", types.MethodType(prefill_attn, attn_module))
            if "attn_" + str(layer_idx) in self.weight_copy_task["attn"]:
                get_weights = True
            else:
                get_weights = False
            attn_wrapper_instance = Attn_Wrapper(
                attn_module,
                layer_idx,
                self.core_engine,
                self.engine_config,
                self.model_config,
                get_weights,
            )
            self.model.model.layers[layer_idx].self_attn = attn_wrapper_instance

    def _config_expert_module(self):
        """
        Replace expert module with the wrapper.
        """
        for layer_idx in range(len(self.model.model.layers)):
            layer = self.model.model.layers[layer_idx]
            for expert_idx in range(len(layer.block_sparse_moe.experts)):
                expert_module = layer.block_sparse_moe.experts[expert_idx]
                if (
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                    in self.weight_copy_task["routed_expert"]
                ):
                    get_weights = True
                else:
                    get_weights = False
                expert_wrapper_instance = Expert_Wrapper(
                    expert_module,
                    layer_idx,
                    expert_idx,
                    self.core_engine,
                    self.engine_config,
                    self.model_config,
                    get_weights,
                )
                layer.block_sparse_moe.experts[expert_idx] = (
                    expert_wrapper_instance
                )
