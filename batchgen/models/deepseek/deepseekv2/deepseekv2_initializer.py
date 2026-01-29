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
from safetensors.torch import load_file
from tqdm import tqdm, trange
from batchgen.config import EngineConfig, ModelConfig
from batchgen.config.model_registry import load_config

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    # jit compile
    from core_engine import batchgen as core_engine
# Use new wrapper system - Attn_Wrapper/Expert_Wrapper are aliases for backward compatibility
from batchgen.models.wrappers import AttnWrapperBase, ExpertWrapperBase
Attn_Wrapper = AttnWrapperBase
Expert_Wrapper = ExpertWrapperBase

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.append(current_dir)
from .modeling_deepseek_v2 import (
    DeepseekV2ForCausalLM,
    apply_rotary_pos_emb,
    rotate_half,
)
from .configuration_deepseek_v2 import DeepseekV2Config as HFDeepseekV2Config


def rotary_pos_emb(t, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = t.shape
    t = t.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    t_embed = (t * cos) + (rotate_half(t) * sin)
    return t_embed


# 	# return query_states, key_states, value_states
def _Pre_Attn(
    self, hidden_states, past_compressed_kv, attention_mask, position_ids
):
    bsz, q_len, _ = hidden_states.size()
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    new_compressed_kv = compressed_kv.clone()

    compressed_kv = torch.cat([past_compressed_kv, compressed_kv], dim=1)
    kv_len = compressed_kv.size(1)
    assert kv_len == attention_mask.size(-1)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(
            bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    kv_seq_len = attention_mask.shape[-1]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids[:, -1].unsqueeze(-1))
    k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(
        bsz, self.num_heads, kv_seq_len, self.q_head_dim
    )
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    return new_compressed_kv, query_states, key_states, value_states


def _Attn_Mechanism(
    self, query_states, key_states, value_states, attention_mask
):
    bsz, _, q_len, _ = query_states.size()
    kv_seq_len = attention_mask.shape[-1]

    attn_weights = (
        torch.matmul(query_states, key_states.transpose(2, 3))
        * self.softmax_scale
    )

    logging.debug(f"Attention weights shape: {attn_weights.shape}")

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    assert attention_mask is not None
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    logging.debug("Starting softmax")

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )

    logging.debug("Starting matmul")
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output


def _Post_Attn(self, attn_output):
    bsz, _, q_len, _ = attn_output.size()
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        bsz, q_len, self.num_heads * self.v_head_dim
    )
    attn_output = self.o_proj(attn_output)
    return attn_output


def prefill_attn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
):
    bsz, q_len, _ = hidden_states.size()
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

    kv_len = compressed_kv.size(1)
    assert kv_len == attention_mask.size(-1)
    compressed_kv_ref, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv_ref))
        .view(
            bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    kv_seq_len = attention_mask.shape[-1]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(
        bsz, self.num_heads, kv_seq_len, self.q_head_dim
    )
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    attn_weights = (
        torch.matmul(query_states, key_states.transpose(2, 3))
        * self.softmax_scale
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    assert attention_mask is not None
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )

    attn_output = torch.matmul(attn_weights, value_states)
    if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(
        bsz, q_len, self.num_heads * self.v_head_dim
    )

    attn_output = self.o_proj(attn_output)
    return attn_output, compressed_kv


def decoding_attn(
    self,
    hidden_states: torch.Tensor,
    past_key_states: torch.Tensor,
    past_value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
):
    bsz, q_len, _ = hidden_states.size()

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

    past_key_states = past_key_states.to(torch.bfloat16)
    compressed_kv_ref = torch.cat([past_key_states, compressed_kv], dim=1)
    kv_len = compressed_kv_ref.size(1)
    assert kv_len == attention_mask.size(-1)
    compressed_kv_ref, k_pe = torch.split(
        compressed_kv_ref, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv_ref))
        .view(
            bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    kv_seq_len = attention_mask.shape[-1]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids[:, -1].unsqueeze(-1))
    k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(
        bsz, self.num_heads, kv_seq_len, self.q_head_dim
    )
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    attn_weights = (
        torch.matmul(query_states, key_states.transpose(2, 3))
        * self.softmax_scale
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    assert attention_mask is not None
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(
        bsz, q_len, self.num_heads * self.v_head_dim
    )

    attn_output = self.o_proj(attn_output)

    return (
        attn_output,
        compressed_kv,
        torch.tensor([], device=hidden_states.device),
    )


def compress_kv(
    self, hidden_states_kv: torch.Tensor, kv_position_ids: torch.LongTensor
) -> torch.Tensor:
    # return the RoPE'ed & compressed kv
    bsz, kv_seq_len, _ = hidden_states_kv.size()
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states_kv)
    new_compressed_kv = compressed_kv.clone()
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    compressed_kv = self.kv_a_layernorm(compressed_kv)
    k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(k_pe, seq_len=kv_seq_len)
    k_pe = rotary_pos_emb(k_pe, cos, sin, kv_position_ids).view(
        bsz, kv_seq_len, self.qk_rope_head_dim
    )
    return torch.cat([compressed_kv, k_pe], dim=-1), new_compressed_kv


def cus_absorbed_mla_decoding_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_states: torch.Tensor,
    past_value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
):
    bsz, q_len, _ = hidden_states.size()

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

    # past_key_states = past_key_states.to(torch.bfloat16)
    compressed_kv_ref = torch.cat([past_key_states, compressed_kv], dim=1)
    kv_len = compressed_kv_ref.size(1)
    assert kv_len == attention_mask.size(-1)
    compressed_kv_ref, k_pe = torch.split(
        compressed_kv_ref, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    compressed_kv_ref = self.kv_a_layernorm(compressed_kv_ref)

    k_pe = k_pe.view(bsz, 1, kv_len, self.qk_rope_head_dim)
    cos, sin = self.rotary_emb(k_pe, seq_len=kv_len)
    k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids)
    # q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    kv_b_proj = self.kv_b_proj.weight.view(
        self.num_heads, -1, self.kv_lora_rank
    )
    q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
    out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

    # cos, sin = self.rotary_emb(q_pe, seq_len=q_len)
    # q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids)
    q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids[:, -1].unsqueeze(-1))

    q_nope = torch.matmul(q_nope, q_absorb)
    # attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)) * self.softmax_scale
    attn_weights = torch.einsum("bhqd,bhcd->bhqc", q_pe, k_pe)
    attn_weights = attn_weights + torch.einsum(
        "bhqd,bhcd->bhqc", q_nope, compressed_kv_ref.unsqueeze(-3)
    )
    attn_weights = attn_weights * self.softmax_scale
    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_len)}, but is"
            f" {attn_weights.size()}"
        )
    attn_weights = attn_weights + attention_mask
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q_nope.dtype)
    attn_output = torch.einsum(
        "bhql,blc->bhqc", attn_weights, compressed_kv_ref
    )
    attn_output = torch.matmul(
        attn_output, out_absorb.mT
    )  # torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        bsz, q_len, self.num_heads * self.v_head_dim
    )
    attn_output = self.o_proj(attn_output)

    return (
        attn_output,
        compressed_kv,
        torch.tensor([], device=hidden_states.device),
    )


class DeepSeek_Initializer:
    def __init__(
        self,
        huggingface_ckpt_name: str,
        hf_cache_dir: str,
        cache_dir: Optional[str],
        engine_config: EngineConfig,
        skeleton_state_dict: Optional[dict],
        shm_name: str,
        tensor_meta_shm_name: str,
        converted_ckpt_dir,
        host_kv_cache_size,
    ):
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.cache_dir = cache_dir
        self.converted_ckpt_dir = converted_ckpt_dir
        self.engine_config = engine_config
        self.skeleton_state_dict = skeleton_state_dict
        self.host_kv_cache_size = host_kv_cache_size
        self.host_kv_cache_byte_size = host_kv_cache_size * 1024 * 1024 * 1024

        self.model = None
        # Use BatchGen's unified config system instead of HuggingFace AutoConfig
        self.loaded_model_config = load_config(huggingface_ckpt_name)

        # Create HuggingFace config for model instantiation (local config class, NOT AutoConfig)
        self.hf_config = HFDeepseekV2Config()
        self.hf_config._name_or_path = huggingface_ckpt_name

        # TODO:
        self.model_config = self._parse_model_config()
        self._default_engine_config()
        self.state_dict_name_map = {}
        self.weight_copy_task = {}

        self.shm_name = shm_name
        self.tensor_meta_shm_name = tensor_meta_shm_name

    def _default_engine_config(self):
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")
        if (
            self.loaded_model_config._name_or_path
            == "deepseek-ai/DeepSeek-V2-Lite-Chat" or
            self.loaded_model_config._name_or_path
            == "deepseek-ai/DeepSeek-V2-Lite"
        ):
            self.engine_config.Basic_Config.log_level = "info"
            self.engine_config.Basic_Config.torch_dtype = torch.bfloat16
            self.engine_config.Basic_Config.dtype_str = "bfloat16"
            self.engine_config.Basic_Config.attn_mode = 1
            self.engine_config.Basic_Config.module_types = [
                "attn",
                "routed_expert",
                "shared_expert",
            ]
            self.engine_config.Basic_Config.num_threads = 16

            self.engine_config.KV_Storage_Config.reserved_length = (
                self.engine_config.Basic_Config.padding_length
                + self.engine_config.Basic_Config.max_decoding_length
            )
            self.engine_config.KV_Storage_Config.slot_byte_size = (
                self.engine_config.KV_Storage_Config.reserved_length
                * self.model_config.compressed_kv_dim
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
                    60 * 544 / context_length
                )
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = math.floor(
                    120 * 544 / context_length
                )
            else:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 60
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 120
            logging.info(
                f"attn_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size}"
            )
            logging.info(
                f"MoE_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size}"
            )
            self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 2048

            if context_length > 544:
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = math.floor(
                    200 * 544 / context_length
                )
            else:
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 200
            logging.info(
                f"attn_decoding_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size}"
            )
            self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size = self.engine_config.Module_Batching_Config.global_batch_size
            self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 2048

            self.engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
                "attn": 1,
                "routed_expert": 32,
                "shared_expert": 1,
            }
            self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
                "attn": 1,
                "routed_expert": 64,
                "shared_expert": 1,
            }
            self.engine_config.GPU_Buffer_Config.num_k_buffer = 2
            self.engine_config.GPU_Buffer_Config.num_v_buffer = 0
            # self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size * 576
            self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
                * (
                    self.engine_config.Basic_Config.max_decoding_length
                    + self.engine_config.Basic_Config.padding_length
                )
            )
            self.engine_config.GPU_Buffer_Config.module_shapes = {
                "attn": {
                    "q_proj.weight": [3072, 2048],
                    "kv_a_proj_with_mqa.weight": [576, 2048],
                    "kv_a_layernorm.weight": [512],
                    "kv_b_proj.weight": [4096, 512],
                    "o_proj.weight": [2048, 2048],
                },
                "routed_expert": {
                    "gate_proj.weight": [1408, 2048],
                    "up_proj.weight": [1408, 2048],
                    "down_proj.weight": [2048, 1408],
                },
                "shared_expert": {
                    "gate_proj.weight": [2816, 2048],
                    "up_proj.weight": [2816, 2048],
                    "down_proj.weight": [2048, 2816],
                },
            }

        elif self.huggingface_ckpt_name in [
            "deepseek-ai/DeepSeek-V2",
            "deepseek-ai/DeepSeek-V2.5",
            "deepseek-ai/DeepSeek-V2.5-1210",
            "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        ]:
            self.engine_config.Basic_Config.log_level = "info"
            self.engine_config.Basic_Config.torch_dtype = torch.bfloat16
            self.engine_config.Basic_Config.dtype_str = "bfloat16"
            self.engine_config.Basic_Config.attn_mode = 1
            self.engine_config.Basic_Config.module_types = [
                "attn",
                "routed_expert",
                "shared_expert",
            ]
            self.engine_config.Basic_Config.num_threads = 16

            # self.engine_config.Module_Batching_Config.prefill_micro_batch_size = 1
            # self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 60
            # self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 8192
            # self.engine_config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 8192
            self.engine_config.KV_Storage_Config.reserved_length = (
                self.engine_config.Basic_Config.padding_length
                + self.engine_config.Basic_Config.max_decoding_length
            )
            self.engine_config.KV_Storage_Config.slot_byte_size = (
                self.engine_config.KV_Storage_Config.reserved_length
                * self.model_config.compressed_kv_dim
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
            if context_length > 768:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = math.floor(
                    15 * 768 / context_length
                )
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = math.floor(
                    30 * 768 / context_length
                )
            else:
                self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 15
                self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 30
            logging.info(
                f"attn_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_prefill_micro_batch_size}"
            )
            logging.info(
                f"MoE_prefill_micro_batch_size: {self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size}"
            )
            self.engine_config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 4096

            if context_length > 768:
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = math.floor(
                    60 * 768 / context_length
                )
            else:
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 60
            logging.info(
                f"attn_decoding_micro_batch_size: {self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size}"
            )
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
                "routed_expert": 160,
                "shared_expert": 1,
            }
            self.engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
                "attn": 1,
                "routed_expert": 160,
                "shared_expert": 1,
            }
            # self.engine_config.GPU_Buffer_Config.num_kv_buffer = 2
            self.engine_config.GPU_Buffer_Config.num_k_buffer = 2
            self.engine_config.GPU_Buffer_Config.num_v_buffer = 0
            self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
                self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
                * (
                    self.engine_config.Basic_Config.max_decoding_length
                    + self.engine_config.Basic_Config.padding_length
                )
            )
            self.engine_config.GPU_Buffer_Config.module_shapes = {
                "attn": {
                    "q_a_proj.weight": [1536, 5120],
                    "q_a_layernorm.weight": [1536],
                    "q_b_proj.weight": [24576, 1536],
                    "kv_a_proj_with_mqa.weight": [576, 5120],
                    "kv_a_layernorm.weight": [512],
                    "kv_b_proj.weight": [32768, 512],
                    "o_proj.weight": [5120, 16384],
                },
                "routed_expert": {
                    "gate_proj.weight": [1536, 5120],
                    "up_proj.weight": [1536, 5120],
                    "down_proj.weight": [5120, 1536],
                },
                "shared_expert": {
                    "gate_proj.weight": [3072, 5120],
                    "up_proj.weight": [3072, 5120],
                    "down_proj.weight": [5120, 3072],
                },
            }

    def _parse_model_config(self):
        model_config = ModelConfig()
        # V2/V2.5
        if "DeepSeek-V2" in self.loaded_model_config._name_or_path:
            if (
                self.loaded_model_config._name_or_path
                == "deepseek-ai/DeepSeek-V2-Lite-Chat" or 
                self.loaded_model_config._name_or_path
                == "deepseek-ai/DeepSeek-V2-Lite"
            ):
                model_config.model_type = "deepseek_v2"
                model_config.num_hidden_layers = 27
                model_config.num_local_experts = 64
                model_config.num_attention_heads = 16
                model_config.num_key_value_heads = 16
                model_config.head_dim = 192
                model_config.compressed_kv_dim = 576

            else:
                model_config.model_type = "deepseek_v2"
                model_config.num_hidden_layers = 60
                model_config.num_local_experts = 160
                model_config.num_attention_heads = 128
                model_config.num_key_value_heads = 128
                model_config.head_dim = 192
                model_config.compressed_kv_dim = 576

        elif (
            "DeepSeek-V3" in self.loaded_model_config._name_or_path
            or "DeepSeek-R1" in self.loaded_model_config._name_or_path
        ):
            model_config.model_type = "deepseek_v3"
            model_config.num_hidden_layers = 61
            model_config.num_local_experts = 256
            model_config.num_attention_heads = 128
            model_config.num_key_value_heads = 128
            model_config.head_dim = 192
            model_config.compressed_kv_dim = 576

        else:
            raise ValueError("Model not supported")

        return model_config

    def save_and_load(self, file_path, save_dir):
        tensor_dict = load_file(file_path)
        torch.save(tensor_dict, save_dir)
        return tensor_dict

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
            dst_dir = os.path.join(
                self.converted_ckpt_dir,
                ckpt.split("/")[-1].replace(".safetensors", ".pt"),
            )
            if os.path.exists(dst_dir):
                continue

            p = Process(target=self.save_and_load, args=(ckpt, dst_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            p.close()
        logging.info("All safetensor loader processes joined")

    def _load_model_skeleton(self, skeleton_state_dict):
        # logging.info(skieleton_state_dict.keys())
        for key, param in skeleton_state_dict.items():
            levels = key.split(".")
            # If the key cannot be found in the model, skip it
            if not hasattr(self.model, levels[0]):
                continue
            weight = self.model.__getattr__(levels[0])
            for level in levels[1:]:
                if not hasattr(weight, level):
                    weight = None
                    break
                weight = weight.__getattr__(level)
            if weight is not None:
                weight.data = param.to(
                    self.engine_config.Basic_Config.torch_dtype
                )
        byte_size = (
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            * 2
            / (1024**3)
        )
        logging.info(f"Model size: {byte_size:.2f} GB")

    def Init(self):
        try:
            # self._save_safetensors_to_pt()
            self._parse_state_dict()
            logging.info("State dict parsed")
            self.core_engine = core_engine(
                self.engine_config, self.model_config
            )
            logging.info("Core engine created")
            if (
                self.loaded_model_config._name_or_path
                == "deepseek-ai/DeepSeek-V2-Lite-Chat" or 
                self.loaded_model_config._name_or_path
                == "deepseek-ai/DeepSeek-V2-Lite"
            ):
                param_byte_size = 32 * 1024 * 1024 * 1024
            else:
                param_byte_size = 236 * 1024 * 1024 * 1024 * 2

            self.core_engine.Init(
                self.shm_name,
                self.tensor_meta_shm_name,
                param_byte_size,
                self.weight_copy_task,
            )
            logging.info("Core engine initialized")
            self._load_model_skeleton(self.skeleton_state_dict)
            self._config_attn_module()
            self._config_expert_module()
            self._config_lm_head_hook()

            self.model.eval()
            # TODO:
            self.model.to(torch.bfloat16)
            self.model.to(self.engine_config.Basic_Config.device_torch)
            # print(self.model, flush=True)
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
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
        # Use HuggingFace config for model instantiation (PretrainedConfig required)
        self.hf_config._attn_implementation = "eager"
        if self.model_config.model_type == "deepseek_v2":
            self.model = DeepseekV2ForCausalLM._from_config(
                self.hf_config
            ).to(self.engine_config.Basic_Config.device_torch)
        else:
            raise ValueError("Model not supported")
        self.model.eval()
        logging.info(
            f"torch module init time: {time.perf_counter() - model_init_start_time} s"
        )

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        for layer_idx in trange(self.model_config.num_hidden_layers):
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

            if layer_idx >= self.loaded_model_config.first_k_dense_replace:
                for name, _ in self.model.model.layers[
                    layer_idx
                ].mlp.shared_experts.named_parameters():
                    tensor_full_name = (
                        "model.layers."
                        + str(layer_idx)
                        + ".mlp.shared_experts."
                        + name
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": "shared_expert_" + str(layer_idx),
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    "shared_expert_" + str(layer_idx)
                )

                for expert_idx in range(self.model_config.num_local_experts):
                    for name, _ in (
                        self.model.model.layers[layer_idx]
                        .mlp.experts[expert_idx]
                        .named_parameters()
                    ):
                        tensor_full_name = (
                            "model.layers."
                            + str(layer_idx)
                            + ".mlp.experts."
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
                        "routed_expert_"
                        + str(layer_idx)
                        + "_"
                        + str(expert_idx)
                    )

    def _config_attn_module(self):
        """
        - Configure the wrapper.
        """
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            setattr(
                attn_module,
                "Pre_Attn",
                types.MethodType(_Pre_Attn, attn_module),
            )
            setattr(
                attn_module,
                "Attn_Mechanism",
                types.MethodType(_Attn_Mechanism, attn_module),
            )
            setattr(
                attn_module,
                "Post_Attn",
                types.MethodType(_Post_Attn, attn_module),
            )
            # setattr(
            # 	attn_module,
            # 	"decoding_attn",
            # 	types.MethodType(decoding_attn, attn_module),
            # )
            setattr(
                attn_module,
                "decoding_attn",
                types.MethodType(
                    cus_absorbed_mla_decoding_forward, attn_module
                ),
            )

            setattr(
                attn_module,
                "prefill_attn",
                types.MethodType(prefill_attn, attn_module),
            )
            setattr(
                attn_module,
                "compress_kv",
                types.MethodType(compress_kv, attn_module),
            )
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
        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            len(self.model.model.layers),
        ):
            layer = self.model.model.layers[layer_idx]
            if (
                "shared_expert_" + str(layer_idx)
                in self.weight_copy_task["shared_expert"]
            ):
                get_weights = True
            else:
                get_weights = False
            layer.mlp.shared_experts = Expert_Wrapper(
                layer.mlp.shared_experts,
                layer_idx,
                -1,
                self.core_engine,
                self.engine_config,
                self.model_config,
                get_weights,
            )
            for expert_idx in range(len(layer.mlp.experts)):
                if (
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                    in self.weight_copy_task["routed_expert"]
                ):
                    get_weights = True
                else:
                    get_weights = False
                layer.mlp.experts[expert_idx] = Expert_Wrapper(
                    layer.mlp.experts[expert_idx],
                    layer_idx,
                    expert_idx,
                    self.core_engine,
                    self.engine_config,
                    self.model_config,
                    get_weights,
                )
