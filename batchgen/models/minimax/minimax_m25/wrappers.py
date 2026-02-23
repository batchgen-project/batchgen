# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""MiniMax-M2.5 wrappers for BatchGen execution.

Provides wrappers for MiniMax-M2.5 with FP8 quantization:
- MiniMaxM25ExpertWrapper: Expert wrapper with FP8 block-wise dequantization
- MiniMaxM25AttnWrapper: Attention wrapper with GQA + QK norm + partial RoPE
"""

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization
from .model import rotate_half


class MiniMaxM25ExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with FP8 dequantization for MiniMax-M2.5.

    Reuses DeepSeek-V3's FP8 block-wise [128,128] dequantization.
    Expert FFN: gate_proj [1536, 3072], up_proj [1536, 3072], down_proj [3072, 1536]
    Activation: standard SwiGLU (silu(gate) * up)
    """

    def __init__(self, module, layer_idx, expert_idx, core_engine, engine_config,
                 model_config, persistent=False, weight_dequant_scale=None):
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def dequantize_weights(self, weights_dict):
        """Dequantize FP8 weights using block-wise scale factors."""
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = deepseek_v3_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def _register_fp8_weights(self):
        self.fp8_gate = self.module.gate_proj.weight.data
        self.fp8_up = self.module.up_proj.weight.data
        self.fp8_down = self.module.down_proj.weight.data

    def _unregister_fp8_weights(self):
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def _forward_impl(self, hidden_states):
        """Forward using deepgemm kernel (FP8 weight × BF16 activation)."""
        return self.module.deepgemm_forward(hidden_states, self.weight_dequant_scale)

    def forward(self, hidden_states):
        rank = self.get_rank_safe()
        if not self.persistent:
            weights = self.load_weights(self.module_key)
            for name, param in self.module.named_parameters():
                param.data = weights[name]
        else:
            self.module.gate_proj.weight.data = self.fp8_gate
            self.module.up_proj.weight.data = self.fp8_up
            self.module.down_proj.weight.data = self.fp8_down

        result = self._forward_impl(hidden_states)

        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        return result


class MiniMaxM25AttnWrapper(AttnWrapperBase):
    """Attention wrapper for MiniMax-M2.5 GQA with QK norm + partial RoPE.

    M2.5 attention is BF16 (not quantized). Only expert FFN weights are FP8.
    """

    def __init__(self, module, layer_idx, core_engine, engine_config,
                 model_config, persistent=True, weight_dequant_scale=None):
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )

    def dequantize_weights(self, weights_dict):
        """M2.5 attention is BF16 — no dequantization needed."""
        return weights_dict

    def _forward_prefill(self, hidden_states, **kwargs):
        """Prefill forward: Q/K/V projection + QK norm + partial RoPE + FA varlen.

        Follows GPT-OSS pattern: all attention logic inline in wrapper,
        calling gqa_prefill_fa directly (no setattr injection).
        """
        from batchgen.attention.gqa import gqa_prefill_fa
        from batchgen.attention.fused_kernels import cuda_rmsnorm

        # Always prepack mode in production prefill
        if hidden_states.dim() == 3:
            assert hidden_states.shape[0] == 1, "Prepack mode expects batch_size=1"
            hidden_states_2d = hidden_states.squeeze(0)
        else:
            hidden_states_2d = hidden_states

        total_tokens = hidden_states_2d.shape[0]
        cu_seqlens = self.prepack_cu_seqlens.to(hidden_states_2d.device)
        max_seqlen = self.prepack_max_seqlen
        position_ids = self.position_ids.to(hidden_states_2d.device)

        # Q/K/V projection
        query = self.module.q_proj(hidden_states_2d)   # [total_tokens, num_heads * head_dim]
        key = self.module.k_proj(hidden_states_2d)     # [total_tokens, num_kv_heads * head_dim]
        value = self.module.v_proj(hidden_states_2d)   # [total_tokens, num_kv_heads * head_dim]

        # QK Norm (on flat projected dims, before reshape)
        query = cuda_rmsnorm(query, self.module.q_norm.weight, self.module.q_norm.eps)
        key = cuda_rmsnorm(key, self.module.k_norm.weight, self.module.k_norm.eps)

        # Reshape to [total_tokens, num_heads, head_dim]
        num_heads = self.module.num_heads
        num_kv_heads = self.module.num_kv_heads
        head_dim = self.module.head_dim
        rotary_dim = self.module.rotary_dim

        query = query.view(total_tokens, num_heads, head_dim)
        key = key.view(total_tokens, num_kv_heads, head_dim)
        value = value.view(total_tokens, num_kv_heads, head_dim)

        # Partial RoPE (rotate first rotary_dim=64 dims, passthrough rest)
        cos, sin = self.module.rotary_emb(value, seq_len=max_seqlen)
        cos = cos[position_ids]  # [total_tokens, rotary_dim]
        sin = sin[position_ids]

        # Apply RoPE to rotary part only
        q_rot = query[..., :rotary_dim]
        q_pass = query[..., rotary_dim:]
        k_rot = key[..., :rotary_dim]
        k_pass = key[..., rotary_dim:]

        cos = cos.unsqueeze(1)  # [total_tokens, 1, rotary_dim]
        sin = sin.unsqueeze(1)

        query = torch.cat([
            q_rot * cos + rotate_half(q_rot) * sin,
            q_pass,
        ], dim=-1)
        key = torch.cat([
            k_rot * cos + rotate_half(k_rot) * sin,
            k_pass,
        ], dim=-1)

        # FlashAttention varlen GQA
        attn_output, lse = gqa_prefill_fa(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )

        # Output projection: [total_tokens, num_heads * head_dim] -> [total_tokens, hidden_size]
        attn_output = attn_output.view(total_tokens, num_heads * head_dim)
        attn_output = self.module.o_proj(attn_output)

        # Offload KV cache to host
        torch.cuda.current_stream().synchronize()
        self._offload_prepacked_kv_gqa(key.view(total_tokens, num_kv_heads, head_dim),
                                        value)

        attn_output = attn_output.unsqueeze(0)
        return (attn_output, None, None)

    def _offload_prepacked_kv_gqa(self, k_cache, v_cache):
        """Offload GQA KV cache per-sequence to host memory."""
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            seq_k = k_cache[start_idx:end_idx].unsqueeze(0)
            seq_v = v_cache[start_idx:end_idx].unsqueeze(0)
            seq_global_id = [global_sequence_ids[seq_idx]]

            self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_k,
                v_tensor=seq_v,
                sequence_lengths=[seq_len],
            )

    def _forward_decode(self, hidden_states, **kwargs):
        """Decode forward: Q/K/V + QK norm + partial RoPE + paged KV FA.

        Follows GPT-OSS pattern: all attention logic inline in wrapper.
        """
        from batchgen.attention.gqa import gqa_decode_fa
        from batchgen.attention.fused_kernels import cuda_rmsnorm

        batch_slice = kwargs.get("batch_slice", None)
        batch, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "Decode expects single token"

        if batch == 0 or hidden_states.numel() == 0:
            return (hidden_states, None, None)

        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        cache_seqlens = AttnWrapperBase.cache_seqlens

        # Micro-batch slicing
        micro_cache_seqlens = cache_seqlens
        if cache_seqlens is not None and batch_slice is not None:
            start_idx, end_idx = batch_slice
            micro_cache_seqlens = cache_seqlens[start_idx:end_idx]

        current_token_position = micro_cache_seqlens - 1 if micro_cache_seqlens is not None else None

        num_heads = self.module.num_heads
        num_kv_heads = self.module.num_kv_heads
        head_dim = self.module.head_dim
        rotary_dim = self.module.rotary_dim

        # Q/K/V projection
        query = self.module.q_proj(hidden_states)
        key = self.module.k_proj(hidden_states)
        value = self.module.v_proj(hidden_states)

        # QK Norm
        query = cuda_rmsnorm(query, self.module.q_norm.weight, self.module.q_norm.eps)
        key = cuda_rmsnorm(key, self.module.k_norm.weight, self.module.k_norm.eps)

        # Reshape: [batch, 1, num_heads, head_dim]
        query = query.view(batch, seq_len, num_heads, head_dim)
        key = key.view(batch, seq_len, num_kv_heads, head_dim)
        value = value.view(batch, seq_len, num_kv_heads, head_dim)

        # Partial RoPE
        max_seqlen = AttnWrapperBase.max_seqlen
        cos, sin = self.module.rotary_emb(value, seq_len=max_seqlen)
        cos = cos[current_token_position].unsqueeze(1)  # [batch, 1, rotary_dim]
        sin = sin[current_token_position].unsqueeze(1)

        q_rot = query[..., :rotary_dim]
        q_pass = query[..., rotary_dim:]
        k_rot = key[..., :rotary_dim]
        k_pass = key[..., rotary_dim:]

        query = torch.cat([
            q_rot * cos + rotate_half(q_rot) * sin,
            q_pass,
        ], dim=-1)
        key = torch.cat([
            k_rot * cos + rotate_half(k_rot) * sin,
            k_pass,
        ], dim=-1)

        # Write new K,V to paged GPU cache
        gpu_kv_manager.update_layer_decode_new_token(
            k_tensor=key,
            v_tensor=value,
            sequence_lengths=current_token_position,
            layer_idx=self.layer_idx,
            batch_slice=batch_slice,
        )

        # Retrieve paged KV cache
        k_cache_layer, v_cache_layer, page_table = gpu_kv_manager.get_layer_kv_with_page_table(
            self.layer_idx
        )
        if batch_slice is not None:
            start_idx, end_idx = batch_slice
            page_table = page_table[start_idx:end_idx]

        cache_seqlens_for_attn = micro_cache_seqlens

        # FlashAttention decode with paged KV
        attn_output, _ = gqa_decode_fa(
            q=query,
            k_cache=k_cache_layer,
            v_cache=v_cache_layer,
            cache_seqlens=cache_seqlens_for_attn,
            block_table=page_table,
        )

        # Output projection
        attn_output = attn_output.view(batch, 1, num_heads * head_dim)
        attn_output = self.module.o_proj(attn_output)

        # Append KV to host
        kv_append_callback = getattr(AttnWrapperBase, 'kv_append_callback', None)
        if kv_append_callback is not None:
            kv_append_callback(self.layer_idx, key, value)

        return (attn_output, None, None)
