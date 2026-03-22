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

"""Qwen3-specific wrappers for BatchGen execution.

Provides wrappers for Qwen3 models (BF16, dense, GQA, no sinks):
- Qwen3AttnWrapper: Attention wrapper with standard GQA (FA2/FA3 auto-detect)
- Qwen3MLPWrapper: Dense MLP wrapper (no quantization)

Target: Single device deployment (Ada 6000, A6000).
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase

from .model import apply_rotary_pos_emb, rotate_half

# Import GQA attention (auto-detects FA2/FA3)
from batchgen.attention.gqa.gqa_attention import gqa_attention_prefill
from batchgen.attention.gqa.fa_prefill import gqa_prefill_fa
from batchgen.attention.gqa.fa_decode import gqa_decode_fa

# GQA decode for paged KV cache
from batchgen.attention.gqa.batchgen_gqa_decode_bf16 import batchgen_gqa_decode_bf16


class Qwen3AttnWrapper(AttnWrapperBase):
    """Attention wrapper for Qwen3 with standard GQA (no sinks, no sliding window).

    Qwen3 attention features:
    - BF16 weights (no quantization)
    - GQA with 32 query heads, 8 KV heads
    - head_dim=128
    - No attention sinks
    - No sliding window — full attention on all layers
    - QK-norm (RMSNorm on Q and K after projection)
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
    ):
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent=persistent, weight_dequant_scale=None
        )

        self.num_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """No-op for BF16 weights."""
        return weights_dict

    def forward(self, hidden_states=None, **kwargs):
        """Override to route prepacked prefill correctly."""
        import logging as _log

        if not self.persistent:
            weights = self.load_weights(self.module_key)
            dequant_weights = self.dequantize_weights(weights)
            self.apply_weights(dequant_weights)

        # Extract hidden_states from kwargs if passed as kwarg
        if hidden_states is None:
            hidden_states = kwargs.pop("hidden_states", None)
        else:
            kwargs.pop("hidden_states", None)

        if self.phase == "prefill" and AttnWrapperBase.prepack_mode:
            result = self._forward_prefill_prepacked(
                hidden_states,
                position_ids=AttnWrapperBase.position_ids,
            )
        elif self.phase == "prefill":
            result = self._forward_prefill(hidden_states, **kwargs)
        else:
            result = self._forward_decode(hidden_states, **kwargs)

        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        return result

    def _forward_prefill_prepacked(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Prepacked prefill with KV offloading to host paged KV cache.

        In prepack mode, hidden_states is [1, total_tokens, hidden_dim],
        with cu_seqlens tracking sequence boundaries.
        KV is split per-sequence and offloaded to host for later decode.
        """
        import logging as _log

        if hidden_states.dim() == 3:
            hidden_states_2d = hidden_states.squeeze(0)
            input_was_3d = True
        else:
            hidden_states_2d = hidden_states
            input_was_3d = False

        total_tokens = hidden_states_2d.shape[0]

        cu_seqlens = AttnWrapperBase.prepack_cu_seqlens
        max_seqlen = AttnWrapperBase.prepack_max_seqlen
        num_sequences = AttnWrapperBase.prepack_num_sequences
        seq_lengths = AttnWrapperBase.prepack_seq_lengths

        # Project Q, K, V: [total_tokens, hidden] -> [total_tokens, heads, dim]
        query = self.module.q_proj(hidden_states_2d)
        key = self.module.k_proj(hidden_states_2d)
        value = self.module.v_proj(hidden_states_2d)

        query = query.view(total_tokens, self.num_heads, self.head_dim)
        key = key.view(total_tokens, self.num_kv_heads, self.head_dim)
        value = value.view(total_tokens, self.num_kv_heads, self.head_dim)

        # QK-norm (reshape to [..., head_dim] for RMSNorm)
        q_shape, k_shape = query.shape, key.shape
        query = self.module.q_norm(query.reshape(-1, self.head_dim)).reshape(q_shape)
        key = self.module.k_norm(key.reshape(-1, self.head_dim)).reshape(k_shape)

        # RoPE per token using position_ids
        if position_ids is not None:
            cos, sin = self.module.rotary_emb(value, seq_len=max_seqlen)
            cos = cos[position_ids]  # [total_tokens, head_dim]
            sin = sin[position_ids]

            half_dim = self.head_dim // 2
            q1, q2 = query[..., :half_dim], query[..., half_dim:]
            k1, k2 = key[..., :half_dim], key[..., half_dim:]

            cos_half = cos[..., :half_dim].unsqueeze(1)  # [total_tokens, 1, half_dim]
            sin_half = sin[..., :half_dim].unsqueeze(1)

            query = torch.cat([
                q1 * cos_half - q2 * sin_half,
                q2 * cos_half + q1 * sin_half
            ], dim=-1)

            key = torch.cat([
                k1 * cos_half - k2 * sin_half,
                k2 * cos_half + k1 * sin_half
            ], dim=-1)

        # Varlen flash attention
        attn_output, lse = gqa_prefill_fa(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens.to(hidden_states_2d.device),
            cu_seqlens_k=cu_seqlens.to(hidden_states_2d.device),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            sinks=None,
            softmax_scale=self.scale,
        )

        # Output projection
        attn_output = attn_output.view(total_tokens, self.num_heads * self.head_dim)
        attn_output = self.module.o_proj(attn_output)

        # CRITICAL: Offload KV to host paged KV cache per sequence
        # Without this, decode reads empty KV and generates gibberish
        global_sequence_ids = AttnWrapperBase.cur_batch
        torch.cuda.current_stream().synchronize()

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            seq_key = key[start_idx:end_idx].unsqueeze(0)      # [1, seq_len, kv_heads, dim]
            seq_value = value[start_idx:end_idx].unsqueeze(0)  # [1, seq_len, kv_heads, dim]
            seq_global_id = [global_sequence_ids[seq_idx]]

            self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_key,
                v_tensor=seq_value,
                sequence_lengths=[seq_len],
            )

        if input_was_3d:
            attn_output = attn_output.unsqueeze(0)

        return attn_output, None, None

    def _apply_qk_norm(self, query, key):
        """Apply QK-norm (per-head RMSNorm) to Q and K tensors.

        Args:
            query: [batch, seq, num_heads, head_dim]
            key: [batch, seq, num_kv_heads, head_dim]

        Returns:
            Normalized (query, key) with same shapes
        """
        # q_norm and k_norm are Qwen3RMSNorm on head_dim
        q_shape = query.shape
        k_shape = key.shape
        # Flatten for norm: [..., head_dim]
        query = self.module.q_norm(query.reshape(-1, self.head_dim)).reshape(q_shape)
        key = self.module.k_norm(key.reshape(-1, self.head_dim)).reshape(k_shape)
        return query, key

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Prefill forward with standard GQA."""
        batch, seq_len, _ = hidden_states.shape

        # Project Q, K, V (separate projections, no fused QKV)
        query = self.module.q_proj(hidden_states)
        key = self.module.k_proj(hidden_states)
        value = self.module.v_proj(hidden_states)

        query = query.view(batch, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # QK-norm
        query, key = self._apply_qk_norm(query, key)

        # RoPE
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1]

        cos, sin = self.module.rotary_emb(value.transpose(1, 2), seq_len=kv_seq_len)
        # apply_rotary_pos_emb expects [B, H, S, D]
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        query_t, key_t = apply_rotary_pos_emb(query_t, key_t, cos, sin, position_ids)

        # Handle KV cache
        if past_key_value is not None:
            key_t = torch.cat([past_key_value[0], key_t], dim=2)
            value_t = torch.cat([past_key_value[1], value.transpose(1, 2)], dim=2)
        else:
            value_t = value.transpose(1, 2)

        new_kv_cache = (key_t, value_t) if use_cache else None

        # GQA attention (no sinks, no sliding window)
        attn_output = gqa_attention_prefill(
            query=query_t,
            key=key_t,
            value=value_t,
            sinks=None,
            scale=self.scale,
            sliding_window=None,
        )

        # Reshape: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.module.o_proj(attn_output)

        return attn_output, None, new_kv_cache

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        batch_slice: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Decode forward for single token generation using gpu_paged_kv_manager."""
        batch, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "Decode expects single token"

        if batch == 0 or hidden_states.numel() == 0:
            return (hidden_states, None, None)

        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        cache_seqlens = AttnWrapperBase.cache_seqlens

        # Micro-batch slicing
        micro_cache_seqlens = cache_seqlens
        if batch_slice is not None:
            start_idx, end_idx = batch_slice
            micro_cache_seqlens = cache_seqlens[start_idx:end_idx]

        if micro_cache_seqlens is not None and micro_cache_seqlens.numel() == 0:
            return (
                torch.empty((0, 1, hidden_states.shape[-1]),
                            device=hidden_states.device, dtype=hidden_states.dtype),
                None, None
            )

        current_token_position = micro_cache_seqlens - 1 if micro_cache_seqlens is not None else None

        # Project Q, K, V
        query = self.module.q_proj(hidden_states)
        key = self.module.k_proj(hidden_states)
        value = self.module.v_proj(hidden_states)

        query = query.view(batch, 1, self.num_heads, self.head_dim)
        key = key.view(batch, 1, self.num_kv_heads, self.head_dim)
        value = value.view(batch, 1, self.num_kv_heads, self.head_dim)

        # QK-norm
        query, key = self._apply_qk_norm(query, key)

        # RoPE — cos/sin need shape [B, 1, 1, D] for [B, S, H, D] tensors
        max_seqlen = AttnWrapperBase.max_seqlen or 1
        cos, sin = self.module.rotary_emb(value.transpose(1, 2), seq_len=max_seqlen)
        if current_token_position is not None:
            # cos[positions] → [B, D], unsqueeze twice for seq and head dims
            cos = cos[current_token_position].unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D]
            sin = sin[current_token_position].unsqueeze(1).unsqueeze(1)
        else:
            cos = cos[:1].unsqueeze(0)  # [1, 1, 1, D]
            sin = sin[:1].unsqueeze(0)

        query, key = self._apply_rotary(query, key, cos, sin)

        # Write new K, V to paged GPU cache
        # Shape: [batch, seq_len=1, num_heads, head_dim]
        gpu_kv_manager.update_layer_decode_new_token(
            k_tensor=key,
            v_tensor=value,
            sequence_lengths=current_token_position if current_token_position is not None else torch.zeros(batch, dtype=torch.int32, device=hidden_states.device),
            layer_idx=self.layer_idx,
            batch_slice=batch_slice,
        )

        # Retrieve paged K, V cache and page table for attention
        k_cache_layer, v_cache_layer, page_table = gpu_kv_manager.get_layer_kv_with_page_table(
            self.layer_idx
        )

        if batch_slice is not None:
            start_idx, end_idx = batch_slice
            page_table = page_table[start_idx:end_idx]

        # GQA decode attention with paged KV
        # batchgen_gqa_decode_bf16 expects q=[B, 1, H, D], not [B, H, 1, D]
        attn_output, _ = batchgen_gqa_decode_bf16(
            q=query,  # [batch, 1, num_heads, head_dim]
            k_cache=k_cache_layer,
            v_cache=v_cache_layer,
            cache_seqlens=micro_cache_seqlens,
            block_table=page_table,
            sinks=None,
            softmax_scale=self.scale,
        )

        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, 1, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.module.o_proj(attn_output)

        return attn_output, None, None

    def _apply_rotary(self, query, key, cos, sin):
        """Apply RoPE to query and key. Handles [B, S, H, D] format."""
        q_embed = (query * cos) + (rotate_half(query) * sin)
        k_embed = (key * cos) + (rotate_half(key) * sin)
        return q_embed, k_embed


class Qwen3MLPWrapper(ExpertWrapperBase):
    """MLP wrapper for Qwen3 dense model (no quantization).

    Standard SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    All weights in BF16.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
    ):
        # expert_idx=0 for dense model (single MLP per layer)
        super().__init__(
            module, layer_idx, expert_idx=0, core_engine=core_engine,
            engine_config=engine_config, model_config=model_config,
            persistent=persistent
        )

    def _build_module_key(self) -> str:
        """Dense MLP uses mlp_{layer} key."""
        return f"mlp_{self.layer_idx}"

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """No-op for BF16 weights."""
        return weights_dict

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard SwiGLU forward."""
        return self.module(hidden_states)
