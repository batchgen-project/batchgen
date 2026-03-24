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

"""MiniMax-M2.5 model implementation for BatchGen.

Key features:
- 62 layers, hidden_size=3072
- GQA: 48 Q heads, 8 KV heads, head_dim=128
- QK Norm: per-layer RMSNorm on Q/K projections before RoPE
- Partial RoPE: rotary_dim=64 (50% of head_dim), theta=5M, no scaling
- 256 experts, Top-8 sigmoid routing with e_score_correction_bias
- FP8 e4m3fn quantization, block_size [128, 128]
- All 62 layers are MoE (no dense layers)
- Standard SwiGLU: silu(gate) * up -> down
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl

from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import MiniMaxM25Config

# CUDA routing kernels
try:
    from batchgen.moe.routing import gate_topk_sigmoid_cuda
    _HAS_CUDA_ROUTING = True
except ImportError:
    _HAS_CUDA_ROUTING = False

# FP8 grouped GEMM kernels (TMA-based, same as DeepSeek-V3)
try:
    from batchgen.gemm.w8a8_grouped_gemm_stage_1 import fused_fp8_moe_stage_1_tma
    from batchgen.moe.fused_grouped_dequant_gemm import fused_dequant_grouped_gemm_fp8_tma
    from batchgen.attention.mla.fa3_backend import act_quant, w8a16_gemm
    from batchgen_kernels.common.mgn import compact_expert_data, fused_moe_token_dispatch
    _HAS_FP8_GROUPED = True
except ImportError:
    _HAS_FP8_GROUPED = False

# FP8 blockwise grouped GEMM (CuTe persistent kernel)
try:
    from batchgen.moe.grouped_fp8_blockwise_moe import (
        grouped_fp8_blockwise_s1_silu,
        grouped_fp8_blockwise_s3,
    )
    _HAS_FP8_BLOCKWISE = True
except ImportError:
    _HAS_FP8_BLOCKWISE = False

# 3D dispatch scatter + CUDA reduce (K2.5 pattern)
try:
    from batchgen.moe.dispatch_scatter_3d import dispatch_scatter_3d, reduce_weighted_scatter
    _HAS_DISPATCH_3D = True
except ImportError:
    _HAS_DISPATCH_3D = False


# ============================================================================
# MoE Triton Kernels (self-contained — no cross-model imports)
# ============================================================================

@triton.jit
def _moe_fp32_accum_kernel_v2(
    outs_ptr,
    inv_idxs_ptr,
    topk_weights_ptr,
    output_ptr,
    total_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """FP32 accumulation kernel for MoE prefill scatter-reduce."""
    token_block_id = tl.program_id(0)
    h_block_id = tl.program_id(1)

    TOKENS_PER_BLOCK: tl.constexpr = 4
    token_start = token_block_id * TOKENS_PER_BLOCK

    h_start = h_block_id * BLOCK_SIZE
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < hidden_dim

    for t_idx in range(TOKENS_PER_BLOCK):
        token_id = token_start + t_idx
        if token_id < total_tokens:
            accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for k in range(topk):
                new_x_idx = token_id * topk + k
                outs_idx = tl.load(inv_idxs_ptr + new_x_idx)
                weight_offset = token_id * topk + k
                weight = tl.load(topk_weights_ptr + weight_offset).to(tl.float32)
                outs_offsets = outs_idx * hidden_dim + h_offsets
                expert_out = tl.load(outs_ptr + outs_offsets, mask=h_mask, other=0.0)
                accum += expert_out.to(tl.float32) * weight
            output_offsets = token_id * hidden_dim + h_offsets
            tl.store(output_ptr + output_offsets, accum.to(output_ptr.dtype.element_ty), mask=h_mask)


def moe_fp32_accum_triton_v2(
    outs: torch.Tensor,
    idxs: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """FP32 accumulation for MoE prefill: weighted scatter-reduce of expert outputs."""
    total_tokens, topk = topk_weights.shape
    hidden_dim = outs.shape[1]

    inv_idxs = torch.empty_like(idxs)
    inv_idxs[idxs] = torch.arange(len(idxs), device=idxs.device, dtype=idxs.dtype)

    output = torch.empty((total_tokens, hidden_dim), device=outs.device, dtype=outs.dtype)

    BLOCK_SIZE = 128
    TOKENS_PER_BLOCK = 4

    grid = lambda META: (
        triton.cdiv(total_tokens, TOKENS_PER_BLOCK),
        triton.cdiv(hidden_dim, META['BLOCK_SIZE'])
    )

    _moe_fp32_accum_kernel_v2[grid](
        outs, inv_idxs, topk_weights, output,
        total_tokens=total_tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@triton.jit
def _scatter_weight_reduce_kernel(
    res_ptr,
    nnz_indices_ptr,
    topk_weight_ptr,
    output_ptr,
    num_tokens,
    num_experts_per_tok,
    hidden_size,
    nnz,
    BLOCK_SIZE_H: tl.constexpr,
):
    """Optimized scatter-reduce using pre-computed inverse mapping."""
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return

    h_offset = tl.program_id(1) * BLOCK_SIZE_H
    h_indices = h_offset + tl.arange(0, BLOCK_SIZE_H)
    h_mask = h_indices < hidden_size

    accumulator = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)

    for k in range(num_experts_per_tok):
        mapping_offset = token_idx * num_experts_per_tok + k
        nnz_idx = tl.load(nnz_indices_ptr + mapping_offset)
        is_valid = (nnz_idx >= 0) & (nnz_idx < nnz)
        weight = tl.load(topk_weight_ptr + mapping_offset)
        safe_nnz_idx = tl.where(is_valid, nnz_idx, 0)
        res_offset = safe_nnz_idx * hidden_size + h_indices
        load_mask = h_mask & is_valid
        res_vals = tl.load(res_ptr + res_offset, mask=load_mask, other=0.0)
        res_vals_fp32 = res_vals.to(tl.float32)
        weighted = tl.where(is_valid, res_vals_fp32 * weight, 0.0)
        accumulator += weighted

    output_offset = token_idx * hidden_size + h_indices
    tl.store(output_ptr + output_offset, accumulator, mask=h_mask)


def _build_inverse_mapping(
    global_indices: torch.Tensor,
    token_topk_pos: torch.Tensor,
    num_tokens: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Build inverse mapping: [num_tokens, num_experts_per_tok] -> nnz_idx."""
    mapping = torch.full((num_tokens, num_experts_per_tok), -1,
                         dtype=torch.int64, device=global_indices.device)
    if global_indices.numel() == 0:
        return mapping
    mapping[global_indices, token_topk_pos] = torch.arange(
        len(global_indices), dtype=torch.int64, device=global_indices.device
    )
    return mapping


def scatter_weight_reduce_optimized(
    res: torch.Tensor,
    global_indices: torch.Tensor,
    token_topk_pos: torch.Tensor,
    topk_weight: torch.Tensor,
    num_tokens: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Optimized scatter-reduce for EP decode MoE combine step."""
    assert topk_weight.dtype == torch.float32, "topk_weight must be float32"
    assert topk_weight.shape == (num_tokens, num_experts_per_tok), \
        f"topk_weight shape mismatch, expected ({num_tokens}, {num_experts_per_tok}), got {topk_weight.shape}"

    nnz, hidden_size = res.shape

    if nnz == 0:
        return torch.zeros((num_tokens, hidden_size), device=res.device, dtype=torch.float32)

    global_indices_sliced = global_indices[:nnz]
    token_topk_pos_sliced = token_topk_pos[:nnz]

    nnz_indices = _build_inverse_mapping(
        global_indices_sliced, token_topk_pos_sliced, num_tokens, num_experts_per_tok
    )

    output = torch.zeros((num_tokens, hidden_size), device=res.device, dtype=torch.float32)

    if num_tokens == 0:
        return output

    BLOCK_SIZE_H = min(triton.next_power_of_2(hidden_size), 256)
    grid = (num_tokens, triton.cdiv(hidden_size, BLOCK_SIZE_H))

    _scatter_weight_reduce_kernel[grid](
        res, nnz_indices, topk_weight,
        output,
        num_tokens, num_experts_per_tok, hidden_size, nnz,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )

    return output


# ============================================================================
# Decode Layer Timing Infrastructure
# ============================================================================

@dataclass
class LayerTimingStats:
    layer_idx: int
    attn_ms: float = 0.0
    moe_ms: float = 0.0
    moe_router_ms: float = 0.0
    moe_dispatch_ms: float = 0.0
    moe_gemm_ms: float = 0.0
    moe_combine_ms: float = 0.0


class DecodeLayerTiming:
    enabled: bool = os.environ.get("BATCHGEN_LAYER_TIMING", "0") == "1"
    layer_stats: List[LayerTimingStats] = []
    current_layer: Optional[LayerTimingStats] = None
    _iteration_count: int = 0

    @classmethod
    def start_layer(cls, layer_idx: int):
        if cls.enabled:
            cls.current_layer = LayerTimingStats(layer_idx=layer_idx)

    @classmethod
    def record_attn(cls, elapsed_ms: float):
        if cls.enabled and cls.current_layer:
            cls.current_layer.attn_ms = elapsed_ms

    @classmethod
    def record_moe(cls, elapsed_ms: float):
        if cls.enabled and cls.current_layer:
            cls.current_layer.moe_ms = elapsed_ms

    @classmethod
    def record_moe_component(cls, component: str, elapsed_ms: float):
        if cls.enabled and cls.current_layer:
            setattr(cls.current_layer, f"moe_{component}_ms", elapsed_ms)

    @classmethod
    def end_layer(cls):
        if cls.enabled and cls.current_layer:
            cls.layer_stats.append(cls.current_layer)
            cls.current_layer = None

    @classmethod
    def print_summary(cls):
        if not cls.layer_stats:
            return
        cls._iteration_count += 1
        total_attn = sum(s.attn_ms for s in cls.layer_stats)
        total_moe = sum(s.moe_ms for s in cls.layer_stats)
        total_time = total_attn + total_moe
        num_layers = len(cls.layer_stats)
        print(f"\n=== Decode Timing (iter {cls._iteration_count}, {num_layers} layers) ===")
        print(f"Total: {total_time:.2f} ms ({1000/total_time:.1f} tokens/sec)")
        print(f"  Attention: {total_attn:.2f} ms ({100*total_attn/total_time:.1f}%)")
        print(f"  MoE:       {total_moe:.2f} ms ({100*total_moe/total_time:.1f}%)")
        print(f"\nPer-layer (first 3):")
        for s in cls.layer_stats[:3]:
            print(f"  L{s.layer_idx}: attn={s.attn_ms:.2f}ms, moe={s.moe_ms:.2f}ms")
        cls.layer_stats.clear()

    @classmethod
    def reset(cls):
        cls.layer_stats.clear()
        cls.current_layer = None
        cls._iteration_count = 0


# ============================================================================
# RMSNorm
# ============================================================================

class MiniMaxM25RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from batchgen.attention.fused_kernels import cuda_rmsnorm
        return cuda_rmsnorm(x, self.weight, self.eps)


# ============================================================================
# Rotary Position Embedding (plain, partial rotation)
# ============================================================================

class MiniMaxM25RotaryEmbedding(nn.Module):
    """Plain RoPE for partial rotation (rotary_dim=64 of head_dim=128).

    No YaRN scaling, no concentration factor. theta=5M.
    cos/sin shape: [seq_len, rotary_dim] — only covers the rotated dims.
    """

    def __init__(self, rotary_dim: int, max_position_embeddings: int = 196608,
                 base: float = 5000000.0, device=None):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

        inv_freq = 1.0 / (base ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # cos/sin cache is deferred to first forward() call — avoids heavy
        # CPU trig computation during model init (runs on GPU instead).

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        # [seq_len, rotary_dim] — cos/sin for the rotated dimensions only
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, rotary_dim]
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if self.cos_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_dim):
    """Apply partial rotary position embedding.

    Only the first rotary_dim dimensions of q/k are rotated.
    The remaining dimensions pass through unchanged.

    Args:
        q: [batch, seq, num_heads, head_dim]
        k: [batch, seq, num_kv_heads, head_dim]
        cos: [seq, rotary_dim]  (already doubled via cat)
        sin: [seq, rotary_dim]
        rotary_dim: number of dimensions to rotate (64)
    """
    cos = cos.unsqueeze(1)  # [seq, 1, rotary_dim]
    sin = sin.unsqueeze(1)

    # Split into rotary and passthrough parts
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    # Apply RoPE to rotary part
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concat rotary and passthrough
    q_out = torch.cat([q_embed, q_pass], dim=-1)
    k_out = torch.cat([k_embed, k_pass], dim=-1)
    return q_out, k_out


# ============================================================================
# GQA Attention with QK Norm and Partial RoPE
# ============================================================================

class MiniMaxM25Attention(nn.Module):
    """Grouped Query Attention with QK norm and partial RoPE.

    - 48 query heads, 8 KV heads (GQA with 6:1 ratio)
    - Head dim = 128
    - QK norm: RMSNorm on full projected Q/K before reshape and RoPE
    - Partial RoPE: rotate first 64 of 128 dims, passthrough rest
    - No sink tokens, no sliding window

    Forward accepts only hidden_states. Masks, position IDs, and KV caches
    are managed by AttnWrapper in production.
    """

    def __init__(self, config: MiniMaxM25Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Projections (no bias)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK Norm — applied to full projected Q/K (before reshape)
        self.q_norm = MiniMaxM25RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM25RMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)

        # RoPE — shared instance assigned by model init
        self.rotary_emb = None

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Apply QK norm (before reshape, on full projected dims)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Reshape: [bsz, seq, num_heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Get RoPE embeddings
        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        else:
            cos = cos[:q_len]
            sin = sin[:q_len]

        # Apply partial RoPE (rotate first 64 dims, passthrough last 64)
        query_states, key_states = apply_partial_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rotary_dim
        )

        # Transpose for attention: [bsz, heads, seq, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled dot-product attention with causal mask
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        kv_len = key_states.shape[-2]
        causal_mask = torch.triu(
            torch.ones((q_len, kv_len), dtype=torch.bool, device=attn_weights.device),
            diagonal=kv_len - q_len + 1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# ============================================================================
# Expert FFN (standard SwiGLU, no bias)
# ============================================================================

class MiniMaxM25Expert(nn.Module):
    """Single expert with standard SwiGLU: silu(gate(x)) * up(x) -> down."""

    def __init__(self, config: MiniMaxM25Config):
        super().__init__()
        # M2.5 uses w1=gate, w2=down, w3=up naming in HF checkpoint
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    @torch.inference_mode()
    def deepgemm_forward(self, x, scale):
        """Forward using deep_gemm FP8 GEMM (w8a16: FP8 weight × BF16 activation)."""
        up = w8a16_gemm(self.up_proj.weight.data, scale['up_proj.weight_scale_inv'], x)
        gate = w8a16_gemm(self.gate_proj.weight.data, scale['gate_proj.weight_scale_inv'], x)
        intermediate = F.silu(gate) * up
        return w8a16_gemm(self.down_proj.weight.data, scale['down_proj.weight_scale_inv'], intermediate)


# ============================================================================
# MoE Buffer Manager (K2.5 pattern: 3D strided layout)
# ============================================================================

_DEFAULT_MTP = 64  # fixed max_tokens_padded for decode batch sizes


class MiniMaxM25MoEBufferManager:
    """Pre-allocated buffers for MiniMax MoE decode pipeline (3D strided layout).

    One instance per model, shared across all 62 MoE layers via class variable.
    Follows K2.5 pattern: dispatch_scatter_3d writes tokens directly into
    fixed-stride slots, GEMM operates inplace, reduce reads via topk_pos.

    Buffer layout: [E_local * max_tokens_padded, dim] (3D strided).
    Each expert e owns rows [e * mtp, (e+1) * mtp) in the activation buffer.
    """

    def __init__(
        self,
        E_local: int,
        max_global_bsz: int,
        H: int,
        N_inter: int,
        topk: int,
        num_tokens_per_rank: int,
        device: torch.device,
        max_tokens_padded: int = _DEFAULT_MTP,
    ):
        self.E_local = E_local
        self.H = H
        self.N_inter = N_inter
        self.topk = topk
        self.max_global_bsz = max_global_bsz
        self.num_tokens_per_rank = num_tokens_per_rank
        self.device = device
        self.max_tokens_padded = max_tokens_padded

        NK = max_global_bsz * topk
        buf_rows = E_local * max_tokens_padded

        # Communication buffers
        self.all_tokens = torch.zeros(max_global_bsz, H, dtype=torch.bfloat16, device=device)
        self.padded = torch.zeros(num_tokens_per_rank, H, dtype=torch.bfloat16, device=device)

        # Routing metadata
        self.expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)

        # 3D strided GEMM buffers
        self.dispatched_x = torch.zeros(buf_rows, H, dtype=torch.bfloat16, device=device)
        self.expert_out = torch.zeros(buf_rows, H, dtype=torch.bfloat16, device=device)

        # Result buffer
        self.result_buffer = torch.empty(max_global_bsz, H, dtype=torch.bfloat16, device=device)

        logging.info(
            f"[MoEBufferManager] 3D strided: E_local={E_local}, mtp={max_tokens_padded}, "
            f"buf_rows={buf_rows}, H={H}, N_inter={N_inter}, "
            f"total={self._total_bytes() / (1024**3):.2f} GiB"
        )

    def resize_if_needed(self, global_bsz: int):
        """Resize communication/routing buffers if global_bsz exceeds capacity."""
        if global_bsz <= self.max_global_bsz:
            return
        logging.info(f"[MoEBufferManager] Resizing: {self.max_global_bsz} -> {global_bsz}")
        self.max_global_bsz = global_bsz
        NK = global_bsz * self.topk
        self.all_tokens = torch.zeros(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=self.device)
        self.result_buffer = torch.empty(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)

    def _total_bytes(self):
        total = 0
        for attr in ['all_tokens', 'padded', 'expert_counts', 'expert_counters',
                      'topk_pos', 'dispatched_x', 'expert_out', 'result_buffer']:
            t = getattr(self, attr)
            total += t.nelement() * t.element_size()
        return total


# ============================================================================
# MoE Layer with Sigmoid Routing + Correction Bias
# ============================================================================

class _ExpertPlaceholder:
    """Lightweight placeholder for expert slots.

    Avoids creating 15872 nn.Module objects (256 experts × 62 layers) during init.
    Replaced by MiniMaxM25ExpertWrapper during _config_expert_module().
    """
    pass


class MiniMaxM25MoE(nn.Module):
    """MoE with sigmoid routing and e_score_correction_bias.

    256 experts, Top-8 selection, no shared experts. Routing:
    1. scores = sigmoid(gate(x)) + e_score_correction_bias
    2. Select top-8 by score
    3. weights = sigmoid(gate(x))[selected]  (original, without bias)
    4. weights /= weights.sum()  (renormalize)

    Three forward paths:
    - Prefill: sequential expert loop via wrappers
    - EP decode (all-persistent): AllGather → gate → dispatch → grouped FP8 GEMM → reduce → AllReduce
    - EP decode (offloading): AllGather → gate → per-expert loop with dynamic weight loading → AllReduce
    """

    def __init__(self, config: MiniMaxM25Config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts  # 256
        self.num_experts_per_tok = config.num_experts_per_tok  # 8
        self.hidden_size = config.hidden_size

        # Router gate (no bias on nn.Linear, but has separate correction bias)
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts),
        )

        # Expert placeholders (replaced by wrappers during PSM config)
        self.experts = [_ExpertPlaceholder() for _ in range(self.num_experts)]

        # EP decode state (set by PSM)
        self.comm = None
        self.device = None
        self.num_tokens_per_rank = None
        self.max_num_tokens_per_rank = None
        self.persistent_expert_ids = []
        self.nonpersistent_expert_ids = []

        # Distributed metadata
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.experts_per_rank = self.num_experts // self.world_size
        self.routed_expert_start_idx = self.rank * self.experts_per_rank
        self.routed_expert_end_idx = (self.rank + 1) * self.experts_per_rank

        # FP8 grouped GEMM pointer arrays (built by init())
        self.gate_list = None
        self.up_list = None
        self.down_list = None
        self.gate_ptrs_ptr = None
        self.up_ptrs_ptr = None
        self.down_ptrs_ptr = None
        self.gate_scale_list = None
        self.up_scale_list = None
        self.down_scale_list = None
        self.gate_scale_ptrs_ptr = None
        self.up_scale_ptrs_ptr = None
        self.down_scale_ptrs_ptr = None

        # Token dispatch buffers (built by init_num_tokens())
        self.token_idx = None
        self.topk_pos = None

    # Class-level shared buffer (one per model, shared across all 62 MoE layers)
    _buf: Optional[MiniMaxM25MoEBufferManager] = None

    def init_num_tokens(self, num_tokens_per_rank):
        """Allocate token dispatch buffers for EP decode."""
        self.num_tokens_per_rank = num_tokens_per_rank
        self.max_num_tokens_per_rank = num_tokens_per_rank
        global_num_tokens = self.num_tokens_per_rank * self.world_size
        K = self.num_experts_per_tok

        # Legacy dispatch buffers (for fused_moe_token_dispatch fallback)
        self.token_idx = torch.arange(
            global_num_tokens, dtype=torch.int32, device=self.device
        ).repeat_interleave(K)
        self.topk_pos = torch.arange(
            K, dtype=torch.int32, device=self.device
        ).repeat(global_num_tokens)

        # 3D buffer manager (K2.5 pattern)
        if _HAS_DISPATCH_3D and self.__class__._buf is None:
            self.__class__._buf = MiniMaxM25MoEBufferManager(
                E_local=self.experts_per_rank,
                max_global_bsz=global_num_tokens,
                H=self.hidden_size,
                N_inter=self.config.moe_intermediate_size,
                topk=K,
                num_tokens_per_rank=num_tokens_per_rank,
                device=self.device,
                max_tokens_padded=_DEFAULT_MTP,
            )

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        """Dynamically update num_tokens_per_rank for reduced communication."""
        if num_tokens_per_rank == self.num_tokens_per_rank:
            return
        if hasattr(self, 'max_num_tokens_per_rank') and num_tokens_per_rank > self.max_num_tokens_per_rank:
            self.max_num_tokens_per_rank = num_tokens_per_rank
        self.num_tokens_per_rank = num_tokens_per_rank
        global_num_tokens = self.num_tokens_per_rank * self.world_size
        K = self.num_experts_per_tok
        self.token_idx = torch.arange(
            global_num_tokens, dtype=torch.int32, device=self.device
        ).repeat_interleave(K)
        self.topk_pos = torch.arange(
            K, dtype=torch.int32, device=self.device
        ).repeat(global_num_tokens)

        # Resize buf.padded to match new num_tokens_per_rank (K2.5 pattern)
        buf = self.__class__._buf
        if buf is not None and buf.padded.shape[0] != num_tokens_per_rank:
            buf.padded = torch.zeros(
                num_tokens_per_rank, buf.H,
                dtype=torch.bfloat16, device=buf.device,
            )
            buf.num_tokens_per_rank = num_tokens_per_rank

    def init(self, micro_batch_size):
        """Build FP8 weight pointer arrays for grouped GEMM.

        Called by PSM after expert wrappers are configured and weights loaded.
        Collects fp8_gate/fp8_up/fp8_down and their scale factors from each
        persistent expert wrapper into contiguous pointer arrays for TMA kernels.
        """
        self.gate_list = []
        self.up_list = []
        self.down_list = []
        self.gate_scale_list = []
        self.up_scale_list = []
        self.down_scale_list = []
        for e in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
            self.gate_list.append(self.experts[e].fp8_gate)
            self.up_list.append(self.experts[e].fp8_up)
            self.down_list.append(self.experts[e].fp8_down)
            self.gate_scale_list.append(
                self.experts[e].weight_dequant_scale['w1.weight_scale_inv']
            )
            self.up_scale_list.append(
                self.experts[e].weight_dequant_scale['w3.weight_scale_inv']
            )
            self.down_scale_list.append(
                self.experts[e].weight_dequant_scale['w2.weight_scale_inv']
            )

        self.gate_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.gate_list], dtype=torch.int64, device=self.device
        )
        self.up_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.up_list], dtype=torch.int64, device=self.device
        )
        self.down_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.down_list], dtype=torch.int64, device=self.device
        )
        self.gate_scale_ptrs_ptr = torch.tensor(
            [s.data_ptr() for s in self.gate_scale_list], dtype=torch.int64, device=self.device
        )
        self.up_scale_ptrs_ptr = torch.tensor(
            [s.data_ptr() for s in self.up_scale_list], dtype=torch.int64, device=self.device
        )
        self.down_scale_ptrs_ptr = torch.tensor(
            [s.data_ptr() for s in self.down_scale_list], dtype=torch.int64, device=self.device
        )

        # Stack FP8 weights for blockwise grouped GEMM kernel
        # gate/up: [E_local, N=moe_intermediate_size, K=hidden_size] fp8
        # down:    [E_local, N=hidden_size, K=moe_intermediate_size] fp8
        self._init_fp8_blockwise_weights()

    def _init_fp8_blockwise_weights(self):
        """Stack per-expert FP8 weights into 3D tensors for blockwise GEMM.

        Creates [E_local, out_dim, in_dim] contiguous fp8 weight tensors
        and [E_local, out_dim/128, (in_dim/128+3)//4*4] padded scale tensors.
        Called once during init() after pointer arrays are built.
        """
        E = self.experts_per_rank
        K = self.hidden_size  # 3072
        N = self.config.moe_intermediate_size  # 1536
        scale_block = 128

        k_blocks = K // scale_block
        n_blocks = N // scale_block
        k_blocks_pad4 = (k_blocks + 3) // 4 * 4
        n_blocks_pad4 = (n_blocks + 3) // 4 * 4

        # Gate: [E, N, K] fp8 — gate_proj maps K→N
        self.fp8_gate_w3d = torch.stack(self.gate_list).contiguous()
        # Up: [E, N, K] fp8
        self.fp8_up_w3d = torch.stack(self.up_list).contiguous()
        # Down: [E, K, N] fp8 — down_proj maps N→K
        self.fp8_down_w3d = torch.stack(self.down_list).contiguous()

        # Gate weight scales: [E, N/128, (K/128+3)//4*4]
        self.fp8_gate_ws3d = torch.zeros(
            E, n_blocks, k_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(self.gate_scale_list):
            self.fp8_gate_ws3d[i, :, :k_blocks] = s

        # Up weight scales: [E, N/128, (K/128+3)//4*4]
        self.fp8_up_ws3d = torch.zeros(
            E, n_blocks, k_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(self.up_scale_list):
            self.fp8_up_ws3d[i, :, :k_blocks] = s

        # Down weight scales: [E, K/128, (N/128+3)//4*4]
        self.fp8_down_ws3d = torch.zeros(
            E, k_blocks, n_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(self.down_scale_list):
            self.fp8_down_ws3d[i, :, :n_blocks] = s

        # Pre-computed cu_seqlens for reserved buffer layout
        # mtp = max_tokens_padded (set later by init_num_tokens)
        self._fp8_blockwise_ready = True

        logging.info(
            f"[MoE] FP8 blockwise weights stacked: "
            f"gate={list(self.fp8_gate_w3d.shape)}, "
            f"down={list(self.fp8_down_w3d.shape)}, "
            f"gate_scale={list(self.fp8_gate_ws3d.shape)}"
        )

    def _gate_sigmoid_topk(self, hidden_states_2d):
        """Sigmoid routing with correction bias on 2D input [N, hidden_size].

        Returns:
            topk_idx: [N, num_experts_per_tok] int32
            topk_weight: [N, num_experts_per_tok] float32
        """
        router_logits = self.gate(hidden_states_2d.to(self.gate.weight.dtype)).to(hidden_states_2d.dtype)

        if _HAS_CUDA_ROUTING:
            topk_idx, topk_weight = gate_topk_sigmoid_cuda(
                router_logits, k=self.num_experts_per_tok,
                e_score_correction=self.e_score_correction_bias,
            )
        else:
            routing_weights = torch.sigmoid(router_logits.float())
            scores_for_choice = routing_weights + self.e_score_correction_bias
            _, topk_idx = torch.topk(
                scores_for_choice, self.num_experts_per_tok, dim=-1, sorted=False
            )
            topk_weight = routing_weights.gather(1, topk_idx)
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        return topk_idx.to(torch.int32), topk_weight.float()

    def grouped_dequant_moe_fp8(self, x, eids, expert_counts, expert_offsets):
        """Process dispatched tokens through local experts with FP8 grouped GEMM.

        Uses CuTe blockwise kernel when available (2.5× faster decode),
        falls back to Triton TMA kernel otherwise.

        Input x is in compacted layout from fused_moe_token_dispatch.
        For blockwise kernel: scatter to uniform [E*mtp, dim] stride, call kernel,
        gather results back to compacted layout.
        """
        actual_num_tokens = expert_offsets[-1]
        if isinstance(actual_num_tokens, torch.Tensor):
            actual_num_tokens_val = actual_num_tokens.item()
        else:
            actual_num_tokens_val = int(actual_num_tokens)

        if actual_num_tokens_val == 0:
            return torch.empty(
                (0, self.hidden_size), device=x.device, dtype=torch.bfloat16
            )

        expert_counts_i32 = expert_counts.to(torch.int32)

        # ── FP8 Blockwise Kernel Path (CuTe persistent) ──
        if _HAS_FP8_BLOCKWISE and getattr(self, '_fp8_blockwise_ready', False):
            return self._grouped_dequant_fp8_blockwise(
                x, expert_counts_i32, expert_offsets, actual_num_tokens_val)

        # ── Fallback: Triton TMA Kernel ──
        return self._grouped_dequant_fp8_triton(
            x, expert_counts_i32, actual_num_tokens_val)

    def _grouped_dequant_fp8_blockwise(self, x, expert_counts, expert_offsets,
                                        actual_num_tokens_val):
        """FP8 blockwise grouped GEMM via CuTe persistent kernel.

        Scatters compacted tokens to uniform [E*mtp, dim] layout,
        runs blockwise GEMM, gathers back.
        """
        E = self.experts_per_rank
        K = self.hidden_size
        N = self.config.moe_intermediate_size
        device = x.device

        # Determine mtp (max tokens padded to 64)
        max_tok = int(expert_counts.max().item())
        mtp = max(((max_tok + 63) // 64) * 64, 64)

        # Build uniform cu_seqlens
        cu_seqlens = torch.arange(
            0, (E + 1) * mtp, mtp, dtype=torch.int32, device=device)

        # Scatter compacted x → uniform [E*mtp, K] layout
        x_uniform = torch.zeros(E * mtp, K, dtype=x.dtype, device=device)
        offsets = expert_offsets.cpu().tolist()
        counts = expert_counts.cpu().tolist()
        for e in range(E):
            m_e = counts[e]
            if m_e > 0:
                src = offsets[e]
                dst = e * mtp
                x_uniform[dst:dst + m_e] = x[src:src + m_e]

        # Quantize to FP8
        x_quant, x_scale = act_quant(x_uniform)

        # Transpose x_scale: [M, K/128] → [K/128, E*mtp]
        x_scale_t = x_scale.t().contiguous()

        seqlens = expert_counts[:E]
        avg = max(int(seqlens.float().mean().item()), 1)

        # S1: gate + up + SiLU
        intermediate = grouped_fp8_blockwise_s1_silu(
            x_quant.view(torch.float8_e4m3fn), x_scale_t,
            self.fp8_gate_w3d.view(torch.float8_e4m3fn),
            self.fp8_up_w3d.view(torch.float8_e4m3fn),
            self.fp8_gate_ws3d, self.fp8_up_ws3d,
            seqlens, cu_seqlens, avg,
        )

        # Re-quantize intermediate for S3
        inter_quant, inter_scale = act_quant(intermediate)
        inter_scale_t = inter_scale.t().contiguous()

        # S3: down projection
        result_uniform = grouped_fp8_blockwise_s3(
            inter_quant.view(torch.float8_e4m3fn), inter_scale_t,
            self.fp8_down_w3d.view(torch.float8_e4m3fn),
            self.fp8_down_ws3d,
            seqlens, cu_seqlens, avg,
        )

        # Gather back to compacted layout
        result = torch.empty(
            actual_num_tokens_val, K, dtype=torch.bfloat16, device=device)
        for e in range(E):
            m_e = counts[e]
            if m_e > 0:
                src = e * mtp
                dst = offsets[e]
                result[dst:dst + m_e] = result_uniform[src:src + m_e]

        return result

    def _grouped_dequant_fp8_triton(self, x, expert_counts, actual_num_tokens_val):
        """Fallback: Triton TMA kernel for FP8 grouped GEMM."""
        group_size, activated_group_idx, group_start_indices, num_active_experts = (
            compact_expert_data(expert_counts)
        )

        if isinstance(num_active_experts, torch.Tensor):
            num_active_val = num_active_experts.item()
        else:
            num_active_val = int(num_active_experts)

        if num_active_val == 0:
            return torch.empty(
                (0, self.hidden_size), device=x.device, dtype=torch.bfloat16
            )

        x_sliced = x[:actual_num_tokens_val]
        x_quant, x_scale = act_quant(x_sliced)

        intermediate = fused_fp8_moe_stage_1_tma(
            x_quant, x_scale,
            self.gate_list, self.gate_ptrs_ptr,
            self.up_list, self.up_ptrs_ptr,
            self.gate_scale_list, self.gate_scale_ptrs_ptr,
            self.up_scale_list, self.up_scale_ptrs_ptr,
            group_size, activated_group_idx, group_start_indices,
            num_active_experts, self.experts_per_rank,
        )

        intermediate, intermediate_scale = act_quant(intermediate)

        res = fused_dequant_grouped_gemm_fp8_tma(
            intermediate, intermediate_scale,
            self.down_list, self.down_ptrs_ptr,
            self.down_scale_list, self.down_scale_ptrs_ptr,
            group_size, activated_group_idx, group_start_indices,
            num_active_experts,
        )
        return res

    def _fp8_blockwise_gemm_3d(self, buf, expert_counts):
        """FP8 blockwise grouped GEMM on 3D strided buffer (no scatter/gather).

        Reads from buf.dispatched_x, writes to buf.expert_out.
        Uses CuTe persistent kernel with uniform cu_seqlens stride.
        """
        if not getattr(self.__class__, '_warned_gemm_3d', False):
            logging.warning("[MoE] HOT PATH: _fp8_blockwise_gemm_3d (CuTe persistent, no scatter/gather)")
            self.__class__._warned_gemm_3d = True
        E = self.experts_per_rank
        K = self.hidden_size
        N = self.config.moe_intermediate_size
        mtp = buf.max_tokens_padded

        cu_seqlens = torch.arange(
            0, (E + 1) * mtp, mtp, dtype=torch.int32, device=buf.dispatched_x.device)

        seqlens = expert_counts[:E]
        avg = max(int(seqlens.float().mean().item()), 1)

        # Quantize input
        x_quant, x_scale = act_quant(buf.dispatched_x[:E * mtp])
        x_scale_t = x_scale.t().contiguous()

        # S1: gate + up + SiLU
        intermediate = grouped_fp8_blockwise_s1_silu(
            x_quant.view(torch.float8_e4m3fn), x_scale_t,
            self.fp8_gate_w3d.view(torch.float8_e4m3fn),
            self.fp8_up_w3d.view(torch.float8_e4m3fn),
            self.fp8_gate_ws3d, self.fp8_up_ws3d,
            seqlens, cu_seqlens, avg,
        )

        # Re-quantize intermediate for S3
        inter_quant, inter_scale = act_quant(intermediate)
        inter_scale_t = inter_scale.t().contiguous()

        # S3: down projection → writes to expert_out buffer
        result = grouped_fp8_blockwise_s3(
            inter_quant.view(torch.float8_e4m3fn), inter_scale_t,
            self.fp8_down_w3d.view(torch.float8_e4m3fn),
            self.fp8_down_ws3d,
            seqlens, cu_seqlens, avg,
        )

        # Copy result to expert_out buffer for reduce
        buf.expert_out[:E * mtp].copy_(result[:E * mtp])

    @torch.inference_mode()
    def moe_infer_allgather_allreduce_bf16_acc(self, x):
        """EP decode: AllGather → gate → 3D dispatch → FP8 GEMM → reduce → AllReduce.

        All-persistent path. Uses K2.5 pattern: dispatch_scatter_3d + reduce_weighted_scatter.
        Falls back to fused_moe_token_dispatch path if dispatch_scatter_3d unavailable.
        """
        buf = self.__class__._buf
        num_tokens, hidden_size = x.shape
        device = x.device
        topk = self.num_experts_per_tok
        num_global = self.num_tokens_per_rank * self.world_size

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        # === K2.5 Pattern: 3D dispatch + CUDA reduce ===
        if buf is not None and _HAS_DISPATCH_3D and _HAS_FP8_BLOCKWISE \
                and getattr(self, '_fp8_blockwise_ready', False):
            if not getattr(self.__class__, '_warned_k25_path', False):
                logging.warning(
                    "[MoE] HOT PATH: dispatch_scatter_3d + reduce_weighted_scatter (K2.5 pattern)")
                self.__class__._warned_k25_path = True
            buf.resize_if_needed(num_global)

            # 1) AllGather into pre-allocated buffer
            all_tokens = buf.all_tokens[:num_global]
            padded = buf.padded
            padded.zero_()
            if num_tokens > 0:
                padded[:num_tokens] = x

            with self.comm.change_state(enable=True):
                self.comm.all_gather(
                    all_tokens, padded,
                    stream=torch.cuda.default_stream(device),
                )

            # 2) Gate: sigmoid routing
            topk_idx, topk_weight = self._gate_sigmoid_topk(all_tokens)

            # 3) 3D dispatch scatter into strided buffer
            buf.dispatched_x.zero_()
            expert_counts, topk_pos = dispatch_scatter_3d(
                all_tokens, topk_idx.to(torch.int32),
                buf.dispatched_x,
                self.routed_expert_start_idx, self.experts_per_rank,
                buf.max_tokens_padded,
                buf.expert_counts, buf.expert_counters,
                buf.topk_pos[:num_global * topk],
            )

            # 4) FP8 blockwise GEMM on 3D buffer
            self._fp8_blockwise_gemm_3d(buf, expert_counts)

            # 5) CUDA reduce: weighted scatter from 3D to flat
            result_buf = buf.result_buffer[:num_global]
            result_buf.zero_()
            global_results = reduce_weighted_scatter(
                buf.expert_out, topk_pos, topk_weight,
                num_global, hidden_size, topk,
                output=result_buf,
            )

            # 6) AllReduce
            with self.comm.change_state(enable=True):
                self.comm.all_reduce(
                    global_results, op=dist.ReduceOp.SUM,
                    stream=torch.cuda.default_stream(device),
                )

            # 7) Slice local tokens
            if num_tokens == 0:
                return torch.empty((0, hidden_size), device=device, dtype=x.dtype)
            start_token_ids = self.rank * self.num_tokens_per_rank
            end_token_ids = start_token_ids + num_tokens
            return global_results[start_token_ids:end_token_ids].to(x.dtype)

        # === Fallback: fused_moe_token_dispatch + Triton reduce ===
        if not getattr(self.__class__, '_warned_fallback', False):
            logging.warning(
                "[MoE] FALLBACK: fused_moe_token_dispatch + Triton reduce "
                f"(dispatch_3d={_HAS_DISPATCH_3D}, blockwise={_HAS_FP8_BLOCKWISE}, "
                f"ready={getattr(self, '_fp8_blockwise_ready', False)}, buf={buf is not None})")
            self.__class__._warned_fallback = True
        if num_tokens == 0:
            padded_hidden_states = torch.zeros(
                (self.num_tokens_per_rank, hidden_size),
                device=self.device, dtype=torch.bfloat16,
            )
        elif num_tokens < self.num_tokens_per_rank:
            padded_hidden_states = torch.zeros(
                (self.num_tokens_per_rank, hidden_size),
                device=self.device, dtype=x.dtype,
            )
            padded_hidden_states[:num_tokens] = x
        else:
            padded_hidden_states = x

        all_tokens = torch.zeros(
            (self.world_size * self.num_tokens_per_rank, hidden_size),
            device=self.device, dtype=torch.bfloat16,
        )
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded_hidden_states,
                stream=torch.cuda.default_stream(self.device),
            )

        topk_idx, topk_weight = self._gate_sigmoid_topk(all_tokens)

        (input_x, input_eids, global_indices, token_topk_pos,
         expert_counts, expert_offsets) = fused_moe_token_dispatch(
            all_tokens, topk_idx, self.token_idx, self.topk_pos,
            self.routed_expert_start_idx, self.routed_expert_end_idx,
        )

        res = self.grouped_dequant_moe_fp8(
            input_x, input_eids, expert_counts, expert_offsets,
        )

        global_results = scatter_weight_reduce_optimized(
            res, global_indices, token_topk_pos, topk_weight,
            num_global, self.num_experts_per_tok,
        )
        global_results = global_results.to(torch.bfloat16)

        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results, op=dist.ReduceOp.SUM,
                stream=torch.cuda.default_stream(self.device),
            )

        if num_tokens == 0:
            return torch.empty((0, hidden_size), device=device, dtype=x.dtype)
        start_token_ids = self.rank * self.num_tokens_per_rank
        end_token_ids = start_token_ids + num_tokens
        final_output = global_results[start_token_ids:end_token_ids]
        return final_output.to(x.dtype)

    @torch.inference_mode()
    def moe_infer_loop_with_offloading(self, x):
        """EP decode with offloading: AllGather → gate → per-expert loop → AllReduce.

        Used when some experts are non-persistent (loaded from host on demand).
        Each expert wrapper handles weight loading based on its persistent flag.
        """
        num_tokens, hidden_size = x.shape
        device = x.device

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        # 1) AllGather
        all_tokens = torch.zeros(
            (self.world_size * self.num_tokens_per_rank, self.hidden_size),
            device=self.device, dtype=torch.bfloat16,
        )
        padded_hidden_states = torch.zeros(
            (self.num_tokens_per_rank, hidden_size),
            device=self.device, dtype=x.dtype,
        )
        if num_tokens > 0:
            padded_hidden_states[:num_tokens] = x

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded_hidden_states,
                stream=torch.cuda.default_stream(self.device),
            )

        # 2) Gate on global tokens
        global_x = all_tokens
        topk_idx, topk_weight = self._gate_sigmoid_topk(global_x)

        # 3) Per-expert loop
        num_global_tokens = global_x.shape[0]
        K = self.num_experts_per_tok
        flat_expert_idx = topk_idx.view(-1)
        token_indices = torch.arange(num_global_tokens, device=device).repeat_interleave(K)
        topk_positions = torch.arange(K, device=device).repeat(num_global_tokens)

        global_results = torch.zeros(
            (num_global_tokens, hidden_size), device=device, dtype=torch.float32,
        )

        for local_e in range(self.experts_per_rank):
            global_e = self.routed_expert_start_idx + local_e
            mask = flat_expert_idx == global_e
            if not mask.any():
                continue
            expert_token_idx = token_indices[mask]
            expert_topk_pos = topk_positions[mask]
            tokens_for_expert = global_x[expert_token_idx]
            expert = self.experts[global_e]
            expert_output = expert(tokens_for_expert)
            expert_weights = topk_weight[expert_token_idx, expert_topk_pos]
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            global_results.index_add_(0, expert_token_idx, weighted_output)

        global_results = global_results.to(torch.bfloat16)

        # 4) AllReduce
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results, op=dist.ReduceOp.SUM,
                stream=torch.cuda.default_stream(self.device),
            )

        # 5) Slice local tokens
        start_token_ids = self.rank * self.num_tokens_per_rank
        end_token_ids = start_token_ids + num_tokens
        final_output = global_results[start_token_ids:end_token_ids]
        return final_output.to(x.dtype)

    def _forward_prefill(self, hidden_states):
        """Prefill: sorted expert loop with deepgemm FP8 GEMM + Triton FP32 accumulation.

        Follows DeepSeek-V3 prefill pattern:
        1. Gate → topk_idx, topk_weight
        2. Sort tokens by expert ID
        3. Sequential expert calls (each uses deepgemm w8a16)
        4. Triton kernel for weighted scatter-reduce (FP32 accumulation)
        """
        # moe_fp32_accum_triton_v2 is defined at module level (self-contained)

        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        topk_idx, topk_weight = self._gate_sigmoid_topk(hidden_flat)

        # Sort tokens by expert ID for sequential processing
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = hidden_flat[idxs // topk_idx.shape[1]]

        # Sequential expert processing with pre-allocated output
        total_tokens = int(tokens_per_expert.sum())
        if total_tokens > 0:
            outs = sorted_tokens.new_empty(total_tokens, hidden_dim)
            read_idx = 0
            write_idx = 0
            for i, num_tokens in enumerate(tokens_per_expert):
                if num_tokens == 0:
                    continue
                expert = self.experts[i]
                tokens_for_expert = sorted_tokens[read_idx:read_idx + num_tokens]
                expert_out = expert(tokens_for_expert)
                outs[write_idx:write_idx + num_tokens] = expert_out
                read_idx += num_tokens
                write_idx += num_tokens
        else:
            outs = sorted_tokens.new_empty(0, sorted_tokens.shape[-1])

        # Weighted accumulation via Triton kernel (FP32 precision)
        final_out = moe_fp32_accum_triton_v2(outs, idxs, topk_weight)
        return final_out.view(batch_size, seq_len, hidden_dim)

    @torch.inference_mode()
    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.comm is not None and self.num_tokens_per_rank is not None:
            # EP decode path
            if getattr(self, 'enable_ep_offloading', False):
                out = self.moe_infer_loop_with_offloading(hidden_states)
            else:
                out = self.moe_infer_allgather_allreduce_bf16_acc(hidden_states)
            return out.view(*orig_shape)
        else:
            # Prefill path (sequential loop)
            return self._forward_prefill(hidden_states.view(*orig_shape))


# ============================================================================
# Decoder Layer
# ============================================================================

class MiniMaxM25DecoderLayer(nn.Module):
    """Single transformer decoder layer with GQA attention + MoE.

    Pre-norm architecture with fused residual + norm between attention and MoE.
    Forward accepts only hidden_states.
    """

    def __init__(self, config: MiniMaxM25Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = MiniMaxM25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MiniMaxM25Attention(config, layer_idx)
        self.post_attention_layernorm = MiniMaxM25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MiniMaxM25MoE(config)

        # CUDA graph support (set externally)
        self.cuda_graph_manager = None
        self._full_attn_segment_name = None

    def enable_cuda_graph(self, manager, full_attn_name: str, moe_name: str = None):
        self.cuda_graph_manager = manager
        self._full_attn_segment_name = full_attn_name

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        DecodeLayerTiming.start_layer(self.layer_idx)

        # ========== ATTENTION ==========
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Fused residual add + post-attention layernorm
        from batchgen.attention.fused_kernels import cuda_add_rmsnorm
        hidden_states, residual = cuda_add_rmsnorm(
            residual, hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )

        # ========== MoE ==========
        timing_enabled = DecodeLayerTiming.enabled
        if timing_enabled:
            torch.cuda.synchronize()
            moe_start = time.perf_counter()

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if timing_enabled:
            torch.cuda.synchronize()
            DecodeLayerTiming.record_moe((time.perf_counter() - moe_start) * 1000)

        DecodeLayerTiming.end_layer()

        return hidden_states, None, None


# ============================================================================
# Model
# ============================================================================

class MiniMaxM25Model(nn.Module):
    """MiniMax-M2.5 inner transformer model.

    Contains embed_tokens, layers, and norm. No lm_head.
    Forward returns tuple: (hidden_states, next_cache, all_hidden_states, all_self_attns).
    """

    def __init__(self, config: MiniMaxM25Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Shared RoPE (partial: rotary_dim=64)
        self._shared_rotary_emb = MiniMaxM25RotaryEmbedding(
            rotary_dim=config.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [MiniMaxM25DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        for layer in self.layers:
            layer.self_attn.rotary_emb = self._shared_rotary_emb

        self.norm = MiniMaxM25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        hidden_states = inputs_embeds

        next_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        DecodeLayerTiming.print_summary()

        return (hidden_states, next_cache, None, None)


class MiniMaxM25(nn.Module):
    """MiniMax-M2.5 model with language modeling head.

    Outer wrapper following GPT-OSS pattern:
    - self.model = MiniMaxM25Model (inner transformer)
    - self.lm_head = nn.Linear (unembedding)
    - Forward returns CausalLMOutputWithPast
    """

    def __init__(self, config: MiniMaxM25Config):
        super().__init__()
        self.config = config
        self.model = MiniMaxM25Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs[1],
            hidden_states=outputs[2],
            attentions=outputs[3],
        )
