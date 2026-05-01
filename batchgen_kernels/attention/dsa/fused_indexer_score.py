"""Fused Indexer Scoring — H20 (SM90a) Only

Version: v3 — CUDA WGMMA wq_b + CUDA RoPE+Hadamard + Triton fused scoring
Hypothesis: Replace PyTorch RoPE+Hadamard with existing CUDA fused kernel
Result: TBD

Pipeline:
  q_a [B, 2048] → CUDA WGMMA wq_b → [B, 4096] → reshape [B, 32, 128]
  → CUDA fused RoPE (interleaved) + Hadamard [128×128]
  → Triton fused Q×K scoring + head_gate + sum + topk

Dimensions (GLM-5):
  B = 32 (decode batch)
  n_heads = 32 (indexer heads)
  head_dim = 128
  q_lora_rank = 2048
  wq_b: [4096, 2048] — M=B, K=2048, N=4096
  max_seqlen = variable (up to 10240+)
  topk = 2048

v1 baseline (scoring kernel only): 15-21× over Torch
v2: CUDA WGMMA wq_b + PyTorch RoPE/Hadamard — 8-17× full pipeline
"""

import torch
import triton
import triton.language as tl
import math

from batchgen_kernels.attention.dsa.fast_topk_cuda import (
    fast_topk_2048,
    fast_topk_2048_out,
)
from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
    build_module,
    FP8IndexerWeightsCUDA,
    _validate_projection_out_buffers,
)

# Import existing CUDA fused RoPE+Hadamard kernel
from batchgen_kernels.attention.dsa.indexer import (
    fused_rope_hadamard as _cuda_fused_rope_hadamard,
    fused_rope_hadamard_out as _cuda_fused_rope_hadamard_out,
)


# ============================================================
# Weight container for wq_b (reuses WP2 CUDA WGMMA infra)
# ============================================================

class FP8WqbWeightsCUDA:
    """Pre-quantized wq_b weights for CUDA WGMMA kernel.

    wq_b: [N=4096, K=2048] — same FP8 block quantization as WP2.
    Reuses FP8IndexerWeightsCUDA which handles arbitrary N, K.
    """

    def __init__(self, wq_b_weight_bf16: torch.Tensor, module, block_k: int = 128):
        # wq_b_weight_bf16: [4096, 2048]
        self.inner = FP8IndexerWeightsCUDA(wq_b_weight_bf16, module, block_k)

    @property
    def w_fp8(self):
        return self.inner.w_fp8

    @property
    def w_scale(self):
        return self.inner.w_scale

    @property
    def tma_desc(self):
        return self.inner.tma_desc

    @property
    def N(self):
        return self.inner.N

    @property
    def K(self):
        return self.inner.K

    @property
    def block_k(self):
        return self.inner.block_k


# ============================================================
# CUDA WGMMA wq_b projection
# ============================================================

def cuda_wq_b_proj(
    q_a: torch.Tensor,           # [B, 2048] BF16
    wq_b_weights: FP8WqbWeightsCUDA,
    module,
) -> torch.Tensor:
    """FP8 WGMMA projection: [B, 2048] → [B, 4096] BF16.

    Reuses WP2's single-WG TMA-both kernel with separate act_quant.
    """
    q_a = q_a.contiguous()
    B, K = q_a.shape
    N = wq_b_weights.N  # 4096

    # Act quant: BF16 → FP8 + per-row scale
    x_fp8 = torch.empty(B, K, dtype=torch.float8_e4m3fn, device=q_a.device)
    x_scale = torch.empty(B, dtype=torch.float32, device=q_a.device)
    module.run_act_quant(q_a, x_fp8, x_scale)

    # Pad to BLOCK_M=64
    B_padded = max(B, 64)
    if B < 64:
        x_fp8_padded = torch.zeros(B_padded, K, dtype=torch.float8_e4m3fn, device=q_a.device)
        x_fp8_padded[:B] = x_fp8
        x_fp8 = x_fp8_padded

    # TMA desc for activation
    a_tma_desc = module.create_tma_desc(x_fp8, B_padded, K, 64, 128)

    return module.indexer_kv_proj_gemm_only(
        a_tma_desc, wq_b_weights.tma_desc,
        wq_b_weights.w_scale, x_scale,
        B, N, K,
    )


def cuda_wq_b_proj_out(
    q_a: torch.Tensor,
    wq_b_weights: FP8WqbWeightsCUDA,
    module,
    x_fp8_padded: torch.Tensor,
    x_scale: torch.Tensor,
    a_tma_desc: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Out-buffer FP8 WGMMA q_b projection for CUDA graph capture."""
    if not q_a.is_contiguous():
        raise ValueError("q_a must be contiguous for graph-captured q_b projection")
    B, K = q_a.shape
    N = wq_b_weights.N
    _validate_projection_out_buffers(B, K, N, x_fp8_padded, x_scale, out)

    module.run_act_quant(q_a, x_fp8_padded[:B], x_scale)
    module.indexer_kv_proj_gemm_only_out(
        a_tma_desc,
        wq_b_weights.tma_desc,
        wq_b_weights.w_scale,
        x_scale,
        out,
        B,
        N,
        K,
    )
    return out


# ============================================================
# Kernel: Fused Q×K scoring + head_gate + sum across heads
# ============================================================

@triton.jit
def _fused_score_kernel(
    Q_ptr,             # [B, n_heads, head_dim] BF16
    K_ptr,             # [B, max_seqlen, head_dim] BF16
    GATES_ptr,         # [B, n_heads] FP32
    SEQLENS_ptr,       # [B] int32
    AGG_ptr,           # [B, max_seqlen] FP32
    max_seqlen,
    B: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Grid: (cdiv(max_seqlen, BLOCK_S), B)"""
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    seqlen = tl.load(SEQLENS_ptr + pid_b)
    s_mask = s_offs < seqlen

    agg = tl.where(s_mask, tl.zeros([BLOCK_S], dtype=tl.float32),
                   float('-inf') + tl.zeros([BLOCK_S], dtype=tl.float32))

    k_base = pid_b * max_seqlen * head_dim
    d_offs = tl.arange(0, BLOCK_D)

    k_ptrs = K_ptr + k_base + s_offs[:, None] * head_dim + d_offs[None, :]
    k_tile = tl.load(k_ptrs, mask=s_mask[:, None] & (d_offs[None, :] < head_dim), other=0.0)
    k_tile = k_tile.to(tl.float32)

    q_base = pid_b * n_heads * head_dim
    gates_base = pid_b * n_heads

    for h in range(n_heads):
        q_ptrs = Q_ptr + q_base + h * head_dim + d_offs
        q_vec = tl.load(q_ptrs, mask=d_offs < head_dim, other=0.0).to(tl.float32)
        gate = tl.load(GATES_ptr + gates_base + h).to(tl.float32)
        scores = tl.sum(k_tile * q_vec[None, :], axis=1)
        agg += tl.where(s_mask, scores * gate, tl.zeros([BLOCK_S], dtype=tl.float32))

    agg_ptrs = AGG_ptr + pid_b * max_seqlen + s_offs
    tl.store(agg_ptrs, agg, mask=s_offs < max_seqlen)


@triton.jit
def _fused_paged_score_kernel(
    Q_ptr,             # [B, n_heads, head_dim] BF16
    K_ptr,             # [num_pages, page_size, 1, head_dim] BF16
    BLOCK_TABLE_ptr,   # [B, max_pages_per_seq] int32/int64
    GATES_ptr,         # [B, n_heads] FP32
    SEQLENS_ptr,       # [B] int32
    AGG_ptr,           # [B, max_seqlen] FP32
    max_seqlen,
    B: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Score Q against paged aux indexer K without dense K materialization.

    Grid: (cdiv(max_seqlen, BLOCK_S), B)
    """
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    seqlen = tl.load(SEQLENS_ptr + pid_b)
    s_mask = s_offs < seqlen
    page_in_range = s_offs < max_seqlen

    logical_page = s_offs // page_size
    page_offset = s_offs - logical_page * page_size
    page_in_table = logical_page < max_pages_per_seq
    safe_logical_page = tl.minimum(logical_page, max_pages_per_seq - 1)
    physical_page = tl.load(
        BLOCK_TABLE_ptr + pid_b * max_pages_per_seq + safe_logical_page,
        mask=page_in_range,
        other=-1,
    ).to(tl.int64)
    k_valid = page_in_table & (physical_page >= 0)
    physical_page = tl.maximum(physical_page, 0)

    agg = tl.where(
        s_mask,
        tl.zeros([BLOCK_S], dtype=tl.float32),
        float("-inf") + tl.zeros([BLOCK_S], dtype=tl.float32),
    )

    d_offs = tl.arange(0, BLOCK_D)
    k_ptrs = K_ptr + (physical_page[:, None] * page_size + page_offset[:, None]) * head_dim + d_offs[None, :]
    k_tile = tl.load(
        k_ptrs,
        mask=page_in_range[:, None] & (d_offs[None, :] < head_dim) & k_valid[:, None],
        other=0.0,
    ).to(tl.float32)

    q_base = pid_b * n_heads * head_dim
    gates_base = pid_b * n_heads

    for h in range(n_heads):
        q_ptrs = Q_ptr + q_base + h * head_dim + d_offs
        q_vec = tl.load(q_ptrs, mask=d_offs < head_dim, other=0.0).to(tl.float32)
        gate = tl.load(GATES_ptr + gates_base + h).to(tl.float32)
        scores = tl.sum(k_tile * q_vec[None, :], axis=1)
        agg += tl.where(s_mask, scores * gate, tl.zeros([BLOCK_S], dtype=tl.float32))

    agg_ptrs = AGG_ptr + pid_b * max_seqlen + s_offs
    tl.store(agg_ptrs, agg, mask=s_offs < max_seqlen)


@triton.jit
def _fused_paged_score_with_slots_kernel(
    Q_ptr,             # [B, n_heads, head_dim] BF16
    K_ptr,             # [num_pages, page_size, 1, head_dim] BF16
    BLOCK_TABLE_ptr,   # [num_slots, max_pages_per_seq] int32/int64
    SLOT_INDICES_ptr,  # [B] int32/int64
    GATES_ptr,         # [B, n_heads] FP32
    SEQLENS_ptr,       # [B] int32
    AGG_ptr,           # [B, max_seqlen] FP32
    max_seqlen,
    B: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Score Q against paged aux K using runtime slot indices."""

    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    slot = tl.load(SLOT_INDICES_ptr + pid_b).to(tl.int64)
    slot_valid = slot >= 0
    safe_slot = tl.maximum(slot, 0)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    seqlen = tl.load(SEQLENS_ptr + pid_b)
    s_mask = (s_offs < seqlen) & slot_valid
    page_in_range = s_offs < max_seqlen

    logical_page = s_offs // page_size
    page_offset = s_offs - logical_page * page_size
    page_in_table = logical_page < max_pages_per_seq
    safe_logical_page = tl.minimum(logical_page, max_pages_per_seq - 1)
    physical_page = tl.load(
        BLOCK_TABLE_ptr + safe_slot * max_pages_per_seq + safe_logical_page,
        mask=page_in_range & slot_valid & page_in_table,
        other=-1,
    ).to(tl.int64)
    k_valid = slot_valid & page_in_table & (physical_page >= 0)
    physical_page = tl.maximum(physical_page, 0)

    agg = tl.where(
        s_mask,
        tl.zeros([BLOCK_S], dtype=tl.float32),
        float("-inf") + tl.zeros([BLOCK_S], dtype=tl.float32),
    )

    d_offs = tl.arange(0, BLOCK_D)
    k_ptrs = K_ptr + (physical_page[:, None] * page_size + page_offset[:, None]) * head_dim + d_offs[None, :]
    k_tile = tl.load(
        k_ptrs,
        mask=page_in_range[:, None] & (d_offs[None, :] < head_dim) & k_valid[:, None],
        other=0.0,
    ).to(tl.float32)

    q_base = pid_b * n_heads * head_dim
    gates_base = pid_b * n_heads

    for h in range(n_heads):
        q_ptrs = Q_ptr + q_base + h * head_dim + d_offs
        q_vec = tl.load(q_ptrs, mask=d_offs < head_dim, other=0.0).to(tl.float32)
        gate = tl.load(GATES_ptr + gates_base + h).to(tl.float32)
        scores = tl.sum(k_tile * q_vec[None, :], axis=1)
        agg += tl.where(s_mask, scores * gate, tl.zeros([BLOCK_S], dtype=tl.float32))

    agg_ptrs = AGG_ptr + pid_b * max_seqlen + s_offs
    tl.store(agg_ptrs, agg, mask=s_offs < max_seqlen)


@triton.jit
def _topk_from_scores_kernel(
    AGG_ptr,           # [B, max_seqlen] FP32
    OUT_ptr,           # [B, topk] int64/int32
    max_seqlen: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Select top-k indices from one row of scores.

    One Triton program owns one batch row. It first finds a score threshold whose
    greater-than set is smaller than ``topk``, compacts that set to the output,
    then repairs the remaining slots with exact repeated maxima from the residual
    values. This keeps the common GLM case near O(log(score_range) * N) instead
    of O(topk * N), while preserving the exact top-k set.
    """

    pid_b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < max_seqlen
    vals = tl.load(
        AGG_ptr + pid_b * max_seqlen + offs,
        mask=mask,
        other=float("-inf"),
    )
    idxs = offs

    lo = tl.min(tl.where(mask, vals, float("inf")), axis=0)
    hi = tl.max(vals, axis=0)

    for _ in range(32):
        mid = (lo + hi) * 0.5
        n_ge = tl.sum(tl.where(vals >= mid, 1, 0), axis=0)
        lo = tl.where(n_ge >= topk, mid, lo)
        hi = tl.where(n_ge >= topk, hi, mid)

    selected_hi = vals > hi
    ranks = tl.cumsum(tl.where(selected_hi, 1, 0), 0) - 1
    tl.store(
        OUT_ptr + pid_b * topk + ranks,
        idxs,
        mask=selected_hi & (ranks < topk),
    )

    vals = tl.where(selected_hi, float("-inf"), vals)
    out_pos = tl.sum(tl.where(selected_hi, 1, 0), axis=0)
    while out_pos < topk:
        best_val = tl.max(vals, axis=0)
        best_idx = tl.max(tl.where(vals == best_val, idxs, 0), axis=0)
        tl.store(OUT_ptr + pid_b * topk + out_pos, best_idx)
        vals = tl.where(idxs == best_idx, float("-inf"), vals)
        out_pos += 1


# ============================================================
# Python wrappers (no CPU-GPU syncs)
# ============================================================

def compute_head_gates(hidden_states, weights_proj_weight, n_heads, head_dim):
    """Compute pre-scaled head gates. Pure GPU, no sync."""
    gates = torch.nn.functional.linear(hidden_states.float(), weights_proj_weight.float())
    scale = (n_heads ** -0.5) * (head_dim ** -0.5)
    return (gates * scale).to(torch.float32)


def _score_into_dense_agg(
    q: torch.Tensor,
    cached_k: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    agg: torch.Tensor,
) -> torch.Tensor:
    B, n_heads, head_dim = q.shape
    max_seqlen = cached_k.shape[1]
    if agg.shape != (B, max_seqlen):
        raise ValueError(
            f"agg must have shape {(B, max_seqlen)}, got {tuple(agg.shape)}"
        )
    if agg.dtype != torch.float32:
        raise TypeError(f"agg must be float32, got {agg.dtype}")

    BLOCK_S = min(128, triton.next_power_of_2(max_seqlen))
    BLOCK_D = head_dim
    grid = (triton.cdiv(max_seqlen, BLOCK_S), B)

    _fused_score_kernel[grid](
        q, cached_k, head_gates, cache_seqlens,
        agg, max_seqlen,
        B=B, n_heads=n_heads, head_dim=head_dim,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )

    return agg


def fused_score_and_topk(
    q: torch.Tensor,
    cached_k: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    topk: int = 2048,
) -> torch.Tensor:
    """Fused scoring + eager top-k.

    The allocation-returning path is used by eager GLM-5 decode.  Keep the
    expensive score computation fused, then use the radix CUDA top-k path for
    production-sized ``index_topk=2048`` contexts; smaller test shapes fall back
    to PyTorch top-k.
    """
    B = q.shape[0]
    max_seqlen = cached_k.shape[1]
    effective_topk = min(topk, max_seqlen)
    agg = torch.empty(B, max_seqlen, dtype=torch.float32, device=q.device)
    _score_into_dense_agg(q, cached_k, head_gates, cache_seqlens, agg)
    if effective_topk == 2048 and cache_seqlens.dtype == torch.int32 and q.is_cuda:
        return fast_topk_2048(agg, cache_seqlens)
    _, top_k_indices = torch.topk(agg, effective_topk, dim=-1)
    return top_k_indices


def fused_score_and_topk_out(
    q: torch.Tensor,
    cached_k: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    agg: torch.Tensor,
    top_k_indices: torch.Tensor,
    topk: int = 2048,
) -> torch.Tensor:
    """Out-buffer fused scoring + custom Triton top-k for CUDA graph capture."""
    B, n_heads, head_dim = q.shape
    max_seqlen = cached_k.shape[1]
    if topk > max_seqlen:
        raise ValueError(f"topk={topk} exceeds max_seqlen={max_seqlen}")
    if agg.shape != (B, max_seqlen):
        raise ValueError(
            f"agg must have shape {(B, max_seqlen)}, got {tuple(agg.shape)}"
        )
    if agg.dtype != torch.float32:
        raise TypeError(f"agg must be float32, got {agg.dtype}")
    if top_k_indices.shape != (B, topk):
        raise ValueError(
            f"top_k_indices must have shape {(B, topk)}, got {tuple(top_k_indices.shape)}"
        )
    if top_k_indices.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"top_k_indices must be int64 or int32, got {top_k_indices.dtype}")

    _score_into_dense_agg(q, cached_k, head_gates, cache_seqlens, agg)

    if topk == 2048 and top_k_indices.dtype == torch.int32:
        fast_topk_2048_out(agg, cache_seqlens, top_k_indices)
    else:
        block_n = triton.next_power_of_2(max_seqlen)
        _topk_from_scores_kernel[(B,)](
            agg,
            top_k_indices,
            max_seqlen=max_seqlen,
            topk=topk,
            BLOCK_N=block_n,
        )

    return top_k_indices


def fused_paged_score_and_topk_out(
    q: torch.Tensor,
    aux_blocked_k: torch.Tensor,
    aux_page_table: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    agg: torch.Tensor,
    top_k_indices: torch.Tensor,
    *,
    topk: int = 2048,
    page_size: int = 64,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Out-buffer score+top-k directly from paged aux indexer K.

    This is the production graph path for GLM-5 DSA scoring.  It avoids the
    temporary dense ``[B, max_seqlen, 128]`` aux-K tensor.
    """
    B, n_heads, head_dim = q.shape
    if max_seqlen is None:
        max_seqlen = agg.shape[1]
    if aux_blocked_k.ndim != 4:
        raise ValueError(
            "aux_blocked_k must have shape [num_pages, page_size, 1, head_dim], "
            f"got {tuple(aux_blocked_k.shape)}"
        )
    if aux_blocked_k.shape[1] != page_size:
        raise ValueError(f"aux page size mismatch: {aux_blocked_k.shape[1]} != {page_size}")
    if aux_blocked_k.shape[2] != 1 or aux_blocked_k.shape[3] != head_dim:
        raise ValueError(
            f"aux_blocked_k must have one head and dim {head_dim}, "
            f"got {tuple(aux_blocked_k.shape)}"
        )
    if aux_page_table.shape[0] != B:
        raise ValueError(
            f"aux_page_table batch dim {aux_page_table.shape[0]} must match q batch {B}"
        )
    if head_gates.shape != (B, n_heads):
        raise ValueError(f"head_gates must have shape {(B, n_heads)}, got {tuple(head_gates.shape)}")
    if cache_seqlens.shape != (B,):
        raise ValueError(f"cache_seqlens must have shape {(B,)}, got {tuple(cache_seqlens.shape)}")
    if agg.shape != (B, max_seqlen) or agg.dtype != torch.float32:
        raise ValueError(
            f"agg must be float32 with shape {(B, max_seqlen)}, got {tuple(agg.shape)} {agg.dtype}"
        )
    if topk > max_seqlen:
        raise ValueError(f"topk={topk} exceeds max_seqlen={max_seqlen}")
    if top_k_indices.shape != (B, topk):
        raise ValueError(
            f"top_k_indices must have shape {(B, topk)}, got {tuple(top_k_indices.shape)}"
        )
    if top_k_indices.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"top_k_indices must be int64 or int32, got {top_k_indices.dtype}")

    BLOCK_S = min(128, triton.next_power_of_2(max_seqlen))
    BLOCK_D = head_dim
    grid = (triton.cdiv(max_seqlen, BLOCK_S), B)
    _fused_paged_score_kernel[grid](
        q,
        aux_blocked_k.reshape(-1, head_dim),
        aux_page_table,
        head_gates,
        cache_seqlens,
        agg,
        max_seqlen,
        B=B,
        n_heads=n_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_pages_per_seq=aux_page_table.shape[1],
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    if topk == 2048 and top_k_indices.dtype == torch.int32:
        fast_topk_2048_out(agg, cache_seqlens, top_k_indices)
    else:
        block_n = triton.next_power_of_2(max_seqlen)
        _topk_from_scores_kernel[(B,)](
            agg,
            top_k_indices,
            max_seqlen=max_seqlen,
            topk=topk,
            BLOCK_N=block_n,
        )
    return top_k_indices


def fused_paged_score_and_topk_with_slots_out(
    q: torch.Tensor,
    aux_blocked_k: torch.Tensor,
    aux_page_table: torch.Tensor,
    aux_slot_indices: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    agg: torch.Tensor,
    top_k_indices: torch.Tensor,
    *,
    topk: int = 2048,
    page_size: int = 64,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Out-buffer score+top-k from full aux page table plus slot indices."""

    B, n_heads, head_dim = q.shape
    if max_seqlen is None:
        max_seqlen = agg.shape[1]
    if aux_blocked_k.ndim != 4:
        raise ValueError(
            "aux_blocked_k must have shape [num_pages, page_size, 1, head_dim], "
            f"got {tuple(aux_blocked_k.shape)}"
        )
    if aux_blocked_k.shape[1] != page_size:
        raise ValueError(f"aux page size mismatch: {aux_blocked_k.shape[1]} != {page_size}")
    if aux_blocked_k.shape[2] != 1 or aux_blocked_k.shape[3] != head_dim:
        raise ValueError(
            f"aux_blocked_k must have one head and dim {head_dim}, "
            f"got {tuple(aux_blocked_k.shape)}"
        )
    if aux_page_table.ndim != 2:
        raise ValueError(f"aux_page_table must be 2-D, got {tuple(aux_page_table.shape)}")
    if aux_slot_indices.shape != (B,):
        raise ValueError(
            f"aux_slot_indices must have shape {(B,)}, got {tuple(aux_slot_indices.shape)}"
        )
    if aux_slot_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"aux_slot_indices must be int32/int64, got {aux_slot_indices.dtype}")
    if head_gates.shape != (B, n_heads):
        raise ValueError(f"head_gates must have shape {(B, n_heads)}, got {tuple(head_gates.shape)}")
    if cache_seqlens.shape != (B,):
        raise ValueError(f"cache_seqlens must have shape {(B,)}, got {tuple(cache_seqlens.shape)}")
    if agg.shape != (B, max_seqlen) or agg.dtype != torch.float32:
        raise ValueError(
            f"agg must be float32 with shape {(B, max_seqlen)}, got {tuple(agg.shape)} {agg.dtype}"
        )
    if topk > max_seqlen:
        raise ValueError(f"topk={topk} exceeds max_seqlen={max_seqlen}")
    if top_k_indices.shape != (B, topk):
        raise ValueError(
            f"top_k_indices must have shape {(B, topk)}, got {tuple(top_k_indices.shape)}"
        )
    if top_k_indices.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"top_k_indices must be int64 or int32, got {top_k_indices.dtype}")

    BLOCK_S = min(128, triton.next_power_of_2(max_seqlen))
    BLOCK_D = head_dim
    grid = (triton.cdiv(max_seqlen, BLOCK_S), B)
    _fused_paged_score_with_slots_kernel[grid](
        q,
        aux_blocked_k.reshape(-1, head_dim),
        aux_page_table,
        aux_slot_indices,
        head_gates,
        cache_seqlens,
        agg,
        max_seqlen,
        B=B,
        n_heads=n_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_pages_per_seq=aux_page_table.shape[1],
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    if topk == 2048 and top_k_indices.dtype == torch.int32:
        fast_topk_2048_out(agg, cache_seqlens, top_k_indices)
    else:
        block_n = triton.next_power_of_2(max_seqlen)
        _topk_from_scores_kernel[(B,)](
            agg,
            top_k_indices,
            max_seqlen=max_seqlen,
            topk=topk,
            BLOCK_N=block_n,
        )
    return top_k_indices


# ============================================================
# Hadamard matrix (cached, built once on GPU)
# ============================================================

_hadamard_cache = {}

def get_hadamard_matrix(dim, device, dtype=torch.bfloat16):
    key = (dim, device, dtype)
    if key not in _hadamard_cache:
        H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
        while H.shape[0] < dim:
            H = torch.cat([torch.cat([H, H], dim=1),
                           torch.cat([H, -H], dim=1)], dim=0)
        _hadamard_cache[key] = (H * (dim ** -0.5)).to(dtype).contiguous()
    return _hadamard_cache[key]


# ============================================================
# RoPE + Hadamard — CUDA fused kernel (from attention/dsa/indexer)
# ============================================================

def apply_rope_interleaved(x, cos, sin):
    """Apply interleaved RoPE. x: [..., dim], cos/sin: [..., rope_dim].
    PyTorch fallback — used only in reference path."""
    rope_dim = cos.shape[-1]
    x_rope = x[..., :rope_dim]
    x_nope = x[..., rope_dim:]
    x1 = x_rope[..., 0::2]
    x2 = x_rope[..., 1::2]
    cos_h = cos[..., :rope_dim // 2]
    sin_h = sin[..., :rope_dim // 2]
    r1 = x1 * cos_h - x2 * sin_h
    r2 = x2 * cos_h + x1 * sin_h
    x_rot = torch.stack([r1, r2], dim=-1).flatten(-2)
    return torch.cat([x_rot, x_nope], dim=-1)


def rope_hadamard_q(q, cos_table, sin_table, positions, rope_dim=64):
    """Apply fused CUDA RoPE + Hadamard to Q [B, n_heads, head_dim].

    Uses the existing CUDA kernel from attention/dsa/indexer.
    The kernel handles [batch, 128] with 1 block/row.
    For multi-head Q [B, 32, 128], we reshape to [B*32, 128]
    and expand positions from [B] to [B*32].

    cos_table/sin_table: [max_pos, 64] — can be BF16 or FP32.
    The CUDA kernel expects float32 cos/sin, so we cast if needed.
    """
    B, n_heads, head_dim = q.shape

    # Expand positions: [B] → [B*n_heads] (each head gets same position)
    positions_expanded = positions.repeat_interleave(n_heads)  # [B*n_heads]

    # Ensure cos/sin are float32 (CUDA kernel requirement)
    cos_f32 = cos_table.float() if cos_table.dtype != torch.float32 else cos_table
    sin_f32 = sin_table.float() if sin_table.dtype != torch.float32 else sin_table

    # [B, 32, 128] → CUDA kernel → [B, 32, 128]
    q_out = _cuda_fused_rope_hadamard(
        q.contiguous(),      # [B, 32, 128] bf16 — kernel reshapes to [B*32, 128]
        cos_f32,             # [max_pos, 64] float32
        sin_f32,             # [max_pos, 64] float32
        positions_expanded,  # [B*32] int64
        128 ** -0.5,         # Hadamard scale
    )
    return q_out


def rope_hadamard_q_out(
    q: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    positions_expanded: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Out-buffer CUDA RoPE + Hadamard for graph capture.

    ``positions_expanded`` must already have shape ``[B * n_heads]`` to avoid
    allocating inside the captured segment.
    """
    B, n_heads, head_dim = q.shape
    if head_dim != 128:
        raise ValueError(f"GLM-5 DSA RoPE+Hadamard requires head_dim=128, got {head_dim}")
    if out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError(f"out must match q shape/dtype, got {out.shape} {out.dtype}")
    if cos_table.dtype != torch.float32 or sin_table.dtype != torch.float32:
        raise TypeError("cos_table and sin_table must be float32 for graph-captured RoPE+Hadamard")
    if positions_expanded.shape != (B * n_heads,) or positions_expanded.dtype != torch.int64:
        raise ValueError(
            f"positions_expanded must be int64 with shape {(B * n_heads,)}, "
            f"got {positions_expanded.shape} {positions_expanded.dtype}"
        )
    return _cuda_fused_rope_hadamard_out(
        q.reshape(B * n_heads, head_dim),
        cos_table,
        sin_table,
        positions_expanded,
        out.reshape(B * n_heads, head_dim),
        128 ** -0.5,
    )


def rope_hadamard_q_pytorch(q, cos_table, sin_table, positions, rope_dim=64):
    """PyTorch fallback for RoPE + Hadamard (reference/test only)."""
    head_dim = q.shape[-1]

    # RoPE
    q_cos = cos_table[positions].unsqueeze(1)
    q_sin = sin_table[positions].unsqueeze(1)
    q = apply_rope_interleaved(q, q_cos, q_sin)

    # Hadamard
    H = get_hadamard_matrix(head_dim, q.device, torch.bfloat16)
    q = torch.matmul(q.to(torch.bfloat16), H)

    return q


# ============================================================
# Full scoring pipeline — v3 (CUDA WGMMA + CUDA RoPE/Hadamard + no sync)
# ============================================================

def fused_score_pipeline(
    q_a,                    # [B, 2048] BF16
    hidden_states,          # [B, 6144] BF16
    cached_k,               # [B, max_seqlen, 128] BF16
    cache_seqlens,          # [B] int32
    wq_b_weights,           # FP8WqbWeightsCUDA
    weights_proj_weight,    # [32, 6144] BF16
    cos_table, sin_table,   # [max_pos, 64] BF16 — RoPE tables
    positions,              # [B] int64
    module,                 # CUDA module from build_module()
    n_heads=32,
    head_dim=128,
    rope_dim=64,
    topk=2048,
):
    """Full fused scoring pipeline — zero CPU-GPU sync in hot path.

    Step 1: CUDA WGMMA wq_b projection [B, 2048] → [B, 4096]
    Step 2-3: RoPE + Hadamard on Q [B, 32, 128]
    Step 4-8: Fused scoring kernel (Triton)
    Step 9: custom Triton top-k on aggregated [B, max_seqlen]
    """
    B = q_a.shape[0]

    # Step 1: CUDA WGMMA wq_b projection
    q_flat = cuda_wq_b_proj(q_a, wq_b_weights, module)  # [B, 4096]
    q = q_flat.view(B, n_heads, head_dim)  # [B, 32, 128]

    # Step 2-3: RoPE + Hadamard
    q = rope_hadamard_q(q, cos_table, sin_table, positions, rope_dim)

    # Step 4-9: Fused scoring + topk
    head_gates = compute_head_gates(hidden_states, weights_proj_weight, n_heads, head_dim)
    top_k_indices = fused_score_and_topk(q, cached_k, head_gates, cache_seqlens, topk)

    return top_k_indices, q


# ============================================================
# Reference (PyTorch, for validation only — has CPU-GPU syncs)
# ============================================================

def reference_score_and_select(
    q_a, hidden_states, cached_k, cache_seqlens,
    wq_b_weight, weights_proj_weight,
    cos_table, sin_table, positions,
    n_heads=32, head_dim=128, rope_dim=64, topk=2048,
):
    """Full PyTorch reference. CPU-GPU syncs allowed (test only)."""
    B = q_a.shape[0]
    max_seqlen = cached_k.shape[1]

    # wq_b projection
    q = torch.nn.functional.linear(q_a, wq_b_weight)
    q = q.view(B, n_heads, head_dim)

    # RoPE + Hadamard (PyTorch reference)
    q = rope_hadamard_q_pytorch(q, cos_table, sin_table, positions, rope_dim)

    # Head gates
    head_gates = compute_head_gates(hidden_states, weights_proj_weight, n_heads, head_dim)

    # Q×K scoring — chunked per-head to avoid OOM at large seqlens
    aggregated = torch.zeros(B, max_seqlen, dtype=torch.float32, device=q.device)
    q_f = q.float()
    for h in range(n_heads):
        s = torch.bmm(q_f[:, h:h+1, :], cached_k.float().transpose(1, 2))
        aggregated += s.squeeze(1) * head_gates[:, h:h+1]
    pos_idx = torch.arange(max_seqlen, device=q.device).unsqueeze(0)
    mask = pos_idx >= cache_seqlens.unsqueeze(1)
    aggregated.masked_fill_(mask, float("-inf"))

    # topk (test code — .item() sync OK)
    min_valid = int(cache_seqlens.min().item())
    effective_topk = min(topk, max_seqlen, min_valid)
    _, top_k_indices = torch.topk(aggregated, effective_topk, dim=-1)

    return top_k_indices, q, aggregated


# ============================================================
# Inline test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"

    n_heads = 32
    head_dim = 128
    rope_dim = 64
    hidden_size = 6144
    q_lora_rank = 2048
    topk = 2048

    # Build CUDA module (reuses WP2)
    module = build_module()

    # RoPE tables
    max_pos = 16384
    theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim))
    t = torch.arange(max_pos, device=device).float()
    angles = t[:, None] * freqs[None, :]
    cos_table = torch.cos(angles).to(torch.bfloat16).repeat(1, 2)
    sin_table = torch.sin(angles).to(torch.bfloat16).repeat(1, 2)

    # Weights
    wq_b_weight_bf16 = torch.randn(n_heads * head_dim, q_lora_rank, dtype=torch.bfloat16, device=device) * 0.01
    weights_proj_weight = torch.randn(n_heads, hidden_size, dtype=torch.bfloat16, device=device) * 0.01

    # FP8 weights for CUDA WGMMA
    wq_b_cuda = FP8WqbWeightsCUDA(wq_b_weight_bf16, module)

    def calc_diff(x, y):
        x, y = x.double(), y.double()
        denom = (x * x + y * y).sum()
        if denom == 0:
            return 0.0
        return (1 - 2 * (x * y).sum() / denom).item()

    def test_wq_b_gemm(B, label=""):
        """Test CUDA WGMMA wq_b projection accuracy."""
        print(f"\n=== wq_b GEMM {label}: B={B} ===")
        q_a = torch.randn(B, q_lora_rank, dtype=torch.bfloat16, device=device) * 0.1

        # Reference: BF16 linear
        ref = torch.nn.functional.linear(q_a, wq_b_weight_bf16)

        # CUDA WGMMA
        out = cuda_wq_b_proj(q_a, wq_b_cuda, module)

        cd = calc_diff(out, ref)
        print(f"  calc_diff vs BF16 ref: {cd:.6f} {'PASS' if cd < 1e-2 else 'FAIL'}")

        # FP8 reference (accounts for quantization)
        N, K = wq_b_weight_bf16.shape
        w_dequant = torch.zeros(N, K, dtype=torch.float32, device=device)
        for n_tile in range(N // 32):
            for kb in range(K // 128):
                ns, ne = n_tile * 32, (n_tile + 1) * 32
                ks, ke = kb * 128, (kb + 1) * 128
                w_dequant[ns:ne, ks:ke] = (
                    wq_b_cuda.w_fp8[ns:ne, ks:ke].float() * wq_b_cuda.w_scale[n_tile, kb]
                )
        ref_fp8 = torch.nn.functional.linear(q_a.float(), w_dequant).to(torch.bfloat16)
        cd_fp8 = calc_diff(out, ref_fp8)
        print(f"  calc_diff vs FP8 ref:  {cd_fp8:.6f} {'PASS' if cd_fp8 < 1e-3 else 'FAIL'}")
        return cd_fp8 < 1e-3

    def test_full_pipeline(B, max_seqlen, label=""):
        """Test full scoring pipeline with CUDA wq_b."""
        print(f"\n=== Full pipeline {label}: B={B}, seqlen={max_seqlen} ===")

        q_a = torch.randn(B, q_lora_rank, dtype=torch.bfloat16, device=device) * 0.1
        hidden_states = torch.randn(B, hidden_size, dtype=torch.bfloat16, device=device) * 0.1
        cached_k = torch.randn(B, max_seqlen, head_dim, dtype=torch.bfloat16, device=device) * 0.1
        cache_seqlens = torch.randint(topk, max_seqlen + 1, (B,), dtype=torch.int32, device=device)
        positions = torch.randint(0, max_pos, (B,), dtype=torch.int64, device=device)

        # Reference (BF16 wq_b)
        ref_indices, ref_q, ref_agg = reference_score_and_select(
            q_a, hidden_states, cached_k, cache_seqlens,
            wq_b_weight_bf16, weights_proj_weight,
            cos_table, sin_table, positions,
        )

        # Fused (CUDA WGMMA wq_b)
        fused_indices, fused_q = fused_score_pipeline(
            q_a, hidden_states, cached_k, cache_seqlens,
            wq_b_cuda, weights_proj_weight,
            cos_table, sin_table, positions,
            module,
        )

        # Q accuracy (FP8 vs BF16 wq_b — expect some diff)
        cd_q = calc_diff(fused_q, ref_q)
        print(f"  Q calc_diff (FP8 vs BF16 wq_b): {cd_q:.6f}")

        # topk overlap
        overlaps = []
        for b in range(B):
            ref_set = set(ref_indices[b].tolist())
            fused_set = set(fused_indices[b].tolist())
            overlaps.append(len(ref_set & fused_set) / max(len(ref_set), 1) * 100)
        avg_overlap = sum(overlaps) / len(overlaps)
        print(f"  topk overlap: avg={avg_overlap:.1f}%, min={min(overlaps):.1f}%")

        passed = avg_overlap > 95.0  # FP8 wq_b → slightly different Q → some topk divergence OK
        print(f"  → {'PASS' if passed else 'FAIL'}")
        return passed

    # Run tests
    all_pass = True
    all_pass &= test_wq_b_gemm(B=1, label="B=1")
    all_pass &= test_wq_b_gemm(B=32, label="B=32")
    all_pass &= test_wq_b_gemm(B=64, label="B=64")
    all_pass &= test_full_pipeline(B=32, max_seqlen=4096, label="medium")
    all_pass &= test_full_pipeline(B=32, max_seqlen=10240, label="long")

    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Benchmark
    print("\n=== Benchmark: wq_b projection ===")
    import time
    for B in [1, 32, 64]:
        q_a = torch.randn(B, q_lora_rank, dtype=torch.bfloat16, device=device) * 0.1

        # Torch BF16
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            torch.nn.functional.linear(q_a, wq_b_weight_bf16)
        torch.cuda.synchronize()
        torch_us = (time.perf_counter() - t0) / 200 * 1e6

        # CUDA WGMMA FP8
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            cuda_wq_b_proj(q_a, wq_b_cuda, module)
        torch.cuda.synchronize()
        cuda_us = (time.perf_counter() - t0) / 200 * 1e6

        print(f"  B={B:>2d}: Torch={torch_us:.1f}µs, CUDA={cuda_us:.1f}µs, speedup={torch_us/cuda_us:.2f}×")

    print("\n=== Benchmark: full scoring pipeline ===")
    for max_seqlen in [2048, 4096, 10240]:
        B = 32
        q_a = torch.randn(B, q_lora_rank, dtype=torch.bfloat16, device=device) * 0.1
        hidden_states = torch.randn(B, hidden_size, dtype=torch.bfloat16, device=device) * 0.1
        cached_k = torch.randn(B, max_seqlen, head_dim, dtype=torch.bfloat16, device=device) * 0.1
        cache_seqlens = torch.full((B,), max_seqlen, dtype=torch.int32, device=device)
        positions = torch.randint(0, max_pos, (B,), dtype=torch.int64, device=device)

        # Torch baseline (BF16 wq_b + PyTorch scoring)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            reference_score_and_select(
                q_a, hidden_states, cached_k, cache_seqlens,
                wq_b_weight_bf16, weights_proj_weight,
                cos_table, sin_table, positions,
            )
        torch.cuda.synchronize()
        torch_us = (time.perf_counter() - t0) / 100 * 1e6

        # Fused (CUDA WGMMA + Triton scoring)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fused_score_pipeline(
                q_a, hidden_states, cached_k, cache_seqlens,
                wq_b_cuda, weights_proj_weight,
                cos_table, sin_table, positions, module,
            )
        torch.cuda.synchronize()
        fused_us = (time.perf_counter() - t0) / 100 * 1e6

        print(f"  seqlen={max_seqlen:>5d}: Torch={torch_us:.1f}µs, Fused={fused_us:.1f}µs, speedup={torch_us/fused_us:.2f}×")
