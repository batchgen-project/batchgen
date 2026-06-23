"""SM120 Triton sparse-MLA decode kernel for DeepSeek-V4-Flash.

Ported from SGLang flash_mla_sm120_triton.py (commit 578f232e), which targets the
identical GPU (RTX PRO 6000, sm120, no wgmma/TMA) and the identical DSv4 paged KV
layout. Replaces the eager flashmla_decode_torch_reference (~533ms/token) with a
fused tiled gather+dequant+flash-decode kernel.

DSv4 page layout (per token): 576 data bytes [0:448]=FP8 nope, [448:576]=BF16 rope
(64 vals); 8 scale bytes (7 UE8M0 groups of 64) at page_size*576 + tok*8.
Validated numerically against flashmla_decode_torch_reference (same correctness oracle).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from batchgen.timing import get_decode_timer

LOG2E = tl.constexpr(1.4426950408889634)

_NOPE_DIM = 448
_ROPE_DIM = 64
_HEAD_DIM = _NOPE_DIM + _ROPE_DIM
_TOKEN_DATA_STRIDE = 576
_SCALE_STRIDE = 8


_PINNED_BLOCK_T = 32
_PINNED_NUM_WARPS = 8
_PINNED_NUM_STAGES = 2


@triton.jit
def _tiled_sparse_decode_kernel(
    Q_ptr,
    cache_fp8_ptr,
    cache_uint8_ptr,
    cache_bf16_ptr,
    indices_ptr,
    topk_len_ptr,
    O_ptr,
    LSE_ptr,
    sm_scale: tl.float32,
    page_size: tl.int32,
    page_bytes: tl.int64,
    scale_section_off: tl.int64,
    H: tl.int32,
    topk: tl.int32,
    topk_rounded: tl.int32,
    has_topk_len: tl.constexpr,
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_ib: tl.int32,
    NOPE_PAD: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM_RT: tl.int32,
    BLOCK_T: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    q_base = bid * stride_qb + hid * stride_qh
    nope_offs = tl.arange(0, NOPE_PAD)
    nope_mask = nope_offs < NOPE_DIM_RT
    rope_offs = tl.arange(0, ROPE_DIM)

    q_nope = tl.load(Q_ptr + q_base + nope_offs, mask=nope_mask, other=0.0)
    q_nope = q_nope.to(tl.float32) * sm_scale
    q_rope = tl.load(Q_ptr + q_base + NOPE_DIM_RT + rope_offs)
    q_rope = q_rope.to(tl.float32) * sm_scale

    valid_topk = topk
    if has_topk_len:
        valid_topk = tl.load(topk_len_ptr + bid).to(tl.int32)
        valid_topk = tl.minimum(valid_topk, topk)

    m_i: tl.float32 = -1e30
    l_i: tl.float32 = 0.0
    acc_nope = tl.zeros([NOPE_PAD], dtype=tl.float32)
    acc_rope = tl.zeros([ROPE_DIM], dtype=tl.float32)

    group_ids = (nope_offs // 64).to(tl.int64)
    t_offs = tl.arange(0, BLOCK_T)

    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + t_offs
        t_in_bounds = t_idx < topk
        t_valid = t_idx < valid_topk

        raw_indices = tl.load(
            indices_ptr + bid * stride_ib + t_idx,
            mask=t_in_bounds,
            other=-1,
        )
        idx_valid = t_valid & (raw_indices >= 0)

        safe_indices = tl.where(
            idx_valid, raw_indices, tl.zeros_like(raw_indices)
        )
        page_ids = (safe_indices // page_size).to(tl.int64)
        page_offs_t = (safe_indices % page_size).to(tl.int64)
        token_data_bases = page_ids * page_bytes + page_offs_t * 576

        nope_addrs = token_data_bases[:, None] + nope_offs[None, :].to(tl.int64)
        nope_2d_mask = idx_valid[:, None] & nope_mask[None, :]
        kv_nope_fp8 = tl.load(
            cache_fp8_ptr + nope_addrs, mask=nope_2d_mask, other=0.0
        )

        scale_bases = (
            page_ids * page_bytes + scale_section_off + page_offs_t * 8
        )
        scale_addrs = scale_bases[:, None] + group_ids[None, :]
        scale_raw = tl.load(
            cache_uint8_ptr + scale_addrs, mask=nope_2d_mask, other=127
        )
        scale_f32 = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)
        kv_nope = tl.where(
            nope_2d_mask, kv_nope_fp8.to(tl.float32) * scale_f32, 0.0
        )

        rope_byte_bases = token_data_bases + 448
        rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)
        rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
        kv_rope = tl.load(
            cache_bf16_ptr + rope_addrs, mask=idx_valid[:, None], other=0.0
        ).to(tl.float32)

        scores = tl.sum(q_nope[None, :] * kv_nope, axis=1) + tl.sum(
            q_rope[None, :] * kv_rope, axis=1
        )
        scores = tl.where(idx_valid, scores, -1e30)

        scores_log2 = scores * LOG2E
        tile_max = tl.max(scores_log2)
        m_new = tl.maximum(m_i, tile_max)

        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores_log2 - m_new)
        p = tl.where(idx_valid, p, 0.0)

        l_i = l_i * alpha + tl.sum(p)
        acc_nope = acc_nope * alpha + tl.sum(p[:, None] * kv_nope, axis=0)
        acc_rope = acc_rope * alpha + tl.sum(p[:, None] * kv_rope, axis=0)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc_nope = acc_nope / safe_l
    acc_rope = acc_rope / safe_l
    lse = tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf"))

    o_base = bid * stride_ob + hid * stride_oh
    tl.store(
        O_ptr + o_base + nope_offs, acc_nope.to(tl.bfloat16), mask=nope_mask
    )
    tl.store(O_ptr + o_base + NOPE_DIM_RT + rope_offs, acc_rope.to(tl.bfloat16))
    tl.store(LSE_ptr + bid * H + hid, lse)


def _run_triton_sparse_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, D = q.shape
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)

    bytes_per_token = _TOKEN_DATA_STRIDE + _SCALE_STRIDE
    if page_bytes < page_size * bytes_per_token:
        raise ValueError(
            "k_cache must be shaped [num_pages, page_size, ..., "
            f"{bytes_per_token}] so stride(0) >= page_size*{bytes_per_token}; "
            f"got page_size(shape[1])={page_size}, stride(0)={page_bytes}. "
            "A flat [num_pages, page_bytes] cache misreads page_size and "
            "causes out-of-bounds scale/rope addressing."
        )

    flat_indices = indices.reshape(B, -1).contiguous()
    topk = flat_indices.shape[1]

    total_elems = num_pages * page_bytes
    raw_flat = k_cache.as_strided((total_elems,), (1,))
    raw_uint8 = raw_flat.view(torch.uint8)
    raw_fp8 = raw_uint8.view(torch.float8_e4m3fn)
    raw_bf16 = raw_uint8.view(torch.bfloat16)

    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    out = torch.zeros(B, H, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.full(
        (B, H), float("-inf"), dtype=torch.float32, device=q.device
    )
    topk_rounded = triton.next_power_of_2(topk)

    grid = (B, H)
    _tiled_sparse_decode_kernel[grid](
        q3,
        raw_fp8,
        raw_uint8,
        raw_bf16,
        flat_indices,
        (
            topk_length
            if topk_length is not None
            else torch.empty(0, device=q.device, dtype=torch.int32)
        ),
        out,
        lse,
        softmax_scale,
        page_size,
        int(page_bytes),
        int(page_size * _TOKEN_DATA_STRIDE),
        H,
        topk,
        topk_rounded,
        topk_length is not None,
        q3.stride(0),
        q3.stride(1),
        out.stride(0),
        out.stride(1),
        flat_indices.stride(0),
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
        BLOCK_T=_PINNED_BLOCK_T,
        num_warps=_PINNED_NUM_WARPS,
        num_stages=_PINNED_NUM_STAGES,
    )
    return out.unsqueeze(1), lse.unsqueeze(1)


def _merge_partial_attn(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_lse = torch.maximum(lse1, lse2)
    w1 = torch.where(
        lse1 > -1e20, torch.exp(lse1 - max_lse), torch.zeros_like(lse1)
    )
    w2 = torch.where(
        lse2 > -1e20, torch.exp(lse2 - max_lse), torch.zeros_like(lse2)
    )
    total = (w1 + w2).clamp(min=1e-20)
    merged = (
        w1.unsqueeze(-1) * out1.float() + w2.unsqueeze(-1) * out2.float()
    ) / total.unsqueeze(-1)
    merged_lse = max_lse + torch.log(total)
    return merged.to(torch.bfloat16), merged_lse


def _apply_attn_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sink_lse = attn_sink.view(1, 1, -1).expand_as(lse)
    combined_lse = torch.logaddexp(lse, sink_lse)
    w = torch.where(
        lse > -1e20, torch.exp(lse - combined_lse), torch.zeros_like(lse)
    )
    return (out.float() * w.unsqueeze(-1)).to(torch.bfloat16), combined_lse


def flash_mla_sparse_decode_sm120(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    layer_idx: int = 0,
) -> torch.Tensor:
    """SM120 sparse MLA decode. Returns attn_out [B, 1, H, head_dim_v] bf16.

    Drop-in for flashmla_decode_torch_reference (same args/semantics): main +
    optional extra (c4/c128) caches via LSE-merge, then attn_sink normalization.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    _dt = get_decode_timer()
    with _dt.timed("attn_sm120_main", layer_idx) if _dt else nullcontext():
        out, lse = _run_triton_sparse_decode(
            q, k_cache, indices, topk_length, softmax_scale
        )

    if extra_k_cache is not None and extra_indices is not None:
        with _dt.timed("attn_sm120_extra", layer_idx) if _dt else nullcontext():
            out_extra, lse_extra = _run_triton_sparse_decode(
                q,
                extra_k_cache,
                extra_indices,
                extra_topk_length,
                softmax_scale,
            )
        with _dt.timed("attn_sm120_merge", layer_idx) if _dt else nullcontext():
            out, lse = _merge_partial_attn(out, lse, out_extra, lse_extra)

    if attn_sink is not None:
        with _dt.timed("attn_sm120_sink", layer_idx) if _dt else nullcontext():
            out, lse = _apply_attn_sink(out, lse, attn_sink)

    return out[..., :head_dim_v]


_WARMUP_TOPK_BUCKETS = (64, 128)
_WARMUP_PAGE_SIZE = 64
_warmup_done: set[tuple[int, int]] = set()


def maybe_warmup_sm120_sparse_decode(
    *,
    num_heads: int = 64,
    head_dim: int = _HEAD_DIM,
    device: torch.device | str | int = "cuda",
) -> None:
    key = (int(num_heads), int(head_dim))
    if key in _warmup_done:
        return
    _warmup_done.add(key)
    warmup_sm120_sparse_decode(
        num_heads=num_heads, head_dim=head_dim, device=device
    )


def warmup_sm120_sparse_decode(
    *,
    num_heads: int = 64,
    head_dim: int = _HEAD_DIM,
    device: torch.device | str | int = "cuda",
    topk_buckets: tuple[int, ...] = _WARMUP_TOPK_BUCKETS,
) -> None:
    """Pre-compile the kernel per topk bucket so the ~377ms first-call JIT and
    the ~371ms topk 64->128 transition do not land on live decode steps 0/64."""
    device = torch.device(device)
    if device.type != "cuda":
        return
    bytes_per_token = _TOKEN_DATA_STRIDE + _SCALE_STRIDE
    num_pages = 4
    k_cache = torch.zeros(
        num_pages,
        _WARMUP_PAGE_SIZE,
        1,
        bytes_per_token,
        dtype=torch.uint8,
        device=device,
    )
    softmax_scale = head_dim**-0.5
    capacity = num_pages * _WARMUP_PAGE_SIZE
    for topk in topk_buckets:
        q = torch.zeros(
            1, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        indices = (
            torch.arange(topk, dtype=torch.int32, device=device) % capacity
        ).view(1, topk)
        topk_length = torch.full(
            (1,), min(topk, capacity), dtype=torch.int32, device=device
        )
        _run_triton_sparse_decode(
            q, k_cache, indices, topk_length, softmax_scale
        )
    torch.cuda.synchronize(device)
