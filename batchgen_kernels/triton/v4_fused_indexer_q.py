# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

MXFP4_BLOCK_SIZE = 32


@triton.jit
def _get_cos_sin(cache_ptr, cache_stride, pos, half_rot_dim: tl.constexpr):
    offset = tl.arange(0, half_rot_dim)
    cos = tl.load(cache_ptr + pos * cache_stride + offset).to(tl.float32)
    sin = tl.load(cache_ptr + pos * cache_stride + offset + half_rot_dim).to(
        tl.float32
    )
    return cos, sin


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 6.0 * (2**-126))
    log2_scale = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
    log2_scale = tl.minimum(tl.maximum(log2_scale, -127.0), 127.0)
    scale = tl.math.exp2(log2_scale)
    ue8m0 = (log2_scale + 127.0).to(tl.uint8)
    packed = _fp32x2_to_fp4x2(x_lo / scale, x_hi / scale)
    return packed, ue8m0


@triton.jit
def _fused_indexer_q_fp8_kernel(
    positions_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    cache_ptr,
    cache_stride0,
    half_rot_dim: tl.constexpr,
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    head_dim: tl.constexpr,
    index_weights_ptr,
    index_weights_stride0,
    softmax_scale,
    head_scale,
    weights_out_ptr,
    weights_out_stride0,
):
    rot_dim: tl.constexpr = 2 * half_rot_dim
    nope_dim: tl.constexpr = head_dim - rot_dim

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(positions_ptr + tok_idx)
    cos, sin = _get_cos_sin(cache_ptr, cache_stride0, pos, half_rot_dim)

    base_ptr = (
        index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    )
    offset = tl.arange(0, half_rot_dim)

    rot_base = base_ptr + nope_dim
    x_even = tl.load(rot_base + offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + offset * 2 + 1).to(tl.float32)
    r_even = (x_even * cos - x_odd * sin).to(tl.bfloat16).to(tl.float32)
    r_odd = (x_odd * cos + x_even * sin).to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if nope_dim > 0:
        nope_offset = tl.arange(0, nope_dim)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))

    log2_q_scale = tl.math.ceil(
        tl.math.log2(tl.maximum(amax, 1e-4) * (1.0 / 448.0))
    )
    q_scale = tl.math.exp2(log2_q_scale)
    q_scale_inv = tl.math.exp2(-log2_q_scale)

    fp8_base = (
        index_q_fp8_ptr
        + tok_idx * index_q_fp8_stride0
        + head_idx * index_q_fp8_stride1
    )
    if nope_dim > 0:
        tl.store(
            fp8_base + nope_offset, (x_nope * q_scale_inv).to(tl.float8e4nv)
        )
    fp8_rot_base = fp8_base + nope_dim
    tl.store(
        fp8_rot_base + offset * 2, (r_even * q_scale_inv).to(tl.float8e4nv)
    )
    tl.store(
        fp8_rot_base + offset * 2 + 1,
        (r_odd * q_scale_inv).to(tl.float8e4nv),
    )

    weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride0 + head_idx
    )
    weights = weights.to(tl.float32) * q_scale_inv * softmax_scale * head_scale
    tl.store(
        weights_out_ptr + tok_idx * weights_out_stride0 + head_idx, weights
    )


@triton.jit
def _fused_indexer_q_mxfp4_kernel(
    positions_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    cache_ptr,
    cache_stride0,
    half_rot_dim: tl.constexpr,
    packed_ptr,
    packed_stride0,
    packed_stride1,
    scale_ptr,
    scale_stride0,
    scale_stride1,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    index_weights_ptr,
    index_weights_stride0,
    softmax_scale,
    head_scale,
    weights_out_ptr,
    weights_out_stride0,
):
    rot_dim: tl.constexpr = 2 * half_rot_dim
    nope_dim: tl.constexpr = head_dim - rot_dim
    num_nope_blocks: tl.constexpr = nope_dim // block_size
    num_rope_blocks: tl.constexpr = rot_dim // block_size
    half_block: tl.constexpr = block_size // 2

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    pos = tl.load(positions_ptr + tok_idx)

    q_base = (
        index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    )
    out_base = packed_ptr + tok_idx * packed_stride0 + head_idx * packed_stride1
    scale_base = scale_ptr + tok_idx * scale_stride0 + head_idx * scale_stride1
    half_offset = tl.arange(0, half_block)

    for block_idx in tl.static_range(num_nope_blocks):
        block_base = block_idx * block_size
        x_lo = tl.load(q_base + block_base + half_offset * 2).to(tl.float32)
        x_hi = tl.load(q_base + block_base + half_offset * 2 + 1).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi)
        tl.store(out_base + block_base // 2 + half_offset, packed)
        tl.store(scale_base + block_idx, ue8m0)

    rot_q_base = q_base + nope_dim
    for block_idx in tl.static_range(num_rope_blocks):
        pair_offset = block_idx * half_block + half_offset
        cos = tl.load(cache_ptr + pos * cache_stride0 + pair_offset).to(
            tl.float32
        )
        sin = tl.load(
            cache_ptr + pos * cache_stride0 + pair_offset + half_rot_dim
        ).to(tl.float32)
        x_even = tl.load(rot_q_base + pair_offset * 2).to(tl.float32)
        x_odd = tl.load(rot_q_base + pair_offset * 2 + 1).to(tl.float32)
        r_even = (x_even * cos - x_odd * sin).to(tl.bfloat16).to(tl.float32)
        r_odd = (x_odd * cos + x_even * sin).to(tl.bfloat16).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
        byte_offset = (nope_dim + block_idx * block_size) // 2
        tl.store(out_base + byte_offset + half_offset, packed)
        tl.store(scale_base + num_nope_blocks + block_idx, ue8m0)

    weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride0 + head_idx
    )
    weights = weights.to(tl.float32) * softmax_scale * head_scale
    tl.store(
        weights_out_ptr + tok_idx * weights_out_stride0 + head_idx, weights
    )


def fused_indexer_q_fp8(
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    index_weights: torch.Tensor,
    softmax_scale: float = 1.0,
    head_scale: float = 1.0,
    rope_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert index_q.ndim == 3 and index_q.is_cuda
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.is_cuda
    assert positions.ndim == 1 and positions.is_cuda
    assert index_weights.ndim == 2 and index_weights.is_cuda
    assert positions.shape[0] == index_q.shape[0] == index_weights.shape[0]
    assert index_q.shape[1] == index_weights.shape[1]
    assert rope_dim % 2 == 0 and index_q.shape[2] >= rope_dim
    assert index_q.stride(-1) == 1 and cos_sin_cache.stride(-1) == 1
    assert positions.dtype == torch.int64

    num_tokens, num_heads, head_dim = index_q.shape
    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    if num_tokens == 0:
        return index_q_fp8, weights_out

    _fused_indexer_q_fp8_kernel[(num_tokens, num_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        rope_dim // 2,
        index_q_fp8,
        index_q_fp8.stride(0),
        index_q_fp8.stride(1),
        head_dim,
        index_weights,
        index_weights.stride(0),
        softmax_scale,
        head_scale,
        weights_out,
        weights_out.stride(0),
        num_warps=1,
        num_stages=1,
    )
    return index_q_fp8, weights_out


def fused_indexer_q_mxfp4(
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    index_weights: torch.Tensor,
    softmax_scale: float = 1.0,
    head_scale: float = 1.0,
    rope_dim: int = 64,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    assert index_q.ndim == 3 and index_q.is_cuda
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.is_cuda
    assert positions.ndim == 1 and positions.is_cuda
    assert index_weights.ndim == 2 and index_weights.is_cuda
    assert positions.shape[0] == index_q.shape[0] == index_weights.shape[0]
    assert index_q.shape[1] == index_weights.shape[1]
    assert rope_dim % 2 == 0 and index_q.shape[2] >= rope_dim
    assert index_q.shape[2] % MXFP4_BLOCK_SIZE == 0
    assert (index_q.shape[2] - rope_dim) % MXFP4_BLOCK_SIZE == 0
    assert rope_dim % MXFP4_BLOCK_SIZE == 0
    assert index_q.stride(-1) == 1 and cos_sin_cache.stride(-1) == 1
    assert positions.dtype == torch.int64

    num_tokens, num_heads, head_dim = index_q.shape
    num_scale_blocks = head_dim // MXFP4_BLOCK_SIZE
    packed = torch.empty(
        (num_tokens, num_heads, head_dim // 2),
        dtype=torch.uint8,
        device=index_q.device,
    )
    scale = torch.empty(
        (num_tokens, num_heads, num_scale_blocks),
        dtype=torch.uint8,
        device=index_q.device,
    )
    weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    if num_tokens == 0:
        return (packed, scale.view(torch.int32).squeeze(-1)), weights_out

    _fused_indexer_q_mxfp4_kernel[(num_tokens, num_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        rope_dim // 2,
        packed,
        packed.stride(0),
        packed.stride(1),
        scale,
        scale.stride(0),
        scale.stride(1),
        head_dim,
        MXFP4_BLOCK_SIZE,
        index_weights,
        index_weights.stride(0),
        softmax_scale,
        head_scale,
        weights_out,
        weights_out.stride(0),
        num_warps=1,
        num_stages=1,
    )
    return (packed, scale.view(torch.int32).squeeze(-1)), weights_out


def fused_indexer_q(
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    index_weights: torch.Tensor,
    softmax_scale: float = 1.0,
    head_scale: float = 1.0,
    rope_dim: int = 64,
    use_fp4: Optional[bool] = None,
):
    if use_fp4 is None:
        use_fp4 = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 12
        )
    if use_fp4:
        return fused_indexer_q_mxfp4(
            index_q,
            cos_sin_cache,
            positions,
            index_weights,
            softmax_scale=softmax_scale,
            head_scale=head_scale,
            rope_dim=rope_dim,
        )
    return fused_indexer_q_fp8(
        index_q,
        cos_sin_cache,
        positions,
        index_weights,
        softmax_scale=softmax_scale,
        head_scale=head_scale,
        rope_dim=rope_dim,
    )


__all__ = [
    "MXFP4_BLOCK_SIZE",
    "fused_indexer_q",
    "fused_indexer_q_fp8",
    "fused_indexer_q_mxfp4",
]
