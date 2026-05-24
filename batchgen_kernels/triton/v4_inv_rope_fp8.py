# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _inv_rope_fp8_kernel(
    o_ptr,
    positions_ptr,
    cache_ptr,
    out_ptr,
    scale_ptr,
    o_stride_token,
    o_stride_head,
    o_stride_dim,
    positions_stride,
    cache_stride_pos,
    cache_stride_dim,
    out_stride_group,
    out_stride_token,
    out_stride_dim,
    scale_stride_group,
    scale_stride_token,
    scale_stride_block,
    heads_per_group,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START_IN_BLOCK: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_group = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    head_in_group = pid_block // CHUNKS_PER_HEAD
    chunk_in_head = pid_block % CHUNKS_PER_HEAD
    head_idx = pid_group * heads_per_group + head_in_group

    offsets = tl.arange(0, BLOCK_SIZE)
    input_base = (
        o_ptr
        + pid_token * o_stride_token
        + head_idx * o_stride_head
        + chunk_in_head * BLOCK_SIZE * o_stride_dim
    )
    x = tl.load(input_base + offsets * o_stride_dim).to(tl.float32)

    if chunk_in_head == CHUNKS_PER_HEAD - 1:
        pos = tl.load(positions_ptr + pid_token * positions_stride)
        cache_base = cache_ptr + pos * cache_stride_pos
        rope_local = offsets - ROPE_START_IN_BLOCK
        is_rope = offsets >= ROPE_START_IN_BLOCK
        partner = offsets ^ 1
        x_partner = tl.load(
            input_base + partner * o_stride_dim,
            mask=is_rope,
            other=0.0,
        ).to(tl.float32)
        cs_idx = tl.maximum(rope_local >> 1, 0)
        cos_v = tl.load(
            cache_base + cs_idx * cache_stride_dim,
            mask=is_rope,
            other=1.0,
        )
        sin_v = tl.load(
            cache_base + (HALF_ROPE + cs_idx) * cache_stride_dim,
            mask=is_rope,
            other=0.0,
        )
        even_out = x * cos_v + x_partner * sin_v
        odd_out = x * cos_v - x_partner * sin_v
        rotated = tl.where((rope_local & 1) == 0, even_out, odd_out)
        x = tl.where(is_rope, rotated, x)

    absmax = tl.max(tl.abs(x), axis=0)
    scale = absmax / FP8_MAX
    safe_scale = tl.where(scale > 0.0, scale, 1.0)
    x_fp8 = tl.clamp(x / safe_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

    out_base = (
        out_ptr
        + pid_group * out_stride_group
        + pid_token * out_stride_token
        + pid_block * BLOCK_SIZE * out_stride_dim
    )
    tl.store(out_base + offsets * out_stride_dim, x_fp8)
    tl.store(
        scale_ptr
        + pid_group * scale_stride_group
        + pid_token * scale_stride_token
        + pid_block * scale_stride_block,
        scale,
    )


def apply_inverse_rope(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    assert o.ndim == 3
    assert positions.ndim == 1 and positions.shape[0] == o.shape[0]
    assert o.shape[-1] >= rope_dim and rope_dim % 2 == 0
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[-1] == rope_dim

    out = o.clone()
    half_rope = rope_dim // 2
    rope = out[..., -rope_dim:].float().view(*out.shape[:-1], half_rope, 2)
    pos_cache = cos_sin_cache.index_select(0, positions)
    cos = pos_cache[:, :half_rope].unsqueeze(1)
    sin = pos_cache[:, half_rope:].unsqueeze(1)

    even = rope[..., 0]
    odd = rope[..., 1]
    inv_even = even * cos + odd * sin
    inv_odd = odd * cos - even * sin
    out[..., -rope_dim:] = (
        torch.stack((inv_even, inv_odd), dim=-1).flatten(-2).to(o.dtype)
    )
    return out


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_groups: int,
    rope_dim: int = 64,
    quant_group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert o.ndim == 3
    assert o.is_cuda and positions.is_cuda and cos_sin_cache.is_cuda
    assert o.dtype == torch.bfloat16
    assert positions.dtype == torch.int64
    assert cos_sin_cache.dtype == torch.float32
    assert positions.ndim == 1 and positions.shape[0] == o.shape[0]
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[-1] == rope_dim
    assert o.shape[-1] % quant_group_size == 0
    assert rope_dim % 2 == 0
    assert o.shape[1] % o_groups == 0

    num_tokens, num_heads, head_dim = o.shape
    heads_per_group = num_heads // o_groups
    d = heads_per_group * head_dim
    num_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size
    rope_start_in_head = head_dim - rope_dim
    last_block_start = (chunks_per_head - 1) * quant_group_size
    rope_start_in_block = rope_start_in_head - last_block_start

    assert d % quant_group_size == 0
    assert head_dim % quant_group_size == 0
    assert last_block_start <= rope_start_in_head < head_dim
    assert 0 <= rope_start_in_block < quant_group_size
    assert rope_start_in_block + rope_dim <= quant_group_size

    o_fp8 = torch.empty(
        (o_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    o_scale = torch.empty(
        (o_groups, num_tokens, num_blocks),
        dtype=torch.float32,
        device=o.device,
    )
    if num_tokens == 0:
        return o_fp8, o_scale

    _inv_rope_fp8_kernel[(num_tokens, o_groups, num_blocks)](
        o,
        positions,
        cos_sin_cache,
        o_fp8,
        o_scale,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        positions.stride(0),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        o_fp8.stride(0),
        o_fp8.stride(1),
        o_fp8.stride(2),
        o_scale.stride(0),
        o_scale.stride(1),
        o_scale.stride(2),
        heads_per_group,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        BLOCK_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START_IN_BLOCK=rope_start_in_block,
        HALF_ROPE=rope_dim // 2,
        num_warps=4,
        num_stages=1,
    )
    return o_fp8, o_scale


__all__ = ["apply_inverse_rope", "fused_inv_rope_fp8_quant"]
