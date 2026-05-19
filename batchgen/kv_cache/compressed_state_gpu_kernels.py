from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compressed_state_update_kernel(
    state_cache,
    state_tokens,
    state_slots,
    cache_stride_slot: tl.constexpr,
    token_stride_batch: tl.constexpr,
    STATE_VEC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(state_slots + row)
    valid_row = slot >= 0
    safe_slot = tl.maximum(slot, 0)

    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        state_tokens + row * token_stride_batch + offsets,
        mask=offsets < STATE_VEC,
        other=0.0,
    )
    tl.store(
        state_cache + safe_slot * cache_stride_slot + offsets,
        values,
        mask=valid_row & (offsets < STATE_VEC),
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 1


def run_compressed_state_update(
    *,
    state_cache: torch.Tensor,
    state_tokens: torch.Tensor,
    state_slots: torch.Tensor,
) -> None:
    """Write raw compressor state rows into precomputed state slots."""

    if state_cache.ndim != 2:
        raise ValueError("state_cache must be a 2D tensor")
    if state_tokens.ndim != 2:
        raise ValueError("state_tokens must be a 2D tensor")
    if state_slots.ndim != 1:
        raise ValueError("state_slots must be a 1D tensor")
    if not state_cache.is_contiguous():
        raise ValueError("state_cache must be contiguous")
    if not state_tokens.is_contiguous():
        raise ValueError("state_tokens must be contiguous")
    if not state_slots.is_contiguous():
        raise ValueError("state_slots must be contiguous")

    batch_size = int(state_tokens.shape[0])
    if int(state_slots.shape[0]) != batch_size:
        raise ValueError("state_slots must align with state_tokens batch size")
    if int(state_tokens.shape[1]) != int(state_cache.shape[1]):
        raise ValueError(
            "state_tokens feature dimension must match state_cache"
        )

    state_vec = int(state_tokens.shape[1])
    block = _next_power_of_2(state_vec)
    _compressed_state_update_kernel[(batch_size,)](
        state_cache,
        state_tokens,
        state_slots.to(device=state_cache.device, dtype=torch.int32),
        int(state_cache.stride(0)),
        int(state_tokens.stride(0)),
        state_vec,
        block,
        num_warps=_num_warps(block),
    )


__all__ = ["run_compressed_state_update"]
