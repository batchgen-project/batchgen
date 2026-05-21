from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_paged_kv_token_update_kernel(
    k_cache,
    k_tokens,
    v_cache,
    v_tokens,
    page_table,
    slot_indices,
    token_indices,
    k_cache_stride_page: tl.constexpr,
    k_cache_stride_token: tl.constexpr,
    k_tokens_stride_batch: tl.constexpr,
    v_cache_stride_page: tl.constexpr,
    v_cache_stride_token: tl.constexpr,
    v_tokens_stride_batch: tl.constexpr,
    page_table_stride_slot: tl.constexpr,
    page_table_stride_page: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_VEC: tl.constexpr,
    V_VEC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_V: tl.constexpr,
):
    row = tl.program_id(0)

    token_idx = tl.load(token_indices + row)
    slot_idx = tl.load(slot_indices + row)
    valid_row = (token_idx >= 0) & (slot_idx >= 0)

    safe_token_idx = tl.maximum(token_idx, 0)
    safe_slot_idx = tl.maximum(slot_idx, 0)
    logical_page_idx = safe_token_idx // PAGE_SIZE
    token_offset = safe_token_idx - logical_page_idx * PAGE_SIZE

    physical_page_idx = tl.load(
        page_table
        + safe_slot_idx * page_table_stride_slot
        + logical_page_idx * page_table_stride_page,
        mask=valid_row,
        other=-1,
    )
    valid_row = valid_row & (physical_page_idx >= 0)
    safe_physical_page_idx = tl.maximum(physical_page_idx, 0)

    k_offsets = tl.arange(0, BLOCK_K)
    k_values = tl.load(
        k_tokens + row * k_tokens_stride_batch + k_offsets,
        mask=k_offsets < K_VEC,
        other=0.0,
    )
    tl.store(
        k_cache
        + safe_physical_page_idx * k_cache_stride_page
        + token_offset * k_cache_stride_token
        + k_offsets,
        k_values,
        mask=valid_row & (k_offsets < K_VEC),
    )

    if HAS_V:
        v_offsets = tl.arange(0, BLOCK_V)
        v_values = tl.load(
            v_tokens + row * v_tokens_stride_batch + v_offsets,
            mask=v_offsets < V_VEC,
            other=0.0,
        )
        tl.store(
            v_cache
            + safe_physical_page_idx * v_cache_stride_page
            + token_offset * v_cache_stride_token
            + v_offsets,
            v_values,
            mask=valid_row & (v_offsets < V_VEC),
        )


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


@triton.jit
def _compressed_state_overlap_update_kernel(
    state_cache,
    state_tokens,
    state_slots,
    cache_stride_slot: tl.constexpr,
    token_stride_batch: tl.constexpr,
    STATE_VEC: tl.constexpr,
    BLOCK: tl.constexpr,
    ROLLING_SIZE: tl.constexpr,
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

    ring_size = 2 * ROLLING_SIZE
    ring_offset = safe_slot - (safe_slot // ring_size) * ring_size
    base_slot = safe_slot - ring_offset
    should_roll = valid_row & (ring_offset == ring_size - 1)

    for rolling_offset in tl.static_range(0, ROLLING_SIZE):
        src_slot = base_slot + ROLLING_SIZE + rolling_offset
        dst_slot = base_slot + rolling_offset
        if rolling_offset == ROLLING_SIZE - 1:
            rolling_values = values
        else:
            rolling_values = tl.load(
                state_cache + src_slot * cache_stride_slot + offsets,
                mask=offsets < STATE_VEC,
                other=0.0,
            )
        tl.store(
            state_cache + dst_slot * cache_stride_slot + offsets,
            rolling_values,
            mask=should_roll & (offsets < STATE_VEC),
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 1


def run_masked_paged_kv_token_update_fused(
    *,
    k_cache: torch.Tensor,
    k_tokens: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size_tokens: int,
    v_cache: Optional[torch.Tensor] = None,
    v_tokens: Optional[torch.Tensor] = None,
) -> None:
    """Write paged KV rows, skipping rows whose token index is negative."""

    if k_tokens.ndim != 2:
        raise ValueError("k_tokens must be a 2D flattened token tensor")
    if not k_cache.is_contiguous() or not k_tokens.is_contiguous():
        raise ValueError("k_cache and k_tokens must be contiguous")
    if not page_table.is_contiguous():
        raise ValueError("page_table must be contiguous")
    if not slot_indices.is_contiguous() or not token_indices.is_contiguous():
        raise ValueError("slot_indices and token_indices must be contiguous")

    batch_size = int(k_tokens.shape[0])
    if int(slot_indices.shape[0]) != batch_size:
        raise ValueError("slot_indices must align with k_tokens batch size")
    if int(token_indices.shape[0]) != batch_size:
        raise ValueError("token_indices must align with k_tokens batch size")

    has_v = v_cache is not None and v_tokens is not None
    if has_v:
        assert v_cache is not None
        assert v_tokens is not None
        if v_tokens.ndim != 2:
            raise ValueError("v_tokens must be a 2D flattened token tensor")
        if not v_cache.is_contiguous() or not v_tokens.is_contiguous():
            raise ValueError("v_cache and v_tokens must be contiguous")
        if int(v_tokens.shape[0]) != batch_size:
            raise ValueError("v_tokens must align with k_tokens batch size")
        v_vec = int(v_tokens.shape[1])
        v_cache_arg = v_cache
        v_tokens_arg = v_tokens
        v_cache_stride_page = int(v_cache.stride(0))
        v_cache_stride_token = int(v_cache.stride(1))
        v_tokens_stride_batch = int(v_tokens.stride(0))
    else:
        v_vec = 1
        v_cache_arg = k_cache
        v_tokens_arg = k_tokens
        v_cache_stride_page = 0
        v_cache_stride_token = 0
        v_tokens_stride_batch = 0

    k_vec = int(k_tokens.shape[1])
    block_k = _next_power_of_2(k_vec)
    block_v = _next_power_of_2(v_vec)
    _masked_paged_kv_token_update_kernel[(batch_size,)](
        k_cache,
        k_tokens,
        v_cache_arg,
        v_tokens_arg,
        page_table,
        slot_indices,
        token_indices,
        int(k_cache.stride(0)),
        int(k_cache.stride(1)),
        int(k_tokens.stride(0)),
        v_cache_stride_page,
        v_cache_stride_token,
        v_tokens_stride_batch,
        int(page_table.stride(0)),
        int(page_table.stride(1)),
        int(page_size_tokens),
        k_vec,
        v_vec,
        block_k,
        block_v,
        has_v,
        num_warps=max(_num_warps(block_k), _num_warps(block_v)),
    )


def run_compressed_state_update(
    *,
    state_cache: torch.Tensor,
    state_tokens: torch.Tensor,
    state_slots: torch.Tensor,
    overlap: bool = False,
    rolling_size: int = 0,
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
    state_slots = state_slots.to(device=state_cache.device, dtype=torch.int32)
    if overlap:
        rolling_size = int(rolling_size)
        if rolling_size <= 0:
            raise ValueError("rolling_size must be positive when overlap=True")
        if int(state_cache.shape[0]) % (2 * rolling_size) != 0:
            raise ValueError(
                "state_cache slot dimension must be divisible by "
                "2 * rolling_size when overlap=True"
            )
        _compressed_state_overlap_update_kernel[(batch_size,)](
            state_cache,
            state_tokens,
            state_slots,
            int(state_cache.stride(0)),
            int(state_tokens.stride(0)),
            state_vec,
            block,
            rolling_size,
            num_warps=_num_warps(block),
        )
    else:
        _compressed_state_update_kernel[(batch_size,)](
            state_cache,
            state_tokens,
            state_slots,
            int(state_cache.stride(0)),
            int(state_tokens.stride(0)),
            state_vec,
            block,
            num_warps=_num_warps(block),
        )


__all__ = [
    "run_compressed_state_update",
    "run_masked_paged_kv_token_update_fused",
]
