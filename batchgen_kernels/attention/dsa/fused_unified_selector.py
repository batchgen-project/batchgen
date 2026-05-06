"""Unified BF16 DSA selected-KV selector.

This Triton kernel combines short-row dense selection and long-row indexer
top-k selection while gathering from the paged MLA KV cache into the fixed
FlashMLA dense contract: ``[B, 2048, 1, 576]`` for GLM-5.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _unified_select_kernel(
    blocked_kv_ptr,
    block_table_ptr,
    slot_indices_ptr,
    cache_seqlens_ptr,
    long_topk_ptr,
    selected_ptr,
    selected_lengths_ptr,
    selected_indices_ptr,
    row_modes_ptr,
    index_topk: tl.constexpr,
    page_size: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    D: tl.constexpr,
    STORE_INDICES: tl.constexpr,
    USE_SLOT_INDICES: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    slot = pid_b
    if USE_SLOT_INDICES:
        slot = tl.load(slot_indices_ptr + pid_b).to(tl.int64)
    slot_valid = slot >= 0
    safe_slot = tl.maximum(slot, 0)

    t_start = pid_t * BLOCK_T
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    seqlen = tl.load(cache_seqlens_ptr + pid_b).to(tl.int64)
    is_long = seqlen > index_topk

    if pid_t == 0:
        selected_len = tl.minimum(seqlen, index_topk).to(tl.int32)
        tl.store(selected_lengths_ptr + pid_b, selected_len)
        tl.store(row_modes_ptr + pid_b, tl.where(slot_valid, is_long.to(tl.int32), 2))

    for ti in range(BLOCK_T):
        t = t_start + ti
        if t < index_topk:
            long_idx = tl.load(long_topk_ptr + pid_b * index_topk + t).to(tl.int64)
            dense_idx = t
            token_idx = tl.where(is_long, long_idx, dense_idx)
            valid = (token_idx >= 0) & (token_idx < seqlen)

            safe_token_idx = tl.maximum(token_idx, 0)
            logical_page = safe_token_idx // page_size
            page_offset = safe_token_idx - logical_page * page_size
            page_in_table = logical_page < max_pages_per_seq
            logical_page = tl.minimum(logical_page, max_pages_per_seq - 1)

            physical_page = tl.load(
                block_table_ptr + safe_slot * max_pages_per_seq + logical_page,
                mask=slot_valid & page_in_table,
                other=-1,
            ).to(tl.int64)
            valid = valid & slot_valid & page_in_table & (physical_page >= 0)
            physical_page = tl.maximum(physical_page, 0)
            flat_idx = physical_page * page_size + page_offset

            src = blocked_kv_ptr + flat_idx * D + d_offs
            vals = tl.load(src, mask=d_mask, other=0.0)
            vals = tl.where(valid, vals, tl.zeros([BLOCK_D], dtype=vals.dtype))
            dst = selected_ptr + (pid_b * index_topk + t) * D + d_offs
            tl.store(dst, vals, mask=d_mask)
            if STORE_INDICES:
                tl.store(
                    selected_indices_ptr + pid_b * index_topk + t,
                    tl.where(valid, token_idx, -1),
                )


def fused_select_mla_kv_bf16(
    primary_blocked_k: torch.Tensor,
    primary_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    long_topk_indices: torch.Tensor,
    page_size: int,
    *,
    return_indices: bool = True,
    primary_slot_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Gather selected GLM-5 BF16 MLA KV for FlashMLA dense decode."""

    primary_blocked_k = primary_blocked_k.contiguous()
    primary_page_table = primary_page_table.contiguous()
    cache_seqlens = cache_seqlens.contiguous()
    long_topk_indices = long_topk_indices.contiguous()

    batch_size, index_topk = long_topk_indices.shape
    num_heads = primary_blocked_k.shape[2]
    head_dim = primary_blocked_k.shape[3]
    dim = num_heads * head_dim
    max_pages_per_seq = primary_page_table.shape[1]

    selected_flat = torch.empty(
        batch_size,
        index_topk,
        dim,
        dtype=primary_blocked_k.dtype,
        device=primary_blocked_k.device,
    )
    selected_lengths = torch.empty(
        batch_size,
        dtype=torch.int32,
        device=primary_blocked_k.device,
    )
    selected_indices = (
        torch.empty(
            batch_size,
            index_topk,
            dtype=long_topk_indices.dtype,
            device=primary_blocked_k.device,
        )
        if return_indices
        else None
    )
    row_modes = torch.empty(
        batch_size,
        dtype=torch.int32,
        device=primary_blocked_k.device,
    )

    fused_select_mla_kv_bf16_out(
        primary_blocked_k,
        primary_page_table,
        cache_seqlens,
        long_topk_indices,
        page_size,
        selected_flat.view(batch_size, index_topk, num_heads, head_dim),
        selected_lengths,
        selected_indices,
        row_modes,
        return_indices=return_indices,
        primary_slot_indices=primary_slot_indices,
    )

    return (
        selected_flat.view(batch_size, index_topk, num_heads, head_dim),
        selected_lengths,
        selected_indices,
        row_modes,
    )


def fused_select_mla_kv_bf16_out(
    primary_blocked_k: torch.Tensor,
    primary_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    long_topk_indices: torch.Tensor,
    page_size: int,
    selected_mla_kv: torch.Tensor,
    selected_lengths: torch.Tensor,
    selected_indices: torch.Tensor | None,
    row_modes: torch.Tensor,
    *,
    return_indices: bool = True,
    primary_slot_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Out-buffer variant for CUDA graph capture."""

    primary_blocked_k = primary_blocked_k.contiguous()
    primary_page_table = primary_page_table.contiguous()
    if primary_slot_indices is not None:
        primary_slot_indices = primary_slot_indices.contiguous()
    cache_seqlens = cache_seqlens.contiguous()
    long_topk_indices = long_topk_indices.contiguous()

    batch_size, index_topk = long_topk_indices.shape
    num_heads = primary_blocked_k.shape[2]
    head_dim = primary_blocked_k.shape[3]
    dim = num_heads * head_dim
    max_pages_per_seq = primary_page_table.shape[1]
    selected_flat = selected_mla_kv.view(batch_size, index_topk, dim)
    if primary_slot_indices is not None:
        if primary_slot_indices.shape != (batch_size,):
            raise ValueError(
                "primary_slot_indices must have shape [B], "
                f"got {tuple(primary_slot_indices.shape)}"
            )
        if primary_slot_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "primary_slot_indices must be int32/int64, "
                f"got {primary_slot_indices.dtype}"
            )

    block_d = triton.next_power_of_2(dim)
    total_work = batch_size * index_topk
    if total_work <= 4096:
        block_t = 1
    elif total_work <= 32768:
        block_t = 8
    else:
        block_t = 32
    grid = (batch_size, triton.cdiv(index_topk, block_t))
    _unified_select_kernel[grid](
        primary_blocked_k.reshape(-1, dim),
        primary_page_table,
        primary_slot_indices if primary_slot_indices is not None else cache_seqlens,
        cache_seqlens,
        long_topk_indices,
        selected_flat,
        selected_lengths,
        selected_indices if selected_indices is not None else selected_flat,
        row_modes,
        index_topk=index_topk,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
        D=dim,
        STORE_INDICES=return_indices,
        USE_SLOT_INDICES=primary_slot_indices is not None,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )

    return (
        selected_mla_kv,
        selected_lengths,
        selected_indices,
        row_modes,
    )
