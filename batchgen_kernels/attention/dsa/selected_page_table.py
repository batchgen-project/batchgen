"""Build an FA3 page-size-1 table from GLM-5 logical top-k positions."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _transform_selected_positions_kernel(
    page_table,
    slot_indices,
    num_valid_tokens,
    cache_seqlens,
    long_topk_indices,
    physical_token_ids,
    selected_lengths,
    max_pages_per_seq: tl.constexpr,
    index_topk: tl.constexpr,
    page_size: tl.constexpr,
    has_slot_indices: tl.constexpr,
    has_valid_tokens: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * block + tl.arange(0, block)
    in_topk = offsets < index_topk

    row_valid = row >= 0
    if has_valid_tokens:
        row_valid = row < tl.load(num_valid_tokens)

    slot = row
    if has_slot_indices:
        slot = tl.load(slot_indices + row).to(tl.int64)
    slot_valid = row_valid & (slot >= 0)
    safe_slot = tl.maximum(slot, 0)

    seqlen = tl.load(cache_seqlens + row).to(tl.int64)
    selected_len = tl.minimum(seqlen, index_topk)
    selected_len = tl.where(slot_valid, selected_len, 0)
    if block_id == 0:
        tl.store(selected_lengths + row, selected_len.to(tl.int32))

    long_idx = tl.load(
        long_topk_indices + row * index_topk + offsets,
        mask=in_topk,
        other=-1,
    ).to(tl.int64)
    logical_idx = tl.where(seqlen > index_topk, long_idx, offsets)
    valid = (
        in_topk
        & slot_valid
        & (offsets < selected_len)
        & (logical_idx >= 0)
        & (logical_idx < seqlen)
    )

    safe_logical_idx = tl.maximum(logical_idx, 0)
    logical_page = safe_logical_idx // page_size
    page_offset = safe_logical_idx - logical_page * page_size
    page_in_table = logical_page < max_pages_per_seq
    safe_logical_page = tl.minimum(logical_page, max_pages_per_seq - 1)
    physical_page = tl.load(
        page_table + safe_slot * max_pages_per_seq + safe_logical_page,
        mask=valid & page_in_table,
        other=-1,
    ).to(tl.int64)
    valid = valid & page_in_table & (physical_page >= 0)
    physical_token = physical_page * page_size + page_offset
    tl.store(
        physical_token_ids + row * index_topk + offsets,
        tl.where(valid, physical_token, -1),
        mask=in_topk,
    )


def transform_selected_positions_out(
    primary_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    long_topk_indices: torch.Tensor,
    physical_token_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    *,
    page_size: int,
    primary_slot_indices: torch.Tensor | None = None,
    num_valid_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map logical selected positions to physical token IDs without copying KV."""

    if primary_page_table.ndim != 2:
        raise ValueError(
            "primary_page_table must have shape [slots, max_pages], "
            f"got {tuple(primary_page_table.shape)}"
        )
    if cache_seqlens.ndim != 1:
        raise ValueError(
            f"cache_seqlens must be 1-D, got {tuple(cache_seqlens.shape)}"
        )
    if long_topk_indices.ndim != 2:
        raise ValueError(
            "long_topk_indices must have shape [B, index_topk], "
            f"got {tuple(long_topk_indices.shape)}"
        )
    batch_size, index_topk = long_topk_indices.shape
    if cache_seqlens.shape != (batch_size,):
        raise ValueError(
            f"cache_seqlens must have shape {(batch_size,)}, "
            f"got {tuple(cache_seqlens.shape)}"
        )
    if physical_token_ids.shape != (batch_size, index_topk):
        raise ValueError(
            f"physical_token_ids must have shape {(batch_size, index_topk)}, "
            f"got {tuple(physical_token_ids.shape)}"
        )
    if selected_lengths.shape != (batch_size,):
        raise ValueError(
            f"selected_lengths must have shape {(batch_size,)}, "
            f"got {tuple(selected_lengths.shape)}"
        )
    tensors = (
        primary_page_table,
        cache_seqlens,
        long_topk_indices,
        physical_token_ids,
        selected_lengths,
    )
    if any(t.device != primary_page_table.device for t in tensors):
        raise ValueError("all selected-page-table tensors must share one device")
    if primary_page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"primary_page_table must be int32/int64, got {primary_page_table.dtype}"
        )
    if cache_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"cache_seqlens must be int32/int64, got {cache_seqlens.dtype}")
    if long_topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"long_topk_indices must be int32/int64, got {long_topk_indices.dtype}"
        )
    if physical_token_ids.dtype != torch.int32:
        raise TypeError(
            f"physical_token_ids must be int32, got {physical_token_ids.dtype}"
        )
    if selected_lengths.dtype != torch.int32:
        raise TypeError(
            f"selected_lengths must be int32, got {selected_lengths.dtype}"
        )
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if primary_page_table.shape[1] <= 0:
        raise ValueError("primary_page_table must have at least one page column")
    if primary_slot_indices is not None:
        if primary_slot_indices.shape != (batch_size,):
            raise ValueError(
                f"primary_slot_indices must have shape {(batch_size,)}, "
                f"got {tuple(primary_slot_indices.shape)}"
            )
        if primary_slot_indices.device != primary_page_table.device:
            raise ValueError("primary_slot_indices must share the page-table device")
        if primary_slot_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "primary_slot_indices must be int32/int64, "
                f"got {primary_slot_indices.dtype}"
            )
    if num_valid_tokens is not None:
        if num_valid_tokens.device != primary_page_table.device:
            raise ValueError("num_valid_tokens must share the page-table device")
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(
                f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}"
            )
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                "num_valid_tokens must contain one element, "
                f"got {tuple(num_valid_tokens.shape)}"
            )

    block = 256
    _transform_selected_positions_kernel[
        (batch_size, triton.cdiv(index_topk, block))
    ](
        primary_page_table,
        primary_slot_indices if primary_slot_indices is not None else cache_seqlens,
        num_valid_tokens if num_valid_tokens is not None else cache_seqlens,
        cache_seqlens,
        long_topk_indices,
        physical_token_ids,
        selected_lengths,
        max_pages_per_seq=primary_page_table.shape[1],
        index_topk=index_topk,
        page_size=page_size,
        has_slot_indices=primary_slot_indices is not None,
        has_valid_tokens=num_valid_tokens is not None,
        block=block,
    )
    return physical_token_ids, selected_lengths
