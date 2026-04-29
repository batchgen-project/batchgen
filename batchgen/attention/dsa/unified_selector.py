"""Unified BF16 selected-KV helpers for GLM-5 DSA decode.

This eager implementation is the correctness contract for the future CUDA
selector kernel. It accepts a mixed local batch and produces one FlashMLA-ready
selected MLA KV tensor where short rows use dense prefix tokens and long rows
use indexer-selected top-k tokens.
"""

from __future__ import annotations

import torch


def select_mla_kv_for_flashmla_bf16(
    primary_blocked_k: torch.Tensor,
    primary_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    long_topk_indices: torch.Tensor,
    *,
    index_topk: int = 2048,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select BF16 MLA KV for one dense FlashMLA call.

    Args:
        primary_blocked_k: Primary MLA paged KV cache with shape
            ``[num_pages, page_size, num_k_heads, kv_dim]``.
        primary_page_table: Batch-aligned logical-to-physical page table with
            shape ``[B, max_pages]`` and ``-1`` for missing pages.
        cache_seqlens: Per-row cache lengths, including the new decode token,
            shape ``[B]``.
        long_topk_indices: Indexer-selected logical token indices for rows with
            ``cache_seqlens > index_topk``, shape ``[B, index_topk]``. Values
            for short rows are ignored.
        index_topk: Fixed selected-token count. GLM-5 uses 2048.
        page_size: KV page size. GLM-5/FlashMLA uses 64.

    Returns:
        ``(selected_mla_kv, selected_lengths, selected_indices, row_modes)``:

        - selected_mla_kv: ``[B, index_topk, num_k_heads, kv_dim]``.
        - selected_lengths: ``min(cache_seqlens, index_topk)`` as int32.
        - selected_indices: logical indices used per output slot, or ``-1``
          for ignored/invalid tail positions.
        - row_modes: int32 ``0`` for dense-short rows and ``1`` for long rows.
    """

    _validate_selector_inputs(
        primary_blocked_k,
        primary_page_table,
        cache_seqlens,
        long_topk_indices,
        index_topk=index_topk,
        page_size=page_size,
    )

    device = primary_blocked_k.device
    batch_size = cache_seqlens.shape[0]
    max_pages = primary_page_table.shape[1]
    num_k_heads = primary_blocked_k.shape[2]
    kv_dim = primary_blocked_k.shape[3]

    seqlens_long = cache_seqlens.to(device=device, dtype=torch.long)
    is_long = seqlens_long > index_topk
    dense_positions = torch.arange(
        index_topk,
        device=device,
        dtype=torch.long,
    ).expand(batch_size, index_topk)
    long_positions = long_topk_indices.to(device=device, dtype=torch.long)
    logical_indices = torch.where(is_long.unsqueeze(1), long_positions, dense_positions)

    valid = (logical_indices >= 0) & (logical_indices < seqlens_long.unsqueeze(1))
    safe_logical = logical_indices.clamp_min(0)
    logical_page = torch.div(safe_logical, page_size, rounding_mode="floor")
    page_offset = safe_logical - logical_page * page_size

    page_in_table = logical_page < max_pages
    logical_page_clamped = logical_page.clamp(max=max_pages - 1)
    physical_page = torch.gather(
        primary_page_table.to(device=device, dtype=torch.long),
        1,
        logical_page_clamped,
    )
    valid = valid & page_in_table & (physical_page >= 0)

    flat_index = physical_page.clamp_min(0) * page_size + page_offset
    flat_k = primary_blocked_k.reshape(-1, num_k_heads * kv_dim)
    gathered = flat_k[flat_index.reshape(-1)].view(
        batch_size,
        index_topk,
        num_k_heads,
        kv_dim,
    )
    selected_mla_kv = torch.where(
        valid.view(batch_size, index_topk, 1, 1),
        gathered,
        torch.zeros((), dtype=gathered.dtype, device=device),
    )
    selected_indices = torch.where(
        valid,
        logical_indices,
        torch.full_like(logical_indices, -1),
    )
    selected_lengths = torch.minimum(
        cache_seqlens.to(device=device, dtype=torch.int32),
        torch.full((batch_size,), index_topk, dtype=torch.int32, device=device),
    )
    row_modes = is_long.to(torch.int32)

    return selected_mla_kv, selected_lengths, selected_indices, row_modes


def view_selected_mla_kv_as_flashmla_pages(
    selected_mla_kv: torch.Tensor,
    *,
    page_size: int = 64,
) -> torch.Tensor:
    """View ``[B, topk, H_KV, D]`` selected KV as FlashMLA paged KV."""

    if selected_mla_kv.ndim != 4:
        raise ValueError(
            "selected_mla_kv must have shape [B, topk, num_k_heads, kv_dim], "
            f"got {tuple(selected_mla_kv.shape)}"
        )
    batch_size, index_topk, num_k_heads, kv_dim = selected_mla_kv.shape
    if index_topk % page_size != 0:
        raise ValueError(
            f"index_topk must be divisible by page_size, got {index_topk=} {page_size=}"
        )
    if not selected_mla_kv.is_contiguous():
        raise ValueError("selected_mla_kv must be contiguous to view as FlashMLA pages")
    return selected_mla_kv.view(
        batch_size * (index_topk // page_size),
        page_size,
        num_k_heads,
        kv_dim,
    )


def make_flashmla_selected_block_table(
    batch_size: int,
    *,
    index_topk: int = 2048,
    page_size: int = 64,
    device: torch.device | str,
) -> torch.Tensor:
    """Create the sequential block table for selected-KV FlashMLA pages."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {index_topk}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if index_topk % page_size != 0:
        raise ValueError(
            f"index_topk must be divisible by page_size, got {index_topk=} {page_size=}"
        )
    pages_per_row = index_topk // page_size
    return torch.arange(
        batch_size * pages_per_row,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, pages_per_row)


def _validate_selector_inputs(
    primary_blocked_k: torch.Tensor,
    primary_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    long_topk_indices: torch.Tensor,
    *,
    index_topk: int,
    page_size: int,
) -> None:
    if primary_blocked_k.ndim != 4:
        raise ValueError(
            "primary_blocked_k must have shape [num_pages, page_size, num_k_heads, kv_dim], "
            f"got {tuple(primary_blocked_k.shape)}"
        )
    if primary_blocked_k.dtype != torch.bfloat16:
        raise TypeError(
            "select_mla_kv_for_flashmla_bf16 expects BF16 primary KV, "
            f"got {primary_blocked_k.dtype}"
        )
    if primary_blocked_k.shape[1] != page_size:
        raise ValueError(
            "primary_blocked_k page dimension must match page_size, "
            f"got {primary_blocked_k.shape[1]} vs {page_size}"
        )
    if primary_page_table.ndim != 2:
        raise ValueError(
            "primary_page_table must have shape [B, max_pages], "
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
    batch_size = cache_seqlens.shape[0]
    if primary_page_table.shape[0] != batch_size:
        raise ValueError(
            "primary_page_table rows must match cache_seqlens, "
            f"got {primary_page_table.shape[0]} vs {batch_size}"
        )
    if long_topk_indices.shape != (batch_size, index_topk):
        raise ValueError(
            "long_topk_indices must have shape [B, index_topk], "
            f"got {tuple(long_topk_indices.shape)} expected {(batch_size, index_topk)}"
        )
    if primary_blocked_k.device != primary_page_table.device:
        raise ValueError("primary_blocked_k and primary_page_table must share device")
    if primary_blocked_k.device != cache_seqlens.device:
        raise ValueError("primary_blocked_k and cache_seqlens must share device")
    if primary_blocked_k.device != long_topk_indices.device:
        raise ValueError("primary_blocked_k and long_topk_indices must share device")
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
    if index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {index_topk}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if primary_page_table.shape[1] <= 0:
        raise ValueError("primary_page_table must have at least one page column")
