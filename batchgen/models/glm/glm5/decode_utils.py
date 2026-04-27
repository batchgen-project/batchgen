from collections.abc import Mapping, Sequence
from typing import Union

import torch


def clamp_token_indices_to_seqlens(
    token_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Clamp token indices so each row never exceeds its last valid token."""
    if token_indices.dim() != 2:
        raise RuntimeError(
            f"token_indices must be 2D [B,K], got shape={tuple(token_indices.shape)}"
        )
    if cache_seqlens.dim() != 1:
        raise RuntimeError(
            f"cache_seqlens must be 1D [B], got shape={tuple(cache_seqlens.shape)}"
        )
    if token_indices.shape[0] != cache_seqlens.shape[0]:
        raise RuntimeError(
            "token_indices/cache_seqlens batch mismatch: "
            f"token_indices.shape[0]={token_indices.shape[0]}, "
            f"cache_seqlens.shape[0]={cache_seqlens.shape[0]}"
        )
    if cache_seqlens.numel() and int(cache_seqlens.min().item()) <= 0:
        raise RuntimeError(
            f"cache_seqlens must be positive for decode, min={int(cache_seqlens.min().item())}"
        )
    cap = (
        cache_seqlens.to(device=token_indices.device, dtype=torch.long).reshape(token_indices.shape[0]) - 1
    ).clamp(min=0).unsqueeze(-1)
    return torch.minimum(token_indices.to(torch.long), cap)


def build_clamped_dense_token_indices(
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    """Build dense token indices capped at each row's last valid token."""
    if cache_seqlens.dim() != 1:
        raise RuntimeError(
            f"cache_seqlens must be 1D [B], got shape={tuple(cache_seqlens.shape)}"
        )
    if int(max_seqlen) <= 0:
        raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")
    if cache_seqlens.numel() and int(cache_seqlens.max().item()) > int(max_seqlen):
        raise RuntimeError(
            f"max_seqlen={max_seqlen} is smaller than cache_seqlens.max()={int(cache_seqlens.max().item())}"
        )
    batch_size = cache_seqlens.shape[0]
    base = torch.arange(
        max_seqlen, device=device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    return clamp_token_indices_to_seqlens(base, cache_seqlens.to(device=device))


def build_batch_slot_indices(
    current_batch: Sequence[int],
    seq_id_to_slot: Mapping[int, int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve current-batch sequence ids to explicit page-table slot indices."""
    if len(current_batch) != batch_size:
        raise RuntimeError(
            "GLM-5 decode batch/slot mismatch: "
            f"batch_size={batch_size}, current_batch_len={len(current_batch)}"
        )

    missing = [int(seq_id) for seq_id in current_batch if int(seq_id) not in seq_id_to_slot]
    if missing:
        raise RuntimeError(
            "GLM-5 decode missing page-table slots for current batch: "
            f"{missing[:8]}"
        )

    return torch.tensor(
        [int(seq_id_to_slot[int(seq_id)]) for seq_id in current_batch],
        dtype=torch.int32,
        device=device,
    )


def reorder_block_table_to_batch_slots(
    block_table: torch.Tensor,
    slot_indices: torch.Tensor,
) -> torch.Tensor:
    """Reorder page-table rows into current-batch order.

    Previously guarded the index_select by a `torch.equal(row_indices,
    arange)` identity-permutation fast-path; nsys showed this produced a
    1-byte DtoH sync every layer per step (bool `.all().item()` under
    the hood). The index_select on an identity permutation is only
    O(batch × max_pages) bytes of D2D copy — cheaper than the sync the
    fast-path was trying to skip. Always do the reorder.
    """
    if block_table.dim() != 2:
        raise RuntimeError(
            f"block_table must be 2D [slots,pages], got shape={tuple(block_table.shape)}"
        )
    if slot_indices.dim() != 1:
        raise RuntimeError(
            f"slot_indices must be 1D [B], got shape={tuple(slot_indices.shape)}"
        )
    if slot_indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"slot_indices must be int32/int64, got {slot_indices.dtype}")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"block_table must be int32/int64, got {block_table.dtype}")
    row_indices = slot_indices.to(device=block_table.device, dtype=torch.long)
    if block_table.shape[0] < row_indices.shape[0]:
        raise RuntimeError(
            "GLM-5 decode block-table row mismatch: "
            f"rows={block_table.shape[0]}, batch={row_indices.shape[0]}"
        )
    if row_indices.numel():
        row_min = int(row_indices.min().item())
        row_max = int(row_indices.max().item())
        if row_min < 0 or row_max >= block_table.shape[0]:
            raise RuntimeError(
                "GLM-5 decode slot index out of range: "
                f"slot_min={row_min}, slot_max={row_max}, rows={block_table.shape[0]}"
            )
    return block_table.index_select(0, row_indices)


def build_paged_gather_cache_key(
    block_table: torch.Tensor,
    max_seqlen: int,
    page_size: int,
    *,
    page_table_version: int,
) -> tuple[int, int, int, int, int, int]:
    """Build a stable cache key for paged gather indices.

    ``block_table.data_ptr()`` alone is insufficient because the GPU page table
    is frequently rebuilt in place, so its contents can change while the tensor
    pointer stays constant.
    """
    return (
        int(page_table_version),
        int(block_table.data_ptr()),
        int(max_seqlen),
        int(page_size),
        int(block_table.shape[0]),
        int(block_table.shape[1]),
    )


def build_flat_paged_gather_indices(
    block_table: torch.Tensor,
    max_seqlen: int,
    page_size: int,
    *,
    return_invalid_mask: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Flatten paged block-table lookups into token gather indices."""
    if block_table.dim() != 2:
        raise RuntimeError(
            f"block_table must be 2D [B,pages], got shape={tuple(block_table.shape)}"
        )
    if block_table.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"block_table must be int32/int64, got {block_table.dtype}")
    if int(max_seqlen) <= 0:
        raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")
    device = block_table.device
    batch_size = block_table.shape[0]
    max_pages = block_table.shape[1]
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    required_pages = (int(max_seqlen) + int(page_size) - 1) // int(page_size)
    if required_pages > max_pages:
        raise RuntimeError(
            "GLM-5 paged gather block-table capacity mismatch: "
            f"max_seqlen={max_seqlen} requires {required_pages} pages, "
            f"but block_table has {max_pages} columns"
        )
    token_positions = torch.arange(max_seqlen, device=device)
    page_indices = (token_positions // page_size).unsqueeze(0).expand(batch_size, -1)
    page_offsets = token_positions % page_size
    physical_pages = torch.gather(block_table, 1, page_indices)
    invalid_mask = physical_pages < 0
    physical_pages = physical_pages.clamp(min=0)
    flat_idx = (physical_pages * page_size + page_offsets.unsqueeze(0)).reshape(-1).long()
    if return_invalid_mask:
        return flat_idx, invalid_mask
    return flat_idx
