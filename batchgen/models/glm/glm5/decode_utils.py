from collections.abc import Mapping, Sequence

import torch


def clamp_token_indices_to_seqlens(
    token_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Clamp token indices so each row never exceeds its last valid token."""
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
    row_indices = slot_indices.to(device=block_table.device, dtype=torch.long)
    if block_table.shape[0] < row_indices.shape[0]:
        raise RuntimeError(
            "GLM-5 decode block-table row mismatch: "
            f"rows={block_table.shape[0]}, batch={row_indices.shape[0]}"
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
) -> torch.Tensor:
    """Flatten paged block-table lookups into token gather indices."""
    device = block_table.device
    batch_size = block_table.shape[0]
    token_positions = torch.arange(max_seqlen, device=device)
    page_indices = (token_positions // page_size).unsqueeze(0).expand(batch_size, -1)
    page_offsets = token_positions % page_size
    max_pages = block_table.shape[1]
    page_indices_clamped = page_indices.clamp(max=max_pages - 1)
    physical_pages = torch.gather(block_table, 1, page_indices_clamped)
    return (physical_pages * page_size + page_offsets.unsqueeze(0)).reshape(-1).long()
