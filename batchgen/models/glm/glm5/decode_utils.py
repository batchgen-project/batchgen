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
    """Reorder page-table rows into current-batch order."""
    row_indices = slot_indices.to(device=block_table.device, dtype=torch.long)
    if block_table.shape[0] < row_indices.shape[0]:
        raise RuntimeError(
            "GLM-5 decode block-table row mismatch: "
            f"rows={block_table.shape[0]}, batch={row_indices.shape[0]}"
        )
    expected = torch.arange(
        row_indices.shape[0], device=block_table.device, dtype=torch.long
    )
    if block_table.shape[0] == row_indices.shape[0] and torch.equal(
        row_indices, expected
    ):
        return block_table
    return block_table.index_select(0, row_indices)
