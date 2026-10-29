"""Per-replay page inputs for the GPT-OSS whole-model decode CUDA graph.

The whole-model graph has the MoE all_gather/all_reduce captured inside it, so
every rank must capture and replay the same bucket, including a rank that
holds no decode row. Such a rank passes zero-row inputs; ``CUDAGraphManager``
then fills every row of the bucket with the capture fill values (page table 0,
slot -1, cache_seqlens 0). Slot -1 makes the paged KV write skip the row, and
the rank still runs the same collective sequence with the same shapes as every
other rank.

A rank that has never held a sequence has no active page table at all, so the
zero-row case must not read it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class WholeModelReplayInputError(RuntimeError):
    """A decode row reached whole-model graph replay without a page table row."""


def whole_model_replay_page_inputs(
    *,
    active_page_table: Optional[torch.Tensor],
    slot_index_tensor: Optional[torch.Tensor],
    batch_size: int,
    max_pages_per_seq: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(page_table, slot_indices)`` for one whole-model graph replay.

    ``batch_size`` is this rank's number of decode rows (not the bucket).
    With rows, the active page table is sliced to them and padded or
    truncated to the captured ``max_pages_per_seq`` columns. With no rows,
    both inputs are empty and the replay pads the whole bucket.
    """
    batch_size = int(batch_size)
    max_pages_per_seq = int(max_pages_per_seq)
    if batch_size < 0:
        raise ValueError(f"batch_size must be >= 0, got {batch_size}")
    if max_pages_per_seq <= 0:
        raise ValueError(f"max_pages_per_seq must be positive, got {max_pages_per_seq}")

    if batch_size == 0:
        return (
            torch.empty((0, max_pages_per_seq), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
        )

    if active_page_table is None:
        raise WholeModelReplayInputError(
            f"GPT-OSS whole-model graph replay has {batch_size} decode row(s) "
            "but no active GPU page table"
        )
    if active_page_table.shape[0] < batch_size:
        raise WholeModelReplayInputError(
            f"GPT-OSS whole-model graph replay has {batch_size} decode row(s) "
            f"but the active GPU page table has {active_page_table.shape[0]} row(s)"
        )

    if slot_index_tensor is None:
        # A rebuild can leave the slot view unset; the active table is then in
        # batch order (unchanged behavior of the replay path).
        slot_index_tensor = torch.arange(
            active_page_table.shape[0], dtype=torch.int32, device=device,
        )

    page_table = active_page_table[:batch_size]
    if page_table.shape[1] < max_pages_per_seq:
        page_table = torch.nn.functional.pad(
            page_table, (0, max_pages_per_seq - page_table.shape[1]), value=0
        )
    elif page_table.shape[1] > max_pages_per_seq:
        page_table = page_table[:, :max_pages_per_seq]
    return page_table, slot_index_tensor[:batch_size]
