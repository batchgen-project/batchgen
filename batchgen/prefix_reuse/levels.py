"""Cut one prepacked chunk into the prefix levels a stateful layer needs.

A full-attention layer serves a whole chunk in one call: it writes every row's
KV first, so a row may read a pooled segment computed beside it in the same
chunk (``same_chunk_ancestors``). A recurrent layer cannot. A row that
continues a segment must start from the state that segment ended with, so it
has to run after it.

The rows of a chunk are therefore grouped by LEVEL: how many pooled segments
the row reads. A level-0 row starts from nothing; a level-``l`` row starts from
the state of its parent segment, which is a level-``l - 1`` row of this chunk
or of an earlier one. Levels run in order, each as one packed call over a
contiguous run of rows, and between two levels the parent's state is forked
into each child's slot.

Pure: index math on CPU lists, and per level per chunk one bounds tensor plus
the state rows its fork copies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Collection, Mapping, Optional, Sequence

import torch

if TYPE_CHECKING:
    from batchgen.prefix_reuse.executor import ChunkBatch
    from batchgen.prefix_reuse.wave_plan import WavePrefixPlan


@dataclass(frozen=True)
class ChunkLevel:
    """One level of a chunk: a contiguous run of rows and their token range."""

    depth: int
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    # The level's rows as cu_seqlens rebased to ``token_start``, so a packed
    # call over ``[token_start, token_end)`` is a legal varlen batch.
    bounds: torch.Tensor


def order_chunk_items_by_level(plan: "WavePrefixPlan") -> "WavePrefixPlan":
    """Reorder every chunk's items so each level is a contiguous run of rows.

    Stable by level, and a row's level is one more than its parent segment's,
    so an ancestor still precedes every row that reads it and every
    ``validate_wave_prefix_plan`` invariant is preserved. Nothing else about a
    plan changes: the same items are computed in the same chunks.
    """

    chunks = tuple(
        replace(
            chunk,
            items=tuple(
                sorted(chunk.items, key=lambda item: len(item.prefix_segments))
            ),
        )
        for chunk in plan.chunks
    )
    return replace(plan, chunks=chunks)


def chunk_levels(batch: "ChunkBatch") -> tuple[ChunkLevel, ...]:
    """The levels of one chunk, in order; raises if they are not contiguous.

    Build this once per chunk and reuse it across the layers: the bounds
    tensors are what a per-level kernel call keys its own plan cache on.
    """

    runs: list[list[int]] = []
    for row, item in enumerate(batch.items):
        depth = len(item.prefix_segments)
        if runs and runs[-1][0] == depth:
            runs[-1][2] = row + 1
        else:
            runs.append([depth, row, row + 1])
    depths = [depth for depth, _, _ in runs]
    if depths != sorted(set(depths)):
        raise ValueError(
            f"chunk rows are not grouped by level: run depths {depths}; "
            "order the plan with order_chunk_items_by_level first"
        )

    bounds = batch.row_bounds
    dtype, device = batch.cu_seqlens_q.dtype, batch.cu_seqlens_q.device
    return tuple(
        ChunkLevel(
            depth=depth,
            row_start=row_start,
            row_end=row_end,
            token_start=bounds[row_start],
            token_end=bounds[row_end],
            bounds=torch.tensor(
                [bound - bounds[row_start] for bound in bounds[row_start:row_end + 1]],
                dtype=dtype, device=device,
            ),
        )
        for depth, row_start, row_end in runs
    )


def fork_pairs(
    batch: "ChunkBatch",
    segment_sids: Mapping[int, int],
    materialized: Collection[int] = (),
) -> tuple[tuple[int, int], ...]:
    """``(row, parent pseudo sequence id)`` for every row that continues a segment.

    The parent is the row's nearest pooled ancestor. A segment's state lives
    under its pseudo sequence id until the wave ends, so this resolves whether
    the parent ran earlier in this chunk or in an earlier chunk of the wave.

    A MATERIALIZED parent is skipped: nothing ran it, so it left no recurrent
    state to fork. Its reader carries the state at the frontier boundary in
    its OWN slot, restored before the wave (M3 Slice 2), and forking a loaded
    segment's unwritten slot over that would replace the restored state with
    zeros -- which no later assertion could see.
    """

    loaded = frozenset(int(segment_id) for segment_id in materialized)
    pairs: list[tuple[int, int]] = []
    for row, item in enumerate(batch.items):
        if not item.prefix_segments:
            continue
        parent = item.prefix_segments[-1]
        if parent in loaded:
            continue
        if parent not in segment_sids:
            raise RuntimeError(
                f"row {row} continues segment {parent}, which has no pseudo "
                "sequence id in this wave"
            )
        pairs.append((row, int(segment_sids[parent])))
    return tuple(pairs)


@dataclass(frozen=True)
class StateLevel:
    """One level of a chunk as a recurrent layer runs it.

    ``bounds``, ``slots`` and ``has_initial_state`` are the varlen arguments of
    one packed call over ``[token_start, token_end)``. ``fork_parents`` and
    ``fork_children`` are the state rows to copy, parent to child, BEFORE that
    call; both are None when no row of the level continues a segment.
    """

    token_start: int
    token_end: int
    bounds: torch.Tensor
    slots: torch.Tensor
    has_initial_state: torch.Tensor
    fork_parents: Optional[torch.Tensor]
    fork_children: Optional[torch.Tensor]


def state_levels(
    batch: "ChunkBatch",
    *,
    segment_sids: Mapping[int, int],
    row_ids: Sequence[int],
    slots: torch.Tensor,
    has_initial_state: torch.Tensor,
    slot_of: Callable[[int], int],
    materialized: Collection[int] = (),
) -> tuple[StateLevel, ...]:
    """Cut a chunk into the levels a recurrent layer runs, state rows included.

    ``row_ids`` are the chunk's rows as sequence ids (a segment's pseudo id, a
    tail's global id) and ``slots`` / ``has_initial_state`` are the per-row
    state slot and resume flag in that same order, as the layer's own slot
    allocation produced them. A row that continues a segment reads an initial
    state even when its slot was just allocated, because the fork puts its
    parent's state there first.

    Build once per chunk and reuse it across the layers: every layer of a chunk
    sees the same rows and the same slots, and a per-level kernel call keys its
    own plan cache on the identity of the bounds tensor it is handed.
    """

    if len(row_ids) != len(batch.items) or slots.numel() != len(batch.items):
        raise ValueError(
            f"chunk has {len(batch.items)} rows but {len(row_ids)} sequence "
            f"ids and {slots.numel()} state slots"
        )
    loaded = frozenset(int(segment_id) for segment_id in materialized)
    parents = dict(fork_pairs(batch, segment_sids, loaded))
    # A row reading a materialized segment resumes from its own slot, so that
    # slot MUST already hold the state at the segment's boundary: the restore
    # ran before this wave. If it did not, the row would start from zeros and
    # attend over a prefix it never produced -- wrong output, no symptom. The
    # ordering the executor relies on is therefore asserted here, at the one
    # point that can still see it.
    if loaded:
        for row, item in enumerate(batch.items):
            if not item.prefix_segments:
                continue
            if item.prefix_segments[-1] not in loaded:
                continue
            if not bool(has_initial_state[row].item()):
                raise RuntimeError(
                    f"row {row} (sequence {row_ids[row]}) reads materialized "
                    f"segment {item.prefix_segments[-1]} but holds no "
                    "recurrent state at its boundary; the trajectory restore "
                    "must bind this row's state slot before the wave's forward"
                )
    resumes = has_initial_state | torch.tensor(
        [row in parents for row in range(len(batch.items))],
        dtype=has_initial_state.dtype, device=has_initial_state.device,
    )
    levels = []
    for level in chunk_levels(batch):
        children = [
            row for row in range(level.row_start, level.row_end) if row in parents
        ]
        levels.append(StateLevel(
            token_start=level.token_start,
            token_end=level.token_end,
            bounds=level.bounds,
            slots=slots[level.row_start:level.row_end],
            has_initial_state=resumes[level.row_start:level.row_end],
            fork_parents=None if not children else torch.tensor(
                [slot_of(parents[row]) for row in children],
                dtype=torch.long, device=slots.device,
            ),
            fork_children=None if not children else torch.tensor(
                [slot_of(row_ids[row]) for row in children],
                dtype=torch.long, device=slots.device,
            ),
        ))
    return tuple(levels)
