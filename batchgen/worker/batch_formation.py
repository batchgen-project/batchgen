"""Rank-assignment planner for batch formation.

Slice 4 of the worker decouple initiative (issue #174). Ports the greedy
bin-packing rank-assignment algorithm previously inlined on
``BatchGenWorker._assign_sequences_to_ranks`` into a pure planner that
returns a typed plan; the worker remains the sole mutator of
``global_batch``.

Out of scope for this slice: ``_tokenize_global_batch`` (229 LOC,
cross-rank ``all_gather_object`` + heavy ``SequenceEntry`` mutation) and
``_build_local_query_book`` (mutates 5 worker fields). Both belong in
later sub-slices once the underlying ``global_batch`` mutation surface
has stabilized.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from batchgen.sequence import SequenceBatch


logger = logging.getLogger(__name__)


# Attention tile granularity (128 tokens) used to balance per-rank workload.
TILE_SIZE: int = 128


@dataclass(frozen=True)
class BatchFormationContext:
    """Frozen snapshot passed to ``BatchFormation`` calls.

    Worker constructs one from ``self.world_size``, ``self.rank``, and
    ``self.global_batch`` per call site.
    """

    world_size: int
    rank: int
    global_batch: "SequenceBatch"


@dataclass(frozen=True)
class RankAssignmentPlan:
    """Greedy bin-packing assignment.

    Attributes:
        assignments: uuid → target rank
        tiles_per_rank: tuple of total predicted attention tiles per rank,
            indexed by rank id. Useful for logging / imbalance diagnostics.
    """

    assignments: Mapping[str, int]
    tiles_per_rank: tuple


class BatchFormation:
    """Stateless planner for cross-rank batch composition."""

    @staticmethod
    def plan_rank_assignment(ctx: BatchFormationContext) -> RankAssignmentPlan:
        """Greedy bin-pack ``global_batch`` sequences across ranks.

        Sort sequences by predicted total context length (descending);
        for each, assign to the rank with the fewest accumulated tiles.
        All ranks execute this identically to maintain consistent
        assignment without explicit cross-rank sync.

        Worker is responsible for applying the plan via
        ``global_batch.assign_rank(uuid, rank)`` for each entry — the
        handler itself is non-mutating.
        """
        if ctx.global_batch is None:
            raise RuntimeError("Global batch not initialized")

        sequences = list(ctx.global_batch)
        # Larger sequences first → better bin-packing balance
        sequences.sort(
            key=lambda s: s.prompt_length + s.max_decode_length,
            reverse=True,
        )

        rank_tiles = [0] * ctx.world_size
        assignments: dict[str, int] = {}

        for seq in sequences:
            predicted_context = seq.prompt_length + seq.max_decode_length
            # ceil_div by TILE_SIZE
            predicted_tiles = (predicted_context + TILE_SIZE - 1) // TILE_SIZE

            target_rank = rank_tiles.index(min(rank_tiles))
            assignments[seq.uuid] = target_rank
            rank_tiles[target_rank] += predicted_tiles

        return RankAssignmentPlan(
            assignments=assignments,
            tiles_per_rank=tuple(rank_tiles),
        )
