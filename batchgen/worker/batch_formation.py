"""Rank-assignment planner for batch formation.

Slice 4 of the worker decouple initiative (issue #174). Ports the greedy
bin-packing rank-assignment algorithm previously inlined on the worker
into a pure planner that returns a typed plan; the worker remains the
sole mutator of ``global_batch``.

NOTE: its only caller was ``BatchGenWorker._assign_sequences_to_ranks``,
which went away with the legacy non-pool path — pool mode assigns ranks
per admission through ``_assign_admitted_sequences_to_ranks``. Nothing in
the runtime calls this planner today.
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
