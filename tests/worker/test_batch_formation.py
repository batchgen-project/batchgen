"""Unit tests for `batchgen.worker.batch_formation`.

Real `SequenceBatch` / `SequenceEntry` fixtures — no mocks per Phase A
§G no-hack rule. CPU-only.
"""

from __future__ import annotations

import pytest

from batchgen.sequence import SequenceBatch, SequenceEntry
from batchgen.worker.batch_formation import (
    BatchFormation,
    BatchFormationContext,
    RankAssignmentPlan,
    TILE_SIZE,
)


def _make_seq(uuid: str, global_idx: int, prompt: int, decode: int) -> SequenceEntry:
    return SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt,
        max_decode_length=decode,
    )


@pytest.fixture
def four_seq_batch() -> SequenceBatch:
    """4 sequences with predicted contexts spanning 1× to 4× TILE_SIZE."""
    batch = SequenceBatch()
    # Predicted context = prompt + decode; in TILE_SIZE=128 units:
    # alpha: 256/128=2 tiles
    # bravo: 128/128=1 tile
    # charlie: 512/128=4 tiles
    # delta: 384/128=3 tiles
    for seq in [
        _make_seq("alpha", 0, prompt=128, decode=128),
        _make_seq("bravo", 1, prompt=64, decode=64),
        _make_seq("charlie", 2, prompt=256, decode=256),
        _make_seq("delta", 3, prompt=192, decode=192),
    ]:
        batch.add_sequence(seq)
    return batch


def test_plan_dataclass_is_frozen():
    plan = RankAssignmentPlan(assignments={"x": 0}, tiles_per_rank=(2,))
    with pytest.raises((AttributeError, Exception)):
        plan.assignments = {"y": 1}  # type: ignore[misc]


def test_assignment_world_size_1_all_one_rank(four_seq_batch):
    ctx = BatchFormationContext(world_size=1, rank=0, global_batch=four_seq_batch)
    plan = BatchFormation.plan_rank_assignment(ctx)
    # Every sequence goes to rank 0
    assert all(r == 0 for r in plan.assignments.values())
    # Sum of tiles = sum of ceil_div((prompt+decode)/TILE_SIZE) = 2+1+4+3 = 10
    assert plan.tiles_per_rank == (10,)


def test_assignment_world_size_2_balanced(four_seq_batch):
    ctx = BatchFormationContext(world_size=2, rank=0, global_batch=four_seq_batch)
    plan = BatchFormation.plan_rank_assignment(ctx)
    # Greedy sorts desc by predicted context:
    #   charlie(4) → rank 0 (tiles: [4,0])
    #   delta(3)   → rank 1 (tiles: [4,3])
    #   alpha(2)   → rank 1 (tiles: [4,5])
    #   bravo(1)   → rank 0 (tiles: [5,5])
    # Final: charlie→0, delta→1, alpha→1, bravo→0
    assert plan.assignments == {
        "charlie": 0,
        "delta": 1,
        "alpha": 1,
        "bravo": 0,
    }
    assert plan.tiles_per_rank == (5, 5)


def test_assignment_is_deterministic(four_seq_batch):
    """All ranks must produce the same plan without explicit sync."""
    ctx_rank0 = BatchFormationContext(world_size=4, rank=0, global_batch=four_seq_batch)
    ctx_rank3 = BatchFormationContext(world_size=4, rank=3, global_batch=four_seq_batch)
    plan_rank0 = BatchFormation.plan_rank_assignment(ctx_rank0)
    plan_rank3 = BatchFormation.plan_rank_assignment(ctx_rank3)
    assert plan_rank0.assignments == plan_rank3.assignments
    assert plan_rank0.tiles_per_rank == plan_rank3.tiles_per_rank


def test_assignment_empty_batch():
    batch = SequenceBatch()
    ctx = BatchFormationContext(world_size=4, rank=0, global_batch=batch)
    plan = BatchFormation.plan_rank_assignment(ctx)
    assert plan.assignments == {}
    assert plan.tiles_per_rank == (0, 0, 0, 0)


def test_assignment_raises_on_none_batch():
    ctx = BatchFormationContext(world_size=2, rank=0, global_batch=None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Global batch not initialized"):
        BatchFormation.plan_rank_assignment(ctx)


def test_assignment_does_not_mutate_global_batch(four_seq_batch):
    """The handler is non-mutating; worker is the sole `assign_rank` caller."""
    pre_assigned = {seq.uuid: seq.assigned_rank for seq in four_seq_batch}
    ctx = BatchFormationContext(world_size=2, rank=0, global_batch=four_seq_batch)
    BatchFormation.plan_rank_assignment(ctx)
    post_assigned = {seq.uuid: seq.assigned_rank for seq in four_seq_batch}
    # No assignment was applied to global_batch — handler is pure.
    assert pre_assigned == post_assigned


def test_assignment_uses_ceil_div_for_tiles():
    """A 129-token sequence consumes 2 tiles, not 1."""
    batch = SequenceBatch()
    batch.add_sequence(_make_seq("just_over", 0, prompt=64, decode=65))  # 129 → ceil(129/128)=2
    ctx = BatchFormationContext(world_size=1, rank=0, global_batch=batch)
    plan = BatchFormation.plan_rank_assignment(ctx)
    assert plan.tiles_per_rank == (2,)


def test_tile_size_constant():
    """The bin-packing granularity is exposed for clarity."""
    assert TILE_SIZE == 128
