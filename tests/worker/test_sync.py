"""Unit tests for `batchgen.worker.sync`.

Single-rank tests with a fake ``CollectiveBackend`` whose all_reduce ops
are no-ops (on a single rank, MAX/MIN of one tensor leaves it unchanged).
Real ``SequenceBatch`` / ``SequenceEntry`` fixtures — no mocks per the
Phase A §G no-hack rule.
"""

from __future__ import annotations

import logging

import pytest
import torch

from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker.sync import (
    CollectiveBackend,
    SyncContext,
    SyncCoordinator,
)


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class FakeCollectiveBackend:
    """Single-rank no-op backend that records call counts.

    On a single rank, ``all_reduce(MAX)`` / ``all_reduce(MIN)`` leave the
    tensor unchanged, so this fake is correct (not just a stub).
    """

    def __init__(self) -> None:
        self.max_calls = 0
        self.min_calls = 0

    def all_reduce_max(self, tensor: torch.Tensor) -> None:
        self.max_calls += 1

    def all_reduce_min(self, tensor: torch.Tensor) -> None:
        self.min_calls += 1


def _make_seq(
    uuid: str,
    global_idx: int,
    rank: int = 0,
    status: SequenceStatus = SequenceStatus.IN_DECODE,
    eos_reached: bool = False,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=8,
        max_decode_length=16,
    )
    seq.assigned_rank = rank
    # SequenceEntry starts in QUEUEING; force to the target status. Many tests
    # need IN_DECODE; we bypass the transition validator by setting directly,
    # which matches how _sync_decode_uuids_tensor reads seq state.
    seq.status = status
    seq.eos_reached = eos_reached
    return seq


@pytest.fixture
def empty_batch() -> SequenceBatch:
    return SequenceBatch()


@pytest.fixture
def batch_with_three_seqs() -> SequenceBatch:
    """3 sequences owned by rank 0, global_idx 0/1/2."""
    batch = SequenceBatch()
    for seq in [
        _make_seq("alpha", global_idx=0),
        _make_seq("bravo", global_idx=1),
        _make_seq("charlie", global_idx=2),
    ]:
        batch.add_sequence(seq)
        batch.assign_rank(seq.uuid, 0)
    return batch


@pytest.fixture
def ctx(batch_with_three_seqs) -> SyncContext:
    return SyncContext(
        rank=0,
        uuid_to_local={"alpha": 0, "bravo": 1, "charlie": 2},
        global_batch=batch_with_three_seqs,
        torch_device=torch.device("cpu"),
    )


@pytest.fixture
def coordinator() -> SyncCoordinator:
    return SyncCoordinator(backend=FakeCollectiveBackend())


# ---------------------------------------------------------------------------
# SyncContext dataclass behavior
# ---------------------------------------------------------------------------

def test_context_is_frozen(ctx):
    with pytest.raises((AttributeError, Exception)):
        ctx.rank = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# sync_completion_status_tensor
# ---------------------------------------------------------------------------

def test_completion_status_empty_input(coordinator, ctx):
    completed, active = coordinator.sync_completion_status_tensor(ctx, [])
    assert completed == set()
    assert active == []


def test_completion_status_no_completed(coordinator, ctx):
    """All 3 sequences IN_DECODE, none flagged as complete → all active."""
    completed, active = coordinator.sync_completion_status_tensor(
        ctx, ["alpha", "bravo", "charlie"]
    )
    assert completed == set()
    assert active == ["alpha", "bravo", "charlie"]
    # No status changes
    for uuid in ["alpha", "bravo", "charlie"]:
        assert ctx.global_batch.get_sequence(uuid).status == SequenceStatus.IN_DECODE


def test_completion_status_one_eos(coordinator, ctx):
    """`bravo` has eos_reached=True; should land in completed set."""
    ctx.global_batch.get_sequence("bravo").eos_reached = True
    completed, active = coordinator.sync_completion_status_tensor(
        ctx, ["alpha", "bravo", "charlie"]
    )
    assert completed == {"bravo"}
    assert active == ["alpha", "charlie"]
    # Status transition applied
    assert ctx.global_batch.get_sequence("bravo").status == SequenceStatus.COMPLETED


def test_completion_status_idempotent_mutation(coordinator, ctx):
    """Running twice in succession yields the same result; status guard prevents double-transition."""
    ctx.global_batch.get_sequence("bravo").eos_reached = True
    c1, _ = coordinator.sync_completion_status_tensor(ctx, ["alpha", "bravo", "charlie"])
    # On the second run, bravo is already COMPLETED — the guard `if seq.status != COMPLETED`
    # suppresses re-transition, and the method still returns bravo in the completed set
    # because the legacy check `(seq.status == SequenceStatus.COMPLETED or seq.eos_reached)`
    # remains true.
    c2, _ = coordinator.sync_completion_status_tensor(ctx, ["alpha", "bravo", "charlie"])
    assert c1 == c2 == {"bravo"}


def test_completion_status_records_backend_calls(coordinator, ctx):
    backend = coordinator._backend
    assert backend.max_calls == 0
    coordinator.sync_completion_status_tensor(ctx, ["alpha"])
    assert backend.max_calls == 1


def test_completion_status_unknown_uuid_skipped(coordinator, ctx):
    """A uuid not in global_batch is silently ignored."""
    completed, active = coordinator.sync_completion_status_tensor(
        ctx, ["alpha", "ghost", "bravo"]
    )
    # ghost is not in global_batch.get_sequence — dropped before tensor build
    assert "ghost" not in completed
    assert "ghost" not in active


# ---------------------------------------------------------------------------
# sync_decode_uuids_tensor
# ---------------------------------------------------------------------------

def test_decode_uuids_empty_input(coordinator, ctx):
    assert coordinator.sync_decode_uuids_tensor(ctx, []) == []


def test_decode_uuids_single_rank_passthrough(coordinator, ctx):
    """On a single rank, all_reduce(MIN) leaves the presence tensor unchanged
    → every input uuid is in the result."""
    result = coordinator.sync_decode_uuids_tensor(ctx, ["alpha", "bravo", "charlie"])
    # Order is by global_idx → alpha(0), bravo(1), charlie(2)
    assert result == ["alpha", "bravo", "charlie"]


def test_decode_uuids_subset(coordinator, ctx):
    """Partial input → only those uuids survive on this rank (single-rank MIN)."""
    result = coordinator.sync_decode_uuids_tensor(ctx, ["charlie", "alpha"])
    # Result sorted by global_idx → alpha(0), charlie(2)
    assert result == ["alpha", "charlie"]


def test_decode_uuids_records_backend_calls(coordinator, ctx):
    backend = coordinator._backend
    assert backend.min_calls == 0
    coordinator.sync_decode_uuids_tensor(ctx, ["alpha"])
    assert backend.min_calls == 1


def test_decode_uuids_unknown_uuid_ignored(coordinator, ctx):
    """A uuid not in global_batch never makes it into the presence tensor."""
    result = coordinator.sync_decode_uuids_tensor(ctx, ["alpha", "ghost"])
    assert "ghost" not in result
    assert "alpha" in result


# ---------------------------------------------------------------------------
# Backend injection
# ---------------------------------------------------------------------------

def test_can_swap_backend():
    class CountingBackend:
        def __init__(self) -> None:
            self.events = []

        def all_reduce_max(self, tensor):
            self.events.append(("max", tuple(tensor.shape)))

        def all_reduce_min(self, tensor):
            self.events.append(("min", tuple(tensor.shape)))

    backend = CountingBackend()
    c = SyncCoordinator(backend=backend)
    batch = SequenceBatch()
    seq = _make_seq("solo", global_idx=0)
    batch.add_sequence(seq)
    batch.assign_rank("solo", 0)
    ctx = SyncContext(
        rank=0,
        uuid_to_local={"solo": 0},
        global_batch=batch,
        torch_device=torch.device("cpu"),
    )

    c.sync_completion_status_tensor(ctx, ["solo"])
    c.sync_decode_uuids_tensor(ctx, ["solo"])

    assert [evt[0] for evt in backend.events] == ["max", "min"]
