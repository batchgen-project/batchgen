"""Unit tests for batchgen.worker.boundary.synchronizer.BoundarySynchronizer."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry
from batchgen.worker.boundary.decisions import (
    BoundaryPlan,
    Evict,
    EvictReason,
    ReleasePages,
)
from batchgen.worker.boundary.synchronizer import BoundarySynchronizer
from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator
from tests.unit.worker.fakes import FakeCollectiveBackend


def _make_state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=rank,
        torch_device=torch.device("cpu"),
    )


def _add(state: WorkerState, uuid: str, *, assigned_rank: int | None = None) -> SequenceEntry:
    seq = SequenceEntry(uuid=uuid, global_idx=0, prompt_length=10, max_decode_length=100, text="")
    seq.original_prompt_length = 10
    seq.decoded_length = 5
    seq.current_context_length = 15
    seq.assigned_rank = assigned_rank if assigned_rank is not None else state.rank
    state.global_batch.add_sequence(seq)
    return seq


# ---------------------------------------------------------------------------
# sync_metadata_in
# ---------------------------------------------------------------------------


class TestSyncMetadataIn:
    def test_delegates_to_sync_coordinator(self) -> None:
        state = _make_state()
        _add(state, "u1")
        col = FakeCollectiveBackend(rank=0, world_size=1)
        sync = SyncCoordinator(state, col)
        bsync = BoundarySynchronizer(state, sync, col)

        bsync.sync_metadata_in(["u1"])

        assert col.call_names() == ["all_gather_object"]

    def test_ctx_drift_propagates(self) -> None:
        state = _make_state()
        seq = _add(state, "u1")
        seq.current_context_length = 99  # drifted
        col = FakeCollectiveBackend(rank=0, world_size=1)
        sync = SyncCoordinator(state, col)
        bsync = BoundarySynchronizer(state, sync, col)

        with pytest.raises(CtxInvariantViolation):
            bsync.sync_metadata_in(["u1"])


# ---------------------------------------------------------------------------
# broadcast_plan
# ---------------------------------------------------------------------------


class TestBroadcastPlan:
    def test_rank_0_broadcasts_local_plan(self) -> None:
        state = _make_state(rank=0, world_size=2)
        col = FakeCollectiveBackend(rank=0, world_size=2)
        sync = SyncCoordinator(state, col)
        bsync = BoundarySynchronizer(state, sync, col)

        plan = BoundaryPlan(
            decisions=(Evict(uuids=("u1",), reason=EvictReason.HOST_KV_WATERMARK),)
        )
        returned = bsync.broadcast_plan(plan)

        assert returned is plan
        assert col.call_names() == ["broadcast_object"]

    def test_non_rank_0_receives_via_injected_response(self) -> None:
        state = _make_state(rank=1, world_size=2)
        inbound = BoundaryPlan(decisions=(ReleasePages(uuids=("u1",)),))
        col = FakeCollectiveBackend(
            rank=1,
            world_size=2,
            broadcast_object_responses=[[inbound]],
        )
        sync = SyncCoordinator(state, col)
        bsync = BoundarySynchronizer(state, sync, col)

        returned = bsync.broadcast_plan(None)

        assert returned is inbound
        assert returned.decisions == (ReleasePages(uuids=("u1",)),)
        assert col.call_names() == ["broadcast_object"]

    def test_none_after_broadcast_raises(self) -> None:
        """A non-None plan must come back from the broadcast. None after
        the collective is a programming error — rank 0 passed None when
        it shouldn't have — and we fail loudly instead of running the
        executor with an empty BoundaryPlan."""
        state = _make_state(rank=1, world_size=2)
        col = FakeCollectiveBackend(
            rank=1,
            world_size=2,
            broadcast_object_responses=[[None]],
        )
        sync = SyncCoordinator(state, col)
        bsync = BoundarySynchronizer(state, sync, col)

        with pytest.raises(RuntimeError, match="must provide a non-None plan"):
            bsync.broadcast_plan(None)

    def test_empty_plan_roundtrips(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        sync = SyncCoordinator(state, col)
        bsync = BoundarySynchronizer(state, sync, col)

        plan = BoundaryPlan()  # no decisions
        assert bsync.broadcast_plan(plan) is plan
