"""Unit tests for batchgen.worker.boundary.guards.BoundaryGuards."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry
from batchgen.worker.boundary.decisions import (
    AsyncLoadHostToGpu,
    BoundaryPlan,
    Evict,
    EvictReason,
    ExtendPages,
    OnHold,
    OnHoldReason,
    ReleasePages,
)
from batchgen.worker.boundary.guards import BoundaryGuards, GuardViolation
from batchgen.worker.state import WorkerState


def _make_state() -> WorkerState:
    return WorkerState(
        rank=0,
        local_rank=0,
        world_size=1,
        device=0,
        torch_device=torch.device("cpu"),
    )


def _add(state: WorkerState, uuid: str, *, decoded: int = 0, prompt: int = 10) -> SequenceEntry:
    seq = SequenceEntry(uuid=uuid, global_idx=0, prompt_length=prompt, max_decode_length=100, text="")
    seq.original_prompt_length = prompt
    seq.decoded_length = decoded
    seq.current_context_length = prompt + decoded
    state.global_batch.add_sequence(seq)
    return seq


# ---------------------------------------------------------------------------
# check_pre — plan references live sequences
# ---------------------------------------------------------------------------


class TestCheckPre:
    def test_empty_plan_passes(self) -> None:
        state = _make_state()
        BoundaryGuards(state).check_pre(BoundaryPlan())  # no raise

    def test_release_pages_live_uuids_passes(self) -> None:
        state = _make_state()
        _add(state, "u1")
        _add(state, "u2")
        plan = BoundaryPlan(decisions=(ReleasePages(uuids=("u1", "u2")),))
        BoundaryGuards(state).check_pre(plan)

    def test_release_pages_ghost_uuid_raises(self) -> None:
        state = _make_state()
        _add(state, "u1")
        plan = BoundaryPlan(decisions=(ReleasePages(uuids=("u1", "ghost")),))
        with pytest.raises(GuardViolation) as exc:
            BoundaryGuards(state).check_pre(plan)
        assert exc.value.check == "pre"
        assert exc.value.invariant == "plan_references_live_sequences"
        assert exc.value.detail["uuid"] == "ghost"
        assert exc.value.detail["decision"] == "ReleasePages"

    def test_evict_ghost_uuid_raises(self) -> None:
        state = _make_state()
        plan = BoundaryPlan(
            decisions=(Evict(uuids=("ghost",), reason=EvictReason.HOST_KV_WATERMARK),)
        )
        with pytest.raises(GuardViolation):
            BoundaryGuards(state).check_pre(plan)

    def test_on_hold_ghost_uuid_raises(self) -> None:
        state = _make_state()
        plan = BoundaryPlan(
            decisions=(OnHold(uuids=("ghost",), reason=OnHoldReason.WATERMARK_TRIGGER),)
        )
        with pytest.raises(GuardViolation):
            BoundaryGuards(state).check_pre(plan)

    def test_extend_pages_ghost_uuid_raises(self) -> None:
        state = _make_state()
        plan = BoundaryPlan(decisions=(ExtendPages(uuid="ghost", additional_pages=2),))
        with pytest.raises(GuardViolation):
            BoundaryGuards(state).check_pre(plan)

    def test_async_load_ghost_uuid_raises(self) -> None:
        state = _make_state()
        plan = BoundaryPlan(decisions=(AsyncLoadHostToGpu(uuid="ghost", host_pages=(1,)),))
        with pytest.raises(GuardViolation):
            BoundaryGuards(state).check_pre(plan)


# ---------------------------------------------------------------------------
# check_post — CTX invariant
# ---------------------------------------------------------------------------


class TestCheckPostCtxInvariant:
    def test_empty_batch_passes(self) -> None:
        state = _make_state()
        BoundaryGuards(state).check_post()

    def test_held_invariant_passes(self) -> None:
        state = _make_state()
        _add(state, "u1", decoded=5, prompt=10)
        _add(state, "u2", decoded=0, prompt=20)
        BoundaryGuards(state).check_post()

    def test_drift_raises_with_uuid_and_values(self) -> None:
        state = _make_state()
        seq = _add(state, "u1", decoded=5, prompt=10)
        seq.current_context_length = 99  # drifted
        with pytest.raises(GuardViolation) as exc:
            BoundaryGuards(state).check_post()
        assert exc.value.check == "post"
        assert exc.value.invariant == "ctx_invariant"
        assert exc.value.detail == {"uuid": "u1", "had": 99, "expected": 15}


# ---------------------------------------------------------------------------
# check_post — index map consistency
# ---------------------------------------------------------------------------


class TestCheckPostIndexMapConsistency:
    def test_consistent_maps_pass(self) -> None:
        state = _make_state()
        state.local_to_uuid_map = {0: "u1", 1: "u2"}
        state.uuid_to_local_map = {"u1": 0, "u2": 1}
        state.free_local_indices = set()
        BoundaryGuards(state).check_post()

    def test_reverse_map_mismatch_raises(self) -> None:
        state = _make_state()
        state.local_to_uuid_map = {0: "u1"}
        state.uuid_to_local_map = {"u1": 99}  # disagrees
        with pytest.raises(GuardViolation) as exc:
            BoundaryGuards(state).check_post()
        assert exc.value.invariant == "index_map_consistency"

    def test_size_mismatch_raises(self) -> None:
        state = _make_state()
        state.local_to_uuid_map = {0: "u1", 1: "u2"}
        state.uuid_to_local_map = {"u1": 0}  # u2 missing
        with pytest.raises(GuardViolation) as exc:
            BoundaryGuards(state).check_post()
        assert exc.value.invariant == "index_map_consistency"

    def test_free_slot_overlaps_live_raises(self) -> None:
        """A local_idx cannot be both live and free at the same time."""
        state = _make_state()
        state.local_to_uuid_map = {0: "u1"}
        state.uuid_to_local_map = {"u1": 0}
        state.free_local_indices = {0}  # also marked free → violation
        with pytest.raises(GuardViolation) as exc:
            BoundaryGuards(state).check_post()
        assert exc.value.invariant == "free_slot_exclusive"
        assert exc.value.detail["overlapping_local_indices"] == [0]

    def test_empty_maps_and_free_set_pass(self) -> None:
        state = _make_state()
        BoundaryGuards(state).check_post()
