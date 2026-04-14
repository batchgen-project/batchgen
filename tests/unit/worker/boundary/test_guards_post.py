"""Unit tests for Phase 2.8.1h guard additions.

Covers:
  * `check_post_page_table_order`: raises on mismatch, no-op on
    match, no-op when the manager is not live yet.
  * `check_pre` accepts the new decision types (HostGrow, HostEvict,
    NewLoadAsync) without raising 'unknown_decision_type'.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from batchgen.sequence import SequenceEntry
from batchgen.worker.boundary.decisions import (
    BoundaryPlan,
    HostEvict,
    HostGrow,
    NewLoadAsync,
)
from batchgen.worker.boundary.guards import BoundaryGuards, GuardViolation
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import FakeLegacyBackend


def _state_with(uuids: list[str]) -> WorkerState:
    state = WorkerState(
        rank=0, local_rank=0, world_size=1, device=0,
        torch_device=torch.device("cpu"),
    )
    for uuid in uuids:
        seq = SequenceEntry(uuid=uuid, global_idx=0, prompt_length=10, max_decode_length=100)
        seq.original_prompt_length = 10
        seq.decoded_length = 0
        seq.current_context_length = 10
        seq.assigned_rank = 0
        state.global_batch.add_sequence(seq)
    return state


def _fake_gpu(
    slot_to_seq_id: list[int] | None = None, is_initialized: bool = True
) -> Any:
    ptm = None
    if slot_to_seq_id is not None:
        ptm = types.SimpleNamespace(slot_to_seq_id=list(slot_to_seq_id))
    return types.SimpleNamespace(
        is_initialized=is_initialized, _gpu_page_table_manager=ptm,
    )


# ---------------------------------------------------------------------------
# check_post_page_table_order
# ---------------------------------------------------------------------------


class _AdapterWithMapping(FakeLegacyBackend):
    def __init__(self, mapping: dict[int, int]) -> None:
        super().__init__()
        self._mapping = mapping

    def local_indices_to_global_seq_ids(self, batch: list[int]) -> list[int]:
        self._record("local_indices_to_global_seq_ids", batch)
        return [self._mapping[i] for i in batch]


class TestPageTableOrderGuard:
    def test_matching_order_passes(self) -> None:
        guards = BoundaryGuards(_state_with(["u"]))
        gpu = _fake_gpu(slot_to_seq_id=[10, 20])
        legacy = _AdapterWithMapping({0: 10, 1: 20})
        guards.check_post_page_table_order(legacy, gpu, [0, 1])

    def test_mismatched_order_raises(self) -> None:
        guards = BoundaryGuards(_state_with(["u"]))
        gpu = _fake_gpu(slot_to_seq_id=[10, 99])
        legacy = _AdapterWithMapping({0: 10, 1: 20})
        with pytest.raises(GuardViolation, match="page_table_order"):
            guards.check_post_page_table_order(legacy, gpu, [0, 1])

    def test_empty_batch_is_noop(self) -> None:
        guards = BoundaryGuards(_state_with(["u"]))
        gpu = _fake_gpu(slot_to_seq_id=[99, 99])
        legacy = _AdapterWithMapping({})
        guards.check_post_page_table_order(legacy, gpu, [])

    def test_uninitialized_gpu_is_noop(self) -> None:
        guards = BoundaryGuards(_state_with(["u"]))
        gpu = _fake_gpu(slot_to_seq_id=[99], is_initialized=False)
        legacy = _AdapterWithMapping({0: 10})
        guards.check_post_page_table_order(legacy, gpu, [0])

    def test_no_page_table_manager_is_noop(self) -> None:
        guards = BoundaryGuards(_state_with(["u"]))
        gpu = _fake_gpu(slot_to_seq_id=None, is_initialized=True)
        legacy = _AdapterWithMapping({0: 10})
        guards.check_post_page_table_order(legacy, gpu, [0])


# ---------------------------------------------------------------------------
# check_pre recognises the new decision types
# ---------------------------------------------------------------------------


class TestCheckPreNewDecisionTypes:
    def test_host_grow_accepted(self) -> None:
        guards = BoundaryGuards(_state_with(["u1", "u2"]))
        plan = BoundaryPlan(
            decisions=(HostGrow(uuids=("u1", "u2"), pages=(3, 5), feasible=True),)
        )
        guards.check_pre(plan)  # no raise

    def test_host_evict_accepted(self) -> None:
        guards = BoundaryGuards(_state_with(["u1"]))
        plan = BoundaryPlan(decisions=(HostEvict(uuids=("u1",)),))
        guards.check_pre(plan)

    def test_new_load_async_accepted(self) -> None:
        guards = BoundaryGuards(_state_with(["u1"]))
        plan = BoundaryPlan(decisions=(NewLoadAsync(uuids=("u1",)),))
        guards.check_pre(plan)

    def test_missing_uuid_still_raises(self) -> None:
        """New types continue to fail the live-uuid check."""
        guards = BoundaryGuards(_state_with([]))
        plan = BoundaryPlan(decisions=(HostEvict(uuids=("ghost",)),))
        with pytest.raises(GuardViolation, match="plan_references_live_sequences"):
            guards.check_pre(plan)
