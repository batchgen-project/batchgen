"""Unit tests for decode/handle_boundary.py (Phase 2.8.2f)."""

from __future__ import annotations

from typing import Any

import torch

from batchgen.worker.boundary import (
    BoundaryHandler,
    BoundaryPlan,
    BoundaryResult,
    OnHold,
    OnHoldReason,
)
from batchgen.worker.decode.handle_boundary import handle_boundary
from batchgen.worker.decode.state import DecodeState
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import FakeLegacyBackend


class _StubBoundaryHandler:
    """Minimal stand-in for :class:`BoundaryHandler` — the decode
    handle_boundary function doesn't touch any handler internals
    beyond ``run_full`` and ``_pending_load_uuids``."""

    def __init__(self, result: BoundaryResult) -> None:
        self._result = result
        self._pending_load_uuids: list[str] = []
        self.run_full_calls: list[dict[str, Any]] = []

    def run_full(self, **kwargs: Any) -> BoundaryResult:
        self.run_full_calls.append(kwargs)
        return self._result


def _state() -> WorkerState:
    return WorkerState(
        rank=0, local_rank=0, world_size=1, device=0,
        torch_device=torch.device("cpu"),
    )


def _decode_state(uuids: list[str], batch: list[int]) -> DecodeState:
    return DecodeState(decode_uuids=list(uuids), batch=list(batch))


class TestHappyPath:
    def test_rebuilds_new_tokens_and_marks_verified(self) -> None:
        boundary = _StubBoundaryHandler(
            BoundaryResult(
                plan=BoundaryPlan(),
                decode_uuids=("u",), batch=(0,),
                watermark_triggered=False,
            )
        )
        state = _state()
        legacy = FakeLegacyBackend()
        ds = _decode_state(["u"], [0])

        outcome = handle_boundary(state, legacy, boundary, ds, gpu_manager=None)

        assert outcome.should_break is False
        assert outcome.should_continue is False
        assert ds.decode_uuids == ["u"]
        assert ds.batch == [0]
        assert ds.cumulative_boundaries == 1
        assert ds.page_table_verified is True
        # rebuild_input_tokens was called
        assert any(c[0] == "rebuild_input_tokens" for c in legacy.calls)


class TestWatermarkBreak:
    def test_planner_watermark_sets_break(self) -> None:
        plan_with_bail = BoundaryPlan(
            decisions=(
                OnHold(uuids=("u",), reason=OnHoldReason.WATERMARK_TRIGGER),
            ),
            watermark_break=True,
        )
        boundary = _StubBoundaryHandler(
            BoundaryResult(plan=plan_with_bail)
        )
        state = _state()
        legacy = FakeLegacyBackend()
        ds = _decode_state(["u"], [0])

        outcome = handle_boundary(state, legacy, boundary, ds, gpu_manager=None)

        assert outcome.should_break is True
        assert outcome.should_continue is False
        # belt-and-braces: wait_pending_kv_append_tasks fires before exit
        assert any(
            c[0] == "wait_pending_kv_append_tasks" for c in legacy.calls
        )

    def test_finalize_watermark_also_triggers_break(self) -> None:
        """The finalize-level watermark bool also trips the break path
        so sequences flushed via finalize get their KV drained."""
        boundary = _StubBoundaryHandler(
            BoundaryResult(
                plan=BoundaryPlan(),
                decode_uuids=("u",), batch=(0,),
                watermark_triggered=True,
            )
        )
        state = _state()
        legacy = FakeLegacyBackend()
        ds = _decode_state(["u"], [0])

        outcome = handle_boundary(state, legacy, boundary, ds, gpu_manager=None)
        assert outcome.should_break is True


class TestEmptyCohort:
    def test_empty_with_no_pending_breaks(self) -> None:
        boundary = _StubBoundaryHandler(
            BoundaryResult(plan=BoundaryPlan(), decode_uuids=(), batch=())
        )
        state = _state()
        legacy = FakeLegacyBackend()
        ds = _decode_state(["u"], [0])

        outcome = handle_boundary(state, legacy, boundary, ds, gpu_manager=None)

        assert outcome.should_break is True
        assert outcome.should_continue is False
        assert ds.decode_uuids == []

    def test_empty_with_pending_loads_continues(self) -> None:
        boundary = _StubBoundaryHandler(
            BoundaryResult(plan=BoundaryPlan(), decode_uuids=(), batch=())
        )
        boundary._pending_load_uuids = ["load_queued"]
        state = _state()
        legacy = FakeLegacyBackend()
        ds = _decode_state(["u"], [0])

        outcome = handle_boundary(state, legacy, boundary, ds, gpu_manager=None)

        assert outcome.should_break is False
        assert outcome.should_continue is True


class TestNoAdmissionPolling:
    def test_adapter_poll_admission_queue_never_called(self) -> None:
        """The L4-fix invariant: decode loop must NOT poll the
        admission queue. AdmissionCoordinator owns that path."""
        boundary = _StubBoundaryHandler(
            BoundaryResult(
                plan=BoundaryPlan(),
                decode_uuids=("u",), batch=(0,),
            )
        )
        state = _state()
        legacy = FakeLegacyBackend()
        ds = _decode_state(["u"], [0])

        handle_boundary(state, legacy, boundary, ds, gpu_manager=None)

        assert not any(
            c[0] == "poll_admission_queue_nowait" for c in legacy.calls
        )
