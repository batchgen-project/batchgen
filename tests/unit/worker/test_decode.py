"""Unit tests for batchgen.worker.decode.DecodeScheduler."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary import (
    BoundaryExecutor,
    BoundaryGuards,
    BoundaryHandler,
    BoundaryPlanner,
    BoundarySynchronizer,
    PlannerConfig,
)
from batchgen.worker.decode import DecodeScheduler, DecodeStepResult
from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator
from tests.unit.worker.fakes import (
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
    FakeModelExecutor,
)


PAGE = SequenceEntry.PAGE_SIZE  # 64
IN_DECODE_PATH = [
    SequenceStatus.IN_PREFILL,
    SequenceStatus.PREFILLED,
    SequenceStatus.IN_DECODE,
]


def _make_state() -> WorkerState:
    return WorkerState(
        rank=0,
        local_rank=0,
        world_size=1,
        device=0,
        torch_device=torch.device("cpu"),
    )


def _add(
    state: WorkerState,
    uuid: str,
    *,
    status_path: list[SequenceStatus] | None = None,
    prompt_length: int = 10,
    decoded_length: int = 0,
    current_context_length: int | None = None,
    gpu_pages_allocated: int = 32,
    global_idx: int = 0,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=10000,
        text="",
    )
    seq.decoded_length = decoded_length
    seq.current_context_length = (
        current_context_length
        if current_context_length is not None
        else prompt_length + decoded_length
    )
    seq.original_prompt_length = prompt_length
    seq.assigned_rank = state.rank
    seq.gpu_pages_allocated = gpu_pages_allocated
    state.global_batch.add_sequence(seq)
    if status_path:
        for s in status_path:
            state.global_batch.update_status(uuid, s)
    return seq


def _make_decode_scheduler(
    state: WorkerState,
    *,
    gpu_free: int = 1024,
    host_free: int = 1024,
    model: FakeModelExecutor | None = None,
    decision_frequency_pages: int = 2,
    initial_gpu_page_buffer: int = 32,
) -> tuple[
    DecodeScheduler,
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
    KVCacheManager,
    FakeModelExecutor,
    BoundaryHandler,
]:
    gpu = FakeGpuKvBackend(free_pages=gpu_free)
    host = FakeHostKvBackend(free_pages=host_free)
    col = FakeCollectiveBackend(rank=state.rank, world_size=state.world_size)
    kv = KVCacheManager(
        state,
        gpu,
        host,
        initial_gpu_page_buffer=initial_gpu_page_buffer,
        extension_gpu_page_buffer=4,
        host_kv_total_pages=10000,
        prefill_watermark_pct=70,
    )
    sync = SyncCoordinator(state, col)
    rebalancer = HostKVRebalancer(state, kv, sync)
    synchronizer = BoundarySynchronizer(state, sync, col)
    planner = BoundaryPlanner(
        PlannerConfig(
            prefill_watermark_pct=70,
            decision_frequency_pages=decision_frequency_pages,
            extension_gpu_page_buffer=4,
            host_total_pages=10000,
        )
    )
    executor = BoundaryExecutor(state, kv, rebalancer)
    guards = BoundaryGuards(state)
    boundary = BoundaryHandler(state, synchronizer, planner, executor, guards, kv)
    me = model or FakeModelExecutor(decode_output="DECODE_OUT")
    scheduler = DecodeScheduler(
        state,
        kv,
        me,
        boundary,
        decision_frequency_pages=decision_frequency_pages,
        initial_gpu_page_buffer=initial_gpu_page_buffer,
    )
    return scheduler, col, gpu, host, kv, me, boundary


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_decision_frequency_zero_raises(self) -> None:
        state = _make_state()
        _s, _c, _g, _h, kv, me, boundary = _make_decode_scheduler(state)
        with pytest.raises(ValueError, match="decision_frequency_pages"):
            DecodeScheduler(
                state, kv, me, boundary,
                decision_frequency_pages=0,
                initial_gpu_page_buffer=32,
            )

    def test_initial_buffer_zero_raises(self) -> None:
        state = _make_state()
        _s, _c, _g, _h, kv, me, boundary = _make_decode_scheduler(state)
        with pytest.raises(ValueError, match="initial_gpu_page_buffer"):
            DecodeScheduler(
                state, kv, me, boundary,
                decision_frequency_pages=2,
                initial_gpu_page_buffer=0,
            )


# ---------------------------------------------------------------------------
# prepare_batch
# ---------------------------------------------------------------------------


class TestPrepareBatch:
    def test_empty_state_returns_empty(self) -> None:
        state = _make_state()
        sch, *_ = _make_decode_scheduler(state)
        assert sch.prepare_batch() == []

    def test_only_prefilled_and_on_hold_are_candidates(self) -> None:
        state = _make_state()
        _add(state, "q1", global_idx=0)  # QUEUEING
        _add(state, "p1", status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED], global_idx=1)
        _add(state, "d1", status_path=IN_DECODE_PATH, global_idx=2)
        _add(state, "h1", status_path=IN_DECODE_PATH + [SequenceStatus.ON_HOLD], global_idx=3)
        sch, *_ = _make_decode_scheduler(state)
        assert sch.prepare_batch() == ["p1", "h1"]

    def test_sorted_by_global_idx_then_uuid(self) -> None:
        state = _make_state()
        _add(state, "uC", status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED], global_idx=2)
        _add(state, "uA", status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED], global_idx=0)
        _add(state, "uB", status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED], global_idx=1)
        sch, *_ = _make_decode_scheduler(state)
        assert sch.prepare_batch() == ["uA", "uB", "uC"]

    def test_ties_broken_by_uuid(self) -> None:
        state = _make_state()
        _add(state, "uB", status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED], global_idx=0)
        _add(state, "uA", status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED], global_idx=0)
        sch, *_ = _make_decode_scheduler(state)
        assert sch.prepare_batch() == ["uA", "uB"]


# ---------------------------------------------------------------------------
# try_load_new
# ---------------------------------------------------------------------------


class TestTryLoadNew:
    def test_loads_on_hold_to_in_decode(self) -> None:
        state = _make_state()
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH + [SequenceStatus.ON_HOLD],
        )
        sch, _col, gpu, _host, _kv, _me, _b = _make_decode_scheduler(state)

        loaded = sch.try_load_new(["u1"])

        assert loaded == ["u1"]
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.IN_DECODE  # type: ignore[union-attr]
        assert gpu.allocated_pages("u1")  # backend has allocation

    def test_stops_when_gpu_full(self) -> None:
        state = _make_state()
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH + [SequenceStatus.ON_HOLD],
            global_idx=0,
        )
        _add(
            state,
            "u2",
            status_path=IN_DECODE_PATH + [SequenceStatus.ON_HOLD],
            global_idx=1,
        )
        # initial_gpu_page_buffer=32, gpu_free=40 → 1 reload fits, the 2nd
        # needs 32 but only 8 remain → stop before u2.
        sch, _c, _g, _h, _kv, _me, _b = _make_decode_scheduler(state, gpu_free=40)

        loaded = sch.try_load_new(["u1", "u2"])

        assert loaded == ["u1"]
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.IN_DECODE  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("u2").status == SequenceStatus.ON_HOLD  # type: ignore[union-attr]

    def test_non_on_hold_uuids_skipped(self) -> None:
        state = _make_state()
        _add(state, "q1")  # QUEUEING
        _add(state, "d1", status_path=IN_DECODE_PATH, global_idx=1)
        sch, *_ = _make_decode_scheduler(state)

        loaded = sch.try_load_new(["q1", "d1"])

        assert loaded == []
        assert state.global_batch.get_sequence("q1").status == SequenceStatus.QUEUEING  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("d1").status == SequenceStatus.IN_DECODE  # type: ignore[union-attr]

    def test_missing_uuid_skipped(self) -> None:
        state = _make_state()
        sch, *_ = _make_decode_scheduler(state)
        assert sch.try_load_new(["ghost"]) == []


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_lazy_sets_flag_on_first_call(self) -> None:
        state = _make_state()
        sch, *_ = _make_decode_scheduler(state)
        assert sch.model_loaded is False
        sch.load_model()
        assert sch.model_loaded is True

    def test_idempotent(self) -> None:
        state = _make_state()
        sch, *_ = _make_decode_scheduler(state)
        sch.load_model()
        sch.load_model()
        sch.load_model()
        assert sch.model_loaded is True


# ---------------------------------------------------------------------------
# config_for_batch
# ---------------------------------------------------------------------------


class TestConfigForBatch:
    def test_records_uuid_list_when_invariant_holds(self) -> None:
        state = _make_state()
        _add(state, "u1", status_path=IN_DECODE_PATH, decoded_length=5)
        sch, *_ = _make_decode_scheduler(state)
        sch.config_for_batch(["u1"])
        assert sch.last_configured == ["u1"]

    def test_ctx_drift_raises_with_sender_side(self) -> None:
        state = _make_state()
        seq = _add(state, "u1", status_path=IN_DECODE_PATH, decoded_length=5)
        seq.current_context_length = 999
        sch, *_ = _make_decode_scheduler(state)
        with pytest.raises(CtxInvariantViolation) as exc:
            sch.config_for_batch(["u1"])
        assert exc.value.uuid == "u1"
        assert exc.value.side == "sender"
        assert exc.value.had == 999
        assert exc.value.expected == 15

    def test_missing_uuid_is_skipped(self) -> None:
        state = _make_state()
        sch, *_ = _make_decode_scheduler(state)
        sch.config_for_batch(["ghost"])  # must not raise


# ---------------------------------------------------------------------------
# run_continuous
# ---------------------------------------------------------------------------


class TestRunContinuous:
    def test_runs_exactly_decision_frequency_iterations(self) -> None:
        state = _make_state()
        _add(state, "u1", status_path=IN_DECODE_PATH, decoded_length=0)
        sch, _col, _gpu, _host, _kv, me, _b = _make_decode_scheduler(
            state, decision_frequency_pages=2
        )

        result = sch.run_continuous(["u1"])

        # 2 pages * 64 tokens/page = 128 forward_decode calls
        assert result.tokens_produced == 128
        assert len(me.decode_batches) == 128

    def test_state_updates_preserve_ctx_invariant(self) -> None:
        state = _make_state()
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH,
            prompt_length=10,
            decoded_length=0,
            gpu_pages_allocated=64,  # ample headroom
        )
        sch, *_ = _make_decode_scheduler(state, decision_frequency_pages=2)

        sch.run_continuous(["u1"])

        seq = state.global_batch.get_sequence("u1")
        assert seq is not None
        assert seq.decoded_length == 128
        # CTX invariant holds after the interval
        assert seq.current_context_length == seq.original_prompt_length + seq.decoded_length

    def test_interval_ends_with_boundary_run(self) -> None:
        """The final step of run_continuous is always BoundaryHandler.run,
        so the returned DecodeStepResult carries a plan. Even a no-op
        plan means the handler executed its sync_metadata + broadcast_plan
        collectives."""
        state = _make_state()
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH,
            gpu_pages_allocated=64,
        )
        sch, col, _gpu, _host, _kv, _me, _boundary = _make_decode_scheduler(
            state, decision_frequency_pages=2
        )

        result = sch.run_continuous(["u1"])

        # Boundary handler always issues sync_metadata + broadcast_plan
        assert "all_gather_object" in col.call_names()
        assert "broadcast_object" in col.call_names()
        # The plan was returned
        assert isinstance(result, DecodeStepResult)

    def test_tokens_produced_matches_decision_frequency_times_page(self) -> None:
        """With decision_frequency_pages=1, produce exactly PAGE_SIZE tokens."""
        state = _make_state()
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH,
            gpu_pages_allocated=64,
        )
        sch, *_ = _make_decode_scheduler(state, decision_frequency_pages=1)
        result = sch.run_continuous(["u1"])
        assert result.tokens_produced == PAGE

    def test_sequence_dropping_out_of_in_decode_mid_interval_is_not_ticked(
        self,
    ) -> None:
        """If a seq transitions out of IN_DECODE mid-interval (e.g. by
        another handler), subsequent iterations must NOT tick its
        decoded_length."""
        state = _make_state()
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH,
            decoded_length=0,
            gpu_pages_allocated=64,
        )
        _add(
            state,
            "u2",
            status_path=IN_DECODE_PATH,
            decoded_length=0,
            gpu_pages_allocated=64,
            global_idx=1,
        )
        sch, *_ = _make_decode_scheduler(state, decision_frequency_pages=1)

        # Simulate u2 dropping out mid-interval by transitioning it BEFORE
        # run_continuous. It should never be ticked.
        state.global_batch.update_status("u2", SequenceStatus.ON_HOLD)

        sch.run_continuous(["u1", "u2"])

        assert state.global_batch.get_sequence("u1").decoded_length == PAGE  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("u2").decoded_length == 0  # type: ignore[union-attr]

    def test_multi_uuid_all_ticked_per_iteration(self) -> None:
        state = _make_state()
        _add(state, "u1", status_path=IN_DECODE_PATH, gpu_pages_allocated=64, global_idx=0)
        _add(state, "u2", status_path=IN_DECODE_PATH, gpu_pages_allocated=64, global_idx=1)
        _add(state, "u3", status_path=IN_DECODE_PATH, gpu_pages_allocated=64, global_idx=2)
        sch, *_ = _make_decode_scheduler(state, decision_frequency_pages=1)

        sch.run_continuous(["u1", "u2", "u3"])

        for uuid in ["u1", "u2", "u3"]:
            assert state.global_batch.get_sequence(uuid).decoded_length == PAGE  # type: ignore[union-attr]
