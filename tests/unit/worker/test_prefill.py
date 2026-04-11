"""Unit tests for batchgen.worker.prefill.PrefillScheduler."""

from __future__ import annotations

import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.prefill import PrefillScheduler
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import (
    FakeGpuKvBackend,
    FakeHostKvBackend,
    FakeModelExecutor,
)


PAGE = SequenceEntry.PAGE_SIZE  # 64


def _make_state() -> WorkerState:
    return WorkerState(
        rank=0,
        local_rank=0,
        world_size=1,
        device=0,
        torch_device=torch.device("cpu"),
    )


def _make_scheduler(
    state: WorkerState,
    *,
    host_free: int = 1000,
    model: FakeModelExecutor | None = None,
) -> tuple[PrefillScheduler, FakeModelExecutor]:
    gpu = FakeGpuKvBackend()
    host = FakeHostKvBackend(free_pages=host_free)
    kv = KVCacheManager(
        state,
        gpu,
        host,
        initial_gpu_page_buffer=8,
        extension_gpu_page_buffer=4,
        host_kv_total_pages=10000,
        prefill_watermark_pct=70,
    )
    me = model or FakeModelExecutor(prefill_output="PREFILL_OUT")
    return PrefillScheduler(state, kv, me), me


def _add(
    state: WorkerState,
    uuid: str,
    *,
    status: SequenceStatus,
    prompt_length: int = 64,
    max_decode_length: int = 64,
    decoded_length: int = 0,
    global_idx: int = 0,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=max_decode_length,
        text="",
    )
    seq.decoded_length = decoded_length
    seq.current_context_length = prompt_length + decoded_length
    seq.original_prompt_length = prompt_length
    state.global_batch.add_sequence(seq)
    # Walk the state machine to the requested status
    if status != SequenceStatus.QUEUEING:
        transitions = {
            SequenceStatus.IN_PREFILL: [SequenceStatus.IN_PREFILL],
            SequenceStatus.PREFILLED: [SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED],
            SequenceStatus.IN_DECODE: [
                SequenceStatus.IN_PREFILL,
                SequenceStatus.PREFILLED,
                SequenceStatus.IN_DECODE,
            ],
            SequenceStatus.EVICTED: [
                SequenceStatus.IN_PREFILL,
                SequenceStatus.PREFILLED,
                SequenceStatus.IN_DECODE,
                SequenceStatus.EVICTED,
            ],
        }
        for s in transitions[status]:
            state.global_batch.update_status(uuid, s)
    return seq


# ---------------------------------------------------------------------------
# prepare_batch
# ---------------------------------------------------------------------------


class TestPrepareBatchEmpty:
    def test_empty_global_batch_returns_empty(self) -> None:
        state = _make_state()
        sch, _ = _make_scheduler(state)
        assert sch.prepare_batch() == []

    def test_only_other_statuses_returns_empty(self) -> None:
        """QUEUEING and EVICTED are the only candidate statuses; others
        must not appear in the prefill batch."""
        state = _make_state()
        _add(state, "u1", status=SequenceStatus.PREFILLED, global_idx=0)
        _add(state, "u2", status=SequenceStatus.IN_DECODE, global_idx=1)
        sch, _ = _make_scheduler(state)
        assert sch.prepare_batch() == []


class TestPrepareBatchQueueing:
    def test_single_queueing_sequence_selected(self) -> None:
        state = _make_state()
        _add(state, "u1", status=SequenceStatus.QUEUEING)
        sch, _ = _make_scheduler(state, host_free=1000)
        assert sch.prepare_batch() == ["u1"]

    def test_queueing_sorted_by_global_idx(self) -> None:
        state = _make_state()
        _add(state, "uC", status=SequenceStatus.QUEUEING, global_idx=2)
        _add(state, "uA", status=SequenceStatus.QUEUEING, global_idx=0)
        _add(state, "uB", status=SequenceStatus.QUEUEING, global_idx=1)
        sch, _ = _make_scheduler(state, host_free=1000)
        assert sch.prepare_batch() == ["uA", "uB", "uC"]


class TestPrepareBatchEvictedPriority:
    def test_evicted_comes_before_queueing(self) -> None:
        state = _make_state()
        _add(state, "q1", status=SequenceStatus.QUEUEING, global_idx=0)
        _add(
            state,
            "e1",
            status=SequenceStatus.EVICTED,
            decoded_length=10,
            global_idx=1,
        )
        sch, _ = _make_scheduler(state, host_free=1000)
        assert sch.prepare_batch() == ["e1", "q1"]

    def test_evicted_sorted_by_decoded_length_descending(self) -> None:
        """Preserve the progress of longer-running sequences — the one
        that has decoded the most is prefilled first."""
        state = _make_state()
        _add(state, "e_short", status=SequenceStatus.EVICTED, decoded_length=5, global_idx=0)
        _add(state, "e_long", status=SequenceStatus.EVICTED, decoded_length=50, global_idx=1)
        _add(state, "e_mid", status=SequenceStatus.EVICTED, decoded_length=20, global_idx=2)
        sch, _ = _make_scheduler(state, host_free=1000)
        assert sch.prepare_batch() == ["e_long", "e_mid", "e_short"]

    def test_evicted_ties_broken_by_uuid_for_determinism(self) -> None:
        state = _make_state()
        _add(state, "eB", status=SequenceStatus.EVICTED, decoded_length=10, global_idx=0)
        _add(state, "eA", status=SequenceStatus.EVICTED, decoded_length=10, global_idx=1)
        _add(state, "eC", status=SequenceStatus.EVICTED, decoded_length=10, global_idx=2)
        sch, _ = _make_scheduler(state, host_free=1000)
        assert sch.prepare_batch() == ["eA", "eB", "eC"]


class TestPrepareBatchCapacity:
    def test_cutoff_when_next_candidate_exceeds_budget(self) -> None:
        """Each sequence needs ceil((prompt + max_decode) / PAGE_SIZE) pages.
        With prompt=PAGE, max_decode=PAGE, that's 2 pages per sequence.
        host_free=5 allows 2 sequences (4 pages) and rejects the third."""
        state = _make_state()
        for i in range(5):
            _add(
                state,
                f"u{i}",
                status=SequenceStatus.QUEUEING,
                prompt_length=PAGE,
                max_decode_length=PAGE,
                global_idx=i,
            )
        sch, _ = _make_scheduler(state, host_free=5)  # 2 seqs max
        selected = sch.prepare_batch()
        assert selected == ["u0", "u1"]

    def test_zero_host_free_returns_empty(self) -> None:
        state = _make_state()
        _add(state, "u1", status=SequenceStatus.QUEUEING)
        sch, _ = _make_scheduler(state, host_free=0)
        assert sch.prepare_batch() == []

    def test_partial_selection_stops_at_exact_boundary(self) -> None:
        """3 sequences at 2 pages each, host_free=4 exactly → first 2 fit."""
        state = _make_state()
        for i in range(3):
            _add(
                state,
                f"u{i}",
                status=SequenceStatus.QUEUEING,
                prompt_length=PAGE,
                max_decode_length=PAGE,
                global_idx=i,
            )
        sch, _ = _make_scheduler(state, host_free=4)
        assert sch.prepare_batch() == ["u0", "u1"]

    def test_capacity_applies_to_evicted_and_queueing_combined(self) -> None:
        """One evicted (priority) + one queueing; budget for only one."""
        state = _make_state()
        _add(
            state,
            "e1",
            status=SequenceStatus.EVICTED,
            decoded_length=100,
            prompt_length=PAGE,
            max_decode_length=PAGE,
            global_idx=0,
        )
        _add(
            state,
            "q1",
            status=SequenceStatus.QUEUEING,
            prompt_length=PAGE,
            max_decode_length=PAGE,
            global_idx=1,
        )
        sch, _ = _make_scheduler(state, host_free=2)  # exactly 1 seq
        assert sch.prepare_batch() == ["e1"]


class TestPrepareBatchPageMath:
    def test_small_seq_rounds_up_to_one_page(self) -> None:
        state = _make_state()
        _add(
            state,
            "u1",
            status=SequenceStatus.QUEUEING,
            prompt_length=1,
            max_decode_length=1,
        )
        sch, _ = _make_scheduler(state, host_free=1)
        assert sch.prepare_batch() == ["u1"]

    def test_exact_page_multiple_no_rounding_waste(self) -> None:
        """prompt + max_decode == 2 * PAGE exactly → 2 pages, no waste."""
        state = _make_state()
        _add(
            state,
            "u1",
            status=SequenceStatus.QUEUEING,
            prompt_length=PAGE,
            max_decode_length=PAGE,
        )
        sch, _ = _make_scheduler(state, host_free=2)
        assert sch.prepare_batch() == ["u1"]

    def test_just_over_page_boundary_allocates_extra(self) -> None:
        """prompt + max_decode == 2*PAGE + 1 → 3 pages."""
        state = _make_state()
        _add(
            state,
            "u1",
            status=SequenceStatus.QUEUEING,
            prompt_length=PAGE + 1,
            max_decode_length=PAGE,
        )
        sch, _ = _make_scheduler(state, host_free=2)
        assert sch.prepare_batch() == []  # needs 3, only 2 available
        sch2, _ = _make_scheduler(state, host_free=3)
        assert sch2.prepare_batch() == ["u1"]


# ---------------------------------------------------------------------------
# config_for_batch / run / run_prepacked
# ---------------------------------------------------------------------------


class TestConfigForBatch:
    def test_records_uuid_list(self) -> None:
        state = _make_state()
        sch, _ = _make_scheduler(state)
        sch.config_for_batch(["a", "b", "c"])
        assert sch.last_configured == ["a", "b", "c"]

    def test_each_call_replaces_previous(self) -> None:
        state = _make_state()
        sch, _ = _make_scheduler(state)
        sch.config_for_batch(["a"])
        sch.config_for_batch(["x", "y"])
        assert sch.last_configured == ["x", "y"]


class TestRun:
    def test_run_delegates_to_model_executor(self) -> None:
        state = _make_state()
        me = FakeModelExecutor(prefill_output="STANDARD")
        sch, _ = _make_scheduler(state, model=me)
        out = sch.run(["u1", "u2"])
        assert out == "STANDARD"
        assert me.prefill_batches == [{"uuids": ["u1", "u2"], "prepacked": False}]

    def test_run_prepacked_flags_prepacked_true(self) -> None:
        state = _make_state()
        me = FakeModelExecutor(prefill_output="PACKED")
        sch, _ = _make_scheduler(state, model=me)
        out = sch.run_prepacked(["u1"])
        assert out == "PACKED"
        assert me.prefill_batches == [{"uuids": ["u1"], "prepacked": True}]

    def test_run_does_not_touch_decode_path(self) -> None:
        state = _make_state()
        me = FakeModelExecutor()
        sch, _ = _make_scheduler(state, model=me)
        sch.run(["u1"])
        assert me.decode_batches == []
