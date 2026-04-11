"""Unit tests for batchgen.worker.orchestrator.WorkerOrchestrator."""

from __future__ import annotations

from queue import Queue

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.config import WorkerConfig
from batchgen.worker.orchestrator import BatchStats, WorkerOrchestrator
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import (
    FakeClock,
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
    FakeModelExecutor,
    FakeResponseSink,
    FakeTokenizer,
    RecordingLifespanLogger,
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


def _make_config(
    *,
    max_decode_length_pages: int = 1,
    model_context_length: int = 4096,
    max_pool_size: int = 0,
) -> WorkerConfig:
    return WorkerConfig(
        decision_frequency_pages=max_decode_length_pages,
        initial_gpu_page_buffer=32,
        extension_gpu_page_buffer=4,
        prefill_watermark_pct=70,
        eviction_watermark_pct=10,
        host_kv_total_pages=10000,
        rep_detection_enabled=False,  # keep deterministic in tests
        preemption_enabled=True,
        ignore_eos=False,
        model_context_length=model_context_length,
        max_pool_size=max_pool_size,
    )


def _build(
    *,
    gpu_free: int = 2000,
    host_free: int = 5000,
    admission_queue: Queue | None = None,
    model: FakeModelExecutor | None = None,
    config: WorkerConfig | None = None,
) -> tuple[WorkerOrchestrator, WorkerState, FakeCollectiveBackend, FakeResponseSink]:
    state = _make_state()
    col = FakeCollectiveBackend(rank=0, world_size=1)
    gpu = FakeGpuKvBackend(free_pages=gpu_free)
    host = FakeHostKvBackend(free_pages=host_free)
    tokenizer = FakeTokenizer(eos_token_ids={99})
    executor = model or FakeModelExecutor(
        prefill_output="P", decode_output="D"
    )
    sink = FakeResponseSink()
    orchestrator = WorkerOrchestrator(
        state,
        config or _make_config(),
        collectives=col,
        gpu_kv=gpu,
        host_kv=host,
        tokenizer=tokenizer,
        model=executor,
        lifespan=RecordingLifespanLogger(),
        sink=sink,
        clock=FakeClock(),
        admission_queue=admission_queue,
    )
    return orchestrator, state, col, sink


def _add_queueing(
    state: WorkerState,
    uuid: str,
    *,
    global_idx: int,
    max_decode_length: int,
    prompt_length: int = 10,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=max_decode_length,
        text="hello world",
    )
    seq.original_prompt_length = prompt_length
    seq.assigned_rank = 0
    # Give ample GPU allocation so the boundary handler never holds or
    # extends during the interval. The test exercises the orchestrator
    # loop, not the per-seq extension math.
    seq.gpu_pages_allocated = 64
    state.global_batch.add_sequence(seq)
    return seq


# ---------------------------------------------------------------------------
# Construction + introspection
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_all_handlers_composed(self) -> None:
        orch, _state, _col, _sink = _build()
        assert orch.index is not None
        assert orch.sync is not None
        assert orch.kv is not None
        assert orch.batch_formation is not None
        assert orch.admission is not None
        assert orch.completion is not None
        assert orch.rebalancer is not None
        assert orch.prefill is not None
        assert orch.boundary is not None
        assert orch.decode is not None

    def test_config_and_state_exposed(self) -> None:
        orch, state, _col, _sink = _build()
        assert orch.state is state
        assert isinstance(orch.config, WorkerConfig)

    def test_init_is_idempotent(self) -> None:
        orch, *_ = _build()
        assert orch.initialized is False
        orch.init()
        assert orch.initialized is True
        orch.init()
        orch.init()
        assert orch.initialized is True
        assert orch.decode.model_loaded is True


# ---------------------------------------------------------------------------
# Single-sequence end-to-end
# ---------------------------------------------------------------------------


class TestRunBatchSingleSequence:
    def test_queueing_sequence_completes_in_one_interval(self) -> None:
        orch, state, _col, sink = _build(
            config=_make_config(max_decode_length_pages=1)
        )
        _add_queueing(state, "u1", global_idx=0, max_decode_length=PAGE)

        stats = orch.run_batch()

        # Completed via length after exactly one decision interval (PAGE iterations).
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.COMPLETED  # type: ignore[union-attr]
        assert "u1" in stats.completed_uuids
        assert "u1" in sink.reported
        assert sink.reported["u1"]["finish_reason"] == "length"
        assert stats.prefill_rounds >= 1
        assert stats.decode_intervals >= 1

    def test_multiple_sequences_all_complete(self) -> None:
        orch, state, _col, sink = _build(
            config=_make_config(max_decode_length_pages=1)
        )
        for i in range(3):
            _add_queueing(state, f"u{i}", global_idx=i, max_decode_length=PAGE)

        orch.run_batch()

        for uuid in ["u0", "u1", "u2"]:
            seq = state.global_batch.get_sequence(uuid)
            assert seq is not None and seq.status == SequenceStatus.COMPLETED
            assert uuid in sink.reported


class TestRunBatchStatusMachine:
    def test_sequence_traverses_queueing_to_completed(self) -> None:
        """Snapshot status at the end — orchestrator drives
        QUEUEING → IN_PREFILL → PREFILLED → IN_DECODE → COMPLETED."""
        orch, state, _col, _sink = _build(
            config=_make_config(max_decode_length_pages=1)
        )
        _add_queueing(state, "u1", global_idx=0, max_decode_length=PAGE)

        assert state.global_batch.get_sequence("u1").status == SequenceStatus.QUEUEING  # type: ignore[union-attr]

        orch.run_batch()

        assert state.global_batch.get_sequence("u1").status == SequenceStatus.COMPLETED  # type: ignore[union-attr]


class TestRunBatchForwardPassDelegation:
    def test_model_executor_sees_prefill_and_decode_calls(self) -> None:
        me = FakeModelExecutor(prefill_output="P", decode_output="D")
        orch, state, _col, _sink = _build(
            model=me,
            config=_make_config(max_decode_length_pages=1),
        )
        _add_queueing(state, "u1", global_idx=0, max_decode_length=PAGE)

        orch.run_batch()

        # Exactly one prefill round
        assert len(me.prefill_batches) == 1
        assert me.prefill_batches[0]["uuids"] == ["u1"]
        # decision_frequency_pages * PAGE decode iterations
        assert len(me.decode_batches) == PAGE


class TestRunBatchEmptyState:
    def test_empty_state_returns_zero_stats(self) -> None:
        orch, _state, _col, _sink = _build()
        stats = orch.run_batch()
        assert stats == BatchStats()
        assert stats.completed_uuids == []


# ---------------------------------------------------------------------------
# generate_persistent (pool mode)
# ---------------------------------------------------------------------------


class TestGeneratePersistent:
    def test_empty_queue_returns_after_one_empty_poll(self) -> None:
        q: Queue = Queue()
        orch, _state, _col, _sink = _build(admission_queue=q)
        iters = orch.generate_persistent(max_iterations=5)
        assert iters == 1  # one empty poll → return

    def test_admitted_sequences_complete_via_pool_loop(self) -> None:
        q: Queue = Queue()
        q.put(
            {
                "sequences": [
                    {"uuid": "p1", "text": "hello", "max_decode_length": PAGE},
                    {"uuid": "p2", "text": "world", "max_decode_length": PAGE},
                ]
            }
        )
        orch, state, _col, sink = _build(
            admission_queue=q,
            config=_make_config(max_decode_length_pages=1, max_pool_size=16),
        )

        iters = orch.generate_persistent(max_iterations=10)

        assert iters >= 1
        for uuid in ["p1", "p2"]:
            assert state.global_batch.get_sequence(uuid).status == SequenceStatus.COMPLETED  # type: ignore[union-attr]
            assert uuid in sink.reported


# ---------------------------------------------------------------------------
# CTX fast-fail propagation
# ---------------------------------------------------------------------------


class TestCtxFastFailPropagation:
    def test_seed_sequence_with_drifted_ctx_raises_in_run_batch(self) -> None:
        from batchgen.worker.exceptions import CtxInvariantViolation

        orch, state, _col, _sink = _build(
            config=_make_config(max_decode_length_pages=1)
        )
        seq = _add_queueing(state, "u1", global_idx=0, max_decode_length=PAGE)
        # Walk the seq to PREFILLED so the decode-phase CTX check is the
        # one that fires (prefill uses different state).
        state.global_batch.update_status("u1", SequenceStatus.IN_PREFILL)
        state.global_batch.update_status("u1", SequenceStatus.PREFILLED)
        seq.current_context_length = 999  # drift

        with pytest.raises(CtxInvariantViolation) as exc:
            orch.run_batch()
        assert exc.value.uuid == "u1"
