"""End-to-end tests for ``BoundaryHandler.run_full`` (Phase 2.8.1i).

Wires every Stage 1 collaborator (synchronizer, planner, executor,
guards, wait_pending, finalize) against the CPU fakes and drives one
boundary cycle. The tests lock in:

  * Collective call count: 1 × all_gather_object + 1 × broadcast_object
    + 1 × all_gather_into_tensor + 1 × barrier per cycle (matches
    legacy `_page_boundary_fast`).
  * BoundaryResult threading: async handle + new-load uuids survive
    the return to the caller.
  * Watermark break propagates from the planner through the handler
    without running the executor/finalize downstream rules.
  * Empty-decode-uuids early return at steps 1 / 8 produces a
    well-formed BoundaryResult (no spurious collectives).
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary import (
    BoundaryExecutor,
    BoundaryGuards,
    BoundaryHandler,
    BoundaryHandlerConfig,
    BoundaryPlanner,
    BoundaryResult,
    BoundarySynchronizer,
    OnHold,
    OnHoldReason,
    PlannerConfig,
)
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator
from tests.unit.worker.fakes import (
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
    FakeLegacyBackend,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGpuManager:
    """Simplified gpu_manager for handler tests. Covers the small
    surface the handler + finalize.check step actually touch."""

    def __init__(self, *, num_free_pages: int = 100) -> None:
        self.is_initialized = True
        self._num_free = num_free_pages
        # Page-table manager state finalize + guards read.
        gpu_table = types.SimpleNamespace(shape=(1,))
        self._gpu_page_table_manager = types.SimpleNamespace(
            gpu_table=gpu_table, slot_to_seq_id=[0]
        )
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def get_stats(self) -> Any:
        return types.SimpleNamespace(num_free_pages=self._num_free)

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def free_pages_for_sequences(self, global_ids: list[int]) -> None:
        self._record("free_pages_for_sequences", global_ids)

    def allocate_pages_for_sequences(
        self, global_ids: list[int], tokens: list[int]
    ) -> None:
        self._record("allocate_pages_for_sequences", global_ids, tokens)

    def rebuild_page_table(self, global_ids: list[int]) -> None:
        self._record("rebuild_page_table", global_ids)

    def get_padded_3d_page_pointers(self) -> tuple[Any, Any]:
        return (object(), object())

    def export_active_sequence_page_counts(self) -> Any:
        return object()


class _PassThroughAdapter(FakeLegacyBackend):
    """Adapter whose `finalize_async_load_minimal` returns the inputs
    so the wait_pending rebuild path can fire when tests exercise it."""

    def finalize_async_load_minimal(self, *args: Any, **kwargs: Any) -> Any:
        self._record("finalize_async_load_minimal", *args, **kwargs)
        return (args[4], args[5])


# ---------------------------------------------------------------------------
# Fixture wiring
# ---------------------------------------------------------------------------


def _add_seq(state: WorkerState, uuid: str, *, global_idx: int = 0) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid, global_idx=global_idx,
        prompt_length=10, max_decode_length=100, text="",
    )
    seq.original_prompt_length = 10
    seq.decoded_length = 5
    seq.current_context_length = 15
    seq.assigned_rank = 0
    seq.gpu_pages_allocated = 2
    seq.host_pages_allocated = 4
    seq.host_token_capacity = 4 * seq.PAGE_SIZE
    seq.input_ids = torch.zeros((1, 15), dtype=torch.int64)
    seq.decoded_tokens = torch.zeros((1, 100), dtype=torch.int64)
    seq.status = SequenceStatus.IN_DECODE
    state.global_batch.sequences[uuid] = seq
    state.global_batch._status_index[SequenceStatus.IN_DECODE].add(uuid)
    return seq


def _build_handler(
    state: WorkerState,
    *,
    adapter: FakeLegacyBackend | None = None,
) -> tuple[BoundaryHandler, FakeCollectiveBackend, FakeLegacyBackend]:
    col = FakeCollectiveBackend(rank=state.rank, world_size=state.world_size)
    sync = SyncCoordinator(state, col)
    synchronizer = BoundarySynchronizer(state, sync, col)
    planner = BoundaryPlanner(
        PlannerConfig(
            prefill_watermark_pct=70,
            decision_frequency_pages=2,
            extension_gpu_page_buffer=4,
            host_total_pages=1000,
        )
    )
    kv = KVCacheManager(
        state,
        FakeGpuKvBackend(),
        FakeHostKvBackend(),
        initial_gpu_page_buffer=8,
        extension_gpu_page_buffer=4,
        host_kv_total_pages=1000,
        prefill_watermark_pct=70,
    )
    rb = HostKVRebalancer(state, kv, sync)
    executor = BoundaryExecutor(state, kv, rb)
    guards = BoundaryGuards(state)
    adapter_impl = adapter if adapter is not None else _PassThroughAdapter()
    handler = BoundaryHandler(
        state, synchronizer, planner, executor, guards, kv,
        adapter=adapter_impl, collectives=col,
        handler_config=BoundaryHandlerConfig(
            enable_host_kv_eviction=False,
            host_kv_eviction_watermark=20,
        ),
    )
    return handler, col, adapter_impl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConstructionGuards:
    def test_run_full_without_adapter_raises(self) -> None:
        state = WorkerState(
            rank=0, local_rank=0, world_size=1, device=0,
            torch_device=torch.device("cpu"),
        )
        col = FakeCollectiveBackend(rank=0, world_size=1)
        sync = SyncCoordinator(state, col)
        synchronizer = BoundarySynchronizer(state, sync, col)
        planner = BoundaryPlanner(PlannerConfig(
            prefill_watermark_pct=70, decision_frequency_pages=2,
            extension_gpu_page_buffer=4, host_total_pages=1000,
        ))
        kv = KVCacheManager(
            state, FakeGpuKvBackend(), FakeHostKvBackend(),
            initial_gpu_page_buffer=8, extension_gpu_page_buffer=4,
            host_kv_total_pages=1000, prefill_watermark_pct=70,
        )
        rb = HostKVRebalancer(state, kv, sync)
        executor = BoundaryExecutor(state, kv, rb)
        guards = BoundaryGuards(state)
        # Construct WITHOUT adapter/collectives/config — the M4 path.
        handler = BoundaryHandler(state, synchronizer, planner, executor, guards, kv)
        with pytest.raises(RuntimeError, match="run_full requires"):
            handler.run_full(
                decode_uuids=["u"], batch=[0], gpu_manager=FakeGpuManager(),
            )


class TestEmptyPathFastReturns:
    def test_empty_decode_uuids_returns_empty_result(self) -> None:
        """Step 1 wait_pending with empty input + no pending load →
        handler returns BoundaryResult(plan=empty) with no collectives."""
        state = WorkerState(
            rank=0, local_rank=0, world_size=1, device=0,
            torch_device=torch.device("cpu"),
        )
        handler, col, _ = _build_handler(state)
        result = handler.run_full(
            decode_uuids=[], batch=[], gpu_manager=FakeGpuManager(),
        )
        assert isinstance(result, BoundaryResult)
        assert result.plan.decisions == ()
        assert result.decode_uuids == ()
        # wait_pending with no pending load emits zero collectives.
        assert col.calls == []


class TestHappyCycle:
    def test_single_rank_cycle_emits_expected_collectives(self) -> None:
        state = WorkerState(
            rank=0, local_rank=0, world_size=1, device=0,
            torch_device=torch.device("cpu"),
        )
        _add_seq(state, "u")
        handler, col, adapter = _build_handler(state)
        adapter._uuid_to_local = {"u": 0}
        gpu = FakeGpuManager()

        result = handler.run_full(
            decode_uuids=["u"], batch=[0], gpu_manager=gpu,
        )
        # Steps 2 + 3 + 6 + 9:
        #   2 × all_gather_object (sync_metadata_in + gather_boundary_state)
        #   1 × broadcast_object (plan broadcast)
        #   1 × all_gather_into_tensor (finalize MoE gather)
        #   1 × barrier (finalize)
        names = col.call_names()
        assert names.count("all_gather_object") == 2
        assert names.count("broadcast_object") == 1
        assert names.count("all_gather_into_tensor") == 1
        assert names.count("barrier") == 1
        # The handler returned a well-formed result.
        assert result.decode_uuids == ("u",)
        assert result.batch == (0,)
        assert result.watermark_triggered is False


class TestPendingLoadStateOnHandler:
    def test_handler_starts_with_empty_pending_state(self) -> None:
        state = WorkerState(
            rank=0, local_rank=0, world_size=1, device=0,
            torch_device=torch.device("cpu"),
        )
        handler, _, _ = _build_handler(state)
        assert handler._pending_async_task is None
        assert handler._pending_load_uuids == []
        assert handler._pending_load_local == []
        assert handler._pending_load_global == []

    def test_pending_state_is_cleared_after_wait_pending(self) -> None:
        """After step 1 integrates any prior pending load the stash is
        empty, so a subsequent executor NewLoadAsync decision cleanly
        records the new load without conflict."""
        state = WorkerState(
            rank=0, local_rank=0, world_size=1, device=0,
            torch_device=torch.device("cpu"),
        )
        _add_seq(state, "u")
        handler, _, adapter = _build_handler(state)
        adapter._uuid_to_local = {"u": 0}
        # Pre-seed a pending load as if the last cycle launched one.
        class _Task:
            def wait(self) -> None:
                pass
        handler._pending_async_task = _Task()
        handler._pending_load_uuids = ["load_from_prev_cycle"]
        handler._pending_load_local = [7]
        handler._pending_load_global = [42]

        handler.run_full(
            decode_uuids=["u"], batch=[0], gpu_manager=FakeGpuManager(),
        )
        # The executor in this happy-path did not fire NewLoadAsync
        # (no load candidates in the snapshot), so after the cycle
        # the stash is back to empty.
        assert handler._pending_async_task is None
        assert handler._pending_load_uuids == []
        assert handler._pending_load_local == []
        assert handler._pending_load_global == []


class TestWatermarkBreakPath:
    def test_watermark_break_returns_without_running_finalize(self) -> None:
        """When the planner flags watermark_break, the executor still
        runs (to OnHold the IN_DECODE uuids) but finalize is skipped
        since decode_uuids ends up empty; result surfaces the plan."""
        state = WorkerState(
            rank=0, local_rank=0, world_size=1, device=0,
            torch_device=torch.device("cpu"),
        )
        _add_seq(state, "u")
        # Add a QUEUEING seq so the planner sees has_pending=True.
        ghost = SequenceEntry(uuid="q", global_idx=1, prompt_length=10, max_decode_length=100)
        ghost.assigned_rank = 0
        state.global_batch.add_sequence(ghost)

        handler, col, adapter = _build_handler(state)
        adapter._uuid_to_local = {"u": 0}

        class _AdapterWithHighWatermark(_PassThroughAdapter):
            def host_paged_kv_worker_view(self) -> Any:
                # Present a view with 90/100 free so planner sees
                # free_pct=90 > watermark=70 → fires the bailout.
                self._record("host_paged_kv_worker_view")
                return types.SimpleNamespace(
                    get_stats=lambda: types.SimpleNamespace(
                        num_total_pages=100, num_free_pages=90
                    )
                )

            def check_host_kv_watermark_trigger(self) -> bool:
                self._record("check_host_kv_watermark_trigger")
                return True

        handler, col, adapter = _build_handler(
            state, adapter=_AdapterWithHighWatermark()
        )
        adapter._uuid_to_local = {"u": 0}
        gpu = FakeGpuManager()

        result = handler.run_full(
            decode_uuids=["u"], batch=[0], gpu_manager=gpu,
        )
        # Planner bailout emitted OnHold(WATERMARK_TRIGGER), executor
        # applied it, decode_uuids is now empty → handler returned
        # without calling finalize.
        assert result.plan.watermark_break is True
        assert any(
            isinstance(d, OnHold) and d.reason is OnHoldReason.WATERMARK_TRIGGER
            for d in result.plan.decisions
        )
        # No MoE gather emitted because finalize was skipped.
        assert col.call_names().count("all_gather_into_tensor") == 0
