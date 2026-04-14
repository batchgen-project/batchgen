"""Unit tests for BoundaryExecutor.apply_full — Phase 2.8.1e port.

Covers each canonical-order branch of ``_boundary_execute_decisions``
+ ``_boundary_async_load`` (batchgen_worker.py:6912-7223). The existing
``test_executor.py`` still exercises the M4 ``apply()`` method; those
tests are unchanged.

Test fakes:
  * ``FakeGpuManager``: tracks free pages + records allocation /
    rebuild / free operations.
  * ``FakeWorkerView``: records grow / release / unregister / async
    load calls; returns an opaque handle from ``async_load_*``.
"""

from __future__ import annotations

import types
from typing import Any

import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary.decisions import (
    BoundaryPlan,
    ExtendPages,
    HostEvict,
    HostGrow,
    NewLoadAsync,
    OnHold,
    OnHoldReason,
    ReleasePages,
)
from batchgen.worker.boundary.executor import BoundaryExecutor
from batchgen.worker.boundary.synchronizer import (
    LoadCandidateState,
    SeqBoundaryState,
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


PAGE = SequenceEntry.PAGE_SIZE  # 64


# ---------------------------------------------------------------------------
# Fakes specific to apply_full
# ---------------------------------------------------------------------------


class FakeGpuManager:
    """Minimal GPU paged-KV manager stub.

    Records the calls apply_full drives: ``free_pages_for_sequences``,
    ``allocate_pages_for_sequences``, ``rebuild_page_table``,
    ``get_padded_3d_page_pointers``, ``export_active_sequence_page_counts``.
    """

    def __init__(self, *, num_free_pages: int = 100, is_initialized: bool = True) -> None:
        self.is_initialized = is_initialized
        self._num_free = num_free_pages
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def get_stats(self) -> Any:
        return types.SimpleNamespace(num_free_pages=self._num_free)

    def free_pages_for_sequences(self, global_ids: list[int]) -> None:
        self.calls.append(("free_pages_for_sequences", (tuple(global_ids),)))

    def allocate_pages_for_sequences(
        self, global_ids: list[int], tokens: list[int]
    ) -> None:
        self.calls.append(
            ("allocate_pages_for_sequences", (tuple(global_ids), tuple(tokens)))
        )

    def rebuild_page_table(self, global_ids: list[int]) -> None:
        self.calls.append(("rebuild_page_table", (tuple(global_ids),)))

    def get_padded_3d_page_pointers(self) -> tuple[Any, Any]:
        self.calls.append(("get_padded_3d_page_pointers", ()))
        return (object(), object())  # opaque tensors

    def export_active_sequence_page_counts(self) -> Any:
        self.calls.append(("export_active_sequence_page_counts", ()))
        return object()


class FakeWorkerView:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def grow_pages_for_sequences(self, requests: list[tuple[int, int]]) -> None:
        self.calls.append(("grow_pages_for_sequences", (tuple(requests),)))

    def release_sequence_pages(self, global_ids: list[int]) -> None:
        self.calls.append(("release_sequence_pages", (tuple(global_ids),)))

    def unregister_sequences(self, global_ids: list[int]) -> None:
        self.calls.append(("unregister_sequences", (tuple(global_ids),)))

    def async_load_layer_paged_kv_to_device(self, **kwargs: Any) -> Any:
        self.calls.append(("async_load_layer_paged_kv_to_device", (kwargs,)))
        return types.SimpleNamespace(name="fake_async_handle")


# ---------------------------------------------------------------------------
# Shared fixture wiring
# ---------------------------------------------------------------------------


def _make_state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=rank,
        torch_device=torch.device("cpu"),
    )


def _add(
    state: WorkerState,
    uuid: str,
    *,
    status: SequenceStatus = SequenceStatus.IN_DECODE,
    global_idx: int = 0,
    prompt_length: int = 10,
    decoded_length: int = 5,
    gpu_pages_allocated: int = 2,
    host_pages_allocated: int = 4,
    assigned_rank: int = 0,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=100,
        text="",
    )
    seq.original_prompt_length = prompt_length
    seq.decoded_length = decoded_length
    seq.current_context_length = prompt_length + decoded_length
    seq.gpu_pages_allocated = gpu_pages_allocated
    seq.host_pages_allocated = host_pages_allocated
    seq.host_token_capacity = host_pages_allocated * seq.PAGE_SIZE
    seq.assigned_rank = assigned_rank
    seq.input_ids = torch.arange(prompt_length + decoded_length).unsqueeze(0)
    seq.decoded_tokens = torch.zeros(
        (1, 100), dtype=torch.int64
    )
    seq.status = status
    state.global_batch.sequences[uuid] = seq
    state.global_batch._status_index[status].add(uuid)
    return seq


def _executor(state: WorkerState) -> BoundaryExecutor:
    col = FakeCollectiveBackend(rank=state.rank, world_size=state.world_size)
    sync = SyncCoordinator(state, col)
    kv = KVCacheManager(
        state=state,
        sync=sync,
        gpu=FakeGpuKvBackend(),
        host=FakeHostKvBackend(),
    )
    rebalancer = HostKVRebalancer(state=state, kv=kv, sync=sync)
    return BoundaryExecutor(state=state, kv=kv, rebalancer=rebalancer)


def _state_for(uuid: str, **overrides) -> SeqBoundaryState:
    base = dict(
        decoded_length=5,
        current_context_length=15,
        gpu_pages_allocated=2,
        eos_reached=False,
        completed=False,
        additional_pages_needed=0,
        assigned_rank=0,
        needs_host_growth=False,
        host_growth_pages=0,
        host_pages_allocated=4,
        host_token_capacity=256,
        prompt_length=10,
        total_decoded_before_eviction=0,
        owning_rank=0,
    )
    base.update(overrides)
    return SeqBoundaryState(**base)


# ---------------------------------------------------------------------------
# ReleasePages branch
# ---------------------------------------------------------------------------


class TestReleaseFull:
    def test_release_transitions_status_and_frees_gpu(self) -> None:
        state = _make_state()
        _add(state, "done", status=SequenceStatus.IN_DECODE)
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"done": 0}
        legacy._sequences_with_gpu_kv = {"done"}

        plan = BoundaryPlan(decisions=(ReleasePages(uuids=("done",)),))
        executor = _executor(state)
        decode_uuids, batch, *_ = executor.apply_full(
            plan,
            decode_uuids=["done"],
            batch=[0],
            gpu_manager=FakeGpuManager(),
            global_seq_state={"done": _state_for("done")},
            global_candidate_info={},
            chunk_size=PAGE,
            adapter=legacy,
        )

        call_names = [c[0] for c in legacy.calls]
        assert "update_batch_status" in call_names
        assert "submit_completed_to_incremental_writer" in call_names
        assert "release_gpu_kv_pages" in call_names
        assert "release_host_kv_pages_for_batch" in call_names
        assert "report_completion" in call_names
        assert "report_chunk_sizer_completion" in call_names
        seq = state.global_batch.get_sequence("done")
        assert seq.gpu_pages_allocated == 0
        assert seq.host_pages_allocated == 0
        assert decode_uuids == []
        assert batch == []


# ---------------------------------------------------------------------------
# HostGrow branch
# ---------------------------------------------------------------------------


class TestHostGrowFull:
    def test_feasible_grow_touches_worker_view_and_updates_scalars(self) -> None:
        state = _make_state()
        seq = _add(state, "u", host_pages_allocated=4)
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}
        view = FakeWorkerView()
        legacy._host_paged_kv_worker_view = view  # type: ignore[attr-defined]

        plan = BoundaryPlan(
            decisions=(HostGrow(uuids=("u",), pages=(3,), feasible=True),)
        )
        executor = _executor(state)
        executor.apply_full(
            plan,
            decode_uuids=["u"],
            batch=[0],
            gpu_manager=FakeGpuManager(),
            global_seq_state={"u": _state_for("u")},
            global_candidate_info={},
            chunk_size=PAGE,
            adapter=legacy,
        )

        assert seq.host_pages_allocated == 4 + 3
        assert seq.host_token_capacity == (4 + 3) * seq.PAGE_SIZE
        grow_calls = [c for c in view.calls if c[0] == "grow_pages_for_sequences"]
        assert len(grow_calls) == 1
        assert grow_calls[0][1][0] == ((seq.global_idx, 3),)

    def test_infeasible_grow_skips_worker_view(self) -> None:
        state = _make_state()
        _add(state, "u")
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}
        view = FakeWorkerView()
        legacy._host_paged_kv_worker_view = view  # type: ignore[attr-defined]

        plan = BoundaryPlan(
            decisions=(HostGrow(uuids=("u",), pages=(3,), feasible=False),)
        )
        executor = _executor(state)
        executor.apply_full(
            plan,
            decode_uuids=["u"],
            batch=[0],
            gpu_manager=FakeGpuManager(),
            global_seq_state={"u": _state_for("u")},
            global_candidate_info={},
            chunk_size=PAGE,
            adapter=legacy,
        )
        assert view.calls == []


# ---------------------------------------------------------------------------
# HostEvict branch
# ---------------------------------------------------------------------------


class TestHostEvictFull:
    def test_evict_updates_status_builds_token_ids_drops_from_decode(self) -> None:
        state = _make_state()
        seq = _add(state, "u", status=SequenceStatus.IN_DECODE)
        seq.reentry_decoded_baseline = 0
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}
        legacy._sequences_with_gpu_kv = {"u"}
        view = FakeWorkerView()
        legacy._host_paged_kv_worker_view = view  # type: ignore[attr-defined]

        plan = BoundaryPlan(decisions=(HostEvict(uuids=("u",)),))
        executor = _executor(state)
        decode_uuids, *_ = executor.apply_full(
            plan,
            decode_uuids=["u"],
            batch=[0],
            gpu_manager=FakeGpuManager(),
            global_seq_state={"u": _state_for("u")},
            global_candidate_info={},
            chunk_size=PAGE,
            adapter=legacy,
        )
        assert decode_uuids == []
        assert seq.status == SequenceStatus.EVICTED
        assert seq.gpu_pages_allocated == 0
        assert seq.host_pages_allocated == 0
        assert seq.evicted_token_ids is not None
        # prompt_length was 10, decoded=5, baseline=0 → new reentry = 15
        assert seq.prompt_length == 15
        assert seq.current_context_length == 15
        # worker_view invoked
        assert any(
            c[0] == "release_sequence_pages" for c in view.calls
        )
        assert any(c[0] == "unregister_sequences" for c in view.calls)


# ---------------------------------------------------------------------------
# OnHold branch (EXTENSION_FAILED + WATERMARK_TRIGGER share path)
# ---------------------------------------------------------------------------


class TestOnHoldFull:
    def test_watermark_trigger_releases_gpu_and_sets_onhold(self) -> None:
        state = _make_state()
        seq = _add(state, "u", status=SequenceStatus.IN_DECODE)
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}
        legacy._sequences_with_gpu_kv = {"u"}
        gpu = FakeGpuManager()

        plan = BoundaryPlan(
            decisions=(OnHold(uuids=("u",), reason=OnHoldReason.WATERMARK_TRIGGER),)
        )
        executor = _executor(state)
        decode_uuids, batch, *_ = executor.apply_full(
            plan,
            decode_uuids=["u"],
            batch=[0],
            gpu_manager=gpu,
            global_seq_state={"u": _state_for("u")},
            global_candidate_info={},
            chunk_size=PAGE,
            adapter=legacy,
        )
        assert seq.status == SequenceStatus.ON_HOLD
        assert seq.gpu_pages_allocated == 0
        assert decode_uuids == []
        assert batch == []
        assert any(
            c[0] == "free_pages_for_sequences" for c in gpu.calls
        )


# ---------------------------------------------------------------------------
# ExtendPages branch
# ---------------------------------------------------------------------------


class TestExtendFull:
    def test_extend_calls_adapter_when_rank_owned(self) -> None:
        state = _make_state()
        _add(state, "u", status=SequenceStatus.IN_DECODE)
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}

        plan = BoundaryPlan(
            decisions=(ExtendPages(uuid="u", additional_pages=3),)
        )
        executor = _executor(state)
        decode_uuids, *_ = executor.apply_full(
            plan,
            decode_uuids=["u"],
            batch=[0],
            gpu_manager=FakeGpuManager(),
            global_seq_state={"u": _state_for("u")},
            global_candidate_info={},
            chunk_size=PAGE,
            adapter=legacy,
        )
        assert ("extend_gpu_kv_allocation", (["u"],), {}) in legacy.calls
        # u stays in decode on success
        assert decode_uuids == ["u"]


# ---------------------------------------------------------------------------
# NewLoadAsync branch
# ---------------------------------------------------------------------------


class TestNewLoadAsyncFull:
    def test_new_load_launches_async_task(self) -> None:
        state = _make_state()
        # Active seq in decode (consumer of existing batch pointer)
        _add(state, "active", status=SequenceStatus.IN_DECODE)
        # Candidate to load
        candidate = _add(
            state, "load", status=SequenceStatus.PREFILLED,
            global_idx=7, host_pages_allocated=5,
        )
        candidate.had_initial_gpu_reservation = False

        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"active": 0, "load": 1}
        legacy._local_to_uuid = {0: "active", 1: "load"}
        view = FakeWorkerView()
        legacy._host_paged_kv_worker_view = view  # type: ignore[attr-defined]
        gpu = FakeGpuManager(num_free_pages=100)

        plan = BoundaryPlan(
            decisions=(
                NewLoadAsync(uuids=("load",), rank_pages=((0, 3),)),
            )
        )
        global_candidate_info = {
            "load": LoadCandidateState(
                pages_needed=3, assigned_rank=0,
                status="PREFILLED", decoded_length=0,
            ),
        }
        executor = _executor(state)
        (
            decode_uuids,
            batch,
            new_async_task,
            new_load_uuids,
            new_load_local,
            new_load_global,
        ) = executor.apply_full(
            plan,
            decode_uuids=["active"],
            batch=[0],
            gpu_manager=gpu,
            global_seq_state={"active": _state_for("active")},
            global_candidate_info=global_candidate_info,
            chunk_size=PAGE,
            adapter=legacy,
        )

        assert new_async_task is not None
        assert new_load_uuids == ["load"]
        assert new_load_local == [1]
        assert new_load_global == [7]
        # rebuild_page_table called twice: once for new, once to restore
        rebuild_calls = [c for c in gpu.calls if c[0] == "rebuild_page_table"]
        assert len(rebuild_calls) == 2
        # async_load_layer_paged_kv_to_device invoked on the view
        assert any(
            c[0] == "async_load_layer_paged_kv_to_device" for c in view.calls
        )


# ---------------------------------------------------------------------------
# Empty plan is a no-op passthrough
# ---------------------------------------------------------------------------


class TestEmptyPlan:
    def test_empty_plan_returns_input_decode_uuids(self) -> None:
        state = _make_state()
        _add(state, "u")
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}

        plan = BoundaryPlan()
        executor = _executor(state)
        result = executor.apply_full(
            plan,
            decode_uuids=["u"],
            batch=[0],
            gpu_manager=FakeGpuManager(),
            global_seq_state={"u": _state_for("u")},
            global_candidate_info={},
            chunk_size=PAGE,
            adapter=legacy,
        )
        assert result == (["u"], [0], None, [], [], [])
