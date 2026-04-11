"""Unit tests for batchgen.worker.boundary.executor.BoundaryExecutor."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
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
from batchgen.worker.boundary.executor import BoundaryExecutor
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator
from tests.unit.worker.fakes import (
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
)


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
    decoded_length: int = 0,
    current_context_length: int | None = None,
    gpu_pages_allocated: int = 0,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=0,
        prompt_length=10,
        max_decode_length=100,
        text="",
    )
    seq.decoded_length = decoded_length
    seq.current_context_length = (
        current_context_length
        if current_context_length is not None
        else seq.prompt_length + decoded_length
    )
    seq.original_prompt_length = seq.prompt_length
    seq.assigned_rank = state.rank
    seq.gpu_pages_allocated = gpu_pages_allocated
    state.global_batch.add_sequence(seq)
    if status_path:
        for s in status_path:
            state.global_batch.update_status(uuid, s)
    return seq


def _make_executor(
    state: WorkerState,
    *,
    gpu_free: int = 128,
    host_free: int = 512,
) -> tuple[
    BoundaryExecutor,
    FakeGpuKvBackend,
    FakeHostKvBackend,
    FakeCollectiveBackend,
    KVCacheManager,
    HostKVRebalancer,
]:
    gpu = FakeGpuKvBackend(free_pages=gpu_free)
    host = FakeHostKvBackend(free_pages=host_free)
    col = FakeCollectiveBackend(rank=state.rank, world_size=state.world_size)
    kv = KVCacheManager(
        state,
        gpu,
        host,
        initial_gpu_page_buffer=8,
        extension_gpu_page_buffer=4,
        host_kv_total_pages=1000,
        prefill_watermark_pct=70,
    )
    sync = SyncCoordinator(state, col)
    rb = HostKVRebalancer(state, kv, sync)
    exe = BoundaryExecutor(state, kv, rb)
    return exe, gpu, host, col, kv, rb


IN_DECODE_PATH = [
    SequenceStatus.IN_PREFILL,
    SequenceStatus.PREFILLED,
    SequenceStatus.IN_DECODE,
]


# ---------------------------------------------------------------------------
# Empty plan
# ---------------------------------------------------------------------------


class TestEmptyPlan:
    def test_empty_plan_is_noop(self) -> None:
        state = _make_state()
        exe, gpu, _host, col, kv, _rb = _make_executor(state)
        exe.apply(BoundaryPlan())
        assert gpu.calls == []
        assert col.calls == []
        assert kv.deferred_count == 0


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


class TestApplyRelease:
    def test_release_frees_gpu_pages_for_completed(self) -> None:
        state = _make_state()
        seq = _add(
            state,
            "u1",
            status_path=[
                *IN_DECODE_PATH,
                SequenceStatus.COMPLETED,
            ],
        )
        exe, gpu, _host, _col, kv, _rb = _make_executor(state)
        # Simulate that the sequence held some GPU pages before completing.
        kv.allocate_two_page_buffer("u1")
        assert gpu.allocated_pages("u1")

        plan = BoundaryPlan(decisions=(ReleasePages(uuids=("u1",)),))
        exe.apply(plan)

        assert not gpu.allocated_pages("u1")
        assert seq.gpu_pages_allocated == 0
        # Status is NOT touched — CompletionHandler already transitioned.
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.COMPLETED  # type: ignore[union-attr]

    def test_release_on_empty_uuids_is_noop(self) -> None:
        state = _make_state()
        exe, _gpu, _host, _col, _kv, _rb = _make_executor(state)
        exe.apply(BoundaryPlan(decisions=(ReleasePages(uuids=()),)))


# ---------------------------------------------------------------------------
# Evict (M4 minimal semantics)
# ---------------------------------------------------------------------------


class TestApplyEvict:
    def test_evict_in_decode_transitions_and_releases(self) -> None:
        state = _make_state()
        _add(state, "u1", status_path=IN_DECODE_PATH)
        exe, gpu, _host, _col, kv, _rb = _make_executor(state)
        kv.allocate_two_page_buffer("u1")

        plan = BoundaryPlan(
            decisions=(Evict(uuids=("u1",), reason=EvictReason.HOST_KV_WATERMARK),)
        )
        exe.apply(plan)

        assert state.global_batch.get_sequence("u1").status == SequenceStatus.EVICTED  # type: ignore[union-attr]
        assert not gpu.allocated_pages("u1")

    def test_evict_skips_sequences_not_in_decode_or_on_hold(self) -> None:
        """Evicting from QUEUEING/PREFILLED/COMPLETED is not a legal
        transition. The executor silently skips rather than raising —
        the planner should not have emitted it, but a skip beats a crash
        on a transient race."""
        state = _make_state()
        _add(state, "u1")  # QUEUEING
        exe, _gpu, _host, _col, _kv, _rb = _make_executor(state)
        plan = BoundaryPlan(
            decisions=(Evict(uuids=("u1",), reason=EvictReason.HOST_KV_WATERMARK),)
        )
        exe.apply(plan)
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.QUEUEING  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# OnHold — MUST route through HostKVRebalancer.put_on_hold
# ---------------------------------------------------------------------------


class TestApplyOnHold:
    def test_onhold_routes_through_host_rebalancer(self) -> None:
        state = _make_state()
        _add(state, "u1", status_path=IN_DECODE_PATH)
        exe, _gpu, _host, col, kv, _rb = _make_executor(state)
        kv.allocate_two_page_buffer("u1")

        # Give kv manager something to flush so the ordering pin has substance.
        kv.append_async("u1", layer=0, kv=torch.zeros(1))
        assert kv.deferred_count == 1

        plan = BoundaryPlan(
            decisions=(OnHold(uuids=("u1",), reason=OnHoldReason.EXTENSION_FAILED),)
        )
        exe.apply(plan)

        # The full Decision #2 ordering must have fired:
        assert kv.deferred_count == 0            # 1. flush
        assert kv.wait_pending_call_count == 1   # 2. wait
        assert state.global_batch.get_sequence("u1").gpu_pages_allocated == 0  # 3. release
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.ON_HOLD  # 4. transition
        assert col.call_names() == ["all_gather_object"]  # 5. sync_metadata


# ---------------------------------------------------------------------------
# ExtendPages
# ---------------------------------------------------------------------------


class TestApplyExtend:
    def test_extend_grows_gpu_allocation(self) -> None:
        state = _make_state()
        _add(state, "u1", status_path=IN_DECODE_PATH)
        exe, _gpu, _host, _col, kv, _rb = _make_executor(state)
        kv.allocate_two_page_buffer("u1")
        before = state.global_batch.get_sequence("u1").gpu_pages_allocated  # type: ignore[union-attr]

        plan = BoundaryPlan(
            decisions=(ExtendPages(uuid="u1", additional_pages=4),)
        )
        exe.apply(plan)

        assert state.global_batch.get_sequence("u1").gpu_pages_allocated == before + 4  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# AsyncLoad — M5 stub
# ---------------------------------------------------------------------------


class TestAsyncLoadStub:
    def test_async_load_raises_not_implemented(self) -> None:
        state = _make_state()
        _add(state, "u1")
        exe, *_ = _make_executor(state)
        plan = BoundaryPlan(
            decisions=(AsyncLoadHostToGpu(uuid="u1", host_pages=(1, 2)),)
        )
        with pytest.raises(NotImplementedError, match="M5"):
            exe.apply(plan)


# ---------------------------------------------------------------------------
# Canonical order
# ---------------------------------------------------------------------------


class TestCanonicalOrder:
    def test_release_before_evict_before_onhold_before_extend(self) -> None:
        """Use the order of gpu backend calls as the order witness.
        Release: release_pages(u_done). Evict: release_pages(u_evict)
        after the EVICTED transition. OnHold: allocate/release around
        the u_hold allocation. Extend: extend_pages(u_extend)."""
        state = _make_state()
        _add(state, "u_done", status_path=[*IN_DECODE_PATH, SequenceStatus.COMPLETED])
        _add(state, "u_evict", status_path=IN_DECODE_PATH)
        _add(state, "u_hold", status_path=IN_DECODE_PATH)
        _add(state, "u_ext", status_path=IN_DECODE_PATH)
        exe, gpu, _host, _col, kv, _rb = _make_executor(state, gpu_free=200)

        # Pre-allocate pages so release/evict/hold/extend all have
        # concrete backend interactions to observe.
        kv.allocate_two_page_buffer("u_done")
        kv.allocate_two_page_buffer("u_evict")
        kv.allocate_two_page_buffer("u_hold")
        kv.allocate_two_page_buffer("u_ext")
        start = len(gpu.calls)

        plan = BoundaryPlan(
            decisions=(
                ExtendPages(uuid="u_ext", additional_pages=4),
                OnHold(uuids=("u_hold",), reason=OnHoldReason.EXTENSION_FAILED),
                ReleasePages(uuids=("u_done",)),
                Evict(uuids=("u_evict",), reason=EvictReason.HOST_KV_WATERMARK),
            )
        )
        exe.apply(plan)

        # Collect the uuid mentioned on each gpu backend call after `start`
        def _first_uuid(call: tuple) -> str | None:
            name, args = call
            if name in {"release_pages", "allocate_pages", "extend_pages"}:
                return args[0] if args else None
            return None

        seen_order: list[str] = []
        for call in gpu.calls[start:]:
            u = _first_uuid(call)
            if u is not None and u in {"u_done", "u_evict", "u_hold", "u_ext"}:
                seen_order.append(u)

        # Release first (u_done), then Evict (u_evict), then OnHold
        # (u_hold — routed through release in put_on_hold), then Extend
        # (u_ext — extend_pages). Allow multiple references to the same
        # uuid, check the FIRST touch of each.
        first_seen: dict[str, int] = {}
        for i, u in enumerate(seen_order):
            if u not in first_seen:
                first_seen[u] = i
        assert first_seen["u_done"] < first_seen["u_evict"]
        assert first_seen["u_evict"] < first_seen["u_hold"]
        assert first_seen["u_hold"] < first_seen["u_ext"]
