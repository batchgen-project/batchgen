"""End-to-end tests for BoundaryHandler wiring the sub-package collaborators."""

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
    ExtendPages,
    OnHold,
    OnHoldReason,
    PlannerConfig,
    ReleasePages,
)
from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator
from tests.unit.worker.fakes import (
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
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
    decoded_length: int = 0,
    current_context_length: int | None = None,
    gpu_pages_allocated: int = 0,
    global_idx: int = 0,
    prompt_length: int = 10,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=100,
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


def _make_handler(
    state: WorkerState,
    *,
    gpu_free: int = 128,
    host_free: int = 200,
    host_total: int = 1000,
    prefill_watermark: int = 70,
) -> tuple[
    BoundaryHandler,
    FakeCollectiveBackend,
    KVCacheManager,
    FakeGpuKvBackend,
    FakeHostKvBackend,
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
        host_kv_total_pages=host_total,
        prefill_watermark_pct=prefill_watermark,
    )
    sync = SyncCoordinator(state, col)
    rb = HostKVRebalancer(state, kv, sync)
    synchronizer = BoundarySynchronizer(state, sync, col)
    planner = BoundaryPlanner(
        PlannerConfig(
            prefill_watermark_pct=prefill_watermark,
            decision_frequency_pages=2,
            extension_gpu_page_buffer=4,
            host_total_pages=host_total,
        )
    )
    executor = BoundaryExecutor(state, kv, rb)
    guards = BoundaryGuards(state)
    handler = BoundaryHandler(state, synchronizer, planner, executor, guards, kv)
    return handler, col, kv, gpu, host


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHandlerHappyPath:
    def test_empty_uuids_runs_sync_and_broadcast_only(self) -> None:
        state = _make_state()
        handler, col, kv, _gpu, _host = _make_handler(state)

        plan = handler.run([])

        assert plan.decisions == ()
        # Step 1: sync_metadata (even for empty). Step 4: broadcast plan.
        assert col.call_names() == ["all_gather_object", "broadcast_object"]

    def test_no_boundary_work_returns_empty_plan(self) -> None:
        """A sequence with plenty of GPU headroom and nothing to complete
        produces an empty plan and a no-op execution."""
        state = _make_state()
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH,
            decoded_length=5,
            current_context_length=15,
            gpu_pages_allocated=32,  # 2048 tokens of headroom
        )
        handler, _col, _kv, gpu, _host = _make_handler(state)
        calls_before = len(gpu.calls)
        plan = handler.run(["u1"])
        assert plan.decisions == ()
        # No gpu backend mutations beyond the initial setup
        assert all(
            c[0] not in {"release_pages", "extend_pages"} for c in gpu.calls[calls_before:]
        )


# ---------------------------------------------------------------------------
# Feature paths end-to-end
# ---------------------------------------------------------------------------


class TestHandlerReleasePath:
    def test_completed_sequence_triggers_release(self) -> None:
        state = _make_state()
        _add(
            state,
            "u_done",
            status_path=[*IN_DECODE_PATH, SequenceStatus.COMPLETED],
            gpu_pages_allocated=4,
        )
        handler, _col, kv, gpu, _host = _make_handler(state)
        # Make the backend think u_done holds pages so release is visible.
        kv.allocate_two_page_buffer("u_done")
        before_free = gpu.free_pages()

        plan = handler.run(["u_done"])

        releases = plan.decisions_of(ReleasePages)
        assert len(releases) == 1
        assert releases[0].uuids == ("u_done",)
        assert gpu.free_pages() > before_free
        assert state.global_batch.get_sequence("u_done").gpu_pages_allocated == 0  # type: ignore[union-attr]


class TestHandlerExtendPath:
    def test_in_decode_near_overflow_triggers_extend(self) -> None:
        state = _make_state()
        # 2 pages allocated, ctx_len just under the page budget → need 4 more
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH,
            current_context_length=127,  # headroom = 128 - 127 = 1 token
            gpu_pages_allocated=2,
        )
        handler, _col, _kv, gpu, _host = _make_handler(state)

        plan = handler.run(["u1"])

        extends = plan.decisions_of(ExtendPages)
        assert len(extends) == 1
        assert extends[0] == ExtendPages(uuid="u1", additional_pages=4)
        seq = state.global_batch.get_sequence("u1")
        assert seq is not None and seq.gpu_pages_allocated == 6

    def test_in_decode_with_zero_gpu_free_onholds(self) -> None:
        state = _make_state()
        _add(
            state,
            "u1",
            status_path=IN_DECODE_PATH,
            decoded_length=50,
            current_context_length=127,
            gpu_pages_allocated=2,
        )
        handler, col, _kv, _gpu, _host = _make_handler(state, gpu_free=0)

        plan = handler.run(["u1"])

        held = plan.decisions_of(OnHold)
        assert len(held) == 1
        assert held[0].reason is OnHoldReason.EXTENSION_FAILED
        assert held[0].uuids == ("u1",)
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.ON_HOLD  # type: ignore[union-attr]
        # The run() produced two collectives: the entry sync_metadata, then
        # the plan broadcast. The OnHold then fires put_on_hold which ALSO
        # calls sync_metadata from HostKVRebalancer. Three total.
        assert col.call_names() == [
            "all_gather_object",  # sync_metadata_in
            "broadcast_object",   # broadcast_plan
            "all_gather_object",  # put_on_hold's tail sync_metadata
        ]


class TestHandlerWatermarkPath:
    def test_watermark_fired_onholds_all_in_decode(self) -> None:
        state = _make_state()
        # IN_DECODE seq and a QUEUEING seq so has_pending=True
        _add(
            state,
            "u_dec",
            status_path=IN_DECODE_PATH,
            decoded_length=5,
            current_context_length=15,
            gpu_pages_allocated=32,
        )
        _add(state, "u_q", global_idx=1)  # QUEUEING
        handler, _col, _kv, _gpu, _host = _make_handler(
            state,
            host_free=800,  # 80% > 70
        )

        plan = handler.run(["u_dec"])  # only the in-decode is in the sync list

        onholds = plan.decisions_of(OnHold)
        assert len(onholds) == 1
        assert onholds[0].reason is OnHoldReason.WATERMARK_TRIGGER
        assert onholds[0].uuids == ("u_dec",)
        assert state.global_batch.get_sequence("u_dec").status == SequenceStatus.ON_HOLD  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Fast-fail paths
# ---------------------------------------------------------------------------


class TestHandlerFastFail:
    def test_ctx_drift_at_sync_raises(self) -> None:
        state = _make_state()
        seq = _add(state, "u1", status_path=IN_DECODE_PATH, decoded_length=5)
        seq.current_context_length = 99  # drifted
        handler, _col, _kv, _gpu, _host = _make_handler(state)

        with pytest.raises(CtxInvariantViolation) as exc:
            handler.run(["u1"])
        assert exc.value.side == "sender"
        # Plan never executed — seq is still IN_DECODE
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.IN_DECODE  # type: ignore[union-attr]
