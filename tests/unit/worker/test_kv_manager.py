"""Unit tests for batchgen.worker.kv_manager.KVCacheManager."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry
from batchgen.worker.kv_manager import DeferredAppend, KVCacheManager
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import FakeGpuKvBackend, FakeHostKvBackend


def _make_state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=rank,
        torch_device=torch.device("cpu"),
    )


def _add_seq(state: WorkerState, uuid: str, global_idx: int = 0) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=10,
        max_decode_length=100,
        text="",
    )
    state.global_batch.add_sequence(seq)
    return seq


def _make_manager(
    state: WorkerState,
    *,
    gpu_free: int = 128,
    host_free: int = 256,
    host_total: int = 1000,
    initial: int = 8,
    extension: int = 4,
    watermark_pct: int = 70,
) -> tuple[KVCacheManager, FakeGpuKvBackend, FakeHostKvBackend]:
    gpu = FakeGpuKvBackend(free_pages=gpu_free)
    host = FakeHostKvBackend(free_pages=host_free)
    mgr = KVCacheManager(
        state,
        gpu,
        host,
        initial_gpu_page_buffer=initial,
        extension_gpu_page_buffer=extension,
        host_kv_total_pages=host_total,
        host_kv_watermark_pct=watermark_pct,
    )
    return mgr, gpu, host


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_initial_zero_raises(self) -> None:
        state = _make_state()
        with pytest.raises(ValueError, match="initial_gpu_page_buffer"):
            KVCacheManager(
                state,
                FakeGpuKvBackend(),
                FakeHostKvBackend(),
                initial_gpu_page_buffer=0,
                extension_gpu_page_buffer=4,
                host_kv_total_pages=100,
                host_kv_watermark_pct=70,
            )

    def test_extension_negative_raises(self) -> None:
        state = _make_state()
        with pytest.raises(ValueError, match="extension_gpu_page_buffer"):
            KVCacheManager(
                state,
                FakeGpuKvBackend(),
                FakeHostKvBackend(),
                initial_gpu_page_buffer=8,
                extension_gpu_page_buffer=-1,
                host_kv_total_pages=100,
                host_kv_watermark_pct=70,
            )

    def test_host_total_zero_raises(self) -> None:
        state = _make_state()
        with pytest.raises(ValueError, match="host_kv_total_pages"):
            KVCacheManager(
                state,
                FakeGpuKvBackend(),
                FakeHostKvBackend(),
                initial_gpu_page_buffer=8,
                extension_gpu_page_buffer=4,
                host_kv_total_pages=0,
                host_kv_watermark_pct=70,
            )

    def test_watermark_out_of_range_raises(self) -> None:
        state = _make_state()
        with pytest.raises(ValueError, match="host_kv_watermark_pct"):
            KVCacheManager(
                state,
                FakeGpuKvBackend(),
                FakeHostKvBackend(),
                initial_gpu_page_buffer=8,
                extension_gpu_page_buffer=4,
                host_kv_total_pages=100,
                host_kv_watermark_pct=101,
            )


# ---------------------------------------------------------------------------
# Allocation: two-page buffer
# ---------------------------------------------------------------------------


class TestAllocateTwoPageBuffer:
    def test_allocates_initial_plus_extension(self) -> None:
        state = _make_state()
        _add_seq(state, "u1")
        mgr, gpu, _ = _make_manager(state, initial=8, extension=4)

        pages = mgr.allocate_two_page_buffer("u1")

        assert len(pages) == 12
        assert gpu.allocated_pages("u1") == list(range(12))
        seq = state.global_batch.get_sequence("u1")
        assert seq is not None
        assert seq.gpu_pages_allocated == 12
        assert seq.had_initial_gpu_reservation is True

    def test_missing_seq_still_calls_backend(self) -> None:
        """The backend call must still happen — the handler does not know
        whether the caller will attach metadata later. Skipping state
        writes silently is OK."""
        state = _make_state()
        mgr, gpu, _ = _make_manager(state)
        pages = mgr.allocate_two_page_buffer("ghost")
        assert len(pages) == 12
        assert gpu.allocated_pages("ghost") == list(range(12))


# ---------------------------------------------------------------------------
# extend_allocation
# ---------------------------------------------------------------------------


class TestExtendAllocation:
    def test_extend_after_initial_grows_count(self) -> None:
        state = _make_state()
        _add_seq(state, "u1")
        mgr, gpu, _ = _make_manager(state, initial=8, extension=4)
        mgr.allocate_two_page_buffer("u1")  # 12 pages

        extra = mgr.extend_allocation("u1", 3)

        assert len(extra) == 3
        seq = state.global_batch.get_sequence("u1")
        assert seq is not None and seq.gpu_pages_allocated == 15

    def test_extend_zero_is_noop(self) -> None:
        state = _make_state()
        _add_seq(state, "u1")
        mgr, gpu, _ = _make_manager(state)
        mgr.allocate_two_page_buffer("u1")
        before_free = gpu.free_pages()

        assert mgr.extend_allocation("u1", 0) == []
        assert gpu.free_pages() == before_free

    def test_extend_negative_is_noop(self) -> None:
        state = _make_state()
        _add_seq(state, "u1")
        mgr, _, _ = _make_manager(state)
        mgr.allocate_two_page_buffer("u1")
        assert mgr.extend_allocation("u1", -5) == []


# ---------------------------------------------------------------------------
# release_pages (multi-uuid)
# ---------------------------------------------------------------------------


class TestReleasePages:
    def test_releases_all_and_clears_state(self) -> None:
        state = _make_state()
        _add_seq(state, "u1")
        _add_seq(state, "u2", global_idx=1)
        mgr, gpu, _ = _make_manager(state, gpu_free=128, initial=8, extension=4)
        mgr.allocate_two_page_buffer("u1")
        mgr.allocate_two_page_buffer("u2")
        before_free = gpu.free_pages()

        mgr.release_pages(["u1", "u2"])

        assert gpu.free_pages() == before_free + 24
        for uuid in ("u1", "u2"):
            seq = state.global_batch.get_sequence(uuid)
            assert seq is not None
            assert seq.gpu_pages_allocated == 0
            assert seq.had_initial_gpu_reservation is False

    def test_release_missing_uuid_is_skipped_on_state(self) -> None:
        state = _make_state()
        _add_seq(state, "u1")
        mgr, gpu, _ = _make_manager(state)
        mgr.allocate_two_page_buffer("u1")
        # Backend is called for every input uuid but state write is skipped
        mgr.release_pages(["u1", "ghost"])
        assert gpu.free_pages() == 128  # all 12 returned, no ghost entry existed


# ---------------------------------------------------------------------------
# Deferred append + flush + wait
# ---------------------------------------------------------------------------


class TestDeferredAppendFlushWait:
    def test_append_async_queues_without_backend_call(self) -> None:
        state = _make_state()
        mgr, gpu, _ = _make_manager(state)
        tensor = torch.zeros(4)

        mgr.append_async("u1", layer=0, kv=tensor)
        mgr.append_async("u1", layer=1, kv=tensor)

        assert mgr.deferred_count == 2
        # No backend call yet
        assert not any(c[0] == "append_kv" for c in gpu.calls)

    def test_flush_deferred_applies_all_and_clears(self) -> None:
        state = _make_state()
        mgr, gpu, _ = _make_manager(state)
        t0 = torch.zeros(4)
        t1 = torch.zeros(4)
        mgr.append_async("u1", layer=0, kv=t0)
        mgr.append_async("u1", layer=1, kv=t1)

        applied = mgr.flush_deferred()

        assert applied == 2
        assert mgr.deferred_count == 0
        append_calls = [c for c in gpu.calls if c[0] == "append_kv"]
        assert len(append_calls) == 2
        assert append_calls[0][1][0] == "u1"
        assert append_calls[0][1][1] == 0

    def test_flush_empty_is_noop(self) -> None:
        state = _make_state()
        mgr, _, _ = _make_manager(state)
        assert mgr.flush_deferred() == 0
        assert mgr.deferred_count == 0

    def test_wait_pending_increments_counter(self) -> None:
        state = _make_state()
        mgr, _, _ = _make_manager(state)
        assert mgr.wait_pending_call_count == 0
        mgr.wait_pending()
        mgr.wait_pending()
        assert mgr.wait_pending_call_count == 2


# ---------------------------------------------------------------------------
# Free-page delegation
# ---------------------------------------------------------------------------


class TestFreePageDelegation:
    def test_get_host_free_pages_delegates(self) -> None:
        state = _make_state()
        mgr, _, host = _make_manager(state, host_free=99)
        assert mgr.get_host_free_pages() == 99
        host.allocate_pages("u1", 5)
        assert mgr.get_host_free_pages() == 94

    def test_get_gpu_free_pages_delegates(self) -> None:
        state = _make_state()
        mgr, gpu, _ = _make_manager(state, gpu_free=64)
        assert mgr.get_gpu_free_pages() == 64
        gpu.allocate_pages("u1", 10)
        assert mgr.get_gpu_free_pages() == 54


# ---------------------------------------------------------------------------
# Watermark trigger
# ---------------------------------------------------------------------------


class TestCheckWatermarkTrigger:
    def test_free_above_watermark_returns_false(self) -> None:
        """800/1000 free = 80%, watermark 70. Above → no trigger."""
        state = _make_state()
        mgr, _, _ = _make_manager(state, host_free=800, host_total=1000, watermark_pct=70)
        assert mgr.check_watermark_trigger() is False

    def test_free_at_watermark_returns_false(self) -> None:
        """700/1000 free = 70%, strict < not <=. At → no trigger."""
        state = _make_state()
        mgr, _, _ = _make_manager(state, host_free=700, host_total=1000, watermark_pct=70)
        assert mgr.check_watermark_trigger() is False

    def test_free_below_watermark_returns_true(self) -> None:
        state = _make_state()
        mgr, _, _ = _make_manager(state, host_free=699, host_total=1000, watermark_pct=70)
        assert mgr.check_watermark_trigger() is True

    def test_zero_free_returns_true(self) -> None:
        state = _make_state()
        mgr, _, _ = _make_manager(state, host_free=0, host_total=1000, watermark_pct=70)
        assert mgr.check_watermark_trigger() is True

    def test_watermark_zero_never_triggers(self) -> None:
        """A watermark of 0% means 'free < 0%' — impossible → never trigger."""
        state = _make_state()
        mgr, _, _ = _make_manager(state, host_free=0, host_total=1000, watermark_pct=0)
        assert mgr.check_watermark_trigger() is False


# ---------------------------------------------------------------------------
# DeferredAppend dataclass sanity
# ---------------------------------------------------------------------------


class TestDeferredAppendDataclass:
    def test_dataclass_is_frozen_and_equal(self) -> None:
        t = torch.zeros(1)
        a = DeferredAppend(uuid="u1", layer=0, kv=t)
        b = DeferredAppend(uuid="u1", layer=0, kv=t)
        assert a == b
        with pytest.raises(Exception):
            a.uuid = "u2"  # type: ignore[misc]
