"""Unit tests for ``boundary/wait_pending.py`` — Phase 2.8.1f port.

Covers each branch of ``_boundary_wait_pending`` (batchgen_worker.py:
6637-6724) as ported to ``wait_pending``:

  * Happy-path drain with no pending load: one
    ``wait_pending_kv_append_tasks`` call, zero collectives.
  * Pending load: async_task.wait + collective barrier + adapter
    ``finalize_async_load_minimal`` + ``rebuild_page_table_for_batch``.
  * Slot-order mismatch repair: ``rebuild_page_table`` is called on
    the gpu_manager to restore parity.
"""

from __future__ import annotations

import types
from typing import Any

import torch

from batchgen.worker.boundary.wait_pending import wait_pending
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import FakeCollectiveBackend, FakeLegacyBackend


class _FakeAsyncTask:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


def _make_state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank, local_rank=rank, world_size=world_size, device=rank,
        torch_device=torch.device("cpu"),
    )


def _fake_gpu_manager(
    *, is_initialized: bool = True, slot_to_seq_id: list[int] | None = None
) -> Any:
    ptm = None
    if slot_to_seq_id is not None:
        ptm = types.SimpleNamespace(slot_to_seq_id=list(slot_to_seq_id))
    rebuild_called: list[tuple[int, ...]] = []

    def rebuild_page_table(global_ids: list[int]) -> None:
        rebuild_called.append(tuple(global_ids))

    return types.SimpleNamespace(
        is_initialized=is_initialized,
        _gpu_page_table_manager=ptm,
        rebuild_page_table=rebuild_page_table,
        _rebuild_called=rebuild_called,
    )


# ---------------------------------------------------------------------------
# Happy path — no pending load
# ---------------------------------------------------------------------------


class TestNoPendingLoad:
    def test_drains_kv_appends_and_returns_input(self) -> None:
        state = _make_state()
        legacy = FakeLegacyBackend()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        decode_uuids, batch = wait_pending(
            state, legacy, col,
            decode_uuids=["u1"], batch=[0],
            gpu_manager=_fake_gpu_manager(),
            pending_async_load_task=None,
            pending_load_uuids=[], pending_load_local=[], pending_load_global=[],
        )
        assert decode_uuids == ["u1"]
        assert batch == [0]
        call_names = [c[0] for c in legacy.calls]
        assert "wait_pending_kv_append_tasks" in call_names
        assert "finalize_async_load_minimal" not in call_names
        # No collective emitted on the happy path.
        assert col.calls == []


# ---------------------------------------------------------------------------
# Pending load branch
# ---------------------------------------------------------------------------


class _PassThroughAdapter(FakeLegacyBackend):
    """Fake adapter that returns the input ``(decode_uuids, batch)``
    from ``finalize_async_load_minimal`` so the rebuild path fires."""

    def finalize_async_load_minimal(self, *args: Any, **kwargs: Any) -> Any:
        self._record("finalize_async_load_minimal", *args, **kwargs)
        # args[4] = decode_uuids, args[5] = batch in wait_pending's call.
        return (args[4], args[5])


class TestPendingLoad:
    def test_waits_on_task_and_finalizes(self) -> None:
        state = _make_state()
        legacy = _PassThroughAdapter()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        task = _FakeAsyncTask()
        gpu = _fake_gpu_manager()

        wait_pending(
            state, legacy, col,
            decode_uuids=["existing"], batch=[0],
            gpu_manager=gpu,
            pending_async_load_task=task,
            pending_load_uuids=["loaded"],
            pending_load_local=[1],
            pending_load_global=[42],
        )
        assert task.waited is True
        # Exactly one barrier emitted (matches legacy line 6700).
        assert col.call_names() == ["barrier"]
        call_names = [c[0] for c in legacy.calls]
        # Order: wait_pending_kv_append_tasks → finalize_async_load_minimal
        # → rebuild_page_table_for_batch.
        assert call_names.index("wait_pending_kv_append_tasks") < \
               call_names.index("finalize_async_load_minimal") < \
               call_names.index("rebuild_page_table_for_batch")

    def test_no_async_task_still_fires_barrier_and_finalize(self) -> None:
        """Legacy treats task==None as 'no wait needed' but still calls
        finalize + barrier. That behaviour is preserved so eagerly
        pre-loaded sequences (no handle) still integrate correctly."""
        state = _make_state()
        legacy = _PassThroughAdapter()
        col = FakeCollectiveBackend(rank=0, world_size=1)

        wait_pending(
            state, legacy, col,
            decode_uuids=["existing"], batch=[0],
            gpu_manager=_fake_gpu_manager(),
            pending_async_load_task=None,
            pending_load_uuids=["loaded"],
            pending_load_local=[1],
            pending_load_global=[42],
        )
        assert col.call_names() == ["barrier"]
        assert "finalize_async_load_minimal" in [c[0] for c in legacy.calls]

    def test_slot_order_mismatch_rebuilds_page_table(self) -> None:
        state = _make_state()

        class _Adapter(FakeLegacyBackend):
            """Override global-id mapping so the slot-order repair path
            has a deterministic target to compare against."""

            def local_indices_to_global_seq_ids(self, batch: list[int]) -> list[int]:
                self._record("local_indices_to_global_seq_ids", batch)
                return [10 if i == 0 else 20 for i in batch]

            def finalize_async_load_minimal(self, *args: Any, **kwargs: Any) -> Any:
                # Preserve the decode set / batch sent in so the test
                # asserts on the page-table repair, not on load
                # integration.
                self._record("finalize_async_load_minimal", *args, **kwargs)
                return (args[4], args[5])  # (decode_uuids, batch)

        legacy = _Adapter()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        # Adapter reports global ids [10, 20] for batch [0, 1] but the
        # page-table manager's slot_to_seq_id disagrees.
        gpu = _fake_gpu_manager(slot_to_seq_id=[99, 99])

        wait_pending(
            state, legacy, col,
            decode_uuids=["existing", "loaded"], batch=[0, 1],
            gpu_manager=gpu,
            pending_async_load_task=_FakeAsyncTask(),
            pending_load_uuids=["loaded"],
            pending_load_local=[1],
            pending_load_global=[20],
        )
        # rebuild_page_table called with the adapter-reported global ids
        # to restore slot ordering parity.
        assert gpu._rebuild_called == [(10, 20)]

    def test_gpu_manager_uninitialized_skips_rebuild(self) -> None:
        state = _make_state()
        legacy = FakeLegacyBackend()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        gpu = _fake_gpu_manager(is_initialized=False)

        wait_pending(
            state, legacy, col,
            decode_uuids=["existing"], batch=[0],
            gpu_manager=gpu,
            pending_async_load_task=None,
            pending_load_uuids=["loaded"],
            pending_load_local=[1],
            pending_load_global=[42],
        )
        call_names = [c[0] for c in legacy.calls]
        assert "rebuild_page_table_for_batch" not in call_names
