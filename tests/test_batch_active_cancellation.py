"""Regression coverage for Batch API cancellation in persistent pool mode."""

from __future__ import annotations

import asyncio
import queue
import sys
import types
from types import SimpleNamespace

import pytest


class _CoreEngineStub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (), {})
        setattr(self, name, value)
        return value


_core_engine_stub = _CoreEngineStub("batchgen.core_engine")
_core_engine_stub.__file__ = __file__
sys.modules.setdefault("batchgen.core_engine", _core_engine_stub)
_worker_manager_stub = types.ModuleType("batchgen.server.worker_manager")
_worker_manager_stub.WorkerManager = type("WorkerManager", (), {})
_worker_manager_stub.WorkerExitState = type("WorkerExitState", (), {})
sys.modules.setdefault("batchgen.server.worker_manager", _worker_manager_stub)

from batchgen.query_book import QueryBookEntry
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.server.batch_scheduler import BatchScheduler
from batchgen.server.intake_pool import IntakeEntry, IntakePool, Priority
from batchgen.server.io_struct import (
    BatchEndpoint,
    BatchObject,
    BatchStatus,
    CompletionWindow,
)
from batchgen.server.scheduling_pool import SchedulingPool
from batchgen.server.storage import StorageManager
from batchgen.worker import cancellation
from batchgen.worker.cancellation import (
    apply_deferred_batch_cancellations,
    cancel_batch_sequences,
    defer_batch_cancellation,
    pause_decode_for_cancellation,
    settle_pending_decode_load,
)


def _entry(request_id: str, batch_id: str) -> IntakeEntry:
    return IntakeEntry(request_id, batch_id, {}, Priority.NORMAL)


def _save_batch(storage: StorageManager, batch_id: str, status: BatchStatus) -> None:
    storage.save_batch(BatchObject(
        id=batch_id,
        endpoint=BatchEndpoint.CHAT_COMPLETIONS,
        input_file_id=f"file-{batch_id}",
        completion_window=CompletionWindow.ONE_DAY,
        status=status,
        created_at=1,
        expires_at=2,
    ))


def _scheduler(tmp_path, *, active: bool):
    storage = StorageManager(tmp_path)
    _save_batch(storage, "batch-a", BatchStatus.IN_PROGRESS)
    intake = IntakePool()
    intake.submit_batch("batch-a", [_entry("a-queued", "batch-a")])
    scheduling = SchedulingPool(4)
    scheduling.register_batch("batch-a", 2)
    if active:
        scheduling.allocate_slot("a-active")
    worker = SimpleNamespace(request_queue=queue.Queue())
    scheduler = object.__new__(BatchScheduler)
    scheduler.storage = storage
    scheduler.worker = worker
    scheduler._pool_mode = True
    scheduler._intake_pool = intake
    scheduler._scheduling_pool = scheduling
    scheduler._pool_request_meta = {
        "batch-a": {
            "a-active": {},
            "a-queued": {},
        }
    }
    scheduler._cancel_pending = {}
    return scheduler, storage, worker


def test_intake_cancel_removes_only_target_batch():
    intake = IntakePool()
    intake.submit_batch("batch-a", [_entry("a-1", "batch-a"), _entry("a-2", "batch-a")])
    intake.submit_batch("batch-b", [_entry("b-1", "batch-b")])

    assert [entry.request_id for entry in intake.cancel_batch("batch-a")] == ["a-1", "a-2"]
    assert [entry.request_id for entry in intake.drain(10)] == ["b-1"]
    assert intake.get_batch_info("batch-a") is None


def test_queued_only_cancel_finishes_without_worker_message(tmp_path):
    scheduler, storage, worker = _scheduler(tmp_path, active=False)
    cancelled = asyncio.run(scheduler.cancel_batch("batch-a"))

    assert cancelled.status == BatchStatus.CANCELLED
    assert scheduler._intake_pool.is_empty()
    assert scheduler._scheduling_pool.get_batch_tracker("batch-a") is None
    assert "batch-a" not in scheduler._pool_request_meta
    assert worker.request_queue.empty()


def test_in_progress_legacy_cancel_fails_loudly(tmp_path):
    scheduler, storage, _ = _scheduler(tmp_path, active=False)
    scheduler._pool_mode = False
    with pytest.raises(RuntimeError, match="requires persistent pool mode"):
        asyncio.run(scheduler.cancel_batch("batch-a"))
    assert storage.load_batch("batch-a").status == BatchStatus.IN_PROGRESS


def test_direct_cancel_rejects_terminal_batch_without_mutating_it(tmp_path):
    scheduler, storage, _ = _scheduler(tmp_path, active=False)
    storage.update_batch_status("batch-a", BatchStatus.COMPLETED)

    with pytest.raises(ValueError, match="cannot be cancelled"):
        asyncio.run(scheduler.cancel_batch("batch-a"))

    assert storage.load_batch("batch-a").status == BatchStatus.COMPLETED


def test_active_cancel_waits_for_worker_ack_and_cleans_slots(tmp_path):
    scheduler, storage, worker = _scheduler(tmp_path, active=True)
    cancelling = asyncio.run(scheduler.cancel_batch("batch-a"))

    assert cancelling.status == BatchStatus.CANCELLING
    assert cancelling.cancelling_at is not None
    assert worker.request_queue.get_nowait() == {"type": "cancel", "batch_id": "batch-a"}
    scheduler._handle_batch_cancelled({
        "type": "batch_cancelled",
        "batch_id": "batch-a",
        "request_ids": ["a-active"],
    })
    cancelled = storage.load_batch("batch-a")
    assert cancelled.status == BatchStatus.CANCELLED
    assert cancelled.cancelled_at >= cancelling.cancelling_at
    assert scheduler._scheduling_pool.num_active_slots() == 0
    assert scheduler._scheduling_pool.get_batch_tracker("batch-a") is None
    assert "batch-a" not in scheduler._pool_request_meta


def test_completion_during_cancelling_is_discarded_but_frees_slot(tmp_path):
    scheduler, storage, worker = _scheduler(tmp_path, active=True)
    asyncio.run(scheduler.cancel_batch("batch-a"))
    worker.request_queue.get_nowait()
    writes = []
    scheduler._write_pool_completion = lambda *args: writes.append(args)

    scheduler._handle_pool_completion({
        "type": "completion",
        "batch_id": "batch-a",
        "request_id": "a-active",
    })

    assert writes == []
    assert scheduler._scheduling_pool.num_active_slots() == 0
    tracker = scheduler._scheduling_pool.get_batch_tracker("batch-a")
    assert tracker.completed_requests == 0


def test_completion_before_ack_drains_pending_and_finishes_cancel(tmp_path):
    scheduler, storage, worker = _scheduler(tmp_path, active=True)
    asyncio.run(scheduler.cancel_batch("batch-a"))
    worker.request_queue.get_nowait()
    writes = []
    scheduler._write_pool_completion = lambda *args: writes.append(args)

    scheduler._handle_pool_completion({
        "type": "completion",
        "batch_id": "batch-a",
        "request_id": "a-active",
    })
    scheduler._handle_batch_cancelled({
        "type": "batch_cancelled",
        "batch_id": "batch-a",
        "request_ids": [],
    })

    assert writes == []
    assert storage.load_batch("batch-a").status == BatchStatus.CANCELLED
    assert scheduler._scheduling_pool.num_active_slots() == 0


def test_incomplete_worker_ack_fails_loudly_without_reusing_slot(tmp_path):
    scheduler, storage, worker = _scheduler(tmp_path, active=True)
    asyncio.run(scheduler.cancel_batch("batch-a"))
    worker.request_queue.get_nowait()

    scheduler._handle_batch_cancelled({
        "type": "batch_cancelled",
        "batch_id": "batch-a",
        "request_ids": [],
    })

    failed = storage.load_batch("batch-a")
    assert failed.status == BatchStatus.FAILED
    assert "omitted active requests" in failed.error
    assert scheduler._scheduling_pool.num_active_slots() == 1
    tracker = scheduler._scheduling_pool.get_batch_tracker("batch-a")
    assert "omitted active requests" in tracker.error

    scheduler._handle_batch_cancelled({
        "type": "batch_cancelled",
        "batch_id": "batch-a",
        "request_ids": ["a-active"],
    })
    assert storage.load_batch("batch-a").status == BatchStatus.FAILED


def test_cancelled_waiter_never_finalizes_output(tmp_path):
    scheduler, storage, _ = _scheduler(tmp_path, active=False)
    storage.update_batch_status("batch-a", BatchStatus.CANCELLED)
    scheduler._batch_timeout = 1
    finalized = []
    scheduler._finalize_batch_output = lambda *args: finalized.append(args)

    asyncio.run(scheduler._wait_and_finalize_batch("batch-a", [], []))
    assert finalized == []
    assert storage.load_batch("batch-a").status == BatchStatus.CANCELLED


class _BufferPool:
    def __init__(self):
        self.freed = []

    def free_slot(self, slot):
        self.freed.append(slot)


def _worker_with_sequence(*, rank: int):
    worker = SimpleNamespace()
    worker.rank = rank
    worker._response_queue = queue.Queue()
    worker._deferred_pool_cancellations = []
    worker.global_batch = SequenceBatch()
    worker.query_book = {0: QueryBookEntry()}
    worker._uuid_to_local_map = {"a": 0}
    worker._local_to_uuid_map = {0: "a"}
    worker._free_local_indices = set()
    worker._buffer_pool = _BufferPool()
    worker._sequences_with_gpu_kv = set()
    worker.gpu_paged_kv_cache_manager = None
    worker.core_engine = SimpleNamespace(host_paged_kv_worker_view=None)
    seq = SequenceEntry("a", 0, 1, 8)
    seq.batch_id = "batch-a"
    seq._buffer_slot = 0
    worker.global_batch.add_sequence(seq)
    return worker


def test_worker_cancel_removes_only_target_sequences_and_acks():
    worker = SimpleNamespace()
    worker.rank = 0
    worker._response_queue = queue.Queue()
    worker.global_batch = SequenceBatch()
    worker.query_book = {0: QueryBookEntry(), 1: QueryBookEntry()}
    worker._uuid_to_local_map = {"a": 0, "b": 1}
    worker._local_to_uuid_map = {0: "a", 1: "b"}
    worker._free_local_indices = set()
    worker._buffer_pool = _BufferPool()
    worker._sequences_with_gpu_kv = set()
    worker.gpu_paged_kv_cache_manager = None
    worker.core_engine = SimpleNamespace(host_paged_kv_worker_view=None)
    for index, (request_id, batch_id) in enumerate((("a", "batch-a"), ("b", "batch-b"))):
        seq = SequenceEntry(request_id, index, 1, 8)
        seq.batch_id = batch_id
        seq._buffer_slot = index
        worker.global_batch.add_sequence(seq)

    cancel_batch_sequences(worker, "batch-a")

    assert worker.global_batch.get_sequence("a") is None
    assert worker.global_batch.get_sequence("b") is not None
    assert worker._uuid_to_local_map == {"b": 1}
    assert worker._buffer_pool.freed == [0]
    assert worker._response_queue.get_nowait() == {
        "type": "batch_cancelled",
        "batch_id": "batch-a",
        "request_ids": ["a"],
    }


def test_worker_cancel_releases_live_resources_before_slots():
    worker = SimpleNamespace()
    worker.rank = 0
    worker._response_queue = queue.Queue()
    worker.global_batch = SequenceBatch()
    worker.query_book = {0: QueryBookEntry()}
    worker._uuid_to_local_map = {"a": 0}
    worker._local_to_uuid_map = {0: "a"}
    worker._free_local_indices = set()
    worker._buffer_pool = _BufferPool()
    worker._sequences_with_gpu_kv = {"a"}
    calls = []
    worker._get_local_indices_for_uuids = lambda values: [0]
    worker._release_gpu_kv_pages = lambda values: calls.append(("gpu", values))
    worker._release_kda_state_slots = lambda values: calls.append(("kda", values))
    worker._release_host_kv_pages_for_batch = lambda values: calls.append(("host", values))
    worker.core_engine = SimpleNamespace(host_paged_kv_worker_view=object())

    seq = SequenceEntry("a", 7, 1, 8)
    seq.batch_id = "batch-a"
    seq.status = SequenceStatus.IN_DECODE
    seq.gpu_pages_allocated = 2
    seq.host_pages_allocated = 3
    seq._buffer_slot = 0
    worker.global_batch.add_sequence(seq)

    cancel_batch_sequences(worker, "batch-a")

    assert calls == [("gpu", [0]), ("host", ["a"])]
    assert worker._buffer_pool.freed == [0]
    assert worker.global_batch.get_sequence("a") is None


def test_worker_cancel_fails_before_partial_cleanup_without_host_view():
    worker = _worker_with_sequence(rank=0)
    seq = worker.global_batch.get_sequence("a")
    seq.host_pages_allocated = 1

    with pytest.raises(RuntimeError, match="Host KV"):
        cancel_batch_sequences(worker, "batch-a")

    assert worker.global_batch.get_sequence("a") is seq
    assert worker._buffer_pool.freed == []
    assert worker._response_queue.empty()


def test_worker_cancel_propagates_missing_host_view_to_non_owner(monkeypatch):
    worker = _worker_with_sequence(rank=1)
    worker.world_size = 2
    worker.torch_device = "cpu"
    worker.global_batch.get_sequence("a").batch_id = "batch-b"
    monkeypatch.setattr(
        cancellation.dist,
        "all_reduce",
        lambda tensor, op: tensor.fill_(1),
    )

    with pytest.raises(RuntimeError, match="Host KV"):
        cancel_batch_sequences(worker, "batch-a")

    assert worker.global_batch.get_sequence("a") is not None


def test_deferred_cancel_keeps_live_sequence_until_safe_boundary():
    worker = _worker_with_sequence(rank=0)
    message = {"type": "cancel", "batch_id": "batch-a"}

    defer_batch_cancellation(worker, message)
    assert worker.global_batch.get_sequence("a") is not None
    assert worker._response_queue.empty()

    assert apply_deferred_batch_cancellations(worker)
    assert worker.global_batch.get_sequence("a") is None
    assert worker._response_queue.get_nowait()["type"] == "batch_cancelled"


def test_nonzero_rank_cleans_without_sending_ack():
    worker = _worker_with_sequence(rank=1)

    cancel_batch_sequences(worker, "batch-a")

    assert worker.global_batch.get_sequence("a") is None
    assert worker._response_queue.empty()


def test_pending_decode_load_settles_before_cancel_boundary(monkeypatch):
    calls = []
    task = SimpleNamespace(wait=lambda: calls.append("wait"))
    worker = SimpleNamespace(
        torch_device="cuda:0",
        _finalize_async_load_minimal=lambda *args: (
            calls.append(("finalize", args)) or (["active", "loaded"], [3, 4])
        ),
    )
    monkeypatch.setattr(
        cancellation.torch.cuda, "synchronize", lambda device: calls.append(("sync", device))
    )
    monkeypatch.setattr(cancellation.dist, "barrier", lambda: calls.append("barrier"))

    decode_uuids, batch = settle_pending_decode_load(
        worker,
        pending_async_task=task,
        pending_load_uuids=["loaded"],
        pending_load_local=[4],
        pending_load_global=[9],
        decode_uuids=["active"],
        batch=[3],
        gpu_manager="manager",
    )

    assert calls[:3] == ["wait", ("sync", "cuda:0"), "barrier"]
    assert calls[3][0] == "finalize"
    assert decode_uuids == ["active", "loaded"]
    assert batch == [3, 4]


def test_surviving_decode_sequences_are_put_on_hold_before_cancel():
    calls = []
    worker = SimpleNamespace(
        _wait_pending_kv_append_tasks=lambda **kwargs: calls.append(("wait", kwargs)),
        _put_sequences_on_hold=lambda uuids: calls.append(("hold", uuids)),
    )

    pause_decode_for_cancellation(worker, ["cancelled", "survivor"])

    assert calls == [
        ("wait", {"sync_distributed_errors": True}),
        ("hold", ["cancelled", "survivor"]),
    ]
