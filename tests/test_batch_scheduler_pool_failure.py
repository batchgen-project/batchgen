"""Regression coverage for persistent-pool batch failure finalization."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from batchgen.server.batch_scheduler import BatchScheduler
from batchgen.server.scheduling_pool import SchedulingPool
from batchgen.server.io_struct import (
    BatchEndpoint,
    BatchObject,
    BatchStatus,
    CompletionWindow,
)
from batchgen.server.storage import StorageManager


def test_pool_worker_failure_writes_terminal_batch_status(tmp_path):
    tracker = SimpleNamespace(is_complete=False, error={"message": "worker failed"})
    storage = StorageManager(tmp_path)
    storage.save_batch(BatchObject(
        id="batch-test",
        endpoint=BatchEndpoint.CHAT_COMPLETIONS,
        input_file_id="file-test",
        completion_window=CompletionWindow.ONE_DAY,
        status=BatchStatus.IN_PROGRESS,
        created_at=1,
        expires_at=2,
    ))
    scheduler = object.__new__(BatchScheduler)
    scheduler._scheduling_pool = SimpleNamespace(
        get_batch_tracker=lambda batch_id: tracker,
    )
    scheduler.storage = storage
    scheduler._batch_timeout = 60

    asyncio.run(
        scheduler._wait_and_finalize_batch(
            "batch-test",
            requests=[],
            prompts=[],
        )
    )

    persisted = storage.load_batch("batch-test")
    assert persisted is not None
    assert persisted.status == BatchStatus.FAILED
    assert persisted.error == str(tracker.error)


def test_capacity_snapshot_does_not_reset_active_scheduling_slots():
    """Free-page snapshots must not reinitialize an active fixed-capacity pool."""
    pool = SchedulingPool(capacity=4)
    pool.allocate_slot("request-1")
    pool.register_batch("batch-1", total_requests=1)

    class ResponseQueue:
        def __init__(self):
            self._results = iter([
                {
                    "type": "trajectory_pool_capacity",
                    "capacity": 3,
                    "free_pages": 255,
                    "active_count": 1,
                    "largest_free_extent_pages": 255,
                },
                {
                    "type": "completion",
                    "request_id": "request-1",
                    "batch_id": "batch-1",
                    "text": "ok",
                },
                {"type": "pool_shutdown"},
            ])

        def get(self, timeout):
            return next(self._results)

    scheduler = object.__new__(BatchScheduler)
    scheduler._stopped = SimpleNamespace(is_set=lambda: False)
    scheduler.worker = SimpleNamespace(response_queue=ResponseQueue())
    scheduler._scheduling_pool = pool
    scheduler._trajectory_pool_info = None
    scheduler.server_args = SimpleNamespace(incremental_output_dir=None)
    scheduler._fail_all_active_batches = lambda error: None

    asyncio.run(scheduler._pool_completion_listener())

    assert pool.capacity == 4
    assert pool.num_active_slots() == 0
    assert pool.num_free_slots() == 4
    assert pool.get_batch_tracker("batch-1").is_complete
    assert scheduler._trajectory_pool_info["free_pages"] == 255
