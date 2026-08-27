"""Regression coverage for persistent-pool batch failure finalization."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from batchgen.server.batch_scheduler import BatchScheduler
from batchgen.server.io_struct import BatchStatus


def test_pool_worker_failure_writes_terminal_batch_status():
    tracker = SimpleNamespace(is_complete=False, error={"message": "worker failed"})
    storage = SimpleNamespace(update_batch_status=Mock())
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

    storage.update_batch_status.assert_called_once_with(
        "batch-test",
        BatchStatus.FAILED,
        error={
            "code": "batch_failed",
            "message": str(tracker.error),
        },
    )
