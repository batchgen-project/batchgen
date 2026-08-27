"""Regression coverage for persistent-pool batch failure finalization."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from batchgen.server.batch_scheduler import BatchScheduler
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
