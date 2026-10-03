"""Pool request IDs are scoped to a Batch while output keeps custom_id."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from batchgen.server.batch_scheduler import BatchScheduler, parse_batch_file
from batchgen.server.intake_pool import IntakePool


def _batch_file(custom_ids: list[str]) -> bytes:
    return "\n".join(
        (
            '{"custom_id": "' + custom_id + '", "method": "POST", '
            '"url": "/v1/completions", "body": '
            '{"model": "m", "prompt": "prompt", "max_tokens": 4}}'
        )
        for custom_id in custom_ids
    ).encode()


def test_parse_rejects_duplicate_custom_id_within_one_batch():
    ok, error, requests = parse_batch_file(_batch_file(["a", "b", "a"]))

    assert not ok
    assert requests == []
    assert error == "Line 3: duplicate custom_id 'a' (first on line 1)"


def test_pool_internal_request_ids_are_batch_scoped(tmp_path):
    ok, error, requests = parse_batch_file(_batch_file(["same"]))
    assert ok, error

    class CapturePool:
        def __init__(self):
            self.entries = []

        def register_batch(self, **kwargs):
            pass

        def submit_batch(self, batch_id, entries, priority):
            self.entries.extend(entries)
            return True

    async def run():
        scheduler = object.__new__(BatchScheduler)
        scheduler.storage = SimpleNamespace(output_dir=tmp_path)
        scheduler.server_args = SimpleNamespace(incremental_output_dir=None)
        scheduler._scheduling_pool = CapturePool()
        scheduler._intake_pool = IntakePool()
        scheduler._pool_request_meta = {}

        async def wait_and_finalize(*args):
            return None

        scheduler._wait_and_finalize_batch = wait_and_finalize
        batch = SimpleNamespace(batchgen_debug=None, max_context_length=None)
        await scheduler._process_batch_pool_mode(
            "batch-1", batch, requests, ["prompt"], [4], [{}]
        )
        await asyncio.sleep(0)
        first_id = scheduler._scheduling_pool.entries[0].request_id

        await scheduler._process_batch_pool_mode(
            "batch-2", batch, requests, ["prompt"], [4], [{}]
        )
        await asyncio.sleep(0)
        second_id = scheduler._scheduling_pool.entries[1].request_id

        assert first_id == "same@batch-1"
        assert second_id == "same@batch-2"
        assert scheduler._pool_request_meta["batch-1"][first_id]["custom_id"] == "same"
        assert scheduler._pool_request_meta["batch-2"][second_id]["custom_id"] == "same"

    asyncio.run(run())
