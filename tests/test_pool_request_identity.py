"""Pool request IDs are scoped to a Batch while output keeps custom_id."""

from __future__ import annotations

import asyncio
import json
from queue import Queue
from types import SimpleNamespace

from batchgen.server.batch_scheduler import BatchScheduler, parse_batch_file
from batchgen.server.intake_pool import IntakePool
from batchgen.server.scheduling_pool import SchedulingPool


def _batch_file(custom_ids: list[str]) -> bytes:
    return "\n".join(
        json.dumps({
            "custom_id": custom_id, "method": "POST", "url": "/v1/completions",
            "body": {"model": "m", "prompt": "prompt", "max_tokens": 4},
        })
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

    async def run():
        scheduler = object.__new__(BatchScheduler)
        scheduler.storage = SimpleNamespace(output_dir=tmp_path)
        scheduler.server_args = SimpleNamespace(incremental_output_dir=str(tmp_path))
        scheduler._scheduling_pool = SchedulingPool(capacity=2)
        scheduler._intake_pool = IntakePool()
        scheduler._pool_request_meta = {}
        scheduler._stopped = asyncio.Event()
        scheduler._pool_initialized = True

        async def wait_and_finalize(*args):
            return None

        scheduler._wait_and_finalize_batch = wait_and_finalize
        batch = SimpleNamespace(batchgen_debug=None, max_context_length=None)
        await scheduler._process_batch_pool_mode(
            "batch-1", batch, requests, ["prompt"], [4], [{}]
        )
        await scheduler._process_batch_pool_mode(
            "batch-2", batch, requests, ["prompt"], [4], [{}]
        )
        admissions = []

        def capture_admission(message):
            admissions.extend(message["entries"])
            scheduler._stopped.set()

        responses = Queue()
        scheduler.worker = SimpleNamespace(
            request_queue=SimpleNamespace(put=capture_admission),
            response_queue=responses,
        )
        await scheduler._drain_intake_to_worker()
        assert len(admissions) == 2
        assert scheduler._scheduling_pool.num_active_slots() == 2
        first_id, second_id = [entry["request_id"] for entry in admissions]

        assert first_id == "same@batch-1"
        assert second_id == "same@batch-2"
        assert scheduler._pool_request_meta["batch-1"][first_id]["custom_id"] == "same"
        assert scheduler._pool_request_meta["batch-2"][second_id]["custom_id"] == "same"

        for entry in admissions:
            responses.put({
                "type": "completion", "request_id": entry["request_id"],
                "batch_id": entry["batch_id"], "text": entry["batch_id"],
                "prompt_length": 1, "decoded_length": 1,
            })
        responses.put({"type": "pool_shutdown"})
        scheduler._stopped.clear()
        await scheduler._pool_completion_listener()
        assert scheduler._scheduling_pool.num_free_slots() == 2
        for batch_id in ("batch-1", "batch-2"):
            assert scheduler._scheduling_pool.get_batch_tracker(batch_id).is_complete
            output = json.loads((tmp_path / f"{batch_id}.jsonl").read_text())
            assert output["custom_id"] == "same"
            assert output["response"]["body"]["choices"][0]["text"] == batch_id

    asyncio.run(run())


def test_intake_drain_failure_fails_batches_and_notifies_worker():
    scheduler = object.__new__(BatchScheduler)
    failures = []
    fatals = []

    async def failing_drain():
        raise RuntimeError("admission failed")

    scheduler._drain_intake_loop = failing_drain
    scheduler._fail_all_active_batches = failures.append
    scheduler.worker = SimpleNamespace(report_worker_fatal=fatals.append)
    asyncio.run(scheduler._drain_intake_to_worker())
    assert failures == fatals == [
        "Server intake drain failed: RuntimeError: admission failed"
    ]
