"""Regression: concurrent pool-mode Batches may reuse the same custom_id.

custom_id is unique only within one Batch, but the pool keyed in-flight
requests by the bare custom_id across batches (SchedulingPool slots, the
worker's sequence uuid, completion routing). Two Batches that both used
miss-000..miss-003 raised `request_id 'miss-000' already allocated` inside the
intake drain task; the task died silently and both Batches stayed in_progress.

The server package imports the JIT engine, so batch_scheduler is loaded by
path with the real io_struct / IntakePool / SchedulingPool and stubs for the
engine-backed modules.
"""
import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVER = Path(__file__).resolve().parents[1] / "batchgen" / "server"


def _load_scheduler_module():
    names = ["batchgen.server"] + [
        f"batchgen.server.{m}" for m in (
            "io_struct", "intake_pool", "scheduling_pool", "server_args",
            "storage", "worker_manager", "batch_scheduler")]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        package = types.ModuleType("batchgen.server")
        package.__path__ = [str(SERVER)]
        sys.modules["batchgen.server"] = package
        for module, attr in (("server_args", "ServerArgs"), ("storage", "StorageManager"),
                             ("worker_manager", "WorkerManager")):
            stub = types.ModuleType(f"batchgen.server.{module}")
            setattr(stub, attr, object)
            sys.modules[stub.__name__] = stub
        loaded = None
        for module in ("io_struct", "intake_pool", "scheduling_pool", "batch_scheduler"):
            spec = importlib.util.spec_from_file_location(
                f"batchgen.server.{module}", SERVER / f"{module}.py")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
        return loaded
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


bs = _load_scheduler_module()
CUSTOM_IDS = [f"miss-{i:03d}" for i in range(4)]


def _batch_file(custom_ids):
    return "\n".join(json.dumps({
        "custom_id": cid, "method": "POST", "url": "/v1/completions",
        "body": {"model": "m", "prompt": f"prompt {cid}", "max_tokens": 8},
    }) for cid in custom_ids).encode()


class _Worker:
    def __init__(self, trace, stopped, completions=()):
        self.trace = trace
        self.request_queue = SimpleNamespace(put=self._put)
        self.response_queue = SimpleNamespace(get=self._get)
        self._stopped = stopped
        self._completions = list(completions)

    def _put(self, msg):
        self.trace.append(("admit", msg))
        self._stopped.set()

    def _get(self, timeout):
        return self._completions.pop(0) if self._completions else None

    def report_worker_fatal(self, reason):
        self.trace.append(("fatal", reason))


def _scheduler(tmp_path, trace):
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)
    scheduler._intake_pool = bs.IntakePool()
    scheduler._scheduling_pool = bs.SchedulingPool(capacity=64)
    scheduler._pool_request_meta = {}
    scheduler._pool_initialized = True
    scheduler.server_args = SimpleNamespace(incremental_output_dir=str(tmp_path))
    scheduler.storage = SimpleNamespace(
        output_dir=tmp_path,
        update_batch_status=lambda batch_id, status, **kw: trace.append(
            ("status", batch_id, status, kw.get("error"))),
    )

    async def no_finalizer(*args):
        pass

    scheduler._wait_and_finalize_batch = no_finalizer
    return scheduler


async def _submit(scheduler, batch_id, custom_ids):
    ok, error, requests = bs.parse_batch_file(_batch_file(custom_ids))
    assert ok, error
    await scheduler._process_batch_pool_mode(
        batch_id, SimpleNamespace(batchgen_debug=None, max_context_length=None),
        requests, [f"prompt {c}" for c in custom_ids], [8] * len(custom_ids), None, {})


def test_parse_rejects_duplicate_custom_id_within_a_batch():
    ok, error, _ = bs.parse_batch_file(_batch_file(["a", "b", "a"]))
    assert not ok
    assert error == "Line 3: duplicate custom_id 'a' (first on line 1)"
    assert bs.parse_batch_file(_batch_file(["a", "b"]))[0]
    assert bs.parse_batch_file(_batch_file(["", ""]))[0]  # index-based fallback


def test_batches_sharing_custom_ids_are_admitted_and_answered_separately(tmp_path):
    trace = []

    async def run():
        scheduler = _scheduler(tmp_path, trace)
        scheduler._stopped = asyncio.Event()
        scheduler.worker = _Worker(trace, scheduler._stopped)
        await _submit(scheduler, "batch_a", CUSTOM_IDS)
        await _submit(scheduler, "batch_b", CUSTOM_IDS)
        await scheduler._drain_intake_to_worker()

        (_, admission), = [t for t in trace if t[0] == "admit"]
        request_ids = [e["request_id"] for e in admission["entries"]]
        assert sorted(request_ids) == sorted(
            f"{cid}@{bid}" for bid in ("batch_a", "batch_b") for cid in CUSTOM_IDS)
        assert scheduler._scheduling_pool.num_active_slots() == 8

        scheduler._stopped = asyncio.Event()
        scheduler.worker = _Worker(trace, scheduler._stopped, completions=[
            {"type": "completion", "request_id": e["request_id"], "batch_id": e["batch_id"],
             "text": f"answer {e['request_id']}"} for e in admission["entries"]])
        await scheduler._pool_completion_listener()
        return scheduler

    scheduler = asyncio.run(run())
    assert not [t for t in trace if t[0] in ("status", "fatal")]
    assert scheduler._scheduling_pool.num_active_slots() == 0
    for batch_id in ("batch_a", "batch_b"):
        assert scheduler._scheduling_pool.get_batch_tracker(batch_id).is_complete
        rows = [json.loads(line) for line in (tmp_path / f"{batch_id}.jsonl").read_text().splitlines()]
        assert sorted(r["custom_id"] for r in rows) == CUSTOM_IDS
        for row in rows:
            text = row["response"]["body"]["choices"][0]["text"]
            assert text == f"answer {row['custom_id']}@{batch_id}"


def test_drain_failure_fails_active_batches_then_stops_the_server(tmp_path, monkeypatch):
    trace = []

    def broken_allocate(request_id):
        raise ValueError(f"request_id {request_id!r} already allocated")

    async def run():
        scheduler = _scheduler(tmp_path, trace)
        scheduler._stopped = asyncio.Event()
        scheduler.worker = _Worker(trace, scheduler._stopped)
        await _submit(scheduler, "batch_a", CUSTOM_IDS)
        monkeypatch.setattr(scheduler._scheduling_pool, "allocate_slot", broken_allocate)
        await scheduler._drain_intake_to_worker()

    asyncio.run(run())
    assert [t[0] for t in trace] == ["status", "fatal"]
    _, batch_id, status, error = trace[0]
    assert (batch_id, status) == ("batch_a", bs.BatchStatus.FAILED)
    assert error["message"] == trace[1][1]
    assert trace[1][1].startswith("Server intake drain failed: ValueError: request_id")


def test_drain_cancellation_is_not_a_failure(tmp_path):
    trace = []

    async def run():
        scheduler = _scheduler(tmp_path, trace)
        scheduler._stopped = asyncio.Event()
        scheduler.worker = _Worker(trace, scheduler._stopped)
        task = asyncio.ensure_future(scheduler._drain_intake_to_worker())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert trace == []
