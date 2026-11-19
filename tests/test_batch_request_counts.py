"""The Batch object carries OpenAI-compatible request_counts.

``request_counts = {total, completed, failed}``: ``total`` is fixed when the
input file is validated, every request is counted once (``completed`` when
its output item carries a response, ``failed`` when it carries an error), so
``completed + failed <= total`` always, with equality once every request has
a result. A client reads the counts on the batch object instead of parsing
the output file.

batch_scheduler and storage are loaded by path with the real io_struct /
IntakePool / SchedulingPool / StorageManager and stubs for the engine-backed
modules; the HTTP handlers are compiled from http_server.py.
"""
import ast
import asyncio
import copy
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "batchgen" / "server"
MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
MAX_MODEL_LEN = 32
LONG = "x" * MAX_MODEL_LEN  # at or above max_model_len: rejected at admission


def _load_server_modules():
    names = ["batchgen", "batchgen.server"] + [
        f"batchgen.server.{m}" for m in (
            "io_struct", "intake_pool", "scheduling_pool", "server_args",
            "storage", "worker_manager", "batch_scheduler")]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name, path in (("batchgen", ROOT / "batchgen"), ("batchgen.server", SERVER)):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        for module, attr in (("server_args", "ServerArgs"),
                             ("worker_manager", "WorkerManager")):
            stub = types.ModuleType(f"batchgen.server.{module}")
            setattr(stub, attr, object)
            sys.modules[stub.__name__] = stub
        loaded = {}
        for module in ("io_struct", "intake_pool", "scheduling_pool", "storage",
                       "batch_scheduler"):
            spec = importlib.util.spec_from_file_location(
                f"batchgen.server.{module}", SERVER / f"{module}.py")
            loaded[module] = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded[module]
            spec.loader.exec_module(loaded[module])
        return SimpleNamespace(**loaded)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


mods = _load_server_modules()
io = mods.io_struct
bs = mods.batch_scheduler
StorageManager = mods.storage.StorageManager


class _HTTPException(Exception):
    def __init__(self, status_code, detail=None, headers=None):
        super().__init__(detail)
        self.status_code = status_code


def _batch_handlers(app=None):
    """The real GET/list/cancel batch handlers of http_server.py.

    With ``app`` (a FastAPI app) they keep their route decorators; without
    one they are plain coroutines taking a request-shaped object.
    """
    tree = ast.parse((SERVER / "http_server.py").read_text())
    wanted = {"retrieve_batch", "list_batches", "cancel_batch"}
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name in wanted:
            node = copy.deepcopy(node)
            if app is None:
                node.decorator_list = []
            nodes.append(node)
    assert {node.name for node in nodes} == wanted
    scope = dict(
        app=app, Optional=Optional, time=time, Request=object,
        HTTPException=_HTTPException, BatchStatus=io.BatchStatus,
        BatchObject=io.BatchObject, ListBatchesRequest=io.ListBatchesRequest,
        ListBatchesResponse=io.ListBatchesResponse,
    )
    if app is not None:
        from fastapi import HTTPException, Request
        scope.update(HTTPException=HTTPException, Request=Request)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "http_server.py", "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in wanted})


def _wire(response_model, value):
    """What FastAPI sends for ``value`` under ``response_model``."""
    return response_model.model_validate(value).model_dump(mode="json")


class _Tokenizer:
    """Encodes completion prompt text one id per character."""

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[ord(ch) for ch in text] for text in texts]}


class _Server:
    """A scheduler with a real store, fed completions through its listener."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "store"
        self.incremental = tmp_path / "incremental"
        self.storage = StorageManager(self.root)
        self.scheduler = bs.BatchScheduler(self.storage, None, SimpleNamespace(
            model=MODEL, max_intake_capacity=64,
            incremental_output_dir=str(self.incremental),
            no_incremental_save=False, parse_thinking=False, parse_tool_call=False,
        ))
        self.scheduler.max_model_len = MAX_MODEL_LEN
        self.scheduler._get_tokenizer = lambda model: _Tokenizer()
        self.handlers = _batch_handlers()
        self.request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            storage=self.storage, scheduler=self.scheduler)))

    def create(self, batch_id, prompts):
        """Upload ``{custom_id: prompt}`` and persist a validating batch."""
        input_id = f"file-{batch_id}"
        (self.storage.files_dir / input_id).write_text("\n".join(json.dumps({
            "custom_id": custom_id, "method": "POST", "url": "/v1/completions",
            "body": {"model": MODEL, "prompt": prompt, "max_tokens": 4},
        }) for custom_id, prompt in prompts.items()))
        self.storage.save_metadata(input_id, {"id": input_id, "purpose": "batch"})
        self.storage.save_batch(io.build_batch_object_from_create_request(
            batch_id=batch_id,
            body=io.CreateBatchRequest(input_file_id=input_id),
            created_at=1, expires_at=2,
        ))

    async def start(self):
        self.scheduler._loop = asyncio.get_running_loop()
        self.scheduler._completion_inbox = asyncio.Queue()
        self.scheduler._ensure_completion_listener()

    async def stop(self):
        self.scheduler._stopped.set()
        await asyncio.wait_for(self.scheduler._completion_listener_task, timeout=5.0)

    async def complete(self, batch_id, custom_id, error=None):
        item = {"type": "completion", "request_id": f"{custom_id}@{batch_id}",
                "batch_id": batch_id}
        if error:
            item["error"] = {"code": error, "message": f"{custom_id} failed",
                             "status_code": 400}
        else:
            item.update(text=f"answer {custom_id}", prompt_length=2,
                        decoded_length=1, finish_reason="stop")
        inbox = self.scheduler._completion_inbox
        inbox.put_nowait((self.scheduler._pump_generation, item))
        while not inbox.empty():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.01)

    async def until_status(self, batch_id, status, timeout=5.0):
        deadline = time.monotonic() + timeout
        while self.storage.load_batch(batch_id).status != status:
            assert time.monotonic() < deadline, self.storage.load_batch(batch_id)
            await asyncio.sleep(0.05)

    async def get(self, batch_id):
        return _wire(io.BatchObject,
                     await self.handlers.retrieve_batch(self.request, batch_id))

    async def list(self):
        return _wire(io.ListBatchesResponse,
                     await self.handlers.list_batches(self.request, None, 20))

    async def cancel(self, batch_id):
        return _wire(io.BatchObject,
                     await self.handlers.cancel_batch(self.request, batch_id))

    def items(self, batch_id):
        path = self.incremental / f"{batch_id}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def output_items(self, batch_id):
        output_id = self.storage.load_batch(batch_id).output_file_id
        text = (self.storage.files_dir / output_id).read_text()
        return [json.loads(line) for line in text.splitlines()]


def counts(total, completed, failed):
    return {"total": total, "completed": completed, "failed": failed}


def test_all_requests_succeed(tmp_path):
    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("b", {"a": "p1", "b": "p2", "c": "p3"})
        assert (await server.get("b"))["request_counts"] == counts(0, 0, 0)
        await server.scheduler._process_batch("b")
        assert (await server.get("b"))["request_counts"] == counts(3, 0, 0)
        for custom_id in ("a", "b", "c"):
            await server.complete("b", custom_id)
        await server.until_status("b", io.BatchStatus.COMPLETED)
        await server.stop()
        return server

    server = asyncio.run(run())
    persisted = server.storage.load_batch("b")
    assert persisted.request_counts.model_dump() == counts(3, 3, 0)
    assert [item["error"] for item in server.output_items("b")] == [None] * 3
    wire = asyncio.run(server.get("b"))
    assert wire["status"] == "completed"
    assert wire["request_counts"] == counts(3, 3, 0)


def test_mixed_success_and_failure_counts_live_and_final(tmp_path):
    """A request rejected at admission and one the worker fails are both
    failed; the live counts are served while the batch runs."""

    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("b", {"ok-1": "p1", "long": LONG, "worker-err": "p3", "ok-2": "p4"})
        await server.scheduler._process_batch("b")
        assert (await server.get("b"))["request_counts"] == counts(4, 0, 1)
        await server.complete("b", "ok-1")
        await server.complete("b", "worker-err", error="context_length_exceeded")
        live = await server.get("b")
        assert live["status"] == "in_progress"
        assert live["request_counts"] == counts(4, 1, 2)
        (listed,) = (await server.list())["data"]
        assert listed["request_counts"] == counts(4, 1, 2)
        # Only the validated total is persisted while the batch runs.
        assert server.storage.load_batch("b").request_counts.model_dump() == counts(4, 0, 0)
        await server.complete("b", "ok-2")
        await server.until_status("b", io.BatchStatus.COMPLETED)
        await server.stop()
        return server

    server = asyncio.run(run())
    assert server.storage.load_batch("b").request_counts.model_dump() == counts(4, 2, 2)
    items = server.output_items("b")
    failed = sorted(item["custom_id"] for item in items if item["error"] is not None)
    assert failed == ["long", "worker-err"]
    assert len(items) == 4
    assert asyncio.run(server.get("b"))["request_counts"] == counts(4, 2, 2)


def test_every_request_rejected_at_admission(tmp_path):
    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("b", {"long-1": LONG, "long-2": LONG + "y"})
        await server.scheduler._process_batch("b")
        await server.until_status("b", io.BatchStatus.COMPLETED)
        await server.stop()
        return server

    server = asyncio.run(run())
    assert server.storage.load_batch("b").request_counts.model_dump() == counts(2, 0, 2)
    assert all(item["error"]["code"] == "invalid_prompt"
               for item in server.output_items("b"))


def test_a_duplicate_result_is_counted_and_written_once(tmp_path):
    """A result delivered twice for the same request does not drift the
    counts from the output file: neither is counted or written again."""

    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("b", {"a": "p1", "b": "p2"})
        await server.scheduler._process_batch("b")
        await server.complete("b", "a")
        await server.complete("b", "a", error="worker_error")
        assert (await server.get("b"))["request_counts"] == counts(2, 1, 0)
        assert [item["custom_id"] for item in server.items("b")] == ["a"]
        await server.complete("b", "b")
        await server.until_status("b", io.BatchStatus.COMPLETED)
        await server.stop()
        return server

    server = asyncio.run(run())
    assert server.storage.load_batch("b").request_counts.model_dump() == counts(2, 2, 0)
    assert len(server.output_items("b")) == 2


def test_cancel_reports_the_counts_reached(tmp_path):
    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("queued", {"a": "p1"})
        cancelled = await server.cancel("queued")
        assert (cancelled["status"], cancelled["request_counts"]) == (
            "cancelled", counts(0, 0, 0))
        await server.scheduler._process_batch("queued")  # skipped: never validated
        assert server.storage.load_batch("queued").request_counts.model_dump() == counts(0, 0, 0)

        server.create("running", {"a": "p1", "b": "p2", "c": "p3"})
        await server.scheduler._process_batch("running")
        await server.complete("running", "a")
        cancelled = await server.cancel("running")
        assert cancelled["status"] == "cancelled"
        assert cancelled["request_counts"] == counts(3, 1, 0)
        await server.stop()

    asyncio.run(run())


def test_a_worker_fatal_persists_the_counts_reached(tmp_path):
    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("b", {"a": "p1", "long": LONG, "c": "p3"})
        await server.scheduler._process_batch("b")
        await server.complete("b", "a")
        server.scheduler._fail_all_active_batches("rank 1 died")
        # The late completion of a failed batch is dropped, not counted.
        await server.complete("b", "c")
        await server.stop()
        return server

    server = asyncio.run(run())
    persisted = server.storage.load_batch("b")
    assert persisted.status == io.BatchStatus.FAILED
    assert persisted.request_counts.model_dump() == counts(3, 1, 1)
    assert asyncio.run(server.get("b"))["request_counts"] == counts(3, 1, 1)


def test_a_batch_failing_validation_has_zero_counts(tmp_path):
    async def run():
        server = _Server(tmp_path)
        server.create("b", {"a": "p1"})
        (server.storage.files_dir / "file-b").write_text("not json")
        await server.scheduler._process_batch("b")
        return server

    server = asyncio.run(run())
    persisted = server.storage.load_batch("b")
    assert persisted.status == io.BatchStatus.FAILED
    assert persisted.request_counts.model_dump() == counts(0, 0, 0)


def test_restart_mid_batch_rebuilds_the_counts_from_the_output_items(tmp_path):
    """The previous process recorded 1 success and 2 failures of 4 before it
    died; the restarted server persists exactly those counts with FAILED.
    A line cut short by the death is not a recorded result."""

    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("b", {"ok": "p1", "long": LONG, "err": "p3", "never": "p4"})
        await server.scheduler._process_batch("b")
        await server.complete("b", "ok")
        await server.complete("b", "err", error="worker_error")
        await server.stop()
        return server

    server = asyncio.run(run())
    with (server.incremental / "b.jsonl").open("a") as handle:
        handle.write('{"id": "batch_req_cut", "custom_id": "nev')

    restarted = StorageManager(server.root)
    assert restarted.fail_orphaned_batches(str(server.incremental)) == ["b"]
    persisted = restarted.load_batch("b")
    assert persisted.status == io.BatchStatus.FAILED
    assert persisted.request_counts.model_dump() == counts(4, 1, 2)
    # A second recovery pass finds nothing active and changes nothing.
    assert restarted.fail_orphaned_batches(str(server.incremental)) == []
    assert restarted.load_batch("b").request_counts.model_dump() == counts(4, 1, 2)


def test_a_finished_batch_reloads_with_its_final_counts(tmp_path):
    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("b", {"ok": "p1", "long": LONG})
        await server.scheduler._process_batch("b")
        await server.complete("b", "ok")
        await server.until_status("b", io.BatchStatus.COMPLETED)
        await server.stop()
        return server

    server = asyncio.run(run())
    restarted = StorageManager(server.root)
    assert restarted.fail_orphaned_batches(str(server.incremental)) == []
    reloaded = restarted.load_batch("b")
    assert reloaded.status == io.BatchStatus.COMPLETED
    assert reloaded.request_counts.model_dump() == counts(2, 1, 1)


def test_a_batch_persisted_without_request_counts_still_loads(tmp_path):
    storage = StorageManager(tmp_path)
    record = {
        "id": "batch-old", "object": "batch", "endpoint": "/v1/completions",
        "input_file_id": "file-old", "completion_window": "24h",
        "status": "completed", "created_at": 1, "expires_at": 2,
    }
    (storage.batches_dir / "batch-old.json").write_text(json.dumps(record))
    active = dict(record, id="batch-active", input_file_id="file-active",
                  status="in_progress")
    (storage.batches_dir / "batch-active.json").write_text(json.dumps(active))

    loaded = storage.load_batch("batch-old")
    assert loaded.request_counts.model_dump() == counts(0, 0, 0)
    assert storage.fail_orphaned_batches(str(tmp_path / "no-incremental")) == ["batch-active"]
    failed = storage.load_batch("batch-active")
    assert failed.status == io.BatchStatus.FAILED
    assert failed.request_counts.model_dump() == counts(0, 0, 0)


def test_record_request_result_counts_each_request_once():
    pool = mods.scheduling_pool.SchedulingPool(capacity=4)
    assert pool.record_request_result("r1", "b", failed=False) is None  # untracked
    pool.register_batch("b", total_requests=2)
    assert pool.record_request_result("r1", "b", failed=True) is False
    assert pool.record_request_result("r1", "b", failed=False) is None  # duplicate
    assert pool.record_request_result("r2", "b", failed=False) is True
    tracker = pool.get_batch_tracker("b")
    assert (tracker.completed_requests, tracker.failed_requests) == (1, 1)


def test_request_counts_reach_the_fastapi_response(tmp_path):
    """The serialisation FastAPI applies to the real routes carries the live
    counts on GET /v1/batches/{id} and GET /v1/batches."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    async def run():
        server = _Server(tmp_path)
        await server.start()
        server.create("b", {"ok": "p1", "long": LONG, "c": "p3"})
        await server.scheduler._process_batch("b")
        await server.complete("b", "ok")
        await server.stop()
        return server

    server = asyncio.run(run())
    app = FastAPI()
    app.state.storage = server.storage
    app.state.scheduler = server.scheduler
    _batch_handlers(app)
    with TestClient(app) as client:
        one = client.get("/v1/batches/b")
        many = client.get("/v1/batches")
    assert one.status_code == 200, one.text
    assert one.json()["request_counts"] == counts(3, 1, 1)
    assert many.json()["data"][0]["request_counts"] == counts(3, 1, 1)
