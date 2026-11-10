"""A batch marked in_progress always reaches a terminal status.

_process_batch sets IN_PROGRESS before it parses and renders the input file,
and _run only logs an exception. A request without max_tokens in a batch with
no max_decoding_length used to raise while building its error message (``.get``
on a pydantic BatchRequestItem), so the batch stayed in_progress forever.

batch_scheduler is loaded by path with the real io_struct / IntakePool /
SchedulingPool and stubs for the engine-backed modules, as in
test_tokenizer_family_admission.py.
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

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


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


def _line(custom_id, max_tokens=None):
    body = {"model": MODEL, "prompt": f"prompt of {custom_id}"}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/completions",
        "body": body,
    })


def _scheduler(tmp_path, lines, max_decoding_length=None):
    input_id = "file-input"
    (tmp_path / input_id).write_bytes("\n".join(lines).encode())
    updates = []
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)
    scheduler.server_args = SimpleNamespace(model=MODEL)
    # The pool state BatchScheduler.__init__ creates; a failure releases it.
    scheduler._scheduling_pool = bs.SchedulingPool(capacity=4)
    scheduler._intake_pool = bs.IntakePool(max_capacity=16)
    scheduler._pool_request_meta = {}
    scheduler.storage = SimpleNamespace(
        files_dir=tmp_path,
        load_batch=lambda batch_id: SimpleNamespace(
            status=bs.BatchStatus.VALIDATING,
            input_file_id=input_id,
            max_decoding_length=max_decoding_length,
            temperature=None,
            top_p=None,
            top_k=None,
        ),
        load_metadata=lambda file_id: {"id": file_id},
        update_batch_status=lambda batch_id, status, **kw: updates.append(
            (status, kw.get("error"))),
    )
    return scheduler, updates


def test_missing_max_tokens_without_a_batch_default_fails_the_batch(tmp_path):
    scheduler, updates = _scheduler(
        tmp_path, [_line("has-limit", 16), _line("no-limit-a"), _line("no-limit-b")]
    )

    async def must_not_submit(*args, **kwargs):
        raise AssertionError("a batch with unbounded requests reached the pool")

    scheduler._process_batch_pool_mode = must_not_submit
    asyncio.run(scheduler._process_batch("batch-1"))

    statuses = [status for status, _ in updates]
    assert statuses == [bs.BatchStatus.IN_PROGRESS, bs.BatchStatus.FAILED]
    error = updates[-1][1]
    assert "2/3 requests have no max_completion_tokens or max_tokens" in error
    assert "['no-limit-a', 'no-limit-b']" in error


def test_a_batch_default_covers_requests_without_max_tokens(tmp_path):
    scheduler, updates = _scheduler(
        tmp_path, [_line("has-limit", 16), _line("no-limit")],
        max_decoding_length=64,
    )
    submitted = {}

    async def record_submit(batch_id, batch, requests, prompts,
                            per_request_max_tokens, sampling_params):
        submitted["max_tokens"] = per_request_max_tokens

    scheduler._process_batch_pool_mode = record_submit
    asyncio.run(scheduler._process_batch("batch-1"))

    assert submitted["max_tokens"] == [16, 64]
    assert [status for status, _ in updates] == [bs.BatchStatus.IN_PROGRESS]


@pytest.mark.parametrize("stage", ["render", "submit"])
def test_any_exception_after_in_progress_ends_in_failed(tmp_path, stage):
    """Whatever raises after IN_PROGRESS, the batch is persisted as FAILED
    with the exception recorded, instead of only being logged by _run."""
    scheduler, updates = _scheduler(tmp_path, [_line("req-1", 16)])

    def broken_render(requests, batch=None):
        raise ValueError("request 'req-1': the chat template rejected it")

    async def broken_submit(*args, **kwargs):
        raise RuntimeError("intake pool unavailable")

    if stage == "render":
        scheduler._convert_requests_to_worker_inputs = broken_render
        expected = "ValueError: request 'req-1': the chat template rejected it"
    else:
        scheduler._process_batch_pool_mode = broken_submit
        expected = "RuntimeError: intake pool unavailable"

    asyncio.run(scheduler._process_batch("batch-1"))

    assert [status for status, _ in updates] == [
        bs.BatchStatus.IN_PROGRESS, bs.BatchStatus.FAILED,
    ]
    assert updates[-1][1] == f"Batch processing failed: {expected}"


def test_an_exception_after_pool_registration_releases_the_batch_state(tmp_path):
    """A batch that fails after the pool registered it must not leave its
    tracker, queued entries or request metadata behind for the drain and the
    listener to act on."""
    scheduler, updates = _scheduler(tmp_path, [_line("req-1", 16)])

    async def register_then_fail(batch_id, batch, requests, prompts,
                                 per_request_max_tokens, sampling_params):
        scheduler._scheduling_pool.register_batch(batch_id, total_requests=1)
        scheduler._intake_pool.submit_batch(batch_id, [bs.IntakeEntry(
            request_id=f"req-1@{batch_id}",
            batch_id=batch_id,
            raw_request={"text": prompts[0], "max_tokens": 16},
            priority=bs.Priority.NORMAL,
        )])
        scheduler._pool_request_meta[batch_id] = {f"req-1@{batch_id}": {}}
        raise OSError("incremental output directory is not writable")

    scheduler._process_batch_pool_mode = register_then_fail
    asyncio.run(scheduler._process_batch("batch-1"))

    assert [status for status, _ in updates] == [
        bs.BatchStatus.IN_PROGRESS, bs.BatchStatus.FAILED,
    ]
    assert "OSError: incremental output directory is not writable" in updates[-1][1]
    assert scheduler._scheduling_pool.get_batch_tracker("batch-1") is None
    assert scheduler._intake_pool.size() == 0
    assert scheduler._intake_pool.get_batch_info("batch-1") is None
    assert scheduler._pool_request_meta == {}
