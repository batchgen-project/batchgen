"""An intake-capacity rejection fails the batch through the single exit.

``_process_batch_pool_mode`` registers the batch with the SchedulingPool (and
persists its validated request total) before offering its entries to the
IntakePool.  A capacity rejection must therefore release that registration
through ``_discard_failed_batch``, the one exit every other failure of an
in-progress batch uses.  Failing the batch directly leaks the tracker: the
scheduler state never shrinks, ``_fail_all_active_batches`` later re-fails the
batch and rewrites its persisted capacity error with a worker-fatal one, and
the live-count view keeps serving a batch that is terminally failed.

batch_scheduler and storage are loaded by path with the real io_struct /
IntakePool / SchedulingPool / StorageManager and stubs for the engine-backed
modules, as in test_batch_request_counts.py.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "batchgen" / "server"
MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


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


class _Tokenizer:
    """Encodes completion prompt text one id per character."""

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[ord(ch) for ch in text] for text in texts]}


def _make_scheduler(tmp_path, max_intake_capacity):
    storage = StorageManager(tmp_path / "store")
    scheduler = bs.BatchScheduler(storage, None, SimpleNamespace(
        model=MODEL, max_intake_capacity=max_intake_capacity,
        incremental_output_dir=None,
        no_incremental_save=False, parse_thinking=False, parse_tool_call=False,
    ))
    scheduler.max_model_len = 32
    scheduler._get_tokenizer = lambda model: _Tokenizer()
    return storage, scheduler


def _create(storage, batch_id, prompts):
    """Upload ``{custom_id: prompt}`` and persist a validating batch."""
    input_id = f"file-{batch_id}"
    (storage.files_dir / input_id).write_text("\n".join(json.dumps({
        "custom_id": custom_id, "method": "POST", "url": "/v1/completions",
        "body": {"model": MODEL, "prompt": prompt, "max_tokens": 4},
    }) for custom_id, prompt in prompts.items()))
    storage.save_metadata(input_id, {"id": input_id, "purpose": "batch"})
    storage.save_batch(io.build_batch_object_from_create_request(
        batch_id=batch_id,
        body=io.CreateBatchRequest(input_file_id=input_id),
        created_at=1, expires_at=2,
    ))


async def _process(scheduler, *batch_ids):
    """Run ``_process_batch`` for each id, then cancel the finalize waiters.

    An accepted batch launches ``_wait_and_finalize_batch`` in the background;
    the tests never complete its requests, so the waiter is cancelled before
    the loop closes.
    """
    for batch_id in batch_ids:
        await scheduler._process_batch(batch_id)
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


def test_intake_capacity_rejection_goes_through_the_single_exit(tmp_path):
    """The refused batch fails with the capacity error and holds no state."""
    storage, scheduler = _make_scheduler(tmp_path, max_intake_capacity=2)
    _create(storage, "fill", {"a": "p1", "b": "p2"})
    _create(storage, "over", {"c": "p3"})

    asyncio.run(_process(scheduler, "fill", "over"))

    # The filling batch was accepted and is still tracked.
    assert scheduler._scheduling_pool.get_batch_tracker("fill") is not None
    assert scheduler._intake_pool.size() == 2

    # The refused batch is FAILED with the capacity error, unchanged.
    over = storage.load_batch("over")
    assert over.status == io.BatchStatus.FAILED
    assert over.error == (
        "capacity_exceeded: Server at capacity: intake pool has 2/2 "
        "requests. Batch with 1 requests rejected. Retry later."
    )
    # Its validated total persists; nothing completed or failed per-request.
    assert over.request_counts.model_dump() == {
        "total": 1, "completed": 0, "failed": 0,
    }
    # The single exit released every piece of scheduler state it held: the
    # tracker (the leak), and the per-request metadata and intake info that
    # were never created for it.
    assert scheduler._scheduling_pool.get_batch_tracker("over") is None
    assert "over" not in scheduler._pool_request_meta
    assert scheduler._intake_pool.get_batch_info("over") is None


def test_worker_fatal_does_not_refail_a_capacity_rejected_batch(tmp_path):
    """A leaked tracker would let a later fatal rewrite the capacity error."""
    storage, scheduler = _make_scheduler(tmp_path, max_intake_capacity=2)
    _create(storage, "fill", {"a": "p1", "b": "p2"})
    _create(storage, "over", {"c": "p3"})

    asyncio.run(_process(scheduler, "fill", "over"))
    scheduler._fail_all_active_batches("rank 1 died")

    # The active batch fails with the fatal; the already-failed one keeps
    # the capacity error it was persisted with.
    fill = storage.load_batch("fill")
    assert fill.status == io.BatchStatus.FAILED
    assert fill.error == "worker_fatal: rank 1 died"
    over = storage.load_batch("over")
    assert over.status == io.BatchStatus.FAILED
    assert over.error.startswith("capacity_exceeded: ")
