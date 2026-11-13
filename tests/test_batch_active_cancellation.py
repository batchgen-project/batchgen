"""Batch API cancellation stops admitted work and owns the terminal status.

POST /v1/batches/{id}/cancel used to write ``cancelled`` to storage and do
nothing else: the intake drain kept admitting the batch's queued requests,
the worker ran every admitted sequence to the end, their results were written
to the output file, and the completed batch then overwrote ``cancelled`` with
``completed``.

Server side, batch_scheduler is loaded by path with the real io_struct /
IntakePool / SchedulingPool / StorageManager and stubs for the engine-backed
modules (as in test_batch_in_progress_failure.py). Worker side, the
cancellation module and the completion predicate are exercised on a real
SequenceBatch; the worker's dispatch and cancel methods are compiled from
its source.
"""
import ast
import asyncio
import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker.completion import CompletionContext, CompletionHandler

ROOT = Path(__file__).resolve().parents[1] / "batchgen"
SERVER = ROOT / "server"


def _load_server_modules():
    names = ["batchgen.server"] + [
        f"batchgen.server.{m}" for m in (
            "io_struct", "intake_pool", "scheduling_pool", "server_args",
            "storage", "worker_manager", "batch_scheduler")]
    saved = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        package = types.ModuleType("batchgen.server")
        package.__path__ = [str(SERVER)]
        sys.modules["batchgen.server"] = package
        for module, attr in (("server_args", "ServerArgs"),
                             ("worker_manager", "WorkerManager")):
            stub = types.ModuleType(f"batchgen.server.{module}")
            setattr(stub, attr, object)
            sys.modules[stub.__name__] = stub
        for module in ("io_struct", "intake_pool", "scheduling_pool",
                       "storage", "batch_scheduler"):
            spec = importlib.util.spec_from_file_location(
                f"batchgen.server.{module}", SERVER / f"{module}.py")
            loaded[module] = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded[module]
            spec.loader.exec_module(loaded[module])
        return loaded
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_mods = _load_server_modules()
bs = _mods["batch_scheduler"]
st = _mods["storage"]
io = _mods["io_struct"]
BatchStatus = io.BatchStatus


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _scheduler(tmp_path, batch_id="b1", n_requests=4, capacity=2):
    """A pool-mode scheduler holding one IN_PROGRESS batch.

    The first ``capacity`` requests are on the worker (slot held); the rest
    are still queued in the IntakePool.
    """
    storage = st.StorageManager(tmp_path)
    storage.save_batch(io.BatchObject(
        id=batch_id,
        endpoint=io.BatchEndpoint.CHAT_COMPLETIONS,
        input_file_id="file-in",
        completion_window=io.CompletionWindow.ONE_DAY,
        status=BatchStatus.IN_PROGRESS,
        created_at=1,
        expires_at=2,
    ))
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)
    scheduler.storage = storage
    scheduler.worker = SimpleNamespace(request_queue=_Queue())
    scheduler.server_args = SimpleNamespace(
        incremental_output_dir=str(tmp_path / "incr"), no_incremental_save=False)
    scheduler._scheduling_pool = bs.SchedulingPool(capacity=capacity)
    scheduler._intake_pool = bs.IntakePool(max_capacity=64)
    scheduler._pool_request_meta = {}
    scheduler._turn_waiters = {}
    scheduler._detached_turns = set()
    scheduler._stale_intake_batches = set()
    scheduler._batch_timeout = 3600
    scheduler._scheduling_pool.register_batch(batch_id, n_requests)
    request_ids = [f"r{i}@{batch_id}" for i in range(n_requests)]
    scheduler._pool_request_meta[batch_id] = {
        rid: {"custom_id": rid.split("@")[0], "url": "/v1/chat/completions",
              "model": "m", "prompt_text": "p"}
        for rid in request_ids
    }
    entries = [bs.IntakeEntry(request_id=rid, batch_id=batch_id,
                              raw_request={"text": "p", "max_tokens": 8},
                              priority=bs.Priority.NORMAL)
               for rid in request_ids]
    assert scheduler._intake_pool.submit_batch(batch_id, entries, bs.Priority.NORMAL)
    for entry in scheduler._intake_pool.drain(capacity):
        scheduler._scheduling_pool.allocate_slot(entry.request_id)
    written = []
    scheduler._write_pool_completion = (
        lambda b, r, result: written.append((b, r)))
    return scheduler, request_ids, written


async def _deliver(scheduler, items):
    """Run the completion listener over *items* and stop it."""
    scheduler._stopped = asyncio.Event()
    scheduler._completion_inbox = asyncio.Queue()
    scheduler._pump_generation = 1
    for item in items:
        scheduler._completion_inbox.put_nowait((1, item))
    task = asyncio.create_task(scheduler._pool_completion_listener())
    while not scheduler._completion_inbox.empty():
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    scheduler._stopped.set()
    await task


def _completion(rid, batch_id="b1", finish="cancelled"):
    return {"type": "completion", "request_id": rid, "batch_id": batch_id,
            "text": "", "finish_reason": finish}


# --------------------------------------------------------------- server side


def test_cancel_stops_admitted_work_and_ends_cancelled(tmp_path):
    scheduler, rids, written = _scheduler(tmp_path)

    batch = scheduler.cancel_batch("b1")

    # Admitted requests are still on the worker: the batch is CANCELLING and
    # the worker is told to stop them; queued requests are gone.
    assert batch.status == BatchStatus.CANCELLING
    assert batch.cancelling_at is not None and batch.cancelled_at is None
    assert scheduler.worker.request_queue.items == [{"type": "cancel", "batch_id": "b1"}]
    assert scheduler._intake_pool.size() == 0

    # The worker returns each admitted request; nothing is published.
    asyncio.run(_deliver(scheduler, [_completion(rids[0])]))
    assert scheduler.storage.load_batch("b1").status == BatchStatus.CANCELLING
    asyncio.run(_deliver(scheduler, [_completion(rids[1], finish="stop")]))

    final = scheduler.storage.load_batch("b1")
    assert final.status == BatchStatus.CANCELLED
    assert final.cancelled_at is not None
    assert written == []
    assert scheduler._scheduling_pool.num_active_slots() == 0
    assert scheduler._scheduling_pool.get_batch_tracker("b1") is None
    assert "b1" not in scheduler._pool_request_meta


def test_batch_with_nothing_on_the_worker_is_cancelled_at_once(tmp_path):
    scheduler, _, _ = _scheduler(tmp_path, capacity=0)

    batch = scheduler.cancel_batch("b1")

    assert batch.status == BatchStatus.CANCELLED
    assert scheduler.worker.request_queue.items == []
    assert scheduler._intake_pool.size() == 0
    assert scheduler._scheduling_pool.get_batch_tracker("b1") is None


def test_validating_batch_is_cancelled_without_worker_traffic(tmp_path):
    scheduler, _, _ = _scheduler(tmp_path)
    scheduler.storage.save_batch(io.BatchObject(
        id="b2", endpoint=io.BatchEndpoint.CHAT_COMPLETIONS,
        input_file_id="file-2", completion_window=io.CompletionWindow.ONE_DAY,
        status=BatchStatus.VALIDATING, created_at=1, expires_at=2))

    assert scheduler.cancel_batch("b2").status == BatchStatus.CANCELLED
    assert scheduler.worker.request_queue.items == []


def test_completed_batch_cannot_be_cancelled(tmp_path):
    scheduler, _, _ = _scheduler(tmp_path)
    scheduler.storage.update_batch_status("b1", BatchStatus.COMPLETED)
    with pytest.raises(ValueError, match="Cannot cancel"):
        scheduler.cancel_batch("b1")
    with pytest.raises(KeyError):
        scheduler.cancel_batch("missing")


def test_cancelled_batch_is_never_finalized_as_completed(tmp_path):
    """The waiter that finalizes a batch must leave a cancelled one alone,
    even when every request already reported before the cancel."""
    scheduler, rids, _ = _scheduler(tmp_path, n_requests=2, capacity=2)
    tracker = scheduler._scheduling_pool.get_batch_tracker("b1")
    for rid in rids:
        scheduler._scheduling_pool.mark_request_completed(rid, "b1")
    assert tracker.is_complete

    scheduler.cancel_batch("b1")
    finalized = []
    scheduler._finalize_batch_output = lambda *a: finalized.append(a)

    async def run():
        waiter = asyncio.create_task(scheduler._wait_and_finalize_batch("b1", [], []))
        await asyncio.sleep(0)
        await _deliver(scheduler, [_completion(rid) for rid in rids])
        await asyncio.wait_for(waiter, timeout=5)

    asyncio.run(run())
    assert finalized == []
    assert scheduler.storage.load_batch("b1").status == BatchStatus.CANCELLED


def test_queued_entries_of_a_cancelled_batch_are_never_admitted(tmp_path):
    scheduler, rids, _ = _scheduler(tmp_path)
    scheduler.cancel_batch("b1")
    # An entry selected before the cancel and requeued afterwards.
    scheduler._intake_pool.requeue(bs.IntakeEntry(
        request_id=rids[3], batch_id="b1", raw_request={},
        priority=bs.Priority.NORMAL))
    drained = scheduler._intake_pool.drain(4)
    assert scheduler._drop_stale_intake_entries(drained) == []


def test_worker_fatal_while_cancelling_ends_cancelled(tmp_path):
    scheduler, _, _ = _scheduler(tmp_path)
    scheduler._fail_pending_turns = lambda error: None
    scheduler.cancel_batch("b1")

    scheduler._fail_all_active_batches("worker died")

    assert scheduler.storage.load_batch("b1").status == BatchStatus.CANCELLED
    assert scheduler._scheduling_pool.num_active_slots() == 0


def test_terminal_status_is_never_overwritten(tmp_path):
    storage = st.StorageManager(tmp_path)
    for status in (BatchStatus.CANCELLED, BatchStatus.COMPLETED, BatchStatus.FAILED):
        storage.save_batch(io.BatchObject(
            id=status.value, endpoint=io.BatchEndpoint.CHAT_COMPLETIONS,
            input_file_id="f", completion_window=io.CompletionWindow.ONE_DAY,
            status=status, created_at=1, expires_at=2))
    with pytest.raises(st.BatchStatusTransitionError):
        storage.update_batch_status("cancelled", BatchStatus.COMPLETED)
    with pytest.raises(st.BatchStatusTransitionError):
        storage.update_batch_status("completed", BatchStatus.CANCELLED)
    with pytest.raises(st.BatchStatusTransitionError):
        storage.update_batch_status("failed", BatchStatus.IN_PROGRESS)
    assert storage.load_batch("cancelled").status == BatchStatus.CANCELLED


def test_detached_turn_is_cancelled_on_the_worker(tmp_path):
    scheduler, _, _ = _scheduler(tmp_path)
    future = SimpleNamespace(done=lambda: True)
    scheduler._turn_waiters["turn-1"] = SimpleNamespace(future=future)

    assert scheduler.cancel_turn("turn-1", "test") == "detached"
    assert scheduler.worker.request_queue.items == [
        {"type": "cancel", "batch_id": "turn-1"}]


# --------------------------------------------------------------- worker side


def cancel_batch_sequences(worker, batch_id):
    return _worker_method("_cancel_batch_sequences")(worker, batch_id)


class _Worker:
    rank = 0

    def __init__(self):
        self.global_batch = SequenceBatch()
        self.reported = []
        self.released = None
        self._uuid_to_local_map = {}

    def _report_completion(self, uuid, gathered_text=None):
        self.reported.append(uuid)

    def _release_prefix_cache_attachments(self, global_ids):
        self.released = list(global_ids)


def _seq(worker, uuid, batch_id, status, idx):
    seq = SequenceEntry(uuid=uuid, global_idx=idx, prompt_length=4,
                        max_decode_length=16)
    seq.batch_id = batch_id
    worker.global_batch.add_sequence(seq)
    path = {
        SequenceStatus.QUEUEING: [],
        SequenceStatus.PREFILLED: [SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED],
        SequenceStatus.IN_DECODE: [SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
                                   SequenceStatus.IN_DECODE],
        SequenceStatus.ON_HOLD: [SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
                                 SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD],
        SequenceStatus.EVICTED: [SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
                                 SequenceStatus.IN_DECODE, SequenceStatus.EVICTED],
        SequenceStatus.IN_PREFILL: [SequenceStatus.IN_PREFILL],
        SequenceStatus.COMPLETED: [SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
                                   SequenceStatus.COMPLETED],
    }[status]
    for step in path:
        worker.global_batch.update_status(uuid, step)
    return seq


def test_worker_cancel_ends_every_live_sequence_of_the_batch():
    worker = _Worker()
    statuses = [SequenceStatus.QUEUEING, SequenceStatus.PREFILLED,
                SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD,
                SequenceStatus.EVICTED, SequenceStatus.COMPLETED]
    seqs = {s: _seq(worker, f"c-{s.name}", "b1", s, i) for i, s in enumerate(statuses)}
    other = _seq(worker, "other", "b2", SequenceStatus.QUEUEING, 10)

    cancelled = cancel_batch_sequences(worker, "b1")

    assert sorted(cancelled) == sorted(
        seqs[s].uuid for s in statuses if s != SequenceStatus.COMPLETED)
    # Never prefilled: completed here and reported, releasing token storage.
    assert worker.reported == ["c-QUEUEING", "c-EVICTED"]
    assert worker.released == []  # this rank holds neither locally
    assert seqs[SequenceStatus.QUEUEING].status == SequenceStatus.COMPLETED
    assert seqs[SequenceStatus.EVICTED].status == SequenceStatus.COMPLETED
    # Holding KV: left to the decode boundary, which now sees them complete.
    ctx = CompletionContext(eos_token_ids=frozenset(), model_context_length=1 << 20, rank=0)
    for status in (SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD):
        seq = seqs[status]
        assert seq.status == status and seq.cancelled
        assert CompletionHandler.is_sequence_completed(ctx, seq)
        assert CompletionHandler.get_finish_reason(ctx, seq) == "cancelled"
    assert not other.cancelled and other.status == SequenceStatus.QUEUEING
    assert not CompletionHandler.is_sequence_completed(ctx, other)


def test_worker_cancel_during_prefill_fails_loud():
    worker = _Worker()
    _seq(worker, "p", "b1", SequenceStatus.IN_PREFILL, 0)
    with pytest.raises(RuntimeError, match="IN_PREFILL"):
        cancel_batch_sequences(worker, "b1")
    with pytest.raises(ValueError, match="batch_id"):
        cancel_batch_sequences(worker, None)


def _worker_method(name):
    source = (ROOT / "batchgen_worker.py").read_text()
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"SequenceStatus": SequenceStatus, "List": list,
                 "logging": __import__("logging")}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(node)], type_ignores=[])), "batchgen_worker.py", "exec"),
        namespace)
    return namespace[name]


def test_worker_dispatches_cancel_messages():
    dispatch = _worker_method("_handle_pool_control_message")
    worker = _Worker()
    worker._cancel_batch_sequences = lambda batch_id: cancel_batch_sequences(worker, batch_id)
    worker._admit_sequences_from_message = lambda msg: worker.reported.append("admit")
    _seq(worker, "q", "b1", SequenceStatus.QUEUEING, 0)

    assert dispatch(worker, {"type": "admit", "entries": []}) is True
    assert dispatch(worker, {"type": "cancel", "batch_id": "b1"}) is False
    assert worker.reported == ["admit", "q"]
    with pytest.raises(ValueError, match="unknown pool-control"):
        dispatch(worker, {"type": "bogus"})


def test_every_admission_poll_accepts_cancel_messages():
    """The rank-0 queue reads that broadcast pool-control messages accept
    ``cancel`` as well as ``admit``; a cancel dropped there never stops work."""
    source = (ROOT / "batchgen_worker.py").read_text()
    assert source.count('msg.get("type") == "admit":\n\t\t\t\t\tmsg_data') == 0
    assert source.count("msg.get(\"type\") in _POOL_CONTROL_TYPES") == 2
