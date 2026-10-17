"""Dev-machine tests for generation-scoped completion delivery and admission.

After a worker-pool restart the new cohort writes to a NEW response queue, and
the previous generation's reader owns a queue that was abandoned with its
cohort. Completion delivery is therefore generation-scoped: one daemon pump
thread per generation owns the blocking read and forwards items to the
scheduler loop, where only the live generation's items are accepted.

These tests load the classes/handlers from source, the way
test_worker_pool_restart.py does, because importing batchgen.server pulls in
GPU-only dependencies.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]

SCHEDULER_SOURCE = ROOT / "batchgen/server/batch_scheduler.py"
HTTP_SOURCE = ROOT / "batchgen/server/http_server.py"


class _FakeHTTPException(Exception):
    """Stands in for fastapi.HTTPException, which is not importable here."""

    def __init__(self, status_code, detail=None, headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


def _load_batch_scheduler_class():
    tree = ast.parse(SCHEDULER_SOURCE.read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "BatchScheduler"
    )
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "asyncio": asyncio,
        "json": json,
        "logging": logging,
        "logger": logging.getLogger(__name__),
        "threading": threading,
        "time": time,
        "uuid": uuid,
        "BatchStatus": SimpleNamespace(FAILED="failed"),
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(SCHEDULER_SOURCE), "exec"), namespace)
    return namespace["BatchScheduler"]


def _load_create_batch_handler():
    tree = ast.parse(HTTP_SOURCE.read_text())
    node = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.AsyncFunctionDef) and item.name == "create_batch"
    )
    node = copy.deepcopy(node)
    node.decorator_list = []
    namespace = {
        "Optional": Optional,
        "HTTPException": _FakeHTTPException,
        "time": time,
        "uuid": uuid,
        "FilePurpose": SimpleNamespace(BATCH=SimpleNamespace(value="batch")),
        "build_batch_object_from_create_request": (
            lambda **kwargs: SimpleNamespace(**kwargs)
        ),
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(HTTP_SOURCE), "exec"), namespace)
    return namespace["create_batch"]


class _SchedulingPoolStub:
    def __init__(self) -> None:
        self._batch_trackers: Dict[str, Any] = {}
        self.freed: List[str] = []
        self.completed: List[Tuple[str, str]] = []
        self.cleared = 0

    def free_slot(self, request_id: str) -> None:
        self.freed.append(request_id)

    def mark_request_completed(self, request_id: str, batch_id: str) -> bool:
        self.completed.append((request_id, batch_id))
        return False

    def num_active_slots(self) -> int:
        return 0

    def num_free_slots(self) -> int:
        return 8

    def clear_active_slots(self) -> None:
        self.cleared += 1


class _StorageStub:
    def __init__(self) -> None:
        self.statuses: List[Tuple[str, Any, Dict[str, Any]]] = []

    def update_batch_status(self, batch_id: str, status: Any, **kwargs: Any) -> None:
        self.statuses.append((batch_id, status, kwargs))


class _WorkerStub:
    def __init__(self) -> None:
        self.response_queue: Any = queue.Queue()
        self.fatals: List[Tuple[str, bool]] = []

    def report_worker_fatal(self, reason: str, *, persisted: bool = False) -> None:
        self.fatals.append((reason, persisted))


class _BrokenQueue:
    """A channel whose reader handle has been closed under the pump."""

    def get(self, timeout: Optional[float] = None):
        raise OSError("handle is closed")


def _completion(request_id: str, batch_id: str) -> Dict[str, Any]:
    return {
        "type": "completion",
        "request_id": request_id,
        "batch_id": batch_id,
        "text": f"output for {request_id}",
        "prompt_length": 3,
        "decoded_length": 2,
        "finish_reason": "stop",
    }


def _scheduler(tmp_path: Path):
    """Build a scheduler with only the completion-path collaborators.

    Must be called from inside a running loop: the asyncio primitives bind to
    the loop that is current when they are constructed.
    """
    cls = _load_batch_scheduler_class()
    scheduler = object.__new__(cls)
    scheduler._loop = asyncio.get_running_loop()
    scheduler._stopped = asyncio.Event()
    scheduler._completion_inbox = asyncio.Queue()
    scheduler._pump_lock = threading.Lock()
    scheduler._pump_generation = 0
    scheduler._pump_thread = None
    scheduler._pump_stop = None
    scheduler._completion_listener_task = None
    scheduler._pool_initialized = False
    scheduler.worker = _WorkerStub()
    scheduler.storage = _StorageStub()
    scheduler._scheduling_pool = _SchedulingPoolStub()
    scheduler._intake_pool = SimpleNamespace(
        size=lambda: 0, remove_batch_entries=lambda batch_id: 0,
    )
    scheduler._stale_intake_batches = set()
    scheduler.server_args = SimpleNamespace(
        incremental_output_dir=str(tmp_path),
    )
    scheduler._pool_request_meta = {
        "batch-1": {
            request_id: {
                "custom_id": request_id,
                # A non-chat endpoint keeps the tokenizer out of this test.
                "url": "/v1/completions",
                "model": "test-model",
                "prompt_text": "prompt",
            }
            for request_id in ("req-1", "req-2")
        }
    }
    return scheduler


async def _await_true(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


async def _shutdown(scheduler) -> None:
    scheduler._stopped.set()
    with scheduler._pump_lock:
        if scheduler._pump_stop is not None:
            scheduler._pump_stop.set()
    task = scheduler._completion_listener_task
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def test_completion_from_the_new_generation_reaches_the_routing_path(tmp_path):
    """A restart must not lose completions: the new generation's queue is read
    by a new pump, and the superseded pump's items are dropped."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        dead_queue = scheduler.worker.response_queue
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()
        assert scheduler._pump_generation == 1

        # The restart replaces the cohort's channel and then announces the new
        # generation from its own thread, never from the scheduler loop.
        fresh_queue: Any = queue.Queue()
        scheduler.worker.response_queue = fresh_queue
        await asyncio.to_thread(scheduler.worker_generation_ready)
        assert scheduler._pump_generation == 2
        assert scheduler._pool_initialized is True
        assert not scheduler._completion_listener_task.done()

        fresh_queue.put(_completion("req-1", "batch-1"))
        assert await _await_true(lambda: scheduler._scheduling_pool.completed)
        assert scheduler._scheduling_pool.completed == [("req-1", "batch-1")]
        assert scheduler._scheduling_pool.freed == ["req-1"]
        lines = (tmp_path / "batch-1.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["custom_id"] == "req-1"

        # Whatever the superseded pump still forwards is dropped, so a stale
        # generation can never mutate the live generation's batch state.
        dead_queue.put(_completion("req-2", "batch-1"))
        assert not await _await_true(
            lambda: len(scheduler._scheduling_pool.completed) > 1, timeout=1.0
        )
        assert scheduler._scheduling_pool.completed == [("req-1", "batch-1")]
        assert scheduler._scheduling_pool.freed == ["req-1"]

        await _shutdown(scheduler)

    asyncio.run(scenario())


def test_listener_survives_pool_shutdown_and_routes_the_next_generation(tmp_path):
    """pool_shutdown still persists the in-flight batches exactly once, but the
    listener does NOT end with it: a listener that exits on a terminal item
    can exit after the next generation's re-arm check saw it alive, leaving
    that generation with a pump and no consumer."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        tracker = SimpleNamespace(is_complete=False, is_failed=False, error=None)
        scheduler._scheduling_pool._batch_trackers["batch-1"] = tracker
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()
        listener = scheduler._completion_listener_task

        scheduler.worker.response_queue.put(
            {"type": "pool_shutdown", "error": "worker fatal"}
        )
        assert await _await_true(lambda: scheduler.worker.fatals)
        assert scheduler.worker.fatals == [("worker fatal", True)]
        assert [entry[0] for entry in scheduler.storage.statuses] == ["batch-1"]
        assert tracker.error == "worker fatal"
        assert scheduler._scheduling_pool.cleared == 1

        # The listener outlives the generation that reported the fatal.
        await asyncio.sleep(0.1)
        assert not listener.done()

        # The next generation reuses the SAME listener task.
        tracker.is_failed = True
        fresh_queue: Any = queue.Queue()
        scheduler.worker.response_queue = fresh_queue
        await asyncio.to_thread(scheduler.worker_generation_ready)
        assert scheduler._completion_listener_task is listener
        assert not listener.done()

        fresh_queue.put(_completion("req-1", "batch-1"))
        assert await _await_true(lambda: scheduler._scheduling_pool.completed)
        assert scheduler._scheduling_pool.completed == [("req-1", "batch-1")]
        # Still exactly one persisted failure for that batch, one report.
        assert [entry[0] for entry in scheduler.storage.statuses] == ["batch-1"]
        assert scheduler.worker.fatals == [("worker fatal", True)]

        await _shutdown(scheduler)

    asyncio.run(scenario())


def test_listener_recreates_a_crashed_task_on_the_next_generation(tmp_path):
    """_ensure_completion_listener remains as crash recovery: a listener task
    that died is recreated when the next generation is announced."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()
        first = scheduler._completion_listener_task
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass

        fresh_queue: Any = queue.Queue()
        scheduler.worker.response_queue = fresh_queue
        await asyncio.to_thread(scheduler.worker_generation_ready)
        assert scheduler._completion_listener_task is not first
        assert not scheduler._completion_listener_task.done()

        fresh_queue.put(_completion("req-1", "batch-1"))
        assert await _await_true(lambda: scheduler._scheduling_pool.completed)

        await _shutdown(scheduler)

    asyncio.run(scenario())


def test_item_queued_before_a_generation_bump_is_dropped(tmp_path):
    """An item accepted while its generation was live can still be waiting in
    the inbox when a restart advances the generation. The listener re-checks
    the tag, so the stale item never mutates live state."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()
        assert scheduler._pump_generation == 1

        # Accepted by the insert-time filter while gen 1 is live, then the
        # generation advances before the listener runs (no await in between).
        scheduler._offer_completion(1, _completion("req-2", "batch-1"))
        assert scheduler._completion_inbox.qsize() == 1
        scheduler.worker.response_queue = queue.Queue()
        scheduler._start_completion_pump()
        assert scheduler._pump_generation == 2

        assert await _await_true(lambda: scheduler._completion_inbox.empty())
        await asyncio.sleep(0.1)
        assert scheduler._scheduling_pool.completed == []
        assert scheduler._scheduling_pool.freed == []
        assert not (tmp_path / "batch-1.jsonl").exists()

        # The live generation's items still route.
        scheduler.worker.response_queue.put(_completion("req-1", "batch-1"))
        assert await _await_true(lambda: scheduler._scheduling_pool.completed)
        assert scheduler._scheduling_pool.completed == [("req-1", "batch-1")]

        await _shutdown(scheduler)

    asyncio.run(scenario())


def test_stale_pool_shutdown_does_not_fail_or_report_the_live_generation(tmp_path):
    """A pool_shutdown from the old cohort that is consumed after the bump
    must not fail the live batches nor take the new cohort down."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        tracker = SimpleNamespace(is_complete=False, is_failed=False, error=None)
        scheduler._scheduling_pool._batch_trackers["batch-1"] = tracker
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()

        scheduler._offer_completion(
            1, {"type": "pool_shutdown", "error": "old cohort fatal"}
        )
        scheduler.worker.response_queue = queue.Queue()
        scheduler._start_completion_pump()

        assert await _await_true(lambda: scheduler._completion_inbox.empty())
        await asyncio.sleep(0.1)
        assert scheduler.worker.fatals == []
        assert scheduler.storage.statuses == []
        assert tracker.error is None
        assert scheduler._scheduling_pool.cleared == 0
        assert not scheduler._completion_listener_task.done()

        await _shutdown(scheduler)

    asyncio.run(scenario())


def test_pump_read_failure_persists_active_batches_as_failed(tmp_path):
    """Abandoning a channel can make a pending read fail. That failure must
    reach the same persistence path the in-task read failure used to take."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        tracker = SimpleNamespace(is_complete=False, is_failed=False, error=None)
        scheduler._scheduling_pool._batch_trackers["batch-1"] = tracker
        scheduler.worker.response_queue = _BrokenQueue()
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()

        assert await _await_true(lambda: scheduler.storage.statuses)
        batch_id, status, kwargs = scheduler.storage.statuses[0]
        assert batch_id == "batch-1"
        assert status == "failed"
        assert "handle is closed" in kwargs["error"]
        # A channel failure is not a worker-reported fatal, so it must not be
        # forwarded as one.
        assert scheduler.worker.fatals == []

        await _shutdown(scheduler)

    asyncio.run(scenario())


def test_retired_pump_read_failure_is_benign(tmp_path):
    """A restart retires the pump before abandoning its channel. The read
    failure that wakes the blocked pump is then expected: it must exit without
    reporting a channel error, failing batches, or ending the listener."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        tracker = SimpleNamespace(is_complete=False, is_failed=False, error=None)
        scheduler._scheduling_pool._batch_trackers["batch-1"] = tracker

        retired = threading.Event()

        class _RetiredQueue:
            """Blocks like a pump mid-read, then fails once the channel is
            closed under it."""

            def get(self, timeout: Optional[float] = None):
                assert retired.wait(5.0)
                raise OSError("handle is closed")

        scheduler.worker.response_queue = _RetiredQueue()
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()
        pump = scheduler._pump_thread

        # The restart's ordering: retire the reader, then close the channel.
        scheduler.stop_completion_pump()
        retired.set()

        assert await _await_true(lambda: not pump.is_alive())
        # No channel error reached the listener: nothing persisted as failed,
        # no worker fatal, and the listener survives into the next generation.
        assert scheduler.storage.statuses == []
        assert scheduler.worker.fatals == []
        assert not scheduler._completion_listener_task.done()

        await _shutdown(scheduler)

    asyncio.run(scenario())


def test_completion_pump_threads_are_daemons_and_do_not_block_exit(tmp_path):
    """A pump blocked on a truncated frame can never return. It must therefore
    be a daemon thread, and a superseded pump must leave on its own."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()
        first = scheduler._pump_thread
        assert first.daemon is True

        scheduler.worker.response_queue = queue.Queue()
        await asyncio.to_thread(scheduler.worker_generation_ready)
        second = scheduler._pump_thread
        assert second is not first
        assert second.daemon is True

        # The superseded pump exits on its own stop event, within one read
        # timeout, without anything joining it.
        assert await _await_true(lambda: not first.is_alive())

        await _shutdown(scheduler)

    asyncio.run(scenario())


def test_completion_listener_never_blocks_in_the_default_executor():
    """A get() that can never return must not sit in the default executor:
    its threads are not daemons, so asyncio.run would wait for them and the
    process could not be stopped gracefully."""
    tree = ast.parse(SCHEDULER_SOURCE.read_text())
    node = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.AsyncFunctionDef)
        and item.name == "_pool_completion_listener"
    )
    calls = {
        ast.unparse(sub.func)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and not isinstance(sub.func, ast.Name)
    }
    assert "asyncio.to_thread" not in calls
    # The pump owns the only blocking read, and it is explicitly a daemon.
    pump = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef)
        and item.name == "_start_completion_pump"
    )
    daemon_kwargs = [
        keyword.value.value
        for sub in ast.walk(pump)
        if isinstance(sub, ast.Call)
        for keyword in sub.keywords
        if keyword.arg == "daemon" and isinstance(keyword.value, ast.Constant)
    ]
    assert daemon_kwargs == [True]


@pytest.mark.parametrize("pool_state", ["restarting", "workers_down"])
def test_batch_admission_is_refused_while_the_pool_is_not_serving(pool_state):
    """A batch accepted during a restart can never run, so it must not be
    accepted: the client gets 503 instead of a success receipt."""
    create_batch = _load_create_batch_handler()

    def unreachable(*args, **kwargs):
        raise AssertionError(
            "admission reached the storage/scheduler layer while the worker "
            f"pool was {pool_state}"
        )

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        storage=SimpleNamespace(
            load_metadata=unreachable,
            get_active_batch_for_file=unreachable,
            save_batch=unreachable,
        ),
        scheduler=SimpleNamespace(
            intake_pool_usage_pct=unreachable,
            enqueue=unreachable,
        ),
        worker=SimpleNamespace(state=pool_state),
    )))

    with pytest.raises(_FakeHTTPException) as excinfo:
        asyncio.run(create_batch(request, body=SimpleNamespace()))

    assert excinfo.value.status_code == 503
    assert pool_state in str(excinfo.value.detail)
    assert excinfo.value.headers == {"Retry-After": "30"}


def test_batch_admission_passes_the_state_gate_while_serving():
    """The gate must only refuse while not serving; a serving pool reaches the
    existing admission checks."""
    create_batch = _load_create_batch_handler()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        storage=SimpleNamespace(load_metadata=lambda file_id: None),
        scheduler=SimpleNamespace(
            intake_pool_usage_pct=lambda: 0.0,
            intake_pool_size=lambda: 0,
            intake_pool_capacity=lambda: 1,
        ),
        worker=SimpleNamespace(state="serving"),
    )))

    with pytest.raises(_FakeHTTPException) as excinfo:
        asyncio.run(create_batch(
            request, body=SimpleNamespace(input_file_id="file-missing")
        ))

    # Past the state gate, into the pre-existing unknown-input-file rejection.
    assert excinfo.value.status_code == 400
    assert "file-missing" in str(excinfo.value.detail)


def test_listener_survives_an_item_that_fails_to_route(tmp_path, caplog):
    """A malformed item is logged with its traceback and skipped; the
    listener keeps routing the items after it."""

    async def scenario():
        scheduler = _scheduler(tmp_path)

        def broken_free_slot(request_id):
            if request_id == "req-bad":
                raise RuntimeError("malformed completion")

        scheduler._scheduling_pool.free_slot = broken_free_slot
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()
        listener = scheduler._completion_listener_task

        scheduler.worker.response_queue.put(_completion("req-bad", "batch-1"))
        scheduler.worker.response_queue.put(_completion("req-1", "batch-1"))
        assert await _await_true(lambda: scheduler._scheduling_pool.completed)
        assert scheduler._scheduling_pool.completed == [("req-1", "batch-1")]
        assert not listener.done()

        await _shutdown(scheduler)

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    failures = [r for r in caplog.records if "failed to route" in r.getMessage()]
    assert len(failures) == 1
    assert failures[0].exc_info is not None


def test_retiring_the_pump_retires_its_generation(tmp_path):
    """Between the retire and the next generation's pump, an item of the
    retired cohort -- already queued, or forwarded by its pump afterwards --
    must not act: retiring advances the generation."""

    async def scenario():
        scheduler = _scheduler(tmp_path)
        tracker = SimpleNamespace(is_complete=False, is_failed=False, error=None)
        scheduler._scheduling_pool._batch_trackers["batch-1"] = tracker
        scheduler._ensure_completion_listener()
        scheduler._start_completion_pump()
        assert scheduler._pump_generation == 1

        # Queued while gen 1 was live, then the restart retires the pump
        # before the listener runs.
        scheduler._offer_completion(1, _completion("req-1", "batch-1"))
        await asyncio.to_thread(scheduler.stop_completion_pump)
        assert scheduler._pump_generation == 2
        # Forwarded by the retired pump after the retire.
        scheduler._offer_completion(
            1, {"type": "pool_shutdown", "error": "old cohort fatal"}
        )

        assert await _await_true(lambda: scheduler._completion_inbox.empty())
        await asyncio.sleep(0.1)
        assert scheduler._scheduling_pool.completed == []
        assert scheduler.worker.fatals == []
        assert tracker.error is None

        # The next cohort's pump gets its own, later generation and routes.
        fresh_queue: Any = queue.Queue()
        scheduler.worker.response_queue = fresh_queue
        await asyncio.to_thread(scheduler.worker_generation_ready)
        assert scheduler._pump_generation == 3
        fresh_queue.put(_completion("req-2", "batch-1"))
        assert await _await_true(lambda: scheduler._scheduling_pool.completed)
        assert scheduler._scheduling_pool.completed == [("req-2", "batch-1")]

        await _shutdown(scheduler)

    asyncio.run(scenario())


def _load_intake_pool_module():
    """Load intake_pool.py by path: it is stdlib-only, while importing it as
    batchgen.server.intake_pool runs the package __init__ (GPU-only deps)."""
    import importlib.util
    import sys

    path = ROOT / "batchgen/server/intake_pool.py"
    spec = importlib.util.spec_from_file_location("_isolated_intake_pool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _intake_entries(module, batch_id: str, count: int) -> List[Any]:
    return [
        module.IntakeEntry(
            request_id=f"req-{idx}@{batch_id}",
            batch_id=batch_id,
            raw_request={"text": "prompt", "max_tokens": 4},
            priority=module.Priority.NORMAL,
        )
        for idx in range(count)
    ]


def test_failing_a_batch_purges_its_queued_intake_entries(tmp_path):
    """A batch persisted as failed must never reach a later worker generation:
    its entries still queued in the intake pool are discarded with it."""

    async def scenario():
        intake_mod = _load_intake_pool_module()
        scheduler = _scheduler(tmp_path)
        intake = intake_mod.IntakePool(max_capacity=100)
        scheduler._intake_pool = intake
        intake.submit_batch("batch-1", _intake_entries(intake_mod, "batch-1", 3))
        intake.submit_batch("batch-2", _intake_entries(intake_mod, "batch-2", 2))
        failing = SimpleNamespace(is_complete=False, is_failed=False, error=None)
        finished = SimpleNamespace(is_complete=True, is_failed=False, error=None)
        scheduler._scheduling_pool._batch_trackers.update(
            {"batch-1": failing, "batch-2": finished}
        )

        assert scheduler._fail_all_active_batches("worker died") == 1

        assert failing.error == "worker died"
        assert intake.size() == 2
        assert intake.batch_remaining("batch-1") == 0
        assert {entry.batch_id for entry in intake.drain(10)} == {"batch-2"}

    asyncio.run(scenario())


class _DrainSchedulingPoolStub(_SchedulingPoolStub):
    def __init__(self) -> None:
        super().__init__()
        self.allocated: List[str] = []

    def select_from_intake(self, intake, max_n=None):
        return intake.drain(max_n)

    def allocate_slot(self, request_id: str) -> int:
        self.allocated.append(request_id)
        return len(self.allocated) - 1

    def get_batch_tracker(self, batch_id: str):
        return self._batch_trackers.get(batch_id)


class _RecordingRequestQueue:
    def __init__(self) -> None:
        self.items: List[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def test_drain_discards_entries_of_an_inactive_batch_without_a_slot(
    tmp_path, caplog
):
    """Defense in depth: an entry of a failed, complete, or finalized batch
    that is still in the intake pool (e.g. requeued after the purge) is
    dropped by the drain, holds no scheduling slot, and is never admitted."""

    async def scenario():
        intake_mod = _load_intake_pool_module()
        scheduler = _scheduler(tmp_path)
        intake = intake_mod.IntakePool(max_capacity=100)
        scheduler._intake_pool = intake
        pool = _DrainSchedulingPoolStub()
        scheduler._scheduling_pool = pool
        scheduler._quiesced = threading.Event()
        scheduler._quiesce_ack = threading.Event()
        scheduler._pool_initialized = True
        scheduler.worker.request_queue = _RecordingRequestQueue()
        pool._batch_trackers.update({
            "live": SimpleNamespace(is_complete=False, is_failed=False),
            "failed": SimpleNamespace(is_complete=False, is_failed=True),
            "complete": SimpleNamespace(is_complete=True, is_failed=False),
        })
        intake.submit_batch("failed", _intake_entries(intake_mod, "failed", 2))
        intake.submit_batch("live", _intake_entries(intake_mod, "live", 1))
        intake.submit_batch("complete", _intake_entries(intake_mod, "complete", 1))
        intake.submit_batch("finalized", _intake_entries(intake_mod, "finalized", 1))

        task = asyncio.create_task(scheduler._drain_intake_loop())
        assert await _await_true(lambda: scheduler.worker.request_queue.items)
        assert await _await_true(intake.is_empty)
        await asyncio.sleep(0.3)
        scheduler._stopped.set()
        await asyncio.wait_for(task, timeout=5.0)

        admitted = [
            entry["request_id"]
            for msg in scheduler.worker.request_queue.items
            for entry in msg["entries"]
        ]
        assert admitted == ["req-0@live"]
        assert pool.allocated == ["req-0@live"]
        assert scheduler._stale_intake_batches == {"failed", "complete", "finalized"}

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    discards = [r for r in caplog.records if "Discarding" in r.getMessage()]
    assert len(discards) == 3
