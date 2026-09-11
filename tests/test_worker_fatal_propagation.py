"""Distributed worker fatals must fail active pool batches with the traceback."""

import ast
import asyncio
import copy
import logging
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MAIN_LOOP = ROOT / "batchgen" / "server_worker_main_loop.py"
SCHEDULER = ROOT / "batchgen" / "server" / "batch_scheduler.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"


def _function(path, name, class_name=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    body = tree.body
    if class_name is not None:
        body = next(
            node for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    return copy.deepcopy(next(
        node for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ))


def _isolated_functions(path, names, globals_):
    module = ast.Module(
        body=[_function(path, name) for name in names],
        type_ignores=[],
    )
    namespace = dict(globals_)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return [namespace[name] for name in names]


def _isolated_scheduler():
    methods = [
        _function(SCHEDULER, "_fail_all_active_batches", "BatchScheduler"),
        _function(SCHEDULER, "_pool_completion_listener", "BatchScheduler"),
    ]
    module = ast.Module(
        body=[ast.ClassDef(
            name="IsolatedBatchScheduler",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )],
        type_ignores=[],
    )
    namespace = {
        "asyncio": asyncio,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.fix_missing_locations(module), str(SCHEDULER), "exec"), namespace)
    return namespace["IsolatedBatchScheduler"]


def test_first_python_fatal_is_published_atomically(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.calls = []

        def compare_set(self, key, expected, desired):
            self.calls.append((key, expected, desired))
            return desired.encode()

    store = FakeStore()
    import torch.distributed.distributed_c10d as c10d
    monkeypatch.setattr(c10d, "_get_default_store", lambda: store)
    (publish,) = _isolated_functions(
        MAIN_LOOP,
        ["_publish_worker_fatal_to_store"],
        {
            "logging": logging,
            "_WORKER_FATAL_STORE_KEY": "batchgen_worker_fatal_v1",
        },
    )

    publish("rank-8 traceback")

    assert store.calls == [
        ("batchgen_worker_fatal_v1", "", "rank-8 traceback")
    ]


def test_pool_mode_exception_reaches_outer_fatal_boundary():
    impl = _function(MAIN_LOOP, "_server_worker_main_impl")
    pool_guard = next(
        node for node in ast.walk(impl)
        if isinstance(node, ast.If)
        and "Pool mode init received" in ast.unparse(node)
    )
    handler = next(
        handler for node in ast.walk(pool_guard)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if "Error in pool mode" in ast.unparse(handler)
    )

    assert any(isinstance(node, ast.Raise) for node in handler.body)
    assert "response_queue.put" not in ast.unparse(handler)


def test_pool_shutdown_error_fails_storage_before_acknowledgement():
    trace = []
    tracker = SimpleNamespace(is_complete=False, is_failed=False, error=None)

    class ResponseQueue:
        def get(self, timeout):
            trace.append(("queue_get", timeout))
            return {"type": "pool_shutdown", "error": "rank 8\nTraceback"}

    class Worker:
        response_queue = ResponseQueue()

        def report_worker_fatal(self, error):
            trace.append(("worker_ack", error))

    scheduler_type = _isolated_scheduler()
    scheduler = scheduler_type()
    scheduler._stopped = SimpleNamespace(is_set=lambda: False)
    scheduler.worker = Worker()
    scheduler._scheduling_pool = SimpleNamespace(
        _batch_trackers={"batch-1": tracker},
    )
    scheduler.storage = SimpleNamespace(
        update_batch=lambda batch_id, **kwargs: trace.append(
            ("storage", batch_id, kwargs)
        ),
    )

    asyncio.run(scheduler._pool_completion_listener())

    assert tracker.error == "rank 8\nTraceback"
    storage_index = next(i for i, item in enumerate(trace) if item[0] == "storage")
    ack_index = next(i for i, item in enumerate(trace) if item[0] == "worker_ack")
    assert storage_index < ack_index
    assert trace[storage_index][2] == {
        "status": "failed",
        "error": {
            "code": "worker_fatal",
            "message": "rank 8\nTraceback",
        },
    }


def test_worker_manager_acknowledges_before_requesting_shutdown():
    method = _function(WORKER_MANAGER, "report_worker_fatal", "WorkerManager")
    module = ast.Module(
        body=[ast.ClassDef(
            name="IsolatedWorkerManager",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )],
        type_ignores=[],
    )
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(module), str(WORKER_MANAGER), "exec"),
        namespace,
    )
    manager = namespace["IsolatedWorkerManager"]()
    trace = []
    manager._fatal_ack_event = SimpleNamespace(
        set=lambda: trace.append("ack")
    )
    manager._handle_worker_failure = lambda reason, exc: trace.append(
        ("shutdown", reason, exc)
    )

    manager.report_worker_fatal("exact traceback")

    assert trace == ["ack", ("shutdown", "exact traceback", None)]
