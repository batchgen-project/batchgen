"""The server's stop lifecycle: a stop request reaches start-up, and a stop
while serving tears down with HTTP still answering.

uvicorn replaces SIGINT/SIGTERM for its whole run and only records that it
should exit; a start-up running inside the lifespan never looked at that, so a
SIGTERM during weight load or store staging was deferred until start-up ended
and the workers had been spawned. And the lifespan's teardown ran on the event
loop after the listener closed, so for the minutes a large cohort takes to
unpin its host memory nothing could ask the server what it was doing.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

uvicorn = pytest.importorskip("uvicorn")

from tests.test_worker_pool_restart import _manager, lifecycle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HTTP_SERVER_PATH = ROOT / "batchgen" / "server" / "http_server.py"


def _http_tree() -> ast.Module:
    return ast.parse(HTTP_SERVER_PATH.read_text(), filename=str(HTTP_SERVER_PATH))


def _load_http_classes():
    """ServingShutdown and _BatchGenServer as written, without the server
    stack importing http_server.py would pull in."""
    tree = _http_tree()
    classes = [
        copy.deepcopy(node) for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"ServingShutdown", "_BatchGenServer"}
    ]
    assert len(classes) == 2
    namespace = {
        "asyncio": asyncio,
        "threading": threading,
        "logging": logging,
        "logger": logging.getLogger(__name__),
        "uvicorn": uvicorn,
        "Optional": Optional,
        "StopRequest": lifecycle.StopRequest,
    }
    module = ast.Module(body=classes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(HTTP_SERVER_PATH), "exec"),
         namespace)
    return namespace["ServingShutdown"], namespace["_BatchGenServer"]


ServingShutdown, BatchGenServer = _load_http_classes()


# ----------------------------------------------------------------- serving stop


def test_serving_teardown_runs_while_the_event_loop_keeps_answering():
    """The worker teardown blocks for as long as unpinning takes; the event
    loop must keep serving requests (here: probes) the whole time, and the
    HTTP server may exit only after teardown finished."""
    release = threading.Event()
    done: List[str] = []
    probes: List[float] = []

    async def scenario():
        shutdown = ServingShutdown(lifecycle.StopRequest())

        async def teardown():
            await asyncio.to_thread(release.wait, 10)

        shutdown.bind(asyncio.get_running_loop(), teardown)
        assert shutdown.begin("SIGTERM", lambda: done.append("exit"))
        assert shutdown.stop_request.is_set()
        assert shutdown.stop_request.reason == "SIGTERM"
        for _ in range(5):
            await asyncio.sleep(0.02)
            probes.append(time.monotonic())
        assert done == []  # teardown still running, listener stays up
        release.set()
        for _ in range(100):
            if done:
                break
            await asyncio.sleep(0.02)

    asyncio.run(scenario())
    assert len(probes) == 5
    assert done == ["exit"]


def test_stop_before_serving_or_a_second_stop_exits_at_once():
    shutdown = ServingShutdown(lifecycle.StopRequest())
    # Start-up: nothing is bound yet; start-up observes the request itself.
    assert not shutdown.begin("SIGTERM", lambda: None)
    assert shutdown.stop_request.is_set()

    async def scenario():
        bound = ServingShutdown(lifecycle.StopRequest())

        async def teardown():
            return None

        bound.bind(asyncio.get_running_loop(), teardown)
        assert bound.begin("SIGTERM", lambda: None)
        # A second signal means "stop answering now".
        assert not bound.begin("SIGINT", lambda: None)

    asyncio.run(scenario())


def test_uvicorn_exit_is_deferred_until_teardown_finished():
    release = threading.Event()

    async def scenario():
        shutdown = ServingShutdown(lifecycle.StopRequest())
        server = BatchGenServer(uvicorn.Config(app=lambda *a: None), shutdown)

        async def teardown():
            await asyncio.to_thread(release.wait, 10)

        shutdown.bind(asyncio.get_running_loop(), teardown)
        server.handle_exit(15, None)
        await asyncio.sleep(0.05)
        assert not server.should_exit
        release.set()
        for _ in range(100):
            if server.should_exit:
                break
            await asyncio.sleep(0.02)
        assert server.should_exit
        # The signal is re-raised by uvicorn after it stops, as before.
        assert server._captured_signals == [15]

    asyncio.run(scenario())


def test_uvicorn_exits_at_once_while_starting():
    shutdown = ServingShutdown(lifecycle.StopRequest())
    server = BatchGenServer(uvicorn.Config(app=lambda *a: None), shutdown)
    server.handle_exit(15, None)
    assert server.should_exit
    assert shutdown.stop_request.reason == "SIGTERM"


def test_lifespan_wires_the_stop_lifecycle():
    """The lifespan binds the serving teardown only once start-up completed,
    runs the worker stop off the event loop, treats a cancelled start-up as a
    stop (not a failure), and the launcher runs the lifecycle-aware server."""
    source = HTTP_SERVER_PATH.read_text()
    lifespan = ast.unparse(next(
        node for node in ast.walk(_http_tree())
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    ))
    assert "await asyncio.to_thread(worker.stop)" in lifespan
    assert "worker.stop()" not in lifespan.replace("asyncio.to_thread(worker.stop)", "")
    assert lifespan.index("health_state.mark_startup_complete()") < lifespan.index(
        "shutdown.bind(asyncio.get_running_loop(), teardown)"
    )
    assert "except StartupCancelled" in lifespan
    assert "stop_request=shutdown.stop_request" in lifespan
    assert "_BatchGenServer(config, shutdown).run()" in source
    assert "uvicorn.run(" not in source


def test_health_reports_shutting_down():
    health = ast.unparse(next(
        node for node in ast.walk(_http_tree())
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "health_check"
    ))
    assert "'shutting_down'" in health
    assert health.index("pool_state == 'stopping'") < health.index("workers_down")


# --------------------------------------------------------------- start-up stop


def _starting_manager(monkeypatch, *, stop_during: str):
    manager = _manager(monkeypatch)
    namespace = type(manager).__init__.__globals__
    events: List[str] = []
    stop = lifecycle.StopRequest()
    manager._stop_request = stop
    manager.started = False
    manager._first_ready = False
    manager.distributed_weight_daemon = None
    manager.args = SimpleNamespace(
        fast_init=False,
        distributed_weight_config=None,
        enable_hugetlbfs=False,
        host_kv_cache_size=None,
        enable_prefix_cache=False,
        runtime_identity=SimpleNamespace(shm_prefix="run.", runtime_dir="/tmp/x"),
        nnodes=1,
        node_rank=0,
    )
    monkeypatch.setitem(namespace, "config_torch_module_initializer", lambda: None)
    monkeypatch.setitem(namespace, "torch", SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: 0)
    ))

    def phase(name):
        def run(*args, **kwargs):
            events.append(name)
            if name == stop_during:
                stop.request("SIGTERM")
        return run

    for name in (
        "_acquire_runtime_admission",
        "_prepare_runtime_dir",
        "_start_startup_phase_watchdog",
        "_stop_startup_phase_watchdog",
        "_exchange_run_identity",
        "_start_restart_follower",
        "_load_model_resources",
        "_spawn_workers",
        "_start_worker_monitor",
        "_shutdown_prefix_cache_owner",
    ):
        setattr(manager, name, phase(name))
    return manager, events


def test_a_stop_during_weight_load_spawns_no_worker(monkeypatch):
    manager, events = _starting_manager(monkeypatch, stop_during="_load_model_resources")
    with pytest.raises(lifecycle.StartupCancelled, match="SIGTERM") as raised:
        manager._start_impl()
    assert raised.value.phase == "spawn"
    assert "_load_model_resources" in events
    assert "_spawn_workers" not in events


def test_a_stop_before_start_up_loads_nothing(monkeypatch):
    manager, events = _starting_manager(
        monkeypatch, stop_during="_acquire_runtime_admission"
    )
    with pytest.raises(lifecycle.StartupCancelled):
        manager._start_impl()
    assert "_load_model_resources" not in events


def test_a_stop_while_waiting_for_readiness_unwinds(monkeypatch):
    manager, events = _starting_manager(monkeypatch, stop_during="_start_worker_monitor")
    manager._monitor_interval_s = 0.01
    manager.worker_process = SimpleNamespace(processes=[])
    manager._collect_worker_exit_reason = lambda: None
    manager._worker_exit_state = SimpleNamespace(is_failed=lambda: False)
    # The real readiness wait, not _manager's stub.
    del manager._wait_for_workers_ready
    with pytest.raises(lifecycle.StartupCancelled) as raised:
        manager._start_impl()
    assert raised.value.phase == "worker-ready"
    # The watchdog is stopped whether the cohort became ready or not.
    assert events[-1] == "_stop_startup_phase_watchdog"


def test_a_stopping_pool_reports_it_and_refuses_a_restart(monkeypatch):
    manager = _manager(monkeypatch)
    manager._stop_request = lifecycle.StopRequest()
    assert manager.state == "serving"
    manager._stop_request.request("SIGTERM")
    assert manager.state == "stopping"
    assert manager.stop_reason == "SIGTERM"
    with pytest.raises(RuntimeError, match="restart conflict: state=stopping"):
        manager.restart_workers(timeout_s=1)
    assert manager._state == "serving"
