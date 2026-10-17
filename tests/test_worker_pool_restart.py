"""Dev-machine lifecycle tests for the restartable worker pool."""

from __future__ import annotations

import ast
import logging
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_worker_manager_class():
    source = (ROOT / "batchgen/server/worker_manager.py").read_text()
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "WorkerManager"
    )
    namespace = {
        "Any": Any,
        "Callable": Callable,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "threading": threading,
        "time": time,
        "logging": logging,
        "logger": logging.getLogger(__name__),
        "cleanup_shm_files": lambda prefix: None,
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(ROOT / "batchgen/server/worker_manager.py"), "exec"), namespace)
    return namespace["WorkerManager"]


def _manager(monkeypatch):
    cls = _load_worker_manager_class()
    manager = object.__new__(cls)
    manager._lock = threading.Lock()
    manager._state = "serving"
    manager._generation = 3
    manager._first_ready = True
    manager._reconciled_failure = False
    manager._main_thread_dispatcher = None
    manager._quiesce_hook = lambda: None
    manager._resume_hook = lambda: None
    manager._fail_in_flight_hook = lambda reason: 2
    manager._scheduler_generation_hook = lambda: None
    manager._pool_init_params_hook = lambda: {
        "max_output_len": 8192, "max_context_length": 65536,
    }
    manager._restart_timeout_s = 10.0
    manager._monitor_stop_event = threading.Event()
    manager._stopping = False
    manager._ready_event = threading.Event()
    manager._fatal_ack_event = threading.Event()
    manager._monitor_thread = None
    manager._worker_exit_state = SimpleNamespace(clear=lambda: None)
    manager.request_queue = queue.Queue()
    manager.response_queue = queue.Queue()
    manager.worker_process = SimpleNamespace(processes=[])
    manager.host_kv_manager = SimpleNamespace(release_all_sequences=lambda: None)
    manager.host_kv_aux_manager = SimpleNamespace(release_all_sequences=lambda: None)
    manager.args = SimpleNamespace(runtime_identity=SimpleNamespace(shm_prefix="run."))
    manager._stop_workers = lambda: None
    manager._spawn_workers = lambda: None
    manager._start_worker_monitor = lambda: None
    manager._wait_for_workers_ready = lambda timeout_s=None: None
    return manager


def test_restart_state_and_generation(monkeypatch):
    manager = _manager(monkeypatch)
    phases = []
    manager._quiesce_hook = lambda: phases.append("quiesce")
    manager._resume_hook = lambda: phases.append("resume")
    manager._stop_workers = lambda: phases.append("teardown")
    manager._spawn_workers = lambda: phases.append("spawn")

    result = manager.restart_workers(timeout_s=1)

    assert manager.state == "serving"
    assert manager.generation == 4
    assert result["generation"] == 4
    assert result["failed_batches"] == 2
    assert phases == ["quiesce", "teardown", "spawn", "resume"]


def test_restart_conflict_is_rejected(monkeypatch):
    manager = _manager(monkeypatch)
    manager._state = "restarting"
    with pytest.raises(RuntimeError, match="restart conflict"):
        manager.restart_workers(timeout_s=1)


def test_queue_drain_and_release_all_after_teardown(monkeypatch):
    manager = _manager(monkeypatch)
    order = []
    manager.request_queue.put("stale-request")
    manager.response_queue.put("stale-response")
    manager.host_kv_manager.release_all_sequences = lambda: order.append("primary")
    manager.host_kv_aux_manager.release_all_sequences = lambda: order.append("aux")

    def teardown():
        order.append("teardown")

    manager._stop_workers = teardown
    original_release = manager._release_host_kv_sequences

    def reconcile_resources():
        order.append("reconcile")
        original_release()

    manager._release_host_kv_sequences = reconcile_resources
    manager._spawn_workers = lambda: order.append("spawn")
    manager.restart_workers(timeout_s=1)

    assert order == ["teardown", "reconcile", "primary", "aux", "spawn"]
    init_msg = manager.request_queue.get_nowait()
    assert init_msg["type"] == "init"
    # Pool capacity must come from the scheduler hook, never defaults.
    assert init_msg["max_output_len"] == 8192
    assert init_msg["max_context_length"] == 65536
    assert manager.request_queue.empty()
    assert manager.response_queue.empty()


def test_spawn_dispatch_from_worker_thread_runs_on_main_thread(monkeypatch):
    manager = _manager(monkeypatch)
    pending = queue.Queue()
    finished = threading.Event()
    observed = []

    def dispatcher(fn, timeout):
        pending.put(fn)
        assert finished.wait(timeout)
        return observed[-1]

    manager.set_main_thread_dispatcher(dispatcher)
    worker = threading.Thread(
        target=lambda: manager._call_spawn_on_main_thread(
            lambda: observed.append(threading.current_thread() is threading.main_thread()), 1
        )
    )
    worker.start()
    pending.get(timeout=1)()
    finished.set()
    worker.join(timeout=1)
    assert observed == [True]


def test_post_ready_failure_enters_workers_down_without_server_signal(monkeypatch):
    manager = _manager(monkeypatch)
    shutdowns = []
    reconciles = []

    class ExitState:
        reason = None

        def set_failure(self, reason, exc):
            self.reason = reason
            return True

    manager._worker_exit_state = ExitState()
    manager._reconcile = lambda reason, fail_in_flight: reconciles.append(
        (reason, fail_in_flight)
    )
    cls = type(manager)
    monkeypatch.setattr(cls, "_request_server_shutdown", staticmethod(lambda: shutdowns.append(True)))

    manager._handle_worker_failure("worker exited", None)

    assert manager.state == "workers_down"
    assert reconciles == [("worker exited", True)]
    assert shutdowns == []


def test_startup_failure_keeps_fail_fast_signal(monkeypatch):
    manager = _manager(monkeypatch)
    manager._first_ready = False
    shutdowns = []

    class ExitState:
        reason = None

        def set_failure(self, reason, exc):
            self.reason = reason
            return True

    manager._worker_exit_state = ExitState()
    cls = type(manager)
    monkeypatch.setattr(cls, "_request_server_shutdown", staticmethod(lambda: shutdowns.append(True)))

    manager._handle_worker_failure("startup worker exited", None)

    assert shutdowns == [True]
    assert manager.state == "serving"


def test_explicit_restart_does_not_double_persist_prior_pool_shutdown(monkeypatch):
    manager = _manager(monkeypatch)
    calls = []
    manager._reconciled_failure = True
    manager._fail_in_flight_hook = lambda reason: calls.append(reason) or 1
    manager._spawn_workers = lambda: None

    manager.restart_workers(timeout_s=1)

    assert calls == []


def test_legacy_reload_symbols_are_absent():
    symbols = (
        "send_" + "reload_command",
        "send_" + "pool_reload",
        "_handle_" + "hot_reload",
        "_write_" + "reload_status",
        "_reload_" + "worker_module",
        "_rebind_" + "all_methods",
        "_validate_" + "reload",
    )
    text = "\n".join(
        path.read_text()
        for path in (ROOT / "batchgen").rglob("*.py")
    )
    text += "\n" + "\n".join(
        path.read_text()
        for path in (ROOT / "tests").rglob("*.py")
        if path.name != Path(__file__).name
    )
    for symbol in symbols:
        assert symbol not in text
    http = (ROOT / "batchgen/server/http_server.py").read_text()
    assert '@app.post("/v1/reload")' not in http
    assert '@app.post("/v1/workers/restart")' in http
    assert '@app.get("/v1/workers/status")' in http


def test_status_broadcast_contract_has_two_slots():
    text = (ROOT / "batchgen/batchgen_worker.py").read_text()
    assert "status[2]" not in text
    assert "[has_new_work, shutdown]" in text
    assert "[1 if has_new else 0, 1 if self._shutdown_requested else 0]" in text


def test_release_all_binding_walks_metadata_without_memset():
    source = (ROOT / "core/KV_Storage/host_paged_kv_backend.cpp").read_text()
    body = source.split(
        "void HostPagedKVBackend::SharedState::ReleaseAllSequences()", 1
    )[1].split("std::vector<std::int32_t>", 1)[0]
    assert "sequence_table" in body
    assert "ReleaseSequence(sequence_id)" in body
    assert "memset" not in body
