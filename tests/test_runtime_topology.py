"""Runtime topology and communication ranges must remain lane-scoped."""

from __future__ import annotations

import ast
import copy
import os
import pickle
import signal
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"
HTTP_SERVER = ROOT / "batchgen" / "server" / "http_server.py"


def _top_level_function(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _worker_method(name: str, globals_=None):
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = copy.deepcopy(
        next(
            node
            for node in worker.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="IsolatedWorker",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace["IsolatedWorker"]


def _worker_manager_method(name: str, globals_=None):
    tree = ast.parse(WORKER_MANAGER.read_text(), filename=str(WORKER_MANAGER))
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerManager"
    )
    method = copy.deepcopy(
        next(
            node
            for node in manager.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="IsolatedManager",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(globals_ or {})
    exec(
        compile(ast.fix_missing_locations(module), str(WORKER_MANAGER), "exec"),
        namespace,
    )
    return namespace["IsolatedManager"]


def test_local_world_size_requires_exact_division_and_visibility():
    resolve = _top_level_function(WORKER_MANAGER, "_resolve_local_world_size")

    assert resolve(8, 1, 8) == 8
    assert resolve(8, 2, 8) == 4
    assert resolve(8, 4, 2, require_exact_visibility=True) == 2
    assert resolve(8, 8, 1, require_exact_visibility=True) == 1

    with pytest.raises(ValueError, match="divisible"):
        resolve(7, 2, 8)
    with pytest.raises(ValueError, match="only 1 GPUs are visible"):
        resolve(4, 1, 1)
    with pytest.raises(ValueError, match="exactly 2 visible GPUs"):
        resolve(4, 2, 8, require_exact_visibility=True)


def test_pynccl_search_never_leaves_assigned_range():
    calls = []

    def _find_available_port(host, start_port, max_attempts):
        calls.append((host, start_port, max_attempts))
        return start_port

    worker_type = _worker_method(
        "_find_available_pynccl_port",
        {"_find_available_port": _find_available_port},
    )
    worker = worker_type()
    worker.pynccl_port_base = 21000
    worker.pynccl_port_span = 4

    assert worker._find_available_pynccl_port("127.0.0.1", 20900) == 21000
    assert calls[-1] == ("127.0.0.1", 21000, 4)
    assert worker._find_available_pynccl_port("127.0.0.1", 21003) == 21003
    assert calls[-1] == ("127.0.0.1", 21003, 1)

    with pytest.raises(RuntimeError, match="range exhausted"):
        worker._find_available_pynccl_port("127.0.0.1", 21004)


def test_worker_rank_math_uses_configured_local_world_size():
    main_loop = (
        ROOT / "batchgen" / "server_worker_main_loop.py"
    ).read_text()
    worker_source = WORKER.read_text()
    migration_source = (ROOT / "batchgen" / "migration.py").read_text()

    assert "args.local_world_size * args.nnode_rank" in main_loop
    assert "NUM_GPUS_PER_NODE" not in worker_source
    assert "NUM_GPUS_PER_NODE" not in migration_source


def test_http_shutdown_has_no_host_global_cleanup_fallback():
    http_source = HTTP_SERVER.read_text()
    manager_source = WORKER_MANAGER.read_text()

    assert "shm_prefix=None" not in http_source
    assert "clean_hugepages=True" not in http_source
    assert "if self._runtime_namespace_owned:" in manager_source
    assert "shm_prefix=(" in manager_source
    assert "self.args.runtime_identity.resource_prefix" in manager_source
    assert "kill_workers=False" in manager_source
    assert http_source.index("worker._acquire_runtime_admission()") < (
        http_source.index("StorageManager(server_args.storage_path)")
    )


def test_worker_start_rolls_back_partial_startup_before_reraising():
    events = []
    manager_type = _worker_manager_method(
        "start",
        {"logger": type("Logger", (), {"exception": lambda *args: None})()},
    )
    manager = manager_type()
    manager.started = False

    def fail_start():
        events.append("start")
        raise RuntimeError("startup failed")

    manager._start_impl = fail_start
    manager.stop = lambda: events.append("stop")

    with pytest.raises(RuntimeError, match="startup failed"):
        manager.start()
    assert events == ["start", "stop"]


def test_worker_stop_preserves_artifacts_and_locks_for_live_owned_pid(
    tmp_path,
):
    events = []
    fake_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    manager_type = _worker_manager_method(
        "stop",
        {
            "cleanup_resources": lambda **kwargs: events.append("cleanup"),
            "logger": fake_logger,
        },
    )
    manager = manager_type()
    manager._stopping = False
    manager.started = True
    manager._runtime_dir_created = True
    manager._runtime_namespace_owned = True
    manager._runtime_locks = SimpleNamespace(
        close=lambda: events.append("runtime-lock-close")
    )
    manager._lane_lease = SimpleNamespace(
        close=lambda: events.append("lane-lease-close")
    )
    manager.worker_process = SimpleNamespace(processes=[SimpleNamespace(pid=123)])
    manager.distributed_weight_daemon = None
    manager.model_info = {}
    manager.skeleton_state_dict_file = None
    manager._monitor_stop_event = SimpleNamespace(set=lambda: None)
    manager._monitor_thread = None
    manager._stop_workers = lambda: (_ for _ in ()).throw(
        RuntimeError("worker teardown left live owned PIDs")
    )
    manager.request_queue = SimpleNamespace(put=lambda value: None)
    manager._join_lock = nullcontext()
    manager._cleanup_skeleton_state_dict_file = lambda: None
    manager.args = SimpleNamespace(
        runtime_identity=SimpleNamespace(
            resource_prefix="batchgen_lane-0_run",
            runtime_dir=tmp_path / "runtime",
        )
    )
    manager._hugepages_enabled = False

    with pytest.raises(RuntimeError, match="live owned PIDs"):
        manager.stop()

    assert "cleanup" not in events
    assert "runtime-lock-close" not in events
    assert "lane-lease-close" not in events
    assert manager._runtime_dir_created
    assert manager._runtime_namespace_owned
    assert manager._runtime_locks is not None
    assert manager._lane_lease is not None


@pytest.mark.parametrize("exits_after_term", [True, False])
def test_worker_stop_signals_only_original_child_handles(exits_after_term):
    events = []

    class Child:
        pid = 123
        exitcode = None

        def join(self, timeout):
            events.append(("join", timeout))
            if exits_after_term and ("signal", 9, 15) in events:
                self.exitcode = 0

    child = Child()
    fake_os = SimpleNamespace(
        pidfd_open=lambda pid: events.append(("open", pid)) or 9,
        close=lambda fd: events.append(("close", fd)),
    )
    fake_signal = SimpleNamespace(
        SIGTERM=15,
        SIGKILL=9,
        pidfd_send_signal=lambda fd, sig: events.append(("signal", fd, sig)),
    )
    fake_logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    manager_type = _worker_manager_method(
        "_stop_workers",
        {
            "logger": fake_logger,
            "os": fake_os,
            "signal": fake_signal,
            "time": __import__("time"),
        },
    )
    manager = manager_type()
    manager.worker_process = SimpleNamespace(processes=[child])
    manager._join_lock = nullcontext()
    manager.request_queue = SimpleNamespace(put=lambda value: events.append("poison"))

    if exits_after_term:
        manager._stop_workers()
        assert ("signal", 9, 9) not in events
    else:
        with pytest.raises(RuntimeError, match="live owned PIDs"):
            manager._stop_workers()
        assert ("signal", 9, 9) in events
    assert events[0:2] == [("open", 123), ("signal", 9, 15)]
    assert events[-1] == ("close", 9)


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
    reason="Linux PIDFD support required",
)
def test_worker_stop_real_child_pidfd():
    child_process = subprocess.Popen(["sleep", "30"])

    class Child:
        pid = child_process.pid

        @property
        def exitcode(self):
            return child_process.poll()

        def join(self, timeout):
            try:
                child_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

    manager_type = _worker_manager_method(
        "_stop_workers",
        {
            "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
            "os": os,
            "signal": signal,
            "time": time,
        },
    )
    manager = manager_type()
    manager.worker_process = SimpleNamespace(processes=[Child()])
    manager._join_lock = nullcontext()
    manager.request_queue = SimpleNamespace(put=lambda value: None)
    try:
        manager._stop_workers()
        assert child_process.poll() is not None
    finally:
        if child_process.poll() is None:
            child_process.kill()
        child_process.wait()


def test_worker_monitor_does_not_invoke_context_auto_kill():
    events = []

    class StopEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            self.stopped = True

    manager_type = _worker_manager_method("_monitor_worker_processes")
    manager = manager_type()
    manager._monitor_stop_event = StopEvent()
    manager.worker_process = SimpleNamespace(
        processes=[SimpleNamespace(join=lambda timeout: events.append("child-join"))],
        join=lambda timeout: events.append("context-join"),
    )
    manager._join_lock = nullcontext()
    manager._monitor_interval_s = 1
    manager._collect_worker_exit_reason = lambda: None

    manager._monitor_worker_processes()

    assert events == ["child-join"]


def test_worker_exit_reason_preserves_python_traceback(tmp_path):
    error_file = tmp_path / "worker-error.pickle"
    error_file.write_bytes(pickle.dumps("Traceback: worker ValueError"))
    manager_type = _worker_manager_method(
        "_collect_worker_exit_reason",
        {"Optional": Optional, "os": os, "pickle": pickle},
    )
    manager = manager_type()
    manager.worker_process = SimpleNamespace(
        processes=[SimpleNamespace(pid=123, exitcode=1)],
        error_files=[str(error_file)],
    )

    reason = manager._collect_worker_exit_reason()

    assert "idx=0 pid=123 exitcode=1" in reason
    assert "Traceback: worker ValueError" in reason
