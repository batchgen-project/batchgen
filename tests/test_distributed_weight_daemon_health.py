"""A latched distributed-weight daemon takes the worker pool out of serving.

The daemon's failure state used to be invisible to Python: a daemon that
latched while the pool was idle left /health at 200 and admission open until
the next streamed prefill fetched a remote module. The worker monitor now
polls the daemon's health with the worker processes, so a latch moves the
pool to workers_down (/health 503, POST /v1/batches 503, in-flight batches
failed, peers told through the down key) within one monitor interval.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_worker_pool_restart import _manager

ROOT = Path(__file__).resolve().parents[1]


class _Daemon:
    def __init__(self, **health) -> None:
        self._health = {
            "transport_failed": False,
            "failure": "",
            "cohort_epoch": 1,
            "cohort_state": "admitting",
            "cohort_faulted": False,
            "cohort_fault": "",
        }
        self._health.update(health)
        self.calls = []

    def health(self) -> str:
        self.calls.append("health")
        return json.dumps(self._health)

    def begin_cohort_swap(self) -> int:
        self.calls.append("begin")
        return self._health["cohort_epoch"]

    def finish_cohort_swap(self, timeout_s) -> str:
        self.calls.append("finish")
        return "{}"


class _ExitState:
    reason = None

    def set_failure(self, reason, exc):
        if self.reason is None:
            self.reason = reason
            return True
        return False


def _run_monitor_until_handled(manager, timeout_s=5.0):
    manager._monitor_interval_s = 0.01
    manager._join_lock = threading.Lock()
    manager.worker_process = SimpleNamespace(
        processes=[SimpleNamespace(join=lambda timeout: None, exitcode=None, pid=1)]
    )
    reconciles = []
    manager._reconcile = lambda reason, fail_in_flight: reconciles.append(
        (reason, fail_in_flight)
    )
    manager._worker_exit_state = _ExitState()
    monitor = threading.Thread(target=manager._monitor_worker_processes)
    monitor.start()
    deadline = time.monotonic() + timeout_s
    while manager.state == "serving" and time.monotonic() < deadline:
        time.sleep(0.01)
    manager._monitor_stop_event.set()
    monitor.join(5)
    assert not monitor.is_alive()
    return reconciles


def test_idle_transport_latch_takes_the_pool_out_of_serving(monkeypatch):
    manager = _manager(monkeypatch)
    daemon = _Daemon(transport_failed=True, failure="ucx get failed: remote access")
    manager.distributed_weight_daemon = daemon

    reconciles = _run_monitor_until_handled(manager)

    assert manager.state == "workers_down"
    reason = manager._worker_exit_state.reason
    assert "transport failed (cold restart required)" in reason
    assert "ucx get failed: remote access" in reason
    # In-flight batches are failed with the daemon's reason.
    assert reconciles == [(reason, True)]
    assert daemon.calls[0] == "health"


def test_cohort_fault_takes_the_pool_out_of_serving(monkeypatch):
    manager = _manager(monkeypatch)
    manager.distributed_weight_daemon = _Daemon(
        cohort_faulted=True, cohort_epoch=4, cohort_fault="timed out waiting for remote module"
    )

    _run_monitor_until_handled(manager)

    assert manager.state == "workers_down"
    assert re.search(
        r"failed worker cohort 4: timed out waiting for remote module",
        manager._worker_exit_state.reason,
    )


def test_unreadable_daemon_health_is_a_failure(monkeypatch):
    manager = _manager(monkeypatch)

    class Broken(_Daemon):
        def health(self):
            raise RuntimeError("daemon object destroyed")

    manager.distributed_weight_daemon = Broken()

    _run_monitor_until_handled(manager)

    assert manager.state == "workers_down"
    assert "health query failed" in manager._worker_exit_state.reason


def test_a_healthy_daemon_keeps_the_pool_serving(monkeypatch):
    manager = _manager(monkeypatch)
    daemon = _Daemon()
    manager.distributed_weight_daemon = daemon

    _run_monitor_until_handled(manager, timeout_s=0.3)

    assert manager.state == "serving"
    assert manager._worker_exit_state.reason is None
    assert daemon.calls.count("health") >= 2


def test_a_dead_worker_is_still_named_before_the_daemon(monkeypatch):
    manager = _manager(monkeypatch)
    manager.distributed_weight_daemon = _Daemon(cohort_faulted=True, cohort_fault="peer closed")
    manager._collect_worker_exit_reason = lambda: "Detected worker process exit: idx=2"

    _run_monitor_until_handled(manager)

    assert manager._worker_exit_state.reason.startswith("Detected worker process exit")


def test_replicated_weights_have_no_daemon_to_poll(monkeypatch):
    manager = _manager(monkeypatch)
    manager.distributed_weight_daemon = None
    assert manager._distributed_weight_daemon_failure() is None


def test_native_daemon_exposes_its_failure_state():
    """The binding the monitor calls exists and reports both latches."""
    binding = (ROOT / "core/batchgen_Binding.cpp").read_text()
    assert re.search(
        r'\.def\("health",\s*&DistributedWeightDaemon::Health,\s*'
        r"py::call_guard<py::gil_scoped_release>\(\)\)",
        binding,
    )
    source = (ROOT / "core/Weights_Storage/distributed_weight_daemon.cpp").read_text()
    body = source[source.index("std::string Health() {"):]
    body = body[: body.index("return health.dump();")]
    for key in (
        "transport_failed", "failure", "cohort_epoch",
        "cohort_state", "cohort_faulted", "cohort_fault",
    ):
        assert f'health["{key}"]' in body, key
