"""A failed distributed-weight daemon fails the pool and /health, while idle.

CPU only. The daemon is the server process's own C++ object and never calls
back into Python; the worker monitor reads its status on every poll. The
manager is the lifted ``WorkerManager`` the restart tests use, the daemon a
fake that reports the native ``status()`` JSON, and /health the real route
body mounted on a bare app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_worker_pool_restart import (  # noqa: E402
    _FakeWeightDaemon,
    _manager,
)
from tests.unit.test_cluster_fail_fast import (  # noqa: E402
    _ExitState,
    _mount_health_route,
)

DAEMON_CPP = ROOT / "core" / "Weights_Storage" / "distributed_weight_daemon.cpp"
DAEMON_H = ROOT / "core" / "Weights_Storage" / "distributed_weight_daemon.h"
BINDING_CPP = ROOT / "core" / "batchgen_Binding.cpp"

# A monitor that never fails the pool would poll forever; every test bounds it.
_MAX_POLLS = 12


class _StatusDaemon(_FakeWeightDaemon):
    """The swap protocol of the restart tests plus the native status().

    ``on_poll(daemon, poll)`` runs before each report, so a test can latch
    the daemon at a chosen poll. A transport failure is permanent, as in the
    daemon: finish refuses it; a cohort fault is cleared by a finished swap.
    """

    def __init__(self, order, *, on_poll=None):
        super().__init__(order)
        self.on_poll = on_poll
        self.polls = 0
        self.failed = False
        self.failure = ""
        self.cohort_faulted = False
        self.cohort_fault = ""
        self.status_error = None

    def status(self):
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self, self.polls)
        if self.status_error is not None:
            raise self.status_error
        return json.dumps({
            "network_ready": True,
            "failed": self.failed,
            "failure": self.failure,
            "failure_class": "transport_fatal" if self.failed else "none",
            "cohort": {
                "epoch": self.epoch,
                "state": "draining" if self.draining else "admitting",
                "faulted": self.cohort_faulted,
                "fault": self.cohort_fault,
                "fault_class": (
                    "cohort_recoverable" if self.cohort_faulted else "none"
                ),
            },
        })

    def finish_cohort_swap(self, timeout_seconds):
        if self.failed:
            self.order.append("finish")
            raise RuntimeError(
                "distributed weights transport failure is not recoverable by "
                "a worker restart (cold restart required): " + self.failure
            )
        report = super().finish_cohort_swap(timeout_seconds)
        self.cohort_faulted = False
        self.cohort_fault = ""
        return report


def _serving_manager(monkeypatch, *, on_poll=None, node_rank=1):
    order = []
    manager = _manager(monkeypatch)
    manager.args.node_rank = node_rank
    manager._monitor_interval_s = 0.0
    manager._join_lock = __import__("threading").Lock()
    manager._worker_exit_state = _ExitState()
    published = []
    manager._restart_coordinator = SimpleNamespace(
        enabled=False,
        epoch=0,
        control_error=None,
        publish_down=lambda generation, reason: published.append(
            (generation, reason)
        ),
    )
    failed_batches = []
    manager._fail_in_flight_hook = (
        lambda reason: failed_batches.append(reason) or 1
    )
    manager._quiesce_hook = lambda: order.append("quiesce")
    manager._stop_workers = lambda: order.append("teardown")
    manager.distributed_weight_daemon = _StatusDaemon(order, on_poll=on_poll)
    # The bound sits on the worker-exit check, which every poll runs, so a
    # monitor that never probes the daemon ends after _MAX_POLLS too.
    collect_exit = manager._collect_worker_exit_reason
    polls = []

    def bounded_collect_exit():
        polls.append(None)
        if len(polls) >= _MAX_POLLS:
            manager._monitor_stop_event.set()
        return collect_exit()

    manager._collect_worker_exit_reason = bounded_collect_exit
    return manager, SimpleNamespace(
        order=order, published=published, failed_batches=failed_batches
    )


def _latch_transport_at(poll_to_latch):
    def on_poll(daemon, poll):
        if poll == poll_to_latch:
            daemon.failed = True
            daemon.failure = (
                "prefetch: ucp_get_nbx completion: Endpoint timeout"
            )

    return on_poll


_TRANSPORT_REASON = (
    "distributed weight daemon on node 1 failed (transport_fatal, cold "
    "restart required): prefetch: ucp_get_nbx completion: Endpoint timeout"
)


def test_a_healthy_daemon_keeps_the_pool_serving(monkeypatch):
    manager, seen = _serving_manager(monkeypatch)

    manager._monitor_worker_processes()

    daemon = manager.distributed_weight_daemon
    assert daemon.polls == _MAX_POLLS
    assert manager.state == "serving"
    assert not manager._worker_exit_state.is_failed()
    assert seen.order == []
    assert seen.published == []
    status = manager.status()["distributed_weights"]
    assert status["failed"] is False
    assert status["cohort"]["faulted"] is False


def test_health_is_200_while_the_daemon_is_healthy(monkeypatch):
    pytest.importorskip("fastapi")
    manager, _seen = _serving_manager(monkeypatch)
    manager._monitor_worker_processes()

    client = _mount_health_route(manager, manager._worker_exit_state)
    assert client.get("/health").status_code == 200


def test_an_idle_transport_latch_fails_the_pool(monkeypatch):
    """No prefill runs: only the monitor's probe can see the latch."""
    manager, seen = _serving_manager(
        monkeypatch, on_poll=_latch_transport_at(3)
    )

    manager._monitor_worker_processes()

    # Latched at poll 3, confirmed at poll 4, nothing polled after it.
    assert manager.distributed_weight_daemon.polls == 4
    assert manager.state == "workers_down"
    assert manager._worker_exit_state.reason == _TRANSPORT_REASON
    # The same ordered path as a worker death: peers told first, then the
    # cohort quiesced, swapped, torn down and its batches failed.
    assert seen.published == [(3, _TRANSPORT_REASON)]
    assert seen.order == ["quiesce", "begin", "teardown", "finish"]
    assert seen.failed_batches == [_TRANSPORT_REASON]


def test_health_names_an_idle_transport_latch(monkeypatch):
    pytest.importorskip("fastapi")
    manager, _seen = _serving_manager(
        monkeypatch, on_poll=_latch_transport_at(3)
    )
    manager._monitor_worker_processes()

    client = _mount_health_route(manager, manager._worker_exit_state)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy", "reason": _TRANSPORT_REASON,
    }


def test_a_transport_latch_is_not_recoverable_by_a_restart(monkeypatch):
    manager, _seen = _serving_manager(
        monkeypatch, on_poll=_latch_transport_at(1)
    )
    manager._monitor_worker_processes()
    assert manager.state == "workers_down"

    with pytest.raises(RuntimeError, match="cold restart required"):
        manager.restart_workers(timeout_s=1)
    assert manager.state == "workers_down"


def test_a_cohort_fault_fails_the_pool_and_a_restart_recovers(monkeypatch):
    def on_poll(daemon, poll):
        if poll == 2:
            daemon.cohort_faulted = True
            daemon.cohort_fault = (
                "worker 3 control connection failed: timed out waiting for "
                "remote module L5E700: the ring holds 32 ready modules this "
                "cohort did not consume"
            )

    manager, seen = _serving_manager(monkeypatch, on_poll=on_poll)

    manager._monitor_worker_processes()

    assert manager.state == "workers_down"
    reason = manager._worker_exit_state.reason
    assert reason.startswith(
        "distributed weight daemon on node 1 failed worker cohort 1 "
        "(cohort_recoverable, worker restart required): worker 3 control "
        "connection failed"
    )
    assert seen.published == [(3, reason)]

    manager.restart_workers(timeout_s=1)
    assert manager.state == "serving"
    assert manager.distributed_weight_daemon.epoch == 3
    assert manager._distributed_weight_failure_reason() is None


def test_a_worker_exit_leads_over_the_daemons_view_of_it(monkeypatch):
    """A dead worker closes its socket before it can be reaped: the daemon
    reports the cohort fault first, yet the exit names the failure."""

    class Proc:
        pid = 4242
        exitcode = None

        def join(self, timeout=None):
            return None

    proc = Proc()

    def on_poll(daemon, poll):
        if poll == 1:
            # Poll 1 already found no exit; the daemon sees the closed
            # socket now, and the process is reapable by the next poll.
            daemon.cohort_faulted = True
            daemon.cohort_fault = (
                "worker 2 control connection failed: peer closed the socket"
            )
            proc.exitcode = -9

    manager, _seen = _serving_manager(monkeypatch, on_poll=on_poll)
    manager.worker_process = SimpleNamespace(processes=[proc])
    manager._attribute_host_oom = lambda exited: None

    manager._monitor_worker_processes()

    assert manager.state == "workers_down"
    assert manager._worker_exit_state.reason == (
        "Detected worker process exit: idx=0 pid=4242 exitcode=-9"
    )


def test_one_report_is_confirmed_before_the_pool_is_failed(monkeypatch):
    def on_poll(daemon, poll):
        # Faulted on every other poll only: never twice in a row.
        daemon.cohort_faulted = poll % 2 == 1
        daemon.cohort_fault = "worker 0 control connection failed: x"

    manager, seen = _serving_manager(monkeypatch, on_poll=on_poll)

    manager._monitor_worker_processes()

    assert manager.distributed_weight_daemon.polls == _MAX_POLLS
    assert manager.state == "serving"
    assert seen.published == []


def test_a_failing_status_probe_fails_the_pool(monkeypatch):
    def on_poll(daemon, poll):
        daemon.status_error = AttributeError(
            "'DistributedWeightDaemon' object has no attribute 'status'"
        )

    manager, _seen = _serving_manager(monkeypatch, on_poll=on_poll)

    manager._monitor_worker_processes()

    assert manager.state == "workers_down"
    assert manager._worker_exit_state.reason == (
        "distributed weight daemon on node 1 could not report its status: "
        "AttributeError: 'DistributedWeightDaemon' object has no attribute "
        "'status'"
    )


def test_a_malformed_report_fails_the_pool_instead_of_the_monitor(monkeypatch):
    """A report missing a field must not end the monitor thread, which would
    also stop the detection of worker exits."""
    manager, _seen = _serving_manager(monkeypatch)
    manager.distributed_weight_daemon.status = lambda: json.dumps(
        {"network_ready": True}
    )

    manager._monitor_worker_processes()

    assert manager.state == "workers_down"
    assert manager._worker_exit_state.reason == (
        "distributed weight daemon on node 1 could not report its status: "
        "KeyError: 'failed'"
    )


def test_without_a_daemon_nothing_is_probed(monkeypatch):
    manager = _manager(monkeypatch)
    assert manager.distributed_weight_daemon is None
    assert manager._distributed_weight_failure_reason() is None
    assert "distributed_weights" not in manager.status()


def test_native_status_reports_both_latches_and_releases_the_gil():
    header = DAEMON_H.read_text()
    assert "std::string Status();" in header

    source = DAEMON_CPP.read_text()
    start = source.index("    std::string Status() {")
    body = source[start:source.index("\n    }\n", start)]
    for field in (
        '"network_ready"', '"failed"', '"failure"', '"failure_class"',
        '"cohort"', '"epoch"', '"state"', '"faulted"', '"fault"',
        '"fault_class"',
    ):
        assert field in body, field
    # A worker-supplied byte in a fault text must not make the probe throw.
    assert "error_handler_t::replace" in body
    # Leaf mutexes only, one at a time: never the ring or the summary.
    assert "ring_mutex" not in body and "summary_mutex" not in body
    assert (
        "std::string DistributedWeightDaemon::Status() "
        "{ return impl_->Status(); }"
    ) in source

    binding = BINDING_CPP.read_text()
    start = binding.index('.def("status", &DistributedWeightDaemon::Status')
    definition = binding[start:binding.index(".def(", start + 1)]
    assert "py::call_guard<py::gil_scoped_release>()" in definition
