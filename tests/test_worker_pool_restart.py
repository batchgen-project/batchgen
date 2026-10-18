"""Dev-machine lifecycle tests for the restartable worker pool."""

from __future__ import annotations

import ast
import inspect
import logging
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]

WORKER_MANAGER_PATH = ROOT / "batchgen/server/worker_manager.py"

# Module constants the real _stop_workers reads.  They are lifted from the
# module itself so a test exercising teardown uses the real values rather than
# a copy that can drift.
_MANAGER_CONSTANTS = (
    "_WORKER_SENTINEL_GRACE_S",
    "_WORKER_KILL_REAP_TIMEOUT_S",
    "_WORKER_EXIT_PATH_SIGNAL",
)


def _worker_manager_tree() -> ast.Module:
    return ast.parse(WORKER_MANAGER_PATH.read_text())


def _load_worker_manager_class():
    tree = _worker_manager_tree()
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "WorkerManager"
    )
    constants = [
        item for item in tree.body
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in _MANAGER_CONSTANTS
            for target in item.targets
        )
    ]
    assert len(constants) == len(_MANAGER_CONSTANTS)
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
        # _stop_workers signals children and names its leftovers through these.
        "os": os,
        "signal": signal,
        "list_run_named_objects": lambda prefix, runtime_dir: [],
    }
    module = ast.Module(body=constants + [node], type_ignores=[])
    exec(compile(module, str(WORKER_MANAGER_PATH), "exec"), namespace)
    return namespace["WorkerManager"]


class _CohortQueue:
    """Stands in for one cohort's mp.Queue.

    Reading a queue whose writer was killed is the defect under test, so every
    read fails loudly instead of returning. Lifecycle calls are recorded.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: List[str] = []
        self.items: List[Any] = []

    def put(self, item: Any) -> None:
        self.calls.append("put")
        self.items.append(item)

    def get(self, *args, **kwargs):
        self.calls.append("get")
        raise AssertionError(
            f"{self.label} was read; a killed cohort's channel may leave a "
            "partial frame, so any read can block forever"
        )

    def get_nowait(self, *args, **kwargs):
        self.calls.append("get_nowait")
        raise AssertionError(
            f"{self.label} was read with get_nowait, which polls the pipe and "
            "then performs an untimed blocking read"
        )

    def cancel_join_thread(self) -> None:
        self.calls.append("cancel_join_thread")

    def close(self) -> None:
        self.calls.append("close")

    def empty(self) -> bool:
        return not self.items


class _RawFlag:
    """Stands in for one cohort's RawValue teardown flag.

    Deliberately offers NOTHING but ``value``: no ``set``, no ``wait``, no
    ``acquire``.  A manager that reached for an Event-shaped API would fail
    here rather than reintroduce the wake handshake that deadlocked teardown.
    """

    def __init__(self) -> None:
        self.value = 0


class _CohortQueueContext:
    """Stub mp context handing out a fresh _CohortQueue per request."""

    def __init__(self) -> None:
        self.created: List[_CohortQueue] = []
        self.flags: List[_RawFlag] = []

    def Queue(self) -> _CohortQueue:  # noqa: N802 - mirrors mp context API
        channel = _CohortQueue(f"channel#{len(self.created)}")
        self.created.append(channel)
        return channel

    def RawValue(self, typecode, *args):  # noqa: N802 - mirrors mp context API
        assert typecode == "b"
        flag = _RawFlag()
        self.flags.append(flag)
        return flag


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
    manager._retire_channels_hook = None
    manager._restart_timeout_s = 10.0
    manager._monitor_stop_event = threading.Event()
    manager._stopping = False
    manager._teardown_lock = threading.Lock()
    manager.started = True
    manager._ready_event = threading.Event()
    manager._fatal_ack_event = threading.Event()
    manager._monitor_thread = None
    manager._worker_exit_state = SimpleNamespace(clear=lambda: None)
    manager._mp_ctx = _CohortQueueContext()
    manager._create_worker_channels()
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


def test_restart_rotates_cohort_queues_and_releases_all_after_teardown(monkeypatch):
    """A restart must hand the new cohort fresh channels.

    The queues the restart force-kills a cohort on can never be read again: a
    writer killed mid-write leaves a frame whose length header is readable but
    whose payload never arrives, so any later read blocks forever with no
    timeout and no exception. They are abandoned unread instead.
    """
    manager = _manager(monkeypatch)
    order = []
    dead_request = manager.request_queue
    dead_response = manager.response_queue
    dead_request.put("stale-request")
    dead_response.put("stale-response")
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

    # Both channels are replaced by fresh objects, after teardown+reconcile
    # and before the spawn that hands them to the new cohort.
    assert manager.request_queue is not dead_request
    assert manager.response_queue is not dead_response

    # The dead pair is abandoned: feeder cancelled, then closed, never read.
    assert dead_request.calls == ["put", "cancel_join_thread", "close"]
    assert dead_response.calls == ["put", "cancel_join_thread", "close"]
    assert dead_request.items == ["stale-request"]
    assert dead_response.items == ["stale-response"]

    # The generation's init message lands in the NEW request queue, carrying
    # the scheduler hook's pool capacity and never a hardcoded default.
    assert manager.request_queue.calls == ["put"]
    assert len(manager.request_queue.items) == 1
    init_msg = manager.request_queue.items[0]
    assert init_msg["type"] == "init"
    assert init_msg["max_output_len"] == 8192
    assert init_msg["max_context_length"] == 65536
    assert manager.response_queue.calls == []
    assert manager.response_queue.empty()


def test_restart_retires_the_scheduler_reader_before_abandoning_channels(monkeypatch):
    """The scheduler's pump may be blocked inside a read of the old response
    queue. Closing that queue under it turns the block into an exception, so
    the restart must announce the retirement first — otherwise the pump
    reports a channel error for a channel the restart itself removed."""
    manager = _manager(monkeypatch)
    order = []
    dead_response = manager.response_queue

    original_close = dead_response.close

    def close_recording():
        order.append("close")
        original_close()

    dead_response.close = close_recording
    manager._retire_channels_hook = lambda: order.append("retire")

    manager.restart_workers(timeout_s=1)

    assert order == ["retire", "close"]


def test_reconcile_never_reads_the_worker_queues(monkeypatch):
    """Reconciliation runs after a force-killed teardown, so it must not read
    the dead cohort's channels at all."""
    manager = _manager(monkeypatch)
    manager.request_queue.put("stale-request")
    manager.response_queue.put("stale-response")

    result = manager._reconcile("cohort died", fail_in_flight=True)

    assert result == {"failed_batches": 2}
    assert "drained" not in result
    assert manager.request_queue.calls == ["put"]
    assert manager.response_queue.calls == ["put"]

    # The untimed blocking read is gone from the module's CODE, not merely
    # unused: get_nowait() polls the pipe and only then performs an untimed
    # recv. Checked on the AST so the comment naming the hazard may stay.
    tree = ast.parse((ROOT / "batchgen/server/worker_manager.py").read_text())
    read_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert "get_nowait" not in read_attrs
    assert "_drain_queues" not in read_attrs
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_drain_queues"
        for node in ast.walk(tree)
    )


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


def test_monitor_failure_defers_to_a_running_restart(monkeypatch):
    """A restart that already set "restarting" owns the cohort's teardown; a
    monitor failure arriving then must not tear down, reconcile, or overwrite
    the state."""
    manager = _manager(monkeypatch)
    manager._state = "restarting"
    calls = []

    class ExitState:
        def set_failure(self, reason, exc):
            calls.append("set_failure")
            return True

    manager._worker_exit_state = ExitState()
    manager._quiesce_hook = lambda: calls.append("quiesce")
    manager._stop_workers = lambda: calls.append("teardown")
    manager._reconcile = lambda reason, fail_in_flight: calls.append("reconcile")

    manager._handle_worker_failure("worker exited", None)

    assert manager.state == "restarting"
    assert calls == []


def test_monitor_claims_teardown_in_the_same_critical_section(monkeypatch):
    """The ownership check and the workers_down transition must be atomic: a
    restart that tries to take the cohort while the monitor is deciding must
    find the state lock held, then see workers_down (never a serving state
    that the monitor is about to overwrite)."""
    manager = _manager(monkeypatch)
    calls = []
    lock_free_during_claim = []

    class ExitState:
        def set_failure(self, reason, exc):
            # A restart arriving now cannot get in between the check and the
            # transition.
            acquired = manager._lock.acquire(blocking=False)
            if acquired:
                manager._lock.release()
            lock_free_during_claim.append(acquired)
            return True

    manager._worker_exit_state = ExitState()
    manager._stop_workers = lambda: calls.append("teardown")
    manager._reconcile = lambda reason, fail_in_flight: calls.append(
        ("reconcile", fail_in_flight)
    )

    manager._handle_worker_failure("worker exited", None)

    assert lock_free_during_claim == [False]
    assert manager.state == "workers_down"
    assert calls == ["teardown", ("reconcile", True)]

    # A second report of the same cohort (e.g. the scheduler's pool_shutdown
    # after the monitor's exit detection) finds workers_down and does nothing.
    manager._handle_worker_failure("pool_shutdown fatal", None)
    assert calls == ["teardown", ("reconcile", True)]


def test_monitor_and_restart_race_has_a_single_teardown_owner(monkeypatch):
    """Thread choreography: the monitor and an explicit restart enter
    concurrently against one serving cohort. Whichever wins the state lock
    owns teardown; the other does not tear down."""
    for restart_first in (True, False):
        manager = _manager(monkeypatch)
        teardowns = []
        entered_restart = threading.Event()
        release_restart = threading.Event()

        class ExitState:
            def set_failure(self, reason, exc):
                return True

            def clear(self):
                pass

            def is_failed(self):
                return False

        manager._worker_exit_state = ExitState()
        manager._reconcile = lambda reason, fail_in_flight: {"failed_batches": 0}
        # The restart runs off the main thread here, as it does in the server.
        manager.set_main_thread_dispatcher(lambda fn, timeout: fn())

        def restart_quiesce():
            entered_restart.set()
            assert release_restart.wait(5)

        manager._quiesce_hook = restart_quiesce if restart_first else (lambda: None)
        manager._stop_workers = lambda: teardowns.append(
            threading.current_thread().name
        )

        if restart_first:
            restarter = threading.Thread(
                target=manager.restart_workers, kwargs={"timeout_s": 1},
                name="restart",
            )
            restarter.start()
            assert entered_restart.wait(5)
            assert manager.state == "restarting"
            monitor = threading.Thread(
                target=manager._handle_worker_failure,
                args=("worker exited", None),
                name="monitor",
            )
            monitor.start()
            monitor.join(5)
            release_restart.set()
            restarter.join(5)
            assert teardowns == ["restart"]
            assert manager.state == "serving"
        else:
            monitor = threading.Thread(
                target=manager._handle_worker_failure,
                args=("worker exited", None),
                name="monitor",
            )
            monitor.start()
            monitor.join(5)
            assert teardowns == ["monitor"]
            assert manager.state == "workers_down"


def test_restart_from_workers_down_waits_for_the_monitors_reconcile(monkeypatch):
    """The monitor claims a failure and is still inside its reconcile (whose
    SHM cleanup must not run beside a respawn) when a restart is requested
    from workers_down. The restart must wait for that reconcile to finish
    before its own teardown, so the two never overlap."""
    manager = _manager(monkeypatch)
    manager.set_main_thread_dispatcher(lambda fn, timeout: fn())
    events = []
    in_reconcile = threading.Event()
    finish_reconcile = threading.Event()

    class ExitState:
        def set_failure(self, reason, exc):
            return True

        def clear(self):
            pass

        def is_failed(self):
            return False

    manager._worker_exit_state = ExitState()

    def reconcile(reason, fail_in_flight):
        who = threading.current_thread().name
        events.append(("reconcile-begin", who))
        if who == "monitor":
            in_reconcile.set()
            assert finish_reconcile.wait(5)
        events.append(("reconcile-end", who))
        return {"failed_batches": 0}

    manager._reconcile = reconcile
    manager._stop_workers = lambda: events.append(
        ("teardown", threading.current_thread().name)
    )

    monitor = threading.Thread(
        target=manager._handle_worker_failure,
        args=("worker exited", None),
        name="monitor",
    )
    monitor.start()
    assert in_reconcile.wait(5)
    assert manager.state == "workers_down"

    restarter = threading.Thread(
        target=manager.restart_workers, kwargs={"timeout_s": 1}, name="restart",
    )
    restarter.start()
    time.sleep(0.2)
    # Blocked on the teardown lock: nothing of the restart's teardown ran.
    assert manager.state == "restarting"
    assert [e for e in events if e[1] == "restart"] == []

    finish_reconcile.set()
    monitor.join(5)
    restarter.join(5)
    assert not restarter.is_alive()
    assert manager.state == "serving"
    assert events.index(("reconcile-end", "monitor")) < events.index(
        ("reconcile-begin", "restart")
    )
    assert not manager._teardown_lock.locked()


def test_teardown_lock_is_released_when_a_restart_fails(monkeypatch):
    manager = _manager(monkeypatch)

    def failing_teardown():
        raise RuntimeError("worker teardown left live owned PIDs")

    manager._stop_workers = failing_teardown
    with pytest.raises(RuntimeError, match="live owned PIDs"):
        manager.restart_workers(timeout_s=1)
    assert manager.state == "workers_down"
    assert not manager._teardown_lock.locked()

    # A later monitor claim can still own a teardown.
    manager._stop_workers = lambda: None
    manager._state = "serving"
    claimed = []

    class ExitState:
        def set_failure(self, reason, exc):
            return True

    manager._worker_exit_state = ExitState()
    manager._reconcile = lambda reason, fail_in_flight: claimed.append(reason)
    manager._handle_worker_failure("worker exited", None)
    assert claimed == ["worker exited"]
    assert not manager._teardown_lock.locked()


def test_monitor_keeps_polling_after_a_declined_failure(monkeypatch):
    """A cohort that dies after readiness but before the restart's serving
    flip is declined (state restarting). The monitor must stay alive and hand
    the failure over again once the pool is serving."""
    manager = _manager(monkeypatch)
    manager._monitor_interval_s = 0.01
    manager._join_lock = threading.Lock()
    manager.worker_process = SimpleNamespace(
        processes=[SimpleNamespace(join=lambda timeout: None)]
    )
    manager._collect_worker_exit_reason = lambda: "Detected worker process exit"
    manager._state = "restarting"
    handled = []

    def handle(reason, exc):
        handled.append(manager.state)
        if manager.state == "serving":
            manager._monitor_stop_event.set()

    manager._handle_worker_failure = handle
    monitor = threading.Thread(target=manager._monitor_worker_processes)
    monitor.start()
    deadline = time.monotonic() + 5
    while len(handled) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert monitor.is_alive()
    assert set(handled) == {"restarting"}

    with manager._lock:
        manager._state = "serving"
    monitor.join(5)
    assert not monitor.is_alive()
    assert handled[-1] == "serving"


def test_explicit_restart_does_not_double_persist_prior_pool_shutdown(monkeypatch):
    manager = _manager(monkeypatch)
    calls = []
    manager._reconciled_failure = True
    manager._fail_in_flight_hook = lambda reason: calls.append(reason) or 1
    manager._spawn_workers = lambda: None

    manager.restart_workers(timeout_s=1)

    assert calls == []


def test_failed_teardown_leaves_recoverable_state_and_retry_persists(monkeypatch):
    """A restart that dies in ANY phase must stay retryable,
    and the retry must still persist the in-flight batches exactly once."""
    manager = _manager(monkeypatch)
    persisted = []
    manager._reconciled_failure = False
    manager._fail_in_flight_hook = lambda reason: persisted.append(reason) or 2

    def failing_teardown():
        raise RuntimeError("worker teardown left live owned PIDs")

    manager._stop_workers = failing_teardown
    with pytest.raises(RuntimeError, match="live owned PIDs"):
        manager.restart_workers(timeout_s=1)

    # Not stuck in "restarting", and the failure path never reached reconcile.
    assert manager.state == "workers_down"
    assert persisted == []
    assert manager._reconciled_failure is False

    manager._stop_workers = lambda: None
    result = manager.restart_workers(timeout_s=1)
    assert manager.state == "serving"
    assert result["failed_batches"] == 2
    assert len(persisted) == 1


def test_stale_exit_state_is_cleared_before_ready_wait(monkeypatch):
    """A fatal recorded by the previous cohort must not abort the readiness
    wait of the cohort a restart spawns."""
    manager = _manager(monkeypatch)
    order = []

    class ExitState:
        reason = "stale fatal from the dead cohort"

        def clear(self):
            self.reason = None
            order.append("clear_exit_state")

        def is_failed(self):
            return self.reason is not None

    manager._state = "workers_down"
    manager._reconciled_failure = True
    manager._worker_exit_state = ExitState()
    manager._wait_for_workers_ready = (
        lambda timeout_s=None: order.append("wait_ready")
    )

    manager.restart_workers(timeout_s=1)

    assert order[0] == "clear_exit_state"
    assert "wait_ready" in order
    assert manager.state == "serving"


@pytest.mark.parametrize("hook", ["generation_ready", "resume"])
def test_failing_scheduler_hook_leaves_workers_down_not_serving(monkeypatch, hook):
    """The scheduler hooks are the last restart phase: if one raises, the
    restart failed and the manager must be retryable, not falsely serving."""
    manager = _manager(monkeypatch)

    def failing():
        raise RuntimeError(f"{hook} hook failed")

    if hook == "generation_ready":
        manager._scheduler_generation_hook = failing
    else:
        manager._resume_hook = failing

    with pytest.raises(RuntimeError, match=f"{hook} hook failed"):
        manager.restart_workers(timeout_s=1)

    assert manager.state == "workers_down"
    assert manager.generation == 3

    # Retryable: a later restart with healthy hooks reaches serving.
    manager._scheduler_generation_hook = lambda: None
    manager._resume_hook = lambda: None
    manager.restart_workers(timeout_s=1)
    assert manager.state == "serving"
    assert manager.generation == 4


def test_scheduler_hooks_run_before_the_serving_flip(monkeypatch):
    manager = _manager(monkeypatch)
    observed = []
    manager._scheduler_generation_hook = lambda: observed.append(
        ("generation_ready", manager.state, manager.generation)
    )
    manager._resume_hook = lambda: observed.append(
        ("resume", manager.state, manager.generation)
    )

    manager.restart_workers(timeout_s=1)

    assert observed == [
        ("generation_ready", "restarting", 3),
        ("resume", "restarting", 3),
    ]
    assert manager.state == "serving"
    assert manager.generation == 4


@pytest.mark.parametrize("stop_phase", ["entry", "after_teardown", "before_spawn"])
def test_restart_aborts_to_workers_down_once_shutdown_begins(monkeypatch, stop_phase):
    """SIGINT cannot cancel the restart thread, so the restart checks for
    shutdown at its phase boundaries and never spawns after stop() began."""
    manager = _manager(monkeypatch)
    phases = []
    manager._quiesce_hook = lambda: phases.append("quiesce")
    manager._resume_hook = lambda: phases.append("resume")

    def teardown():
        phases.append("teardown")
        if stop_phase == "after_teardown":
            manager._stopping = True

    def channels_created():
        phases.append("channels")
        if stop_phase == "before_spawn":
            # stop() ran to completion during this phase: _stopping is
            # cleared again on its way out, only started stays False.
            manager.started = False

    manager._stop_workers = teardown
    original_create = manager._create_worker_channels

    def create_channels():
        original_create()
        channels_created()

    manager._create_worker_channels = create_channels
    manager._spawn_workers = lambda: phases.append("spawn")
    if stop_phase == "entry":
        manager._stopping = True

    with pytest.raises(RuntimeError, match="server is stopping"):
        manager.restart_workers(timeout_s=1)

    assert manager.state == "workers_down"
    assert manager.generation == 3
    assert "spawn" not in phases
    assert "resume" not in phases
    if stop_phase == "entry":
        assert phases == []
    elif stop_phase == "after_teardown":
        assert phases == ["quiesce", "teardown"]
    else:
        assert phases == ["quiesce", "teardown", "channels"]


def test_restart_result_reports_all_phase_timings(monkeypatch):
    manager = _manager(monkeypatch)
    result = manager.restart_workers(timeout_s=1)
    assert set(result["phase_timings"]) == {
        "quiesce_s", "teardown_s", "reconcile_s", "spawn_to_ready_s",
    }


def test_teardown_budgets_cover_pinned_page_release():
    """Ordered release of a large cudaHostRegister'd footprint takes minutes,
    and a SIGKILL'd worker is unreapable until the kernel unpins it."""
    source = (ROOT / "batchgen/server/worker_manager.py").read_text()
    assert "_WORKER_SENTINEL_GRACE_S = 180.0" in source
    assert "_WORKER_KILL_REAP_TIMEOUT_S = 900.0" in source
    assert "deadline = time.monotonic() + _WORKER_KILL_REAP_TIMEOUT_S" in source


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


# --------------------------------------------------------------------------
# NCCL-abort watchdog on shutdown
#
# A worker whose peer died mid-collective is blocked inside NCCL: it consumes
# neither the queue sentinel (only rank 0 reads the queue) nor the status
# broadcast the other ranks learn shutdown from (itself a collective).  The
# cohort shutdown flag reaches it out of band, and a watchdog thread aborts its
# communicators ONE GROUP AT A TIME, one daemon thread each, so the blocked
# collective returns and the rank retires through the ordered teardown.
#
# Three measured facts bind these tests; each is pinned by name below:
#   C1  multiprocessing.Event cannot carry the signal -- set() blocks in
#       Condition.notify while a registered waiter is a dead process, so the
#       teardown owner would deadlock on the very failure this signal is for.
#   C2  the all-groups abort IS the bug -- it walks groups in reverse name
#       order inside one ncclGroupStart/End region and blocks on the first
#       communicator, which is generally not the wedged one.
#   C3  an aborted collective RETURNS, it does not raise, so a flag check at
#       the loop boundary is the primary exit and the guard is the secondary.
# --------------------------------------------------------------------------

WORKER_LOOP_PATH = ROOT / "batchgen/server_worker_main_loop.py"

_WORKER_LOOP_HELPERS = (
    "_shutdown_requested",
    "_live_process_groups",
    "_abort_one_process_group",
    "_flush_worker_logs",
    "_start_shutdown_abort_watchdog",
    "_resolve_main_loop_exit",
)
_WORKER_LOOP_CONSTANTS = (
    "_SHUTDOWN_ABORT_GRACE_S",
    "_SHUTDOWN_FLAG_POLL_S",
    "_SHUTDOWN_SELF_EXIT_GRACE_S",
    "_WORKER_SELF_EXIT_CODE",
    "_EXIT_PATH_SENTINEL",
    "_EXIT_PATH_FLAG",
    "_EXIT_PATH_ABORT",
    "_EXIT_PATH_DECODE_BOUNDARY",
    "_EXIT_PATH_SELF_EXIT",
)


def _worker_loop_tree() -> ast.Module:
    return ast.parse(WORKER_LOOP_PATH.read_text())


def _worker_loop_function(name: str) -> ast.FunctionDef:
    return next(
        node for node in _worker_loop_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _main_loop_guards(impl: ast.FunctionDef) -> List[ast.Try]:
    """Every try at the main-loop boundary, i.e. wrapping the ``while True``.

    ``_server_worker_main_impl`` contains other, older try blocks (process-group
    init, NCCL warmup), so the guard is identified by what it wraps, never by
    being the first one.  Takes the impl node so callers that also index
    ``impl.body`` compare nodes from the SAME parse.
    """
    return [
        node for node in impl.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(inner, ast.While)
            and isinstance(inner.test, ast.Constant)
            and inner.test.value is True
            for inner in node.body
        )
    ]


def _main_loop_guard() -> ast.Try:
    guards = _main_loop_guards(_worker_loop_function("_server_worker_main_impl"))
    assert len(guards) == 1
    return guards[0]


def _load_worker_loop_helpers() -> Dict[str, Any]:
    """Execute the shutdown helpers without importing the worker module.

    ``server_worker_main_loop`` imports the whole worker, which a dev machine
    cannot load, so the helpers are lifted out of the source exactly as
    ``_load_worker_manager_class`` lifts out WorkerManager.  The module
    constants come from the same source, so a changed grace window or marker
    cannot pass unnoticed.
    """
    tree = _worker_loop_tree()
    constants = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in _WORKER_LOOP_CONSTANTS
            for target in node.targets
        )
    ]
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _WORKER_LOOP_HELPERS
    ]
    assert {node.name for node in functions} == set(_WORKER_LOOP_HELPERS)
    assert len(constants) == len(_WORKER_LOOP_CONSTANTS)
    namespace: Dict[str, Any] = {
        "Any": Any,
        "List": List,
        "Optional": Optional,
        "threading": threading,
        "time": time,
        "logging": logging,
        # The self-exit safety net ends the process through os._exit and flushes
        # the log streams first, so the lifted helpers need both modules.  Every
        # test that exercises that path injects its own exit hook, so the real
        # os._exit here is never reached -- but it must still resolve, or the
        # watchdog would fail for the wrong reason.
        "os": os,
        "sys": sys,
    }
    module = ast.Module(body=constants + functions, type_ignores=[])
    exec(compile(module, str(WORKER_LOOP_PATH), "exec"), namespace)
    return namespace


class _Flag:
    """The worker's view of the cohort flag: a plain shared byte."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


class _Group:
    """A stand-in ProcessGroup recording its own abort."""

    def __init__(self, name: str, block: Optional[threading.Event] = None) -> None:
        self.name = name
        self.aborts = 0
        self._block = block

    def abort(self) -> None:
        self.aborts += 1
        if self._block is not None:
            # A per-group abort whose group is not the wedged one can block
            # indefinitely (C2); nothing may wait on it.
            self._block.wait(timeout=30)


def _abort_threads() -> List[threading.Thread]:
    return [
        thread for thread in threading.enumerate()
        if thread.name.startswith("worker-shutdown-abort-")
    ]


# ---- C1: the transport is lock-free shared memory, never an Event ---------

def test_cohort_flag_is_lock_free_shared_memory():
    """C1.  The real spawn context's RawValue is what the manager binds, and it
    must expose no lock and no wake handshake: a plain store is the whole
    protocol.  multiprocessing.Event's set() goes through Condition.notify,
    which blocks while a registered waiter is a dead process -- the teardown
    owner would deadlock on exactly the failure this signal exists for."""
    flag = multiprocessing.get_context("spawn").RawValue("b", 0)
    for forbidden in ("get_lock", "acquire", "release", "set", "wait", "is_set"):
        assert not hasattr(flag, forbidden), (
            f"the cohort flag must not expose {forbidden}; it is a plain "
            "shared byte, not a synchronization primitive"
        )
    assert flag.value == 0
    flag.value = 1
    assert flag.value == 1


def test_the_shutdown_signal_uses_no_multiprocessing_event_or_condition():
    """C1, by construction.  Nothing in the cohort-flag mechanism may create an
    mp.Event or Condition, on either side."""
    manager_tree = _worker_manager_tree()

    # Manager side: the flag is created by RawValue and nothing else.
    create = next(
        node for node in ast.walk(manager_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_create_worker_channels"
    )
    flag_sources = [
        ast.unparse(node.value) for node in ast.walk(create)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_cohort_shutdown_flag"
            for target in node.targets
        )
    ]
    assert flag_sources == ["self._mp_ctx.RawValue('b', 0)"]

    # Manager side: the flag is only ever stored to / loaded from.
    touched = {
        node.attr
        for node in ast.walk(manager_tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_cohort_shutdown_flag"
    }
    assert touched == {"value"}, (
        f"the cohort flag was used as more than a shared byte: {touched}"
    )

    # Worker side: no mp primitive anywhere in the shutdown helpers.
    loop_source = WORKER_LOOP_PATH.read_text()
    helpers = loop_source.split("def _shutdown_requested(", 1)[1]
    helpers = helpers.split("def _setup_nccl_env(", 1)[0]
    for forbidden in ("mp.Event", "mp.Condition", "multiprocessing"):
        assert forbidden not in helpers, (
            f"{forbidden} appears in the shutdown watchdog; the teardown "
            "signal must carry no wake handshake"
        )

    # And the flag is read with a plain load, not waited on.
    read = _worker_loop_function("_shutdown_requested")
    assert {
        node.attr for node in ast.walk(read) if isinstance(node, ast.Attribute)
    } == {"value"}


def test_shutdown_requested_is_a_plain_load():
    ns = _load_worker_loop_helpers()
    assert ns["_shutdown_requested"](None) is False
    assert ns["_shutdown_requested"](_Flag(0)) is False
    assert ns["_shutdown_requested"](_Flag(1)) is True


# ---- C2: never the no-group abort; one daemon thread per group ------------

def test_the_no_group_abort_api_is_absent_from_the_worker_loop():
    """C2.  ``_abort_process_group()`` with no group is the bug: torch walks the
    groups in reverse NAME order inside one ncclGroupStart/End region, so
    ProcessGroupNCCL::abort blocks in waitForFutureOrTimeout on a communicator
    that carries no wedged work and never reaches the wedged one, leaving the
    blocked rank blocked and the abort call itself never returning.

    Passing the DEFAULT group explicitly is the same branch by another name --
    torch tests ``group is None or group == GroupMember.WORLD`` and
    GroupMember.WORLD *is* the default group -- so the only safe construction
    is not to reference that wrapper at all."""
    source = WORKER_LOOP_PATH.read_text()
    assert "_abort_process_group" not in source, (
        "the all-groups abort wrapper must be unreachable from this module: "
        "called with the default group it re-enters the reverse-name-order, "
        "group-wrapped branch that blocks instead of aborting"
    )


def test_abort_helper_requires_an_explicit_group():
    """C2.  The no-group call is impossible by construction: the helper's group
    parameter is required and has no default, and a None group is refused with
    a named error rather than silently widened."""
    ns = _load_worker_loop_helpers()
    abort = ns["_abort_one_process_group"]

    parameters = list(inspect.signature(abort).parameters.values())
    assert len(parameters) == 1
    assert parameters[0].name == "group"
    assert parameters[0].default is inspect.Parameter.empty
    assert parameters[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    with pytest.raises(TypeError):
        abort()
    with pytest.raises(RuntimeError, match="explicit process group"):
        abort(None)

    group = _Group("0")
    abort(group)
    assert group.aborts == 1


def test_abort_helper_calls_the_single_group_primitive():
    """C2.  ``ProcessGroup.abort()`` is the frame directly under torch's own
    per-group branch (2.8.0 distributed_c10d.py:2298, 2.14.1 :3161, production
    2.9.0 :2307) -- the same primitive, without the ordering that blocks."""
    fn = _worker_loop_function("_abort_one_process_group")
    calls = [
        ast.unparse(node) for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls == ["group.abort()"]


def test_group_enumeration_reads_torchs_own_registry_as_a_snapshot():
    """Grounded in the installed torch: ``_world.pg_names`` maps every process
    group THIS rank joined to its name, which is what torch itself walks
    (2.8.0 distributed_c10d.py:2271-2273, 2.14.1 :3136-3139, production 2.9.0
    :2280-2283).  It must be snapshotted, because the concurrent aborts below
    make torch mutate ``_world`` while we iterate."""
    fn = _worker_loop_function("_live_process_groups")
    assert ast.unparse(fn.body[-1]) == "return list(_world.pg_names)"
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "torch.distributed.distributed_c10d"
        and [alias.name for alias in node.names] == ["_world"]
        for node in ast.walk(fn)
    )

    torch_c10d = pytest.importorskip("torch.distributed.distributed_c10d")
    # The attribute path exists in the installed torch and is a mapping keyed
    # by process group; with no group initialized it is simply empty.
    assert hasattr(torch_c10d._world, "pg_names")
    assert isinstance(dict(torch_c10d._world.pg_names), dict)

    ns = _load_worker_loop_helpers()
    groups = ns["_live_process_groups"]()
    assert isinstance(groups, list)


def test_shutdown_watchdog_aborts_every_group_on_its_own_daemon_thread():
    """C2.  One throwaway daemon thread per communicator, started concurrently,
    each aborting exactly one group.  Fired this way the thread that happens to
    hold the wedged communicator frees the main thread regardless of what the
    others are doing, so the caller needs no knowledge of which group wedged."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    ns["_SHUTDOWN_SELF_EXIT_GRACE_S"] = 0.01
    groups = [_Group("0"), _Group("1"), _Group("2")]
    seen: List[Any] = []
    launched = threading.Semaphore(0)
    flag = _Flag(0)
    aborted = threading.Event()

    def record(group):
        seen.append(group)
        launched.release()

    watchdog = ns["_start_shutdown_abort_watchdog"](
        3, flag, threading.Event(), aborted,
        live_process_groups=lambda: list(groups),
        abort_one_group=record,
        # The loop never retires in this test, so the watchdog reaches its
        # self-exit net; without a hook it would end the test process.
        exit_process=lambda code: None,
    )
    flag.value = 1
    watchdog.join(timeout=10)
    assert not watchdog.is_alive()
    for _ in groups:
        assert launched.acquire(timeout=10)

    # Exactly one abort per group, each with its own group object.
    assert sorted(group.name for group in seen) == ["0", "1", "2"]
    assert len(seen) == len(set(id(group) for group in seen))
    assert aborted.is_set()


def test_abort_threads_are_daemons_and_are_never_joined():
    """C2.  A per-group abort can block indefinitely when its group is not the
    wedged one, so no thread may stand in front of another's call and nothing
    may wait on any of them.  The watchdog must return while a blocking abort
    is still in flight, and every thread it started must be a daemon."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    ns["_SHUTDOWN_SELF_EXIT_GRACE_S"] = 0.01
    wedged = threading.Event()        # released only at the end of the test
    entered = threading.Semaphore(0)
    flag = _Flag(0)

    def blocking_abort(group):
        entered.release()
        wedged.wait(timeout=30)

    try:
        watchdog = ns["_start_shutdown_abort_watchdog"](
            0, flag, threading.Event(), threading.Event(),
            live_process_groups=lambda: [_Group("0"), _Group("1")],
            abort_one_group=blocking_abort,
            exit_process=lambda code: None,
        )
        flag.value = 1
        # Both aborts are in flight...
        assert entered.acquire(timeout=10)
        assert entered.acquire(timeout=10)
        # ...and the watchdog is already gone, having joined neither.
        watchdog.join(timeout=10)
        assert not watchdog.is_alive()

        in_flight = _abort_threads()
        assert len(in_flight) >= 2
        for thread in in_flight:
            assert thread.daemon, (
                f"{thread.name} is not a daemon; a blocked abort thread would "
                "then keep the process alive after its teardown"
            )
    finally:
        wedged.set()

    # No join call anywhere in the watchdog, on any path.
    fn = _worker_loop_function("_start_shutdown_abort_watchdog")
    assert not [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
    ]


def test_shutdown_watchdog_is_silent_when_the_loop_exits_in_the_window():
    """The graceful path stays the normal one: a worker that is not stuck leaves
    its loop inside the window, and the watchdog must abort nothing at all."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    aborts: List[Any] = []
    flag = _Flag(0)
    loop_exited = threading.Event()
    aborted = threading.Event()

    watchdog = ns["_start_shutdown_abort_watchdog"](
        0, flag, loop_exited, aborted,
        live_process_groups=lambda: [_Group("0")],
        abort_one_group=aborts.append,
    )
    assert watchdog is not None
    flag.value = 1
    loop_exited.set()

    watchdog.join(timeout=ns["_SHUTDOWN_ABORT_GRACE_S"] + 10)
    assert not watchdog.is_alive()
    assert aborts == []
    assert not aborted.is_set()


def test_shutdown_watchdog_is_silent_when_the_loop_was_already_gone():
    """The flag may be set after the loop ended -- the watchdog must notice the
    finished loop while polling, not start a grace window for a dead loop."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    aborts: List[Any] = []
    loop_exited = threading.Event()
    loop_exited.set()

    watchdog = ns["_start_shutdown_abort_watchdog"](
        0, _Flag(0), loop_exited, threading.Event(),
        live_process_groups=lambda: [_Group("0")],
        abort_one_group=aborts.append,
    )
    watchdog.join(timeout=10)
    assert not watchdog.is_alive()
    assert aborts == []


def test_shutdown_watchdog_never_fires_without_the_cohort_flag():
    """The watchdog's only trigger is the cohort flag, which nothing but
    teardown sets.  With no flag there is nothing to arm, and with an unset
    flag it polls forever rather than acting on a timeout."""
    ns = _load_worker_loop_helpers()
    aborts: List[Any] = []

    assert ns["_start_shutdown_abort_watchdog"](
        0, None, threading.Event(), threading.Event(),
        live_process_groups=lambda: [_Group("0")],
        abort_one_group=aborts.append,
    ) is None

    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    watchdog = ns["_start_shutdown_abort_watchdog"](
        0, _Flag(0), threading.Event(), threading.Event(),
        live_process_groups=lambda: [_Group("0")],
        abort_one_group=aborts.append,
    )
    # Far longer than the (shortened) grace window: an unset flag must not
    # start the window at all.
    watchdog.join(timeout=0.5)
    assert watchdog.is_alive()
    assert aborts == []


def test_shutdown_watchdog_survives_a_failing_abort():
    """An abort that raises must not take the watchdog down silently; the
    manager's signal fallback still covers the worker, so each thread logs its
    own failure and the others still run."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    ns["_SHUTDOWN_SELF_EXIT_GRACE_S"] = 0.01
    reached: List[str] = []
    done = threading.Semaphore(0)
    flag = _Flag(0)

    def exploding_abort(group):
        reached.append(group.name)
        done.release()
        raise RuntimeError(f"abort of {group.name} failed")

    watchdog = ns["_start_shutdown_abort_watchdog"](
        1, flag, threading.Event(), threading.Event(),
        live_process_groups=lambda: [_Group("0"), _Group("1")],
        abort_one_group=exploding_abort,
        exit_process=lambda code: None,
    )
    flag.value = 1
    watchdog.join(timeout=10)
    assert not watchdog.is_alive()
    assert done.acquire(timeout=10) and done.acquire(timeout=10)
    assert sorted(reached) == ["0", "1"]


def test_shutdown_watchdog_survives_a_failing_group_enumeration():
    """If torch's registry cannot be read the watchdog says so and ends; it
    must never fall through to an unspecified abort."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    aborts: List[Any] = []
    flag = _Flag(0)
    aborted = threading.Event()

    def exploding_enumeration():
        raise RuntimeError("torch internals moved")

    watchdog = ns["_start_shutdown_abort_watchdog"](
        2, flag, threading.Event(), aborted,
        live_process_groups=exploding_enumeration,
        abort_one_group=aborts.append,
    )
    flag.value = 1
    watchdog.join(timeout=10)
    assert not watchdog.is_alive()
    assert aborts == []
    assert not aborted.is_set()


def test_shutdown_watchdog_claims_nothing_when_there_is_no_live_group():
    """A stuck loop holding no communicator cannot be freed by an abort, so the
    watchdog must not publish one: the exit marker would otherwise claim an
    abort retired a rank when none was ever issued."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    aborts: List[Any] = []
    flag = _Flag(0)
    aborted = threading.Event()

    watchdog = ns["_start_shutdown_abort_watchdog"](
        4, flag, threading.Event(), aborted,
        live_process_groups=lambda: [],
        abort_one_group=aborts.append,
    )
    flag.value = 1
    watchdog.join(timeout=10)
    assert not watchdog.is_alive()
    assert aborts == []
    assert not aborted.is_set()


# ---- C3: the boundary check is the primary exit, the guard the secondary --

def test_main_loop_checks_the_shutdown_flag_at_its_boundary_without_raising():
    """C3.  The wedged collective RETURNS when its communicator is aborted; it
    does not raise -- the kernel exits and the event fires.  So the loop must
    TEST the flag at its boundary and leave by a normal break; an
    exception-only guard never fires at all, which is how the first attempt at
    this fix failed validation with nothing routed through its guard."""
    guard = _main_loop_guard()
    outer = next(node for node in guard.body if isinstance(node, ast.While))

    # The very first thing the loop does, before any collective.
    first = outer.body[0]
    assert isinstance(first, ast.If)
    assert ast.unparse(first.test) == "_shutdown_requested(shutdown_flag)"
    assert any(isinstance(node, ast.Break) for node in first.body)
    assert not [node for node in ast.walk(first) if isinstance(node, ast.Raise)], (
        "the boundary exit must be a normal break; release from an aborted "
        "collective is not an exception"
    )

    # The inner polling loop -- where an idle rank actually blocks, and where
    # an aborted broadcast returns -- stops on the same flag...
    inner = next(node for node in ast.walk(outer) if isinstance(node, ast.While)
                 and node is not outer)
    inner_first = inner.body[0]
    assert isinstance(inner_first, ast.If)
    assert ast.unparse(inner_first.test) == "_shutdown_requested(shutdown_flag)"
    assert [type(node) for node in inner_first.body] == [ast.Break]

    # ...and control returns to the single check above rather than falling
    # through into the next collective.
    after_inner = outer.body[outer.body.index(inner) + 1]
    assert isinstance(after_inner, ast.If)
    assert ast.unparse(after_inner.test) == "not work_available"
    assert [type(node) for node in after_inner.body] == [ast.Continue]


def test_boundary_exit_names_the_abort_only_when_the_abort_fired():
    """The marker must stay honest.  A healthy rank that simply saw the flag did
    not need an abort, and node validation counts the ranks that did."""
    outer = next(
        node for node in _main_loop_guard().body if isinstance(node, ast.While)
    )
    assignment = next(
        node for node in ast.walk(outer.body[0]) if isinstance(node, ast.Assign)
    )
    assert ast.unparse(assignment) == (
        "exit_path = _EXIT_PATH_ABORT if aborted.is_set() else _EXIT_PATH_FLAG"
    )


def test_main_loop_guard_routes_an_abort_to_the_ordered_teardown():
    ns = _load_worker_loop_helpers()

    exit_path = ns["_resolve_main_loop_exit"](
        2, _Flag(1), RuntimeError("NCCL communicator was aborted"),
    )

    assert exit_path == ns["_EXIT_PATH_ABORT"] == "ordered-via-abort"


@pytest.mark.parametrize("shutdown_flag", [None, _Flag(0)])
def test_main_loop_guard_reraises_a_real_collective_failure(shutdown_flag):
    """Without the cohort flag the same exception shape is a genuine NCCL
    failure and must stay exactly as fatal as it is today."""
    ns = _load_worker_loop_helpers()
    error = RuntimeError("NCCL communicator was aborted")

    with pytest.raises(RuntimeError) as excinfo:
        ns["_resolve_main_loop_exit"](2, shutdown_flag, error)

    assert excinfo.value is error


def test_main_loop_has_exactly_one_abort_guard_at_its_boundary():
    """The guard belongs at the main-loop boundary, not sprinkled over the
    collective call sites, and the ordered teardown must run after it rather
    than inside the handler."""
    impl = _worker_loop_function("_server_worker_main_impl")
    guards = _main_loop_guards(impl)
    assert len(guards) == 1
    guard = guards[0]

    # One handler, catching everything, delegating the decision to the
    # grounded helper rather than re-deciding inline.
    assert len(guard.handlers) == 1
    handler = guard.handlers[0]
    assert isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resolve_main_loop_exit"
        for node in ast.walk(handler)
    )

    # The watchdog is told the loop ended on BOTH paths, and before the
    # ordered teardown, which can take minutes and must not be aborted.
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "loop_exited"
        for node in ast.walk(ast.Module(body=guard.finalbody, type_ignores=[]))
    )

    # The ordered teardown sits after the guard, so every exit path runs the
    # SAME release sequence.
    guard_index = impl.body.index(guard)
    tail = ast.Module(body=impl.body[guard_index + 1:], type_ignores=[])
    teardown_calls = [
        node for node in ast.walk(tail)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_shutdown_worker_runtime"
    ]
    assert len(teardown_calls) == 1
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_shutdown_worker_runtime"
        for node in ast.walk(ast.Module(body=guard.body, type_ignores=[]))
    )


def test_ordered_teardown_is_not_poisoned_by_the_abort():
    """``destroy_process_group()`` shuts groups down in the SAME reverse-name
    order the all-groups abort uses (2.8.0 distributed_c10d.py:2183-2186,
    2.14.1 :3017-3020), so after an abort it is a fresh chance to block on a
    sibling communicator.  The ordered teardown must skip it once this worker
    aborted, and the C++ release must still run first."""
    impl = _worker_loop_function("_server_worker_main_impl")
    guard_at = next(
        i for i, stmt in enumerate(impl.body)
        if isinstance(stmt, ast.Try)
        and any(isinstance(inner, ast.While) for inner in stmt.body)
    )
    statements = impl.body[guard_at + 1:]

    def _index_of(predicate) -> int:
        return next(
            i for i, stmt in enumerate(statements)
            if any(predicate(node) for node in ast.walk(stmt))
        )

    release_at = _index_of(
        lambda node: isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_shutdown_worker_runtime"
    )
    destroy_at = _index_of(
        lambda node: isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "destroy_process_group"
    )
    # The C++ release (storage unregister/munmap) runs first and is untouched
    # by the abort, which only tears down communicators.
    assert release_at < destroy_at

    destroy_stmt = statements[destroy_at]
    assert isinstance(destroy_stmt, ast.If)
    assert ast.unparse(destroy_stmt.test) == (
        "dist.is_available() and dist.is_initialized() and (not aborted.is_set())"
    )


def test_worker_exit_paths_are_all_logged():
    ns = _load_worker_loop_helpers()
    assert ns["_EXIT_PATH_SENTINEL"] == "ordered-via-sentinel"
    assert ns["_EXIT_PATH_FLAG"] == "ordered-via-flag"
    assert ns["_EXIT_PATH_ABORT"] == "ordered-via-abort"
    # A few seconds, as designed: long enough for a healthy worker to leave its
    # loop, short enough that the 180s grace is no longer paid.
    assert 0 < ns["_SHUTDOWN_ABORT_GRACE_S"] <= 30
    # Short enough that a flag set at teardown is noticed promptly, long enough
    # not to spin.
    assert 0 < ns["_SHUTDOWN_FLAG_POLL_S"] <= 1.0

    source = WORKER_LOOP_PATH.read_text()
    assert source.count('"[shutdown] worker exit path=%s rank=%s"') == 1
    assert "exit_path, global_rank" in source

    # The self-exit marker is a SEPARATE line with its own format, because it
    # is not an ordered exit and carries the pid: the watchdog thread logs it
    # while the main thread is still stuck below Python.
    assert source.count('"[shutdown] worker exit path=%s rank=%s pid=%s"') == 1

    # The worker never claims the signal-fallback path: it cannot log it (it is
    # force-killed inside a native collective, so no Python runs in it again),
    # so the manager owns that marker and this module must not contain it as a
    # value anywhere.  Checked over string literals, not prose.
    literals = {
        node.value for node in ast.walk(_worker_loop_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "signal-fallback" not in literals
    assert {
        "ordered-via-sentinel",
        "ordered-via-flag",
        "ordered-via-abort",
        "ordered-via-decode-boundary",
        "self-exit",
    } <= literals

    # ...and it IS the manager's, exactly once, with the rank index and pid.
    manager_source = WORKER_MANAGER_PATH.read_text()
    assert '_WORKER_EXIT_PATH_SIGNAL = "signal-fallback"' in manager_source
    assert manager_source.count(
        '"[shutdown] worker exit path=%s rank-idx=%s pid=%s"'
    ) == 1


# ---- C4/C5: the manager side ---------------------------------------------

def test_teardown_sets_the_cohort_flag_before_the_sentinel(monkeypatch):
    """C5.  A rank blocked in a collective reads neither the queue sentinel nor
    the status broadcast, so the out-of-band flag must be set first -- and with
    a plain store, which cannot block the teardown owner."""
    manager = _manager(monkeypatch)
    order: List[str] = []

    class _RecordingFlag:
        def __init__(self) -> None:
            self._value = 0

        @property
        def value(self) -> int:
            return self._value

        @value.setter
        def value(self, new: int) -> None:
            order.append(f"cohort-flag-store:{new}")
            self._value = new

    manager._cohort_shutdown_flag = _RecordingFlag()
    channel = manager.request_queue
    original_put = channel.put

    def recording_put(item):
        order.append(f"sentinel:{item!r}")
        original_put(item)

    channel.put = recording_put
    manager._join_lock = threading.Lock()
    manager.worker_process = SimpleNamespace(processes=[
        SimpleNamespace(exitcode=0, pid=4242, join=lambda timeout=None: None),
    ])
    # Use the real teardown, not the _manager() stub.
    del manager._stop_workers

    manager._stop_workers()

    assert order == ["cohort-flag-store:1", "sentinel:None"]
    assert manager._cohort_shutdown_flag.value == 1


def test_manager_logs_the_signal_fallback_marker_per_forced_worker(
    monkeypatch, caplog
):
    """C4.  A force-killed worker is stopped while a native collective owns its
    thread, so it never runs Python again and can never log its own exit
    marker.  The manager emits it, naming the rank index and pid, for each
    worker whose ordered exit a signal replaced -- and for no other."""
    manager = _manager(monkeypatch)

    class _Proc:
        def __init__(self, pid: int, exits_on_sentinel: bool) -> None:
            self.pid = pid
            self.exitcode = 0 if exits_on_sentinel else None

        def join(self, timeout=None) -> None:
            pass

        def terminate(self) -> None:
            self.exitcode = -15

        def kill(self) -> None:
            self.exitcode = -9

    graceful = _Proc(1001, exits_on_sentinel=True)
    forced = _Proc(1002, exits_on_sentinel=False)
    manager._join_lock = threading.Lock()
    manager.worker_process = SimpleNamespace(processes=[graceful, forced])
    manager.args = SimpleNamespace(runtime_identity=SimpleNamespace(
        shm_prefix="run.", runtime_dir="/tmp/does-not-matter",
    ))
    del manager._stop_workers

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        manager._stop_workers()

    markers = [
        record.getMessage() for record in caplog.records
        if "worker exit path=" in record.getMessage()
    ]
    assert markers == [
        "[shutdown] worker exit path=signal-fallback rank-idx=1 pid=1002"
    ], markers
    # The marker is emitted by the manager, never claimed by the worker.
    assert forced.exitcode == -15 and graceful.exitcode == 0


def test_restart_rotates_the_cohort_shutdown_flag(monkeypatch):
    """The flag is cohort-scoped like the channel pair: a new generation must
    never start up already holding the previous teardown's flag."""
    manager = _manager(monkeypatch)
    retired = manager._cohort_shutdown_flag
    assert retired is not None
    retired.value = 1

    manager.restart_workers(timeout_s=1)

    fresh = manager._cohort_shutdown_flag
    assert fresh is not None
    assert fresh is not retired
    assert fresh.value == 0
    # The retired flag is abandoned with its cohort, never reused or cleared.
    assert retired.value == 1


def test_spawn_hands_the_cohort_flag_to_every_worker():
    """The flag must reach every LOCAL rank, so it travels as a spawn arg
    beside the channels and the readiness events."""
    source = WORKER_MANAGER_PATH.read_text()
    spawn_args = source.split("self.worker_process = mp.spawn(", 1)[1]
    spawn_args = spawn_args.split("nprocs=", 1)[0]
    assert "self.request_queue" in spawn_args
    assert "self._ready_event" in spawn_args
    assert "self._cohort_shutdown_flag" in spawn_args

    loop_source = WORKER_LOOP_PATH.read_text()
    for name in ("server_worker_main", "_server_worker_main_impl"):
        signature = loop_source.split(f"def {name}(", 1)[1].split("):", 1)[0]
        assert "shutdown_flag" in signature


def test_sentinel_grace_is_unchanged_by_the_abort_watchdog():
    """The watchdog shortens the expected teardown; the grace stays the
    unchanged safety net and must not be tuned down in this change."""
    source = WORKER_MANAGER_PATH.read_text()
    assert "_WORKER_SENTINEL_GRACE_S = 180.0" in source
    assert "_WORKER_SENTINEL_GRACE_S if sentinel_sent else 0.0" in source


def test_abort_watchdog_adds_no_env_vars():
    """No new env var gates this path: the cohort flag is the only trigger.
    None of torch's own error-handling switches (ASYNC_ERROR_HANDLING,
    BLOCKING_WAIT, the heartbeat monitor) rescues a rank blocked this way, so
    there is nothing to turn on instead of this watchdog."""
    loop_source = WORKER_LOOP_PATH.read_text()
    helpers = loop_source.split("def _shutdown_requested(", 1)[1]
    helpers = helpers.split("def _setup_nccl_env(", 1)[0]
    assert "os.environ" not in helpers
    assert "getenv" not in helpers


def _cohort_flag_probe_child(flag, observed, child_pid, deadline_s):
    """Worker side of the cohort-flag transport test, in a real spawned child.

    Builds the same zero-argument probe the supervisor installs on the worker
    and polls it, exactly as the decode loop does, then records whether the
    manager's store ever became visible here.  Module level and picklable
    because spawn re-imports this module in the child.
    """
    child_pid.value = os.getpid()
    probe = lambda: flag is not None and bool(flag.value)
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if probe():
            observed.value = 1
            return
        time.sleep(0.01)
    observed.value = 0


# --------------------------------------------------------------------------
# Attempt 3 (v3 spec): the decode-step boundary exit and the self-exit net.
#
# What the W5 bench established and these tests pin:
#   B3  the abort's release is SILENT -- the wedged sync returns normally and
#       in graph mode nothing ever raises, so only a flag test on the freed
#       rank's own execution path can retire it.
#   B4  that path has a FINITE window (the freed rank resumes, then re-wedges
#       permanently), so the test must sit on the first code after the sync.
#   B5  os._exit from a side thread ends a wedged process promptly with the
#       status as set, which makes a bounded self-exit net viable.
# --------------------------------------------------------------------------

BATCHGEN_WORKER_PATH = ROOT / "batchgen/batchgen_worker.py"

_worker_tree_cache: Dict[str, ast.Module] = {}


def _worker_tree() -> ast.Module:
    """Parse batchgen_worker.py once; it is ~17.5k lines."""
    if "tree" not in _worker_tree_cache:
        _worker_tree_cache["tree"] = ast.parse(BATCHGEN_WORKER_PATH.read_text())
    return _worker_tree_cache["tree"]


def _worker_class() -> ast.ClassDef:
    return next(
        node for node in _worker_tree().body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )


def _worker_method(name: str) -> ast.FunctionDef:
    return next(
        node for node in _worker_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _decode_loop() -> ast.While:
    """The decode loop whose per-step sync is the measured production wedge."""
    return next(
        node for node in ast.walk(_worker_method("decoding_continuous"))
        if isinstance(node, ast.While) and ast.unparse(node.test) == "decode_uuids"
    )


def _decode_shutdown_checks() -> List[ast.If]:
    return [
        node for node in _decode_loop().body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self._cohort_shutdown_requested()"
    ]


def _generate_outer_loop() -> ast.While:
    return next(
        node for node in _worker_method("generate").body
        if isinstance(node, ast.While)
        and isinstance(node.test, ast.Constant) and node.test.value is True
    )


def _collective_calls(node: ast.AST) -> List[Any]:
    """Every call in this subtree that reaches the GPU or a communicator."""
    found = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        name = ast.unparse(item.func)
        if (
            name.startswith("dist.")
            or name.startswith("torch.cuda.")
            or "synchronize" in name
            or "barrier" in name
            or "all_gather" in name
            or "all_reduce" in name
            or "broadcast" in name
        ):
            found.append((item.lineno, name))
    return found


def _load_worker_shutdown_seam():
    """Lift the worker's cohort-shutdown seam out of its source.

    ``batchgen_worker`` cannot be imported on a dev machine, so the three
    members that make up the seam are compiled into a bare class exactly as
    ``_load_worker_manager_class`` lifts out WorkerManager.  Taking them from
    the real source is the point: a renamed probe or a dropped latch fails here.
    """
    wanted = {
        "set_cohort_shutdown_probe",
        "_cohort_shutdown_requested",
        "cohort_shutdown_observed",
    }
    members = [
        node for node in _worker_class().body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in members} == wanted, (
        f"the worker's cohort-shutdown seam changed shape: "
        f"{sorted(node.name for node in members)}"
    )
    cls = ast.ClassDef(
        name="_Seam",
        bases=[],
        keywords=[],
        body=members,
        decorator_list=[],
        type_params=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[cls], type_ignores=[])
    )
    namespace: Dict[str, Any] = {"property": property}
    exec(compile(module, str(BATCHGEN_WORKER_PATH), "exec"), namespace)
    seam = namespace["_Seam"]

    def _new():
        obj = seam()
        obj._cohort_shutdown_probe = None
        obj._cohort_shutdown_observed = False
        return obj

    return _new


# ---- B3/B4: the decode-step boundary is where a freed rank is caught -----

def test_decode_loop_tests_the_cohort_flag_at_the_top_of_every_step():
    """The ordinary case: teardown began between steps, so the step must not
    start at all.  First statement in the loop, before any work is issued."""
    loop = _decode_loop()
    first = loop.body[0]
    assert isinstance(first, ast.If)
    assert ast.unparse(first.test) == "self._cohort_shutdown_requested()"


def test_decode_loop_tests_the_cohort_flag_right_after_the_token_sync():
    """B3/B4.  ``_new_tokens_ready.synchronize()`` IS the production wedge
    point: a rank whose peer died mid-decode sits there, below Python, and the
    cohort abort releases it by making that call return -- silently, with no
    exception then or ever in graph mode.  The freed rank has a finite window
    before it re-wedges, so the flag test must sit as close to that sync as the
    step's own invariants allow.

    "As close as the invariants allow" is exactly one statement further than
    the sync: the step's host-KV append drain MUST run first (see
    test_the_decode_exit_never_strands_in_flight_local_work).  Nothing else may
    come between.
    """
    loop = _decode_loop()
    sync_index = next(
        index for index, node in enumerate(loop.body)
        if isinstance(node, ast.Expr)
        and ast.unparse(node.value) == "_new_tokens_ready.synchronize()"
    )
    checks = [
        index for index, node in enumerate(loop.body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self._cohort_shutdown_requested()"
    ]
    after_sync = [index for index in checks if index > sync_index]
    assert after_sync, "no cohort-flag test follows the per-step token sync"
    check_index = min(after_sync)

    # Between the sync and the test: host-side timing, and the append drain
    # whose completion the teardown depends on.  Nothing else -- in particular
    # nothing that issues new GPU or collective work, which a rank that is
    # already supposed to be stopping must not do.
    allowed = {
        "time.perf_counter",
        "self._wait_pending_kv_append_tasks",
    }
    between = loop.body[sync_index + 1:check_index]
    for node in between:
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                assert ast.unparse(call.func) in allowed, (
                    f"{ast.unparse(call.func)} runs between the wedge-point "
                    "sync and the shutdown test"
                )


def test_the_decode_exit_never_strands_in_flight_local_work():
    """The decode teardown exit must quiesce what this rank already launched.

    The host-KV appends are std::async threads holding unlocked pointers into
    the worker view's page table, and an async host->GPU KV load can be in
    flight too.  The ordered teardown this exit leads to releases those
    mappings, so returning with either still running is a use-after-free during
    teardown -- and the existing per-step drain invariant
    (tests/test_worker_host_kv_append_completion.py) exists because an earlier
    version of that race corrupted KV pages in production.

    Both exits therefore route through one quiesce helper, and every wait in it
    is local: a cross-rank drain here would hang on the dead peer, which is the
    original defect.
    """
    checks = _decode_shutdown_checks()
    assert len(checks) == 2
    for check in checks:
        quiesce = [
            node for node in ast.walk(check)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func)
            == "self._quiesce_decode_for_cohort_shutdown"
        ]
        assert len(quiesce) == 1, (
            "each decode teardown exit must quiesce in-flight local work"
        )
        # It is handed the async load so the helper can wait on it.
        assert [ast.unparse(arg) for arg in quiesce[0].args] == [
            "pending_async_task"
        ]

    helper = _worker_method("_quiesce_decode_for_cohort_shutdown")

    # Local waits only.  No collective may appear here at all.
    assert not [
        name for _, name in _collective_calls(helper)
        if name.startswith("dist.")
    ], "the decode teardown quiesce must issue no collective"

    # The append drain defers its errors, which is what keeps it off the
    # cross-rank error sync.
    drains = [
        node for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self._wait_pending_kv_append_tasks"
    ]
    assert len(drains) == 1
    assert [(kw.arg, ast.unparse(kw.value)) for kw in drains[0].keywords] == [
        ("defer_errors", "True")
    ]

    # And it cannot raise: this runs on a path whose purpose is to unwind
    # normally, so a failed quiesce must not become an escaping exception.
    handlers = [
        node for node in ast.walk(helper) if isinstance(node, ast.ExceptHandler)
    ]
    assert len(handlers) == 2
    for handler in handlers:
        assert isinstance(handler.type, ast.Name)
        assert handler.type.id == "BaseException"
        assert not [
            node for node in ast.walk(handler) if isinstance(node, ast.Raise)
        ]


def test_decode_loop_shutdown_exit_returns_rather_than_breaking():
    """The exit must RETURN out of decoding_continuous.

    Breaking would fall into this loop's own post-loop cleanup, which drains
    the KV-append tasks with sync_distributed_errors=True -- i.e. issues a
    fresh dist.all_gather_object on the communicator that was just aborted.
    Returning issues no further GPU or collective work at all, which is the
    entire purpose of stopping at this boundary.
    """
    checks = _decode_shutdown_checks()
    assert len(checks) == 2, f"expected two boundary tests, found {len(checks)}"
    for check in checks:
        kinds = [type(node) for node in ast.walk(check)]
        assert ast.Return in kinds, "the boundary exit must return"
        assert ast.Break not in kinds, (
            "a break would run the post-loop cross-rank drain; see the "
            "sync_distributed_errors hazard this test's docstring names"
        )
        assert ast.Raise not in kinds, (
            "the boundary exit must unwind normally; the release from an "
            "aborted collective is not an exception (B3)"
        )
        # It latches, so the supervisor can name the path this rank took.
        assert any(
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "self._cohort_shutdown_observed"
            and ast.unparse(node.value) == "True"
            for node in ast.walk(check)
        )
        # And it issues nothing on the way out.
        assert _collective_calls(check) == []


def test_the_post_loop_drain_a_break_would_hit_is_really_a_collective():
    """Grounds the previous test's reason in the code rather than in prose.

    If this drain ever stops being a cross-rank operation the return-vs-break
    argument changes, and whoever changes it should see this test.
    """
    fn = _worker_method("decoding_continuous")
    loop = _decode_loop()
    tail = [node for node in fn.body if node.lineno > loop.end_lineno]
    drains = [
        node for node in ast.walk(ast.Module(body=tail, type_ignores=[]))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self._wait_pending_kv_append_tasks"
        and any(
            keyword.arg == "sync_distributed_errors"
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    assert len(drains) == 1, (
        "the post-loop cross-rank drain moved; re-derive whether the decode "
        "boundary may break instead of return"
    )
    inner = _worker_method("_wait_pending_kv_append_tasks")
    assert [
        name for _, name in _collective_calls(inner)
        if "all_gather" in name
    ], "the drain no longer issues a collective"


def test_decode_boundary_probe_and_latch_behave_as_a_plain_load():
    """Executable check of the seam itself, lifted from the worker's source."""
    make = _load_worker_shutdown_seam()

    unarmed = make()
    assert unarmed._cohort_shutdown_requested() is False
    assert unarmed.cohort_shutdown_observed is False

    armed = make()
    armed.set_cohort_shutdown_probe(lambda: False)
    assert armed._cohort_shutdown_requested() is False
    armed.set_cohort_shutdown_probe(lambda: True)
    assert armed._cohort_shutdown_requested() is True

    # The latch is what the supervisor reads, and it is not the probe: a rank
    # that never reached the decode boundary must not claim that exit path.
    assert armed.cohort_shutdown_observed is False
    armed._cohort_shutdown_observed = True
    assert armed.cohort_shutdown_observed is True


def test_decode_step_costs_one_probe_call_when_no_teardown_is_running():
    """The healthy path pays a None test, or one shared-byte load, per step --
    no lock, no allocation, no logging.  Counted on the real lifted seam."""
    make = _load_worker_shutdown_seam()
    worker = make()
    calls = []

    class _Byte:
        """Stands in for the lock-free shared byte, counting its loads."""

        def __init__(self) -> None:
            self._v = 0

        @property
        def value(self):
            calls.append("load")
            return self._v

    byte = _Byte()
    worker.set_cohort_shutdown_probe(lambda: bool(byte.value))

    for _ in range(1000):
        assert worker._cohort_shutdown_requested() is False

    assert calls == ["load"] * 1000


# ---- the unwind out of generate() ----------------------------------------

def test_generate_carries_the_decode_exit_out_before_the_next_collective():
    """The decode phase's cleanup ends in dist.barrier().  A rank that left the
    decode loop because of teardown must break BEFORE that barrier, or it
    issues a fresh collective on an aborted communicator."""
    outer = _generate_outer_loop()
    decode_phase = next(
        node for node in outer.body
        if isinstance(node, ast.While) and "has_in_decode" in ast.unparse(node.test)
    )
    latch_checks = [
        node for node in decode_phase.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self._cohort_shutdown_observed"
    ]
    assert len(latch_checks) == 1
    check = latch_checks[0]
    assert [type(node) for node in check.body] == [ast.Break]

    barriers = [
        lineno for lineno, name in _collective_calls(decode_phase)
        if name == "dist.barrier"
    ]
    assert barriers, "the decode phase no longer issues the cleanup barrier"
    assert check.lineno < min(barriers), (
        "the teardown break must precede the decode cleanup barrier"
    )


def test_generate_outer_boundary_tests_the_latch_and_never_the_probe():
    """Load-bearing distinction, and the reason this is not a flag test.

    An IDLE pool leaves through the queue sentinel a few lines below this
    check and is counted as ordered-via-sentinel.  The manager sets the cohort
    flag immediately BEFORE that sentinel, so a probe test here would quietly
    take over every clean teardown and relabel it.  Only a rank that actually
    left the decode loop at its boundary may exit here.
    """
    outer = _generate_outer_loop()
    first = outer.body[0]
    assert isinstance(first, ast.If)
    assert ast.unparse(first.test) == "self._cohort_shutdown_observed"
    assert [type(node) for node in first.body] == [ast.Break]

    assert "_cohort_shutdown_requested" not in ast.unparse(first), (
        "the outer generate() boundary must test the latch, not the probe, or "
        "a clean idle teardown stops being ordered-via-sentinel"
    )


def test_generate_issues_no_collective_after_its_loop():
    """Having broken out, the rank must reach its supervisor without touching
    a communicator again."""
    fn = _worker_method("generate")
    outer = _generate_outer_loop()
    tail = [node for node in fn.body if node.lineno > outer.end_lineno]
    assert tail, "generate() has no tail after its persistent loop"
    assert _collective_calls(ast.Module(body=tail, type_ignores=[])) == []


# ---- the plumbing seam ---------------------------------------------------

def test_the_supervisor_installs_the_probe_before_the_main_loop():
    """The seam, and where it is armed.  Set before the loop and never
    cleared, so it cannot be missing on any path that reaches decode."""
    impl = _worker_loop_function("_server_worker_main_impl")
    installs = [
        node for node in ast.walk(impl)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "worker.set_cohort_shutdown_probe"
    ]
    assert len(installs) == 1
    install = installs[0]

    # It hands down a zero-argument probe over the module's own flag helper,
    # so the flag's shape stays in the supervisor.
    assert len(install.args) == 1
    probe = install.args[0]
    assert isinstance(probe, ast.Lambda)
    assert probe.args.args == []
    assert ast.unparse(probe.body) == "_shutdown_requested(shutdown_flag)"

    guard = _main_loop_guard()
    assert install.lineno < guard.lineno, (
        "the probe must be installed before the main loop starts"
    )


def test_the_cohort_flag_reaches_a_spawned_child_as_the_worker_sees_it():
    """End-to-end transport: the manager's store must be visible through the
    probe in a real spawned process, which is how workers are started."""
    ctx = multiprocessing.get_context("spawn")
    flag = ctx.RawValue("b", 0)
    observed = ctx.RawValue("b", -1)
    child_pid = ctx.RawValue("i", 0)

    child = ctx.Process(
        target=_cohort_flag_probe_child,
        args=(flag, observed, child_pid, 30.0),
    )
    child.start()
    try:
        # The child is already polling the flag it was handed; this is the
        # manager's plain store, from the other side of the spawn boundary.
        flag.value = 1
        child.join(timeout=60)
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)

    assert child.exitcode == 0
    # Proof this really crossed a process boundary rather than reading the
    # parent's own memory: the observation was recorded by another pid.
    assert child_pid.value not in (0, os.getpid()), (
        f"the probe did not run in a separate process (pid={child_pid.value})"
    )
    assert observed.value == 1, (
        "a spawned worker did not see the cohort flag the manager stored to"
    )


# ---- B5: the self-exit safety net ---------------------------------------

def test_watchdog_self_exits_when_the_loop_never_leaves(caplog):
    """B5.  The abort frees a rank in milliseconds but cannot be acknowledged,
    and a rank it did NOT free will never leave on its own.  Holding the
    teardown open for it only trades this bounded window for the manager's
    grace and a SIGKILL, so the watchdog ends the process itself -- and says
    so first, which it can do because it holds the GIL."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    ns["_SHUTDOWN_SELF_EXIT_GRACE_S"] = 0.05
    exits: List[int] = []
    flag = _Flag(0)

    with caplog.at_level(logging.ERROR):
        watchdog = ns["_start_shutdown_abort_watchdog"](
            6, flag, threading.Event(), threading.Event(),
            live_process_groups=lambda: [_Group("0")],
            abort_one_group=lambda group: None,
            exit_process=exits.append,
        )
        flag.value = 1
        watchdog.join(timeout=30)

    assert not watchdog.is_alive()
    assert exits == [ns["_WORKER_SELF_EXIT_CODE"]]

    markers = [
        record.getMessage() for record in caplog.records
        if "worker exit path=" in record.getMessage()
    ]
    assert markers == [
        f"[shutdown] worker exit path=self-exit rank=6 pid={os.getpid()}"
    ], markers


def test_watchdog_does_not_self_exit_when_the_loop_leaves_in_the_window():
    """The net is a fallback, not the path.  A rank the abort freed leaves
    through the decode boundary well inside this window, and must then be
    allowed to run its ordered teardown -- which releases a large registered
    mapping and can take minutes."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    ns["_SHUTDOWN_SELF_EXIT_GRACE_S"] = 30.0
    exits: List[int] = []
    flag = _Flag(0)
    loop_exited = threading.Event()
    started = threading.Semaphore(0)

    def abort(group):
        started.release()

    watchdog = ns["_start_shutdown_abort_watchdog"](
        7, flag, loop_exited, threading.Event(),
        live_process_groups=lambda: [_Group("0")],
        abort_one_group=abort,
        exit_process=exits.append,
    )
    flag.value = 1
    # The abort went out, so the watchdog is now in its self-exit window...
    assert started.acquire(timeout=30)
    # ...and the loop retires inside it.
    loop_exited.set()
    watchdog.join(timeout=30)

    assert not watchdog.is_alive()
    assert exits == []


def test_a_clean_teardown_never_arms_the_self_exit():
    """Zero change to the healthy path: a worker that leaves its loop inside
    the abort grace is never aborted and never self-exits."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_FLAG_POLL_S"] = 0.01
    ns["_SHUTDOWN_SELF_EXIT_GRACE_S"] = 0.01
    exits: List[int] = []
    aborts: List[Any] = []
    flag = _Flag(0)
    loop_exited = threading.Event()

    watchdog = ns["_start_shutdown_abort_watchdog"](
        0, flag, loop_exited, threading.Event(),
        live_process_groups=lambda: [_Group("0")],
        abort_one_group=aborts.append,
        exit_process=exits.append,
    )
    flag.value = 1
    loop_exited.set()
    watchdog.join(timeout=30)

    assert not watchdog.is_alive()
    assert aborts == []
    assert exits == []


def test_the_self_exit_marker_is_flushed_before_the_process_ends():
    """``os._exit`` runs no handlers and flushes nothing, so the one line that
    attributes the exit is the most likely record to be lost."""
    fn = _worker_loop_function("_start_shutdown_abort_watchdog")
    flushes = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_flush_worker_logs"
    ]
    exits = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self_exit"
    ]
    assert len(flushes) == 1 and len(exits) == 1
    assert flushes[0].lineno < exits[0].lineno

    marker_logs = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "logging.error"
    ]
    assert len(marker_logs) == 1
    assert marker_logs[0].lineno < flushes[0].lineno


def test_the_log_flush_cannot_raise_on_the_way_out():
    """It runs while a teardown is already failing; raising there would
    replace a bounded self-exit with an unhandled error in a daemon thread."""
    ns = _load_worker_loop_helpers()

    class _Hostile(logging.Handler):
        def emit(self, record):
            pass

        def flush(self):
            raise RuntimeError("handler is wedged too")

    handler = _Hostile()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        ns["_flush_worker_logs"]()
    finally:
        root.removeHandler(handler)


# ---- the exit status, and what the manager does with it ------------------

def test_the_self_exit_status_collides_with_nothing_the_manager_reads():
    """Grounded choice, not a guess.  The manager reports a worker's exitcode
    verbatim as the reason a cohort went down, so the self-exit status must be
    distinguishable from every other status this worker can produce:
    0 the clean pool exit, 1 every fatal path, 128+n the signal translation in
    process_utils, and -n multiprocessing's rendering of a signal death.
    """
    ns = _load_worker_loop_helpers()
    code = ns["_WORKER_SELF_EXIT_CODE"]

    assert isinstance(code, int)
    assert code not in (0, 1), "collides with the clean and fatal exits"
    assert 1 < code < 128, (
        "128 and above collide with process_utils' 128+signum translation"
    )

    # Every other os._exit status this worker can take, from the source.
    loop_source = WORKER_LOOP_PATH.read_text()
    worker_source = BATCHGEN_WORKER_PATH.read_text()
    others = set()
    for source in (loop_source, worker_source):
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "os._exit"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                others.add(node.args[0].value)
    assert code not in others, f"{code} is already used as an exit status"


def test_the_manager_interprets_no_exit_status_specially():
    """The manager needs no change to reap a self-exited worker: it compares
    exitcode only against None (has it exited), never against a value."""
    tree = _worker_manager_tree()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        parts = [node.left] + node.comparators
        if not any(
            isinstance(part, ast.Attribute) and part.attr == "exitcode"
            for part in parts
        ):
            continue
        rendered = ast.unparse(node)
        assert all(
            isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
        ), f"the manager branches on an exitcode VALUE: {rendered}"
        assert all(
            isinstance(part, ast.Constant) and part.value is None
            for part in parts[1:]
        ), f"the manager compares exitcode to a value: {rendered}"


def test_the_signal_fallback_marker_skips_a_worker_that_exited_itself(
    monkeypatch, caplog
):
    """A self-exited worker is already dead when the fallback leg runs, so it
    is filtered out before the markers are logged and is never named as
    signal-fallback.  That filter is what makes the self-exit honest in the
    per-cohort marker accounting."""
    ns = _load_worker_loop_helpers()
    manager = _manager(monkeypatch)

    class _Proc:
        def __init__(self, pid: int, exitcode) -> None:
            self.pid = pid
            self.exitcode = exitcode

        def join(self, timeout=None) -> None:
            pass

        def terminate(self) -> None:
            self.exitcode = -15

        def kill(self) -> None:
            self.exitcode = -9

    self_exited = _Proc(2001, ns["_WORKER_SELF_EXIT_CODE"])
    forced = _Proc(2002, None)
    manager._join_lock = threading.Lock()
    manager.worker_process = SimpleNamespace(processes=[self_exited, forced])
    manager.args = SimpleNamespace(runtime_identity=SimpleNamespace(
        shm_prefix="run.", runtime_dir="/tmp/does-not-matter",
    ))
    del manager._stop_workers

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        manager._stop_workers()

    markers = [
        record.getMessage() for record in caplog.records
        if "worker exit path=" in record.getMessage()
    ]
    assert markers == [
        "[shutdown] worker exit path=signal-fallback rank-idx=1 pid=2002"
    ], markers
    # The self-exited worker was reaped untouched, keeping its own status.
    assert self_exited.exitcode == ns["_WORKER_SELF_EXIT_CODE"]


# ---- marker accounting ---------------------------------------------------

def test_the_decode_boundary_marker_joins_the_exit_path_enumeration():
    """Per-cohort accounting sums ordered-via-{sentinel,flag,abort,
    decode-boundary} plus the manager's signal-fallback, so the new marker has
    to exist as a value and be distinct from the others."""
    ns = _load_worker_loop_helpers()
    ordered = [
        ns["_EXIT_PATH_SENTINEL"],
        ns["_EXIT_PATH_FLAG"],
        ns["_EXIT_PATH_ABORT"],
        ns["_EXIT_PATH_DECODE_BOUNDARY"],
    ]
    assert ordered == [
        "ordered-via-sentinel",
        "ordered-via-flag",
        "ordered-via-abort",
        "ordered-via-decode-boundary",
    ]
    assert len(set(ordered)) == 4
    # Not an ordered exit, and deliberately named so it cannot be summed into
    # one by mistake.
    assert ns["_EXIT_PATH_SELF_EXIT"] == "self-exit"
    assert not ns["_EXIT_PATH_SELF_EXIT"].startswith("ordered-via-")

    literals = {
        node.value for node in ast.walk(_worker_loop_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert set(ordered) | {"self-exit"} <= literals


def test_the_supervisor_names_the_decode_boundary_exit():
    """The rank reports having taken that exit; the supervisor does not infer
    it from the flag.  A pool shutdown that merely coincided with teardown is
    a different path and must not claim this marker."""
    impl = _worker_loop_function("_server_worker_main_impl")
    checks = [
        node for node in ast.walk(impl)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "worker.cohort_shutdown_observed"
    ]
    assert len(checks) == 1
    check = checks[0]

    assert any(
        isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "exit_path"
        and ast.unparse(node.value) == "_EXIT_PATH_DECODE_BOUNDARY"
        for node in check.body
    )
    assert any(isinstance(node, ast.Break) for node in check.body)
    assert not [node for node in ast.walk(check) if isinstance(node, ast.Raise)]

    # It runs after generate_persistent returns and before the pool_shutdown
    # sentinel, whose cohort is the one being torn down.
    persistent = next(
        node for node in ast.walk(impl)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "worker.generate_persistent"
    )
    sentinel = next(
        node for node in ast.walk(impl)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "response_queue.put"
        and "pool_shutdown" in ast.unparse(node)
    )
    assert persistent.lineno < check.lineno < sentinel.lineno


def test_the_self_exit_window_is_bounded_and_needs_no_env_var():
    ns = _load_worker_loop_helpers()
    # Long enough for a freed rank to notice the flag and leave on its own
    # (measured working window after the abort: a few seconds), short enough
    # that it is not the manager's grace by another name.
    assert 0 < ns["_SHUTDOWN_SELF_EXIT_GRACE_S"] <= 60
    assert ns["_SHUTDOWN_SELF_EXIT_GRACE_S"] > ns["_SHUTDOWN_ABORT_GRACE_S"]

    helpers = WORKER_LOOP_PATH.read_text().split("def _shutdown_requested(", 1)[1]
    helpers = helpers.split("def _setup_nccl_env(", 1)[0]
    assert "os.environ" not in helpers
    assert "getenv" not in helpers
