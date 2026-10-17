"""Dev-machine lifecycle tests for the restartable worker pool."""

from __future__ import annotations

import ast
import logging
import os
import queue
import signal
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
    # The teardown budgets come from the module itself, so a test exercising
    # _stop_workers uses the real values rather than a copy that can drift.
    budgets = [
        item for item in tree.body
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {
                "_WORKER_SENTINEL_GRACE_S", "_WORKER_KILL_REAP_TIMEOUT_S",
            }
            for target in item.targets
        )
    ]
    assert len(budgets) == 2
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
        # _stop_workers signals children through these directly.
        "os": os,
        "signal": signal,
    }
    module = ast.Module(body=budgets + [node], type_ignores=[])
    exec(compile(module, str(ROOT / "batchgen/server/worker_manager.py"), "exec"), namespace)
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


class _CohortQueueContext:
    """Stub mp context handing out a fresh _CohortQueue per request."""

    def __init__(self) -> None:
        self.created: List[_CohortQueue] = []
        self.events: List[threading.Event] = []

    def Queue(self) -> _CohortQueue:  # noqa: N802 - mirrors mp context API
        channel = _CohortQueue(f"channel#{len(self.created)}")
        self.created.append(channel)
        return channel

    def Event(self) -> threading.Event:  # noqa: N802 - mirrors mp context API
        """The cohort shutdown flag. A threading.Event has the same surface
        (set/is_set/wait) as the mp.Event the real context returns."""
        event = threading.Event()
        self.events.append(event)
        return event


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
# cohort shutdown Event reaches it out of band, and a watchdog thread aborts
# the communicators so the blocked collective returns with an error the main
# loop retires through the ordered teardown.
# --------------------------------------------------------------------------

WORKER_LOOP_PATH = ROOT / "batchgen/server_worker_main_loop.py"

_WORKER_LOOP_HELPERS = (
    "_abort_all_process_groups",
    "_resolve_main_loop_exit",
    "_start_shutdown_abort_watchdog",
)
_WORKER_LOOP_CONSTANTS = (
    "_SHUTDOWN_ABORT_GRACE_S",
    "_EXIT_PATH_SENTINEL",
    "_EXIT_PATH_ABORT",
    "_EXIT_PATH_SIGNAL",
)


def _worker_loop_tree() -> ast.Module:
    return ast.parse(WORKER_LOOP_PATH.read_text())


def _load_worker_loop_helpers() -> Dict[str, Any]:
    """Execute the shutdown helpers without importing the worker module.

    ``server_worker_main_loop`` imports torch and the whole worker, neither of
    which a dev machine can load, so the helpers are lifted out of the source
    exactly as ``_load_worker_manager_class`` lifts out WorkerManager.  The
    module constants come from the same source, so a changed grace window or
    marker cannot pass unnoticed.
    """
    tree = _worker_loop_tree()
    body: List[Any] = [
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
    assert len(body) == len(_WORKER_LOOP_CONSTANTS)
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Optional": Optional,
        "threading": threading,
        "time": time,
        "logging": logging,
        # Only ever touched in annotations here; threading.Event has the same
        # set/is_set/wait surface as the mp.Event the manager really passes.
        "mp": SimpleNamespace(Event=threading.Event),
        "torch": SimpleNamespace(__version__="stub-for-dev-machine"),
    }
    module = ast.Module(body=body + functions, type_ignores=[])
    exec(compile(module, str(WORKER_LOOP_PATH), "exec"), namespace)
    return namespace


def test_shutdown_watchdog_is_silent_when_the_loop_exits_in_the_window():
    """The graceful path must stay the normal one: a worker that is not stuck
    consumes the sentinel/broadcast inside the window, and the watchdog must
    then abort nothing at all."""
    ns = _load_worker_loop_helpers()
    aborts: List[str] = []
    shutdown_event = threading.Event()
    loop_exited = threading.Event()

    thread = ns["_start_shutdown_abort_watchdog"](
        0,
        shutdown_event,
        loop_exited,
        abort_process_groups=lambda: aborts.append("abort"),
    )
    assert thread is not None

    shutdown_event.set()
    # The main loop left on its own, well inside the grace window.
    loop_exited.set()

    thread.join(timeout=ns["_SHUTDOWN_ABORT_GRACE_S"] + 10)
    assert not thread.is_alive()
    assert aborts == []


def test_shutdown_watchdog_never_fires_without_the_cohort_event():
    """The watchdog's only trigger is the cohort Event, which nothing but
    teardown sets. With no Event there is nothing to arm, and with an unset
    Event the thread waits forever rather than acting on a timeout."""
    ns = _load_worker_loop_helpers()
    aborts: List[str] = []

    assert ns["_start_shutdown_abort_watchdog"](
        0, None, threading.Event(),
        abort_process_groups=lambda: aborts.append("abort"),
    ) is None

    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    unset = threading.Event()
    thread = ns["_start_shutdown_abort_watchdog"](
        0, unset, threading.Event(),
        abort_process_groups=lambda: aborts.append("abort"),
    )
    # Far longer than the (shortened) grace window: an unset Event must not
    # start the window at all.
    thread.join(timeout=0.5)
    assert thread.is_alive()
    assert aborts == []


def test_shutdown_watchdog_aborts_once_when_the_loop_is_stuck():
    """A loop still running after the window is blocked in a collective, and
    only an abort makes that collective return. The grace window is shortened
    here for speed; its real value is asserted separately."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    aborts: List[str] = []
    shutdown_event = threading.Event()
    # Never set: this stands in for a main thread blocked inside NCCL.
    stuck_loop = threading.Event()

    thread = ns["_start_shutdown_abort_watchdog"](
        3,
        shutdown_event,
        stuck_loop,
        abort_process_groups=lambda: aborts.append("abort"),
    )
    shutdown_event.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert aborts == ["abort"]


def test_shutdown_watchdog_survives_a_failing_abort():
    """An abort that raises must not kill the watchdog silently; the manager's
    signal fallback still covers the worker, so the thread logs and ends."""
    ns = _load_worker_loop_helpers()
    ns["_SHUTDOWN_ABORT_GRACE_S"] = 0.01
    shutdown_event = threading.Event()

    def exploding_abort():
        raise RuntimeError("abort failed")

    thread = ns["_start_shutdown_abort_watchdog"](
        1, shutdown_event, threading.Event(),
        abort_process_groups=exploding_abort,
    )
    shutdown_event.set()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_abort_uses_the_all_groups_torch_api_in_one_call():
    """Every process group this worker created must be aborted, default group
    included, and in one call: ``_abort_process_group()`` with no group aborts
    all registered groups inside one ncclGroupStart/End region, which is what
    keeps two communicators' aborts from deadlocking each other. Aborting
    group by group from Python would reintroduce that deadlock, so the
    traversal order is torch's and this asserts the single all-groups call."""
    tree = _worker_loop_tree()
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_abort_all_process_groups"
    )
    imported = [
        alias.name
        for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom)
        and node.module == "torch.distributed.distributed_c10d"
        for alias in node.names
    ]
    assert imported == ["_abort_process_group"]

    calls = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_abort_process_group"
    ]
    assert len(calls) == 1
    # No group argument: the all-groups form, never one group at a time.
    assert calls[0].args == [] and calls[0].keywords == []

    # A missing API must fail with that API named, never degrade to no abort.
    source = WORKER_LOOP_PATH.read_text()
    assert "_abort_process_group is " in source


def test_main_loop_guard_routes_an_abort_to_the_ordered_teardown():
    ns = _load_worker_loop_helpers()
    shutdown_event = threading.Event()
    shutdown_event.set()

    exit_path = ns["_resolve_main_loop_exit"](
        2, shutdown_event, RuntimeError("NCCL communicator was aborted"),
    )

    assert exit_path == ns["_EXIT_PATH_ABORT"] == "ordered-via-abort"


@pytest.mark.parametrize("shutdown_event", [None, threading.Event()])
def test_main_loop_guard_reraises_a_real_collective_failure(shutdown_event):
    """Without the cohort Event the same exception shape is a genuine NCCL
    failure and must stay exactly as fatal as it is today."""
    ns = _load_worker_loop_helpers()
    error = RuntimeError("NCCL communicator was aborted")

    with pytest.raises(RuntimeError) as excinfo:
        ns["_resolve_main_loop_exit"](2, shutdown_event, error)

    assert excinfo.value is error


def test_main_loop_has_exactly_one_abort_guard_at_its_boundary():
    """The guard belongs at the main-loop boundary, not sprinkled over the
    collective call sites, and the ordered teardown must run after it rather
    than inside the handler."""
    tree = _worker_loop_tree()
    impl = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_server_worker_main_impl"
    )
    guards = [
        node for node in impl.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(inner, ast.While)
            and isinstance(inner.test, ast.Constant)
            and inner.test.value is True
            for inner in node.body
        )
    ]
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

    # The ordered teardown sits after the guard, so the sentinel path and the
    # abort path run the SAME release sequence.
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
    """``_abort_process_group()`` clears torch's default group, so a later
    unconditional ``destroy_process_group()`` asserts. The ordered teardown
    must guard it, and the C++ release must still run first."""
    tree = _worker_loop_tree()
    impl = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_server_worker_main_impl"
    )
    # Only the ordered teardown, i.e. everything after the main-loop guard.
    # The signal handler defined earlier in the function destroys the group
    # too, and that path is deliberately unchanged.
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

    # destroy_process_group() is reached only through the initialized guard:
    # after an abort torch has already dropped the default group and the
    # unconditional call would assert, taking the ordered exit down with it.
    destroy_stmt = statements[destroy_at]
    assert isinstance(destroy_stmt, ast.If)
    guard = ast.unparse(destroy_stmt.test)
    assert guard == "dist.is_available() and dist.is_initialized()"


def test_worker_exit_paths_are_all_logged():
    ns = _load_worker_loop_helpers()
    assert ns["_EXIT_PATH_SENTINEL"] == "ordered-via-sentinel"
    assert ns["_EXIT_PATH_ABORT"] == "ordered-via-abort"
    assert ns["_EXIT_PATH_SIGNAL"] == "signal-fallback"
    # A few seconds, as designed: long enough for a healthy worker to leave
    # its loop, short enough that the 180s grace is no longer paid.
    assert 0 < ns["_SHUTDOWN_ABORT_GRACE_S"] <= 30

    source = WORKER_LOOP_PATH.read_text()
    assert source.count('"[shutdown] worker exit path=%s rank=%s"') == 2
    assert "_EXIT_PATH_SIGNAL," in source
    assert "exit_path, global_rank" in source


def test_teardown_sets_the_cohort_event_before_the_sentinel(monkeypatch):
    """A rank blocked in a collective reads neither the queue sentinel nor the
    status broadcast, so the out-of-band Event must be set first."""
    manager = _manager(monkeypatch)
    order: List[str] = []

    class _RecordingEvent:
        def __init__(self) -> None:
            self._set = False

        def set(self) -> None:
            order.append("cohort-event-set")
            self._set = True

        def is_set(self) -> bool:
            return self._set

    manager._cohort_shutdown_event = _RecordingEvent()
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

    assert order == ["cohort-event-set", "sentinel:None"]
    assert manager._cohort_shutdown_event.is_set()


def test_restart_rotates_the_cohort_shutdown_event(monkeypatch):
    """The Event is cohort-scoped like the channel pair: a new generation must
    never start up already holding the previous teardown's flag."""
    manager = _manager(monkeypatch)
    retired = manager._cohort_shutdown_event
    assert retired is not None
    retired.set()

    manager.restart_workers(timeout_s=1)

    fresh = manager._cohort_shutdown_event
    assert fresh is not None
    assert fresh is not retired
    assert not fresh.is_set()
    # The retired Event is abandoned with its cohort, never reused or cleared.
    assert retired.is_set()


def test_spawn_hands_the_cohort_event_to_every_worker():
    """The Event must reach every LOCAL rank, so it travels as a spawn arg
    beside the channels and the readiness events."""
    source = (ROOT / "batchgen/server/worker_manager.py").read_text()
    spawn_args = source.split("self.worker_process = mp.spawn(", 1)[1]
    spawn_args = spawn_args.split("nprocs=", 1)[0]
    assert "self.request_queue" in spawn_args
    assert "self._ready_event" in spawn_args
    assert "self._cohort_shutdown_event" in spawn_args

    loop_source = WORKER_LOOP_PATH.read_text()
    for name in ("server_worker_main", "_server_worker_main_impl"):
        signature = loop_source.split(f"def {name}(", 1)[1].split("):", 1)[0]
        assert "shutdown_event" in signature


def test_sentinel_grace_is_unchanged_by_the_abort_watchdog():
    """The watchdog shortens the expected teardown; the grace stays the
    unchanged safety net and must not be tuned down in this change."""
    source = (ROOT / "batchgen/server/worker_manager.py").read_text()
    assert "_WORKER_SENTINEL_GRACE_S = 180.0" in source
    assert "_WORKER_SENTINEL_GRACE_S if sentinel_sent else 0.0" in source


def test_abort_watchdog_adds_no_env_vars():
    """No new env var gates this path: the cohort Event is the only trigger."""
    loop_source = WORKER_LOOP_PATH.read_text()
    guard = loop_source.split("def _start_shutdown_abort_watchdog(", 1)[1]
    guard = guard.split("def _setup_nccl_env", 1)[0]
    assert "os.environ" not in guard
    assert "getenv" not in guard
