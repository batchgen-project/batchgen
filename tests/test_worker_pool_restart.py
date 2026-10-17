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

    def Queue(self) -> _CohortQueue:  # noqa: N802 - mirrors mp context API
        channel = _CohortQueue(f"channel#{len(self.created)}")
        self.created.append(channel)
        return channel


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
