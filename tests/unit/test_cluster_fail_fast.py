"""Cluster fail-fast: one node's fatal error or loss ends every node.

CPU only. The follower's real poll runs against an in-memory control store on
an injected clock, so a 600 s join grace costs no real time; the manager is
the lifted ``WorkerManager`` class the restart tests use, with its teardown
and its process exit replaced by recorders.
"""

from __future__ import annotations

import ast
import copy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_restart_coordinator import (  # noqa: E402
    FakeStore,
    coordinator_module,
    server_args_module,
)
from tests.test_worker_pool_restart import _manager  # noqa: E402

RestartCoordinator = coordinator_module.RestartCoordinator
FollowerHooks = coordinator_module._FollowerHooks
HTTP_SERVER_PATH = ROOT / "batchgen" / "server" / "http_server.py"


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _coordinator(store, clock, *, node_rank=1, nnodes=2, **kwargs):
    node = RestartCoordinator(
        nnodes,
        node_rank,
        "127.0.0.1",
        0,
        store_factory=lambda *_args, **_kw: store,
        poll_interval_s=0.01,
        request_wait_s=0.5,
        error_backoff_s=0.01,
        clock=clock,
        **kwargs,
    )
    node.start()
    return node


class Seen:
    """Every report the follower hands to its manager, in order."""

    def __init__(self) -> None:
        self.restarts: list = []
        self.down: list = []
        self.lost: list = []
        self.control_lost: list = []


def _watch(node, *, state="serving", generation=1, ready=True, seen=None):
    """Begin a watch without its threads, so each poll is stepped by hand.

    Mirrors what start_follower sets up before it launches the threads; the
    threaded case below covers start_follower itself.
    """
    seen = seen or Seen()
    node._watch_started_at = node._clock()
    node._last_store_ok = node._watch_started_at
    node._epoch = node._read_epoch()
    hooks = FollowerHooks(
        state=lambda: state,
        generation=lambda: generation,
        ready=lambda: ready,
        on_peer_restart=seen.restarts.append,
        on_peer_down=lambda *args: seen.down.append(args),
        on_peer_lost=lambda *args: seen.lost.append(args),
        on_control_lost=seen.control_lost.append,
    )
    return hooks, seen


def _run_watchdog(node, hooks, clock, seconds, *, poll=False, beat_every=None):
    """Let ``seconds`` pass with the watchdog ticking on schedule.

    The clock moves one watchdog tick at a time, as it does for a watchdog
    that is not starved; a single jump past CONTROL_DEAD_S is the starved
    case instead, which the watchdog deliberately does not judge. ``poll``
    runs a follower poll every tick, ``beat_every`` a heartbeat that often.
    """
    tick = node._watchdog_tick_s
    elapsed = since_beat = 0.0
    while elapsed < seconds - 1e-9:
        step = min(tick, seconds - elapsed)
        clock.advance(step)
        elapsed += step
        since_beat += step
        if poll:
            node._follow_once(hooks)
        if beat_every is not None and since_beat >= beat_every - 1e-9:
            node._beat()
            since_beat = 0.0
        node._check_control(hooks)


# --------------------------------------------------------------- heartbeats


def test_a_peer_whose_heartbeat_stops_is_lost_after_peer_dead_s_once():
    store, clock = FakeStore(), FakeClock()
    node0 = _coordinator(store, clock, node_rank=0)
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1)

    node0._beat()
    node1._follow_once(hooks)
    clock.advance(coordinator_module.PEER_DEAD_S - 0.1)
    node1._follow_once(hooks)
    assert seen.lost == []

    clock.advance(0.1)
    node1._follow_once(hooks)
    assert seen.lost == [(0, pytest.approx(coordinator_module.PEER_DEAD_S))]

    clock.advance(60)
    node1._follow_once(hooks)
    assert len(seen.lost) == 1


def test_a_beating_peer_is_never_lost_however_long_this_follower_was_busy():
    """Liveness is whether the value moved, so a follower that was blocked in
    a restart for minutes does not mistake a live peer for a dead one."""
    store, clock = FakeStore(), FakeClock()
    node0 = _coordinator(store, clock, node_rank=0)
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1)

    node0._beat()
    node1._follow_once(hooks)
    clock.advance(300)
    node0._beat()
    node1._follow_once(hooks)
    clock.advance(coordinator_module.PEER_DEAD_S - 1)
    node0._beat()
    node1._follow_once(hooks)

    assert seen.lost == []


def test_a_peer_that_never_beats_is_lost_only_after_the_join_grace():
    store, clock = FakeStore(), FakeClock()
    node0 = _coordinator(store, clock, node_rank=0, nnodes=3)
    _node2 = _coordinator(store, clock, node_rank=2, nnodes=3)
    hooks, seen = _watch(node0)

    # Node 1 beats once and then stays silent; node 2 never beats at all.
    store.add("alive/node1", 1)
    node0._follow_once(hooks)
    clock.advance(coordinator_module.PEER_DEAD_S)
    node0._follow_once(hooks)
    assert seen.lost == [(1, pytest.approx(coordinator_module.PEER_DEAD_S))]

    clock.advance(coordinator_module.JOIN_GRACE_S - coordinator_module.PEER_DEAD_S - 1)
    node0._follow_once(hooks)
    assert len(seen.lost) == 1

    clock.advance(1)
    node0._follow_once(hooks)
    assert seen.lost[1] == (2, pytest.approx(coordinator_module.JOIN_GRACE_S))


def test_the_heartbeat_runs_on_its_own_thread_and_a_closed_peer_is_lost():
    """The real threads: node 1's heartbeat keeps node 0's watch quiet until
    node 1 closes, and only the clock advancing then makes it lost."""
    store, clock = FakeStore(), FakeClock()
    node0 = _coordinator(store, clock, node_rank=0, heartbeat_s=0.005)
    node1 = _coordinator(store, clock, node_rank=1, heartbeat_s=0.005)
    lost = []
    for node in (node0, node1):
        node.start_follower(
            state=lambda: "serving",
            generation=lambda: 1,
            on_peer_restart=lambda request: None,
            on_peer_down=lambda *args: None,
            on_peer_lost=lambda *args: lost.append(args),
            on_control_lost=lambda age: None,
        )
    names = {thread.name for thread in threading.enumerate()}
    assert {"restart-coordinator", "restart-coordinator-heartbeat"} <= names
    assert _wait_until(lambda: int(store.snapshot().get("alive/node1", 0)) > 3)

    # Beating: a jump far past the limit changes nothing.
    clock.advance(10 * coordinator_module.PEER_DEAD_S)
    time.sleep(0.05)
    assert lost == []

    node1.close()
    time.sleep(0.05)
    clock.advance(coordinator_module.PEER_DEAD_S)
    assert _wait_until(lambda: lost and lost[0][0] == 1)
    node0.close()


BINDING_PATH = ROOT / "core" / "batchgen_Binding.cpp"

# Native calls the server process makes while its heartbeat thread runs that
# block for longer than PEER_DEAD_S: the Host-KV region creation (tens of
# seconds for a large budget) and the wait for the distributed weight store's
# cross-node bootstrap (minutes). Holding the GIL there stops the heartbeat on
# every node at once, and the peers declare each other lost.
_LONG_STARTUP_BINDINGS = (
    ('"initialize", &Manager::Initialize', 'py::arg("create_region")'),
    ('"wait_ready", &DistributedWeightDaemon::WaitReady',
     'py::arg("timeout_seconds")'),
    ('"initialize", &kv::HostPrefixCacheCoordinator::Initialize',
     'py::arg("create_region")'),
)


def _binding_def(source: str, head: str) -> str:
    start = source.index(head)
    end = source.find(".def(", start + 1)
    return source[start:end if end != -1 else len(source)]


@pytest.mark.parametrize("head,arg", _LONG_STARTUP_BINDINGS)
def test_long_startup_native_calls_release_the_gil(head, arg):
    definition = _binding_def(BINDING_PATH.read_text(), head)
    assert arg in definition
    assert "py::call_guard<py::gil_scoped_release>()" in definition, (
        f"{head} must release the GIL, or the cluster heartbeat stalls"
    )


def test_a_single_node_server_starts_no_heartbeat_and_opens_no_store():
    def refuse(*_args, **_kwargs):
        raise AssertionError("a single-node server must not open a control store")

    node = RestartCoordinator(1, 0, "127.0.0.1", 0, store_factory=refuse)
    node.start()
    node.start_follower(
        state=lambda: "serving",
        generation=lambda: 1,
        on_peer_restart=lambda request: pytest.fail("no follower may run"),
        on_peer_down=lambda *args: pytest.fail("no follower may run"),
        on_peer_lost=lambda *args: pytest.fail("no follower may run"),
        on_control_lost=lambda age: pytest.fail("no follower may run"),
    )
    assert not [
        thread for thread in threading.enumerate()
        if thread.name.startswith("restart-coordinator")
    ]
    assert node.control_error is None
    node.close()


# ------------------------------------------------------ control-store loss


def test_store_errors_shorter_than_control_dead_s_then_recovery_never_trip():
    store, clock = FakeStore(), FakeClock()
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1)

    store.error = ConnectionError("connection reset")
    _run_watchdog(node1, hooks, clock, coordinator_module.CONTROL_DEAD_S - 0.5,
                  poll=True)
    assert node1.control_error == "coordinator unreachable: connection reset"

    store.error = None
    node1._follow_once(hooks)
    assert node1.control_error is None

    # The count restarts after a recovery: a second short outage is short.
    store.error = ConnectionError("connection reset")
    _run_watchdog(node1, hooks, clock, coordinator_module.CONTROL_DEAD_S - 0.5,
                  poll=True)
    assert seen.control_lost == []
    assert not node1._store_dead


def test_store_errors_for_control_dead_s_report_node_zero_lost_once():
    store, clock = FakeStore(), FakeClock()
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1)

    store.error = ConnectionError("connection refused")
    _run_watchdog(node1, hooks, clock, coordinator_module.CONTROL_DEAD_S,
                  poll=True)
    assert seen.control_lost == [pytest.approx(coordinator_module.CONTROL_DEAD_S)]
    assert node1._store_dead
    assert node1.control_error == "coordinator unreachable: no store reply for 10s"

    _run_watchdog(node1, hooks, clock, 60, poll=True)
    assert len(seen.control_lost) == 1


def test_node_zero_never_trips_the_control_store_detector():
    """Node 0 hosts the store in its own process; failing calls there mean
    this very server is stopping, never that node 0 is gone."""
    store, clock = FakeStore(), FakeClock()
    node0 = _coordinator(store, clock, node_rank=0)
    hooks, seen = _watch(node0)

    store.error = ConnectionError("store closed")
    _run_watchdog(node0, hooks, clock, 5 * coordinator_module.CONTROL_DEAD_S,
                  poll=True)
    assert seen.control_lost == []


def test_the_heartbeat_alone_keeps_the_store_alive_while_the_follower_is_busy():
    """A followed restart holds the follower for minutes; the heartbeat's
    successes are what tell the watchdog node 0 still answers."""
    store, clock = FakeStore(), FakeClock()
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1)

    _run_watchdog(node1, hooks, clock, 10 * coordinator_module.CONTROL_DEAD_S,
                  beat_every=coordinator_module.HEARTBEAT_S)
    assert seen.control_lost == []


def _starved(node1, hooks, clock):
    """The watchdog ticks once, then neither it nor anything else that stamps
    a store reply runs for longer than CONTROL_DEAD_S -- a long call holding
    the interpreter -- and the watchdog is the first thread to run again."""
    node1._check_control(hooks)
    clock.advance(coordinator_module.CONTROL_DEAD_S + 5)
    node1._check_control(hooks)


def test_a_starved_watchdog_waits_one_tick_and_a_reply_landing_meanwhile_wins():
    store, clock = FakeStore(), FakeClock()
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1)

    _starved(node1, hooks, clock)
    # The silence is stale on both sides, so this tick does not judge it.
    assert seen.control_lost == []
    assert not node1._store_dead

    # The heartbeat lands before the next tick: node 0 was alive all along.
    node1._beat()
    _run_watchdog(node1, hooks, clock, 3 * coordinator_module.CONTROL_DEAD_S,
                  beat_every=coordinator_module.HEARTBEAT_S)
    assert seen.control_lost == []


def test_a_starved_watchdog_still_finds_a_dead_store_on_the_next_tick():
    store, clock = FakeStore(), FakeClock()
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1)

    store.error = ConnectionError("connection reset")
    _starved(node1, hooks, clock)
    assert seen.control_lost == []

    _run_watchdog(node1, hooks, clock, node1._watchdog_tick_s, poll=True)
    assert seen.control_lost == [
        pytest.approx(coordinator_module.CONTROL_DEAD_S + 5 + node1._watchdog_tick_s)
    ]
    assert node1._store_dead


def test_a_dead_store_is_never_called_again_and_close_waits_on_nothing():
    """Once declared dead, every path the exit takes returns at once: a hung
    store would otherwise hang the exit too."""
    store, clock = FakeStore(), FakeClock()
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1)
    closed = []
    store.close = lambda: closed.append(True)
    _run_watchdog(node1, hooks, clock, coordinator_module.CONTROL_DEAD_S)
    assert seen.control_lost and node1._store_dead
    calls = list(store.calls)

    node1.fail_phase(1, "P0", "stopping")
    node1.publish_down(3, "rank 0 exited")
    node1.publish_result(1, {"state": "workers_down"})
    node1._beat()
    assert node1._follow_once(hooks) == node1._error_backoff_s
    stuck = threading.Thread(target=threading.Event().wait, daemon=True)
    stuck.start()
    node1._follower = stuck
    started = time.monotonic()
    node1.close()

    assert time.monotonic() - started < 1.0
    assert store.calls == calls
    assert closed == []
    assert node1._store is None and node1._abandoned_store is store


# ----------------------------------------------- reports before readiness


def test_a_down_key_is_reported_with_its_generation_before_readiness():
    """Start-up is watched too: the peer's down key reaches a node that is
    not ready yet, while a restart epoch only moves its baseline."""
    store, clock = FakeStore(), FakeClock()
    node0 = _coordinator(store, clock, node_rank=0)
    node1 = _coordinator(store, clock, node_rank=1)
    hooks, seen = _watch(node1, generation=0, ready=False)

    node0.begin_restart(5.0)
    node0.publish_down(0, "rank 3 failed to load weights")
    node1._follow_once(hooks)

    assert seen.down == [(0, "rank 3 failed to load weights", 0)]
    assert seen.restarts == []
    assert node1.epoch == 1


# ------------------------------------------------------- the manager side


class _Exited(BaseException):
    """Stands in for os._exit: nothing after it runs."""


class _ExitState:
    reason = None

    def set_failure(self, reason, exc):
        if self.reason is not None:
            return False
        self.reason = reason
        return True

    def is_failed(self):
        return self.reason is not None

    def clear(self):
        self.reason = None


def _node_manager(
    monkeypatch, *, node_rank=1, fail_fast=True, first_ready=True,
    store=None, clock=None, **coordinator_options,
):
    """A lifted WorkerManager on a real coordinator, every exit recorded."""
    manager = _manager(monkeypatch)
    store = store or FakeStore()
    clock = clock or FakeClock()
    manager.args.nnodes = 2
    manager.args.node_rank = node_rank
    manager._cluster_fail_fast = fail_fast
    manager._first_ready = first_ready
    manager._monitor_interval_s = 0.01
    coordinator = _coordinator(
        store, clock, node_rank=node_rank, **coordinator_options
    )
    manager._restart_coordinator = coordinator
    manager._worker_exit_state = _ExitState()
    events = []
    manager._stop_workers = lambda: events.append("teardown")
    manager._reconcile = lambda reason, fail_in_flight: events.append("reconcile")
    manager.stop = lambda: events.append("stop")

    def exit_process(code, reason, **kwargs):
        events.append(("exit", code, reason))
        raise _Exited

    def record_exit(code, reason, **kwargs):
        events.append(("record_exit", code, reason))

    cls = type(manager)
    monkeypatch.setitem(
        cls._exit_cluster.__globals__,
        "eventlog_process",
        SimpleNamespace(record_exit=record_exit, exit_process=exit_process),
    )
    monkeypatch.setattr(
        cls, "_request_server_shutdown",
        staticmethod(lambda: events.append("sigint")),
    )
    return manager, store, clock, coordinator, events


def _manager_hooks(manager):
    """The hooks _start_restart_follower installs, minus the threads."""
    return FollowerHooks(
        state=lambda: manager.state,
        generation=lambda: manager.generation,
        ready=lambda: manager._first_ready,
        on_peer_restart=manager._run_followed_restart,
        on_peer_down=manager._handle_peer_cohort_down,
        on_peer_lost=manager._handle_peer_lost,
        on_control_lost=manager._handle_control_lost,
    )


def test_the_manager_installs_every_liveness_hook_on_its_follower(monkeypatch):
    manager, *_rest = _node_manager(monkeypatch)
    installed = {}
    manager._restart_coordinator = SimpleNamespace(
        start_follower=lambda **kwargs: installed.update(kwargs)
    )
    manager._start_restart_follower()
    assert installed["on_peer_lost"] == manager._handle_peer_lost
    assert installed["on_control_lost"] == manager._handle_control_lost
    assert installed["on_peer_down"] == manager._handle_peer_cohort_down
    manager._first_ready = False
    assert installed["ready"]() is False


def test_the_watch_starts_before_the_model_loads_not_at_readiness():
    """The follower used to start only once the pool was serving, so a peer
    failing during start-up was never seen there."""
    from tests.test_worker_pool_restart import _worker_manager_tree

    start_impl = next(
        node for node in ast.walk(_worker_manager_tree())
        if isinstance(node, ast.FunctionDef) and node.name == "_start_impl"
    )
    calls = [
        ast.unparse(node.func) for node in ast.walk(start_impl)
        if isinstance(node, ast.Call)
    ]
    lines = {
        ast.unparse(node.func): node.lineno for node in ast.walk(start_impl)
        if isinstance(node, ast.Call)
    }
    assert calls.count("self._start_restart_follower") == 1
    assert lines["self._start_restart_follower"] > lines["self._exchange_run_identity"]
    assert lines["self._start_restart_follower"] < lines["self._load_model_resources"]


def test_a_lost_peer_heartbeat_after_readiness_exits_with_the_peer_reason(
    monkeypatch,
):
    manager, store, clock, coordinator, events = _node_manager(
        monkeypatch, node_rank=1
    )
    hooks = _manager_hooks(manager)
    coordinator._watch_started_at = clock()
    store.add("alive/node0", 1)
    coordinator._follow_once(hooks)
    clock.advance(coordinator_module.PEER_DEAD_S)

    coordinator._follow_once(hooks)

    assert manager.state == "workers_down"
    assert events == [
        "teardown", "reconcile", "stop",
        ("exit", 1, "peer node 0 heartbeat lost (15s)"),
    ]
    # No echo: every survivor reads the same silent heartbeat itself.
    assert "down/3/node1" not in store.snapshot()


def test_a_peer_down_key_after_readiness_exits_naming_the_peer(monkeypatch):
    manager, store, _clock, coordinator, events = _node_manager(
        monkeypatch, node_rank=0
    )
    store.compare_set("down/3/node1", "", "rank 9 exited: CUDA error")

    coordinator._follow_once(_manager_hooks(manager))

    assert events[-1] == (
        "exit", 1, "peer node 1 fatal: rank 9 exited: CUDA error"
    )
    assert events[:-1] == ["teardown", "reconcile", "stop"]


def test_a_local_fatal_after_readiness_tells_the_peers_then_exits(monkeypatch):
    manager, store, _clock, _coordinator, events = _node_manager(
        monkeypatch, node_rank=1
    )

    with pytest.raises(_Exited):
        manager._handle_worker_failure("Detected worker process exit", None)

    assert store.snapshot()["down/3/node1"] == "Detected worker process exit"
    assert events == [
        "teardown", "reconcile", "stop",
        ("exit", 1, "worker fatal: Detected worker process exit"),
    ]


def test_losing_node_zeros_store_exits_this_node(monkeypatch):
    manager, store, clock, coordinator, events = _node_manager(
        monkeypatch, node_rank=1
    )
    hooks = _manager_hooks(manager)
    coordinator._watch_started_at = clock()
    coordinator._last_store_ok = clock()
    store.error = ConnectionError("connection reset by peer")

    _run_watchdog(coordinator, hooks, clock, coordinator_module.CONTROL_DEAD_S,
                  poll=True)

    assert events[-1] == (
        "exit", 1, "control store unreachable (node 0 down?)"
    )
    assert events[:-1] == ["teardown", "reconcile", "stop"]


@pytest.mark.parametrize("fail_fast", [True, False])
def test_a_down_key_during_startup_aborts_through_the_recorded_shutdown(
    monkeypatch, fail_fast
):
    """Before readiness the existing SIGINT path aborts start-up, flag or not,
    and it leaves its exit record before it signals."""
    manager, store, clock, coordinator, events = _node_manager(
        monkeypatch, node_rank=1, fail_fast=fail_fast, first_ready=False
    )
    manager._generation = 0
    store.compare_set("down/0/node0", "", "rank 2 failed to load weights")
    hooks = _manager_hooks(manager)
    coordinator._watch_started_at = clock()

    coordinator._follow_once(hooks)

    assert events == [
        ("record_exit", 1, "peer node 0 fatal: rank 2 failed to load weights"),
        "sigint",
    ]
    assert manager._worker_exit_state.reason == (
        "peer node 0 worker cohort failed: rank 2 failed to load weights"
    )
    # A second report during the same abort neither records nor signals again.
    manager._handle_peer_lost(0, 15.0)
    assert len(events) == 2


def test_a_peer_that_never_joins_aborts_startup_after_the_grace(monkeypatch):
    manager, _store, clock, coordinator, events = _node_manager(
        monkeypatch, node_rank=0, first_ready=False
    )
    manager._generation = 0
    hooks = _manager_hooks(manager)
    coordinator._watch_started_at = clock()

    clock.advance(coordinator_module.JOIN_GRACE_S - 1)
    coordinator._follow_once(hooks)
    assert events == []

    clock.advance(1)
    coordinator._follow_once(hooks)
    assert events == [
        ("record_exit", 1, "peer node 1 heartbeat lost (600s)"), "sigint",
    ]


def test_a_local_startup_failure_tells_the_peers_and_leaves_a_record(
    monkeypatch,
):
    manager, store, _clock, _coordinator, events = _node_manager(
        monkeypatch, node_rank=1, first_ready=False
    )
    manager._generation = 0

    manager._handle_worker_failure("Detected worker process exit", None)

    assert store.snapshot()["down/0/node1"] == "Detected worker process exit"
    assert events == [
        ("record_exit", 1, "worker fatal: Detected worker process exit"),
        "sigint",
    ]


def test_a_startup_exception_is_published_to_the_peers(monkeypatch):
    manager, store, _clock, _coordinator, _events = _node_manager(
        monkeypatch, node_rank=1, first_ready=False
    )
    manager._generation = 0
    manager.started = False

    def fail():
        raise RuntimeError("checkpoint shard 7 unreadable")

    manager._start_impl = fail
    with pytest.raises(RuntimeError):
        manager.start()

    assert store.snapshot()["down/0/node1"] == (
        "startup failed: checkpoint shard 7 unreadable"
    )


def test_fail_fast_off_keeps_the_survivor_up_in_workers_down(monkeypatch):
    manager, store, clock, coordinator, events = _node_manager(
        monkeypatch, node_rank=1, fail_fast=False
    )
    hooks = _manager_hooks(manager)
    store.compare_set("down/3/node0", "", "rank 0 exited")

    coordinator._follow_once(hooks)
    assert manager.state == "workers_down"
    assert events == ["teardown", "reconcile"]

    # Later losses change nothing more: the pool is already down.
    coordinator._watch_started_at = clock()
    coordinator._last_store_ok = clock()
    store.error = ConnectionError("connection reset")
    _run_watchdog(coordinator, hooks, clock, coordinator_module.CONTROL_DEAD_S,
                  poll=True)
    assert coordinator._store_dead
    assert events == ["teardown", "reconcile"]
    assert manager.state == "workers_down"


def test_fail_fast_exits_from_a_workers_down_left_by_a_failed_restart(
    monkeypatch,
):
    manager, _store, _clock, _coordinator, events = _node_manager(
        monkeypatch, node_rank=0
    )
    manager._state = "workers_down"

    with pytest.raises(_Exited):
        manager._handle_peer_lost(1, 15.2)

    assert events == ["stop", ("exit", 1, "peer node 1 heartbeat lost (15s)")]


def test_a_down_key_read_before_a_restart_spares_the_new_cohort(monkeypatch):
    """Node 0's follower waits out its own restart before it acts, and a down
    key it read for the old cohort must not take down the new one."""
    manager, _store, _clock, _coordinator, events = _node_manager(
        monkeypatch, node_rank=0
    )
    manager._state = "restarting"

    def finish_restart():
        time.sleep(0.05)
        with manager._lock:
            manager._generation = 4
            manager._state = "serving"

    thread = threading.Thread(target=finish_restart)
    thread.start()
    manager._handle_peer_cohort_down(1, "rank 9 exited", 3)
    thread.join()

    assert manager.state == "serving"
    assert events == []


def _failing_restart(monkeypatch, *, fail_fast):
    """Node 0 runs a coordinated restart whose respawn fails after P1."""
    from tests.test_restart_coordinator import _healthy_peer

    manager, store, _clock, _coordinator, events = _node_manager(
        monkeypatch, node_rank=0, fail_fast=fail_fast
    )

    def spawn():
        raise RuntimeError("spawn failed: rank 4 CUDA OOM")

    manager._spawn_workers = spawn
    stop_peer, _acked = _healthy_peer(store, epoch=1)
    return manager, store, events, stop_peer


def test_a_failed_restart_with_fail_fast_exits_after_telling_the_peers(
    monkeypatch,
):
    manager, store, events, stop_peer = _failing_restart(
        monkeypatch, fail_fast=True
    )
    try:
        with pytest.raises(_Exited):
            manager.restart_workers(timeout_s=5)
    finally:
        stop_peer.set()

    assert manager.state == "workers_down"
    assert store.snapshot()["restart/1/P1/node0"] == "ok"
    assert store.snapshot()["restart/1/P2/node0"] == (
        "failed:spawn failed: rank 4 CUDA OOM"
    )
    assert events[-2:] == [
        "stop", ("exit", 1, "restart failed: spawn failed: rank 4 CUDA OOM"),
    ]


def test_a_failed_restart_without_fail_fast_stays_in_workers_down(monkeypatch):
    manager, _store, events, stop_peer = _failing_restart(
        monkeypatch, fail_fast=False
    )
    try:
        with pytest.raises(RuntimeError, match="spawn failed"):
            manager.restart_workers(timeout_s=5)
    finally:
        stop_peer.set()

    assert manager.state == "workers_down"
    assert "stop" not in events
    assert not [event for event in events if isinstance(event, tuple)]


def test_a_restart_aborted_by_shutdown_never_exits_the_process(monkeypatch):
    """The shutdown that aborted the restart owns the process's end; a
    fail-fast exit there would cut its ordered teardown short."""
    manager, _store, _clock, _coordinator, events = _node_manager(
        monkeypatch, node_rank=0
    )
    manager.started = False

    with pytest.raises(RuntimeError, match="server is stopping"):
        manager.restart_workers(timeout_s=5)

    assert manager.state == "workers_down"
    assert not [event for event in events if isinstance(event, tuple)]


class HangingStore(FakeStore):
    """Answers until ``hang`` is set; from then on every call blocks, the way
    a client does on a node 0 that lost power: no error, no reply."""

    def __init__(self) -> None:
        super().__init__()
        self.hang = threading.Event()
        self.release = threading.Event()
        self.calls_while_hung: list = []

    def _checkpoint(self, name: str) -> None:
        if self.hang.is_set():
            self.calls_while_hung.append((name, threading.current_thread().name))
            self.release.wait()
            raise ConnectionError("released at test teardown")
        super()._checkpoint(name)

    def check(self, keys) -> bool:
        # The base class checks under its lock; a hung check must not hold it.
        self._checkpoint("check")
        with self._lock:
            return all(key in self._values for key in keys)

    def add(self, key, amount) -> int:
        self._checkpoint("add")
        with self._lock:
            value = int(self._values.get(key, 0)) + int(amount)
            self._values[key] = str(value)
            return value


def test_a_hung_node_zero_ends_this_node_without_touching_the_store_again(
    monkeypatch, tmp_path,
):
    """The real follower, heartbeat and watchdog threads and the real stop():
    with every store call blocked forever, the watchdog still declares node 0
    gone after CONTROL_DEAD_S and the exit completes without one more store
    call. Bounded in real time, so a regression fails instead of hanging."""
    store, clock = HangingStore(), FakeClock()
    manager, _store, _clock, coordinator, events = _node_manager(
        monkeypatch, node_rank=1, store=store, clock=clock,
        heartbeat_s=0.005, watchdog_tick_s=0.005,
    )
    # The real stop(), on a manager whose cohort the failure path reaps.
    del manager.stop
    real_stop = manager.stop
    manager.stop = lambda: (real_stop(), events.append("stop returned"))
    manager.args.runtime_identity.runtime_dir = tmp_path / "unused"
    manager._runtime_dir_created = False
    manager._runtime_namespace_owned = False
    manager._runtime_locks = None
    manager._lane_lease = None
    manager._run_lock = None
    manager._staged_weight_store = None
    manager.parameter_server_instance = None
    manager.model_info = {}
    manager.skeleton_state_dict_file = None
    manager.prefix_cache_runtime_config = None
    manager._close_skeleton_memfd = lambda: None
    # Inside a restart phase, so stop() would tell the peers -- if it could.
    manager._coordinated_phase = (1, "P1")

    manager._start_restart_follower()
    try:
        assert _wait_until(lambda: int(store.snapshot().get("alive/node1", 0)) >= 2)
        store.hang.set()
        assert _wait_until(
            lambda: {name for _op, name in store.calls_while_hung} >= {
                "restart-coordinator", "restart-coordinator-heartbeat",
            }
        )

        clock.advance(coordinator_module.CONTROL_DEAD_S - 0.5)
        time.sleep(0.05)
        assert events == []

        clock.advance(0.5)
        started = time.monotonic()
        assert _wait_until(lambda: any(isinstance(e, tuple) for e in events), 10.0)
        elapsed = time.monotonic() - started

        assert events == [
            "teardown", "reconcile", "stop returned",
            ("exit", 1, "control store unreachable (node 0 down?)"),
        ]
        # Neither the threads parked in the store nor the exit were waited on.
        assert elapsed < 2.0
        # The only calls since the hang are the two that never returned.
        assert sorted(name for _op, name in store.calls_while_hung) == [
            "restart-coordinator", "restart-coordinator-heartbeat",
        ]
        assert coordinator._store is None
        assert coordinator._abandoned_store is store
    finally:
        store.release.set()


# ----------------------------------------------------------------- /health


def _mount_health_route(worker, worker_exit=None):
    """The real /health body on a bare app -- no server stack, no lifespan."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    route = copy.deepcopy(next(
        node for node in ast.walk(ast.parse(HTTP_SERVER_PATH.read_text()))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "health_check"
    ))
    route.decorator_list = []
    namespace = {"JSONResponse": JSONResponse, "Request": Request}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[route], type_ignores=[])),
            str(HTTP_SERVER_PATH),
            "exec",
        ),
        namespace,
    )
    app = FastAPI()
    app.state.health = SimpleNamespace(is_healthy=lambda: (True, None))
    app.state.worker_exit_state = worker_exit or _ExitState()
    app.state.worker = worker
    app.get("/health")(namespace["health_check"])
    return TestClient(app)


def test_health_is_200_only_for_a_serving_node_with_its_control_store():
    pytest.importorskip("fastapi")
    healthy = _mount_health_route(
        SimpleNamespace(state="serving", coordinator_error=None)
    )
    assert healthy.get("/health").status_code == 200

    down = _mount_health_route(
        SimpleNamespace(state="workers_down", coordinator_error=None)
    )
    assert down.get("/health").status_code == 503

    cut_off = _mount_health_route(
        SimpleNamespace(
            state="serving",
            coordinator_error="coordinator unreachable: connection reset",
        )
    )
    response = cut_off.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "reason": "coordinator unreachable: connection reset",
    }


# ------------------------------------------------------------- the flag


@pytest.fixture
def _skip_port_probes(monkeypatch):
    """Validation binds real sockets; the flag rules under test do not."""
    monkeypatch.setattr(
        server_args_module, "_ensure_local_port_free", lambda port, label: None
    )


def _args(tmp_path, **overrides):
    values = {
        "model": "openai/gpt-oss-120b",
        "host_kv_cache_size": 1,
        "storage_path": tmp_path / "storage",
        "nnodes": 2,
        "node_rank": 1,
        "world_size": 16,
        "dist_init_addr": "10.0.0.1:12355",
    }
    values.update(overrides)
    return server_args_module.ServerArgs(**values)


@pytest.mark.parametrize(
    "nnodes, world_size, given, resolved",
    [
        (1, 8, None, False),
        (2, 16, None, True),
        (4, 32, None, True),
        (2, 16, False, False),
        (2, 16, True, True),
        (1, 8, False, False),
    ],
)
def test_the_flag_defaults_on_for_a_cluster_and_off_for_one_node(
    tmp_path, _skip_port_probes, nnodes, world_size, given, resolved
):
    args = _args(
        tmp_path, nnodes=nnodes, world_size=world_size, node_rank=0,
        cluster_fail_fast=given,
    )
    assert server_args_module.cluster_fail_fast_enabled(args) is resolved
    server_args_module.validate_server_args(args)
    assert args.cluster_fail_fast is resolved


def test_the_flag_on_a_single_node_is_refused(tmp_path, _skip_port_probes):
    with pytest.raises(ValueError, match="--cluster-fail-fast needs --nnodes > 1"):
        server_args_module.validate_server_args(
            _args(tmp_path, nnodes=1, world_size=8, node_rank=0,
                  cluster_fail_fast=True)
        )


@pytest.mark.parametrize(
    "argv, parsed",
    [
        ([], None),
        (["--cluster-fail-fast"], True),
        (["--no-cluster-fail-fast"], False),
    ],
)
def test_the_cli_spells_the_flag_both_ways_and_leaves_the_default_open(
    argv, parsed
):
    namespace = server_args_module._build_parser().parse_args(
        ["--model", "openai/gpt-oss-120b"] + argv
    )
    assert namespace.cluster_fail_fast is parsed


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False
