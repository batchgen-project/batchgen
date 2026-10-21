"""The node-0 coordinated worker-pool restart protocol.

Everything here runs on the dev machine: an in-memory control store stands in
for the real one, one case uses a real ``TCPStore`` on the loopback interface,
and nothing needs a GPU or a worker process. Timings are injected through the
coordinator's constructor, never through an environment variable, so the whole
file finishes in well under a second apart from the loopback case.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
COORDINATOR_PATH = ROOT / "batchgen" / "server" / "restart_coordinator.py"
HTTP_SERVER_PATH = ROOT / "batchgen" / "server" / "http_server.py"


def _load_coordinator():
    """Import restart_coordinator alone (batchgen/__init__ pulls kernels)."""
    spec = importlib.util.spec_from_file_location(
        "_batchgen_restart_coordinator_test", COORDINATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_server_args_module():
    """Import the leaf module without running the HTTP server package init."""
    package_name = "batchgen.server"
    previous = sys.modules.get(package_name)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return importlib.import_module("batchgen.server.server_args")
    finally:
        if previous is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous


coordinator_module = _load_coordinator()
RestartCoordinator = coordinator_module.RestartCoordinator
PeerRestartError = coordinator_module.PeerRestartError
PHASE_ARMED = coordinator_module.PHASE_ARMED
PHASE_TORN_DOWN = coordinator_module.PHASE_TORN_DOWN
PHASE_READY = coordinator_module.PHASE_READY
RESTART_PHASES = coordinator_module.RESTART_PHASES

server_args_module = _load_server_args_module()
ServerArgs = server_args_module.ServerArgs


# --------------------------------------------------------------------------
# An in-memory control store with the same blocking rules as the real one:
# get() on a missing key would block, so the coordinator must reach for
# check() first, and every operation can be made to fail on demand.
# --------------------------------------------------------------------------


class FakeStore:
    def __init__(self) -> None:
        self._values: dict = {}
        self._lock = threading.Lock()
        self._event = threading.Condition(self._lock)
        self.error: BaseException | None = None
        self.calls: list = []

    def _checkpoint(self, name: str) -> None:
        self.calls.append(name)
        if self.error is not None:
            raise self.error

    def set(self, key, value) -> None:
        with self._event:
            self._checkpoint("set")
            self._values[key] = value if isinstance(value, bytes) else str(value)
            self._event.notify_all()

    def get(self, key):
        with self._event:
            self._checkpoint("get")
            if key not in self._values:
                raise AssertionError(
                    f"get({key!r}) on a missing key; the real store blocks "
                    "there until its own timeout, so check() must come first"
                )
            value = self._values[key]
        return value if isinstance(value, bytes) else value.encode()

    def check(self, keys) -> bool:
        with self._event:
            self._checkpoint("check")
            return all(key in self._values for key in keys)

    def add(self, key, amount) -> int:
        with self._event:
            self._checkpoint("add")
            value = int(self._values.get(key, 0)) + int(amount)
            self._values[key] = str(value)
            self._event.notify_all()
            return value

    def compare_set(self, key, expected, desired):
        with self._event:
            self._checkpoint("compare_set")
            current = self._values.get(key)
            if current is None and expected == "":
                self._values[key] = desired
            elif current == expected:
                self._values[key] = desired
            self._event.notify_all()
            return self._values[key].encode()

    def wait(self, keys, timeout) -> None:
        seconds = getattr(timeout, "total_seconds", lambda: float(timeout))()
        deadline = time.monotonic() + seconds
        with self._event:
            self._checkpoint("wait")
            while not all(key in self._values for key in keys):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"wait timed out for {keys}")
                self._event.wait(remaining)

    # Test-side helpers.
    def snapshot(self) -> dict:
        with self._lock:
            return {
                key: value.decode() if isinstance(value, bytes) else value
                for key, value in self._values.items()
            }


def _cluster(nnodes=2, store=None, **kwargs):
    """One shared store, one coordinator per node, all already started."""
    store = store or FakeStore()
    options = {
        "poll_interval_s": 0.01,
        "request_wait_s": 0.5,
        "error_backoff_s": 0.01,
    }
    options.update(kwargs)
    nodes = []
    for node_rank in range(nnodes):
        node = RestartCoordinator(
            nnodes,
            node_rank,
            "127.0.0.1",
            0,
            store_factory=lambda *_args, **_kw: store,
            **options,
        )
        node.start()
        nodes.append(node)
    return store, nodes


def _in_thread(fn, *args):
    """Run fn in a thread; returns (thread, outcome list of (ok, value))."""
    outcome = []

    def run():
        try:
            outcome.append((True, fn(*args)))
        except BaseException as exc:
            outcome.append((False, exc))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ------------------------------------------------- the single-node no-op path


def test_a_single_node_coordinator_opens_no_store_and_every_call_is_a_no_op():
    """The whole point of the single-node case: no store, no port, no thread,
    and no phase that can fail -- so a one-node restart keeps exactly the
    phases and timings it had before any of this existed."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("a single-node server must not open a control store")

    node = RestartCoordinator(1, 0, "127.0.0.1", 12356, store_factory=refuse)
    assert node.enabled is False

    node.start()
    request = node.begin_restart(7.0)
    assert request.epoch == 0
    assert request.timeout_s == 7.0
    # Every phase call returns immediately and raises nothing.
    node.barrier(request.epoch, PHASE_ARMED, deadline=0.0)
    node.fail_phase(request.epoch, PHASE_ARMED, "ignored")
    node.raise_if_peer_failed(request.epoch, PHASE_READY)
    assert node.peer_failed(request.epoch, PHASE_READY) is None
    node.publish_down(3, "ignored")
    node.publish_result(request.epoch, {"state": "serving"})
    assert node.peer_acks() == {}
    assert node.control_error is None

    node.start_follower(
        state=lambda: "serving",
        generation=lambda: 1,
        on_peer_restart=lambda request: pytest.fail("no follower may run"),
        on_peer_down=lambda node, reason: pytest.fail("no follower may run"),
    )
    assert not [
        thread for thread in threading.enumerate()
        if thread.name == "restart-coordinator"
    ]
    node.close()


# ------------------------------------------------------------ epoch + request


def test_node_zero_claims_sequential_epochs_and_publishes_the_request():
    store, (node0, _node1) = _cluster()

    first = node0.begin_restart(30.0)
    second = node0.begin_restart(30.0)

    assert (first.epoch, second.epoch) == (1, 2)
    assert node0.epoch == 2
    published = json.loads(store.snapshot()[f"restart/{first.epoch}/request"])
    assert published["timeout_s"] == 30.0
    # The deadline is wall-clock and derived from the requester's clock, so
    # every node gives up at the same instant rather than each waiting out its
    # own full timeout.
    assert first.deadline == pytest.approx(published["requested_at"] + 30.0)


def test_the_result_is_published_under_the_epoch_that_produced_it():
    store, (node0, _node1) = _cluster()
    request = node0.begin_restart(5.0)

    node0.publish_result(request.epoch, {"state": "serving", "generation": 4})

    result = json.loads(store.snapshot()[f"restart/{request.epoch}/result"])
    assert result == {"generation": 4, "state": "serving"}


# ------------------------------------------------------------------- barriers


def test_barrier_releases_only_once_every_node_acked():
    store, (node0, node1) = _cluster()
    request = node0.begin_restart(5.0)

    thread, outcome = _in_thread(
        node0.barrier, request.epoch, PHASE_ARMED, request.deadline
    )
    # Node 0 acked itself and is waiting: one ack, no release.
    assert _wait_until(
        lambda: f"restart/{request.epoch}/{PHASE_ARMED}/node0" in store.snapshot()
    )
    time.sleep(0.05)
    assert outcome == []

    node1.barrier(request.epoch, PHASE_ARMED, request.deadline)
    thread.join(5)
    assert outcome == [(True, None)]
    assert store.snapshot()[f"restart/{request.epoch}/{PHASE_ARMED}/node1"] == "ok"


def test_barrier_raises_peer_restart_error_naming_the_failed_node_and_phase():
    store, (node0, node1) = _cluster()
    request = node0.begin_restart(5.0)

    node1.fail_phase(request.epoch, PHASE_TORN_DOWN, "worker teardown left live PIDs")
    with pytest.raises(PeerRestartError) as caught:
        node0.barrier(request.epoch, PHASE_TORN_DOWN, request.deadline)

    assert caught.value.node == 1
    assert caught.value.phase == PHASE_TORN_DOWN
    assert "worker teardown left live PIDs" in caught.value.reason
    assert store.snapshot()[
        f"restart/{request.epoch}/{PHASE_TORN_DOWN}/node1"
    ].startswith("failed:")


def test_barrier_gives_up_at_the_shared_deadline_not_its_own_timeout():
    """A node that enters a barrier late must still stop when the epoch's
    deadline passes, so every node abandons the restart together."""
    _store, (node0, _node1) = _cluster()
    request = coordinator_module.RestartRequest(
        epoch=1, timeout_s=0.2, requested_at=time.time()
    )

    time.sleep(0.15)
    started = time.monotonic()
    with pytest.raises(PeerRestartError, match="no phase ack"):
        node0.barrier(request.epoch, PHASE_ARMED, request.deadline)

    # Bounded by what was left of the shared window, not by 0.2s from here.
    assert time.monotonic() - started < 0.15


def test_an_ok_ack_is_a_promise_and_is_never_overwritten_by_a_failure():
    """Peers act on an ``ok``: a node that already published one and then
    failed must report that failure against the phase it has NOT acked, or the
    peers would be told to roll back a phase they already left."""
    store, (node0, node1) = _cluster()
    request = node0.begin_restart(5.0)

    thread, _outcome = _in_thread(
        node1.barrier, request.epoch, PHASE_ARMED, request.deadline
    )
    node0.barrier(request.epoch, PHASE_ARMED, request.deadline)
    thread.join(5)

    node0.fail_phase(request.epoch, PHASE_ARMED, "too late to take it back")
    assert store.snapshot()[f"restart/{request.epoch}/{PHASE_ARMED}/node0"] == "ok"

    # The next phase is where the failure belongs, and it lands there.
    node0.fail_phase(request.epoch, PHASE_TORN_DOWN, "reconcile failed")
    assert store.snapshot()[
        f"restart/{request.epoch}/{PHASE_TORN_DOWN}/node0"
    ] == "failed:reconcile failed"


def test_a_published_failure_is_bounded_to_one_line():
    store, (node0, _node1) = _cluster()
    reason = "traceback\n" + "x" * 1000

    node0.fail_phase(1, PHASE_READY, reason)

    value = store.snapshot()[f"restart/1/{PHASE_READY}/node0"]
    assert "\n" not in value
    assert len(value) < 500


def test_peer_failed_polls_without_waiting_for_the_missing_acks():
    """The readiness wait calls this every interval: it must answer from what
    the store already has, never block on a key that may never appear."""
    _store, (node0, node1) = _cluster()

    assert node0.peer_failed(1, PHASE_READY) is None
    node0.raise_if_peer_failed(1, PHASE_READY)

    node1.fail_phase(1, PHASE_READY, "cohort never became ready")
    failure = node0.peer_failed(1, PHASE_READY)
    assert isinstance(failure, PeerRestartError)
    assert failure.node == 1
    with pytest.raises(PeerRestartError):
        node0.raise_if_peer_failed(1, PHASE_READY)


def test_a_store_error_fails_the_barrier_at_once_and_is_reported():
    """A TCPStore client never reconnects, so a failed call means node 0's
    store is gone; retrying until the deadline would only hold every peer in
    ``restarting`` for the whole restart timeout."""
    store, (node0, node1) = _cluster()
    deadline = time.time() + 30
    store.error = RuntimeError("Broken pipe")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="could not be coordinated"):
        node0.barrier(1, PHASE_ARMED, deadline)
    with pytest.raises(PeerRestartError) as caught:
        node1.barrier(1, PHASE_ARMED, deadline)
    assert time.monotonic() - started < 1.0
    assert caught.value.node == 0
    assert "restart coordinator unreachable" in caught.value.reason
    assert "coordinator unreachable" in node1.control_error


def test_a_lost_coordinator_fails_the_readiness_poll_naming_node_zero():
    store, (_node0, node1) = _cluster()
    store.error = RuntimeError("Broken pipe")

    failure = node1.peer_failed(1, PHASE_READY)

    assert isinstance(failure, PeerRestartError)
    assert failure.node == 0


# ------------------------------------------------------------- down-key fanout


def test_publish_down_is_keyed_by_generation_and_the_first_writer_wins():
    store, (node0, _node1) = _cluster()

    node0.publish_down(4, "Detected worker process exit: idx=0")
    node0.publish_down(4, "a later, less useful reason")

    assert store.snapshot()["down/4/node0"] == (
        "Detected worker process exit: idx=0"
    )
    # A different cohort is a different key, never a retry of the same one.
    node0.publish_down(5, "the next cohort died too")
    assert store.snapshot()["down/5/node0"] == "the next cohort died too"


def test_peer_acks_report_the_newest_phase_each_node_reached():
    _store, (node0, node1) = _cluster()
    request = node0.begin_restart(5.0)

    thread, _outcome = _in_thread(
        node1.barrier, request.epoch, PHASE_ARMED, request.deadline
    )
    node0.barrier(request.epoch, PHASE_ARMED, request.deadline)
    thread.join(5)
    node0.fail_phase(request.epoch, PHASE_TORN_DOWN, "reconcile failed")

    acks = node0.peer_acks()
    assert acks["node0"] == {
        "phase": PHASE_TORN_DOWN, "ack": "failed:reconcile failed",
    }
    assert acks["node1"] == {"phase": PHASE_ARMED, "ack": "ok"}


# ------------------------------------------------------------------- follower


def _follower(node, *, state="serving", generation=1):
    seen_requests = []
    seen_down = []
    node.start_follower(
        state=lambda: state() if callable(state) else state,
        generation=lambda: generation() if callable(generation) else generation,
        on_peer_restart=seen_requests.append,
        on_peer_down=lambda failed, reason: seen_down.append((failed, reason)),
    )
    return seen_requests, seen_down


def test_the_follower_runs_a_new_epoch_on_a_peer_and_never_on_node_zero():
    _store, (node0, node1) = _cluster()
    node0_requests, _ = _follower(node0)
    node1_requests, _ = _follower(node1)

    request = node0.begin_restart(5.0)

    assert _wait_until(lambda: len(node1_requests) == 1)
    assert node1_requests[0].epoch == request.epoch
    assert node1_requests[0].timeout_s == 5.0
    assert node1_requests[0].deadline == pytest.approx(request.deadline)
    # Node 0 entered the restart through its own HTTP route; a second entry
    # from its follower would tear the same cohort down twice.
    time.sleep(0.1)
    assert node0_requests == []
    node0.close()
    node1.close()


def test_the_follower_ignores_an_epoch_it_has_already_run():
    _store, (node0, node1) = _cluster()
    requests, _ = _follower(node1)

    node0.begin_restart(5.0)
    assert _wait_until(lambda: len(requests) == 1)
    time.sleep(0.1)
    assert len(requests) == 1

    node0.begin_restart(5.0)
    assert _wait_until(lambda: len(requests) == 2)
    assert [request.epoch for request in requests] == [1, 2]
    node1.close()


def test_a_follower_started_after_a_restart_does_not_replay_it():
    """A baseline is taken when the follower starts, so an epoch that is
    already history is not mistaken for a fresh request."""
    _store, (node0, node1) = _cluster()
    node0.begin_restart(5.0)

    requests, _ = _follower(node1)
    time.sleep(0.1)

    assert requests == []
    node1.close()


def test_the_follower_reports_a_peer_down_key_once_and_only_while_serving():
    _store, (node0, node1) = _cluster()
    state = {"value": "restarting"}
    _requests, down = _follower(node1, state=lambda: state["value"], generation=7)

    node0.publish_down(7, "Detected worker process exit")
    time.sleep(0.1)
    # A node that is already restarting or down owns its own recovery; the
    # peer's failure must not be handed to it again.
    assert down == []

    state["value"] = "serving"
    assert _wait_until(lambda: len(down) == 1)
    assert down[0] == (0, "Detected worker process exit")
    time.sleep(0.1)
    assert len(down) == 1
    node1.close()


def test_the_follower_only_reports_the_down_key_of_the_live_generation():
    _store, (node0, node1) = _cluster()
    _requests, down = _follower(node1, generation=9)

    node0.publish_down(8, "a previous cohort's failure")
    time.sleep(0.1)
    assert down == []

    node0.publish_down(9, "this cohort's failure")
    assert _wait_until(lambda: len(down) == 1)
    assert down[0] == (0, "this cohort's failure")
    node1.close()


def test_the_follower_survives_store_errors_backs_off_and_records_them():
    store, (_node0, node1) = _cluster()
    requests, _down = _follower(node1)
    store.error = RuntimeError("control store is gone")

    assert _wait_until(lambda: node1.control_error is not None)
    assert "coordinator unreachable: control store is gone" == node1.control_error
    # Backs off and keeps running: the store coming back is the only way this
    # node ever hears about a restart again.
    time.sleep(0.1)
    assert node1._follower.is_alive()

    # It recovers on its own once the store answers again.
    store.error = None
    store.add("epoch", 1)
    store.set("restart/1/request", json.dumps(
        {"timeout_s": 5.0, "requested_at": time.time()}
    ))
    assert _wait_until(lambda: len(requests) == 1)
    node1.close()


def test_the_follower_outlives_a_failed_restart_so_the_next_epoch_is_seen():
    """A coordinated restart that fails leaves the pool in the retryable
    workers_down state; the follower is how the retry reaches this node."""
    _store, (node0, node1) = _cluster()
    attempts = []

    def failing_restart(request):
        attempts.append(request.epoch)
        raise RuntimeError("worker teardown left live owned PIDs")

    node1.start_follower(
        state=lambda: "serving",
        generation=lambda: 1,
        on_peer_restart=failing_restart,
        on_peer_down=lambda failed, reason: None,
    )
    node0.begin_restart(5.0)
    assert _wait_until(lambda: attempts == [1])
    node0.begin_restart(5.0)
    assert _wait_until(lambda: attempts == [1, 2])
    node1.close()


def test_the_follower_is_a_daemon_thread_and_never_the_asyncio_loop():
    """A coordinated restart runs to completion inline on this thread and takes
    minutes, so it may never be the event loop; and a stuck poll must not keep
    the process alive after shutdown."""
    _store, (_node0, node1) = _cluster()
    _follower(node1)

    follower = next(
        thread for thread in threading.enumerate()
        if thread.name == "restart-coordinator"
    )
    assert follower.daemon
    assert follower is not threading.main_thread()

    node1.close()
    assert _wait_until(lambda: not follower.is_alive())


# -------------------------------------------------- the real loopback store


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_protocol_runs_over_a_real_control_store():
    """The fake store above encodes this store's contract; this case is what
    proves the contract, over a real TCPStore on the loopback interface."""
    pytest.importorskip("torch")
    port = _free_port()
    node0 = RestartCoordinator(2, 0, "127.0.0.1", port, poll_interval_s=0.02)
    node1 = RestartCoordinator(2, 1, "127.0.0.1", port, poll_interval_s=0.02)
    node0.start()
    node1.start()
    try:
        requests, _down = _follower(node1)
        request = node0.begin_restart(20.0)
        assert _wait_until(lambda: len(requests) == 1, timeout=20)
        assert requests[0].epoch == request.epoch
        assert requests[0].deadline == pytest.approx(request.deadline)

        thread, outcome = _in_thread(
            node1.barrier, request.epoch, PHASE_ARMED, request.deadline
        )
        node0.barrier(request.epoch, PHASE_ARMED, request.deadline)
        thread.join(20)
        assert outcome == [(True, None)]

        node1.fail_phase(request.epoch, PHASE_TORN_DOWN, "spawn failed")
        with pytest.raises(PeerRestartError) as caught:
            node0.barrier(request.epoch, PHASE_TORN_DOWN, request.deadline)
        assert (caught.value.node, caught.value.phase) == (1, PHASE_TORN_DOWN)

        node1.publish_down(3, "Detected worker process exit")
        assert node0.peer_acks()["node1"]["ack"].startswith("failed:")
    finally:
        node1.close()
        node0.close()


# ------------------------------------------- the manager running the phases


def _multi_node_manager(monkeypatch, node_rank=0, nnodes=2, store=None):
    """A WorkerManager wired to a real coordinator over the fake store.

    The manager itself comes from the lifted class the single-node restart
    tests use, so the local phases under test here are the same ones.
    """
    from tests.test_worker_pool_restart import _manager

    manager = _manager(monkeypatch)
    store = store or FakeStore()
    manager.args.nnodes = nnodes
    manager.args.node_rank = node_rank
    coordinator = RestartCoordinator(
        nnodes,
        node_rank,
        "127.0.0.1",
        0,
        store_factory=lambda *_args, **_kw: store,
        poll_interval_s=0.01,
        request_wait_s=0.5,
        error_backoff_s=0.01,
    )
    coordinator.start()
    manager._restart_coordinator = coordinator
    return manager, store, coordinator


def _healthy_peer(store, epoch, *, watch=0, write=1, phases=RESTART_PHASES):
    """A peer that acks each phase as soon as the node under test does.

    It writes straight to the store, so what it proves is the other side: the
    node under test cannot leave a phase until this ack lands.
    """
    stop = threading.Event()
    acked = []

    def run():
        for phase in phases:
            while not stop.is_set():
                if store.check([f"restart/{epoch}/{phase}/node{watch}"]):
                    store.set(f"restart/{epoch}/{phase}/node{write}", "ok")
                    acked.append(phase)
                    break
                time.sleep(0.005)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return stop, acked


class _ExitState:
    reason = None

    def set_failure(self, reason, exc):
        self.reason = reason
        return True

    def clear(self):
        self.reason = None

    def is_failed(self):
        return self.reason is not None


def test_node_zero_runs_every_phase_and_reports_the_epoch_and_node_count(
    monkeypatch,
):
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    stop, acked = _healthy_peer(store, epoch=1)
    order = []
    manager._stop_workers = lambda: order.append("teardown")
    manager._spawn_workers = lambda: order.append("spawn")
    try:
        result = manager.restart_workers(timeout_s=5)
    finally:
        stop.set()

    assert manager.state == "serving"
    assert result["epoch"] == 1
    assert result["nodes"] == 2
    assert order == ["teardown", "spawn"]
    # Every phase was acked by both nodes, in order.
    assert acked == list(RESTART_PHASES)
    snapshot = store.snapshot()
    for phase in RESTART_PHASES:
        assert snapshot[f"restart/1/{phase}/node0"] == "ok"
    assert json.loads(snapshot["restart/1/result"])["state"] == "serving"


def test_no_node_spawns_before_every_peer_reaped_its_old_cohort(monkeypatch):
    """A node that spawned early would rendezvous into a world that still
    holds a peer's dying ranks, which is the hang this phase exists to stop."""
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    manager.set_main_thread_dispatcher(lambda fn, timeout: fn())
    spawned = threading.Event()
    manager._spawn_workers = spawned.set
    stop, _acked = _healthy_peer(store, epoch=1, phases=(PHASE_ARMED,))

    thread, outcome = _in_thread(lambda: manager.restart_workers(timeout_s=5))
    try:
        # Node 0 finished its own teardown and is waiting for the peer's.
        assert _wait_until(
            lambda: f"restart/1/{PHASE_TORN_DOWN}/node0" in store.snapshot()
        )
        time.sleep(0.05)
        assert not spawned.is_set()

        store.set(f"restart/1/{PHASE_TORN_DOWN}/node1", "ok")
        store.set(f"restart/1/{PHASE_READY}/node1", "ok")
        thread.join(5)
    finally:
        stop.set()

    assert spawned.is_set()
    assert outcome and outcome[0][0] is True
    assert manager.state == "serving"


def test_a_peer_that_never_arms_still_tears_this_node_down_then_fails(
    monkeypatch,
):
    """Ranks cannot serve without their peers, so a cohort whose peers are not
    arming is dead either way: it is reaped rather than left holding GPUs, and
    the restart still fails into the retryable state."""
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    order = []
    manager._quiesce_hook = lambda: order.append("quiesce")
    manager._stop_workers = lambda: order.append("teardown")
    manager._reconcile = lambda reason, fail_in_flight: order.append(
        "reconcile"
    ) or {"failed_batches": 0}
    manager._spawn_workers = lambda: order.append("spawn")
    store.set(f"restart/1/{PHASE_ARMED}/node1", "failed:quiesce hook failed")

    with pytest.raises(PeerRestartError) as caught:
        manager.restart_workers(timeout_s=5)

    assert caught.value.node == 1
    assert caught.value.phase == PHASE_ARMED
    assert order == ["quiesce", "teardown", "reconcile"]
    assert manager.state == "workers_down"
    # The local channels were rotated, so the retry starts from a clean cohort.
    assert manager.request_queue is not None
    # This node's own P0 ack stands; its failure is reported against the phase
    # it never acked, which is where the peers are waiting.
    snapshot = store.snapshot()
    assert snapshot[f"restart/1/{PHASE_ARMED}/node0"] == "ok"
    assert snapshot[f"restart/1/{PHASE_TORN_DOWN}/node0"].startswith("failed:")


def test_a_local_failure_is_published_for_the_phase_it_happened_in(monkeypatch):
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    stop, _acked = _healthy_peer(store, epoch=1, phases=(PHASE_ARMED,))

    def failing_teardown():
        raise RuntimeError("worker teardown left live owned PIDs")

    manager._stop_workers = failing_teardown
    try:
        with pytest.raises(RuntimeError, match="live owned PIDs"):
            manager.restart_workers(timeout_s=5)
    finally:
        stop.set()

    assert manager.state == "workers_down"
    assert store.snapshot()[f"restart/1/{PHASE_TORN_DOWN}/node0"] == (
        "failed:worker teardown left live owned PIDs"
    )
    assert manager._last_restart_error == "worker teardown left live owned PIDs"
    assert json.loads(store.snapshot()["restart/1/result"])["state"] == (
        "workers_down"
    )


def test_a_shutdown_abort_is_published_as_stopping(monkeypatch):
    """The peers cannot retry this epoch against a node that is going away, so
    they are told that in the one word they can act on."""
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    manager._stopping = True

    with pytest.raises(RuntimeError, match="server is stopping"):
        manager.restart_workers(timeout_s=5)

    assert manager.state == "workers_down"
    assert store.snapshot()[f"restart/1/{PHASE_ARMED}/node0"] == "failed:stopping"


def test_a_server_stop_mid_restart_fails_every_phase_this_node_has_left(
    monkeypatch,
):
    """stop() drops the control store; before it does, the peers must learn
    that this node is gone, or they wait at their next barrier until the
    shared deadline. The current phase may already carry this node's ``ok``,
    so the later phases are failed too."""
    manager, store, coordinator = _multi_node_manager(monkeypatch)
    coordinator._write_ack(1, PHASE_TORN_DOWN, "ok")
    manager._coordinated_phase = (1, PHASE_TORN_DOWN)

    manager._publish_restart_stopping()

    snapshot = store.snapshot()
    assert f"restart/1/{PHASE_ARMED}/node0" not in snapshot
    assert snapshot[f"restart/1/{PHASE_TORN_DOWN}/node0"] == "ok"
    assert snapshot[f"restart/1/{PHASE_READY}/node0"] == "failed:stopping"


def test_a_server_stop_outside_a_restart_publishes_nothing(monkeypatch):
    manager, store, _coordinator = _multi_node_manager(monkeypatch)

    manager._publish_restart_stopping()

    assert not any(key.startswith("restart/") for key in store.snapshot())


def test_stop_tells_the_peers_before_it_drops_the_control_store():
    path = ROOT / "batchgen" / "server" / "worker_manager.py"
    text = path.read_text()
    manager_class = next(
        node for node in ast.parse(text).body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerManager"
    )
    stop = next(
        node for node in manager_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "stop"
    )
    source = ast.get_source_segment(text, stop)
    assert source.index("self._publish_restart_stopping()") < source.index(
        "self._restart_coordinator.close()"
    )


def test_a_restart_ends_in_workers_down_even_when_the_peers_cannot_be_told(
    monkeypatch,
):
    """The control store may be the very thing that failed; the local
    recovery (workers_down, teardown lock released) must not depend on it."""
    manager, store, coordinator = _multi_node_manager(monkeypatch)
    stop, _acked = _healthy_peer(store, epoch=1, phases=(PHASE_ARMED,))

    def failing_teardown():
        raise RuntimeError("teardown failed")

    def unreachable(*_args, **_kwargs):
        raise ConnectionError("control store unreachable")

    manager._stop_workers = failing_teardown
    monkeypatch.setattr(coordinator, "publish_result", unreachable)
    try:
        with pytest.raises(RuntimeError, match="teardown failed"):
            manager.restart_workers(timeout_s=5)
    finally:
        stop.set()

    assert manager.state == "workers_down"
    assert manager._coordinated_phase is None
    assert manager._teardown_lock.acquire(blocking=False)
    manager._teardown_lock.release()


def test_a_followed_restart_runs_under_node_zeros_epoch_and_deadline(monkeypatch):
    manager, store, _coordinator = _multi_node_manager(monkeypatch, node_rank=1)
    # The stand-in is node 0 here: it acks each phase once this node has, so
    # the roles are the only difference from the node-0 case.
    stop, _acked = _healthy_peer(store, epoch=3, watch=1, write=0)
    try:
        result = manager.restart_workers(
            timeout_s=5, epoch=3, deadline=time.time() + 5
        )
    finally:
        stop.set()

    assert manager.state == "serving"
    assert result["epoch"] == 3
    # Only node 0 answers a caller, so only node 0 publishes the result.
    assert "nodes" not in result
    assert "restart/3/result" not in store.snapshot()


def test_a_followed_restart_respawns_through_the_persistent_dispatcher(
    monkeypatch,
):
    """Workers arm a parent-death signal tied to the thread that spawned them,
    so every respawn has to reach the main thread. A peer's restart arrives
    from the coordinator's thread with no request behind it, so a per-request
    bridge is not enough: without the persistent one the respawn refuses."""
    manager, store, _coordinator = _multi_node_manager(monkeypatch, node_rank=1)
    request = coordinator_module.RestartRequest(
        epoch=2, timeout_s=5, requested_at=time.time()
    )
    stop, _acked = _healthy_peer(store, epoch=2, watch=1, write=0)
    try:
        thread, outcome = _in_thread(manager._run_followed_restart, request)
        thread.join(5)
        assert outcome and outcome[0][0] is False
        assert "requires a main-thread dispatcher" in str(outcome[0][1])
        assert manager.state == "workers_down"

        # With the bridge the lifespan installs, the same restart completes.
        bridged = []
        manager.set_main_thread_dispatcher(
            lambda fn, timeout: bridged.append(timeout) or fn()
        )
        spawn_threads = []
        manager._spawn_workers = lambda: spawn_threads.append(
            threading.current_thread().name
        )
        store.set(f"restart/2/{PHASE_TORN_DOWN}/node0", "ok")
        store.set(f"restart/2/{PHASE_READY}/node0", "ok")
        thread, outcome = _in_thread(manager._run_followed_restart, request)
        thread.join(5)
    finally:
        stop.set()

    assert outcome == [(True, None)]
    assert manager.state == "serving"
    assert bridged == [5.0]
    assert spawn_threads and spawn_threads[0] != threading.main_thread().name


def test_a_followed_restart_that_hits_the_state_guard_publishes_the_conflict(
    monkeypatch,
):
    """A follower has no caller to refuse, and node 0 is already waiting on
    its ack, so skipping the epoch silently would hang the whole restart."""
    manager, store, _coordinator = _multi_node_manager(monkeypatch, node_rank=1)
    manager._state = "restarting"

    with pytest.raises(RuntimeError, match="restart conflict"):
        manager.restart_workers(timeout_s=5, epoch=4, deadline=time.time() + 5)

    assert store.snapshot()[f"restart/4/{PHASE_ARMED}/node1"].startswith(
        "failed:worker restart conflict"
    )


def test_the_readiness_wait_polls_the_peers_verdict_each_interval(monkeypatch):
    """A local rank blocked in a rendezvous its peers will never join shows no
    failure of its own, so without this the restart waits out its timeout."""
    manager, store, coordinator = _multi_node_manager(monkeypatch)
    del manager._wait_for_workers_ready
    manager._monitor_interval_s = 0.01
    manager._ready_event = threading.Event()
    manager.worker_process = SimpleNamespace(processes=[object()])
    manager._worker_exit_state = _ExitState()
    manager._collect_worker_exit_reason = lambda: None
    store.set(f"restart/1/{PHASE_READY}/node1", "failed:cohort never became ready")

    started = time.monotonic()
    with pytest.raises(PeerRestartError) as caught:
        manager._wait_for_workers_ready(
            timeout_s=30,
            peer_check=lambda: coordinator.raise_if_peer_failed(1, PHASE_READY),
        )

    assert caught.value.node == 1
    assert time.monotonic() - started < 5


def test_a_cohort_failure_is_published_under_the_generation_that_failed(
    monkeypatch,
):
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    manager._worker_exit_state = _ExitState()
    manager._reconcile = lambda reason, fail_in_flight: {"failed_batches": 0}

    manager._handle_worker_failure("Detected worker process exit", None)

    assert manager.state == "workers_down"
    # _manager's cohort is generation 3; a peer keys the failure by the cohort
    # it belongs to, never by whatever generation it reaches later.
    assert store.snapshot()["down/3/node0"] == "Detected worker process exit"


def test_the_reaction_to_a_peers_down_key_does_not_echo_one_back(monkeypatch):
    """First writer wins anyway, but a second node's key for an echo would
    make every node look like an independent origin of the failure."""
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    manager._worker_exit_state = _ExitState()
    manager._reconcile = lambda reason, fail_in_flight: {"failed_batches": 0}

    manager._handle_peer_cohort_down(1, "Detected worker process exit")

    assert manager.state == "workers_down"
    assert manager._worker_exit_state.reason == (
        "peer node 1 worker cohort failed: Detected worker process exit"
    )
    assert "down/3/node0" not in store.snapshot()


# ------------------------------------------------ the control address itself


def _args(tmp_path, **overrides):
    values = {
        "model": "openai/gpt-oss-120b",
        "host_kv_cache_size": 1,
        "storage_path": tmp_path / "storage",
        "nnodes": 2,
        "node_rank": 0,
        "world_size": 16,
    }
    values.update(overrides)
    return ServerArgs(**values)


@pytest.fixture
def _skip_port_probes(monkeypatch):
    """Validation binds real sockets; the address rules under test do not."""
    monkeypatch.setattr(
        server_args_module, "_ensure_local_port_free", lambda port, label: None
    )


def test_the_control_address_defaults_to_the_rendezvous_host_and_next_port(
    tmp_path,
):
    args = _args(tmp_path, dist_init_addr="10.0.0.1:12355")
    assert args.restart_control_addr == "10.0.0.1:12356"

    # An explicit value is kept untouched.
    explicit = _args(
        tmp_path,
        dist_init_addr="10.0.0.1:12355",
        restart_control_addr="10.0.0.1:23000",
    )
    assert explicit.restart_control_addr == "10.0.0.1:23000"


@pytest.mark.parametrize(
    "addr, message",
    [
        ("10.0.0.1:0", "Invalid restart_control_addr port"),
        ("10.0.0.1:70000", "Invalid restart_control_addr port"),
        ("10.0.0.1:10900", "must differ from the listen port"),
        ("10.0.0.1:12355", "must differ from the listen port"),
        ("10.0.0.1:20050", "must not fall inside"),
    ],
)
def test_a_colliding_or_out_of_range_control_port_is_refused(
    tmp_path, _skip_port_probes, addr, message
):
    """Every one of these would break only the recovery path, and only once a
    restart was attempted -- long after the launch that chose the port."""
    with pytest.raises(ValueError, match=message):
        server_args_module.validate_server_args(
            _args(tmp_path, dist_init_addr="10.0.0.1:12355",
                  restart_control_addr=addr)
        )


def test_a_single_node_launch_never_validates_or_binds_the_control_port(
    tmp_path, monkeypatch
):
    probed = []
    monkeypatch.setattr(
        server_args_module,
        "_ensure_local_port_free",
        lambda port, label: probed.append(label),
    )

    server_args_module.validate_server_args(
        _args(
            tmp_path,
            nnodes=1,
            world_size=1,
            # Colliding on purpose: with one node there is no store to collide.
            restart_control_addr="10.0.0.1:10900",
        )
    )

    assert "restart_control_addr" not in probed


def test_node_zero_is_the_only_node_that_binds_the_control_port(tmp_path, monkeypatch):
    probed = []
    monkeypatch.setattr(
        server_args_module,
        "_ensure_local_port_free",
        lambda port, label: probed.append(label),
    )

    server_args_module.validate_server_args(
        _args(tmp_path, node_rank=1, dist_init_addr="10.0.0.1:12355")
    )
    assert "restart_control_addr" not in probed

    probed.clear()
    server_args_module.validate_server_args(
        _args(tmp_path, node_rank=0, dist_init_addr="10.0.0.1:12355")
    )
    assert "restart_control_addr" in probed


# ------------------------------------------------------------- the HTTP route


def _http_tree() -> ast.Module:
    return ast.parse(HTTP_SERVER_PATH.read_text(), filename=str(HTTP_SERVER_PATH))


def _http_function(name: str):
    return next(
        node for node in ast.walk(_http_tree())
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _mount_restart_route(worker, server):
    """The real route body on a bare app -- no server stack, no lifespan."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    route = copy.deepcopy(_http_function("restart_workers"))
    route.decorator_list = []
    namespace = {
        "asyncio": importlib.import_module("asyncio"),
        "HTTPException": HTTPException,
        "JSONResponse": JSONResponse,
        "Request": Request,
        "PeerRestartError": PeerRestartError,
        "make_main_thread_dispatcher": lambda loop: (lambda fn, timeout: fn()),
    }
    module = ast.Module(body=[route], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(HTTP_SERVER_PATH), "exec"),
        namespace,
    )

    app = FastAPI()
    app.state.worker = worker
    app.state.server_args = server
    app.post("/v1/workers/restart")(namespace["restart_workers"])
    return TestClient(app)


def test_a_peer_node_refuses_the_restart_and_names_the_coordinator():
    calls = []
    worker = SimpleNamespace(
        restart_workers=lambda timeout_s=None: calls.append(timeout_s),
        set_main_thread_dispatcher=lambda dispatcher: None,
        state="serving",
        generation=3,
        status=lambda: {},
    )
    client = _mount_restart_route(worker, SimpleNamespace(node_rank=1))

    response = client.post("/v1/workers/restart", json={"timeout": 5})

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "message": "worker restart is coordinated by node 0; send it there",
        "coordinator_node": 0,
    }
    # Refused before anything touched the pool.
    assert calls == []


def test_a_peer_failure_is_a_500_naming_the_node_and_phase():
    def restart(timeout_s=None):
        raise PeerRestartError(1, PHASE_READY, "cohort never became ready")

    worker = SimpleNamespace(
        restart_workers=restart,
        set_main_thread_dispatcher=lambda dispatcher: None,
        state="workers_down",
        generation=3,
        status=lambda: {"restart_epoch": 7},
    )
    client = _mount_restart_route(worker, SimpleNamespace(node_rank=0))

    response = client.post("/v1/workers/restart", json={"timeout": 5})

    assert response.status_code == 500
    assert response.json() == {
        "state": "workers_down",
        "generation": 3,
        "epoch": 7,
        "error": (
            "peer node 1 did not complete restart phase P2: "
            "cohort never became ready"
        ),
        "failed_node": 1,
        "failed_phase": PHASE_READY,
    }


def test_a_persistent_main_thread_dispatcher_is_installed_before_the_pool():
    """A peer's restart comes from the coordinator's thread, with no request
    behind it, so the respawn bridge cannot be per-request only: workers arm a
    parent-death signal tied to the thread that spawned them."""
    lifespan = _http_function("lifespan")
    calls = {
        ast.unparse(node): node.lineno
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Call)
    }
    install = next(
        call for call in calls
        if call.startswith("worker.set_main_thread_dispatcher")
    )
    assert "make_main_thread_dispatcher(asyncio.get_running_loop())" in install
    assert calls[install] < calls["worker.start()"]

    # And the per-request install still happens, through the same factory.
    route = _http_function("restart_workers")
    assert any(
        ast.unparse(node).startswith(
            "worker_manager.set_main_thread_dispatcher("
            "make_main_thread_dispatcher("
        )
        for node in ast.walk(route)
        if isinstance(node, ast.Call)
    )


def test_a_peer_failing_the_last_phase_leaves_this_scheduler_quiesced(
    monkeypatch,
):
    """P2 is acked before the node-local scheduler hooks, so a peer that fails
    there never leaves this node resumed under a workers_down state."""
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    resumed = []
    manager._scheduler_generation_hook = lambda: resumed.append("generation")
    manager._resume_hook = lambda: resumed.append("resume")
    stop, _acked = _healthy_peer(
        store, epoch=1, phases=(PHASE_ARMED, PHASE_TORN_DOWN)
    )
    store.set(f"restart/1/{PHASE_READY}/node1", "failed:spawn failed")
    try:
        with pytest.raises(PeerRestartError) as caught:
            manager.restart_workers(timeout_s=5)
    finally:
        stop.set()

    assert caught.value.phase == PHASE_READY
    assert resumed == []
    assert manager.state == "workers_down"


def test_a_failure_after_every_node_was_ready_is_published_as_the_new_cohort_down(
    monkeypatch,
):
    """Past P2 the peers serve the next generation and no longer wait on any
    phase ack, so only that generation's down key can reach them."""
    manager, store, _coordinator = _multi_node_manager(monkeypatch)
    stop, _acked = _healthy_peer(store, epoch=1)

    def failing_resume():
        raise RuntimeError("scheduler resume failed")

    manager._resume_hook = failing_resume
    generation = manager.generation
    try:
        with pytest.raises(RuntimeError, match="scheduler resume failed"):
            manager.restart_workers(timeout_s=5)
    finally:
        stop.set()

    assert manager.state == "workers_down"
    assert store.snapshot()[f"down/{generation + 1}/node0"] == (
        "scheduler resume failed"
    )


def test_a_peer_down_key_survives_a_failed_read_and_is_reported_later(
    monkeypatch,
):
    store, (node0, node1) = _cluster()
    node1.publish_down(3, "rank 9 exited")

    # The key is seen (check succeeds) but reading it fails once.
    real_get = store.get

    def failing_get(key):
        monkeypatch.setattr(store, "get", real_get)
        raise ConnectionError("transient")

    monkeypatch.setattr(store, "get", failing_get)
    with pytest.raises(ConnectionError):
        node0._poll_peer_down(3)

    assert node0._poll_peer_down(3) == (1, "rank 9 exited")
    assert node0._poll_peer_down(3) is None


def test_a_barrier_on_a_closed_coordinator_fails_at_once_not_at_the_deadline():
    _store, (node0, _node1) = _cluster()
    node0.close()

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="store is closed"):
        node0.barrier(1, PHASE_ARMED, time.time() + 30)

    assert time.monotonic() - started < 1.0


def test_a_shutdown_does_not_wait_out_an_open_restart_request():
    """uvicorn's graceful shutdown waits for open requests before the lifespan
    shutdown runs WorkerManager.stop(); an open POST /v1/workers/restart would
    defer every SIGTERM until the restart itself ended."""
    tree = _http_tree()
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "uvicorn.run"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "timeout_graceful_shutdown" in keywords
    constant = next(
        node.value.value for node in tree.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == ast.unparse(keywords["timeout_graceful_shutdown"])
    )
    assert 0 < constant <= 30
