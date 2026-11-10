"""A cross-node rendezvous is charged from when every node reached it.

On a 4x8 launch one node spent most of its start-up allocating host KV
before spawning a worker. The ranks already spawned on the other nodes
entered the communicator rendezvous, and the startup watchdog failed the
cohort when its 900 s budget ran out on THEIR clock, naming the earliest
arrivals rather than the node everyone was waiting for. The budget of a
cross-node phase now starts once this node has seen every node reach it;
until then the waiting ranks are not charged and the missing nodes are
logged with what they are doing.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batchgen.worker.startup_phases import (  # noqa: E402
    PHASE_COMMUNICATOR_INIT,
    PHASE_HOST_KV_ALLOCATION,
    PHASE_MODEL_BUILD,
    StartupPhaseRecord,
    StartupPhaseWatchdog,
    phase_budget_s,
)
from tests.test_restart_coordinator import FakeStore, RestartCoordinator  # noqa: E402

WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"
BUDGET = phase_budget_s(PHASE_COMMUNICATOR_INIT)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _node(node_rank, store, clock, slots=3):
    coordinator = RestartCoordinator(
        2, node_rank, "127.0.0.1", 0,
        store_factory=lambda *_a, **_k: store, clock=clock,
    )
    coordinator.start()
    record = StartupPhaseRecord(
        slots, [0.0] * (slots * StartupPhaseRecord._FIELDS), clock=clock
    )
    labels = [f"node {node_rank} server"] + [
        f"node {node_rank} rank {node_rank * (slots - 1) + i}"
        for i in range(slots - 1)
    ]
    breaches = []
    watchdog = StartupPhaseWatchdog(
        record, labels, breaches.append, cluster=coordinator, clock=clock,
        wait_log_interval_s=0.0,
    )
    return coordinator, record, watchdog


def _exchange(*nodes):
    """One round of what the heartbeat and follower threads do."""
    store = nodes[0][0]._store
    for coordinator, _, watchdog in nodes:
        watchdog.poll()
        coordinator._beat()
    for coordinator, _, _ in nodes:
        coordinator._read_peer_startup(store)


def test_a_rendezvous_waiting_for_a_node_still_loading_is_not_charged(caplog):
    store, clock = FakeStore(), Clock()
    node0 = _node(0, store, clock)
    node1 = _node(1, store, clock)
    node0[1].enter(1, PHASE_COMMUNICATOR_INIT)
    node0[1].enter(2, PHASE_COMMUNICATOR_INIT)
    node1[1].enter(0, PHASE_HOST_KV_ALLOCATION)
    _exchange(node0, node1)

    clock.now += BUDGET + 300.0
    with caplog.at_level(logging.WARNING, logger="batchgen.worker.startup_phases"):
        _exchange(node0, node1)
        assert node0[2].poll() is None

    waiting = [r.getMessage() for r in caplog.records if "not reached it" in r.getMessage()]
    assert waiting, caplog.text
    assert "node 1 (node 1 server in host-kv-allocation for" in waiting[-1]


def test_the_budget_starts_once_the_last_node_reaches_the_rendezvous():
    store, clock = FakeStore(), Clock()
    node0 = _node(0, store, clock)
    node1 = _node(1, store, clock)
    node0[1].enter(1, PHASE_COMMUNICATOR_INIT)
    node1[1].enter(0, PHASE_HOST_KV_ALLOCATION)
    _exchange(node0, node1)
    clock.now += BUDGET + 300.0

    node1[1].leave(0)
    node1[1].enter(1, PHASE_COMMUNICATOR_INIT)
    _exchange(node0, node1)
    assert node0[2].poll() is None

    clock.now += BUDGET - 1.0
    _exchange(node0, node1)
    assert node0[2].poll() is None

    clock.now += 2.0
    reason = node0[2].poll()
    assert reason is not None
    assert "node 0 rank 0 stuck in communicator-init" in reason
    assert f"for {BUDGET + 1.0:.0f}s" in reason


def test_a_rank_that_passed_the_rendezvous_between_polls_counts_as_reached():
    store, clock = FakeStore(), Clock()
    node0 = _node(0, store, clock)
    node1 = _node(1, store, clock)
    node1[1].enter(1, PHASE_COMMUNICATOR_INIT)
    node1[1].enter(1, PHASE_MODEL_BUILD)  # moved on before any poll
    node0[1].enter(1, PHASE_COMMUNICATOR_INIT)
    _exchange(node0, node1)
    assert node0[0].all_nodes_reached_for_s(PHASE_COMMUNICATOR_INIT) == 0.0


def test_a_node_that_never_published_is_named_as_such():
    store, clock = FakeStore(), Clock()
    node0 = _node(0, store, clock)
    _node(1, store, clock)
    node0[1].enter(1, PHASE_COMMUNICATOR_INIT)
    node0[2].poll()
    assert node0[0].nodes_not_reached(PHASE_COMMUNICATOR_INIT) == [
        "node 1 (no start-up progress published yet)"
    ]


def test_the_manager_watches_host_kv_allocation_and_shares_progress():
    source = WORKER_MANAGER.read_text()
    tree = ast.parse(source)
    start = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_impl"
    )
    wrapped = [
        node for node in ast.walk(start)
        if isinstance(node, ast.With)
        and "PHASE_HOST_KV_ALLOCATION" in ast.unparse(node.items[0])
        and "allocate_host_kv_cache" in ast.unparse(node)
    ]
    assert wrapped
    arm = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_start_startup_phase_watchdog"
    )
    assert "cluster=" in ast.unparse(arm)
