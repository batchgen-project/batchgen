"""The startup phase record and the watchdog that bounds each phase.

CPU only. The record is driven directly (a plain array stands in for the
shared one, and a fake clock for the monotonic one), so a rank stuck inside a
phase, a rank that is merely slow, a torn read of a slot being written and a
scaled budget are all reproduced without a cohort.

The last group of gates reads the source: that the phase names reported by
the worker are declared, that the watchdog is armed before the first long
phase of a start-up, and that a breach takes the existing worker-fatal route
rather than one of its own.
"""

import ast
import logging
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batchgen.worker.startup_phases import (  # noqa: E402
    DEFAULT_PHASE_BUDGET_S,
    NULL_STARTUP_PHASE_REPORTER,
    PHASE_BUDGETS_S,
    PHASE_COMMUNICATOR_INIT,
    PHASE_DECODE_SHARD_BUILD,
    PHASE_PREWARM,
    PHASE_STORE_STAGING,
    STARTUP_PHASES,
    StartupPhaseRecord,
    StartupPhaseWatchdog,
    phase_budget_s,
)

WORKER = ROOT / "batchgen" / "batchgen_worker.py"
MAIN_LOOP = ROOT / "batchgen" / "server_worker_main_loop.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def record(slots=3, clock=None):
    clock = clock or FakeClock()
    return StartupPhaseRecord(
        slots, [0.0] * (slots * StartupPhaseRecord._FIELDS), clock=clock
    ), clock


def labels(slots):
    return ["node 0 server"] + [f"node 0 rank {rank}" for rank in range(slots - 1)]


# --------------------------------------------------------------------------- #
#  the record                                                                 #
# --------------------------------------------------------------------------- #


def test_a_fresh_record_has_every_slot_idle():
    phases, _ = record()
    assert [phases.read(slot) for slot in range(phases.slots)] == [None, None, None]


def test_a_slot_reports_its_phase_and_how_long_it_has_been_in_it():
    phases, clock = record()
    phases.enter(1, PHASE_COMMUNICATOR_INIT)
    clock.advance(42.0)
    assert phases.read(1) == (PHASE_COMMUNICATOR_INIT, 42.0)
    # Only that slot.
    assert phases.read(0) is None


def test_leaving_a_phase_returns_the_slot_to_idle():
    phases, clock = record()
    phases.enter(0, PHASE_STORE_STAGING)
    phases.leave(0)
    clock.advance(10_000.0)
    assert phases.read(0) is None


def test_an_undeclared_phase_name_is_refused():
    phases, _ = record()
    with pytest.raises(ValueError, match="unknown startup phase"):
        phases.enter(0, "store_staging")
    with pytest.raises(ValueError, match="unknown startup phase"):
        NULL_STARTUP_PHASE_REPORTER.enter("not-a-phase")


def test_every_declared_phase_round_trips():
    phases, _ = record(slots=1)
    for phase in STARTUP_PHASES:
        phases.enter(0, phase)
        assert phases.read(0)[0] == phase


def test_a_slot_being_written_reads_as_unavailable_rather_than_stuck():
    """Cross-process the (phase, entry time) pair can be read half written;
    an odd sequence counter is what keeps that from looking like a stall."""
    phases, clock = record()
    phases.enter(2, PHASE_DECODE_SHARD_BUILD)
    clock.advance(5.0)
    assert phases.read(2) == (PHASE_DECODE_SHARD_BUILD, 5.0)
    # Simulate a writer caught between its two counter bumps.
    phases._array[2 * StartupPhaseRecord._FIELDS] += 1.0
    assert phases.read(2) is None


def test_a_record_rejects_a_mismatched_array():
    with pytest.raises(ValueError):
        StartupPhaseRecord(2, [0.0] * 5)
    with pytest.raises(ValueError):
        StartupPhaseRecord(0, [])


def test_a_slot_outside_the_record_is_refused():
    phases, _ = record(slots=2)
    with pytest.raises(IndexError):
        phases.enter(2, PHASE_STORE_STAGING)


def test_reset_clears_only_the_named_slots():
    phases, clock = record()
    for slot in range(3):
        phases.enter(slot, PHASE_STORE_STAGING)
    clock.advance(1.0)
    phases.reset(range(1, 3))
    assert phases.read(0) == (PHASE_STORE_STAGING, 1.0)
    assert phases.read(1) is None and phases.read(2) is None


def test_the_reporter_wraps_a_body_in_one_phase():
    phases, clock = record()
    reporter = phases.reporter(1)
    with reporter.phase(PHASE_COMMUNICATOR_INIT):
        clock.advance(3.0)
        assert phases.read(1) == (PHASE_COMMUNICATOR_INIT, 3.0)
    assert phases.read(1) is None


def test_the_reporter_goes_idle_even_when_the_body_raises():
    phases, _ = record()
    reporter = phases.reporter(1)
    with pytest.raises(RuntimeError):
        with reporter.phase(PHASE_COMMUNICATOR_INIT):
            raise RuntimeError("the phase failed")
    assert phases.read(1) is None


def test_the_null_reporter_runs_every_body_and_records_nothing():
    ran = []
    with NULL_STARTUP_PHASE_REPORTER.phase(PHASE_PREWARM):
        ran.append(True)
    NULL_STARTUP_PHASE_REPORTER.leave()
    assert ran == [True]
    assert NULL_STARTUP_PHASE_REPORTER.slot == -1


# --------------------------------------------------------------------------- #
#  the budgets                                                                #
# --------------------------------------------------------------------------- #


def test_the_rendezvous_budget_is_tighter_than_the_default():
    assert PHASE_BUDGETS_S[PHASE_COMMUNICATOR_INIT] < DEFAULT_PHASE_BUDGET_S
    assert phase_budget_s(PHASE_STORE_STAGING) == DEFAULT_PHASE_BUDGET_S


def test_the_scale_multiplies_every_budget():
    for phase in STARTUP_PHASES:
        assert phase_budget_s(phase, scale=2.5) == phase_budget_s(phase) * 2.5


def test_a_non_positive_scale_is_refused():
    for scale in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            phase_budget_s(PHASE_STORE_STAGING, scale=scale)
        phases, _ = record()
        with pytest.raises(ValueError, match="must be positive"):
            StartupPhaseWatchdog(phases, labels(3), lambda reason: None, scale=scale)


def test_a_watchdog_needs_one_label_per_slot():
    phases, _ = record(slots=3)
    with pytest.raises(ValueError, match="slot labels"):
        StartupPhaseWatchdog(phases, ["only one"], lambda reason: None)


# --------------------------------------------------------------------------- #
#  the watchdog                                                               #
# --------------------------------------------------------------------------- #


def watchdog(phases, breaches, **kwargs):
    return StartupPhaseWatchdog(
        phases, labels(phases.slots), breaches.append, **kwargs
    )


def test_a_phase_inside_its_budget_is_not_breached():
    phases, clock = record()
    guard = watchdog(phases, [])
    phases.enter(1, PHASE_COMMUNICATOR_INIT)
    clock.advance(phase_budget_s(PHASE_COMMUNICATOR_INIT))
    assert guard.poll() is None


def test_a_phase_past_its_budget_names_the_rank_the_phase_and_the_seconds(caplog):
    phases, clock = record()
    guard = watchdog(phases, [])
    phases.enter(2, PHASE_COMMUNICATOR_INIT)
    stuck_for = phase_budget_s(PHASE_COMMUNICATOR_INIT) + 60.0
    clock.advance(stuck_for)
    with caplog.at_level(logging.CRITICAL, logger="batchgen.worker.startup_phases"):
        reason = guard.poll()
    assert reason is not None
    assert "node 0 rank 1" in reason
    assert PHASE_COMMUNICATOR_INIT in reason
    assert f"{stuck_for:.0f}s" in reason
    # The evidence reaches the log BEFORE anything is failed: that is the
    # whole point of this path.
    crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(crit) == 1
    message = crit[0].getMessage()
    assert "node 0 rank 1" in message and PHASE_COMMUNICATOR_INIT in message


def test_every_stuck_rank_is_named_once(caplog):
    phases, clock = record(slots=4)
    guard = watchdog(phases, [])
    for slot in (1, 3):
        phases.enter(slot, PHASE_COMMUNICATOR_INIT)
    clock.advance(phase_budget_s(PHASE_COMMUNICATOR_INIT) + 1.0)
    with caplog.at_level(logging.CRITICAL, logger="batchgen.worker.startup_phases"):
        reason = guard.poll()
    assert [r for r in caplog.records if r.levelno == logging.CRITICAL].__len__() == 2
    assert "node 0 rank 0" in reason and "node 0 rank 2" in reason


def test_a_phase_change_restarts_the_budget():
    """A startup that is slow but MOVING is never failed."""
    phases, clock = record()
    guard = watchdog(phases, [])
    phases.enter(1, PHASE_COMMUNICATOR_INIT)
    clock.advance(phase_budget_s(PHASE_COMMUNICATOR_INIT) - 1.0)
    assert guard.poll() is None
    phases.enter(1, PHASE_DECODE_SHARD_BUILD)
    clock.advance(phase_budget_s(PHASE_COMMUNICATOR_INIT))
    assert guard.poll() is None
    assert phases.read(1)[0] == PHASE_DECODE_SHARD_BUILD


def test_the_scale_moves_the_breach_point():
    phases, clock = record()
    tight = watchdog(phases, [], scale=0.5)
    loose = watchdog(phases, [], scale=2.0)
    phases.enter(1, PHASE_COMMUNICATOR_INIT)
    clock.advance(phase_budget_s(PHASE_COMMUNICATOR_INIT) * 0.75)
    assert tight.poll() is not None
    assert loose.poll() is None


def test_the_servers_own_staging_slot_is_watched_too():
    phases, clock = record()
    guard = watchdog(phases, [])
    phases.enter(0, PHASE_STORE_STAGING)
    clock.advance(phase_budget_s(PHASE_STORE_STAGING) + 1.0)
    reason = guard.poll()
    assert "node 0 server" in reason and PHASE_STORE_STAGING in reason


def test_the_thread_fails_the_startup_once_and_then_stops():
    phases, clock = record()
    breaches = []
    guard = watchdog(phases, breaches, poll_interval_s=0.01)
    phases.enter(1, PHASE_COMMUNICATOR_INIT)
    clock.advance(phase_budget_s(PHASE_COMMUNICATOR_INIT) + 1.0)
    guard.start()
    deadline = time.monotonic() + 5.0
    while not breaches and time.monotonic() < deadline:
        time.sleep(0.01)
    guard.stop()
    assert len(breaches) == 1
    assert guard.fired
    # Still stuck, but the failure it reported owns the teardown now.
    time.sleep(0.05)
    assert len(breaches) == 1


def test_a_thread_stopped_before_any_breach_reports_nothing():
    phases, clock = record()
    breaches = []
    guard = watchdog(phases, breaches, poll_interval_s=0.01)
    phases.enter(1, PHASE_COMMUNICATOR_INIT)
    guard.start()
    time.sleep(0.05)
    guard.stop()
    assert breaches == []
    assert not guard.fired


def test_a_breach_handler_that_raises_does_not_kill_the_thread(caplog):
    phases, clock = record()

    def explode(reason):
        raise RuntimeError("teardown failed")

    guard = StartupPhaseWatchdog(
        phases, labels(3), explode, poll_interval_s=0.01
    )
    phases.enter(1, PHASE_COMMUNICATOR_INIT)
    clock.advance(phase_budget_s(PHASE_COMMUNICATOR_INIT) + 1.0)
    with caplog.at_level(logging.ERROR, logger="batchgen.worker.startup_phases"):
        guard.start()
        deadline = time.monotonic() + 5.0
        while not guard.fired and time.monotonic() < deadline:
            time.sleep(0.01)
        guard.stop()
    assert guard.fired
    assert any("did not complete" in r.getMessage() for r in caplog.records)


def test_a_record_that_cannot_be_read_is_logged_and_retried(caplog):
    class Hostile(StartupPhaseRecord):
        def __init__(self):
            super().__init__(1, [0.0] * StartupPhaseRecord._FIELDS)
            self.reads = 0

        def read(self, slot):
            self.reads += 1
            raise OSError("the shared mapping is gone")

    hostile = Hostile()
    guard = StartupPhaseWatchdog(
        hostile, ["node 0 server"], lambda reason: None, poll_interval_s=0.01
    )
    with caplog.at_level(logging.ERROR, logger="batchgen.worker.startup_phases"):
        guard.start()
        deadline = time.monotonic() + 5.0
        while hostile.reads < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        guard.stop()
    assert hostile.reads >= 2
    assert not guard.fired


def test_the_snapshot_lists_every_slot_in_a_phase():
    phases, clock = record()
    guard = watchdog(phases, [])
    phases.enter(0, PHASE_STORE_STAGING)
    phases.enter(2, PHASE_DECODE_SHARD_BUILD)
    clock.advance(7.0)
    assert guard.snapshot() == [
        ("node 0 server", PHASE_STORE_STAGING, 7.0),
        ("node 0 rank 1", PHASE_DECODE_SHARD_BUILD, 7.0),
    ]


def test_the_record_is_shared_across_processes_without_a_lock():
    """The shape the manager depends on: a spawn-context RawArray, so a slot
    can be read while its writer is wedged below Python or already dead."""
    import multiprocessing as real_mp

    ctx = real_mp.get_context("spawn")
    shared = StartupPhaseRecord.create(2, ctx)
    shared.enter(1, PHASE_COMMUNICATOR_INIT)
    phase, seconds = shared.read(1)
    assert phase == PHASE_COMMUNICATOR_INIT
    assert 0.0 <= seconds < 5.0
    assert not hasattr(shared._array, "get_lock")


# --------------------------------------------------------------------------- #
#  the wiring, read from the source                                           #
# --------------------------------------------------------------------------- #


def _function(path, function_name, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    return next(
        node
        for node in body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _reported_phase_names(path):
    """Every phase name literal handed to a phase reporter in ``path``.

    Only calls that actually go to a reporter: the module has unrelated
    ``.phase()`` and ``.enter()`` consumers of its own.
    """
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not node.args:
            continue
        receiver = ast.unparse(node.func.value)
        reports = node.func.attr == "_startup_phase" or (
            node.func.attr in {"phase", "enter"}
            and ("phase_reporter" in receiver or "startup_phase" in receiver)
        )
        if not reports:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(first.value)
        elif isinstance(first, ast.Name) and first.id.startswith("PHASE_"):
            names.add(first.id)
    return names


def test_every_phase_the_worker_reports_is_declared():
    """A typo in a phase name must fail here, not on a GPU node."""
    reported = _reported_phase_names(WORKER)
    assert reported, "the worker's startup pass reports no phase at all"
    assert reported <= set(STARTUP_PHASES)
    # The pass covers each of its long steps.
    assert reported >= {
        "model-build",
        "extension-preload",
        "model-extension-preload",
        "communicator-init",
        "decode-shard-build",
        "prefill-configure",
        "prewarm",
    }


def test_the_supervisor_reports_the_rendezvous_and_the_model_build():
    source = MAIN_LOOP.read_text()
    impl = _function(MAIN_LOOP, "_server_worker_main_impl")
    calls = {
        node.func.attr
        for node in ast.walk(impl)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "set_startup_phase_reporter" in calls
    assert "PHASE_COMMUNICATOR_INIT" in source
    assert "PHASE_MODEL_BUILD" in source
    # The slot a rank reports into is the one after the server's.
    assert "phase_record.reporter(1 + rank_idx)" in source


def test_the_startup_pass_phases_wrap_their_own_steps():
    startup = _function(WORKER, "prepare_kimi_k3_startup", "BatchGenWorker")
    wrapped = {}
    for node in ast.walk(startup):
        if not isinstance(node, ast.With):
            continue
        item = node.items[0].context_expr
        if not (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "_startup_phase"
        ):
            continue
        wrapped[item.args[0].value] = {
            inner.func.attr
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
        }
    assert "Init" in wrapped["model-build"]
    assert "_preload_kimi_k3_runtime_extensions" in wrapped["extension-preload"]
    assert "_ensure_pynccl_communicator" in wrapped["communicator-init"]
    assert "_load_decode_model" in wrapped["decode-shard-build"]
    assert "configure_prefill" in wrapped["prefill-configure"]


def test_the_watchdog_is_armed_before_the_first_long_phase():
    impl = _function(WORKER_MANAGER, "_start_impl", "WorkerManager")
    calls = {}
    for node in ast.walk(impl):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.setdefault(node.func.attr, []).append(node.lineno)
    assert min(calls["_start_startup_phase_watchdog"]) < min(
        calls["_stage_distributed_weight_store"]
    )
    assert min(calls["_start_startup_phase_watchdog"]) < min(
        calls["_spawn_workers"]
    )
    assert min(calls["_stop_startup_phase_watchdog"]) > min(
        calls["_wait_for_workers_ready"]
    )


def test_the_staging_step_is_reported_as_a_phase():
    """Inside the staging method, not around its call site: the start-up
    path's own call order is pinned elsewhere by AST walk order."""
    stage = _function(
        WORKER_MANAGER, "_stage_distributed_weight_store", "WorkerManager"
    )
    staged = next(
        node
        for node in ast.walk(stage)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and getattr(node.items[0].context_expr.func, "attr", None)
        == "_server_startup_phase"
    )
    assert staged.items[0].context_expr.args[0].id == "PHASE_STORE_STAGING"
    assert "stage_distributed_weight_store" in {
        node.func.id
        for node in ast.walk(staged)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_a_breach_takes_the_existing_worker_fatal_route():
    """Not a teardown of its own: the fatal path already publishes this
    node's reason to the peers and ends the process, which is what aborts
    the whole multi-node group."""
    breach = _function(
        WORKER_MANAGER, "_fail_startup_phase_breach", "WorkerManager"
    )
    calls = {
        node.func.attr
        for node in ast.walk(breach)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls == {"report_worker_fatal"}


def test_a_restart_does_not_inherit_the_previous_cohorts_phases():
    spawn = _function(WORKER_MANAGER, "_spawn_workers", "WorkerManager")
    calls = {}
    for node in ast.walk(spawn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.setdefault(node.func.attr, []).append(node.lineno)
    assert min(calls["reset"]) < min(calls["spawn"])
    assert "self._startup_phases" in ast.unparse(spawn)


# --------------------------------------------------------------------------- #
#  the CLI knob                                                               #
# --------------------------------------------------------------------------- #


def _load_server_args_module():
    import importlib
    import types

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


server_args_module = _load_server_args_module()


@pytest.fixture(autouse=True)
def _skip_port_probes(monkeypatch):
    monkeypatch.setattr(
        server_args_module, "_ensure_local_port_free", lambda port, label: None
    )


def launch(tmp_path, *extra):
    return server_args_module.prepare_server_args(
        [
            "--model", "moonshotai/Kimi-K3",
            "--nnodes", "2",
            "--world-size", "16",
            "--node-rank", "0",
            "--host-kv-cache-size", "1",
            "--replicated-weights",
            "--storage-path", str(tmp_path / "storage"),
            *[str(item) for item in extra],
        ]
    )


def test_the_budget_scale_defaults_to_one(tmp_path):
    assert launch(tmp_path).startup_phase_timeout_scale == 1.0


def test_the_budget_scale_reaches_the_server_args(tmp_path):
    args = launch(tmp_path, "--startup-phase-timeout-scale", "3")
    assert args.startup_phase_timeout_scale == 3.0


@pytest.mark.parametrize("scale", ["0", "-2"])
def test_a_non_positive_budget_scale_is_refused_at_launch(tmp_path, scale):
    with pytest.raises(ValueError, match="startup_phase_timeout_scale"):
        launch(tmp_path, "--startup-phase-timeout-scale", scale)


def test_the_manager_passes_the_launch_scale_to_the_watchdog():
    arm = _function(
        WORKER_MANAGER, "_start_startup_phase_watchdog", "WorkerManager"
    )
    assert "self.args.startup_phase_timeout_scale" in ast.unparse(arm)
