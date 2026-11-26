"""The worker hang watchdog arms at the READINESS boundary, never at spawn.

A real 4-node start was killed DURING weight load by ``worker-N watchdog
timeout (watchdog_timeout=300.0, soft=False)``: weight loading legitimately
exceeds the timeout with no heartbeats, and startup already has its own
independent guards (the server-side startup-phase watchdog, the
--startup-timeout whole-startup deadline, the restart coordinator's peer
heartbeats). The hang watchdog's job is the SERVING phase only.

Behaviour contract, run against the REAL batchgen/server/watchdog.py with
real timing in subprocesses (the watchdog kills by SIGQUIT to its parent, so
each lifecycle runs under a disposable parent-harness process):

(a) no heartbeats before readiness for longer than the timeout -> no kill;
(b) after readiness, a worker silent past the timeout -> killed exactly as
    today (SIGQUIT reaches the parent process);
(c) a worker-pool restart window is a new startup window: the old cohort's
    watchdog dies with its process, nothing fires for the whole window or
    during the next cohort's slow startup, and the watchdog re-arms when
    serving resumes (a post-readiness silence then fires again).

The subprocess children execute the arming lifecycle the production worker
main loop now uses; the ordering test at the bottom pins that lifecycle to
the production seam itself (``_server_worker_main_impl``): every
``Watchdog.create`` sits after the all-rank barrier and the ready signal,
before the serving loop. The decode-step watchdog is additionally proven to
start disabled and count only inside decode loops.
"""

import ast
import importlib.util
import logging
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MAIN_LOOP = ROOT / "batchgen" / "server_worker_main_loop.py"
WATCHDOG = ROOT / "batchgen" / "server" / "watchdog.py"

# Small enough for fast tests, large enough that scheduler jitter on a loaded
# CI host cannot starve a feeding loop past it.
TIMEOUT = 0.5
# The watchdog sleeps 5s between logging the timeout and signalling the
# parent (watchdog.py), so a fire takes ~ TIMEOUT * 1.5 + 5s end to end.
FIRE_BUDGET_S = TIMEOUT * 1.5 + 5.0 + 2.0


# --------------------------------------------------------------------------- #
#  subprocess harness: a disposable parent so SIGQUIT never reaches pytest
# --------------------------------------------------------------------------- #


_CHILD_PRELUDE = """\
import importlib.util, os, sys, time

spec = importlib.util.spec_from_file_location(
    "batchgen_server_watchdog", {watchdog_path!r})
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)
TIMEOUT = {timeout}
"""


def _write(tmp_path, name, body, **fmt):
    path = tmp_path / name
    path.write_text(body.format(
        watchdog_path=str(WATCHDOG), timeout=TIMEOUT, **fmt))
    return path


def _run(script, *args, timeout_s):
    return subprocess.run(
        [sys.executable, str(script), *[str(a) for a in args]],
        capture_output=True, text=True, timeout=timeout_s,
    )


# (a) -- a startup longer than the timeout, with no heartbeats, is never
# killed: the watchdog does not exist before the readiness boundary.
_PARENT_A = """\
import os, signal, subprocess, sys

def _on_quit(signum, frame):
    os._exit(9)  # a kill before/during startup is exactly the defect

signal.signal(signal.SIGQUIT, _on_quit)
child = subprocess.Popen([sys.executable, sys.argv[1]])
sys.exit(child.wait())
"""

_CHILD_A = _CHILD_PRELUDE + """\
# Startup window: three timeouts of silence, UNARMED (the production main
# loop creates the watchdog only at the readiness boundary).
time.sleep(3 * TIMEOUT)

# Readiness boundary: arm, then serve with heartbeats.
watchdog = wd.Watchdog.create(
    debug_name="worker-0", watchdog_timeout=TIMEOUT, soft=False)
for _ in range(5):
    watchdog.feed()
    time.sleep(TIMEOUT / 4)
os._exit(0)
"""


def test_no_heartbeats_before_readiness_longer_than_timeout_is_not_killed(
    tmp_path,
):
    child = _write(tmp_path, "child_a.py", _CHILD_A)
    parent = _write(tmp_path, "parent_a.py", _PARENT_A)

    done = _run(parent, child, timeout_s=60)

    assert done.returncode == 0, (done.returncode, done.stderr)


# (b) -- after readiness a silent worker is killed exactly as today: the
# watchdog SIGQUITs the parent process.
_PARENT_B = """\
import os, signal, subprocess, sys

child = None

def _on_quit(signum, frame):
    if child is not None:
        child.kill()
    os._exit(7)  # the kill arrived, as it must

signal.signal(signal.SIGQUIT, _on_quit)
child = subprocess.Popen([sys.executable, sys.argv[1]])
sys.exit(child.wait())
"""

_CHILD_B = _CHILD_PRELUDE + """\
# Readiness boundary: armed, one heartbeat, then silence forever.
watchdog = wd.Watchdog.create(
    debug_name="worker-0", watchdog_timeout=TIMEOUT, soft=False)
watchdog.feed()
time.sleep(3600)
"""


def test_silence_past_the_timeout_after_readiness_is_killed_as_today(tmp_path):
    child = _write(tmp_path, "child_b.py", _CHILD_B)
    parent = _write(tmp_path, "parent_b.py", _PARENT_B)

    started = time.monotonic()
    done = _run(parent, child, timeout_s=60)
    elapsed = time.monotonic() - started

    assert done.returncode == 7, (done.returncode, done.stderr)
    assert elapsed < FIRE_BUDGET_S + 10, elapsed


# (c) -- a restart window is a new startup window. The serving cohort's
# watchdog dies with its process at teardown; nothing fires through the
# whole window nor through the next cohort's slow (unarmed) startup; the
# watchdog re-arms at the new readiness and a silence then fires again.
_PARENT_C = """\
import os, signal, subprocess, sys, time

marker = sys.argv[2]
cohort = None

def _on_quit(signum, frame):
    if cohort is not None:
        cohort.kill()
    # Fired before the new cohort's readiness: the disarm contract is broken.
    os._exit(7 if os.path.exists(marker) else 9)

signal.signal(signal.SIGQUIT, _on_quit)

# Serving cohort: armed and feeding. Tear it down the way a restart does --
# the process is killed, and its watchdog must die with it.
ready1 = sys.argv[3]
cohort = subprocess.Popen([sys.executable, sys.argv[1], "serve", ready1])
deadline = time.monotonic() + 15
while not os.path.exists(ready1):
    if time.monotonic() > deadline:
        os._exit(8)
    time.sleep(0.05)
cohort.kill()
cohort.wait()
cohort = None

# Restart window: longer than timeout + the watchdog's 5s pre-kill sleep, so
# a watchdog that survived the teardown would land its SIGQUIT here (exit 9).
time.sleep({window})

# Respawned cohort: slow unarmed startup, marker at readiness, then silence.
cohort = subprocess.Popen([sys.executable, sys.argv[1], "restart", marker])
sys.exit(cohort.wait())
"""

_CHILD_C = _CHILD_PRELUDE + """\
mode, marker = sys.argv[1], sys.argv[2]

if mode == "serve":
    # First cohort: readiness, then keep serving (feeding) until killed.
    watchdog = wd.Watchdog.create(
        debug_name="worker-0", watchdog_timeout=TIMEOUT, soft=False)
    watchdog.feed()
    open(marker, "w").close()
    while True:
        watchdog.feed()
        time.sleep(TIMEOUT / 4)

# Respawned cohort: a slow startup, unarmed -- the restart window's weight
# reload. Then readiness (marker + arm), then silence, which must fire.
time.sleep(3 * TIMEOUT)
open(marker, "w").close()
watchdog = wd.Watchdog.create(
    debug_name="worker-0", watchdog_timeout=TIMEOUT, soft=False)
watchdog.feed()
time.sleep(3600)
"""


def test_restart_window_is_disarmed_and_rearms_when_serving_resumes(tmp_path):
    window = FIRE_BUDGET_S
    child = _write(tmp_path, "child_c.py", _CHILD_C)
    parent = _write(tmp_path, "parent_c.py", _PARENT_C, window=window)

    done = _run(
        parent, child, tmp_path / "ready2.marker", tmp_path / "ready1.marker",
        timeout_s=120,
    )

    assert done.returncode == 7, (done.returncode, done.stderr)


# --------------------------------------------------------------------------- #
#  decode-step watchdog: never counts outside a decode loop
# --------------------------------------------------------------------------- #


def _load_watchdog_module():
    spec = importlib.util.spec_from_file_location(
        "batchgen_server_watchdog_inproc", str(WATCHDOG))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_step_watchdog_counts_only_inside_decode_loops(caplog):
    from batchgen.worker.worker_lifecycle_engine import WorkerLifecycleEngine

    wd = _load_watchdog_module()
    engine = WorkerLifecycleEngine()
    worker = SimpleNamespace(_decode_watchdog=None)
    # Soft mode: the semantics under test are WHEN it counts, and a hard
    # watchdog would SIGQUIT the pytest process's parent.
    watchdog = wd.Watchdog.create(
        debug_name="decode-0", watchdog_timeout=TIMEOUT, soft=True)

    try:
        with caplog.at_level(logging.ERROR):
            # The production seam: set_decode_watchdog starts it DISABLED, so
            # the whole startup/restart window (and idle serving) is silent
            # for it without a single feed.
            engine.set_decode_watchdog(worker, watchdog)
            time.sleep(3 * TIMEOUT)
            assert "decode-0 watchdog timeout" not in caplog.text

            # Inside a decode loop it counts: enabled and then starved, it
            # fires.
            engine.enable_decode_watchdog(worker)
            deadline = time.monotonic() + 3 * TIMEOUT + 30
            while "decode-0 watchdog timeout" not in caplog.text:
                assert time.monotonic() < deadline, caplog.text
                time.sleep(0.05)
    finally:
        # Quiesce the daemon thread's trigger for the rest of the session.
        engine.disable_decode_watchdog(worker)


# --------------------------------------------------------------------------- #
#  the production seam: arming sits at the readiness boundary, one code path
# --------------------------------------------------------------------------- #


def _calls(node, name):
    """Line numbers of every ``name(...)`` or ``*.name(...)`` call."""
    return [
        n.lineno
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == name)
            or (isinstance(n.func, ast.Name) and n.func.id == name)
        )
    ]


def _watchdog_creates(node):
    return [
        n.lineno
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "create"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "Watchdog"
    ]


def test_main_loop_arms_both_watchdogs_at_the_readiness_boundary():
    tree = ast.parse(MAIN_LOOP.read_text())
    impl = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_server_worker_main_impl"
    )

    # One code path: the module's only two Watchdog.create sites are the
    # hang and decode watchdogs inside the worker main impl.
    creates = _watchdog_creates(impl)
    assert len(creates) == 2
    assert sorted(_watchdog_creates(tree)) == sorted(creates)

    (ready,) = _calls(impl, "_signal_local_worker_manager_ready")
    (deploy,) = _calls(impl, "prepare_deployment")
    (probe,) = _calls(impl, "set_cohort_shutdown_probe")
    (set_hang,) = _calls(impl, "set_watchdog")
    (set_decode,) = _calls(impl, "set_decode_watchdog")

    # Not at spawn: nothing is created or attached until the weight load,
    # the all-rank barrier and the ready signal are all behind the cohort.
    barrier_before_ready = max(
        line for line in _calls(impl, "barrier") if line < ready
    )
    assert deploy < barrier_before_ready < ready < min(creates)

    # ...and armed before the serving loop begins, on every respawn (initial
    # start, worker-pool restart, debug-state rewarm all re-run this impl).
    assert max(creates) < probe
    assert ready < set_hang < probe
    assert ready < set_decode < probe
