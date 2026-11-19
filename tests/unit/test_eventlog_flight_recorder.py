"""``batchgen.eventlog.flight_recorder``: the per-rank ring of recent events
and the evidence it leaves on a fatal exception, a native crash and a hang.

Process deaths and the watchdog run in child processes; the rest runs in this
process with ``faulthandler`` and ``threading.excepthook`` restored after each
test. CPU only; no CUDA, no torch.
"""

from __future__ import annotations

import ast
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()
fr = tu.import_flight_recorder()

NODE_DIR = Path(tu.RUN_DIR) / "node0"
MAIN_LOOP = tu.REPO_ROOT / "batchgen" / "server_worker_main_loop.py"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    # install() points faulthandler at its ring and takes over the thread
    # hook; neither may leak into the rest of the test session.
    enabled = []
    monkeypatch.setattr(fr.faulthandler, "enable",
                        lambda **kw: enabled.append(kw))
    monkeypatch.setattr(fr.faulthandler, "disable", lambda: None)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(fr, "_recorder", None)
    with tu.isolated_logging(ev):
        yield enabled
        recorder = fr._recorder
        if recorder is not None:
            recorder.close()


def _configure(tmp_path: Path, *, rank: int = 0, cohort: int = 1) -> None:
    ev.configure("rank", node=0, rank=rank, local_rank=rank, cohort=cohort,
                 run_id=tu.RUN_ID, run_started_at=tu.RUN_STARTED_AT,
                 log_dir=tmp_path, console=False)


def _flight_dir(tmp_path: Path) -> Path:
    return tmp_path / NODE_DIR / fr.DIR_NAME


def _read_dump(path: Path) -> tuple[dict, list[dict]]:
    header, *events = tu.read_jsonl(path)
    return header, events


def _ticks(events: list[dict]) -> list[int]:
    return [e["data"]["index"] for e in events
            if e["event"] == "worker.test.tick"]


# ------------------------------------------------------------ the ring


def test_ring_holds_exactly_the_events_the_jsonl_file_holds(tmp_path):
    _configure(tmp_path)
    recorder = fr.install()
    logger = ev.get_logger("batchgen.worker")
    for index in range(20):
        ev.log_event(logger, "worker.test.tick", index=index)
    logger.warning("plain text record %d", 7)
    assert ev.flush()

    ring = fr.read_ring(recorder.path)
    jsonl = tu.read_jsonl(ev.jsonl_path())
    # Everything logged from install() on is in both, record for record.
    assert ring.events == jsonl[-len(ring.events):]
    assert _ticks(ring.events) == list(range(20))
    assert ring.events[-1]["msg"] == "plain text record 7"
    assert ring.torn == 0 and ring.overwritten == 0
    assert ring.header["rank"] == 0 and ring.header["cohort"] == 1
    assert ring.header["pid"] == os.getpid()
    assert recorder.path == (
        _flight_dir(tmp_path) / f"rank0-cohort1-pid{os.getpid()}.ring"
    )


def test_ring_bound_in_slots_and_bytes(tmp_path):
    slots, slot_bytes = 8, 512
    _configure(tmp_path)
    recorder = fr.install(slots=slots, slot_bytes=slot_bytes)
    logger = ev.get_logger("batchgen.worker")
    for index in range(50):
        ev.log_event(logger, "worker.test.tick", index=index)
    ev.log_event(logger, "worker.test.big", blob="x" * 10_000)
    assert ev.flush()

    assert recorder.path.stat().st_size == (slots + 1) * slot_bytes
    ring = fr.read_ring(recorder.path)
    assert len(ring.events) == slots
    assert _ticks(ring.events) == list(range(43, 50))
    big = ring.events[-1]
    assert big["event"] == "worker.test.big"
    assert big["data"]["eventlog_truncated_bytes"] > 10_000
    # install() logged one line before the 51 events above.
    assert ring.recorded == 52
    assert ring.overwritten == 52 - slots


def test_torn_slot_is_skipped_not_fatal(tmp_path):
    _configure(tmp_path)
    recorder = fr.install(slots=4, slot_bytes=256)
    logger = ev.get_logger("batchgen.worker")
    for index in range(3):
        ev.log_event(logger, "worker.test.tick", index=index)
    assert ev.flush()
    # A write cut short by the process dying: the head of a slot only.
    with open(recorder.path, "r+b") as fh:
        fh.seek(2 * 256 + 20)
        fh.write(b"\x00" * 40)

    ring = fr.read_ring(recorder.path)
    assert ring.torn == 1
    assert len(ring.events) == 3


# -------------------------------------------------- trigger (a): exception


def _decode_step():
    raise RuntimeError("injected decode fault")


def test_injected_exception_dump_holds_last_events_and_traceback(tmp_path):
    _configure(tmp_path)
    recorder = fr.install()
    logger = ev.get_logger("batchgen.worker")
    for index in range(30):
        ev.log_event(logger, "worker.test.tick", index=index)
    try:
        _decode_step()
    except RuntimeError as exc:
        logging.critical("[FATAL] unhandled", exc_info=True)
        path = fr.dump(fr.TRIGGER_EXCEPTION, "unhandled worker exception", exc)

    assert path == recorder.path.with_name(
        f"rank0-cohort1-pid{os.getpid()}-dump1.jsonl"
    )
    header, events = _read_dump(path)
    assert header["v"] == 1
    assert header["event"] == fr.DUMP_EVENT
    assert header["level"] == "CRITICAL"
    assert header["rank"] == 0 and header["cohort"] == 1
    assert header["thread"] == "MainThread"
    assert header["data"]["trigger"] == "exception"
    assert header["data"]["flushed"] is True
    assert "RuntimeError: injected decode fault" in header["exc"]
    assert "_decode_step" in header["exc"]
    main = header["data"]["threads"][0]
    assert main["name"] == "MainThread"
    assert any("test_injected_exception" in line for line in main["stack"])
    # The last events, in order, the fatal record (with its traceback) last.
    assert _ticks(events) == list(range(30))
    assert events[-1]["level"] == "CRITICAL"
    assert "injected decode fault" in events[-1]["exc"]


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_uncaught_thread_exception_logs_and_dumps(tmp_path):
    _configure(tmp_path)
    recorder = fr.install()
    logger = ev.get_logger("batchgen.worker")
    ev.log_event(logger, "worker.test.tick", index=0)

    def _worker_thread():
        raise ValueError("thread fault")

    thread = threading.Thread(target=_worker_thread, name="h2d-worker")
    thread.start()
    thread.join(10)

    (path,) = sorted(recorder.path.parent.glob("*-dump*.jsonl"))
    header, events = _read_dump(path)
    assert header["data"]["trigger"] == "thread_exception"
    assert header["thread"] == "h2d-worker"
    assert "ValueError: thread fault" in header["exc"]
    assert header["data"]["threads"][0]["name"] == "h2d-worker"
    assert _ticks(events) == [0]
    assert events[-1]["msg"] == "Uncaught exception in thread h2d-worker"
    assert ev.flush()
    jsonl = tu.read_jsonl(ev.jsonl_path())
    assert any(rec["msg"].startswith("Flight recorder dump (thread_exception)")
               for rec in jsonl)


def test_dump_failure_does_not_mask_the_original_error(tmp_path, monkeypatch,
                                                       capsys):
    _configure(tmp_path)
    recorder = fr.install()

    def _broken(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(recorder, "_write_dump", _broken)
    with pytest.raises(RuntimeError, match="injected decode fault"):
        try:
            _decode_step()
        except RuntimeError as exc:
            assert fr.dump(fr.TRIGGER_EXCEPTION, "fatal", exc) is None
            raise
    assert "No space left on device" in capsys.readouterr().err


def test_a_wedged_dump_returns_within_its_timeout(tmp_path, monkeypatch):
    _configure(tmp_path)
    recorder = fr.install()
    release = threading.Event()
    monkeypatch.setattr(recorder, "_write_dump",
                        lambda *args: release.wait(30))
    started = time.monotonic()
    try:
        assert recorder.dump(fr.TRIGGER_HANG, "wedged fs", timeout_s=0.2) \
            is None
        assert time.monotonic() - started < 5
    finally:
        release.set()


def test_dumps_per_process_are_bounded(tmp_path):
    _configure(tmp_path)
    recorder = fr.install()
    paths = [recorder.dump(fr.TRIGGER_HANG, f"hang {n}")
             for n in range(fr.MAX_DUMPS_PER_PROCESS + 2)]
    assert all(p is not None for p in paths[:fr.MAX_DUMPS_PER_PROCESS])
    assert paths[fr.MAX_DUMPS_PER_PROCESS:] == [None, None]
    assert len(set(paths[:fr.MAX_DUMPS_PER_PROCESS])) == \
        fr.MAX_DUMPS_PER_PROCESS


def test_dump_without_a_recorder_is_a_no_op():
    assert fr.dump(fr.TRIGGER_EXCEPTION, "before install") is None


def test_install_requires_a_bound_jsonl_log(tmp_path):
    ev.configure("rank", node=0, rank=0, log_dir=tmp_path, console=False)
    with pytest.raises(fr.FlightRecorderError, match="bound to a run"):
        fr.install()


def test_install_points_faulthandler_at_the_ring(tmp_path, _isolated):
    _configure(tmp_path)
    recorder = fr.install()
    assert _isolated == [{"file": recorder.ring.fault_fd, "all_threads": True}]
    with pytest.raises(fr.FlightRecorderError, match="already installed"):
        fr.install()


# ---------------------------------------------------------- retention


def test_retention_keeps_the_newest_generations_of_this_rank(tmp_path):
    flight = _flight_dir(tmp_path)
    flight.mkdir(parents=True)
    now = time.time()
    for age, cohort in enumerate(range(10, 16)):
        for name in (f"rank0-cohort{cohort}-pid{100 + cohort}.ring",
                     f"rank0-cohort{cohort}-pid{100 + cohort}-dump1.jsonl"):
            (flight / name).write_text("x")
            os.utime(flight / name, (now - 100 + age, now - 100 + age))
    (flight / "rank1-cohort10-pid5.ring").write_text("other rank")
    (flight / "rank0-notes.txt").write_text("not ours")

    _configure(tmp_path, cohort=16)
    recorder = fr.install()

    rank0 = sorted(p.name for p in flight.glob("rank0-cohort*.ring"))
    assert rank0 == sorted([
        "rank0-cohort13-pid113.ring", "rank0-cohort14-pid114.ring",
        "rank0-cohort15-pid115.ring", recorder.path.name,
    ])
    assert len(rank0) == fr.KEEP_GENERATIONS
    assert len(list(flight.glob("rank0-cohort*-dump1.jsonl"))) == 3
    assert (flight / "rank1-cohort10-pid5.ring").exists()
    assert (flight / "rank0-notes.txt").exists()


# ------------------------------------------------ child-process triggers

_CHILD_PRELUDE = """
import faulthandler, logging, os, sys, time
sys.path.insert(0, {tests!r})
import eventlog_testutil as tu
ev = tu.import_eventlog()
fr = tu.import_flight_recorder()
ev.configure("rank", node=0, rank=0, local_rank=0, cohort={cohort},
             run_id=tu.RUN_ID, run_started_at=tu.RUN_STARTED_AT,
             log_dir={log_dir!r}, console=False)
recorder = fr.install()
logger = ev.get_logger("batchgen.worker")
for index in range({count}):
    ev.log_event(logger, "worker.test.tick", index=index, cohort={cohort})
"""

_CRASH_BODY = """
# The decode thread waits on the device (the GIL is released), then a
# native call faults.
time.sleep(0.3)

def _native_kernel_launch():
    faulthandler._sigsegv()

_native_kernel_launch()
"""

_HANG_BODY = """
wd = tu.import_under_stub_packages(
    "batchgen.server.watchdog", ("batchgen", "batchgen.server"))
watchdog = wd.Watchdog.create(
    debug_name="decode-0", watchdog_timeout=0.3, soft=True,
    dump_info=recorder.watchdog_dump_info("decode-0"))
watchdog.feed()

def _simulated_collective_hang():
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if list(recorder.path.parent.glob("*-dump1.jsonl")):
            return
        time.sleep(0.05)

_simulated_collective_hang()
time.sleep(0.5)  # let the watchdog log the dump path
ev.flush(10.0)
os._exit(0)
"""


def _run_child(tmp_path: Path, body: str, *, cohort: int = 1,
               count: int = 40) -> subprocess.CompletedProcess:
    script = _CHILD_PRELUDE.format(
        tests=str(tu.TESTS_DIR), log_dir=str(tmp_path), cohort=cohort,
        count=count,
    ) + body
    env = dict(os.environ)
    env.pop("PYTHONFAULTHANDLER", None)
    return subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=60, env=env)


def _crash_signals() -> set[int]:
    return {-signal.SIGSEGV, -signal.SIGBUS}


def test_native_crash_leaves_decodable_events_and_thread_stacks(tmp_path):
    done = _run_child(tmp_path, _CRASH_BODY)
    assert done.returncode in _crash_signals(), done.stderr

    (ring_path,) = _flight_dir(tmp_path).glob("rank0-cohort1-pid*.ring")
    ring = fr.read_ring(ring_path)
    assert _ticks(ring.events) == list(range(40))
    assert ring.torn == 0
    assert "Fatal Python error" in ring.fatal_text
    assert "most recent call first" in ring.fatal_text
    assert "_native_kernel_launch" in ring.fatal_text
    # No Python ran after the fault: there is no dump file.
    assert not list(_flight_dir(tmp_path).glob("*-dump*.jsonl"))


def test_watchdog_hang_dumps_the_hung_threads_stack(tmp_path):
    done = _run_child(tmp_path, _HANG_BODY)
    assert done.returncode == 0, done.stderr

    (path,) = _flight_dir(tmp_path).glob("rank0-cohort1-pid*-dump1.jsonl")
    header, events = _read_dump(path)
    assert header["data"]["trigger"] == "hang"
    assert header["data"]["reason"] == "decode-0 watchdog timeout"
    assert header["thread"] != "MainThread"  # the watchdog's own thread
    stacks = {t["name"]: t["stack"] for t in header["data"]["threads"]}
    assert any("_simulated_collective_hang" in line
               for line in stacks["MainThread"])
    assert _ticks(events) == list(range(40))
    # The watchdog logged the dump as its debug info.
    jsonl = tu.read_jsonl(tmp_path / NODE_DIR / "rank0.jsonl")
    assert any(f"flight recorder dump: {path}" in rec["msg"] for rec in jsonl)


def test_two_generations_of_a_rank_keep_separate_evidence(tmp_path):
    first = _run_child(tmp_path, _CRASH_BODY, cohort=1, count=5)
    second = _run_child(tmp_path, _CRASH_BODY, cohort=2, count=7)
    assert first.returncode in _crash_signals(), first.stderr
    assert second.returncode in _crash_signals(), second.stderr

    rings = {
        path.name.split("-")[1]: fr.read_ring(path)
        for path in _flight_dir(tmp_path).glob("rank0-cohort*-pid*.ring")
    }
    assert sorted(rings) == ["cohort1", "cohort2"]
    assert _ticks(rings["cohort1"].events) == list(range(5))
    assert _ticks(rings["cohort2"].events) == list(range(7))
    assert {e["data"]["cohort"] for e in rings["cohort2"].events
            if e["event"] == "worker.test.tick"} == {2}
    for ring in rings.values():
        assert "_native_kernel_launch" in ring.fatal_text


# ------------------------------------------------- worker entry wiring


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(MAIN_LOOP.read_text(), filename=str(MAIN_LOOP))
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def test_every_rank_installs_the_recorder_right_after_logging():
    impl = _function("_server_worker_main_impl")
    calls = [ast.unparse(node) for node in impl.body
             if isinstance(node, (ast.Expr, ast.Assign))]
    configure = next(i for i, text in enumerate(calls)
                     if "eventlog_process.configure_process(" in text)
    assert calls[configure + 1] == "recorder = flight_recorder.install()"


def test_both_worker_watchdogs_dump_on_timeout():
    impl = ast.unparse(_function("_server_worker_main_impl"))
    assert impl.count("Watchdog.create(") == 2
    assert "dump_info=recorder.watchdog_dump_info(f'worker-{args.global_rank}')" in impl
    assert "dump_info=recorder.watchdog_dump_info(f'decode-{args.global_rank}')" in impl


def test_shutdown_self_exit_dumps_before_the_log_flush():
    fn = ast.unparse(_function("_start_shutdown_abort_watchdog"))
    dump = fn.index("flight_recorder.dump(flight_recorder.TRIGGER_HANG")
    assert dump < fn.index("flush_logs_before_exit()") < fn.index(
        "self_exit(WORKER_SELF_EXIT_CODE)")


def test_worker_fatal_boundary_dumps_before_the_fatal_leaves_the_process():
    calls = []

    def _impl(*args):
        raise RuntimeError("injected worker fault")

    namespace = {
        "_server_worker_main_impl": _impl,
        "logging": SimpleNamespace(
            critical=lambda *a, **k: calls.append("log_critical")),
        "traceback": SimpleNamespace(format_exc=lambda: "tb"),
        "_format_worker_fatal": lambda *a: "message",
        "dist": SimpleNamespace(is_available=lambda: True,
                                is_initialized=lambda: True),
        "_publish_worker_fatal_to_store": lambda m: calls.append("publish"),
        "_send_worker_fatal_to_local_server":
            lambda *a: calls.append("send_local"),
        "flight_recorder": SimpleNamespace(
            TRIGGER_EXCEPTION=fr.TRIGGER_EXCEPTION,
            dump=lambda trigger, reason, exc: calls.append(
                ("dump", trigger, str(exc)))),
        "eventlog_process": SimpleNamespace(
            exit_process=lambda code, reason: calls.append(("exit", code))),
    }
    module = ast.Module(body=[_function("server_worker_main")],
                        type_ignores=[])
    exec(compile(module, str(MAIN_LOOP), "exec"), namespace)
    args = SimpleNamespace(global_rank=3)
    namespace["server_worker_main"](0, None, None, args)

    assert calls == [
        "log_critical",
        ("dump", "exception", "injected worker fault"),
        "publish",
        "send_local",
        ("exit", 1),
    ]
