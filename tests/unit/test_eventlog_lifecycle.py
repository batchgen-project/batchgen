"""``batchgen.eventlog`` lifecycle: configure/shutdown, flush, deferred run
identity with its bounded buffer, and ``IntervalGate``.

CPU only; no CUDA, no torch.
"""

from __future__ import annotations

import io
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()


@pytest.fixture(autouse=True)
def _isolated():
    with tu.isolated_logging(ev):
        yield


def _configure_server(tmp_path: Path, **kwargs) -> None:
    kwargs.setdefault("console", False)
    ev.configure("server", node=0, run_id=tu.RUN_ID,
                 run_started_at=tu.RUN_STARTED_AT, log_dir=tmp_path,
                 **kwargs)


# --------------------------------------------------------------------------- #
#  (6) configure / shutdown
# --------------------------------------------------------------------------- #
def test_configure_twice_leaves_exactly_one_root_handler(tmp_path):
    root = logging.getLogger()
    stray = logging.StreamHandler(io.StringIO())
    root.addHandler(stray)

    _configure_server(tmp_path)
    first = root.handlers[0]
    assert root.handlers == [first]
    assert isinstance(first, ev.handlers.EventQueueHandler)

    _configure_server(tmp_path, level="DEBUG")
    assert len(root.handlers) == 1
    assert root.handlers[0] is not first
    assert root.level == logging.DEBUG


def test_reconfigure_keeps_writing_to_the_same_file(tmp_path):
    _configure_server(tmp_path)
    logger = ev.get_logger("batchgen.server")
    logger.info("one")
    _configure_server(tmp_path)
    logger.info("two")
    assert ev.flush()
    msgs = [rec["msg"] for rec in tu.read_jsonl(ev.jsonl_path())]
    assert msgs == ["one", "two"]


def test_shutdown_is_idempotent_and_safe_unconfigured(tmp_path):
    ev.shutdown()  # never configured in this test
    _configure_server(tmp_path)
    ev.shutdown()
    ev.shutdown()
    assert ev.jsonl_path() is None
    assert ev.flush() is True  # nothing pending


def test_shutdown_drains_pending_records(tmp_path):
    _configure_server(tmp_path)
    path = ev.jsonl_path()
    logger = ev.get_logger("batchgen.server")
    for index in range(200):
        logger.info("record %d", index)
    ev.shutdown()
    assert len(tu.read_jsonl(path)) == 200
    assert logging.getLogger().handlers == []


@pytest.mark.parametrize("args,kwargs", [
    (("worker",), {"node": 0}),
    (("server",), {"node": -1}),
    (("server",), {"node": 0, "run_id": "7F2A" * 8,
                   "run_started_at": 0.0}),
    (("server",), {"node": 0, "run_id": "abc", "run_started_at": 0.0}),
    (("server",), {"node": 0, "run_id": tu.RUN_ID}),
    (("server",), {"node": 0, "run_started_at": 0.0}),
    (("server",), {"node": 0, "run_id": tu.RUN_ID,
                   "run_started_at": datetime(2026, 10, 27)}),
    (("rank",), {"node": 0}),
    (("rank",), {"node": 0, "rank": -1}),
    (("server",), {"node": 0, "rank": 0}),
    (("paramserver",), {"node": 0, "local_rank": 0}),
    (("server",), {"node": 0, "level": "LOUD"}),
])
def test_configure_rejects_bad_arguments(tmp_path, args, kwargs):
    _configure_server(tmp_path)
    installed = list(logging.getLogger().handlers)
    with pytest.raises(ValueError):
        ev.configure(*args, log_dir=tmp_path, console=False, **kwargs)
    # A rejected call leaves the previous setup in place.
    assert logging.getLogger().handlers == installed
    assert ev.jsonl_path() is not None


def test_warnings_are_captured(tmp_path):
    _configure_server(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("deprecated knob", UserWarning)
    assert ev.flush()
    (rec,) = tu.read_jsonl(ev.jsonl_path())
    assert rec["logger"] == "py.warnings"
    assert "deprecated knob" in rec["msg"]


# --------------------------------------------------------------------------- #
#  (7) flush
# --------------------------------------------------------------------------- #
def test_flush_puts_records_on_disk(tmp_path):
    _configure_server(tmp_path)
    ev.get_logger("batchgen.server").critical("on disk")
    assert ev.flush() is True
    # Read through an independent file object.
    with open(ev.jsonl_path(), encoding="utf-8") as fh:
        assert '"msg":"on disk"' in fh.read()


def test_flush_returns_false_without_hanging_when_listener_stopped(tmp_path):
    _configure_server(tmp_path)
    assert ev._state.listener.stop()
    started = time.monotonic()
    assert ev.flush(timeout_s=5.0) is False
    assert time.monotonic() - started < 1.0


def test_flush_from_the_listener_thread_does_not_deadlock(tmp_path):
    results = []

    def renderer(data):
        results.append(ev.flush(timeout_s=5.0))
        return "rendered"

    ev.register_console_renderer("test.reentrant", renderer)
    _configure_server(tmp_path)
    started = time.monotonic()
    ev.log_event(ev.get_logger("batchgen.t"), "test.reentrant")
    assert ev.flush()
    assert results == [False]
    assert time.monotonic() - started < 1.0


# --------------------------------------------------------------------------- #
#  (8) Deferred run identity
# --------------------------------------------------------------------------- #
def test_records_before_bind_run_are_written_in_order(tmp_path):
    stream = io.StringIO()
    ev.configure("rank", node=1, rank=8, log_dir=tmp_path,
                 console_stream=stream)
    logger = ev.get_logger("batchgen.worker")
    for index in range(5):
        logger.info("early %d", index)
    assert ev.flush()
    assert ev.jsonl_path() is None
    assert not (tmp_path / tu.RUN_DIR).exists()
    # The console does not wait for the run identity.
    assert stream.getvalue().count("worker: early") == 5

    path = ev.bind_run(tu.RUN_ID, tu.RUN_STARTED_AT)
    assert path == tmp_path / tu.RUN_DIR / "node1" / "rank8.jsonl"
    for index in range(2):
        logger.info("late %d", index)
    assert ev.flush()

    recs = tu.read_jsonl(path)
    assert [rec["msg"] for rec in recs] == [
        "early 0", "early 1", "early 2", "early 3", "early 4",
        "late 0", "late 1",
    ]
    assert {rec["run_id"] for rec in recs} == {tu.RUN_ID}
    assert {rec["rank"] for rec in recs} == {8}


def test_bind_run_rules(tmp_path):
    with pytest.raises(RuntimeError):
        ev.bind_run(tu.RUN_ID, tu.RUN_STARTED_AT)  # not configured
    ev.configure("server", node=0, log_dir=tmp_path, console=False)
    with pytest.raises(ValueError):
        ev.bind_run("not-a-run-id", tu.RUN_STARTED_AT)
    path = ev.bind_run(tu.RUN_ID, tu.RUN_STARTED_AT)
    assert ev.bind_run(tu.RUN_ID, tu.RUN_STARTED_AT) == path
    with pytest.raises(RuntimeError):
        ev.bind_run("0" * 32, tu.RUN_STARTED_AT)


def test_bind_run_without_jsonl_returns_none(tmp_path):
    ev.configure("server", node=0, log_dir=tmp_path, console=False,
                 jsonl=False)
    assert ev.bind_run(tu.RUN_ID, tu.RUN_STARTED_AT) is None
    assert ev.jsonl_path() is None
    assert not any(tmp_path.iterdir())


def test_buffer_overflow_is_reported_first(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "DEFAULT_BUFFER_LIMIT", 3)
    ev.configure("server", node=0, log_dir=tmp_path, console=False)
    logger = ev.get_logger("batchgen.server")
    for index in range(5):
        logger.info("early %d", index)
    path = ev.bind_run(tu.RUN_ID, tu.RUN_STARTED_AT)
    logger.info("late")
    assert ev.flush()

    recs = tu.read_jsonl(path)
    overflow = recs[0]
    assert overflow["event"] == "eventlog.buffer.overflow"
    assert overflow["level"] == "WARNING"
    assert overflow["data"] == {"dropped": 2, "buffer_limit": 3}
    assert overflow["run_id"] == tu.RUN_ID
    assert [rec["msg"] for rec in recs[1:]] == [
        "early 2", "early 3", "early 4", "late"]


# --------------------------------------------------------------------------- #
#  (12) IntervalGate
# --------------------------------------------------------------------------- #
class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_interval_gate_fires_at_most_once_per_interval():
    clock = _Clock()
    gate = ev.counters.IntervalGate(10.0, clock=clock)
    assert gate.due() is False  # not before one interval has elapsed
    clock.now = 109.9
    assert gate.due() is False
    clock.now = 110.0
    assert gate.due() is True
    assert gate.due() is False
    clock.now = 119.9
    assert gate.due() is False
    clock.now = 135.0  # a long gap fires once, not once per missed interval
    assert gate.due() is True
    assert gate.due() is False
    clock.now = 145.0
    assert gate.due() is True


def test_interval_gate_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        ev.counters.IntervalGate(0)
