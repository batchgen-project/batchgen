"""``batchgen.eventlog`` console rendering: the approved template lines,
labels, colour, and tracebacks in both sinks.

CPU only; no CUDA, no torch.
"""

from __future__ import annotations

import io
import logging
import queue
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()
schema = ev.schema
ProcessInfo = schema.ProcessInfo

SERVER0 = ProcessInfo(proc="server", node=0)
RANK0 = ProcessInfo(proc="rank", node=0, rank=0)
RANK11 = ProcessInfo(proc="rank", node=1, rank=11)


@pytest.fixture(autouse=True)
def _isolated():
    with tu.isolated_logging(ev):
        yield


def _at(hour: int, minute: int, second: int, millis: int) -> float:
    return datetime(2026, 10, 27, hour, minute, second, millis * 1000,
                    tzinfo=timezone.utc).timestamp()


def _record(name, level, msg, args=(), *, created=None, pathname="x.py",
            lineno=1, exc_info=None, ctx=None, event=None, data=None):
    """A record as the listener sees it: prepared by the queue handler."""
    record = logging.LogRecord(name, level, pathname, lineno, msg, args,
                               exc_info)
    if created is not None:
        record.created = created
    for key, value in (ctx or {}).items():
        setattr(record, f"bg_{key}", value)
    if event is not None:
        setattr(record, schema.ATTR_EVENT, event)
        setattr(record, schema.ATTR_DATA, data or {})
    return ev.handlers.EventQueueHandler(queue.SimpleQueue()).prepare(record)


def _render(info, record, stream=None):
    return ev.console.ConsoleFormatter(info, stream).format(record)


# --------------------------------------------------------------------------- #
#  (4) The approved template
# --------------------------------------------------------------------------- #
def test_template_line_server_start():
    record = _record("batchgen.server", logging.INFO,
                     "started run %s on port %d",
                     ("20261027-140205-7f2a9c1e", 8000),
                     created=_at(14, 2, 5, 1))
    assert _render(SERVER0, record) == (
        "2026-10-27 14:02:05.001Z INFO  [server node=0] server: "
        "started run 20261027-140205-7f2a9c1e on port 8000"
    )


def test_template_line_event_with_renderer():
    ev.register_console_renderer(
        "batch.progress",
        lambda d: f"batch progress: {d['done']}/{d['total']} sequences",
    )
    record = _record("batchgen.scheduler", logging.INFO, "batch.progress",
                     created=_at(14, 2, 11, 482),
                     ctx={"batch_id": "batch_7f2a", "wave": 2},
                     event="batch.progress", data={"done": 512, "total": 2048})
    assert _render(RANK0, record) == (
        "2026-10-27 14:02:11.482Z INFO  [rank=0 node=0] "
        "[batch=batch_7f2a wave=2] batch progress: 512/2048 sequences"
    )


def test_template_line_warning_with_location():
    record = _record(
        "batchgen.kv_cache.host", logging.WARNING,
        "free host-KV pages below plan margin (free=%d, planned=%d)",
        (212, 256), created=_at(14, 2, 11, 517),
        pathname="/src/batchgen/kv_cache/host_paged.py", lineno=812,
        ctx={"batch_id": "batch_7f2a", "wave": 2, "step": 1840})
    assert _render(RANK11, record) == (
        "2026-10-27 14:02:11.517Z WARN  [rank=11 node=1] "
        "[batch=batch_7f2a wave=2 step=1840] kv_cache.host: "
        "free host-KV pages below plan margin (free=212, planned=256)"
        "  at host_paged.py:812"
    )


def _gpu_failure_exc_info():
    try:
        raise RuntimeError("GPU page table is not initialized")
    except RuntimeError:
        return sys.exc_info()


def test_template_line_error_with_traceback():
    record = _record(
        "batchgen.worker", logging.ERROR, "decode step failed",
        created=_at(14, 2, 12, 3),
        pathname="/src/batchgen/worker/batchgen_worker.py", lineno=19021,
        exc_info=_gpu_failure_exc_info(),
        ctx={"batch_id": "batch_7f2a", "wave": 2, "step": 1841})
    lines = _render(RANK11, record).split("\n")
    assert lines[0] == (
        "2026-10-27 14:02:12.003Z ERROR [rank=11 node=1] "
        "[batch=batch_7f2a wave=2 step=1841] worker: decode step failed"
        "  at batchgen_worker.py:19021"
    )
    assert lines[1] == "    Traceback (most recent call last):"
    assert lines[-1] == "    RuntimeError: GPU page table is not initialized"
    assert all(line.startswith("    ") for line in lines[1:])


def test_restart_suffix_only_for_positive_cohort():
    record = _record("batchgen.worker", logging.INFO, "hi",
                     created=_at(0, 0, 0, 0))
    restarted = ProcessInfo(proc="rank", node=1, rank=11, cohort=2)
    first = ProcessInfo(proc="rank", node=1, rank=11, cohort=0)
    param = ProcessInfo(proc="paramserver", node=3, cohort=1)
    assert "[rank=11 node=1 restart=2] worker: hi" in _render(restarted,
                                                              record)
    assert "[rank=11 node=1] worker: hi" in _render(first, record)
    assert "[param-server node=3 restart=1] worker: hi" in _render(param,
                                                                   record)


def test_empty_context_block_is_omitted():
    record = _record("batchgen.server", logging.INFO, "hi",
                     created=_at(0, 0, 0, 0))
    assert _render(SERVER0, record) == (
        "2026-10-27 00:00:00.000Z INFO  [server node=0] server: hi")


@pytest.mark.parametrize("level,label", [
    (logging.DEBUG, "DEBUG"), (logging.INFO, "INFO "),
    (logging.WARNING, "WARN "), (logging.ERROR, "ERROR"),
    (logging.CRITICAL, "CRIT "),
])
def test_level_labels(level, label):
    record = _record("batchgen.server", level, "hi",
                     created=_at(0, 0, 0, 0))
    line = _render(SERVER0, record)
    assert line.startswith(f"2026-10-27 00:00:00.000Z {label} [server")


@pytest.mark.parametrize("name,component", [
    ("batchgen.server.http", "server.http"),
    ("uvicorn.error", "uvicorn.error"),
    ("root", "root"),
])
def test_component_strips_batchgen_prefix(name, component):
    record = _record(name, logging.INFO, "hi", created=_at(0, 0, 0, 0))
    assert _render(SERVER0, record).endswith(f"] {component}: hi")


def test_default_event_body_quotes_strings_with_spaces():
    record = _record("batchgen.t", logging.INFO, "kv_cache.host.low",
                     created=_at(0, 0, 0, 0), event="kv_cache.host.low",
                     data={"free": 212, "note": "two words", "flag": True})
    assert _render(SERVER0, record).endswith(
        '] kv_cache.host.low free=212 note="two words" flag=True')


def test_failing_renderer_falls_back_to_default_body():
    ev.register_console_renderer("test.broken", lambda d: d["missing"])
    record = _record("batchgen.t", logging.INFO, "test.broken",
                     created=_at(0, 0, 0, 0), event="test.broken",
                     data={"n": 1})
    assert _render(SERVER0, record).endswith("] test.broken n=1")


def test_formatter_never_raises():
    record = logging.LogRecord("batchgen.t", logging.INFO, "x.py", 1,
                               "%d items", ("not a number",), None)
    assert _render(SERVER0, record) == repr(record)


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_colour_only_on_tty_without_no_color(monkeypatch):
    record = _record("batchgen.server", logging.ERROR, "hi",
                     created=_at(0, 0, 0, 0))
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert "\x1b[" not in _render(SERVER0, record, io.StringIO())
    coloured = _render(SERVER0, record, _FakeTTY())
    assert coloured.startswith("2026-10-27 00:00:00.000Z \x1b[")
    assert "ERROR\x1b[0m [server node=0] server: hi" in coloured
    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b[" not in _render(SERVER0, record, _FakeTTY())


def test_register_renderer_validates_arguments():
    with pytest.raises(ValueError):
        ev.register_console_renderer("Bad Name", str)
    with pytest.raises(TypeError):
        ev.register_console_renderer("test.ok", "not callable")


# --------------------------------------------------------------------------- #
#  (5) Exceptions through both sinks
# --------------------------------------------------------------------------- #
def test_logger_exception_reaches_both_sinks(tmp_path):
    stream = io.StringIO()
    ev.configure("rank", node=1, rank=11, run_id=tu.RUN_ID,
                 run_started_at=tu.RUN_STARTED_AT, log_dir=tmp_path,
                 console_stream=stream)
    logger = ev.get_logger("batchgen.worker")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("decode step failed")
    assert ev.flush()

    (rec,) = tu.read_jsonl(ev.jsonl_path())
    assert rec["level"] == "ERROR"
    assert rec["msg"] == "decode step failed"
    assert rec["exc"].startswith("Traceback (most recent call last):")
    assert rec["exc"].rstrip().endswith("ZeroDivisionError: division by zero")

    lines = stream.getvalue().rstrip("\n").split("\n")
    assert lines[0].endswith("] worker: decode step failed  at "
                             f"test_eventlog_console.py:{rec['line']}")
    assert lines[1] == "    Traceback (most recent call last):"
    assert lines[-1] == "    ZeroDivisionError: division by zero"


def test_stack_info_is_kept_as_exc(tmp_path):
    ev.configure("server", node=0, run_id=tu.RUN_ID,
                 run_started_at=tu.RUN_STARTED_AT, log_dir=tmp_path,
                 console=False)
    ev.get_logger("batchgen.server").info("where", stack_info=True)
    assert ev.flush()
    (rec,) = tu.read_jsonl(ev.jsonl_path())
    assert rec["msg"] == "where"
    assert rec["exc"].startswith("Stack (most recent call last):")
