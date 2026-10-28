"""Robustness cases for batchgen.eventlog: payload snapshots, strict JSON,
console quoting, failure rollback, fork detach, and concurrent producers.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()


@pytest.fixture(autouse=True)
def _isolated():
    with tu.isolated_logging(ev):
        yield


def _configure(tmp_path: Path, **kwargs) -> None:
    kwargs.setdefault("console", False)
    ev.configure("server", node=0, run_id=tu.RUN_ID,
                 run_started_at=tu.RUN_STARTED_AT, log_dir=tmp_path,
                 **kwargs)


def _records() -> list[dict]:
    assert ev.flush()
    return tu.read_jsonl(ev.jsonl_path())


def test_payload_is_snapshotted_on_the_caller_thread(tmp_path):
    _configure(tmp_path)
    logger = ev.get_logger("batchgen.t")
    stats = {"done": 1}
    seen = [1]
    ev.log_event(logger, "batch.progress", stats=stats, seen=seen)
    stats["done"] = 2
    seen.append(2)
    (rec,) = _records()
    assert rec["data"] == {"stats": {"done": 1}, "seen": [1]}


def test_nested_nan_and_large_repr_stay_strict_json(tmp_path):
    _configure(tmp_path)
    logger = ev.get_logger("batchgen.t")

    class Big:
        def __repr__(self) -> str:
            return "x" * 5000

    ev.log_event(logger, "probe.values",
                 nested={"a": {"b": math.nan}, "ok": 1}, big=Big())
    (rec,) = _records()
    assert rec["data"]["nested"] == {"a": {"b": "nan"}, "ok": 1}
    assert rec["data"]["big"].endswith("...<truncated>")
    assert len(rec["data"]["big"]) < 1100


def test_lone_surrogate_does_not_lose_the_record(tmp_path):
    _configure(tmp_path)
    ev.get_logger("batchgen.t").info("bad \udcff text")
    (rec,) = _records()
    assert rec["msg"].startswith("bad ")


@pytest.mark.parametrize(
    "value, rendered",
    [
        ("", '""'),
        ("a\tb", '"a\\tb"'),
        ("a\nb", '"a\\nb"'),
        ("k=v", '"k=v"'),
        ('say "hi"', '"say \\"hi\\""'),
        ("plain", "plain"),
    ],
)
def test_default_event_body_quotes_ambiguous_strings(value, rendered):
    assert ev.console._default_body("probe.values", {"v": value}) == (
        f"probe.values v={rendered}"
    )


def test_renderer_returning_none_falls_back_to_default(tmp_path):
    ev.register_console_renderer("probe.values", lambda data: None)
    _configure(tmp_path)
    ev.log_event(ev.get_logger("batchgen.t"), "probe.values", n=3)
    (rec,) = _records()
    assert rec["msg"] == "probe.values n=3"


def test_configure_rolls_back_when_bind_fails(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    with pytest.raises(OSError):
        _configure(blocker)
    assert ev.jsonl_path() is None
    assert not any(
        isinstance(h, ev.handlers.EventQueueHandler)
        for h in logging.getLogger().handlers
    )


def test_listener_survives_a_failing_sink(tmp_path, capsys):
    class Boom(logging.Handler):
        def emit(self, record):
            raise RuntimeError("sink failure")

        def handle(self, record):
            raise RuntimeError("sink failure")

    _configure(tmp_path)
    listener = ev._state.listener
    listener.handlers = (Boom(),) + tuple(listener.handlers)
    logger = ev.get_logger("batchgen.t")
    logger.info("first")
    assert ev.flush()
    assert listener.alive
    assert "listener error" in capsys.readouterr().err


def test_shutdown_reports_a_listener_that_does_not_stop(tmp_path, capsys,
                                                         monkeypatch):
    _configure(tmp_path)
    listener = ev._state.listener
    monkeypatch.setattr(listener, "stop", lambda timeout_s=0: False)
    ev.shutdown()
    assert "did not stop" in capsys.readouterr().err


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_detaches_from_the_dead_listener(tmp_path):
    _configure(tmp_path)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            attached = any(
                isinstance(h, ev.handlers.EventQueueHandler)
                for h in logging.getLogger().handlers
            )
            os.write(write_fd, b"1" if attached or ev._state else b"0")
        finally:
            os._exit(0)
    os.close(write_fd)
    result = os.read(read_fd, 1)
    os.close(read_fd)
    os.waitpid(pid, 0)
    assert result == b"0"
    assert ev._state is not None  # parent keeps its event log


def test_concurrent_producers_lose_nothing_and_keep_order(tmp_path):
    _configure(tmp_path)
    logger = ev.get_logger("batchgen.t")
    threads, per_thread = 8, 500

    def produce(t: int) -> None:
        for i in range(per_thread):
            ev.log_event(logger, "probe.seq", t=t, i=i)

    workers = [threading.Thread(target=produce, args=(t,))
               for t in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    recs = _records()
    assert len(recs) == threads * per_thread
    for t in range(threads):
        seq = [r["data"]["i"] for r in recs if r["data"]["t"] == t]
        assert seq == list(range(per_thread))


def test_disabled_library_logger_records_are_not_enqueued(tmp_path):
    _configure(tmp_path, level="INFO")
    lib = logging.getLogger("thirdparty.chatty")
    lib.setLevel(logging.DEBUG)
    try:
        lib.debug("noise")
        lib.info("kept")
        msgs = [r["msg"] for r in _records()]
        assert msgs == ["kept"]
    finally:
        lib.setLevel(logging.NOTSET)


def test_fixture_does_not_accumulate_root_handlers():
    # Runs after many configure()/shutdown() cycles in this module: only
    # pytest's own capture handlers for the current phase may be attached.
    handlers = logging.getLogger().handlers
    assert not [h for h in handlers
                if isinstance(h, ev.handlers.EventQueueHandler)]
    assert len([h for h in handlers
                if type(h).__module__ == "_pytest.logging"]) <= 2


def test_json_lines_parse_independently(tmp_path):
    _configure(tmp_path)
    ev.get_logger("batchgen.t").info("multi\nline")
    assert ev.flush()
    lines = ev.jsonl_path().read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["msg"] == "multi\nline"
