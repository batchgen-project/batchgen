"""``batchgen.eventlog`` schema v1: the JSONL envelope, event names and the
file layout.

CPU only; no CUDA, no torch.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()
schema = ev.schema

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


@pytest.fixture(autouse=True)
def _isolated():
    with tu.isolated_logging(ev):
        yield


def _configure_rank(tmp_path: Path, **kwargs) -> None:
    kwargs.setdefault("console", False)
    ev.configure(
        "rank",
        node=1,
        rank=3,
        local_rank=1,
        cohort=2,
        run_id=tu.RUN_ID,
        run_started_at=tu.RUN_STARTED_AT,
        log_dir=tmp_path,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
#  (1) JSONL round trip
# --------------------------------------------------------------------------- #
def test_event_record_carries_every_envelope_field(tmp_path):
    _configure_rank(tmp_path)
    logger = ev.get_logger("batchgen.kv_cache.host")
    warn = logging.WARNING
    payload = {"free": 212, "planned": 256}
    with ev.bind(batch_id="batch_7f2a", wave=2, step=1840):
        line = sys._getframe().f_lineno + 1
        ev.log_event(logger, "kv_cache.host.low", level=warn, **payload)
    assert ev.flush()

    (rec,) = tu.read_jsonl(ev.jsonl_path())
    assert set(rec) == set(schema.ENVELOPE_FIELDS) - {schema.F_EXC}
    assert rec["v"] == schema.SCHEMA_VERSION == 1
    assert TS_RE.match(rec["ts"])
    assert rec["level"] == "WARNING"
    assert rec["event"] == "kv_cache.host.low"
    assert rec["msg"] == "kv_cache.host.low free=212 planned=256"
    assert rec["logger"] == "batchgen.kv_cache.host"
    assert rec["src"] == "py"
    assert rec["proc"] == "rank"
    assert rec["pid"] == os.getpid()
    assert rec["thread"] == threading.current_thread().name
    assert rec["run_id"] == tu.RUN_ID
    assert rec["node"] == 1
    assert rec["rank"] == 3
    assert rec["local_rank"] == 1
    assert rec["cohort"] == 2
    assert rec["batch_id"] == "batch_7f2a"
    assert rec["wave"] == 2
    assert rec["step"] == 1840
    assert rec["file"] == "test_eventlog_schema.py"
    assert rec["line"] == line
    assert rec["data"] == {"free": 212, "planned": 256}


def test_plain_log_omits_none_fields(tmp_path):
    ev.configure("server", node=0, run_id=tu.RUN_ID,
                 run_started_at=tu.RUN_STARTED_AT, log_dir=tmp_path,
                 console=False)
    ev.get_logger("batchgen.server").info("listening on port %d", 8000)
    assert ev.flush()

    (rec,) = tu.read_jsonl(ev.jsonl_path())
    assert rec["event"] == schema.TEXT_EVENT == "log.text"
    assert rec["msg"] == "listening on port 8000"
    assert rec["proc"] == "server"
    assert rec["node"] == 0
    for absent in ("rank", "local_rank", "cohort", "batch_id", "wave",
                   "step", "data", "exc"):
        assert absent not in rec
    assert None not in rec.values()


def test_non_json_payload_values_fall_back_to_repr(tmp_path):
    _configure_rank(tmp_path)
    marker = object()
    ev.log_event(ev.get_logger("batchgen.t"), "test.payload", obj=marker,
                 ratio=float("nan"), ok=1)
    assert ev.flush()

    path = ev.jsonl_path()
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text  # strict JSON
    (rec,) = tu.read_jsonl(path)
    assert rec["data"]["obj"] == repr(marker)
    assert rec["data"]["ratio"] == "nan"
    assert rec["data"]["ok"] == 1


def test_format_ts_is_utc_rfc3339_with_microseconds():
    created = datetime(2026, 10, 27, 14, 2, 5, 1000,
                       tzinfo=timezone.utc).timestamp()
    assert schema.format_ts(created) == "2026-10-27T14:02:05.001000Z"


# --------------------------------------------------------------------------- #
#  (2) Event names and the disabled-level fast path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [
    "kv_cache.host", "batch.progress", "worker.decode_step.failed",
    "a.b.c.d", "server.http2.ready",
])
def test_valid_event_names(name):
    assert schema.validate_event_name(name) == name
    assert schema.EVENT_NAME_RE.fullmatch(name)


@pytest.mark.parametrize("name", [
    "single", "a.b.c.d.e", "Kv.host", "kv-cache.host", "1a.b", "a..b",
    "a.b.", ".a.b", "a.B", "a._b", "", None, 3,
])
def test_invalid_event_names(name):
    with pytest.raises(ValueError):
        schema.validate_event_name(name)


@pytest.mark.parametrize("name", ["log.text", "eventlog.buffer.overflow"])
def test_reserved_prefixes(name):
    with pytest.raises(ValueError, match="reserved"):
        schema.validate_event_name(name)
    assert schema.validate_event_name(name, allow_reserved=True) == name


def test_log_event_rejects_bad_names_when_enabled(tmp_path):
    _configure_rank(tmp_path)
    logger = ev.get_logger("batchgen.t")
    with pytest.raises(ValueError):
        ev.log_event(logger, "NotAnEvent")
    with pytest.raises(ValueError):
        ev.log_event(logger, "log.text")


def test_disabled_level_skips_rendering_and_writing(tmp_path):
    calls = []

    def renderer(data):
        calls.append(dict(data))
        return f"rendered {data['n']}"

    ev.register_console_renderer("test.render", renderer)
    stream = io.StringIO()
    _configure_rank(tmp_path, console_stream=stream, console=True)
    logger = ev.get_logger("batchgen.t")

    ev.log_event(logger, "test.render", level=logging.DEBUG, n=1)
    # The level check comes first: a disabled call does no work at all.
    ev.log_event(logger, "Not A Name", level=logging.DEBUG)
    assert ev.flush()
    assert calls == []
    assert tu.read_jsonl(ev.jsonl_path()) == []
    assert stream.getvalue() == ""

    ev.log_event(logger, "test.render", n=2)
    assert ev.flush()
    # Rendered once and shared by both sinks.
    assert calls == [{"n": 2}]
    (rec,) = tu.read_jsonl(ev.jsonl_path())
    assert rec["msg"] == "rendered 2"
    assert stream.getvalue().rstrip("\n").endswith("] rendered 2")


# --------------------------------------------------------------------------- #
#  (11) File layout and run_dir naming
# --------------------------------------------------------------------------- #
def test_run_dir_uses_utc_start_and_run_id_prefix():
    float_start = tu.RUN_STARTED_AT.timestamp()
    cst = timezone(timedelta(hours=8))
    aware_cst = datetime(2026, 10, 27, 22, 2, 5, tzinfo=cst)
    for start in (float_start, tu.RUN_STARTED_AT, aware_cst):
        assert ev.handlers.run_dir_name(tu.RUN_ID, start) == tu.RUN_DIR
    with pytest.raises(ValueError):
        ev.handlers.run_dir_name(tu.RUN_ID, datetime(2026, 10, 27))


@pytest.mark.parametrize("proc,extra,relpath", [
    ("server", {}, "node2/server.jsonl"),
    ("paramserver", {}, "node2/paramserver.jsonl"),
    ("rank", {"rank": 13, "local_rank": 5}, "node2/rank13.jsonl"),
])
def test_jsonl_path_layout(tmp_path, proc, extra, relpath):
    ev.configure(proc, node=2, log_dir=tmp_path, console=False, **extra)
    assert ev.jsonl_path() is None
    path = ev.bind_run(tu.RUN_ID, tu.RUN_STARTED_AT.timestamp())
    assert path == tmp_path / tu.RUN_DIR / relpath
    assert ev.jsonl_path() == path
    assert path.is_file()


def test_default_log_dir_resolves_against_cwd_at_configure(tmp_path,
                                                           monkeypatch):
    monkeypatch.chdir(tmp_path)
    ev.configure("server", node=0, run_id=tu.RUN_ID,
                 run_started_at=tu.RUN_STARTED_AT, console=False)
    expected = Path(os.getcwd()) / "batchgen_logs" / tu.RUN_DIR / "node0"
    monkeypatch.chdir("/")
    path = ev.jsonl_path()
    assert path.is_absolute()
    assert path == expected / "server.jsonl"
    assert path.is_file()


def test_encode_is_one_compact_line():
    line = schema.encode({"v": 1, "msg": "a\nb", "data": {"k": "é"}})
    assert "\n" not in line
    assert json.loads(line) == {"v": 1, "msg": "a\nb", "data": {"k": "é"}}
