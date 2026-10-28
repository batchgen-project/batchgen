# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Schema v1 of the BatchGen JSONL event log.

Every line of a ``.jsonl`` file is one JSON object built by
:func:`to_envelope`; fields whose value is ``None`` are omitted. The console
line is a rendering of the same record, so this module is the single
definition of what a log record contains.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1

# Envelope field names, in the order they are written.
F_VERSION = "v"
F_TS = "ts"
F_LEVEL = "level"
F_EVENT = "event"
F_MSG = "msg"
F_LOGGER = "logger"
F_SRC = "src"
F_PROC = "proc"
F_PID = "pid"
F_THREAD = "thread"
F_RUN_ID = "run_id"
F_NODE = "node"
F_RANK = "rank"
F_LOCAL_RANK = "local_rank"
F_COHORT = "cohort"
F_BATCH_ID = "batch_id"
F_WAVE = "wave"
F_STEP = "step"
F_FILE = "file"
F_LINE = "line"
F_DATA = "data"
F_EXC = "exc"

ENVELOPE_FIELDS = (
    F_VERSION, F_TS, F_LEVEL, F_EVENT, F_MSG, F_LOGGER, F_SRC, F_PROC, F_PID,
    F_THREAD, F_RUN_ID, F_NODE, F_RANK, F_LOCAL_RANK, F_COHORT, F_BATCH_ID,
    F_WAVE, F_STEP, F_FILE, F_LINE, F_DATA, F_EXC,
)

SRC_PY = "py"
PROCS = ("server", "rank", "paramserver")

# Event of every plain ``logger.info(...)`` call.
TEXT_EVENT = "log.text"
# Written first when the pre-``bind_run`` buffer dropped records.
BUFFER_OVERFLOW_EVENT = "eventlog.buffer.overflow"
RESERVED_EVENT_PREFIXES = ("log.", "eventlog.")

EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){1,3}$")
# Same contract as ``batchgen/server/runtime_identity.py``.
RUN_ID_RE = re.compile(r"[0-9a-f]{32}\Z")

# LogRecord attributes owned by this package.
ATTR_EVENT = "bg_event"
ATTR_DATA = "bg_data"
ATTR_EXC = "bg_exc"
ATTR_BODY = "bg_body"
ATTR_BATCH_ID = "bg_batch_id"
ATTR_WAVE = "bg_wave"
ATTR_STEP = "bg_step"


def validate_event_name(name: str, *, allow_reserved: bool = False) -> str:
    """Return ``name`` or raise ``ValueError`` when it breaks the rule."""
    if not isinstance(name, str) or not EVENT_NAME_RE.fullmatch(name):
        raise ValueError(
            f"event name {name!r} must be 2-4 dot-separated lowercase "
            "snake_case segments, e.g. 'kv_cache.host.low'"
        )
    if not allow_reserved and name.startswith(RESERVED_EVENT_PREFIXES):
        raise ValueError(
            f"event name {name!r} uses a prefix reserved for the event log"
        )
    return name


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must be exactly 32 lowercase hexadecimal characters"
        )
    return run_id


@dataclass(frozen=True)
class ProcessInfo:
    """Identity of the logging process, stamped on every record."""

    proc: str
    node: int
    rank: int | None = None
    local_rank: int | None = None
    cohort: int | None = None
    run_id: str | None = None

    @property
    def file_stem(self) -> str:
        if self.proc == "rank":
            return f"rank{self.rank}"
        return self.proc


def format_ts(created: float) -> str:
    """UTC RFC3339 with microseconds, e.g. ``2026-10-27T14:02:05.001000Z``."""
    moment = datetime.fromtimestamp(created, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def to_envelope(
    record: logging.LogRecord, info: ProcessInfo, msg: str
) -> dict[str, Any]:
    """Build the v1 envelope of a prepared record; ``None`` fields dropped."""
    data = getattr(record, ATTR_DATA, None) or None
    envelope = {
        F_VERSION: SCHEMA_VERSION,
        F_TS: format_ts(record.created),
        F_LEVEL: record.levelname,
        F_EVENT: getattr(record, ATTR_EVENT, None) or TEXT_EVENT,
        F_MSG: msg,
        F_LOGGER: record.name,
        F_SRC: SRC_PY,
        F_PROC: info.proc,
        F_PID: record.process,
        F_THREAD: record.threadName,
        F_RUN_ID: info.run_id,
        F_NODE: info.node,
        F_RANK: info.rank,
        F_LOCAL_RANK: info.local_rank,
        F_COHORT: info.cohort,
        F_BATCH_ID: getattr(record, ATTR_BATCH_ID, None),
        F_WAVE: getattr(record, ATTR_WAVE, None),
        F_STEP: getattr(record, ATTR_STEP, None),
        F_FILE: record.filename,
        F_LINE: record.lineno,
        F_DATA: data,
        F_EXC: getattr(record, ATTR_EXC, None),
    }
    return {k: v for k, v in envelope.items() if v is not None}


MAX_REPR_CHARS = 1000


def _repr(value: Any) -> str:
    text = repr(value)
    if len(text) <= MAX_REPR_CHARS:
        return text
    return text[:MAX_REPR_CHARS] + "...<truncated>"


def _dumps(value: Any) -> str:
    return json.dumps(
        value,
        default=_repr,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    try:
        _dumps(value)
    except (TypeError, ValueError):
        return _repr(value)
    return value


def encode(envelope: dict[str, Any]) -> str:
    """One JSON line (no newline); unencodable payload values -> ``repr``."""
    try:
        return _dumps(envelope)
    except (TypeError, ValueError):
        data = envelope.get(F_DATA)
        if isinstance(data, dict):
            data = _safe_value(data)
        else:
            data = _repr(data)
        return _dumps({**envelope, F_DATA: data})
