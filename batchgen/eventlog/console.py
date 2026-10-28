# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Human console rendering of a prepared log record.

One line per record::

    <utc time> <level> [rank=R node=N] [batch=B wave=W step=S] <body>

``WARN`` and above append ``  at <file>:<line>``; a traceback follows on
lines indented by four spaces. The body of an event record is also the JSONL
``msg``, so both sinks say the same thing.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TextIO

from . import schema

_RENDERERS: dict[str, Callable[[dict], str]] = {}

_LEVEL_LABELS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO ",
    logging.WARNING: "WARN ",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRIT ",
}
_LEVEL_COLOURS = {
    logging.DEBUG: "2",
    logging.INFO: "32",
    logging.WARNING: "33",
    logging.ERROR: "31",
    logging.CRITICAL: "1;31",
}
_TRACE_INDENT = "    "


def register_console_renderer(
    event: str, fn: Callable[[dict], str]
) -> None:
    """Render ``event`` records with ``fn(data)`` instead of ``k=v`` pairs."""
    schema.validate_event_name(event, allow_reserved=True)
    if not callable(fn):
        raise TypeError("console renderer must be callable")
    _RENDERERS[event] = fn


def _format_value(value: Any) -> str:
    text = str(value)
    if isinstance(value, str) and (
        not value
        or any(c.isspace() or c in '"=' for c in value)
    ):
        return json.dumps(value, ensure_ascii=False)
    return text


def _default_body(event: str, data: dict) -> str:
    pairs = " ".join(f"{k}={_format_value(v)}" for k, v in data.items())
    return f"{event} {pairs}" if pairs else event


def event_body(record: logging.LogRecord) -> str:
    """Body of an event record; computed once and cached on the record."""
    body = getattr(record, schema.ATTR_BODY, None)
    if body is not None:
        return body
    event = getattr(record, schema.ATTR_EVENT)
    data = getattr(record, schema.ATTR_DATA, None) or {}
    renderer = _RENDERERS.get(event)
    body = None
    if renderer is not None:
        try:
            rendered = renderer(data)
            body = None if rendered is None else str(rendered)
        except Exception:
            body = None
    if body is None:
        body = _default_body(event, data)
    setattr(record, schema.ATTR_BODY, body)
    return body


def message_text(record: logging.LogRecord) -> str:
    """The JSONL ``msg``: event body, or the formatted plain message."""
    if getattr(record, schema.ATTR_EVENT, None) is not None:
        return event_body(record)
    return record.getMessage()


def component(logger_name: str) -> str:
    if not logger_name:
        return "root"
    if logger_name.startswith("batchgen."):
        return logger_name[len("batchgen."):]
    return logger_name


def _timestamp(created: float) -> str:
    moment = datetime.fromtimestamp(created, tz=timezone.utc)
    millis = moment.microsecond // 1000
    return f"{moment:%Y-%m-%d %H:%M:%S}.{millis:03d}Z"


def _level_label(levelno: int) -> str:
    label = _LEVEL_LABELS.get(levelno)
    if label is not None:
        return label
    return logging.getLevelName(levelno)[:5].ljust(5)


def process_block(info: schema.ProcessInfo) -> str:
    if info.proc == "rank":
        text = f"rank={info.rank} node={info.node}"
    elif info.proc == "paramserver":
        text = f"param-server node={info.node}"
    else:
        text = f"{info.proc} node={info.node}"
    if info.cohort is not None and info.cohort > 0:
        text += f" restart={info.cohort}"
    return f"[{text}]"


def context_block(record: logging.LogRecord) -> str:
    parts = []
    batch_id = getattr(record, schema.ATTR_BATCH_ID, None)
    wave = getattr(record, schema.ATTR_WAVE, None)
    step = getattr(record, schema.ATTR_STEP, None)
    if batch_id is not None:
        parts.append(f"batch={batch_id}")
    if wave is not None:
        parts.append(f"wave={wave}")
    if step is not None:
        parts.append(f"step={step}")
    return f"[{' '.join(parts)}]" if parts else ""


def _wants_colour(stream: TextIO | None) -> bool:
    if stream is None or "NO_COLOR" in os.environ:
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class ConsoleFormatter(logging.Formatter):
    """Formats prepared records with the approved console template."""

    def __init__(
        self, info: schema.ProcessInfo, stream: TextIO | None = None
    ) -> None:
        super().__init__()
        self._process = process_block(info)
        self._colour = _wants_colour(stream)

    def format(self, record: logging.LogRecord) -> str:
        try:
            return self._format(record)
        except Exception:
            return repr(record)

    def _format(self, record: logging.LogRecord) -> str:
        label = _level_label(record.levelno)
        if self._colour:
            code = _LEVEL_COLOURS.get(record.levelno, "0")
            label = f"\x1b[{code}m{label}\x1b[0m"
        if getattr(record, schema.ATTR_EVENT, None) is not None:
            body = event_body(record)
        else:
            body = f"{component(record.name)}: {record.getMessage()}"
        parts = [_timestamp(record.created), label, self._process]
        context = context_block(record)
        if context:
            parts.append(context)
        parts.append(body)
        line = " ".join(parts)
        if record.levelno >= logging.WARNING:
            line += f"  at {record.filename}:{record.lineno}"
        trace = getattr(record, schema.ATTR_EXC, None)
        if trace:
            line += "".join(
                f"\n{_TRACE_INDENT}{text}" for text in trace.splitlines()
            )
        return line
