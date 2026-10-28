# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Queue handler, listener thread and the JSONL sink.

The caller thread only snapshots the record (message, traceback, context) and
enqueues it. One listener thread renders and writes it to the console and the
JSONL file, so it is the only writer of that file. Flush and run binding are
control items sent through the same queue, which keeps them ordered with the
records logged before them.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import logging
import logging.handlers
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from . import console, schema

DEFAULT_BUFFER_LIMIT = 50_000
_STOP_TIMEOUT_S = 5.0
_FORMATTER = logging.Formatter()


def run_dir_name(run_id: str, run_started_at: datetime | float) -> str:
    """``%Y%m%d-%H%M%S`` (UTC run start) + ``-`` + first 8 run_id chars."""
    if isinstance(run_started_at, datetime):
        if run_started_at.utcoffset() is None:
            raise ValueError("run_started_at datetime must be timezone-aware")
        started = run_started_at.astimezone(timezone.utc)
    elif isinstance(run_started_at, (int, float)) and not isinstance(
        run_started_at, bool
    ):
        started = datetime.fromtimestamp(run_started_at, tz=timezone.utc)
    else:
        raise TypeError(
            "run_started_at must be an aware datetime or POSIX seconds"
        )
    return f"{started:%Y%m%d-%H%M%S}-{run_id[:8]}"


def jsonl_file_path(
    log_dir: Path,
    info: schema.ProcessInfo,
    run_id: str,
    run_started_at: datetime | float,
) -> Path:
    return (
        log_dir
        / run_dir_name(run_id, run_started_at)
        / f"node{info.node}"
        / f"{info.file_stem}.jsonl"
    )


class EventQueueHandler(logging.handlers.QueueHandler):
    """Snapshots a record on the caller thread; no I/O, no JSON."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        record = copy.copy(record)
        msg = record.getMessage()
        record.message = msg
        record.msg = msg
        record.args = None
        trace = []
        if record.exc_info:
            trace.append(_FORMATTER.formatException(record.exc_info))
        elif record.exc_text:
            trace.append(record.exc_text)
        if record.stack_info:
            trace.append(_FORMATTER.formatStack(record.stack_info))
        setattr(record, schema.ATTR_EXC, "\n".join(trace) or None)
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        data = getattr(record, schema.ATTR_DATA, None)
        if data is not None:
            # Rendering happens later on the listener thread; copy one
            # level so callers may mutate what they passed in.
            setattr(
                record,
                schema.ATTR_DATA,
                {k: _snapshot(v) for k, v in data.items()},
            )
        return record


def _snapshot(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, set)):
        return type(value)(value)
    return value


class JsonlHandler(logging.Handler):
    """Writes envelopes to one file; buffers until the run is bound."""

    def __init__(
        self,
        info: schema.ProcessInfo,
        buffer_limit: int = DEFAULT_BUFFER_LIMIT,
    ) -> None:
        super().__init__()
        self.info = info
        self._stream: TextIO | None = None
        self._buffer: collections.deque = collections.deque(
            maxlen=buffer_limit
        )
        self._dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._stream is None:
                if len(self._buffer) == self._buffer.maxlen:
                    self._dropped += 1
                self._buffer.append(record)
                return
            self._write(record)
        except Exception:
            self.handleError(record)

    def _write(self, record: logging.LogRecord) -> None:
        envelope = schema.to_envelope(
            record, self.info, console.message_text(record)
        )
        self._stream.write(schema.encode(envelope) + "\n")

    def open_stream(self, stream: TextIO, run_id: str) -> None:
        """Start writing; called on the listener thread only."""
        with self.lock:
            self.info = dataclasses.replace(self.info, run_id=run_id)
            self._stream = stream
            if self._dropped:
                self._write(_overflow_record(self._dropped, self._buffer))
                self._dropped = 0
            while self._buffer:
                self._write(self._buffer.popleft())
            stream.flush()

    def flush(self) -> None:
        with self.lock:
            if self._stream is not None:
                self._stream.flush()

    def close(self) -> None:
        with self.lock:
            if self._stream is not None:
                try:
                    self._stream.flush()
                finally:
                    self._stream.close()
                    self._stream = None
        super().close()


def _overflow_record(
    dropped: int, buffer: collections.deque
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="batchgen.eventlog",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=schema.BUFFER_OVERFLOW_EVENT,
        args=None,
        exc_info=None,
    )
    setattr(record, schema.ATTR_EVENT, schema.BUFFER_OVERFLOW_EVENT)
    setattr(
        record,
        schema.ATTR_DATA,
        {"dropped": dropped, "buffer_limit": buffer.maxlen},
    )
    return record


class FlushRequest:
    """Control item: set ``done`` once every earlier record is flushed."""

    def __init__(self) -> None:
        self.done = threading.Event()

    def run(self, listener: "EventListener") -> None:
        listener.flush_sinks()
        self.done.set()


class BindRequest:
    """Control item: hand the opened JSONL stream to the sink."""

    def __init__(self, stream: TextIO, run_id: str) -> None:
        self.stream = stream
        self.run_id = run_id

    def run(self, listener: "EventListener") -> None:
        if listener.jsonl is not None:
            listener.jsonl.open_stream(self.stream, self.run_id)
        else:
            self.stream.close()


_CONTROL = (FlushRequest, BindRequest)


class EventListener(logging.handlers.QueueListener):
    """The one thread that renders and writes records."""

    def __init__(
        self,
        queue,
        console_handler: logging.Handler | None,
        jsonl_handler: JsonlHandler | None,
    ) -> None:
        sinks = [h for h in (console_handler, jsonl_handler) if h is not None]
        super().__init__(queue, *sinks, respect_handler_level=True)
        self.jsonl = jsonl_handler

    @property
    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def thread(self) -> threading.Thread | None:
        return self._thread

    def handle(self, record) -> None:
        # An exception here would end the listener thread for good.
        try:
            if isinstance(record, _CONTROL):
                record.run(self)
                return
            if getattr(record, schema.ATTR_EVENT, None) is not None:
                console.event_body(record)
            for handler in self.handlers:
                if record.levelno >= handler.level:
                    handler.handle(record)
            if self.queue.empty():
                self.flush_sinks()
        except Exception as exc:
            _report_internal_error(exc)

    def flush_sinks(self) -> None:
        for handler in self.handlers:
            try:
                handler.flush()
            except Exception as exc:
                _report_internal_error(exc)

    def stop(self, timeout_s: float = _STOP_TIMEOUT_S) -> bool:
        """Drain and stop; ``False`` if the thread is still running."""
        thread = self._thread
        if thread is None:
            return True
        if thread.is_alive():
            self.enqueue_sentinel()
            thread.join(timeout_s)
        if thread.is_alive():
            return False
        self._thread = None
        return True


def _report_internal_error(exc: BaseException) -> None:
    try:
        sys.stderr.write(f"batchgen.eventlog: listener error: {exc!r}\n")
    except Exception:
        pass
