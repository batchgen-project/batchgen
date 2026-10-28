# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Unified BatchGen logging: one JSONL file per process plus a console view.

Each process calls :func:`configure` once. The root logger then has exactly
one handler, which snapshots records on the caller thread and queues them; a
listener thread writes every record to the console and to
``<log_dir>/<run_dir>/node<node>/<proc_file>.jsonl``. When the run identity
is not known yet (non-zero nodes), file records are buffered in memory until
:func:`bind_run` opens the file.

Structured events go through :func:`log_event`; per-batch context comes from
:func:`bind` / :func:`set_context`, never from call-site arguments.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from . import counters, schema
from .console import ConsoleFormatter, register_console_renderer
from .context import ContextFilter, bind, clear_context, set_context
from .handlers import (
    DEFAULT_BUFFER_LIMIT,
    BindRequest,
    EventListener,
    EventQueueHandler,
    FlushRequest,
    JsonlHandler,
    _report_internal_error,
    jsonl_file_path,
    run_dir_name,
)

__all__ = [
    "bind",
    "bind_run",
    "clear_context",
    "configure",
    "flush",
    "get_logger",
    "jsonl_path",
    "log_event",
    "register_console_renderer",
    "set_context",
    "shutdown",
]

DEFAULT_LOG_DIR = "./batchgen_logs"


@dataclass
class _Runtime:
    info: schema.ProcessInfo
    log_dir: Path
    queue: queue.SimpleQueue
    handler: EventQueueHandler
    listener: EventListener
    console_handler: logging.Handler | None
    jsonl_handler: JsonlHandler | None
    run_id: str | None = None
    path: Path | None = None


_LOCK = threading.RLock()
_state: _Runtime | None = None
_atexit_registered = False
_VALID_EVENTS: set[str] = set()


def _resolve_level(level: int | str) -> int:
    if isinstance(level, str):
        mapping = logging.getLevelNamesMapping()
        if level.upper() not in mapping:
            raise ValueError(f"unknown log level {level!r}")
        return mapping[level.upper()]
    if isinstance(level, int) and not isinstance(level, bool):
        return level
    raise TypeError("level must be a level name or number")


def _check_index(name: str, value: object, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")


def _process_info(
    proc: str,
    node: int,
    rank: int | None,
    local_rank: int | None,
    cohort: int | None,
) -> schema.ProcessInfo:
    if proc not in schema.PROCS:
        raise ValueError(f"proc must be one of {schema.PROCS}, got {proc!r}")
    _check_index("node", node, required=True)
    is_rank = proc == "rank"
    _check_index("rank", rank, required=is_rank)
    _check_index("local_rank", local_rank, required=False)
    _check_index("cohort", cohort, required=False)
    if not is_rank and (rank is not None or local_rank is not None):
        raise ValueError("rank/local_rank only apply to proc='rank'")
    return schema.ProcessInfo(
        proc=proc, node=node, rank=rank, local_rank=local_rank, cohort=cohort
    )


def configure(
    proc: str,
    *,
    node: int,
    run_id: str | None = None,
    run_started_at: datetime | float | None = None,
    rank: int | None = None,
    local_rank: int | None = None,
    cohort: int | None = None,
    log_dir: str | os.PathLike | None = None,
    level: int | str = "INFO",
    console: bool = True,
    console_stream: TextIO | None = None,
    jsonl: bool = True,
) -> None:
    """Install the event log for this process, replacing any previous one.

    With ``run_id=None`` the console starts immediately and file records are
    buffered until :func:`bind_run`.
    """
    global _atexit_registered, _state
    info = _process_info(proc, node, rank, local_rank, cohort)
    if run_id is not None:
        schema.validate_run_id(run_id)
        if run_started_at is None:
            raise ValueError("run_started_at is required with run_id")
        run_dir_name(run_id, run_started_at)
    elif run_started_at is not None:
        raise ValueError("run_started_at requires run_id")
    levelno = _resolve_level(level)
    resolved_dir = Path(os.path.abspath(log_dir or DEFAULT_LOG_DIR))

    with _LOCK:
        shutdown()
        console_handler = None
        if console:
            stream = sys.stderr if console_stream is None else console_stream
            console_handler = logging.StreamHandler(stream)
            console_handler.setFormatter(ConsoleFormatter(info, stream))
            console_handler.setLevel(levelno)
        jsonl_handler = None
        if jsonl:
            jsonl_handler = JsonlHandler(info, DEFAULT_BUFFER_LIMIT)
            jsonl_handler.setLevel(levelno)

        record_queue: queue.SimpleQueue = queue.SimpleQueue()
        handler = EventQueueHandler(record_queue)
        handler.setLevel(levelno)
        handler.addFilter(ContextFilter())
        listener = EventListener(record_queue, console_handler, jsonl_handler)

        root = logging.getLogger()
        for existing in list(root.handlers):
            root.removeHandler(existing)
        root.addHandler(handler)
        root.setLevel(levelno)
        listener.start()
        logging.captureWarnings(True)

        _state = _Runtime(
            info=info,
            log_dir=resolved_dir,
            queue=record_queue,
            handler=handler,
            listener=listener,
            console_handler=console_handler,
            jsonl_handler=jsonl_handler,
        )
        if not _atexit_registered:
            atexit.register(shutdown)
            os.register_at_fork(after_in_child=_detach_after_fork)
            _atexit_registered = True
        if run_id is not None:
            try:
                bind_run(run_id, run_started_at)
            except BaseException:
                shutdown()
                raise


def _detach_after_fork() -> None:
    """A forked child has no listener thread; stop enqueueing into a dead
    queue so ``logging.lastResort`` shows WARNING+ until it reconfigures.
    """
    global _state
    state, _state = _state, None
    if state is not None:
        logging.getLogger().removeHandler(state.handler)


def bind_run(
    run_id: str, run_started_at: datetime | float
) -> Path | None:
    """Open this process's JSONL file and write the buffered records.

    Returns the file path, or ``None`` when JSONL output is disabled.
    Binding again to the same ``run_id`` returns the same path.
    """
    schema.validate_run_id(run_id)
    run_dir_name(run_id, run_started_at)
    with _LOCK:
        state = _state
        if state is None:
            raise RuntimeError("batchgen.eventlog is not configured")
        if state.run_id is not None:
            if state.run_id == run_id:
                return state.path
            raise RuntimeError(
                f"event log already bound to run {state.run_id}"
            )
        path = None
        if state.jsonl_handler is not None:
            path = jsonl_file_path(
                state.log_dir, state.info, run_id, run_started_at
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = open(
                path, "a", encoding="utf-8", errors="backslashreplace"
            )
            state.queue.put(BindRequest(stream, run_id))
        state.run_id = run_id
        state.path = path
        return path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    event: str,
    /,
    level: int = logging.INFO,
    **data: object,
) -> None:
    """Log structured event ``event`` with JSON payload ``data``.

    Returns before any rendering when ``level`` is disabled for ``logger``.
    """
    if not logger.isEnabledFor(level):
        return
    if event not in _VALID_EVENTS:
        _VALID_EVENTS.add(schema.validate_event_name(event))
    logger.log(
        level,
        event,
        extra={schema.ATTR_EVENT: event, schema.ATTR_DATA: data},
        stacklevel=2,
    )


def flush(timeout_s: float = 2.0) -> bool:
    """Wait until every record logged before the call is written and flushed.

    Lock-free so it is usable on exit paths (e.g. right before
    ``os._exit``). Returns ``False`` on timeout or when the listener is not
    running.
    """
    state = _state
    if state is None:
        return True
    listener = state.listener
    if not listener.alive or threading.current_thread() is listener.thread:
        return False
    request = FlushRequest()
    state.queue.put(request)
    return request.done.wait(timeout_s)


def shutdown() -> None:
    """Drain and stop the listener, close the file, remove the handler."""
    global _state
    with _LOCK:
        state, _state = _state, None
        if state is None:
            return
        logging.getLogger().removeHandler(state.handler)
        logging.captureWarnings(False)
        stopped = state.listener.stop()
        if not stopped:
            _report_internal_error(
                RuntimeError(
                    "listener did not stop in time; queued records may "
                    "be lost and sinks were left open"
                )
            )
        else:
            if state.console_handler is not None:
                state.console_handler.flush()
                state.console_handler.close()
            if state.jsonl_handler is not None:
                state.jsonl_handler.close()
        state.handler.close()


def jsonl_path() -> Path | None:
    state = _state
    return None if state is None else state.path
