"""Per-process wiring of :mod:`batchgen.eventlog` for the BatchGen processes.

Every BatchGen process configures the event log exactly once, here:

* the HTTP server (``proc="server"``) right after its arguments parse;
* every GPU worker (``proc="rank"``) at the top of its spawned entry;
* the standalone parameter server (``proc="paramserver"``).

All processes of one run write under one run directory, named from a
:class:`RunIdentity`. Node 0's server creates it; every other node learns it
over the restart control store and binds to it once known, buffering its
earlier records until then. Workers receive it in their spawn arguments.

This module imports only the standard library and :mod:`batchgen.eventlog`,
so it loads without the server stack and without torch.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator, NoReturn, Optional, Tuple

from batchgen import eventlog
from batchgen.eventlog import schema

_logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = eventlog.DEFAULT_LOG_DIR
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOG_LEVEL = "INFO"

# Python level -> the spdlog level name the native (C++) loggers take, most
# severe first. ``native_log_level`` walks it, so a non-standard numeric level
# resolves to the nearest standard one instead of failing.
_NATIVE_LEVEL_BY_PYTHON_LEVEL = (
    (logging.CRITICAL, "critical"),
    (logging.ERROR, "error"),
    (logging.WARNING, "warning"),
    (logging.INFO, "info"),
    (logging.DEBUG, "debug"),
)

EXIT_EVENT = "process.exit"


def native_log_level() -> str:
    """The level the native (C++/spdlog) loggers of this process are built at.

    The core components take their level as a string argument because they
    have no access to the Python logging configuration. Rather than letting
    each call site hard-code one, this reads back the level the Python
    handlers were already configured with (``--log-level``), so one knob
    drives both logging stacks. ``core/utils.cpp``'s ``init_logger`` accepts
    these names case-insensitively.
    """
    effective = logging.getLogger().getEffectiveLevel()
    for threshold, native_name in _NATIVE_LEVEL_BY_PYTHON_LEVEL:
        if effective >= threshold:
            return native_name
    return "trace"


@dataclass(frozen=True)
class RunIdentity:
    """Names the run directory every process of one run writes under.

    ``run_started_at`` is node 0's server wall clock (POSIX seconds) at the
    moment it configured logging. It is published verbatim so every node and
    every worker derives the same directory name from it.
    """

    run_id: str
    run_started_at: float

    def __post_init__(self) -> None:
        schema.validate_run_id(self.run_id)
        if isinstance(self.run_started_at, bool) or not isinstance(
            self.run_started_at, (int, float)
        ):
            raise TypeError("run_started_at must be POSIX seconds")

    def to_json(self) -> str:
        return json.dumps(
            {"run_id": self.run_id, "run_started_at": self.run_started_at},
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> "RunIdentity":
        values = json.loads(payload)
        return cls(
            run_id=values["run_id"],
            run_started_at=float(values["run_started_at"]),
        )


def new_run_identity(run_id: str) -> RunIdentity:
    """A run identity for ``run_id`` that starts now."""
    return RunIdentity(run_id=run_id, run_started_at=time.time())


def add_logging_args(parser) -> None:
    """Add ``--log-dir`` and ``--log-level`` to an argparse parser."""
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Root directory of the structured event log; each run writes "
            "<log-dir>/<UTC start>-<run id prefix>/node<N>/<process>.jsonl. "
            "Relative paths resolve against the launch directory. "
            f"Default: {DEFAULT_LOG_DIR}"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default=DEFAULT_LOG_LEVEL,
        help=f"Minimum level of both the console and the JSONL log "
             f"(default: {DEFAULT_LOG_LEVEL})",
    )


def resolve_log_dir(value: Optional[str]) -> str:
    """The absolute log root, resolved against the current directory."""
    return os.path.abspath(value or DEFAULT_LOG_DIR)


_run: Optional[RunIdentity] = None
_settings: Optional[Tuple[str, str]] = None
_fallback_run_id: Optional[str] = None


def configure_process(
    proc: str,
    *,
    node: int,
    log_dir: Optional[str],
    level: str,
    run: Optional[RunIdentity] = None,
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    cohort: Optional[int] = None,
    fallback_run_id: Optional[str] = None,
) -> None:
    """Install the event log for this process.

    With ``run=None`` file records are buffered until :func:`bind_run`;
    ``fallback_run_id`` names the run :func:`ensure_run_bound` binds if the
    process ends before that.
    """
    global _run, _settings, _fallback_run_id
    resolved = resolve_log_dir(log_dir)
    eventlog.configure(
        proc,
        node=node,
        run_id=None if run is None else run.run_id,
        run_started_at=None if run is None else run.run_started_at,
        rank=rank,
        local_rank=local_rank,
        cohort=cohort,
        log_dir=resolved,
        level=level,
    )
    _settings = (resolved, level)
    _run = run
    _fallback_run_id = fallback_run_id


def bind_run(run: RunIdentity) -> None:
    """Open this process's JSONL file under ``run`` and write the buffer.

    Binding again to the same run is a no-op; a different run raises.
    """
    global _run
    eventlog.bind_run(run.run_id, run.run_started_at)
    _run = run


def ensure_run_bound(fallback_run_id: Optional[str] = None) -> None:
    """Bind a run of this process's own if none is bound yet.

    A node other than 0 buffers its records until node 0 names the run; when
    it ends before that, this writes the buffer under ``fallback_run_id``
    (default: the one given to :func:`configure_process`) instead of losing
    it at exit. Never raises.
    """
    run_id = fallback_run_id or _fallback_run_id
    if _run is not None or _settings is None or run_id is None:
        return
    try:
        bind_run(new_run_identity(run_id))
    except Exception:
        _logger.exception("Binding a fallback log run failed")


def current_run() -> Optional[RunIdentity]:
    return _run


def current_settings() -> Optional[Tuple[str, str]]:
    """``(log_dir, level)`` of the last :func:`configure_process` call."""
    return _settings


def record_exit(
    code: int,
    reason: str,
    *,
    logger: Optional[logging.Logger] = None,
    exc_info: bool = False,
    flush_timeout_s: float = 2.0,
    stacklevel: int = 2,
) -> None:
    """Log a CRITICAL ``process.exit`` record and flush every sink.

    For a caller about to ``os._exit``, which runs no atexit hook and flushes
    nothing. Never raises. Unconfigured, the record reaches stderr through
    ``logging.lastResort``.
    """
    try:
        target = logger if logger is not None else _logger
        target.critical(
            "%s code=%s reason=%s",
            EXIT_EVENT,
            code,
            reason,
            exc_info=exc_info,
            extra={
                schema.ATTR_EVENT: EXIT_EVENT,
                schema.ATTR_DATA: {"code": code, "reason": reason},
            },
            stacklevel=stacklevel,
        )
        ensure_run_bound()
        eventlog.flush(flush_timeout_s)
    except BaseException:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except BaseException:
            pass


def exit_process(
    code: int,
    reason: str,
    *,
    logger: Optional[logging.Logger] = None,
    exc_info: bool = False,
    flush_timeout_s: float = 2.0,
) -> NoReturn:
    """:func:`record_exit`, then ``os._exit(code)`` whatever happened."""
    record_exit(
        code,
        reason,
        logger=logger,
        exc_info=exc_info,
        flush_timeout_s=flush_timeout_s,
        stacklevel=3,
    )
    os._exit(code)


@contextlib.contextmanager
def recording_exit() -> Iterator[None]:
    """Run a process body; leave a ``process.exit`` record when it fails.

    A nonzero ``SystemExit`` or an unhandled exception is recorded at CRITICAL
    before it propagates. Either way, a process that never bound a run binds
    its fallback run, so its buffered records reach disk.
    """
    try:
        yield
    except SystemExit as exc:
        if exc.code not in (None, 0):
            code = exc.code if isinstance(exc.code, int) else 1
            record_exit(code, "exit with an error", stacklevel=3)
        raise
    except Exception:
        record_exit(1, "unhandled exception", exc_info=True, stacklevel=3)
        raise
    finally:
        ensure_run_bound()
