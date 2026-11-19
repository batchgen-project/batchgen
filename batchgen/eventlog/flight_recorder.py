# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Per-process flight recorder: the last schema-v1 events, kept for a crash.

:func:`install` gives the process a fixed-size ring of the most recent event
log envelopes. The ring is a file of fixed-width text slots under the
process's log directory::

    <log_dir>/<run_dir>/node<N>/flight-recorder/<proc>-cohort<C>-pid<P>.ring

Each slot is one line, ``<16-digit sequence> <envelope JSON><spaces>\\n``; the
first slot is a JSON header naming the geometry and the process. The event
log's listener thread writes every envelope it writes to the JSONL file into
the next slot with one ``pwrite``, so the ring sees exactly the records the
event log keeps, through the one place they are encoded, and the threads that
log pay nothing for it. Slot writes land in the kernel's page cache, so the
ring outlives the process however it dies, a native crash or ``SIGKILL``
included, and :func:`read_ring` decodes it afterwards.

Three triggers leave evidence:

* a fatal Python exception (the caller of :func:`dump`, or an uncaught
  exception in any thread through ``threading.excepthook``) and
* a hang reported by a watchdog (:meth:`FlightRecorder.watchdog_dump_info`)
  write a dump: a JSONL file whose first line is an ``eventlog.flight.dump``
  envelope carrying the reason, the traceback and the stack of every thread,
  followed by the ring's envelopes, oldest first;
* a fatal signal runs no Python: ``faulthandler`` appends the stacks of all
  threads to the ring file itself, after the last slot, and the slots already
  hold the events.

A dump never raises into the failing path and never blocks it for longer
than :data:`DUMP_TIMEOUT_S`.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from batchgen import eventlog
from batchgen.eventlog import schema

__all__ = [
    "DUMP_EVENT",
    "FlightRecorder",
    "FlightRecorderError",
    "Ring",
    "dump",
    "install",
    "read_ring",
]

_logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
DIR_NAME = "flight-recorder"
DUMP_EVENT = "eventlog.flight.dump"

# 2048 slots of 2 KiB: a 4 MiB file per process generation, holding minutes
# of a worker's INFO-level events. An envelope longer than a slot is stored
# without its payload and traceback (see ``_fit``).
DEFAULT_SLOTS = 2048
DEFAULT_SLOT_BYTES = 2048
_SEQ_DIGITS = 16
_SLOT_OVERHEAD = _SEQ_DIGITS + 2  # sequence, separating space, newline

# Retention: dumps one process may write, and the generations (processes) of
# one ``<proc>`` kept per run directory, the new one included.
MAX_DUMPS_PER_PROCESS = 4
KEEP_GENERATIONS = 4

# A dump waits this long for the listener to write what is queued, and its
# caller waits at most DUMP_TIMEOUT_S for the whole dump.
FLUSH_TIMEOUT_S = 2.0
DUMP_TIMEOUT_S = 5.0

TRIGGER_EXCEPTION = "exception"
TRIGGER_THREAD_EXCEPTION = "thread_exception"
TRIGGER_HANG = "hang"


class FlightRecorderError(RuntimeError):
    """The flight recorder cannot be installed in this process."""


# ------------------------------------------------------------- ring file


def _blank_slot(slot_bytes: int) -> bytes:
    return b" " * (slot_bytes - 1) + b"\n"


def _fit(envelope: dict, payload: bytes, limit: int) -> bytes:
    """``payload`` if it fits ``limit`` bytes, else a reduced envelope.

    The reduced envelope keeps every scalar field, drops ``data`` and
    ``exc``, records the original size in ``data`` and shortens ``msg``
    until it fits.
    """
    if len(payload) <= limit:
        return payload
    reduced = {
        k: v for k, v in envelope.items()
        if k not in (schema.F_DATA, schema.F_EXC)
    }
    reduced[schema.F_DATA] = {"eventlog_truncated_bytes": len(payload)}
    msg = str(reduced.get(schema.F_MSG, ""))
    while True:
        out = schema.encode(reduced).encode("utf-8", "backslashreplace")
        if len(out) <= limit:
            return out
        if not msg:
            break
        msg = msg[: len(msg) // 2]
        reduced[schema.F_MSG] = msg
    minimal = {
        k: reduced[k]
        for k in (schema.F_VERSION, schema.F_TS, schema.F_LEVEL, schema.F_EVENT)
        if k in reduced
    }
    minimal[schema.F_DATA] = reduced[schema.F_DATA]
    return schema.encode(minimal).encode("utf-8", "backslashreplace")[:limit]


class _RingFile:
    """The slot file. ``record`` runs on the event log's listener thread."""

    def __init__(self, path: Path, slots: int, slot_bytes: int,
                 header: dict) -> None:
        if slots < 1:
            raise FlightRecorderError(f"slots must be >= 1, got {slots}")
        if slot_bytes < 256:
            raise FlightRecorderError(
                f"slot_bytes must be >= 256, got {slot_bytes}"
            )
        self.path = path
        self.slots = slots
        self.slot_bytes = slot_bytes
        self._payload_limit = slot_bytes - _SLOT_OVERHEAD
        self._seq = 0
        self.write_errors = 0
        head = json.dumps(header, sort_keys=True).encode("utf-8")
        if len(head) > slot_bytes - 1:
            raise FlightRecorderError("flight recorder header exceeds a slot")
        # The whole file is written once here, so every later slot write
        # overwrites allocated blocks instead of growing the file.
        image = bytearray(_blank_slot(slot_bytes) * (slots + 1))
        image[: len(head)] = head
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            view = memoryview(image)
            while view:
                view = view[os.write(self._fd, view):]
            # faulthandler writes at the end of the file, after the slots.
            self.fault_fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        except BaseException:
            os.close(self._fd)
            raise

    def record(self, envelope: dict, line: str) -> None:
        """Write one envelope into the next slot. Never raises.

        One ``os.pwrite``, which releases the GIL for the system call, so a
        slow file system stalls this thread and not the threads that log.
        """
        try:
            payload = _fit(
                envelope,
                line.encode("utf-8", "backslashreplace"),
                self._payload_limit,
            )
            self._seq += 1
            seq = self._seq
            slot = (b"%0*d " % (_SEQ_DIGITS, seq)) + payload
            slot = slot.ljust(self.slot_bytes - 1) + b"\n"
            index = 1 + (seq - 1) % self.slots
            os.pwrite(self._fd, slot, index * self.slot_bytes)
        except Exception as exc:
            self.write_errors += 1
            if self.write_errors == 1:
                _report(f"ring write failed: {exc!r}")

    def close(self) -> None:
        for fd in (self._fd, self.fault_fd):
            try:
                os.close(fd)
            except OSError:
                pass


@dataclass
class Ring:
    """A decoded ring file."""

    path: Path
    header: dict
    events: list[dict] = field(default_factory=list)
    recorded: int = 0
    torn: int = 0
    fatal_text: str = ""

    @property
    def overwritten(self) -> int:
        return max(0, self.recorded - int(self.header.get("slots", 0)))


def read_ring(path: str | os.PathLike) -> Ring:
    """Decode a ring file, live or left behind by a dead process.

    ``events`` holds the envelopes still in the ring, oldest first;
    ``recorded`` is how many were ever written; ``torn`` counts slots whose
    content does not decode (a write cut short by the process dying);
    ``fatal_text`` is what ``faulthandler`` appended after the slots.
    """
    path = Path(path)
    data = path.read_bytes()
    end = data.find(b"\n")
    if end < 0:
        raise ValueError(f"{path} is not a flight recorder ring")
    header = json.loads(data[:end])
    if header.get("flight_recorder") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a v{FORMAT_VERSION} ring")
    slots = int(header["slots"])
    slot_bytes = int(header["slot_bytes"])
    ring = Ring(path=path, header=header)
    entries = []
    for index in range(1, slots + 1):
        raw = data[index * slot_bytes:(index + 1) * slot_bytes]
        if not raw.strip():
            continue
        try:
            seq = int(raw[:_SEQ_DIGITS])
            if raw[_SEQ_DIGITS:_SEQ_DIGITS + 1] != b" " or seq < 1:
                raise ValueError("bad sequence field")
            event = json.loads(raw[_SEQ_DIGITS + 1:].rstrip(b" \n"))
            if not isinstance(event, dict):
                raise ValueError("slot is not an envelope")
        except ValueError:
            ring.torn += 1
            continue
        entries.append((seq, event))
    entries.sort(key=lambda entry: entry[0])
    ring.events = [event for _, event in entries]
    ring.recorded = entries[-1][0] if entries else 0
    ring.fatal_text = data[(slots + 1) * slot_bytes:].decode(
        "utf-8", "backslashreplace"
    )
    return ring


# --------------------------------------------------------------- dumping


def _report(message: str) -> None:
    try:
        sys.stderr.write(f"batchgen.eventlog.flight_recorder: {message}\n")
    except Exception:
        pass


def _thread_stacks(first: int | None) -> list[dict]:
    """The Python stack of every thread, ``first`` (an ident) leading."""
    threads = {thread.ident: thread for thread in threading.enumerate()}
    stacks = []
    for ident, frame in sys._current_frames().items():
        thread = threads.get(ident)
        lines = "".join(traceback.format_stack(frame)).splitlines()
        stacks.append({
            "name": thread.name if thread is not None else f"<{ident}>",
            "ident": ident,
            "daemon": thread.daemon if thread is not None else None,
            "stack": lines,
        })
    stacks.sort(key=lambda item: (
        item["ident"] != first, item["name"] != "MainThread", item["name"]
    ))
    return stacks


def _format_exc(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    return "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).rstrip("\n")


class FlightRecorder:
    """The installed recorder of this process; see :func:`install`."""

    def __init__(self, ring: _RingFile, info: schema.ProcessInfo,
                 stem: str) -> None:
        self.ring = ring
        self.info = info
        self.stem = stem
        self._dumps = 0
        self._dump_lock = threading.Lock()
        self._previous_excepthook = None
        self._faulthandler_enabled = False

    @property
    def path(self) -> Path:
        return self.ring.path

    def dump(self, trigger: str, reason: str,
             exc: BaseException | None = None,
             *, timeout_s: float = DUMP_TIMEOUT_S) -> Path | None:
        """Write a dump of the ring; the path, or ``None`` if none was written.

        The stacks are taken on the calling thread, then the dump is written
        by a helper thread this call waits at most ``timeout_s`` for. Never
        raises.
        """
        try:
            current = threading.current_thread()
            stacks = _thread_stacks(current.ident)
            result: list[Path | None] = [None]
            helper = threading.Thread(
                target=self._dump_guarded,
                args=(result, trigger, reason, _format_exc(exc), stacks,
                      current.name),
                name="flight-recorder-dump",
                daemon=True,
            )
            helper.start()
            helper.join(timeout_s)
            if helper.is_alive():
                _report(f"{trigger} dump still running after {timeout_s}s")
            return result[0]
        except Exception as error:
            _report(f"{trigger} dump failed: {error!r}")
            return None

    def watchdog_dump_info(self, debug_name: str) -> Callable[[], str]:
        """A ``Watchdog`` ``dump_info`` callback that writes a hang dump.

        The watchdog calls it from its own thread when the watched thread
        stopped feeding it, and logs the string it returns.
        """
        def _dump_info() -> str:
            path = self.dump(TRIGGER_HANG, f"{debug_name} watchdog timeout")
            return f"flight recorder dump: {path}" if path is not None else ""

        return _dump_info

    def close(self) -> None:
        """Stop recording and release the ring file.

        Restores the thread hook, disables ``faulthandler`` (it must not keep
        a closed descriptor) and detaches the ring from the event log. A
        worker never calls this; its descriptors close when it exits.
        """
        global _recorder
        if self._previous_excepthook is not None:
            if threading.excepthook == self._thread_excepthook:
                threading.excepthook = self._previous_excepthook
            self._previous_excepthook = None
        if self._faulthandler_enabled:
            faulthandler.disable()
            self._faulthandler_enabled = False
        try:
            eventlog.set_jsonl_mirror(None)
            eventlog.flush(FLUSH_TIMEOUT_S)
        except RuntimeError:
            pass  # the event log was shut down first; nothing writes the ring
        self.ring.close()
        if _recorder is self:
            _recorder = None

    # -- internals

    def _dump_guarded(self, result, trigger, reason, exc_text, stacks,
                      thread_name) -> None:
        try:
            result[0] = self._write_dump(
                trigger, reason, exc_text, stacks, thread_name
            )
        except Exception as error:
            _report(f"{trigger} dump failed: {error!r}")
            return
        if result[0] is not None:
            try:
                _logger.error(
                    "Flight recorder dump (%s): %s", trigger, result[0]
                )
            except Exception:
                pass

    def _next_dump_path(self) -> Path | None:
        with self._dump_lock:
            index = self._dumps + 1
            self._dumps = index
        if index > MAX_DUMPS_PER_PROCESS:
            if index == MAX_DUMPS_PER_PROCESS + 1:
                _report(
                    f"further dumps skipped: this process already wrote "
                    f"{MAX_DUMPS_PER_PROCESS}"
                )
            return None
        return self.ring.path.with_name(f"{self.stem}-dump{index}.jsonl")

    def _write_dump(self, trigger, reason, exc_text, stacks,
                    thread_name) -> Path | None:
        path = self._next_dump_path()
        if path is None:
            return None
        flushed = eventlog.flush(FLUSH_TIMEOUT_S)
        ring = read_ring(self.ring.path)
        record = logging.LogRecord(
            name=__name__,
            level=logging.CRITICAL,
            pathname=__file__,
            lineno=0,
            msg=f"flight recorder dump: trigger={trigger} reason={reason}",
            args=None,
            exc_info=None,
        )
        record.threadName = thread_name
        setattr(record, schema.ATTR_EVENT, DUMP_EVENT)
        setattr(record, schema.ATTR_EXC, exc_text)
        setattr(record, schema.ATTR_DATA, {
            "trigger": trigger,
            "reason": reason,
            "flushed": flushed,
            "ring": {
                "path": str(ring.path),
                "slots": self.ring.slots,
                "slot_bytes": self.ring.slot_bytes,
                "recorded": ring.recorded,
                "overwritten": ring.overwritten,
                "torn": ring.torn,
                "write_errors": self.ring.write_errors,
                "events": len(ring.events),
            },
            "threads": stacks,
        })
        header = schema.to_envelope(record, self.info, record.msg)
        lines = [schema.encode(header)]
        lines.extend(schema.encode(event) for event in ring.events)
        body = ("\n".join(lines) + "\n").encode("utf-8", "backslashreplace")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            view = memoryview(body)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
        return path

    def _thread_excepthook(self, args) -> None:
        previous = self._previous_excepthook or threading.__excepthook__
        if args.exc_type is SystemExit:
            previous(args)
            return
        name = args.thread.name if args.thread is not None else "<unknown>"
        try:
            _logger.critical(
                "Uncaught exception in thread %s", name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        except Exception:
            pass
        self.dump(
            TRIGGER_THREAD_EXCEPTION,
            f"uncaught exception in thread {name}",
            args.exc_value,
        )
        # The default hook prints the traceback the record above already
        # carries to the console; any other hook still runs.
        if previous is not threading.__excepthook__:
            previous(args)


# ---------------------------------------------------- install and retention

_GENERATION_RE = r"^{stem}-cohort\d+-pid\d+(\.ring|-dump\d+\.jsonl)$"


def _generation_stem(name: str) -> str:
    return re.sub(r"(\.ring|-dump\d+\.jsonl)$", "", name)


def _prune(directory: Path, proc_stem: str, keep: int) -> None:
    """Delete all but the newest ``keep`` generations of ``proc_stem``."""
    pattern = re.compile(_GENERATION_RE.format(stem=re.escape(proc_stem)))
    newest: dict[str, float] = {}
    files: dict[str, list[Path]] = {}
    for entry in directory.iterdir():
        if not pattern.match(entry.name):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        stem = _generation_stem(entry.name)
        newest[stem] = max(newest.get(stem, mtime), mtime)
        files.setdefault(stem, []).append(entry)
    ordered = sorted(newest, key=lambda stem: newest[stem], reverse=True)
    for stem in ordered[keep:]:
        for entry in files[stem]:
            try:
                entry.unlink()
            except OSError as exc:
                _report(f"could not prune {entry}: {exc!r}")


_recorder: FlightRecorder | None = None


def install(*, slots: int = DEFAULT_SLOTS,
            slot_bytes: int = DEFAULT_SLOT_BYTES) -> FlightRecorder:
    """Install the flight recorder of this process.

    Requires a configured event log bound to its run, so the ring lands next
    to this process's JSONL file. Prunes the older generations of this
    process kind (``KEEP_GENERATIONS``), creates this generation's ring,
    mirrors the event log into it, points ``faulthandler`` at it and takes
    over ``threading.excepthook``. Raises :class:`FlightRecorderError` when
    there is no bound JSONL event log or a recorder is already installed.
    """
    global _recorder
    if _recorder is not None:
        raise FlightRecorderError("a flight recorder is already installed")
    info = eventlog.process_info()
    jsonl = eventlog.jsonl_path()
    if info is None or jsonl is None:
        raise FlightRecorderError(
            "the flight recorder needs a configured event log bound to a run "
            "with JSONL output"
        )
    directory = jsonl.parent / DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    cohort = 0 if info.cohort is None else info.cohort
    stem = f"{info.file_stem}-cohort{cohort}-pid{os.getpid()}"
    _prune(directory, info.file_stem, KEEP_GENERATIONS - 1)
    header = {
        "flight_recorder": FORMAT_VERSION,
        "slots": slots,
        "slot_bytes": slot_bytes,
        "created": schema.format_ts(time.time()),
        "pid": os.getpid(),
        schema.F_PROC: info.proc,
        schema.F_RUN_ID: info.run_id,
        schema.F_NODE: info.node,
        schema.F_RANK: info.rank,
        schema.F_LOCAL_RANK: info.local_rank,
        schema.F_COHORT: info.cohort,
    }
    ring = _RingFile(directory / f"{stem}.ring", slots, slot_bytes, header)
    recorder = FlightRecorder(ring, info, stem)
    try:
        eventlog.set_jsonl_mirror(ring)
        faulthandler.enable(file=ring.fault_fd, all_threads=True)
    except BaseException:
        eventlog.set_jsonl_mirror(None)
        ring.close()
        raise
    recorder._faulthandler_enabled = True
    recorder._previous_excepthook = threading.excepthook
    threading.excepthook = recorder._thread_excepthook
    _recorder = recorder
    _logger.info("Flight recorder ring: %s", ring.path)
    return recorder


def dump(trigger: str, reason: str,
         exc: BaseException | None = None) -> Path | None:
    """:meth:`FlightRecorder.dump` of the installed recorder.

    ``None`` when this process installed none (it failed before
    :func:`install`). Never raises.
    """
    recorder = _recorder
    if recorder is None:
        return None
    return recorder.dump(trigger, reason, exc)
