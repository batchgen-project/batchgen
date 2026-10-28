# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Hot-path helpers for rate-limited progress events."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable

STEP_COUNTERS_EVENT = "step.counters"
# Seconds between two ``step.counters`` records of one process. INFO
# periodic summaries stay at or below one record per 30 s per rank in steady
# decode (schema v1 level policy).
STEP_COUNTERS_INTERVAL_S = 30.0

_NO_ROWS = sys.maxsize


class IntervalGate:
    """``due()`` is true at most once per ``interval_s``.

    The first ``True`` comes once ``interval_s`` has elapsed since
    construction. A call is one clock read and a float compare.
    """

    __slots__ = ("_interval_s", "_clock", "_next")

    def __init__(
        self,
        interval_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval_s = interval_s
        self._clock = clock
        self._next = clock() + interval_s

    def due(self) -> bool:
        now = self._clock()
        if now < self._next:
            return False
        self._next = now + self._interval_s
        return True


class StepCounters:
    """Per-process decode step counters, emitted as one rate-limited event.

    The decode loop calls :meth:`begin_round` before its first step,
    :meth:`add_step` and :meth:`maybe_emit` once per step, and
    :meth:`end_round` after it exits. Every per-step call only bumps plain
    int/float fields and compares one time; the ``step.counters`` payload is
    built only when a record is emitted. All times are seconds of the
    caller's clock (the loop's ``time.perf_counter``).

    One record carries the deltas since the previous record:

    * ``steps``: decode steps begun;
    * ``row_steps``: decode rows those steps forwarded (a row that finished
      since the last page boundary still counts until the boundary drops it);
    * ``batch_min`` / ``batch_max``: fewest / most rows in one step;
    * ``decode_wall_s``: wall time spent inside decode loops;
    * ``window_s``: wall time since the previous record, decode or not;
    * ``total_steps``: steps reported by this instance, this record included.
    """

    __slots__ = (
        "_logger",
        "_log_event",
        "steps",
        "row_steps",
        "batch_min",
        "batch_max",
        "decode_wall_s",
        "total_steps",
        "_window_start_s",
        "_last_step_s",
    )

    def __init__(self, logger: logging.Logger, now_s: float) -> None:
        # Resolved here, not at module import: the package imports this
        # module before it defines ``log_event``.
        from . import log_event

        self._logger = logger
        self._log_event = log_event
        self.total_steps = 0
        self._last_step_s = now_s
        self._reset(now_s)

    def _reset(self, now_s: float) -> None:
        self.steps = 0
        self.row_steps = 0
        self.batch_min = _NO_ROWS
        self.batch_max = 0
        self.decode_wall_s = 0.0
        self._window_start_s = now_s

    def begin_round(self, now_s: float) -> None:
        """Start the decode-loop wall clock; time between rounds is not
        decode time."""
        self._last_step_s = now_s

    def add_step(self, now_s: float, rows: int) -> None:
        """Count one decode step of ``rows`` rows that begins at ``now_s``."""
        self.steps += 1
        self.row_steps += rows
        if rows > self.batch_max:
            self.batch_max = rows
        if rows < self.batch_min:
            self.batch_min = rows
        self.decode_wall_s += now_s - self._last_step_s
        self._last_step_s = now_s

    def end_round(self, now_s: float) -> None:
        """Charge the last step of the round, up to ``now_s``."""
        self.decode_wall_s += now_s - self._last_step_s
        self._last_step_s = now_s

    def maybe_emit(self, now_s: float, interval_s: float) -> bool:
        """Close the window when ``interval_s`` has elapsed since the
        previous record and a step was counted: emit one ``step.counters``
        record (skipped inside ``log_event`` when INFO is disabled), reset
        the deltas and return ``True``. Below the interval this is one
        subtraction and one compare.
        """
        if now_s - self._window_start_s < interval_s:
            return False
        if self.steps == 0:
            return False
        self._emit(now_s)
        return True

    def _emit(self, now_s: float) -> None:
        self.total_steps += self.steps
        self._log_event(
            self._logger,
            STEP_COUNTERS_EVENT,
            steps=self.steps,
            row_steps=self.row_steps,
            batch_min=self.batch_min,
            batch_max=self.batch_max,
            decode_wall_s=round(self.decode_wall_s, 6),
            window_s=round(now_s - self._window_start_s, 6),
            total_steps=self.total_steps,
        )
        self._reset(now_s)
