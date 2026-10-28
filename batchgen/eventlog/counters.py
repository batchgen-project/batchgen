# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Hot-path helpers for rate-limited progress events."""

from __future__ import annotations

import time
from collections.abc import Callable


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
