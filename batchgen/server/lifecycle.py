"""The one stop request of a server process, observed by every phase.

A server stops in one of two lifecycle phases, and the request reaches both
through this object:

* **Start-up** runs synchronously inside the HTTP lifespan, so the event loop
  is not running while weights load or the weight store is staged, and the
  HTTP server's own signal handler only records that it should exit once
  start-up returns. Start-up therefore polls :meth:`StopRequest.raise_if_set`
  at its phase boundaries (and the staging copy between segments) and unwinds
  with :class:`StartupCancelled`; nothing is spawned after the request.
* **Serving**: the server keeps answering HTTP while the scheduler and the
  workers tear down (``/health`` reports ``shutting_down``), and only stops
  the HTTP listener once teardown has finished. A second signal stops the
  listener at once; teardown still completes before the process exits.
"""

from __future__ import annotations

import threading
from typing import Optional


class StartupCancelled(Exception):
    """Start-up unwound because the server was asked to stop."""

    def __init__(self, phase: str, reason: str) -> None:
        super().__init__(f"startup cancelled during {phase}: {reason}")
        self.phase = phase
        self.reason = reason


class StopRequest:
    """A thread-safe, set-once stop request with the reason it was made."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._reason: Optional[str] = None

    def request(self, reason: str) -> bool:
        """Record the request; True only for the first one."""
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._event.set()
            return True

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def event(self) -> threading.Event:
        """The underlying event, for loops that poll it between work items."""
        return self._event

    @property
    def reason(self) -> Optional[str]:
        with self._lock:
            return self._reason

    def raise_if_set(self, phase: str) -> None:
        if self._event.is_set():
            raise StartupCancelled(phase, self.reason or "stop requested")


__all__ = ["StartupCancelled", "StopRequest"]
