"""Thread-safe server health state for watchdog and /health endpoint integration."""

from __future__ import annotations

import threading
from typing import Tuple


class ServerHealthState:
    """Tracks server health across threads (watchdog, startup timer, uvicorn handler).

    Set unhealthy from any thread; the /health HTTP endpoint reads this to
    return 503 with a reason string so external orchestrators (K8s, spot
    instance managers) can detect failure.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._healthy = True
        self._reason = ""
        self._startup_complete = False

    def set_unhealthy(self, reason: str) -> None:
        """Mark the server as unhealthy. First call wins (reason is not overwritten)."""
        with self._lock:
            if self._healthy:
                self._healthy = False
                self._reason = reason

    def is_healthy(self) -> Tuple[bool, str]:
        """Return (healthy, reason). Thread-safe."""
        with self._lock:
            return self._healthy, self._reason

    def mark_startup_complete(self) -> None:
        """Called when the server is fully ready (lifespan yield reached)."""
        with self._lock:
            self._startup_complete = True

    def is_startup_complete(self) -> bool:
        with self._lock:
            return self._startup_complete
