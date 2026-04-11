"""TorchResponseSink — wraps main's response queue + incremental writer.

Main sends completed-sequence payloads through ``self._response_queue``
(an ``mp.Queue``) and also writes to an ``_incremental_writer`` when
available. The adapter takes both as optional constructor args and
forwards every :meth:`put` call through both.

Missing response queue or incremental writer is treated as a no-op
for that half — main already handles that case at startup (e.g.,
single-process test runs without the queue).
"""

from __future__ import annotations

from typing import Any


class TorchResponseSink:
    """Production adapter for :class:`ResponseSinkBackend`."""

    def __init__(
        self,
        response_queue: Any | None = None,
        incremental_writer: Any | None = None,
    ) -> None:
        self._queue = response_queue
        self._writer = incremental_writer

    def put(self, uuid: str, payload: dict[str, Any]) -> None:
        # Forward to the response queue (rank-0 only in main's flow).
        if self._queue is not None:
            try:
                self._queue.put_nowait({"uuid": uuid, **payload})
            except Exception:
                # Queue full or closed — fall through to the writer.
                pass
        # Also forward to the incremental writer when present.
        if self._writer is not None:
            write = getattr(self._writer, "write_completion", None) or getattr(
                self._writer, "write", None
            )
            if write is not None:
                try:
                    write(uuid, payload)
                except Exception:
                    # Writer errors should not abort the hot path in
                    # production; main's existing logging captures them.
                    pass


__all__ = ["TorchResponseSink"]
