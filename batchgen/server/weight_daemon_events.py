"""Server-log events of the node's distributed-weight daemon.

The daemon keeps a bounded, sequence-numbered queue of what it did (config,
network readiness, failures, cohort swaps, worker admission, stalled
acquires, stop). :class:`WeightDaemonWatcher` drains that queue on one thread
and logs each entry as a named ``weight_daemon.*`` event.

Each record carries the event name and its fields as the ``bg_event`` and
``bg_data`` record attributes, which a structured event handler reads; its
message is the same event rendered as ``name k=v ...`` for plain logging.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# One poll blocks in the daemon, without the GIL, for at most this long when
# nothing happens; it returns as soon as an event is pushed. It is also how
# long the watcher may take to notice stop().
DAEMON_POLL_TIMEOUT_S = 1.0
# stop() waits for the poll in flight plus the logging of what it returned.
_STOP_JOIN_TIMEOUT_S = 2 * DAEMON_POLL_TIMEOUT_S

# Daemon event kind -> (event name, log level).
EVENT_NAMES: Dict[str, Tuple[str, int]] = {
    "config": ("weight_daemon.config", logging.INFO),
    "network_ready": ("weight_daemon.network_ready", logging.INFO),
    "failed": ("weight_daemon.failed", logging.ERROR),
    "cohort_draining": ("weight_daemon.cohort.draining", logging.INFO),
    "cohort_admitting": ("weight_daemon.cohort.admitting", logging.INFO),
    "cohort_faulted": ("weight_daemon.cohort.faulted", logging.WARNING),
    "worker_admitted": ("weight_daemon.cohort.worker_admitted", logging.INFO),
    "hello_rejected": ("weight_daemon.hello.rejected", logging.WARNING),
    "acquire_stalled": ("weight_daemon.acquire.stalled", logging.WARNING),
    "stopped": ("weight_daemon.stopped", logging.INFO),
}
# Not a daemon kind: the watcher's own report of events lost to overflow.
EVENTS_DROPPED = "weight_daemon.events.dropped"
# A kind this module does not know yet; logged rather than lost.
UNKNOWN_EVENT = "weight_daemon.unknown"


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        if value and not any(c.isspace() or c in '="' for c in value):
            return value
        return json.dumps(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def render_event(name: str, data: Dict[str, Any]) -> str:
    """``name k=v ...``, the message body of an event record."""
    fields = " ".join(f"{key}={_render_value(value)}" for key, value in data.items())
    return f"{name} {fields}" if fields else name


def event_fields(event: Dict[str, Any]) -> Tuple[str, int, Dict[str, Any]]:
    """The event name, log level and data of one polled daemon event.

    A JSON object in ``detail`` (the config, or a swap report) is merged into
    the data; the event's own fields win on a clash. Detail that is not a JSON
    object is kept as the ``detail`` string.
    """
    kind = event.get("kind")
    name, level = EVENT_NAMES.get(kind, (UNKNOWN_EVENT, logging.WARNING))
    data = {key: value for key, value in event.items() if key not in ("kind", "detail")}
    if name == UNKNOWN_EVENT:
        data["kind"] = kind
    detail = event.get("detail")
    if detail:
        try:
            parsed = json.loads(detail)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                data.setdefault(key, value)
        else:
            data["detail"] = detail
    return name, level, data


class WeightDaemonWatcher:
    """Logs the daemon's events from one thread until it stops.

    The daemon must outlive the watcher: a poll runs inside the daemon with
    the GIL released, so :meth:`stop` is called after the daemon's own stop
    and before the daemon object is dropped.
    """

    def __init__(self, daemon: Any, event_logger: logging.Logger) -> None:
        self._daemon = daemon
        self._logger = event_logger
        self._last_seq = 0
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="weight-daemon-events", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Join the thread, then drain what it left without waiting.

        The daemon's stop pushes its last event before returning, so the
        final drain logs ``weight_daemon.stopped`` if the thread did not.
        Never raises: shutdown continues whatever happens here.
        """
        self._stop_requested.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
            if thread.is_alive():
                # It still drains; a second drainer would log events twice.
                self._logger.warning(
                    "weight daemon event watcher did not stop within %.1fs",
                    _STOP_JOIN_TIMEOUT_S,
                )
                return
            self._thread = None
        try:
            self._drain(0.0)
        except Exception:
            self._logger.exception("Final drain of weight daemon events failed")

    def _run(self) -> None:
        try:
            while not self._stop_requested.is_set():
                if self._drain(DAEMON_POLL_TIMEOUT_S):
                    return
        except Exception:
            self._logger.exception("Weight daemon event watcher failed")

    def _drain(self, timeout_s: float) -> bool:
        """Log every event after the last one seen; True once stopped is."""
        events, dropped = self._daemon.poll_events(self._last_seq, timeout_s)
        if dropped:
            self._emit(
                EVENTS_DROPPED,
                logging.WARNING,
                {"dropped": int(dropped), "after_seq": self._last_seq},
            )
        stopped = False
        for event in events:
            self._last_seq = max(self._last_seq, int(event["seq"]))
            stopped = stopped or event.get("kind") == "stopped"
            try:
                name, level, data = event_fields(event)
                self._emit(name, level, data)
            except Exception:
                # One event that cannot be rendered or logged must not end
                # the watcher: report it and keep draining the rest.
                self._logger.exception(
                    "Could not log weight daemon event seq=%s",
                    event.get("seq"),
                )
        return stopped

    def _emit(self, name: str, level: int, data: Dict[str, Any]) -> None:
        self._logger.log(
            level,
            "%s",
            render_event(name, data),
            extra={"bg_event": name, "bg_data": data},
            stacklevel=2,
        )
