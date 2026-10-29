"""Server-side registry of agent sessions for the per-turn API.

An agent harness opens a session, sends its model turns one at a time through
POST /v1/turns, and closes or aborts the session out of band. The registry
records each session's lifecycle state and validates every turn against it:

    open --turn--> turn_in_flight --done/failed--> idle --turn--> ...
    open | idle --close--> closed
    open | idle | turn_in_flight --abort--> aborted

``closed`` and ``aborted`` are terminal. A turn is accepted only for a
session that was opened explicitly (no implicit open).

A terminal session is kept as a tombstone so its id cannot be reused, for
``terminal_ttl_s`` seconds (default 3600) after it ended or until
``max_terminal`` (default 65536) later sessions have ended, whichever comes
first. Past that window the id is forgotten: a turn for it gets
``404 session_not_found`` and it can be opened again. This bounds the
registry's memory under a long run of short sessions. Holding KV across turns is not done here; the registry only keeps state.

A turn in flight has its own terminal point: when the scheduler routes its
worker completion (``complete_turn``) the turn is finished even though its
HTTP response has not been sent yet. An abort in that window gets 409
``already_completed`` and leaves the session as it is: the turn's answer is
returned, and the session goes idle when the handler ends the turn.

The registry is confined to the server's event loop: the HTTP handlers are its
only callers, so it takes no lock.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class TurnError(Exception):
    """A turn or session request the server answers with an HTTP error."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message

    def detail(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}


class SessionState(str, Enum):
    OPEN = "open"
    TURN_IN_FLIGHT = "turn_in_flight"
    IDLE = "idle"
    CLOSED = "closed"
    ABORTED = "aborted"


_TERMINAL = (SessionState.CLOSED, SessionState.ABORTED)
_ACCEPTS_TURN = (SessionState.OPEN, SessionState.IDLE)


@dataclass
class SessionRecord:
    traj_id: str
    session_id: str
    parent_session_id: Optional[str]
    state: SessionState = SessionState.OPEN
    turns: int = 0
    in_flight_turn_id: Optional[str] = None
    # The in-flight turn's completion was routed; only its response remains.
    in_flight_turn_completed: bool = False
    end_reason: Optional[str] = None
    opened_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": "session",
            "traj_id": self.traj_id,
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "state": self.state.value,
            "turns": self.turns,
            "in_flight_turn_id": self.in_flight_turn_id,
            "end_reason": self.end_reason,
        }

    def _set(self, state: SessionState) -> None:
        self.state = state
        self.updated_at = time.time()


DEFAULT_TERMINAL_TTL_S = 3600.0
DEFAULT_MAX_TERMINAL = 65536


class SessionRegistry:
    """Lifecycle state of every live session and of recently ended ones."""

    def __init__(
        self,
        *,
        terminal_ttl_s: float = DEFAULT_TERMINAL_TTL_S,
        max_terminal: int = DEFAULT_MAX_TERMINAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not terminal_ttl_s > 0:
            raise ValueError(f"terminal_ttl_s must be > 0, got {terminal_ttl_s}")
        if max_terminal < 1:
            raise ValueError(f"max_terminal must be >= 1, got {max_terminal}")
        self._terminal_ttl_s = float(terminal_ttl_s)
        self._max_terminal = int(max_terminal)
        self._clock = clock
        # Live sessions: open, turn_in_flight, idle.
        self._sessions: Dict[str, SessionRecord] = {}
        # Tombstones of closed/aborted sessions, oldest first:
        # session_id -> (ended at, by ``clock``; record).
        self._terminal: "OrderedDict[str, Tuple[float, SessionRecord]]" = OrderedDict()

    def get(self, session_id: str) -> Optional[SessionRecord]:
        self._prune()
        return self._lookup(session_id)

    def _lookup(self, session_id: str) -> Optional[SessionRecord]:
        record = self._sessions.get(session_id)
        if record is None:
            tombstone = self._terminal.get(session_id)
            record = None if tombstone is None else tombstone[1]
        return record

    def _retire(self, record: SessionRecord) -> None:
        """Move a session that just became terminal to the tombstones."""
        del self._sessions[record.session_id]
        self._terminal[record.session_id] = (self._clock(), record)
        self._prune()

    def _prune(self) -> None:
        """Drop tombstones past the TTL, then the oldest beyond the cap."""
        cutoff = self._clock() - self._terminal_ttl_s
        while self._terminal:
            ended_at, _ = next(iter(self._terminal.values()))
            if ended_at > cutoff:
                break
            self._terminal.popitem(last=False)
        while len(self._terminal) > self._max_terminal:
            self._terminal.popitem(last=False)

    def open(
        self,
        traj_id: str,
        session_id: str,
        parent_session_id: Optional[str] = None,
    ) -> SessionRecord:
        self._prune()
        existing = self._lookup(session_id)
        if existing is not None:
            raise TurnError(
                409, "session_exists",
                f"session {session_id!r} already exists "
                f"(state={existing.state.value}); session ids are not reused "
                "while the server still holds them",
            )
        if parent_session_id is not None:
            parent = self._lookup(parent_session_id)
            if parent is None:
                raise TurnError(
                    404, "parent_session_not_found",
                    f"parent session {parent_session_id!r} was never opened",
                )
            if parent.traj_id != traj_id:
                raise TurnError(
                    400, "parent_session_other_trajectory",
                    f"parent session {parent_session_id!r} belongs to trajectory "
                    f"{parent.traj_id!r}, not {traj_id!r}",
                )
        record = SessionRecord(
            traj_id=traj_id,
            session_id=session_id,
            parent_session_id=parent_session_id,
        )
        self._sessions[session_id] = record
        return record

    def begin_turn(
        self,
        traj_id: str,
        session_id: str,
        parent_session_id: Optional[str],
        turn_id: str,
    ) -> SessionRecord:
        """Validate a turn's identity and mark its session turn_in_flight."""
        self._prune()
        record = self._require(session_id)
        if record.traj_id != traj_id:
            raise TurnError(
                400, "session_identity_mismatch",
                f"session {session_id!r} belongs to trajectory "
                f"{record.traj_id!r}, but the turn names {traj_id!r}",
            )
        if record.parent_session_id != parent_session_id:
            raise TurnError(
                400, "session_identity_mismatch",
                f"session {session_id!r} was opened with parent "
                f"{record.parent_session_id!r}, but the turn names "
                f"{parent_session_id!r}",
            )
        if record.state == SessionState.TURN_IN_FLIGHT:
            raise TurnError(
                409, "turn_in_flight",
                f"session {session_id!r} already has turn "
                f"{record.in_flight_turn_id!r} in flight",
            )
        if record.state not in _ACCEPTS_TURN:
            raise TurnError(
                409, f"session_{record.state.value}",
                f"session {session_id!r} is {record.state.value}; "
                "it accepts no more turns",
            )
        record.in_flight_turn_id = turn_id
        record.in_flight_turn_completed = False
        record.turns += 1
        record._set(SessionState.TURN_IN_FLIGHT)
        return record

    def complete_turn(self, session_id: str, turn_id: str) -> None:
        """The turn's worker completion was routed: it can no longer be aborted.

        Called synchronously by the scheduler before it resolves the turn's
        future, so no abort can run between the two.
        """
        record = self._lookup(session_id)
        if record is None or record.in_flight_turn_id != turn_id:
            return
        record.in_flight_turn_completed = True

    def end_turn(self, session_id: str, turn_id: str) -> None:
        """A turn finished (answered or failed): its session becomes idle.

        A session that was aborted while the turn ran stays aborted.
        """
        record = self._lookup(session_id)
        if record is None or record.in_flight_turn_id != turn_id:
            return
        record.in_flight_turn_id = None
        record.in_flight_turn_completed = False
        if record.state == SessionState.TURN_IN_FLIGHT:
            record._set(SessionState.IDLE)

    def close(self, session_id: str, reason: str) -> List[SessionRecord]:
        """Close a session; ``end_of_trajectory`` closes its whole trajectory.

        All-or-nothing: if any session to close has a turn in flight, nothing
        is closed. Returns the sessions closed by this call.
        """
        self._prune()
        record = self._require(session_id)
        if record.state in _TERMINAL:
            raise TurnError(
                409, f"session_{record.state.value}",
                f"session {session_id!r} is already {record.state.value}",
            )
        if reason == "end_of_trajectory":
            targets = [
                r for r in self._sessions.values() if r.traj_id == record.traj_id
            ]
        else:
            targets = [record]
        busy = [r.session_id for r in targets if r.state == SessionState.TURN_IN_FLIGHT]
        if busy:
            raise TurnError(
                409, "turn_in_flight",
                f"cannot close while a turn is in flight in session(s) {busy}; "
                "wait for the turn or abort the session",
            )
        for target in targets:
            target.end_reason = reason
            target._set(SessionState.CLOSED)
            self._retire(target)
        return targets

    def abort(
        self, session_id: str, reason: Optional[str]
    ) -> Tuple[SessionRecord, Optional[str]]:
        """Abort a session; return it and the id of the turn it had in flight.

        Raises 409 ``already_completed`` (and changes nothing) when the turn in
        flight has had its completion routed: that turn returns its answer.
        """
        self._prune()
        record = self._require(session_id)
        if record.state in _TERMINAL:
            raise TurnError(
                409, f"session_{record.state.value}",
                f"session {session_id!r} is already {record.state.value}",
            )
        if record.in_flight_turn_completed:
            raise TurnError(
                409, "already_completed",
                f"turn {record.in_flight_turn_id!r} of session {session_id!r} "
                "already completed and its response is being returned; the "
                "session was not aborted (abort it again once the turn has "
                "returned to end it)",
            )
        in_flight = record.in_flight_turn_id
        record.in_flight_turn_id = None
        record.end_reason = reason or "aborted"
        record._set(SessionState.ABORTED)
        self._retire(record)
        return record, in_flight

    def _require(self, session_id: str) -> SessionRecord:
        record = self._lookup(session_id)
        if record is None:
            raise TurnError(
                404, "session_not_found",
                f"session {session_id!r} was never opened; "
                "POST /v1/sessions first",
            )
        return record
