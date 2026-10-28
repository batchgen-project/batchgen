"""Server-side registry of agent sessions for the per-turn API.

An agent harness opens a session, sends its model turns one at a time through
POST /v1/turns, and closes or aborts the session out of band. The registry
records each session's lifecycle state and validates every turn against it:

    open --turn--> turn_in_flight --done/failed--> idle --turn--> ...
    open | idle --close--> closed
    open | idle | turn_in_flight --abort--> aborted

``closed`` and ``aborted`` are terminal. A session id is never reused, and a
turn is accepted only for a session that was opened explicitly (no implicit
open). Holding KV across turns is not done here; the registry only keeps state.

The registry is confined to the server's event loop: the HTTP handlers are its
only callers, so it takes no lock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


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


class SessionRegistry:
    """Lifecycle state of every session this server has seen."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}

    def get(self, session_id: str) -> Optional[SessionRecord]:
        return self._sessions.get(session_id)

    def open(
        self,
        traj_id: str,
        session_id: str,
        parent_session_id: Optional[str] = None,
    ) -> SessionRecord:
        existing = self._sessions.get(session_id)
        if existing is not None:
            raise TurnError(
                409, "session_exists",
                f"session {session_id!r} already exists "
                f"(state={existing.state.value}); session ids are not reused",
            )
        if parent_session_id is not None:
            parent = self._sessions.get(parent_session_id)
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
        record.turns += 1
        record._set(SessionState.TURN_IN_FLIGHT)
        return record

    def end_turn(self, session_id: str, turn_id: str) -> None:
        """A turn finished (answered or failed): its session becomes idle.

        A session that was aborted while the turn ran stays aborted.
        """
        record = self._sessions.get(session_id)
        if record is None or record.in_flight_turn_id != turn_id:
            return
        record.in_flight_turn_id = None
        if record.state == SessionState.TURN_IN_FLIGHT:
            record._set(SessionState.IDLE)

    def close(self, session_id: str, reason: str) -> List[SessionRecord]:
        """Close a session; ``end_of_trajectory`` closes its whole trajectory.

        All-or-nothing: if any session to close has a turn in flight, nothing
        is closed. Returns the sessions closed by this call.
        """
        record = self._require(session_id)
        if record.state in _TERMINAL:
            raise TurnError(
                409, f"session_{record.state.value}",
                f"session {session_id!r} is already {record.state.value}",
            )
        if reason == "end_of_trajectory":
            targets = [
                r for r in self._sessions.values()
                if r.traj_id == record.traj_id and r.state not in _TERMINAL
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
        return targets

    def abort(
        self, session_id: str, reason: Optional[str]
    ) -> Tuple[SessionRecord, Optional[str]]:
        """Abort a session; return it and the id of the turn it had in flight."""
        record = self._require(session_id)
        if record.state in _TERMINAL:
            raise TurnError(
                409, f"session_{record.state.value}",
                f"session {session_id!r} is already {record.state.value}",
            )
        in_flight = record.in_flight_turn_id
        record.in_flight_turn_id = None
        record.end_reason = reason or "aborted"
        record._set(SessionState.ABORTED)
        return record, in_flight

    def _require(self, session_id: str) -> SessionRecord:
        record = self._sessions.get(session_id)
        if record is None:
            raise TurnError(
                404, "session_not_found",
                f"session {session_id!r} was never opened; "
                "POST /v1/sessions first",
            )
        return record
