"""State transitions of the per-turn API's session registry.

open --turn--> turn_in_flight --done--> idle --turn--> ...; open/idle --close-->
closed; open/idle/turn_in_flight --abort--> aborted. closed and aborted are
terminal, ids are not reused while the registry holds their tombstone,
and there is no implicit open.

agent_sessions.py is stdlib-only, but importing it as batchgen.server.* would
run the server package __init__ (the GPU stack), so it is loaded by path.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "batchgen" / "server" / "agent_sessions.py"
_spec = importlib.util.spec_from_file_location("agent_sessions_under_test", _PATH)
sessions = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sessions
_spec.loader.exec_module(sessions)

SessionRegistry = sessions.SessionRegistry
SessionState = sessions.SessionState
TurnError = sessions.TurnError


def _error(fn, *args):
    with pytest.raises(TurnError) as info:
        fn(*args)
    return info.value.status_code, info.value.code


def test_turn_lifecycle_open_in_flight_idle_closed():
    reg = SessionRegistry()
    record = reg.open("t1", "s1")
    assert record.state is SessionState.OPEN
    assert reg.begin_turn("t1", "s1", None, "turn-a").state is SessionState.TURN_IN_FLIGHT
    assert record.in_flight_turn_id == "turn-a"
    reg.end_turn("s1", "turn-a")
    assert record.state is SessionState.IDLE and record.in_flight_turn_id is None
    reg.begin_turn("t1", "s1", None, "turn-b")
    reg.end_turn("s1", "turn-b")
    assert record.turns == 2
    closed = reg.close("s1", "end_of_session")
    assert [r.session_id for r in closed] == ["s1"]
    assert record.state is SessionState.CLOSED
    assert record.to_dict()["end_reason"] == "end_of_session"


def test_unknown_session_gets_404_and_no_implicit_open():
    reg = SessionRegistry()
    assert _error(reg.begin_turn, "t1", "nope", None, "turn-a") == (404, "session_not_found")
    assert reg.get("nope") is None
    assert _error(reg.close, "nope", "end_of_session") == (404, "session_not_found")
    assert _error(reg.abort, "nope", None) == (404, "session_not_found")


def test_closed_and_aborted_sessions_accept_no_turn_and_ids_are_not_reused():
    reg = SessionRegistry()
    reg.open("t1", "s1")
    reg.close("s1", "end_of_session")
    assert _error(reg.begin_turn, "t1", "s1", None, "x") == (409, "session_closed")
    assert _error(reg.close, "s1", "end_of_session") == (409, "session_closed")
    assert _error(reg.open, "t1", "s1") == (409, "session_exists")
    reg.open("t1", "s2")
    reg.abort("s2", "tool_crash")
    assert _error(reg.begin_turn, "t1", "s2", None, "x") == (409, "session_aborted")
    assert _error(reg.abort, "s2", None) == (409, "session_aborted")


def test_one_turn_in_flight_per_session_and_close_waits_for_it():
    reg = SessionRegistry()
    reg.open("t1", "s1")
    reg.begin_turn("t1", "s1", None, "turn-a")
    assert _error(reg.begin_turn, "t1", "s1", None, "turn-b") == (409, "turn_in_flight")
    assert _error(reg.close, "s1", "end_of_session") == (409, "turn_in_flight")
    assert reg.get("s1").state is SessionState.TURN_IN_FLIGHT


def test_turn_identity_must_match_the_opened_session():
    reg = SessionRegistry()
    reg.open("t1", "main")
    reg.open("t1", "sub", "main")
    assert _error(reg.begin_turn, "t2", "main", None, "x") == (400, "session_identity_mismatch")
    assert _error(reg.begin_turn, "t1", "sub", None, "x") == (400, "session_identity_mismatch")
    assert reg.begin_turn("t1", "sub", "main", "x").state is SessionState.TURN_IN_FLIGHT


def test_parent_must_exist_in_the_same_trajectory():
    reg = SessionRegistry()
    assert _error(reg.open, "t1", "sub", "ghost") == (404, "parent_session_not_found")
    reg.open("t1", "main")
    assert _error(reg.open, "t2", "sub", "main") == (400, "parent_session_other_trajectory")
    assert reg.get("sub") is None


def test_abort_returns_the_in_flight_turn_and_a_late_end_turn_keeps_aborted():
    reg = SessionRegistry()
    reg.open("t1", "s1")
    reg.begin_turn("t1", "s1", None, "turn-a")
    record, turn_id = reg.abort("s1", "harness_error")
    assert turn_id == "turn-a"
    assert record.state is SessionState.ABORTED and record.end_reason == "harness_error"
    reg.end_turn("s1", "turn-a")
    assert record.state is SessionState.ABORTED
    reg.open("t1", "s2")
    assert reg.abort("s2", None)[1] is None


def test_end_of_trajectory_closes_every_live_session_of_the_trajectory():
    reg = SessionRegistry()
    reg.open("t1", "main")
    reg.open("t1", "sub-a", "main")
    reg.open("t1", "sub-b", "main")
    reg.open("t2", "other")
    reg.close("sub-a", "end_of_session")
    closed = reg.close("main", "end_of_trajectory")
    assert sorted(r.session_id for r in closed) == ["main", "sub-b"]
    assert reg.get("sub-a").end_reason == "end_of_session"
    assert reg.get("other").state is SessionState.OPEN


def test_end_of_trajectory_is_all_or_nothing_while_a_turn_is_in_flight():
    reg = SessionRegistry()
    reg.open("t1", "main")
    reg.open("t1", "sub", "main")
    reg.begin_turn("t1", "sub", "main", "turn-a")
    assert _error(reg.close, "main", "end_of_trajectory") == (409, "turn_in_flight")
    assert reg.get("main").state is SessionState.OPEN
    assert reg.get("sub").state is SessionState.TURN_IN_FLIGHT


def test_abort_after_the_completion_is_routed_is_already_completed():
    reg = SessionRegistry()
    reg.open("t1", "s1")
    reg.begin_turn("t1", "s1", None, "turn-a")
    reg.complete_turn("s1", "turn-a")
    assert _error(reg.abort, "s1", "late") == (409, "already_completed")
    record = reg.get("s1")
    assert record.state is SessionState.TURN_IN_FLIGHT and record.end_reason is None
    reg.end_turn("s1", "turn-a")
    assert record.state is SessionState.IDLE
    # The next turn is abortable again until its own completion is routed.
    reg.begin_turn("t1", "s1", None, "turn-b")
    assert reg.abort("s1", "tool_crash")[1] == "turn-b"
    assert record.state is SessionState.ABORTED


def test_complete_turn_ignores_a_turn_that_is_not_in_flight():
    reg = SessionRegistry()
    reg.open("t1", "s1")
    reg.begin_turn("t1", "s1", None, "turn-a")
    reg.complete_turn("s1", "turn-stale")
    reg.complete_turn("ghost", "turn-a")
    assert reg.abort("s1", None)[1] == "turn-a"


def test_terminal_sessions_are_bounded_by_default():
    # A long run of short sessions must not grow the registry without bound:
    # with the defaults, at most 65,536 ended sessions are retained.
    reg = SessionRegistry()
    for i in range(65536 + 100):
        reg.open("t", f"s{i}")
        reg.close(f"s{i}", "end_of_session")
    retained = len(reg._sessions) + len(getattr(reg, "_terminal", {}))
    assert retained <= 65536
    assert reg.get("s0") is None and reg.get(f"s{65536 + 99}") is not None


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_tombstone_blocks_reuse_until_ttl_then_the_id_is_forgotten():
    clock = _Clock()
    reg = SessionRegistry(terminal_ttl_s=60.0, max_terminal=100, clock=clock)
    reg.open("t1", "s1")
    reg.close("s1", "end_of_session")
    reg.open("t1", "s2")
    reg.abort("s2", "tool_crash")
    clock.now += 59.0
    assert _error(reg.open, "t1", "s1") == (409, "session_exists")
    assert _error(reg.begin_turn, "t1", "s2", None, "x") == (409, "session_aborted")
    assert reg.get("s1").state is SessionState.CLOSED
    clock.now += 1.0
    assert reg.get("s1") is None and reg.get("s2") is None
    assert _error(reg.begin_turn, "t1", "s1", None, "x") == (404, "session_not_found")
    assert reg.open("t1", "s1").state is SessionState.OPEN


def test_tombstone_cap_evicts_the_oldest_and_never_live_sessions():
    clock = _Clock()
    reg = SessionRegistry(terminal_ttl_s=3600.0, max_terminal=2, clock=clock)
    reg.open("t1", "live")
    for sid in ("a", "b", "c"):
        reg.open("t1", sid)
        reg.close(sid, "end_of_session")
        clock.now += 1.0
    assert reg.get("a") is None
    assert [reg.get(s).state for s in ("b", "c")] == [SessionState.CLOSED] * 2
    assert reg.get("live").state is SessionState.OPEN
    # end_of_trajectory retires every live session of the trajectory at once.
    closed = reg.close("live", "end_of_trajectory")
    assert [r.session_id for r in closed] == ["live"]
    assert reg.get("live").state is SessionState.CLOSED and reg.get("b") is None


def test_an_aborted_in_flight_session_still_ends_its_turn_after_retirement():
    reg = SessionRegistry(max_terminal=1)
    reg.open("t1", "s1")
    reg.begin_turn("t1", "s1", None, "turn-a")
    record, turn_id = reg.abort("s1", None)
    reg.end_turn("s1", turn_id)
    assert record.state is SessionState.ABORTED


def test_registry_limits_are_validated():
    with pytest.raises(ValueError):
        SessionRegistry(terminal_ttl_s=0)
    with pytest.raises(ValueError):
        SessionRegistry(max_terminal=0)
