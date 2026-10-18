"""Unit tests for the one-step-lookahead state machine.

The state object is what makes the reordering safe: it carries the row
identity of the in-flight step (so a composition change cannot mis-attribute
tokens), the per-row accept mask (so a retired row's computed token is
discarded rather than appended), and the per-row ``+1`` context offset the
next forward needs while the host has not yet advanced ``decoded_length``.
"""
import pytest

from batchgen.worker.decode_lookahead import (
    DecodeLookaheadError,
    DecodeLookaheadState,
    LookaheadArmError,
    LookaheadCompositionError,
    effective_cache_seqlens,
)


def test_nothing_in_flight_means_no_offsets():
    state = DecodeLookaheadState()
    assert not state.is_armed
    # The first step of a round, and every step after a flush, must build the
    # synchronous vector exactly.
    assert state.context_offsets([3, 7, 11]) == (0, 0, 0)
    assert state.context_offsets([]) == ()


def test_offsets_are_one_per_accepting_row():
    state = DecodeLookaheadState()
    state.arm([3, 7, 11], ["a", "b", "c"], [True, False, True], 0)
    assert state.is_armed
    # The retired row contributes 0: its context is frozen, which is what the
    # synchronous path does for a row it keeps forwarding until the boundary.
    assert state.context_offsets([3, 7, 11]) == (1, 0, 1)


def test_offsets_are_stable_across_repeated_reads():
    state = DecodeLookaheadState()
    state.arm([1], ["a"], [True], 1)
    assert state.context_offsets([1]) == state.context_offsets([1]) == (1,)


def test_a_row_set_change_while_a_step_is_in_flight_is_a_named_error():
    state = DecodeLookaheadState()
    state.arm([3, 7, 11], ["a", "b", "c"], [True, True, True], 0)
    for moved in ([3, 7], [3, 7, 11, 12], [11, 7, 3], [4, 7, 11]):
        with pytest.raises(LookaheadCompositionError):
            state.context_offsets(moved)
    # ... and the state is not consumed by the failed read, so the flush that
    # should have happened can still happen.
    assert state.is_armed


def test_arming_twice_is_a_named_error():
    state = DecodeLookaheadState()
    state.arm([1, 2], ["a", "b"], [True, True], 0)
    with pytest.raises(LookaheadArmError):
        state.arm([1, 2], ["a", "b"], [True, True], 1)


def test_arm_shape_mismatch_is_rejected():
    state = DecodeLookaheadState()
    with pytest.raises(LookaheadArmError):
        state.arm([1, 2], ["a"], [True, True], 0)
    with pytest.raises(LookaheadArmError):
        state.arm([1, 2], ["a", "b"], [True], 0)


def test_take_disarms_and_returns_the_step():
    state = DecodeLookaheadState()
    state.arm([5, 6], ["x", "y"], [True, False], 1)
    pending = state.take()
    assert pending.rows == (5, 6)
    assert pending.sequences == ("x", "y")
    assert pending.active == (True, False)
    assert pending.num_rows == 2
    assert pending.slot == 1
    assert not state.is_armed
    # A second take is a lost-flush bug, not an empty result.
    with pytest.raises(LookaheadArmError):
        state.take()


def test_take_then_arm_round_trips():
    state = DecodeLookaheadState()
    for step in range(4):
        state.arm([1, 2], ["a", "b"], [True, True], step % 2)
        assert state.context_offsets([1, 2]) == (1, 1)
        assert state.take().slot == step % 2
        assert state.context_offsets([1, 2]) == (0, 0)


def test_effective_cache_seqlens_adds_the_in_flight_token():
    assert effective_cache_seqlens([10, 20, 30], (1, 0, 1)) == [11, 20, 31]
    assert effective_cache_seqlens([], ()) == []


def test_effective_cache_seqlens_rejects_bad_shapes_and_offsets():
    with pytest.raises(DecodeLookaheadError):
        effective_cache_seqlens([1, 2], (1,))
    with pytest.raises(DecodeLookaheadError):
        # A two-step lag would mean two unapplied tokens per row; the design
        # allows exactly one, so this must fail loud rather than silently
        # advance the context by 2.
        effective_cache_seqlens([1], (2,))
    with pytest.raises(DecodeLookaheadError):
        effective_cache_seqlens([1], (-1,))


def test_the_pending_record_is_immutable():
    state = DecodeLookaheadState()
    state.arm([1], ["a"], [True], 0)
    pending = state.take()
    with pytest.raises(Exception):
        pending.rows = (2,)
