"""Eviction re-entry baseline fuzzer — plan invariant #2.

Invariant: after any number of evict + re-enter cycles, the reconstructed
prompt length equals the original prompt length plus the cumulative count
of tokens decoded across every prior cycle. Never more, never less.

    prompt_length == original_prompt_length + total_decoded_before_eviction

The old scheduler-split branch lost this invariant and grew prompt_length
geometrically: each eviction cycle appended the full current
``decoded_length`` to the history, but ``decoded_length`` already included
the tokens restored into the decoded-tokens buffer from the previous
re-entry. The fix is the ``reentry_decoded_baseline`` field — on eviction,
only ``decoded_length - baseline`` are genuinely new tokens.

This fuzzer tests the mathematical model of the evict + re-enter cycle
directly. When ``BoundaryExecutor._apply_evict`` lands in M4, an
additional integration test will run the real handler through the same
cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings, strategies as st


@dataclass
class _SeqModel:
    """Minimal SequenceEntry stand-in — only the fields the invariant touches."""

    original_prompt_length: int
    prompt_length: int
    decoded_length: int
    reentry_decoded_baseline: int
    total_decoded_before_eviction: int


def _fresh(original_prompt_length: int) -> _SeqModel:
    return _SeqModel(
        original_prompt_length=original_prompt_length,
        prompt_length=original_prompt_length,
        decoded_length=0,
        reentry_decoded_baseline=0,
        total_decoded_before_eviction=0,
    )


def _decode(seq: _SeqModel, n: int) -> None:
    """Produce `n` more tokens beyond the current decoded_length."""
    if n < 0:
        raise ValueError("cannot decode a negative number of tokens")
    seq.decoded_length += n


def _evict(seq: _SeqModel) -> None:
    """Append the genuinely-new tokens to history, drop GPU-side state.

    ``decoded_length - baseline`` is the count of tokens produced SINCE
    the last re-entry — every token at position ``< baseline`` was already
    baked into the reconstructed prompt_length, so appending those again
    would double-count and produce geometric prompt growth.
    """
    newly = seq.decoded_length - seq.reentry_decoded_baseline
    assert newly >= 0, (
        f"decoded_length ({seq.decoded_length}) dropped below baseline "
        f"({seq.reentry_decoded_baseline})"
    )
    seq.total_decoded_before_eviction += newly
    seq.decoded_length = 0
    seq.reentry_decoded_baseline = 0


def _reenter(seq: _SeqModel) -> None:
    """Simulate prefill re-entry: reconstruct prompt + restore decoded buffer.

    The reconstructed prompt contains the original prompt plus every token
    ever decoded on this sequence. The decoded-tokens buffer is restored
    with the same tokens so the model can continue producing output from
    where it left off — hence ``decoded_length`` AND ``baseline`` both
    start at ``total_decoded_before_eviction``.
    """
    seq.prompt_length = seq.original_prompt_length + seq.total_decoded_before_eviction
    seq.decoded_length = seq.total_decoded_before_eviction
    seq.reentry_decoded_baseline = seq.total_decoded_before_eviction


# ---------------------------------------------------------------------------
# Concrete sanity test
# ---------------------------------------------------------------------------


class TestConcreteMultiCycle:
    def test_three_cycles_no_geometric_growth(self) -> None:
        s = _fresh(original_prompt_length=100)

        _decode(s, 30)
        _evict(s)
        _reenter(s)
        # After cycle 1: prompt = 100 + 30 = 130, buffer restored to 30 tokens
        assert s.prompt_length == 130
        assert s.decoded_length == 30
        assert s.reentry_decoded_baseline == 30
        assert s.total_decoded_before_eviction == 30

        _decode(s, 20)  # now decoded_length = 50
        _evict(s)
        _reenter(s)
        # Cycle 2: ONLY the 20 new tokens should land in the history
        # Total new = 30 + 20 = 50; prompt = 100 + 50 = 150
        assert s.prompt_length == 150
        assert s.decoded_length == 50
        assert s.reentry_decoded_baseline == 50
        assert s.total_decoded_before_eviction == 50

        _decode(s, 7)  # decoded_length = 57
        _evict(s)
        _reenter(s)
        # Cycle 3: 7 more new tokens; total 57; prompt = 157
        assert s.prompt_length == 157
        assert s.total_decoded_before_eviction == 57

    def test_zero_new_decoded_in_cycle_is_fixed_point(self) -> None:
        """An evict-reenter with no new decoding is a fixed point.
        The plan tolerates this — sometimes a boundary fires while the
        sequence just re-entered and produced nothing yet."""
        s = _fresh(original_prompt_length=50)
        _decode(s, 10)
        _evict(s)
        _reenter(s)
        before_prompt = s.prompt_length
        before_total = s.total_decoded_before_eviction

        # Immediately evict + reenter again with zero new decoding
        _evict(s)
        _reenter(s)
        assert s.prompt_length == before_prompt
        assert s.total_decoded_before_eviction == before_total


# ---------------------------------------------------------------------------
# Property fuzzer
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    original_prompt_length=st.integers(min_value=1, max_value=4096),
    per_cycle_decoded=st.lists(
        st.integers(min_value=0, max_value=500),
        min_size=1,
        max_size=5,
    ),
)
def test_invariant_holds_after_every_cycle(
    original_prompt_length: int, per_cycle_decoded: list[int]
) -> None:
    """For any prompt length and any sequence of 1-5 cycles each producing
    0-500 new tokens, the invariant holds after every evict + re-enter:

        prompt_length == original_prompt_length + sum(per_cycle_decoded[:k])

    for every prefix k in [1, len(per_cycle_decoded)].
    """
    s = _fresh(original_prompt_length)
    cumulative = 0
    for n in per_cycle_decoded:
        _decode(s, n)
        _evict(s)
        _reenter(s)
        cumulative += n
        assert s.prompt_length == original_prompt_length + cumulative, (
            f"invariant broken after cycle with {n} new tokens: "
            f"prompt_length={s.prompt_length}, "
            f"original_prompt_length={original_prompt_length}, "
            f"cumulative={cumulative}"
        )
        assert s.total_decoded_before_eviction == cumulative
        assert s.reentry_decoded_baseline == cumulative
        assert s.decoded_length == cumulative  # buffer holds all prior tokens


@settings(max_examples=50, deadline=None)
@given(
    original_prompt_length=st.integers(min_value=1, max_value=1000),
    first_cycle_decoded=st.integers(min_value=0, max_value=500),
    extra_after_reenter=st.integers(min_value=1, max_value=500),
)
def test_decoding_after_reenter_does_not_retro_shift_prompt_until_next_evict(
    original_prompt_length: int,
    first_cycle_decoded: int,
    extra_after_reenter: int,
) -> None:
    """``prompt_length`` is FIXED for the duration of the decode phase and
    only moves at the next evict-then-reenter. This test runs one
    evict+reenter, then decodes more, and asserts ``prompt_length`` is
    unchanged until the NEXT evict+reenter (at which point it jumps by
    ``extra_after_reenter``).
    """
    s = _fresh(original_prompt_length)
    _decode(s, first_cycle_decoded)
    _evict(s)
    _reenter(s)
    prompt_after_first_reenter = s.prompt_length
    assert prompt_after_first_reenter == original_prompt_length + first_cycle_decoded

    # Decode more AFTER re-entry — prompt_length must stay put.
    _decode(s, extra_after_reenter)
    assert s.prompt_length == prompt_after_first_reenter

    # Next evict+reenter picks up the newly-decoded tokens.
    _evict(s)
    _reenter(s)
    assert s.prompt_length == (
        original_prompt_length + first_cycle_decoded + extra_after_reenter
    )
