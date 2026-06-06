"""Unit tests for `batchgen.worker.completion`.

Real fixtures only — no mocks of `SequenceEntry` per the Phase A §G
no-hack rule. Tests run CPU-only and require no GPU.
"""

from __future__ import annotations

import pytest

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.completion import CompletionContext, CompletionHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_seq(
    uuid: str = "seq-1",
    global_idx: int = 0,
    prompt_length: int = 8,
    max_decode_length: int = 16,
    decoded_length: int = 0,
    eos_reached: bool = False,
    rep_detected: bool = False,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=max_decode_length,
    )
    seq.decoded_length = decoded_length
    seq.current_context_length = prompt_length + decoded_length
    seq.eos_reached = eos_reached
    seq._rep_detected = rep_detected
    return seq


@pytest.fixture
def ctx_strict() -> CompletionContext:
    """Production-like context: ignore_eos=False, real model_context_length."""
    return CompletionContext(
        ignore_eos=False,
        eos_token_ids=frozenset({0, 2, 1024}),
        model_context_length=4096,
        rank=0,
    )


@pytest.fixture
def ctx_ignore_eos() -> CompletionContext:
    """ignore_eos=True context for the override path."""
    return CompletionContext(
        ignore_eos=True,
        eos_token_ids=frozenset({0, 2, 1024}),
        model_context_length=4096,
        rank=0,
    )


# ---------------------------------------------------------------------------
# CompletionContext dataclass behavior
# ---------------------------------------------------------------------------

def test_context_is_frozen(ctx_strict):
    with pytest.raises((AttributeError, Exception)):
        ctx_strict.rank = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# should_stop_at_eos
# ---------------------------------------------------------------------------

def test_should_stop_at_eos_hit(ctx_strict):
    assert CompletionHandler.should_stop_at_eos(ctx_strict, 2) is True
    assert CompletionHandler.should_stop_at_eos(ctx_strict, 1024) is True


def test_should_stop_at_eos_miss(ctx_strict):
    assert CompletionHandler.should_stop_at_eos(ctx_strict, 42) is False


def test_should_stop_at_eos_ignored_when_ignore_eos(ctx_ignore_eos):
    # Even EOS tokens return False under ignore_eos
    assert CompletionHandler.should_stop_at_eos(ctx_ignore_eos, 2) is False
    assert CompletionHandler.should_stop_at_eos(ctx_ignore_eos, 1024) is False


# ---------------------------------------------------------------------------
# is_sequence_completed
# ---------------------------------------------------------------------------

def test_is_sequence_completed_max_decode(ctx_strict):
    seq = _make_seq(max_decode_length=16, decoded_length=16)
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is True


def test_is_sequence_completed_context_limit(ctx_strict):
    # prompt_length=8, decoded_length=4088, ctx=4096 == model_context_length=4096
    seq = _make_seq(prompt_length=8, decoded_length=4088, max_decode_length=10_000)
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is True


def test_is_sequence_completed_eos(ctx_strict):
    seq = _make_seq(eos_reached=True, decoded_length=2)
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is True


def test_is_sequence_completed_eos_ignored_under_ignore_eos(ctx_ignore_eos):
    seq = _make_seq(eos_reached=True, decoded_length=2)
    assert CompletionHandler.is_sequence_completed(ctx_ignore_eos, seq) is False


def test_is_sequence_completed_repetition(ctx_strict):
    seq = _make_seq(rep_detected=True, decoded_length=2)
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is True


def test_is_sequence_completed_repetition_overrides_ignore_eos(ctx_ignore_eos):
    # ignore_eos doesn't suppress repetition detection
    seq = _make_seq(rep_detected=True, decoded_length=2)
    assert CompletionHandler.is_sequence_completed(ctx_ignore_eos, seq) is True


def test_is_sequence_completed_active_sequence(ctx_strict):
    seq = _make_seq(decoded_length=4, max_decode_length=16)
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is False


# ---------------------------------------------------------------------------
# get_finish_reason
# ---------------------------------------------------------------------------

def test_get_finish_reason_repetition(ctx_strict):
    seq = _make_seq(rep_detected=True, decoded_length=4)
    assert CompletionHandler.get_finish_reason(ctx_strict, seq) == "repetition"


def test_get_finish_reason_length_max_decode(ctx_strict):
    seq = _make_seq(max_decode_length=16, decoded_length=16)
    assert CompletionHandler.get_finish_reason(ctx_strict, seq) == "length"


def test_get_finish_reason_length_context_limit(ctx_strict):
    seq = _make_seq(prompt_length=8, decoded_length=4088, max_decode_length=10_000)
    assert CompletionHandler.get_finish_reason(ctx_strict, seq) == "length"


def test_get_finish_reason_stop(ctx_strict):
    seq = _make_seq(eos_reached=True, decoded_length=4)
    assert CompletionHandler.get_finish_reason(ctx_strict, seq) == "stop"


def test_get_finish_reason_stop_becomes_length_under_ignore_eos(ctx_ignore_eos):
    # Sequence's only completion signal is eos_reached; with ignore_eos=True
    # the function falls through to the "length" branch.
    seq = _make_seq(eos_reached=True, decoded_length=4, max_decode_length=16)
    assert CompletionHandler.get_finish_reason(ctx_ignore_eos, seq) == "length"


def test_get_finish_reason_precedence_repetition_beats_length(ctx_strict):
    # Both rep_detected and max-decode triggered — repetition wins.
    seq = _make_seq(rep_detected=True, decoded_length=16, max_decode_length=16)
    assert CompletionHandler.get_finish_reason(ctx_strict, seq) == "repetition"


# ---------------------------------------------------------------------------
# Statelessness
# ---------------------------------------------------------------------------

def test_handler_is_stateless(ctx_strict):
    seq = _make_seq(decoded_length=4)
    for _ in range(10):
        CompletionHandler.should_stop_at_eos(ctx_strict, 2)
        CompletionHandler.is_sequence_completed(ctx_strict, seq)
    # ctx snapshot unchanged
    assert ctx_strict.rank == 0
    assert ctx_strict.eos_token_ids == frozenset({0, 2, 1024})
    # seq fields we read (not written) unchanged
    assert seq.decoded_length == 4
    assert seq.eos_reached is False


def test_is_sequence_completed_honors_per_sequence_ignore_eos(ctx_strict):
    """Per-sequence ignore_eos (vendor extension via extra_body) overrides a real
    EOS even when the global ctx.ignore_eos is False; length limits still apply."""
    seq = _make_seq(eos_reached=True)
    # Baseline under strict (global ignore_eos=False) ctx: real EOS completes.
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is True
    # Per-sequence override: not completed by EOS.
    seq.ignore_eos = True
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is False
    # Length limit still completes regardless of ignore_eos.
    seq.decoded_length = seq.max_decode_length
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is True


def test_get_finish_reason_per_sequence_ignore_eos_becomes_length(ctx_strict):
    """A real EOS reports finish_reason 'stop' under strict ctx, but 'length' when
    the sequence sets ignore_eos (per-request), without touching the global flag."""
    seq = _make_seq(eos_reached=True, decoded_length=4, max_decode_length=16)
    assert CompletionHandler.get_finish_reason(ctx_strict, seq) == "stop"
    seq.ignore_eos = True
    assert CompletionHandler.get_finish_reason(ctx_strict, seq) == "length"
