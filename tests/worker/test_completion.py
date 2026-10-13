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
    """Production-like context with a real model_context_length."""
    return CompletionContext(
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


def test_should_stop_at_eos_ignored_when_the_sequence_sets_the_flag(ctx_strict):
    # Even EOS tokens return False for a request that asked to ignore EOS
    assert CompletionHandler.should_stop_at_eos(ctx_strict, 2, True) is False
    assert CompletionHandler.should_stop_at_eos(ctx_strict, 1024, True) is False


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


def test_is_sequence_completed_repetition(ctx_strict):
    seq = _make_seq(rep_detected=True, decoded_length=2)
    assert CompletionHandler.is_sequence_completed(ctx_strict, seq) is True


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
