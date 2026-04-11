"""Unit tests for batchgen.worker.completion.CompletionHandler."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.completion import CompletionHandler
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import (
    FakeCollectiveBackend,
    FakeResponseSink,
    FakeTokenizer,
)


def _make_state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=rank,
        torch_device=torch.device("cpu"),
    )


def _make_seq(
    state: WorkerState,
    uuid: str = "u1",
    *,
    prompt_length: int = 10,
    max_decode_length: int = 100,
    global_idx: int = 0,
    assigned_rank: int | None = None,
    decoded: list[int] | None = None,
    status: SequenceStatus | None = None,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=max_decode_length,
        text="",
    )
    seq.current_context_length = prompt_length
    if assigned_rank is None:
        seq.assigned_rank = state.rank
    else:
        seq.assigned_rank = assigned_rank
    if decoded is not None:
        buf = torch.zeros(max(len(decoded), max_decode_length), dtype=torch.long)
        for i, v in enumerate(decoded):
            buf[i] = v
        seq.decoded_tokens = buf
        seq.decoded_length = len(decoded)
        seq.current_context_length = prompt_length + len(decoded)
    state.global_batch.add_sequence(seq)
    if status is not None:
        # Need to advance through valid transitions
        from batchgen.sequence import SequenceStatus as S

        order = [S.QUEUEING, S.IN_PREFILL, S.PREFILLED, S.IN_DECODE]
        cur = S.QUEUEING
        while cur != status and cur in order and order.index(cur) < order.index(status):
            nxt = order[order.index(cur) + 1]
            state.global_batch.update_status(uuid, nxt)
            cur = nxt
    return seq


def _make_ch(
    state: WorkerState,
    *,
    tokenizer: FakeTokenizer | None = None,
    collectives: FakeCollectiveBackend | None = None,
    sink: FakeResponseSink | None = None,
    model_context_length: int = 1024,
    ignore_eos: bool = False,
    rep_detection_enabled: bool = True,
) -> CompletionHandler:
    return CompletionHandler(
        state=state,
        tokenizer=tokenizer or FakeTokenizer(eos_token_ids={99}),
        collectives=collectives or FakeCollectiveBackend(rank=state.rank, world_size=state.world_size),
        sink=sink or FakeResponseSink(),
        model_context_length=model_context_length,
        ignore_eos=ignore_eos,
        rep_detection_enabled=rep_detection_enabled,
    )


# ---------------------------------------------------------------------------
# is_completed
# ---------------------------------------------------------------------------


class TestIsCompleted:
    def test_max_decode_length_reached(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1] * 100, max_decode_length=100)
        assert _make_ch(state).is_completed(seq)

    def test_max_decode_length_not_reached(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1] * 50, max_decode_length=100)
        assert not _make_ch(state).is_completed(seq)

    def test_context_limit_reached(self) -> None:
        state = _make_state()
        seq = _make_seq(state, prompt_length=1000, max_decode_length=200, decoded=[1] * 30)
        # ctx = 1000 + 30 = 1030 >= 1024
        assert _make_ch(state, model_context_length=1024).is_completed(seq)

    def test_eos_reached_with_default_stops(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1])
        seq.eos_reached = True
        assert _make_ch(state).is_completed(seq)

    def test_eos_reached_with_ignore_eos_does_not_stop(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1])
        seq.eos_reached = True
        assert not _make_ch(state, ignore_eos=True).is_completed(seq)

    def test_repetition_detected_completes(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1])
        seq._rep_detected = True
        assert _make_ch(state).is_completed(seq)

    def test_no_completion_condition(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1, 2, 3], max_decode_length=100)
        assert not _make_ch(state).is_completed(seq)


# ---------------------------------------------------------------------------
# get_finish_reason
# ---------------------------------------------------------------------------


class TestGetFinishReason:
    def test_repetition_priority_over_stop(self) -> None:
        """Plan Decision #4: repetition also sets eos_reached, so repetition must win."""
        state = _make_state()
        seq = _make_seq(state, decoded=[1])
        seq._rep_detected = True
        seq.eos_reached = True
        assert _make_ch(state).get_finish_reason(seq) == "repetition"

    def test_stop_when_eos_reached(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1])
        seq.eos_reached = True
        assert _make_ch(state).get_finish_reason(seq) == "stop"

    def test_stop_suppressed_by_ignore_eos(self) -> None:
        """ignore_eos makes EOS contribute nothing; length must be the reason."""
        state = _make_state()
        seq = _make_seq(state, decoded=[1] * 100, max_decode_length=100)
        seq.eos_reached = True
        assert _make_ch(state, ignore_eos=True).get_finish_reason(seq) == "length"

    def test_length_when_max_decode_reached(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1] * 100, max_decode_length=100)
        assert _make_ch(state).get_finish_reason(seq) == "length"

    def test_length_when_context_limit_reached(self) -> None:
        state = _make_state()
        seq = _make_seq(state, prompt_length=1000, max_decode_length=200, decoded=[1] * 30)
        assert _make_ch(state, model_context_length=1024).get_finish_reason(seq) == "length"

    def test_not_completed_raises(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1])
        with pytest.raises(ValueError, match="non-completed"):
            _make_ch(state).get_finish_reason(seq)


# ---------------------------------------------------------------------------
# should_stop_at_eos
# ---------------------------------------------------------------------------


class TestShouldStopAtEos:
    def test_token_in_eos_set_returns_true(self) -> None:
        state = _make_state()
        seq = _make_seq(state)
        tok = FakeTokenizer(eos_token_ids={99, 100})
        assert _make_ch(state, tokenizer=tok).should_stop_at_eos(seq, 100)

    def test_token_not_in_eos_set_returns_false(self) -> None:
        state = _make_state()
        seq = _make_seq(state)
        tok = FakeTokenizer(eos_token_ids={99})
        assert not _make_ch(state, tokenizer=tok).should_stop_at_eos(seq, 42)

    def test_ignore_eos_short_circuits(self) -> None:
        state = _make_state()
        seq = _make_seq(state)
        tok = FakeTokenizer(eos_token_ids={99})
        assert not _make_ch(state, tokenizer=tok, ignore_eos=True).should_stop_at_eos(seq, 99)

    def test_plural_eos_convention(self) -> None:
        """conventions.md: eos_token_ids is a SET (plural). Multiple tokens all stop."""
        state = _make_state()
        seq = _make_seq(state)
        tok = FakeTokenizer(eos_token_ids={1, 2, 3, 4, 5})
        ch = _make_ch(state, tokenizer=tok)
        for t in (1, 2, 3, 4, 5):
            assert ch.should_stop_at_eos(seq, t)
        assert not ch.should_stop_at_eos(seq, 6)


# ---------------------------------------------------------------------------
# check_repeating_pattern
# ---------------------------------------------------------------------------


class TestCheckRepeatingPattern:
    def test_too_short_is_not_checked(self) -> None:
        state = _make_state()
        seq = _make_seq(state, decoded=[1] * 32, max_decode_length=200)
        assert not _make_ch(state).check_repeating_pattern(seq)
        assert not seq._rep_detected

    def test_trailing_ab_ab_repeat_detected(self) -> None:
        state = _make_state()
        tokens = [5, 6] * 32 + [7, 8, 7, 8]  # ... 7 8 7 8 at the tail
        seq = _make_seq(state, decoded=tokens, max_decode_length=200)
        detected = _make_ch(state).check_repeating_pattern(seq)
        assert detected
        assert seq._rep_detected
        assert seq.eos_reached

    def test_long_pattern_detected(self) -> None:
        state = _make_state()
        pattern = list(range(10, 30))  # 20-token pattern
        # 80 tokens total: 40 padding + (20)(20) matched pair at end
        tokens = [0] * 40 + pattern + pattern
        seq = _make_seq(state, decoded=tokens, max_decode_length=200)
        assert _make_ch(state).check_repeating_pattern(seq)

    def test_no_repetition_leaves_flags_clear(self) -> None:
        state = _make_state()
        tokens = list(range(64, 128))  # strictly increasing — no repetition
        seq = _make_seq(state, decoded=tokens, max_decode_length=200)
        assert not _make_ch(state).check_repeating_pattern(seq)
        assert not seq._rep_detected
        assert not seq.eos_reached

    def test_disabled_short_circuits(self) -> None:
        state = _make_state()
        tokens = [1, 2] * 50
        seq = _make_seq(state, decoded=tokens, max_decode_length=200)
        assert not _make_ch(state, rep_detection_enabled=False).check_repeating_pattern(seq)
        assert not seq._rep_detected

    def test_missing_decoded_tokens_buffer(self) -> None:
        state = _make_state()
        seq = _make_seq(state)
        seq.decoded_length = 100
        seq.decoded_tokens = None
        assert not _make_ch(state).check_repeating_pattern(seq)


# ---------------------------------------------------------------------------
# gather_tokens
# ---------------------------------------------------------------------------


class TestGatherTokens:
    def test_single_rank_returns_own_decoded(self) -> None:
        state = _make_state()
        _make_seq(state, uuid="u1", decoded=[97, 98, 99])  # "abc"
        tok = FakeTokenizer()
        ch = _make_ch(state, tokenizer=tok)
        result = ch.gather_tokens(["u1"])
        assert result == {"u1": "abc"}

    def test_skips_uuids_assigned_to_other_rank(self) -> None:
        state = _make_state(rank=0, world_size=2)
        _make_seq(state, uuid="u0", decoded=[97], assigned_rank=0)
        _make_seq(state, uuid="u1", decoded=[98], assigned_rank=1, global_idx=1)
        col = FakeCollectiveBackend(rank=0, world_size=2)
        ch = _make_ch(state, collectives=col)

        result = ch.gather_tokens(["u0", "u1"])

        # Only u0 belongs to this rank. u1 is not contributed by rank 0
        # and the default fake all_gather just returns the self-dict on
        # other slots, so u1 does not appear.
        assert result == {"u0": "a"}

    def test_merges_injected_other_rank_contributions(self) -> None:
        state = _make_state(rank=0, world_size=2)
        _make_seq(state, uuid="u0", decoded=[97], assigned_rank=0)
        _make_seq(state, uuid="u1", decoded=[], assigned_rank=1, global_idx=1)
        col = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_gather_object_responses=[[None, {"u1": "rank1-text"}]],
        )
        ch = _make_ch(state, collectives=col)
        result = ch.gather_tokens(["u0", "u1"])
        assert result == {"u0": "a", "u1": "rank1-text"}

    def test_empty_decoded_buffer_returns_empty_string(self) -> None:
        state = _make_state()
        seq = _make_seq(state, uuid="u1")
        seq.decoded_length = 0
        seq.decoded_tokens = None
        ch = _make_ch(state)
        result = ch.gather_tokens(["u1"])
        assert result == {"u1": ""}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestReport:
    def test_sends_payload_to_sink(self) -> None:
        state = _make_state()
        sink = FakeResponseSink()
        ch = _make_ch(state, sink=sink)
        ch.report("u1", "hello", "stop")
        assert sink.reported == {"u1": {"text": "hello", "finish_reason": "stop"}}


# ---------------------------------------------------------------------------
# check_and_handle
# ---------------------------------------------------------------------------


class TestCheckAndHandle:
    def test_transitions_completed_sequences_and_reports_on_rank0(self) -> None:
        state = _make_state(rank=0, world_size=1)
        # Seq u1: decoded_length == max → length completion
        _make_seq(
            state,
            uuid="u1",
            max_decode_length=10,
            decoded=[ord("a") for _ in range(10)],
            status=SequenceStatus.IN_DECODE,
        )
        sink = FakeResponseSink()
        ch = _make_ch(state, sink=sink)

        completed = ch.check_and_handle(["u1"])

        assert completed == {"u1"}
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.COMPLETED  # type: ignore[union-attr]
        assert "u1" in sink.reported
        assert sink.reported["u1"]["finish_reason"] == "length"

    def test_skips_non_decode_status_sequences(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _make_seq(state, uuid="u1", decoded=[1], status=None)  # stays QUEUEING
        ch = _make_ch(state)

        completed = ch.check_and_handle(["u1"])

        assert completed == set()
        # Not reported, not transitioned
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.QUEUEING  # type: ignore[union-attr]

    def test_empty_uuids_returns_empty_without_collectives(self) -> None:
        state = _make_state()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        ch = _make_ch(state, collectives=col)
        assert ch.check_and_handle([]) == set()
        # No gather issued when nothing to complete.
        assert col.call_names() == []

    def test_no_completions_returns_empty_without_collectives(self) -> None:
        state = _make_state()
        _make_seq(state, uuid="u1", decoded=[1, 2, 3], max_decode_length=100, status=SequenceStatus.IN_DECODE)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        ch = _make_ch(state, collectives=col)
        assert ch.check_and_handle(["u1"]) == set()
        assert col.call_names() == []

    def test_non_rank0_does_not_report_to_sink(self) -> None:
        state = _make_state(rank=1, world_size=2)
        _make_seq(
            state,
            uuid="u1",
            assigned_rank=1,
            max_decode_length=10,
            decoded=[ord("a") for _ in range(10)],
            status=SequenceStatus.IN_DECODE,
        )
        sink = FakeResponseSink()
        ch = _make_ch(state, sink=sink)

        completed = ch.check_and_handle(["u1"])

        assert completed == {"u1"}
        # Rank 1 must NOT report to the sink — only rank 0 talks to the queue.
        assert sink.reported == {}
        # But the transition still happens on every rank.
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.COMPLETED  # type: ignore[union-attr]
