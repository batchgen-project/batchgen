"""Unit tests for batchgen.worker.batch_formation.BatchFormation."""

from __future__ import annotations

from typing import Any

import torch

from batchgen.sequence import SequenceEntry
from batchgen.worker.batch_formation import BatchFormation
from batchgen.worker.indexing import IndexManager
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import (
    FakeCollectiveBackend,
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


def _add_seq(
    state: WorkerState, uuid: str, text: str, global_idx: int = 0
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=len(text),  # placeholder until tokenize runs
        max_decode_length=32,
        text=text,
    )
    state.global_batch.add_sequence(seq)
    return seq


def _make_batch_formation(
    state: WorkerState,
    *,
    tokenizer: FakeTokenizer | None = None,
    collectives: FakeCollectiveBackend | None = None,
    model_context_length: int = 1_000_000,
) -> BatchFormation:
    tok = tokenizer or FakeTokenizer()
    col = collectives or FakeCollectiveBackend(rank=state.rank, world_size=state.world_size)
    idx = IndexManager(state)
    return BatchFormation(state, tok, col, idx, model_context_length)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_single_rank_encodes_every_uuid_and_writes_back(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "u1", "ab")
        _add_seq(state, "u2", "xyz")
        tok = FakeTokenizer()
        bf = _make_batch_formation(state, tokenizer=tok)

        bf.tokenize(["u1", "u2"])

        seq1 = state.global_batch.get_sequence("u1")
        seq2 = state.global_batch.get_sequence("u2")
        assert seq1 is not None and seq2 is not None
        assert torch.equal(seq1.input_ids, torch.tensor([97, 98], dtype=torch.long))
        assert torch.equal(seq2.input_ids, torch.tensor([120, 121, 122], dtype=torch.long))
        assert seq1.prompt_length == 2
        assert seq2.prompt_length == 3
        assert seq1.current_context_length == 2
        assert seq2.current_context_length == 3
        assert seq1.original_prompt_length == 2
        assert seq2.original_prompt_length == 3
        assert tok.encode_calls == ["ab", "xyz"]

    def test_multi_rank_only_self_slice_encoded_locally(self) -> None:
        """Rank 1 of world=2 should only encode uuids at positions 1, 3, ..."""
        state = _make_state(rank=1, world_size=2)
        for i, uuid in enumerate(["u0", "u1", "u2", "u3"]):
            _add_seq(state, uuid, f"t{i}")
        tok = FakeTokenizer()
        # Inject rank-0's tokenized results so all_gather_object has something to merge.
        other = {"u0": [200], "u2": [202]}
        col = FakeCollectiveBackend(
            rank=1,
            world_size=2,
            all_gather_object_responses=[[other, None]],
        )
        bf = _make_batch_formation(state, tokenizer=tok, collectives=col)

        bf.tokenize(["u0", "u1", "u2", "u3"])

        # Only u1 and u3 went through this rank's tokenizer
        assert tok.encode_calls == ["t1", "t3"]
        # But every sequence has input_ids after the gather merge
        for uuid, expected in {
            "u0": [200],
            "u1": [ord("t"), ord("1")],
            "u2": [202],
            "u3": [ord("t"), ord("3")],
        }.items():
            seq = state.global_batch.get_sequence(uuid)
            assert seq is not None
            assert torch.equal(seq.input_ids, torch.tensor(expected, dtype=torch.long))
            assert seq.prompt_length == len(expected)

    def test_missing_text_is_skipped(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "u1", "ab")
        seq2 = _add_seq(state, "u2", "placeholder")
        seq2.text = None  # simulate admission with no text yet
        tok = FakeTokenizer()
        bf = _make_batch_formation(state, tokenizer=tok)

        bf.tokenize(["u1", "u2"])

        assert tok.encode_calls == ["ab"]
        seq2_after = state.global_batch.get_sequence("u2")
        assert seq2_after is not None
        assert seq2_after.input_ids is None

    def test_missing_uuid_is_skipped(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "u1", "ab")
        bf = _make_batch_formation(state)
        # "ghost" is not in global_batch — tokenize must not raise
        bf.tokenize(["u1", "ghost"])
        seq1 = state.global_batch.get_sequence("u1")
        assert seq1 is not None
        assert seq1.prompt_length == 2

    def test_tokenize_issues_exactly_one_all_gather_object(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "u1", "ab")
        col = FakeCollectiveBackend(rank=0, world_size=1)
        bf = _make_batch_formation(state, collectives=col)

        bf.tokenize(["u1"])

        assert col.call_names() == ["all_gather_object"]


# ---------------------------------------------------------------------------
# assign_ranks
# ---------------------------------------------------------------------------


class TestAssignRanks:
    def test_round_robin_world_size_2(self) -> None:
        state = _make_state(rank=0, world_size=2)
        for uuid in ["u0", "u1", "u2", "u3"]:
            _add_seq(state, uuid, "x")
        bf = _make_batch_formation(state)

        bf.assign_ranks(["u0", "u1", "u2", "u3"])

        # Sorted order: u0, u1, u2, u3 → ranks 0, 1, 0, 1
        expected = {"u0": 0, "u1": 1, "u2": 0, "u3": 1}
        for uuid, rank in expected.items():
            seq = state.global_batch.get_sequence(uuid)
            assert seq is not None and seq.assigned_rank == rank

    def test_round_robin_world_size_4(self) -> None:
        state = _make_state(rank=0, world_size=4)
        for uuid in ["s1", "s2", "s3", "s4", "s5"]:
            _add_seq(state, uuid, "x")
        bf = _make_batch_formation(state)

        bf.assign_ranks(["s1", "s2", "s3", "s4", "s5"])

        # Sorted: s1..s5 → ranks 0, 1, 2, 3, 0
        assert state.global_batch.get_sequence("s1").assigned_rank == 0  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("s2").assigned_rank == 1  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("s3").assigned_rank == 2  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("s4").assigned_rank == 3  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("s5").assigned_rank == 0  # type: ignore[union-attr]

    def test_assign_ranks_is_order_independent(self) -> None:
        """Calling with shuffled input gives the same result as sorted input."""
        state_a = _make_state(rank=0, world_size=2)
        state_b = _make_state(rank=0, world_size=2)
        for uuid in ["a", "b", "c", "d"]:
            _add_seq(state_a, uuid, "x")
            _add_seq(state_b, uuid, "x")
        _make_batch_formation(state_a).assign_ranks(["d", "b", "a", "c"])
        _make_batch_formation(state_b).assign_ranks(["a", "b", "c", "d"])
        for uuid in ["a", "b", "c", "d"]:
            assert (
                state_a.global_batch.get_sequence(uuid).assigned_rank  # type: ignore[union-attr]
                == state_b.global_batch.get_sequence(uuid).assigned_rank  # type: ignore[union-attr]
            )

    def test_ghost_uuid_skipped(self) -> None:
        state = _make_state(rank=0, world_size=2)
        _add_seq(state, "u1", "x")
        bf = _make_batch_formation(state)
        bf.assign_ranks(["u1", "ghost"])  # must not raise
        assert state.global_batch.get_sequence("u1").assigned_rank == 0  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# build_query_book
# ---------------------------------------------------------------------------


class TestBuildQueryBook:
    def test_registers_only_uuids_assigned_to_this_rank(self) -> None:
        state = _make_state(rank=0, world_size=2)
        for uuid in ["u0", "u1", "u2", "u3"]:
            _add_seq(state, uuid, "x")
        bf = _make_batch_formation(state)
        bf.assign_ranks(["u0", "u1", "u2", "u3"])
        # Rank 0 owns u0 and u2

        allocated = bf.build_query_book(["u0", "u1", "u2", "u3"])

        assert allocated == [0, 1]
        assert state.uuid_to_local_map == {"u0": 0, "u2": 1}

    def test_returns_local_indices_in_input_order(self) -> None:
        state = _make_state(rank=0, world_size=2)
        for uuid in ["u0", "u1", "u2", "u3"]:
            _add_seq(state, uuid, "x")
        bf = _make_batch_formation(state)
        bf.assign_ranks(["u0", "u1", "u2", "u3"])

        # Reverse input order; IndexManager still allocates lowest-first,
        # so u2 registers first (local 0), u0 second (local 1).
        allocated = bf.build_query_book(["u3", "u2", "u1", "u0"])

        # u3, u1 → rank 1, skipped; u2 → 0, u0 → 1 (order: u2 before u0)
        assert allocated == [0, 1]
        assert state.uuid_to_local_map == {"u2": 0, "u0": 1}

    def test_skips_already_registered_uuids(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "u1", "x")
        bf = _make_batch_formation(state)
        bf.assign_ranks(["u1"])
        bf.build_query_book(["u1"])

        # Re-calling with the same uuid is idempotent — no new allocation.
        allocated = bf.build_query_book(["u1"])
        assert allocated == []
        assert state.uuid_to_local_map == {"u1": 0}

    def test_missing_uuid_is_skipped(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "u1", "x")
        bf = _make_batch_formation(state)
        bf.assign_ranks(["u1"])
        allocated = bf.build_query_book(["ghost", "u1"])
        assert allocated == [0]


# ---------------------------------------------------------------------------
# reject_overflow
# ---------------------------------------------------------------------------


class TestRejectOverflow:
    def test_empty_when_all_fit(self) -> None:
        state = _make_state()
        seq = _add_seq(state, "u1", "x")
        seq.prompt_length = 100
        bf = _make_batch_formation(state, model_context_length=1024)
        assert bf.reject_overflow(["u1"]) == set()

    def test_filters_sequences_at_or_above_threshold(self) -> None:
        state = _make_state()
        for uuid, length in [("u1", 100), ("u2", 500), ("u3", 1024), ("u4", 2000)]:
            seq = _add_seq(state, uuid, "x")
            seq.prompt_length = length
        bf = _make_batch_formation(state, model_context_length=1024)

        rejected = bf.reject_overflow(["u1", "u2", "u3", "u4"])

        # gte, not gt: u3 (==1024) is rejected.
        assert rejected == {"u3", "u4"}

    def test_boundary_strict_equality_is_rejected(self) -> None:
        state = _make_state()
        seq = _add_seq(state, "u1", "x")
        seq.prompt_length = 42
        bf = _make_batch_formation(state, model_context_length=42)
        assert bf.reject_overflow(["u1"]) == {"u1"}

    def test_boundary_one_below_is_accepted(self) -> None:
        state = _make_state()
        seq = _add_seq(state, "u1", "x")
        seq.prompt_length = 41
        bf = _make_batch_formation(state, model_context_length=42)
        assert bf.reject_overflow(["u1"]) == set()

    def test_missing_uuid_is_not_rejected(self) -> None:
        state = _make_state()
        bf = _make_batch_formation(state, model_context_length=1024)
        assert bf.reject_overflow(["ghost"]) == set()
