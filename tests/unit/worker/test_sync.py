"""Unit tests for batchgen.worker.sync.SyncCoordinator."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry
from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SeqSnapshot, SyncCoordinator
from tests.unit.worker.fakes import FakeCollectiveBackend


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
    uuid: str,
    *,
    prompt_length: int = 10,
    decoded_length: int = 0,
    assigned_rank: int | None = None,
    eos_reached: bool = False,
    current_context_length: int | None = None,
    gpu_pages_allocated: int = 0,
    host_pages_allocated: int = 0,
    global_idx: int = 0,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=100,
        text="",
    )
    seq.original_prompt_length = prompt_length
    seq.decoded_length = decoded_length
    seq.current_context_length = (
        current_context_length
        if current_context_length is not None
        else prompt_length + decoded_length
    )
    seq.assigned_rank = assigned_rank if assigned_rank is not None else state.rank
    seq.eos_reached = eos_reached
    seq.gpu_pages_allocated = gpu_pages_allocated
    seq.host_pages_allocated = host_pages_allocated
    state.global_batch.add_sequence(seq)
    return seq


# ---------------------------------------------------------------------------
# sync_metadata — happy path + CTX fast-fail
# ---------------------------------------------------------------------------


class TestSyncMetadataHappyPath:
    def test_single_rank_preserves_state_and_records_one_call(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _make_seq(state, "u1", prompt_length=10, decoded_length=5)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        SyncCoordinator(state, col).sync_metadata(["u1"])

        seq = state.global_batch.get_sequence("u1")
        assert seq is not None
        assert seq.current_context_length == 15  # unchanged
        assert col.call_names() == ["all_gather_object"]

    def test_absorbs_cross_rank_updates_for_non_owned_sequence(self) -> None:
        """Rank 0 holds a shadow of u1 owned by rank 1. After sync_metadata,
        the shadow must reflect rank 1's authoritative decoded_length."""
        state = _make_state(rank=0, world_size=2)
        # u0 is ours; u1 is rank 1's — start with stale shadow metadata
        _make_seq(state, "u0", prompt_length=10, decoded_length=3, assigned_rank=0)
        _make_seq(
            state,
            "u1",
            prompt_length=20,
            decoded_length=0,  # stale
            assigned_rank=1,
            global_idx=1,
        )
        rank1_payload = {
            "u1": SeqSnapshot(
                uuid="u1",
                prompt_length=20,
                original_prompt_length=20,
                decoded_length=7,
                current_context_length=27,
                gpu_pages_allocated=3,
                host_pages_allocated=5,
                eos_reached=False,
            )
        }
        col = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_gather_object_responses=[[None, rank1_payload]],
        )
        SyncCoordinator(state, col).sync_metadata(["u0", "u1"])

        u1 = state.global_batch.get_sequence("u1")
        assert u1 is not None
        assert u1.decoded_length == 7
        assert u1.current_context_length == 27
        assert u1.gpu_pages_allocated == 3
        assert u1.host_pages_allocated == 5

    def test_does_not_overwrite_own_sequences(self) -> None:
        """Even if a peer accidentally sends a snapshot for a sequence we own,
        the receiver absorb step must NOT touch our authoritative copy."""
        state = _make_state(rank=0, world_size=2)
        _make_seq(state, "u0", prompt_length=10, decoded_length=5, assigned_rank=0)
        stale = {
            "u0": SeqSnapshot(
                uuid="u0",
                prompt_length=10,
                original_prompt_length=10,
                decoded_length=999,  # attempted overwrite
                current_context_length=1009,
                gpu_pages_allocated=0,
                host_pages_allocated=0,
                eos_reached=False,
            )
        }
        col = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_gather_object_responses=[[None, stale]],
        )
        SyncCoordinator(state, col).sync_metadata(["u0"])
        u0 = state.global_batch.get_sequence("u0")
        assert u0 is not None
        assert u0.decoded_length == 5
        assert u0.current_context_length == 15

    def test_empty_uuids_still_issues_gather(self) -> None:
        """Empty input must still gather (collective ordering invariant).
        All ranks must call all_gather_object in lockstep per plan invariant #6."""
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        SyncCoordinator(state, col).sync_metadata([])
        assert col.call_names() == ["all_gather_object"]


class TestSyncMetadataCtxFastFail:
    def test_sender_drift_raises_before_any_collective(self) -> None:
        state = _make_state(rank=0, world_size=2)
        seq = _make_seq(
            state,
            "u0",
            prompt_length=10,
            decoded_length=5,
            current_context_length=14,  # drifted by -1
        )
        assert seq.original_prompt_length + seq.decoded_length == 15
        col = FakeCollectiveBackend(rank=0, world_size=2)

        with pytest.raises(CtxInvariantViolation) as exc:
            SyncCoordinator(state, col).sync_metadata(["u0"])

        assert exc.value.uuid == "u0"
        assert exc.value.side == "sender"
        assert exc.value.had == 14
        assert exc.value.expected == 15
        # Critical: no collective was issued after the guard failed.
        assert col.calls == []

    def test_receiver_drift_raises_after_gather(self) -> None:
        state = _make_state(rank=0, world_size=2)
        _make_seq(state, "u0", prompt_length=10, decoded_length=5, assigned_rank=0)
        broken_rank1 = {
            "u1": SeqSnapshot(
                uuid="u1",
                prompt_length=20,
                original_prompt_length=20,
                decoded_length=7,
                current_context_length=99,  # drifted
                gpu_pages_allocated=0,
                host_pages_allocated=0,
                eos_reached=False,
            )
        }
        col = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_gather_object_responses=[[None, broken_rank1]],
        )
        with pytest.raises(CtxInvariantViolation) as exc:
            SyncCoordinator(state, col).sync_metadata(["u0"])

        assert exc.value.uuid == "u1"
        assert exc.value.side == "receiver"
        assert exc.value.had == 99
        assert exc.value.expected == 27
        # The collective DID happen before the receiver guard tripped.
        assert col.call_names() == ["all_gather_object"]

    def test_sender_skips_non_owned_sequences(self) -> None:
        """A rank should NOT verify CTX for sequences it does not own."""
        state = _make_state(rank=0, world_size=2)
        # u1 is rank-1-owned with drifted ctx — rank 0 must not check it.
        _make_seq(
            state,
            "u1",
            prompt_length=20,
            decoded_length=7,
            assigned_rank=1,
            current_context_length=999,  # drifted on the shadow copy
        )
        col = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_gather_object_responses=[[{}, {}]],
        )
        # Rank 0 does not own u1, so no sender-side raise. The gather
        # returns an empty payload from rank 1 so there's nothing to
        # receiver-check either.
        SyncCoordinator(state, col).sync_metadata(["u1"])


# ---------------------------------------------------------------------------
# sync_completion_status
# ---------------------------------------------------------------------------


class TestSyncCompletionStatus:
    def test_empty_uuids_no_collective(self) -> None:
        state = _make_state()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        result = SyncCoordinator(state, col).sync_completion_status([])
        assert result == set()
        assert col.calls == []

    def test_single_rank_returns_local_completions(self) -> None:
        state = _make_state()
        _make_seq(state, "u0", eos_reached=True)
        _make_seq(state, "u1", eos_reached=False, global_idx=1)
        _make_seq(state, "u2", eos_reached=True, global_idx=2)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        result = SyncCoordinator(state, col).sync_completion_status(["u0", "u1", "u2"])
        assert result == {"u0", "u2"}
        assert col.call_names() == ["all_reduce_max"]

    def test_union_across_ranks_via_injected_max_delta(self) -> None:
        """Rank 0 sees u0 complete, rank 1 sees u1 complete. Union = {u0, u1}."""
        state = _make_state(rank=0, world_size=2)
        _make_seq(state, "u0", eos_reached=True, assigned_rank=0)
        _make_seq(state, "u1", eos_reached=False, assigned_rank=1, global_idx=1)
        # Inject: rank 1 reports [0, 1] for these two candidates.
        col = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_reduce_max_deltas=[torch.tensor([0, 1], dtype=torch.int32)],
        )
        result = SyncCoordinator(state, col).sync_completion_status(["u0", "u1"])
        assert result == {"u0", "u1"}


# ---------------------------------------------------------------------------
# sync_decode_uuids
# ---------------------------------------------------------------------------


class TestSyncDecodeUuids:
    def test_empty_candidates_no_collective(self) -> None:
        state = _make_state()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        assert SyncCoordinator(state, col).sync_decode_uuids([]) == set()
        assert col.calls == []

    def test_single_rank_returns_locally_present(self) -> None:
        state = _make_state()
        _make_seq(state, "u0")
        _make_seq(state, "u1", global_idx=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        result = SyncCoordinator(state, col).sync_decode_uuids(["u0", "u1", "ghost"])
        assert result == {"u0", "u1"}
        assert col.call_names() == ["all_reduce_min"]

    def test_intersection_across_ranks(self) -> None:
        """u0 present on both; u1 only on rank 0 (rank 1 reports 0 for u1)."""
        state = _make_state(rank=0, world_size=2)
        _make_seq(state, "u0")
        _make_seq(state, "u1", global_idx=1)
        col = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_reduce_min_deltas=[torch.tensor([1, 0], dtype=torch.int32)],
        )
        result = SyncCoordinator(state, col).sync_decode_uuids(["u0", "u1"])
        assert result == {"u0"}


# ---------------------------------------------------------------------------
# gather_rank_token_counts
# ---------------------------------------------------------------------------


class TestGatherRankTokenCounts:
    def test_single_rank_returns_self(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        result = SyncCoordinator(state, col).gather_rank_token_counts(42)
        assert result == [42]
        assert col.call_names() == ["all_gather_into_tensor"]

    def test_multi_rank_returns_per_rank_list(self) -> None:
        state = _make_state(rank=0, world_size=4)
        col = FakeCollectiveBackend(
            rank=0,
            world_size=4,
            all_gather_into_tensor_responses=[
                torch.tensor([3, 7, 5, 1], dtype=torch.int64),
            ],
        )
        result = SyncCoordinator(state, col).gather_rank_token_counts(3)
        assert result == [3, 7, 5, 1]

    def test_zero_local_count_ok(self) -> None:
        state = _make_state(rank=1, world_size=2)
        col = FakeCollectiveBackend(
            rank=1,
            world_size=2,
            all_gather_into_tensor_responses=[torch.tensor([5, 0], dtype=torch.int64)],
        )
        result = SyncCoordinator(state, col).gather_rank_token_counts(0)
        assert result == [5, 0]
