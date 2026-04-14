"""Unit tests for decode/update_sequences.py (Phase 2.8.2e)."""

from __future__ import annotations

import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.decode.update_sequences import update_sequences
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import FakeLegacyBackend


def _state(uuids: list[str]) -> WorkerState:
    s = WorkerState(
        rank=0, local_rank=0, world_size=1, device=0,
        torch_device=torch.device("cpu"),
    )
    for i, uuid in enumerate(uuids):
        seq = SequenceEntry(
            uuid=uuid, global_idx=i, prompt_length=10, max_decode_length=100,
        )
        seq.original_prompt_length = 10
        seq.decoded_length = 0
        seq.current_context_length = 10
        seq.assigned_rank = 0
        seq.status = SequenceStatus.IN_DECODE
        s.global_batch.sequences[uuid] = seq
        s.global_batch._status_index[SequenceStatus.IN_DECODE].add(uuid)
    return s


class _Adapter(FakeLegacyBackend):
    """FakeLegacyBackend with a deterministic EOS token for these tests."""

    def should_stop_at_eos(self, token_id: int) -> bool:
        self._record("should_stop_at_eos", token_id)
        return token_id == 2  # treat 2 as the stop token


class TestHappyPath:
    def test_writes_token_advances_counters(self) -> None:
        state = _state(["u"])
        seq = state.global_batch.get_sequence("u")
        legacy = _Adapter()
        legacy._uuid_to_local = {"u": 0}
        legacy._local_to_uuid = {0: "u"}

        tokens = torch.tensor([[5]], dtype=torch.int64)
        update_sequences(
            state, legacy, batch=[0], new_tokens_cpu=tokens, local_iteration=1,
        )
        assert seq.decoded_length == 1
        assert seq.current_context_length == 11
        assert seq.eos_reached is False
        # record_decoded_token was called with correct params
        rec = next(c for c in legacy.calls if c[0] == "record_decoded_token")
        assert rec[2]["local_idx"] == 0
        assert rec[2]["decode_pos"] == 0
        assert rec[2]["token"].item() == 5

    def test_batch_with_multiple_sequences(self) -> None:
        state = _state(["a", "b"])
        legacy = _Adapter()
        legacy._uuid_to_local = {"a": 0, "b": 1}
        legacy._local_to_uuid = {0: "a", 1: "b"}

        tokens = torch.tensor([[10], [20]], dtype=torch.int64)
        update_sequences(
            state, legacy, batch=[0, 1], new_tokens_cpu=tokens, local_iteration=1,
        )
        assert state.global_batch.get_sequence("a").decoded_length == 1
        assert state.global_batch.get_sequence("b").decoded_length == 1
        recs = [c for c in legacy.calls if c[0] == "record_decoded_token"]
        assert len(recs) == 2


class TestCompletionPaths:
    def test_eos_token_flags_seq(self) -> None:
        state = _state(["u"])
        seq = state.global_batch.get_sequence("u")
        legacy = _Adapter()
        legacy._uuid_to_local = {"u": 0}
        legacy._local_to_uuid = {0: "u"}

        tokens = torch.tensor([[2]], dtype=torch.int64)  # EOS
        update_sequences(
            state, legacy, batch=[0], new_tokens_cpu=tokens, local_iteration=1,
        )
        assert seq.eos_reached is True

    def test_max_decode_length_flags_seq(self) -> None:
        state = _state(["u"])
        seq = state.global_batch.get_sequence("u")
        seq.decoded_length = seq.max_decode_length - 1  # one short of cap
        legacy = _Adapter()
        legacy._uuid_to_local = {"u": 0}
        legacy._local_to_uuid = {0: "u"}

        tokens = torch.tensor([[7]], dtype=torch.int64)
        update_sequences(
            state, legacy, batch=[0], new_tokens_cpu=tokens, local_iteration=1,
        )
        assert seq.decoded_length == seq.max_decode_length
        assert seq.eos_reached is True

    def test_already_completed_skipped(self) -> None:
        state = _state(["u"])
        seq = state.global_batch.get_sequence("u")
        seq.eos_reached = True  # is_sequence_completed returns True
        legacy = _Adapter()
        legacy._uuid_to_local = {"u": 0}
        legacy._local_to_uuid = {0: "u"}

        tokens = torch.tensor([[5]], dtype=torch.int64)
        update_sequences(
            state, legacy, batch=[0], new_tokens_cpu=tokens, local_iteration=1,
        )
        # decoded_length not advanced; no buffer write.
        assert seq.decoded_length == 0
        assert not any(
            c[0] == "record_decoded_token" for c in legacy.calls
        )


class TestRepetitionDetection:
    def test_32_same_tokens_triggers_eos(self) -> None:
        state = _state(["u"])
        seq = state.global_batch.get_sequence("u")
        seq._rep_last_token = 9
        seq._rep_count = 31
        legacy = _Adapter()
        legacy._uuid_to_local = {"u": 0}
        legacy._local_to_uuid = {0: "u"}

        tokens = torch.tensor([[9]], dtype=torch.int64)  # same token
        update_sequences(
            state, legacy, batch=[0], new_tokens_cpu=tokens, local_iteration=1,
        )
        assert seq._rep_detected is True
        assert seq.eos_reached is True

    def test_different_token_resets_counter(self) -> None:
        state = _state(["u"])
        seq = state.global_batch.get_sequence("u")
        seq._rep_last_token = 9
        seq._rep_count = 15
        legacy = _Adapter()
        legacy._uuid_to_local = {"u": 0}
        legacy._local_to_uuid = {0: "u"}

        tokens = torch.tensor([[7]], dtype=torch.int64)
        update_sequences(
            state, legacy, batch=[0], new_tokens_cpu=tokens, local_iteration=1,
        )
        assert seq._rep_last_token == 7
        assert seq._rep_count == 1
        assert seq._rep_detected is False
        assert seq.eos_reached is False
