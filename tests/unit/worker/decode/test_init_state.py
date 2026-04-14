"""Unit tests for decode/init_state.py (Phase 2.8.2b)."""

from __future__ import annotations

import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.decode.init_state import init_decode_state
from batchgen.worker.decode.state import DecodeState
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
        seq.assigned_rank = 0
        seq.status = SequenceStatus.IN_DECODE
        s.global_batch.sequences[uuid] = seq
        s.global_batch._status_index[SequenceStatus.IN_DECODE].add(uuid)
    return s


class TestInitDecodeState:
    def test_counters_initialise_to_zero(self) -> None:
        state = _state(["u"])
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}
        legacy._local_to_uuid = {0: "u"}

        ds = init_decode_state(state, legacy, decode_uuids=["u"], batch=[0])

        assert isinstance(ds, DecodeState)
        assert ds.local_iteration == 0
        assert ds.last_boundary == 0
        assert ds.global_batch_size == 1
        assert ds.decode_uuids == ["u"]
        assert ds.batch == [0]
        assert ds.cumulative_iterations == 0
        assert ds.cumulative_boundaries == 0

    def test_batch_uuids_get_registered_in_gpu_kv_set(self) -> None:
        """Legacy 7906-7909 — anything in ``batch`` that isn't in the
        gpu_kv set gets added. The native port keeps that safety."""
        state = _state(["alpha", "beta"])
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"alpha": 0, "beta": 1}
        legacy._local_to_uuid = {0: "alpha", 1: "beta"}
        legacy._sequences_with_gpu_kv = {"alpha"}  # "beta" missing initially

        init_decode_state(
            state, legacy, decode_uuids=["alpha", "beta"], batch=[0, 1],
        )

        assert legacy.sequences_with_gpu_kv() == {"alpha", "beta"}

    def test_decode_uuids_not_in_batch_also_registered(self) -> None:
        """Any rank-owned uuid in decode_uuids should be tracked, even
        when the local batch happens not to include its local_idx
        (can happen during async-load integration)."""
        state = _state(["alpha", "beta"])
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"alpha": 0, "beta": 1}
        legacy._local_to_uuid = {0: "alpha", 1: "beta"}
        legacy._sequences_with_gpu_kv = set()

        init_decode_state(
            state, legacy, decode_uuids=["alpha", "beta"], batch=[0],
        )

        # Both land in the tracking set.
        assert legacy.sequences_with_gpu_kv() == {"alpha", "beta"}

    def test_non_rank_owned_uuid_skipped(self) -> None:
        """A uuid in decode_uuids but not in uuid_to_local_map belongs
        to another rank; it should NOT be added to this rank's
        sequences_with_gpu_kv set."""
        state = _state(["alpha"])
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {}  # no rank-owned uuids
        legacy._local_to_uuid = {}
        legacy._sequences_with_gpu_kv = set()

        init_decode_state(state, legacy, decode_uuids=["alpha"], batch=[])

        assert legacy.sequences_with_gpu_kv() == set()
