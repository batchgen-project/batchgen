"""Unit tests for decode/moe_sync.py (Phase 2.8.2c)."""

from __future__ import annotations

import torch

from batchgen.worker.decode.moe_sync import initial_moe_sync
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import FakeCollectiveBackend, FakeLegacyBackend


def _state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank, local_rank=rank, world_size=world_size, device=rank,
        torch_device=torch.device("cpu"),
    )


class TestInitialMoeSync:
    def test_single_rank_sets_tokens_per_rank(self) -> None:
        state = _state()
        legacy = FakeLegacyBackend()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        initial_moe_sync(state, legacy, col, batch=[0, 1, 2])

        names = [c[0] for c in legacy.calls]
        assert names == ["set_num_tokens_per_rank", "set_rank_token_counts"]
        # Value passed to set_num_tokens_per_rank is the max = 3.
        assert legacy.calls[0][1] == (3,)
        assert col.call_names() == ["all_gather_into_tensor"]

    def test_max_is_taken_across_ranks(self) -> None:
        state = _state(rank=0, world_size=2)
        # Inject a gather response where rank-1 has 7 tokens.
        response = torch.tensor([3, 7], dtype=torch.int64)
        col = FakeCollectiveBackend(
            rank=0, world_size=2,
            all_gather_into_tensor_responses=[response],
        )
        legacy = FakeLegacyBackend()
        initial_moe_sync(state, legacy, col, batch=[0, 1, 2])

        assert legacy.calls[0] == ("set_num_tokens_per_rank", (7,), {})

    def test_empty_batch_skips_setters(self) -> None:
        """When every rank has 0 tokens, legacy skips the parallel_manager
        setters (batchgen_worker.py:7935). No-op is correct."""
        state = _state()
        legacy = FakeLegacyBackend()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        initial_moe_sync(state, legacy, col, batch=[])

        names = [c[0] for c in legacy.calls]
        assert "set_num_tokens_per_rank" not in names
        assert "set_rank_token_counts" not in names
        assert col.call_names() == ["all_gather_into_tensor"]

    def test_single_collective_per_call(self) -> None:
        """The collective-ordering fuzzer depends on exactly one
        all_gather_into_tensor per decode entry — lock it down."""
        state = _state()
        legacy = FakeLegacyBackend()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        initial_moe_sync(state, legacy, col, batch=[0, 1])
        assert col.call_names().count("all_gather_into_tensor") == 1
