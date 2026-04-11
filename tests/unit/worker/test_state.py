"""Unit tests for batchgen.worker.state.WorkerState."""

from __future__ import annotations

import pytest
import torch

from batchgen.worker.state import WorkerState


def _make(**overrides: object) -> WorkerState:
    """Construct a WorkerState with sensible single-rank defaults."""
    base = dict(
        rank=0,
        local_rank=0,
        world_size=1,
        device=0,
        torch_device=torch.device("cpu"),
    )
    base.update(overrides)  # type: ignore[arg-type]
    return WorkerState(**base)  # type: ignore[arg-type]


class TestConstruction:
    def test_single_rank_defaults(self) -> None:
        state = _make()
        assert state.rank == 0
        assert state.local_rank == 0
        assert state.world_size == 1
        assert state.device == 0
        assert state.torch_device == torch.device("cpu")

    def test_multi_rank_happy_path(self) -> None:
        state = _make(rank=3, local_rank=3, world_size=4, device=3)
        assert state.rank == 3
        assert state.local_rank == 3
        assert state.world_size == 4
        assert state.device == 3


class TestValidation:
    def test_world_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="world_size"):
            _make(world_size=0, rank=0, local_rank=0)

    def test_world_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="world_size"):
            _make(world_size=-1, rank=0, local_rank=0)

    def test_rank_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            _make(rank=-1)

    def test_rank_equal_world_size_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            _make(rank=2, local_rank=0, world_size=2)

    def test_rank_above_world_size_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            _make(rank=5, local_rank=0, world_size=2)

    def test_local_rank_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="local_rank"):
            _make(local_rank=-1)

    def test_local_rank_above_world_size_raises(self) -> None:
        with pytest.raises(ValueError, match="local_rank"):
            _make(rank=0, local_rank=3, world_size=2)

    def test_device_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="device"):
            _make(device=-1)


class TestMutation:
    def test_fields_are_mutable(self) -> None:
        """WorkerState is a plain dataclass; handlers write through it."""
        state = _make(rank=0, local_rank=0, world_size=2, device=0)
        # Post-init validation only runs at construction. Field reassignment
        # is allowed so handlers can update state during the worker lifetime.
        state.rank = 1  # type: ignore[misc]
        assert state.rank == 1
