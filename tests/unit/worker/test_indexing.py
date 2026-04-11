"""Unit tests for batchgen.worker.indexing.IndexManager."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.indexing import (
    DuplicateSequenceError,
    IndexManager,
    UnknownSequenceError,
)
from batchgen.worker.state import WorkerState


def _make_state() -> WorkerState:
    return WorkerState(
        rank=0,
        local_rank=0,
        world_size=1,
        device=0,
        torch_device=torch.device("cpu"),
    )


def _make_seq(uuid: str, global_idx: int) -> SequenceEntry:
    """Minimal SequenceEntry for IndexManager tests.

    Uses SequenceEntry's actual constructor signature. Only the fields
    IndexManager reads (uuid, global_idx) matter here.
    """
    return SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=1,
        max_decode_length=1,
        input_ids=[0],
        text="",
    )


class TestRegister:
    def test_first_register_returns_zero(self) -> None:
        state = _make_state()
        idx = IndexManager(state).register("u1")
        assert idx == 0
        assert state.uuid_to_local_map == {"u1": 0}
        assert state.local_to_uuid_map == {0: "u1"}
        assert state.next_local_idx == 1
        assert state.free_local_indices == set()

    def test_sequential_registers_grow_next_idx(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        assert im.register("u1") == 0
        assert im.register("u2") == 1
        assert im.register("u3") == 2
        assert state.next_local_idx == 3
        assert state.free_local_indices == set()

    def test_duplicate_register_raises(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        im.register("u1")
        with pytest.raises(DuplicateSequenceError, match="u1"):
            im.register("u1")
        # State must be unchanged after the failed call
        assert state.next_local_idx == 1


class TestUnregister:
    def test_unregister_frees_slot_and_removes_maps(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        im.register("u1")
        im.register("u2")
        im.unregister("u1")
        assert "u1" not in state.uuid_to_local_map
        assert 0 not in state.local_to_uuid_map
        assert state.free_local_indices == {0}
        # u2's slot is untouched
        assert state.uuid_to_local_map == {"u2": 1}

    def test_unregister_unknown_raises(self) -> None:
        state = _make_state()
        with pytest.raises(UnknownSequenceError, match="u1"):
            IndexManager(state).unregister("u1")


class TestFreeSlotReuse:
    def test_next_register_pops_from_free_set_before_growing(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        im.register("u1")  # 0
        im.register("u2")  # 1
        im.register("u3")  # 2
        assert state.next_local_idx == 3

        im.unregister("u2")
        assert state.free_local_indices == {1}

        reused = im.register("u4")
        assert reused == 1
        assert state.next_local_idx == 3  # not grown
        assert state.free_local_indices == set()

    def test_lowest_free_index_reused_first(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        for u in ("u0", "u1", "u2", "u3"):
            im.register(u)
        im.unregister("u3")  # free {3}
        im.unregister("u1")  # free {1, 3}
        assert im.register("u4") == 1
        assert im.register("u5") == 3
        assert state.next_local_idx == 4  # still not grown

    def test_register_after_all_freed_reuses_slots(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        im.register("u1")
        im.register("u2")
        im.unregister("u1")
        im.unregister("u2")
        assert state.free_local_indices == {0, 1}
        assert im.register("u3") == 0
        assert im.register("u4") == 1
        assert state.next_local_idx == 2


class TestLookups:
    def test_local_for_uuid_roundtrip(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        im.register("u1")
        im.register("u2")
        assert im.local_for_uuid("u1") == 0
        assert im.local_for_uuid("u2") == 1

    def test_local_for_uuid_unknown_raises(self) -> None:
        im = IndexManager(_make_state())
        with pytest.raises(UnknownSequenceError, match="u1"):
            im.local_for_uuid("u1")

    def test_uuid_for_local_roundtrip(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        im.register("u1")
        im.register("u2")
        assert im.uuid_for_local(0) == "u1"
        assert im.uuid_for_local(1) == "u2"

    def test_uuid_for_local_unknown_raises(self) -> None:
        im = IndexManager(_make_state())
        with pytest.raises(UnknownSequenceError, match="local_idx 42"):
            im.uuid_for_local(42)


class TestGlobalIdsForLocal:
    def test_preserves_order_and_maps_via_global_batch(self) -> None:
        state = _make_state()
        state.global_batch.add_sequence(_make_seq("u1", global_idx=100))
        state.global_batch.add_sequence(_make_seq("u2", global_idx=200))
        state.global_batch.add_sequence(_make_seq("u3", global_idx=300))
        im = IndexManager(state)
        im.register("u1")  # 0
        im.register("u2")  # 1
        im.register("u3")  # 2
        # Out-of-order input preserved
        assert im.global_ids_for_local([2, 0, 1]) == [300, 100, 200]

    def test_unknown_local_idx_raises(self) -> None:
        im = IndexManager(_make_state())
        with pytest.raises(UnknownSequenceError, match="local_idx 5"):
            im.global_ids_for_local([5])

    def test_uuid_not_in_global_batch_raises(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        im.register("u1")  # registered but global_batch doesn't have it
        with pytest.raises(UnknownSequenceError, match="not in state.global_batch"):
            im.global_ids_for_local([0])


class TestIntrospection:
    def test_is_registered(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        assert not im.is_registered("u1")
        im.register("u1")
        assert im.is_registered("u1")
        im.unregister("u1")
        assert not im.is_registered("u1")

    def test_live_count(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        assert im.live_count() == 0
        im.register("u1")
        im.register("u2")
        assert im.live_count() == 2
        im.unregister("u1")
        assert im.live_count() == 1


class TestPoolModeReusePattern:
    """Exercise the plan Decision #7 lockstep invariant at the IndexManager level.

    Simulates 100 admissions + 50 completions + 50 more admissions, asserts
    `next_local_idx` never exceeds 100 (the max live-count peak).
    """

    def test_100_admissions_50_completions_50_more(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        for i in range(100):
            im.register(f"u{i}")
        assert state.next_local_idx == 100

        for i in range(50):
            im.unregister(f"u{i}")
        assert len(state.free_local_indices) == 50

        for i in range(100, 150):
            im.register(f"u{i}")

        assert state.next_local_idx == 100, (
            "next_local_idx must not grow past peak live count; free slots "
            "should have been reused"
        )
        assert state.free_local_indices == set()
        assert im.live_count() == 100

    def test_churn_many_cycles_bounded(self) -> None:
        state = _make_state()
        im = IndexManager(state)
        for cycle in range(10):
            for i in range(20):
                im.register(f"c{cycle}-u{i}")
            for i in range(20):
                im.unregister(f"c{cycle}-u{i}")
        assert state.next_local_idx == 20, (
            "10 cycles of 20 admissions+completions must stay at peak 20"
        )
