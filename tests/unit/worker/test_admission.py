"""Unit tests for batchgen.worker.admission.AdmissionCoordinator."""

from __future__ import annotations

from queue import Queue

import torch

from batchgen.worker.admission import AdmissionCoordinator
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import FakeCollectiveBackend


def _make_state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=rank,
        torch_device=torch.device("cpu"),
    )


def _queue_with(*messages: dict) -> Queue:
    q: Queue = Queue()
    for m in messages:
        q.put(m)
    return q


# ---------------------------------------------------------------------------
# Rank 0: polling behavior
# ---------------------------------------------------------------------------


class TestRank0Polling:
    def test_empty_queue_returns_empty_list_and_broadcasts_none(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        ac = AdmissionCoordinator(state, col, admission_queue=Queue())

        assert ac.poll_and_broadcast() == []
        assert col.call_names() == ["broadcast_object"]
        # No sequences materialized
        assert len(state.global_batch.sequences) == 0

    def test_no_queue_returns_empty_list_and_still_broadcasts_none(self) -> None:
        """Non-rank-0 ranks construct with `admission_queue=None` — rank 0
        may also be constructed without a queue in bench/test setups."""
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        ac = AdmissionCoordinator(state, col, admission_queue=None)

        assert ac.poll_and_broadcast() == []
        assert col.call_names() == ["broadcast_object"]

    def test_single_admission_materializes_sequence(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        q = _queue_with({
            "sequences": [
                {"uuid": "u1", "text": "hello", "max_decode_length": 64},
            ],
        })
        ac = AdmissionCoordinator(state, col, admission_queue=q)

        admitted = ac.poll_and_broadcast()

        assert admitted == ["u1"]
        seq = state.global_batch.get_sequence("u1")
        assert seq is not None
        assert seq.text == "hello"
        assert seq.max_decode_length == 64
        assert seq.global_idx == 0

    def test_multi_admission_assigns_sequential_global_idx(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        q = _queue_with({
            "sequences": [
                {"uuid": "u1", "text": "a"},
                {"uuid": "u2", "text": "b"},
                {"uuid": "u3", "text": "c"},
            ],
        })
        ac = AdmissionCoordinator(state, col, admission_queue=q)

        admitted = ac.poll_and_broadcast()

        assert admitted == ["u1", "u2", "u3"]
        assert state.global_batch.get_sequence("u1").global_idx == 0  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("u2").global_idx == 1  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("u3").global_idx == 2  # type: ignore[union-attr]

    def test_subsequent_admission_continues_global_idx(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        q = _queue_with(
            {"sequences": [{"uuid": "u1"}, {"uuid": "u2"}]},
            {"sequences": [{"uuid": "u3"}]},
        )
        ac = AdmissionCoordinator(state, col, admission_queue=q)

        ac.poll_and_broadcast()
        ac.poll_and_broadcast()

        assert state.global_batch.get_sequence("u1").global_idx == 0  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("u2").global_idx == 1  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("u3").global_idx == 2  # type: ignore[union-attr]

    def test_max_decode_length_defaults_to_32(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        q = _queue_with({"sequences": [{"uuid": "u1", "text": "hi"}]})
        ac = AdmissionCoordinator(state, col, admission_queue=q)
        ac.poll_and_broadcast()
        assert state.global_batch.get_sequence("u1").max_decode_length == 32  # type: ignore[union-attr]

    def test_missing_text_is_accepted(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        q = _queue_with({"sequences": [{"uuid": "u1"}]})
        ac = AdmissionCoordinator(state, col, admission_queue=q)
        ac.poll_and_broadcast()
        seq = state.global_batch.get_sequence("u1")
        assert seq is not None
        assert seq.text is None

    def test_empty_sequences_list_returns_empty(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        q = _queue_with({"sequences": []})
        ac = AdmissionCoordinator(state, col, admission_queue=q)
        assert ac.poll_and_broadcast() == []
        assert len(state.global_batch.sequences) == 0


# ---------------------------------------------------------------------------
# Non-rank-0: broadcast-receive behavior
# ---------------------------------------------------------------------------


class TestNonRank0Receive:
    def test_non_rank_0_does_not_touch_queue(self) -> None:
        """Even if a queue is passed, non-rank-0 must not poll it."""
        state = _make_state(rank=1, world_size=2)
        col = FakeCollectiveBackend(
            rank=1,
            world_size=2,
            broadcast_object_responses=[[None]],
        )
        q = _queue_with({"sequences": [{"uuid": "u1"}]})
        ac = AdmissionCoordinator(state, col, admission_queue=q)

        assert ac.poll_and_broadcast() == []
        # Queue is still populated — non-rank-0 never consumed it.
        assert q.qsize() == 1

    def test_non_rank_0_materializes_from_broadcast(self) -> None:
        state = _make_state(rank=1, world_size=2)
        admission_msg = {
            "sequences": [
                {"uuid": "u1", "text": "foo", "max_decode_length": 16},
                {"uuid": "u2", "text": "bar"},
            ],
        }
        col = FakeCollectiveBackend(
            rank=1,
            world_size=2,
            broadcast_object_responses=[[admission_msg]],
        )
        ac = AdmissionCoordinator(state, col, admission_queue=None)

        admitted = ac.poll_and_broadcast()

        assert admitted == ["u1", "u2"]
        assert state.global_batch.get_sequence("u1").max_decode_length == 16  # type: ignore[union-attr]
        assert state.global_batch.get_sequence("u2").max_decode_length == 32  # type: ignore[union-attr]

    def test_non_rank_0_with_none_broadcast_returns_empty(self) -> None:
        state = _make_state(rank=1, world_size=2)
        col = FakeCollectiveBackend(
            rank=1,
            world_size=2,
            broadcast_object_responses=[[None]],
        )
        ac = AdmissionCoordinator(state, col, admission_queue=None)
        assert ac.poll_and_broadcast() == []
        assert len(state.global_batch.sequences) == 0


# ---------------------------------------------------------------------------
# Collective ordering
# ---------------------------------------------------------------------------


class TestCollectiveOrdering:
    def test_always_issues_exactly_one_broadcast_object(self) -> None:
        state = _make_state(rank=0, world_size=1)
        col = FakeCollectiveBackend(rank=0, world_size=1)
        q = _queue_with({"sequences": [{"uuid": "u1"}]})
        ac = AdmissionCoordinator(state, col, admission_queue=q)

        ac.poll_and_broadcast()  # with msg
        ac.poll_and_broadcast()  # without msg

        assert col.call_names() == ["broadcast_object", "broadcast_object"]
