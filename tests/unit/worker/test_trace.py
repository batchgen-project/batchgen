"""Unit tests for batchgen.worker.trace primitives."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.state import WorkerState
from batchgen.worker.trace import (
    SeqSpec,
    SeqStateSnapshot,
    StateSnapshot,
    Trace,
    TraceCheckpoint,
    TraceDivergence,
    diff_traces,
)


def _make_state() -> WorkerState:
    return WorkerState(
        rank=0,
        local_rank=0,
        world_size=1,
        device=0,
        torch_device=torch.device("cpu"),
    )


def _add(
    state: WorkerState,
    uuid: str,
    *,
    global_idx: int = 0,
    decoded_length: int = 0,
    prompt_length: int = 10,
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
    seq.current_context_length = prompt_length + decoded_length
    seq.assigned_rank = 0
    state.global_batch.add_sequence(seq)
    return seq


def _cp(
    index: int,
    label: str,
    state: WorkerState,
    *,
    collectives: tuple[str, ...] = (),
    reported: tuple[str, ...] = (),
) -> TraceCheckpoint:
    return TraceCheckpoint(
        index=index,
        label=label,
        state=StateSnapshot.from_state(state),
        collective_calls=collectives,
        reported_uuids=reported,
    )


# ---------------------------------------------------------------------------
# SeqStateSnapshot + StateSnapshot
# ---------------------------------------------------------------------------


class TestSeqStateSnapshot:
    def test_from_seq_captures_every_field(self) -> None:
        state = _make_state()
        seq = _add(state, "u1", global_idx=7, decoded_length=5, prompt_length=10)
        seq.gpu_pages_allocated = 4
        seq.host_pages_allocated = 8

        snap = SeqStateSnapshot.from_seq(seq)

        assert snap.uuid == "u1"
        assert snap.global_idx == 7
        assert snap.status == int(SequenceStatus.QUEUEING)
        assert snap.assigned_rank == 0
        assert snap.prompt_length == 10
        assert snap.original_prompt_length == 10
        assert snap.decoded_length == 5
        assert snap.current_context_length == 15
        assert snap.gpu_pages_allocated == 4
        assert snap.host_pages_allocated == 8
        assert snap.eos_reached is False
        assert snap.rep_detected is False

    def test_equality_by_value(self) -> None:
        state_a = _make_state()
        state_b = _make_state()
        seq_a = _add(state_a, "u1", decoded_length=3)
        seq_b = _add(state_b, "u1", decoded_length=3)
        assert SeqStateSnapshot.from_seq(seq_a) == SeqStateSnapshot.from_seq(seq_b)

    def test_frozen(self) -> None:
        state = _make_state()
        seq = _add(state, "u1")
        snap = SeqStateSnapshot.from_seq(seq)
        with pytest.raises(Exception):
            snap.uuid = "u2"  # type: ignore[misc]


class TestStateSnapshot:
    def test_sequences_are_sorted_by_uuid(self) -> None:
        state = _make_state()
        _add(state, "uC", global_idx=2)
        _add(state, "uA", global_idx=0)
        _add(state, "uB", global_idx=1)
        snap = StateSnapshot.from_state(state)
        assert [s.uuid for s in snap.sequences] == ["uA", "uB", "uC"]

    def test_captures_index_map_state(self) -> None:
        state = _make_state()
        state.next_local_idx = 5
        state.free_local_indices = {2, 4}
        state.local_to_uuid_map = {0: "a", 1: "b", 3: "c"}
        snap = StateSnapshot.from_state(state)
        assert snap.next_local_idx == 5
        assert snap.free_local_indices == (2, 4)
        assert snap.local_to_uuid_map == ((0, "a"), (1, "b"), (3, "c"))

    def test_equality_across_independent_states_with_same_values(self) -> None:
        state_a = _make_state()
        state_b = _make_state()
        _add(state_a, "u1")
        _add(state_b, "u1")
        assert StateSnapshot.from_state(state_a) == StateSnapshot.from_state(state_b)

    def test_inequality_on_differing_decoded_length(self) -> None:
        state_a = _make_state()
        state_b = _make_state()
        _add(state_a, "u1", decoded_length=3)
        _add(state_b, "u1", decoded_length=4)
        assert StateSnapshot.from_state(state_a) != StateSnapshot.from_state(state_b)


# ---------------------------------------------------------------------------
# diff_traces
# ---------------------------------------------------------------------------


def _trace_with(state: WorkerState, *, label: str = "only") -> Trace:
    return Trace(
        name="self",
        initial_sequences=(),
        checkpoints=(_cp(0, label, state),),
    )


class TestDiffTracesEmpty:
    def test_identical_traces_return_no_divergence(self) -> None:
        state_a = _make_state()
        state_b = _make_state()
        _add(state_a, "u1", decoded_length=2)
        _add(state_b, "u1", decoded_length=2)
        assert diff_traces(_trace_with(state_a), _trace_with(state_b)) == []


class TestDiffTracesLabels:
    def test_name_mismatch_reported_once(self) -> None:
        t_a = Trace(name="A", initial_sequences=(), checkpoints=())
        t_b = Trace(name="B", initial_sequences=(), checkpoints=())
        divergences = diff_traces(t_a, t_b)
        assert [d.path for d in divergences] == ["name"]
        assert divergences[0].expected == "A"
        assert divergences[0].actual == "B"

    def test_length_mismatch_short_circuits_per_checkpoint_walk(self) -> None:
        state = _make_state()
        _add(state, "u1")
        t_a = _trace_with(state)
        t_b = Trace(name="self", initial_sequences=(), checkpoints=())
        divergences = diff_traces(t_a, t_b)
        assert len(divergences) == 1
        assert divergences[0].path == "len(checkpoints)"
        assert divergences[0].expected == 1
        assert divergences[0].actual == 0

    def test_label_mismatch_reported(self) -> None:
        state = _make_state()
        _add(state, "u1")
        t_a = _trace_with(state, label="before")
        t_b = _trace_with(state, label="after")
        divergences = diff_traces(t_a, t_b)
        assert any(d.path == "label" for d in divergences)


class TestDiffTracesSeqFields:
    def test_decoded_length_divergence_reported_with_precise_path(self) -> None:
        state_a = _make_state()
        state_b = _make_state()
        _add(state_a, "u1", decoded_length=5)
        _add(state_b, "u1", decoded_length=7)
        divergences = diff_traces(_trace_with(state_a), _trace_with(state_b))
        paths = [d.path for d in divergences]
        assert "state.sequences['u1'].decoded_length" in paths
        # Also current_context_length moves because both depend on decoded_length
        assert "state.sequences['u1'].current_context_length" in paths

    def test_uuid_set_divergence_short_circuits_field_walk(self) -> None:
        state_a = _make_state()
        state_b = _make_state()
        _add(state_a, "u1")
        _add(state_b, "u2")
        divergences = diff_traces(_trace_with(state_a), _trace_with(state_b))
        assert any(d.path == "state.sequences.uuids" for d in divergences)
        # No per-field noise because the uuid sets differ
        assert not any(".decoded_length" in d.path for d in divergences)


class TestDiffTracesCollectiveAndReported:
    def test_collective_call_order_divergence(self) -> None:
        state = _make_state()
        _add(state, "u1")
        cp_a = _cp(0, "run", state, collectives=("all_gather_object", "broadcast_object"))
        cp_b = _cp(0, "run", state, collectives=("broadcast_object", "all_gather_object"))
        t_a = Trace(name="self", initial_sequences=(), checkpoints=(cp_a,))
        t_b = Trace(name="self", initial_sequences=(), checkpoints=(cp_b,))
        assert any(d.path == "collective_calls" for d in diff_traces(t_a, t_b))

    def test_reported_uuids_divergence(self) -> None:
        state = _make_state()
        _add(state, "u1")
        cp_a = _cp(0, "run", state, reported=("u1",))
        cp_b = _cp(0, "run", state, reported=())
        t_a = Trace(name="self", initial_sequences=(), checkpoints=(cp_a,))
        t_b = Trace(name="self", initial_sequences=(), checkpoints=(cp_b,))
        divergences = diff_traces(t_a, t_b)
        assert any(d.path == "reported_uuids" for d in divergences)


class TestSeqSpecAndTraceFrozen:
    def test_seq_spec_frozen(self) -> None:
        spec = SeqSpec(uuid="u1", global_idx=0, prompt_length=10, max_decode_length=100, text="")
        with pytest.raises(Exception):
            spec.uuid = "u2"  # type: ignore[misc]

    def test_trace_frozen(self) -> None:
        t = Trace(name="t", initial_sequences=(), checkpoints=())
        with pytest.raises(Exception):
            t.name = "t2"  # type: ignore[misc]

    def test_trace_divergence_frozen(self) -> None:
        d = TraceDivergence(checkpoint_index=0, label="", path="x", expected=1, actual=2)
        with pytest.raises(Exception):
            d.path = "y"  # type: ignore[misc]
