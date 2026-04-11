"""Trace primitives for the orchestrator equivalence harness.

M7 bundles three concerns:

  1. **Snapshot**: project ``WorkerState.global_batch`` into a small
     frozen dataclass that captures every field a correctness test
     cares about (status, decoded_length, current_context_length,
     gpu_pages_allocated, host_pages_allocated, eos_reached).
     Snapshots are equal by value, hashable, and deterministic.

  2. **Checkpoint**: a snapshot plus the recorded collective call
     names up to that point in the run and the set of UUIDs reported
     as completed. One checkpoint represents "the state of the world
     at a well-defined point in the scheduling loop".

  3. **Trace**: an ordered list of checkpoints with a label and a
     list of initial sequence specs so another run can be seeded
     identically. Traces are compared with :func:`diff_traces` which
     returns a list of (checkpoint_index, field, expected, actual)
     tuples instead of raising on the first mismatch — useful when
     the first divergence wants to be pinpointed but the caller wants
     the full divergence story.

The M7 replay harness uses these primitives for orchestrator
self-consistency (M7-S2). The same primitives will carry the real
main → orchestrator replay once main is instrumented with a
``TraceRecorder`` on wechat_87 (M7-S3 design doc, production work).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.protocols import UUID
from batchgen.worker.state import WorkerState


@dataclass(frozen=True)
class SeqStateSnapshot:
    """Per-sequence state captured at a trace checkpoint.

    Frozen + equality-by-value so ``snapshot_a == snapshot_b`` is all
    a test needs to say "both runs agree on this sequence".
    """

    uuid: UUID
    global_idx: int
    status: int
    assigned_rank: int
    prompt_length: int
    original_prompt_length: int
    decoded_length: int
    current_context_length: int
    gpu_pages_allocated: int
    host_pages_allocated: int
    eos_reached: bool
    rep_detected: bool

    @classmethod
    def from_seq(cls, seq: SequenceEntry) -> "SeqStateSnapshot":
        return cls(
            uuid=seq.uuid,
            global_idx=seq.global_idx,
            status=int(seq.status),
            assigned_rank=seq.assigned_rank if seq.assigned_rank is not None else -1,
            prompt_length=seq.prompt_length,
            original_prompt_length=seq.original_prompt_length,
            decoded_length=seq.decoded_length,
            current_context_length=seq.current_context_length,
            gpu_pages_allocated=seq.gpu_pages_allocated,
            host_pages_allocated=seq.host_pages_allocated,
            eos_reached=seq.eos_reached,
            rep_detected=seq._rep_detected,
        )


@dataclass(frozen=True)
class StateSnapshot:
    """Full orchestrator state captured at a trace checkpoint.

    Stores sequences in a sorted tuple for a deterministic equality
    comparison regardless of insertion order. Index map fields come
    from the WorkerState directly so post-check-op state lines up.
    """

    sequences: tuple[SeqStateSnapshot, ...]
    next_local_idx: int
    free_local_indices: tuple[int, ...]
    local_to_uuid_map: tuple[tuple[int, UUID], ...]

    @classmethod
    def from_state(cls, state: WorkerState) -> "StateSnapshot":
        seqs = sorted(
            (
                SeqStateSnapshot.from_seq(seq)
                for seq in state.global_batch.sequences.values()
            ),
            key=lambda s: s.uuid,
        )
        return cls(
            sequences=tuple(seqs),
            next_local_idx=state.next_local_idx,
            free_local_indices=tuple(sorted(state.free_local_indices)),
            local_to_uuid_map=tuple(
                sorted(state.local_to_uuid_map.items())
            ),
        )


@dataclass(frozen=True)
class TraceCheckpoint:
    """One trace event: label + full state snapshot + cumulative collectives.

    ``collective_calls`` is the cumulative list of collective op names
    observed on the fake collective backend up to this point; it lets
    the replay assert that both runs issued the same collectives in
    the same order (plan invariant #6) AND reached the same state
    (snapshots equal).

    ``reported_uuids`` is the cumulative set of UUIDs the response
    sink saw. It grows monotonically across a trace.
    """

    index: int
    label: str
    state: StateSnapshot
    collective_calls: tuple[str, ...]
    reported_uuids: tuple[UUID, ...]


@dataclass(frozen=True)
class SeqSpec:
    """Seed spec for reproducing a trace's initial conditions.

    :meth:`Trace.from_run` captures the pre-run batch state via
    :meth:`SeqStateSnapshot.from_seq`, then the replayer can install
    identical sequences in a fresh WorkerState via :func:`install_seq`.
    """

    uuid: UUID
    global_idx: int
    prompt_length: int
    max_decode_length: int
    text: str


@dataclass(frozen=True)
class Trace:
    """An ordered list of checkpoints + enough info to seed a replay."""

    name: str
    initial_sequences: tuple[SeqSpec, ...]
    checkpoints: tuple[TraceCheckpoint, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TraceDivergence:
    """One divergence discovered by :func:`diff_traces`.

    Intentionally a flat record so a human (or a log formatter) can
    read the field without recursing into nested dataclasses. The
    ``path`` string uses dotted notation, e.g.
    ``"checkpoints[3].state.sequences[2].decoded_length"``.
    """

    checkpoint_index: int
    label: str
    path: str
    expected: object
    actual: object


def diff_traces(expected: Trace, actual: Trace) -> list[TraceDivergence]:
    """Return every divergence between two traces.

    Returns an empty list if the traces match. When they don't, the
    returned list walks every mismatching field so the caller can
    pick the first one (for a crisp error) or log all of them (for
    debug output).

    Mismatch on trace length is reported as a single divergence at
    the first missing checkpoint; subsequent checkpoints are not
    walked.
    """
    out: list[TraceDivergence] = []
    if expected.name != actual.name:
        out.append(
            TraceDivergence(
                checkpoint_index=-1,
                label="",
                path="name",
                expected=expected.name,
                actual=actual.name,
            )
        )
    n_exp = len(expected.checkpoints)
    n_act = len(actual.checkpoints)
    if n_exp != n_act:
        out.append(
            TraceDivergence(
                checkpoint_index=min(n_exp, n_act),
                label="(length mismatch)",
                path="len(checkpoints)",
                expected=n_exp,
                actual=n_act,
            )
        )
        return out

    for i, (c_exp, c_act) in enumerate(
        zip(expected.checkpoints, actual.checkpoints)
    ):
        out.extend(_diff_checkpoint(i, c_exp, c_act))
    return out


def _diff_checkpoint(
    i: int, exp: TraceCheckpoint, act: TraceCheckpoint
) -> Iterable[TraceDivergence]:
    def _make(path: str, expected: object, actual: object) -> TraceDivergence:
        return TraceDivergence(
            checkpoint_index=i,
            label=exp.label,
            path=path,
            expected=expected,
            actual=actual,
        )

    if exp.label != act.label:
        yield _make("label", exp.label, act.label)
    if exp.collective_calls != act.collective_calls:
        yield _make("collective_calls", exp.collective_calls, act.collective_calls)
    if exp.reported_uuids != act.reported_uuids:
        yield _make("reported_uuids", exp.reported_uuids, act.reported_uuids)
    if exp.state.next_local_idx != act.state.next_local_idx:
        yield _make(
            "state.next_local_idx",
            exp.state.next_local_idx,
            act.state.next_local_idx,
        )
    if exp.state.free_local_indices != act.state.free_local_indices:
        yield _make(
            "state.free_local_indices",
            exp.state.free_local_indices,
            act.state.free_local_indices,
        )
    if exp.state.local_to_uuid_map != act.state.local_to_uuid_map:
        yield _make(
            "state.local_to_uuid_map",
            exp.state.local_to_uuid_map,
            act.state.local_to_uuid_map,
        )

    # Sequence-level diff
    exp_by_uuid = {s.uuid: s for s in exp.state.sequences}
    act_by_uuid = {s.uuid: s for s in act.state.sequences}
    if exp_by_uuid.keys() != act_by_uuid.keys():
        yield _make(
            "state.sequences.uuids",
            sorted(exp_by_uuid.keys()),
            sorted(act_by_uuid.keys()),
        )
        return
    for uuid in sorted(exp_by_uuid.keys()):
        exp_seq = exp_by_uuid[uuid]
        act_seq = act_by_uuid[uuid]
        if exp_seq == act_seq:
            continue
        # Walk fields for a precise diff
        for field_name in (
            "status",
            "assigned_rank",
            "prompt_length",
            "original_prompt_length",
            "decoded_length",
            "current_context_length",
            "gpu_pages_allocated",
            "host_pages_allocated",
            "eos_reached",
            "rep_detected",
        ):
            exp_v = getattr(exp_seq, field_name)
            act_v = getattr(act_seq, field_name)
            if exp_v != act_v:
                yield _make(
                    f"state.sequences[{uuid!r}].{field_name}", exp_v, act_v
                )


__all__ = [
    "SeqStateSnapshot",
    "StateSnapshot",
    "TraceCheckpoint",
    "SeqSpec",
    "Trace",
    "TraceDivergence",
    "diff_traces",
]
