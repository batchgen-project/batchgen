"""Self-consistency replayer — M7 scaffold.

Two concerns live here:

  1. ``record_run``: drive a :class:`WorkerOrchestrator` through a
     scenario and capture a :class:`Trace` at every logical
     checkpoint (end of run_batch, final state after
     generate_persistent, etc.). The recorder walks the orchestrator's
     :class:`FakeCollectiveBackend` and :class:`FakeResponseSink` at
     checkpoint time so cumulative call order and reported UUIDs
     ride along with the state snapshot.

  2. ``replay_roundtrip``: builds two fresh orchestrators, seeds both
     with identical sequences, records a trace from each, and
     returns a :class:`ReplayResult` listing any divergences found
     by :func:`batchgen.worker.trace.diff_traces`. The replayer is
     deliberately symmetric — it doesn't know which run is the
     "expected" and which is the "actual"; the test decides.

This is the scaffold the M7-S3 design doc extends. The main-side
``TraceRecorder`` writes the same :class:`Trace` shape, so future
real-vs-new replays feed into the same :func:`diff_traces` without
changing anything in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue

import torch

from batchgen.sequence import SequenceEntry
from batchgen.worker.config import WorkerConfig
from batchgen.worker.orchestrator import WorkerOrchestrator
from batchgen.worker.state import WorkerState
from batchgen.worker.trace import (
    SeqSpec,
    StateSnapshot,
    Trace,
    TraceCheckpoint,
    TraceDivergence,
    diff_traces,
)
from tests.unit.worker.fakes import (
    FakeClock,
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
    FakeModelExecutor,
    FakeResponseSink,
    FakeTokenizer,
    RecordingLifespanLogger,
)


@dataclass
class ReplayResult:
    expected: Trace
    actual: Trace
    divergences: list[TraceDivergence]

    @property
    def matches(self) -> bool:
        return not self.divergences


def _install_seq(state: WorkerState, spec: SeqSpec) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=spec.uuid,
        global_idx=spec.global_idx,
        prompt_length=spec.prompt_length,
        max_decode_length=spec.max_decode_length,
        text=spec.text,
    )
    seq.original_prompt_length = spec.prompt_length
    seq.assigned_rank = 0
    # Give ample GPU headroom so the boundary handler does not emit
    # Extend / OnHold decisions that would make the two independent
    # runs take different executor branches. Determinism is the
    # point of self-consistency replay.
    seq.gpu_pages_allocated = 64
    state.global_batch.add_sequence(seq)
    return seq


def _build_orchestrator(
    *,
    specs: list[SeqSpec],
    config: WorkerConfig,
    admission_messages: list[dict] | None = None,
) -> tuple[WorkerOrchestrator, WorkerState, FakeCollectiveBackend, FakeResponseSink, Queue]:
    state = WorkerState(
        rank=0,
        local_rank=0,
        world_size=1,
        device=0,
        torch_device=torch.device("cpu"),
    )
    col = FakeCollectiveBackend(rank=0, world_size=1)
    gpu = FakeGpuKvBackend(free_pages=4000)
    host = FakeHostKvBackend(free_pages=8000)
    tokenizer = FakeTokenizer(eos_token_ids={99})
    model = FakeModelExecutor(prefill_output="P", decode_output="D")
    sink = FakeResponseSink()
    queue: Queue = Queue()
    for msg in admission_messages or []:
        queue.put(msg)

    orch = WorkerOrchestrator(
        state,
        config,
        collectives=col,
        gpu_kv=gpu,
        host_kv=host,
        tokenizer=tokenizer,
        model=model,
        lifespan=RecordingLifespanLogger(),
        sink=sink,
        clock=FakeClock(),
        admission_queue=queue if admission_messages else None,
    )

    for spec in specs:
        _install_seq(state, spec)

    return orch, state, col, sink, queue


def _take_checkpoint(
    index: int,
    label: str,
    state: WorkerState,
    col: FakeCollectiveBackend,
    sink: FakeResponseSink,
) -> TraceCheckpoint:
    return TraceCheckpoint(
        index=index,
        label=label,
        state=StateSnapshot.from_state(state),
        collective_calls=tuple(col.call_names()),
        reported_uuids=tuple(sorted(sink.reported.keys())),
    )


def record_run(
    name: str,
    specs: list[SeqSpec],
    *,
    config: WorkerConfig,
    drive_generate_persistent: bool = False,
    admission_messages: list[dict] | None = None,
    persistent_max_iterations: int | None = 5,
) -> Trace:
    """Drive an orchestrator through `specs` and capture a Trace.

    Checkpoints:
      - ``"initial"``: before any handler runs (just after construction + seed).
      - ``"post_run_batch"`` / ``"post_generate_persistent"``: after the
        chosen top-level driver returns.
      - ``"final"``: an extra snapshot for symmetry with future
        multi-phase replays (identical to the post-* checkpoint for
        single-phase runs).
    """
    orch, state, col, sink, _q = _build_orchestrator(
        specs=specs, config=config, admission_messages=admission_messages
    )

    checkpoints: list[TraceCheckpoint] = []
    checkpoints.append(_take_checkpoint(0, "initial", state, col, sink))

    if drive_generate_persistent:
        orch.generate_persistent(max_iterations=persistent_max_iterations)
        checkpoints.append(
            _take_checkpoint(1, "post_generate_persistent", state, col, sink)
        )
    else:
        orch.run_batch()
        checkpoints.append(_take_checkpoint(1, "post_run_batch", state, col, sink))

    checkpoints.append(_take_checkpoint(2, "final", state, col, sink))

    return Trace(
        name=name,
        initial_sequences=tuple(specs),
        checkpoints=tuple(checkpoints),
    )


def replay_roundtrip(
    name: str,
    specs: list[SeqSpec],
    *,
    config: WorkerConfig,
    drive_generate_persistent: bool = False,
    admission_messages: list[dict] | None = None,
) -> ReplayResult:
    """Record two independent runs with identical seeds and diff them.

    Self-consistency: the orchestrator must be deterministic. Two
    independent instances seeded with identical sequences and driven
    identically MUST produce byte-identical traces. Any divergence
    exposes non-determinism (dict iteration order, set order, float
    noise, etc.) in one of the handlers.
    """
    expected = record_run(
        name,
        specs,
        config=config,
        drive_generate_persistent=drive_generate_persistent,
        admission_messages=admission_messages,
    )
    actual = record_run(
        name,
        specs,
        config=config,
        drive_generate_persistent=drive_generate_persistent,
        admission_messages=admission_messages,
    )
    divergences = diff_traces(expected, actual)
    return ReplayResult(expected=expected, actual=actual, divergences=divergences)


__all__ = [
    "ReplayResult",
    "record_run",
    "replay_roundtrip",
]
