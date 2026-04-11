"""The inner forward-pass-and-tick loop for :class:`DecodeScheduler`.

Extracted from :class:`DecodeScheduler.run_continuous` so the loop body
is readable on its own and the scheduler file stays short. The helper
is a free function that takes the state, model backend, and the uuid
list explicitly — no hidden dependency on a scheduler instance.

One invocation drives exactly one decision interval:

    iterations = decision_frequency_pages * PAGE_SIZE

Each iteration calls :meth:`ModelExecutorBackend.forward_decode` with
the current uuid batch and ticks ``decoded_length`` +
``current_context_length`` for every sequence still in IN_DECODE.
Sequences that transition out of IN_DECODE mid-interval are no longer
ticked (matches main's guarded update path).

The helper does NOT call the boundary handler — the caller does that
after the loop returns. The split is deliberate so tests can exercise
the tick math independently of the boundary orchestration.
"""

from __future__ import annotations

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.protocols import UUID, ModelExecutorBackend
from batchgen.worker.state import WorkerState


def run_decode_interval(
    state: WorkerState,
    model: ModelExecutorBackend,
    uuids: list[UUID],
    *,
    decision_frequency_pages: int,
) -> int:
    """Run one decision interval over `uuids`. Returns tokens produced.

    Tokens produced is the number of forward_decode iterations the
    interval executed. It does NOT attempt to account for sequences
    that fell out of IN_DECODE mid-interval — the caller uses
    state.global_batch to learn which sequences were actually advanced.
    """
    page_size = SequenceEntry.PAGE_SIZE
    iterations = decision_frequency_pages * page_size
    tokens_produced = 0

    for _ in range(iterations):
        model.forward_decode({"uuids": list(uuids)})
        for uuid in uuids:
            seq = state.global_batch.get_sequence(uuid)
            if seq is None or seq.status != SequenceStatus.IN_DECODE:
                continue
            seq.decoded_length += 1
            seq.current_context_length += 1
        tokens_produced += 1

    return tokens_produced


__all__ = ["run_decode_interval"]
