"""AdmissionCoordinator — pool-mode admission polling and broadcast.

Small handler (~50 LOC) that owns two concerns:

  1. Rank-0 polls an `admission_queue` (typically an mp.Queue) for pending
     admission messages without blocking.
  2. The message (or `None`) is broadcast to every rank via
     `CollectiveBackend.broadcast_object` so every rank sees the same
     admission decision at the same outer-loop tick.

On receipt, the coordinator materializes `SequenceEntry` objects into
`state.global_batch` and returns the list of newly-admitted UUIDs. It does
NOT tokenize, assign ranks, or build the query book — the orchestrator calls
`BatchFormation.tokenize / assign_ranks / build_query_book` on the returned
uuids afterwards.

Expected message shape (same as main's `_admit_sequences_from_message`):

    {
        "sequences": [
            {"uuid": "...", "text": "...", "max_decode_length": int},
            ...
        ],
    }

`max_decode_length` defaults to 32 if omitted. Missing `text` is accepted —
tokenize will skip it. Unknown top-level keys are ignored.
"""

from __future__ import annotations

from queue import Empty
from typing import Any, Protocol

from batchgen.sequence import SequenceEntry
from batchgen.worker.protocols import UUID, CollectiveBackend
from batchgen.worker.state import WorkerState


class AdmissionQueueBackend(Protocol):
    """Structural surface of the admission queue on rank 0.

    Matches `queue.Queue.get_nowait` / `multiprocessing.Queue.get_nowait`
    — raises `queue.Empty` when empty. Tests use a plain `queue.Queue`.
    """

    def get_nowait(self) -> Any: ...


class AdmissionCoordinator:
    def __init__(
        self,
        state: WorkerState,
        collectives: CollectiveBackend,
        admission_queue: AdmissionQueueBackend | None = None,
    ) -> None:
        self._state = state
        self._collectives = collectives
        self._queue = admission_queue

    def poll_and_broadcast(self) -> list[UUID]:
        """Poll (rank 0) + broadcast (all ranks) + materialize sequences.

        Returns:
            List of newly-admitted UUIDs in the order they appeared in the
            message. Empty list when no admission is pending on rank 0
            (and therefore nothing is broadcast downstream either).
        """
        msg: dict[str, Any] | None = None
        if self._state.rank == 0 and self._queue is not None:
            try:
                msg = self._queue.get_nowait()
            except Empty:
                msg = None

        obj_list: list[Any] = [msg]
        self._collectives.broadcast_object(obj_list, src=0)
        msg = obj_list[0]

        if not msg:
            return []
        specs = msg.get("sequences", []) or []
        if not specs:
            return []

        base_idx = len(self._state.global_batch.sequences)
        admitted: list[UUID] = []
        for offset, spec in enumerate(specs):
            uuid = spec["uuid"]
            global_idx = base_idx + offset
            seq = SequenceEntry(
                uuid=uuid,
                global_idx=global_idx,
                prompt_length=0,
                max_decode_length=int(spec.get("max_decode_length", 32)),
                text=spec.get("text"),
            )
            self._state.global_batch.add_sequence(seq)
            admitted.append(uuid)
        return admitted


__all__ = ["AdmissionCoordinator", "AdmissionQueueBackend"]
