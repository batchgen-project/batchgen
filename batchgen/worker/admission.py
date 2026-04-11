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

**Message shape — matches main's ``_admit_sequences_from_message`` contract
(verified against batch_scheduler.py at the drain site)**:

    {
        "type": "admit",
        "entries": [
            {
                "request_id": "...",       # becomes SequenceEntry.uuid
                "text": "...",
                "max_tokens": int,          # becomes max_decode_length
                "batch_id": "...",          # routing hint, preserved on seq
                "priority": int,            # 0=NORMAL, 1=HIGH
                "sampling_params": {...},   # optional, stored for future use
            },
            ...
        ],
    }

The legacy unit-test shape ``{"sequences": [{"uuid", "text",
"max_decode_length"}, ...]}`` is still accepted for backward compatibility
with the existing M1 test suite. Production messages are detected by the
``"type": "admit"`` key + the ``"entries"`` list. A ``None`` message
signals shutdown and is propagated via
``WorkerState.shutdown_requested``.
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
        shutdown_signal = False
        if self._state.rank == 0 and self._queue is not None:
            try:
                raw = self._queue.get_nowait()
            except Empty:
                raw = None
            if raw is None and self._queue is not None:
                # Distinguish "queue empty" from "explicit None shutdown".
                # queue.get_nowait() only returns here if it DID pull a
                # None; Empty is caught above. Main uses None as the
                # shutdown sentinel.
                # Note: queue.Queue.get_nowait raises Empty on empty, so
                # the `raw is None` branch only fires on explicit None.
                shutdown_signal = True
            else:
                msg = raw

        # Broadcast the message (may be None if no admission pending).
        obj_list: list[Any] = [msg]
        self._collectives.broadcast_object(obj_list, src=0)
        msg = obj_list[0]

        # Broadcast the shutdown flag separately so non-rank-0 see it too.
        # Uses broadcast_object for simplicity; production would use a
        # small tensor but this path is out-of-hot-loop.
        shutdown_obj: list[Any] = [shutdown_signal]
        self._collectives.broadcast_object(shutdown_obj, src=0)
        if shutdown_obj[0]:
            # Signal shutdown via the orchestrator's _shutdown flag, if
            # it exists on state. Handlers check state.shutdown_requested
            # in the outer generate_persistent loop.
            setattr(self._state, "shutdown_requested", True)
            return []

        if not msg:
            return []

        # Determine message shape and extract the entries list.
        entries = self._extract_entries(msg)
        if not entries:
            return []

        base_idx = len(self._state.global_batch.sequences)
        admitted: list[UUID] = []
        for offset, spec in enumerate(entries):
            # Production shape uses request_id + max_tokens; legacy test
            # shape uses uuid + max_decode_length. Support both.
            uuid = spec.get("request_id") or spec["uuid"]
            max_dec = int(spec.get("max_tokens", spec.get("max_decode_length", 32)))
            global_idx = base_idx + offset
            seq = SequenceEntry(
                uuid=uuid,
                global_idx=global_idx,
                prompt_length=0,
                max_decode_length=max_dec,
                text=spec.get("text"),
            )
            # Preserve routing hints from the production message shape so
            # downstream reporting can correlate by batch_id / priority.
            if "batch_id" in spec:
                seq.batch_id = spec["batch_id"]
            if "priority" in spec:
                seq.priority = int(spec["priority"])
            self._state.global_batch.add_sequence(seq)
            admitted.append(uuid)
        return admitted

    @staticmethod
    def _extract_entries(msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the list of entry dicts regardless of message shape.

        Accepted shapes:
          - Production: ``{"type": "admit", "entries": [...]}``
          - Legacy test: ``{"sequences": [...]}``
        """
        if msg.get("type") == "admit" and "entries" in msg:
            return list(msg.get("entries") or [])
        return list(msg.get("sequences") or [])


__all__ = ["AdmissionCoordinator", "AdmissionQueueBackend"]
