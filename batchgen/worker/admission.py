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
from batchgen.worker.protocols import UUID, CollectiveBackend, LegacyInfraBackend
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
        legacy_infra: LegacyInfraBackend | None = None,
    ) -> None:
        """
        legacy_infra: Phase-F2 production adapter. When set,
            :meth:`poll_and_broadcast` runs the FULL admission cycle
            natively (poll + broadcast in this module) and uses the
            adapter for the infrastructure-heavy steps
            (tokenization, rank assignment, query_book build,
            max_input_length propagation). The adapter is the only
            callback into legacy `BatchGenWorker` — the control
            flow lives here.
            Unit tests pass ``legacy_infra=None`` and rely on the
            lightweight path that only materializes
            ``SequenceEntry`` objects (no tokenization, no query_book).
        """
        self._state = state
        self._collectives = collectives
        self._queue = admission_queue
        self._legacy = legacy_infra

    def poll_and_broadcast(self) -> list[UUID]:
        """Poll (rank 0) + broadcast (all ranks) + materialize sequences.

        Full native flow (when ``legacy_infra`` is set):

          1. Rank 0 polls ``admission_queue`` (or via
             ``legacy_infra.poll_admission_queue_nowait`` if no direct
             queue is wired).
          2. Single ``broadcast_object`` sends the message (or
             ``None``/shutdown sentinel) to every rank — invariant #6.
          3. All ranks parse entries, create ``SequenceEntry`` objects,
             and add them to ``state.global_batch``.
          4. When ``legacy_infra`` is set, all ranks then run
             tokenization + rank-assignment + query_book build +
             max_input_length propagation via the adapter.

        Returns:
            List of newly-admitted UUIDs in the order they appeared in
            the message. Empty list when no admission is pending.
        """
        msg: dict[str, Any] | None = None
        if self._state.rank == 0 and self._queue is not None:
            try:
                raw = self._queue.get_nowait()
                if raw is None:
                    # Explicit shutdown sentinel from main. Convert to
                    # a typed message so the single broadcast carries
                    # the signal without a second collective.
                    msg = {"type": "shutdown"}
                elif isinstance(raw, dict):
                    msg = raw
            except Empty:
                msg = None

        obj_list: list[Any] = [msg]
        self._collectives.broadcast_object(obj_list, src=0)
        msg = obj_list[0]

        if not msg:
            return []

        # Shutdown takes precedence over admission parsing.
        if isinstance(msg, dict) and msg.get("type") == "shutdown":
            setattr(self._state, "shutdown_requested", True)
            return []

        # Step 1: parse entries + materialize SequenceEntry objects
        entries = self._extract_entries(msg)
        if not entries:
            return []

        admitted: list[UUID] = self._materialize_entries(entries)

        # Step 2 (native path, legacy_infra set): tokenize + assign +
        # query_book + propagate max_input_length. This replaces the
        # legacy `_admit_sequences_from_message` pipeline entirely —
        # the adapter calls into battle-tested legacy infrastructure
        # without going through a `admission_delegate` closure.
        if admitted and self._legacy is not None:
            self._legacy.tokenize_admitted_sequences(admitted)

            # Propagate max_input_length from the admitted prompts.
            # Mirrors legacy `_admit_sequences_from_message` step 2.5.
            max_prompt = max(
                (
                    self._state.global_batch.get_sequence(u).prompt_length
                    for u in admitted
                    if self._state.global_batch.get_sequence(u) is not None
                ),
                default=0,
            )
            self._legacy.update_max_input_length(max_prompt)

            self._legacy.assign_admitted_sequences_to_ranks(admitted)
            self._legacy.build_local_query_book_for_admitted(admitted)

        return admitted

    def _materialize_entries(
        self, entries: list[dict[str, Any]]
    ) -> list[UUID]:
        """Build `SequenceEntry` objects from entries and add to batch.

        Mirrors legacy `_admit_sequences_from_message` steps 1 (global_idx
        assignment + `SequenceEntry` construction). Shared by both the
        native (legacy_infra) path and the lightweight CPU test path.
        """
        # global_idx must continue from existing batch state, mirroring
        # legacy (which uses `max(..., default=-1) + 1`). This keeps
        # cross-rank determinism when the batch already has sequences.
        existing_max = -1
        for seq in self._state.global_batch:
            if seq.global_idx > existing_max:
                existing_max = seq.global_idx
        start_idx = existing_max + 1

        admitted: list[UUID] = []
        for offset, spec in enumerate(entries):
            # Production shape uses request_id + max_tokens; legacy test
            # shape uses uuid + max_decode_length. Support both.
            uuid = spec.get("request_id") or spec["uuid"]
            max_dec = int(spec.get("max_tokens", spec.get("max_decode_length", 32)))
            global_idx = start_idx + offset
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
