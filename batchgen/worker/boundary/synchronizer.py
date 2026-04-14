"""BoundarySynchronizer — metadata sync in + plan broadcast out.

Primitives the boundary handler uses to keep every rank in lockstep:

  - ``sync_metadata_in(uuids)``: delegates to
    :meth:`SyncCoordinator.sync_metadata` so every rank has the same
    authoritative view of the batch before the planner runs.
  - ``gather_boundary_state(decode_uuids, gpu_manager, adapter)``:
    single ``all_gather_object`` that collects per-rank free-page counts,
    per-rank-owned sequence state, and per-rank-owned load candidates
    (PREFILLED / ON_HOLD sequences that are not already in decode). Maps
    legacy ``_boundary_gather_state`` (batchgen_worker.py:6726-6804).
  - ``absorb_cross_rank_metadata(decode_uuids, all_payloads, adapter)``:
    applies the gathered state to local shadow ``SequenceEntry`` objects
    for sequences owned by other ranks; handles the ``missing_uuids``
    orphan path by forcibly completing the stragglers and pruning them
    from ``decode_uuids``. Maps legacy metadata-absorption slice in
    ``_boundary_merge_and_decide`` (batchgen_worker.py:6836-6888).
  - ``broadcast_plan(plan)``: rank 0 broadcasts the computed
    :class:`BoundaryPlan`; other ranks pass ``None`` and receive the
    plan through the shared collective backend.

The decision logic that consumes the gathered state lives in
:mod:`batchgen.worker.boundary.planner`. The synchronizer deliberately
does not compute any decisions; it only moves data between ranks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from batchgen.lifespan import SeqEvent
from batchgen.sequence import SequenceStatus
from batchgen.worker.boundary.decisions import BoundaryPlan
from batchgen.worker.protocols import UUID, CollectiveBackend, LegacyInfraBackend
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator


# ---------------------------------------------------------------------------
# Gather payload types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeqBoundaryState:
    """Per-sequence state shared from the owning rank to every other rank.

    Mirrors the legacy ``local_seq_state`` dict at
    ``batchgen_worker.py:6755-6769``. Every field is what the planner
    needs to decide completion / host-growth / GPU-extension / eviction
    for this sequence. ``owning_rank`` is stamped during absorption so
    the planner can route per-rank-local operations back to the right
    rank without consulting any other map.
    """

    decoded_length: int
    current_context_length: int
    gpu_pages_allocated: int
    eos_reached: bool
    completed: bool
    additional_pages_needed: int
    assigned_rank: int
    needs_host_growth: bool
    host_growth_pages: int
    host_pages_allocated: int
    host_token_capacity: int
    prompt_length: int
    total_decoded_before_eviction: int
    owning_rank: int = -1


@dataclass(frozen=True)
class LoadCandidateState:
    """Per-sequence state for a PREFILLED / ON_HOLD load candidate.

    Legacy builds this from ``_uuid_to_local_map`` uuids not already in
    ``decode_uuids`` whose status is in ``{PREFILLED, ON_HOLD}``; see
    ``batchgen_worker.py:6775-6790``. The planner uses ``pages_needed``
    to fit new loads into per-rank free budgets and ``decoded_length``
    to prioritise longer-running sequences first.
    """

    pages_needed: int
    assigned_rank: int
    status: str                 # ``SequenceStatus`` name
    decoded_length: int


@dataclass(frozen=True)
class BoundaryPayload:
    """One rank's contribution to the boundary all_gather.

    Mirrors the legacy ``local_payload`` dict at
    ``batchgen_worker.py:6793-6797``. Always carries the rank's own free
    pages plus the two per-uuid state tables (empty dicts when the rank
    owns no sequences of the relevant kind). The boundary all_gather
    emits exactly one of these per rank.
    """

    free_pages: int
    seq_state: dict[UUID, SeqBoundaryState] = field(default_factory=dict)
    candidate_state: dict[UUID, LoadCandidateState] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BoundarySynchronizer
# ---------------------------------------------------------------------------


class BoundarySynchronizer:
    def __init__(
        self,
        state: WorkerState,
        sync: SyncCoordinator,
        collectives: CollectiveBackend,
    ) -> None:
        self._state = state
        self._sync = sync
        self._collectives = collectives

    # ------------------------------------------------------------------
    # sync_metadata_in
    # ------------------------------------------------------------------

    def sync_metadata_in(self, uuids: list[UUID]) -> None:
        """Refresh cross-rank metadata for `uuids` before planning."""
        self._sync.sync_metadata(uuids)

    # ------------------------------------------------------------------
    # gather_boundary_state
    # ------------------------------------------------------------------

    def gather_boundary_state(
        self,
        decode_uuids: list[UUID],
        gpu_manager: object,
        adapter: LegacyInfraBackend,
    ) -> tuple[list[BoundaryPayload | None], int]:
        """Phase 1 port: single all_gather of seq state + free pages.

        Every rank contributes a :class:`BoundaryPayload` describing its
        free GPU pages, the sequences it owns from ``decode_uuids``, and
        any PREFILLED / ON_HOLD candidates it could asynchronously load
        onto the GPU. Returns the per-rank payload list plus the
        effective chunk size (needed by host-growth accounting in the
        planner and executor).

        The collective pattern matches legacy: one
        ``all_gather_object``; no tensor-backed all_reduce is used
        because the payloads carry heterogeneous types.

        Parameters:
            decode_uuids: The IN_DECODE cohort entering this boundary.
            gpu_manager: The rank's local GPU paged KV manager. Legacy
                reads ``get_stats().num_free_pages`` on it; callers
                pass ``None`` (or a manager with ``is_initialized =
                False``) when the manager is not yet live — the
                payload records zero free pages in that case.
            adapter: LegacyInfraBackend for ``uuid_to_local_map()`` and
                ``effective_chunk_size()`` lookups.
        """
        rank = self._state.rank

        local_free_pages = 0
        if gpu_manager is not None and getattr(gpu_manager, "is_initialized", False):
            stats = gpu_manager.get_stats()
            local_free_pages = int(getattr(stats, "num_free_pages", 0))

        chunk_size = int(adapter.effective_chunk_size())
        uuid_to_local = adapter.uuid_to_local_map()

        local_seq_state: dict[UUID, SeqBoundaryState] = {}
        for uuid in decode_uuids:
            if uuid not in uuid_to_local:
                continue
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            is_completed = bool(adapter.is_sequence_completed(seq))
            local_seq_state[uuid] = SeqBoundaryState(
                decoded_length=seq.decoded_length,
                current_context_length=seq.current_context_length,
                gpu_pages_allocated=seq.gpu_pages_allocated,
                eos_reached=seq.eos_reached,
                completed=is_completed,
                additional_pages_needed=seq.get_additional_gpu_pages_needed(),
                assigned_rank=seq.assigned_rank,
                needs_host_growth=seq.needs_host_kv_growth(chunk_size),
                host_growth_pages=seq.get_host_growth_pages(chunk_size),
                host_pages_allocated=seq.host_pages_allocated,
                host_token_capacity=seq.host_token_capacity,
                prompt_length=seq.prompt_length,
                total_decoded_before_eviction=seq.total_decoded_before_eviction,
            )

        decode_set = set(decode_uuids)
        valid_load_statuses = {SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD}
        local_candidate_state: dict[UUID, LoadCandidateState] = {}
        for uuid in uuid_to_local.keys():
            if uuid in decode_set:
                continue
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            if seq.status == SequenceStatus.COMPLETED:
                continue
            if seq.status not in valid_load_statuses:
                continue
            local_candidate_state[uuid] = LoadCandidateState(
                pages_needed=seq.get_gpu_pages_for_two_page_buffer(),
                assigned_rank=seq.assigned_rank,
                status=seq.status.name,
                decoded_length=seq.decoded_length,
            )

        local_payload = BoundaryPayload(
            free_pages=local_free_pages,
            seq_state=local_seq_state,
            candidate_state=local_candidate_state,
        )

        all_payloads: list[BoundaryPayload | None] = [None] * self._state.world_size
        # Seed self-slot; the collective backend is responsible for
        # populating the peer slots.
        all_payloads[rank] = local_payload
        self._collectives.all_gather_object(all_payloads, local_payload)  # type: ignore[arg-type]
        return all_payloads, chunk_size

    # ------------------------------------------------------------------
    # absorb_cross_rank_metadata
    # ------------------------------------------------------------------

    def absorb_cross_rank_metadata(
        self,
        decode_uuids: list[UUID],
        all_payloads: list[BoundaryPayload | None],
        adapter: LegacyInfraBackend,
    ) -> tuple[list[UUID], dict[UUID, SeqBoundaryState], dict[UUID, LoadCandidateState], list[int]]:
        """Port the metadata-absorption slice of _boundary_merge_and_decide.

        Walks every rank's payload and (a) stamps ``owning_rank`` on
        each :class:`SeqBoundaryState` as we merge into the global
        table, (b) copies the gathered scalars back onto local shadow
        ``SequenceEntry`` objects for sequences this rank does not own,
        fast-fixing CTX drift in the process, and (c) handles the
        ``missing_uuids`` orphan path by force-completing the missing
        sequences and pruning them from ``decode_uuids``.

        Returns:
            ``(decode_uuids, global_seq_state, global_candidate_info,
            per_rank_free)``. The ``decode_uuids`` list is the same
            input list when no orphans were found, otherwise a new list
            with orphaned uuids removed.
        """
        rank = self._state.rank
        uuid_to_local = adapter.uuid_to_local_map()
        sequences_with_gpu_kv = adapter.sequences_with_gpu_kv()

        per_rank_free: list[int] = [0] * self._state.world_size
        global_seq_state: dict[UUID, SeqBoundaryState] = {}
        global_candidate_info: dict[UUID, LoadCandidateState] = {}

        for rank_idx, payload in enumerate(all_payloads):
            if payload is None:
                continue
            per_rank_free[rank_idx] = int(payload.free_pages)
            for uuid, seq_state in payload.seq_state.items():
                # Re-stamp ``owning_rank`` so the planner can route
                # rank-local work without a secondary lookup. Frozen
                # dataclass → rebuild via dataclasses.replace equivalent.
                global_seq_state[uuid] = SeqBoundaryState(
                    decoded_length=seq_state.decoded_length,
                    current_context_length=seq_state.current_context_length,
                    gpu_pages_allocated=seq_state.gpu_pages_allocated,
                    eos_reached=seq_state.eos_reached,
                    completed=seq_state.completed,
                    additional_pages_needed=seq_state.additional_pages_needed,
                    assigned_rank=seq_state.assigned_rank,
                    needs_host_growth=seq_state.needs_host_growth,
                    host_growth_pages=seq_state.host_growth_pages,
                    host_pages_allocated=seq_state.host_pages_allocated,
                    host_token_capacity=seq_state.host_token_capacity,
                    prompt_length=seq_state.prompt_length,
                    total_decoded_before_eviction=seq_state.total_decoded_before_eviction,
                    owning_rank=rank_idx,
                )
            for uuid, cand in payload.candidate_state.items():
                global_candidate_info[uuid] = cand

        # Orphan detection + force-completion. Matches legacy
        # batchgen_worker.py:6836-6867 line-for-line in intent: the
        # planner must only see decode_uuids it has authoritative state
        # for, and the worker must not keep KV around for sequences
        # nobody reported on.
        missing_uuids = [u for u in decode_uuids if u not in global_seq_state]
        if missing_uuids:
            for orphan_uuid in missing_uuids:
                orphan_seq = self._state.global_batch.get_sequence(orphan_uuid)
                if orphan_seq is None:
                    continue
                orphan_seq.gpu_pages_allocated = 0
                orphan_seq.host_pages_allocated = 0
                orphan_seq.host_token_capacity = 0
                sequences_with_gpu_kv.discard(orphan_uuid)
                if orphan_seq.status != SequenceStatus.COMPLETED:
                    try:
                        self._state.global_batch.update_status(
                            orphan_uuid, SequenceStatus.COMPLETED
                        )
                    except ValueError:
                        # Transition already rejected by the state
                        # machine (e.g. QUEUEING → COMPLETED is invalid).
                        # Swallowing matches legacy behaviour: the
                        # orphan has no state to preserve, so the
                        # status machine's verdict stays authoritative.
                        pass
            decode_uuids = [u for u in decode_uuids if u in global_seq_state]

        # Copy gathered fields onto local shadow SequenceEntry objects
        # for sequences this rank does not own. Legacy lines
        # batchgen_worker.py:6869-6888.
        for uuid, seq_state in global_seq_state.items():
            if uuid in uuid_to_local:
                continue
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            seq.decoded_length = seq_state.decoded_length
            seq.current_context_length = seq_state.current_context_length
            seq.gpu_pages_allocated = seq_state.gpu_pages_allocated
            seq.eos_reached = seq_state.eos_reached
            seq.host_pages_allocated = seq_state.host_pages_allocated
            seq.host_token_capacity = seq_state.host_token_capacity
            seq.prompt_length = seq_state.prompt_length
            seq.total_decoded_before_eviction = seq_state.total_decoded_before_eviction
            expected_ctx = seq.original_prompt_length + seq.decoded_length
            if seq.current_context_length != expected_ctx:
                seq.log_event(
                    SeqEvent.CTX_MISMATCH,
                    rank,
                    f"gathered_ctx={seq.current_context_length}, expected={expected_ctx}",
                )
                seq.current_context_length = expected_ctx

        return decode_uuids, global_seq_state, global_candidate_info, per_rank_free

    # ------------------------------------------------------------------
    # broadcast_plan
    # ------------------------------------------------------------------

    def broadcast_plan(self, plan: BoundaryPlan | None) -> BoundaryPlan:
        """Broadcast a BoundaryPlan from rank 0 to every rank.

        On rank 0 the caller passes the locally-computed plan; every
        other rank passes ``None`` and receives the broadcast. Returns
        the plan value every rank must feed into the executor.

        Non-rank-0 without an injected response raises
        ``AssertionError`` via the fake — production uses
        `torch.distributed.broadcast_object_list`.
        """
        obj_list: list[BoundaryPlan | None] = [plan]
        self._collectives.broadcast_object(obj_list, src=0)
        received = obj_list[0]
        if received is None:
            # Every rank should end up with a real plan after the broadcast.
            # None here means the broadcast payload was mis-constructed on
            # rank 0; fail loudly rather than quietly returning an empty
            # BoundaryPlan.
            raise RuntimeError(
                "BoundarySynchronizer.broadcast_plan: received None after "
                "broadcast_object; rank 0 must provide a non-None plan"
            )
        return received


__all__ = [
    "BoundaryPayload",
    "SeqBoundaryState",
    "LoadCandidateState",
    "BoundarySynchronizer",
]
