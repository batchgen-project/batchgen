"""BoundaryHandler — orchestrates a full page-boundary cycle.

Public surface of the ``batchgen.worker.boundary`` sub-package. The
handler wires the four collaborators the sub-package splits into:

  - :class:`BoundarySynchronizer` — cross-rank sync + plan broadcast.
  - :class:`BoundaryPlanner` — pure rank-0 decision computation.
  - :class:`BoundaryExecutor` — applies the plan in canonical order.
  - :class:`BoundaryGuards` — pre/post invariant checks.

``run(uuids)`` drives one boundary cycle:

  1. ``synchronizer.sync_metadata_in(uuids)`` — every rank agrees on
     the per-sequence metadata; CTX fast-fail lands here.
  2. Build the planner snapshot from ``state.global_batch`` for the
     caller-supplied ``uuids``.
  3. On rank 0, call ``planner.plan`` with the snapshot + free page
     counts + ``has_pending`` flag. Other ranks pass ``None`` to the
     broadcast (the fake path in tests; production uses
     ``torch.distributed.broadcast_object_list``).
  4. ``synchronizer.broadcast_plan`` returns the authoritative plan on
     every rank.
  5. ``guards.check_pre(plan)`` — all plan UUIDs exist in state.
  6. ``executor.apply(plan)`` — canonical-order application.
  7. ``guards.check_post()`` — CTX invariant + index map consistency.

Every step is required; skipping any one of them has broken main in
prior bug tails. The sequence is intentionally linear and trivial to
read — complexity lives inside the collaborators.
"""

from __future__ import annotations

from batchgen.sequence import SequenceStatus
from batchgen.worker.boundary.decisions import (
    AsyncLoadHostToGpu,
    BoundaryPlan,
    Evict,
    EvictReason,
    ExtendPages,
    OnHold,
    OnHoldReason,
    PageBoundaryDecision,
    ReleasePages,
    SeqMetadata,
)
from batchgen.worker.boundary.executor import BoundaryExecutor
from batchgen.worker.boundary.guards import BoundaryGuards, GuardViolation
from batchgen.worker.boundary.planner import BoundaryPlanner, PlannerConfig
from batchgen.worker.boundary.synchronizer import BoundarySynchronizer
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import UUID
from batchgen.worker.state import WorkerState


class BoundaryHandler:
    def __init__(
        self,
        state: WorkerState,
        synchronizer: BoundarySynchronizer,
        planner: BoundaryPlanner,
        executor: BoundaryExecutor,
        guards: BoundaryGuards,
        kv: KVCacheManager,
    ) -> None:
        self._state = state
        self._sync = synchronizer
        self._planner = planner
        self._executor = executor
        self._guards = guards
        self._kv = kv

    def run(self, uuids: list[UUID]) -> BoundaryPlan:
        """Run one full boundary cycle, return the executed plan.

        Steps are executed in the fixed order documented at the module
        level. Returns the plan so callers (DecodeScheduler, tests)
        can inspect what was applied.
        """
        # 1. Cross-rank metadata sync — CTX fast-fail here
        self._sync.sync_metadata_in(list(uuids))

        # 2. Build the snapshot from the now-synchronized state
        snapshot = self._build_snapshot(uuids)

        # 3. Plan on rank 0 (other ranks pass None to the broadcast)
        if self._state.rank == 0:
            local_plan: BoundaryPlan | None = self._planner.plan(
                snapshot,
                gpu_free=self._kv.get_gpu_free_pages(),
                host_free=self._kv.get_host_free_pages(),
                has_pending=self._has_pending_work(),
            )
        else:
            local_plan = None

        # 4. Broadcast from rank 0 to every rank
        plan = self._sync.broadcast_plan(local_plan)

        # 5. Pre-execution sanity
        self._guards.check_pre(plan)

        # 6. Canonical-order execution
        self._executor.apply(plan)

        # 7. Post-execution invariants
        self._guards.check_post()

        return plan

    # ------------------------------------------------------------------
    # Snapshot construction
    # ------------------------------------------------------------------

    def _build_snapshot(self, uuids: list[UUID]) -> dict[UUID, SeqMetadata]:
        """Project ``state.global_batch[uuid]`` onto a per-sequence
        :class:`SeqMetadata` view for every uuid in `uuids`. Missing
        uuids are skipped silently (they may have just completed and
        been removed)."""
        snapshot: dict[UUID, SeqMetadata] = {}
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            snapshot[uuid] = SeqMetadata(
                uuid=seq.uuid,
                global_idx=seq.global_idx,
                status=int(seq.status),
                assigned_rank=seq.assigned_rank if seq.assigned_rank is not None else -1,
                prompt_length=seq.prompt_length,
                max_decode_length=seq.max_decode_length,
                decoded_length=seq.decoded_length,
                current_context_length=seq.current_context_length,
                gpu_pages_allocated=seq.gpu_pages_allocated,
                host_pages_allocated=seq.host_pages_allocated,
                had_initial_gpu_reservation=seq.had_initial_gpu_reservation,
                eos_reached=seq.eos_reached,
                rep_detected=seq._rep_detected,
            )
        return snapshot

    def _has_pending_work(self) -> bool:
        """True if any QUEUEING or EVICTED sequence is waiting for prefill.

        Drives the planner's watermark-trigger bailout — we only switch
        from decode to prefill if there is actual work to prefill.
        """
        gb = self._state.global_batch
        return bool(
            gb.get_sequences_by_status(SequenceStatus.QUEUEING)
            or gb.get_sequences_by_status(SequenceStatus.EVICTED)
        )


__all__ = [
    "AsyncLoadHostToGpu",
    "BoundaryPlan",
    "Evict",
    "EvictReason",
    "ExtendPages",
    "OnHold",
    "OnHoldReason",
    "PageBoundaryDecision",
    "ReleasePages",
    "SeqMetadata",
    "BoundaryExecutor",
    "BoundaryGuards",
    "BoundaryHandler",
    "BoundaryPlanner",
    "BoundarySynchronizer",
    "GuardViolation",
    "PlannerConfig",
]
