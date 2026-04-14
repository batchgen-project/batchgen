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

from dataclasses import dataclass
from typing import Any

from batchgen.sequence import SequenceStatus
from batchgen.worker.boundary.decisions import (
    AsyncLoadHostToGpu,
    BoundaryPlan,
    BoundaryResult,
    Evict,
    EvictReason,
    ExtendPages,
    HostEvict,
    HostGrow,
    NewLoadAsync,
    OnHold,
    OnHoldReason,
    PageBoundaryDecision,
    ReleasePages,
    SeqMetadata,
)
from batchgen.worker.boundary.executor import BoundaryExecutor
from batchgen.worker.boundary.finalize import finalize as _finalize
from batchgen.worker.boundary.guards import BoundaryGuards, GuardViolation
from batchgen.worker.boundary.planner import (
    BoundaryPlanner,
    PlannerConfig,
    WorkerViewStats,
)
from batchgen.worker.boundary.synchronizer import BoundarySynchronizer
from batchgen.worker.boundary.wait_pending import wait_pending as _wait_pending
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import UUID, CollectiveBackend, LegacyInfraBackend
from batchgen.worker.state import WorkerState


@dataclass(frozen=True)
class BoundaryHandlerConfig:
    """Knobs the Phase 2.8.1 ``run_full`` handler reads.

    Kept distinct from :class:`PlannerConfig` so the planner can be
    constructed without the handler-specific runtime switches. Every
    field is plumbed from ``BatchGenWorker.__init__`` via the
    orchestrator; handlers never read ``os.environ``.
    """

    enable_host_kv_eviction: bool
    host_kv_eviction_watermark: int   # e.g. 20 (%)


class BoundaryHandler:
    def __init__(
        self,
        state: WorkerState,
        synchronizer: BoundarySynchronizer,
        planner: BoundaryPlanner,
        executor: BoundaryExecutor,
        guards: BoundaryGuards,
        kv: KVCacheManager,
        *,
        adapter: LegacyInfraBackend | None = None,
        collectives: CollectiveBackend | None = None,
        handler_config: BoundaryHandlerConfig | None = None,
    ) -> None:
        self._state = state
        self._sync = synchronizer
        self._planner = planner
        self._executor = executor
        self._guards = guards
        self._kv = kv
        # Phase 2.8.1i: the ``run_full`` path requires an adapter +
        # collectives + handler config. The M4 ``run`` path does not —
        # tests still construct the handler without the new args.
        self._adapter = adapter
        self._collectives = collectives
        self._handler_config = handler_config

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

    # ==================================================================
    # Phase 2.8.1i — native boundary orchestration
    # ==================================================================

    def run_full(
        self,
        *,
        decode_uuids: list[UUID],
        batch: list[int],
        gpu_manager: Any,
        pending_async_task: Any = None,
        pending_load_uuids: list[UUID] | None = None,
        pending_load_local: list[int] | None = None,
        pending_load_global: list[int] | None = None,
    ) -> BoundaryResult:
        """Run one full boundary cycle through the Stage 1 native path.

        Signature matches legacy ``_page_boundary_fast``
        (batchgen_worker.py:7336) so the Stage 2 decode-loop port can
        swap the call 1:1. Order mirrors
        ``docs/phase_2.8_stage1_design.md §3.7``:

          1. ``wait_pending`` drains deferred KV writes + integrates
             any prior-cycle async load.
          2. ``synchronizer.sync_metadata_in`` (CTX fast-fail).
          3. ``synchronizer.gather_boundary_state`` — one
             ``all_gather_object``.
          4. ``synchronizer.absorb_cross_rank_metadata`` — apply peer
             scalars + handle orphan path.
          5. ``_build_snapshot`` for the planner.
          6. Rank-0 ``planner.plan_full`` + broadcast.
          7. ``guards.check_pre(plan)``.
          8. ``executor.apply_full(plan)`` returns the updated
             ``(decode_uuids, batch, new_async_task, new_load_uuids,
             new_load_local, new_load_global)``.
          9. ``finalize`` — page-table rebuild, MoE sync, barrier,
             watermark check.
         10. ``guards.check_post() + check_post_page_table_order``.
         11. Return :class:`BoundaryResult`.

        Raises:
            RuntimeError: when the handler was not constructed with
                ``adapter=`` + ``collectives=`` + ``handler_config=``.
                The M4 ``run`` method stays usable without these; only
                ``run_full`` requires them.
        """
        if (
            self._adapter is None
            or self._collectives is None
            or self._handler_config is None
        ):
            raise RuntimeError(
                "BoundaryHandler.run_full requires adapter, collectives, "
                "and handler_config to be passed at construction."
            )
        adapter = self._adapter
        collectives = self._collectives
        cfg = self._handler_config

        pending_load_uuids = list(pending_load_uuids or [])
        pending_load_local = list(pending_load_local or [])
        pending_load_global = list(pending_load_global or [])

        # Step 1 — drain + integrate prior async load.
        decode_uuids, batch = _wait_pending(
            self._state, adapter, collectives,
            decode_uuids=list(decode_uuids), batch=list(batch),
            gpu_manager=gpu_manager,
            pending_async_load_task=pending_async_task,
            pending_load_uuids=pending_load_uuids,
            pending_load_local=pending_load_local,
            pending_load_global=pending_load_global,
        )
        if not decode_uuids:
            return BoundaryResult(plan=BoundaryPlan())

        # Step 2 — metadata sync (CTX fast-fail).
        self._sync.sync_metadata_in(list(decode_uuids))

        # Step 3 — gather per-rank payloads.
        all_payloads, chunk_size = self._sync.gather_boundary_state(
            list(decode_uuids), gpu_manager, adapter,
        )

        # Step 4 — absorb cross-rank metadata + orphan handling.
        (
            decode_uuids,
            global_seq_state,
            global_candidate_info,
            per_rank_free,
        ) = self._sync.absorb_cross_rank_metadata(
            list(decode_uuids), all_payloads, adapter,
        )

        # Step 5 — build the planner snapshot.
        snapshot = self._build_snapshot(decode_uuids)

        # Step 6 — rank-0 plans + broadcast to peers.
        priority_by_uuid, global_idx_by_uuid = self._planner_aux_dicts(
            list(global_seq_state.keys())
        )
        worker_view_stats = self._worker_view_stats(adapter)
        local_plan: BoundaryPlan | None = None
        if self._state.rank == 0:
            local_plan = self._planner.plan_full(
                list(decode_uuids),
                global_seq_state=global_seq_state,
                global_candidate_info=global_candidate_info,
                per_rank_free=list(per_rank_free),
                chunk_size=chunk_size,
                worker_view_stats=worker_view_stats,
                has_pending=self._has_pending_work(),
                world_size=self._state.world_size,
                enable_host_kv_eviction=cfg.enable_host_kv_eviction,
                host_kv_eviction_watermark=cfg.host_kv_eviction_watermark,
                priority_by_uuid=priority_by_uuid,
                global_idx_by_uuid=global_idx_by_uuid,
            )
        plan = self._sync.broadcast_plan(local_plan)

        # Step 7 — pre-execution sanity.
        self._guards.check_pre(plan)

        # Step 8 — canonical-order apply.
        (
            decode_uuids,
            batch,
            new_async_task,
            new_load_uuids,
            new_load_local,
            new_load_global,
        ) = self._executor.apply_full(
            plan,
            decode_uuids=list(decode_uuids),
            batch=list(batch),
            gpu_manager=gpu_manager,
            global_seq_state=global_seq_state,
            global_candidate_info=global_candidate_info,
            chunk_size=chunk_size,
            adapter=adapter,
        )
        if not decode_uuids:
            return BoundaryResult(
                plan=plan,
                new_async_task=new_async_task,
                new_load_uuids=tuple(new_load_uuids),
                new_load_local=tuple(new_load_local),
                new_load_global=tuple(new_load_global),
            )

        # Step 9 — finalize (page table + MoE + watermark).
        batch, watermark_triggered = _finalize(
            self._state, adapter, collectives,
            decode_uuids=list(decode_uuids), batch=list(batch),
            gpu_manager=gpu_manager,
        )

        # Step 10 — post guards.
        self._guards.check_post()
        self._guards.check_post_page_table_order(adapter, gpu_manager, batch)

        # Step 11 — result.
        return BoundaryResult(
            plan=plan,
            decode_uuids=tuple(decode_uuids),
            batch=tuple(batch),
            new_async_task=new_async_task,
            new_load_uuids=tuple(new_load_uuids),
            new_load_local=tuple(new_load_local),
            new_load_global=tuple(new_load_global),
            watermark_triggered=watermark_triggered,
        )

    # ------------------------------------------------------------------
    # Helpers for run_full
    # ------------------------------------------------------------------

    def _planner_aux_dicts(
        self, uuids: list[UUID]
    ) -> tuple[dict[UUID, int], dict[UUID, int]]:
        """Build the priority + global_idx aux dicts from state.

        Threads the two legacy tie-breaker fields into ``plan_full``
        without the planner having to reach into ``state.global_batch``
        (plan stays pure). Missing uuids are simply absent; the planner
        falls back to ``priority=0`` / ``global_idx=inf``.
        """
        priorities: dict[UUID, int] = {}
        global_ids: dict[UUID, int] = {}
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            priorities[uuid] = getattr(seq, "priority", 0)
            global_ids[uuid] = seq.global_idx
        return priorities, global_ids

    def _worker_view_stats(
        self, adapter: LegacyInfraBackend
    ) -> WorkerViewStats | None:
        """Read host-KV occupancy on rank 0 only.

        The planner reads stats on rank 0; peers get the plan broadcast
        from rank 0 anyway, so they do not need to re-query the view.
        Returns ``None`` when the worker view is not yet live (pre-
        registration; tests).
        """
        if self._state.rank != 0:
            return None
        view = adapter.host_paged_kv_worker_view()
        if view is None:
            return None
        stats = view.get_stats()
        return WorkerViewStats(
            num_total_pages=int(getattr(stats, "num_total_pages", 0)),
            num_free_pages=int(getattr(stats, "num_free_pages", 0)),
        )


__all__ = [
    "AsyncLoadHostToGpu",
    "BoundaryPlan",
    "BoundaryResult",
    "Evict",
    "EvictReason",
    "ExtendPages",
    "HostEvict",
    "HostGrow",
    "NewLoadAsync",
    "OnHold",
    "OnHoldReason",
    "PageBoundaryDecision",
    "ReleasePages",
    "SeqMetadata",
    "BoundaryExecutor",
    "BoundaryGuards",
    "BoundaryHandler",
    "BoundaryHandlerConfig",
    "BoundaryPlanner",
    "BoundarySynchronizer",
    "GuardViolation",
    "PlannerConfig",
    "WorkerViewStats",
]
