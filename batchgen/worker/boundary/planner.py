"""BoundaryPlanner — rank-0 pure-function decision computation.

Given a snapshot of per-sequence metadata, the current GPU/host free-page
counts, a ``has_pending`` flag (queued or evicted work exists), and a
small :class:`PlannerConfig`, produce a :class:`BoundaryPlan` listing
the boundary operations to execute on every rank.

The planner **never** touches ``state.global_batch``, issues a
collective, or mutates any backend. It is a pure function from
``(snapshot, free counts, config)`` to ``BoundaryPlan``. This keeps it
trivially testable — fixture in, plan out, compare — and makes the
cross-rank behavior deterministic (every rank computes the same plan
from the same inputs, so rank-0's broadcast is a no-op on happy-path).

Rules implemented in M4 (executor order drives the sequence):

  1. **Release** — every sequence whose ``status == COMPLETED``.

  2. **Prefill watermark trigger** — if ``has_pending`` AND host-free
     percentage > ``prefill_watermark_pct``, OnHold every IN_DECODE
     sequence with ``reason=WATERMARK_TRIGGER``. This matches main's
     ``_check_host_kv_watermark_trigger`` bailout (plan Decision #3 +
     POIS's Q1a). When the bailout fires, no Extend / AsyncLoad
     decisions are emitted — the whole batch is yielding to prefill.

  3. **Per-seq Extend vs OnHold(EXTENSION_FAILED)** — for every
     IN_DECODE sequence, look at how much head-room its GPU allocation
     has (in pages). If the next decision interval would overflow
     (POIS Q4: "next decision would overflow current allocation"),
     try to extend by ``decision_frequency_pages`` pages. Sequences
     that cannot be extended because ``gpu_free`` is exhausted get
     bundled into a single OnHold(EXTENSION_FAILED) decision and are
     sorted by ``ShortestDecodedFirstStrategy`` so the shortest-decoded
     sequences are the ones held (preserve longer-running progress).

  4. **AsyncLoadHostToGpu** — out of scope for M4. Landing in M5
     alongside DecodeScheduler.try_load_new where the async handle
     machinery lives.

  5. **Evict** — out of scope for M4. Per POIS Q1c, Evict fires when
     a sequence can't expand its host-KV chunk reservation and has to
     re-prefill from ``evicted_token_ids``. That path is tied to the
     host-side chunk reservation state which DecodeScheduler owns;
     lands alongside DecodeScheduler in M5.

``Evict`` and ``AsyncLoadHostToGpu`` still appear in the sealed
:class:`PageBoundaryDecision` union and in :class:`BoundaryExecutor`,
so adding the missing planner paths later is purely additive.
"""

from __future__ import annotations

from dataclasses import dataclass

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary.decisions import (
    BoundaryPlan,
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
from batchgen.worker.boundary.synchronizer import (
    LoadCandidateState,
    SeqBoundaryState,
)
from batchgen.worker.protocols import UUID


@dataclass(frozen=True)
class WorkerViewStats:
    """Snapshot of host paged-KV occupancy the planner consumes.

    Fills the role of ``worker_view.get_stats()`` inside legacy
    ``_compute_boundary_decisions`` (batchgen_worker.py:6484/6498)
    without requiring the planner to hold a live reference. The
    BoundaryHandler reads the stats on rank 0 before calling
    ``plan_full`` so the planner stays a pure function.
    """

    num_total_pages: int
    num_free_pages: int


@dataclass(frozen=True)
class PlannerConfig:
    """Static knobs the planner reads.

    Every field here is plumbed from :class:`BatchGenWorker.__init__`
    via the orchestrator; handlers never read ``os.environ``.
    """

    prefill_watermark_pct: int          # e.g. 70
    decision_frequency_pages: int       # e.g. 2 (main default)
    extension_gpu_page_buffer: int      # e.g. 4 (main default)
    host_total_pages: int               # for watermark % math
    page_size_tokens: int = SequenceEntry.PAGE_SIZE  # 64


class BoundaryPlanner:
    def __init__(self, config: PlannerConfig) -> None:
        if config.decision_frequency_pages < 1:
            raise ValueError(
                f"decision_frequency_pages must be >= 1, got {config.decision_frequency_pages}"
            )
        if config.host_total_pages < 1:
            raise ValueError(
                f"host_total_pages must be >= 1, got {config.host_total_pages}"
            )
        self._cfg = config

    def plan(
        self,
        snapshot: dict[UUID, SeqMetadata],
        *,
        gpu_free: int,
        host_free: int,
        has_pending: bool,
    ) -> BoundaryPlan:
        """Compute the boundary plan from the given snapshot + budgets.

        Pure function — every input is a value, every output is a
        frozen dataclass. Run this on rank 0 and broadcast the result.
        """
        decisions: list[PageBoundaryDecision] = []

        # Rule 1: ReleasePages for completed sequences.
        release = self._find_releases(snapshot)
        if release is not None:
            decisions.append(release)

        # Rule 2: prefill watermark trigger — OnHold all IN_DECODE, bail.
        if self._prefill_watermark_fired(host_free, has_pending):
            onhold = self._onhold_all_in_decode(
                snapshot, OnHoldReason.WATERMARK_TRIGGER
            )
            if onhold is not None:
                decisions.append(onhold)
            return BoundaryPlan(
                decisions=tuple(decisions),
                metadata_snapshot=dict(snapshot),
            )

        # Rule 3: per-seq Extend vs OnHold(EXTENSION_FAILED).
        extend_decisions, held = self._plan_per_seq_extension(
            snapshot, gpu_free_after_release=gpu_free
        )
        if held:
            decisions.append(
                OnHold(uuids=tuple(held), reason=OnHoldReason.EXTENSION_FAILED)
            )
        # Extend decisions are added AFTER the OnHold so the executor's
        # canonical order (Release > Evict > OnHold > Extend > AsyncLoad)
        # lines up with the plan's list order 1:1.
        decisions.extend(extend_decisions)

        return BoundaryPlan(
            decisions=tuple(decisions),
            metadata_snapshot=dict(snapshot),
        )

    # ------------------------------------------------------------------
    # Rule 1: releases
    # ------------------------------------------------------------------

    @staticmethod
    def _find_releases(
        snapshot: dict[UUID, SeqMetadata],
    ) -> ReleasePages | None:
        uuids = sorted(
            m.uuid
            for m in snapshot.values()
            if m.status == SequenceStatus.COMPLETED
        )
        if not uuids:
            return None
        return ReleasePages(uuids=tuple(uuids))

    # ------------------------------------------------------------------
    # Rule 2: prefill watermark trigger
    # ------------------------------------------------------------------

    def _prefill_watermark_fired(self, host_free: int, has_pending: bool) -> bool:
        if not has_pending:
            return False
        free_pct = (host_free * 100) // self._cfg.host_total_pages
        return free_pct > self._cfg.prefill_watermark_pct

    @staticmethod
    def _onhold_all_in_decode(
        snapshot: dict[UUID, SeqMetadata], reason: OnHoldReason
    ) -> OnHold | None:
        uuids = sorted(
            m.uuid for m in snapshot.values() if m.status == SequenceStatus.IN_DECODE
        )
        if not uuids:
            return None
        return OnHold(uuids=tuple(uuids), reason=reason)

    # ------------------------------------------------------------------
    # Rule 3: per-seq Extend vs OnHold(EXTENSION_FAILED)
    # ------------------------------------------------------------------

    def _plan_per_seq_extension(
        self, snapshot: dict[UUID, SeqMetadata], gpu_free_after_release: int
    ) -> tuple[list[ExtendPages], list[UUID]]:
        """Pick the IN_DECODE sequences that need more GPU pages this
        boundary; split into ``extends`` (fit) and ``held`` (don't fit).

        "Need more pages" means the next decision interval would push
        ``current_context_length`` past the sequence's current GPU page
        allocation:

            allocated_tokens = gpu_pages_allocated * PAGE_SIZE
            headroom_tokens = allocated_tokens - current_context_length
            need_more = headroom_tokens < decision_frequency_pages * PAGE_SIZE

        Sequences are considered in ``global_idx`` order so cross-rank
        determinism holds. Sequences we can't fit are sorted by
        shortest-decoded-first (``ShortestDecodedFirstStrategy`` in main)
        so the longest-running sequences survive.
        """
        page = self._cfg.page_size_tokens
        freq = self._cfg.decision_frequency_pages
        ext = self._cfg.extension_gpu_page_buffer
        interval_tokens = freq * page

        candidates_ordered = sorted(
            (m for m in snapshot.values() if m.status == SequenceStatus.IN_DECODE),
            key=lambda m: (m.global_idx, m.uuid),
        )

        need_more: list[SeqMetadata] = []
        for meta in candidates_ordered:
            allocated_tokens = meta.gpu_pages_allocated * page
            headroom = allocated_tokens - meta.current_context_length
            if headroom < interval_tokens:
                need_more.append(meta)

        # Try to allocate `ext` more pages per needy sequence, in order.
        extends: list[ExtendPages] = []
        held: list[UUID] = []
        remaining = gpu_free_after_release
        for meta in need_more:
            if remaining >= ext:
                extends.append(ExtendPages(uuid=meta.uuid, additional_pages=ext))
                remaining -= ext
            else:
                held.append(meta.uuid)

        # Sort held shortest-decoded-first; main's behavior is to preserve
        # longer-running sequences by holding the smaller-progress ones.
        if held:
            held_meta = {m.uuid: m for m in need_more}
            held.sort(key=lambda u: (held_meta[u].decoded_length, u))

        return extends, held


    # ==================================================================
    # plan_full — Phase 2.8.1d native boundary port
    # ==================================================================

    def plan_full(
        self,
        decode_uuids: list[UUID],
        *,
        global_seq_state: dict[UUID, SeqBoundaryState],
        global_candidate_info: dict[UUID, LoadCandidateState],
        per_rank_free: list[int],
        chunk_size: int,
        worker_view_stats: WorkerViewStats | None,
        has_pending: bool,
        world_size: int,
        enable_host_kv_eviction: bool,
        host_kv_eviction_watermark: int,
        priority_by_uuid: dict[UUID, int] | None = None,
        global_idx_by_uuid: dict[UUID, int] | None = None,
    ) -> BoundaryPlan:
        """Legacy-faithful rank-0 planning, producing a Stage 1 BoundaryPlan.

        Ports ``_compute_boundary_decisions`` (batchgen_worker.py:6443-6629)
        with two additions native to the Stage 1 schema:

          * The watermark-trigger bailout runs *before* the HostGrow /
            HostEvict / Extend / NewLoadAsync rules. When both the host
            watermark has fired AND ``has_pending`` is true, the planner
            emits ``OnHold(WATERMARK_TRIGGER)`` for every IN_DECODE
            uuid *only* (the L4 root-cause fix — never target a
            QUEUEING / EVICTED / ON_HOLD uuid) and surfaces
            ``watermark_break=True`` on the plan. Decode loop reads
            the break flag to yield back to prefill.
          * ``decode_uuids_final`` is pre-computed on the plan so the
            handler / executor never redoes the subtractions.

        This remains a pure function — no collective, no state mutation.
        The handler is expected to gather ``global_seq_state`` /
        ``global_candidate_info`` / ``per_rank_free`` via the
        synchronizer before calling.

        Parameters:
            decode_uuids: IN_DECODE cohort entering this boundary.
            global_seq_state: gather output from
                :meth:`BoundarySynchronizer.gather_boundary_state`.
            global_candidate_info: gather output for PREFILLED / ON_HOLD
                load candidates.
            per_rank_free: Free GPU pages per rank, index-aligned with
                ``rank``; ``sum(per_rank_free)`` ≈ total cluster free.
            chunk_size: legacy ``_get_effective_chunk_size()`` value
                the planner pipes into host-growth accounting.
            worker_view_stats: host-KV occupancy snapshot; pass
                ``None`` when the worker view is not yet live (tests,
                pre-registration paths).
            has_pending: Whether there are queued / evicted sequences
                waiting for host KV room. Required for the watermark
                bailout rule.
            world_size: number of ranks.
            enable_host_kv_eviction: master switch for Rule 4
                (legacy ``self.enable_host_kv_eviction`` at 6497).
            host_kv_eviction_watermark: % free below which eviction
                fires (legacy ``self.host_kv_eviction_watermark``).
            priority_by_uuid: per-uuid priority (0=NORMAL, 1=HIGH).
                Defaults to 0 when uuid not present.
            global_idx_by_uuid: per-uuid global index. Defaults to
                ``inf`` (never chosen for eviction) when uuid not
                present — matches legacy ``float('inf')`` fallback.
        """
        priorities = priority_by_uuid or {}
        global_ids = global_idx_by_uuid or {}

        decisions: list[PageBoundaryDecision] = []

        # ----- Rule 1: completed vs active split + Release -----
        completed_uuids: list[UUID] = []
        active_uuids: list[UUID] = []
        for uuid in decode_uuids:
            st = global_seq_state.get(uuid)
            if st is not None and st.completed:
                completed_uuids.append(uuid)
            else:
                active_uuids.append(uuid)
        if completed_uuids:
            decisions.append(ReleasePages(uuids=tuple(completed_uuids)))

        # ----- Watermark-trigger bailout (L4 root-cause fix) -----
        # Emitted BEFORE grow/evict/extend/new-load so those rules are
        # short-circuited when the whole batch is yielding back to prefill.
        watermark_break = False
        if (
            has_pending
            and worker_view_stats is not None
            and worker_view_stats.num_total_pages > 0
        ):
            free_pct = (
                worker_view_stats.num_free_pages * 100
                // worker_view_stats.num_total_pages
            )
            if free_pct > self._cfg.prefill_watermark_pct:
                watermark_break = True
                # IN_DECODE-only filter: legacy _put_sequences_on_hold
                # was status-blind, which is the L4 crash. Filter here
                # once, at the planner, so the executor sees a safe
                # set (belt-and-braces: put_on_hold in Stage 3 also
                # re-asserts).
                in_decode_active = tuple(
                    sorted(
                        u for u in active_uuids
                        if global_seq_state.get(u) is not None
                        and global_seq_state[u].gpu_pages_allocated > 0
                    )
                )
                if in_decode_active:
                    decisions.append(
                        OnHold(
                            uuids=in_decode_active,
                            reason=OnHoldReason.WATERMARK_TRIGGER,
                        )
                    )
                return BoundaryPlan(
                    decisions=tuple(decisions),
                    decode_uuids_final=(),
                    watermark_break=True,
                )

        # ----- Rule 2: HostGrow (legacy 6469-6492) -----
        host_growth_uuids: list[UUID] = []
        host_growth_pages: list[int] = []
        total_growth_needed = 0
        for uuid in active_uuids:
            st = global_seq_state.get(uuid)
            if st is not None and st.needs_host_growth and st.host_growth_pages > 0:
                host_growth_uuids.append(uuid)
                host_growth_pages.append(st.host_growth_pages)
                total_growth_needed += st.host_growth_pages

        growth_feasible = False
        if total_growth_needed > 0 and worker_view_stats is not None:
            safety_margin = int(worker_view_stats.num_total_pages * 0.05)
            growth_feasible = total_growth_needed <= (
                worker_view_stats.num_free_pages - safety_margin
            )

        if host_growth_uuids:
            decisions.append(
                HostGrow(
                    uuids=tuple(host_growth_uuids),
                    pages=tuple(host_growth_pages),
                    feasible=growth_feasible,
                )
            )

        # ----- Rule 3: HostEvict (legacy 6494-6528) -----
        host_evicted_uuids: list[UUID] = []
        decode_after_eviction = list(active_uuids)
        if (
            enable_host_kv_eviction
            and active_uuids
            and worker_view_stats is not None
            and worker_view_stats.num_total_pages > 0
        ):
            free_pct = (
                worker_view_stats.num_free_pages
                / worker_view_stats.num_total_pages
                * 100
            )
            if free_pct < host_kv_eviction_watermark:
                eviction_candidates: list[tuple[UUID, dict]] = []
                completed_set = set(completed_uuids)
                for uuid in active_uuids:
                    if uuid in completed_set:
                        continue
                    st = global_seq_state.get(uuid)
                    if st is None:
                        continue
                    eviction_candidates.append(
                        (
                            uuid,
                            {
                                "decoded_length": st.decoded_length,
                                "host_pages_allocated": st.host_pages_allocated,
                                "global_idx": global_ids.get(uuid, float("inf")),
                                "priority": priorities.get(uuid, 0),
                            },
                        )
                    )
                target_free = int(
                    worker_view_stats.num_total_pages
                    * host_kv_eviction_watermark
                    / 100
                )
                pages_to_free = max(0, target_free - worker_view_stats.num_free_pages)
                if pages_to_free > 0 and eviction_candidates:
                    from batchgen.continuous_batching import (
                        EvictionStrategy,
                        select_sequences_for_eviction,
                    )

                    host_evicted_uuids, _ = select_sequences_for_eviction(
                        eviction_candidates,
                        pages_to_free,
                        strategy=EvictionStrategy.SHORTEST_FIRST,
                        page_key="host_pages_allocated",
                    )
                    if host_evicted_uuids:
                        evicted_set = set(host_evicted_uuids)
                        decode_after_eviction = [
                            u for u in active_uuids if u not in evicted_set
                        ]
                        decisions.append(
                            HostEvict(uuids=tuple(host_evicted_uuids))
                        )

        # ----- Rule 4: GPU extension + per-rank OnHold(EXTENSION_FAILED) -----
        # Legacy 6530-6580. Per-rank arithmetic: the planner knows
        # each rank's free pages and each sequence's owning rank, so
        # it can pick which uuids land on hold without any collective.
        seqs_needing_extension: list[UUID] = []
        total_additional_by_rank = [0] * world_size
        for uuid in decode_after_eviction:
            st = global_seq_state.get(uuid)
            if st is not None and st.additional_pages_needed > 0:
                r = st.assigned_rank
                if 0 <= r < world_size:
                    total_additional_by_rank[r] += st.additional_pages_needed
                    seqs_needing_extension.append(uuid)

        all_can_extend = all(
            total_additional_by_rank[r] <= per_rank_free[r]
            for r in range(world_size)
        )

        onhold_uuids: list[UUID] = []
        actual_extension_by_rank = [0] * world_size
        if all_can_extend:
            actual_extension_by_rank = list(total_additional_by_rank)
        else:
            for r in range(world_size):
                if total_additional_by_rank[r] > per_rank_free[r]:
                    rank_seqs = [
                        (u, global_seq_state[u])
                        for u in decode_after_eviction
                        if u in global_seq_state
                        and global_seq_state[u].assigned_rank == r
                    ]
                    # Priority-aware: NORMAL (0) evicted before HIGH (1),
                    # tied by shortest-decoded-first, finally by
                    # global_idx. Matches legacy sort at 6560-6564.
                    rank_seqs.sort(
                        key=lambda item: (
                            priorities.get(item[0], 0),
                            item[1].decoded_length,
                            global_ids.get(item[0], float("inf")),
                        )
                    )
                    pages_to_free = (
                        total_additional_by_rank[r] - per_rank_free[r]
                    )
                    freed = 0
                    for uuid, st in rank_seqs:
                        if freed >= pages_to_free:
                            break
                        onhold_uuids.append(uuid)
                        freed += st.gpu_pages_allocated

            onhold_set = set(onhold_uuids)
            for uuid in seqs_needing_extension:
                if uuid in onhold_set:
                    continue
                st = global_seq_state.get(uuid)
                if st is not None:
                    r = st.assigned_rank
                    if 0 <= r < world_size:
                        actual_extension_by_rank[r] += st.additional_pages_needed

        if onhold_uuids:
            decisions.append(
                OnHold(
                    uuids=tuple(onhold_uuids),
                    reason=OnHoldReason.EXTENSION_FAILED,
                )
            )

        onhold_set = set(onhold_uuids)
        # Emit one ExtendPages per sequence (legacy executor iterates
        # over `seqs_needing_extension` minus `onhold_set`). The
        # executor reads ``st.additional_pages_needed`` through the
        # global_seq_state snapshot, so carrying the page count on the
        # decision keeps the executor stateless.
        for uuid in seqs_needing_extension:
            if uuid in onhold_set:
                continue
            st = global_seq_state.get(uuid)
            if st is None:
                continue
            decisions.append(
                ExtendPages(uuid=uuid, additional_pages=st.additional_pages_needed)
            )

        decode_uuids_final = [
            u for u in decode_after_eviction if u not in onhold_set
        ]

        # ----- Rule 5: NewLoadAsync (legacy 6582-6616) -----
        if global_candidate_info and decode_uuids_final:
            completed_set = set(completed_uuids)
            evicted_set = set(host_evicted_uuids)
            load_candidates_synced = sorted(
                (
                    u for u in global_candidate_info.keys()
                    if u not in completed_set
                    and u not in onhold_set
                    and u not in evicted_set
                ),
                key=lambda u: (
                    -global_candidate_info[u].decoded_length,
                    global_ids.get(u, float("inf")),
                ),
            )

            adjusted_per_rank_free = [
                per_rank_free[r] - actual_extension_by_rank[r]
                for r in range(world_size)
            ]
            rank_pages_used = [0] * world_size
            new_load_uuids: list[UUID] = []
            new_load_rank_pages: list[tuple[int, int]] = []
            for uuid in load_candidates_synced:
                info = global_candidate_info.get(uuid)
                if info is None:
                    continue
                req_pages = info.pages_needed
                r = info.assigned_rank
                if req_pages <= 0:
                    continue
                if not (0 <= r < world_size):
                    continue
                if (
                    rank_pages_used[r] + req_pages
                    <= adjusted_per_rank_free[r]
                ):
                    new_load_uuids.append(uuid)
                    new_load_rank_pages.append((r, req_pages))
                    rank_pages_used[r] += req_pages

            if new_load_uuids:
                decisions.append(
                    NewLoadAsync(
                        uuids=tuple(new_load_uuids),
                        rank_pages=tuple(new_load_rank_pages),
                    )
                )

        return BoundaryPlan(
            decisions=tuple(decisions),
            decode_uuids_final=tuple(decode_uuids_final),
            watermark_break=watermark_break,
        )


__all__ = ["PlannerConfig", "WorkerViewStats", "BoundaryPlanner"]
