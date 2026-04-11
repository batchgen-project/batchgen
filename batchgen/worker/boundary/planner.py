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
    OnHold,
    OnHoldReason,
    PageBoundaryDecision,
    ReleasePages,
    SeqMetadata,
)
from batchgen.worker.protocols import UUID


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


__all__ = ["PlannerConfig", "BoundaryPlanner"]
