"""BoundaryExecutor — apply a BoundaryPlan in canonical order.

The planner produces a list of :class:`PageBoundaryDecision`; the
executor applies them on every rank in exactly this order (POIS Q2):

    ReleasePages > Evict > OnHold > ExtendPages > AsyncLoadHostToGpu

Why grouped-by-type application rather than list order: the planner
emits decisions in any order, but page-budget math requires freeing
before spending. Release frees GPU pages held by completed sequences.
Evict frees both host and GPU pages when a seq can't expand its host
reservation. OnHold goes through ``HostKVRebalancer.put_on_hold`` (the
load-bearing Decision #2 ordering). Only after all three have run does
the executor spend remaining GPU pages on Extend and AsyncLoad.

Delegations:
  - Release / Extend → ``KVCacheManager`` directly.
  - OnHold → ``HostKVRebalancer.put_on_hold`` (enforces the 5-step
    flush → wait → release → transition → sync sequence).
  - Evict → M4 minimal: status transition + GPU release. Stashing
    ``evicted_token_ids`` + host-chunk release is M5 work alongside
    DecodeScheduler where the host reservation state lives.
  - AsyncLoadHostToGpu → M4 raises ``NotImplementedError``; lands
    alongside DecodeScheduler in M5 (host→GPU async handle machinery).
"""

from __future__ import annotations

from batchgen.sequence import SequenceStatus
from batchgen.worker.boundary.decisions import (
    AsyncLoadHostToGpu,
    BoundaryPlan,
    Evict,
    ExtendPages,
    OnHold,
    ReleasePages,
)
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.state import WorkerState


class BoundaryExecutor:
    def __init__(
        self,
        state: WorkerState,
        kv: KVCacheManager,
        rebalancer: HostKVRebalancer,
    ) -> None:
        self._state = state
        self._kv = kv
        self._rebalancer = rebalancer

    def apply(self, plan: BoundaryPlan) -> None:
        """Apply every decision in the canonical order.

        Iterating plan.decisions_of(kind) is deliberate: the planner
        can emit the decisions in any order but the executor always
        frees pages before spending them. An empty decision list for
        any type is a no-op.
        """
        for decision in plan.decisions_of(ReleasePages):
            self._apply_release(decision)  # type: ignore[arg-type]
        for decision in plan.decisions_of(Evict):
            self._apply_evict(decision)  # type: ignore[arg-type]
        for decision in plan.decisions_of(OnHold):
            self._apply_onhold(decision)  # type: ignore[arg-type]
        for decision in plan.decisions_of(ExtendPages):
            self._apply_extend(decision)  # type: ignore[arg-type]
        for decision in plan.decisions_of(AsyncLoadHostToGpu):
            self._apply_async_load(decision)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Per-decision handlers
    # ------------------------------------------------------------------

    def _apply_release(self, decision: ReleasePages) -> None:
        """Free GPU pages for already-COMPLETED sequences.

        CompletionHandler has already transitioned these sequences to
        COMPLETED before the boundary runs, so the executor does not
        touch status here — it only returns GPU pages.
        """
        self._kv.release_pages(list(decision.uuids))

    def _apply_evict(self, decision: Evict) -> None:
        """M4 minimal: transition IN_DECODE/ON_HOLD → EVICTED + release GPU.

        Host-chunk release and evicted_token_ids stashing land in M5
        alongside DecodeScheduler — the chunk reservation state is
        DecodeScheduler's domain and wiring it through BoundaryExecutor
        before that handler exists would create an orphan code path.
        """
        to_release: list[str] = []
        for uuid in decision.uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            if seq.status in (SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD):
                self._state.global_batch.update_status(uuid, SequenceStatus.EVICTED)
                to_release.append(uuid)
        if to_release:
            self._kv.release_pages(to_release)

    def _apply_onhold(self, decision: OnHold) -> None:
        """Route through HostKVRebalancer.put_on_hold to preserve the
        plan Decision #2 5-step ordering (flush → wait → release →
        transition → sync). Never call put_on_hold elsewhere."""
        self._rebalancer.put_on_hold(list(decision.uuids))

    def _apply_extend(self, decision: ExtendPages) -> None:
        """Grow `uuid`'s GPU page allocation by `additional_pages`."""
        self._kv.extend_allocation(decision.uuid, decision.additional_pages)

    def _apply_async_load(self, decision: AsyncLoadHostToGpu) -> None:
        """**M5 stub**: async host→GPU transfer requires the async handle
        machinery DecodeScheduler owns. Raising here is intentional so
        the orchestrator cannot silently wire a planner that emits
        AsyncLoadHostToGpu before M5 lands."""
        raise NotImplementedError(
            "BoundaryExecutor._apply_async_load: AsyncLoadHostToGpu is "
            "deferred to M5 alongside DecodeScheduler.try_load_new"
        )


__all__ = ["BoundaryExecutor"]
