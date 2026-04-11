"""HostKVRebalancer — eviction victim selection, put-on-hold, migration.

Owns three concerns:

  1. **Victim selection**: ``select_for_onhold(count)`` picks IN_DECODE
     sequences via a pluggable :class:`EvictionStrategy`. The default
     :class:`ShortestDecodedFirstStrategy` matches main's behavior (preserve
     sequences that have made the most progress). Swapping strategies does
     not touch the handler — plan Decision #1 "pluggable, default
     shortest-first".

  2. **put_on_hold** orchestration: transitioning sequences from IN_DECODE
     to ON_HOLD is only safe AFTER the async KV-append callbacks have
     drained so the host has the latest KV for every in-flight sequence.
     The method therefore enforces the plan Decision #2 ordering:

        flush_deferred → wait_pending → release_gpu_pages →
        update_status(ON_HOLD) → sync_metadata(all_uuids)

     Any handler that wants to put sequences on hold MUST go through this
     method — the ordering is a load-bearing invariant and there is no
     "fast path" that skips a step.

  3. **Migration** (M3 stub → M9 full impl): :class:`MigrationOp`,
     ``plan_migration``, ``execute_migrations``, and ``rebalance`` are
     structurally present so the orchestrator can wire them, but the
     migration planning body is deferred to M9 along with the CUDA
     migration bug root-cause. The stubs are documented with a TODO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import UUID
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator


# ---------------------------------------------------------------------------
# Eviction strategy
# ---------------------------------------------------------------------------


class EvictionStrategy(Protocol):
    """Picks the subset of sequences to put on hold or evict.

    The `sequences` input is the full candidate set in some arbitrary
    deterministic order; the strategy returns `count` UUIDs (or fewer
    if the candidate pool is smaller).
    """

    def select(self, sequences: list[SequenceEntry], count: int) -> list[UUID]: ...


class ShortestDecodedFirstStrategy:
    """Default eviction strategy — evict the sequences that have decoded the
    fewest tokens, preserving the progress of longer-running sequences.

    Matches main's behavior at ``batchgen_worker.py`` host-KV watermark
    path. Ties (equal decoded_length) are broken by `uuid` so the order
    is identical across ranks without a collective.
    """

    def select(self, sequences: list[SequenceEntry], count: int) -> list[UUID]:
        if count <= 0:
            return []
        ordered = sorted(sequences, key=lambda s: (s.decoded_length, s.uuid))
        return [s.uuid for s in ordered[:count]]


# ---------------------------------------------------------------------------
# Migration op + stub
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationOp:
    """One host-KV page migration from one rank to another.

    The ``uuid`` is the owning sequence; the ``page_count`` is the number
    of host pages to move. Execution semantics live in
    :meth:`HostKVRebalancer.execute_migrations` (M9 stub until the CUDA
    migration bug from the old scheduler-split branch is root-caused).
    """

    from_rank: int
    to_rank: int
    uuid: UUID
    page_count: int


# ---------------------------------------------------------------------------
# HostKVRebalancer
# ---------------------------------------------------------------------------


class HostKVRebalancer:
    def __init__(
        self,
        state: WorkerState,
        kv: KVCacheManager,
        sync: SyncCoordinator,
        *,
        eviction_strategy: EvictionStrategy | None = None,
    ) -> None:
        self._state = state
        self._kv = kv
        self._sync = sync
        self._strategy: EvictionStrategy = (
            eviction_strategy if eviction_strategy is not None else ShortestDecodedFirstStrategy()
        )

    # ------------------------------------------------------------------
    # Victim selection
    # ------------------------------------------------------------------

    def select_for_onhold(self, count: int) -> list[UUID]:
        """Return `count` IN_DECODE UUIDs to put on hold, via the strategy.

        Reads the IN_DECODE candidate set from ``state.global_batch``; only
        sequences still holding the IN_DECODE status are considered.
        Sequences missing from the batch are skipped. Returns fewer than
        ``count`` UUIDs if the candidate pool is smaller.
        """
        if count <= 0:
            return []
        candidate_uuids = self._state.global_batch.get_sequences_by_status(
            SequenceStatus.IN_DECODE
        )
        candidates: list[SequenceEntry] = []
        for uuid in candidate_uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is not None:
                candidates.append(seq)
        return self._strategy.select(candidates, count)

    # ------------------------------------------------------------------
    # put_on_hold — the load-bearing ordering invariant
    # ------------------------------------------------------------------

    def put_on_hold(self, uuids: list[UUID]) -> None:
        """Transition `uuids` IN_DECODE → ON_HOLD with the correct ordering.

        Plan Decision #2 ordering (every step is required):

          1. ``KVCacheManager.flush_deferred``: drain queued kv-append ops
             so the host has the latest KV for every IN_DECODE sequence.
          2. ``KVCacheManager.wait_pending``: wait for async transfers to
             complete; after this the host KV is fully consistent.
          3. ``KVCacheManager.release_pages(uuids)``: return GPU pages to
             the free pool. Must happen BEFORE the status transition so a
             concurrent reader racing on the status flag never sees a
             half-released allocation.
          4. ``SequenceBatch.update_status(uuid, ON_HOLD)``: goes through
             ``status_transition()`` — plan invariant #9 (atomic
             transitions, metadata BEFORE status change).
          5. ``SyncCoordinator.sync_metadata(uuids)``: every rank learns
             the new on-hold list and absorbs any peer metadata updates.

        Empty input is a no-op (no collectives issued). UUIDs missing from
        ``global_batch`` are skipped silently; UUIDs in a status other than
        ``IN_DECODE`` are also skipped (they may already be ON_HOLD if a
        previous call was interrupted mid-way).
        """
        if not uuids:
            return

        self._kv.flush_deferred()
        self._kv.wait_pending()
        self._kv.release_pages(uuids)

        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            if seq.status != SequenceStatus.IN_DECODE:
                continue
            self._state.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

        self._sync.sync_metadata(uuids)

    # ------------------------------------------------------------------
    # Migration (M9 stubs)
    # ------------------------------------------------------------------

    def plan_migration(self) -> list[MigrationOp]:
        """Compute migration ops to rebalance host KV across ranks.

        **M9 stub**: returns empty list. The full impl reads per-rank
        host-free counts via ``sync.gather_rank_token_counts`` and
        computes a set of :class:`MigrationOp`s to equalize capacity.
        Deferred to M9 along with the CUDA migration bug root-cause
        (plan Known Risks item #3).
        """
        return []

    def execute_migrations(self, ops: list[MigrationOp]) -> int:
        """Apply each migration op via the host KV backend.

        **M9 stub**: accepts only an empty list; raises
        ``NotImplementedError`` if non-empty ops are passed so the
        orchestrator cannot silently mis-wire before M9 lands.

        Returns the number of ops applied (0 for the empty case).
        """
        if not ops:
            return 0
        raise NotImplementedError(
            "HostKVRebalancer.execute_migrations: non-empty ops are deferred "
            "to M9 (host-rebalancer sub-package split + CUDA migration fix)"
        )

    def rebalance(self) -> int:
        """Run plan → execute. **M9 stub**: no-op returning 0."""
        return self.execute_migrations(self.plan_migration())


__all__ = [
    "EvictionStrategy",
    "ShortestDecodedFirstStrategy",
    "MigrationOp",
    "HostKVRebalancer",
]
