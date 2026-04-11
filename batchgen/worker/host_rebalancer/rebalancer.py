"""HostKVRebalancer — the class itself.

External contract unchanged across the M9 sub-package split.
:class:`EvictionStrategy` / :class:`ShortestDecodedFirstStrategy` moved
to :mod:`.eviction`; :class:`MigrationOp` moved to :mod:`.migration`.
Everything is re-exported from ``batchgen.worker.host_rebalancer`` so
existing callers see no change.
"""

from __future__ import annotations

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.host_rebalancer.eviction import (
    EvictionStrategy,
    ShortestDecodedFirstStrategy,
)
from batchgen.worker.host_rebalancer.migration import MigrationOp
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import UUID
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator


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
            eviction_strategy
            if eviction_strategy is not None
            else ShortestDecodedFirstStrategy()
        )

    # ------------------------------------------------------------------
    # Victim selection
    # ------------------------------------------------------------------

    def select_for_onhold(self, count: int) -> list[UUID]:
        """Return `count` IN_DECODE UUIDs to put on hold, via the strategy."""
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
          1. KVCacheManager.flush_deferred
          2. KVCacheManager.wait_pending
          3. KVCacheManager.release_pages(uuids)
          4. SequenceBatch.update_status(uuid, ON_HOLD)
          5. SyncCoordinator.sync_metadata(uuids)

        Empty input is a no-op. UUIDs missing from ``global_batch``
        are skipped silently; UUIDs not in IN_DECODE are skipped as
        well.
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
    # Migration (still a stub — awaits real-hardware root-cause)
    # ------------------------------------------------------------------

    def plan_migration(self) -> list[MigrationOp]:
        """**M9 stub**: returns empty list until the real-hardware
        session roots out the CUDA migration bug from the old
        scheduler-split branch."""
        return []

    def execute_migrations(self, ops: list[MigrationOp]) -> int:
        """**M9 stub**: accepts only empty list; non-empty raises
        NotImplementedError so the orchestrator cannot mis-wire
        before the root-cause lands."""
        if not ops:
            return 0
        raise NotImplementedError(
            "HostKVRebalancer.execute_migrations: non-empty ops are deferred "
            "to M9 hardware root-cause"
        )

    def rebalance(self) -> int:
        """Run plan → execute. **Stub**: no-op returning 0."""
        return self.execute_migrations(self.plan_migration())


__all__ = ["HostKVRebalancer"]
