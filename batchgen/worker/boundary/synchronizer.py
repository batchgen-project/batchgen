"""BoundarySynchronizer — metadata sync in + plan broadcast out.

Two primitives the boundary handler uses to keep every rank in lockstep:

  - ``sync_metadata_in(uuids)``: delegates to
    :meth:`SyncCoordinator.sync_metadata` so every rank has the same
    authoritative view of the batch before the planner runs. The
    CTX fast-fail on sender or receiver propagates directly.
  - ``broadcast_plan(plan)``: rank 0 broadcasts the computed
    :class:`BoundaryPlan`; other ranks pass ``None`` and receive the
    plan through the shared collective backend. Returns the plan on
    every rank.

Decoupled from the planner and executor so the BoundaryHandler can wire
them in whatever order the caller prefers without touching collective
code.
"""

from __future__ import annotations

from batchgen.worker.boundary.decisions import BoundaryPlan
from batchgen.worker.protocols import UUID, CollectiveBackend
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator


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

    def sync_metadata_in(self, uuids: list[UUID]) -> None:
        """Refresh cross-rank metadata for `uuids` before planning."""
        self._sync.sync_metadata(uuids)

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


__all__ = ["BoundarySynchronizer"]
