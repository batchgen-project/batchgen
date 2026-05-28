"""Cross-rank synchronization helpers.

Slice 3 of the worker decouple initiative (issue #173). Ports the
tensor-based completion / uuid sync methods previously inlined on
``BatchGenWorker`` (``_sync_completion_status_tensor``,
``_sync_decode_uuids_tensor``) into a single sibling module.

Design departs slightly from earlier slices because NCCL collectives
must be called consistently across ranks — the same op, same order,
same shapes. The handler does NOT call ``torch.distributed`` directly;
it accepts a ``CollectiveBackend`` Protocol whose implementation is
wired by the worker. This makes the unit-test surface tractable
(production wires the real ``torch.distributed``; tests can stub the
Protocol or skip when single-rank).

Compare-mode caveat (vs Phases 1–2): the mutating sync method updates
``seq.eos_reached`` / ``seq.status`` as a side effect, but those
mutations are idempotent (set once, stays set; status transitions are
guarded against double-COMPLETED). Running legacy and native in
compare-mode produces the same final state; the NCCL collective runs
twice (acceptable migration-window cost).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Set, Tuple

import torch

if TYPE_CHECKING:
    from batchgen.sequence import SequenceBatch


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CollectiveBackend Protocol — narrow surface this slice consumes
# ---------------------------------------------------------------------------


class CollectiveBackend(Protocol):
    """Cross-rank primitive ops used by ``SyncCoordinator``.

    Production wires a thin wrapper around ``torch.distributed``. Tests
    that don't have a process group can stub this Protocol structurally.
    """

    def all_reduce_max(self, tensor: torch.Tensor) -> None: ...

    def all_reduce_min(self, tensor: torch.Tensor) -> None: ...


class TorchDistCollectiveBackend:
    """Production ``CollectiveBackend`` impl backed by ``torch.distributed``."""

    def __init__(self) -> None:
        # Lazy import so unit tests on machines without a working
        # torch.distributed installation can still import this module.
        import torch.distributed as dist
        self._dist = dist

    def all_reduce_max(self, tensor: torch.Tensor) -> None:
        self._dist.all_reduce(tensor, op=self._dist.ReduceOp.MAX)

    def all_reduce_min(self, tensor: torch.Tensor) -> None:
        self._dist.all_reduce(tensor, op=self._dist.ReduceOp.MIN)


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncContext:
    """Frozen snapshot passed to each ``SyncCoordinator`` call.

    Worker constructs one from its canonical fields (``self.rank``,
    ``self._uuid_to_local_map``, ``self.global_batch``,
    ``self.torch_device``) per call site.
    """

    rank: int
    uuid_to_local: "object"  # Mapping[str, int]; "object" avoids Mapping import bloat
    global_batch: "SequenceBatch"
    torch_device: torch.device


# ---------------------------------------------------------------------------
# SyncCoordinator
# ---------------------------------------------------------------------------


class SyncCoordinator:
    """Tensor-based cross-rank sync handlers.

    The instance holds a ``CollectiveBackend`` so the worker can inject
    a fake during tests. Each method takes a frozen ``SyncContext``
    snapshot built fresh per call.
    """

    def __init__(self, *, backend: CollectiveBackend) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # sync_completion_status_tensor
    # ------------------------------------------------------------------
    def sync_completion_status_tensor(
        self,
        ctx: SyncContext,
        decode_uuids: List[str],
    ) -> Tuple[Set[str], List[str]]:
        """Synchronize completion status across all ranks using tensor ops.

        Replaces the expensive ``all_gather_object`` with a tensor-based
        ``all_reduce(MAX)``. On any rank that marks a sequence complete,
        ALL ranks observe completion and mark the sequence accordingly.

        Side effects on completed sequences: ``seq.eos_reached = True``;
        ``seq.status`` transitions to ``COMPLETED`` when not already there.
        Both mutations are idempotent.

        Returns:
            ``(global_completed_uuids, active_decode_uuids)`` — both
            sorted by ``global_idx``.
        """
        from batchgen.sequence import SequenceStatus  # local import to avoid module-load cost

        if not decode_uuids:
            return set(), []

        # Build global_idx ↔ uuid mapping for decode candidates
        idx_to_uuid = {}
        uuid_to_idx = {}
        for uuid in decode_uuids:
            seq = ctx.global_batch.get_sequence(uuid)
            if seq is not None:
                idx_to_uuid[seq.global_idx] = uuid
                uuid_to_idx[uuid] = seq.global_idx

        if not idx_to_uuid:
            return set(), []

        max_idx = max(idx_to_uuid.keys())

        # Completion tensor: 1 = completed, 0 = not. Each rank marks its
        # LOCAL sequences' completion status.
        completion_tensor = torch.zeros(max_idx + 1, dtype=torch.int32, device=ctx.torch_device)
        for uuid in decode_uuids:
            if uuid in ctx.uuid_to_local:
                seq = ctx.global_batch.get_sequence(uuid)
                if seq is not None and uuid in uuid_to_idx:
                    is_completed = (seq.status == SequenceStatus.COMPLETED or seq.eos_reached)
                    if is_completed:
                        completion_tensor[uuid_to_idx[uuid]] = 1

        # all_reduce MAX: if ANY rank marks a sequence complete, result is 1
        self._backend.all_reduce_max(completion_tensor)

        global_completed: Set[str] = set()
        active_uuids: List[str] = []
        for global_idx in sorted(idx_to_uuid.keys()):
            uuid = idx_to_uuid[global_idx]
            if completion_tensor[global_idx].item() == 1:
                global_completed.add(uuid)
                seq = ctx.global_batch.get_sequence(uuid)
                if seq is not None:
                    seq.eos_reached = True
                    if seq.status != SequenceStatus.COMPLETED:
                        try:
                            ctx.global_batch.update_status(uuid, SequenceStatus.COMPLETED)
                        except ValueError as e:
                            logger.debug(
                                f"Rank {ctx.rank}: Could not update {uuid[:8]} to COMPLETED: {e}"
                            )
            else:
                active_uuids.append(uuid)

        return global_completed, active_uuids

    # ------------------------------------------------------------------
    # sync_decode_uuids_tensor
    # ------------------------------------------------------------------
    def sync_decode_uuids_tensor(
        self,
        ctx: SyncContext,
        decode_uuids: List[str],
    ) -> List[str]:
        """Synchronize ``decode_uuids`` across all ranks using tensor ops.

        Uses ``global_idx`` as the common identifier and ``all_reduce(MIN)``
        to find the intersection. Only sequences present on ALL ranks
        survive. No state mutation.

        Returns:
            Sorted list of UUIDs that ALL ranks agree on.
        """
        if not decode_uuids:
            return []

        # Build global_idx ↔ uuid mapping for the FULL batch (not just decode_uuids)
        # because the presence tensor is indexed by global_idx and we want a
        # well-defined dimension regardless of which slice of the batch is in
        # decode_uuids on this rank.
        idx_to_uuid = {}
        uuid_to_idx = {}
        for seq in ctx.global_batch:
            idx_to_uuid[seq.global_idx] = seq.uuid
            uuid_to_idx[seq.uuid] = seq.global_idx

        max_idx = max(idx_to_uuid.keys()) if idx_to_uuid else 0

        # Presence tensor: 1 = uuid is in this rank's decode_uuids, 0 = not
        presence_tensor = torch.zeros(max_idx + 1, dtype=torch.int32, device=ctx.torch_device)
        for uuid in decode_uuids:
            if uuid in uuid_to_idx:
                presence_tensor[uuid_to_idx[uuid]] = 1

        # all_reduce MIN: only sequences present on ALL ranks survive
        self._backend.all_reduce_min(presence_tensor)

        synced_uuids: List[str] = []
        for global_idx in sorted(idx_to_uuid.keys()):
            if presence_tensor[global_idx].item() == 1:
                synced_uuids.append(idx_to_uuid[global_idx])

        return synced_uuids
