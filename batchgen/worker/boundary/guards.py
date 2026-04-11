"""Boundary-time invariant checks.

Runs as `check_pre(state, plan)` before the executor applies a plan and
`check_post(state)` after application. Every violation raises
:class:`GuardViolation` — failing loudly is the point (plan Decision #6).

The guards deliberately do NOT repair state. A handler that silently
fixes a broken invariant is exactly the failure mode the old
scheduler-split branch kept creating.
"""

from __future__ import annotations

from batchgen.worker.boundary.decisions import (
    AsyncLoadHostToGpu,
    BoundaryPlan,
    Evict,
    ExtendPages,
    OnHold,
    ReleasePages,
)
from batchgen.worker.state import WorkerState


class GuardViolation(RuntimeError):
    """Raised when a boundary-time invariant is broken.

    Carries a `check` name (``"pre"`` / ``"post"``), an `invariant` name
    identifying which rule tripped, and a `detail` dict with the offending
    values so traces show what happened without re-running.
    """

    def __init__(
        self,
        *,
        check: str,
        invariant: str,
        detail: dict,
    ) -> None:
        self.check = check
        self.invariant = invariant
        self.detail = detail
        super().__init__(
            f"Boundary guard violation ({check}/{invariant}): {detail}"
        )


class BoundaryGuards:
    """Pure invariant checks — no state writes, no collectives."""

    def __init__(self, state: WorkerState) -> None:
        self._state = state

    # ------------------------------------------------------------------
    # Pre-check: plan sanity before execution
    # ------------------------------------------------------------------

    def check_pre(self, plan: BoundaryPlan) -> None:
        """Verify the plan references only sequences currently live in
        `state.global_batch`. Raises on the first violation.

        Does NOT verify page budgets — the executor's allocate/extend
        calls will raise RuntimeError if the GPU backend is out of pages,
        and the guard doesn't duplicate that accounting.
        """
        for decision in plan.decisions:
            uuids = self._uuids_of(decision)
            for uuid in uuids:
                if self._state.global_batch.get_sequence(uuid) is None:
                    raise GuardViolation(
                        check="pre",
                        invariant="plan_references_live_sequences",
                        detail={
                            "decision": type(decision).__name__,
                            "uuid": uuid,
                        },
                    )

    @staticmethod
    def _uuids_of(decision) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
        if isinstance(decision, (ReleasePages, Evict, OnHold)):
            return tuple(decision.uuids)
        if isinstance(decision, ExtendPages):
            return (decision.uuid,)
        if isinstance(decision, AsyncLoadHostToGpu):
            return (decision.uuid,)
        raise GuardViolation(
            check="pre",
            invariant="unknown_decision_type",
            detail={"type": type(decision).__name__},
        )

    # ------------------------------------------------------------------
    # Post-check: invariants after execution
    # ------------------------------------------------------------------

    def check_post(self) -> None:
        """Run all post-execution invariant checks in a fixed order.

        Stops at the first violation so traces are loud. The invariants
        are additive — new ones are added here and tests lock them down.
        """
        self._check_ctx_invariant()
        self._check_index_map_consistency()

    def _check_ctx_invariant(self) -> None:
        """Every live sequence must satisfy
        ``current_context_length == original_prompt_length + decoded_length``.

        The same invariant `SyncCoordinator.sync_metadata` enforces on
        the wire — here we re-check after boundary execution so a broken
        executor surfaces immediately.
        """
        for uuid in list(self._state.global_batch.sequences.keys()):
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            expected = seq.original_prompt_length + seq.decoded_length
            if seq.current_context_length != expected:
                raise GuardViolation(
                    check="post",
                    invariant="ctx_invariant",
                    detail={
                        "uuid": uuid,
                        "had": seq.current_context_length,
                        "expected": expected,
                    },
                )

    def _check_index_map_consistency(self) -> None:
        """IndexManager's dual maps plus the free set must be consistent.

          - For every ``(local_idx, uuid)`` in ``local_to_uuid_map``,
            ``uuid_to_local_map[uuid] == local_idx``.
          - The reverse must also hold.
          - ``free_local_indices`` and ``local_to_uuid_map`` must be
            disjoint sets of ints (a slot is either free or in use).
        """
        l2u = self._state.local_to_uuid_map
        u2l = self._state.uuid_to_local_map

        for local_idx, uuid in l2u.items():
            if u2l.get(uuid) != local_idx:
                raise GuardViolation(
                    check="post",
                    invariant="index_map_consistency",
                    detail={
                        "local_idx": local_idx,
                        "uuid": uuid,
                        "uuid_to_local_map_says": u2l.get(uuid),
                    },
                )
        if len(l2u) != len(u2l):
            raise GuardViolation(
                check="post",
                invariant="index_map_consistency",
                detail={
                    "local_to_uuid_size": len(l2u),
                    "uuid_to_local_size": len(u2l),
                },
            )

        free = self._state.free_local_indices
        live = set(l2u.keys())
        overlap = free & live
        if overlap:
            raise GuardViolation(
                check="post",
                invariant="free_slot_exclusive",
                detail={"overlapping_local_indices": sorted(overlap)},
            )


__all__ = ["GuardViolation", "BoundaryGuards"]
