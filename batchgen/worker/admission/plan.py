"""Status transition plans: the vocabulary of sequence status changes.

A sequence's status changes only through ``SequenceBatch.apply(plan)``. Each
decision that moves sequences between statuses builds a ``TransitionPlan``
tagged with the ``StatusEdge`` that produced it, and the batch applies it in
order. ``apply`` returns the transitions the sequences rejected instead of
raising, so each caller keeps its own policy for an invalid transition.

``StatusEdge``, ``Transition``, ``TransitionPlan`` and ``RejectedTransition`` are defined in
``batchgen.sequence`` next to ``SequenceBatch`` so that module never imports
this package; they are re-exported here as the public spelling.
"""

from __future__ import annotations

import logging

from batchgen.sequence import (
    RejectedTransition,
    StatusEdge,
    Transition,
    TransitionPlan,
)

__all__ = [
    "RejectedTransition",
    "StatusEdge",
    "Transition",
    "TransitionPlan",
    "apply_or_raise",
    "update_batch_status",
]


def update_batch_status(pool, rank, uuids, new_status, edge):
    """Update status for sequences, skipping if already in target status."""
    if isinstance(uuids, str):
        uuids = [uuids]
    transitions, missing = pool.plan_transitions(uuids, new_status)
    for uuid in missing:
        logging.warning(f"Rank {rank}: Sequence {uuid} not found in global_batch")
    rejected = pool.apply(TransitionPlan(edge, transitions))
    for rejection in rejected:
        logging.warning(
            f"Rank {rank}: Invalid status transition for "
            f"{rejection.transition.uuid}: {rejection.reason}"
        )


def apply_or_raise(batch, plan: TransitionPlan) -> None:
    """Apply ``plan`` to ``batch``; a rejected transition is fatal.

    Every transition is attempted; the first rejection is then raised as the
    ``ValueError`` the sequence produced for it.
    """
    rejected = batch.apply(plan)
    if rejected:
        raise ValueError(rejected[0].reason)
