"""Sealed union of `PageBoundaryDecision` types + planner snapshot types.

The planner emits a list of these decisions; the executor consumes them
in the canonical order defined in `BoundaryExecutor.apply`:

    ReleasePages > Evict > OnHold > ExtendPages > AsyncLoadHostToGpu

Every new boundary operation (e.g. an eager-prefetch for the next
sequence) becomes a new frozen dataclass here plus a matching executor
branch and a guard update — see plan "BoundaryHandler, modularized".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from batchgen.worker.protocols import UUID, PageId


# ---------------------------------------------------------------------------
# Reason enums (plan "BoundaryHandler, modularized")
# ---------------------------------------------------------------------------


class EvictReason(Enum):
    """Why a set of sequences is being evicted to host KV / dropped."""

    HOST_KV_WATERMARK = auto()          # host KV free % dropped under watermark
    PREEMPT_FOR_QUEUED_PREFILL = auto() # make room for queued QUEUEING prefill
    CONTEXT_OVERFLOW = auto()           # seq exceeded model_context_length at boundary


class OnHoldReason(Enum):
    """Why a set of sequences is being put on hold (kept warm on host)."""

    EXTENSION_FAILED = auto()   # GPU page extension couldn't fit during decode
    WATERMARK_TRIGGER = auto()  # host KV watermark forced preemption of decode


# ---------------------------------------------------------------------------
# Planner snapshot types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeqMetadata:
    """Per-sequence metadata consumed by `BoundaryPlanner.plan`.

    Keeps the planner a pure function — it reads only the fields listed
    here, never reaching into `state.global_batch`. Adding a field is
    additive; callers fill the snapshot at the boundary-sync step.
    """

    uuid: UUID
    global_idx: int
    status: int                  # SequenceStatus int value
    assigned_rank: int
    prompt_length: int
    max_decode_length: int
    decoded_length: int
    current_context_length: int
    gpu_pages_allocated: int
    host_pages_allocated: int
    had_initial_gpu_reservation: bool
    eos_reached: bool
    rep_detected: bool

    @property
    def is_completed_by_length(self) -> bool:
        return self.decoded_length >= self.max_decode_length


# ---------------------------------------------------------------------------
# Decision types (sealed union)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleasePages:
    """Free GPU pages for sequences that already finished (post-completion).

    Does NOT change sequence status — CompletionHandler has already
    transitioned the sequence to COMPLETED before the boundary runs. The
    executor calls `KVCacheManager.release_pages` and clears the
    gpu_pages_allocated field.
    """

    uuids: tuple[UUID, ...]


@dataclass(frozen=True)
class Evict:
    """Transition sequences to EVICTED; stash evicted_token_ids; free GPU."""

    uuids: tuple[UUID, ...]
    reason: EvictReason


@dataclass(frozen=True)
class OnHold:
    """Transition IN_DECODE → ON_HOLD via `HostKVRebalancer.put_on_hold`.

    Executor MUST route this through `HostKVRebalancer.put_on_hold` so
    the load-bearing flush→wait→release→transition→sync ordering (plan
    Decision #2) is preserved.
    """

    uuids: tuple[UUID, ...]
    reason: OnHoldReason


@dataclass(frozen=True)
class ExtendPages:
    """Grow `uuid`'s GPU page allocation by `additional_pages`."""

    uuid: UUID
    additional_pages: int


@dataclass(frozen=True)
class AsyncLoadHostToGpu:
    """Kick off an async host→GPU KV transfer for `uuid`.

    Each `host_pages` element is an opaque `PageId` from the host KV
    backend. The executor returns the async handle to the caller so
    follow-up waits can complete the transfer.
    """

    uuid: UUID
    host_pages: tuple[PageId, ...]


PageBoundaryDecision = ReleasePages | Evict | OnHold | ExtendPages | AsyncLoadHostToGpu


# ---------------------------------------------------------------------------
# BoundaryPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryPlan:
    """The planner's output: ordered list of decisions + input snapshot.

    `decisions` is an immutable tuple so the planner cannot mutate the
    plan after returning it. `metadata_snapshot` is the same per-sequence
    view the planner consumed — guards use it to verify the executor
    ended in a state consistent with the plan's premises.
    """

    decisions: tuple[PageBoundaryDecision, ...] = field(default_factory=tuple)
    metadata_snapshot: dict[UUID, SeqMetadata] = field(default_factory=dict)

    def decisions_of(self, kind: type) -> tuple[PageBoundaryDecision, ...]:
        """Return just the decisions of a given dataclass type."""
        return tuple(d for d in self.decisions if isinstance(d, kind))


__all__ = [
    "EvictReason",
    "OnHoldReason",
    "SeqMetadata",
    "ReleasePages",
    "Evict",
    "OnHold",
    "ExtendPages",
    "AsyncLoadHostToGpu",
    "PageBoundaryDecision",
    "BoundaryPlan",
]
