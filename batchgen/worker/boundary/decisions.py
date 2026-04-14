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

    Retained for the decode-layer prefetch path; the boundary executor
    uses :class:`NewLoadAsync` instead (it batches every candidate into
    one async launch and carries per-rank page budgets).
    """

    uuid: UUID
    host_pages: tuple[PageId, ...]


@dataclass(frozen=True)
class HostGrow:
    """Rank-0 decision: grow host KV allocation for N sequences.

    Legacy ``_boundary_execute_decisions`` (batchgen_worker.py:6957-6987)
    computes per-sequence ``host_growth_pages`` and a single
    ``growth_feasible`` flag derived from a 5% safety-margin check on
    host free pages. If ``feasible=False`` the executor SKIPS growth
    entirely; the planner packs both signals into this decision so the
    executor stays ignorant of the watermark math.

    ``uuids[i]`` grows by ``pages[i]`` host pages. Tuples are parallel.
    """

    uuids: tuple[UUID, ...]
    pages: tuple[int, ...]
    feasible: bool


@dataclass(frozen=True)
class HostEvict:
    """Rank-0 decision: evict N sequences from host KV to EVICTED.

    Legacy selects these via
    ``select_host_kv_eviction(..., SHORTEST_FIRST)`` in
    ``_compute_boundary_decisions`` when host-free drops below the
    watermark. The executor runs the reentry math
    (``prompt_length = prompt_length + new_decoded_count``) on every
    rank; the rank that owns each uuid additionally releases its GPU +
    host pages via the worker view.
    """

    uuids: tuple[UUID, ...]


@dataclass(frozen=True)
class NewLoadAsync:
    """Rank-0 decision: async-load N host-resident sequences onto GPU.

    ``rank_pages`` is the per-rank page budget the planner precomputed,
    serialized as a tuple of ``(rank, pages_allocated)`` so the
    executor does not need a second collective to match uuids to
    allocations. Matches legacy ``new_load_uuids`` +
    ``actual_extension_by_rank`` bookkeeping at
    batchgen_worker.py:7075-7131.
    """

    uuids: tuple[UUID, ...]
    rank_pages: tuple[tuple[int, int], ...] = ()


PageBoundaryDecision = (
    ReleasePages
    | Evict
    | OnHold
    | ExtendPages
    | HostGrow
    | HostEvict
    | NewLoadAsync
    | AsyncLoadHostToGpu
)


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

    ``watermark_break`` is set by the planner when the host-KV watermark
    has fired; the decode loop reads it to decide whether to break
    back to prefill after this boundary applies. ``decode_uuids_final``
    is the ordered post-boundary IN_DECODE cohort — saves callers from
    re-deriving it by subtracting decisions from the input uuids.
    """

    decisions: tuple[PageBoundaryDecision, ...] = field(default_factory=tuple)
    metadata_snapshot: dict[UUID, SeqMetadata] = field(default_factory=dict)
    decode_uuids_final: tuple[UUID, ...] = field(default_factory=tuple)
    watermark_break: bool = False

    def decisions_of(self, kind: type) -> tuple[PageBoundaryDecision, ...]:
        """Return just the decisions of a given dataclass type."""
        return tuple(d for d in self.decisions if isinstance(d, kind))


# ---------------------------------------------------------------------------
# BoundaryResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryResult:
    """Return value of ``BoundaryHandler.run``.

    Matches the legacy ``_page_boundary_fast`` output 1:1 so the Stage 2
    decode-loop port can swap the call in place without reshaping the
    caller. Fields:

      * ``plan``: the ``BoundaryPlan`` the handler applied.
      * ``decode_uuids``: post-boundary IN_DECODE cohort.
      * ``batch``: post-boundary local-index list matching ``decode_uuids``.
      * ``new_async_task``: async handle returned by the executor's
        new-load sub-step (``None`` when no new loads launched).
      * ``new_load_uuids`` / ``new_load_local`` / ``new_load_global``:
        the uuids / local indices / global ids that were handed to the
        async load; the decode loop threads these back into the next
        boundary cycle via ``BoundaryHandler.run``.
      * ``watermark_triggered``: final host-KV watermark bool (from
        ``adapter.check_host_kv_watermark_trigger()`` in
        ``boundary/finalize.py``). The decode loop reads this to decide
        whether to break back to prefill.
    """

    plan: BoundaryPlan
    decode_uuids: tuple[UUID, ...] = ()
    batch: tuple[int, ...] = ()
    new_async_task: object | None = None
    new_load_uuids: tuple[UUID, ...] = ()
    new_load_local: tuple[int, ...] = ()
    new_load_global: tuple[int, ...] = ()
    watermark_triggered: bool = False


__all__ = [
    "EvictReason",
    "OnHoldReason",
    "SeqMetadata",
    "ReleasePages",
    "Evict",
    "OnHold",
    "ExtendPages",
    "AsyncLoadHostToGpu",
    "HostGrow",
    "HostEvict",
    "NewLoadAsync",
    "PageBoundaryDecision",
    "BoundaryPlan",
    "BoundaryResult",
]
