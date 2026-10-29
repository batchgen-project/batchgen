"""Footprint admission (invariant I-A) for a session-hold server.

With ``--enable-session-kv-hold`` a finished turn keeps its Host KV until its
session's next turn, close or abort (``batchgen.worker.session_hold``). The
per-prefill Host reservation (``compute_prefill_host_reservation``) only
charges a wave its *initial* pages and knows nothing about held pages, so a
hold server can admit work whose later Host growth no longer fits: the
decode boundary then evicts running sequences (recompute), and a resumed
turn that is charged its full reservation can wait forever behind its own
held pages.

This module is the pure decision that replaces that optimism on a hold
server. Every admission candidate is charged its *projected* KV footprint:

* a new turn (or a batch request, or an evicted sequence) needs its whole
  KV budget, ``ceil((prompt + max_decode) / page)`` pages;
* a resumed turn of a held session needs that budget minus the pages its
  session already holds: the LCP pages are reused in place and the held
  pages past the LCP are freed when it commits, so the net change is
  ``budget_pages - held_pages`` ("suffix + output budget").

and is admitted only if the footprint fits the free capacity of its pools:

    Host  (per node):  total - running - held - margin
    GPU   (per rank):  total - running

"running" is the projected budget of every sequence that owns KV (in
prefill, prefilled, decoding or ON_HOLD), not its current allocation, so
admitted work can always grow to its budget. The Host margin keeps the pool
out of the band where the decode boundary evicts running sequences
(``--host-kv-eviction-watermark``) and above the growth planner's own 5 %
safety reserve; held KV lives only on Host in this build (its GPU pages are
released when the turn ends), so it costs no GPU pages and the GPU side has
no watermark that preempts below a free threshold -- a projected budget is
an upper bound of a sequence's GPU allocation, so GPU needs no margin.

Ordering (no deadlock): evicted sequences first (unchanged), then resumed
turns, then new work. Resumed turns need only an increment and free nothing
by waiting, so while any resumed turn cannot fit, no new work is admitted
(``resumed_turn_waiting``). A resumed turn that cannot fit waits for running
sequences to finish. If nothing is running and it still cannot fit, every
free page is held by other sessions: a genuine overcommit, reported loudly
and counted. Held KV is never dropped for it (choosing a victim is A4).

Pure and deterministic: the worker passes replicated inputs only (the global
batch, the replicated hold book, gathered pool totals), so every rank makes
the same decision without communication.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Default Host safety margin, percent of the node's Host pool. Equal to the
# default --host-kv-eviction-watermark: the decode boundary evicts running
# sequences whenever free Host pages drop below that share, so admission must
# never project free pages below it. It also exceeds the boundary growth
# planner's own 5 % reserve (``boundary.py``: ``safety_margin = total * 0.05``).
DEFAULT_ADMISSION_MARGIN_PERCENT = 10.0
# The boundary growth planner keeps this share of the Host pool free on top of
# any pending growth; a smaller admission margin would let growth fail.
HOST_GROWTH_SAFETY_PERCENT = 5.0

# Why a candidate was not admitted this wave (counted per candidate and wave).
DEFER_HOST = "host_footprint"
DEFER_GPU = "gpu_footprint"
DEFER_RESUMED_FIRST = "resumed_turn_waiting"
DEFER_OVERCOMMIT = "overcommit"
DEFER_REASONS = (DEFER_HOST, DEFER_GPU, DEFER_RESUMED_FIRST, DEFER_OVERCOMMIT)

# Per-request error code for a request whose projected footprint exceeds its
# whole pool (after the margin): no amount of waiting admits it.
REJECT_NEVER_FITS = "kv_footprint_exceeds_capacity"

KIND_EVICTED = "evicted"
KIND_RESUMED = "resumed"
KIND_NEW = "new"


class KVAdmissionConfigError(ValueError):
    """The footprint admission margin cannot keep its guarantee."""


def ceil_div(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor))


def validate_margin_percent(
    margin_percent: float, *, eviction_watermark_percent: float
) -> float:
    """Return the margin, or refuse one the decode boundary would undercut."""
    margin = float(margin_percent)
    if not 0.0 <= margin < 100.0:
        raise KVAdmissionConfigError(
            f"session KV admission margin must be in [0, 100), got {margin}"
        )
    floor = max(float(eviction_watermark_percent), HOST_GROWTH_SAFETY_PERCENT)
    if margin < floor:
        raise KVAdmissionConfigError(
            f"session KV admission margin {margin}% is below "
            f"{floor}% (max of --host-kv-eviction-watermark="
            f"{eviction_watermark_percent}% and the boundary's "
            f"{HOST_GROWTH_SAFETY_PERCENT}% growth reserve): admitted work "
            "would be evicted by the decode boundary"
        )
    return margin


def margin_pages(total_pages: int, margin_percent: float) -> int:
    return int(math.ceil(int(total_pages) * float(margin_percent) / 100.0))


@dataclass(frozen=True)
class FootprintCandidate:
    """One admission candidate's projected KV footprint."""

    uuid: str
    kind: str  # KIND_EVICTED / KIND_RESUMED / KIND_NEW
    node_id: int
    rank: int
    global_idx: int
    # Net Host pages admission adds on its node (resumed: budget - held).
    host_pages: int
    # GPU pages on its rank: the whole budget (held KV has no GPU pages).
    gpu_pages: int
    # Whole budget in pages; decides "can never fit" independent of state.
    budget_pages: int
    # Pages its session already holds (resumed only).
    held_pages: int = 0
    total_decoded_before_eviction: int = 0

    def order_key(self) -> Tuple:
        if self.kind == KIND_EVICTED:
            return (0, -self.total_decoded_before_eviction, self.global_idx)
        if self.kind == KIND_RESUMED:
            return (1, self.global_idx)
        return (2, self.global_idx)


@dataclass(frozen=True)
class FootprintState:
    """Pool totals and the projected footprint already committed."""

    host_total_by_node: Tuple[int, ...]
    host_running_by_node: Tuple[int, ...]
    host_held_by_node: Tuple[int, ...]
    host_margin_by_node: Tuple[int, ...]
    # Per rank; None (or a negative entry) = GPU pool size not known yet.
    gpu_total_by_rank: Optional[Tuple[int, ...]]
    gpu_running_by_rank: Tuple[int, ...]

    def host_free(self) -> List[int]:
        return [
            int(total) - int(running) - int(held) - int(margin)
            for total, running, held, margin in zip(
                self.host_total_by_node, self.host_running_by_node,
                self.host_held_by_node, self.host_margin_by_node,
            )
        ]

    def gpu_known(self, rank: int) -> bool:
        return (
            self.gpu_total_by_rank is not None
            and 0 <= rank < len(self.gpu_total_by_rank)
            and int(self.gpu_total_by_rank[rank]) >= 0
        )

    def gpu_free(self) -> List[Optional[int]]:
        return [
            (int(self.gpu_total_by_rank[rank]) - int(running))
            if self.gpu_known(rank) else None
            for rank, running in enumerate(self.gpu_running_by_rank)
        ]


@dataclass(frozen=True)
class FootprintPlan:
    """The decision: admitted uuids in priority order, and why the rest wait."""

    admitted: Tuple[str, ...]
    deferred: Mapping[str, str]
    never_fits: Mapping[str, str]
    # A resumed turn cannot fit, nothing is running and nothing was admitted.
    overcommit: bool
    order: Tuple[str, ...]
    host_free_after: Tuple[int, ...]
    blocked_resumed: Tuple[str, ...] = ()

    def deferred_by_reason(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for reason in self.deferred.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts


def plan_footprint_admission(
    candidates: Sequence[FootprintCandidate],
    state: FootprintState,
    *,
    live: bool,
) -> FootprintPlan:
    """Decide which candidates fit their projected footprint, in priority order.

    ``live`` is whether any sequence owns KV anywhere (rank-uniform OR): a
    resumed turn that cannot fit while something runs waits for it to finish;
    with nothing running it is a genuine overcommit.
    """
    num_nodes = len(state.host_total_by_node)
    num_ranks = len(state.gpu_running_by_rank)
    for field_name in ("host_running_by_node", "host_held_by_node", "host_margin_by_node"):
        if len(getattr(state, field_name)) != num_nodes:
            raise ValueError(f"{field_name} must have one value per node")
    if state.gpu_total_by_rank is not None and len(state.gpu_total_by_rank) != num_ranks:
        raise ValueError("gpu_total_by_rank must have one value per rank")

    ordered = sorted(candidates, key=lambda c: (c.order_key(), c.uuid))
    host_free = state.host_free()
    gpu_free = state.gpu_free()
    admitted: List[str] = []
    deferred: Dict[str, str] = {}
    never_fits: Dict[str, str] = {}
    blocked_resumed: List[str] = []

    for c in ordered:
        if not 0 <= c.node_id < num_nodes:
            raise ValueError(f"candidate {c.uuid} has invalid node_id={c.node_id}")
        if not 0 <= c.rank < num_ranks:
            raise ValueError(f"candidate {c.uuid} has invalid rank={c.rank}")
        if c.host_pages < 0 or c.gpu_pages < 0 or c.budget_pages <= 0:
            raise ValueError(f"candidate {c.uuid} has an invalid footprint {c}")
        usable_host = (
            int(state.host_total_by_node[c.node_id])
            - int(state.host_margin_by_node[c.node_id])
        )
        if c.budget_pages > usable_host:
            never_fits[c.uuid] = (
                f"projected KV footprint {c.budget_pages} Host pages exceeds the "
                f"usable Host pool {usable_host} pages (total "
                f"{state.host_total_by_node[c.node_id]} minus margin "
                f"{state.host_margin_by_node[c.node_id]}) on node {c.node_id}"
            )
            continue
        if state.gpu_known(c.rank) and c.gpu_pages > int(state.gpu_total_by_rank[c.rank]):
            never_fits[c.uuid] = (
                f"projected KV footprint {c.gpu_pages} GPU pages exceeds the GPU "
                f"KV pool {state.gpu_total_by_rank[c.rank]} pages of rank {c.rank}"
            )
            continue
        if c.kind == KIND_NEW and blocked_resumed:
            deferred[c.uuid] = DEFER_RESUMED_FIRST
            continue
        host_ok = c.host_pages <= host_free[c.node_id]
        gpu_ok = gpu_free[c.rank] is None or c.gpu_pages <= gpu_free[c.rank]
        if host_ok and gpu_ok:
            admitted.append(c.uuid)
            host_free[c.node_id] -= c.host_pages
            if gpu_free[c.rank] is not None:
                gpu_free[c.rank] -= c.gpu_pages
            continue
        deferred[c.uuid] = DEFER_HOST if not host_ok else DEFER_GPU
        if c.kind == KIND_RESUMED:
            blocked_resumed.append(c.uuid)

    overcommit = bool(blocked_resumed) and not live and not admitted
    if overcommit:
        for uuid in blocked_resumed:
            deferred[uuid] = DEFER_OVERCOMMIT
    return FootprintPlan(
        admitted=tuple(admitted),
        deferred=deferred,
        never_fits=never_fits,
        overcommit=overcommit,
        order=tuple(c.uuid for c in ordered),
        host_free_after=tuple(host_free),
        blocked_resumed=tuple(blocked_resumed),
    )


@dataclass
class KVAdmissionStats:
    """Footprint admission metrics; identical on every rank."""

    margin_percent: float
    page_tokens: int
    waves: int = 0
    admitted: Dict[str, int] = field(default_factory=dict)
    deferred: Dict[str, int] = field(default_factory=dict)
    rejected_never_fits: int = 0
    overcommit_waits: int = 0
    overcommit_episodes: int = 0
    overcommit_active: bool = False
    # Projected (budget) vs actual (final context) footprint at completion.
    error_samples: int = 0
    projected_pages_sum: int = 0
    actual_pages_sum: int = 0
    error_pages_max: int = 0
    error_pages_min: Optional[int] = None
    # Peaks, sampled at every admission and decode boundary (node 0 / max node).
    host_total_pages: int = 0
    peak_host_used_pages: int = 0
    peak_host_projected_pages: int = 0
    peak_held_pages: int = 0
    peak_running_projected_pages: int = 0
    gpu_total_pages: int = -1
    peak_gpu_projected_pages: int = 0
    host_used_over_total_events: int = 0

    def note_plan(self, plan: FootprintPlan, kinds: Mapping[str, str]) -> None:
        self.waves += 1
        for uuid in plan.admitted:
            kind = kinds.get(uuid, KIND_NEW)
            self.admitted[kind] = self.admitted.get(kind, 0) + 1
        for reason, count in plan.deferred_by_reason().items():
            self.deferred[reason] = self.deferred.get(reason, 0) + count
        self.rejected_never_fits += len(plan.never_fits)

    def note_overcommit(self, active: bool) -> bool:
        """Track an overcommit wait; True when a new episode starts."""
        started = active and not self.overcommit_active
        if active:
            self.overcommit_waits += 1
            if started:
                self.overcommit_episodes += 1
        self.overcommit_active = bool(active)
        return started

    def note_completion(self, *, projected_pages: int, actual_pages: int) -> None:
        error = int(projected_pages) - int(actual_pages)
        self.error_samples += 1
        self.projected_pages_sum += int(projected_pages)
        self.actual_pages_sum += int(actual_pages)
        self.error_pages_max = max(self.error_pages_max, error)
        self.error_pages_min = (
            error if self.error_pages_min is None else min(self.error_pages_min, error)
        )

    def note_sample(
        self,
        *,
        host_total: int,
        host_used: int,
        held: int,
        running_projected: int,
        gpu_total: Optional[int] = None,
        gpu_projected_max: Optional[int] = None,
    ) -> None:
        self.host_total_pages = int(host_total)
        self.peak_host_used_pages = max(self.peak_host_used_pages, int(host_used))
        self.peak_held_pages = max(self.peak_held_pages, int(held))
        self.peak_running_projected_pages = max(
            self.peak_running_projected_pages, int(running_projected)
        )
        self.peak_host_projected_pages = max(
            self.peak_host_projected_pages, int(held) + int(running_projected)
        )
        if int(held) + int(running_projected) > int(host_total):
            self.host_used_over_total_events += 1
        if gpu_total is not None and int(gpu_total) >= 0:
            self.gpu_total_pages = int(gpu_total)
        if gpu_projected_max is not None:
            self.peak_gpu_projected_pages = max(
                self.peak_gpu_projected_pages, int(gpu_projected_max)
            )

    def snapshot(self) -> Dict[str, object]:
        mean_error = (
            (self.projected_pages_sum - self.actual_pages_sum) / self.error_samples
            if self.error_samples else None
        )
        ratio = (
            self.projected_pages_sum / self.actual_pages_sum
            if self.actual_pages_sum else None
        )
        total = self.host_total_pages
        return {
            "margin_percent": self.margin_percent,
            "page_tokens": self.page_tokens,
            "waves": self.waves,
            "admitted": dict(sorted(self.admitted.items())),
            "deferred": dict(sorted(self.deferred.items())),
            "rejected_never_fits": self.rejected_never_fits,
            "overcommit_active": self.overcommit_active,
            "overcommit_episodes": self.overcommit_episodes,
            "overcommit_waits": self.overcommit_waits,
            "footprint_error": {
                "samples": self.error_samples,
                "projected_pages_sum": self.projected_pages_sum,
                "actual_pages_sum": self.actual_pages_sum,
                "mean_error_pages": None if mean_error is None else round(mean_error, 3),
                "max_error_pages": self.error_pages_max,
                "min_error_pages": self.error_pages_min,
                "projected_over_actual": None if ratio is None else round(ratio, 4),
            },
            "host_total_pages": total,
            "peak_host_used_pages": self.peak_host_used_pages,
            "peak_held_pages": self.peak_held_pages,
            "peak_running_projected_pages": self.peak_running_projected_pages,
            "peak_held_plus_running_projected_pages": self.peak_host_projected_pages,
            "peak_host_used_over_capacity": (
                round(self.peak_host_used_pages / total, 4) if total else None
            ),
            "peak_held_plus_running_projected_over_capacity": (
                round(self.peak_host_projected_pages / total, 4) if total else None
            ),
            "projected_over_capacity_events": self.host_used_over_total_events,
            "gpu_total_pages": self.gpu_total_pages,
            "peak_gpu_projected_pages": self.peak_gpu_projected_pages,
        }


def log_metrics(stats: KVAdmissionStats, rank: int, event: str) -> None:
    snap = stats.snapshot()
    logging.info(
        "[KV_ADMISSION] metrics rank=%s event=%s waves=%s admitted=%s deferred=%s "
        "rejected_never_fits=%s overcommit_active=%s overcommit_episodes=%s "
        "overcommit_waits=%s footprint_error=%s host_total_pages=%s "
        "peak_host_used_pages=%s peak_held_pages=%s "
        "peak_held_plus_running_projected_pages=%s gpu_total_pages=%s "
        "peak_gpu_projected_pages=%s",
        rank, event, snap["waves"], snap["admitted"], snap["deferred"],
        snap["rejected_never_fits"], snap["overcommit_active"],
        snap["overcommit_episodes"], snap["overcommit_waits"],
        snap["footprint_error"], snap["host_total_pages"],
        snap["peak_host_used_pages"], snap["peak_held_pages"],
        snap["peak_held_plus_running_projected_pages"], snap["gpu_total_pages"],
        snap["peak_gpu_projected_pages"],
    )
