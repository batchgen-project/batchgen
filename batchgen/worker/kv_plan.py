"""Pure host/GPU KV capacity arithmetic of one ``BatchGenWorker``.

``kv_plan`` owns the capacity terms the worker used to compute inline,
in some cases twice:

* the per-node capacity verdict over gathered rank reports
  (``verdict_node_capacity``). The prefill admission preflight and the
  boundary growth preflight each carried a copy of the same loop; both now
  call this one definition. The same-node capacity-mismatch check is part
  of the verdict for both callers.
* the prefill owner demand in host pages (``prefill_owner_demand_pages``).
* the watermark credit for reclaimable retained prefix pages
  (``credit_reclaimable_pages``), the pure part only.
* the provisional GPU KV pool formula ``total * frac - used``
  (``provisional_pool_gb``). Its two call sites keep their intentionally
  different failure policies (warn-and-floor vs. fail-loud) on the worker.
* the page-rounded effective host-KV chunk size (``effective_chunk_size``).
* the "this row has no next-token runway" predicate
  (``host_capacity_due``).

What it deliberately does NOT own: collectives (all_gather / all_reduce),
tensor ``.item()`` conversions, allocator transactions, coordinator reads and
their gating, logging, warnings and raises, the single-rank early returns of
the preflights, the adaptive chunk sizer, the prefill pool wave (callers pass
its already-computed page count), host growth pages (owned by
``SequenceEntry.get_host_growth_pages``), and KV geometry. The pure types
already extracted into ``worker/kv_manager.py`` stay there; this module only
adds new names. It holds no state.

Every input is a plain int, float or mapping; arithmetic is copied verbatim
from the worker, including its float-division rounding, so a multi-rank
consensus computed through it is bit-identical to the old inline code.

Imports: stdlib only (no torch, no batchgen).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple


# ----------------------------------------------------- node capacity verdict


@dataclass(frozen=True)
class NodeCapacityReport:
    """One rank's gathered capacity row.

    ``free`` and ``total`` are ``-1`` on a rank that does not report the
    node's shared allocator (only the node leader does). ``snapshot_error``
    is non-zero when the rank's stats call failed.
    """

    node_id: int
    need: int
    free: int
    total: int
    snapshot_error: int


@dataclass(frozen=True)
class NodeCapacityVerdict:
    """Local (pre-reduction) verdict over all gathered rows.

    ``ok``: every node's summed need fits its free and total pages and the
    snapshot is sound. ``rows_usable`` / ``rows`` are only tracked when the
    caller asked for them (``want_rows=True``); otherwise both are ``None``.
    ``rows`` maps node id to ``(need, free)`` and is ``None`` when the
    snapshot is unusable (a stats failure, a node with no capacity report,
    or two reports of one node that disagree).
    """

    ok: bool
    rows_usable: Optional[bool]
    rows: Optional[Dict[int, Tuple[int, int]]]


def verdict_node_capacity(
    reports: Iterable[NodeCapacityReport], *, want_rows: bool
) -> NodeCapacityVerdict:
    """Aggregate gathered rank reports per node and decide whether they fit.

    Iteration order is the gather order, and dict insertion order follows
    it, exactly as the worker's inline loops did.
    """
    node_needs: Dict[int, int] = {}
    node_capacity: Dict[int, Tuple[int, int]] = {}
    ok = True
    # A rejection that is merely "short of pages" is retryable by reclaim; a
    # broken snapshot is not, and the two must not be confused.
    rows_usable = True
    for report in reports:
        reported_node = report.node_id
        node_needs[reported_node] = node_needs.get(reported_node, 0) + report.need
        if report.snapshot_error != 0:
            ok = False
            rows_usable = False
        if report.free >= 0:
            capacity = (report.free, report.total)
            previous = node_capacity.get(reported_node)
            if previous is not None and previous != capacity:
                ok = False
                rows_usable = False
            node_capacity[reported_node] = capacity
    for node, needed in node_needs.items():
        free_total = node_capacity.get(node)
        if free_total is None:
            ok = False
            rows_usable = False
            continue
        free_pages, total_pages = free_total
        if needed < 0 or needed > free_pages or needed > total_pages:
            ok = False
    if not want_rows:
        return NodeCapacityVerdict(ok=ok, rows_usable=None, rows=None)
    if not rows_usable:
        return NodeCapacityVerdict(ok=ok, rows_usable=False, rows=None)
    return NodeCapacityVerdict(
        ok=ok,
        rows_usable=True,
        rows={
            node: (needed, node_capacity[node][0])
            for node, needed in node_needs.items()
        },
    )


# ------------------------------------------------------- prefill owner demand


def prefill_owner_demand_pages(
    owner_request_tokens: Iterable[int],
    *,
    page_size: int,
    pool_segment_pages: int,
) -> int:
    """Host pages one rank's prefill owner requests need.

    ``pool_segment_pages`` is the prefill pool wave's computed-segment page
    count, already computed by the caller (and 0 for a rank with no owner
    demand: only the wave's host-KV leader allocates the segments).
    """
    return (
        sum(math.ceil(tokens / page_size) for tokens in owner_request_tokens)
        + pool_segment_pages
    )


# ------------------------------------------------- watermark reclaim credit


def credit_reclaimable_pages(
    stats: Mapping[str, int], releasable: int
) -> Mapping[str, int]:
    """Report ``releasable`` retained prefix pages as free, not as pressure.

    Returns ``stats`` itself (not a copy) when there is nothing to credit;
    otherwise a new dict with ``num_used_pages`` / ``num_free_pages`` moved
    by the credited count, ``num_reclaimable_prefix_pages`` set, and
    ``free_percent`` recomputed.
    """
    releasable = max(0, releasable)
    if releasable == 0:
        return stats
    total = int(stats["num_total_pages"])
    used = int(stats["num_used_pages"])
    # A retained page is charged to `used`, so crediting it moves the same
    # count. Clamped so a stale read can never make `used` negative or
    # free% exceed 100 -- a watermark fed an impossible snapshot would
    # preempt decode on every step.
    credited = min(releasable, used)
    free = int(stats["num_free_pages"]) + credited
    updated = dict(stats)
    updated["num_used_pages"] = used - credited
    updated["num_free_pages"] = free
    updated["num_reclaimable_prefix_pages"] = credited
    updated["free_percent"] = (
        int((free / total) * 100) if total > 0 else 100
    )
    return updated


# ----------------------------------------------------- provisional pool size


def provisional_pool_gb(
    total_gb: float,
    frac: float,
    used_gb: float,
    *,
    floor_gb: Optional[float],
) -> float:
    """Provisional GPU KV pool size: ``total * frac - used``.

    ``floor_gb=None`` returns the raw value, non-positive included; the
    caller decides (fail-loud). A float ``floor_gb`` is substituted when the
    raw value is non-positive (``<= 0``, so a NaN passes through unchanged,
    as it did inline). Warnings and raises stay at the call sites.
    """
    size_gb = total_gb * frac - used_gb
    if floor_gb is not None and size_gb <= 0:
        return floor_gb
    return size_gb


# ------------------------------------------------------ host-KV chunk sizing


def effective_chunk_size(
    chunk: int,
    *,
    max_decoding_length: int,
    decision_interval: int,
    page_size: int,
) -> int:
    """Clamp a base host-KV chunk and round it up to a page boundary."""
    # Cap by max_decoding_length — no point reserving more than max decode
    if max_decoding_length > 0:
        chunk = min(chunk, max_decoding_length)
    # A boundary is the only place where host pages may be allocated.  Keep
    # one complete decision interval of runway so the next boundary is
    # reached while the metadata invariant still holds.  This removes the
    # unsafe post-sample emergency growth path.
    chunk = max(chunk, decision_interval)
    # Round up to page boundary
    chunk = math.ceil(chunk / page_size) * page_size
    return chunk


def host_capacity_due(host_token_capacity: int, current_context_length: int) -> bool:
    """Whether a decode row has no host-KV runway for its next token."""
    return host_token_capacity <= current_context_length
