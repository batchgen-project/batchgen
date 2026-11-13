"""KV cache helpers — read-only stats + page-table capacity tiers.

Slice 5 of the worker decouple initiative (issue #175). Sub-slice 5.1
landed the read-only stats subset:

  - ``_get_host_kv_free_pages`` (5 LOC)  → ``KVCacheManager.get_host_free_pages``
  - ``_get_gpu_kv_free_pages`` (6 LOC)   → ``KVCacheManager.get_gpu_free_pages``
  - ``_get_host_kv_utilization`` (65 LOC) → ``KVCacheManager.get_host_utilization``

Sub-slice 5.2 adds the page-table capacity helpers (pure math; no
state mutation, no backend, no NCCL):

  - ``_cuda_graph_page_table_token_capacity`` (27 LOC) →
    ``KVCacheManager.page_table_token_capacity``
  - ``_cuda_graph_page_table_slot_capacity`` (19 LOC) →
    ``KVCacheManager.page_table_slot_capacity``
  - ``_with_cuda_graph_page_table_capacity`` (22 LOC) →
    ``KVCacheManager.apply_page_table_capacity``

Sub-slice 5.3 adds the token-budget cache helpers:

  - ``_get_sequence_token_budget`` (20 LOC) →
    ``KVCacheManager.get_sequence_token_budget``
  - ``_compute_host_kv_sequence_tokens`` (3 LOC) →
    ``KVCacheManager.compute_host_kv_sequence_tokens``

The token-budget helpers memoize a per-sequence ``kv_token_budget`` on
the ``QueryBookEntry`` passed via snapshot. That mutation is *on the
entry object the worker passes in*, not on worker state — handler stays
stateless w.r.t. worker.

Sub-slice 5.4a adds the GPU-KV-manager allocation *planning* extracted
from ``_ensure_gpu_paged_kv_manager``:

  - ``KVCacheManager.plan_gpu_kv_manager`` — builds the primary (+ aux,
    for DSA models) ``GPUPagedKVConfig`` and decides whether the worker
    can reuse the existing manager or must destroy + recreate it.

Only the *decision* (config sizing + reuse/recreate) is ported; the GPU
side effects (``GPUPagedKVCacheManager`` create / destroy / initialize /
bind) stay on the worker, which applies the returned ``GpuKvManagerPlan``.
The allocation/IO methods are ~90% irreducible side effects, so wrapping
them behind a Backend Protocol would add scaffolding without testability
gain — we extract only the genuinely pure planning step here.

Sub-slice 5.5 adds the host-KV watermark *trigger decision* extracted
from ``_check_host_kv_watermark_trigger``:

  - ``KVCacheManager.plan_watermark_trigger`` — given the per-node host-KV
    stats gathered across ranks, decides whether decode should be
    preempted for prefill (free space above watermark AND work waiting),
    and aggregates the global page stats for logging.

Only the *decision* is ported; the cross-rank ``dist.all_gather_object``
that collects the per-node stats stays on the worker (NCCL side effect).
The worker applies the returned ``WatermarkTriggerPlan`` — stores the
aggregated stats and emits the log lines.

Sub-slice 5.6 adds the host-KV migration *planner* extracted from
``_plan_kv_migration``:

  - ``KVCacheManager.plan_kv_migration`` — greedy rebalance of host-KV
    pages across nodes: pick the smallest sequences off overloaded nodes
    and assign them (round-robin across ranks) to the least-loaded nodes
    until balanced.

Only the *decision* is ported; the cross-rank ``dist.all_gather_object``
that gathers per-node stats and the candidate enumeration over
``global_batch`` stay on the worker, which builds the request and applies
the returned ``MigrationOp`` list via the NCCL migration executors.

The KV Cache Helper section has 27 methods totaling ~1330 LOC; they
span four distinct concerns (read-only stats, allocation, planning,
migration execution). Porting them all in one slice would be too risky —
each concern landed in its own sub-slice (5.1–5.6).

Design follows the per-slice Backend Protocol pattern introduced by
``SyncCoordinator`` (Slice 3): the handler takes a ``KVStatsBackend``
that the worker wires to its real KV managers; tests wire a fake.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Protocol, Sequence, Tuple

if TYPE_CHECKING:
    from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig
    from batchgen.migration import MigrationOp
    from batchgen.query_book import QueryBookEntry
    from batchgen.sequence import SequenceBatch


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU KV pool sizing constants
# ---------------------------------------------------------------------------

#: Device memory deliberately left free ABOVE the GPU KV pool, in GiB.
#:
#: The post-capture steady-state measurement already contains every persistent
#: allocation (weights, resident experts, graph pools and their static buffers,
#: NCCL/JIT first-touch caches), so the pool no longer needs the accidental
#: ``(1 - gpu_memory_frac)`` slack that used to absorb the graph footprint the
#: pre-capture measurement could not see. What still needs covering is only the
#: memory nothing else accounts for: decode transients outside the captured
#: graphs (eager-fallback activations, sampling temporaries) and allocator
#: rounding. That is an explicit, auditable constant — not a fraction of the
#: device whose size nobody chose.
#:
#: 2.0 GiB is the measured anchor: on the 2026-11-03 H200 GLM-5.3 server
#: (frac 0.93, 139.81 GiB device) the first decode round ran its whole batch
#: with 3.96 GiB free above an identically-captured steady state, so the
#: non-pool demand fits inside 2.0 GiB with the other ~2 GiB handed to pool
#: capacity. Raising this constant costs capacity; lowering it is only
#: defensible against a measured decode-transient peak.
GPU_KV_POOL_HEADROOM_GB: float = 2.0


# ---------------------------------------------------------------------------
# Stats dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVStats:
    """C++-backed page-counter snapshot for a paged KV manager."""

    num_free_pages: int
    num_used_pages: int
    num_total_pages: int


@dataclass(frozen=True)
class HostKVUtilization:
    """Per-node host-KV utilization view.

    Host KV is shared across all ranks on a node; the counts here are
    aggregated over the node-local rank range, not the whole world.
    """

    rank: int
    node_id: int
    num_free_pages: int
    num_total_pages: int
    num_used_pages: int
    free_percent: int
    num_in_decode: int
    num_onhold: int
    num_prefilled: int
    num_valid_sequences: int


@dataclass(frozen=True)
class KVUtilizationRequest:
    """Frozen snapshot passed to ``get_host_utilization``."""

    rank: int
    world_size: int
    local_rank: int
    num_gpus_per_node: int
    global_batch: "SequenceBatch"


# ---------------------------------------------------------------------------
# Backend Protocol
# ---------------------------------------------------------------------------


class KVStatsBackend(Protocol):
    """Tier-1 KV backend: read-only stats.

    Production wires the worker's ``host_paged_kv_worker_view`` /
    ``gpu_paged_kv_cache_manager``. Tests wire a fake.
    """

    def get_host_stats(self) -> KVStats: ...

    def get_gpu_stats(self) -> Optional[KVStats]: ...


# ---------------------------------------------------------------------------
# Page-table capacity (Phase 5.2): frozen request snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageTableCapacityRequest:
    """Frozen snapshot for the page-table capacity helpers.

    Token capacity is the resolved deployment bound, carried in the historical
    ``model_max_position_embeddings`` field. Slot capacity takes the maximum
    of the relevant batch and graph settings. Optional fields are ``None``
    when the underlying worker attribute is unset. The snapshot carries only
    what the two capacity helpers read: per-batch lengths have no place here,
    because the page table's shape is a deployment constant.
    """

    engine_module_global_batch_size: Optional[int]
    engine_module_attn_decoding_micro_batch_size: Optional[int]
    engine_basic_num_queries: Optional[int]
    model_max_position_embeddings: Optional[int]
    args_cuda_graph_max_bucket_size: Optional[int]
    engine_basic_decode_graph_max_bucket: Optional[int] = None


# ---------------------------------------------------------------------------
# Token-budget cache (Phase 5.3): frozen request snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenBudgetRequest:
    """Frozen snapshot for the per-sequence host-KV token-budget cache.

    The budget is read from ``query_book[sequence_id].kv_token_budget``,
    which admission sets.
    """

    query_book: Mapping[int, "QueryBookEntry"]


# ---------------------------------------------------------------------------
# GPU-KV-manager allocation planning (Phase 5.4a): frozen request + plan
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Host-KV watermark trigger (Phase 5.5): frozen request + plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatermarkTriggerRequest:
    """Frozen snapshot for ``plan_watermark_trigger``.

    ``node_stats`` is the per-node host-KV utilization view, one entry per
    node representative (already gathered across ranks by the worker via
    ``dist.all_gather_object`` and filtered to non-None). Each entry is the
    dict produced by ``get_host_utilization`` (``HostKVUtilization`` fields):
    ``free_percent``, ``num_used_pages``, ``num_total_pages``,
    ``num_free_pages``, ``node_id``.
    """

    node_stats: Tuple[Mapping[str, int], ...]
    host_kv_watermark: int
    has_queued: bool
    has_evicted: bool


@dataclass(frozen=True)
class WatermarkGlobalStats:
    """Aggregated host-KV page stats across all nodes (for logging)."""

    used: int
    total: int
    free_percent: int
    num_nodes: int


@dataclass(frozen=True)
class WatermarkTriggerPlan:
    """What ``plan_watermark_trigger`` returns; the worker applies it.

    * ``should_trigger`` — interrupt decode and switch to prefill.
    * ``max_free_percent`` — highest per-node free %; ``None`` when there
      are no node stats (worker returns False without aggregating).
    * ``global_stats`` — aggregated page stats the worker stores on
      ``self._host_kv_page_stats`` (rank-0 logging). ``None`` when there
      are no node stats.
    """

    should_trigger: bool
    max_free_percent: Optional[int]
    global_stats: Optional[WatermarkGlobalStats]


# ---------------------------------------------------------------------------
# Host-KV migration planning (Phase 5.6): frozen request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationCandidate:
    """A sequence eligible for host-KV migration (PREFILLED or ON_HOLD).

    The worker enumerates these from ``global_batch`` (one snapshot per
    candidate); the planner buckets them by source node and selects.
    """

    uuid: str
    assigned_rank: int
    global_idx: int
    kv_token_budget: int
    host_pages_allocated: int


@dataclass(frozen=True)
class MigrationPlanRequest:
    """Frozen snapshot for ``plan_kv_migration``.

    ``node_stats`` maps ``node_id -> {num_used_pages, num_total_pages}``
    (gathered across ranks). ``candidates`` is every PREFILLED / ON_HOLD
    sequence across all ranks; the planner sorts them by ``global_idx`` so
    enumeration order is irrelevant.
    """

    node_stats: Mapping[int, Mapping[str, int]]
    candidates: Tuple[MigrationCandidate, ...]
    num_gpus_per_node: int
    world_size: int


# ---------------------------------------------------------------------------
# KVCacheManager (stats + page-table capacity tiers)
# ---------------------------------------------------------------------------


class KVCacheManager:
    """KV cache helper — stats + page-table capacity (Phases 5.1, 5.2).

    Future sub-slices add allocation, planning, and migration; the
    constructor accepts only ``KVStatsBackend`` for now.
    """

    def __init__(self, *, backend: KVStatsBackend) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Free-page counters
    # ------------------------------------------------------------------
    def get_host_free_pages(self) -> int:
        """Free pages on the host-side KV worker view."""
        return self._backend.get_host_stats().num_free_pages

    def get_gpu_free_pages(self) -> int:
        """Free pages on the GPU paged KV manager; ``0`` if not yet bound."""
        stats = self._backend.get_gpu_stats()
        return stats.num_free_pages if stats is not None else 0

    # ------------------------------------------------------------------
    # Host-KV utilization (aggregated per-node)
    # ------------------------------------------------------------------
    def get_host_utilization(self, req: KVUtilizationRequest) -> HostKVUtilization:
        """Aggregate host KV stats counting sequences with KV in host memory.

        Valid statuses = ``PREFILLED``, ``ON_HOLD``, ``IN_DECODE`` (all
        have KV in host). Host KV is shared per-node, so we count
        sequences from ALL ranks on this node, not just this rank.

        Uses C++ ground truth for page counts — shared-memory atomic
        counters are accurate per-node, unlike per-sequence
        ``host_pages_allocated`` which is stale on non-owner ranks
        between metadata syncs.
        """
        # Local import to avoid a module-load cycle with batchgen.sequence
        # (which transitively pulls torch — heavy at module init).
        from batchgen.sequence import SequenceStatus

        stats = self._backend.get_host_stats()

        node_id = req.rank // req.num_gpus_per_node
        node_rank_start = node_id * req.num_gpus_per_node
        node_rank_end = min(node_rank_start + req.num_gpus_per_node, req.world_size)

        # CRITICAL: IN_DECODE sequences also have KV in host (streams after each layer)
        valid_statuses = {
            SequenceStatus.PREFILLED,
            SequenceStatus.ON_HOLD,
            SequenceStatus.IN_DECODE,
        }

        status_counts: dict = {status: [] for status in valid_statuses}
        for rank_on_node in range(node_rank_start, node_rank_end):
            for status in valid_statuses:
                seqs = req.global_batch.get_sequences_for_rank_with_status(
                    rank_on_node, status
                )
                status_counts[status].extend(seqs)

        valid_sequences = []
        for seqs in status_counts.values():
            valid_sequences.extend(seqs)

        used_pages = stats.num_used_pages
        free_pages = stats.num_free_pages
        free_percent = (
            int((free_pages / stats.num_total_pages) * 100)
            if stats.num_total_pages > 0
            else 100
        )

        if req.local_rank == 0:
            logger.debug(
                f"[HOST_KV_UTIL] C++ stats: used={used_pages}, free={free_pages}, "
                f"total={stats.num_total_pages}, {len(valid_sequences)} valid seqs"
            )

        return HostKVUtilization(
            rank=req.rank,
            node_id=node_id,
            num_free_pages=free_pages,
            num_total_pages=stats.num_total_pages,
            num_used_pages=used_pages,
            free_percent=free_percent,
            num_in_decode=len(status_counts[SequenceStatus.IN_DECODE]),
            num_onhold=len(status_counts[SequenceStatus.ON_HOLD]),
            num_prefilled=len(status_counts[SequenceStatus.PREFILLED]),
            num_valid_sequences=len(valid_sequences),
        )

    # ------------------------------------------------------------------
    # Page-table capacity (Phase 5.2)
    # ------------------------------------------------------------------
    @staticmethod
    def page_table_token_capacity(req: PageTableCapacityRequest) -> int:
        """The deployment bound, independent of any batch's prompt/decode mix.

        The snapshot's historical ``model_max_position_embeddings`` field
        carries the resolved deployment length, including startup auto-fit.
        """
        length = req.model_max_position_embeddings
        if length is None or length < 2:
            raise ValueError("page-table sizing requires a positive deployment max_model_len")
        return int(length)

    @staticmethod
    def page_table_slot_capacity(req: PageTableCapacityRequest) -> int:
        """Maximum number of slots a CUDA-graph page table must address.

        Takes the max of (``args.cuda_graph_max_bucket_size`` if set,
        ``engine.global_batch_size``, ``engine.attn_decoding_micro_batch_size``,
        ``engine.num_queries``, the largest planner ``decode_graph_buckets``
        entry). Falls back to ``1`` when none are set.
        """
        candidates: list = []
        v = req.args_cuda_graph_max_bucket_size
        if v is not None and int(v) > 0:
            candidates.append(int(v))
        for v in (
            req.engine_module_global_batch_size,
            req.engine_module_attn_decoding_micro_batch_size,
            req.engine_basic_num_queries,
            req.engine_basic_decode_graph_max_bucket,
        ):
            if v is not None and int(v) > 0:
                candidates.append(int(v))
        return max(candidates) if candidates else 1

    @staticmethod
    def apply_page_table_capacity(req: PageTableCapacityRequest, config: Any) -> Any:
        """Return a copy of ``config`` with CUDA-graph page-table fields populated.

        Requires ``config`` to be a dataclass with ``num_pages``,
        ``page_size_tokens``, ``cuda_graph_max_pages_per_sequence``, and
        ``cuda_graph_max_slots`` attributes — uses ``dataclasses.replace``
        to construct the updated copy.
        """
        token_capacity = KVCacheManager.page_table_token_capacity(req)
        # Decoupled from pool size (measure-once-cache contract): the page
        # table's SHAPE comes from token/slot capacity only, so captured
        # graphs never depend on the pool's page count and a pool resize
        # cannot invalidate the baked table shape. Values stored in the table
        # are page ids bounded by the live pool at runtime; a shape wider
        # than the pool is unused columns (KB-MB scale), never an error.
        page_capacity = max(
            1,
            math.ceil(token_capacity / int(config.page_size_tokens)),
        )
        slot_capacity = max(1, KVCacheManager.page_table_slot_capacity(req))
        return replace(
            config,
            max_model_len=token_capacity,
            cuda_graph_max_pages_per_sequence=page_capacity,
            cuda_graph_max_slots=slot_capacity,
        )

    # ------------------------------------------------------------------
    # Token-budget cache (Phase 5.3)
    # ------------------------------------------------------------------
    @staticmethod
    def get_sequence_token_budget(req: TokenBudgetRequest, sequence_id: int) -> int:
        """Return the host-allocation token budget recorded at admission.

        Reads ``query_book[sequence_id].kv_token_budget``. Admission always
        sets it (clamped to the model context), so a missing value is a bug.

        Raises ``RuntimeError`` if ``query_book`` is empty (means the worker
        has not initialized it yet) or the entry has no budget, and
        ``KeyError`` if the sequence is missing.
        """
        if not req.query_book:
            raise RuntimeError("query_book is not initialized before KV allocation")
        query_entry = req.query_book.get(sequence_id)
        if query_entry is None or query_entry.encoded is None:
            raise KeyError(f"Missing query entry for sequence {sequence_id}")
        if query_entry.kv_token_budget is None:
            raise RuntimeError(f"Query entry for sequence {sequence_id} has no kv_token_budget")
        return query_entry.kv_token_budget

    @staticmethod
    def compute_host_kv_sequence_tokens(
        req: TokenBudgetRequest, sequence_ids: Sequence[int]
    ) -> List[int]:
        """Reuse cached token budgets so host/GPU allocations stay consistent."""
        return [
            KVCacheManager.get_sequence_token_budget(req, sequence_id)
            for sequence_id in sequence_ids
        ]

    # ------------------------------------------------------------------
    # GPU-KV-manager allocation planning (Phase 5.4a)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Measure-once-cache pool sizing (pure helpers; the worker applies them).
    # The provisional formula ``total * frac - used`` has one owner,
    # ``batchgen.worker.kv_plan.provisional_pool_gb``.
    # ------------------------------------------------------------------
    @staticmethod
    def compute_final_pool_gb(
        total_gb: float,
        used_gb: float,
        provisional_pool_gb: float,
        headroom_gb: float = GPU_KV_POOL_HEADROOM_GB,
    ) -> float:
        """Final pool size from the post-capture steady-state measurement.

        ``used_gb`` is sampled AFTER the full graph warmup/capture with the
        provisional pool still allocated; subtracting the provisional pool
        yields the steady residency (weights + graph pools + runtime), and the
        pool is everything left on the device above it, minus the explicit
        ``headroom_gb``.

        ``gpu_memory_frac`` is deliberately NOT applied here. It bounds the
        PRE-capture provisional measurement, where the graph footprint is not
        yet allocated and its slack is the only thing covering it. Applying it
        again after the capture double-charges that footprint: the graphs are
        inside ``steady_used_gb`` already, and ``(1 - frac) * total`` then sits
        permanently free (9.79 GiB of a 139.81 GiB device at frac 0.93) for no
        named consumer. The explicit constant replaces that accidental slack.

        Pure and deterministic, and independent of the provisional pool it was
        measured against: ``used_gb - provisional_pool_gb`` is the same steady
        residency whatever pool was resident, so re-measuring with the final
        pool in place returns the final pool again (measure-once-cache
        contract, POIS 2026-10-05). The worker caches the result and reuses it
        for every later pool (re)creation.

        Fail-loud on inputs that cannot describe a real measurement; the
        caller fail-louds on a result below the usable floor.
        """
        if total_gb <= 0:
            raise RuntimeError(
                f"[GPU-KV] Invalid device total for pool sizing: {total_gb} GB"
            )
        if headroom_gb < 0:
            raise RuntimeError(
                f"[GPU-KV] Invalid pool headroom: {headroom_gb} GB (must be >= 0)"
            )
        steady_used_gb = used_gb - provisional_pool_gb
        if steady_used_gb <= 0:
            raise RuntimeError(
                f"[GPU-KV] Post-capture measurement is not a steady state: "
                f"used {used_gb:.2f} GB minus provisional pool "
                f"{provisional_pool_gb:.2f} GB leaves {steady_used_gb:.2f} GB "
                "of residency, but the weights alone outweigh the pool. The "
                "measurement did not see the pool that is allocated."
            )
        return total_gb - steady_used_gb - headroom_gb

    @staticmethod
    def pool_pages_for_gb(pool_gb: float, bytes_per_page: float) -> int:
        """Page count a ``pool_gb`` pool funds at ``bytes_per_page``.

        Non-positive sizes return 0 so the caller's floor check reports the
        shortfall instead of a division artifact.
        """
        if bytes_per_page <= 0:
            raise RuntimeError(
                f"[GPU-KV] Invalid bytes-per-page for pool sizing: {bytes_per_page}"
            )
        if pool_gb <= 0:
            return 0
        return int(pool_gb * (1024 ** 3) / bytes_per_page)

    @staticmethod
    def check_cached_pool_fits(
        cached_pool_gb: float, free_gb: float, *, rank: int = 0
    ) -> None:
        """Fail loud when the cached pool no longer fits the free memory.

        Startup fixes the pool budget. Later residency may consume reserved
        slack, but cannot take capacity promised by that pool. Name a shortfall
        here rather than surface a bare allocation error or silently shrink
        the pool across decode rounds.
        """
        if cached_pool_gb <= free_gb:
            return
        raise RuntimeError(
            f"[GPU-KV] Rank {rank}: cached pool size {cached_pool_gb:.2f} GB "
            f"no longer fits {free_gb:.2f} GB of free device memory "
            f"(short by {cached_pool_gb - free_gb:.2f} GB). Resident memory "
            "grew after the post-capture steady-state measurement; find the "
            "growth instead of shrinking the pool."
        )

    @staticmethod
    def pool_floor_pages(max_bucket_size: int, min_pages_per_sequence: int = 4) -> int:
        """Fail-loud floor: the pool must at least hold one full decode bucket.

        Every active sequence needs pages; ``min_pages_per_sequence`` = 4 at
        page size 64 gives each sequence 256 tokens of headroom — a
        conservative minimum, documented as tunable. A computed pool below
        this is a configuration error, never a silently accepted remainder.
        """
        return max(1, int(max_bucket_size)) * max(1, int(min_pages_per_sequence))

    # ------------------------------------------------------------------
    # Host-KV watermark trigger (Phase 5.5)
    # ------------------------------------------------------------------
    @staticmethod
    def plan_watermark_trigger(req: WatermarkTriggerRequest) -> WatermarkTriggerPlan:
        """Decide whether to preempt decode for prefill based on host-KV free %.

        Trigger when ANY node's free space exceeds the watermark (host KV is
        underutilized) AND there is work waiting (queued or, if eviction is
        enabled, evicted sequences). Also aggregates the global page stats the
        worker logs on rank 0.

        Pure: reads only the gathered ``node_stats`` + scalar inputs. The
        cross-rank ``dist.all_gather_object`` stays on the worker.
        """
        if not req.node_stats:
            return WatermarkTriggerPlan(
                should_trigger=False,
                max_free_percent=None,
                global_stats=None,
            )

        max_free_percent = max(int(s["free_percent"]) for s in req.node_stats)
        above_watermark = max_free_percent > req.host_kv_watermark
        should_trigger = above_watermark and (req.has_queued or req.has_evicted)

        total_used_pages = sum(int(s["num_used_pages"]) for s in req.node_stats)
        total_pages = sum(int(s["num_total_pages"]) for s in req.node_stats)
        global_used_percent = (
            int((total_used_pages / total_pages) * 100) if total_pages > 0 else 0
        )
        global_stats = WatermarkGlobalStats(
            used=total_used_pages,
            total=total_pages,
            free_percent=100 - global_used_percent,
            num_nodes=len(req.node_stats),
        )

        return WatermarkTriggerPlan(
            should_trigger=should_trigger,
            max_free_percent=max_free_percent,
            global_stats=global_stats,
        )

    # ------------------------------------------------------------------
    # Host-KV migration planning (Phase 5.6)
    # ------------------------------------------------------------------
    @staticmethod
    def plan_kv_migration(req: MigrationPlanRequest) -> List["MigrationOp"]:
        """Greedily rebalance host-KV pages across nodes.

        Target = total used pages / num nodes. Repeatedly take the smallest
        (by ``kv_token_budget``, then ``global_idx``) PREFILLED / ON_HOLD
        sequence off the most-overloaded node and assign it — round-robin
        across the destination node's ranks — to the least-loaded node that
        still has room, until every node is at or below target.

        Pure: reads only the gathered ``node_stats`` + the candidate
        snapshots. The cross-rank gather, the candidate enumeration over
        ``global_batch``, and the NCCL migration execution stay on the
        worker. Deterministic across ranks (candidates sorted by
        ``global_idx``; ties broken by node id), so every rank produces the
        identical plan without communication.
        """
        from batchgen.migration import MigrationOp

        node_stats = req.node_stats
        if len(node_stats) <= 1:
            return []

        total_used = sum(s["num_used_pages"] for s in node_stats.values())
        num_nodes = len(node_stats)
        target_per_node = total_used // num_nodes

        overloaded = [
            (nid, s) for nid, s in node_stats.items()
            if s["num_used_pages"] > target_per_node
        ]
        underutilized = [
            (nid, s) for nid, s in node_stats.items()
            if s["num_used_pages"] < target_per_node
        ]
        if not overloaded or not underutilized:
            return []

        overloaded.sort(key=lambda x: x[1]["num_used_pages"], reverse=True)
        underutilized.sort(key=lambda x: x[1]["num_used_pages"])

        # Bucket candidates by source node (assigned_rank // gpus_per_node).
        cand_by_node: dict = {}
        for c in req.candidates:
            nid = c.assigned_rank // req.num_gpus_per_node
            cand_by_node.setdefault(nid, []).append(c)

        migrations: list = []
        used_by_node = {nid: s["num_used_pages"] for nid, s in node_stats.items()}
        migrated_uuids: set = set()
        dest_rank_counter: dict = {}

        for src_node_id, _ in overloaded:
            while used_by_node[src_node_id] > target_per_node and underutilized:
                candidate_sequences = [
                    c for c in cand_by_node.get(src_node_id, [])
                    if c.uuid not in migrated_uuids
                ]
                if not candidate_sequences:
                    break

                # Deterministic selection: smallest budget, global_idx tie-break.
                candidate_sequences.sort(key=lambda c: c.global_idx)
                cand = min(
                    candidate_sequences,
                    key=lambda c: (c.kv_token_budget, c.global_idx),
                )

                # Use actual host pages allocated (chunked growth means this
                # is < ceil(kv_token_budget / page_size)).
                pages_needed = cand.host_pages_allocated
                if pages_needed <= 0:
                    migrated_uuids.add(cand.uuid)  # don't retry
                    continue

                # Dest node = most free space (lowest used), node id tie-break.
                dest_node_id = min(
                    underutilized, key=lambda x: (used_by_node[x[0]], x[0])
                )[0]

                dest_total = node_stats[dest_node_id]["num_total_pages"]
                dest_free = dest_total - used_by_node[dest_node_id]
                if pages_needed > dest_free:
                    underutilized = [
                        (nid, s) for nid, s in underutilized if nid != dest_node_id
                    ]
                    if not underutilized:
                        break
                    continue

                # Round-robin across dest node's ranks for load balancing.
                if dest_node_id not in dest_rank_counter:
                    dest_rank_counter[dest_node_id] = 0
                dest_rank_offset = dest_rank_counter[dest_node_id] % req.num_gpus_per_node
                dest_rank = dest_node_id * req.num_gpus_per_node + dest_rank_offset
                if dest_rank >= req.world_size:
                    dest_rank = dest_node_id * req.num_gpus_per_node  # fallback
                dest_rank_counter[dest_node_id] += 1

                migrations.append(MigrationOp(
                    uuid=cand.uuid,
                    from_rank=cand.assigned_rank,
                    to_rank=dest_rank,
                    pages=pages_needed,
                    host_pages=pages_needed,
                ))
                migrated_uuids.add(cand.uuid)

                used_by_node[src_node_id] -= pages_needed
                used_by_node[dest_node_id] += pages_needed

                if used_by_node[dest_node_id] >= target_per_node:
                    underutilized = [
                        (nid, s) for nid, s in underutilized if nid != dest_node_id
                    ]

        # Defensive dedup (mirrors legacy): keep first occurrence per uuid.
        migration_uuids = [m.uuid for m in migrations]
        if len(migration_uuids) != len(set(migration_uuids)):
            seen: set = set()
            unique: list = []
            for mig in migrations:
                if mig.uuid not in seen:
                    seen.add(mig.uuid)
                    unique.append(mig)
            migrations = unique

        return migrations
