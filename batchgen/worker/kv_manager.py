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

The KV Cache Helper section has 27 methods totaling ~1330 LOC; they
span four distinct concerns (read-only stats, allocation, planning,
migration execution). Porting them all in one slice would be too risky.
Allocators, planners, and migration executors land in later sub-slices
(5.3+).

Design follows the per-slice Backend Protocol pattern introduced by
``SyncCoordinator`` (Slice 3): the handler takes a ``KVStatsBackend``
that the worker wires to its real KV managers; tests wire a fake.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Tuple

if TYPE_CHECKING:
    from batchgen.sequence import SequenceBatch


logger = logging.getLogger(__name__)


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

    Token / slot capacity are pure max-of-candidates computations; the
    worker passes the union of relevant config sources through this
    snapshot. Optional fields are ``None`` when the underlying worker
    attribute is unset (legacy uses ``getattr(self, "...", 0)`` or
    similar; we normalize to typed ``Optional`` here).
    """

    sequence_tokens: Tuple[int, ...]
    max_input_length: int
    max_decoding_length: int
    engine_max_prompt: Optional[int]
    engine_max_decode: Optional[int]
    engine_module_global_batch_size: Optional[int]
    engine_module_attn_decoding_micro_batch_size: Optional[int]
    engine_basic_num_queries: Optional[int]
    model_max_position_embeddings: Optional[int]
    args_cuda_graph_max_bucket_size: Optional[int]


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
        """Maximum token capacity any CUDA-graph page table must cover.

        Takes the max of:
          * a 16384 floor,
          * any explicit ``sequence_tokens`` from the call site,
          * ``max_input_length + max_decoding_length`` from the worker,
          * engine-config-derived prompt/decode budgets,
          * the model's ``max_position_embeddings``.
        """
        candidates: list = [16384]
        candidates.extend(int(t) for t in req.sequence_tokens if int(t) > 0)
        if req.max_input_length > 0:
            candidates.append(
                req.max_input_length + max(0, req.max_decoding_length)
            )
        emp = req.engine_max_prompt
        emd = req.engine_max_decode
        if emp is not None and emd is not None:
            candidates.append(int(emp) + int(emd))
        elif emp is not None:
            candidates.append(int(emp))
        elif emd is not None:
            candidates.append(int(emd))
        mmp = req.model_max_position_embeddings
        if mmp is not None and int(mmp) > 0:
            candidates.append(int(mmp))
        return max(candidates)

    @staticmethod
    def page_table_slot_capacity(req: PageTableCapacityRequest) -> int:
        """Maximum number of slots a CUDA-graph page table must address.

        Takes the max of (``args.cuda_graph_max_bucket_size`` if set,
        ``engine.global_batch_size``, ``engine.attn_decoding_micro_batch_size``,
        ``engine.num_queries``). Falls back to ``1`` when none are set.
        """
        candidates: list = []
        v = req.args_cuda_graph_max_bucket_size
        if v is not None and int(v) > 0:
            candidates.append(int(v))
        for v in (
            req.engine_module_global_batch_size,
            req.engine_module_attn_decoding_micro_batch_size,
            req.engine_basic_num_queries,
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
        page_capacity = max(
            1,
            min(
                int(config.num_pages),
                math.ceil(token_capacity / int(config.page_size_tokens)),
            ),
        )
        slot_capacity = max(
            1,
            min(int(config.num_pages), KVCacheManager.page_table_slot_capacity(req)),
        )
        return replace(
            config,
            cuda_graph_max_pages_per_sequence=page_capacity,
            cuda_graph_max_slots=slot_capacity,
        )
