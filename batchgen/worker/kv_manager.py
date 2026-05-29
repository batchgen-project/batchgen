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

The KV Cache Helper section has 27 methods totaling ~1330 LOC; they
span four distinct concerns (read-only stats, allocation, planning,
migration execution). Porting them all in one slice would be too risky.
Watermark/eviction planners and migration executors land in later
sub-slices (5.5+).

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
    from batchgen.query_book import QueryBookEntry
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
# Token-budget cache (Phase 5.3): frozen request snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenBudgetRequest:
    """Frozen snapshot for the per-sequence host-KV token-budget cache.

    The handler memoizes the computed budget on
    ``query_book[sequence_id].kv_token_budget``. That mutation is on the
    entry object the worker passes in; worker state is not touched.
    """

    query_book: Mapping[int, "QueryBookEntry"]
    local_to_uuid: Mapping[int, str]
    global_batch: "SequenceBatch"
    max_decoding_length: int


# ---------------------------------------------------------------------------
# GPU-KV-manager allocation planning (Phase 5.4a): frozen request + plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuKvManagerRequest:
    """Frozen snapshot for ``plan_gpu_kv_manager``.

    ``current_num_pages`` is the page count of the manager the worker
    currently holds (0 when none is bound). ``capacity`` carries the
    page-table capacity inputs so the plan can populate the CUDA-graph
    fields on the configs it returns.
    """

    model_name: str
    sequence_tokens: Tuple[int, ...]
    has_manager: bool
    current_num_pages: int
    capacity: PageTableCapacityRequest


@dataclass(frozen=True)
class GpuKvManagerPlan:
    """What ``plan_gpu_kv_manager`` returns; the worker applies it.

    * ``reuse`` — the existing manager already has enough pages; the
      worker just re-initializes and binds it (no configs needed).
    * ``destroy_existing`` — an existing manager must be torn down before
      the new one is created (only set when ``reuse`` is False).
    * ``primary_config`` / ``aux_config`` — sized configs for the new
      managers. ``aux_config`` is non-None only for DSA models. Both are
      ``None`` when ``reuse`` is True.
    """

    reuse: bool
    destroy_existing: bool
    primary_config: Optional["GPUPagedKVConfig"]
    aux_config: Optional["GPUPagedKVConfig"]


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

    # ------------------------------------------------------------------
    # Token-budget cache (Phase 5.3)
    # ------------------------------------------------------------------
    @staticmethod
    def get_sequence_token_budget(req: TokenBudgetRequest, sequence_id: int) -> int:
        """Return cached host-allocation tokens for a sequence, computing once.

        Reads ``query_book[sequence_id]``; if ``kv_token_budget`` is set,
        returns it. Otherwise computes ``prompt_length + max_decoding_length``
        from sequence metadata and memoizes it on the entry.

        Raises ``RuntimeError`` if ``query_book`` is empty (means the worker
        has not initialized it yet) or ``KeyError`` if the sequence is missing.
        """
        if not req.query_book:
            raise RuntimeError("query_book is not initialized before KV allocation")
        query_entry = req.query_book.get(sequence_id)
        if query_entry is None or query_entry.encoded is None:
            raise KeyError(f"Missing query entry for sequence {sequence_id}")
        if query_entry.kv_token_budget is not None:
            return query_entry.kv_token_budget
        # Fallback: compute from sequence metadata (attention_mask removed)
        uuid = req.local_to_uuid.get(sequence_id, "")
        seq = req.global_batch.get_sequence(uuid) if uuid else None
        if seq is None:
            raise KeyError(f"No sequence metadata available for sequence {sequence_id}")
        # NO truncation: KV budget must cover the FULL prompt + decode budget.
        # An earlier min(...) here silently undersized KV when max_input_length
        # lagged behind the actual prompt length on multi-batch admits.
        input_tokens = seq.prompt_length
        total_tokens = input_tokens + req.max_decoding_length
        query_entry.kv_token_budget = total_tokens
        return total_tokens

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
    @staticmethod
    def plan_gpu_kv_manager(req: GpuKvManagerRequest) -> GpuKvManagerPlan:
        """Decide whether to reuse or recreate the GPU paged KV manager.

        Builds the primary (and, for DSA models, auxiliary) ``GPUPagedKVConfig``
        sized for ``req.sequence_tokens`` with the CUDA-graph page-table
        capacity applied. If the worker already holds a manager with at least
        ``primary_config.num_pages`` pages, the plan says to reuse it; otherwise
        the worker must (destroy the old manager and) create new ones.

        Pure: calls the pure ``build_gpu_kv_config`` / ``build_gpu_kv_config_aux``
        config builders and ``apply_page_table_capacity``. No GPU allocation, no
        worker state — the worker applies the returned plan.
        """
        # Local import: the config builders pull torch + the model registry,
        # which are heavy at module-load time and could cycle back here.
        from batchgen.kv_cache.host_kv_mananger_config import (
            build_gpu_kv_config,
            build_gpu_kv_config_aux,
        )

        primary_config = build_gpu_kv_config(
            model_name=req.model_name,
            sequence_tokens=req.sequence_tokens,
        )
        primary_config = KVCacheManager.apply_page_table_capacity(
            req.capacity, primary_config
        )
        required_pages = primary_config.num_pages

        if req.has_manager and req.current_num_pages >= required_pages:
            return GpuKvManagerPlan(
                reuse=True,
                destroy_existing=False,
                primary_config=None,
                aux_config=None,
            )

        aux_config = build_gpu_kv_config_aux(
            model_name=req.model_name,
            sequence_tokens=req.sequence_tokens,
        )
        if aux_config is not None:
            aux_config = KVCacheManager.apply_page_table_capacity(
                req.capacity, aux_config
            )

        return GpuKvManagerPlan(
            reuse=False,
            destroy_existing=req.has_manager,
            primary_config=primary_config,
            aux_config=aux_config,
        )
