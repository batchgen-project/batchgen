"""Continuous batching scheduler components for efficient GPU utilization.

This module provides data structures and utilities for page boundary scheduling
in continuous batching inference. The main scheduling logic remains in
BatchGenWorker due to deep coupling with worker state, but these structures
help organize the scheduling process.
"""

import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Debug flag for continuous batching logging
BATCHGEN_CB_DEBUG = os.environ.get("BATCHGEN_CB_LOG", "").upper() == "DEBUG"


@dataclass
class FastBoundaryTimingStats:
    """Detailed timing statistics for optimized page boundary operations.

    This dataclass tracks timing for each phase of the page boundary process,
    enabling performance analysis and bottleneck identification.
    """

    total_ms: float = 0.0

    # Phase 0: Async wait
    wait_kv_append_ms: float = 0.0
    num_kv_append_tasks: int = 0  # Track number of tasks waited
    wait_async_load_ms: float = 0.0
    finalize_load_ms: float = 0.0

    # Phase 0.5: Sync decode_uuids
    sync_decode_uuids_ms: float = 0.0

    # Phase 1: Gather
    gather_ms: float = 0.0

    # Phase 2: Process
    process_ms: float = 0.0
    extension_ms: float = 0.0

    # Phase 3: Async load launch
    load_select_ms: float = 0.0
    load_alloc_ms: float = 0.0
    load_launch_ms: float = 0.0

    # Phase 4: Rebuild + MoE buffer + barrier
    rebuild_ms: float = 0.0
    moe_buffer_update_ms: float = 0.0  # Time to sync and update MoE buffer size
    barrier_ms: float = 0.0

    # Counts
    num_completed: int = 0
    num_onhold: int = 0
    num_loaded: int = 0

    # Status counts
    total_active: int = 0
    total_prefilled: int = 0
    total_completed_cumulative: int = 0

    def summary(self) -> str:
        """Return a one-line summary of key timing metrics."""
        return (
            f"total={self.total_ms:.1f}ms, gather={self.gather_ms:.1f}ms, "
            f"process={self.process_ms:.1f}ms, barrier={self.barrier_ms:.1f}ms, "
            f"completed={self.num_completed}, onhold={self.num_onhold}, loaded={self.num_loaded}"
        )


@dataclass
class BoundaryDecisions:
    """Rank-0 computed decisions, broadcast to all ranks.

    Centralizes all batching decisions to rank 0 to prevent desync.
    All ranks receive identical decisions via broadcast_object_list.
    """
    completed_uuids: List[str]           # Sequences to mark COMPLETED
    active_uuids: List[str]              # Remaining decode sequences after completions (ordered)
    host_growth_uuids: List[str]         # Sequences needing host KV growth
    host_growth_pages: List[int]         # Pages to grow per sequence (parallel to host_growth_uuids)
    growth_feasible: bool                # Whether host growth is feasible
    host_evicted_uuids: List[str]        # Sequences to evict from host KV
    onhold_uuids: List[str]              # Sequences to put ON_HOLD (GPU eviction)
    seqs_needing_extension: List[str]    # Sequences needing GPU page extension
    new_load_uuids: List[str]            # Sequences to async-load into GPU
    decode_uuids_final: List[str]        # Final decode_uuids after all decisions
    scheduler_error: Optional[str] = None # Fatal scheduler invariant violation, raised after broadcast


def _format_ranked_reports(reports: Dict[str, List[int]], limit: int = 8) -> str:
    items = sorted(reports.items(), key=lambda item: item[0])[:limit]
    return ", ".join(f"{uuid}:{ranks}" for uuid, ranks in items)


def _reported_by_exact_group(ranks: List[int], group: Any, group_size: int) -> bool:
    """True iff ``ranks`` is EXACTLY the G contiguous ranks of decode group ``group``.

    Option 1 (G>1): a replicated sequence in group g must be reported by every rank
    in ``[g*G, (g+1)*G)`` and no other -- neither a missing group member nor a
    stray cross-group report is tolerated.
    """
    if not isinstance(group, int):
        return False
    expected = set(range(group * group_size, (group + 1) * group_size))
    return set(ranks) == expected


def validate_boundary_payload_alignment(
    decode_uuids: List[str],
    all_payloads: List[Optional[Dict[str, Any]]],
    group_size: int = 1,
) -> None:
    """Fail fast when boundary all-gather ownership metadata is inconsistent.

    ``group_size`` (G) is the decode attention TP size:

    * G==1 (validated pure-DP path): every UUID in the active decode list must be
      reported EXACTLY ONCE in ``seq_state`` by its ``assigned_rank``. Load
      candidates likewise once, and must not overlap active decode UUIDs.
    * G>1 (Option 1 unified resident TP): a sequence is replicated onto ALL G
      ranks of its ``decode_dp_group`` g, so it must be reported by EXACTLY the G
      contiguous ranks ``[g*G, (g+1)*G)`` (``set(ranks) == set(range(g*G,(g+1)*G))``)
      and ownership is GROUP MEMBERSHIP (``rank // G == g``), not
      ``assigned_rank == rank``.

    Violations mean scheduler state is already corrupt enough that silently
    completing or dropping rows would hide data loss and can lead to long-tail
    stalls.
    """
    G = int(group_size)
    errors: List[str] = []
    active_reports: Dict[str, List[int]] = {}
    candidate_reports: Dict[str, List[int]] = {}
    active_group: Dict[str, Any] = {}       # uuid -> decode_dp_group (G>1)
    candidate_group: Dict[str, Any] = {}
    active_wrong_owner: List[Tuple[str, int, Any]] = []
    candidate_wrong_owner: List[Tuple[str, int, Any]] = []
    candidate_bad_status: List[Tuple[str, int, Any]] = []

    for rank_idx, payload in enumerate(all_payloads):
        if not isinstance(payload, dict):
            errors.append(f"rank {rank_idx} payload missing or invalid: {type(payload).__name__}")
            continue

        seq_state = payload.get("seq_state") or {}
        candidate_state = payload.get("candidate_state") or {}

        for uuid, state in seq_state.items():
            active_reports.setdefault(uuid, []).append(rank_idx)
            if G > 1:
                group = state.get("decode_dp_group") if isinstance(state, dict) else None
                if group is not None:
                    active_group[uuid] = group
                # Owner check is group membership: rank // G == decode_dp_group.
                if not isinstance(group, int) or rank_idx // G != group:
                    active_wrong_owner.append((uuid, rank_idx, group))
            else:
                assigned_rank = state.get("assigned_rank") if isinstance(state, dict) else None
                if assigned_rank is not None:
                    try:
                        assigned_rank_int = int(assigned_rank)
                    except (TypeError, ValueError):
                        active_wrong_owner.append((uuid, rank_idx, assigned_rank))
                    else:
                        if assigned_rank_int != rank_idx:
                            active_wrong_owner.append((uuid, rank_idx, assigned_rank))

        for uuid, state in candidate_state.items():
            candidate_reports.setdefault(uuid, []).append(rank_idx)
            if G > 1:
                group = state.get("decode_dp_group") if isinstance(state, dict) else None
                if group is not None:
                    candidate_group[uuid] = group
                if not isinstance(group, int) or rank_idx // G != group:
                    candidate_wrong_owner.append((uuid, rank_idx, group))
            else:
                assigned_rank = state.get("assigned_rank") if isinstance(state, dict) else None
                if assigned_rank is not None:
                    try:
                        assigned_rank_int = int(assigned_rank)
                    except (TypeError, ValueError):
                        candidate_wrong_owner.append((uuid, rank_idx, assigned_rank))
                    else:
                        if assigned_rank_int != rank_idx:
                            candidate_wrong_owner.append((uuid, rank_idx, assigned_rank))
            status = state.get("status") if isinstance(state, dict) else None
            if status not in (None, "PREFILLED", "ON_HOLD"):
                candidate_bad_status.append((uuid, rank_idx, status))

    missing_active = [uuid for uuid in decode_uuids if uuid not in active_reports]
    if missing_active:
        errors.append(
            "decode UUIDs missing from gathered seq_state: "
            f"count={len(missing_active)} first={missing_active[:8]}"
        )

    if G > 1:
        # Each active UUID must be reported by EXACTLY its group's G contiguous ranks.
        bad_active_card = {
            uuid: ranks for uuid, ranks in active_reports.items()
            if not _reported_by_exact_group(ranks, active_group.get(uuid), G)
        }
        if bad_active_card:
            errors.append(
                "active UUIDs not reported by exactly their decode group's G ranks: "
                f"{_format_ranked_reports(bad_active_card)}"
            )
        bad_candidate_card = {
            uuid: ranks for uuid, ranks in candidate_reports.items()
            if not _reported_by_exact_group(ranks, candidate_group.get(uuid), G)
        }
        if bad_candidate_card:
            errors.append(
                "load candidates not reported by exactly their decode group's G ranks: "
                f"{_format_ranked_reports(bad_candidate_card)}"
            )
    else:
        duplicate_active = {
            uuid: ranks for uuid, ranks in active_reports.items() if len(ranks) != 1
        }
        if duplicate_active:
            errors.append(
                "active UUIDs reported by multiple ranks: "
                f"{_format_ranked_reports(duplicate_active)}"
            )

        duplicate_candidates = {
            uuid: ranks for uuid, ranks in candidate_reports.items() if len(ranks) != 1
        }
        if duplicate_candidates:
            errors.append(
                "load candidates reported by multiple ranks: "
                f"{_format_ranked_reports(duplicate_candidates)}"
            )

    active_candidate_overlap = [
        uuid for uuid in decode_uuids if uuid in candidate_reports
    ]
    if active_candidate_overlap:
        errors.append(
            "decode UUIDs also reported as load candidates: "
            f"count={len(active_candidate_overlap)} first={active_candidate_overlap[:8]}"
        )

    if active_wrong_owner:
        _key = "group" if G > 1 else "assigned_rank"
        errors.append(
            "active UUID owner/rank mismatch: "
            + ", ".join(
                f"{uuid}(reported_rank={rank}, {_key}={assigned})"
                for uuid, rank, assigned in active_wrong_owner[:8]
            )
        )

    if candidate_wrong_owner:
        _key = "group" if G > 1 else "assigned_rank"
        errors.append(
            "candidate UUID owner/rank mismatch: "
            + ", ".join(
                f"{uuid}(reported_rank={rank}, {_key}={assigned})"
                for uuid, rank, assigned in candidate_wrong_owner[:8]
            )
        )

    if candidate_bad_status:
        errors.append(
            "load candidates have invalid status: "
            + ", ".join(
                f"{uuid}(rank={rank}, status={status})"
                for uuid, rank, status in candidate_bad_status[:8]
            )
        )

    if errors:
        raise RuntimeError("[SCHED_INVARIANT] boundary gather misalignment: " + " | ".join(errors))


class LoadingStrategy(Enum):
    """Strategy for selecting which sequences to load from host."""

    LONGEST_FIRST = "longest_first"  # Prioritize longest-decoded (closer to completion)
    SHORTEST_FIRST = "shortest_first"  # Prioritize shortest-decoded (new sequences)
    FIFO = "fifo"  # First-in-first-out based on global_idx


class EvictionStrategy(Enum):
    """Strategy for selecting which sequences to evict to host."""

    SHORTEST_FIRST = "shortest_first"  # Evict shortest-decoded first (keep longer)
    LONGEST_FIRST = "longest_first"  # Evict longest-decoded first
    FIFO = "fifo"  # First-in-first-out


class AdaptiveChunkSizer:
    """Adapts host KV chunk size based on observed decode lengths (EMA).

    Tracks a running exponential moving average of completed sequence decode
    lengths and adjusts the chunk size to reduce waste (over-reservation) and
    eviction frequency (under-reservation).
    """

    def __init__(
        self,
        initial_chunk: int = 8192,
        min_chunk: int = 1024,
        max_chunk: int = 65536,
        ema_alpha: float = 0.1,
        multiplier: float = 1.5,
    ):
        self.current_chunk = initial_chunk
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.ema_alpha = ema_alpha
        self.multiplier = multiplier
        self.ema_decode_length: Optional[float] = None
        self.completed_count = 0

    def report_completion(self, decoded_length: int) -> None:
        """Called when a sequence completes. Updates EMA and chunk size."""
        if self.ema_decode_length is None:
            self.ema_decode_length = float(decoded_length)
        else:
            self.ema_decode_length = (
                self.ema_alpha * decoded_length
                + (1 - self.ema_alpha) * self.ema_decode_length
            )
        self.completed_count += 1
        # Only adapt after enough samples to have a reasonable estimate
        if self.completed_count >= 10:
            old_chunk = self.current_chunk
            target = self.ema_decode_length * self.multiplier
            self.current_chunk = int(
                max(self.min_chunk, min(self.max_chunk, target))
            )
            # Round up to page boundary (64 tokens per page)
            self.current_chunk = math.ceil(self.current_chunk / 64) * 64
            if self.current_chunk != old_chunk and BATCHGEN_CB_DEBUG:
                logger.debug(
                    f"[ADAPTIVE_CHUNK] {old_chunk} -> {self.current_chunk} "
                    f"ema={self.ema_decode_length:.0f} completed={self.completed_count}"
                )

    def get_chunk_size(self) -> int:
        """Return current chunk size in tokens."""
        return self.current_chunk


def select_sequences_for_eviction(
    sequences: List[Tuple[str, Dict[str, Any]]],
    pages_to_free: int,
    strategy: EvictionStrategy = EvictionStrategy.SHORTEST_FIRST,
    page_key: str = "gpu_pages_allocated",
) -> Tuple[List[str], int]:
    """Select sequences to evict to free pages (GPU or host).

    Uses deterministic sorting to ensure all ranks select the same sequences.

    Args:
        sequences: List of (uuid, state_dict) tuples for sequences on a rank
        pages_to_free: Minimum number of pages to free
        strategy: Eviction strategy to use
        page_key: Key in state_dict for page count (default: gpu_pages_allocated,
                  use 'host_pages_allocated' for host KV eviction)

    Returns:
        (list of uuids to evict, total pages freed)
    """
    if not sequences or pages_to_free <= 0:
        return [], 0

    # Sort based on strategy - CRITICAL: Use tie-breaker for determinism
    # Priority-aware: NORMAL (0) evicted before HIGH (1) within same strategy
    if strategy == EvictionStrategy.SHORTEST_FIRST:
        sorted_seqs = sorted(
            sequences, key=lambda x: (x[1].get("priority", 0), x[1].get("decoded_length", 0), x[0])
        )
    elif strategy == EvictionStrategy.LONGEST_FIRST:
        sorted_seqs = sorted(
            sequences, key=lambda x: (x[1].get("priority", 0), -x[1].get("decoded_length", 0), x[0])
        )
    else:  # FIFO
        sorted_seqs = sorted(
            sequences, key=lambda x: (x[1].get("priority", 0), x[1].get("global_idx", float("inf")), x[0])
        )

    evict_uuids = []
    freed = 0

    for uuid, state in sorted_seqs:
        if freed >= pages_to_free:
            break
        evict_uuids.append(uuid)
        freed += state.get(page_key, 0)

    return evict_uuids, freed


@dataclass
class HostKVGrowthEvictionPlan:
    """Deterministic host-KV eviction plan that preserves growth headroom."""

    evicted_uuids: List[str]
    freed_pages: int
    remaining_growth_uuids: List[str]
    remaining_growth_pages: List[int]
    remaining_growth_needed: int
    expected_free_pages: int
    required_free_pages: int
    growth_feasible_after_eviction: bool


def plan_host_kv_growth_evictions(
    *,
    active_uuids: List[str],
    completed_uuids: List[str],
    host_growth_uuids: List[str],
    host_growth_pages: List[int],
    eviction_candidates: List[Tuple[str, Dict[str, Any]]],
    free_pages: int,
    total_pages: int,
    completed_pages: int,
    watermark_percent: float,
    safety_margin: int,
    strategy: EvictionStrategy = EvictionStrategy.SHORTEST_FIRST,
    page_key: str = "host_pages_allocated",
) -> HostKVGrowthEvictionPlan:
    """Plan host evictions so remaining active rows can grow before appending.

    The dynamic host-KV invariant is: after a boundary, every still-active row
    that needs host growth must either have enough free pages to grow, or must
    have been removed from decode. This planner accounts for pages that will be
    released by completed rows before growth, then selects deterministic
    eviction candidates until both the free-page watermark and the remaining
    growth debt plus reserve are satisfied.
    """
    if len(host_growth_uuids) != len(host_growth_pages):
        raise ValueError("host_growth_uuids and host_growth_pages must align")

    completed_set = set(completed_uuids)
    base_free = min(max(total_pages, 0), max(free_pages, 0) + max(completed_pages, 0))
    target_free = int(max(total_pages, 0) * max(watermark_percent, 0.0) / 100.0)
    growth_by_uuid = {
        uuid: pages
        for uuid, pages in zip(host_growth_uuids, host_growth_pages)
        if pages > 0 and uuid in active_uuids and uuid not in completed_set
    }
    remaining_growth = sum(growth_by_uuid.values())

    if strategy == EvictionStrategy.SHORTEST_FIRST:
        sorted_candidates = sorted(
            eviction_candidates,
            key=lambda x: (
                x[1].get("priority", 0),
                x[1].get("decoded_length", 0),
                x[1].get("global_idx", float("inf")),
                x[0],
            ),
        )
    elif strategy == EvictionStrategy.LONGEST_FIRST:
        sorted_candidates = sorted(
            eviction_candidates,
            key=lambda x: (
                x[1].get("priority", 0),
                -x[1].get("decoded_length", 0),
                x[1].get("global_idx", float("inf")),
                x[0],
            ),
        )
    else:
        sorted_candidates = sorted(
            eviction_candidates,
            key=lambda x: (x[1].get("priority", 0), x[1].get("global_idx", float("inf")), x[0]),
        )

    evicted_uuids: List[str] = []
    evicted_set: Set[str] = set()
    freed_pages = 0
    candidate_idx = 0

    while True:
        expected_free = min(max(total_pages, 0), base_free + freed_pages)
        growth_required_free = remaining_growth + max(safety_margin, 0) if remaining_growth > 0 else 0
        required_free = max(target_free, growth_required_free)
        if expected_free >= required_free:
            break

        while candidate_idx < len(sorted_candidates):
            uuid, state = sorted_candidates[candidate_idx]
            candidate_idx += 1
            if uuid in completed_set or uuid in evicted_set:
                continue
            pages = int(state.get(page_key, 0) or 0)
            if pages <= 0:
                continue
            break
        else:
            break

        evicted_uuids.append(uuid)
        evicted_set.add(uuid)
        freed_pages += pages
        remaining_growth -= growth_by_uuid.pop(uuid, 0)

    remaining_growth_uuids = [uuid for uuid in host_growth_uuids if uuid in growth_by_uuid]
    remaining_growth_pages = [growth_by_uuid[uuid] for uuid in remaining_growth_uuids]
    expected_free = min(max(total_pages, 0), base_free + freed_pages)
    required_free = max(
        target_free,
        remaining_growth + max(safety_margin, 0) if remaining_growth > 0 else 0,
    )
    growth_feasible = remaining_growth == 0 or expected_free >= remaining_growth + max(safety_margin, 0)

    return HostKVGrowthEvictionPlan(
        evicted_uuids=evicted_uuids,
        freed_pages=freed_pages,
        remaining_growth_uuids=remaining_growth_uuids,
        remaining_growth_pages=remaining_growth_pages,
        remaining_growth_needed=remaining_growth,
        expected_free_pages=expected_free,
        required_free_pages=required_free,
        growth_feasible_after_eviction=growth_feasible,
    )


def select_sequences_for_loading(
    candidates: Dict[str, Dict[str, Any]],
    per_rank_free_pages: List[int],
    exclude_uuids: Set[str],
    strategy: LoadingStrategy = LoadingStrategy.LONGEST_FIRST,
    get_global_idx_fn: Optional[callable] = None,
    group_size: int = 1,
) -> Tuple[List[str], Dict[int, int]]:
    """Select sequences to load from host to GPU.

    Uses deterministic sorting to ensure all ranks select the same sequences.

    Args:
        candidates: Dict of uuid -> candidate_info dict
        per_rank_free_pages: Free pages available on each rank
        exclude_uuids: UUIDs to exclude (completed, just evicted, etc.)
        strategy: Loading strategy to use
        get_global_idx_fn: Function to get global_idx for a uuid (for tie-breaking)

    Returns:
        (list of uuids to load, dict of capacity-group -> pages used)

    ``group_size`` is the decode attention TP size.  With ``group_size > 1``
    a sequence is replicated on the contiguous ranks named by
    ``decode_dp_group``; it therefore consumes one page bucket per group, with
    capacity equal to the tightest physical rank in that group.  The default
    keeps the validated pure-DP behavior unchanged.
    """
    if not candidates:
        return [], {}

    # Filter out excluded sequences
    valid_candidates = [
        (uuid, info) for uuid, info in candidates.items() if uuid not in exclude_uuids
    ]

    if not valid_candidates:
        return [], {}

    if group_size <= 0 or len(per_rank_free_pages) % group_size != 0:
        raise ValueError(
            f"group_size={group_size} must divide the number of ranks="
            f"{len(per_rank_free_pages)}"
        )

    num_capacity_groups = len(per_rank_free_pages) // group_size
    if group_size == 1:
        capacity_free_pages = list(per_rank_free_pages)
    else:
        capacity_free_pages = [
            min(per_rank_free_pages[g * group_size:(g + 1) * group_size])
            for g in range(num_capacity_groups)
        ]

    # Sort based on strategy - CRITICAL: Use tie-breaker for determinism
    def sort_key(item):
        uuid, info = item
        decoded_len = info.get("decoded_length", 0)
        global_idx = (
            get_global_idx_fn(uuid) if get_global_idx_fn else float("inf")
        )

        if strategy == LoadingStrategy.LONGEST_FIRST:
            return (-decoded_len, global_idx)  # Descending by decoded_length
        elif strategy == LoadingStrategy.SHORTEST_FIRST:
            return (decoded_len, global_idx)  # Ascending by decoded_length
        else:  # FIFO
            return (global_idx, uuid)

    sorted_candidates = sorted(valid_candidates, key=sort_key)

    load_uuids = []
    rank_pages_used: Dict[int, int] = {
        r: 0 for r in range(num_capacity_groups)
    }

    for uuid, info in sorted_candidates:
        req_pages = info.get("pages_needed", 0)
        if group_size > 1:
            capacity_group = info.get("decode_dp_group")
            if not isinstance(capacity_group, int):
                raise ValueError(
                    f"candidate {uuid} has no decode_dp_group for "
                    f"group_size={group_size}"
                )
        else:
            capacity_group = info.get("assigned_rank", 0)

        if req_pages == 0:
            continue

        if capacity_group < 0 or capacity_group >= num_capacity_groups:
            logger.warning(
                f"Invalid capacity group {capacity_group} for {uuid}"
            )
            continue

        if (
            rank_pages_used[capacity_group] + req_pages
            <= capacity_free_pages[capacity_group]
        ):
            load_uuids.append(uuid)
            rank_pages_used[capacity_group] += req_pages

    return load_uuids, rank_pages_used


def check_extension_feasibility(
    sequences_needing_extension: List[Tuple[str, Dict[str, Any]]],
    per_rank_free_pages: List[int],
    world_size: int,
) -> Tuple[bool, Dict[int, int]]:
    """Check if all sequences can extend their GPU page allocation.

    Args:
        sequences_needing_extension: List of (uuid, state_dict) for sequences needing more pages
        per_rank_free_pages: Free pages available on each rank
        world_size: Number of ranks

    Returns:
        (all_can_extend, dict of rank -> pages needed)
    """
    total_by_rank: Dict[int, int] = {r: 0 for r in range(world_size)}

    for uuid, state in sequences_needing_extension:
        assigned_rank = state.get("assigned_rank", 0)
        pages_needed = state.get("additional_pages_needed", 0)

        if assigned_rank < world_size:
            total_by_rank[assigned_rank] += pages_needed

    all_can_extend = all(
        total_by_rank.get(r, 0) <= per_rank_free_pages[r]
        for r in range(world_size)
    )

    return all_can_extend, total_by_rank


def validate_decode_uuid_sync(
    local_decode_set: Set[str],
    all_decode_sets: List[Set[str]],
    rank: int,
) -> Tuple[bool, Set[str]]:
    """Validate that decode_uuids are synchronized across ranks.

    Args:
        local_decode_set: This rank's decode_uuids as a set
        all_decode_sets: Gathered decode_uuids from all ranks
        rank: This rank's ID for logging

    Returns:
        (all_synced, union_set if not synced)
    """
    all_sets_equal = all(
        s == local_decode_set for s in all_decode_sets if s is not None
    )

    if all_sets_equal:
        return True, local_decode_set

    # Log desync details
    for r, s in enumerate(all_decode_sets):
        if s != local_decode_set:
            diff_in_r = s - local_decode_set if s else set()
            diff_in_local = local_decode_set - s if s else local_decode_set
            logger.error(
                f"Rank {rank}: decode_uuids DESYNC detected! "
                f"Rank {r} has {len(diff_in_r)} extra, "
                f"Rank {rank} has {len(diff_in_local)} extra"
            )

    # Return union to ensure all sequences are processed
    global_decode_set: Set[str] = set()
    for s in all_decode_sets:
        if s:
            global_decode_set.update(s)

    return False, global_decode_set
