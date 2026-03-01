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
    if strategy == EvictionStrategy.SHORTEST_FIRST:
        # Shortest decoded_length first (ascending)
        sorted_seqs = sorted(
            sequences, key=lambda x: (x[1].get("decoded_length", 0), x[0])
        )
    elif strategy == EvictionStrategy.LONGEST_FIRST:
        # Longest decoded_length first (descending)
        sorted_seqs = sorted(
            sequences, key=lambda x: (-x[1].get("decoded_length", 0), x[0])
        )
    else:  # FIFO
        # By global_idx ascending
        sorted_seqs = sorted(
            sequences, key=lambda x: (x[1].get("global_idx", float("inf")), x[0])
        )

    evict_uuids = []
    freed = 0

    for uuid, state in sorted_seqs:
        if freed >= pages_to_free:
            break
        evict_uuids.append(uuid)
        freed += state.get(page_key, 0)

    return evict_uuids, freed


def select_sequences_for_loading(
    candidates: Dict[str, Dict[str, Any]],
    per_rank_free_pages: List[int],
    exclude_uuids: Set[str],
    strategy: LoadingStrategy = LoadingStrategy.LONGEST_FIRST,
    get_global_idx_fn: Optional[callable] = None,
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
        (list of uuids to load, dict of rank -> pages used)
    """
    if not candidates:
        return [], {}

    # Filter out excluded sequences
    valid_candidates = [
        (uuid, info) for uuid, info in candidates.items() if uuid not in exclude_uuids
    ]

    if not valid_candidates:
        return [], {}

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
    rank_pages_used: Dict[int, int] = {r: 0 for r in range(len(per_rank_free_pages))}

    for uuid, info in sorted_candidates:
        req_pages = info.get("pages_needed", 0)
        assigned_rank = info.get("assigned_rank", 0)

        if req_pages == 0:
            continue

        if assigned_rank >= len(per_rank_free_pages):
            logger.warning(f"Invalid assigned_rank {assigned_rank} for {uuid}")
            continue

        if rank_pages_used[assigned_rank] + req_pages <= per_rank_free_pages[assigned_rank]:
            load_uuids.append(uuid)
            rank_pages_used[assigned_rank] += req_pages

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
