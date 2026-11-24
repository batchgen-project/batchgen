"""Frozen state bundles passed across the decode admission seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class RankFacts:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    rank_ctx: Optional[Any]
    attn_tp_size: int
    device: Any
    page_size: int
    decision_interval: int
    enable_prefix_cache: bool
    enable_host_kv_eviction: bool
    host_kv_chunk_size: int
    prefix_cache_debug_stats: bool = False
    kv_stats_native: bool = False
    kv_stats_compare: bool = False
    shared_prefix_admission: bool = True


@dataclass(frozen=True)
class SchedulingState:
    pool: Any
    seq_table: Any
    trajectories: Any
    prefix: Any = None


@dataclass(frozen=True)
class DecodePhaseContext:
    facts: RankFacts
    sched: SchedulingState
    gpu_kv: Optional[Any]
    host_view: Optional[Any]
    core_engine: Optional[Any]
    gpu_resident: Optional[Any]
    prefix_coordinator: Optional[Any]
    prefix_runtime: Optional[Any]
    chunk_sizer: Optional[Any]
    parallel_manager: Optional[Any] = None
    host_view_aux: Optional[Any] = None


@dataclass(frozen=True)
class WaitOutcome:
    num_tasks: int
    new_deferred_errors: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class MoeCountsOutcome:
    local_count_tensor: Any
    all_rank_counts: Any
    current_local_batch_size: int
    max_batch_size: int
    rank_token_counts: Any
