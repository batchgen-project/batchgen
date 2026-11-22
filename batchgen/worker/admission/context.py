"""Frozen state bundles passed across the decode admission seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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


@dataclass(frozen=True)
class SchedulingState:
    pool: Any
    seq_table: Any
    trajectories: Any


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
