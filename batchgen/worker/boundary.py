"""Page-boundary scheduling — the rank-0 boundary decision.

Slice 8 of the worker decouple initiative (issue #175). Extracts
``_compute_boundary_decisions`` — the single rank-0-only function that,
at each decode page boundary, decides for the whole cluster: which
sequences completed, which need host-KV growth, which to evict from host
KV, which to put ON_HOLD (GPU eviction), which need GPU page extension,
and which to async-load. The result (``BoundaryDecisions``) is broadcast
to all ranks, which then execute their local portion.

The decision is already centralized to rank 0 with explicit dict inputs,
so it ports cleanly to a pure handler. The only worker couplings replaced
here:

  - ``self.global_batch.get_sequence(uuid).{global_idx,priority}`` → the
    ``seq_meta`` snapshot map (rank 0's global_batch has every sequence's
    static ``global_idx``; this avoids enlarging the broadcast payload).
  - ``self._get_node_for_rank(rank)`` → ``rank // num_gpus_per_node``
    (equivalent: the single-node case collapses to 0).
  - ``self.{world_size, enable_host_kv_eviction, host_kv_eviction_watermark}``
    → request fields.

The actual per-node growth/eviction math and the loading selection are
the pre-existing pure helpers ``plan_host_kv_growth_evictions`` and
``select_sequences_for_loading`` from ``batchgen.continuous_batching``.
The NCCL gather/broadcast around this decision and the execution of the
returned plan stay on the worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from batchgen.continuous_batching import (
    BoundaryDecisions,
    EvictionStrategy,
    LoadingStrategy,
    plan_host_kv_growth_evictions,
    select_sequences_for_loading,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundarySeqMeta:
    """Per-sequence metadata the decision reads from ``global_batch``.

    Sourced on rank 0 (whose ``global_batch`` holds every sequence). Carries
    only the static/synced fields the boundary decision needs beyond what
    ``global_seq_state`` already provides.
    """

    global_idx: int
    priority: int
    current_context_length: int
    host_token_capacity: int
    host_pages_allocated: int


@dataclass(frozen=True)
class BoundaryDecisionRequest:
    """Frozen snapshot for ``compute_decisions`` (rank-0 boundary decision)."""

    decode_uuids: Tuple[str, ...]
    global_seq_state: Mapping[str, dict]
    global_candidate_info: Mapping[str, dict]
    per_rank_free: Tuple[int, ...]
    chunk_size: int
    per_node_host_stats: Optional[Tuple[dict, ...]]
    seq_meta: Mapping[str, BoundarySeqMeta]
    world_size: int
    num_gpus_per_node: int
    enable_host_kv_eviction: bool
    host_kv_eviction_watermark: int
    # Decode attention TP size.  G==1 is pure DP; G>1 makes each sequence
    # resident on the contiguous G ranks of ``decode_dp_group``.
    attn_tp_size: int = 1


class BoundaryHandler:
    """Rank-0 page-boundary decision — pure, deterministic across ranks."""

    @staticmethod
    def compute_decisions(req: BoundaryDecisionRequest) -> BoundaryDecisions:
        """Compute ALL batching decisions for one page boundary.

        Pure: reads only the snapshot (state dicts + per-seq meta + scalars).
        The cross-rank gather that builds the dicts, the broadcast of the
        result, and the execution of the plan stay on the worker.
        """
        decode_uuids = req.decode_uuids
        global_seq_state = req.global_seq_state
        global_candidate_info = req.global_candidate_info
        per_rank_free = req.per_rank_free
        per_node_host_stats = req.per_node_host_stats
        seq_meta = req.seq_meta
        world_size = req.world_size
        gpn = req.num_gpus_per_node
        enable_host_kv_eviction = req.enable_host_kv_eviction
        host_kv_eviction_watermark = req.host_kv_eviction_watermark
        group_size = int(req.attn_tp_size)
        if group_size <= 0 or world_size % group_size != 0:
            raise ValueError(
                f"attn_tp_size={group_size} must divide world_size={world_size}"
            )

        def capacity_group(state: Mapping[str, object], uuid: str) -> int:
            """Return the physical GPU capacity bucket for one sequence."""
            if group_size > 1:
                group = state.get('decode_dp_group')
                if not isinstance(group, int):
                    raise ValueError(
                        f"sequence {uuid} has no decode_dp_group for "
                        f"attn_tp_size={group_size}"
                    )
                return group
            rank = state.get('assigned_rank')
            if not isinstance(rank, int):
                raise ValueError(f"sequence {uuid} has no assigned_rank")
            return rank

        num_capacity_groups = world_size // group_size
        capacity_free_pages = [
            min(per_rank_free[g * group_size:(g + 1) * group_size])
            for g in range(num_capacity_groups)
        ]

        def node_for_rank(rank: int) -> int:
            return rank // gpn

        def meta_global_idx(uuid: str):
            m = seq_meta.get(uuid)
            return m.global_idx if m is not None else float("inf")

        def meta_priority(uuid: str) -> int:
            m = seq_meta.get(uuid)
            return m.priority if m is not None else 0

        # Identify completed sequences
        completed_uuids = []
        active_uuids = []
        for uuid in decode_uuids:
            state = global_seq_state.get(uuid)
            if state and state['completed']:
                completed_uuids.append(uuid)
            else:
                active_uuids.append(uuid)

        # Host KV growth + eviction decisions. Growth and eviction must be
        # planned together: if growth is needed, watermark-only eviction is not
        # enough. The plan reserves enough free pages for remaining growth debt
        # after completed and evicted rows release their host pages.
        host_growth_uuids = []
        host_growth_pages_list = []
        for uuid in active_uuids:
            state = global_seq_state.get(uuid)
            if state and state.get('needs_host_growth'):
                growth_pages = state.get('host_growth_pages', 0)
                if growth_pages > 0:
                    host_growth_uuids.append(uuid)
                    host_growth_pages_list.append(growth_pages)

        host_evicted_uuids = []
        decode_after_eviction = list(active_uuids)
        growth_feasible = False
        scheduler_error = None
        per_node_growth_plans = {}
        growth_pages_by_uuid = dict(zip(host_growth_uuids, host_growth_pages_list))
        remaining_growth_by_uuid = dict(growth_pages_by_uuid)
        if per_node_host_stats:
            host_stats_by_node = {
                int(stats.get('node_id', idx)): stats
                for idx, stats in enumerate(per_node_host_stats)
            }
            completed_set = set(completed_uuids)
            active_nodes = {
                node_for_rank(global_seq_state[uuid]['assigned_rank'])
                for uuid in active_uuids
                if uuid in global_seq_state and global_seq_state[uuid].get('assigned_rank') is not None
            }
            completed_nodes = {
                node_for_rank(global_seq_state[uuid]['assigned_rank'])
                for uuid in completed_uuids
                if uuid in global_seq_state and global_seq_state[uuid].get('assigned_rank') is not None
            }
            for node in sorted(active_nodes | completed_nodes | set(host_stats_by_node.keys())):
                node_stats = host_stats_by_node.get(node)
                node_active_uuids = [
                    uuid for uuid in active_uuids
                    if uuid in global_seq_state
                    and global_seq_state[uuid].get('assigned_rank') is not None
                    and node_for_rank(global_seq_state[uuid]['assigned_rank']) == node
                ]
                node_completed_uuids = [
                    uuid for uuid in completed_uuids
                    if uuid in global_seq_state
                    and global_seq_state[uuid].get('assigned_rank') is not None
                    and node_for_rank(global_seq_state[uuid]['assigned_rank']) == node
                ]
                node_growth_uuids = [
                    uuid for uuid in host_growth_uuids
                    if uuid in global_seq_state
                    and global_seq_state[uuid].get('assigned_rank') is not None
                    and node_for_rank(global_seq_state[uuid]['assigned_rank']) == node
                ]
                if not node_active_uuids and not node_completed_uuids and not node_growth_uuids:
                    continue
                if node_stats is None or int(node_stats.get('num_total_pages', 0) or 0) <= 0:
                    if node_growth_uuids:
                        scheduler_error = (
                            f"[HOST_KV_GROWTH_PLAN] node {node} has growth requests but no host KV stats"
                        )
                        logger.error(scheduler_error)
                    continue

                total_pages = int(node_stats.get('num_total_pages', 0) or 0)
                free_pages = int(node_stats.get('num_free_pages', 0) or 0)
                safety_margin = int(total_pages * 0.05)
                completed_host_pages = sum(
                    int(global_seq_state.get(uuid, {}).get('host_pages_allocated', 0) or 0)
                    for uuid in node_completed_uuids
                )
                eviction_candidates = []
                if node_active_uuids and enable_host_kv_eviction:
                    for uuid in node_active_uuids:
                        state = global_seq_state.get(uuid)
                        if state and uuid not in completed_set:
                            eviction_candidates.append((uuid, {
                                'decoded_length': state['decoded_length'],
                                'host_pages_allocated': state.get('host_pages_allocated', 0),
                                'global_idx': meta_global_idx(uuid),
                                'priority': meta_priority(uuid),
                            }))

                growth_plan = plan_host_kv_growth_evictions(
                    active_uuids=node_active_uuids,
                    completed_uuids=node_completed_uuids,
                    host_growth_uuids=node_growth_uuids,
                    host_growth_pages=[growth_pages_by_uuid[uuid] for uuid in node_growth_uuids],
                    eviction_candidates=eviction_candidates,
                    free_pages=free_pages,
                    total_pages=total_pages,
                    completed_pages=completed_host_pages,
                    watermark_percent=host_kv_eviction_watermark if enable_host_kv_eviction else 0,
                    safety_margin=safety_margin,
                    strategy=EvictionStrategy.SHORTEST_FIRST,
                    page_key='host_pages_allocated',
                )
                host_evicted_uuids.extend(growth_plan.evicted_uuids)
                for uuid in growth_plan.evicted_uuids:
                    remaining_growth_by_uuid.pop(uuid, None)
                for uuid in node_growth_uuids:
                    if uuid not in growth_plan.remaining_growth_uuids:
                        remaining_growth_by_uuid.pop(uuid, None)

                if growth_plan.remaining_growth_needed > 0 or growth_plan.evicted_uuids:
                    growth_eviction_overlap = len(set(growth_plan.evicted_uuids) & set(node_growth_uuids))
                    logger.info(
                        f"[HOST_KV_GROWTH_PLAN] node={node} active={len(node_active_uuids)} "
                        f"growth_rows_total={len(node_growth_uuids)} "
                        f"growth_pages_total={sum(growth_pages_by_uuid[uuid] for uuid in node_growth_uuids)} "
                        f"growth_rows_remaining={len(growth_plan.remaining_growth_uuids)} "
                        f"growth_pages_remaining={growth_plan.remaining_growth_needed} "
                        f"free={free_pages} completed_pages={completed_host_pages} "
                        f"evict_rows={len(growth_plan.evicted_uuids)} evict_pages={growth_plan.freed_pages} "
                        f"growth_rows_evicted={growth_eviction_overlap} "
                        f"expected_free={growth_plan.expected_free_pages} "
                        f"required_free={growth_plan.required_free_pages} "
                        f"safety={safety_margin} feasible={growth_plan.growth_feasible_after_eviction}"
                    )
                    if node_growth_uuids and (growth_plan.evicted_uuids or not growth_plan.growth_feasible_after_eviction):
                        detail_rows = []
                        for uuid in node_growth_uuids:
                            state = global_seq_state.get(uuid, {})
                            m = seq_meta.get(uuid)
                            context_len = int(state.get('current_context_length', m.current_context_length if m else 0) or 0)
                            capacity = int(state.get('host_token_capacity', m.host_token_capacity if m else 0) or 0)
                            detail_rows.append((
                                capacity - context_len,
                                uuid,
                                m.global_idx if m else None,
                                state.get('assigned_rank'),
                                context_len,
                                capacity,
                                int(state.get('host_pages_allocated', m.host_pages_allocated if m else 0) or 0),
                                growth_pages_by_uuid.get(uuid, 0),
                            ))
                        detail_rows.sort(key=lambda x: (x[0], str(x[1])))
                        logger.warning(
                            f"[HOST_KV_GROWTH_PLAN_DETAIL] node={node} tightest_rows="
                            + "; ".join(
                                f"{uuid[:8]}(gid={gid},rank={rank},ctx={ctx},cap={cap},"
                                f"runway={runway},host_pages={host_pages},growth_pages={growth_pages})"
                                for runway, uuid, gid, rank, ctx, cap, host_pages, growth_pages in detail_rows[:8]
                            )
                        )
                per_node_growth_plans[node] = {
                    'expected_free_pages': growth_plan.expected_free_pages,
                    'safety_margin': safety_margin,
                    'num_candidates': len(eviction_candidates),
                }
        elif host_growth_uuids:
            scheduler_error = (
                "[HOST_KV_GROWTH_PLAN] host growth requested but per-node host KV stats are missing"
            )
            logger.error(scheduler_error)

        evicted_set = set(host_evicted_uuids)
        host_evicted_uuids = [uuid for uuid in active_uuids if uuid in evicted_set]
        decode_after_eviction = [u for u in active_uuids if u not in evicted_set]

        # GPU page extension / on-hold decisions
        seqs_needing_extension = []
        total_additional_by_rank = [0] * world_size

        for uuid in decode_after_eviction:
            state = global_seq_state.get(uuid)
            if state and state['additional_pages_needed'] > 0:
                capacity = capacity_group(state, uuid)
                total_additional_by_rank[capacity] += state['additional_pages_needed']
                seqs_needing_extension.append(uuid)

        all_can_extend = all(
            total_additional_by_rank[g] <= capacity_free_pages[g]
            for g in range(num_capacity_groups)
        )

        onhold_uuids = []
        actual_extension_by_rank = [0] * world_size

        if all_can_extend:
            actual_extension_by_rank = list(total_additional_by_rank)
        elif not all_can_extend:
            for g in range(num_capacity_groups):
                if total_additional_by_rank[g] > capacity_free_pages[g]:
                    rank_seqs = [
                        (uuid, global_seq_state[uuid])
                        for uuid in decode_after_eviction
                        if uuid in global_seq_state
                        and capacity_group(global_seq_state[uuid], uuid) == g
                    ]
                    # Priority-aware: NORMAL (0) evicted before HIGH (1)
                    rank_seqs.sort(
                        key=lambda x: (meta_priority(x[0]),
                                    x[1]['decoded_length'],
                                    meta_global_idx(x[0]))
                    )
                    pages_to_free = (
                        total_additional_by_rank[g] - capacity_free_pages[g]
                    )
                    freed = 0
                    for uuid, state in rank_seqs:
                        if freed >= pages_to_free:
                            break
                        onhold_uuids.append(uuid)
                        freed += state['gpu_pages_allocated']

            # Compute actual extension for remaining sequences
            onhold_set = set(onhold_uuids)
            for uuid in seqs_needing_extension:
                if uuid not in onhold_set:
                    state = global_seq_state.get(uuid, {})
                    g = capacity_group(state, uuid)
                    actual_extension_by_rank[g] += state.get('additional_pages_needed', 0)

        # Rows moved ON_HOLD are removed from decode before the next append, so
        # they no longer need immediate host growth at this boundary.
        onhold_set = set(onhold_uuids)
        for uuid in onhold_set:
            remaining_growth_by_uuid.pop(uuid, None)

        host_growth_uuids = [uuid for uuid in host_growth_uuids if uuid in remaining_growth_by_uuid]
        host_growth_pages_list = [remaining_growth_by_uuid[uuid] for uuid in host_growth_uuids]
        total_growth_needed = sum(host_growth_pages_list)
        if total_growth_needed > 0:
            remaining_growth_by_node = {}
            for uuid in host_growth_uuids:
                state = global_seq_state.get(uuid, {})
                assigned_rank = state.get('assigned_rank')
                if assigned_rank is None:
                    continue
                node = node_for_rank(assigned_rank)
                remaining_growth_by_node[node] = (
                    remaining_growth_by_node.get(node, 0)
                    + remaining_growth_by_uuid[uuid]
                )
            for node, node_growth_pages in sorted(remaining_growth_by_node.items()):
                plan_info = per_node_growth_plans.get(node)
                if plan_info is None:
                    scheduler_error = (
                        f"[HOST_KV_GROWTH_PLAN] node {node} has remaining growth but no host KV plan"
                    )
                    logger.error(scheduler_error)
                    break
                required_free = node_growth_pages + int(plan_info['safety_margin'])
                if int(plan_info['expected_free_pages']) < required_free:
                    scheduler_error = (
                        f"[HOST_KV_GROWTH_PLAN] node {node} infeasible after eviction/on-hold planning; "
                        f"growth_pages={node_growth_pages}, "
                        f"expected_free={plan_info['expected_free_pages']}, "
                        f"safety={plan_info['safety_margin']}, "
                        f"candidates={plan_info['num_candidates']}"
                    )
                    logger.error(scheduler_error)
                    break

        growth_feasible = total_growth_needed > 0 and scheduler_error is None

        # Load candidate selection
        onhold_set = set(onhold_uuids)
        completed_set = set(completed_uuids)
        evicted_set = set(host_evicted_uuids)
        decode_uuids_final = [u for u in decode_after_eviction if u not in onhold_set]

        new_load_uuids = []
        if global_candidate_info:
            # Compute adjusted free pages after extensions (arithmetic, no collective needed).
            adjusted_per_rank_free = [
                per_rank_free[r]
                - actual_extension_by_rank[r // group_size]
                for r in range(world_size)
            ]
            new_load_uuids, _ = select_sequences_for_loading(
                candidates=global_candidate_info,
                per_rank_free_pages=adjusted_per_rank_free,
                exclude_uuids=completed_set | onhold_set | evicted_set,
                strategy=LoadingStrategy.LONGEST_FIRST,
                get_global_idx_fn=meta_global_idx,
                group_size=group_size,
            )

        return BoundaryDecisions(
            completed_uuids=completed_uuids,
            active_uuids=active_uuids,
            host_growth_uuids=host_growth_uuids,
            host_growth_pages=host_growth_pages_list,
            growth_feasible=growth_feasible,
            host_evicted_uuids=host_evicted_uuids,
            onhold_uuids=onhold_uuids,
            seqs_needing_extension=seqs_needing_extension,
            new_load_uuids=new_load_uuids,
            decode_uuids_final=decode_uuids_final,
            scheduler_error=scheduler_error,
        )
