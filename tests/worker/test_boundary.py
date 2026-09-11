"""Unit tests for `batchgen.worker.boundary.BoundaryHandler`.

Pure CPU tests over the rank-0 page-boundary decision with fabricated
snapshot inputs — no server, no NCCL, no global_batch. Topology: 8
GPUs/node (ranks 0-7 = node 0, 8-15 = node 1).

The deeper host-KV growth/eviction integration (which composes the
pre-existing pure helper `plan_host_kv_growth_evictions`) is exercised
end-to-end on real boundaries by the compare-mode gate (every decode
boundary runs both legacy and native and asserts the BoundaryDecisions
are equal). These unit tests pin the deterministically-predictable
phases: completion split, GPU extension/on-hold, loading, scheduler error.
"""

from __future__ import annotations

import pytest

from batchgen.worker.boundary import (
    BoundaryDecisionRequest,
    BoundaryHandler,
    BoundarySeqMeta,
)

_GPN = 8


def _state(
    *, assigned_rank, completed=False, decoded_length=10,
    additional_pages_needed=0, gpu_pages_allocated=4, host_pages_allocated=2,
    needs_host_growth=False, host_growth_pages=0,
    current_context_length=100, host_token_capacity=4096, decode_dp_group=None,
):
    return {
        "completed": completed,
        "active": not completed,
        "assigned_rank": assigned_rank,
        "decoded_length": decoded_length,
        "additional_pages_needed": additional_pages_needed,
        "gpu_pages_allocated": gpu_pages_allocated,
        "host_pages_allocated": host_pages_allocated,
        "needs_host_growth": needs_host_growth,
        "host_growth_pages": host_growth_pages,
        "current_context_length": current_context_length,
        "host_token_capacity": host_token_capacity,
        "decode_dp_group": decode_dp_group,
    }


def _meta(gidx, *, priority=0, ctx=100, cap=4096, host_pages=2):
    return BoundarySeqMeta(
        global_idx=gidx, priority=priority,
        current_context_length=ctx, host_token_capacity=cap,
        host_pages_allocated=host_pages,
    )


def _req(
    *, decode_uuids, global_seq_state, per_rank_free, world_size,
    global_candidate_info=None, per_node_host_stats=None, seq_meta=None,
    chunk_size=64, enable_host_kv_eviction=False, host_kv_eviction_watermark=10,
    attn_tp_size=1,
):
    if seq_meta is None:
        seq_meta = {u: _meta(i) for i, u in enumerate(decode_uuids)}
    return BoundaryDecisionRequest(
        decode_uuids=tuple(decode_uuids),
        global_seq_state=global_seq_state,
        global_candidate_info=global_candidate_info or {},
        per_rank_free=tuple(per_rank_free),
        chunk_size=chunk_size,
        per_node_host_stats=tuple(per_node_host_stats) if per_node_host_stats else None,
        seq_meta=seq_meta,
        world_size=world_size,
        num_gpus_per_node=_GPN,
        enable_host_kv_eviction=enable_host_kv_eviction,
        host_kv_eviction_watermark=host_kv_eviction_watermark,
        attn_tp_size=attn_tp_size,
    )


def test_completion_split():
    state = {
        "a": _state(assigned_rank=0, completed=True),
        "b": _state(assigned_rank=0, completed=False),
        "c": _state(assigned_rank=1, completed=True),
    }
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=["a", "b", "c"], global_seq_state=state,
             per_rank_free=[100] * 8, world_size=8)
    )
    assert plan.completed_uuids == ["a", "c"]
    assert plan.active_uuids == ["b"]
    assert plan.decode_uuids_final == ["b"]  # no onhold, no eviction
    assert plan.onhold_uuids == []
    assert plan.host_evicted_uuids == []
    assert plan.scheduler_error is None


def test_no_stats_no_growth_is_clean():
    state = {"a": _state(assigned_rank=0), "b": _state(assigned_rank=1)}
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=["a", "b"], global_seq_state=state,
             per_rank_free=[100] * 8, world_size=8)
    )
    assert plan.active_uuids == ["a", "b"]
    assert plan.host_growth_uuids == []
    assert plan.growth_feasible is False  # no growth needed
    assert plan.scheduler_error is None


def test_growth_requested_without_stats_sets_scheduler_error():
    state = {
        "a": _state(assigned_rank=0, needs_host_growth=True, host_growth_pages=5),
    }
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=["a"], global_seq_state=state,
             per_rank_free=[100] * 8, world_size=8, per_node_host_stats=None)
    )
    # growth requested with no stats → a scheduler error is raised. (The
    # "stats are missing" error fires first, then the downstream "remaining
    # growth but no host KV plan" check overwrites it — last writer wins,
    # exactly as in the legacy control flow.)
    assert plan.scheduler_error is not None
    assert "HOST_KV_GROWTH_PLAN" in plan.scheduler_error
    assert plan.growth_feasible is False


def test_gpu_extension_all_fit_no_onhold():
    state = {
        "a": _state(assigned_rank=0, additional_pages_needed=3),
        "b": _state(assigned_rank=1, additional_pages_needed=2),
    }
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=["a", "b"], global_seq_state=state,
             per_rank_free=[10] * 8, world_size=8)
    )
    assert set(plan.seqs_needing_extension) == {"a", "b"}
    assert plan.onhold_uuids == []
    assert set(plan.decode_uuids_final) == {"a", "b"}


def test_gpu_onhold_when_rank_free_insufficient():
    # rank 0 needs 8 pages total but only 5 free → must shed
    state = {
        "a": _state(assigned_rank=0, additional_pages_needed=8,
                    gpu_pages_allocated=4, decoded_length=5),
        "b": _state(assigned_rank=0, additional_pages_needed=0,
                    gpu_pages_allocated=4, decoded_length=50),
    }
    free = [5] + [100] * 7
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=["a", "b"], global_seq_state=state,
             per_rank_free=free, world_size=8)
    )
    # something on rank 0 must be put on hold to free pages
    assert len(plan.onhold_uuids) >= 1
    # on-hold sequences are excluded from the final decode set
    for u in plan.onhold_uuids:
        assert u not in plan.decode_uuids_final


def test_onhold_priority_then_decoded_length_order():
    # both on rank 0; need to shed exactly one. Lower priority shed first;
    # within equal priority, smaller decoded_length shed first.
    state = {
        "lo_short": _state(assigned_rank=0, additional_pages_needed=8,
                           gpu_pages_allocated=10, decoded_length=5),
        "lo_long": _state(assigned_rank=0, additional_pages_needed=0,
                          gpu_pages_allocated=10, decoded_length=99),
    }
    seq_meta = {"lo_short": _meta(0, priority=0), "lo_long": _meta(1, priority=0)}
    free = [2] + [100] * 7  # rank0 has 2 free, needs 8 → shed ~1 seq (10 pages)
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=["lo_short", "lo_long"], global_seq_state=state,
             per_rank_free=free, world_size=8, seq_meta=seq_meta)
    )
    # equal priority → smaller decoded_length ("lo_short") shed first
    assert plan.onhold_uuids[0] == "lo_short"


def test_loading_selection_from_candidates():
    state = {"a": _state(assigned_rank=0)}
    candidates = {
        "x": {"decoded_length": 50, "pages_needed": 3, "assigned_rank": 1},
        "y": {"decoded_length": 90, "pages_needed": 3, "assigned_rank": 1},
    }
    seq_meta = {"a": _meta(0), "x": _meta(10), "y": _meta(11)}
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=["a"], global_seq_state=state, per_rank_free=[100] * 8,
             world_size=8, global_candidate_info=candidates, seq_meta=seq_meta)
    )
    # LONGEST_FIRST → both fit on rank 1; "y" (decoded 90) ordered before "x"
    assert set(plan.new_load_uuids) == {"x", "y"}
    assert plan.new_load_uuids[0] == "y"


def test_loading_excludes_completed_and_onhold():
    state = {
        "done": _state(assigned_rank=1, completed=True),
        "a": _state(assigned_rank=0),
    }
    # candidate "done" is completed → must be excluded from loading
    candidates = {
        "done": {"decoded_length": 50, "pages_needed": 2, "assigned_rank": 1},
        "fresh": {"decoded_length": 60, "pages_needed": 2, "assigned_rank": 1},
    }
    seq_meta = {"done": _meta(0), "a": _meta(1), "fresh": _meta(2)}
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=["done", "a"], global_seq_state=state,
             per_rank_free=[100] * 8, world_size=8,
             global_candidate_info=candidates, seq_meta=seq_meta)
    )
    assert "done" not in plan.new_load_uuids
    assert "fresh" in plan.new_load_uuids


def test_tp_loading_uses_group_capacity_and_not_assigned_rank():
    state = {
        "active": _state(assigned_rank=1),
    }
    candidates = {
        # Both candidates are in group 0.  A rank-based planner would admit
        # both because their stale assigned_rank values differ; TP8 must
        # charge them to the same replicated GPU-capacity bucket.
        "x": {
            "decoded_length": 50, "pages_needed": 60,
            "assigned_rank": 1, "decode_dp_group": 0,
        },
        "y": {
            "decoded_length": 40, "pages_needed": 60,
            "assigned_rank": 6, "decode_dp_group": 0,
        },
        "z": {
            "decoded_length": 30, "pages_needed": 60,
            "assigned_rank": 8, "decode_dp_group": 1,
        },
    }
    seq_meta = {u: _meta(i) for i, u in enumerate(["active", "x", "y", "z"])}
    plan = BoundaryHandler.compute_decisions(
        _req(
            decode_uuids=["active"],
            global_seq_state=state,
            per_rank_free=[100] * 16,
            world_size=16,
            global_candidate_info=candidates,
            seq_meta=seq_meta,
            attn_tp_size=8,
        )
    )
    assert plan.new_load_uuids == ["x", "z"]


def test_tp_extension_uses_tightest_rank_in_group():
    state = {
        "a": _state(
            assigned_rank=3, decode_dp_group=0,
            additional_pages_needed=8,
        ),
    }
    # Group 0's rank 0 has only five free pages, so the replicated extension
    # cannot be admitted even though the other seven ranks have capacity.
    plan = BoundaryHandler.compute_decisions(
        _req(
            decode_uuids=["a"],
            global_seq_state=state,
            per_rank_free=[5] + [100] * 7 + [100] * 8,
            world_size=16,
            attn_tp_size=8,
        )
    )
    assert plan.onhold_uuids == ["a"]
    assert plan.decode_uuids_final == []


def test_request_and_meta_are_frozen():
    req = _req(decode_uuids=["a"], global_seq_state={"a": _state(assigned_rank=0)},
               per_rank_free=[1] * 8, world_size=8)
    with pytest.raises((AttributeError, Exception)):
        req.world_size = 16  # type: ignore[misc]
    m = _meta(0)
    with pytest.raises((AttributeError, Exception)):
        m.global_idx = 9  # type: ignore[misc]


def test_empty_decode_uuids():
    plan = BoundaryHandler.compute_decisions(
        _req(decode_uuids=[], global_seq_state={}, per_rank_free=[100] * 8, world_size=8)
    )
    assert plan.completed_uuids == []
    assert plan.active_uuids == []
    assert plan.decode_uuids_final == []
    assert plan.new_load_uuids == []
    assert plan.scheduler_error is None
