"""Host-KV growth planning credits only pages a release actually returns.

With prefix cache on, ``host_pages_allocated`` counts attached shared prefix
pages and pages retained into prefix residency, neither of which is in the
sequence's own Host chain, so releasing the row does not free them. The
rank-0 boundary plan must credit completed/evicted rows with
``host_owned_pages`` instead.
"""

from __future__ import annotations

from batchgen.worker.boundary import (
    BoundaryDecisionRequest,
    BoundaryHandler,
    BoundarySeqMeta,
)

_GPN = 8
_TOTAL_PAGES = 455  # safety margin = int(455 * 0.05) = 22


def _state(*, completed, allocated, owned, growth_pages=0, decoded_length=10):
    return {
        "completed": completed,
        "assigned_rank": 0,
        "decode_dp_group": None,
        "decoded_length": decoded_length,
        "additional_pages_needed": 0,
        "gpu_pages_allocated": 4,
        "host_pages_allocated": allocated,
        "host_owned_pages": owned,
        "needs_host_growth": growth_pages > 0,
        "host_growth_pages": growth_pages,
        "current_context_length": 100,
        "host_token_capacity": allocated * 64,
    }


def _plan(state, *, free_pages, enable_host_kv_eviction=False):
    uuids = list(state)
    return BoundaryHandler.compute_decisions(BoundaryDecisionRequest(
        decode_uuids=tuple(uuids),
        global_seq_state=state,
        global_candidate_info={},
        per_rank_free=(100,) * _GPN,
        chunk_size=64,
        per_node_host_stats=({
            "node_id": 0,
            "num_total_pages": _TOTAL_PAGES,
            "num_free_pages": free_pages,
        },),
        seq_meta={
            uuid: BoundarySeqMeta(
                global_idx=i, priority=0, current_context_length=100,
                host_token_capacity=state[uuid]["host_token_capacity"],
                host_pages_allocated=state[uuid]["host_pages_allocated"],
            )
            for i, uuid in enumerate(uuids)
        },
        world_size=_GPN,
        num_gpus_per_node=_GPN,
        enable_host_kv_eviction=enable_host_kv_eviction,
        host_kv_eviction_watermark=0,
    ))


def _completed_and_growers(owned):
    state = {
        f"done{i}": _state(completed=True, allocated=61, owned=owned)
        for i in range(4)
    }
    state.update({
        f"grow{i}": _state(completed=False, allocated=8, owned=8, growth_pages=32)
        for i in range(6)
    })
    return state


def test_prefix_rows_credit_only_owned_pages():
    # 13 free + 4 * 29 owned = 129 < 6 * 32 growth + 22 safety = 214.
    # Crediting host_pages_allocated would see 13 + 4 * 61 = 257 (feasible).
    plan = _plan(_completed_and_growers(owned=29), free_pages=13)
    assert plan.growth_feasible is False
    assert plan.scheduler_error is not None
    assert "infeasible" in plan.scheduler_error


def test_prefix_free_rows_plan_unchanged():
    # owned == allocated: 13 + 4 * 61 = 257 >= 214.
    plan = _plan(_completed_and_growers(owned=61), free_pages=13)
    assert plan.growth_feasible is True
    assert plan.scheduler_error is None
    assert plan.host_growth_pages == [32] * 6


def test_eviction_credits_only_owned_pages():
    # Need 2 * 32 + 22 = 86 free. Victims each own 29 of 61 allocated pages:
    # 13 + 29k >= 86 needs k = 3 (allocated credit would stop at k = 2).
    state = {
        f"victim{i}": _state(completed=False, allocated=61, owned=29, decoded_length=1)
        for i in range(4)
    }
    state.update({
        f"grow{i}": _state(
            completed=False, allocated=8, owned=8, growth_pages=32,
            decoded_length=100,
        )
        for i in range(2)
    })
    plan = _plan(state, free_pages=13, enable_host_kv_eviction=True)
    assert plan.host_evicted_uuids == ["victim0", "victim1", "victim2"]
    assert plan.growth_feasible is True
    assert plan.scheduler_error is None
