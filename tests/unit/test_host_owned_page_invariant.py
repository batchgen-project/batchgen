"""HOST_OWNED_PAGE: a decode-capable sequence owns >= 1 page of its Host chain.

Admission reserves at least ``ceil((prompt + 1) / page)`` Host pages while
prefix attach and prefix commit share at most ``floor(prompt / page)`` full
prompt pages, so the page holding the first decode position stays in the
sequence's own chain. The host-KV growth planner relies on it: with eviction
enabled every growing row is an eviction candidate that frees >= 1 page, so
evicting every candidate clears all growth debt and the boundary's
"infeasible after eviction" branch cannot fire.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from batchgen.continuous_batching import (
    HostKVGrowthEvictionPlan,
    plan_host_kv_growth_evictions,
)
from batchgen.prefix_reuse.commit import (
    aligned_prefix_tokens,
    build_prefix_commit_request,
    retain_inserted_prefix_pages,
)
from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker import boundary as boundary_module
from batchgen.worker.boundary import (
    BoundaryDecisionRequest,
    BoundaryHandler,
    BoundarySeqMeta,
)
from batchgen.worker.prefill import (
    compute_prefill_host_reservation,
    host_owned_pages_after_sharing,
)

PAGE = SequenceEntry.PAGE_SIZE
_GPN = 8


# --------------------------------------------------------------------------
# (a) reservation minus shared pages >= 1, through the real helpers
# --------------------------------------------------------------------------


class _RetainView:
    def retain_sequence_pages(self, sequence_id, page_ids):
        return list(page_ids)


def _owned_after_admission(*, prompt, max_decode, shared, chunk, buffer):
    """Run admission (reserve + attach) and prefix commit (retain) for one row.

    ``shared`` is the cached prefix length the lookup matched. The commit is
    the worst case for ownership: every committed page the row did not attach
    is newly inserted, hence retained out of the row's chain.
    """
    pages, _ = compute_prefill_host_reservation(
        prompt_length=prompt,
        kv_token_budget=prompt + max_decode,
        page_size=PAGE,
        chunk_size=chunk,
        initial_gpu_page_buffer=buffer,
    )
    attached_pages = aligned_prefix_tokens(shared, PAGE) // PAGE
    owned = host_owned_pages_after_sharing(
        chain_pages=pages,
        shared_pages=attached_pages,
        sequence_id=0,
        where="test reservation",
    )
    commit_pages = aligned_prefix_tokens(prompt, PAGE) // PAGE
    request = build_prefix_commit_request(
        namespace_digest=(0, 0, 0, 0),
        token_ids=list(range(prompt)),
        publish_boundary_tokens=PAGE,
        pages_by_group={0: list(range(commit_pages))},
    )
    if request is None:
        return owned
    commit_result = SimpleNamespace(inserted_group_pages=[
        SimpleNamespace(
            group_id=0,
            pages=request.page_ids_by_group[0][attached_pages:],
        )
    ])
    retained = retain_inserted_prefix_pages(
        commit_result=commit_result,
        request=request,
        worker_views_by_group={0: _RetainView()},
        sequence_id=0,
    )
    return host_owned_pages_after_sharing(
        chain_pages=owned,
        shared_pages=len(retained.get(0, ())),
        sequence_id=0,
        where="test commit",
    )


def _share_lengths(prompt):
    if prompt <= 2 * PAGE:
        return range(prompt + 1)
    middle = (prompt // 2 // PAGE) * PAGE
    edges = {
        0, 1, PAGE - 1, PAGE, PAGE + 1, middle - 1, middle, middle + 1,
        prompt - PAGE - 1, prompt - PAGE, prompt - 1, prompt,
    }
    return sorted(s for s in edges if 0 <= s <= prompt)


@pytest.mark.parametrize("prompt", [0, 1, 63, 64, 65, 127, 128, 129, 8191, 8192, 32768])
@pytest.mark.parametrize("max_decode", [1, 2, 63, 64, 65, 8192])
@pytest.mark.parametrize("chunk,buffer", [(0, 0), (64, 0), (64, 32), (8192, 32)])
def test_admission_and_commit_leave_one_owned_page(prompt, max_decode, chunk, buffer):
    for shared in _share_lengths(prompt):
        owned = _owned_after_admission(
            prompt=prompt, max_decode=max_decode, shared=shared,
            chunk=chunk, buffer=buffer,
        )
        assert owned >= 1, (prompt, max_decode, shared, chunk, buffer, owned)


def test_invariant_needs_decode_budget():
    # The exact boundary of the invariant: with no decode budget
    # (kv_token_budget == prompt, which max_tokens >= 1 forbids at the API)
    # a page-aligned prompt fully covered by a prefix leaves no owned page,
    # and the establishing helper refuses it instead of committing zero.
    with pytest.raises(RuntimeError, match=r"\[SCHED_INVARIANT\] HOST_OWNED_PAGE"):
        _owned_after_admission(
            prompt=2 * PAGE, max_decode=0, shared=2 * PAGE, chunk=64, buffer=32,
        )


def test_helper_reports_numbers():
    with pytest.raises(RuntimeError) as err:
        host_owned_pages_after_sharing(
            chain_pages=3, shared_pages=3, sequence_id=17, where="unit",
        )
    message = str(err.value)
    assert "HOST_OWNED_PAGE broken at unit" in message
    assert "sequence 17 would own 0" in message
    assert "chain_pages=3" in message and "shared_pages=3" in message


# --------------------------------------------------------------------------
# (b) planner: owned >= 1 for every row makes "evict all" feasible
# --------------------------------------------------------------------------


def _row(*, owned, growth, completed=False, decoded_length=10):
    return {
        "completed": completed,
        "assigned_rank": 0,
        "decode_dp_group": None,
        "decoded_length": decoded_length,
        "additional_pages_needed": 0,
        "gpu_pages_allocated": 4,
        "host_pages_allocated": max(owned, 1),
        "host_owned_pages": owned,
        "needs_host_growth": growth > 0,
        "host_growth_pages": growth,
        "current_context_length": 100,
        "host_token_capacity": max(owned, 1) * PAGE,
    }


def _decide(state, *, free_pages, total_pages, enable_host_kv_eviction):
    uuids = list(state)
    return BoundaryHandler.compute_decisions(BoundaryDecisionRequest(
        decode_uuids=tuple(uuids),
        global_seq_state=state,
        global_candidate_info={},
        per_rank_free=(100,) * _GPN,
        chunk_size=64,
        per_node_host_stats=({
            "node_id": 0,
            "num_total_pages": total_pages,
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
    ))


def test_planner_feasible_when_every_row_owns_a_page():
    rng = random.Random(20261110)
    for _ in range(2000):
        n = rng.randint(1, 8)
        uuids = [f"s{i}" for i in range(n)]
        owned = [rng.randint(1, 6) for _ in uuids]
        growth = [rng.choice([0, 0, 1, 8, 32, 500]) for _ in uuids]
        completed = [u for u in uuids if rng.random() < 0.2]
        total = rng.randint(1, 600)
        plan = plan_host_kv_growth_evictions(
            active_uuids=[u for u in uuids if u not in completed],
            completed_uuids=completed,
            host_growth_uuids=uuids,
            host_growth_pages=growth,
            eviction_candidates=[
                (u, {"decoded_length": rng.randint(0, 100), "host_owned_pages": o,
                     "global_idx": i, "priority": rng.randint(0, 1)})
                for i, (u, o) in enumerate(zip(uuids, owned))
                if u not in completed
            ],
            free_pages=rng.randint(0, total),
            total_pages=total,
            completed_pages=sum(o for u, o in zip(uuids, owned) if u in completed),
            safety_margin=rng.randint(0, 2 * total),
            page_key="host_owned_pages",
        )
        assert plan.growth_feasible_after_eviction, plan


def test_boundary_never_infeasible_when_every_row_owns_a_page():
    # An empty pool, a pool too small for any growth, and a safety margin
    # larger than the pool all stay feasible: evicting the growers clears
    # their debt, and with no debt the margin is not required.
    for free_pages, total_pages in [(0, 20), (0, 455), (13, 455), (2, 2)]:
        state = {f"grow{i}": _row(owned=1, growth=32) for i in range(6)}
        state.update({f"idle{i}": _row(owned=3, growth=0) for i in range(3)})
        plan = _decide(
            state, free_pages=free_pages, total_pages=total_pages,
            enable_host_kv_eviction=True,
        )
        assert plan.scheduler_error is None, plan.scheduler_error
        # Every grower is either evicted or grows in this boundary.
        growers = {f"grow{i}" for i in range(6)}
        assert growers <= set(plan.host_evicted_uuids) | set(plan.host_growth_uuids)
        assert plan.growth_feasible == bool(plan.host_growth_uuids)


def test_boundary_infeasible_only_with_zero_owned_grower():
    # 0 free + 0 freed < 32 growth + int(455 * 0.05) = 22 safety: the grower
    # owns no page, the planner skips it as a candidate, and its debt stays.
    state = {"grow": _row(owned=0, growth=32), "idle": _row(owned=1, growth=0)}
    plan = _decide(state, free_pages=0, total_pages=455, enable_host_kv_eviction=True)
    assert plan.scheduler_error is not None
    assert "[SCHED_INVARIANT]" in plan.scheduler_error
    assert "grow(gid=0,owned=0,growth=32)" in plan.scheduler_error


def test_boundary_infeasible_with_eviction_disabled():
    # The other exact condition: with eviction disabled no row is a
    # candidate, so pool sizing alone decides. Every row owns >= 1 page:
    # 13 free + 4 * 29 completed = 129 < 6 * 32 growth + 22 safety = 214.
    # Production derives eviction from a positive host-KV chunk size, which
    # server argument validation requires, so this needs a direct request.
    state = {
        f"done{i}": _row(owned=29, growth=0, completed=True) for i in range(4)
    }
    state.update({f"grow{i}": _row(owned=8, growth=32) for i in range(6)})
    plan = _decide(state, free_pages=13, total_pages=455, enable_host_kv_eviction=False)
    assert plan.scheduler_error is not None
    assert "eviction_enabled=False" in plan.scheduler_error
    assert "candidates=0" in plan.scheduler_error
    assert "expected_free=129" in plan.scheduler_error
    assert "required_free=214" in plan.scheduler_error


# --------------------------------------------------------------------------
# (c) validator
# --------------------------------------------------------------------------


def _seq(status, *, owned):
    seq = SequenceEntry(uuid="u", global_idx=0, prompt_length=100, max_decode_length=50)
    seq.assigned_rank = 0
    seq.status = status
    seq.host_owned_pages = owned
    if status == SequenceStatus.EVICTED:
        seq._buffer_slot = 0
        return seq
    seq.host_pages_allocated = 2
    seq.host_token_capacity = 2 * PAGE
    if status == SequenceStatus.IN_DECODE:
        seq.gpu_pages_allocated = 2
    return seq


@pytest.mark.parametrize(
    "status",
    [SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD],
)
def test_validator_rejects_zero_owned_host_pages(status):
    with pytest.raises(RuntimeError, match=r"host_owned_pages >= 1 \(HOST_OWNED_PAGE"):
        _seq(status, owned=0).validate_metadata("test")
    _seq(status, owned=1).validate_metadata("test")


def test_validator_accepts_evicted_with_zero_owned_host_pages():
    _seq(SequenceStatus.EVICTED, owned=0).validate_metadata("test")


# --------------------------------------------------------------------------
# (d) the boundary turns an infeasible plan into the named invariant error
# --------------------------------------------------------------------------


def test_infeasible_plan_raises_named_invariant(monkeypatch):
    def infeasible_plan(**kwargs):
        return HostKVGrowthEvictionPlan(
            evicted_uuids=[],
            freed_pages=0,
            remaining_growth_uuids=["grow"],
            remaining_growth_pages=[32],
            remaining_growth_needed=32,
            expected_free_pages=5,
            required_free_pages=32 + 22,
            growth_feasible_after_eviction=False,
        )

    monkeypatch.setattr(
        boundary_module, "plan_host_kv_growth_evictions", infeasible_plan,
    )
    state = {"grow": _row(owned=2, growth=32)}
    plan = _decide(state, free_pages=5, total_pages=455, enable_host_kv_eviction=True)
    error = plan.scheduler_error
    assert error is not None
    assert error.startswith("[SCHED_INVARIANT] [HOST_KV_GROWTH_PLAN] node 0")
    assert "HOST_OWNED_PAGE" in error
    for field in (
        "eviction_enabled=True", "candidates=1", "free=5", "expected_free=5",
        "growth_pages=32", "safety=22", "required_free=54",
        "grow(gid=0,owned=2,growth=32)",
    ):
        assert field in error, field
    assert plan.growth_feasible is False
