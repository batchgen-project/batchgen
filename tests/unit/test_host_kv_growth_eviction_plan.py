"""Host-KV boundary eviction is growth-driven, never watermark-driven.

Pins the regression where a host pool legitimately filled by full
reservations (zero growth debt) was drained of its whole active decode set
at every boundary to restore a free-page watermark: each evicted row
returned only its own chain while losing all decode progress, so the batch
thrashed through evict/full-recompute cycles instead of decoding to
completion.
"""

from batchgen.continuous_batching import (
    EvictionStrategy,
    plan_host_kv_growth_evictions,
)


def _candidates(n, pages_each=4, decoded=128):
    return [
        (
            f"seq{i}",
            {
                "decoded_length": decoded,
                "host_owned_pages": pages_each,
                "global_idx": i,
                "priority": 0,
            },
        )
        for i in range(n)
    ]


def _plan(**overrides):
    kwargs = dict(
        active_uuids=[f"seq{i}" for i in range(24)],
        completed_uuids=[],
        host_growth_uuids=[],
        host_growth_pages=[],
        eviction_candidates=_candidates(24),
        free_pages=639,
        total_pages=126787,
        completed_pages=0,
        safety_margin=6339,
        strategy=EvictionStrategy.SHORTEST_FIRST,
        page_key="host_owned_pages",
    )
    kwargs.update(overrides)
    return plan_host_kv_growth_evictions(**kwargs)


def test_full_pool_zero_growth_debt_evicts_nothing():
    # The regression shape: pool ~100% used by full reservations, free pages
    # far below any percentage watermark, and no row needs host growth.
    plan = _plan()
    assert plan.evicted_uuids == []
    assert plan.freed_pages == 0
    assert plan.required_free_pages == 0
    assert plan.growth_feasible_after_eviction is True


def test_growth_debt_covered_by_free_pages_evicts_nothing():
    plan = _plan(
        host_growth_uuids=["seq0", "seq1"],
        host_growth_pages=[2, 2],
        free_pages=20,
        safety_margin=10,
    )
    assert plan.evicted_uuids == []
    assert plan.required_free_pages == 4 + 10
    assert plan.growth_feasible_after_eviction is True


def test_growth_debt_beyond_free_pages_evicts_only_enough():
    # 6 pages of growth + 1 safety against 2 free. Evicting seq0 (first in
    # shortest-decoded order) returns its 3 own pages AND retires its 2-page
    # debt, leaving 4 remaining debt + 1 safety against 5 expected free: the
    # plan must stop after exactly that one eviction.
    plan = _plan(
        active_uuids=[f"seq{i}" for i in range(4)],
        eviction_candidates=_candidates(4, pages_each=3),
        host_growth_uuids=["seq0", "seq1", "seq2"],
        host_growth_pages=[2, 2, 2],
        free_pages=2,
        safety_margin=1,
    )
    assert plan.evicted_uuids == ["seq0"]
    assert plan.growth_feasible_after_eviction is True
    assert plan.expected_free_pages >= plan.required_free_pages


def test_completed_pages_credit_satisfies_growth_without_eviction():
    plan = _plan(
        completed_uuids=["done0"],
        host_growth_uuids=["seq0"],
        host_growth_pages=[4],
        free_pages=0,
        completed_pages=20,
        safety_margin=10,
    )
    assert plan.evicted_uuids == []
    assert plan.growth_feasible_after_eviction is True
