"""Host-KV boundary eviction is growth-driven, never watermark-driven.

Pins the regression where a host pool legitimately filled by full
reservations (zero growth debt) was drained of its whole active decode set
at every boundary to restore a free-page watermark: each evicted row
returned only its own chain while losing all decode progress, so the batch
thrashed through evict/full-recompute cycles instead of decoding to
completion.

Also pins the no-thrash growth contract: the debt is covered exactly (no
pool-proportional margin) by free, completed, reclaimable retained prefix
and evicted owned pages, and cost-aware victims close the most deficit
pages per recompute token, so a debt of D pages evicts at most D rows.
"""

import math
import random
from fractions import Fraction

import pytest

from batchgen.continuous_batching import (
    EvictionStrategy,
    plan_host_kv_growth_evictions,
    select_sequences_for_eviction,
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
    )
    assert plan.evicted_uuids == []
    assert plan.required_free_pages == 4
    assert plan.growth_feasible_after_eviction is True


def test_growth_debt_beyond_free_pages_evicts_only_enough():
    # 6 pages of growth against 2 free. Evicting seq0 (first in
    # shortest-decoded order) returns its 3 own pages AND retires its 2-page
    # debt, leaving 4 remaining debt against 5 expected free: the plan must
    # stop after exactly that one eviction.
    plan = _plan(
        active_uuids=[f"seq{i}" for i in range(4)],
        eviction_candidates=_candidates(4, pages_each=3),
        host_growth_uuids=["seq0", "seq1", "seq2"],
        host_growth_pages=[2, 2, 2],
        free_pages=2,
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
    )
    assert plan.evicted_uuids == []
    assert plan.growth_feasible_after_eviction is True


# --------------------------------------------------------------------------
# No-thrash growth: exact debt cover, reclaimable supply, cost-aware victims
# --------------------------------------------------------------------------


def _cost_candidate(uuid, *, owned, growth=0, ctx=1000, gidx=0, priority=0):
    return (
        uuid,
        {
            "decoded_length": 0,
            "host_owned_pages": owned,
            "growth_pages": growth,
            "recompute_tokens": ctx,
            "global_idx": gidx,
            "priority": priority,
        },
    )


def _cost_plan(candidates, *, growth, free_pages, total_pages,
               completed_pages=0, reclaimable_pages=0):
    uuids = [uuid for uuid, _ in candidates]
    return plan_host_kv_growth_evictions(
        active_uuids=uuids,
        completed_uuids=[],
        host_growth_uuids=list(growth),
        host_growth_pages=list(growth.values()),
        eviction_candidates=candidates,
        free_pages=free_pages,
        total_pages=total_pages,
        completed_pages=completed_pages,
        reclaimable_pages=reclaimable_pages,
        strategy=EvictionStrategy.GROWTH_COST_AWARE,
        page_key="host_owned_pages",
    )


def test_margin_identity_one_page_debt_one_free_page_evicts_nothing():
    # The deleted 5%-of-pool margin demanded 1 + 500 free pages here; the
    # exact identity needs only the 1 page of debt.
    candidates = [_cost_candidate("grow", owned=8, growth=1)]
    plan = _cost_plan(candidates, growth={"grow": 1}, free_pages=1,
                      total_pages=10_000)
    assert plan.evicted_uuids == []
    assert plan.required_free_pages == 1
    assert plan.growth_feasible_after_eviction is True


def test_reclaimable_prefix_pages_cover_the_debt_without_eviction():
    debt = 37
    candidates = [_cost_candidate("grow", owned=4, growth=debt)] + [
        _cost_candidate(f"b{i}", owned=8, gidx=i + 1) for i in range(30)
    ]
    plan = _cost_plan(candidates, growth={"grow": debt}, free_pages=0,
                      total_pages=10_000, reclaimable_pages=debt)
    assert plan.evicted_uuids == []
    assert plan.expected_free_pages == debt
    assert plan.growth_feasible_after_eviction is True
    # One page short of reclaimable supply is closed by eviction instead.
    short = _cost_plan(candidates, growth={"grow": debt}, free_pages=0,
                       total_pages=10_000, reclaimable_pages=debt - 1)
    assert len(short.evicted_uuids) == 1


@pytest.mark.parametrize("grower_wins", [True, False])
def test_one_page_debt_on_full_pool_evicts_exactly_one_row(grower_wins):
    # Full pool, one page of debt, 70 bystanders owning 8 pages each. The
    # deleted margin (1 + int(0.05 * total)) forced a wave of
    # ceil(501 / 8) = 63 bystander evictions; the exact cover evicts one row.
    total = 10_000
    grower_ctx = 1000 if grower_wins else 4000
    candidates = [_cost_candidate("grow", owned=8, growth=1, ctx=grower_ctx)] + [
        _cost_candidate(f"b{i:02d}", owned=8, ctx=1000, gidx=i + 1)
        for i in range(70)
    ]
    plan = _cost_plan(candidates, growth={"grow": 1}, free_pages=0,
                      total_pages=total)
    old_margin_wave = math.ceil((1 + int(total * 0.05)) / 8)
    assert old_margin_wave == 63
    assert len(plan.evicted_uuids) == 1 < old_margin_wave
    assert plan.growth_feasible_after_eviction is True
    if grower_wins:
        # 9 deficit pages per 1000 tokens beats 8: the grower retires its
        # own debt, and no bystander is touched.
        assert plan.evicted_uuids == ["grow"]
        assert plan.remaining_growth_needed == 0
    else:
        assert plan.evicted_uuids == ["b00"]
        assert plan.remaining_growth_uuids == ["grow"]


def test_cost_aware_order_is_benefit_per_recompute_token_then_priority():
    candidates = [
        # benefit/cost: 10/1000, 4/100, 6/100 (+ growth), 50/100 but HIGH.
        _cost_candidate("long", owned=10, ctx=1000, gidx=0),
        _cost_candidate("short", owned=4, ctx=100, gidx=1),
        _cost_candidate("grower", owned=2, growth=4, ctx=100, gidx=2),
        _cost_candidate("high", owned=50, ctx=100, gidx=3, priority=1),
    ]
    plan = _cost_plan(candidates, growth={"grower": 4}, free_pages=0,
                      total_pages=1000)
    # grower (6/100) first: it alone closes the whole 4-page deficit.
    assert plan.evicted_uuids == ["grower"]
    # A deficit beyond every NORMAL row reaches HIGH rows only afterwards,
    # and within HIGH the better score (50/100 over 61/10000) goes first.
    candidates_deep = [c for c in candidates if c[0] != "grower"] + [
        _cost_candidate("g2", owned=1, growth=60, ctx=10_000, gidx=9, priority=1)
    ]
    deep = _cost_plan(candidates_deep, growth={"g2": 60}, free_pages=0,
                      total_pages=1000)
    assert deep.evicted_uuids == ["short", "long", "high"]
    assert deep.growth_feasible_after_eviction is True


def test_cost_aware_ties_and_input_order_are_deterministic():
    base = [
        _cost_candidate(f"r{i}", owned=2 * (i % 3) + 1, growth=i % 2,
                        ctx=100 * (i % 4 + 1), gidx=i)
        for i in range(24)
    ]
    growth = {uuid: s["growth_pages"] for uuid, s in base if s["growth_pages"]}
    reference = _cost_plan(base, growth=growth, free_pages=0, total_pages=500)
    rng = random.Random(11)
    for _ in range(20):
        shuffled = list(base)
        rng.shuffle(shuffled)
        assert _cost_plan(shuffled, growth=growth, free_pages=0,
                          total_pages=500) == reference
    # Exact ties (2/200 == 1/100) break by global_idx, never by float noise.
    tie = [
        _cost_candidate("b", owned=2, ctx=200, gidx=5),
        _cost_candidate("a", owned=1, ctx=100, gidx=7),
    ]
    plan = _cost_plan(tie + [_cost_candidate("g", owned=1, growth=3, ctx=10**6, gidx=9)],
                      growth={"g": 3}, free_pages=0, total_pages=100)
    assert plan.evicted_uuids[:2] == ["b", "a"]
    assert Fraction(2, 200) == Fraction(1, 100)


def test_cost_aware_candidate_without_cost_fields_fails_loud():
    bad = [("x", {"host_owned_pages": 3, "global_idx": 0, "priority": 0})]
    with pytest.raises(ValueError, match="growth_pages"):
        _cost_plan(bad, growth={"x": 1}, free_pages=0, total_pages=10)


def test_gpu_eviction_selector_rejects_growth_cost_aware():
    with pytest.raises(ValueError, match="GROWTH_COST_AWARE"):
        select_sequences_for_eviction(
            [("x", {"gpu_pages_allocated": 1})], 1,
            strategy=EvictionStrategy.GROWTH_COST_AWARE,
        )


def test_no_thrash_property_victims_bounded_by_debt():
    """Each victim closes >= its owned pages >= 1 deficit page.

    With zero other supply, a debt of D pages evicts at most
    ceil(D / min benefit) <= D rows, and the plan is always feasible when
    every grower is a candidate. With reclaimable supply >= D it evicts none.
    """
    rng = random.Random(20261110)
    for _ in range(3000):
        n = rng.randint(1, 40)
        candidates = []
        growth = {}
        for i in range(n):
            owned = rng.randint(1, 12)
            g = rng.choice([0, 0, 0, 1, 2, 7, 64])
            if g:
                growth[f"r{i}"] = g
            candidates.append(_cost_candidate(
                f"r{i}", owned=owned, growth=g, ctx=rng.randint(0, 70_000),
                gidx=rng.randint(0, 3 * n), priority=rng.randint(0, 1),
            ))
        debt = sum(growth.values())
        total = sum(s["host_owned_pages"] for _, s in candidates) + rng.randint(0, 50)
        plan = _cost_plan(candidates, growth=growth, free_pages=0,
                          total_pages=total)
        assert plan.growth_feasible_after_eviction, plan
        min_benefit = min(
            s["host_owned_pages"] + s["growth_pages"] for _, s in candidates
        )
        bound = math.ceil(debt / min_benefit) if debt else 0
        assert len(plan.evicted_uuids) <= bound <= debt, (plan, debt, min_benefit)
        covered = _cost_plan(candidates, growth=growth, free_pages=0,
                             total_pages=total + debt, reclaimable_pages=debt)
        assert covered.evicted_uuids == []
        assert covered.growth_feasible_after_eviction
