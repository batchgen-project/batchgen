# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear                                                       #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""M2b Tier-1 (CPU): decode DP-group assignment + MoE ntp G-awareness.

Validates the pure arithmetic behind piece 1 (``seq.decode_dp_group``) and
piece 2 (``ntp = ceil(max_B_grp / G)``) without a worker or a GPU:

  * assignment balances N sequences over ``num_dp = world/G`` groups;
  * the G ranks of a group resolve the SAME sequence set (global_rank // G);
  * ranks in different groups resolve DISJOINT sets;
  * ntp is exactly the post-scatter per-rank share.

Each carries a NON-VACUITY control: a wrong group map / wrong divisor FAILS.

Run: python -m pytest tests/test_kimi_linear_m2b_decode_dp_group.py -x -q -rA
"""

from __future__ import annotations

import math

import pytest

from batchgen.decode_dp_group import (
    assign_decode_dp_groups,
    moe_ntp_from_group_max,
    num_decode_dp_groups,
    rank_in_decode_group,
)


def _group_load(lengths, groups, num_dp):
    load = [0.0] * num_dp
    for L, g in zip(lengths, groups):
        load[g] += float(L) * float(L)
    return load


# --------------------------------------------------------------------------- #
#  Piece 1: decode DP-group assignment                                        #
# --------------------------------------------------------------------------- #

def test_num_decode_dp_groups():
    assert num_decode_dp_groups(32, 8) == 4
    assert num_decode_dp_groups(32, 1) == 32   # pure DP
    assert num_decode_dp_groups(8, 8) == 1     # single node
    with pytest.raises(ValueError):
        num_decode_dp_groups(32, 5)            # not divisible


def test_assignment_balanced_over_groups():
    world, G = 32, 8
    num_dp = num_decode_dp_groups(world, G)    # 4
    # Deterministic pseudo-random lengths.
    lengths = [((i * 37 + 11) % 500) + 1 for i in range(40)]
    groups = assign_decode_dp_groups(lengths, num_dp)

    assert len(groups) == len(lengths)
    assert all(0 <= g < num_dp for g in groups)

    load = _group_load(lengths, groups, num_dp)
    balanced_max = max(load)
    # NON-VACUITY control: the worst legal map (everything in one group) must be
    # strictly worse, and the balancer must be far below it.
    worst_max = sum(L * L for L in lengths)
    assert balanced_max < worst_max
    # FFD + L^2 argmin keeps the heaviest group within 1.5x of a perfect split.
    perfect = worst_max / num_dp
    assert balanced_max <= 1.5 * perfect, (load, perfect)


def test_group_ranks_resolve_same_sequence_set():
    world, G = 32, 8
    num_dp = num_decode_dp_groups(world, G)
    lengths = [((i * 13 + 5) % 300) + 1 for i in range(25)]
    groups = assign_decode_dp_groups(lengths, num_dp)

    # For each rank, the set of sequences it owns via the decode predicate.
    def owned(rank):
        return {i for i, g in enumerate(groups)
                if rank_in_decode_group(g, rank, G)}

    # Every rank maps to group rank//G; all G ranks of a group own the SAME set.
    for grp in range(num_dp):
        ranks = list(range(grp * G, (grp + 1) * G))
        assert all(r // G == grp for r in ranks)
        sets = [owned(r) for r in ranks]
        expected = {i for i, g in enumerate(groups) if g == grp}
        for s in sets:
            assert s == expected                 # same set on every rank
        # ranks of DIFFERENT groups are disjoint
        for other in range(num_dp):
            if other == grp:
                continue
            other_set = {i for i, g in enumerate(groups) if g == other}
            assert expected.isdisjoint(other_set)

    # Union over all groups == all sequences (nothing dropped/duplicated).
    union = set()
    for grp in range(num_dp):
        union |= {i for i, g in enumerate(groups) if g == grp}
    assert union == set(range(len(lengths)))


def test_assignment_nonvacuity_wrong_map_unbalanced():
    """A length-blind all-to-group-0 map fails the same balance gate."""
    world, G = 32, 8
    num_dp = num_decode_dp_groups(world, G)
    lengths = [((i * 37 + 11) % 500) + 1 for i in range(40)]

    good = assign_decode_dp_groups(lengths, num_dp)
    bad = [0] * len(lengths)                      # wrong: everything on group 0

    good_max = max(_group_load(lengths, good, num_dp))
    bad_max = max(_group_load(lengths, bad, num_dp))
    perfect = sum(L * L for L in lengths) / num_dp
    assert good_max <= 1.5 * perfect
    assert bad_max > 1.5 * perfect               # the wrong map is caught


def test_assignment_incremental_prior_load():
    """Prior-load keeps incremental admissions balanced vs the standing batch."""
    num_dp = 4
    # Group 0 already heavily loaded; new short seqs must avoid it.
    prior = [1_000_000.0, 0.0, 0.0, 0.0]
    lengths = [10, 10, 10]
    groups = assign_decode_dp_groups(lengths, num_dp, prior_load=prior)
    assert 0 not in groups                        # never piles onto the hot group

    with pytest.raises(ValueError):
        assign_decode_dp_groups([1, 2], num_dp, prior_load=[0.0, 0.0])  # wrong len


def test_rank_in_decode_group_mapping():
    G = 8
    # group id g == global_rank // G for every rank in [g*G,(g+1)*G).
    for g in range(4):
        for r in range(g * G, (g + 1) * G):
            assert rank_in_decode_group(g, r, G)
            for other in range(4):
                if other != g:
                    assert not rank_in_decode_group(other, r, G)
    assert rank_in_decode_group(None, 0, G) is False   # unassigned owns nothing


# --------------------------------------------------------------------------- #
#  Piece 2: MoE ntp = ceil(max_B_grp / G)                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("max_b,G", [
    (1, 8), (7, 8), (8, 8), (9, 8), (16, 8), (17, 8),
    (10, 4), (3, 4), (5, 2), (100, 8), (256, 32),
])
def test_moe_ntp_is_ceil_share(max_b, G):
    assert moe_ntp_from_group_max(max_b, G) == math.ceil(max_b / G)


def test_moe_ntp_g1_identity_and_nonvacuity():
    # G==1 (pure DP): ntp is the group max unchanged.
    for b in (1, 5, 33, 256):
        assert moe_ntp_from_group_max(b, 1) == b
    # NON-VACUITY: for G>1 the padded per-rank share is strictly smaller than
    # the group batch, i.e. using B_grp directly would over-pad the collective.
    assert moe_ntp_from_group_max(64, 8) == 8
    assert moe_ntp_from_group_max(64, 8) < 64
    with pytest.raises(ValueError):
        moe_ntp_from_group_max(64, 0)
