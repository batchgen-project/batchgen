"""``RankContext``: the single owner of "node of a rank" and of decode-group
sequence ownership.

The ownership predicates moved here from ``BatchGenWorker._owns_local_sequence``
and ``_owns_host_kv`` unchanged. These tests pin today's behaviour for G=1
(pure DP, the validated path) and G=8 (Kimi-K3 resident TP-G), so a later
slice that means to change the G>1 rules has to change a test on purpose.
"""

import math
from types import SimpleNamespace

import pytest

from batchgen.worker.rank_context import RankContext, node_of_rank


def _ctx(rank, *, world_size=32, local_world_size=8, decode_group_size=1):
    return RankContext(
        rank=rank,
        world_size=world_size,
        local_rank=rank % local_world_size,
        local_world_size=local_world_size,
        decode_group_size=decode_group_size,
    )


def _seq(*, assigned_rank=0, decode_dp_group=None):
    return SimpleNamespace(
        uuid="u", global_idx=1,
        assigned_rank=assigned_rank, decode_dp_group=decode_dp_group,
    )


# ------------------------------------------------------------ node of a rank


@pytest.mark.parametrize(
    "rank,local_world_size,expected",
    [(0, 8, 0), (7, 8, 0), (8, 8, 1), (31, 8, 3), (0, 1, 0), (5, 1, 5)],
)
def test_node_of_rank_is_floor_division(rank, local_world_size, expected):
    assert node_of_rank(rank, local_world_size) == expected


def test_node_of_matches_the_module_function_for_every_rank():
    ctx = _ctx(0, world_size=32, local_world_size=8)
    assert [ctx.node_of(r) for r in range(32)] == [
        node_of_rank(r, 8) for r in range(32)
    ]


def test_node_of_answers_for_any_rank_not_only_this_one():
    ctx = _ctx(3, world_size=32, local_world_size=8)
    assert ctx.node_of(3) == 0 and ctx.node_of(9) == 1 and ctx.node_of(24) == 3


def test_node_id_is_this_ranks_node():
    assert _ctx(9, local_world_size=8).node_id == 1


# ------------------------------------------------------------ node count


@pytest.mark.parametrize(
    "world_size,local_world_size,expected",
    [
        (1, 1, 1),     # single GPU
        (8, 8, 1),     # one full node (G=8 resident TP lives inside it)
        (32, 8, 4),    # four full nodes
        (4, 8, 1),     # a partial single node still counts as one
        (9, 8, 2),     # uneven: a partial last node is counted
        (15, 8, 2),
        (17, 8, 3),
        (7, 2, 4),
        (5, 1, 5),     # pure per-GPU "nodes"
    ],
)
def test_num_nodes_is_ceiling_division_at_least_one(world_size, local_world_size, expected):
    ctx = _ctx(0, world_size=world_size, local_world_size=local_world_size)
    assert ctx.num_nodes == expected
    # The old inline worker formula (batchgen_worker.py:6733 / :5610 @ ad4a0942).
    assert ctx.num_nodes == max(1, math.ceil(world_size / local_world_size))


@pytest.mark.parametrize("decode_group_size", [1, 8])
def test_num_nodes_is_independent_of_the_decode_group_size(decode_group_size):
    ctx = _ctx(0, world_size=32, local_world_size=8, decode_group_size=decode_group_size)
    assert ctx.num_nodes == 4


def test_every_rank_node_is_below_num_nodes():
    for world_size, local_world_size in [(32, 8), (9, 8), (7, 2), (1, 1)]:
        ctx = _ctx(0, world_size=world_size, local_world_size=local_world_size)
        assert {ctx.node_of(r) for r in range(world_size)} == set(range(ctx.num_nodes))


def test_single_node_runs_put_every_rank_on_node_zero():
    """The retired ``world_size <= local_world_size -> 0`` special case in
    ``BatchGenWorker._get_node_for_rank`` is subsumed by floor division: every
    valid rank of a single-node run is below ``local_world_size``."""
    ctx = _ctx(0, world_size=8, local_world_size=8)
    assert {ctx.node_of(rank) for rank in range(8)} == {0}


def test_rank_context_is_frozen():
    ctx = _ctx(0)
    with pytest.raises(Exception):
        ctx.rank = 1


# ----------------------------------------------------- owns_sequence, G == 1


def test_g1_owns_sequence_is_the_assigned_rank():
    ctx = _ctx(3, decode_group_size=1)
    assert ctx.owns_sequence(_seq(assigned_rank=3))
    assert not ctx.owns_sequence(_seq(assigned_rank=4))


def test_g1_ignores_the_decode_group_entirely():
    """Pure DP never consults ``decode_dp_group``, stamped or not."""
    ctx = _ctx(3, decode_group_size=1)
    assert ctx.owns_sequence(_seq(assigned_rank=3, decode_dp_group=7))
    assert not ctx.owns_sequence(_seq(assigned_rank=4, decode_dp_group=0))


# ----------------------------------------------------- owns_sequence, G == 8


def test_g8_all_eight_ranks_of_the_group_own_the_sequence():
    seq = _seq(assigned_rank=0, decode_dp_group=1)
    owners = [
        rank for rank in range(32)
        if _ctx(rank, decode_group_size=8).owns_sequence(seq)
    ]
    assert owners == list(range(8, 16))


def test_g8_owns_sequence_does_not_depend_on_assigned_rank():
    """Every rank of group 1 owns it however ``assigned_rank`` is stamped."""
    ctx = _ctx(9, decode_group_size=8)
    assert ctx.owns_sequence(_seq(assigned_rank=0, decode_dp_group=1))
    assert ctx.owns_sequence(_seq(assigned_rank=15, decode_dp_group=1))
    assert not ctx.owns_sequence(_seq(assigned_rank=9, decode_dp_group=2))


def test_g8_unstamped_group_is_owned_by_nobody():
    """An unstamped sequence is not owned anywhere, so a missing stamp shows
    up as "no rank binds it" rather than as every rank binding it."""
    seq = _seq(assigned_rank=0, decode_dp_group=None)
    assert not any(
        _ctx(rank, decode_group_size=8).owns_sequence(seq) for rank in range(32)
    )


# -------------------------------------------------------------- owns_host_kv


def test_g1_owns_host_kv_is_the_assigned_rank():
    ctx = _ctx(3, decode_group_size=1)
    assert ctx.owns_host_kv(_seq(assigned_rank=3))
    assert not ctx.owns_host_kv(_seq(assigned_rank=4))


def test_g8_exactly_the_group_leader_owns_the_shared_host_region():
    seq = _seq(assigned_rank=11, decode_dp_group=1)
    owners = [
        rank for rank in range(32)
        if _ctx(rank, decode_group_size=8).owns_host_kv(seq)
    ]
    assert owners == [8]


@pytest.mark.parametrize("group", [0, 1, 2, 3])
def test_g8_every_group_has_exactly_one_host_kv_owner(group):
    seq = _seq(assigned_rank=0, decode_dp_group=group)
    owners = [
        rank for rank in range(32)
        if _ctx(rank, decode_group_size=8).owns_host_kv(seq)
    ]
    assert owners == [group * 8]


def test_g8_unstamped_group_has_no_host_kv_owner():
    seq = _seq(assigned_rank=0, decode_dp_group=None)
    assert not any(
        _ctx(rank, decode_group_size=8).owns_host_kv(seq) for rank in range(32)
    )


def test_g8_host_kv_ownership_is_narrower_than_sequence_ownership():
    """The invariant the shared host region depends on: every host-KV owner
    also owns the sequence, and for G>1 strictly fewer ranks do."""
    seq = _seq(assigned_rank=0, decode_dp_group=2)
    ctxs = [_ctx(rank, decode_group_size=8) for rank in range(32)]
    binds = {c.rank for c in ctxs if c.owns_sequence(seq)}
    host = {c.rank for c in ctxs if c.owns_host_kv(seq)}
    assert host < binds
    assert len(host) == 1 and len(binds) == 8
