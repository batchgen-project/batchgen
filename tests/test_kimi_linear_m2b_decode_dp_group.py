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

import ast
import copy
import logging
import math
import types
from pathlib import Path
from typing import List

import pytest

from batchgen.decode_dp_group import (
    assign_decode_dp_groups,
    host_kv_owner_rank,
    moe_ntp_from_group_max,
    num_decode_dp_groups,
    rank_in_decode_group,
)


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"


def _isolated_worker_method(function_name, globals_=None):
    """Compile one worker method without importing the GPU/JIT-heavy module."""
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = copy.deepcopy(next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ))
    module = ast.Module(
        body=[ast.ClassDef(
            name="Isolated",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )],
        type_ignores=[],
    )
    namespace = {
        "List": List,
        "MigrationOp": object,
        **(globals_ or {}),
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return getattr(namespace["Isolated"], function_name)


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


def test_option1_single_group_all_ranks_own_all():
    """Option 1 parity-gate config: world=8, G=8 -> num_dp=1 (one serve-group).

    The moved binding (admission/prefill, not the decode transition) uses
    rank_in_decode_group as the per-rank ownership predicate. With a single
    group EVERY rank owns EVERY sequence, so all 8 ranks bind + prefill + decode
    the same sequences in TP-8 lockstep. This is exactly the invariant the G=8
    parity gate exercises.
    """
    world, G = 8, 8
    num_dp = num_decode_dp_groups(world, G)
    assert num_dp == 1
    lengths = [((i * 17 + 3) % 200) + 1 for i in range(4)]  # the 4 gate prompts
    groups = assign_decode_dp_groups(lengths, num_dp)
    assert groups == [0, 0, 0, 0]                # only group 0 exists
    # Every rank resolves the FULL set (replicated), none is empty.
    for r in range(world):
        owned = {i for i, g in enumerate(groups)
                 if rank_in_decode_group(g, r, G)}
        assert owned == set(range(len(lengths)))
    # MoE post-scatter share: 4 rows over 8 ranks -> ceil = 1.
    assert moe_ntp_from_group_max(len(lengths), G) == 1


# --------------------------------------------------------------------------- #
#  Piece 1b: host-KV owner = group leader (per-node SHARED region)            #
# --------------------------------------------------------------------------- #

def test_host_kv_owner_is_group_leader():
    """The single rank allowed to touch the shared host-KV region is g*G, and it
    is a member of the group (so it already holds the sequence's replica)."""
    G = 8
    for g in range(4):                       # world=32, num_dp=4
        leader = host_kv_owner_rank(g, G)
        assert leader == g * G
        # leader is inside the group's contiguous rank band
        assert rank_in_decode_group(g, leader, G)
        # exactly one rank per group is the owner: no OTHER group rank matches
        owners = [r for r in range(g * G, (g + 1) * G)
                  if r == host_kv_owner_rank(g, G)]
        assert owners == [leader]


def test_host_kv_owner_single_group_gate_config():
    """G=8 parity-gate: world=8, one group -> leader is rank 0, the other 7 ranks
    replicate the seq but MUST NOT release the shared entry (that is the
    double-free IndexError this gate reproduces)."""
    world, G = 8, 8
    assert host_kv_owner_rank(0, G) == 0
    non_leaders = [r for r in range(world) if r != host_kv_owner_rank(0, G)]
    assert non_leaders == [1, 2, 3, 4, 5, 6, 7]   # all skip the shared region
    with pytest.raises(ValueError):
        host_kv_owner_rank(0, 0)


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


# --------------------------------------------------------------------------- #
#  Piece 1c: legacy host-KV migration is disabled under G>1                    #
# --------------------------------------------------------------------------- #

def _bare_worker(G, *, rank=0, world_size=16):
    """A BatchGenWorker with ONLY the fields _plan_kv_migration reads."""
    return types.SimpleNamespace(
        rank=rank,
        local_rank=rank % 8,
        world_size=world_size,
        _decode_attn_tp_size=lambda: G,
    )


def test_plan_kv_migration_disabled_for_decode_tp_group(caplog):
    """G>1: the legacy planner keys from_rank/to_rank on the SINGLE assigned_rank,
    which under unified resident TP may hold no HostKVPageTable registration at
    all (the world16/G=8 IndexError). _plan_kv_migration must fail safe to [] and
    must not touch host-utilization / NCCL / the planner on the way there."""
    def _must_not_run(*args, **kwargs):
        raise AssertionError("legacy single-rank migration path ran under G>1")

    worker = _bare_worker(8, rank=0)
    worker._get_host_kv_utilization = _must_not_run
    worker._make_migration_plan_request = _must_not_run
    plan = _isolated_worker_method(
        "_plan_kv_migration",
        {
            "logging": logging,
            "dist": types.SimpleNamespace(all_gather_object=_must_not_run),
            "KVCacheManager": types.SimpleNamespace(
                plan_kv_migration=_must_not_run
            ),
        },
    ).__get__(worker)

    with caplog.at_level(logging.INFO):
        assert plan() == []
        assert plan() == []   # still safe on re-entry

    # One-shot: informative, and logged exactly once for the whole run.
    skips = [r for r in caplog.records
             if "skipping legacy single-rank" in r.getMessage()]
    assert len(skips) == 1, [r.getMessage() for r in caplog.records]
    assert "attn_tp_size=8" in skips[0].getMessage()


def test_plan_kv_migration_g1_still_plans():
    """NON-VACUITY control: with G==1 (validated pure-DP path) the guard is
    inert -- stats are gathered and the planner's decision is returned verbatim."""
    worker = _bare_worker(1, rank=0, world_size=2)
    worker._get_host_kv_utilization = lambda: {"node_id": 0}
    worker._make_migration_plan_request = lambda node_stats: node_stats

    def _fake_all_gather_object(object_list, obj, group=None):
        object_list[0] = {"node_id": 0}
        object_list[1] = {"node_id": 1}

    planned = [types.SimpleNamespace(
        uuid="deadbeefcafe", from_rank=0, to_rank=1, pages=3)]
    plan = _isolated_worker_method(
        "_plan_kv_migration",
        {
            "logging": logging,
            "dist": types.SimpleNamespace(
                all_gather_object=_fake_all_gather_object
            ),
            "KVCacheManager": types.SimpleNamespace(
                plan_kv_migration=lambda req: planned
            ),
        },
    ).__get__(worker)

    assert plan() is planned


# --------------------------------------------------------------------------- #
#  Piece 1d: ON_HOLD releases every locally replicated GPU-KV entry            #
# --------------------------------------------------------------------------- #

def test_put_on_hold_frees_tp_replica_on_non_assigned_rank():
    """A TP-group member owns local GPU KV even when assigned_rank points at a
    different rank. The local binding, not the legacy rank, controls release."""
    on_hold = object()
    calls = types.SimpleNamespace(freed=[], barriers=0)

    class _Seq:
        uuid = "u"
        global_idx = 17
        assigned_rank = 0
        decode_dp_group = 0
        gpu_pages_allocated = 12

        def log_event(self, *args, **kwargs):
            pass

    seq = _Seq()

    class _Batch:
        def get_sequence(self, uuid):
            assert uuid == "u"
            return seq

        def update_status(self, uuid, status):
            assert uuid == "u"
            assert status is on_hold

    class _Manager:
        _sequences = {17: object()}

        def free_pages_for_sequences(self, global_ids):
            calls.freed.append(list(global_ids))

    worker = types.SimpleNamespace(
        rank=1,
        global_batch=_Batch(),
        gpu_paged_kv_cache_manager=_Manager(),
        _uuid_to_local_map={"u": 3},
        _sequences_with_gpu_kv={"u"},
        _sync_sequence_metadata=lambda uuids: None,
    )
    method = _isolated_worker_method(
        "_put_sequences_on_hold",
        {
            "List": List,
            "logging": logging,
            "SequenceStatus": types.SimpleNamespace(ON_HOLD=on_hold),
            "SeqEvent": types.SimpleNamespace(ON_HOLD=object()),
            "dist": types.SimpleNamespace(
                barrier=lambda: setattr(calls, "barriers", calls.barriers + 1)
            ),
        },
    ).__get__(worker)

    method(["u"])

    assert calls.freed == [[17]]
    assert worker._sequences_with_gpu_kv == set()
    assert seq.gpu_pages_allocated == 0
    assert calls.barriers == 1


def _assignment_targets(method_name, value_name):
    """Names read by every assignment to `value_name` inside `method_name`."""
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    found = []
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == value_name for t in node.targets
        ):
            found.append({
                n.attr for n in ast.walk(node.value) if isinstance(n, ast.Attribute)
            })
    return found


def test_page_boundary_eviction_releases_host_kv_on_owner_only():
    """Host-KV eviction must filter by _owns_host_kv (single-owner invariant).

    Regression for the G=8 decode failure `IndexError: Sequence ID N not found
    during release`: `_page_boundary_fast` built `evicted_global_ids` from ALL
    locally-evicted uuids and called `release_sequence_pages` on every rank.
    Host KV is ONE shared per-node shm region keyed by global_idx, so under G>1
    the first releaser tombstones the entry and the other G-1 ranks raise.
    The validated release path already filters by `_owns_host_kv`; the eviction
    path must too. Only reachable once GPU-KV pressure makes eviction routine
    (a high decode admission cap), which is why 32-slot runs never hit it.
    """
    assignments = _assignment_targets("_page_boundary_fast", "evicted_global_ids")
    assert assignments, "evicted_global_ids assignment not found"
    for attrs in assignments:
        assert "_owns_host_kv" in attrs, (
            "evicted_global_ids feeds release_sequence_pages/unregister_sequences "
            "on the SHARED per-node host-KV region; it must be filtered by "
            "_owns_host_kv or G>1 ranks double-release"
        )
