"""End-to-end migration-pipeline tests with *fabricated* inputs.

This is the payoff of the worker decouple: the migration **planner**
(`KVCacheManager.plan_kv_migration`, Slice 5.6) and the **round grouping**
(`HostKVRebalancer.group_migrations_for_parallel_execution`, Slice 7) are
now pure functions over frozen snapshots — so the full host-KV migration
decision path can be exercised here with synthetic multi-node imbalance,
with NO server, NO NCCL, and NO real host-KV pressure.

Each test:
  1. builds a fabricated `MigrationPlanRequest` (imbalanced node stats +
     candidate sequences),
  2. runs `plan_kv_migration` → a `MigrationOp` list,
  3. runs `group_migrations_for_parallel_execution` → parallel rounds,
  4. asserts the plan rebalances toward target AND every round is
     conflict-free (no rank or source-node reused within a round).
"""

from __future__ import annotations

from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.kv_manager import (
    KVCacheManager,
    MigrationCandidate,
    MigrationPlanRequest,
)

_GPN = 8


def _node(used, total=400):
    return {"num_used_pages": used, "num_total_pages": total}


def _cand(uuid, rank, gidx, host_pages, budget=2048):
    return MigrationCandidate(
        uuid=uuid,
        assigned_rank=rank,
        global_idx=gidx,
        kv_token_budget=budget,
        host_pages_allocated=host_pages,
    )


def _assert_rounds_conflict_free(rounds):
    """No rank (src or dst) and no source node reused within a round."""
    seen_uuids = set()
    for rnd in rounds:
        ranks = set()
        src_nodes = set()
        for m in rnd:
            assert m.from_rank not in ranks, "from_rank reused in round"
            assert m.to_rank not in ranks, "to_rank reused in round"
            src_node = m.from_rank // _GPN
            assert src_node not in src_nodes, "source node reused in round"
            ranks.add(m.from_rank)
            ranks.add(m.to_rank)
            src_nodes.add(src_node)
            seen_uuids.add(m.uuid)
    return seen_uuids


def _plan_and_group(node_stats, candidates, world_size):
    req = MigrationPlanRequest(
        node_stats=node_stats,
        candidates=tuple(candidates),
        num_gpus_per_node=_GPN,
        world_size=world_size,
    )
    migrations = KVCacheManager.plan_kv_migration(req)
    rounds = HostKVRebalancer.group_migrations_for_parallel_execution(migrations, _GPN)
    return migrations, rounds


def test_two_overloaded_nodes_group_into_one_parallel_round():
    """node0 + node1 overloaded → node2; the two moves come from distinct
    source nodes with no shared rank → a single parallel round."""
    node_stats = {0: _node(70), 1: _node(70), 2: _node(10)}
    cands = [
        _cand("a", rank=0, gidx=0, host_pages=20),  # node 0
        _cand("b", rank=8, gidx=1, host_pages=20),  # node 1
    ]
    migrations, rounds = _plan_and_group(node_stats, cands, world_size=24)

    # target = (70+70+10)/3 = 50; each node sheds one 20-page seq → 50.
    assert len(migrations) == 2
    movers = {m.uuid for m in migrations}
    assert movers == {"a", "b"}
    # both destined for node 2 (the only underutilized node), distinct ranks
    assert all(m.to_rank // _GPN == 2 for m in migrations)
    assert len({m.to_rank for m in migrations}) == 2
    # distinct source nodes + no rank overlap → one parallel round
    assert len(rounds) == 1
    assert _assert_rounds_conflict_free(rounds) == {"a", "b"}


def test_same_source_node_serializes_into_multiple_rounds():
    """All movers sit on node0 → source-node exclusivity forces one per round."""
    node_stats = {0: _node(150), 1: _node(0)}
    cands = [
        _cand("a", rank=0, gidx=0, host_pages=20),
        _cand("b", rank=1, gidx=1, host_pages=20),
        _cand("c", rank=2, gidx=2, host_pages=20),
    ]
    migrations, rounds = _plan_and_group(node_stats, cands, world_size=16)

    # target = 75; node0 150 → shed until <=75: 4×20 would be needed but only
    # 3 candidates exist → 3 migrations (150→90 after 3).
    assert len(migrations) == 3
    # all from node 0 → every round holds exactly one migration
    assert all(len(rnd) == 1 for rnd in rounds)
    assert len(rounds) == 3
    seen = _assert_rounds_conflict_free(rounds)
    assert seen == {"a", "b", "c"}


def test_balanced_cluster_plans_nothing():
    node_stats = {0: _node(50), 1: _node(50)}
    migrations, rounds = _plan_and_group(node_stats, [], world_size=16)
    assert migrations == []
    assert rounds == []


def test_smallest_budget_migrated_first_then_grouped():
    """Planner picks smallest kv_token_budget first; grouping preserves them."""
    node_stats = {0: _node(80), 1: _node(0)}
    cands = [
        _cand("big", rank=0, gidx=0, host_pages=15, budget=4096),
        _cand("small", rank=1, gidx=1, host_pages=15, budget=512),
    ]
    migrations, rounds = _plan_and_group(node_stats, cands, world_size=16)
    # target = 40; node0 80 → shed until <=40: 15+15=30 → 80-30=50>40,
    # need a 3rd but only 2 candidates → both migrate.
    assert {m.uuid for m in migrations} == {"big", "small"}
    # "small" (budget 512) is selected first by the planner
    assert migrations[0].uuid == "small"
    # both from node 0 → serialized into 2 rounds, conflict-free
    assert len(rounds) == 2
    assert _assert_rounds_conflict_free(rounds) == {"big", "small"}
