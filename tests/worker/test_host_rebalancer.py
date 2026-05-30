"""Unit tests for `batchgen.worker.host_rebalancer.HostKVRebalancer`.

Pure CPU tests over the round-grouping decision. Uses the real
``MigrationOp`` (imports cleanly, no JIT). Topology: 8 GPUs/node, so
ranks 0-7 = node 0, ranks 8-15 = node 1, ranks 16-23 = node 2.
"""

from __future__ import annotations

from batchgen.migration import MigrationOp
from batchgen.worker.host_rebalancer import HostKVRebalancer

_GPN = 8


def _mig(uuid, from_rank, to_rank, pages=10):
    return MigrationOp(
        uuid=uuid, from_rank=from_rank, to_rank=to_rank,
        pages=pages, host_pages=pages,
    )


def _group(migrations):
    return HostKVRebalancer.group_migrations_for_parallel_execution(migrations, _GPN)


def test_empty_returns_empty():
    assert _group([]) == []


def test_disjoint_ranks_and_nodes_single_round():
    # node0→node1 and node2→node3: no shared rank, distinct src nodes
    m1 = _mig("a", 0, 8)    # src node 0 → dst node 1
    m2 = _mig("b", 16, 24)  # src node 2 → dst node 3
    rounds = _group([m1, m2])
    assert len(rounds) == 1
    assert rounds[0] == [m1, m2]


def test_shared_from_rank_splits_rounds():
    m1 = _mig("a", 0, 8)
    m2 = _mig("b", 0, 16)  # same from_rank 0
    rounds = _group([m1, m2])
    assert len(rounds) == 2
    assert rounds[0] == [m1]
    assert rounds[1] == [m2]


def test_shared_to_rank_splits_rounds():
    m1 = _mig("a", 0, 16)
    m2 = _mig("b", 8, 16)  # same to_rank 16
    rounds = _group([m1, m2])
    assert len(rounds) == 2
    assert rounds[0] == [m1]
    assert rounds[1] == [m2]


def test_same_source_node_splits_rounds():
    # from_rank 0 and 1 are both on node 0 → cannot run in same round
    m1 = _mig("a", 0, 16)
    m2 = _mig("b", 1, 24)
    rounds = _group([m1, m2])
    assert len(rounds) == 2
    assert rounds[0] == [m1]
    assert rounds[1] == [m2]


def test_from_rank_collides_with_other_to_rank():
    # m1 uses ranks {0, 8}; m2's from_rank 8 collides with m1's to_rank
    m1 = _mig("a", 0, 8)
    m2 = _mig("b", 8, 16)
    rounds = _group([m1, m2])
    assert len(rounds) == 2
    assert rounds[0] == [m1]
    assert rounds[1] == [m2]


def test_multiple_parallel_across_distinct_nodes():
    # three migrations, each from a distinct node to a distinct rank,
    # no rank reuse → all parallel in one round
    m1 = _mig("a", 0, 8)    # node0 → rank8
    m2 = _mig("b", 16, 9)   # node2 → rank9
    m3 = _mig("c", 24, 10)  # node3 → rank10
    rounds = _group([m1, m2, m3])
    assert len(rounds) == 1
    assert rounds[0] == [m1, m2, m3]


def test_greedy_order_preserving_packing():
    # m1(node0), m2(node0 again→next round), m3(node1→packs with m1)
    m1 = _mig("a", 0, 8)    # src node0, ranks {0,8}
    m2 = _mig("b", 1, 16)   # src node0 (rank1) → conflicts node0 with m1
    m3 = _mig("c", 9, 17)   # src node1, ranks {9,17} → ok with m1
    rounds = _group([m1, m2, m3])
    # round 0: m1 (node0), then m3 (node1, no rank overlap). m2 deferred (node0 reused).
    assert rounds[0] == [m1, m3]
    assert rounds[1] == [m2]


def test_all_serialized_when_all_share_source_node():
    migs = [_mig(f"m{i}", i, 16 + i) for i in range(3)]  # ranks 0,1,2 all node 0
    rounds = _group(migs)
    assert len(rounds) == 3
    assert [r[0].uuid for r in rounds] == ["m0", "m1", "m2"]


def test_every_migration_appears_exactly_once():
    migs = [_mig(f"m{i}", i, 100 + i) for i in range(5)]  # ranks 0..4 (node0) + distinct dsts
    rounds = _group(migs)
    flat = [m.uuid for r in rounds for m in r]
    assert sorted(flat) == [f"m{i}" for i in range(5)]
    assert len(flat) == 5  # no dupes, no drops
