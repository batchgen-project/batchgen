"""Unit tests for `batchgen.worker.prefill.PrefillScheduler`.

Pure CPU tests — the scheduler reads only frozen snapshots, no torch /
NCCL / global_batch. Capacity math constants used below (page_size=64,
initial_gpu_page_buffer=32):

    a prompt_length=100 candidate needs:
      post = 101; gpu_pages = ceil(101/64)+32 = 2+32 = 34
      gpu_tokens = 34*64 = 2176
      capacity  = max(100+chunk, 2176) capped at kv_token_budget
      → with chunk=128, budget>=2176: capacity=2176, req_pages=34
"""

from __future__ import annotations

import pytest

from batchgen.worker.prefill import (
    PrefillCandidate,
    PrefillScheduler,
    PrefillSelectionRequest,
)

_PAGE = 64
_BUF = 32
_GPN = 8  # gpus per node


def _cand(
    uuid,
    *,
    rank=0,
    node=None,
    evicted=False,
    gidx=0,
    decoded=0,
    prompt=100,
    budget=100000,
    host_kv_replication_factor=1,
):
    return PrefillCandidate(
        uuid=uuid,
        assigned_rank=rank,
        node_id=rank // _GPN if node is None else node,
        is_evicted=evicted,
        global_idx=gidx,
        total_decoded_before_eviction=decoded,
        prompt_length=prompt,
        kv_token_budget=budget,
        page_size=_PAGE,
        host_kv_replication_factor=host_kv_replication_factor,
    )


def _req(
    candidates,
    per_node_free,
    *,
    chunk=128,
    gpus_per_node=_GPN,
    per_rank_sequence_free=None,
    per_node_sequence_free=None,
):
    return PrefillSelectionRequest(
        candidates=tuple(candidates),
        per_node_host_free=tuple(per_node_free),
        chunk_size=chunk,
        num_nodes=len(per_node_free),
        gpus_per_node=gpus_per_node,
        initial_gpu_page_buffer=_BUF,
        per_rank_sequence_free=(
            tuple(per_rank_sequence_free)
            if per_rank_sequence_free is not None else None
        ),
        per_node_sequence_free=(
            tuple(per_node_sequence_free)
            if per_node_sequence_free is not None else None
        ),
    )


def test_empty_candidates_returns_empty():
    assert PrefillScheduler.select_prefill_batch(_req([], [100])) == []


def test_single_candidate_fits_exactly():
    # prompt 100 → 34 pages; node free 34 → fits
    plan = PrefillScheduler.select_prefill_batch(_req([_cand("a", prompt=100)], [34]))
    assert plan == ["a"]


def test_single_candidate_one_page_short():
    # 34 pages needed, only 33 free → not selected
    plan = PrefillScheduler.select_prefill_batch(_req([_cand("a", prompt=100)], [33]))
    assert plan == []


def test_kv_token_budget_caps_capacity():
    # budget 200 → initial_capacity = min(2176, 200) = 200 → req_pages = ceil(200/64)=4
    plan = PrefillScheduler.select_prefill_batch(
        _req([_cand("a", prompt=100, budget=200)], [4])
    )
    assert plan == ["a"]
    # one page short of the capped requirement
    plan2 = PrefillScheduler.select_prefill_batch(
        _req([_cand("a", prompt=100, budget=200)], [3])
    )
    assert plan2 == []


def test_chunk_size_dominates_when_large():
    # chunk 4096 → prompt+chunk = 4196 > gpu_tokens 2176 → capacity 4196
    # req_pages = ceil(4196/64) = 66
    plan = PrefillScheduler.select_prefill_batch(
        _req([_cand("a", prompt=100)], [66], chunk=4096)
    )
    assert plan == ["a"]
    plan2 = PrefillScheduler.select_prefill_batch(
        _req([_cand("a", prompt=100)], [65], chunk=4096)
    )
    assert plan2 == []


def test_evicted_admitted_before_queueing():
    # node free fits exactly one 34-page candidate
    q = _cand("q", evicted=False, gidx=0, prompt=100)
    e = _cand("e", evicted=True, gidx=1, decoded=50, prompt=100)
    plan = PrefillScheduler.select_prefill_batch(_req([q, e], [34]))
    assert plan == ["e"]  # evicted wins the single slot


def test_evicted_ordered_most_decoded_first():
    e1 = _cand("e1", evicted=True, gidx=0, decoded=10, prompt=100)
    e2 = _cand("e2", evicted=True, gidx=1, decoded=50, prompt=100)
    plan = PrefillScheduler.select_prefill_batch(_req([e1, e2], [34]))
    assert plan == ["e2"]  # 50 > 10 decoded → higher priority


def test_evicted_tiebreak_global_idx():
    e1 = _cand("e1", evicted=True, gidx=5, decoded=50, prompt=100)
    e2 = _cand("e2", evicted=True, gidx=2, decoded=50, prompt=100)
    plan = PrefillScheduler.select_prefill_batch(_req([e1, e2], [34]))
    assert plan == ["e2"]  # equal decoded → lower global_idx


def test_queueing_ordered_by_global_idx():
    q1 = _cand("q1", gidx=5, prompt=100)
    q2 = _cand("q2", gidx=2, prompt=100)
    plan = PrefillScheduler.select_prefill_batch(_req([q1, q2], [34]))
    assert plan == ["q2"]


def test_per_node_capacity_independent():
    # node0 (rank 0) and node1 (rank 8) each have room for one
    c0 = _cand("c0", rank=0, prompt=100)
    c1 = _cand("c1", rank=8, prompt=100)
    plan = PrefillScheduler.select_prefill_batch(_req([c0, c1], [34, 34]))
    assert set(plan) == {"c0", "c1"}


def test_full_node_blocks_only_its_sequences():
    c0 = _cand("c0", rank=0, prompt=100)   # node 0
    c1 = _cand("c1", rank=8, prompt=100)   # node 1, no free pages
    plan = PrefillScheduler.select_prefill_batch(_req([c0, c1], [34, 0]))
    assert plan == ["c0"]


def test_greedy_fill_until_node_exhausted():
    # node free for two 34-page candidates = 68; third is dropped
    cands = [_cand(f"q{i}", gidx=i, prompt=100) for i in range(3)]
    plan = PrefillScheduler.select_prefill_batch(_req(cands, [68]))
    assert plan == ["q0", "q1"]


def test_tp8_host_kv_replication_is_charged_to_node_capacity():
    # Each candidate needs 34 pages per rank.  A TP8 serve group consumes
    # 8 * 34 = 272 pages from the node-level host KV allocator.
    cands = [
        _cand(
            f"q{i}",
            gidx=i,
            prompt=100,
            host_kv_replication_factor=8,
        )
        for i in range(2)
    ]
    assert PrefillScheduler.select_prefill_batch(_req(cands, [271])) == []
    assert PrefillScheduler.select_prefill_batch(_req(cands, [272])) == ["q0"]
    assert PrefillScheduler.select_prefill_batch(_req(cands, [544])) == [
        "q0",
        "q1",
    ]


def test_host_kv_replication_factor_must_be_positive():
    with pytest.raises(ValueError, match="host_kv_replication_factor=0"):
        PrefillScheduler.select_prefill_batch(
            _req(
                [_cand("q0", host_kv_replication_factor=0)],
                [1000],
            )
        )


def test_no_eviction_candidates_pure_queueing_order():
    cands = [_cand(f"q{i}", gidx=2 - i, prompt=100) for i in range(3)]  # gidx 2,1,0
    plan = PrefillScheduler.select_prefill_batch(_req(cands, [200]))
    # all fit (200 >= 3*34=102), order by global_idx ascending
    assert plan == ["q2", "q1", "q0"]  # uuids q2(gidx0), q1(gidx1), q0(gidx2)


def test_persistent_kda_limit_is_per_rank_when_requested():
    cands = [
        _cand("r0-a", rank=0, gidx=0),
        _cand("r0-b", rank=0, gidx=1),
        _cand("r0-c", rank=0, gidx=2),
        _cand("r1-a", rank=1, gidx=3),
    ]
    plan = PrefillScheduler.select_prefill_batch(
        _req(cands, [200], per_rank_sequence_free=[2, 2])
    )
    assert plan == ["r0-a", "r0-b", "r1-a"]


def test_persistent_kda_limit_is_per_node_for_tp8_group():
    cands = [
        _cand("n0-a", rank=0, gidx=0),
        _cand("n0-b", rank=1, gidx=1),
        _cand("n0-c", rank=7, gidx=2),
        _cand("n1-a", rank=8, gidx=3),
    ]
    plan = PrefillScheduler.select_prefill_batch(
        _req(cands, [200, 200], per_node_sequence_free=[2, 2])
    )
    assert plan == ["n0-a", "n0-b", "n1-a"]


def test_tp8_capacity_uses_group_node_not_legacy_assigned_rank():
    # The legacy rank balancer can assign all requests to ranks 0..15 while
    # TP8 decode groups place four requests on each physical node. Admission
    # must follow the latter or a 16-request W2 batch is split unnecessarily.
    cands = [
        _cand(f"q{i}", rank=i, node=i % 4, gidx=i)
        for i in range(16)
    ]
    plan = PrefillScheduler.select_prefill_batch(
        _req(
            cands,
            [1000, 1000, 1000, 1000],
            per_node_sequence_free=[4, 4, 4, 4],
        )
    )
    assert plan == [f"q{i}" for i in range(16)]


def test_asymmetric_node_slot_capacity_is_applied_from_gathered_vector():
    cands = [
        _cand("n0", rank=0, node=0, gidx=0),
        _cand("n1-a", rank=0, node=1, gidx=1),
        _cand("n1-b", rank=0, node=1, gidx=2),
    ]
    plan = PrefillScheduler.select_prefill_batch(
        _req(
            cands,
            [200, 200],
            per_node_sequence_free=[0, 2],
        )
    )
    assert plan == ["n1-a", "n1-b"]


def test_persistent_kda_limits_cannot_have_two_scopes():
    with pytest.raises(ValueError, match="scoped to rank or node"):
        PrefillScheduler.select_prefill_batch(
            _req(
                [_cand("a")],
                [34],
                per_rank_sequence_free=[1],
                per_node_sequence_free=[1],
            )
        )


def test_request_and_candidate_are_frozen():
    req = _req([_cand("a")], [34])
    with pytest.raises((AttributeError, Exception)):
        req.chunk_size = 1  # type: ignore[misc]
    c = _cand("a")
    with pytest.raises((AttributeError, Exception)):
        c.uuid = "b"  # type: ignore[misc]
