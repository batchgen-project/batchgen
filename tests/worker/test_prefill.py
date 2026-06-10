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
    PrefillWaveGateRequest,
)

_PAGE = 64
_BUF = 32
_GPN = 8  # gpus per node


def _cand(uuid, *, rank=0, evicted=False, gidx=0, decoded=0, prompt=100, budget=100000):
    return PrefillCandidate(
        uuid=uuid,
        assigned_rank=rank,
        is_evicted=evicted,
        global_idx=gidx,
        total_decoded_before_eviction=decoded,
        prompt_length=prompt,
        kv_token_budget=budget,
        page_size=_PAGE,
    )


def _prefix_cand(
    uuid,
    *,
    rank=0,
    gidx=0,
    prompt=4096,
    cached=0,
    page_ids=(),
    budget=100000,
):
    return PrefillCandidate(
        uuid=uuid,
        assigned_rank=rank,
        is_evicted=False,
        global_idx=gidx,
        total_decoded_before_eviction=0,
        prompt_length=prompt,
        kv_token_budget=budget,
        page_size=_PAGE,
        estimated_shared_prefix_tokens=cached,
        estimated_shared_prefix_page_ids=tuple(page_ids),
    )


def _req(
    candidates,
    per_node_free,
    *,
    chunk=128,
    gpus_per_node=_GPN,
    charge_shared_prefix_pages=False,
):
    return PrefillSelectionRequest(
        candidates=tuple(candidates),
        per_node_host_free=tuple(per_node_free),
        chunk_size=chunk,
        num_nodes=len(per_node_free),
        gpus_per_node=gpus_per_node,
        initial_gpu_page_buffer=_BUF,
        charge_shared_prefix_pages=charge_shared_prefix_pages,
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


def test_no_eviction_candidates_pure_queueing_order():
    cands = [_cand(f"q{i}", gidx=2 - i, prompt=100) for i in range(3)]  # gidx 2,1,0
    plan = PrefillScheduler.select_prefill_batch(_req(cands, [200]))
    # all fit (200 >= 3*34=102), order by global_idx ascending
    assert plan == ["q2", "q1", "q0"]  # uuids q2(gidx0), q1(gidx1), q0(gidx2)


def test_prefix_estimate_reduces_admission_pages():
    # Without prefix estimate, prompt 4096 needs 97 pages:
    # max(prompt + chunk = 4224, gpu_tokens = (65 + 32) * 64).
    # With a 3072-token page-aligned hit, the private charge drops to
    # ceil((6208 - 3072) / 64) = 49 pages, so two candidates fit in 98 pages.
    c0 = _prefix_cand("c0", gidx=0, prompt=4096, cached=3072)
    c1 = _prefix_cand("c1", gidx=1, prompt=4096, cached=3072)
    plan = PrefillScheduler.select_prefill_batch(_req([c0, c1], [98]))
    assert plan == ["c0", "c1"]
    plan = PrefillScheduler.select_prefill_batch(_req([c0, c1], [97]))
    assert plan == ["c0"]


def test_non_page_aligned_prefix_estimate_is_conservative():
    # A full-hit compute path may normalize to prompt_length - 1. Admission
    # must only credit fully page-aligned shared pages.
    c = _prefix_cand("c", prompt=4096, cached=4095)
    plan = PrefillScheduler.select_prefill_batch(_req([c], [33]))
    assert plan == []
    plan = PrefillScheduler.select_prefill_batch(_req([c], [34]))
    assert plan == ["c"]


def test_prefix_admission_charges_unique_shared_pages_when_requested():
    shared_pages = tuple((0, page_id) for page_id in range(48))
    c0 = _prefix_cand("c0", gidx=0, prompt=4096, cached=3072, page_ids=shared_pages)
    c1 = _prefix_cand("c1", gidx=1, prompt=4096, cached=3072, page_ids=shared_pages)

    plan = PrefillScheduler.select_prefill_batch(
        _req([c0, c1], [145], charge_shared_prefix_pages=True)
    )
    assert plan == ["c0"]

    plan = PrefillScheduler.select_prefill_batch(
        _req([c0, c1], [146], charge_shared_prefix_pages=True)
    )
    assert plan == ["c0", "c1"]


def test_prefix_admission_does_not_charge_shared_pages_against_free_capacity():
    shared_pages = tuple((0, page_id) for page_id in range(48))
    c0 = _prefix_cand("c0", gidx=0, prompt=4096, cached=3072, page_ids=shared_pages)
    c1 = _prefix_cand("c1", gidx=1, prompt=4096, cached=3072, page_ids=shared_pages)

    plan = PrefillScheduler.select_prefill_batch(
        _req([c0, c1], [98], charge_shared_prefix_pages=False)
    )
    assert plan == ["c0", "c1"]


def test_request_and_candidate_are_frozen():
    req = _req([_cand("a")], [34])
    with pytest.raises((AttributeError, Exception)):
        req.chunk_size = 1  # type: ignore[misc]
    c = _cand("a")
    with pytest.raises((AttributeError, Exception)):
        c.uuid = "b"  # type: ignore[misc]


def test_prefix_cache_wave_gate_allows_first_wave():
    req = PrefillWaveGateRequest(
        selected_count=1,
        prefix_cache_enabled=True,
        has_active_work=False,
        world_size=8,
    )
    assert PrefillScheduler.should_run_prefill_wave(req)


def test_prefix_cache_wave_gate_defers_small_wave_with_active_work():
    req = PrefillWaveGateRequest(
        selected_count=35,
        prefix_cache_enabled=True,
        has_active_work=True,
        world_size=8,
    )
    assert not PrefillScheduler.should_run_prefill_wave(req)


def test_prefix_cache_wave_gate_allows_large_wave_with_active_work():
    req = PrefillWaveGateRequest(
        selected_count=128,
        prefix_cache_enabled=True,
        has_active_work=True,
        world_size=8,
    )
    assert PrefillScheduler.should_run_prefill_wave(req)


def test_prefix_cache_wave_gate_does_not_change_non_prefix_cache_path():
    req = PrefillWaveGateRequest(
        selected_count=1,
        prefix_cache_enabled=False,
        has_active_work=True,
        world_size=8,
    )
    assert PrefillScheduler.should_run_prefill_wave(req)


def test_prefix_cache_wave_gate_uses_world_size_threshold():
    assert PrefillScheduler.min_prefix_cache_wave_sequences(world_size=1) == 128
    assert PrefillScheduler.min_prefix_cache_wave_sequences(world_size=32) == 512
