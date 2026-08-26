"""Unit tests for `batchgen.worker.decode.DecodeScheduler`.

Pure CPU tests over the decode batch selection — no torch / NCCL /
global_batch. Capacity = int(total_pages * 0.9) per rank.
"""

from __future__ import annotations

import pytest

from batchgen.worker.decode import (
    DecodeBatchRequest,
    DecodeCandidate,
    DecodeScheduler,
)


def _cand(uuid, *, rank=0, gidx=0, req_pages=10, decode_dp_group=None):
    return DecodeCandidate(
        uuid=uuid,
        assigned_rank=rank,
        global_idx=gidx,
        req_pages=req_pages,
        decode_dp_group=decode_dp_group,
    )


def _req(candidates, total_pages, world_size=8, attn_tp_size=1):
    return DecodeBatchRequest(
        candidates=tuple(candidates),
        total_pages=total_pages,
        world_size=world_size,
        attn_tp_size=attn_tp_size,
    )


def test_empty_candidates_returns_empty():
    assert DecodeScheduler.select_decode_batch(_req([], 1000)) == []


def test_single_candidate_fits():
    # capacity = int(100*0.9) = 90; req 10 fits
    plan = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=10)], 100))
    assert plan == ["a"]


def test_candidate_too_big_for_watermark():
    # capacity = 90; req 91 → excluded
    plan = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=91)], 100))
    assert plan == []


def test_ninety_percent_watermark_boundary():
    # capacity = int(1000*0.9) = 900; exactly 900 fits
    plan = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=900)], 1000))
    assert plan == ["a"]
    # 901 does not
    plan2 = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=901)], 1000))
    assert plan2 == []


def test_global_idx_ordering():
    # all on rank 0, capacity fits only 2 of 3 (each 40, cap=90)
    cands = [
        _cand("c", gidx=2, req_pages=40),
        _cand("a", gidx=0, req_pages=40),
        _cand("b", gidx=1, req_pages=40),
    ]
    plan = DecodeScheduler.select_decode_batch(_req(cands, 100))
    # sorted by global_idx → a(0), b(1) fit (80), c(2) would be 120 > 90
    assert plan == ["a", "b"]


def test_per_rank_capacity_independent():
    # rank 0 and rank 1 each fill independently
    cands = [
        _cand("r0a", rank=0, gidx=0, req_pages=80),
        _cand("r0b", rank=0, gidx=1, req_pages=80),  # 160 > 90 → excluded
        _cand("r1a", rank=1, gidx=2, req_pages=80),
    ]
    plan = DecodeScheduler.select_decode_batch(_req(cands, 100))
    assert set(plan) == {"r0a", "r1a"}
    assert "r0b" not in plan


def test_tp_group_replicas_share_one_page_capacity():
    # Both candidates are replicated onto every rank of group 0. Although their
    # legacy assigned ranks differ, their cumulative 120 pages exceed the
    # per-rank capacity of 90 pages, so only the first candidate may enter.
    cands = [
        _cand("a", rank=0, gidx=0, req_pages=60, decode_dp_group=0),
        _cand("b", rank=1, gidx=1, req_pages=60, decode_dp_group=0),
        _cand("c", rank=8, gidx=2, req_pages=60, decode_dp_group=1),
    ]
    plan = DecodeScheduler.select_decode_batch(
        _req(cands, 100, world_size=16, attn_tp_size=8)
    )
    assert plan == ["a", "c"]

    # NON-VACUITY: pure DP still charges the two legacy ranks independently.
    assert DecodeScheduler.select_decode_batch(
        _req(cands[:2], 100, world_size=16, attn_tp_size=1)
    ) == ["a", "b"]


def test_greedy_fill_until_rank_full():
    cands = [_cand(f"s{i}", rank=0, gidx=i, req_pages=30) for i in range(5)]
    # cap = int(100*0.9)=90 → 3 fit (90), 4th would be 120
    plan = DecodeScheduler.select_decode_batch(_req(cands, 100))
    assert plan == ["s0", "s1", "s2"]


def test_zero_total_pages_admits_nothing():
    # capacity = 0; any positive req_pages excluded
    plan = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=1)], 0))
    assert plan == []


def test_request_and_candidate_are_frozen():
    req = _req([_cand("a")], 100)
    with pytest.raises((AttributeError, Exception)):
        req.total_pages = 1  # type: ignore[misc]
    c = _cand("a")
    with pytest.raises((AttributeError, Exception)):
        c.uuid = "b"  # type: ignore[misc]
