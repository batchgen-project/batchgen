"""Unit tests for `batchgen.worker.decode.DecodeScheduler`.

Pure CPU tests over decode batch selection — no torch / NCCL / global_batch.
Admission uses live physical free pages; the old 90% value is not a safety
bound.
"""

from __future__ import annotations

import pytest

from batchgen.worker.decode import (
    DecodeBatchRequest,
    DecodeCandidate,
    DecodeCapacityError,
    DecodeScheduler,
    reduce_decode_capacity_snapshot,
    estimate_max_decode_replica_batch,
)


def _cand(uuid, *, rank=0, gidx=0, req_pages=10, decode_dp_group=None):
    return DecodeCandidate(
        uuid=uuid,
        assigned_rank=rank,
        global_idx=gidx,
        req_pages=req_pages,
        decode_dp_group=decode_dp_group,
    )


def _req(
    candidates,
    total_pages,
    world_size=8,
    attn_tp_size=1,
    existing_pages=(),
    free_pages=None,
    existing_sequence_counts=(),
):
    groups = world_size // attn_tp_size
    resident = tuple(existing_pages) if existing_pages else (0,) * groups
    if free_pages is None:
        free_pages = tuple(total_pages - pages for pages in resident)
    return DecodeBatchRequest(
        candidates=tuple(candidates),
        total_pages=total_pages,
        world_size=world_size,
        attn_tp_size=attn_tp_size,
        existing_pages=tuple(existing_pages),
        free_pages=tuple(free_pages),
        existing_sequence_counts=tuple(existing_sequence_counts),
    )


def test_empty_candidates_returns_empty():
    assert DecodeScheduler.select_decode_batch(_req([], 1000)) == []


def test_decode_replica_batch_estimate_accounts_for_tp_replication():
    assert estimate_max_decode_replica_batch(64, 16, 8) == 32
    assert estimate_max_decode_replica_batch(65, 16, 8) == 33
    # Pure DP preserves the legacy total/world_size estimate.
    assert estimate_max_decode_replica_batch(64, 16, 1) == 4


def test_single_candidate_fits():
    # Live free capacity is 100 pages; req 10 fits.
    plan = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=10)], 100))
    assert plan == ["a"]


def test_candidate_over_watermark_is_admitted_as_singleton():
    # req 91 still fits in the physical 100-page pool. The retired watermark
    # must not strand this candidate.
    plan = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=91)], 100))
    assert plan == ["a"]


def test_ninety_percent_watermark_boundary():
    # The old 900-page watermark is not a physical capacity bound.
    plan = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=900)], 1000))
    assert plan == ["a"]
    # 901 still fits physically and is admitted as the singleton.
    plan2 = DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=901)], 1000))
    assert plan2 == ["a"]


def test_candidate_over_physical_capacity_is_refused():
    with pytest.raises(DecodeCapacityError, match="more GPU KV pages"):
        DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=101)], 100))


def test_candidates_share_bucket_when_both_fit_physical_capacity():
    # The old singleton rule was an artifact of the 90% target. Both rows fit
    # in the live physical pool and may share the round.
    cands = [
        _cand("wide", gidx=0, req_pages=91),
        _cand("small", gidx=1, req_pages=1),
    ]
    assert DecodeScheduler.select_decode_batch(_req(cands, 100)) == ["wide", "small"]


def test_oversized_singleton_waits_behind_existing_decode_pages():
    # An active row already owns pages in the capacity bucket. The singleton
    # fallback must not overcommit the physical pool beside that row.
    cands = [_cand("wide", req_pages=91)]
    assert DecodeScheduler.select_decode_batch(
        _req(cands, 100, existing_pages=(20, 0, 0, 0, 0, 0, 0, 0))
    ) == []


def test_candidate_fitting_live_free_pages_is_admitted_past_old_watermark():
    # Metadata says five pages are resident, while the allocator reports 95
    # live free pages. The physical snapshot, not the old 90% target, decides.
    assert DecodeScheduler.select_decode_batch(
        _req(
            [_cand("wide", req_pages=91)],
            100,
            existing_pages=(5, 0, 0, 0, 0, 0, 0, 0),
            free_pages=(95, 100, 100, 100, 100, 100, 100, 100),
        )
    ) == ["wide"]


def test_stale_metadata_cannot_override_live_free_pages():
    # A stale resident count must not allow a request to overrun the allocator.
    assert DecodeScheduler.select_decode_batch(
        _req(
            [_cand("wide", req_pages=20)],
            100,
            existing_pages=(5, 0, 0, 0, 0, 0, 0, 0),
            free_pages=(10, 100, 100, 100, 100, 100, 100, 100),
        )
    ) == []


def test_existing_decode_rows_consume_sequence_cap():
    # Existing IN_DECODE rows count against the padded batch limit before new
    # candidates are considered.
    request = DecodeBatchRequest(
        candidates=(_cand("new", req_pages=1),),
        total_pages=100,
        world_size=1,
        free_pages=(100,),
        max_rank_bsz=1,
        existing_sequence_counts=(1,),
    )
    assert DecodeScheduler.select_decode_batch(request) == []


def test_capacity_snapshot_reduces_tp_free_pages_and_rejects_total_mismatch():
    snapshot = reduce_decode_capacity_snapshot(
        (100, 100, 100, 100),
        (90, 70, 80, 95),
        world_size=4,
        attn_tp_size=2,
    )
    assert snapshot.total_pages == 100
    assert snapshot.free_pages == (70, 80)
    with pytest.raises(ValueError, match="total page count diverged"):
        reduce_decode_capacity_snapshot(
            (100, 99, 100, 100),
            (90, 70, 80, 95),
            world_size=4,
            attn_tp_size=2,
        )


def test_zero_page_pool_keeps_empty_selection():
    # A zero-page manager is an unsized/torn-down pool, not an oversized
    # sequence; preserve the existing empty-selection behavior.
    assert DecodeScheduler.select_decode_batch(_req([_cand("a", req_pages=1)], 0)) == []


def test_global_idx_ordering():
    # all on rank 0, physical capacity fits only 2 of 3 (each 40)
    cands = [
        _cand("c", gidx=2, req_pages=40),
        _cand("a", gidx=0, req_pages=40),
        _cand("b", gidx=1, req_pages=40),
    ]
    plan = DecodeScheduler.select_decode_batch(_req(cands, 100))
    # sorted by global_idx → a(0), b(1) fit (80), c(2) would be 120 > 100
    assert plan == ["a", "b"]


def test_per_rank_capacity_independent():
    # rank 0 and rank 1 each fill independently
    cands = [
        _cand("r0a", rank=0, gidx=0, req_pages=80),
        _cand("r0b", rank=0, gidx=1, req_pages=80),  # 160 > 100 → excluded
        _cand("r1a", rank=1, gidx=2, req_pages=80),
    ]
    plan = DecodeScheduler.select_decode_batch(_req(cands, 100))
    assert set(plan) == {"r0a", "r1a"}
    assert "r0b" not in plan


def test_tp_group_replicas_share_one_page_capacity():
    # Both candidates are replicated onto every rank of group 0. Although their
    # legacy assigned ranks differ, their cumulative 120 pages exceed the
    # physical capacity of 100 pages, so only the first candidate may enter.
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
    # Physical capacity is 100 → 3 fit (90), 4th would be 120.
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
