"""Admission arithmetic for the per-sequence KDA record reservation (方案甲).

The record pages join the SAME combined host budget as KV, exactly: the pure
selection charges them per candidate, the owner-demand preflight charges them
per owned sequence (zero-token requests included), and the shared-prefix
ledger never discounts them -- the record is per sequence, never shared.
"""

import math

from batchgen.worker.kv_plan import prefill_owner_demand_pages
from batchgen.worker.prefill import (
    PREFILL_CAP_HOST_KV_PAGES,
    PrefillCandidate,
    PrefillScheduler,
    PrefillSelectionRequest,
    compute_prefill_host_reservation,
)

_PAGE = 64
_BUF = 2
_GPN = 8
RECORD_PAGES = 256  # Kimi-K3 at U=8 on the H200 profile; any constant works


def _cand(uuid, *, prompt=100, gidx=0, shared=()):
    return PrefillCandidate(
        uuid=uuid,
        assigned_rank=0,
        node_id=0,
        is_evicted=False,
        global_idx=gidx,
        total_decoded_before_eviction=0,
        prompt_length=prompt,
        kv_token_budget=100000,
        page_size=_PAGE,
        shared_prefix_segments=tuple(shared),
    )


def _req(candidates, per_node_free, *, kda_record_pages=0):
    return PrefillSelectionRequest(
        candidates=tuple(candidates),
        per_node_host_free=tuple(per_node_free),
        chunk_size=128,
        num_nodes=len(per_node_free),
        gpus_per_node=_GPN,
        initial_gpu_page_buffer=_BUF,
        kda_record_pages=kda_record_pages,
    )


def _kv_pages(prompt=100):
    pages, _ = compute_prefill_host_reservation(
        prompt_length=prompt,
        kv_token_budget=100000,
        page_size=_PAGE,
        chunk_size=128,
        initial_gpu_page_buffer=_BUF,
    )
    return pages


def test_required_host_pages_includes_the_record_constant():
    candidate = _cand("a")
    kv_only = PrefillScheduler.required_host_pages(
        candidate, _req([candidate], [10_000])
    )
    with_record = PrefillScheduler.required_host_pages(
        candidate, _req([candidate], [10_000], kda_record_pages=RECORD_PAGES)
    )
    assert with_record == kv_only + RECORD_PAGES


def test_admission_refuses_when_budget_holds_kv_but_not_the_record():
    """The named refusal: free pages fit the KV reservation alone, so with
    the record charge the candidate must NOT be admitted -- a committed
    write may never discover the shortage later."""

    candidate = _cand("a")
    kv_only = _kv_pages()
    # Sanity: without a record the same budget admits the candidate.
    assert PrefillScheduler.select_prefill_batch(
        _req([candidate], [kv_only])
    ) == ["a"]
    assert PrefillScheduler.select_prefill_batch(
        _req([candidate], [kv_only], kda_record_pages=RECORD_PAGES)
    ) == []
    # One more page than KV+record admits it again -- the charge is exact.
    assert PrefillScheduler.select_prefill_batch(
        _req(
            [candidate],
            [kv_only + RECORD_PAGES],
            kda_record_pages=RECORD_PAGES,
        )
    ) == ["a"]


def test_blocked_candidate_names_the_record_inclusive_need():
    candidate = _cand("a")
    kv_only = _kv_pages()
    selection = PrefillScheduler.select_prefill_wave(
        _req([candidate], [kv_only], kda_record_pages=RECORD_PAGES)
    )
    record = selection.cap_record
    assert record.cap == PREFILL_CAP_HOST_KV_PAGES
    assert record.first_blocked_uuid == "a"
    assert record.first_blocked_needed == kv_only + RECORD_PAGES


def test_shared_prefix_ledger_never_discounts_the_record():
    """Two candidates sharing their whole prefix: the second row's marginal
    KV cost collapses to the ledger floor, but each still pays its own
    record -- records are per sequence, never shared."""

    from batchgen.worker.shared_prefix_charge import SharedPrefixLedger

    shared = ((7, _kv_pages() - 1),)
    first = _cand("a", gidx=0, shared=shared)
    second = _cand("b", gidx=1, shared=shared)
    req = _req([first, second], [10_000], kda_record_pages=RECORD_PAGES)
    ledger = SharedPrefixLedger()
    first_cost = PrefillScheduler.marginal_host_pages(first, req, ledger)
    ledger.charge(
        total_pages=_kv_pages(),
        segments=[(7, _kv_pages() - 1)],
    )
    second_cost = PrefillScheduler.marginal_host_pages(second, req, ledger)
    assert first_cost == _kv_pages() + RECORD_PAGES
    # The second row's KV collapses to the one private floor page, and the
    # record constant survives in full.
    assert second_cost == 1 + RECORD_PAGES


def test_owner_demand_counts_records_for_every_owned_sequence():
    tokens = [640, 0, 128]  # a fully attached prefix (0) still needs a record
    kv_demand = sum(math.ceil(t / _PAGE) for t in tokens)
    assert prefill_owner_demand_pages(
        tokens, page_size=_PAGE, pool_segment_pages=5
    ) == kv_demand + 5
    assert prefill_owner_demand_pages(
        tokens,
        page_size=_PAGE,
        pool_segment_pages=5,
        kda_record_pages_per_sequence=RECORD_PAGES,
    ) == kv_demand + 5 + RECORD_PAGES * len(tokens)


def test_zero_record_pages_changes_nothing():
    candidate = _cand("a")
    req_without = _req([candidate], [10_000])
    req_with = _req([candidate], [10_000], kda_record_pages=0)
    assert PrefillScheduler.required_host_pages(
        candidate, req_without
    ) == PrefillScheduler.required_host_pages(candidate, req_with)
    assert prefill_owner_demand_pages(
        [100], page_size=_PAGE, pool_segment_pages=0
    ) == prefill_owner_demand_pages(
        [100],
        page_size=_PAGE,
        pool_segment_pages=0,
        kda_record_pages_per_sequence=0,
    )
