"""Shared-prefix-aware admission accounting — charge, release and arithmetic.

Pure CPU. The identity under test is that a prefix several sequences read from
ONE physical copy is charged ONCE per resource, so prefix sharing raises the
number of admitted sequences instead of only saving compute.

The nqa64k shape anchors the arithmetic: 512 rows = 64 books x 8 questions, a
64k shared prefix per book, Host KV 700 GB = 126,787 pages, GPU pool 3,645
pages per rank.
"""

from __future__ import annotations

import pytest

from batchgen.prefix_reuse.admission import (
    plan_admission_shared_segments,
    shared_pages_per_row,
)
from batchgen.worker.decode import (
    DecodeBatchRequest,
    DecodeCandidate,
    DecodeScheduler,
)
from batchgen.worker.prefill import (
    PrefillCandidate,
    PrefillScheduler,
    PrefillSelectionRequest,
    compute_prefill_host_reservation,
)
from batchgen.worker.shared_prefix_charge import (
    SharedPrefixChargeError,
    SharedPrefixLedger,
    SharedPrefixRefcountError,
    SharedSegment,
    plan_shared_runs,
)

# ---------------------------------------------------------------- nqa64k shape

PAGE_SIZE = 64
CHUNK_SIZE = 2048
GPU_PAGE_BUFFER = 32
MAX_DECODE = 4096

BOOKS = 64
QUESTIONS_PER_BOOK = 8
HOST_POOL_PAGES = 126_787          # 700 GB of Host KV
GPU_POOL_PAGES = 3_645             # per rank

SHARED_PREFIX_PAGES = 1_002        # the shared book, page aligned
QUESTION_TOKENS = 88
PROMPT_LENGTH = SHARED_PREFIX_PAGES * PAGE_SIZE + QUESTION_TOKENS
KV_TOKEN_BUDGET = PROMPT_LENGTH + MAX_DECODE


def host_reservation_pages() -> int:
    pages, _tokens = compute_prefill_host_reservation(
        prompt_length=PROMPT_LENGTH,
        kv_token_budget=KV_TOKEN_BUDGET,
        page_size=PAGE_SIZE,
        chunk_size=CHUNK_SIZE,
        initial_gpu_page_buffer=GPU_PAGE_BUFFER,
    )
    return pages


def test_nqa64k_reservation_is_the_audited_page_count():
    """The per-row reservation the admission walk charges, stated once."""
    assert host_reservation_pages() == 1_036
    # Tail = what a row costs once its book is already paid for.
    assert host_reservation_pages() - SHARED_PREFIX_PAGES == 34


# ------------------------------------------------------------------- 1. ledger


def test_first_reader_pays_the_prefix_and_later_readers_pay_the_tail():
    ledger = SharedPrefixLedger()
    segments = (SharedSegment(key=7, pages=SHARED_PREFIX_PAGES),)
    total = host_reservation_pages()

    first = ledger.charge(total_pages=total, segments=segments)
    assert first.marginal_pages == total
    assert first.credited_pages == 0
    assert first.first_reader_keys == (7,)

    for _ in range(QUESTIONS_PER_BOOK - 1):
        later = ledger.charge(total_pages=total, segments=segments)
        assert later.marginal_pages == total - SHARED_PREFIX_PAGES
        assert later.credited_pages == SHARED_PREFIX_PAGES
        assert later.first_reader_keys == ()

    assert ledger.refcount(7) == QUESTIONS_PER_BOOK
    assert ledger.charged_pages() == SHARED_PREFIX_PAGES


def test_preview_does_not_charge_so_a_skipped_row_leaves_no_credit():
    ledger = SharedPrefixLedger()
    segments = (SharedSegment(key=1, pages=100),)
    ledger.preview(total_pages=200, segments=segments)
    assert not ledger.is_charged(1)
    # The next row is therefore still a first reader and pays in full.
    assert ledger.charge(
        total_pages=200, segments=segments
    ).marginal_pages == 200


def test_pages_come_back_only_at_the_last_reader():
    ledger = SharedPrefixLedger()
    segments = (SharedSegment(key=5, pages=300),)
    for _ in range(3):
        ledger.charge(total_pages=400, segments=segments)

    assert ledger.release(segments) == 0   # two readers left
    assert ledger.release(segments) == 0   # one reader left
    assert ledger.refcount(5) == 1
    assert ledger.release(segments) == 300  # last reader frees the copy
    assert ledger.refcount(5) == 0
    assert ledger.charged_pages() == 0


def test_resident_prefix_is_free_to_join_and_its_holder_is_never_released():
    """A prefix the resource already holds costs a joining row nothing."""
    ledger = SharedPrefixLedger(resident={9: SHARED_PREFIX_PAGES})
    segments = (SharedSegment(key=9, pages=SHARED_PREFIX_PAGES),)
    total = host_reservation_pages()

    joined = ledger.charge(total_pages=total, segments=segments)
    assert joined.marginal_pages == total - SHARED_PREFIX_PAGES

    assert ledger.release(segments) == 0
    # The holder's own reference remains: releasing it would free pages the
    # prefix cache still owns.
    with pytest.raises(SharedPrefixRefcountError, match="holder's own"):
        ledger.release(segments)


def test_release_without_a_charge_fails_loudly():
    ledger = SharedPrefixLedger()
    with pytest.raises(SharedPrefixRefcountError, match="never charged"):
        ledger.release((SharedSegment(key=3, pages=10),))


def test_rollback_undoes_every_charge_a_failed_second_phase_took():
    """A rejected reservation must leave the ledger as the wave found it."""
    ledger = SharedPrefixLedger()
    kept = (SharedSegment(key=1, pages=50),)
    ledger.charge(total_pages=80, segments=kept)
    savepoint = ledger.savepoint()

    # Second phase: charge a wave, then have the allocator reject it.
    speculative = (SharedSegment(key=2, pages=70),)
    ledger.charge(total_pages=100, segments=speculative)
    ledger.charge(total_pages=100, segments=speculative)
    ledger.charge(total_pages=80, segments=kept)
    assert ledger.refcount(2) == 2
    assert ledger.refcount(1) == 2

    ledger.rollback(savepoint)
    assert ledger.refcount(1) == 1
    assert ledger.refcount(2) == 0
    assert ledger.charged_pages() == 50
    # The rolled-back segment is a first reader again, not a leaked charge.
    assert ledger.charge(
        total_pages=100, segments=speculative
    ).marginal_pages == 100


def test_a_fully_shared_row_still_pays_one_private_page():
    """No row is ever free: the last page holds the token it produces."""
    ledger = SharedPrefixLedger()
    segments = (SharedSegment(key=4, pages=100),)
    ledger.charge(total_pages=100, segments=segments)
    second = ledger.charge(total_pages=100, segments=segments)
    assert second.marginal_pages == 1
    assert second.credited_pages == 99


def test_a_segment_cannot_claim_two_page_counts_in_one_reservation():
    ledger = SharedPrefixLedger()
    with pytest.raises(SharedPrefixChargeError, match="appears with"):
        ledger.charge(
            total_pages=100,
            segments=(
                SharedSegment(key=1, pages=10),
                SharedSegment(key=1, pages=20),
            ),
        )


def test_a_zero_page_segment_is_refused():
    with pytest.raises(SharedPrefixChargeError, match=">= 1 page"):
        SharedSegment(key=1, pages=0)


# --------------------------------------------------- 2. in-wave segment plan


def test_segment_plan_finds_one_shared_run_per_book():
    """Rows of a book share one run; rows of different books share nothing."""
    page_tokens = 4
    prompts = []
    for book in range(3):
        body = list(range(1000 * (book + 1), 1000 * (book + 1) + 12))
        for question in range(4):
            prompts.append(body + [90_000 + question, 90_001 + question])

    chains = plan_admission_shared_segments(prompts, page_tokens)
    assert len(chains) == 12
    # Each row shares its book's three whole pages (12 tokens / 4).
    assert shared_pages_per_row(chains) == (3,) * 12
    keys = [chain[-1][0] for chain in chains]
    # One key per book, equal within a book and distinct across books.
    assert len(set(keys)) == 3
    assert keys[0] == keys[1] == keys[2] == keys[3]
    assert keys[0] != keys[4]


def test_segment_plan_gives_a_singleton_prompt_no_chain():
    chains = plan_admission_shared_segments(
        [[1, 2, 3, 4, 5], [9, 9, 9, 9, 9]], 2
    )
    assert chains == ((), ())


def test_segment_plan_is_identical_for_identical_input():
    prompts = [[1, 2, 3, 4, 7, 7], [1, 2, 3, 4, 8, 8], [5, 6, 1, 2, 3, 3]]
    assert plan_admission_shared_segments(
        prompts, 2
    ) == plan_admission_shared_segments(prompts, 2)


# ------------------------------------------- 3. site A: host wave arithmetic


def _nqa64k_prefill_request(*, shared: bool) -> PrefillSelectionRequest:
    candidates = []
    for book in range(BOOKS):
        for question in range(QUESTIONS_PER_BOOK):
            index = book * QUESTIONS_PER_BOOK + question
            candidates.append(
                PrefillCandidate(
                    uuid=f"seq-{index:04d}",
                    assigned_rank=0,
                    node_id=0,
                    is_evicted=False,
                    global_idx=index,
                    total_decoded_before_eviction=0,
                    prompt_length=PROMPT_LENGTH,
                    kv_token_budget=KV_TOKEN_BUDGET,
                    page_size=PAGE_SIZE,
                    # Prefix-order stamping keeps a book's rows contiguous.
                    prefill_order=index,
                    shared_prefix_segments=(
                        ((book, SHARED_PREFIX_PAGES),) if shared else ()
                    ),
                )
            )
    return PrefillSelectionRequest(
        candidates=tuple(candidates),
        per_node_host_free=(HOST_POOL_PAGES,),
        chunk_size=CHUNK_SIZE,
        num_nodes=1,
        gpus_per_node=8,
        initial_gpu_page_buffer=GPU_PAGE_BUFFER,
    )


def test_per_row_charge_admits_one_wave_of_122_rows():
    """The defect, stated as arithmetic: the pool looks 8x fuller than it is."""
    admitted = PrefillScheduler.select_prefill_batch(
        _nqa64k_prefill_request(shared=False)
    )
    assert len(admitted) == HOST_POOL_PAGES // host_reservation_pages() == 122


def test_shared_charge_admits_the_whole_batch_in_one_wave():
    admitted = PrefillScheduler.select_prefill_batch(
        _nqa64k_prefill_request(shared=True)
    )
    assert len(admitted) == BOOKS * QUESTIONS_PER_BOOK == 512

    # Closed numerically: one prefix copy per book plus one tail per row.
    tail = host_reservation_pages() - SHARED_PREFIX_PAGES
    charged = BOOKS * (
        host_reservation_pages() + (QUESTIONS_PER_BOOK - 1) * tail
    )
    assert charged == 81_536
    assert charged <= HOST_POOL_PAGES


def test_shared_charge_still_respects_the_pool_bound():
    """Credit raises admission; it does not lift the capacity bound."""
    base = _nqa64k_prefill_request(shared=True)
    tiny = PrefillSelectionRequest(
        candidates=base.candidates,
        per_node_host_free=(host_reservation_pages() + 3,),
        chunk_size=base.chunk_size,
        num_nodes=1,
        gpus_per_node=8,
        initial_gpu_page_buffer=base.initial_gpu_page_buffer,
    )
    # Room for the first book's copy and nothing else: 3 spare pages cannot
    # cover a 34-page tail.
    assert len(PrefillScheduler.select_prefill_batch(tiny)) == 1


def test_required_host_pages_stays_the_standalone_cost():
    """Rejection, reclaim and blocker text must keep asking about one row."""
    req = _nqa64k_prefill_request(shared=True)
    for candidate in req.candidates:
        assert (
            PrefillScheduler.required_host_pages(candidate, req)
            == host_reservation_pages()
        )


def test_prefill_selection_is_identical_across_ranks():
    """Pure and deterministic: every rank derives the same wave."""
    req = _nqa64k_prefill_request(shared=True)
    shuffled = PrefillSelectionRequest(
        candidates=tuple(reversed(req.candidates)),
        per_node_host_free=req.per_node_host_free,
        chunk_size=req.chunk_size,
        num_nodes=1,
        gpus_per_node=8,
        initial_gpu_page_buffer=req.initial_gpu_page_buffer,
    )
    first = PrefillScheduler.select_prefill_batch(req)
    assert first == PrefillScheduler.select_prefill_batch(req)
    # The snapshot's own order must not change the decision: admission order
    # comes from prefill_order, not from how the worker listed candidates.
    assert first == PrefillScheduler.select_prefill_batch(shuffled)


def test_a_second_node_does_not_reuse_the_first_nodes_prefix_copy():
    """Host KV is per-node: a prefix on node 0 saves nothing on node 1."""
    candidates = tuple(
        PrefillCandidate(
            uuid=f"seq-{index}",
            assigned_rank=index,
            node_id=index,
            is_evicted=False,
            global_idx=index,
            total_decoded_before_eviction=0,
            prompt_length=PROMPT_LENGTH,
            kv_token_budget=KV_TOKEN_BUDGET,
            page_size=PAGE_SIZE,
            prefill_order=index,
            shared_prefix_segments=((0, SHARED_PREFIX_PAGES),),
        )
        for index in range(2)
    )
    req = PrefillSelectionRequest(
        candidates=candidates,
        # Each node can hold exactly one full copy, not a credited tail.
        per_node_host_free=(host_reservation_pages(),) * 2,
        chunk_size=CHUNK_SIZE,
        num_nodes=2,
        gpus_per_node=1,
        initial_gpu_page_buffer=GPU_PAGE_BUFFER,
    )
    assert len(PrefillScheduler.select_prefill_batch(req)) == 2


# ----------------------------------------- 4. site B: decode GPU arithmetic


def _decode_request(*, shared: bool, rows: int) -> DecodeBatchRequest:
    req_pages, _tokens = compute_prefill_host_reservation(
        prompt_length=PROMPT_LENGTH,
        kv_token_budget=KV_TOKEN_BUDGET,
        page_size=PAGE_SIZE,
        chunk_size=CHUNK_SIZE,
        initial_gpu_page_buffer=GPU_PAGE_BUFFER,
    )
    candidates = tuple(
        DecodeCandidate(
            uuid=f"seq-{index}",
            assigned_rank=0,
            global_idx=index,
            req_pages=req_pages,
            shared_prefix_segments=(
                ((4242, SHARED_PREFIX_PAGES),) if shared else ()
            ),
        )
        for index in range(rows)
    )
    return DecodeBatchRequest(
        candidates=candidates,
        total_pages=GPU_POOL_PAGES,
        world_size=1,
        free_pages=(GPU_POOL_PAGES,),
    )


def test_per_row_decode_charge_caps_a_rank_at_three_rows():
    admitted = DecodeScheduler.select_decode_batch(
        _decode_request(shared=False, rows=QUESTIONS_PER_BOOK)
    )
    assert len(admitted) == GPU_POOL_PAGES // 1_036 == 3


def test_shared_decode_charge_admits_every_row_of_one_book():
    admitted = DecodeScheduler.select_decode_batch(
        _decode_request(shared=True, rows=QUESTIONS_PER_BOOK)
    )
    assert len(admitted) == QUESTIONS_PER_BOOK == 8

    tail = 1_036 - SHARED_PREFIX_PAGES
    charged = 1_036 + (QUESTIONS_PER_BOOK - 1) * tail
    assert charged == 1_274
    assert charged <= GPU_POOL_PAGES


def test_resident_shared_pages_credit_a_row_joining_a_mapped_prefix():
    """A prefix a decoding sibling already mapped costs a joiner nothing."""
    ledger = SharedPrefixLedger()
    candidate = DecodeCandidate(
        uuid="seq-0",
        assigned_rank=0,
        global_idx=0,
        req_pages=1_036,
        resident_shared_pages=SHARED_PREFIX_PAGES,
    )
    assert DecodeScheduler.marginal_req_pages(candidate, ledger) == 34


def test_decode_credit_is_the_max_not_the_sum_of_both_sources():
    """Both credits describe a PREFIX of one page list; summing would lie."""
    ledger = SharedPrefixLedger()
    segments = ((4242, SHARED_PREFIX_PAGES),)
    first = DecodeCandidate(
        uuid="seq-0",
        assigned_rank=0,
        global_idx=0,
        req_pages=1_036,
        shared_prefix_segments=segments,
        resident_shared_pages=500,
    )
    # Resident covers 500 pages; the run is not charged yet, so the credit is
    # the larger of the two, never 1502.
    assert DecodeScheduler.marginal_req_pages(first, ledger) == 1_036 - 500
    ledger.charge(total_pages=1_036, segments=segments)
    assert DecodeScheduler.marginal_req_pages(first, ledger) == 34


def test_decode_selection_is_identical_across_ranks():
    req = _decode_request(shared=True, rows=QUESTIONS_PER_BOOK)
    first = DecodeScheduler.select_decode_batch(req)
    assert first == DecodeScheduler.select_decode_batch(req)


def test_separate_capacity_groups_each_pay_for_their_own_copy():
    candidates = tuple(
        DecodeCandidate(
            uuid=f"seq-{index}",
            assigned_rank=index,
            global_idx=index,
            req_pages=1_036,
            shared_prefix_segments=((4242, SHARED_PREFIX_PAGES),),
        )
        for index in range(2)
    )
    req = DecodeBatchRequest(
        candidates=candidates,
        total_pages=1_036,
        world_size=2,
        free_pages=(1_036, 1_036),
    )
    # Each rank's pool holds exactly one full copy; the credit must not let
    # rank 1 ride on rank 0's pages.
    assert len(DecodeScheduler.select_decode_batch(req)) == 2


# ------------------------------------------------- 5. host page id grouping


def test_shared_runs_come_from_physical_host_page_identity():
    """Rows share a run exactly when their Host page tables agree."""
    tables = {
        10: [1, 2, 3, 40],
        11: [1, 2, 3, 41],
        12: [1, 2, 3, 42],
        20: [7, 8, 90],
        21: [7, 8, 91],
        30: [500, 501],
    }
    runs = plan_shared_runs(tables)
    assert runs[10] == runs[11] == runs[12] == (3, 3)
    assert runs[20] == runs[21] == (8, 2)
    # A row sharing no first page shares nothing.
    assert 30 not in runs


def test_equal_length_prefix_on_distinct_physical_pages_is_not_shared():
    """Two Host copies of the same tokens are two copies, not one."""
    runs = plan_shared_runs({1: [10, 11, 12], 2: [20, 21, 22]})
    assert runs == {}
