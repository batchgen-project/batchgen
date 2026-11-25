# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Unit tests for the ``SequenceEntry`` resource-scalar lifecycle methods.

Each method owns one group of per-sequence resource-scalar writes that the
worker and the engines previously assigned directly (pinned by
``test_sequence_resource_scalars_single_writer``).  These tests pin the exact
fields each method writes, the one observable write order (a held group is
saved before it is cleared; the eviction snapshot reads the decoded length,
not the re-entry length), and the fields each method must NOT touch.
"""

import pytest

from batchgen.sequence import HostReservationCommitError, SequenceEntry

PAGE = SequenceEntry.PAGE_SIZE


def _entry(**overrides):
    seq = SequenceEntry(
        uuid="u" * 32, global_idx=7, prompt_length=100, max_decode_length=50
    )
    for name, value in overrides.items():
        setattr(seq, name, value)
    return seq


# --------------------------------------------------------------------------- #
#  drop_decode_group                                                           #
# --------------------------------------------------------------------------- #

def test_drop_decode_group_saves_the_held_group_before_clearing():
    seq = _entry(decode_dp_group=3, evicted_from_dp_group=None)
    seq.drop_decode_group()
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (None, 3)


def test_drop_decode_group_keeps_the_record_when_no_group_is_held():
    # A redundant drop (already evicted, never re-grouped) must not erase the
    # earlier record.
    seq = _entry(decode_dp_group=None, evicted_from_dp_group=5)
    seq.drop_decode_group()
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (None, 5)


def test_drop_decode_group_saves_nothing_when_never_grouped():
    seq = _entry()
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (None, None)
    seq.drop_decode_group()
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (None, None)


# --------------------------------------------------------------------------- #
#  grow_host_kv                                                                #
# --------------------------------------------------------------------------- #

def test_grow_host_kv_increments_exactly_the_three_host_counters():
    seq = _entry(
        host_pages_allocated=2,
        host_owned_pages=1,
        host_token_capacity=2 * PAGE,
        gpu_pages_allocated=4,
    )
    seq.grow_host_kv(3)
    assert seq.host_token_capacity == 5 * PAGE
    assert seq.host_pages_allocated == 5
    assert seq.host_owned_pages == 4
    # Not a GPU-side or length write.
    assert seq.gpu_pages_allocated == 4
    assert seq.current_context_length == 100


# --------------------------------------------------------------------------- #
#  release_gpu_pages                                                           #
# --------------------------------------------------------------------------- #

def test_release_gpu_pages_zeroes_only_the_gpu_page_count():
    seq = _entry(
        gpu_pages_allocated=12,
        had_initial_gpu_reservation=True,
        host_pages_allocated=4,
        host_owned_pages=4,
        host_token_capacity=4 * PAGE,
    )
    seq.release_gpu_pages()
    assert seq.gpu_pages_allocated == 0
    # Unlike reset_gpu_allocation: a resumed sequence keeps its reservation
    # history, so the next load takes the extension buffer.
    assert seq.had_initial_gpu_reservation is True
    # Host counters are untouched (host KV is retained on hold).
    assert seq.host_pages_allocated == 4
    assert seq.host_owned_pages == 4
    assert seq.host_token_capacity == 4 * PAGE


def test_release_gpu_pages_is_idempotent():
    seq = _entry(gpu_pages_allocated=12)
    seq.release_gpu_pages()
    seq.release_gpu_pages()
    assert seq.gpu_pages_allocated == 0


# --------------------------------------------------------------------------- #
#  release_all_kv                                                              #
# --------------------------------------------------------------------------- #

def test_release_all_kv_zeroes_the_four_kv_counters_and_nothing_else():
    seq = _entry(
        gpu_pages_allocated=6,
        host_pages_allocated=4,
        host_owned_pages=3,
        host_token_capacity=4 * PAGE,
        decode_dp_group=1,
        evicted_from_dp_group=2,
        decoded_length=30,
        current_context_length=130,
    )
    seq.release_all_kv()
    assert seq.gpu_pages_allocated == 0
    assert seq.host_pages_allocated == 0
    assert seq.host_owned_pages == 0
    assert seq.host_token_capacity == 0
    # The eviction path keeps the group record and the lengths; they are
    # restamped by their own owners.
    assert seq.decode_dp_group == 1
    assert seq.evicted_from_dp_group == 2
    assert seq.decoded_length == 30
    assert seq.current_context_length == 130


# --------------------------------------------------------------------------- #
#  finish_and_release                                                          #
# --------------------------------------------------------------------------- #

def test_finish_and_release_zeroes_kv_and_clears_the_saved_group():
    seq = _entry(
        gpu_pages_allocated=6,
        host_pages_allocated=4,
        host_owned_pages=3,
        host_token_capacity=4 * PAGE,
        decode_dp_group=1,
        evicted_from_dp_group=2,
        decoded_length=30,
        current_context_length=130,
    )
    seq.finish_and_release()
    assert seq.gpu_pages_allocated == 0
    assert seq.host_pages_allocated == 0
    assert seq.host_owned_pages == 0
    assert seq.host_token_capacity == 0
    assert seq.evicted_from_dp_group is None
    # Completion does not touch the live group binding or the lengths.
    assert seq.decode_dp_group == 1
    assert seq.decoded_length == 30
    assert seq.current_context_length == 130


# --------------------------------------------------------------------------- #
#  begin_eviction                                                              #
# --------------------------------------------------------------------------- #

def test_begin_eviction_restamps_the_three_length_scalars():
    seq = _entry(decoded_length=30, current_context_length=130)
    assert seq.prompt_length == 100
    seq.begin_eviction(120)
    # The snapshot reads the decoded length (30), never the re-entry length.
    assert seq.total_decoded_before_eviction == 30
    assert seq.prompt_length == 120
    assert seq.current_context_length == 120
    # Untouched: the caller computes the re-entry length from these.
    assert seq.decoded_length == 30
    assert seq.reentry_decoded_baseline == 0
    assert seq.original_prompt_length == 100
    # No resource counter is written here; the KV release is a separate call.
    seq2 = _entry(decoded_length=30, gpu_pages_allocated=6, host_pages_allocated=4)
    seq2.begin_eviction(120)
    assert seq2.gpu_pages_allocated == 6
    assert seq2.host_pages_allocated == 4


# --------------------------------------------------------------------------- #
#  commit_host_reservation                                                     #
# --------------------------------------------------------------------------- #

def test_commit_host_reservation_stamps_exactly_the_three_host_counters():
    seq = _entry(
        host_pages_allocated=1,
        host_owned_pages=1,
        host_token_capacity=PAGE,
        gpu_pages_allocated=4,
        decoded_length=30,
        current_context_length=130,
    )
    seq.commit_host_reservation(7, 7 * PAGE, 5)
    assert seq.host_pages_allocated == 7
    assert seq.host_token_capacity == 7 * PAGE
    assert seq.host_owned_pages == 5
    # Not a GPU-side or length write.
    assert seq.gpu_pages_allocated == 4
    assert seq.decoded_length == 30
    assert seq.current_context_length == 130


def test_commit_host_reservation_migration_form_owns_every_page():
    # The migration rebase re-stamps all three from the destination's
    # allocation: (p, p * PAGE_SIZE, p).
    seq = _entry(host_pages_allocated=3, host_owned_pages=2,
                 host_token_capacity=3 * PAGE)
    seq.commit_host_reservation(9, 9 * PAGE, 9)
    assert (seq.host_pages_allocated, seq.host_owned_pages) == (9, 9)
    assert seq.host_token_capacity == 9 * PAGE


def test_commit_host_reservation_refuses_a_broken_capacity_identity():
    seq = _entry(host_pages_allocated=3, host_owned_pages=2,
                 host_token_capacity=3 * PAGE)
    with pytest.raises(HostReservationCommitError, match="tokens="):
        seq.commit_host_reservation(7, 7 * PAGE - 1, 7)
    # Refused before any write: the commit is all-or-nothing.
    assert seq.host_pages_allocated == 3
    assert seq.host_owned_pages == 2
    assert seq.host_token_capacity == 3 * PAGE


def test_commit_host_reservation_refuses_owning_more_than_allocated():
    seq = _entry(host_pages_allocated=3, host_owned_pages=2,
                 host_token_capacity=3 * PAGE)
    with pytest.raises(HostReservationCommitError, match="exceeds pages"):
        seq.commit_host_reservation(7, 7 * PAGE, 8)
    assert seq.host_pages_allocated == 3
    assert seq.host_owned_pages == 2
    assert seq.host_token_capacity == 3 * PAGE


# --------------------------------------------------------------------------- #
#  share_owned_pages                                                           #
# --------------------------------------------------------------------------- #

def test_share_owned_pages_deducts_only_the_owned_count():
    seq = _entry(host_pages_allocated=5, host_owned_pages=5,
                 host_token_capacity=5 * PAGE)
    seq.share_owned_pages(2, where="unit share")
    assert seq.host_owned_pages == 3
    # The chain allocation and capacity are not what sharing moves.
    assert seq.host_pages_allocated == 5
    assert seq.host_token_capacity == 5 * PAGE


def test_share_owned_pages_zero_share_still_passes_the_invariant():
    # A rank that shares nothing (a non-owner at reservation, an empty retain at prefix commit)
    # still goes through HOST_OWNED_PAGE, so an empty chain is refused.
    seq = _entry(host_pages_allocated=4, host_owned_pages=4)
    seq.share_owned_pages(0, where="unit zero share")
    assert seq.host_owned_pages == 4

    empty = _entry(host_pages_allocated=0, host_owned_pages=0)
    with pytest.raises(RuntimeError, match=r"HOST_OWNED_PAGE broken at unit zero"):
        empty.share_owned_pages(0, where="unit zero share on empty chain")


def test_share_owned_pages_refuses_sharing_the_whole_chain():
    seq = _entry(host_pages_allocated=3, host_owned_pages=3)
    with pytest.raises(RuntimeError, match=r"\[SCHED_INVARIANT\] HOST_OWNED_PAGE"):
        seq.share_owned_pages(3, where="unit full share")
    # Refused before the write.
    assert seq.host_owned_pages == 3


# --------------------------------------------------------------------------- #
#  grant_gpu_pages                                                             #
# --------------------------------------------------------------------------- #

def test_grant_gpu_pages_initial_stamps_the_total_and_marks_the_reservation():
    seq = _entry(
        host_pages_allocated=4,
        host_owned_pages=4,
        host_token_capacity=4 * PAGE,
        decoded_length=30,
        current_context_length=130,
    )
    assert seq.had_initial_gpu_reservation is False
    seq.grant_gpu_pages(6, initial=True)
    assert seq.gpu_pages_allocated == 6
    assert seq.had_initial_gpu_reservation is True
    # Not a host-side or length write.
    assert seq.host_pages_allocated == 4
    assert seq.host_owned_pages == 4
    assert seq.host_token_capacity == 4 * PAGE
    assert seq.decoded_length == 30
    assert seq.current_context_length == 130


def test_grant_gpu_pages_extension_writes_the_total_and_keeps_the_flag():
    # The page-boundary extension commits the new TOTAL (not a delta) and
    # never touches the reservation flag, in either state.
    seq = _entry(gpu_pages_allocated=6, had_initial_gpu_reservation=True)
    seq.grant_gpu_pages(8, initial=False)
    assert seq.gpu_pages_allocated == 8
    assert seq.had_initial_gpu_reservation is True

    fresh = _entry(gpu_pages_allocated=2)
    fresh.grant_gpu_pages(3, initial=False)
    assert fresh.gpu_pages_allocated == 3
    assert fresh.had_initial_gpu_reservation is False


def test_grant_gpu_pages_marks_after_the_scalar_write():
    # Inside the method the scalar is committed BEFORE the reservation is
    # marked, the statement order of every site it replaced; any sizing that
    # reads the flag therefore happens in the caller, before the call.
    seen = []

    class _Observed(SequenceEntry):
        def mark_initial_gpu_reservation_done(self):
            seen.append(self.gpu_pages_allocated)
            super().mark_initial_gpu_reservation_done()

    seq = _Observed(
        uuid="u" * 32, global_idx=7, prompt_length=100, max_decode_length=50
    )
    seq.grant_gpu_pages(5, initial=True)
    assert seen == [5]


def test_grant_gpu_pages_redundant_initial_restamp_is_idempotent():
    # Decode entry re-stamps a sequence the allocator already granted: the
    # recomputed total overwrites and the flag simply stays set.
    seq = _entry()
    seq.grant_gpu_pages(6, initial=True)
    seq.grant_gpu_pages(6, initial=True)
    assert seq.gpu_pages_allocated == 6
    assert seq.had_initial_gpu_reservation is True


# --------------------------------------------------------------------------- #
#  rebase_decoded_length                                                       #
# --------------------------------------------------------------------------- #

def test_rebase_decoded_length_restores_the_context_identity():
    seq = _entry(decoded_length=5, current_context_length=105)
    seq.rebase_decoded_length(42)
    assert seq.decoded_length == 42
    # ccl == original_prompt_length + dl, from the ORIGINAL prompt length.
    assert seq.current_context_length == 100 + 42
    # Not a prompt or resource write.
    assert seq.prompt_length == 100
    assert seq.original_prompt_length == 100
    assert seq.host_pages_allocated == 0
    assert seq.gpu_pages_allocated == 0


def test_rebase_decoded_length_uses_original_prompt_after_reentry():
    # After an eviction cycle prompt_length holds the re-entry length; the
    # identity is anchored on original_prompt_length, exactly as the
    # migration-receive site computed it.
    seq = _entry(decoded_length=30, current_context_length=130)
    seq.begin_eviction(120)
    seq.rebase_decoded_length(30)
    assert seq.prompt_length == 120
    assert seq.current_context_length == 100 + 30
