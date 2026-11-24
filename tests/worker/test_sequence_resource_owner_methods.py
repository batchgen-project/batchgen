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

from batchgen.sequence import SequenceEntry

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
