"""
Unit tests for the GLM-5 128K single-sequence KV capacity-abort guard.

Issue: batchgen-project/batchgen-internal#1

These tests cover the pieces that don't require spinning up a real worker:
- SequenceEntry.get_admission_pages_required() arithmetic
- _finish_capacity slot exists and defaults to False
- BoundaryDecisions carries the new capacity_aborted_uuids field
- The capacity-guard constants exist with the expected default values
"""

import math

import pytest

from batchgen.sequence import (
    SINGLE_SEQ_PAGE_HEADROOM,
    SequenceEntry,
)
from batchgen.continuous_batching import BoundaryDecisions


def test_capacity_constants_have_sensible_defaults():
    # Tight headroom (8 pages = 512 tokens) absorbs page-table /
    # two-page-buffer overhead. This is the only safety slack — the cap
    # itself is just num_total_pages (no percentage margin).
    assert SINGLE_SEQ_PAGE_HEADROOM == 8


def test_get_admission_pages_required_uses_kv_token_budget():
    # 128K decode + 2K prompt in real binary tokens (131_072) — matches the
    # issue's "128K completion requires 2048 pages before prompt tokens".
    seq = SequenceEntry(uuid="u", global_idx=0, prompt_length=2048, max_decode_length=131_072)
    expected = math.ceil((2048 + 131_072) / SequenceEntry.PAGE_SIZE) + SINGLE_SEQ_PAGE_HEADROOM
    assert seq.get_admission_pages_required() == expected
    # The 128K-over-cap regression: 2080 pages of payload + 8 headroom.
    assert seq.get_admission_pages_required() > 2048


def test_get_admission_pages_required_short_sequence_does_not_explode():
    seq = SequenceEntry(uuid="u", global_idx=0, prompt_length=128, max_decode_length=64)
    # ceil(192/64) + 8 = 11
    assert seq.get_admission_pages_required() == 11


def test_finish_capacity_defaults_to_false():
    seq = SequenceEntry(uuid="u", global_idx=0, prompt_length=64, max_decode_length=64)
    assert seq._finish_capacity is False


def test_finish_capacity_is_settable_and_persistent():
    seq = SequenceEntry(uuid="u", global_idx=0, prompt_length=64, max_decode_length=64)
    seq._finish_capacity = True
    assert seq._finish_capacity is True


def test_boundary_decisions_default_capacity_aborted_uuids_is_empty():
    d = BoundaryDecisions(
        completed_uuids=[],
        active_uuids=[],
        host_growth_uuids=[],
        host_growth_pages=[],
        growth_feasible=False,
        host_evicted_uuids=[],
        onhold_uuids=[],
        seqs_needing_extension=[],
        new_load_uuids=[],
        decode_uuids_final=[],
    )
    assert d.capacity_aborted_uuids == []


def test_boundary_decisions_carries_capacity_aborted_uuids():
    d = BoundaryDecisions(
        completed_uuids=["a", "b"],
        active_uuids=[],
        host_growth_uuids=[],
        host_growth_pages=[],
        growth_feasible=False,
        host_evicted_uuids=[],
        onhold_uuids=[],
        seqs_needing_extension=[],
        new_load_uuids=[],
        decode_uuids_final=[],
        capacity_aborted_uuids=["a", "b"],
    )
    assert d.capacity_aborted_uuids == ["a", "b"]
    # Capacity-aborted uuids must always also appear in completed_uuids so
    # the existing release/report path picks them up.
    assert set(d.capacity_aborted_uuids).issubset(set(d.completed_uuids))


def test_glm5_128k_request_triggers_abort_threshold():
    """Regression for the run failure: a 128K decode on a 2011-page rank.

    The capacity-abort guard compares get_admission_pages_required() to
    safe_single_seq_per_rank_capacity = num_total_pages (no margin).
    A 2K-prompt + 128K-decode request needs ceil((2048+131072)/64) + 8 = 2088
    pages. On a 2088-page rank it just fits; on the failing run's 2011-page
    rank it does NOT, so the guard fires.
    """
    seq = SequenceEntry(uuid="u", global_idx=0, prompt_length=2048, max_decode_length=131_072)
    required = seq.get_admission_pages_required()
    assert required == 2088, f"expected 2088 admission pages, got {required}"

    # On a comfortably-large rank, no abort.
    assert required <= 2200, "should fit on a 2200-page rank"

    # On the failing run's actual rank (2011 / 1765 pages), guard must fire.
    for total_pages_failing in (2011, 1765):
        assert required > total_pages_failing, (
            f"with safe_cap = {total_pages_failing}, admission {required} "
            f"must exceed cap for the guard to fire"
        )
