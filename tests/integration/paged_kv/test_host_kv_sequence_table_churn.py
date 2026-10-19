"""The host-KV sequence table must survive churn and must never leak pages.

`ReleaseSequence` turns a sequence's slot into a tombstone, and nothing ever
turns a tombstone back into an empty slot. A linear probe that ends only on an
empty slot therefore starts refusing every insert once the table has been
walked over a single time, although every slot is reusable. `AcquirePages`
pops its pages off the free stack before it inserts the sequence entry, so each
of those refusals also stranded the popped pages permanently.

Both halves are covered here: a churn loop that must never be refused and must
stay page-neutral, and a genuinely full table whose refusal must still return
the pages the allocator had already popped.
"""

from __future__ import annotations

import sys

import pytest


if sys.platform != "linux":
    pytest.skip(
        "host paged KV backend is Linux-only", allow_module_level=True
    )

try:
    from batchgen.models.engine_loader import core_engine as bg
except ImportError:  # compiled extension not built in this environment
    pytest.skip("core_engine extension is unavailable", allow_module_level=True)


# Deliberately tiny: an 8-slot table is churned over several times in a loop
# short enough to stay a unit-speed test, and 64 pages keep every acquire
# satisfiable so a refusal can only come from the sequence table.
SEQUENCE_TABLE_CAPACITY = 8
PAGE_SIZE_TOKENS = 4


def _make_config(num_pages: int):
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = "/batchgen_kv_seq_table_churn"
    cfg.num_layers = 1
    cfg.num_pages = num_pages
    cfg.page_size_tokens = PAGE_SIZE_TOKENS
    cfg.num_k_heads = 1
    cfg.k_head_dim = 8
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = 2
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = SEQUENCE_TABLE_CAPACITY
    cfg.alignment_bytes = 64
    return cfg


def test_sequence_table_reuses_tombstones_across_churn():
    """Insert/release cycles far past capacity: no refusal, no page loss."""
    iterations = 6 * SEQUENCE_TABLE_CAPACITY
    manager = bg.MLAHostPagedKVManager(_make_config(num_pages=64))
    try:
        manager.initialize(True)
        total_pages = manager.get_stats().num_total_pages
        assert total_pages == 64

        for i in range(iterations):
            sequence_id = 1000 + i
            # A tombstoned slot is reusable, so this must never be refused.
            manager.allocate_pages(sequence_id, 1)
            manager.free_sequence(sequence_id)

            stats = manager.get_stats()
            assert stats.num_free_pages == total_pages, i
            assert stats.num_active_sequences == 0, i
    finally:
        del manager


def test_full_sequence_table_returns_the_pages_it_popped():
    """A refused insert must not keep the pages AcquirePages already popped."""
    manager = bg.MLAHostPagedKVManager(_make_config(num_pages=16))
    try:
        manager.initialize(True)
        live = [7000 + i for i in range(SEQUENCE_TABLE_CAPACITY)]
        for sequence_id in live:
            manager.allocate_pages(sequence_id, 1)

        before = manager.get_stats()
        assert before.num_active_sequences == SEQUENCE_TABLE_CAPACITY
        assert before.num_free_pages == 16 - SEQUENCE_TABLE_CAPACITY

        # Every slot holds a live sequence, so this insert is impossible for a
        # real reason. The pages popped for it must come back regardless.
        with pytest.raises(RuntimeError, match="Sequence table is full"):
            manager.allocate_pages(99999, 1)

        assert manager.get_stats().num_free_pages == before.num_free_pages

        # The restored pages must be allocatable again, not merely counted.
        # Freeing one slot returns its page and makes the table insertable, so
        # the retried acquire consumes exactly that page.
        manager.free_sequence(live[0])
        manager.allocate_pages(99999, 1)
        assert manager.get_stats().num_free_pages == before.num_free_pages
    finally:
        del manager


def test_audit_tracks_tombstones_and_lifetime_counters_across_churn():
    """The monitoring view of the same churn: tombstones accumulate, empty
    slots only ever fall, the page identity holds, and the lifetime counters
    match the operations performed."""
    iterations = 6 * SEQUENCE_TABLE_CAPACITY
    manager = bg.MLAHostPagedKVManager(_make_config(num_pages=64))
    try:
        manager.initialize(True)
        assert manager.get_stats().audit is None  # the O(1) path never walks
        previous_empty = SEQUENCE_TABLE_CAPACITY
        for i in range(iterations):
            sequence_id = 3000 + i
            manager.allocate_pages(sequence_id, 1)
            manager.free_sequence(sequence_id)

            stats = manager.get_audited_stats()
            audit = stats.audit
            assert (
                audit.live_entries + audit.tombstone_entries + audit.empty_entries
                == SEQUENCE_TABLE_CAPACITY
            ), i
            assert audit.empty_entries <= previous_empty, i
            previous_empty = audit.empty_entries
            assert audit.page_identity_delta == 0, i
            assert stats.sequences_inserted == i + 1, i
            assert stats.sequences_released == i + 1, i
        assert audit.live_entries == 0
        assert audit.tombstone_entries > 0
        assert audit.empty_entries < SEQUENCE_TABLE_CAPACITY
        assert stats.acquire_failures == 0
    finally:
        del manager


def test_a_refused_acquire_is_counted_and_leaves_the_identity_intact():
    manager = bg.MLAHostPagedKVManager(_make_config(num_pages=16))
    try:
        manager.initialize(True)
        for i in range(SEQUENCE_TABLE_CAPACITY):
            manager.allocate_pages(8000 + i, 1)
        with pytest.raises(RuntimeError, match="Sequence table is full"):
            manager.allocate_pages(99999, 2 * PAGE_SIZE_TOKENS)

        stats = manager.get_audited_stats()
        assert stats.acquire_failures == 1
        assert stats.sequences_inserted == SEQUENCE_TABLE_CAPACITY
        assert stats.audit.live_entries == SEQUENCE_TABLE_CAPACITY
        assert stats.audit.empty_entries == 0
        assert stats.audit.live_sequence_pages == SEQUENCE_TABLE_CAPACITY
        assert stats.audit.page_identity_delta == 0
    finally:
        del manager
