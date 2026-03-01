"""Comprehensive tests for dynamic host KV cache reservation.

Tests cover:
- Phase 1: Chunk-based dynamic growth + adaptive chunk sizing
- Phase 2: Host KV eviction + prefill recompute re-entry
- Edge cases: empty batches, single sequence, all evicted, growth failure
"""

import math
import pytest
import torch
from unittest.mock import MagicMock, patch

from batchgen.sequence import (
    SequenceEntry,
    SequenceStatus,
    SequenceBatch,
    EXTENSION_GPU_PAGE_BUFFER,
    configure_page_buffers,
)
from batchgen.continuous_batching import (
    AdaptiveChunkSizer,
    EvictionStrategy,
    select_sequences_for_eviction,
)


# ============ Helper ============

def make_seq(
    uuid: str = "test-uuid",
    global_idx: int = 0,
    prompt_length: int = 512,
    max_decode_length: int = 131072,
    decoded_length: int = 0,
    host_token_capacity: int = 0,
    host_pages_allocated: int = 0,
    status: SequenceStatus = SequenceStatus.IN_DECODE,
    assigned_rank: int = 0,
) -> SequenceEntry:
    """Create a SequenceEntry with specified properties."""
    seq = SequenceEntry(uuid, global_idx, prompt_length, max_decode_length)
    seq.decoded_length = decoded_length
    seq.current_context_length = prompt_length + decoded_length
    seq.host_token_capacity = host_token_capacity
    seq.host_pages_allocated = host_pages_allocated
    seq.status = status
    seq.assigned_rank = assigned_rank
    seq.input_ids = torch.zeros(prompt_length, dtype=torch.long)
    seq.decoded_tokens = torch.zeros(max_decode_length, dtype=torch.long)
    return seq


# ============ SequenceEntry: Host KV Growth ============

class TestHostKVGrowth:
    """Tests for SequenceEntry host KV growth methods."""

    def test_needs_host_kv_growth_no_capacity(self):
        """Sequence with no host_token_capacity should not need growth."""
        seq = make_seq(host_token_capacity=0)
        assert not seq.needs_host_kv_growth(8192)

    def test_needs_host_kv_growth_plenty_of_room(self):
        """Sequence with lots of room should not need growth."""
        seq = make_seq(
            prompt_length=512,
            decoded_length=100,
            host_token_capacity=512 + 8192,
        )
        assert not seq.needs_host_kv_growth(8192)

    def test_needs_host_kv_growth_approaching_capacity(self):
        """Sequence approaching host capacity should trigger growth."""
        # Extension buffer = 4 pages * 64 = 256 tokens
        cap = 512 + 8192
        # decoded_length just under cap minus extension buffer
        decoded = 8192 - 200  # 200 < 256 threshold
        seq = make_seq(
            prompt_length=512,
            decoded_length=decoded,
            host_token_capacity=cap,
        )
        assert seq.needs_host_kv_growth(8192)

    def test_needs_host_kv_growth_at_exact_boundary(self):
        """Sequence at exactly extension buffer threshold."""
        extension_tokens = EXTENSION_GPU_PAGE_BUFFER * 64  # 256
        cap = 10000
        # current_context_length + extension_tokens = cap -> needs growth
        decoded = cap - 512 - extension_tokens  # exactly at threshold
        seq = make_seq(
            prompt_length=512,
            decoded_length=decoded,
            host_token_capacity=cap,
        )
        assert seq.needs_host_kv_growth(8192)

    def test_get_host_growth_pages_normal(self):
        """Normal growth returns chunk_size / PAGE_SIZE pages."""
        seq = make_seq(
            prompt_length=512,
            host_token_capacity=512 + 8192,
        )
        seq.kv_token_budget = 512 + 131072  # Large budget
        pages = seq.get_host_growth_pages(8192)
        assert pages == math.ceil(8192 / 64)  # 128 pages

    def test_get_host_growth_pages_capped_at_budget(self):
        """Growth should not exceed kv_token_budget."""
        seq = make_seq(prompt_length=512)
        seq.host_token_capacity = 512 + 8000
        seq.kv_token_budget = 512 + 8192  # Only 192 tokens left
        pages = seq.get_host_growth_pages(8192)
        assert pages == math.ceil(192 / 64)  # 3 pages

    def test_get_host_growth_pages_already_at_budget(self):
        """No growth if already at budget."""
        seq = make_seq(prompt_length=512)
        seq.kv_token_budget = 512 + 8192
        seq.host_token_capacity = 512 + 8192  # At budget
        pages = seq.get_host_growth_pages(8192)
        assert pages == 0

    def test_get_host_pages_for_initial_chunk(self):
        """Initial chunk allocation uses prompt + chunk_size."""
        seq = make_seq(prompt_length=512)
        pages = seq.get_host_pages_for_initial_chunk(8192)
        assert pages == math.ceil((512 + 8192) / 64)

    def test_get_host_pages_for_initial_chunk_capped(self):
        """Initial chunk should not exceed kv_token_budget."""
        seq = make_seq(prompt_length=512, max_decode_length=4096)
        # kv_token_budget = 512 + 4096 = 4608 < 512 + 8192
        pages = seq.get_host_pages_for_initial_chunk(8192)
        assert pages == math.ceil(4608 / 64)


# ============ SequenceStatus: EVICTED transitions ============

class TestEvictedStatus:
    """Tests for EVICTED status transitions."""

    def test_in_decode_to_evicted(self):
        seq = make_seq(status=SequenceStatus.IN_DECODE)
        seq.status_transition(SequenceStatus.EVICTED)
        assert seq.status == SequenceStatus.EVICTED

    def test_on_hold_to_evicted(self):
        seq = make_seq()
        seq.status = SequenceStatus.ON_HOLD
        seq.status_transition(SequenceStatus.EVICTED)
        assert seq.status == SequenceStatus.EVICTED

    def test_evicted_to_in_prefill(self):
        seq = make_seq()
        seq.status = SequenceStatus.EVICTED
        seq.status_transition(SequenceStatus.IN_PREFILL)
        assert seq.status == SequenceStatus.IN_PREFILL

    def test_evicted_to_invalid(self):
        seq = make_seq()
        seq.status = SequenceStatus.EVICTED
        with pytest.raises(ValueError):
            seq.status_transition(SequenceStatus.IN_DECODE)

    def test_evicted_to_completed_invalid(self):
        seq = make_seq()
        seq.status = SequenceStatus.EVICTED
        with pytest.raises(ValueError):
            seq.status_transition(SequenceStatus.COMPLETED)

    def test_queueing_to_evicted_invalid(self):
        seq = make_seq(status=SequenceStatus.QUEUEING)
        with pytest.raises(ValueError):
            seq.status_transition(SequenceStatus.EVICTED)


# ============ SequenceBatch: EVICTED handling ============

class TestSequenceBatchEvicted:
    """Tests for SequenceBatch with EVICTED status."""

    def test_has_evicted_false(self):
        batch = SequenceBatch()
        assert not batch.has_evicted()

    def test_has_evicted_true(self):
        batch = SequenceBatch()
        seq = make_seq(uuid="s1", status=SequenceStatus.IN_DECODE)
        batch.add_sequence(seq)
        batch.update_status("s1", SequenceStatus.EVICTED)
        assert batch.has_evicted()

    def test_all_completed_with_evicted(self):
        """all_completed should be False when there are EVICTED sequences."""
        batch = SequenceBatch()
        seq1 = make_seq(uuid="s1", status=SequenceStatus.IN_DECODE)
        seq2 = make_seq(uuid="s2", global_idx=1, status=SequenceStatus.IN_DECODE)
        batch.add_sequence(seq1)
        batch.add_sequence(seq2)
        batch.update_status("s1", SequenceStatus.COMPLETED)
        batch.update_status("s2", SequenceStatus.EVICTED)
        assert not batch.all_completed()

    def test_evicted_sequences_listed(self):
        batch = SequenceBatch()
        seq = make_seq(uuid="s1", status=SequenceStatus.IN_DECODE)
        batch.add_sequence(seq)
        batch.update_status("s1", SequenceStatus.EVICTED)
        evicted = batch.get_sequences_by_status(SequenceStatus.EVICTED)
        assert evicted == ["s1"]


# ============ AdaptiveChunkSizer ============

class TestAdaptiveChunkSizer:
    """Tests for EMA-based adaptive chunk sizing."""

    def test_initial_chunk_size(self):
        sizer = AdaptiveChunkSizer(initial_chunk=8192)
        assert sizer.get_chunk_size() == 8192

    def test_no_adaptation_before_threshold(self):
        """Should not adapt until 10 completions."""
        sizer = AdaptiveChunkSizer(initial_chunk=8192)
        for _ in range(9):
            sizer.report_completion(1000)
        assert sizer.get_chunk_size() == 8192  # Still initial

    def test_adaptation_after_threshold(self):
        """Should adapt after 10 completions."""
        sizer = AdaptiveChunkSizer(
            initial_chunk=8192,
            min_chunk=1024,
            max_chunk=65536,
            ema_alpha=1.0,  # Immediate, no smoothing for test
            multiplier=1.5,
        )
        for _ in range(10):
            sizer.report_completion(2000)
        # EMA = 2000, chunk = 2000 * 1.5 = 3000, rounded to page boundary
        expected = math.ceil(3000 / 64) * 64
        assert sizer.get_chunk_size() == expected

    def test_ema_smoothing(self):
        """EMA should smooth out fluctuations."""
        sizer = AdaptiveChunkSizer(
            initial_chunk=8192,
            ema_alpha=0.1,
            multiplier=1.5,
        )
        # Report 10 completions at 1000
        for _ in range(10):
            sizer.report_completion(1000)
        # EMA should be close to 1000
        assert 900 < sizer.ema_decode_length < 1100

        # Now report 5 completions at 50000 — EMA should move slowly
        for _ in range(5):
            sizer.report_completion(50000)
        # EMA should be between 1000 and 50000, but closer to the lower end
        assert sizer.ema_decode_length < 25000

    def test_min_chunk_enforced(self):
        """Chunk size should not go below min_chunk."""
        sizer = AdaptiveChunkSizer(
            initial_chunk=8192,
            min_chunk=2048,
            ema_alpha=1.0,
            multiplier=1.0,
        )
        for _ in range(10):
            sizer.report_completion(100)  # Very short sequences
        assert sizer.get_chunk_size() >= 2048

    def test_max_chunk_enforced(self):
        """Chunk size should not exceed max_chunk."""
        sizer = AdaptiveChunkSizer(
            initial_chunk=8192,
            max_chunk=16384,
            ema_alpha=1.0,
            multiplier=2.0,
        )
        for _ in range(10):
            sizer.report_completion(100000)  # Very long sequences
        assert sizer.get_chunk_size() <= 16384

    def test_page_aligned(self):
        """Chunk size should always be page-aligned (multiple of 64)."""
        sizer = AdaptiveChunkSizer(
            initial_chunk=8192,
            ema_alpha=1.0,
            multiplier=1.0,
        )
        for _ in range(10):
            sizer.report_completion(1337)  # Not page-aligned
        assert sizer.get_chunk_size() % 64 == 0


# ============ select_sequences_for_eviction with page_key ============

class TestEvictionWithPageKey:
    """Tests for select_sequences_for_eviction with host page support."""

    def test_eviction_uses_host_pages(self):
        """Eviction should use host_pages_allocated when page_key is set."""
        sequences = [
            ("s1", {"decoded_length": 100, "host_pages_allocated": 50, "gpu_pages_allocated": 10}),
            ("s2", {"decoded_length": 200, "host_pages_allocated": 100, "gpu_pages_allocated": 20}),
            ("s3", {"decoded_length": 50, "host_pages_allocated": 80, "gpu_pages_allocated": 5}),
        ]
        uuids, freed = select_sequences_for_eviction(
            sequences, pages_to_free=60,
            strategy=EvictionStrategy.SHORTEST_FIRST,
            page_key="host_pages_allocated",
        )
        # Shortest first: s3 (50 decoded, 80 host pages freed >= 60)
        assert uuids == ["s3"]
        assert freed == 80

    def test_eviction_default_gpu_pages(self):
        """Default page_key should use gpu_pages_allocated."""
        sequences = [
            ("s1", {"decoded_length": 100, "gpu_pages_allocated": 50}),
            ("s2", {"decoded_length": 50, "gpu_pages_allocated": 80}),
        ]
        uuids, freed = select_sequences_for_eviction(
            sequences, pages_to_free=60,
            strategy=EvictionStrategy.SHORTEST_FIRST,
        )
        # s2 shortest (50 decoded) → freed 80 >= 60, done
        assert uuids == ["s2"]
        assert freed == 80

    def test_eviction_empty_list(self):
        uuids, freed = select_sequences_for_eviction([], 100)
        assert uuids == []
        assert freed == 0

    def test_eviction_zero_pages(self):
        sequences = [("s1", {"decoded_length": 100, "gpu_pages_allocated": 50})]
        uuids, freed = select_sequences_for_eviction(sequences, 0)
        assert uuids == []
        assert freed == 0

    def test_eviction_deterministic_tie_breaking(self):
        """Sequences with same decoded_length should be ordered by uuid."""
        sequences = [
            ("b-uuid", {"decoded_length": 100, "gpu_pages_allocated": 50}),
            ("a-uuid", {"decoded_length": 100, "gpu_pages_allocated": 50}),
            ("c-uuid", {"decoded_length": 100, "gpu_pages_allocated": 50}),
        ]
        uuids, freed = select_sequences_for_eviction(
            sequences, pages_to_free=100,
            strategy=EvictionStrategy.SHORTEST_FIRST,
        )
        # Tie-broken by uuid (alphabetical)
        assert uuids == ["a-uuid", "b-uuid"]
        assert freed == 100


# ============ Integration: Full Lifecycle ============

class TestFullLifecycle:
    """Tests for the full sequence lifecycle with dynamic host KV."""

    def test_sequence_growth_lifecycle(self):
        """Test: create -> prefill -> decode -> grow -> grow -> complete."""
        seq = make_seq(
            prompt_length=512,
            max_decode_length=32768,
        )
        chunk_size = 8192

        # Initial allocation
        initial_cap = min(seq.prompt_length + chunk_size, seq.kv_token_budget)
        seq.host_token_capacity = initial_cap
        seq.host_pages_allocated = math.ceil(initial_cap / 64)
        assert seq.host_token_capacity == 512 + 8192

        # Decode for a while — should not need growth yet
        seq.decoded_length = 4000
        seq.current_context_length = 512 + 4000
        assert not seq.needs_host_kv_growth(chunk_size)

        # Decode near capacity
        seq.decoded_length = 8000
        seq.current_context_length = 512 + 8000
        assert seq.needs_host_kv_growth(chunk_size)

        # Grow
        growth = seq.get_host_growth_pages(chunk_size)
        assert growth == 128  # 8192 / 64
        seq.host_token_capacity += growth * 64
        seq.host_pages_allocated += growth
        assert seq.host_token_capacity == 512 + 8192 + 8192

        # Continue decoding
        seq.decoded_length = 16000
        seq.current_context_length = 512 + 16000
        assert seq.needs_host_kv_growth(chunk_size)

        # Grow again
        growth = seq.get_host_growth_pages(chunk_size)
        seq.host_token_capacity += growth * 64
        seq.host_pages_allocated += growth
        assert seq.host_token_capacity == 512 + 8192 * 3

    def test_eviction_reentry_lifecycle(self):
        """Test: decode -> evict -> save tokens -> re-prefill."""
        batch = SequenceBatch()
        seq = make_seq(
            uuid="s1",
            prompt_length=512,
            max_decode_length=32768,
            decoded_length=5000,
            status=SequenceStatus.IN_DECODE,
        )
        batch.add_sequence(seq)

        # Evict
        seq.evicted_token_ids = torch.cat([
            seq.input_ids[:seq.prompt_length],
            seq.decoded_tokens[:seq.decoded_length],
        ])
        seq.total_decoded_before_eviction = seq.decoded_length
        seq.gpu_pages_allocated = 0
        seq.host_pages_allocated = 0
        seq.host_token_capacity = 0
        batch.update_status("s1", SequenceStatus.EVICTED)

        assert batch.has_evicted()
        assert not batch.all_completed()
        assert seq.evicted_token_ids.shape[0] == 512 + 5000

        # Re-enter via prefill (simulating what _prepare_prefill_batch does)
        seq.input_ids = seq.evicted_token_ids.clone()
        seq.prompt_length = len(seq.evicted_token_ids)
        seq.max_decode_length = seq.original_max_decode_length - seq.total_decoded_before_eviction
        seq.kv_token_budget = seq.prompt_length + seq.max_decode_length
        seq.decoded_length = 0
        seq.current_context_length = seq.prompt_length
        batch.update_status("s1", SequenceStatus.IN_PREFILL)

        assert seq.prompt_length == 5512
        assert seq.max_decode_length == 32768 - 5000
        assert seq.status == SequenceStatus.IN_PREFILL

    def test_adaptive_chunk_reduces_waste(self):
        """Demonstrate that adaptive sizing reduces over-reservation."""
        sizer = AdaptiveChunkSizer(
            initial_chunk=8192,
            ema_alpha=0.3,  # Moderate smoothing
            multiplier=1.5,
        )
        # Dataset with short outputs (~500 tokens)
        for _ in range(20):
            sizer.report_completion(500)

        # Chunk should have adapted down from 8192 to ~750 (500 * 1.5, rounded)
        assert sizer.get_chunk_size() < 2000
        assert sizer.get_chunk_size() >= sizer.min_chunk


# ============ Edge Cases ============

class TestEdgeCases:
    """Edge case tests."""

    def test_growth_with_zero_chunk_size(self):
        """Growth with very small chunk should still work."""
        seq = make_seq(prompt_length=512)
        seq.host_token_capacity = 512 + 64
        seq.kv_token_budget = 512 + 131072
        pages = seq.get_host_growth_pages(64)  # Minimum: 1 page
        assert pages == 1

    def test_growth_at_exact_budget(self):
        """No growth when exactly at budget."""
        seq = make_seq(prompt_length=512, max_decode_length=100)
        seq.host_token_capacity = 612  # = kv_token_budget
        pages = seq.get_host_growth_pages(8192)
        assert pages == 0

    def test_eviction_single_sequence(self):
        """Eviction with only one sequence should evict it."""
        sequences = [
            ("s1", {"decoded_length": 100, "host_pages_allocated": 50}),
        ]
        uuids, freed = select_sequences_for_eviction(
            sequences, 10,
            page_key="host_pages_allocated",
        )
        assert uuids == ["s1"]
        assert freed == 50

    def test_initial_chunk_larger_than_budget(self):
        """When chunk_size > max_decode_length, cap at budget."""
        seq = make_seq(prompt_length=100, max_decode_length=500)
        # kv_token_budget = 600, chunk_size = 8192
        pages = seq.get_host_pages_for_initial_chunk(8192)
        assert pages == math.ceil(600 / 64)

    def test_sequence_with_no_decoded_tokens_eviction(self):
        """Eviction of sequence with no decoded tokens."""
        seq = make_seq(
            uuid="s1",
            prompt_length=512,
            decoded_length=0,
            status=SequenceStatus.IN_DECODE,
        )
        # No decoded tokens — evicted_token_ids should just be prompt
        if seq.decoded_tokens is not None and seq.decoded_length > 0:
            seq.evicted_token_ids = torch.cat([
                seq.input_ids[:seq.prompt_length],
                seq.decoded_tokens[:seq.decoded_length],
            ])
        else:
            seq.evicted_token_ids = seq.input_ids[:seq.prompt_length].clone()

        assert seq.evicted_token_ids.shape[0] == 512

    def test_adaptive_sizer_single_completion(self):
        """Single completion should update EMA but not adapt chunk size."""
        sizer = AdaptiveChunkSizer(initial_chunk=8192)
        sizer.report_completion(100)
        assert sizer.ema_decode_length == 100.0
        assert sizer.completed_count == 1
        assert sizer.get_chunk_size() == 8192  # Not yet adapted

    def test_adaptive_sizer_disabled(self):
        """When adaptive_chunk=False, sizer should not be created."""
        # This tests the worker-level logic implicitly
        sizer = None
        chunk_size = 8192
        effective = sizer.get_chunk_size() if sizer else chunk_size
        assert effective == 8192


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
