"""
Unit tests for QueryBookBufferPool.

Tests correctness and performance of the buffer pool design:
- View shape preservation (critical for downstream code)
- Simulated migration (clone → send → copy into dest slot)
- Slot reuse after free
- Eviction re-entry
- Performance: buffer pool vs per-sequence allocation
"""

import time
import torch
import pytest


class QueryBookBufferPool:
    """Pre-allocated contiguous buffers for query book tensors."""

    def __init__(self, num_sequences: int, model_context_length: int, max_decoding_length: int):
        self.input_ids_buffer = torch.zeros((num_sequences, model_context_length), dtype=torch.long)
        self.decoded_tokens_buffer = torch.zeros((num_sequences, max_decoding_length), dtype=torch.int64)
        self.num_sequences = num_sequences
        self.model_context_length = model_context_length
        self.max_decoding_length = max_decoding_length
        self._free_slots: set = set()
        self._next_slot: int = 0

    def allocate_slot(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        slot = self._next_slot
        if slot >= self.num_sequences:
            raise RuntimeError(f"Buffer pool exhausted: {self.num_sequences} slots used")
        self._next_slot += 1
        return slot

    def free_slot(self, slot: int):
        self._free_slots.add(slot)

    def get_input_ids_view(self, slot: int, seq_extended_size: int) -> torch.Tensor:
        return self.input_ids_buffer[slot:slot+1, :seq_extended_size]

    def get_decoded_tokens_view(self, slot: int) -> torch.Tensor:
        return self.decoded_tokens_buffer[slot:slot+1, :]


# ============ Test Cases ============


class TestAllocationCorrectness:
    """Verify views have correct shapes and writes go to underlying buffer."""

    def test_view_shapes(self):
        pool = QueryBookBufferPool(num_sequences=10, model_context_length=1024, max_decoding_length=128)
        slot = pool.allocate_slot()
        seq_ext = 512  # smaller than model_context_length

        input_view = pool.get_input_ids_view(slot, seq_ext)
        decoded_view = pool.get_decoded_tokens_view(slot)

        assert input_view.shape == (1, 512), f"Expected (1, 512), got {input_view.shape}"
        assert decoded_view.shape == (1, 128), f"Expected (1, 128), got {decoded_view.shape}"

    def test_write_through_view(self):
        pool = QueryBookBufferPool(num_sequences=10, model_context_length=1024, max_decoding_length=128)
        slot = pool.allocate_slot()
        input_view = pool.get_input_ids_view(slot, 512)

        # Write prompt tokens
        prompt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        input_view[0, :5] = prompt

        # Verify underlying buffer has the data
        assert torch.equal(pool.input_ids_buffer[slot, :5], prompt)
        # Verify padding is still zero
        assert pool.input_ids_buffer[slot, 5:512].sum() == 0

    def test_decoded_tokens_sequential_write(self):
        pool = QueryBookBufferPool(num_sequences=10, model_context_length=1024, max_decoding_length=128)
        slot = pool.allocate_slot()
        decoded_view = pool.get_decoded_tokens_view(slot)

        # Simulate decode loop
        for i in range(10):
            decoded_view[:, i] = i + 100

        expected = torch.zeros(128, dtype=torch.int64)
        expected[:10] = torch.arange(100, 110, dtype=torch.int64)
        assert torch.equal(pool.decoded_tokens_buffer[slot], expected)

    def test_multiple_slots_independent(self):
        pool = QueryBookBufferPool(num_sequences=10, model_context_length=1024, max_decoding_length=128)
        s0 = pool.allocate_slot()
        s1 = pool.allocate_slot()

        v0 = pool.get_input_ids_view(s0, 256)
        v1 = pool.get_input_ids_view(s1, 512)

        v0[0, 0] = 42
        v1[0, 0] = 99

        assert pool.input_ids_buffer[s0, 0] == 42
        assert pool.input_ids_buffer[s1, 0] == 99
        assert v0.shape == (1, 256)
        assert v1.shape == (1, 512)


class TestViewShapePreservation:
    """Critical: .shape[1] must return seq_extended_size, not model_context_length."""

    def test_shape_is_seq_extended_not_model_ctx(self):
        pool = QueryBookBufferPool(num_sequences=5, model_context_length=131072, max_decoding_length=8192)
        slot = pool.allocate_slot()

        seq_ext = 4096 + 8192  # prompt + max_decode
        view = pool.get_input_ids_view(slot, seq_ext)

        assert view.shape[1] == seq_ext, f"shape[1]={view.shape[1]}, expected {seq_ext}"
        assert view.shape[1] != 131072, "shape[1] should NOT be model_context_length"

    def test_contiguous_is_a_view_not_clone(self):
        """CRITICAL FINDING: .contiguous() on a row slice returns the SAME tensor
        (since buffer[i:i+1, :N] is already contiguous). Migration MUST use .clone()."""
        pool = QueryBookBufferPool(num_sequences=5, model_context_length=131072, max_decoding_length=8192)
        slot = pool.allocate_slot()
        seq_ext = 12345

        view = pool.get_input_ids_view(slot, seq_ext)
        view[0, :10] = torch.arange(10, dtype=torch.long)

        # .contiguous() does NOT clone — it's already contiguous
        cont = view.contiguous()
        assert cont.data_ptr() == view.data_ptr(), "contiguous() should return same tensor for contiguous view"

        # Migration must use .clone() to get an independent copy
        cloned = view.clone()
        assert cloned.shape == (1, seq_ext)
        assert cloned.data_ptr() != view.data_ptr(), ".clone() must create independent copy"
        assert torch.equal(cloned[0, :10], torch.arange(10, dtype=torch.long))

        # Verify clone is independent: mutate buffer, clone should be unaffected
        view[0, 0] = 999
        assert cloned[0, 0] == 0, "clone must be independent of buffer"

    def test_view_is_actually_a_view(self):
        """Verify the view shares storage with the buffer (no copy)."""
        pool = QueryBookBufferPool(num_sequences=5, model_context_length=1024, max_decoding_length=128)
        slot = pool.allocate_slot()
        view = pool.get_input_ids_view(slot, 512)

        # Write via buffer, read via view
        pool.input_ids_buffer[slot, 100] = 777
        assert view[0, 100] == 777

        # Write via view, read via buffer
        view[0, 200] = 888
        assert pool.input_ids_buffer[slot, 200] == 888


class TestSimulatedMigration:
    """End-to-end migration: source writes → clone → recv → copy into dest."""

    def test_migration_roundtrip(self):
        model_ctx = 4096
        max_decode = 128
        src_pool = QueryBookBufferPool(num_sequences=10, model_context_length=model_ctx, max_decoding_length=max_decode)
        dst_pool = QueryBookBufferPool(num_sequences=10, model_context_length=model_ctx, max_decoding_length=max_decode)

        # Source: populate slot with prompt + decoded tokens
        src_slot = src_pool.allocate_slot()
        prompt_len = 100
        seq_ext = prompt_len + max_decode
        prompt_tokens = torch.arange(1, prompt_len + 1, dtype=torch.long)

        src_input = src_pool.get_input_ids_view(src_slot, seq_ext)
        src_input[0, :prompt_len] = prompt_tokens

        src_decoded = src_pool.get_decoded_tokens_view(src_slot)
        decoded_count = 15
        for i in range(decoded_count):
            src_decoded[:, i] = 1000 + i

        # Migration send: .clone() to get independent copy (NOT .contiguous() — see test above)
        send_input = src_input.clone()
        send_decoded = src_decoded.clone()

        assert send_input.shape == (1, seq_ext)
        assert send_decoded.shape == (1, max_decode)

        # Source frees slot
        src_pool.free_slot(src_slot)

        # Destination: allocate slot, copy received data
        dst_slot = dst_pool.allocate_slot()
        budget = seq_ext
        dst_pool.input_ids_buffer[dst_slot, :budget] = send_input[0, :budget]
        dst_pool.decoded_tokens_buffer[dst_slot, :] = send_decoded[0, :]

        # Create views
        dst_input = dst_pool.get_input_ids_view(dst_slot, budget)
        dst_decoded = dst_pool.get_decoded_tokens_view(dst_slot)

        # Verify correctness
        assert dst_input.shape == (1, seq_ext)
        assert torch.equal(dst_input[0, :prompt_len], prompt_tokens)
        assert dst_input[0, prompt_len:].sum() == 0  # padding zero

        for i in range(decoded_count):
            assert dst_decoded[0, i] == 1000 + i
        assert dst_decoded[0, decoded_count:].sum() == 0


class TestSlotReuse:

    def test_freed_slot_is_reused(self):
        pool = QueryBookBufferPool(num_sequences=10, model_context_length=1024, max_decoding_length=128)
        slots = [pool.allocate_slot() for _ in range(5)]
        assert slots == [0, 1, 2, 3, 4]

        pool.free_slot(2)
        reused = pool.allocate_slot()
        assert reused == 2

    def test_exhaust_and_reuse(self):
        pool = QueryBookBufferPool(num_sequences=3, model_context_length=64, max_decoding_length=16)
        s0 = pool.allocate_slot()
        s1 = pool.allocate_slot()
        s2 = pool.allocate_slot()

        with pytest.raises(RuntimeError, match="exhausted"):
            pool.allocate_slot()

        pool.free_slot(s1)
        reused = pool.allocate_slot()
        assert reused == s1


class TestEvictionReEntry:

    def test_reuse_slot_after_eviction(self):
        pool = QueryBookBufferPool(num_sequences=5, model_context_length=4096, max_decoding_length=128)
        slot = pool.allocate_slot()
        budget = 2048

        # Initial: write prompt
        view = pool.get_input_ids_view(slot, budget)
        view[0, :50] = torch.arange(1, 51, dtype=torch.long)

        decoded = pool.get_decoded_tokens_view(slot)
        for i in range(20):
            decoded[:, i] = 500 + i

        # Eviction: save tokens
        evicted_prompt = view[0, :50].clone()
        evicted_decoded = decoded[0, :20].clone()
        evicted_ids = torch.cat([evicted_prompt, evicted_decoded])

        # Re-entry: clear and rewrite same slot
        pool.input_ids_buffer[slot, :] = 0
        pool.decoded_tokens_buffer[slot, :] = 0

        new_prompt_len = len(evicted_ids)  # 70
        pool.input_ids_buffer[slot, :new_prompt_len] = evicted_ids
        new_view = pool.get_input_ids_view(slot, budget)

        # Verify
        assert torch.equal(new_view[0, :new_prompt_len], evicted_ids)
        assert new_view[0, new_prompt_len:].sum() == 0


class TestPerformance:
    """Benchmark buffer pool vs per-sequence allocation."""

    @pytest.mark.parametrize("num_seqs,ctx_len", [
        (100, 131072),    # small batch, full context
        (1000, 131072),   # medium batch
        (12000, 4096),    # large batch, short context
    ])
    def test_allocation_speedup(self, num_seqs, ctx_len):
        max_decode = 8192

        # Baseline: per-sequence allocation in a loop
        t0 = time.perf_counter()
        tensors = []
        for i in range(num_seqs):
            seq_ext = min(1000 + i % 3000, ctx_len)  # variable sizes
            t = torch.zeros((1, seq_ext), dtype=torch.long)
            t[0, :min(500, seq_ext)] = torch.arange(min(500, seq_ext), dtype=torch.long)
            d = torch.zeros((1, max_decode), dtype=torch.int64)
            tensors.append((t, d))
        baseline_ms = (time.perf_counter() - t0) * 1000
        del tensors

        # Buffer pool: single allocation + views
        t0 = time.perf_counter()
        pool = QueryBookBufferPool(
            num_sequences=num_seqs,
            model_context_length=ctx_len,
            max_decoding_length=max_decode,
        )
        views = []
        for i in range(num_seqs):
            slot = pool.allocate_slot()
            seq_ext = min(1000 + i % 3000, ctx_len)
            v = pool.get_input_ids_view(slot, seq_ext)
            v[0, :min(500, seq_ext)] = torch.arange(min(500, seq_ext), dtype=torch.long)
            d = pool.get_decoded_tokens_view(slot)
            views.append((v, d))
        pool_ms = (time.perf_counter() - t0) * 1000
        del views, pool

        speedup = baseline_ms / pool_ms if pool_ms > 0 else float('inf')
        print(f"\n  N={num_seqs}, ctx={ctx_len}: baseline={baseline_ms:.1f}ms, pool={pool_ms:.1f}ms, speedup={speedup:.1f}x")

    def test_realistic_production_workload(self):
        """Reproduce the actual production scenario: 12K seqs, 131072 ctx, ~2000 token prompts.
        This matches what we see in server logs: 12032 sequences, max_prompt=2657, max_decode=8192."""
        import gc

        num_seqs = 12000
        ctx_len = 131072
        max_decode = 8192
        # Realistic prompt lengths: uniform 878-2657 (matching server logs)
        prompt_lengths = [878 + (i * 1779) % 1780 for i in range(num_seqs)]

        print(f"\n  Production scenario: N={num_seqs}, ctx={ctx_len}, "
              f"prompts={min(prompt_lengths)}-{max(prompt_lengths)}, max_decode={max_decode}")

        # ---- Baseline: per-sequence allocation (matches current Phase 3) ----
        gc.collect()
        t_total = time.perf_counter()
        t_zeros_input = 0.0
        t_zeros_decoded = 0.0
        t_list_to_tensor = 0.0
        t_copy = 0.0

        tensors = []
        for i in range(num_seqs):
            prompt_len = prompt_lengths[i]
            seq_ext = min(prompt_len + max_decode, ctx_len)
            # Simulated token list (like tokenizer output)
            token_list = list(range(prompt_len))

            t0 = time.perf_counter()
            inp = torch.zeros((1, seq_ext), dtype=torch.long)
            t1 = time.perf_counter()
            t_zeros_input += t1 - t0

            t0 = time.perf_counter()
            token_tensor = torch.tensor(token_list, dtype=torch.long)
            t1 = time.perf_counter()
            t_list_to_tensor += t1 - t0

            t0 = time.perf_counter()
            inp[0, :prompt_len] = token_tensor
            t1 = time.perf_counter()
            t_copy += t1 - t0

            t0 = time.perf_counter()
            dec = torch.zeros((1, max_decode), dtype=torch.int64)
            t1 = time.perf_counter()
            t_zeros_decoded += t1 - t0

            tensors.append((inp, dec))

            if (i + 1) % 3000 == 0:
                elapsed = time.perf_counter() - t_total
                print(f"    Baseline progress: {i+1}/{num_seqs} ({elapsed:.1f}s)")

        baseline_ms = (time.perf_counter() - t_total) * 1000
        total_input_gb = sum(min(prompt_lengths[i] + max_decode, ctx_len) for i in range(num_seqs)) * 8 / (1024**3)
        total_decoded_gb = num_seqs * max_decode * 8 / (1024**3)
        print(f"  Baseline: {baseline_ms:.0f}ms total ({total_input_gb:.2f}GB input + {total_decoded_gb:.2f}GB decoded)")
        print(f"    torch.zeros input:  {t_zeros_input*1000:.0f}ms")
        print(f"    torch.zeros decoded:{t_zeros_decoded*1000:.0f}ms")
        print(f"    list→tensor:        {t_list_to_tensor*1000:.0f}ms")
        print(f"    copy:               {t_copy*1000:.0f}ms")
        del tensors
        gc.collect()

        # ---- Buffer pool ----
        t_total = time.perf_counter()
        t_alloc = time.perf_counter()
        pool = QueryBookBufferPool(
            num_sequences=num_seqs,
            model_context_length=ctx_len,
            max_decoding_length=max_decode,
        )
        t_alloc_ms = (time.perf_counter() - t_alloc) * 1000

        t_list_to_tensor_pool = 0.0
        t_copy_pool = 0.0
        t_view_pool = 0.0

        views = []
        for i in range(num_seqs):
            prompt_len = prompt_lengths[i]
            seq_ext = min(prompt_len + max_decode, ctx_len)
            token_list = list(range(prompt_len))

            t0 = time.perf_counter()
            slot = pool.allocate_slot()
            v = pool.get_input_ids_view(slot, seq_ext)
            d = pool.get_decoded_tokens_view(slot)
            t1 = time.perf_counter()
            t_view_pool += t1 - t0

            t0 = time.perf_counter()
            token_tensor = torch.tensor(token_list, dtype=torch.long)
            t1 = time.perf_counter()
            t_list_to_tensor_pool += t1 - t0

            t0 = time.perf_counter()
            v[0, :prompt_len] = token_tensor
            t1 = time.perf_counter()
            t_copy_pool += t1 - t0

            views.append((v, d))

            if (i + 1) % 3000 == 0:
                elapsed = time.perf_counter() - t_total
                print(f"    Pool progress: {i+1}/{num_seqs} ({elapsed:.1f}s)")

        pool_ms = (time.perf_counter() - t_total) * 1000
        print(f"  Buffer pool: {pool_ms:.0f}ms total")
        print(f"    buffer alloc:       {t_alloc_ms:.0f}ms")
        print(f"    view creation:      {t_view_pool*1000:.0f}ms")
        print(f"    list→tensor:        {t_list_to_tensor_pool*1000:.0f}ms")
        print(f"    copy:               {t_copy_pool*1000:.0f}ms")

        speedup = baseline_ms / pool_ms if pool_ms > 0 else float('inf')
        print(f"  SPEEDUP: {speedup:.1f}x")
        del views, pool
        gc.collect()

    def test_migration_overhead(self):
        """Measure clone + copy overhead for simulated migration."""
        pool = QueryBookBufferPool(num_sequences=100, model_context_length=131072, max_decoding_length=8192)

        # Populate 100 slots
        for i in range(100):
            slot = pool.allocate_slot()
            seq_ext = 4096 + (i * 1000) % 127000
            v = pool.get_input_ids_view(slot, seq_ext)
            v[0, :100] = torch.arange(100, dtype=torch.long)

        # Measure migration of 10 sequences
        t0 = time.perf_counter()
        for i in range(10):
            seq_ext = 4096 + (i * 1000) % 127000
            src_view = pool.get_input_ids_view(i, seq_ext)
            # Clone for send
            cloned = src_view.contiguous()
            # Simulate recv + copy into new slot (reuse slot 90+i)
            dst_slot = 90 + i
            pool.input_ids_buffer[dst_slot, :seq_ext] = cloned[0, :]

        migrate_ms = (time.perf_counter() - t0) * 1000
        print(f"\n  Migration 10 seqs: {migrate_ms:.1f}ms ({migrate_ms/10:.2f}ms/seq)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
