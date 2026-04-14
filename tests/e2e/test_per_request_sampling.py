"""Unit tests for per-request sampling parameters (T26).

Tests the vectorized sample_tokens() function with per-sequence temperature,
top_p, and top_k tensors. Covers correctness, corner cases, and backward
compatibility.

NOTE: Performance tests require GPU and should be run on remote machines only.
"""

import torch
import pytest
from batchgen.sampling import sample_tokens, greedy_decode


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_logits(B: int, V: int = 1000, device: str = "cpu") -> torch.Tensor:
    """Create deterministic logits with a clear peak at position 0."""
    logits = torch.randn(B, V, device=device)
    # Make position 0 clearly the argmax for each row
    logits[:, 0] = 10.0
    return logits


def _make_skewed_logits(B: int, V: int = 1000, device: str = "cpu") -> torch.Tensor:
    """Logits with most mass in top few tokens (for top-k/top-p testing)."""
    logits = torch.full((B, V), -100.0, device=device)
    # Top 5 tokens have high logits, rest are -100
    logits[:, 0] = 10.0
    logits[:, 1] = 9.0
    logits[:, 2] = 8.0
    logits[:, 3] = 7.0
    logits[:, 4] = 6.0
    return logits


# ===========================================================================
# 1. Greedy-only batch
# ===========================================================================
class TestGreedyOnly:
    def test_all_greedy_scalar(self):
        logits = _make_logits(4)
        result = sample_tokens(logits, temperature=None)
        expected = torch.argmax(logits, dim=-1, keepdim=True)
        assert torch.equal(result, expected)

    def test_all_greedy_zero_scalar(self):
        logits = _make_logits(4)
        result = sample_tokens(logits, temperature=0.0)
        expected = torch.argmax(logits, dim=-1, keepdim=True)
        assert torch.equal(result, expected)

    def test_all_greedy_tensor(self):
        logits = _make_logits(4)
        temps = torch.tensor([0.0, 0.0, -1.0, 0.0])
        result = sample_tokens(logits, temperature=temps)
        expected = torch.argmax(logits, dim=-1, keepdim=True)
        assert torch.equal(result, expected)


# ===========================================================================
# 2. Sampling-only batch with fixed seed
# ===========================================================================
class TestSamplingOnly:
    def test_different_temps_produce_different_distributions(self):
        """With same logits but different temps, outputs should differ over many runs."""
        B, V = 2, 100
        logits = torch.randn(B, V)
        logits[0] = logits[1]  # Same logits for both sequences

        temps = torch.tensor([0.1, 2.0])  # Very different temperatures
        # Low temp should mostly pick argmax, high temp should be more random
        low_temp_results = []
        high_temp_results = []
        for _ in range(50):
            result = sample_tokens(logits.clone(), temperature=temps.clone())
            low_temp_results.append(result[0].item())
            high_temp_results.append(result[1].item())

        # Low temp should have much less diversity than high temp
        low_unique = len(set(low_temp_results))
        high_unique = len(set(high_temp_results))
        assert low_unique <= high_unique, (
            f"Low temp ({low_unique} unique) should have <= diversity than "
            f"high temp ({high_unique} unique)"
        )


# ===========================================================================
# 3. Mixed greedy + sampling
# ===========================================================================
class TestMixedGreedySampling:
    def test_greedy_rows_deterministic(self):
        """Greedy rows should always return argmax regardless of sampling rows."""
        B = 4
        logits = _make_logits(B)
        # Sequences 0, 2 are greedy; 1, 3 are sampling
        temps = torch.tensor([0.0, 1.0, 0.0, 0.8])
        expected_greedy = torch.argmax(logits, dim=-1, keepdim=True)

        for _ in range(10):
            result = sample_tokens(logits.clone(), temperature=temps.clone())
            # Greedy rows must match argmax
            assert result[0].item() == expected_greedy[0].item()
            assert result[2].item() == expected_greedy[2].item()

    def test_sampling_rows_stochastic(self):
        """Sampling rows with high temp should vary across runs."""
        B = 2
        logits = torch.randn(B, 100)
        temps = torch.tensor([0.0, 2.0])  # Row 0 greedy, row 1 high temp

        results_row1 = set()
        for _ in range(50):
            result = sample_tokens(logits.clone(), temperature=temps.clone())
            results_row1.add(result[1].item())

        assert len(results_row1) > 1, "High temperature should produce varied results"


# ===========================================================================
# 4. Top-k correctness
# ===========================================================================
class TestTopK:
    def test_top_k_1_equals_argmax(self):
        """top_k=1 should be equivalent to greedy."""
        logits = _make_logits(4)
        temps = torch.tensor([1.0, 1.0, 1.0, 1.0])
        top_ks = torch.tensor([1, 1, 1, 1])
        result = sample_tokens(logits, temperature=temps, top_k=top_ks)
        expected = torch.argmax(logits, dim=-1, keepdim=True)
        assert torch.equal(result, expected)

    def test_top_k_limits_candidates(self):
        """With top_k=5, only top-5 tokens should ever be selected."""
        logits = _make_skewed_logits(4)
        temps = torch.tensor([1.0, 1.0, 1.0, 1.0])
        top_ks = torch.tensor([5, 5, 5, 5])

        for _ in range(50):
            result = sample_tokens(logits.clone(), temperature=temps.clone(), top_k=top_ks.clone())
            for i in range(4):
                assert result[i].item() < 5, f"Token {result[i].item()} should be in top-5"

    def test_per_seq_top_k(self):
        """Different top_k per sequence."""
        logits = _make_skewed_logits(2)
        temps = torch.tensor([1.0, 1.0])
        top_ks = torch.tensor([1, 5])  # Seq 0: top-1 (argmax), Seq 1: top-5

        for _ in range(50):
            result = sample_tokens(logits.clone(), temperature=temps.clone(), top_k=top_ks.clone())
            assert result[0].item() == 0, "top_k=1 should always select argmax"
            assert result[1].item() < 5, "top_k=5 should select from top-5"


# ===========================================================================
# 5. Top-p correctness
# ===========================================================================
class TestTopP:
    def test_top_p_very_small(self):
        """Very small top_p should behave like greedy (only top-1 token)."""
        logits = _make_skewed_logits(4)
        temps = torch.tensor([1.0, 1.0, 1.0, 1.0])
        top_ps = torch.tensor([0.01, 0.01, 0.01, 0.01])

        for _ in range(20):
            result = sample_tokens(logits.clone(), temperature=temps.clone(), top_p=top_ps.clone())
            for i in range(4):
                assert result[i].item() == 0, "Very small top_p should select top-1 token"

    def test_top_p_1_allows_all(self):
        """top_p=1.0 should allow all tokens."""
        B, V = 4, 100
        logits = torch.randn(B, V)
        temps = torch.tensor([1.0, 1.0, 1.0, 1.0])
        top_ps = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # Should not crash and should produce valid results
        result = sample_tokens(logits, temperature=temps, top_p=top_ps)
        assert result.shape == (B, 1)
        assert (result >= 0).all() and (result < V).all()

    def test_per_seq_top_p(self):
        """Different top_p per sequence."""
        logits = _make_skewed_logits(2)
        temps = torch.tensor([1.0, 1.0])
        top_ps = torch.tensor([0.01, 1.0])  # Seq 0: only top-1, Seq 1: all tokens

        for _ in range(20):
            result = sample_tokens(logits.clone(), temperature=temps.clone(), top_p=top_ps.clone())
            assert result[0].item() == 0, "Very small top_p should select argmax"


# ===========================================================================
# 6. Backward compatibility
# ===========================================================================
class TestBackwardCompat:
    def test_scalar_matches_tensor_uniform(self):
        """Scalar params should produce same results as uniform tensor params."""
        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        # Scalar path
        torch.manual_seed(123)
        result_scalar = sample_tokens(logits.clone(), temperature=0.7, top_p=0.9, top_k=10)

        # Tensor path (uniform values)
        torch.manual_seed(123)
        temps = torch.tensor([0.7, 0.7, 0.7, 0.7])
        top_ps = torch.tensor([0.9, 0.9, 0.9, 0.9])
        top_ks = torch.tensor([10, 10, 10, 10])
        result_tensor = sample_tokens(logits.clone(), temperature=temps, top_p=top_ps, top_k=top_ks)

        assert torch.equal(result_scalar, result_tensor), (
            f"Scalar and tensor paths should produce identical results with same seed. "
            f"Scalar: {result_scalar.flatten().tolist()}, Tensor: {result_tensor.flatten().tolist()}"
        )


# ===========================================================================
# 7. Corner case: NaN fallback
# ===========================================================================
class TestNaNFallback:
    def test_all_filtered_fallback_to_argmax(self):
        """If top_k + top_p filters everything, should fallback to argmax."""
        B, V = 2, 10
        # Uniform logits — after extreme filtering, may result in all -inf
        logits = torch.zeros(B, V)
        logits[:, 5] = 1.0  # Slight peak at position 5
        temps = torch.tensor([1.0, 1.0])
        top_ks = torch.tensor([1, 1])
        top_ps = torch.tensor([0.001, 0.001])

        result = sample_tokens(logits, temperature=temps, top_k=top_ks, top_p=top_ps)
        assert result.shape == (B, 1)
        # Should not crash — either returns valid token or argmax fallback


# ===========================================================================
# 8. Corner case: B=1
# ===========================================================================
class TestBatchSize1:
    def test_single_sequence_greedy(self):
        logits = _make_logits(1)
        result = sample_tokens(logits, temperature=torch.tensor([0.0]))
        assert result.shape == (1, 1)
        assert result.item() == torch.argmax(logits).item()

    def test_single_sequence_sampling(self):
        logits = _make_logits(1)
        result = sample_tokens(logits, temperature=torch.tensor([1.0]))
        assert result.shape == (1, 1)
        assert 0 <= result.item() < logits.shape[1]


# ===========================================================================
# 9. Corner case: B=0
# ===========================================================================
class TestBatchSize0:
    def test_empty_batch(self):
        logits = torch.randn(0, 100)
        result = sample_tokens(logits, temperature=None)
        assert result.shape == (0, 1)


# ===========================================================================
# 10. Corner case: large top_k
# ===========================================================================
class TestLargeTopK:
    def test_top_k_larger_than_vocab(self):
        B, V = 2, 100
        logits = torch.randn(B, V)
        temps = torch.tensor([1.0, 1.0])
        top_ks = torch.tensor([9999, 9999])  # Much larger than V

        result = sample_tokens(logits, temperature=temps, top_k=top_ks)
        assert result.shape == (B, 1)
        assert (result >= 0).all() and (result < V).all()


# ===========================================================================
# 11. Combined per-sequence params
# ===========================================================================
class TestCombinedPerSequence:
    def test_mixed_all_params(self):
        """Each sequence has different temperature, top_p, and top_k."""
        B, V = 4, 100
        logits = _make_skewed_logits(B, V)
        temps = torch.tensor([0.0, 0.5, 1.0, 2.0])
        top_ps = torch.tensor([1.0, 0.9, 0.5, 1.0])
        top_ks = torch.tensor([0, 10, 5, 0])

        result = sample_tokens(logits, temperature=temps, top_p=top_ps, top_k=top_ks)
        assert result.shape == (B, 1)
        # Seq 0 is greedy
        assert result[0].item() == 0


# ===========================================================================
# 12. Sequence manager integration
# ===========================================================================
class TestSequenceEntryParams:
    def test_sequence_entry_stores_params(self):
        from batchgen.sequence_manager.batch_defs import SequenceEntry
        seq = SequenceEntry(
            uuid="test-1",
            input_ids=[1, 2, 3],
            max_output_length=10,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
        )
        assert seq.temperature == 0.7
        assert seq.top_p == 0.95
        assert seq.top_k == 40

    def test_sequence_entry_default_params(self):
        from batchgen.sequence_manager.batch_defs import SequenceEntry
        seq = SequenceEntry(
            uuid="test-2",
            input_ids=[1, 2, 3],
            max_output_length=10,
        )
        assert seq.temperature is None
        assert seq.top_p is None
        assert seq.top_k is None


class TestActiveBatchSamplingTensors:
    def test_build_sampling_tensors(self):
        from batchgen.sequence_manager.batch_defs import (
            SequenceEntry, GlobalSequenceRegistry, ActiveBatch
        )
        seqs = [
            SequenceEntry("s1", [1, 2], 10, temperature=0.7, top_p=0.9, top_k=10),
            SequenceEntry("s2", [3, 4], 10, temperature=None, top_p=None, top_k=None),
            SequenceEntry("s3", [5, 6], 10, temperature=1.0, top_p=0.5, top_k=20),
        ]
        registry = GlobalSequenceRegistry(seqs)
        batch = ActiveBatch(registry, ["s1", "s2", "s3"], torch.device("cpu"), 0)

        assert batch.sampling_temps is not None
        torch.testing.assert_close(
            batch.sampling_temps, torch.tensor([0.7, 0.0, 1.0])
        )  # None → 0.0
        torch.testing.assert_close(
            batch.sampling_top_ps, torch.tensor([0.9, 1.0, 0.5])
        )  # None → 1.0
        assert batch.sampling_top_ks.tolist() == [10, 0, 20]  # None → 0


# ===========================================================================
# Performance test (GPU only — run on remote machine)
# ===========================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
class TestPerformance:
    def test_sampling_latency(self):
        """Measure per-seq sampling overhead vs scalar sampling."""
        device = "cuda"
        V = 152064  # Typical vocab size

        for B in [1, 8, 32, 64, 128]:
            logits = torch.randn(B, V, device=device)

            # Scalar sampling (current path)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                sample_tokens(logits.clone(), temperature=0.7, top_p=0.9)
            end.record()
            torch.cuda.synchronize()
            scalar_ms = start.elapsed_time(end) / 100

            # Per-seq tensor sampling
            temps = torch.full((B,), 0.7, device=device)
            top_ps = torch.full((B,), 0.9, device=device)
            top_ks = torch.zeros(B, dtype=torch.int64, device=device)

            start.record()
            for _ in range(100):
                sample_tokens(logits.clone(), temperature=temps.clone(),
                            top_p=top_ps.clone(), top_k=top_ks.clone())
            end.record()
            torch.cuda.synchronize()
            tensor_ms = start.elapsed_time(end) / 100

            overhead = (tensor_ms - scalar_ms) / scalar_ms * 100
            print(f"B={B:4d}: scalar={scalar_ms:.3f}ms, tensor={tensor_ms:.3f}ms, "
                  f"overhead={overhead:+.1f}%")

            # Tensor path should not be >50% slower than scalar for uniform params
            assert tensor_ms < scalar_ms * 1.5, (
                f"B={B}: Tensor path too slow ({tensor_ms:.3f}ms vs {scalar_ms:.3f}ms)"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
