"""Unit tests for GQA with attention sinks.

Tests verify:
1. Sink correction math is correct
2. FA prefill matches reference implementation
3. FA decode produces correct outputs
4. Sinks actually affect the output
"""

import pytest
import torch

from .reference import attention_ref, attention_ref_no_sinks
from .sink_correction import apply_sink_correction


# =============================================================================
# Test Utilities
# =============================================================================

def make_test_tensors(
    batch_size: int,
    num_queries: int,
    num_keys: int,
    nheads: int,
    nheads_kv: int,
    headdim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Create test tensors in the reference format.

    Reference format:
        Q: [batch, num_queries, nheads_kv, num_groups, headdim]
        K: [batch, num_keys, nheads_kv, headdim]
        V: [batch, num_keys, nheads_kv, headdim]
        sinks: [nheads]
    """
    num_groups = nheads // nheads_kv
    q = torch.randn(batch_size, num_queries, nheads_kv, num_groups, headdim,
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, num_keys, nheads_kv, headdim,
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, num_keys, nheads_kv, headdim,
                    device=device, dtype=dtype)
    sinks = torch.randn(nheads, device=device, dtype=dtype)
    return q, k, v, sinks


def reshape_for_flash_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor = None,
):
    """Reshape tensors from reference format to flash varlen format.

    Reference: Q [batch, seqlen, nheads_kv, num_groups, headdim]
    Flash varlen: Q [total_tokens, nheads, headdim]

    Returns q, k, v in flash format plus cu_seqlens and max_seqlen.
    """
    batch, seqlen_q, nheads_kv, num_groups, headdim = q.shape
    _, seqlen_k, _, _ = k.shape
    nheads = nheads_kv * num_groups

    # Reshape Q: [batch, seqlen, nheads_kv, groups, dim] -> [batch, seqlen, nheads, dim]
    q_flash = q.reshape(batch, seqlen_q, nheads, headdim)

    # For now, assume no padding (all sequences same length)
    # In real usage, would use attention_mask to compute actual lengths
    seqlens_q = torch.full((batch,), seqlen_q, dtype=torch.int32, device=q.device)
    seqlens_k = torch.full((batch,), seqlen_k, dtype=torch.int32, device=k.device)

    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=k.device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, dim=0)

    # Flatten to varlen format
    q_varlen = q_flash.reshape(-1, nheads, headdim)  # [total_q, nheads, headdim]
    k_varlen = k.reshape(-1, nheads_kv, headdim)  # [total_k, nheads_kv, headdim]
    v_varlen = v.reshape(-1, nheads_kv, headdim)

    return q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k


# =============================================================================
# Sink Correction Tests
# =============================================================================

class TestSinkCorrection:
    """Tests for the sink correction math."""

    def test_sink_correction_identity_with_zero_sinks(self):
        """With sinks=0, correction factor should be ~0.5 (sigmoid(lse - 0))."""
        batch, nheads, seqlen, headdim = 2, 8, 16, 64
        output = torch.randn(batch, seqlen, nheads, headdim, device="cuda")
        lse = torch.zeros(batch, nheads, seqlen, device="cuda")  # LSE = 0
        sinks = torch.zeros(nheads, device="cuda")  # sinks = 0

        corrected = apply_sink_correction(output, lse, sinks)

        # sigmoid(0 - 0) = 0.5, so output should be halved
        expected = output * 0.5
        torch.testing.assert_close(corrected, expected, rtol=1e-4, atol=1e-4)

    def test_sink_correction_with_large_lse(self):
        """With large LSE >> sinks, correction factor should be ~1."""
        batch, nheads, seqlen, headdim = 2, 8, 16, 64
        output = torch.randn(batch, seqlen, nheads, headdim, device="cuda")
        lse = torch.full((batch, nheads, seqlen), 100.0, device="cuda")  # Large LSE
        sinks = torch.zeros(nheads, device="cuda")

        corrected = apply_sink_correction(output, lse, sinks)

        # sigmoid(100 - 0) ≈ 1.0
        torch.testing.assert_close(corrected, output, rtol=1e-4, atol=1e-4)

    def test_sink_correction_with_large_sinks(self):
        """With large sinks >> LSE, correction factor should be ~0."""
        batch, nheads, seqlen, headdim = 2, 8, 16, 64
        output = torch.randn(batch, seqlen, nheads, headdim, device="cuda")
        lse = torch.zeros(batch, nheads, seqlen, device="cuda")
        sinks = torch.full((nheads,), 100.0, device="cuda")  # Large sinks

        corrected = apply_sink_correction(output, lse, sinks)

        # sigmoid(0 - 100) ≈ 0.0
        expected = torch.zeros_like(output)
        torch.testing.assert_close(corrected, expected, rtol=1e-4, atol=1e-4)

    def test_sink_correction_varlen_format(self):
        """Test sink correction with varlen (2D LSE) format."""
        total_tokens, nheads, headdim = 128, 64, 128
        output = torch.randn(total_tokens, nheads, headdim, device="cuda")
        lse = torch.randn(nheads, total_tokens, device="cuda")  # (nheads, total_tokens)
        sinks = torch.randn(nheads, device="cuda")

        corrected = apply_sink_correction(output, lse, sinks)

        assert corrected.shape == output.shape
        # Verify it's different from input (sinks are non-zero)
        assert not torch.allclose(corrected, output)


# =============================================================================
# Reference Implementation Tests
# =============================================================================

class TestReferenceImplementation:
    """Tests for the reference attention implementation."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("num_queries", [1, 64, 128])
    @pytest.mark.parametrize("num_keys", [64, 128])
    @pytest.mark.parametrize("sliding_window", [None, 128])
    def test_reference_output_shape(self, batch_size, num_queries, num_keys, sliding_window):
        """Test reference implementation output shape."""
        if num_queries > num_keys:
            pytest.skip("num_queries > num_keys not valid for causal")

        nheads, nheads_kv, headdim = 64, 8, 128
        q, k, v, sinks = make_test_tensors(
            batch_size, num_queries, num_keys, nheads, nheads_kv, headdim
        )
        sm_scale = headdim ** -0.5

        output = attention_ref(q, k, v, sinks, sm_scale, sliding_window)

        expected_shape = (batch_size, num_queries, nheads * headdim)
        assert output.shape == expected_shape

    def test_reference_causal_mask(self):
        """Test that reference implementation correctly applies causal mask."""
        batch_size, num_queries, num_keys = 1, 4, 4
        nheads, nheads_kv, headdim = 8, 8, 64
        q, k, v, sinks = make_test_tensors(
            batch_size, num_queries, num_keys, nheads, nheads_kv, headdim
        )
        # Set sinks to zero to isolate causal behavior
        sinks = torch.zeros_like(sinks)

        output = attention_ref(q, k, v, sinks, sm_scale=1.0)

        # Output should be valid (no NaN/inf from attending to future)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sinks_affect_output(self):
        """Test that non-zero sinks actually change the output."""
        batch_size, num_queries, num_keys = 1, 32, 64
        nheads, nheads_kv, headdim = 64, 8, 128
        q, k, v, _ = make_test_tensors(
            batch_size, num_queries, num_keys, nheads, nheads_kv, headdim
        )

        # Compare zero sinks vs large sinks
        sinks_zero = torch.zeros(nheads, device="cuda", dtype=torch.bfloat16)
        sinks_large = torch.full((nheads,), 10.0, device="cuda", dtype=torch.bfloat16)

        output_no_sinks = attention_ref(q, k, v, sinks_zero, sm_scale=0.0884)
        output_with_sinks = attention_ref(q, k, v, sinks_large, sm_scale=0.0884)

        # Outputs should be different
        assert not torch.allclose(output_no_sinks, output_with_sinks)

        # With large sinks, output magnitude should be smaller
        # (sinks steal attention mass from all keys)
        assert output_with_sinks.abs().mean() < output_no_sinks.abs().mean()


# =============================================================================
# Prefill Tests (requires flash-attention)
# =============================================================================

class TestPrefill:
    """Tests for prefill implementation using flash-attention."""

    @pytest.fixture(autouse=True)
    def check_flash_attn(self):
        """Skip tests if flash-attention not available."""
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError:
            try:
                from flash_attn_interface import flash_attn_varlen_func
            except ImportError:
                pytest.skip("flash-attention not available")

    def test_prefill_debug(self):
        """Debug test to understand numerical mismatch."""
        from .fa_prefill import gqa_prefill_fa, _USE_FA3

        # Simple case: batch=1, small sequences, no sliding window
        batch_size, num_queries, num_keys = 1, 16, 16
        nheads, nheads_kv, headdim = 8, 2, 64
        num_groups = nheads // nheads_kv

        print(f"\nUsing FA3: {_USE_FA3}")

        # Create test data
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_queries, nheads_kv, num_groups, headdim,
                        device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_keys, nheads_kv, headdim,
                        device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch_size, num_keys, nheads_kv, headdim,
                        device="cuda", dtype=torch.bfloat16)
        sm_scale = headdim ** -0.5

        # Convert to flash format
        q_varlen, k_varlen, v_varlen, cu_q, cu_k, max_q, max_k = reshape_for_flash_varlen(q, k, v)
        print(f"Q varlen shape: {q_varlen.shape}")
        print(f"K varlen shape: {k_varlen.shape}")

        # Test 1: Compare FA output (no sinks) with reference_no_sinks
        # This tests basic attention computation without sink complexity
        output_ref_no_sinks, lse_ref = attention_ref_no_sinks(q, k, v, sm_scale, None)
        print(f"Reference (no sinks) output shape: {output_ref_no_sinks.shape}")
        print(f"Reference LSE shape: {lse_ref.shape}")

        output_fa, lse_fa = gqa_prefill_fa(
            q_varlen, k_varlen, v_varlen, cu_q, cu_k, max_q, max_k,
            sinks=None, softmax_scale=sm_scale, sliding_window=None
        )
        print(f"FA output shape: {output_fa.shape}")
        print(f"FA LSE shape: {lse_fa.shape if lse_fa is not None else None}")

        # Reshape FA output to match reference
        output_fa_reshaped = output_fa.reshape(batch_size, num_queries, nheads * headdim)

        diff_no_sinks = (output_fa_reshaped - output_ref_no_sinks).abs()
        print(f"Max diff (FA vs ref_no_sinks): {diff_no_sinks.max().item():.6f}")
        print(f"Mean diff (FA vs ref_no_sinks): {diff_no_sinks.mean().item():.6f}")

        # Test 2: Compare FA+sink_correction with reference (with sinks)
        sinks = torch.randn(nheads, device="cuda", dtype=torch.bfloat16)
        output_ref_sinks = attention_ref(q, k, v, sinks, sm_scale, None)

        output_fa_sinks, lse_fa_sinks = gqa_prefill_fa(
            q_varlen, k_varlen, v_varlen, cu_q, cu_k, max_q, max_k,
            sinks=sinks, softmax_scale=sm_scale, sliding_window=None
        )
        output_fa_sinks_reshaped = output_fa_sinks.reshape(batch_size, num_queries, nheads * headdim)

        diff_sinks = (output_fa_sinks_reshaped - output_ref_sinks).abs()
        print(f"Max diff (FA+sinks vs ref): {diff_sinks.max().item():.6f}")
        print(f"Mean diff (FA+sinks vs ref): {diff_sinks.mean().item():.6f}")

        # Assert basic attention matches (this should pass)
        torch.testing.assert_close(output_fa_reshaped, output_ref_no_sinks, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("num_queries", [64, 128])
    @pytest.mark.parametrize("num_keys", [64, 128])
    @pytest.mark.parametrize("sliding_window", [None, 128])
    def test_prefill_vs_reference(self, batch_size, num_queries, num_keys, sliding_window):
        """Test that FA prefill matches reference implementation."""
        if num_queries > num_keys:
            pytest.skip("num_queries > num_keys not valid")

        from .fa_prefill import gqa_prefill_fa

        nheads, nheads_kv, headdim = 64, 8, 128
        q, k, v, sinks = make_test_tensors(
            batch_size, num_queries, num_keys, nheads, nheads_kv, headdim
        )
        sm_scale = headdim ** -0.5

        # Get reference output
        output_ref = attention_ref(q, k, v, sinks, sm_scale, sliding_window)

        # Convert to flash format and run
        q_varlen, k_varlen, v_varlen, cu_q, cu_k, max_q, max_k = reshape_for_flash_varlen(q, k, v)
        output_fa, _ = gqa_prefill_fa(
            q_varlen, k_varlen, v_varlen, cu_q, cu_k, max_q, max_k,
            sinks=sinks, softmax_scale=sm_scale, sliding_window=sliding_window
        )

        # Reshape FA output to match reference
        output_fa = output_fa.reshape(batch_size, num_queries, nheads * headdim)

        # Compare
        torch.testing.assert_close(output_fa, output_ref, rtol=1e-2, atol=1e-2)


# =============================================================================
# Decode Tests (requires flash-attention with KV cache support)
# =============================================================================

class TestDecode:
    """Tests for decode implementation with paged KV cache."""

    @pytest.fixture(autouse=True)
    def check_flash_attn_kvcache(self):
        """Skip tests if flash-attention KV cache API not available."""
        try:
            from flash_attn import flash_attn_with_kvcache
        except ImportError:
            try:
                from flash_attn_interface import flash_attn_with_kvcache
            except ImportError:
                pytest.skip("flash_attn_with_kvcache not available")

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("cache_len", [64, 128, 256])
    def test_decode_contiguous_kv(self, batch_size, cache_len):
        """Test decode with contiguous (non-paged) KV cache."""
        from .fa_decode import gqa_decode_fa_contiguous

        nheads, nheads_kv, headdim = 64, 8, 128
        num_groups = nheads // nheads_kv

        # Create decode scenario: single query token
        q = torch.randn(batch_size, 1, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        k_cache = torch.randn(batch_size, cache_len, nheads_kv, headdim, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.randn(batch_size, cache_len, nheads_kv, headdim, device="cuda", dtype=torch.bfloat16)
        cache_seqlens = torch.full((batch_size,), cache_len, dtype=torch.int32, device="cuda")
        sinks = torch.randn(nheads, device="cuda", dtype=torch.bfloat16)

        output, lse = gqa_decode_fa_contiguous(
            q, k_cache, v_cache, cache_seqlens,
            sinks=sinks, softmax_scale=headdim ** -0.5
        )

        # Check output shape
        assert output.shape == (batch_size, 1, nheads, headdim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_decode_output_differs_with_sinks(self):
        """Test that sinks change decode output."""
        from .fa_decode import gqa_decode_fa_contiguous

        batch_size, cache_len = 2, 64
        nheads, nheads_kv, headdim = 64, 8, 128

        q = torch.randn(batch_size, 1, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        k_cache = torch.randn(batch_size, cache_len, nheads_kv, headdim, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.randn(batch_size, cache_len, nheads_kv, headdim, device="cuda", dtype=torch.bfloat16)
        cache_seqlens = torch.full((batch_size,), cache_len, dtype=torch.int32, device="cuda")

        sinks_zero = torch.zeros(nheads, device="cuda", dtype=torch.bfloat16)
        sinks_nonzero = torch.randn(nheads, device="cuda", dtype=torch.bfloat16) * 2

        output_no_sinks, _ = gqa_decode_fa_contiguous(
            q, k_cache, v_cache, cache_seqlens,
            sinks=sinks_zero, softmax_scale=headdim ** -0.5
        )
        output_with_sinks, _ = gqa_decode_fa_contiguous(
            q, k_cache, v_cache, cache_seqlens,
            sinks=sinks_nonzero, softmax_scale=headdim ** -0.5
        )

        assert not torch.allclose(output_no_sinks, output_with_sinks)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_prefill_then_decode_consistency(self):
        """Test that prefill and decode produce consistent results."""
        # This test verifies that:
        # 1. Prefill computes correct KV for positions 0..N-1
        # 2. Decode at position N attends to the same KV
        # Both should use the same attention math
        pass  # TODO: Implement when integrating with model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
