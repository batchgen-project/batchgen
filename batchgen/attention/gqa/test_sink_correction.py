"""Sanity check: compare FA3+sink_correction vs OpenAI reference sdpa.

This test validates that the FlashAttention + sigmoid post-correction approach
produces numerically equivalent results to the reference OpenAI implementation
that uses inline softmax with sink concatenation.

Usage:
    python -m batchgen.attention.gqa.test_sink_correction

Reference Implementation (gpt-oss/torch/model.py):
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K) * sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)  # Add sink column
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]  # Remove sink
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
"""

import torch
import math


def reference_sdpa_with_sinks(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    sliding_window: int = 0,
) -> torch.Tensor:
    """OpenAI reference implementation from gpt-oss/torch/model.py.

    This is the "ground truth" for how sink attention should work.

    Args:
        Q: Query tensor [n_tokens, n_heads, q_mult, head_dim]
            For standard GQA: q_mult = num_q_heads / num_kv_heads
        K: Key tensor [n_tokens, n_heads, head_dim]
            n_heads here is num_kv_heads
        V: Value tensor [n_tokens, n_heads, head_dim]
        sinks: Per-head sink values [n_q_heads] where n_q_heads = n_heads * q_mult
        sm_scale: Softmax scale (typically 1/sqrt(head_dim))
        sliding_window: Window size (0 = no sliding window)

    Returns:
        Attention output [n_tokens, n_q_heads * head_dim]
    """
    n_tokens, n_kv_heads, q_mult, head_dim = Q.shape
    assert K.shape == (n_tokens, n_kv_heads, head_dim)
    assert V.shape == (n_tokens, n_kv_heads, head_dim)

    # Expand K, V for GQA: [n_tokens, n_kv_heads, head_dim] -> [n_tokens, n_kv_heads, q_mult, head_dim]
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)

    # Reshape sinks: [n_q_heads] -> [n_kv_heads, q_mult, 1, 1] -> expand to [n_kv_heads, q_mult, n_tokens, 1]
    n_q_heads = n_kv_heads * q_mult
    S = sinks.view(n_kv_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)

    # Causal mask
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)

    # Sliding window mask
    if sliding_window > 0:
        mask = mask + torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")),
            diagonal=-sliding_window
        )

    # Compute attention: QK = Q @ K^T * scale
    # [n_tokens, n_kv_heads, q_mult, head_dim] @ [n_tokens, n_kv_heads, q_mult, head_dim]
    # -> [n_kv_heads, q_mult, n_tokens, n_tokens]
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK = QK * sm_scale

    # Apply causal mask
    QK = QK + mask[None, None, :, :]

    # Concatenate sink as extra column: [h, m, q, k] -> [h, m, q, k+1]
    QK = torch.cat([QK, S], dim=-1)

    # Softmax over k+1 dimension (including sink)
    W = torch.softmax(QK, dim=-1)

    # Remove sink column before value multiplication
    W = W[..., :-1]

    # Apply attention to values
    # [n_kv_heads, q_mult, n_tokens, n_tokens] @ [n_tokens, n_kv_heads, q_mult, head_dim]
    # -> [n_tokens, n_kv_heads, q_mult, head_dim]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)

    # Reshape output: [n_tokens, n_kv_heads, q_mult, head_dim] -> [n_tokens, n_q_heads * head_dim]
    return attn.reshape(n_tokens, -1)


def fa_style_sdpa_with_sinks(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    sliding_window: int = 0,
) -> torch.Tensor:
    """FlashAttention-style implementation with sigmoid post-correction.

    This simulates what FA3 does: compute standard attention, get LSE,
    then apply sink correction as post-processing.

    Args:
        Same as reference_sdpa_with_sinks

    Returns:
        Attention output [n_tokens, n_q_heads * head_dim]
    """
    from .sink_correction import apply_sink_correction

    n_tokens, n_kv_heads, q_mult, head_dim = Q.shape
    n_q_heads = n_kv_heads * q_mult

    # Expand K, V for GQA
    K_exp = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V_exp = V[:, :, None, :].expand(-1, -1, q_mult, -1)

    # Causal mask
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)

    # Sliding window mask
    if sliding_window > 0:
        mask = mask + torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")),
            diagonal=-sliding_window
        )

    # Compute attention WITHOUT sinks (simulating FA3 native behavior)
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K_exp)
    QK = QK * sm_scale
    QK = QK + mask[None, None, :, :]

    # Standard softmax (no sink)
    W = torch.softmax(QK, dim=-1)

    # Compute LSE for correction
    # logsumexp over the last dimension
    LSE = torch.logsumexp(QK, dim=-1)  # [n_kv_heads, q_mult, n_tokens]

    # Apply attention to values
    attn = torch.einsum("hmqk,khmd->qhmd", W, V_exp)
    attn = attn.reshape(n_tokens, n_q_heads, head_dim)  # [n_tokens, n_q_heads, head_dim]

    # Apply sink correction using the same function as FA3 path
    # LSE shape for sink_correction: (nheads, total_tokens)
    LSE_for_correction = LSE.view(n_q_heads, n_tokens)  # [n_q_heads, n_tokens]

    output_corrected = apply_sink_correction(attn, LSE_for_correction, sinks)

    return output_corrected.reshape(n_tokens, -1)


def test_sink_correction_numerical_equivalence():
    """Test that FA-style correction matches reference implementation."""
    print("=" * 60)
    print("Testing FA3 sink correction vs reference implementation")
    print("=" * 60)

    # Test parameters matching GPT-OSS-120B
    n_tokens = 64
    n_kv_heads = 8
    q_mult = 8  # 64 query heads / 8 kv heads
    n_q_heads = n_kv_heads * q_mult
    head_dim = 64
    sm_scale = 1.0 / math.sqrt(head_dim)

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Config: n_tokens={n_tokens}, n_kv_heads={n_kv_heads}, q_mult={q_mult}, head_dim={head_dim}")

    # Random inputs
    torch.manual_seed(42)
    Q = torch.randn(n_tokens, n_kv_heads, q_mult, head_dim, dtype=dtype, device=device)
    K = torch.randn(n_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
    V = torch.randn(n_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)

    # Test with various sink value ranges
    test_cases = [
        ("Small sinks (mean=0)", torch.randn(n_q_heads, dtype=dtype, device=device) * 0.1),
        ("Medium sinks (mean=2)", torch.randn(n_q_heads, dtype=dtype, device=device) + 2.0),
        ("Large sinks (mean=5)", torch.randn(n_q_heads, dtype=dtype, device=device) + 5.0),
        ("GPT-OSS-like (mixed)", torch.randn(n_q_heads, dtype=dtype, device=device) * 3.0 + 1.0),
    ]

    all_passed = True

    for case_name, sinks in test_cases:
        print(f"\n--- Test case: {case_name} ---")
        print(f"    Sinks range: [{sinks.min().item():.4f}, {sinks.max().item():.4f}], mean={sinks.mean().item():.4f}")

        # Reference implementation
        ref_output = reference_sdpa_with_sinks(Q, K, V, sinks, sm_scale)

        # FA-style with correction
        fa_output = fa_style_sdpa_with_sinks(Q, K, V, sinks, sm_scale)

        # Compare
        max_diff = (ref_output - fa_output).abs().max().item()
        mean_diff = (ref_output - fa_output).abs().mean().item()
        rel_diff = (ref_output - fa_output).abs() / (ref_output.abs() + 1e-8)
        max_rel_diff = rel_diff.max().item()

        print(f"    Max absolute diff: {max_diff:.6e}")
        print(f"    Mean absolute diff: {mean_diff:.6e}")
        print(f"    Max relative diff: {max_rel_diff:.6e}")

        # Check if within tolerance
        tolerance = 1e-4
        if max_diff < tolerance:
            print(f"    PASSED (max_diff < {tolerance})")
        else:
            print(f"    FAILED (max_diff >= {tolerance})")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


def test_padding_handling():
    """Test that padding tokens are handled correctly."""
    print("\n" + "=" * 60)
    print("Testing padding token handling")
    print("=" * 60)

    from .sink_correction import apply_sink_correction

    n_tokens = 32
    n_heads = 64
    head_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output with some values
    output = torch.randn(n_tokens, n_heads, head_dim, device=device)

    # Create LSE with some -inf values (simulating fully masked positions)
    lse = torch.randn(n_heads, n_tokens, device=device)
    lse[:, -8:] = float("-inf")  # Last 8 positions are "padding"

    sinks = torch.randn(n_heads, device=device) + 2.0

    print(f"LSE shape: {lse.shape}")
    print(f"LSE -inf positions: last 8 tokens")
    print(f"Sinks range: [{sinks.min().item():.4f}, {sinks.max().item():.4f}]")

    # Apply correction
    corrected = apply_sink_correction(output, lse, sinks)

    # Check that -inf positions produce valid (not NaN) output
    has_nan = torch.isnan(corrected).any().item()
    has_inf = torch.isinf(corrected).any().item()

    print(f"Output has NaN: {has_nan}")
    print(f"Output has Inf: {has_inf}")

    # The -inf LSE positions should have near-zero output
    # (since sigmoid(-inf - sink) -> 0)
    padding_output_mean = corrected[-8:].abs().mean().item()
    valid_output_mean = corrected[:-8].abs().mean().item()

    print(f"Mean |output| for padding positions: {padding_output_mean:.6e}")
    print(f"Mean |output| for valid positions: {valid_output_mean:.4f}")

    passed = not has_nan and not has_inf and padding_output_mean < 1e-6

    if passed:
        print("PASSED")
    else:
        print("FAILED")

    return passed


if __name__ == "__main__":
    test1 = test_sink_correction_numerical_equivalence()
    test2 = test_padding_handling()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Numerical equivalence: {'PASSED' if test1 else 'FAILED'}")
    print(f"Padding handling: {'PASSED' if test2 else 'FAILED'}")

    if test1 and test2:
        print("\nAll tests passed!")
        exit(0)
    else:
        print("\nSome tests failed!")
        exit(1)
