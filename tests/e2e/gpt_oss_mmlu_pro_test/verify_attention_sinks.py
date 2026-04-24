"""Verify attention sink correction produces equivalent results to OpenAI reference.

This test compares:
1. OpenAI: Inline softmax with sinks (concatenate sink column, softmax, remove)
2. BatchGen: FlashAttention + sigmoid post-correction

Math derivation:
- OpenAI: W[i] = exp(qk[i]) / (sum(exp(qk)) + exp(sink))
- BatchGen: output = FA_output * sigmoid(lse - sink)
  where lse = log(sum(exp(qk)))
  sigmoid(lse - sink) = exp(lse) / (exp(lse) + exp(sink))
                      = sum(exp(qk)) / (sum(exp(qk)) + exp(sink))

This is a per-row scaling factor that should give equivalent results.

Usage:
    python verify_attention_sinks.py
    BATCHGEN_VANILLA_SINKS=1 python verify_attention_sinks.py  # Compare vanilla path
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add BatchGen to path
BATCHGEN_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BATCHGEN_PATH))


def openai_sdpa_with_sinks(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    sliding_window: int = 0,
) -> torch.Tensor:
    """OpenAI's reference SDPA with sinks (from gpt-oss/torch/model.py).

    Args:
        Q: [n_tokens, n_heads, q_mult, head_dim] - query states
        K: [n_tokens, n_heads, head_dim] - key states
        V: [n_tokens, n_heads, head_dim] - value states
        sinks: [n_heads] - per-head sink parameters
        sm_scale: softmax scaling factor (1/sqrt(head_dim))
        sliding_window: window size for sliding attention (0 = full)

    Returns:
        Attention output [n_tokens, n_heads * q_mult * head_dim]
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)

    # Expand K, V for GQA
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)

    # Reshape sinks for broadcasting: [n_heads, q_mult, 1, 1]
    S = sinks.reshape(n_heads, 1, 1, 1).expand(-1, q_mult, n_tokens, -1)

    # Create causal mask
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )

    # Compute attention scores: [n_heads, q_mult, q_tokens, k_tokens]
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]

    # Concatenate sink column
    QK = torch.cat([QK, S], dim=-1)  # [n_heads, q_mult, q_tokens, k_tokens + 1]

    # Softmax (includes sink)
    W = torch.softmax(QK, dim=-1, dtype=torch.float32)

    # Remove sink column before matmul
    W = W[..., :-1].to(Q.dtype)

    # Compute output
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


def batchgen_sdpa_with_sink_correction(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    sliding_window: int = 0,
) -> torch.Tensor:
    """BatchGen's approach: Standard attention + sigmoid post-correction.

    Args:
        Q: [n_tokens, n_heads, q_mult, head_dim] - query states
        K: [n_tokens, n_heads, head_dim] - key states
        V: [n_tokens, n_heads, head_dim] - value states
        sinks: [n_heads] - per-head sink parameters
        sm_scale: softmax scaling factor (1/sqrt(head_dim))
        sliding_window: window size for sliding attention (0 = full)

    Returns:
        Attention output [n_tokens, n_heads * q_mult * head_dim]
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape

    # Expand K, V for GQA
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)

    # Create causal mask
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )

    # Compute attention scores: [n_heads, q_mult, q_tokens, k_tokens]
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]

    # Standard softmax (NO sinks)
    W = torch.softmax(QK, dim=-1, dtype=torch.float32)

    # Compute LSE for sink correction: log(sum(exp(qk)))
    # For numerical stability, use logsumexp
    lse = torch.logsumexp(QK.float(), dim=-1)  # [n_heads, q_mult, q_tokens]

    # Compute output without sink correction
    attn_no_sink = torch.einsum("hmqk,khmd->qhmd", W.to(Q.dtype), V)

    # Apply sink correction: sigmoid(lse - sink)
    # sinks: [n_heads] -> [n_heads, 1, 1] for broadcasting
    sink_broadcast = sinks.float().view(n_heads, 1, 1)
    correction = torch.sigmoid(lse - sink_broadcast)  # [n_heads, q_mult, q_tokens]

    # Transpose correction for multiplication: [n_heads, q_mult, q_tokens] -> [q_tokens, n_heads, q_mult, 1]
    correction = correction.permute(2, 0, 1).unsqueeze(-1)

    # Apply correction
    attn = attn_no_sink * correction.to(Q.dtype)

    return attn.reshape(n_tokens, -1)


def test_sink_equivalence():
    """Test that BatchGen's sigmoid post-correction matches OpenAI's inline approach."""
    print("=" * 60)
    print("Attention Sink Equivalence Test")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # GPT-OSS-120B configuration
    n_tokens = 10
    n_heads = 8  # num_kv_heads for GQA
    q_mult = 8   # query heads per KV head (64 / 8)
    head_dim = 64

    print(f"\nConfiguration:")
    print(f"  n_tokens: {n_tokens}")
    print(f"  n_heads (KV): {n_heads}")
    print(f"  q_mult: {q_mult} (total Q heads: {n_heads * q_mult})")
    print(f"  head_dim: {head_dim}")
    print(f"  device: {device}")

    # Create random Q, K, V, sinks
    Q = torch.randn(n_tokens, n_heads, q_mult, head_dim, dtype=dtype, device=device)
    K = torch.randn(n_tokens, n_heads, head_dim, dtype=dtype, device=device)
    V = torch.randn(n_tokens, n_heads, head_dim, dtype=dtype, device=device)

    # Test with various sink values
    sink_test_cases = [
        ("zeros", torch.zeros(n_heads, dtype=dtype, device=device)),
        ("small_positive", torch.full((n_heads,), 1.0, dtype=dtype, device=device)),
        ("small_negative", torch.full((n_heads,), -1.0, dtype=dtype, device=device)),
        ("large_positive", torch.full((n_heads,), 5.0, dtype=dtype, device=device)),
        ("large_negative", torch.full((n_heads,), -5.0, dtype=dtype, device=device)),
        ("random", torch.randn(n_heads, dtype=dtype, device=device)),
        ("realistic", torch.randn(n_heads, dtype=dtype, device=device) * 2),  # ~N(0, 2)
    ]

    sm_scale = 1.0 / math.sqrt(head_dim)
    all_pass = True

    for name, sinks in sink_test_cases:
        print(f"\n--- Test: {name} sinks ---")
        print(f"  Sink values: min={sinks.min().item():.4f}, max={sinks.max().item():.4f}, mean={sinks.float().mean().item():.4f}")

        # OpenAI reference
        out_openai = openai_sdpa_with_sinks(Q, K, V, sinks, sm_scale)

        # BatchGen approach
        out_batchgen = batchgen_sdpa_with_sink_correction(Q, K, V, sinks, sm_scale)

        # Compare
        diff = (out_openai.float() - out_batchgen.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Tolerances
        atol = 1e-2  # bfloat16 precision
        rtol = 1e-2

        match = torch.allclose(out_openai, out_batchgen, atol=atol, rtol=rtol)

        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Match (atol={atol}, rtol={rtol}): {match}")

        if not match:
            all_pass = False
            # Debug: Show sample values
            print(f"  OpenAI[0,:8]: {out_openai[0,:8].float().tolist()}")
            print(f"  BatchGen[0,:8]: {out_batchgen[0,:8].float().tolist()}")

    print("\n" + "=" * 60)
    if all_pass:
        print("✓ All sink equivalence tests PASSED")
        print("\nConclusion: Sigmoid post-correction is mathematically equivalent")
        print("            to OpenAI's inline softmax with sinks.")
    else:
        print("✗ Sink equivalence tests FAILED")
        print("\nThe post-correction approach may have numerical differences.")
    print("=" * 60)

    return all_pass


def test_extreme_sinks():
    """Test behavior with extreme sink values (edge cases)."""
    print("\n" + "=" * 60)
    print("Extreme Sink Values Test")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    n_tokens = 5
    n_heads = 4
    q_mult = 2
    head_dim = 32

    Q = torch.randn(n_tokens, n_heads, q_mult, head_dim, dtype=dtype, device=device)
    K = torch.randn(n_tokens, n_heads, head_dim, dtype=dtype, device=device)
    V = torch.randn(n_tokens, n_heads, head_dim, dtype=dtype, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim)

    # Extreme positive sinks should make output nearly zero
    # (sink absorbs all attention mass)
    extreme_positive = torch.full((n_heads,), 20.0, dtype=dtype, device=device)
    out_extreme_pos = batchgen_sdpa_with_sink_correction(Q, K, V, extreme_positive, sm_scale)

    print(f"\nExtreme positive sinks (value=20):")
    print(f"  Output norm: {out_extreme_pos.norm().item():.6f}")
    print(f"  Output max abs: {out_extreme_pos.abs().max().item():.6f}")

    # Extreme negative sinks should have almost no effect
    # (sink takes no attention mass)
    extreme_negative = torch.full((n_heads,), -20.0, dtype=dtype, device=device)
    out_extreme_neg = batchgen_sdpa_with_sink_correction(Q, K, V, extreme_negative, sm_scale)

    # Reference without sinks
    zero_sinks = torch.zeros(n_heads, dtype=dtype, device=device)
    out_no_sinks = batchgen_sdpa_with_sink_correction(Q, K, V, zero_sinks, sm_scale)

    print(f"\nExtreme negative sinks (value=-20):")
    print(f"  Output norm: {out_extreme_neg.norm().item():.6f}")
    print(f"  Compare to zero sinks norm: {out_no_sinks.norm().item():.6f}")

    neg_match = torch.allclose(out_extreme_neg, out_no_sinks, atol=0.1, rtol=0.1)
    print(f"  Similar to no-sink output: {neg_match}")

    # The key insight: large positive sinks should scale output toward zero
    ratio_pos = out_extreme_pos.norm() / out_no_sinks.norm()
    print(f"\nExtreme positive / no-sink ratio: {ratio_pos.item():.6f}")

    if ratio_pos < 0.01:
        print("WARNING: Extreme positive sinks scale output to near-zero!")
        print("         If checkpoint has large sink values, this could cause")
        print("         degenerate output (residual stream dominates).")


def test_sliding_window():
    """Test that sliding window attention works with sinks."""
    print("\n" + "=" * 60)
    print("Sliding Window + Sinks Test")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    n_tokens = 20
    n_heads = 4
    q_mult = 2
    head_dim = 32
    sliding_window = 8  # GPT-OSS uses 128, but smaller for test

    print(f"\nConfiguration:")
    print(f"  n_tokens: {n_tokens}")
    print(f"  sliding_window: {sliding_window}")

    Q = torch.randn(n_tokens, n_heads, q_mult, head_dim, dtype=dtype, device=device)
    K = torch.randn(n_tokens, n_heads, head_dim, dtype=dtype, device=device)
    V = torch.randn(n_tokens, n_heads, head_dim, dtype=dtype, device=device)

    sinks = torch.randn(n_heads, dtype=dtype, device=device)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # OpenAI reference with sliding window
    out_openai = openai_sdpa_with_sinks(Q, K, V, sinks, sm_scale, sliding_window=sliding_window)

    # BatchGen approach with sliding window
    out_batchgen = batchgen_sdpa_with_sink_correction(Q, K, V, sinks, sm_scale, sliding_window=sliding_window)

    # Compare
    diff = (out_openai.float() - out_batchgen.float()).abs()
    max_diff = diff.max().item()
    match = torch.allclose(out_openai, out_batchgen, atol=1e-2, rtol=1e-2)

    print(f"\n  Max diff: {max_diff:.6f}")
    print(f"  Match: {match}")

    if match:
        print("✓ Sliding window + sinks equivalence PASSED")
    else:
        print("✗ Sliding window + sinks equivalence FAILED")

    return match


def main():
    parser = argparse.ArgumentParser(description="Verify attention sink correction")
    parser.add_argument("--extreme", action="store_true", help="Test extreme sink values")
    parser.add_argument("--sliding", action="store_true", help="Test sliding window")
    args = parser.parse_args()

    results = {}

    # Always run basic equivalence test
    results["equivalence"] = test_sink_equivalence()

    if args.extreme:
        test_extreme_sinks()

    if args.sliding:
        results["sliding"] = test_sliding_window()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
