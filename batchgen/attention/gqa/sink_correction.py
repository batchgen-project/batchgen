"""Attention sink correction for GQA.

Attention sinks modify the softmax normalization by adding a learned per-head
value to the denominator. This can be applied as a post-correction to standard
flash-attention output.

Math:
    Standard softmax: scores = exp(logits) / sum(exp(logits))
    With sinks: scores = exp(logits) / (sum(exp(logits)) + exp(sinks))

    Given LSE = logsumexp(logits), the correction factor is:
    factor = exp(LSE) / (exp(LSE) + exp(sinks)) = sigmoid(LSE - sinks)

    output_with_sinks = output * factor

Numerical Stability:
    - LSE can be -inf when all attention positions are masked (padding)
    - LSE - sinks can be very large/small, causing sigmoid saturation
    - This implementation clamps extreme values to avoid NaN/overflow
"""

import os
import torch

# Debug counter to limit output
_debug_call_count = 0


def apply_sink_correction(
    output: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Apply attention sink correction to flash-attention output.

    Numerically stable version that handles:
    - Very large/small LSE values
    - -inf LSE from fully masked positions (padding)
    - Extreme sigmoid inputs that would cause saturation

    Args:
        output: Attention output from flash-attention.
            - Prefill (varlen): (total_tokens, nheads, headdim)
            - Decode: (batch, seqlen, nheads, headdim)
        lse: Log-sum-exp values from flash-attention.
            - Prefill (varlen): (nheads, total_tokens) or (batch, nheads, seqlen)
            - Decode: (batch, nheads, seqlen)
        sinks: Per-head sink values, shape (nheads,).

    Returns:
        Corrected output with same shape as input.
    """
    nheads = sinks.shape[0]
    sinks = sinks.float()  # Ensure float32 for numerical stability

    output_dtype = output.dtype

    # Clamp bounds for sigmoid stability
    # sigmoid(x) saturates at ~1e-9 for x < -20, ~1-1e-9 for x > 20
    # Using +-20 to avoid numerical issues while preserving meaningful gradients
    SIGMOID_CLAMP_MIN = -20.0
    SIGMOID_CLAMP_MAX = 20.0

    # Floor for -inf replacement: large enough negative that sigmoid(-1e10 - sink) -> 0
    # This effectively zeros output for fully-masked positions (expected behavior)
    NEG_INF_REPLACEMENT = -1e10

    if lse.dim() == 2:
        # Varlen prefill: lse is (nheads, total_tokens)
        # output is (total_tokens, nheads, headdim)
        # Transpose lse to (total_tokens, nheads) for broadcasting
        lse_transposed = lse.T.float()  # (total_tokens, nheads)
        sinks_broadcast = sinks.view(1, nheads)  # (1, nheads)

        # Handle -inf LSE (from fully masked positions / padding)
        # Replace -inf with large negative to avoid NaN in subsequent operations
        lse_safe = torch.where(
            torch.isinf(lse_transposed) & (lse_transposed < 0),
            torch.full_like(lse_transposed, NEG_INF_REPLACEMENT),
            lse_transposed
        )

        # Compute sigmoid with clamping for numerical stability
        diff = lse_safe - sinks_broadcast
        diff_clamped = torch.clamp(diff, min=SIGMOID_CLAMP_MIN, max=SIGMOID_CLAMP_MAX)
        factor = torch.sigmoid(diff_clamped)  # (total_tokens, nheads)

        # DEBUG: Print sink correction details for prefill
        global _debug_call_count
        if os.environ.get("BATCHGEN_DEBUG_SINK", "0") == "1" and _debug_call_count < 3:
            _debug_call_count += 1
            with torch.no_grad():
                lse_min = lse_transposed.min().item()
                lse_max = lse_transposed.max().item()
                lse_mean = lse_transposed.mean().item()
                sink_min = sinks.min().item()
                sink_max = sinks.max().item()
                factor_min = factor.min().item()
                factor_max = factor.max().item()
                factor_mean = factor.mean().item()
                num_near_zero = (factor < 0.1).sum().item()
                total_factors = factor.numel()

                print(f"\n[SINK CORRECTION PREFILL] Call #{_debug_call_count}")
                print(f"[SINK PREFILL] LSE shape={lse_transposed.shape}, min={lse_min:.4f}, max={lse_max:.4f}, mean={lse_mean:.4f}")
                print(f"[SINK PREFILL] Sinks min={sink_min:.4f}, max={sink_max:.4f}")
                print(f"[SINK PREFILL] Factor min={factor_min:.6f}, max={factor_max:.6f}, mean={factor_mean:.6f}")
                print(f"[SINK PREFILL] Factor near zero (<0.1): {num_near_zero}/{total_factors}")

        # Apply correction
        result = output.float() * factor.unsqueeze(-1)  # (total_tokens, nheads, 1)
        return result.to(output_dtype)

    elif lse.dim() == 3:
        # Padded prefill or decode: lse is (batch, nheads, seqlen)
        # output is (batch, seqlen, nheads, headdim)
        lse_float = lse.float()  # (batch, nheads, seqlen)
        sinks_broadcast = sinks.view(1, nheads, 1)  # (1, nheads, 1)

        # Handle -inf LSE (from fully masked positions / padding)
        lse_safe = torch.where(
            torch.isinf(lse_float) & (lse_float < 0),
            torch.full_like(lse_float, NEG_INF_REPLACEMENT),
            lse_float
        )

        # Compute sigmoid with clamping for numerical stability
        diff = lse_safe - sinks_broadcast
        diff_clamped = torch.clamp(diff, min=SIGMOID_CLAMP_MIN, max=SIGMOID_CLAMP_MAX)
        factor = torch.sigmoid(diff_clamped)  # (batch, nheads, seqlen)

        # DEBUG: Print sink correction details
        global _debug_call_count
        if os.environ.get("BATCHGEN_DEBUG_SINK", "0") == "1" and _debug_call_count < 5:
            _debug_call_count += 1
            with torch.no_grad():
                # LSE stats
                lse_min = lse_float.min().item()
                lse_max = lse_float.max().item()
                lse_mean = lse_float.mean().item()

                # Sink stats
                sink_min = sinks.min().item()
                sink_max = sinks.max().item()
                sink_mean = sinks.mean().item()

                # Factor stats (before transpose)
                factor_min = factor.min().item()
                factor_max = factor.max().item()
                factor_mean = factor.mean().item()

                # Check for problematic values
                num_near_zero = (factor < 0.1).sum().item()
                num_near_one = (factor > 0.9).sum().item()
                total_factors = factor.numel()

                # Output stats before correction
                out_min = output.min().item()
                out_max = output.max().item()
                out_mean = output.float().mean().item()

                print(f"\n[SINK CORRECTION DEBUG] Call #{_debug_call_count}")
                print(f"[SINK] LSE shape={lse_float.shape}, min={lse_min:.4f}, max={lse_max:.4f}, mean={lse_mean:.4f}")
                print(f"[SINK] Sinks shape={sinks.shape}, min={sink_min:.4f}, max={sink_max:.4f}, mean={sink_mean:.4f}")
                print(f"[SINK] First 8 sink values: {sinks[:8].tolist()}")
                print(f"[SINK] Factor min={factor_min:.6f}, max={factor_max:.6f}, mean={factor_mean:.6f}")
                print(f"[SINK] Factor distribution: {num_near_zero}/{total_factors} near 0, {num_near_one}/{total_factors} near 1")
                print(f"[SINK] Output before correction: min={out_min:.4f}, max={out_max:.4f}, mean={out_mean:.4f}")

                # Sample LSE and factor for first batch, first head
                if lse_float.shape[0] > 0 and lse_float.shape[1] > 0:
                    lse_sample = lse_float[0, 0, :].tolist()  # batch 0, head 0
                    factor_sample = factor[0, 0, :].tolist()  # batch 0, head 0
                    print(f"[SINK] LSE[0,0,:] (batch0, head0): {lse_sample}")
                    print(f"[SINK] Factor[0,0,:] (batch0, head0): {factor_sample}")

        # Transpose factor to match output: (batch, seqlen, nheads)
        factor = factor.transpose(1, 2)  # (batch, seqlen, nheads)
        result = output.float() * factor.unsqueeze(-1)  # (batch, seqlen, nheads, 1)
        return result.to(output_dtype)

    else:
        raise ValueError(f"Unexpected LSE shape: {lse.shape}, expected 2D or 3D")
