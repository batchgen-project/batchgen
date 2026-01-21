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

import torch


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

        # Transpose factor to match output: (batch, seqlen, nheads)
        factor = factor.transpose(1, 2)  # (batch, seqlen, nheads)
        result = output.float() * factor.unsqueeze(-1)  # (batch, seqlen, nheads, 1)
        return result.to(output_dtype)

    else:
        raise ValueError(f"Unexpected LSE shape: {lse.shape}, expected 2D or 3D")
