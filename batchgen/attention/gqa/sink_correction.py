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
"""

import torch


def apply_sink_correction(
    output: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Apply attention sink correction to flash-attention output.

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

    if lse.dim() == 2:
        # Varlen prefill: lse is (nheads, total_tokens)
        # output is (total_tokens, nheads, headdim)
        # Transpose lse to (total_tokens, nheads) for broadcasting
        lse_transposed = lse.T.float()  # (total_tokens, nheads)
        sinks_broadcast = sinks.view(1, nheads)  # (1, nheads)
        factor = torch.sigmoid(lse_transposed - sinks_broadcast)  # (total_tokens, nheads)
        result = output.float() * factor.unsqueeze(-1)  # (total_tokens, nheads, 1)
        return result.to(output_dtype)

    elif lse.dim() == 3:
        # Padded prefill or decode: lse is (batch, nheads, seqlen)
        # output is (batch, seqlen, nheads, headdim)
        batch, seqlen = output.shape[0], output.shape[1]
        lse_float = lse.float()  # (batch, nheads, seqlen)
        sinks_broadcast = sinks.view(1, nheads, 1)  # (1, nheads, 1)
        factor = torch.sigmoid(lse_float - sinks_broadcast)  # (batch, nheads, seqlen)
        # Transpose factor to match output: (batch, seqlen, nheads)
        factor = factor.transpose(1, 2)  # (batch, seqlen, nheads)
        result = output.float() * factor.unsqueeze(-1)  # (batch, seqlen, nheads, 1)
        return result.to(output_dtype)

    else:
        raise ValueError(f"Unexpected LSE shape: {lse.shape}, expected 2D or 3D")
