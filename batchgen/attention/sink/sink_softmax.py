# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Softmax with learned attention sinks.

Implements softmax that includes sink tokens in the normalization without
multiplying them with values. This is used in GPT-OSS-style attention.

The sink token is a learnable per-head parameter that acts as a virtual
key position, absorbing attention weight but not contributing to output.

Mathematical formulation:
    Let s be the sink value for a head, and x be the attention logits.

    Standard softmax: softmax(x)_i = exp(x_i) / sum(exp(x))

    Sink softmax: sink_softmax(x)_i = exp(x_i) / (sum(exp(x)) + exp(s))

    The sink "absorbs" exp(s) worth of probability mass, which is subtracted
    from all other positions proportionally.
"""

import torch
from typing import Tuple, Optional


def softmax_with_sinks(
    logits: torch.Tensor,
    sinks: torch.Tensor,
    dim: int = -1,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute softmax with sink tokens included in normalization.

    The sink tokens are added to the softmax denominator but NOT multiplied
    with values. This effectively makes the sink "absorb" attention weight.

    Args:
        logits: Attention logits [batch, num_heads, seq_q, seq_k]
        sinks: Per-head sink parameters [num_heads]
        dim: Dimension to apply softmax over (default: -1, the key dimension)
        return_lse: If True, also return log-sum-exp for use in correction

    Returns:
        Tuple of:
            - Attention weights [batch, num_heads, seq_q, seq_k]
            - LSE [batch, num_heads, seq_q] if return_lse=True, else None
    """
    # Reshape sinks for broadcasting: [num_heads] -> [1, num_heads, 1, 1]
    sinks_expanded = sinks.view(1, -1, 1, 1).to(logits.dtype)

    # Compute max for numerical stability, including sinks
    logits_max = logits.max(dim=dim, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks_expanded, logits_max)

    # Compute exp values
    exp_logits = torch.exp(logits - logits_or_sinks_max)
    exp_sinks = torch.exp(sinks_expanded - logits_or_sinks_max)

    # Normalize: denominator includes sink
    # Sum over key dimension
    sum_exp = exp_logits.sum(dim=dim, keepdim=True)
    normalizer = sum_exp + exp_sinks

    # Attention weights (sink weight is NOT returned, just absorbed)
    attn_weights = exp_logits / normalizer

    # Compute LSE if requested
    lse = None
    if return_lse:
        # LSE = max + log(sum(exp(x - max)) + exp(sink - max))
        lse = logits_or_sinks_max.squeeze(dim) + torch.log(normalizer.squeeze(dim))

    return attn_weights, lse


def softmax_with_sinks_gqa(
    logits: torch.Tensor,
    sinks: torch.Tensor,
    num_kv_heads: int,
    num_q_heads: int,
) -> torch.Tensor:
    """Softmax with sinks for Grouped Query Attention.

    Handles the case where there are more query heads than KV heads,
    with sink parameters defined per query head.

    Args:
        logits: Attention logits [batch, num_q_heads, seq_q, seq_k]
        sinks: Per query head sink parameters [num_q_heads]
        num_kv_heads: Number of KV heads
        num_q_heads: Number of query heads (must be multiple of num_kv_heads)

    Returns:
        Attention weights [batch, num_q_heads, seq_q, seq_k]
    """
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be multiple of num_kv_heads"

    # Use the standard sink softmax with all query heads
    attn_weights, _ = softmax_with_sinks(logits, sinks, dim=-1, return_lse=False)

    return attn_weights


def fused_softmax_with_sinks_and_mask(
    logits: torch.Tensor,
    sinks: torch.Tensor,
    causal_mask: torch.Tensor,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Fused softmax with sinks, causal mask, and optional sliding window.

    This is a convenience function that applies the causal mask and sliding
    window before computing softmax with sinks.

    Args:
        logits: Attention logits [batch, num_heads, seq_q, seq_k]
        sinks: Per-head sink parameters [num_heads]
        causal_mask: Boolean mask, True where positions should be masked
        sliding_window: If not None, mask positions beyond this distance

    Returns:
        Attention weights [batch, num_heads, seq_q, seq_k]
    """
    # Apply causal mask
    masked_logits = logits.masked_fill(causal_mask, float("-inf"))

    # Apply sliding window mask if specified
    if sliding_window is not None and sliding_window > 0:
        batch, num_heads, seq_q, seq_k = logits.shape
        device = logits.device

        # Create distance matrix
        q_idx = torch.arange(seq_q, device=device).unsqueeze(1)  # [seq_q, 1]
        k_idx = torch.arange(seq_k, device=device).unsqueeze(0)  # [1, seq_k]

        # In decoding, seq_k > seq_q, so we need offset
        offset = seq_k - seq_q
        distance = k_idx - (q_idx + offset)

        # Mask positions too far back
        sliding_mask = distance < -sliding_window
        sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_q, seq_k]

        masked_logits = masked_logits.masked_fill(sliding_mask, float("-inf"))

    # Compute softmax with sinks
    attn_weights, _ = softmax_with_sinks(masked_logits, sinks, dim=-1, return_lse=False)

    return attn_weights
