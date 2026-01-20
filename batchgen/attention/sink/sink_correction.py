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

"""Sink correction for attention output.

In some implementations (like FlashAttention with sinks), the sink token
may need to be corrected after the attention computation. This module
provides utilities for that correction.

Note: When using the softmax_with_sinks approach where sinks are included
in the softmax denominator, no additional correction is needed. This module
is provided for alternative implementations that may need post-hoc correction.
"""

import torch


def apply_sink_correction(
    attn_output: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Apply sink correction to attention output.

    This is used when the attention computation includes sink tokens in the
    output but they need to be factored out. The correction multiplies the
    output by a sigmoid-based factor.

    Formula: corrected_output = output * sigmoid(LSE - sinks)

    This corrects for sink tokens that were included in softmax normalization
    but shouldn't contribute to the actual output values.

    Args:
        attn_output: Attention output [batch, seq, hidden_size] or
                     [batch, seq, num_heads, head_dim]
        lse: Log-sum-exp from attention [batch, num_heads, seq]
        sinks: Per-head sink parameters [num_heads]

    Returns:
        Corrected attention output with same shape as input
    """
    # Reshape sinks for broadcasting: [num_heads] -> [1, num_heads, 1]
    sinks_expanded = sinks.view(1, -1, 1).to(lse.dtype)

    # Compute correction factor: sigmoid(LSE - sinks)
    # Shape: [batch, num_heads, seq]
    correction = torch.sigmoid(lse - sinks_expanded)

    # Apply correction
    if attn_output.dim() == 4:
        # Shape: [batch, seq, num_heads, head_dim]
        # Transpose correction to match: [batch, num_heads, seq] -> [batch, seq, num_heads, 1]
        correction = correction.transpose(1, 2).unsqueeze(-1)
        corrected = attn_output * correction
    else:
        # Shape: [batch, seq, hidden_size]
        # Need to reshape hidden_size to [num_heads, head_dim]
        batch, seq, hidden_size = attn_output.shape
        num_heads = sinks.shape[0]
        head_dim = hidden_size // num_heads

        output_reshaped = attn_output.view(batch, seq, num_heads, head_dim)
        correction = correction.transpose(1, 2).unsqueeze(-1)  # [batch, seq, num_heads, 1]
        corrected = (output_reshaped * correction).view(batch, seq, hidden_size)

    return corrected


def compute_sink_weight(
    logits_max: torch.Tensor,
    logits_sum_exp: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Compute the attention weight absorbed by sink tokens.

    This returns the probability mass that goes to the sink token(s),
    which can be useful for analysis and debugging.

    Args:
        logits_max: Max of attention logits [batch, num_heads, seq_q]
        logits_sum_exp: Sum of exp(logits - max) [batch, num_heads, seq_q]
        sinks: Per-head sink parameters [num_heads]

    Returns:
        Sink attention weights [batch, num_heads, seq_q] - the probability
        mass absorbed by each sink token
    """
    # Reshape sinks: [num_heads] -> [1, num_heads, 1]
    sinks_expanded = sinks.view(1, -1, 1).to(logits_max.dtype)

    # Compute exp(sink - max)
    exp_sink = torch.exp(sinks_expanded - logits_max)

    # Normalize: sink_weight = exp(sink - max) / (sum_exp + exp(sink - max))
    normalizer = logits_sum_exp + exp_sink
    sink_weight = exp_sink / normalizer

    return sink_weight
