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

"""Grouped Query Attention implementation with sink token support.

This module provides GQA attention for GPT-OSS-style models with:
- Grouped Query Attention (GQA) with configurable head ratio
- Learned sink tokens that absorb attention weight
- Sliding window attention support
- Both prefill and decode modes

Architecture reference: GPT-OSS-120B
- 64 query heads, 8 KV heads (8:1 ratio)
- head_dim = 64
- Alternating sliding (128 tokens) / full attention per layer

Uses FlashAttention when available for memory-efficient attention.
Falls back to vanilla PyTorch for debugging/testing only.

SINK TOKEN CONFIGURATION:
- USE_VANILLA_FOR_SINKS=False (default): Use FlashAttention with numerically stable
  sigmoid post-correction (fast, handles padding correctly)
- USE_VANILLA_FOR_SINKS=True: Use vanilla PyTorch with inline softmax_with_sinks
  (slower, for debugging only)

Set BATCHGEN_VANILLA_SINKS=1 environment variable to enable vanilla path.
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..sink import softmax_with_sinks

# Configuration: Use vanilla PyTorch vs FlashAttention for sink attention.
# - False (default): Use FlashAttention + numerically stable sigmoid post-correction (fast)
# - True: Use vanilla PyTorch with inline softmax_with_sinks (slower, for debugging)
# Can be overridden via environment variable: BATCHGEN_VANILLA_SINKS=1
USE_VANILLA_FOR_SINKS = os.environ.get("BATCHGEN_VANILLA_SINKS", "0") == "1"

# Log the flag value at import time for debugging
if os.environ.get("BATCHGEN_VANILLA_SINKS", "0") == "1" or os.environ.get("BATCHGEN_DEBUG_SINK", "0") == "1":
    print(f"[GQA ATTENTION] BATCHGEN_VANILLA_SINKS={os.environ.get('BATCHGEN_VANILLA_SINKS', '0')}, USE_VANILLA_FOR_SINKS={USE_VANILLA_FOR_SINKS}")

# Detect which flash attention version is available
_USE_FA3 = False
_flash_attn_func = None

try:
    from flash_attn_interface import flash_attn_func as _fa3_func
    _USE_FA3 = True
    _flash_attn_func = _fa3_func
except ImportError:
    pass

if _flash_attn_func is None:
    try:
        from flash_attn import flash_attn_func as _fa2_func
        _flash_attn_func = _fa2_func
    except ImportError:
        pass

# Import gqa_prefill_fa for the FA path with sink correction
from .fa_prefill import gqa_prefill_fa


def gqa_attention_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GQA attention for prefill phase.

    Computes full sequence attention with optional sliding window and sinks.
    Uses FlashAttention when available for memory efficiency.

    Args:
        query: Query tensor [batch, num_q_heads, seq_q, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_k, head_dim]
        value: Value tensor [batch, num_kv_heads, seq_k, head_dim]
        sinks: Per-head sink parameters [num_q_heads] or None
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        sliding_window: Window size for sliding attention, None for full
        attention_mask: Additional attention mask [batch, 1, seq_q, seq_k] or None

    Returns:
        Attention output [batch, num_q_heads, seq_q, head_dim]
    """
    batch, num_q_heads, seq_q, head_dim = query.shape
    _, num_kv_heads, seq_k, _ = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Use FlashAttention if available, UNLESS:
    # - sinks are present AND BATCHGEN_VANILLA_SINKS=1
    # The vanilla path uses correct inline softmax_with_sinks,
    # while FA path uses sigmoid post-correction which may have issues.
    # Check env var at runtime (not import time) for worker process compatibility
    use_vanilla_for_sinks = os.environ.get("BATCHGEN_VANILLA_SINKS", "0") == "1"
    use_vanilla = (sinks is not None and use_vanilla_for_sinks)

    if _flash_attn_func is not None and not use_vanilla:
        # Convert padded input to varlen format for gqa_prefill_fa
        # Input: [batch, heads, seq, dim] -> varlen: [total, heads, dim]
        # Use permute to get [batch, seq, heads, dim] then reshape
        q_varlen = query.permute(0, 2, 1, 3).reshape(batch * seq_q, num_q_heads, head_dim)
        k_varlen = key.permute(0, 2, 1, 3).reshape(batch * seq_k, num_kv_heads, head_dim)
        v_varlen = value.permute(0, 2, 1, 3).reshape(batch * seq_k, num_kv_heads, head_dim)

        # Create cumulative sequence lengths for uniform batches
        cu_seqlens_q = torch.arange(
            0, (batch + 1) * seq_q, seq_q,
            dtype=torch.int32, device=query.device
        )
        cu_seqlens_k = torch.arange(
            0, (batch + 1) * seq_k, seq_k,
            dtype=torch.int32, device=query.device
        )

        # Use gqa_prefill_fa which handles FA3/FA2 and sink correction
        output_varlen, lse = gqa_prefill_fa(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seq_q,
            max_seqlen_k=seq_k,
            sinks=sinks,
            softmax_scale=scale,
            sliding_window=sliding_window,
        )

        # Convert output back to padded format: [total, heads, dim] -> [batch, heads, seq, dim]
        # Reshape to [batch, seq, heads, dim] then permute to [batch, heads, seq, dim]
        attn_output = output_varlen.reshape(batch, seq_q, num_q_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        return attn_output

    # Vanilla PyTorch attention path
    # Used when: FlashAttention not available, OR (sinks present AND USE_VANILLA_FOR_SINKS=True)
    # This path uses correct inline softmax_with_sinks for accurate sink handling.
    if _flash_attn_func is None:
        import warnings
        warnings.warn(
            "FlashAttention not available, falling back to vanilla PyTorch attention. "
            "This will use excessive memory for long sequences!",
            RuntimeWarning
        )

    num_groups = num_q_heads // num_kv_heads

    # Repeat KV heads to match query heads
    key = key.repeat_interleave(num_groups, dim=1)
    value = value.repeat_interleave(num_groups, dim=1)

    # Compute attention scores
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Create causal mask
    causal_mask = torch.triu(
        torch.ones((seq_q, seq_k), dtype=torch.bool, device=query.device),
        diagonal=seq_k - seq_q + 1
    )

    # Apply sliding window mask if specified
    if sliding_window is not None and sliding_window > 0:
        q_idx = torch.arange(seq_q, device=query.device).unsqueeze(1)
        k_idx = torch.arange(seq_k, device=query.device).unsqueeze(0)
        offset = seq_k - seq_q
        distance = k_idx - (q_idx + offset)
        sliding_mask = distance < -sliding_window
        causal_mask = causal_mask | sliding_mask

    # Apply masks
    attn_scores = attn_scores.masked_fill(
        causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
    )

    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    # Compute attention weights with or without sinks
    if sinks is not None:
        attn_weights, _ = softmax_with_sinks(attn_scores, sinks, dim=-1, return_lse=False)
    else:
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, value)

    return attn_output


def gqa_attention_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GQA attention for decode phase (single token generation).

    Optimized for generating one token at a time with KV cache.

    Args:
        query: Query tensor [batch, num_q_heads, 1, head_dim]
        key: Key cache [batch, num_kv_heads, cache_len, head_dim]
        value: Value cache [batch, num_kv_heads, cache_len, head_dim]
        sinks: Per-head sink parameters [num_q_heads] or None
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        sliding_window: Window size for sliding attention, None for full
        cache_seqlens: Actual sequence lengths in cache [batch]

    Returns:
        Attention output [batch, num_q_heads, 1, head_dim]
    """
    batch, num_q_heads, seq_q, head_dim = query.shape
    batch, num_kv_heads, cache_len, head_dim = key.shape
    num_groups = num_q_heads // num_kv_heads

    assert seq_q == 1, "Decode mode expects single query token"

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Repeat KV heads to match query heads
    key = key.repeat_interleave(num_groups, dim=1)
    value = value.repeat_interleave(num_groups, dim=1)

    # Compute attention scores
    # [batch, num_q_heads, 1, cache_len]
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply sliding window mask if specified
    if sliding_window is not None and sliding_window > 0:
        k_idx = torch.arange(cache_len, device=query.device)
        # Current position is at cache_len (0-indexed)
        current_pos = cache_len - 1
        if cache_seqlens is not None:
            # Use actual sequence length
            current_pos = cache_seqlens.unsqueeze(-1) - 1  # [batch, 1]
            distance = k_idx.unsqueeze(0) - current_pos  # [batch, cache_len]
        else:
            distance = k_idx - current_pos

        sliding_mask = distance < -sliding_window
        if sliding_mask.dim() == 1:
            sliding_mask = sliding_mask.unsqueeze(0)  # [1, cache_len]
        sliding_mask = sliding_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, cache_len]

        attn_scores = attn_scores.masked_fill(sliding_mask, float("-inf"))

    # Mask future positions (shouldn't exist in decode but safety check)
    if cache_seqlens is not None:
        k_idx = torch.arange(cache_len, device=query.device)
        valid_mask = k_idx.unsqueeze(0) < cache_seqlens.unsqueeze(-1)  # [batch, cache_len]
        valid_mask = valid_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, cache_len]
        attn_scores = attn_scores.masked_fill(~valid_mask, float("-inf"))

    # Compute attention weights with or without sinks
    if sinks is not None:
        attn_weights, _ = softmax_with_sinks(attn_scores, sinks, dim=-1, return_lse=False)
    else:
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, value)

    return attn_output


def gqa_attention_with_sinks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    is_decode: bool = False,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified GQA attention with sinks for both prefill and decode.

    This is the main entry point for GPT-OSS-style attention with sinks.

    Args:
        query: Query tensor [batch, num_q_heads, seq_q, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_k, head_dim]
        value: Value tensor [batch, num_kv_heads, seq_k, head_dim]
        sinks: Per-head sink parameters [num_q_heads]
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        sliding_window: Window size for sliding attention, None for full
        is_decode: If True, use decode-optimized path
        cache_seqlens: For decode, actual sequence lengths in cache [batch]

    Returns:
        Attention output [batch, num_q_heads, seq_q, head_dim]
    """
    if is_decode:
        return gqa_attention_decode(
            query, key, value, sinks, scale, sliding_window, cache_seqlens
        )
    else:
        return gqa_attention_prefill(
            query, key, value, sinks, scale, sliding_window
        )


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [..., head_dim]
        k: Key tensor [..., head_dim]
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    head_dim = q.shape[-1]
    half_dim = head_dim // 2

    # Split into first and second half
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    # Get cos/sin for the positions
    cos_half = cos[..., :half_dim]
    sin_half = sin[..., :half_dim]

    # Apply rotation
    q_rot = torch.cat([q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=-1)
    k_rot = torch.cat([k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=-1)

    return q_rot, k_rot
