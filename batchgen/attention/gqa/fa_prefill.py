"""GQA prefill using flash-attention with optional sink correction.

Uses flash_attn_varlen_func for variable-length (unpadded) sequences.
Supports both FA2 (Ampere) and FA3 (Hopper).
"""

import torch
from typing import Optional, Tuple

# Detect which flash attention version is available
_USE_FA3 = False
try:
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen_func
    _USE_FA3 = True
except ImportError:
    _fa3_varlen_func = None

try:
    from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
except ImportError:
    _fa2_varlen_func = None


def gqa_prefill_fa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GQA prefill using flash-attention with optional sink correction.

    Args:
        q: Query tensor, unpadded (total_q, nheads, headdim)
        k: Key tensor, unpadded (total_k, nheads_kv, headdim)
        v: Value tensor, unpadded (total_k, nheads_kv, headdim)
        cu_seqlens_q: Cumulative sequence lengths for Q (batch + 1,)
        cu_seqlens_k: Cumulative sequence lengths for K (batch + 1,)
        max_seqlen_q: Maximum sequence length in Q batch
        max_seqlen_k: Maximum sequence length in K batch
        sinks: Optional per-head sink values (nheads,)
        softmax_scale: Scale factor for QK^T (default: 1/sqrt(headdim))
        sliding_window: Optional sliding window size for local attention

    Returns:
        Tuple of:
            - output: Attention output (total_q, nheads, headdim)
            - lse: Log-sum-exp values (nheads, total_q) or None if sinks not provided
    """
    # Set up window_size parameter
    # Flash attention uses (window_size_left, window_size_right)
    # For causal with sliding window: (sliding_window - 1, 0)
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)  # No windowing

    # Default scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lse = None

    if _USE_FA3 and _fa3_varlen_func is not None:
        # FA3 (Hopper) - uses return_attn_probs to return (output, softmax_lse)
        if sinks is not None:
            # Need LSE for sink correction
            output, lse = _fa3_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                return_attn_probs=True,
            )
        else:
            # No sinks, don't need LSE
            output = _fa3_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )
    elif _fa2_varlen_func is not None:
        # FA2 (Ampere) - uses return_softmax_lse
        if sinks is not None:
            output, lse, *_ = _fa2_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                return_softmax_lse=True,
            )
        else:
            output = _fa2_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )
    else:
        raise ImportError("Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available")

    # Apply sink correction if sinks provided
    if sinks is not None and lse is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse
