"""GQA prefill using flash-attention with optional sink correction.

Uses flash_attn_varlen_func for variable-length (unpadded) sequences.
Supports both FA2 (Ampere) and FA3 (Hopper).
"""

import torch
from typing import Optional, Tuple


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
            - lse: Log-sum-exp values (nheads, total_q)
    """
    # Try FA3 first (Hopper), fall back to FA2
    try:
        from flash_attn_interface import flash_attn_varlen_func
    except ImportError:
        from flash_attn import flash_attn_varlen_func

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

    # Call flash attention
    result = flash_attn_varlen_func(
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

    # Unpack result - FA returns (output, lse, ...) when return_softmax_lse=True
    if isinstance(result, tuple):
        output = result[0]
        lse = result[1]
    else:
        raise RuntimeError("flash_attn_varlen_func did not return LSE. "
                          "Make sure return_softmax_lse=True is supported.")

    # Apply sink correction if sinks provided
    if sinks is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse
