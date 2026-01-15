"""GQA prefill using flash-attention with optional sink correction.

Uses flash_attn_varlen_func for variable-length (unpadded) sequences.
Supports both FA2 (Ampere) and FA3 (Hopper).
"""

import torch
from typing import Optional, Tuple

# Detect which flash attention version is available
_USE_FA3 = False
_flash_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen_func
    _USE_FA3 = True
    _flash_varlen_func = _fa3_varlen_func
except ImportError:
    pass

if _flash_varlen_func is None:
    try:
        from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
        _flash_varlen_func = _fa2_varlen_func
    except ImportError:
        pass


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
            - lse: Log-sum-exp values or None if sinks not provided
    """
    if _flash_varlen_func is None:
        raise ImportError("Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available")

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

    if _USE_FA3:
        # FA3 interface - check what it actually returns
        result = _flash_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
        )
        # FA3 returns (output, lse) tuple
        if isinstance(result, tuple):
            output = result[0]
            lse = result[1] if len(result) > 1 else None
            # Debug: print what we got
            # print(f"FA3 returned tuple of length {len(result)}, lse shape: {lse.shape if lse is not None else None}")
        else:
            output = result
            # Debug: print result type
            # print(f"FA3 returned non-tuple: {type(result)}")
    else:
        # FA2 needs return_softmax_lse=True to get LSE
        if sinks is not None:
            result = _flash_varlen_func(
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
            if isinstance(result, tuple):
                output = result[0]
                lse = result[1]
            else:
                output = result
        else:
            output = _flash_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )

    # Apply sink correction if sinks provided
    if sinks is not None and lse is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse
