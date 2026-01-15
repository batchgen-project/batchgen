"""GQA decode using flash-attention with paged KV cache and sink correction.

Uses flash_attn_with_kvcache for efficient decode with paged attention.
Supports both FA2 (Ampere) and FA3 (Hopper).
"""

import torch
from typing import Optional, Tuple

# Detect which flash attention version is available
_USE_FA3 = False
try:
    from flash_attn_interface import flash_attn_with_kvcache as _fa3_with_kvcache
    _USE_FA3 = True
except ImportError:
    _fa3_with_kvcache = None

try:
    from flash_attn import flash_attn_with_kvcache as _fa2_with_kvcache
except ImportError:
    _fa2_with_kvcache = None


def gqa_decode_fa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GQA decode using flash-attention with paged KV cache and sink correction.

    Args:
        q: Query tensor (batch, seqlen_q, nheads, headdim)
            For standard decode, seqlen_q = 1
        k_cache: Paged key cache (num_blocks, page_size, nheads_kv, headdim)
        v_cache: Paged value cache (num_blocks, page_size, nheads_kv, headdim)
        cache_seqlens: Current sequence lengths (batch,) int32
        block_table: Page table mapping (batch, max_blocks_per_seq) int32
        sinks: Optional per-head sink values (nheads,)
        softmax_scale: Scale factor for QK^T (default: 1/sqrt(headdim))
        sliding_window: Optional sliding window size for local attention

    Returns:
        Tuple of:
            - output: Attention output (batch, seqlen_q, nheads, headdim)
            - lse: Log-sum-exp values (batch, nheads, seqlen_q) or None if sinks not provided
    """
    # Set up window_size parameter
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    # Default scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lse = None

    if _USE_FA3 and _fa3_with_kvcache is not None:
        # FA3 (Hopper) - uses return_attn_probs to return (output, softmax_lse)
        if sinks is not None:
            output, lse = _fa3_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                return_attn_probs=True,
            )
        else:
            output = _fa3_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )
    elif _fa2_with_kvcache is not None:
        # FA2 (Ampere) - uses return_softmax_lse
        if sinks is not None:
            result = _fa2_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                return_softmax_lse=True,
            )
            if isinstance(result, tuple):
                output, lse = result[0], result[1]
            else:
                output = result
        else:
            output = _fa2_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
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


def gqa_decode_fa_contiguous(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GQA decode with contiguous (non-paged) KV cache.

    Use this when KV cache is stored contiguously without page tables.

    Args:
        q: Query tensor (batch, seqlen_q, nheads, headdim)
        k_cache: Contiguous key cache (batch, max_seqlen, nheads_kv, headdim)
        v_cache: Contiguous value cache (batch, max_seqlen, nheads_kv, headdim)
        cache_seqlens: Current sequence lengths (batch,) int32
        sinks: Optional per-head sink values (nheads,)
        softmax_scale: Scale factor for QK^T
        sliding_window: Optional sliding window size

    Returns:
        Tuple of:
            - output: Attention output (batch, seqlen_q, nheads, headdim)
            - lse: Log-sum-exp values (batch, nheads, seqlen_q) or None if sinks not provided
    """
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lse = None

    if _USE_FA3 and _fa3_with_kvcache is not None:
        if sinks is not None:
            output, lse = _fa3_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                return_attn_probs=True,
            )
        else:
            output = _fa3_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )
    elif _fa2_with_kvcache is not None:
        if sinks is not None:
            result = _fa2_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                return_softmax_lse=True,
            )
            if isinstance(result, tuple):
                output, lse = result[0], result[1]
            else:
                output = result
        else:
            output = _fa2_with_kvcache(
                q,
                k_cache,
                v_cache,
                cache_seqlens=cache_seqlens,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )
    else:
        raise ImportError("Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available")

    if sinks is not None and lse is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse
