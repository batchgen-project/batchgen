"""GQA decode using flash-attention with paged KV cache and sink correction.

Uses flash_attn_with_kvcache for efficient decode with paged attention.
Supports both FA2 (Ampere) and FA3 (Hopper).
"""

import torch
from typing import Optional, Tuple


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
            - lse: Log-sum-exp values (batch, nheads, seqlen_q)
    """
    # Try FA3 first (Hopper), fall back to FA2
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        from flash_attn import flash_attn_with_kvcache

    # Set up window_size parameter
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    # Default scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # Call flash attention with KV cache
    result = flash_attn_with_kvcache(
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

    # Unpack result
    if isinstance(result, tuple):
        output = result[0]
        lse = result[1]
    else:
        raise RuntimeError("flash_attn_with_kvcache did not return LSE. "
                          "Make sure return_softmax_lse=True is supported.")

    # Apply sink correction if sinks provided
    if sinks is not None:
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
            - lse: Log-sum-exp values (batch, nheads, seqlen_q)
    """
    # Try FA3 first (Hopper), fall back to FA2
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        from flash_attn import flash_attn_with_kvcache

    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # Call without block_table for contiguous cache
    result = flash_attn_with_kvcache(
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
        output = result[0]
        lse = result[1]
    else:
        raise RuntimeError("flash_attn_with_kvcache did not return LSE.")

    if sinks is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse
