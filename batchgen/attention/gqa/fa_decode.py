"""GQA decode using flash-attention with paged KV cache and sink correction.

Uses flash_attn_with_kvcache for efficient decode with paged attention.
Supports both FA2 (Ampere) and FA3 (Hopper).
"""

import torch
from typing import Optional, Tuple

# Detect which flash attention version is available
_USE_FA3 = False
_flash_with_kvcache = None

try:
    from flash_attn_interface import flash_attn_with_kvcache as _fa3_with_kvcache
    _USE_FA3 = True
    _flash_with_kvcache = _fa3_with_kvcache
except ImportError:
    pass

if _flash_with_kvcache is None:
    try:
        from flash_attn import flash_attn_with_kvcache as _fa2_with_kvcache
        _flash_with_kvcache = _fa2_with_kvcache
    except ImportError:
        pass


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
            - lse: Log-sum-exp values or None if sinks not provided
    """
    if _flash_with_kvcache is None:
        raise ImportError("Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available")

    # Set up window_size parameter
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    # Default scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lse = None

    # FA3 uses page_table, FA2 uses block_table
    page_table_kwarg = "page_table" if _USE_FA3 else "block_table"

    if sinks is not None:
        # Need LSE for sink correction
        result = _flash_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            return_softmax_lse=True,
            **{page_table_kwarg: block_table},
        )
        # Handle return value - could be (output, lse) or (output, lse, ...)
        if isinstance(result, tuple):
            output = result[0]
            lse = result[1]
        else:
            output = result
    else:
        output = _flash_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            **{page_table_kwarg: block_table},
        )

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
            - lse: Log-sum-exp values or None if sinks not provided
    """
    if _flash_with_kvcache is None:
        raise ImportError("Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available")

    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lse = None

    if sinks is not None:
        result = _flash_with_kvcache(
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
            output = result
    else:
        output = _flash_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
        )

    if sinks is not None and lse is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse
