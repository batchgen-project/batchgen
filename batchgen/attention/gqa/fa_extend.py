"""GQA suffix-prefill attention over a paged prefix KV cache."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


_USE_FA3 = False
_flash_with_kvcache = None

try:
    from flash_attn_interface import (
        flash_attn_with_kvcache as _fa3_with_kvcache,
    )

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


def gqa_extend_fa(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Attend suffix queries to full prefix+suffix paged KV.

    The caller writes the newly computed suffix K/V into ``k_cache`` and
    ``v_cache`` before this call. ``cache_seqlens`` therefore contains full
    prompt lengths while ``cu_seqlens_q`` segments only the computed suffix.
    """

    if _flash_with_kvcache is None:
        raise ImportError(
            "Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available"
        )

    window_size = (
        (int(sliding_window) - 1, 0)
        if sliding_window is not None and sliding_window > 0
        else (-1, -1)
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    page_table_kwarg = "page_table" if _USE_FA3 else "block_table"
    result = _flash_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=int(max_seqlen_q),
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window_size,
        return_softmax_lse=sinks is not None,
        **{page_table_kwarg: page_table},
    )

    if isinstance(result, tuple):
        output = result[0]
        lse = result[1] if len(result) > 1 else None
    else:
        output = result
        lse = None

    if sinks is not None and lse is not None:
        from .sink_correction import apply_sink_correction

        output = apply_sink_correction(output, lse, sinks)
    return output, lse
