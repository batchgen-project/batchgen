"""GQA extend prefill using FlashAttention paged KV cache."""

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
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run batched suffix-prefill attention over paged prefix+suffix KV.

    This is the paged-KV extend counterpart of varlen prefill attention.  The
    caller is responsible for writing the freshly computed suffix K/V into the
    paged cache before calling this function. ``cache_seqlens`` and
    ``cu_seqlens_k`` therefore describe the full logical KV lengths, while
    ``cu_seqlens_q`` describes only the suffix query lengths.
    """

    if _flash_with_kvcache is None:
        raise ImportError(
            "Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available"
        )

    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    page_table_kwarg = "page_table" if _USE_FA3 else "block_table"
    extra_kwargs = {page_table_kwarg: page_table}
    if _USE_FA3:
        # Variable-length extend prefill has two independent boundaries:
        # suffix query tokens and newly appended suffix KV tokens. BatchGen
        # writes suffix K/V into the paged cache before calling FA3, but FA3
        # still needs the new-KV segmentation to map varlen rows correctly.
        extra_kwargs["cu_seqlens_k_new"] = cu_seqlens_q
    result = _flash_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window_size,
        return_softmax_lse=sinks is not None,
        **extra_kwargs,
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
