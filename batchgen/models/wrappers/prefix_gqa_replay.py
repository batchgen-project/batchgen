"""Common GQA prefix-cache replay helpers for attention wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from batchgen.models.wrappers.prefix_cache import PrefixCachePrepackMetadata


@dataclass(frozen=True)
class GqaReplaySpec:
    """Static GQA dimensions and optional attention modifiers."""

    num_kv_heads: int
    head_dim: int
    sinks: Optional[torch.Tensor] = None
    softmax_scale: Optional[float] = None
    sliding_window: Optional[int] = None


def run_prefix_gqa_prefill_attention(
    *,
    wrapper: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    spec: GqaReplaySpec,
) -> torch.Tensor:
    """Run GQA prefill attention with optional cached prefix K/V."""
    from batchgen.attention.gqa import gqa_prefill_fa

    cu_q = metadata.cu_seqlens.to(query.device)
    max_seqlen_q = metadata.max_seqlen
    if metadata.full_hit_mode:
        key_for_attn, value_for_attn, cu_k, max_seqlen_k = (
            wrapper.prefix_attention_kv_builder().build_gqa_full_hit_kv(
                metadata=metadata,
                num_heads=spec.num_kv_heads,
                head_dim=spec.head_dim,
                dtype=key.dtype,
                device=key.device,
            )
        )
    elif metadata.prefix_reuse_mode:
        key_for_attn, value_for_attn, cu_k, max_seqlen_k = (
            wrapper.prefix_attention_kv_builder().build_gqa_prefix_kv(
                key=key,
                value=value,
                metadata=metadata,
                num_heads=spec.num_kv_heads,
                head_dim=spec.head_dim,
            )
        )
    else:
        key_for_attn = key
        value_for_attn = value
        cu_k = cu_q
        max_seqlen_k = metadata.max_seqlen

    attn_output, _ = gqa_prefill_fa(
        q=query,
        k=key_for_attn,
        v=value_for_attn,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sinks=spec.sinks,
        softmax_scale=spec.softmax_scale,
        sliding_window=spec.sliding_window,
    )
    return attn_output
