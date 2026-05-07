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
    from batchgen.attention.prefix_aware_backend import (
        GqaPrefixAwareAttentionBackend,
    )

    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=wrapper.prefix_attention_kv_builder(),
        num_kv_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
        sinks=spec.sinks,
        softmax_scale=spec.softmax_scale,
        sliding_window=spec.sliding_window,
    )
    return backend.forward_prefill(
        query=query,
        key=key,
        value=value,
        metadata=metadata,
    )
