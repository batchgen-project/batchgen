"""Common MLA prefix-cache extend-prefill helpers for attention wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from batchgen.models.wrappers.prefix_cache import (
    PrefixCachePrepackMetadata,
    ensure_prefix_cache_prepack_metadata,
)


@dataclass(frozen=True)
class MlaExtendSpec:
    """Static MLA dimensions needed by the prefix extend-prefill path."""

    num_heads: int
    kv_lora_rank: int
    softmax_scale: float


OutputProjectMlaFn = Callable[[torch.Tensor], torch.Tensor]


def run_prefix_mla_suffix_prefill_with_projected(
    *,
    wrapper: object,
    query_states: torch.Tensor,
    offload_kv: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaExtendSpec,
    output_projection: OutputProjectMlaFn,
    prefill_prefix_materialization: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run suffix-only MLA prefill from already projected suffix Q/KV."""
    if prefill_prefix_materialization is None:
        raise RuntimeError(
            "MLA prefix-cache suffix prefill requires GPU paged materialization"
        )
    attn_out = run_projected_mla_prefix_attention_from_gpu_pages(
        layer_idx=int(wrapper.layer_idx),
        query_states=query_states,
        offload_kv=offload_kv,
        metadata=metadata,
        spec=spec,
        materialization=prefill_prefix_materialization,
    )
    return output_projection(attn_out), offload_kv


def run_projected_mla_prefix_attention_from_gpu_pages(
    *,
    layer_idx: int,
    query_states: torch.Tensor,
    offload_kv: torch.Tensor | None,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaExtendSpec,
    materialization: object,
) -> torch.Tensor:
    """Run MLA prefix attention from materialized GPU compressed KV."""

    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    manager = materialization.manager
    if manager.config.has_v_cache:
        raise RuntimeError(
            "MLA GPU prefix materialization requires K-only compressed KV pages"
        )

    layer_idx = int(layer_idx)
    materialization.wait_for_layer(layer_idx)

    if not metadata.prefix_reuse_mode:
        raise RuntimeError(
            "MLA GPU prefix materialization requires prefix reuse"
        )
    if offload_kv is None:
        raise RuntimeError("MLA GPU prefix extend requires suffix KV")
    manager.append_layer_prefill_suffix_tokens(
        k_tensor=offload_kv,
        v_tensor=None,
        append_plan=materialization.append_plan,
        layer_idx=layer_idx,
    )

    blocked_k, blocked_v, block_table = manager.get_layer_kv_with_page_table(
        layer_idx
    )
    if blocked_v is not None:
        raise RuntimeError(
            "MLA GPU prefix materialization unexpectedly has V cache"
        )
    if block_table is None:
        raise RuntimeError("MLA GPU prefix materialization requires page table")

    return _run_flashinfer_mla_prefix_attention(
        query_states=query_states.contiguous(),
        blocked_k=blocked_k,
        block_table=block_table,
        cache_seqlens=materialization.append_plan.cache_seqlens,
        slot_indices=materialization.append_plan.slot_indices,
        metadata=metadata,
        spec=spec,
    )


def _run_flashinfer_mla_prefix_attention(
    *,
    query_states: torch.Tensor,
    blocked_k: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    slot_indices: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaExtendSpec,
) -> torch.Tensor:
    """Run FlashInfer MLA paged attention against materialized prefix pages."""
    from batchgen.attention.mla.flashinfer_extend import (
        run_flashinfer_mla_extend_prefill,
    )

    return run_flashinfer_mla_extend_prefill(
        query_states=query_states,
        compressed_kv_cache=blocked_k,
        page_table=block_table,
        slot_indices=slot_indices,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=metadata.cu_seqlens,
        kv_lora_rank=int(spec.kv_lora_rank),
        num_heads=int(spec.num_heads),
        softmax_scale=float(spec.softmax_scale),
    )
