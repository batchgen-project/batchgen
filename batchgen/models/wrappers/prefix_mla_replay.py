"""Common MLA prefix-cache replay helpers for attention wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from batchgen.models.wrappers.prefix_cache import (
    PrefixCachePrepackMetadata,
    ensure_prefix_cache_prepack_metadata,
)


@dataclass(frozen=True)
class MlaReplaySpec:
    """Static MLA dimensions needed by the prefix replay kernel path."""

    kv_dim: int
    num_heads: int
    kv_lora_rank: int
    softmax_scale: float


ProjectSuffixMlaFn = Callable[
    [torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
]
ProjectQueryMlaFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
OutputProjectMlaFn = Callable[[torch.Tensor], torch.Tensor]
PrefixMlaAttentionFn = Callable[..., torch.Tensor]


def run_prefix_mla_suffix_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaReplaySpec,
    project_suffix_query_and_kv: ProjectSuffixMlaFn,
    output_projection: OutputProjectMlaFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run suffix-only MLA prefill using cached prefix KV."""
    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    if (
        metadata.prefix_shared_tokens is None
        or metadata.full_seq_lengths is None
    ):
        raise RuntimeError("MLA prefix replay requires prefix metadata")

    query_states, offload_kv = project_suffix_query_and_kv(
        hidden_states_2d,
        position_ids,
        max(metadata.full_seq_lengths),
    )
    return run_prefix_mla_suffix_prefill_with_projected(
        wrapper=wrapper,
        query_states=query_states,
        offload_kv=offload_kv,
        metadata=metadata,
        spec=spec,
        output_projection=output_projection,
    )


def run_prefix_mla_suffix_prefill_with_projected(
    *,
    wrapper: object,
    query_states: torch.Tensor,
    offload_kv: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaReplaySpec,
    output_projection: OutputProjectMlaFn,
    prefill_prefix_materialization: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run suffix-only MLA prefill from already projected suffix Q/KV."""
    metadata = ensure_prefix_cache_prepack_metadata(metadata)

    if prefill_prefix_materialization is None:
        raise RuntimeError(
            "MLA prefix-cache suffix prefill requires GPU paged materialization"
        )
    attn_out = run_projected_mla_prefix_attention_from_gpu_pages(
        prefix_kv_builder=wrapper.prefix_attention_kv_builder(),
        query_states=query_states,
        offload_kv=offload_kv,
        metadata=metadata,
        spec=spec,
        materialization=prefill_prefix_materialization,
    )
    return output_projection(attn_out), offload_kv


def run_prefix_mla_full_hit_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaReplaySpec,
    project_query: ProjectQueryMlaFn,
    output_projection: OutputProjectMlaFn,
) -> torch.Tensor:
    """Run exact full-hit MLA prefill using fully cached prompt KV."""
    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    if metadata.full_seq_lengths is None:
        raise RuntimeError("MLA full-hit replay requires full sequence lengths")

    query_states = project_query(
        hidden_states_2d,
        position_ids,
        max(metadata.full_seq_lengths),
    )
    return run_prefix_mla_full_hit_prefill_with_query(
        wrapper=wrapper,
        query_states=query_states,
        metadata=metadata,
        spec=spec,
        output_projection=output_projection,
    )


def run_prefix_mla_full_hit_prefill_with_query(
    *,
    wrapper: object,
    query_states: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaReplaySpec,
    output_projection: OutputProjectMlaFn,
    prefill_prefix_materialization: object | None = None,
) -> torch.Tensor:
    """Run exact full-hit MLA prefill from already projected query states."""
    metadata = ensure_prefix_cache_prepack_metadata(metadata)

    if prefill_prefix_materialization is None:
        raise RuntimeError(
            "MLA full-hit prefix prefill requires GPU paged materialization"
        )
    attn_out = run_projected_mla_prefix_attention_from_gpu_pages(
        prefix_kv_builder=wrapper.prefix_attention_kv_builder(),
        query_states=query_states,
        offload_kv=None,
        metadata=metadata,
        spec=spec,
        materialization=prefill_prefix_materialization,
    )
    return output_projection(attn_out)


def run_projected_mla_prefix_attention(
    *,
    prefix_kv_builder: object,
    query_states: torch.Tensor,
    offload_kv: torch.Tensor | None,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaReplaySpec,
    page_size: int,
    attention_fn: PrefixMlaAttentionFn | None = None,
    prefill_prefix_materialization: object | None = None,
) -> torch.Tensor:
    """Run MLA prefix/no-prefix attention from projected query and compressed KV."""

    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    del page_size
    if prefill_prefix_materialization is None:
        raise RuntimeError(
            "MLA prefix attention requires GPU paged materialization"
        )
    return run_projected_mla_prefix_attention_from_gpu_pages(
        prefix_kv_builder=prefix_kv_builder,
        query_states=query_states,
        offload_kv=offload_kv,
        metadata=metadata,
        spec=spec,
        materialization=prefill_prefix_materialization,
        attention_fn=attention_fn,
    )


def run_projected_mla_prefix_attention_from_gpu_pages(
    *,
    prefix_kv_builder: object,
    query_states: torch.Tensor,
    offload_kv: torch.Tensor | None,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaReplaySpec,
    materialization: object,
    attention_fn: PrefixMlaAttentionFn | None = None,
) -> torch.Tensor:
    """Run MLA prefix/full-hit attention from materialized GPU compressed KV."""

    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    manager = materialization.manager
    if manager.config.has_v_cache:
        raise RuntimeError(
            "MLA GPU prefix materialization requires K-only compressed KV pages"
        )

    layer_idx = int(prefix_kv_builder.reader.layer_idx)
    materialization.wait_for_layer(layer_idx)

    if metadata.prefix_reuse_mode:
        if offload_kv is None:
            raise RuntimeError("MLA GPU prefix replay requires suffix KV")
        manager.append_layer_prefill_suffix_tokens(
            k_tensor=offload_kv,
            v_tensor=None,
            append_plan=materialization.append_plan,
            layer_idx=layer_idx,
        )
        blocked_k, blocked_v, block_table = (
            manager.get_layer_kv_with_page_table(layer_idx)
        )
        if blocked_v is not None:
            raise RuntimeError(
                "MLA GPU prefix materialization unexpectedly has V cache"
            )
        if block_table is None:
            raise RuntimeError(
                "MLA GPU prefix materialization requires page table"
            )
        if attention_fn is not None:
            raise RuntimeError(
                "MLA prefix-cache suffix prefill must use FlashInfer paged "
                "MLA attention"
            )
        return _run_flashinfer_mla_prefix_attention(
            query_states=query_states,
            blocked_k=blocked_k,
            block_table=block_table,
            cache_seqlens=materialization.append_plan.cache_seqlens,
            slot_indices=materialization.append_plan.slot_indices,
            metadata=metadata,
            spec=spec,
        )
    if metadata.full_hit_mode:
        if offload_kv is not None:
            raise RuntimeError(
                "MLA full-hit prefix replay does not accept suffix KV"
            )
        if attention_fn is not None:
            raise RuntimeError(
                "MLA full-hit prefix replay must use FlashInfer paged MLA attention"
            )
    else:
        raise RuntimeError(
            "MLA GPU prefix materialization requires prefix reuse or full hit"
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
    spec: MlaReplaySpec,
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
