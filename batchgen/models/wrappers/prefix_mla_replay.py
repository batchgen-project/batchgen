"""Common MLA prefix-cache replay helpers for attention wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

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
    [torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]
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
    if metadata.prefix_shared_tokens is None or metadata.full_seq_lengths is None:
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

    prefix_kv_builder = wrapper.prefix_attention_kv_builder()
    if prefill_prefix_materialization is not None:
        attn_out = run_projected_mla_prefix_attention_from_gpu_pages(
            prefix_kv_builder=prefix_kv_builder,
            query_states=query_states,
            offload_kv=offload_kv,
            metadata=metadata,
            spec=spec,
            materialization=prefill_prefix_materialization,
        )
    else:
        attn_out = _run_projected_mla_prefix_attention_normalized(
            prefix_kv_builder=prefix_kv_builder,
            query_states=query_states,
            offload_kv=offload_kv,
            metadata=metadata,
            spec=spec,
            page_size=wrapper.host_prefix_reader().page_size(),
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

    prefix_kv_builder = wrapper.prefix_attention_kv_builder()
    if prefill_prefix_materialization is not None:
        attn_out = run_projected_mla_prefix_attention_from_gpu_pages(
            prefix_kv_builder=prefix_kv_builder,
            query_states=query_states,
            offload_kv=None,
            metadata=metadata,
            spec=spec,
            materialization=prefill_prefix_materialization,
        )
    else:
        attn_out = _run_projected_mla_prefix_attention_normalized(
            prefix_kv_builder=prefix_kv_builder,
            query_states=query_states,
            offload_kv=None,
            metadata=metadata,
            spec=spec,
            page_size=wrapper.host_prefix_reader().page_size(),
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
    if prefill_prefix_materialization is not None:
        return run_projected_mla_prefix_attention_from_gpu_pages(
            prefix_kv_builder=prefix_kv_builder,
            query_states=query_states,
            offload_kv=offload_kv,
            metadata=metadata,
            spec=spec,
            materialization=prefill_prefix_materialization,
            attention_fn=attention_fn,
        )
    return _run_projected_mla_prefix_attention_normalized(
        prefix_kv_builder=prefix_kv_builder,
        query_states=query_states,
        offload_kv=offload_kv,
        metadata=metadata,
        spec=spec,
        page_size=page_size,
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

    materialization.wait_for_load()
    layer_idx = int(prefix_kv_builder.reader.layer_idx)

    if metadata.full_hit_mode:
        query_len = 1
    elif metadata.prefix_reuse_mode:
        if offload_kv is None:
            raise RuntimeError("MLA GPU prefix replay requires suffix KV")
        manager.append_layer_prefill_suffix_tokens(
            k_tensor=offload_kv,
            v_tensor=None,
            append_plan=materialization.append_plan,
            layer_idx=layer_idx,
        )
        query_len = int(metadata.max_seqlen)
    else:
        raise RuntimeError(
            "MLA GPU prefix materialization requires prefix reuse or full hit"
        )

    blocked_k, blocked_v, block_table = manager.get_layer_kv_with_page_table(
        layer_idx
    )
    if blocked_v is not None:
        raise RuntimeError("MLA GPU prefix materialization unexpectedly has V cache")
    if block_table is None:
        raise RuntimeError("MLA GPU prefix materialization requires page table")

    attention_fn = attention_fn or run_flash_mla_prefix_attention
    return attention_fn(
        query_states=query_states,
        blocked_k=blocked_k,
        block_table=block_table,
        cache_seqlens=materialization.append_plan.cache_seqlens,
        query_len=query_len,
        spec=spec,
    )


def _run_projected_mla_prefix_attention_normalized(
    *,
    prefix_kv_builder: object,
    query_states: torch.Tensor,
    offload_kv: torch.Tensor | None,
    metadata: PrefixCachePrepackMetadata,
    spec: MlaReplaySpec,
    page_size: int,
    attention_fn: PrefixMlaAttentionFn | None = None,
) -> torch.Tensor:
    if metadata.full_hit_mode:
        compressed_kv, _, _ = prefix_kv_builder.build_mla_full_hit_kv(
            metadata=metadata,
            kv_dim=spec.kv_dim,
            dtype=query_states.dtype,
            device=query_states.device,
        )
        if metadata.full_seq_lengths is None:
            raise RuntimeError("MLA full-hit replay requires full sequence lengths")
        cu_k_values = _build_cu_seqlens_values(metadata.full_seq_lengths)
        query_len = 1
    elif metadata.prefix_reuse_mode:
        if offload_kv is None:
            raise RuntimeError("MLA prefix replay requires suffix KV")
        compressed_kv, _, _ = prefix_kv_builder.build_mla_prefix_kv(
            key=offload_kv,
            metadata=metadata,
            kv_dim=spec.kv_dim,
        )
        if metadata.full_seq_lengths is None:
            raise RuntimeError("MLA prefix replay requires full sequence lengths")
        cu_k_values = _build_cu_seqlens_values(metadata.full_seq_lengths)
        query_len = int(metadata.max_seqlen)
    else:
        if offload_kv is None:
            raise RuntimeError("MLA prefill requires KV")
        compressed_kv = offload_kv
        if compressed_kv.dim() == 2:
            compressed_kv = compressed_kv.unsqueeze(1)
        cu_k_values = metadata.cu_seqlens_list()
        query_len = int(metadata.max_seqlen)

    blocked_k, block_table, cache_seqlens = block_mla_kv_by_sequence(
        compressed_kv=compressed_kv,
        cu_k_values=cu_k_values,
        page_size=page_size,
    )
    attention_fn = attention_fn or run_flash_mla_prefix_attention
    return attention_fn(
        query_states=query_states,
        blocked_k=blocked_k,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        query_len=query_len,
        spec=spec,
    )


def run_flash_mla_prefix_attention(
    *,
    query_states: torch.Tensor,
    blocked_k: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    query_len: int,
    spec: MlaReplaySpec,
) -> torch.Tensor:
    """Run FlashMLA against cached-prefix page blocks."""
    from batchgen.attention.mla.flashmla_backend import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens,
        int(spec.num_heads),
        int(query_len),
    )
    attn_out, _ = flash_mla_with_kvcache(
        query_states,
        blocked_k,
        block_table,
        cache_seqlens,
        int(spec.kv_lora_rank),
        tile_scheduler_metadata,
        num_splits,
        float(spec.softmax_scale),
        True,
    )
    return attn_out


def block_mla_kv_by_sequence(
    *,
    compressed_kv: torch.Tensor,
    cu_k_values: Sequence[int],
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert packed per-sequence MLA KV into FlashMLA page blocks."""
    if compressed_kv.dim() != 3:
        raise RuntimeError(
            f"MLA compressed KV must be [tokens, 1, dim], got "
            f"{tuple(compressed_kv.shape)}"
        )
    page_size = int(page_size)
    cu_values = [int(value) for value in cu_k_values]
    if len(cu_values) < 2:
        raise RuntimeError("MLA blocked KV build requires at least one sequence")

    page_blocks = []
    block_rows = []
    cache_lengths = []
    next_page_idx = 0
    for seq_idx in range(len(cu_values) - 1):
        start = cu_values[seq_idx]
        end = cu_values[seq_idx + 1]
        seq_len = end - start
        if seq_len <= 0:
            raise RuntimeError(
                f"MLA blocked KV build got empty sequence at index {seq_idx}"
            )
        segment = compressed_kv[start:end]
        num_pages = (seq_len + page_size - 1) // page_size
        padded_tokens = num_pages * page_size
        if padded_tokens != seq_len:
            padding = torch.zeros(
                padded_tokens - seq_len,
                compressed_kv.shape[1],
                compressed_kv.shape[2],
                dtype=compressed_kv.dtype,
                device=compressed_kv.device,
            )
            segment = torch.cat([segment, padding], dim=0)
        page_blocks.append(
            segment.contiguous().view(
                num_pages,
                page_size,
                compressed_kv.shape[1],
                compressed_kv.shape[2],
            )
        )
        block_rows.append(
            torch.arange(
                next_page_idx,
                next_page_idx + num_pages,
                dtype=torch.int32,
                device=compressed_kv.device,
            )
        )
        cache_lengths.append(seq_len)
        next_page_idx += num_pages

    blocked_k = torch.cat(page_blocks, dim=0)
    max_pages = max(int(row.numel()) for row in block_rows)
    block_table = torch.zeros(
        (len(block_rows), max_pages),
        dtype=torch.int32,
        device=compressed_kv.device,
    )
    for row_idx, row in enumerate(block_rows):
        block_table[row_idx, : row.numel()] = row
    cache_seqlens = torch.tensor(
        cache_lengths,
        dtype=torch.int32,
        device=compressed_kv.device,
    )
    return blocked_k, block_table, cache_seqlens


def _build_cu_seqlens_values(seq_lengths: Sequence[int]) -> list[int]:
    values = [0]
    running = 0
    for length in seq_lengths:
        running += int(length)
        values.append(running)
    return values
