"""Common MLA prefix-cache replay helpers for attention wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

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
    if not metadata.prefix_reuse_mode:
        raise RuntimeError("MLA prefix replay requires prefix reuse mode")
    if metadata.num_sequences != 1:
        raise RuntimeError(
            "MLA prefix replay currently requires single-sequence suffix "
            "micro-batches"
        )
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run suffix-only MLA prefill from already projected suffix Q/KV."""
    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    if not metadata.prefix_reuse_mode:
        raise RuntimeError("MLA prefix replay requires prefix reuse mode")
    if metadata.num_sequences != 1:
        raise RuntimeError(
            "MLA prefix replay currently requires single-sequence suffix "
            "micro-batches"
        )
    if metadata.prefix_shared_tokens is None or metadata.full_seq_lengths is None:
        raise RuntimeError("MLA prefix replay requires prefix metadata")

    from batchgen.attention.prefix_aware_backend import (
        MlaProjectedPrefixAwareAttentionBackend,
    )

    backend = MlaProjectedPrefixAwareAttentionBackend(
        prefix_kv_builder=wrapper.prefix_attention_kv_builder(),
        page_size=wrapper.host_prefix_reader().page_size(),
        kv_dim=spec.kv_dim,
        num_heads=spec.num_heads,
        kv_lora_rank=spec.kv_lora_rank,
        softmax_scale=spec.softmax_scale,
        output_projection=output_projection,
        layer_idx=getattr(wrapper, "layer_idx", None),
    )
    attn_out = backend.forward_prefill(
        query=query_states,
        key=offload_kv,
        value=None,
        metadata=metadata,
    )
    return attn_out, offload_kv


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
    if not metadata.full_hit_mode:
        raise RuntimeError("MLA full-hit replay requires full-hit mode")
    if metadata.full_seq_lengths is None:
        raise RuntimeError("MLA full-hit replay requires full sequence lengths")
    metadata.validate_full_hit_query_lengths()

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
) -> torch.Tensor:
    """Run exact full-hit MLA prefill from already projected query states."""
    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    if not metadata.full_hit_mode:
        raise RuntimeError("MLA full-hit replay requires full-hit mode")
    if metadata.full_seq_lengths is None:
        raise RuntimeError("MLA full-hit replay requires full sequence lengths")
    metadata.validate_full_hit_query_lengths()

    compressed_kv, cu_k, _ = (
        wrapper.prefix_attention_kv_builder().build_mla_full_hit_kv(
            metadata=metadata,
            kv_dim=spec.kv_dim,
            dtype=query_states.dtype,
            device=query_states.device,
        )
    )
    blocked_k, block_table, cache_seqlens = block_mla_kv_by_sequence(
        compressed_kv=compressed_kv,
        cu_k=cu_k,
        page_size=wrapper.host_prefix_reader().page_size(),
    )
    attn_out = run_flash_mla_prefix_attention(
        query_states=query_states,
        blocked_k=blocked_k,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        query_len=1,
        spec=spec,
    )
    return output_projection(attn_out)


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
    cu_k: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert packed per-sequence MLA KV into FlashMLA page blocks."""
    if compressed_kv.dim() != 3:
        raise RuntimeError(
            f"MLA compressed KV must be [tokens, 1, dim], got "
            f"{tuple(compressed_kv.shape)}"
        )
    page_size = int(page_size)
    cu_values = [int(value) for value in cu_k.detach().cpu().tolist()]
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
