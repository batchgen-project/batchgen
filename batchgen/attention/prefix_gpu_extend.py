"""GPU paged-KV extend helpers for prefix-aware prefill."""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch


def gpu_page_table_attention_enabled() -> bool:
    """Whether prefix prefill should attend directly from GPU paged KV."""

    return os.environ.get("BATCHGEN_PREFIX_REUSE_GPU_EXTEND_ATTENTION", "0") == "1"


def current_kv_cache_metadata():
    from batchgen.attention.forward_metadata_context import (
        get_current_forward_batch_metadata,
    )

    forward_metadata = get_current_forward_batch_metadata(required=True)
    kv_cache = forward_metadata.kv_cache
    if kv_cache is None:
        raise RuntimeError("Current ForwardBatchMetadata has no KV cache metadata")
    return kv_cache


def append_suffix_to_gpu_kv(
    *,
    kv_cache_metadata,
    k_tensor: torch.Tensor,
    v_tensor: Optional[torch.Tensor],
    layer_idx: int,
    metadata,
    manager_attr: str,
    context: str,
) -> object:
    """Append packed suffix K/V into a GPU paged-KV manager and return the plan."""

    manager = kv_manager_from_metadata(
        kv_cache_metadata=kv_cache_metadata,
        manager_attr=manager_attr,
        context=context,
    )
    append_plan = manager.prepare_prefill_suffix_append(
        sequence_ids=metadata.global_sequence_ids,
        prefix_lens=_prefix_lens_for_metadata(metadata),
        suffix_lens=metadata.seq_lengths,
    )
    manager.append_layer_prefill_suffix_tokens(
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        append_plan=append_plan,
        layer_idx=int(layer_idx),
    )
    return append_plan


def gqa_prefill_with_gpu_paged_kv(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata,
    kv_cache_metadata,
    layer_idx: Optional[int],
    paged_attention_fn: Optional[Callable[..., tuple[torch.Tensor, object]]],
    sinks: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    sliding_window: Optional[int],
) -> torch.Tensor:
    """Run single-sequence GQA suffix prefill against GPU paged KV."""

    if metadata.full_hit_mode:
        raise RuntimeError(
            "GQA GPU page-table prefill does not support full-hit batches yet"
        )
    _require_single_sequence_suffix(metadata, "GQA GPU page-table prefill")
    if layer_idx is None:
        raise RuntimeError("GQA GPU page-table prefill requires layer_idx")
    if kv_cache_metadata is None:
        kv_cache_metadata = current_kv_cache_metadata()

    manager = kv_manager_from_metadata(
        kv_cache_metadata=kv_cache_metadata,
        manager_attr="gpu_paged_kv_manager",
        context="GQA GPU page-table prefill",
    )
    append_plan = append_suffix_to_gpu_kv(
        kv_cache_metadata=kv_cache_metadata,
        k_tensor=key,
        v_tensor=value,
        layer_idx=int(layer_idx),
        metadata=metadata,
        manager_attr="gpu_paged_kv_manager",
        context="GQA GPU page-table prefill",
    )
    k_cache, v_cache, _ = manager.get_layer_kv_with_page_table(int(layer_idx))
    if v_cache is None:
        raise RuntimeError("GQA GPU page-table prefill requires V cache")

    q_len = int(metadata.seq_lengths[0])
    paged_q = query.contiguous().view(1, q_len, query.shape[1], query.shape[2])
    if paged_attention_fn is None:
        from batchgen.attention.gqa.fa_decode import gqa_decode_fa

        paged_attention_fn = gqa_decode_fa
    attn_output, _ = paged_attention_fn(
        q=paged_q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=append_plan.cache_seqlens,
        block_table=append_plan.page_table,
        sinks=sinks,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )
    return attn_output.view(q_len, query.shape[1], query.shape[2])


def mla_prefill_with_gpu_paged_kv(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    metadata,
    kv_cache_metadata,
    layer_idx: Optional[int],
    kv_dim: int,
    num_heads: int,
    kv_lora_rank: int,
    softmax_scale: float,
    attention_fn: Optional[Callable[..., torch.Tensor]],
) -> torch.Tensor:
    """Run single-sequence MLA suffix prefill against GPU paged KV."""

    if metadata.full_hit_mode:
        raise RuntimeError(
            "MLA GPU page-table prefill does not support full-hit batches yet"
        )
    _require_single_sequence_suffix(metadata, "MLA GPU page-table prefill")
    if layer_idx is None:
        raise RuntimeError("MLA GPU page-table prefill requires layer_idx")
    if kv_cache_metadata is None:
        kv_cache_metadata = current_kv_cache_metadata()

    manager = kv_manager_from_metadata(
        kv_cache_metadata=kv_cache_metadata,
        manager_attr="gpu_paged_kv_manager",
        context="MLA GPU page-table prefill",
    )
    append_plan = append_suffix_to_gpu_kv(
        kv_cache_metadata=kv_cache_metadata,
        k_tensor=key,
        v_tensor=None,
        layer_idx=int(layer_idx),
        metadata=metadata,
        manager_attr="gpu_paged_kv_manager",
        context="MLA GPU page-table prefill",
    )
    blocked_k, _, _ = manager.get_layer_kv_with_page_table(int(layer_idx))

    from batchgen.models.wrappers.prefix_mla_replay import MlaReplaySpec

    spec = MlaReplaySpec(
        kv_dim=int(kv_dim),
        num_heads=int(num_heads),
        kv_lora_rank=int(kv_lora_rank),
        softmax_scale=float(softmax_scale),
    )
    if attention_fn is None:
        from batchgen.models.wrappers.prefix_mla_replay import (
            run_flash_mla_prefix_attention,
        )

        attention_fn = run_flash_mla_prefix_attention
    return attention_fn(
        query_states=query,
        blocked_k=blocked_k,
        block_table=append_plan.page_table,
        cache_seqlens=append_plan.cache_seqlens,
        query_len=int(metadata.max_seqlen),
        spec=spec,
    )


def kv_manager_from_metadata(
    *,
    kv_cache_metadata,
    manager_attr: str,
    context: str,
):
    manager = getattr(kv_cache_metadata, manager_attr, None)
    if manager is None:
        raise RuntimeError(f"{context} requires kv_cache_metadata.{manager_attr}")
    return manager


def _prefix_lens_for_metadata(metadata) -> list[int]:
    if metadata.prefix_shared_tokens is None:
        return [0] * int(metadata.num_sequences)
    return [int(tokens) for tokens in metadata.prefix_shared_tokens]


def _require_single_sequence_suffix(metadata, context: str) -> None:
    if metadata.num_sequences != 1:
        raise RuntimeError(
            f"{context} currently requires single-sequence suffix micro-batches"
        )
