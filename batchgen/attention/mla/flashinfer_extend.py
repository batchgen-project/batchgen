"""FlashInfer MLA paged-KV extend helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from flashinfer import BatchMLAPagedAttentionWrapper

_DEFAULT_WORKSPACE_BYTES = 384 * 1024 * 1024
_WORKSPACE_CACHE: dict[tuple[str, Optional[int]], torch.Tensor] = {}
_WRAPPER_CACHE: dict[tuple[str, Optional[int], str], object] = {}
_PLAN_CACHE_KEY = "flashinfer_mla_extend_prefill"


@dataclass
class _FlashInferMlaExtendPlanState:
    signature: tuple[object, ...]
    wrapper: object
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_len_arr: torch.Tensor


def run_flashinfer_mla_extend_prefill(
    *,
    query_states: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_lora_rank: int,
    num_heads: int,
    softmax_scale: float,
    plan_cache: dict[str, object] | None = None,
) -> torch.Tensor:
    """Run prefix-hit MLA extend prefill through FlashInfer paged attention.

    ``compressed_kv_cache`` is BatchGen's materialized GPU paged MLA cache with
    shape ``[num_pages, page_size, 1, kv_lora_rank + rope_dim]``. The returned
    tensor is packed as ``[1, tokens, heads, rank]`` so existing MLA
    output-projection glue can stay unchanged. Exact full hits are represented
    as one query token per sequence.
    """

    packed_query = _packed_query_view(query_states)
    q_nope = packed_query[..., :kv_lora_rank].contiguous()
    q_pe = packed_query[..., kv_lora_rank:].contiguous()
    ckv_cache, kpe_cache = _split_compressed_mla_cache(
        compressed_kv_cache,
        kv_lora_rank=kv_lora_rank,
    )
    device = packed_query.device
    page_size = int(compressed_kv_cache.shape[1])
    kv_len_arr = cache_seqlens.to(device=device, dtype=torch.int32)
    kv_indptr, kv_indices = _build_flashinfer_page_metadata(
        page_table=page_table,
        slot_indices=slot_indices,
        cache_seqlens=kv_len_arr,
        page_size=page_size,
    )
    qo_indptr = cu_seqlens_q.to(device=device, dtype=torch.int32)

    wrapper = _get_or_plan_flashinfer_mla_wrapper(
        device=device,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_arr,
        num_heads=int(num_heads),
        kv_lora_rank=int(kv_lora_rank),
        rope_head_dim=int(q_pe.shape[-1]),
        page_size=page_size,
        softmax_scale=float(softmax_scale),
        q_dtype=q_nope.dtype,
        kv_dtype=ckv_cache.dtype,
        plan_cache=plan_cache,
    )
    output = wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)
    return output.unsqueeze(0).contiguous()


def _packed_query_view(query_states: torch.Tensor) -> torch.Tensor:
    if query_states.dim() == 4 and query_states.shape[0] == 1:
        return query_states.squeeze(0)
    if query_states.dim() == 4 and query_states.shape[1] == 1:
        return query_states.reshape(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[3],
        )
    if query_states.dim() == 3:
        return query_states
    raise RuntimeError(
        "FlashInfer MLA extend prefill expects packed query states shaped "
        "[1, tokens, heads, dim], [batch, 1, heads, dim], or "
        "[tokens, heads, dim]"
    )


def _split_compressed_mla_cache(
    compressed_kv_cache: torch.Tensor,
    *,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if compressed_kv_cache.dim() != 4 or compressed_kv_cache.shape[2] != 1:
        raise RuntimeError(
            "FlashInfer MLA extend prefill expects K-only compressed MLA cache "
            "shaped [pages, page_size, 1, dim]"
        )
    cache = compressed_kv_cache.squeeze(2)
    return cache[..., :kv_lora_rank], cache[..., kv_lora_rank:]


def _build_flashinfer_page_metadata(
    *,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = cache_seqlens.device
    slot_indices = slot_indices.to(device=page_table.device, dtype=torch.long)
    selected_table = page_table.index_select(0, slot_indices).to(
        dtype=torch.int32
    )
    pages_per_sequence = torch.div(
        cache_seqlens + (int(page_size) - 1),
        int(page_size),
        rounding_mode="floor",
    )
    kv_indptr = torch.empty(
        pages_per_sequence.numel() + 1,
        dtype=torch.int32,
        device=device,
    )
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(pages_per_sequence, dim=0, dtype=torch.int32)

    page_offsets = torch.arange(
        selected_table.shape[1],
        dtype=torch.int32,
        device=selected_table.device,
    )
    valid_pages = page_offsets.unsqueeze(0) < pages_per_sequence.to(
        device=selected_table.device
    ).unsqueeze(1)
    kv_indices = selected_table[valid_pages].to(
        device=device, dtype=torch.int32
    )
    return kv_indptr, kv_indices.contiguous()


def _get_flashinfer_mla_wrapper(device: torch.device) -> object:
    backend = os.getenv("BATCHGEN_FLASHINFER_MLA_BACKEND", "auto")
    key = _cache_key(device) + (backend,)
    wrapper = _WRAPPER_CACHE.get(key)
    if wrapper is not None:
        return wrapper

    workspace = _get_workspace(device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend=backend)
    _WRAPPER_CACHE[key] = wrapper
    return wrapper


def _get_or_plan_flashinfer_mla_wrapper(
    *,
    device: torch.device,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr: torch.Tensor,
    num_heads: int,
    kv_lora_rank: int,
    rope_head_dim: int,
    page_size: int,
    softmax_scale: float,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    plan_cache: dict[str, object] | None,
) -> object:
    signature = (
        _cache_key(device),
        os.getenv("BATCHGEN_FLASHINFER_MLA_BACKEND", "auto"),
        tuple(qo_indptr.shape),
        tuple(kv_indptr.shape),
        tuple(kv_indices.shape),
        tuple(kv_len_arr.shape),
        int(num_heads),
        int(kv_lora_rank),
        int(rope_head_dim),
        int(page_size),
        float(softmax_scale),
        str(q_dtype),
        str(kv_dtype),
    )
    if plan_cache is not None:
        cached = plan_cache.get(_PLAN_CACHE_KEY)
        if (
            isinstance(cached, _FlashInferMlaExtendPlanState)
            and cached.signature == signature
        ):
            return cached.wrapper

    if plan_cache is None:
        wrapper = _get_flashinfer_mla_wrapper(device)
    else:
        wrapper = _new_flashinfer_mla_wrapper(device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        int(num_heads),
        int(kv_lora_rank),
        int(rope_head_dim),
        int(page_size),
        True,
        float(softmax_scale),
        q_dtype,
        kv_dtype,
    )
    if plan_cache is not None:
        plan_cache[_PLAN_CACHE_KEY] = _FlashInferMlaExtendPlanState(
            signature=signature,
            wrapper=wrapper,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
        )
    return wrapper


def _new_flashinfer_mla_wrapper(device: torch.device) -> object:
    backend = os.getenv("BATCHGEN_FLASHINFER_MLA_BACKEND", "auto")
    return BatchMLAPagedAttentionWrapper(_get_workspace(device), backend=backend)


def _get_workspace(device: torch.device) -> torch.Tensor:
    key = _cache_key(device)
    workspace = _WORKSPACE_CACHE.get(key)
    if workspace is None:
        workspace_bytes = int(
            os.getenv("BATCHGEN_FLASHINFER_WORKSPACE_BYTES", "0")
            or "0"
        )
        if workspace_bytes <= 0:
            workspace_mb = int(
                os.getenv("BATCHGEN_FLASHINFER_WORKSPACE_MB", "0")
                or "0"
            )
            workspace_bytes = (
                workspace_mb * 1024 * 1024
                if workspace_mb > 0
                else _DEFAULT_WORKSPACE_BYTES
            )
        workspace = torch.empty(
            workspace_bytes,
            dtype=torch.uint8,
            device=device,
        )
        _WORKSPACE_CACHE[key] = workspace
    return workspace


def _cache_key(device: torch.device) -> tuple[str, Optional[int]]:
    normalized = torch.device(device)
    return normalized.type, normalized.index


def _reset_flashinfer_mla_extend_prefill_cache_for_tests() -> None:
    _WORKSPACE_CACHE.clear()
    _WRAPPER_CACHE.clear()
