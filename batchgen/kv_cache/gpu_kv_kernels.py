from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 256

@triton.jit
def _paged_cache_update_kernel(
    cache_ptr,
    src_ptr,
    page_indices_ptr,
    token_offsets_ptr,
    layer_offset,
    page_stride,
    token_stride,
    elements_per_token,
    num_tokens,
    num_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    if token_id >= num_tokens or chunk_id >= num_chunks:
        return

    page = tl.load(page_indices_ptr + token_id)
    slot = tl.load(token_offsets_ptr + token_id)

    dest_base = (
        cache_ptr + layer_offset + page * page_stride + slot * token_stride
    )
    src_base = src_ptr + token_id * elements_per_token
    element_offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_offsets < elements_per_token

    values = tl.load(src_base + element_offsets, mask=mask, other=0)
    tl.store(dest_base + element_offsets, values, mask=mask)


def _launch_single_cache_update(
    cache_tensor: torch.Tensor,
    src_tokens: torch.Tensor,
    page_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    layer_idx: int,
) -> None:
    if cache_tensor is None:
        raise ValueError("cache_tensor must be defined")
    if cache_tensor.device != src_tokens.device:
        raise ValueError("cache_tensor and src_tokens must live on the same device")
    if cache_tensor.device != page_indices.device or cache_tensor.device != token_offsets.device:
        raise ValueError("page_indices/token_offsets must be on the same device as the cache tensor")
    if src_tokens.ndim != 2:
        raise ValueError("src_tokens must be flattened to [batch, elements]")
    if page_indices.ndim != 1 or token_offsets.ndim != 1:
        raise ValueError("page_indices and token_offsets must be 1-D tensors")
    if page_indices.shape[0] != src_tokens.shape[0] or token_offsets.shape[0] != src_tokens.shape[0]:
        raise ValueError("page_indices/token_offsets must match the batch size")

    num_tokens = src_tokens.shape[0]
    if num_tokens == 0:
        return

    if src_tokens.shape[1] <= 0:
        raise ValueError("Token vectors must contain at least one element")

    layer_stride, page_stride, token_stride = cache_tensor.stride()[:3]
    layer_offset = layer_idx * layer_stride
    elements_per_token = src_tokens.shape[1]
    num_chunks = (elements_per_token + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    grid = (num_tokens, num_chunks)

    _paged_cache_update_kernel[grid](
        cache_tensor,
        src_tokens,
        page_indices,
        token_offsets,
        layer_offset,
        page_stride,
        token_stride,
        elements_per_token,
        num_tokens,
        num_chunks,
        BLOCK_SIZE=_BLOCK_SIZE,
    )


def run_paged_kv_token_update(
    *,
    k_cache: torch.Tensor,
    k_tokens: torch.Tensor,
    page_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    layer_idx: int,
    v_cache: Optional[torch.Tensor] = None,
    v_tokens: Optional[torch.Tensor] = None,
) -> None:
    """Launches batched token copies into the paged KV cache.

    Args:
        k_cache: Destination K cache tensor on device.
        k_tokens: Flattened source K tokens shaped [batch, elements].
        page_indices: Per-token GPU page indices (int32 tensor).
        token_offsets: Per-token offsets inside each page (int32 tensor).
        layer_idx: Target transformer layer for all tokens.
        stream: CUDA stream used to enqueue the Triton kernels.
        v_cache: Optional V cache tensor on device.
        v_tokens: Optional flattened V tokens matching ``v_cache``.
    """
    if k_tokens.device != k_cache.device:
        raise ValueError("k_tokens must be placed on the same device as k_cache")
    if page_indices.dtype != torch.int32:
        raise ValueError("page_indices must be int32")
    if token_offsets.dtype != torch.int32:
        raise ValueError("token_offsets must be int32")

    _launch_single_cache_update(
        cache_tensor=k_cache,
        src_tokens=k_tokens,
        page_indices=page_indices,
        token_offsets=token_offsets,
        layer_idx=layer_idx,
    )

    if v_cache is None:
        if v_tokens is not None:
            raise ValueError("v_tokens provided but v_cache is None")
        return

    if v_tokens is None:
        raise ValueError("v_cache requires matching v_tokens input")
    if v_tokens.device != v_cache.device:
        raise ValueError("v_tokens must be on the same device as v_cache")

    _launch_single_cache_update(
        cache_tensor=v_cache,
        src_tokens=v_tokens,
        page_indices=page_indices,
        token_offsets=token_offsets,
        layer_idx=layer_idx,
    )
