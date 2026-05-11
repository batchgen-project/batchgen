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

    dest_base = cache_ptr + page * page_stride + slot * token_stride
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
) -> None:
    if cache_tensor is None:
        raise ValueError("cache_tensor must be defined")
    if cache_tensor.device != src_tokens.device:
        raise ValueError(
            "cache_tensor and src_tokens must live on the same device"
        )
    if (
        cache_tensor.device != page_indices.device
        or cache_tensor.device != token_offsets.device
    ):
        raise ValueError(
            "page_indices/token_offsets must be on the same device as the cache tensor"
        )
    if src_tokens.ndim != 2:
        raise ValueError("src_tokens must be flattened to [batch, elements]")
    if page_indices.ndim != 1 or token_offsets.ndim != 1:
        raise ValueError("page_indices and token_offsets must be 1-D tensors")
    if (
        page_indices.shape[0] != src_tokens.shape[0]
        or token_offsets.shape[0] != src_tokens.shape[0]
    ):
        raise ValueError("page_indices/token_offsets must match the batch size")

    num_tokens = src_tokens.shape[0]
    if num_tokens == 0:
        return

    if src_tokens.shape[1] <= 0:
        raise ValueError("Token vectors must contain at least one element")

    strides = cache_tensor.stride()
    if len(strides) < 2:
        raise ValueError("cache_tensor must have at least two dimensions")
    page_stride = strides[0]
    token_stride = strides[1]
    elements_per_token = src_tokens.shape[1]
    num_chunks = (elements_per_token + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    grid = (num_tokens, num_chunks)

    _paged_cache_update_kernel[grid](
        cache_tensor,
        src_tokens,
        page_indices,
        token_offsets,
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
    v_cache: Optional[torch.Tensor] = None,
    v_tokens: Optional[torch.Tensor] = None,
) -> None:
    """Launches batched token copies into the paged KV cache.

    Args:
        k_cache: Destination K cache tensor on device.
        k_tokens: Flattened source K tokens shaped [batch, elements].
        page_indices: Per-token GPU page indices (int32 tensor).
        token_offsets: Per-token offsets inside each page (int32 tensor).
        stream: CUDA stream used to enqueue the Triton kernels.
        v_cache: Optional V cache tensor on device.
        v_tokens: Optional flattened V tokens matching ``v_cache``.
    """
    if k_tokens.device != k_cache.device:
        raise ValueError(
            "k_tokens must be placed on the same device as k_cache"
        )
    if page_indices.dtype != torch.int32:
        raise ValueError("page_indices must be int32")
    if token_offsets.dtype != torch.int32:
        raise ValueError("token_offsets must be int32")

    _launch_single_cache_update(
        cache_tensor=k_cache,
        src_tokens=k_tokens,
        page_indices=page_indices,
        token_offsets=token_offsets,
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
    )


@triton.jit
def _paged_cache_update_with_page_table_kernel(
    cache_ptr,
    src_ptr,
    page_table_ptr,
    slot_indices_ptr,
    token_indices_ptr,
    num_valid_tokens_ptr,
    page_stride,
    token_stride,
    page_table_cols,
    page_size_tokens,
    elements_per_token,
    num_tokens,
    num_chunks,
    HAS_VALID_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    if token_id >= num_tokens or chunk_id >= num_chunks:
        return
    if HAS_VALID_TOKENS:
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        if token_id >= num_valid_tokens:
            return

    slot = tl.load(slot_indices_ptr + token_id)
    if slot < 0:  # sentinel: skip padding tokens (CUDA graph bucketing)
        return
    token_index = tl.load(token_indices_ptr + token_id)

    page_slot = token_index // page_size_tokens
    offset = token_index - page_slot * page_size_tokens

    # page_table is 2-D flattened row-major: [num_slots, page_table_cols]
    page = tl.load(page_table_ptr + slot * page_table_cols + page_slot)

    dest_base = cache_ptr + page * page_stride + offset * token_stride
    src_base = src_ptr + token_id * elements_per_token
    element_offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_offsets < elements_per_token

    values = tl.load(src_base + element_offsets, mask=mask, other=0)
    tl.store(dest_base + element_offsets, values, mask=mask)


def _launch_single_cache_update_with_page_table(
    cache_tensor: torch.Tensor,
    src_tokens: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size_tokens: int,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> None:
    num_tokens = src_tokens.shape[0]
    if num_tokens == 0:
        return

    if src_tokens.shape[1] <= 0:
        raise ValueError("Token vectors must contain at least one element")

    strides = cache_tensor.stride()
    if len(strides) < 2:
        raise ValueError("cache_tensor must have at least two dimensions")
    page_stride = strides[0]
    token_stride = strides[1]
    elements_per_token = src_tokens.shape[1]
    num_chunks = (elements_per_token + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    page_table_cols = page_table.shape[1]
    if num_valid_tokens is not None:
        if num_valid_tokens.device != src_tokens.device:
            raise ValueError("num_valid_tokens must be on the same device as src_tokens")
        if num_valid_tokens.dtype != torch.int32:
            raise ValueError("num_valid_tokens must be int32")
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                f"num_valid_tokens must contain one element, got {tuple(num_valid_tokens.shape)}"
            )

    grid = (num_tokens, num_chunks)

    _paged_cache_update_with_page_table_kernel[grid](
        cache_tensor,
        src_tokens,
        page_table,
        slot_indices,
        token_indices,
        num_valid_tokens if num_valid_tokens is not None else slot_indices,
        page_stride,
        token_stride,
        page_table_cols,
        page_size_tokens,
        elements_per_token,
        num_tokens,
        num_chunks,
        HAS_VALID_TOKENS=num_valid_tokens is not None,
        BLOCK_SIZE=_BLOCK_SIZE,
    )


def run_paged_kv_token_update_fused(
    *,
    k_cache: torch.Tensor,
    k_tokens: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size_tokens: int,
    v_cache: Optional[torch.Tensor] = None,
    v_tokens: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> None:
    """Fused variant that looks up page numbers from a GPU-side page table.

    Args:
        page_table: int32 tensor on device with shape [num_slots, max_pages_per_sequence]
        slot_indices: int32 tensor [batch] mapping each token to a slot row
        token_indices: int32 tensor [batch] giving the absolute token index in sequence
    """

    _launch_single_cache_update_with_page_table(
        cache_tensor=k_cache,
        src_tokens=k_tokens,
        page_table=page_table,
        slot_indices=slot_indices,
        token_indices=token_indices,
        page_size_tokens=page_size_tokens,
        num_valid_tokens=num_valid_tokens,
    )

    if v_cache is None:
        if v_tokens is not None:
            raise ValueError("v_tokens provided but v_cache is None")
        return

    if v_tokens is None:
        raise ValueError("v_cache requires matching v_tokens input")
    if v_tokens.device != v_cache.device:
        raise ValueError("v_tokens must be on the same device as v_cache")

    _launch_single_cache_update_with_page_table(
        cache_tensor=v_cache,
        src_tokens=v_tokens,
        page_table=page_table,
        slot_indices=slot_indices,
        token_indices=token_indices,
        page_size_tokens=page_size_tokens,
        num_valid_tokens=num_valid_tokens,
    )
