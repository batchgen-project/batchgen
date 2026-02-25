from __future__ import annotations

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 256

# ---------------------------------------------------------------------------
# CUDA kernel for paged KV cache update (lower launch overhead than Triton)
# ---------------------------------------------------------------------------
_cuda_kv_module = None
_cuda_kv_available = None  # None = not checked yet


def _get_cuda_kv_module():
    global _cuda_kv_module, _cuda_kv_available
    if _cuda_kv_available is not None:
        return _cuda_kv_module
    try:
        from torch.utils.cpp_extension import load_inline
        _CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void paged_kv_token_update_kernel(
    scalar_t* __restrict__ cache,
    const scalar_t* __restrict__ src,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ slot_indices,
    const int32_t* __restrict__ token_indices,
    const int64_t page_stride,
    const int64_t token_stride,
    const int32_t page_table_cols,
    const int32_t page_size_tokens,
    const int32_t elements_per_token,
    const int32_t num_tokens
) {
    const int token_id = blockIdx.x;
    if (token_id >= num_tokens) return;

    const int32_t slot = __ldg(&slot_indices[token_id]);
    if (slot < 0) return;

    const int32_t token_index = __ldg(&token_indices[token_id]);
    const int32_t page_slot = token_index / page_size_tokens;
    const int32_t offset = token_index - page_slot * page_size_tokens;

    const int32_t page = __ldg(&page_table[slot * page_table_cols + page_slot]);
    if (page < 0) return;

    scalar_t* dst = cache + page * page_stride + offset * token_stride;
    const scalar_t* src_row = src + (int64_t)token_id * elements_per_token;

    for (int e = threadIdx.x; e < elements_per_token; e += blockDim.x) {
        dst[e] = src_row[e];
    }
}

void paged_kv_token_update_cuda(
    torch::Tensor cache,
    torch::Tensor src_tokens,
    torch::Tensor page_table,
    torch::Tensor slot_indices,
    torch::Tensor token_indices,
    int64_t page_size_tokens
) {
    const int num_tokens = src_tokens.size(0);
    if (num_tokens == 0) return;

    const int elements_per_token = src_tokens.size(1);
    const int64_t page_stride = cache.stride(0);
    const int64_t token_stride = cache.stride(1);
    const int page_table_cols = page_table.size(1);

    const int threads = 128;
    const int blocks = num_tokens;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        cache.scalar_type(), "paged_kv_token_update", [&] {
            paged_kv_token_update_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                cache.data_ptr<scalar_t>(),
                src_tokens.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(),
                slot_indices.data_ptr<int32_t>(),
                token_indices.data_ptr<int32_t>(),
                page_stride,
                token_stride,
                page_table_cols,
                static_cast<int32_t>(page_size_tokens),
                elements_per_token,
                num_tokens
            );
        }
    );
}
"""
        _CPP_SRC = r"""
void paged_kv_token_update_cuda(
    torch::Tensor cache, torch::Tensor src_tokens,
    torch::Tensor page_table, torch::Tensor slot_indices,
    torch::Tensor token_indices, int64_t page_size_tokens);
"""
        _cuda_kv_module = load_inline(
            name="paged_kv_write_cuda",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["paged_kv_token_update_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        _cuda_kv_available = True
        logging.info("CUDA paged KV write kernel compiled successfully")
    except Exception as e:
        _cuda_kv_available = False
        logging.warning(f"CUDA paged KV write kernel unavailable, using Triton: {e}")
    return _cuda_kv_module


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
    page_stride,
    token_stride,
    page_table_cols,
    page_size_tokens,
    elements_per_token,
    num_tokens,
    num_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    if token_id >= num_tokens or chunk_id >= num_chunks:
        return

    slot = tl.load(slot_indices_ptr + token_id)
    if slot < 0:  # sentinel: skip padding tokens (CUDA graph bucketing)
        return
    token_index = tl.load(token_indices_ptr + token_id)

    page_slot = token_index // page_size_tokens
    offset = token_index - page_slot * page_size_tokens

    # page_table is 2-D flattened row-major: [num_slots, page_table_cols]
    page = tl.load(page_table_ptr + slot * page_table_cols + page_slot)
    if page < 0:  # sentinel -1: unallocated page slot
        return

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
) -> None:
    num_tokens = src_tokens.shape[0]
    if num_tokens == 0:
        return

    # Use CUDA kernel if available (lower launch overhead)
    mod = _get_cuda_kv_module()
    if mod is not None:
        mod.paged_kv_token_update_cuda(
            cache_tensor, src_tokens, page_table,
            slot_indices, token_indices, page_size_tokens,
        )
        return

    # Fallback to Triton
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

    grid = (num_tokens, num_chunks)

    _paged_cache_update_with_page_table_kernel[grid](
        cache_tensor,
        src_tokens,
        page_table,
        slot_indices,
        token_indices,
        page_stride,
        token_stride,
        page_table_cols,
        page_size_tokens,
        elements_per_token,
        num_tokens,
        num_chunks,
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
    )
