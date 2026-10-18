"""The one producer of a primary MLA KV cache record, for every GLM-5 variant.

GLM-5, GLM-5.1 and GLM-5.3 store the compressed KV latent as bf16 token-major
records (576 elements); GLM-5.2 stores FlashMLA's packed FP8 record (656 bytes:
512 e4m3 NoPE, 4 fp32 block scales, 64 bf16 RoPE). The same decode and prefill
code serves all of them, so -- exactly as for the aux/indexer cache -- the
format is read off the CACHE the KV manager allocated rather than re-derived
from a model name, and anything that is neither layout fails loud.

Every writer funnels through ``prepare_primary_mla_kv``, which produces the
record bytes ONCE into the caller's destination buffer. That destination is
the same tensor the caller then (a) stores into the GPU paged cache with the
dtype-agnostic bare-copy kernel and (b) hands to the host append. One
production point is what makes "GPU cache row == staged host row == shm page"
an assertable invariant: there is no second quantization anywhere downstream,
so every later hop is a byte copy.

Callers keep their own paged-write and masking mechanics, which legitimately
differ (the captured graph writes through a GPU page table and selects invalid
rows to zero bytes; eager decode goes through the KV manager and relies on the
-1 slot sentinel). What they must NOT differ on is the bytes, and that is what
lives here.
"""

from __future__ import annotations

import torch

from batchgen_kernels.attention.mla_kv_packed import (
    SOURCE_DIM as PACKED_SOURCE_DIM,
    TOKEN_BYTES as PACKED_TOKEN_BYTES,
    cache_is_packed_fp8 as primary_cache_is_packed_fp8,
    dequantize_mla_kv_packed,
)

__all__ = [
    "PACKED_SOURCE_DIM",
    "PACKED_TOKEN_BYTES",
    "primary_cache_is_packed_fp8",
    "primary_record_dim",
    "primary_record_dtype",
    "dequantize_mla_kv_packed",
    "prepare_primary_mla_kv",
]


def primary_record_dim(primary_blocked_k: torch.Tensor) -> int:
    """Trailing size of ONE record of this cache: 656 bytes or 576 elements."""

    return int(primary_blocked_k.shape[-1])


def primary_record_dtype(primary_blocked_k: torch.Tensor) -> torch.dtype:
    """dtype of the record buffers a writer must allocate for this cache.

    Equal to the cache's own dtype by construction -- the paged store is a
    byte copy, and the dtype-equality guard on
    ``run_paged_kv_token_update_fused`` rejects anything else -- so deriving it
    here keeps the graph's static output spec, the whole-model staging buffer
    and the host append from each picking their own.
    """

    return primary_blocked_k.dtype


def prepare_primary_mla_kv(
    *,
    primary_blocked_k: torch.Tensor,
    compressed_kv: torch.Tensor,
    out: torch.Tensor,
    source_dim: int,
) -> torch.Tensor:
    """Produce this cache's record bytes for a batch of tokens, once.

    Args:
        primary_blocked_k: the allocated paged cache; the format discriminator.
        compressed_kv: bf16 compressed KV straight out of the fused
            RMSNorm+RoPE kernel, viewable as ``[tokens, source_dim]``.
        out: destination, ``[tokens, record_dim]`` of the cache's dtype. This
            is the tensor the caller stores AND stages, so the record exists
            exactly once.
        source_dim: ``kv_lora_rank + qk_rope_head_dim`` for this model.

    Returns:
        ``out``.
    """

    packed = primary_cache_is_packed_fp8(primary_blocked_k, source_dim)
    tokens = int(compressed_kv.shape[0]) if compressed_kv.ndim else 0
    flat_source = compressed_kv.view(tokens, source_dim)

    if not packed:
        # bf16 AoS: the record IS the latent. Keep the plain copy.
        out.view(tokens, source_dim).copy_(flat_source)
        return out

    # Imported lazily: Triton is absent on CPU-only machines, and a bf16 AoS
    # profile must stay importable there.
    from batchgen_kernels.triton.mla_kv_packed import quantize_mla_kv_packed

    quantize_mla_kv_packed(flat_source, out=out.view(tokens, PACKED_TOKEN_BYTES))
    return out
