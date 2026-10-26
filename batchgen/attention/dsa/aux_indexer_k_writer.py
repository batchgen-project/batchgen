"""The one GPU writer of the DSA indexer (aux) paged K cache.

GLM-5, GLM-5.1 and GLM-5.3 store indexer keys as bf16 token-major records;
GLM-5.2 stores them in DeepGEMM's page-split FP8 layout (per page: the e4m3
K bytes, then one fp32 scale per token). The same decode code serves all of
them, so the format is read off the CACHE the KV manager allocated rather than
re-derived from a model name, and anything that is neither layout fails loud.

The format also decides the Hadamard incoherence rotation
(``aux_cache_applies_hadamard``): the page-split FP8 route follows upstream
V3.2 and drops it on BOTH the Q index vector and every index-K producer, while
the bf16 AoS route keeps it. Rotating only one side would not preserve the
indexer logit, so the flag is read off the cache rather than passed per site.

Both the eager decode selector and the captured DSA graph segment call
``write_aux_indexer_k``, so the two paths cannot drift apart: a page-split
cache gets one fused quantize+write launch that also produces the
``(k_fp8, scales)`` pair the host KV region's append path requires, while an
AoS cache keeps the plain bf16 paged token update and no scales.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from batchgen_kernels.attention.dsa.indexer_k_quant import (
    cache_applies_hadamard as aux_cache_applies_hadamard,
    cache_data_dim as aux_cache_data_dim,
    cache_is_page_split as aux_cache_is_page_split,
)
from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused

__all__ = [
    "aux_cache_applies_hadamard",
    "aux_manager_applies_hadamard",
    "aux_cache_is_page_split",
    "aux_cache_data_dim",
    "write_aux_indexer_k",
    "write_aux_indexer_k_norm_rope",
]


def aux_manager_applies_hadamard(manager, data_dim: int) -> bool:
    """``aux_cache_applies_hadamard`` resolved from the aux KV manager config.

    Some call sites must know the rotation BEFORE they hold a layer cache: the
    eager selector ropes and rotates indexer K before it resolves the page
    table, and its graph/eager compare prologue runs with no page table at all.
    ``config.kv_dtype`` and ``config.k_head_dim`` are the two facts that
    allocated the cache, so this builds a zero-element meta probe with exactly
    that element type and carrier and asks the SAME discriminator. No second
    rule: a mis-sized configuration still fails loud here.
    """

    config = manager.config
    probe = torch.empty(
        (0, 0, 0, int(config.k_head_dim)),
        dtype=config.kv_dtype,
        device="meta",
    )
    return aux_cache_applies_hadamard(probe, data_dim)


def write_aux_indexer_k(
    *,
    aux_blocked_k: torch.Tensor,
    indexer_k: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size_tokens: int,
    k_fp8_out: Optional[torch.Tensor] = None,
    scales_out: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Write this step's indexer keys into the aux paged cache.

    Args:
        aux_blocked_k: ``[num_pages, page_size, 1, carrier]`` aux cache of the
            layer being written.
        indexer_k: ``[N, 1, 1, data_dim]`` bf16 keys of this step.
        page_table / slot_indices / token_indices / page_size_tokens: the GPU
            page-table lookup, identical to the primary cache's writer.
        k_fp8_out / scales_out: page-split only, pre-allocated host-append
            sources; a captured graph passes its static buffers so nothing is
            allocated during replay.
        num_valid_tokens: optional ``[1]`` int32 device tensor bounding the
            rows that may reach the cache.

    Returns:
        ``(k_for_host, scales_for_host)``: the tensors the aux host-append
        path must receive. ``scales_for_host`` is None for an AoS cache (the
        host region rejects scales) and the fp32 per-token scales for a
        page-split cache (the host region requires them).
    """

    data_dim = int(indexer_k.shape[-1])
    if aux_cache_is_page_split(aux_blocked_k, data_dim):
        # Imported lazily: the indexer package JIT-loads its CUDA extensions at
        # import time, which only a CUDA host can do.
        from batchgen_kernels.attention.dsa.indexer.indexer_k_write_fp8 import (
            fused_indexer_k_write_fp8,
        )

        return fused_indexer_k_write_fp8(
            aux_cache=aux_blocked_k,
            k_src=indexer_k,
            page_table=page_table,
            slot_indices=slot_indices,
            token_indices=token_indices,
            page_size_tokens=int(page_size_tokens),
            k_fp8_out=k_fp8_out,
            scales_out=scales_out,
            num_valid_tokens=num_valid_tokens,
        )

    if k_fp8_out is not None or scales_out is not None:
        raise ValueError(
            "host-append FP8 buffers were passed for an AoS aux indexer "
            "cache; the host region rejects scales in that format"
        )
    run_paged_kv_token_update_fused(
        k_cache=aux_blocked_k,
        k_tokens=indexer_k.reshape(indexer_k.shape[0], -1),
        page_table=page_table,
        slot_indices=slot_indices,
        token_indices=token_indices,
        page_size_tokens=int(page_size_tokens),
        num_valid_tokens=num_valid_tokens,
    )
    return indexer_k, None


def write_aux_indexer_k_norm_rope(
    *,
    aux_blocked_k: torch.Tensor,
    k_raw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    ln_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size_tokens: int,
    apply_hadamard: bool,
    k_fp8_out: torch.Tensor,
    scales_out: torch.Tensor,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize, rope, rotate, quantize and page-write this step's keys.

    The page-split-only counterpart of ``write_aux_indexer_k``: it takes the
    UNNORMALIZED ``wk`` output and owns the whole K tail in one launch
    (``fused_indexer_k_norm_rope_write_fp8``), instead of the ATen LayerNorm +
    fused RoPE/Hadamard + fused quantize/write triple.

    A bf16 AoS cache has no fused path -- its records are not quantized -- so
    an AoS cache fails loud here rather than silently reverting to the triple.

    Returns the ``(k_fp8, scales)`` pair the aux host-append path requires.
    """

    data_dim = int(k_raw.shape[-1])
    if not aux_cache_is_page_split(aux_blocked_k, data_dim):
        raise ValueError(
            "write_aux_indexer_k_norm_rope needs a page-split FP8 aux indexer "
            "cache; a bf16 AoS cache stores unquantized records and must use "
            "write_aux_indexer_k with the separate LayerNorm and RoPE passes"
        )

    # Imported lazily: the indexer package JIT-loads its CUDA extensions at
    # import time, which only a CUDA host can do.
    from batchgen_kernels.attention.dsa.indexer.indexer_k_norm_rope_write_fp8 import (
        fused_indexer_k_norm_rope_write_fp8,
    )

    return fused_indexer_k_norm_rope_write_fp8(
        aux_cache=aux_blocked_k,
        k_raw=k_raw,
        ln_weight=ln_weight,
        ln_bias=ln_bias,
        ln_eps=float(ln_eps),
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        positions=positions,
        page_table=page_table,
        slot_indices=slot_indices,
        token_indices=token_indices,
        page_size_tokens=int(page_size_tokens),
        apply_hadamard=bool(apply_hadamard),
        k_fp8_out=k_fp8_out,
        scales_out=scales_out,
        num_valid_tokens=num_valid_tokens,
    )
