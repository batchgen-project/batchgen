"""The one GPU writer of the DSA indexer (aux) paged K cache.

GLM-5, GLM-5.1 and GLM-5.3 store indexer keys as bf16 token-major records;
GLM-5.2 stores them in DeepGEMM's page-split FP8 layout (per page: the e4m3
K bytes, then one fp32 scale per token). The same decode code serves all of
them, so the format is read off the CACHE the KV manager allocated rather than
re-derived from a model name, and anything that is neither layout fails loud.

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
    cache_data_dim as aux_cache_data_dim,
    cache_is_page_split as aux_cache_is_page_split,
)
from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused

__all__ = [
    "aux_cache_is_page_split",
    "aux_cache_data_dim",
    "write_aux_indexer_k",
]


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
