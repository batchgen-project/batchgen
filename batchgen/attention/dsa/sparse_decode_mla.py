"""Sparse FlashMLA decode: run FlashMLA on DSA-selected token subset.

Packs sparse-gathered MLA KV into synthetic paged format so we can reuse
the standard `flash_mla_with_kvcache` kernel without modifications.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_mla import flash_mla_with_kvcache, get_mla_metadata


def sparse_flash_mla_decode(
    query_states: torch.Tensor,
    sparse_mla_kv: torch.Tensor,
    sparse_seqlens: torch.Tensor,
    num_heads: int,
    softmax_scale: float,
    head_dim_v: int = 512,
    page_size: int = 64,
) -> torch.Tensor:
    """Run FlashMLA on sparse-gathered KV packed into synthetic pages.

    Args:
        query_states: [batch, 1, num_heads, qk_head_dim] — absorbed Q (nope@absorb || rope)
        sparse_mla_kv: [batch, topk, 1, compressed_kv_dim] — gathered MLA KV
        sparse_seqlens: [batch] — actual number of valid tokens per sequence
            (min of topk and cache_seqlens+1)
        num_heads: number of attention heads
        softmax_scale: attention softmax scale
        head_dim_v: V head dimension for FlashMLA (kv_lora_rank, typically 512)
        page_size: FlashMLA page size (must match blocked_k layout)

    Returns:
        attn_out: [batch, 1, num_heads, kv_lora_rank] — raw FlashMLA output
            (caller applies out_absorb and o_proj)
    """
    batch, topk, _, kv_dim = sparse_mla_kv.shape
    num_pages_per_seq = (topk + page_size - 1) // page_size
    padded_topk = num_pages_per_seq * page_size

    # Pad to multiple of page_size
    if topk < padded_topk:
        pad = padded_topk - topk
        sparse_mla_kv = F.pad(sparse_mla_kv, (0, 0, 0, 0, 0, pad))

    # Pack as blocked KV: [total_pages, page_size, 1, kv_dim]
    total_pages = batch * num_pages_per_seq
    blocked_k = sparse_mla_kv.reshape(total_pages, page_size, 1, kv_dim)

    # Synthetic block table: sequential page indices per sequence
    block_table = torch.arange(
        total_pages, dtype=torch.int32, device=sparse_mla_kv.device
    ).view(batch, num_pages_per_seq)

    sparse_seqlens = sparse_seqlens.to(dtype=torch.int32, device=sparse_mla_kv.device)

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        sparse_seqlens, num_heads, 1
    )

    attn_out, _ = flash_mla_with_kvcache(
        query_states,
        blocked_k,
        block_table,
        sparse_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        False,  # causal=False: sparse-gathered KV is NOT sequential
    )

    return attn_out
