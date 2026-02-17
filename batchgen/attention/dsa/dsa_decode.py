"""DSA decode path.

During decode, DSA performs:
1. Compute indexer K for the new token → update auxiliary cache
2. Score all cached tokens via indexer Q @ K^T → select top-K
3. Compute MLA compressed KV for the new token → update primary cache
4. Gather MLA KV at top-K positions from primary cache
5. Run MLA attention (FlashMLA) on the sparse subset

The MLA compression and FlashMLA kernel calls are reused from
batchgen/attention/mla/flashmla_backend.py. This module orchestrates
the indexer scoring and sparse gather steps.
"""

from __future__ import annotations

from typing import Tuple

import torch

from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv


def dsa_decode_score_and_gather(
	indexer,
	hidden_states: torch.Tensor,
	gpu_paged_kv_manager_aux,
	gpu_paged_kv_manager,
	cache_seqlens: torch.Tensor,
	layer_idx: int,
	page_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Run the DSA indexer scoring and sparse MLA KV gather for decode.

	This is called as part of the decode attention forward. It handles:
	1. Update auxiliary (indexer) cache with new token's indexer K
	2. Score all cached tokens → top-K selection
	3. Gather MLA KV at top-K positions from primary cache

	The caller is responsible for:
	- Computing and writing the MLA KV to the primary cache
	- Running FlashMLA on the sparse KV subset

	Args:
		indexer: LightningIndexer module for this layer
		hidden_states: [batch, 1, hidden_size] — current decode token
		gpu_paged_kv_manager_aux: Auxiliary (indexer) paged KV cache manager
		gpu_paged_kv_manager: Primary (MLA) paged KV cache manager
		cache_seqlens: [batch] — current cache lengths (before this token)
		layer_idx: Layer index
		page_size: Tokens per page

	Returns:
		sparse_mla_kv: [batch, topk, num_k_heads, k_head_dim] — gathered MLA KV
		top_k_indices: [batch, topk] — selected token positions
	"""
	# 1. Compute and store indexer K for new token
	indexer_kv = indexer.compute_indexer_kv(hidden_states)  # [batch, 1, 1, index_dim]
	gpu_paged_kv_manager_aux.update_layer_decode_new_token(
		k_tensor=indexer_kv.squeeze(1),  # [batch, 1, index_dim]
		v_tensor=None,
		sequence_lengths=cache_seqlens,
		layer_idx=layer_idx,
	)

	# 2. Score all cached tokens and select top-K
	indexer_blocked_k, _, idx_block_table = (
		gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(layer_idx)
	)
	# After writing the new token, cache_seqlens effectively becomes cache_seqlens + 1
	updated_seqlens = cache_seqlens + 1
	top_k_indices = indexer.score_and_select(
		hidden_states, indexer_blocked_k, idx_block_table,
		updated_seqlens, page_size,
	)

	# 3. Gather MLA KV at top-K positions from primary cache
	mla_blocked_k, _, mla_block_table = (
		gpu_paged_kv_manager.get_layer_kv_with_page_table(layer_idx)
	)
	sparse_mla_kv = sparse_gather_from_paged_kv(
		mla_blocked_k, mla_block_table, top_k_indices, page_size,
	)

	return sparse_mla_kv, top_k_indices
