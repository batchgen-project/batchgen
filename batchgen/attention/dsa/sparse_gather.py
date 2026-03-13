"""Sparse KV gather from paged cache.

Given top-K token indices from the Lightning Indexer, gathers the
corresponding MLA KV entries from the paged GPU cache.
"""

from __future__ import annotations

import torch

try:
	from batchgen_kernels.attention.dsa import fused_paged_gather as _fused_paged_gather
except (ImportError, Exception):
	_fused_paged_gather = None


def sparse_gather_from_paged_kv(
	blocked_k: torch.Tensor,
	block_table: torch.Tensor,
	token_indices: torch.Tensor,
	page_size: int,
) -> torch.Tensor:
	"""Gather KV entries at specific token positions from paged cache.

	Converts token positions to (page_index, offset) pairs, uses the block
	table to map logical pages to physical pages, then gathers from the
	blocked cache tensor.

	Args:
		blocked_k: [num_pages, page_size, num_k_heads, k_head_dim]
			The full paged KV cache.
		block_table: [batch, max_num_pages_per_seq]
			Maps (batch, logical_page) → physical_page_index.
		token_indices: [batch, topk]
			Absolute token positions to gather (from indexer top-K).
		page_size: Number of tokens per page.

	Returns:
		gathered_kv: [batch, topk, num_k_heads, k_head_dim]
			KV entries at the requested positions.
	"""
	batch_size, topk = token_indices.shape
	num_k_heads = blocked_k.shape[2]
	k_head_dim = blocked_k.shape[3]

	# Use fused Triton kernel if available
	if _fused_paged_gather is not None:
		return _fused_paged_gather(blocked_k, block_table, token_indices, page_size)

	# Convert absolute token positions to page index and offset
	logical_page_idx = token_indices // page_size  # [batch, topk]
	page_offset = token_indices % page_size  # [batch, topk]

	# Clamp logical page indices to valid block_table range
	max_pages = block_table.shape[1]
	logical_page_idx = logical_page_idx.clamp(max=max_pages - 1)

	# Map logical page to physical page via block_table
	physical_page_idx = torch.gather(
		block_table, 1, logical_page_idx.long()
	)  # [batch, topk]

	# Compute flat index into blocked_k reshaped as [num_pages * page_size, ...]
	flat_idx = physical_page_idx * page_size + page_offset  # [batch, topk]

	# Reshape blocked_k for flat indexing
	blocked_flat = blocked_k.reshape(-1, num_k_heads * k_head_dim)

	# Gather
	flat_idx_expanded = flat_idx.reshape(-1).long()  # [batch * topk]
	gathered_flat = blocked_flat[flat_idx_expanded]  # [batch * topk, num_k_heads * k_head_dim]
	gathered_kv = gathered_flat.view(batch_size, topk, num_k_heads, k_head_dim)

	return gathered_kv
