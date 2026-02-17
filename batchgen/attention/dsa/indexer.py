"""Lightning Indexer for Dynamic Sparse Attention (DSA).

The indexer is a lightweight per-layer module that scores all cached tokens
and selects the top-K most relevant positions for full MLA attention.

Architecture (per layer):
  - q_proj: [hidden_size, index_n_heads * index_head_dim]
  - k_proj: [hidden_size, index_n_heads * index_head_dim]
  - Scoring: Q_idx @ K_idx^T → softmax → top-K selection

The indexer maintains its own KV cache (auxiliary) separate from the MLA cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IndexerConfig:
	"""Configuration for the Lightning Indexer."""

	hidden_size: int = 7168
	index_n_heads: int = 64
	index_head_dim: int = 128
	index_topk: int = 2048
	rope_interleave: bool = False


class LightningIndexer(nn.Module):
	"""Lightning Indexer for token scoring and top-K selection.

	During prefill: computes indexer K for all tokens (stored in auxiliary cache).
	During decode: computes indexer Q for the new token, scores against all cached
	indexer K entries, and returns top-K token indices.
	"""

	def __init__(self, config: IndexerConfig, layer_idx: int = 0) -> None:
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx

		self.index_n_heads = config.index_n_heads
		self.index_head_dim = config.index_head_dim
		self.index_topk = config.index_topk
		self.index_dim = config.index_n_heads * config.index_head_dim

		self.q_proj = nn.Linear(config.hidden_size, self.index_dim, bias=False)
		self.k_proj = nn.Linear(config.hidden_size, self.index_dim, bias=False)

	def compute_indexer_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
		"""Compute indexer K for cache storage.

		Called during both prefill and decode to populate the auxiliary KV cache.

		Args:
			hidden_states: [batch, seq_len, hidden_size]

		Returns:
			indexer_k: [batch, seq_len, 1, index_dim] — shaped for paged KV manager
				(num_k_heads=1, k_head_dim=index_dim)
		"""
		k = self.k_proj(hidden_states)  # [batch, seq_len, index_dim]
		return k.unsqueeze(2)  # [batch, seq_len, 1, index_dim]

	def score_and_select(
		self,
		hidden_states: torch.Tensor,
		indexer_blocked_k: torch.Tensor,
		block_table: torch.Tensor,
		cache_seqlens: torch.Tensor,
		page_size: int = 64,
	) -> torch.Tensor:
		"""Score all cached tokens and return top-K indices.

		Args:
			hidden_states: [batch, 1, hidden_size] — current decode token
			indexer_blocked_k: [num_pages, page_size, 1, index_dim] — paged indexer cache
			block_table: [batch, max_num_pages_per_seq] — page mapping
			cache_seqlens: [batch] — number of valid tokens per sequence
			page_size: tokens per page

		Returns:
			top_k_indices: [batch, index_topk] — absolute token positions of top-K
		"""
		batch_size = hidden_states.shape[0]
		device = hidden_states.device

		# Compute query
		q = self.q_proj(hidden_states)  # [batch, 1, index_dim]
		q = q.view(batch_size, self.index_n_heads, self.index_head_dim)

		# Gather all valid K entries from paged cache into a contiguous tensor
		# Shape: [batch, max_seq_len, n_heads, head_dim]
		max_seqlen = int(cache_seqlens.max().item())
		gathered_k = _gather_all_from_paged_cache(
			indexer_blocked_k, block_table, cache_seqlens, page_size, max_seqlen
		)
		# gathered_k: [batch, max_seqlen, 1, index_dim]
		# Reshape to [batch, max_seqlen, n_heads, head_dim]
		gathered_k = gathered_k.squeeze(2).view(
			batch_size, max_seqlen, self.index_n_heads, self.index_head_dim
		)

		# Compute attention scores: [batch, n_heads, 1, max_seqlen]
		# q: [batch, n_heads, head_dim] → [batch, n_heads, 1, head_dim]
		q = q.unsqueeze(2)
		# gathered_k: [batch, max_seqlen, n_heads, head_dim] → [batch, n_heads, max_seqlen, head_dim]
		gathered_k = gathered_k.permute(0, 2, 1, 3)
		scores = torch.matmul(q, gathered_k.transpose(-2, -1))  # [batch, n_heads, 1, max_seqlen]
		scores = scores.squeeze(2)  # [batch, n_heads, max_seqlen]

		# Mask out positions beyond actual sequence length
		position_indices = torch.arange(max_seqlen, device=device).unsqueeze(0)  # [1, max_seqlen]
		mask = position_indices >= cache_seqlens.unsqueeze(1)  # [batch, max_seqlen]
		scores.masked_fill_(mask.unsqueeze(1), float("-inf"))

		# Aggregate across heads (sum scores)
		aggregated_scores = scores.sum(dim=1)  # [batch, max_seqlen]

		# Top-K selection
		effective_topk = min(self.index_topk, max_seqlen)
		_, top_k_indices = torch.topk(aggregated_scores, effective_topk, dim=-1)  # [batch, topk]

		return top_k_indices


def _gather_all_from_paged_cache(
	blocked_k: torch.Tensor,
	block_table: torch.Tensor,
	cache_seqlens: torch.Tensor,
	page_size: int,
	max_seqlen: int,
) -> torch.Tensor:
	"""Gather all valid tokens from paged cache into a contiguous tensor.

	Args:
		blocked_k: [num_pages, page_size, num_k_heads, k_head_dim]
		block_table: [batch, max_num_pages_per_seq]
		cache_seqlens: [batch]
		page_size: tokens per page
		max_seqlen: maximum sequence length to gather

	Returns:
		gathered: [batch, max_seqlen, num_k_heads, k_head_dim]
	"""
	batch_size = block_table.shape[0]
	num_k_heads = blocked_k.shape[2]
	k_head_dim = blocked_k.shape[3]
	device = blocked_k.device

	# Pre-allocate output
	gathered = torch.zeros(
		batch_size, max_seqlen, num_k_heads, k_head_dim,
		dtype=blocked_k.dtype, device=device,
	)

	# For each token position, compute which page and offset it maps to
	token_positions = torch.arange(max_seqlen, device=device)  # [max_seqlen]
	page_indices = token_positions // page_size  # [max_seqlen]
	page_offsets = token_positions % page_size  # [max_seqlen]

	# Expand for batch: [batch, max_seqlen]
	page_indices = page_indices.unsqueeze(0).expand(batch_size, -1)

	# Clamp page indices to valid range for gather (will be masked later)
	max_pages = block_table.shape[1]
	page_indices_clamped = page_indices.clamp(max=max_pages - 1)

	# Map logical page index to physical page via block_table
	physical_pages = torch.gather(block_table, 1, page_indices_clamped)  # [batch, max_seqlen]

	# Flatten to index into blocked_k
	# blocked_k is [num_pages, page_size, num_k_heads, k_head_dim]
	# We need: blocked_k[physical_pages[b, t], page_offsets[t], :, :]
	flat_idx = physical_pages * page_size + page_offsets.unsqueeze(0)  # [batch, max_seqlen]

	# Reshape blocked_k to [num_pages * page_size, num_k_heads * k_head_dim]
	blocked_flat = blocked_k.reshape(-1, num_k_heads * k_head_dim)

	# Gather
	flat_idx_expanded = flat_idx.reshape(-1).long()  # [batch * max_seqlen]
	gathered_flat = blocked_flat[flat_idx_expanded]  # [batch * max_seqlen, num_k_heads * k_head_dim]
	gathered = gathered_flat.view(batch_size, max_seqlen, num_k_heads, k_head_dim)

	return gathered
