"""DSA prefill path.

During prefill, DSA uses full attention (no sparsity) — identical to standard
MLA prefill via FlashAttention3. The only addition is computing indexer K
values and returning them for auxiliary cache population.

The actual MLA prefill logic is reused from batchgen/attention/mla/fa3_backend.py.
This module provides the wrapper that additionally handles the indexer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def dsa_prefill_compute_indexer_kv(
	indexer,
	hidden_states: torch.Tensor,
) -> torch.Tensor:
	"""Compute indexer KV during prefill for auxiliary cache population.

	This is called alongside the standard MLA prefill to produce the
	indexer K values that will be stored in the auxiliary KV cache.

	Args:
		indexer: LightningIndexer module for this layer
		hidden_states: [batch, seq_len, hidden_size] — input to the layer

	Returns:
		indexer_kv: [batch, seq_len, 1, index_dim] — shaped for paged KV manager
	"""
	return indexer.compute_indexer_kv(hidden_states)
