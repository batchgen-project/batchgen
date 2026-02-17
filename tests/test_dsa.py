"""Unit tests for Dynamic Sparse Attention (DSA) components.

Tests cover:
1. DualKVCacheCoordinator — lifecycle sync between two managers
2. LightningIndexer — shape correctness, top-K selection
3. sparse_gather_from_paged_kv — correctness against direct indexing
4. DSA end-to-end mock — shape flow through prefill + decode
"""

import pytest
import torch

from batchgen.attention.dsa.indexer import (
	IndexerConfig,
	LightningIndexer,
	_gather_all_from_paged_cache,
)
from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def indexer_config():
	return IndexerConfig(
		hidden_size=256,  # Small for testing
		index_n_heads=4,
		index_head_dim=32,
		index_topk=8,
	)


@pytest.fixture
def indexer(indexer_config):
	return LightningIndexer(indexer_config, layer_idx=0)


@pytest.fixture
def page_size():
	return 4  # Small page size for testing


# ---------------------------------------------------------------------------
# Test: LightningIndexer
# ---------------------------------------------------------------------------

class TestLightningIndexer:

	def test_init(self, indexer, indexer_config):
		assert indexer.index_dim == indexer_config.index_n_heads * indexer_config.index_head_dim
		assert indexer.q_proj.in_features == indexer_config.hidden_size
		assert indexer.q_proj.out_features == indexer.index_dim
		assert indexer.k_proj.in_features == indexer_config.hidden_size
		assert indexer.k_proj.out_features == indexer.index_dim

	def test_compute_indexer_kv_shape(self, indexer, indexer_config):
		batch, seq_len = 2, 16
		hidden = torch.randn(batch, seq_len, indexer_config.hidden_size)
		kv = indexer.compute_indexer_kv(hidden)
		assert kv.shape == (batch, seq_len, 1, indexer.index_dim)

	def test_compute_indexer_kv_single_token(self, indexer, indexer_config):
		"""Decode: single token."""
		batch = 3
		hidden = torch.randn(batch, 1, indexer_config.hidden_size)
		kv = indexer.compute_indexer_kv(hidden)
		assert kv.shape == (batch, 1, 1, indexer.index_dim)

	def test_score_and_select_shape(self, indexer, indexer_config, page_size):
		batch = 2
		seq_len = 32
		num_pages_per_seq = (seq_len + page_size - 1) // page_size
		total_pages = batch * num_pages_per_seq + 4  # Extra pages

		# Create mock paged cache
		blocked_k = torch.randn(total_pages, page_size, 1, indexer.index_dim)

		# Block table: maps logical pages to physical pages
		block_table = torch.zeros(batch, num_pages_per_seq, dtype=torch.int32)
		for b in range(batch):
			for p in range(num_pages_per_seq):
				block_table[b, p] = b * num_pages_per_seq + p

		cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32)
		hidden = torch.randn(batch, 1, indexer_config.hidden_size)

		top_k = indexer.score_and_select(
			hidden, blocked_k, block_table, cache_seqlens, page_size
		)

		expected_topk = min(indexer_config.index_topk, seq_len)
		assert top_k.shape == (batch, expected_topk)

	def test_score_and_select_topk_clamped(self, indexer_config, page_size):
		"""When seq_len < index_topk, returns seq_len indices."""
		config = IndexerConfig(
			hidden_size=indexer_config.hidden_size,
			index_n_heads=indexer_config.index_n_heads,
			index_head_dim=indexer_config.index_head_dim,
			index_topk=100,  # Larger than seq_len
		)
		idx = LightningIndexer(config)

		batch, seq_len = 1, 10
		num_pages = (seq_len + page_size - 1) // page_size
		blocked_k = torch.randn(num_pages, page_size, 1, idx.index_dim)
		block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
		cache_seqlens = torch.tensor([seq_len], dtype=torch.int32)
		hidden = torch.randn(batch, 1, config.hidden_size)

		top_k = idx.score_and_select(
			hidden, blocked_k, block_table, cache_seqlens, page_size
		)
		assert top_k.shape == (1, seq_len)

	def test_score_and_select_indices_in_range(self, indexer, indexer_config, page_size):
		"""All returned indices should be < cache_seqlens."""
		batch, seq_len = 2, 20
		num_pages_per_seq = (seq_len + page_size - 1) // page_size
		total_pages = batch * num_pages_per_seq

		blocked_k = torch.randn(total_pages, page_size, 1, indexer.index_dim)
		block_table = torch.zeros(batch, num_pages_per_seq, dtype=torch.int32)
		for b in range(batch):
			for p in range(num_pages_per_seq):
				block_table[b, p] = b * num_pages_per_seq + p

		cache_seqlens = torch.tensor([seq_len, seq_len - 5], dtype=torch.int32)
		hidden = torch.randn(batch, 1, indexer_config.hidden_size)

		top_k = indexer.score_and_select(
			hidden, blocked_k, block_table, cache_seqlens, page_size
		)

		# All indices for batch 0 should be < seq_len
		assert (top_k[0] < seq_len).all()
		# All indices for batch 1 should be < seq_len - 5
		assert (top_k[1] < seq_len - 5).all()


# ---------------------------------------------------------------------------
# Test: sparse_gather_from_paged_kv
# ---------------------------------------------------------------------------

class TestSparseGather:

	def test_basic_gather(self, page_size):
		"""Gather specific positions and verify against direct indexing."""
		num_pages = 8
		num_k_heads = 1
		k_head_dim = 64

		# Create cache with known values: each token gets a unique vector
		blocked_k = torch.zeros(num_pages, page_size, num_k_heads, k_head_dim)
		for page in range(num_pages):
			for offset in range(page_size):
				token_id = page * page_size + offset
				blocked_k[page, offset, 0, :] = token_id  # Fill with token_id

		# Simple block table: identity mapping
		batch = 1
		block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)

		# Gather tokens at positions [0, 5, 10, 31]
		token_indices = torch.tensor([[0, 5, 10, 31]], dtype=torch.int32)

		gathered = sparse_gather_from_paged_kv(
			blocked_k, block_table, token_indices, page_size
		)

		assert gathered.shape == (1, 4, 1, 64)
		# Verify values match expected token IDs
		for i, expected_id in enumerate([0, 5, 10, 31]):
			assert (gathered[0, i, 0, :] == expected_id).all(), (
				f"Position {i}: expected {expected_id}, got {gathered[0, i, 0, 0].item()}"
			)

	def test_batched_gather(self, page_size):
		"""Multiple sequences with different page mappings."""
		num_pages = 16
		num_k_heads = 1
		k_head_dim = 8

		blocked_k = torch.randn(num_pages, page_size, num_k_heads, k_head_dim)

		# Two sequences, each using 4 pages but mapped differently
		block_table = torch.tensor([
			[0, 1, 2, 3],  # Seq 0: pages 0-3
			[4, 5, 6, 7],  # Seq 1: pages 4-7
		], dtype=torch.int32)

		# Gather first token from each sequence
		token_indices = torch.tensor([[0], [0]], dtype=torch.int32)

		gathered = sparse_gather_from_paged_kv(
			blocked_k, block_table, token_indices, page_size
		)

		assert gathered.shape == (2, 1, 1, 8)
		# Seq 0, token 0 → page 0, offset 0
		torch.testing.assert_close(gathered[0, 0], blocked_k[0, 0])
		# Seq 1, token 0 → page 4, offset 0
		torch.testing.assert_close(gathered[1, 0], blocked_k[4, 0])

	def test_cross_page_boundary(self, page_size):
		"""Gather tokens that span page boundaries."""
		num_pages = 4
		num_k_heads = 1
		k_head_dim = 4

		blocked_k = torch.randn(num_pages, page_size, num_k_heads, k_head_dim)
		block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)

		# Last token of page 0, first token of page 1
		last_of_page0 = page_size - 1
		first_of_page1 = page_size
		token_indices = torch.tensor([[last_of_page0, first_of_page1]], dtype=torch.int32)

		gathered = sparse_gather_from_paged_kv(
			blocked_k, block_table, token_indices, page_size
		)

		assert gathered.shape == (1, 2, 1, 4)
		torch.testing.assert_close(gathered[0, 0], blocked_k[0, page_size - 1])
		torch.testing.assert_close(gathered[0, 1], blocked_k[1, 0])


# ---------------------------------------------------------------------------
# Test: _gather_all_from_paged_cache (internal helper)
# ---------------------------------------------------------------------------

class TestGatherAll:

	def test_basic(self, page_size):
		"""Verify full gather matches sequential access."""
		num_pages = 4
		seq_len = num_pages * page_size
		num_k_heads = 1
		k_head_dim = 8

		# Fill with known pattern
		blocked_k = torch.zeros(num_pages, page_size, num_k_heads, k_head_dim)
		for p in range(num_pages):
			for o in range(page_size):
				blocked_k[p, o, 0, :] = p * page_size + o

		block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
		cache_seqlens = torch.tensor([seq_len], dtype=torch.int32)

		gathered = _gather_all_from_paged_cache(
			blocked_k, block_table, cache_seqlens, page_size, seq_len
		)

		assert gathered.shape == (1, seq_len, 1, 8)
		for t in range(seq_len):
			assert (gathered[0, t, 0, :] == t).all()


# ---------------------------------------------------------------------------
# Test: DSA end-to-end shape flow (mock, no real kernels)
# ---------------------------------------------------------------------------

class TestDSAEndToEnd:

	def test_prefill_shapes(self, indexer, indexer_config):
		"""Prefill produces correctly shaped indexer KV for cache storage."""
		batch, seq_len = 2, 64
		hidden = torch.randn(batch, seq_len, indexer_config.hidden_size)

		indexer_kv = indexer.compute_indexer_kv(hidden)
		assert indexer_kv.shape == (batch, seq_len, 1, indexer.index_dim)

	def test_decode_flow_shapes(self, indexer, indexer_config, page_size):
		"""Mock decode: indexer score → top-K → sparse gather → verify shapes."""
		batch = 2
		seq_len = 24  # Existing cache
		topk = indexer_config.index_topk

		# Mock indexer paged cache
		idx_num_pages_per_seq = (seq_len + page_size - 1) // page_size
		idx_total_pages = batch * idx_num_pages_per_seq
		idx_blocked_k = torch.randn(
			idx_total_pages, page_size, 1, indexer.index_dim
		)
		idx_block_table = torch.zeros(batch, idx_num_pages_per_seq, dtype=torch.int32)
		for b in range(batch):
			for p in range(idx_num_pages_per_seq):
				idx_block_table[b, p] = b * idx_num_pages_per_seq + p

		cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32)
		hidden = torch.randn(batch, 1, indexer_config.hidden_size)

		# Step 1: Score and select
		top_k_indices = indexer.score_and_select(
			hidden, idx_blocked_k, idx_block_table, cache_seqlens, page_size
		)
		effective_topk = min(topk, seq_len)
		assert top_k_indices.shape == (batch, effective_topk)

		# Step 2: Mock MLA paged cache and sparse gather
		mla_k_head_dim = 576
		mla_total_pages = idx_total_pages  # Same page count
		mla_blocked_k = torch.randn(mla_total_pages, page_size, 1, mla_k_head_dim)
		mla_block_table = idx_block_table.clone()

		sparse_kv = sparse_gather_from_paged_kv(
			mla_blocked_k, mla_block_table, top_k_indices, page_size
		)
		assert sparse_kv.shape == (batch, effective_topk, 1, mla_k_head_dim)


# ---------------------------------------------------------------------------
# Test: DualKVCacheCoordinator
# ---------------------------------------------------------------------------

class TestDualKVCacheCoordinator:
	"""Tests for DualKVCacheCoordinator.

	These tests use mock managers since real GPUPagedKVCacheManager
	requires CUDA. The coordinator's job is purely delegation, so
	verifying call forwarding is sufficient.
	"""

	def test_init(self):
		from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator

		primary = _MockManager("primary")
		auxiliary = _MockManager("auxiliary")
		coord = DualKVCacheCoordinator(primary, auxiliary)

		assert coord.primary is primary
		assert coord.auxiliary is auxiliary

	def test_initialize_delegates(self):
		from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator

		primary = _MockManager("primary")
		auxiliary = _MockManager("auxiliary")
		coord = DualKVCacheCoordinator(primary, auxiliary)
		coord.initialize()

		assert primary.initialized
		assert auxiliary.initialized

	def test_destroy_delegates(self):
		from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator

		primary = _MockManager("primary")
		auxiliary = _MockManager("auxiliary")
		coord = DualKVCacheCoordinator(primary, auxiliary)
		coord.destroy(empty_cuda_cache=True)

		assert primary.destroyed
		assert auxiliary.destroyed

	def test_allocate_pages_delegates(self):
		from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator

		primary = _MockManager("primary")
		auxiliary = _MockManager("auxiliary")
		coord = DualKVCacheCoordinator(primary, auxiliary)

		result = coord.allocate_pages(42, 128)

		assert result == [0, 1]  # From primary mock
		assert primary.allocate_calls == [(42, 128)]
		assert auxiliary.allocate_calls == [(42, 128)]

	def test_free_pages_delegates(self):
		from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator

		primary = _MockManager("primary")
		auxiliary = _MockManager("auxiliary")
		coord = DualKVCacheCoordinator(primary, auxiliary)

		coord.free_pages_for_sequences([1, 2, 3])

		assert primary.free_calls == [[1, 2, 3]]
		assert auxiliary.free_calls == [[1, 2, 3]]

	def test_rebuild_page_table_delegates(self):
		from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator

		primary = _MockManager("primary")
		auxiliary = _MockManager("auxiliary")
		coord = DualKVCacheCoordinator(primary, auxiliary)

		result = coord.rebuild_page_table([10, 20])

		assert result == "primary_table"  # From primary mock
		assert primary.rebuild_calls == [[10, 20]]
		assert auxiliary.rebuild_calls == [[10, 20]]


class _MockManager:
	"""Mock GPUPagedKVCacheManager for testing DualKVCacheCoordinator."""

	def __init__(self, name: str):
		self.name = name
		self.initialized = False
		self.destroyed = False
		self.allocate_calls = []
		self.free_calls = []
		self.rebuild_calls = []

	def initialize(self):
		self.initialized = True

	def destroy(self, *, empty_cuda_cache=False):
		self.destroyed = True

	def is_initialized(self):
		return self.initialized

	def allocate_pages(self, sequence_id, num_tokens):
		self.allocate_calls.append((sequence_id, num_tokens))
		return [0, 1]  # Mock pages

	def allocate_pages_for_sequences(self, sequence_ids, num_tokens):
		return [[0, 1]] * len(sequence_ids)

	def free_pages_for_sequences(self, sequence_ids):
		self.free_calls.append(list(sequence_ids))

	def rebuild_page_table(self, sequence_ids):
		self.rebuild_calls.append(list(sequence_ids))
		return f"{self.name}_table"

	def grow_sequence_pages(self, sequence_id, additional_tokens):
		return [0]

	def grow_pages_for_sequences(self, sequence_ids, additional_tokens):
		return [[0]] * len(sequence_ids)

	def clear_page_table(self):
		pass

	def get_stats(self):
		return None
