"""Coordinator for dual KV caches (MLA + Indexer) used by DSA models.

DSA models (DeepSeek-V3.2, GLM-5) require two paged KV caches per layer:
  - Primary: MLA compressed KV (dim=576)
  - Auxiliary: Indexer KV for token scoring (dim=128)

This coordinator wraps two GPUPagedKVCacheManager instances and delegates
all lifecycle operations to both, keeping them synchronized. It duck-types
the GPUPagedKVCacheManager API so callers (batchgen_worker.py) can use it
as a drop-in replacement.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import (
	GPUPagedKVCacheManager,
	GPUPagedKVConfig,
	GPUPagedKVStats,
)

logger = logging.getLogger(__name__)


class DualKVCacheCoordinator:
	"""Synchronizes a primary and auxiliary GPUPagedKVCacheManager.

	Both managers track identical sequences with identical token counts.
	All lifecycle operations are mirrored. Return values come from the
	primary manager for backward compatibility.

	Attributes:
		primary: MLA KV cache manager (dim=576 for DeepSeek-V3.2)
		auxiliary: Indexer KV cache manager (dim=128 for DeepSeek-V3.2)
	"""

	def __init__(
		self,
		primary: GPUPagedKVCacheManager,
		auxiliary: GPUPagedKVCacheManager,
	) -> None:
		self.primary = primary
		self.auxiliary = auxiliary

	# -- Lifecycle --

	def initialize(self) -> None:
		self.primary.initialize()
		self.auxiliary.initialize()
		logger.info(
			"DualKVCacheCoordinator initialized: primary=%s, auxiliary=%s",
			self.primary.get_stats(),
			self.auxiliary.get_stats(),
		)

	def destroy(self, *, empty_cuda_cache: bool = False) -> None:
		self.primary.destroy(empty_cuda_cache=empty_cuda_cache)
		self.auxiliary.destroy(empty_cuda_cache=empty_cuda_cache)

	@property
	def is_initialized(self) -> bool:
		return self.primary.is_initialized and self.auxiliary.is_initialized

	# -- Page allocation --

	def allocate_pages(self, sequence_id: int, num_tokens: int) -> List[int]:
		pages = self.primary.allocate_pages(sequence_id, num_tokens)
		self.auxiliary.allocate_pages(sequence_id, num_tokens)
		return pages

	def allocate_pages_for_sequences(
		self,
		sequence_ids: Sequence[int],
		num_tokens: Sequence[int],
	) -> List[List[int]]:
		result = self.primary.allocate_pages_for_sequences(sequence_ids, num_tokens)
		self.auxiliary.allocate_pages_for_sequences(sequence_ids, num_tokens)
		return result

	def grow_sequence_pages(self, sequence_id: int, additional_tokens: int) -> List[int]:
		pages = self.primary.grow_sequence_pages(sequence_id, additional_tokens)
		self.auxiliary.grow_sequence_pages(sequence_id, additional_tokens)
		return pages

	def grow_pages_for_sequences(
		self,
		sequence_ids: Sequence[int],
		additional_tokens: Sequence[int],
	) -> List[List[int]]:
		result = self.primary.grow_pages_for_sequences(sequence_ids, additional_tokens)
		self.auxiliary.grow_pages_for_sequences(sequence_ids, additional_tokens)
		return result

	def extend_pages_for_sequence(self, sequence_id: int, new_total_tokens: int) -> int:
		result = self.primary.extend_pages_for_sequence(sequence_id, new_total_tokens)
		self.auxiliary.extend_pages_for_sequence(sequence_id, new_total_tokens)
		return result

	# -- Page table --

	def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
		table = self.primary.rebuild_page_table(sequence_ids)
		self.auxiliary.rebuild_page_table(sequence_ids)
		return table

	def clear_page_table(self) -> None:
		self.primary.clear_page_table()
		self.auxiliary.clear_page_table()

	# -- Page freeing --

	def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
		self.primary.free_pages_for_sequences(sequence_ids)
		self.auxiliary.free_pages_for_sequences(sequence_ids)

	# -- Query (primary only) --

	def get_stats(self) -> GPUPagedKVStats:
		return self.primary.get_stats()

	def get_kv_tensors(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		return self.primary.get_kv_tensors()

	def get_layer_kv_with_page_table(self, layer_idx: int):
		return self.primary.get_layer_kv_with_page_table(layer_idx)

	def update_layer_decode_new_token(self, *args, **kwargs):
		return self.primary.update_layer_decode_new_token(*args, **kwargs)

	@property
	def config(self) -> GPUPagedKVConfig:
		return self.primary.config

	@property
	def device(self):
		return self.primary.device

	@property
	def _gpu_page_table_manager(self):
		return self.primary._gpu_page_table_manager

	@property
	def _sequences(self):
		# Both managers track identical sequences — primary is authoritative.
		return self.primary._sequences

	def copy_kv_to_tensor(self, sequence_id: int) -> torch.Tensor:
		return self.primary.copy_kv_to_tensor(sequence_id)

	def copy_tensor_to_kv(self, sequence_id: int, k_tensor: torch.Tensor) -> None:
		self.primary.copy_tensor_to_kv(sequence_id, k_tensor)

	# -- Delegation helpers for less common methods --

	def get_context_kv_page_ptrs(self, *args, **kwargs):
		return self.primary.get_context_kv_page_ptrs(*args, **kwargs)

	def get_sequence_layer_page_pointers(self, *args, **kwargs):
		return self.primary.get_sequence_layer_page_pointers(*args, **kwargs)

	def export_layer_page_pointer_table(self, *args, **kwargs):
		return self.primary.export_layer_page_pointer_table(*args, **kwargs)

	def export_active_sequence_page_counts(self) -> torch.Tensor:
		return self.primary.export_active_sequence_page_counts()

	def get_padded_3d_page_pointers(self, *args, **kwargs):
		return self.primary.get_padded_3d_page_pointers(*args, **kwargs)
