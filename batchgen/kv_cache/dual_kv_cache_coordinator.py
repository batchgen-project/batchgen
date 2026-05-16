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
from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator

logger = logging.getLogger(__name__)


class DualKVCacheCoordinator(GPUKVCoordinator):
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
		super().__init__(primary_component_name="primary")
		self.auxiliary = auxiliary
		self.register_component("primary", primary)
		self.register_component("auxiliary", auxiliary)
		if primary.config.page_size_tokens != auxiliary.config.page_size_tokens:
			raise ValueError(
				"DSA primary/aux GPU KV page size mismatch: "
				f"primary={primary.config.page_size_tokens}, "
				f"aux={auxiliary.config.page_size_tokens}"
			)

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
		self._preflight_allocate([sequence_id], [num_tokens], "allocate_pages")
		pages = {}
		try:
			pages = self.primary.allocate_pages_for_sequences([sequence_id], [num_tokens])
			self.auxiliary.allocate_pages_for_sequences([sequence_id], [num_tokens])
		except Exception:
			self._rollback_allocations(self.primary, pages)
			raise
		self._assert_mirrored_sequence_pages(sequence_id, "allocate_pages")
		return pages.get(sequence_id, [])

	def allocate_pages_for_sequences(
		self,
		sequence_ids: Sequence[int],
		num_tokens: Sequence[int],
	) -> List[List[int]]:
		self._preflight_allocate(sequence_ids, num_tokens, "allocate_pages_for_sequences")
		result = {}
		try:
			result = self.primary.allocate_pages_for_sequences(sequence_ids, num_tokens)
			self.auxiliary.allocate_pages_for_sequences(sequence_ids, num_tokens)
		except Exception:
			self._rollback_allocations(self.primary, result)
			raise
		self.assert_mirrored_state("allocate_pages_for_sequences", sequence_ids)
		return result

	def grow_sequence_pages(self, sequence_id: int, additional_tokens: int) -> List[int]:
		self._preflight_grow([sequence_id], [additional_tokens], "grow_sequence_pages")
		pages = {}
		try:
			pages = self.primary.grow_pages_for_sequences([sequence_id], [additional_tokens])
			self.auxiliary.grow_pages_for_sequences([sequence_id], [additional_tokens])
		except Exception:
			self._rollback_allocations(self.primary, pages)
			raise
		self._assert_mirrored_sequence_pages(sequence_id, "grow_sequence_pages")
		return pages.get(sequence_id, [])

	def grow_pages_for_sequences(
		self,
		sequence_ids: Sequence[int],
		additional_tokens: Sequence[int],
	) -> List[List[int]]:
		self._preflight_grow(sequence_ids, additional_tokens, "grow_pages_for_sequences")
		result = {}
		try:
			result = self.primary.grow_pages_for_sequences(sequence_ids, additional_tokens)
			self.auxiliary.grow_pages_for_sequences(sequence_ids, additional_tokens)
		except Exception:
			self._rollback_allocations(self.primary, result)
			raise
		self.assert_mirrored_state("grow_pages_for_sequences", sequence_ids)
		return result

	def extend_pages_for_sequence(self, sequence_id: int, new_total_tokens: int) -> int:
		primary_state = self.primary._sequences.get(sequence_id)
		if primary_state is None:
			raise KeyError(f"extend_pages_for_sequence: sequence {sequence_id} is not allocated")
		required = int(self.primary._geometry.required_pages(new_total_tokens))
		missing = max(0, required - int(primary_state.pages.numel()))
		if missing > 0:
			self._preflight_grow([sequence_id], [missing], "extend_pages_for_sequence")
		result = {}
		try:
			added = self.primary.extend_pages_for_sequence(sequence_id, new_total_tokens)
			if added:
				state = self.primary._sequences[sequence_id]
				result = {sequence_id: state.pages[-added:].tolist()}
			self.auxiliary.extend_pages_for_sequence(sequence_id, new_total_tokens)
		except Exception:
			self._rollback_allocations(self.primary, result)
			raise
		self._assert_mirrored_sequence_pages(sequence_id, "extend_pages_for_sequence")
		return added

	# -- Page table --

	def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
		table = self.primary.rebuild_page_table(sequence_ids)
		self.auxiliary.rebuild_page_table(sequence_ids)
		self._assert_mirrored_page_tables("rebuild_page_table")
		return table

	def clear_page_table(self) -> None:
		self.primary.clear_page_table()
		self.auxiliary.clear_page_table()
		self._assert_mirrored_page_tables("clear_page_table")

	# -- Page freeing --

	def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
		self._assert_sequences_present(self.primary, sequence_ids, "free_pages_for_sequences")
		self._assert_sequences_present(self.auxiliary, sequence_ids, "free_pages_for_sequences")
		self.primary.free_pages_for_sequences(sequence_ids)
		self.auxiliary.free_pages_for_sequences(sequence_ids)
		self.assert_mirrored_state("free_pages_for_sequences")

	# -- Query (primary only) --

	def get_stats(self) -> GPUPagedKVStats:
		self.assert_mirrored_state("get_stats")
		return self.primary.get_stats()

	def get_kv_tensors(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		raise RuntimeError(
			"DualKVCacheCoordinator.get_kv_tensors() is primary-only and invalid for DSA; "
			"use explicit primary/auxiliary managers"
		)

	def get_layer_kv_with_page_table(self, layer_idx: int):
		raise RuntimeError(
			"DualKVCacheCoordinator.get_layer_kv_with_page_table() is primary-only and invalid for DSA; "
			"use explicit primary/auxiliary managers"
		)

	def update_layer_decode_new_token(self, *args, **kwargs):
		raise RuntimeError(
			"DualKVCacheCoordinator.update_layer_decode_new_token() is primary-only and invalid for DSA; "
			"write primary and auxiliary KV explicitly"
		)

	def get_page_table_version(self) -> int:
		return self.primary.get_page_table_version()

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
		raise RuntimeError(
			"DualKVCacheCoordinator.export_active_sequence_page_counts() is primary-only and invalid for DSA; "
			"use explicit primary/auxiliary managers"
		)

	def get_padded_3d_page_pointers(self, *args, **kwargs):
		raise RuntimeError(
			"DualKVCacheCoordinator.get_padded_3d_page_pointers() is primary-only and invalid for DSA; "
			"use explicit primary/auxiliary managers"
		)

	def _preflight_allocate(
		self, sequence_ids: Sequence[int], num_tokens: Sequence[int], op_name: str
	) -> None:
		if len(sequence_ids) != len(num_tokens):
			raise ValueError(f"{op_name}: sequence_ids and num_tokens must be the same length")
		for manager_name, manager in (
			("primary", self.primary),
			("auxiliary", self.auxiliary),
		):
			manager._ensure_initialized()
			required_pages = manager._geometry.required_pages(num_tokens).tolist()
			missing_total = 0
			for seq_id, required in zip(sequence_ids, required_pages):
				required_int = int(required)
				if required_int <= 0:
					raise ValueError(
						f"{op_name}: required pages must be positive for seq {seq_id}, "
						f"got {required_int}"
					)
				state = manager._sequences.get(seq_id)
				current = int(state.pages.numel()) if state else 0
				missing_total += max(0, required_int - current)
			if missing_total > manager._free_pages.size:
				raise RuntimeError(
					f"{op_name}: insufficient {manager_name} free pages: "
					f"need {missing_total}, free {manager._free_pages.size}"
				)

	def _preflight_grow(
		self, sequence_ids: Sequence[int], num_pages: Sequence[int], op_name: str
	) -> None:
		if len(sequence_ids) != len(num_pages):
			raise ValueError(f"{op_name}: sequence_ids and num_pages must be the same length")
		for manager_name, manager in (
			("primary", self.primary),
			("auxiliary", self.auxiliary),
		):
			manager._ensure_initialized()
			self._assert_sequences_present(manager, sequence_ids, op_name)
			needed = 0
			for seq_id, count in zip(sequence_ids, num_pages):
				if count <= 0:
					raise ValueError(
						f"{op_name}: num_pages must be positive for seq {seq_id}, got {count}"
					)
				needed += int(count)
			if needed > manager._free_pages.size:
				raise RuntimeError(
					f"{op_name}: insufficient {manager_name} free pages: "
					f"need {needed}, free {manager._free_pages.size}"
				)

	def _assert_sequences_present(
		self, manager: GPUPagedKVCacheManager, sequence_ids: Sequence[int], op_name: str
	) -> None:
		missing = [seq_id for seq_id in sequence_ids if seq_id not in manager._sequences]
		if missing:
			raise KeyError(
				f"{op_name}: sequences not allocated: "
				+ ", ".join(str(seq_id) for seq_id in missing)
			)

	def _rollback_allocations(
		self, manager: GPUPagedKVCacheManager, allocations
	) -> None:
		if not allocations:
			return
		reclaimed = []
		for seq_id, pages in allocations.items():
			if not pages:
				continue
			state = manager._sequences.get(seq_id)
			if state is None:
				continue
			count = len(pages)
			tail = state.pages[-count:].tolist()
			if tail != pages:
				raise RuntimeError(
					f"Cannot rollback KV allocation for seq {seq_id}: "
					f"tail={tail}, allocated={pages}"
				)
			reclaimed.append(state.pages[-count:].clone())
			if state.pages.numel() == count:
				del manager._sequences[seq_id]
			else:
				state.pages = state.pages[:-count].clone()
		if reclaimed:
			manager._free_pages.push(torch.cat(reclaimed, dim=0))
			manager._clear_active_page_pointer_tables()

	def _assert_mirrored_sequence_pages(self, sequence_id: int, op_name: str) -> None:
		primary_state = self.primary._sequences.get(sequence_id)
		aux_state = self.auxiliary._sequences.get(sequence_id)
		if primary_state is None or aux_state is None:
			raise RuntimeError(
				f"{op_name}: primary/auxiliary allocation mismatch for seq {sequence_id}: "
				f"primary={primary_state is not None}, auxiliary={aux_state is not None}"
			)
		primary_pages = int(primary_state.pages.numel())
		aux_pages = int(aux_state.pages.numel())
		if primary_pages != aux_pages:
			raise RuntimeError(
				f"{op_name}: primary/auxiliary page-count mismatch for seq {sequence_id}: "
				f"primary={primary_pages}, auxiliary={aux_pages}"
			)
		if not torch.equal(primary_state.pages, aux_state.pages):
			raise RuntimeError(
				f"{op_name}: primary/auxiliary page-vector mismatch for seq {sequence_id}: "
				f"primary={primary_state.pages.tolist()[:10]}, "
				f"auxiliary={aux_state.pages.tolist()[:10]}"
			)

	def assert_mirrored_state(
		self, op_name: str, sequence_ids: Optional[Sequence[int]] = None
	) -> None:
		primary_ids = set(self.primary._sequences.keys())
		aux_ids = set(self.auxiliary._sequences.keys())
		if primary_ids != aux_ids:
			raise RuntimeError(
				f"{op_name}: primary/auxiliary sequence-id set mismatch: "
				f"primary_only={sorted(primary_ids - aux_ids)[:10]}, "
				f"aux_only={sorted(aux_ids - primary_ids)[:10]}"
			)
		check_ids = list(sequence_ids) if sequence_ids is not None else sorted(primary_ids)
		for seq_id in check_ids:
			if seq_id in primary_ids:
				self._assert_mirrored_sequence_pages(seq_id, op_name)
		self._assert_mirrored_page_tables(op_name)

	def _assert_mirrored_page_tables(self, op_name: str) -> None:
		primary_mgr = self.primary._gpu_page_table_manager
		aux_mgr = self.auxiliary._gpu_page_table_manager
		primary_order = list(primary_mgr.slot_to_seq_id)
		aux_order = list(aux_mgr.slot_to_seq_id)
		if primary_order != aux_order:
			raise RuntimeError(
				f"{op_name}: primary/auxiliary slot order mismatch: "
				f"primary={primary_order[:10]} len={len(primary_order)}, "
				f"auxiliary={aux_order[:10]} len={len(aux_order)}"
			)
		primary_shape = None if primary_mgr.gpu_table is None else tuple(primary_mgr.gpu_table.shape)
		aux_shape = None if aux_mgr.gpu_table is None else tuple(aux_mgr.gpu_table.shape)
		if primary_shape != aux_shape:
			raise RuntimeError(
				f"{op_name}: primary/auxiliary page-table shape mismatch: "
				f"primary={primary_shape}, auxiliary={aux_shape}"
			)
		if primary_mgr.gpu_table is None or aux_mgr.gpu_table is None:
			if primary_mgr.gpu_table is not aux_mgr.gpu_table:
				raise RuntimeError(
					f"{op_name}: primary/auxiliary page-table presence mismatch"
				)
			return
		if not torch.equal(primary_mgr.gpu_table, aux_mgr.gpu_table):
			raise RuntimeError(
				f"{op_name}: primary/auxiliary page-table contents mismatch"
			)
