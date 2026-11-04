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

import contextlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import (
	GPUPagedKVCacheManager,
	GPUPagedKVConfig,
	GPUPagedKVStats,
	PrefixGPUShareStats,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DualPoolPageKeys:
	"""Host page-key rows for the primary pool and for the auxiliary pool.

	The two GPU pools are backed by two SEPARATE Host KV regions, and each
	region allocates its own pages (``DualHostKVCoordinator`` mirrors the page
	COUNTS, not the ids). A Host page id therefore identifies content only
	inside its own region, so the sharing identity has to be built per pool
	and the two key spaces may never be crossed.

	``primary[i]`` and ``auxiliary[i]`` describe the same sequence, in the
	same logical page order, which is why both rows must have the same length:
	the coordinator keeps the two pools' page counts mirrored.
	"""

	primary: Tuple[Tuple[int, ...], ...]
	auxiliary: Tuple[Tuple[int, ...], ...]

	@classmethod
	def from_rows(cls, primary, auxiliary) -> "DualPoolPageKeys":
		return cls(
			primary=tuple(tuple(int(key) for key in row) for row in primary),
			auxiliary=tuple(tuple(int(key) for key in row) for row in auxiliary),
		)

	def __post_init__(self) -> None:
		if len(self.primary) != len(self.auxiliary):
			raise ValueError(
				"DualPoolPageKeys: primary/auxiliary row count mismatch: "
				f"primary={len(self.primary)}, auxiliary={len(self.auxiliary)}"
			)
		for index, (primary_row, aux_row) in enumerate(
			zip(self.primary, self.auxiliary)
		):
			if len(primary_row) != len(aux_row):
				raise ValueError(
					"DualPoolPageKeys: primary/auxiliary page-key count "
					f"mismatch for row {index}: primary={len(primary_row)}, "
					f"auxiliary={len(aux_row)}"
				)


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
		if primary.config.page_size_tokens != auxiliary.config.page_size_tokens:
			raise ValueError(
				"DSA primary/aux GPU KV page size mismatch: "
				f"primary={primary.config.page_size_tokens}, "
				f"aux={auxiliary.config.page_size_tokens}"
			)
		# STICKY: has keyed (deduplicated) allocation ever touched either
		# pool's free stack? Once it has, the two pools' physical page ids are
		# permanently independent and may never be compared again until both
		# pools are rebuilt from empty. See `_page_ids_must_mirror`. Derived
		# from the pools rather than hardcoded so wrapping managers that
		# already hold keyed pages starts out relaxed, not crashing.
		self._keyed_pages_assigned = bool(
			primary._shared_page_key_to_gpu_page
			or auxiliary._shared_page_key_to_gpu_page
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
		# Both pools are back to empty: `_reset_runtime_state` (allocator
		# pools) and `_release_fixed_address_pool` (fixed-address pools) both
		# rebuild the free stack and drop the shared-page registry, so the
		# next page handed out is the two pools' first again and strict
		# id mirroring is meaningful once more. `initialize()` is NOT such a
		# point: on an already-initialized manager it warns and returns
		# without touching allocator state.
		self._keyed_pages_assigned = False

	@property
	def is_initialized(self) -> bool:
		return self.primary.is_initialized and self.auxiliary.is_initialized

	@property
	def uses_fixed_address_pool(self) -> bool:
		"""True when BOTH pools unmap and map one reserved range each.

		Mirrored like every other lifecycle property: a pair where only one
		side keeps its addresses is a configuration error, not a half-usable
		state, because the decode graphs bake pointers into both.
		"""
		primary = self.primary.uses_fixed_address_pool
		auxiliary = self.auxiliary.uses_fixed_address_pool
		if primary != auxiliary:
			raise RuntimeError(
				"DSA primary/aux GPU KV pools disagree on fixed addressing: "
				f"primary={primary}, auxiliary={auxiliary}"
			)
		return primary

	@property
	def pool_is_mapped(self) -> bool:
		"""True when both pools' physical memory is currently mapped."""
		primary = self.primary.pool_is_mapped
		auxiliary = self.auxiliary.pool_is_mapped
		if primary != auxiliary:
			raise RuntimeError(
				"DSA primary/aux GPU KV pools diverged in mapped state: "
				f"primary={primary}, auxiliary={auxiliary}"
			)
		return primary

	def assert_pool_mapped(self, context: str) -> None:
		"""Fail loud before anything dereferences an unmapped KV pool."""
		self.primary.assert_pool_mapped(f"{context} (primary)")
		self.auxiliary.assert_pool_mapped(f"{context} (auxiliary)")

	@contextlib.contextmanager
	def capacity_snapshot(self):
		"""Hold both pools mapped for a decode admission / capacity read (#424)."""
		with self.primary.capacity_snapshot(), self.auxiliary.capacity_snapshot():
			yield self

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

	# -- Page allocation with Host page identity (dedup) --

	def _pool_page_keys(self, page_keys_by_sequence, op_name: str):
		"""The (name, manager, key rows) triples of a keyed call.

		The dual coordinator needs one key space per pool, so it accepts only
		``DualPoolPageKeys``. A plain row list -- what a single manager takes
		-- would silently key the indexer pool with MLA Host page ids, which
		is exactly the cross-pool identity error that must never be guessed
		at.
		"""

		if not isinstance(page_keys_by_sequence, DualPoolPageKeys):
			raise TypeError(
				f"{op_name}: the dual KV coordinator requires DualPoolPageKeys "
				"(one Host page-key space per pool), got "
				f"{type(page_keys_by_sequence).__name__}"
			)
		return (
			("primary", self.primary, page_keys_by_sequence.primary),
			("auxiliary", self.auxiliary, page_keys_by_sequence.auxiliary),
		)

	def count_new_shared_pages(
		self,
		sequence_ids: Sequence[int],
		num_tokens: Sequence[int],
		page_keys_by_sequence: "DualPoolPageKeys",
	) -> int:
		"""Physical pages one keyed allocation would take from a pool.

		Both pools must have room for it, and both pools charge the same
		number of pages for the same batch (their page counts are mirrored and
		their keys group the rows identically), so one number describes the
		demand. A disagreement means the two pools' page identity no longer
		groups the rows the same way: that is a bug in the key sources, not a
		capacity question, so it is named instead of being hidden by a max().
		"""

		demands = {}
		for name, manager, rows in self._pool_page_keys(
			page_keys_by_sequence, "count_new_shared_pages"
		):
			demands[name] = int(
				manager.count_new_shared_pages(sequence_ids, num_tokens, rows)
			)
		if demands["primary"] != demands["auxiliary"]:
			raise RuntimeError(
				"count_new_shared_pages: primary/auxiliary shared-page demand "
				f"mismatch: primary={demands['primary']}, "
				f"auxiliary={demands['auxiliary']}"
			)
		return demands["primary"]

	def allocate_pages_for_sequences_with_page_keys(
		self,
		sequence_ids: Sequence[int],
		num_tokens: Sequence[int],
		page_keys_by_sequence: "DualPoolPageKeys",
	) -> Dict[int, List[int]]:
		"""Allocate in BOTH pools, deduplicating each pool on its own keys.

		Each wrapped manager maps one physical page per distinct Host page id
		of ITS region and refcounts it, so a page is reclaimed when the last
		reader IN THAT POOL releases it (``_release_gpu_pages``). The pools
		are independent registries; nothing crosses.

		Atomicity: both pools are previewed first (the preview runs the
		allocator's own validation, so a demand it accepts cannot fail
		validation later), and if the auxiliary allocation still raises, the
		sequences created in the primary pool are freed through the refcounted
		release -- which is why a rollback cannot take a page a sibling row
		still reads.
		"""

		op_name = "allocate_pages_for_sequences_with_page_keys"
		pools = self._pool_page_keys(page_keys_by_sequence, op_name)
		created = [int(seq_id) for seq_id in sequence_ids]
		# Armed BEFORE anything pops: one keyed allocation makes the two pools'
		# page ids independent for good. Even a rolled-back attempt does,
		# because the primary pool popped and pushed pages its peer never
		# touched, so the two free stacks no longer hand out the same ids.
		# Releasing every shared page later does NOT undo that.
		self._keyed_pages_assigned = True
		for name, manager, rows in pools:
			manager._ensure_initialized()
			demand = int(
				manager.count_new_shared_pages(sequence_ids, num_tokens, rows)
			)
			if demand > manager._free_pages.size:
				raise RuntimeError(
					f"{op_name}: insufficient {name} free pages: "
					f"need {demand}, free {manager._free_pages.size}"
				)
		allocations = self.primary.allocate_pages_for_sequences_with_page_keys(
			sequence_ids, num_tokens, page_keys_by_sequence.primary
		)
		try:
			self.auxiliary.allocate_pages_for_sequences_with_page_keys(
				sequence_ids, num_tokens, page_keys_by_sequence.auxiliary
			)
		except Exception:
			self.primary.free_pages_for_sequences(created)
			raise
		self.assert_mirrored_state(op_name, sequence_ids)
		return allocations

	def get_prefix_gpu_share_stats(self) -> PrefixGPUShareStats:
		"""Sharing in the primary pool, which both pools mirror page for page.

		Reported from the primary manager for the same reason ``get_stats()``
		is: the two pools hold the same number of pages per sequence and the
		same grouping of shared pages, so one side's counts describe both.
		"""

		return self.primary.get_prefix_gpu_share_stats()

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

	def reset_allocations(self) -> None:
		self.primary.reset_allocations()
		self.auxiliary.reset_allocations()
		# Every page returns and both shared-page registries are cleared (see
		# GPUPagedKVCacheManager.reset_allocations), so the pools are in
		# lockstep again: re-arm strict page-id mirroring, and let the
		# assertion below run under it.
		self._keyed_pages_assigned = False
		self.assert_mirrored_state("reset_allocations")

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
		"""Undo one pool's just-made allocation, through the refcount.

		The pages go back with ``_release_gpu_pages``, not straight onto the
		free stack: a page this pool deduplicated may still be read by a
		resident sequence, so only the last reference may reclaim it -- and a
		page pushed back while its refcount still stands would be handed out
		twice.
		"""

		if not allocations:
			return
		touched = False
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
			released = state.pages[-count:].clone()
			if state.pages.numel() == count:
				del manager._sequences[seq_id]
			else:
				state.pages = state.pages[:-count].clone()
			manager._release_gpu_pages(released)
			touched = True
		if touched:
			manager._clear_active_page_pointer_tables()

	def has_shared_pages(self) -> bool:
		"""Is either pool deduplicating pages by Host page key RIGHT NOW?

		A live read, for observability only. It must never gate a mirroring
		assertion: it goes back to False as soon as the last shared page is
		released, while the page ids it caused to diverge stay diverged. That
		is what `_keyed_pages_assigned` is for.
		"""

		return bool(
			self.primary._shared_page_key_to_gpu_page
			or self.auxiliary._shared_page_key_to_gpu_page
		)

	def _page_ids_must_mirror(self) -> bool:
		"""Must both pools hold the SAME physical page ids for a sequence?

		Without page keys, yes: the two pools have the same page count and
		every allocation, growth and release is mirrored, so their free stacks
		stay in lockstep and any id difference is a real mirroring bug worth
		crashing on.

		Keyed allocation breaks that coincidence on purpose. Each pool dedups
		on the Host page ids of its OWN region, and the physical page a new
		key gets follows the order of those ids, so two pools that group the
		rows identically can still assign different ids. Page ids are not
		required to agree for correctness: each pool is addressed through its
		own page table (the indexer path reads
		``gpu_paged_kv_manager_aux._gpu_page_table_manager.gpu_table``,
		batchgen/attention/dsa/glm5_decode_selector.py:922, and the dual host
		load takes each pool's own pointers,
		batchgen_worker.py:5376-5379). What must stay mirrored is the SHAPE --
		the sequence set, the page count per sequence, the slot order and the
		table shape -- and those are checked unconditionally.

		STICKY, never live. The first version of this read
		`not has_shared_pages()` and crashed a GLM-5.3 lcb run inside
		`free_pages_for_sequences`: the rows shared one chat-template prefix
		page, their ids diverged, then the last reader released that page, the
		registries went empty and the CONTENT assertion re-armed against
		tables that had already diverged. Divergence is not undone by
		releasing the pages that caused it -- only by rebuilding both pools
		from empty (`destroy`, `reset_allocations`), which is where the flag
		is cleared.
		"""

		return not self._keyed_pages_assigned

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
		if self._page_ids_must_mirror() and not torch.equal(
			primary_state.pages, aux_state.pages
		):
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
		if self._page_ids_must_mirror() and not torch.equal(
			primary_mgr.gpu_table, aux_mgr.gpu_table
		):
			raise RuntimeError(
				f"{op_name}: primary/auxiliary page-table contents mismatch"
			)
