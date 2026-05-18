from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager


def _ceil_div(value: int, divisor: int) -> int:
	if divisor <= 0:
		raise ValueError("divisor must be positive")
	return -(-value // divisor)


def _as_int_list(values: Sequence[int] | torch.Tensor) -> list[int]:
	if isinstance(values, torch.Tensor):
		return [int(v) for v in values.detach().cpu().tolist()]
	return [int(v) for v in values]


@dataclass
class _SWASequenceState:
	window_start_page: int = 0
	max_seen_raw_pos: int = -1
	has_tokens: bool = False


@dataclass(frozen=True)
class _WindowForRawEnd:
	window_start_page: int
	active_tokens: int
	required_pages: int


class SWAGPUPagedKVCacheManager:
	"""Page-level sliding-window wrapper for ``GPUPagedKVCacheManager``.

	The backing manager still owns storage, page allocation, page-table rebuilds,
	and token update kernels. This wrapper only converts raw token positions into
	window-local storage positions and releases old prefix pages when the
	page-aligned SWA window moves forward.
	"""

	def __init__(
		self,
		base_manager: GPUPagedKVCacheManager,
		*,
		window_size_tokens: int,
	) -> None:
		if base_manager is None:
			raise ValueError("base_manager must be set")
		self._base = base_manager
		self.window_size_tokens = int(window_size_tokens)
		self.page_size_tokens = int(base_manager.config.page_size_tokens)
		if self.page_size_tokens <= 0:
			raise ValueError("page_size_tokens must be > 0")
		if self.window_size_tokens <= 0:
			raise ValueError("window_size_tokens must be > 0")
		if self.window_size_tokens % self.page_size_tokens != 0:
			raise ValueError(
				"window_size_tokens must be divisible by page_size_tokens "
				"for page-level SWA"
			)
		self.window_pages = self.window_size_tokens // self.page_size_tokens
		self._states: dict[int, _SWASequenceState] = {}
		self._last_page_table_order: Optional[list[int]] = None

	def __getattr__(self, name: str):
		return getattr(self._base, name)

	@property
	def base_manager(self) -> GPUPagedKVCacheManager:
		return self._base

	@property
	def config(self):
		return self._base.config

	@property
	def device(self):
		return self._base.device

	@property
	def is_initialized(self) -> bool:
		return self._base.is_initialized

	@property
	def uses_logical_layer_mapping(self) -> bool:
		return self._base.uses_logical_layer_mapping

	def initialize(self) -> None:
		return self._base.initialize()

	def destroy(self, *, empty_cuda_cache: bool = False) -> None:
		self._states.clear()
		self._last_page_table_order = None
		return self._base.destroy(empty_cuda_cache=empty_cuda_cache)

	def allocate_pages(self, sequence_id: int, num_tokens: int) -> list[int]:
		allocations = self.allocate_pages_for_sequences(
			[int(sequence_id)], [int(num_tokens)]
		)
		return allocations.get(int(sequence_id), [])

	def allocate_pages_for_sequences(
		self, sequence_ids: Sequence[int], num_tokens: Sequence[int]
	) -> dict[int, list[int]]:
		if len(sequence_ids) != len(num_tokens):
			raise ValueError(
				"allocate_pages_for_sequences: sequence_ids and num_tokens "
				"must have the same length"
			)
		sequence_ids = [int(seq_id) for seq_id in sequence_ids]
		raw_tokens = [int(tokens) for tokens in num_tokens]
		active_tokens: list[int] = []
		windows: list[_WindowForRawEnd] = []
		for token_count in raw_tokens:
			window = self._compute_window_for_raw_end(token_count)
			if window.active_tokens <= 0:
				raise ValueError(
					"allocate_pages_for_sequences: num_tokens entries must "
					"be > 0"
				)
			active_tokens.append(window.active_tokens)
			windows.append(window)

		allocations = self._base.allocate_pages_for_sequences(
			sequence_ids, active_tokens
		)
		for seq_id, token_count, window in zip(
			sequence_ids, raw_tokens, windows
		):
			state = self._states.setdefault(seq_id, _SWASequenceState())
			state.window_start_page = window.window_start_page
			state.max_seen_raw_pos = token_count - 1
			state.has_tokens = True
		return allocations

	def grow_sequence_pages(self, sequence_id: int, num_pages: int) -> list[int]:
		return self._base.grow_sequence_pages(int(sequence_id), int(num_pages))

	def grow_pages_for_sequences(
		self, sequence_ids: Sequence[int], num_pages: Sequence[int]
	):
		return self._base.grow_pages_for_sequences(sequence_ids, num_pages)

	def extend_pages_for_sequence(
		self, sequence_id: int, new_total_tokens: int
	) -> int:
		_, added_pages = self._update_window_for_raw_end(
			int(sequence_id), int(new_total_tokens)
		)
		self._rebuild_last_page_table_if_available()
		return added_pages

	def release_sequence_prefix_pages(
		self, sequence_id: int, num_pages: int
	) -> list[int]:
		sequence_id = int(sequence_id)
		num_pages = int(num_pages)
		released = self._base.release_sequence_prefix_pages(
			sequence_id, num_pages
		)
		state = self._states.setdefault(sequence_id, _SWASequenceState())
		state.window_start_page += num_pages
		self._rebuild_last_page_table_if_available()
		return released

	def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
		sequence_ids = [int(seq_id) for seq_id in sequence_ids]
		self._base.free_pages_for_sequences(sequence_ids)
		for seq_id in sequence_ids:
			self._states.pop(seq_id, None)
		if self._last_page_table_order is not None:
			freed = set(sequence_ids)
			self._last_page_table_order = [
				seq_id
				for seq_id in self._last_page_table_order
				if seq_id not in freed
			]

	def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
		self._last_page_table_order = [int(seq_id) for seq_id in sequence_ids]
		return self._base.rebuild_page_table(self._last_page_table_order)

	def clear_page_table(self) -> None:
		self._last_page_table_order = None
		return self._base.clear_page_table()

	def ensure_cuda_graph_page_table(
		self, sequence_ids: Sequence[int]
	) -> torch.Tensor:
		self._last_page_table_order = [int(seq_id) for seq_id in sequence_ids]
		return self._base.ensure_cuda_graph_page_table(
			self._last_page_table_order
		)

	def get_context_kv_page_ptrs(
		self, sequence_id: int, layer_idx: int, context_length: int
	):
		active_tokens, _ = self._update_window_for_raw_end(
			int(sequence_id), int(context_length)
		)
		self._rebuild_last_page_table_if_available()
		return self._base.get_context_kv_page_ptrs(
			int(sequence_id), int(layer_idx), active_tokens
		)

	def update_layer_decode_new_token(
		self,
		k_tensor: torch.Tensor,
		v_tensor: Optional[torch.Tensor],
		sequence_lengths: torch.Tensor,
		layer_idx: int,
		batch_slice: Optional[tuple] = None,
		slot_indices: Optional[torch.Tensor] = None,
	) -> None:
		"""Append decode KV using raw token positions.

		``sequence_lengths`` keeps the old manager's argument name, but for this
		wrapper it is interpreted as raw decode write positions. The backing
		manager receives window-local storage positions.
		"""

		batch_size = int(k_tensor.shape[0])
		resolved_slots = self._resolve_slot_indices(
			batch_size=batch_size,
			batch_slice=batch_slice,
			slot_indices=slot_indices,
		)
		raw_positions = self._resolve_raw_positions(
			batch_size=batch_size,
			batch_slice=batch_slice,
			sequence_lengths=sequence_lengths,
		)
		storage_positions = self._prepare_storage_positions(
			raw_positions=raw_positions,
			slot_indices=resolved_slots,
			like=sequence_lengths,
		)
		self._rebuild_last_page_table_if_available()
		return self._base.update_layer_decode_new_token(
			k_tensor=k_tensor,
			v_tensor=v_tensor,
			sequence_lengths=storage_positions,
			layer_idx=int(layer_idx),
			batch_slice=None,
			slot_indices=resolved_slots,
		)

	def map_raw_lengths_to_window_local_lengths(
		self,
		sequence_ids: Sequence[int],
		raw_lengths: Sequence[int] | torch.Tensor,
		*,
		device: Optional[torch.device | str] = None,
		dtype: torch.dtype = torch.int32,
	) -> torch.Tensor:
		"""Return page-level SWA lengths aligned with ``sequence_ids``.

		This is the value that attention should consume as cache length when it
		uses the page table owned by this SWA manager.
		"""

		sequence_ids = [int(seq_id) for seq_id in sequence_ids]
		raw_values = _as_int_list(raw_lengths)
		if len(sequence_ids) != len(raw_values):
			raise ValueError(
				"map_raw_lengths_to_window_local_lengths: sequence_ids and "
				"raw_lengths must have the same length"
			)
		local_lengths: list[int] = []
		for seq_id, raw_length in zip(sequence_ids, raw_values):
			state = self._states.get(seq_id)
			if state is None or not state.has_tokens:
				window = self._compute_window_for_raw_end(raw_length)
				local_lengths.append(window.active_tokens)
				continue
			window_start_token = state.window_start_page * self.page_size_tokens
			local_lengths.append(max(0, int(raw_length) - window_start_token))
		output_device = device
		if output_device is None and isinstance(raw_lengths, torch.Tensor):
			output_device = raw_lengths.device
		return torch.as_tensor(
			local_lengths,
			dtype=dtype,
			device=output_device if output_device is not None else self.device,
		)

	def window_start_page(self, sequence_id: int) -> int:
		state = self._states.get(int(sequence_id))
		if state is None:
			raise KeyError(
				f"window_start_page: unknown sequence id {sequence_id}"
			)
		return state.window_start_page

	def window_start_pages(self, sequence_ids: Sequence[int]) -> list[int]:
		return [self.window_start_page(seq_id) for seq_id in sequence_ids]

	def _compute_window_for_raw_end(self, raw_end_tokens: int) -> _WindowForRawEnd:
		raw_end_tokens = int(raw_end_tokens)
		if raw_end_tokens <= 0:
			return _WindowForRawEnd(0, 0, 0)
		first_needed_token = max(0, raw_end_tokens - self.window_size_tokens)
		window_start_page = first_needed_token // self.page_size_tokens
		window_start_token = window_start_page * self.page_size_tokens
		active_tokens = raw_end_tokens - window_start_token
		required_pages = _ceil_div(active_tokens, self.page_size_tokens)
		if required_pages > self.window_pages + 1:
			raise RuntimeError(
				f"SWA active pages {required_pages} exceed window_pages + 1 "
				f"({self.window_pages + 1})"
			)
		return _WindowForRawEnd(
			window_start_page=window_start_page,
			active_tokens=active_tokens,
			required_pages=required_pages,
		)

	def _update_window_for_raw_end(
		self, sequence_id: int, raw_end_tokens: int
	) -> tuple[int, int]:
		window = self._compute_window_for_raw_end(raw_end_tokens)
		state = self._states.setdefault(sequence_id, _SWASequenceState())
		if state.has_tokens and window.window_start_page < state.window_start_page:
			raise ValueError(
				"SWA GPU manager does not support writing tokens older than "
				"the current page-level window"
			)
		if state.has_tokens and window.window_start_page > state.window_start_page:
			pages_to_release = window.window_start_page - state.window_start_page
			self._base.release_sequence_prefix_pages(
				sequence_id, pages_to_release
			)

		state.window_start_page = window.window_start_page
		added_pages = self._ensure_active_capacity(
			sequence_id, window.active_tokens
		)
		if raw_end_tokens > 0:
			state.max_seen_raw_pos = max(
				state.max_seen_raw_pos, raw_end_tokens - 1
			)
			state.has_tokens = True
		return window.active_tokens, added_pages

	def _ensure_active_capacity(
		self, sequence_id: int, active_tokens: int
	) -> int:
		if active_tokens <= 0:
			return 0
		required_pages = _ceil_div(active_tokens, self.page_size_tokens)
		if required_pages > self.window_pages + 1:
			raise ValueError(
				f"sequence {sequence_id} requires {required_pages} active "
				f"pages, exceeding window_pages + 1 ({self.window_pages + 1})"
			)
		state = self._base._sequences.get(sequence_id)
		current_pages = int(state.pages.numel()) if state is not None else 0
		missing_pages = max(0, required_pages - current_pages)
		if missing_pages:
			if state is None:
				self._base.allocate_pages(
					sequence_id, missing_pages * self.page_size_tokens
				)
			else:
				self._base.grow_sequence_pages(sequence_id, missing_pages)
		return missing_pages

	def _resolve_slot_indices(
		self,
		*,
		batch_size: int,
		batch_slice: Optional[tuple],
		slot_indices: Optional[torch.Tensor],
	) -> torch.Tensor:
		page_table = self._base._gpu_page_table_manager.gpu_table
		if page_table is None:
			raise RuntimeError(
				"SWAGPUPagedKVCacheManager: GPU page table is not initialized"
			)
		if slot_indices is None:
			slot_indices = self._base._gpu_page_table_manager._slot_index_tensor
			if slot_indices is None:
				raise RuntimeError(
					"SWAGPUPagedKVCacheManager: slot indices are unavailable"
				)
		else:
			slot_indices = slot_indices.to(
				device=page_table.device, dtype=torch.int32
			)
		if batch_slice is not None and slot_indices.shape[0] != batch_size:
			start_idx, end_idx = batch_slice
			slot_indices = slot_indices[start_idx:end_idx]
		if slot_indices.shape[0] != batch_size:
			raise ValueError(
				"SWAGPUPagedKVCacheManager: slot_indices must align with "
				f"batch size, got {slot_indices.shape[0]} vs {batch_size}"
			)
		return slot_indices.contiguous()

	def _resolve_raw_positions(
		self,
		*,
		batch_size: int,
		batch_slice: Optional[tuple],
		sequence_lengths: torch.Tensor,
	) -> torch.Tensor:
		if not isinstance(sequence_lengths, torch.Tensor):
			raise TypeError("sequence_lengths must be a torch.Tensor")
		raw_positions = sequence_lengths
		if batch_slice is not None and raw_positions.shape[0] != batch_size:
			start_idx, end_idx = batch_slice
			raw_positions = raw_positions[start_idx:end_idx]
		if raw_positions.shape[0] != batch_size:
			raise ValueError(
				"SWAGPUPagedKVCacheManager: sequence_lengths must align with "
				f"batch size, got {raw_positions.shape[0]} vs {batch_size}"
			)
		return raw_positions.contiguous()

	def _prepare_storage_positions(
		self,
		*,
		raw_positions: torch.Tensor,
		slot_indices: torch.Tensor,
		like: torch.Tensor,
	) -> torch.Tensor:
		slot_values = _as_int_list(slot_indices)
		raw_values = _as_int_list(raw_positions)
		slot_order = self._base._gpu_page_table_manager.slot_to_seq_id
		storage_positions: list[int] = []
		for slot, raw_pos in zip(slot_values, raw_values):
			if slot < 0:
				storage_positions.append(0)
				continue
			if slot >= len(slot_order):
				raise IndexError(
					f"slot index {slot} exceeds active slot order length "
					f"{len(slot_order)}"
				)
			sequence_id = int(slot_order[slot])
			active_tokens, _ = self._update_window_for_raw_end(
				sequence_id, int(raw_pos) + 1
			)
			storage_positions.append(active_tokens - 1)
		return torch.as_tensor(
			storage_positions,
			dtype=like.dtype,
			device=like.device,
		)

	def _rebuild_last_page_table_if_available(self) -> None:
		order = self._last_page_table_order
		if order is None:
			order = list(self._base._gpu_page_table_manager.slot_to_seq_id)
		if order:
			self._last_page_table_order = [int(seq_id) for seq_id in order]
			self._base.rebuild_page_table(self._last_page_table_order)


__all__ = ["SWAGPUPagedKVCacheManager"]
