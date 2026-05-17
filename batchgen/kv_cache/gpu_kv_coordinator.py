"""Lightweight registry for GPU-side heterogeneous KV managers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from batchgen.kv_cache.coordinator_utils import GPUKVComponent
logger = logging.getLogger(__name__)


class GPUKVCoordinator:
	"""Component-aware facade for GPU-side heterogeneous KV managers."""

	def __init__(self) -> None:
		self._components: Dict[str, GPUKVComponent] = {}

	def register_component(
		self,
		component: GPUKVComponent | str,
		manager: Any = None,
		**kwargs: Any,
	) -> GPUKVComponent:
		if isinstance(component, GPUKVComponent):
			if manager is not None or kwargs:
				raise ValueError(
					"Pass either a GPUKVComponent or name/manager/kwargs, not both"
				)
			item = component
		else:
			if kwargs:
				raise ValueError(
					"GPUKVCoordinator does not accept component metadata; "
					"configure the backing GPUPagedKVCacheManager instead"
				)
			item = GPUKVComponent(name=component, manager=manager, **kwargs)
		if item.name in self._components:
			raise ValueError(f"GPU KV component already registered: {item.name}")
		self._components[item.name] = item
		setattr(self, item.name, item.manager)
		return item

	@property
	def component_names(self) -> list[str]:
		return list(self._components.keys())

	def components(self) -> Iterator[GPUKVComponent]:
		return iter(self._components.values())

	def get_component(self, name: str) -> GPUKVComponent:
		try:
			return self._components[name]
		except KeyError as exc:
			raise KeyError(f"Unknown GPU KV component: {name}") from exc

	def get_manager(self, name: str) -> Any:
		return self.get_component(name).manager

	def call_all(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
		"""Call the same method on every backing manager."""

		results: dict[str, Any] = {}
		for component in self.components():
			method = getattr(component.manager, method_name)
			results[component.name] = method(*args, **kwargs)
		return results

	def _component_for_op(self, component_name: str, context: str) -> GPUKVComponent:
		if component_name is None:
			raise KeyError(f"{context}: component_name is required")
		try:
			return self.get_component(component_name)
		except KeyError as exc:
			raise KeyError(
				f"{context}: unknown GPU KV component {component_name!r}"
			) from exc

	def initialize(self) -> dict[str, Any]:
		return self.call_all("initialize")

	def destroy(self, *, empty_cuda_cache: bool = False) -> dict[str, Any]:
		results: dict[str, Any] = {}
		for component in reversed(list(self.components())):
			results[component.name] = component.manager.destroy(
				empty_cuda_cache=empty_cuda_cache
			)
		return results

	@property
	def is_initialized(self) -> bool:
		return all(
			bool(getattr(component.manager, "is_initialized", False))
			for component in self.components()
		)

	def allocate_pages(
		self,
		sequence_id: int,
		num_tokens: int,
		*,
		component_name: Optional[str] = None,
	):
		if component_name is not None:
			component = self._component_for_op(component_name, "allocate_pages")
			return component.manager.allocate_pages(
				int(sequence_id), int(num_tokens)
			)

		return self.allocate_pages_for_sequences([sequence_id], [num_tokens])

	def allocate_pages_for_sequences(
		self,
		sequence_ids: Sequence[int],
		num_tokens: Sequence[int],
		*,
		component_name: Optional[str] = None,
	):
		sequence_ids = [int(seq_id) for seq_id in sequence_ids]
		num_tokens = [int(tokens) for tokens in num_tokens]
		if component_name is not None:
			component = self._component_for_op(
				component_name, "allocate_pages_for_sequences"
			)
			return component.manager.allocate_pages_for_sequences(
				sequence_ids, num_tokens
			)

		results: dict[str, Any] = {}
		allocated: list[tuple[GPUKVComponent, Any]] = []
		try:
			for component in self.components():
				result = component.manager.allocate_pages_for_sequences(
					sequence_ids, num_tokens
				)
				results[component.name] = result
				allocated.append((component, result))
		except Exception:
			for component, allocations in reversed(allocated):
				try:
					_rollback_gpu_allocations(component.manager, allocations)
				except Exception:
					logger.exception(
						"Failed to rollback GPU KV allocation for %s on %s",
						sequence_ids[:10],
						component.name,
					)
			raise
		return results

	def grow_sequence_pages(
		self,
		sequence_id: int,
		num_pages: int,
		*,
		component_name: Optional[str] = None,
	):
		if component_name is not None:
			component = self._component_for_op(component_name, "grow_sequence_pages")
			return component.manager.grow_sequence_pages(int(sequence_id), int(num_pages))
		return self.grow_pages_for_sequences([sequence_id], [num_pages])

	def grow_pages_for_sequences(
		self,
		sequence_ids: Sequence[int],
		num_pages: Sequence[int],
		*,
		component_name: Optional[str] = None,
	):
		sequence_ids = [int(seq_id) for seq_id in sequence_ids]
		num_pages = [int(count) for count in num_pages]
		if component_name is not None:
			component = self._component_for_op(
				component_name, "grow_pages_for_sequences"
			)
			return component.manager.grow_pages_for_sequences(sequence_ids, num_pages)
		return self.call_all("grow_pages_for_sequences", sequence_ids, num_pages)

	def extend_pages_for_sequence(
		self,
		sequence_id: int,
		new_total_tokens: int,
		*,
		component_name: Optional[str] = None,
	):
		if component_name is not None:
			component = self._component_for_op(
				component_name, "extend_pages_for_sequence"
			)
			return component.manager.extend_pages_for_sequence(
				int(sequence_id), int(new_total_tokens)
			)
		results: dict[str, Any] = {}
		for component in self.components():
			results[component.name] = component.manager.extend_pages_for_sequence(
				int(sequence_id), int(new_total_tokens)
			)
		return results

	def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> dict[str, Any]:
		return self.call_all(
			"free_pages_for_sequences", [int(seq_id) for seq_id in sequence_ids]
		)

	def rebuild_page_table(self, sequence_ids: Sequence[int]):
		return self.call_all(
			"rebuild_page_table", [int(seq_id) for seq_id in sequence_ids]
		)

	def clear_page_table(self) -> dict[str, Any]:
		return self.call_all("clear_page_table")

	def get_page_table_version(self, *, component_name: str) -> int:
		component = self._component_for_op(component_name, "get_page_table_version")
		return component.manager.get_page_table_version()

	def get_cuda_graph_page_table(self, *, component_name: str):
		component = self._component_for_op(
			component_name, "get_cuda_graph_page_table"
		)
		return component.manager.get_cuda_graph_page_table()

	def get_cuda_graph_page_table_storage(self, *, component_name: str):
		component = self._component_for_op(
			component_name, "get_cuda_graph_page_table_storage"
		)
		return component.manager.get_cuda_graph_page_table_storage()

	def ensure_cuda_graph_page_table(
		self,
		sequence_ids: Sequence[int],
		*,
		component_name: str,
	):
		component = self._component_for_op(
			component_name, "ensure_cuda_graph_page_table"
		)
		return component.manager.ensure_cuda_graph_page_table(
			[int(seq_id) for seq_id in sequence_ids]
		)

	def get_cuda_graph_page_table_state(self, *, component_name: str):
		component = self._component_for_op(
			component_name, "get_cuda_graph_page_table_state"
		)
		return component.manager.get_cuda_graph_page_table_state()

	def get_stats(self):
		return self.call_all("get_stats")

	def get_stats_by_component(self) -> dict[str, Any]:
		return self.get_stats()

	def get_kv_tensors(self, *, component_name: str):
		component = self._component_for_op(component_name, "get_kv_tensors")
		return component.manager.get_kv_tensors()

	def get_layer_kv_with_page_table(
		self,
		layer_idx: int,
		*,
		component_name: str,
	):
		component = self._component_for_op(
			component_name, "get_layer_kv_with_page_table"
		)
		return component.manager.get_layer_kv_with_page_table(int(layer_idx))

	def update_layer_decode_new_token(
		self,
		k_tensor,
		v_tensor,
		sequence_lengths,
		layer_idx: int,
		batch_slice: Optional[tuple] = None,
		slot_indices=None,
		*,
		component_name: str,
	) -> None:
		component = self._component_for_op(
			component_name, "update_layer_decode_new_token"
		)
		return component.manager.update_layer_decode_new_token(
			k_tensor=k_tensor,
			v_tensor=v_tensor,
			sequence_lengths=sequence_lengths,
			layer_idx=int(layer_idx),
			batch_slice=batch_slice,
			slot_indices=slot_indices,
		)

	def get_context_kv_page_ptrs(
		self,
		sequence_id: int,
		layer_idx: int,
		context_length: int,
		*,
		component_name: str,
	):
		component = self._component_for_op(
			component_name, "get_context_kv_page_ptrs"
		)
		return component.manager.get_context_kv_page_ptrs(
			int(sequence_id),
			int(layer_idx),
			int(context_length),
		)

	def get_sequence_layer_page_pointers(
		self,
		sequence_id: int,
		layer_idx: int,
		*,
		component_name: str,
	):
		component = self._component_for_op(
			component_name, "get_sequence_layer_page_pointers"
		)
		return component.manager.get_sequence_layer_page_pointers(
			int(sequence_id), int(layer_idx)
		)

	def export_layer_page_pointer_table(self, *, component_name: str):
		component = self._component_for_op(
			component_name, "export_layer_page_pointer_table"
		)
		return component.manager.export_layer_page_pointer_table()

	def export_active_sequence_page_counts(self, *, component_name: str):
		component = self._component_for_op(
			component_name, "export_active_sequence_page_counts"
		)
		return component.manager.export_active_sequence_page_counts()

	def get_padded_3d_page_pointers(self, *, component_name: str):
		component = self._component_for_op(
			component_name, "get_padded_3d_page_pointers"
		)
		return component.manager.get_padded_3d_page_pointers()

	def copy_kv_to_tensor(self, sequence_id: int, *, component_name: str):
		component = self._component_for_op(component_name, "copy_kv_to_tensor")
		return component.manager.copy_kv_to_tensor(int(sequence_id))

	def copy_tensor_to_kv(
		self,
		sequence_id: int,
		k_tensor,
		*,
		component_name: str,
	) -> None:
		component = self._component_for_op(component_name, "copy_tensor_to_kv")
		return component.manager.copy_tensor_to_kv(int(sequence_id), k_tensor)


def _rollback_gpu_allocations(manager: Any, allocations: Any) -> None:
	if not isinstance(allocations, Mapping):
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
		if tail != list(pages):
			raise RuntimeError(
				f"Cannot rollback GPU KV allocation for seq {seq_id}: "
				f"tail={tail}, allocated={pages}"
			)
		reclaimed.append(state.pages[-count:].clone())
		if state.pages.numel() == count:
			del manager._sequences[seq_id]
		else:
			state.pages = state.pages[:-count].clone()
	if reclaimed:
		import torch

		manager._free_pages.push(torch.cat(reclaimed, dim=0))
		manager._clear_active_page_pointer_tables()
