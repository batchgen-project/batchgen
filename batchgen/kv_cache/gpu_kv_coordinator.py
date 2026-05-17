"""Lightweight registry for GPU-side heterogeneous KV managers."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

from batchgen.kv_cache.coordinator_utils import resolve_from_layer_mapping


LayerMapping = Mapping[int, int] | Sequence[int]
TokenCapacityFn = Callable[[int], int]
logger = logging.getLogger(__name__)


@dataclass
class GPUKVComponent:
	"""One named GPU KV component backed by one paged manager."""

	name: str
	manager: Any
	logical_to_physical_layer: Optional[LayerMapping] = None
	token_capacity_scale: float = 1.0
	token_capacity_fn: Optional[TokenCapacityFn] = None

	def __post_init__(self) -> None:
		if not self.name:
			raise ValueError("GPUKVComponent.name must be non-empty")
		if self.manager is None:
			raise ValueError(f"GPUKVComponent({self.name}): manager must be set")
		if self.token_capacity_scale <= 0:
			raise ValueError(
				f"GPUKVComponent({self.name}): token_capacity_scale must be > 0"
			)

	def token_capacity(self, num_tokens: int) -> int:
		if num_tokens <= 0:
			raise ValueError(
				f"GPUKVComponent({self.name}): num_tokens must be > 0, got {num_tokens}"
			)
		if self.token_capacity_fn is not None:
			capacity = int(self.token_capacity_fn(int(num_tokens)))
		else:
			capacity = int(math.ceil(int(num_tokens) * self.token_capacity_scale))
		if capacity <= 0:
			raise ValueError(
				f"GPUKVComponent({self.name}): token capacity must be > 0, got {capacity}"
			)
		return capacity

	def resolve_physical_layer(self, logical_layer_id: int) -> int:
		if logical_layer_id < 0:
			raise IndexError(
				f"GPUKVComponent({self.name}): logical layer id must be >= 0"
			)
		if self.logical_to_physical_layer is None:
			return int(logical_layer_id)
		return resolve_from_layer_mapping(
			"GPU KV", self.name, self.logical_to_physical_layer, logical_layer_id
		)

	def map_token_counts(self, token_counts: Sequence[int]) -> list[int]:
		return [self.token_capacity(int(count)) for count in token_counts]


class GPUKVCoordinator:
	"""Component-aware facade for GPU-side heterogeneous KV managers."""

	def __init__(self) -> None:
		self._components: Dict[str, GPUKVComponent] = {}
		self.default_component_name: Optional[str] = None

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
			item = GPUKVComponent(name=component, manager=manager, **kwargs)
		if item.name in self._components:
			raise ValueError(f"GPU KV component already registered: {item.name}")
		self._components[item.name] = item
		if self.default_component_name is None:
			self.default_component_name = item.name
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

	def set_default_component(self, component_name: Optional[str]) -> str:
		if component_name is None:
			if self.default_component_name is None:
				raise KeyError("GPU KV coordinator has no default component")
			return self.default_component_name
		self.get_component(component_name)
		self.default_component_name = component_name
		return component_name

	def resolve_physical_layer(self, component_name: str, logical_layer_id: int) -> int:
		return self.get_component(component_name).resolve_physical_layer(logical_layer_id)

	def map_token_counts(self, component_name: str, token_counts: Sequence[int]) -> list[int]:
		return self.get_component(component_name).map_token_counts(token_counts)

	def call_all(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
		"""Call the same method on every backing manager."""

		results: dict[str, Any] = {}
		for component in self.components():
			method = getattr(component.manager, method_name)
			results[component.name] = method(*args, **kwargs)
		return results

	def _component_or_default(
		self, component_name: Optional[str], context: str
	) -> GPUKVComponent:
		name = self.default_component_name if component_name is None else component_name
		if name is None:
			raise KeyError(f"{context}: GPU KV coordinator has no components")
		try:
			return self.get_component(name)
		except KeyError as exc:
			raise KeyError(f"{context}: unknown GPU KV component {name!r}") from exc

	def _manager_for_layer(
		self, component_name: Optional[str], logical_layer_id: int, context: str
	) -> tuple[Any, int]:
		component = self._component_or_default(component_name, context)
		return component.manager, component.resolve_physical_layer(int(logical_layer_id))

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

	@property
	def config(self):
		return self._component_or_default(None, "config").manager.config

	@property
	def device(self):
		return self._component_or_default(None, "device").manager.device

	@property
	def _gpu_page_table_manager(self):
		return self._component_or_default(
			None, "_gpu_page_table_manager"
		).manager._gpu_page_table_manager

	@property
	def _sequences(self):
		return self._component_or_default(None, "_sequences").manager._sequences

	def allocate_pages(
		self,
		sequence_id: int,
		num_tokens: int,
		*,
		component_name: Optional[str] = None,
	):
		if component_name is not None:
			component = self._component_or_default(component_name, "allocate_pages")
			return component.manager.allocate_pages(
				int(sequence_id), component.token_capacity(int(num_tokens))
			)

		results = self.allocate_pages_for_sequences([sequence_id], [num_tokens])
		return results.get(self.set_default_component(None), {}).get(int(sequence_id), [])

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
			component = self._component_or_default(
				component_name, "allocate_pages_for_sequences"
			)
			return component.manager.allocate_pages_for_sequences(
				sequence_ids, component.map_token_counts(num_tokens)
			)

		results: dict[str, Any] = {}
		allocated: list[tuple[GPUKVComponent, Any]] = []
		try:
			for component in self.components():
				result = component.manager.allocate_pages_for_sequences(
					sequence_ids, component.map_token_counts(num_tokens)
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
			component = self._component_or_default(component_name, "grow_sequence_pages")
			return component.manager.grow_sequence_pages(int(sequence_id), int(num_pages))
		results = self.grow_pages_for_sequences([sequence_id], [num_pages])
		return results.get(self.set_default_component(None), {}).get(int(sequence_id), [])

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
			component = self._component_or_default(
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
			component = self._component_or_default(
				component_name, "extend_pages_for_sequence"
			)
			return component.manager.extend_pages_for_sequence(
				int(sequence_id), component.token_capacity(int(new_total_tokens))
			)
		results: dict[str, Any] = {}
		for component in self.components():
			results[component.name] = component.manager.extend_pages_for_sequence(
				int(sequence_id), component.token_capacity(int(new_total_tokens))
			)
		return results.get(self.set_default_component(None), 0)

	def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> dict[str, Any]:
		return self.call_all(
			"free_pages_for_sequences", [int(seq_id) for seq_id in sequence_ids]
		)

	def rebuild_page_table(self, sequence_ids: Sequence[int]):
		results = self.call_all(
			"rebuild_page_table", [int(seq_id) for seq_id in sequence_ids]
		)
		return results[self.set_default_component(None)]

	def clear_page_table(self) -> dict[str, Any]:
		return self.call_all("clear_page_table")

	def get_page_table_version(self) -> int:
		component = self._component_or_default(None, "get_page_table_version")
		return component.manager.get_page_table_version()

	def get_cuda_graph_page_table(
		self, *, component_name: Optional[str] = None
	):
		component = self._component_or_default(
			component_name, "get_cuda_graph_page_table"
		)
		return component.manager.get_cuda_graph_page_table()

	def get_cuda_graph_page_table_storage(
		self, *, component_name: Optional[str] = None
	):
		component = self._component_or_default(
			component_name, "get_cuda_graph_page_table_storage"
		)
		return component.manager.get_cuda_graph_page_table_storage()

	def ensure_cuda_graph_page_table(
		self,
		sequence_ids: Sequence[int],
		*,
		component_name: Optional[str] = None,
	):
		component = self._component_or_default(
			component_name, "ensure_cuda_graph_page_table"
		)
		return component.manager.ensure_cuda_graph_page_table(
			[int(seq_id) for seq_id in sequence_ids]
		)

	def get_cuda_graph_page_table_state(
		self, *, component_name: Optional[str] = None
	):
		component = self._component_or_default(
			component_name, "get_cuda_graph_page_table_state"
		)
		return component.manager.get_cuda_graph_page_table_state()

	def get_stats(self):
		return self._component_or_default(None, "get_stats").manager.get_stats()

	def get_stats_by_component(self) -> dict[str, Any]:
		return self.call_all("get_stats")

	def get_kv_tensors(self, *, component_name: Optional[str] = None):
		component = self._component_or_default(component_name, "get_kv_tensors")
		return component.manager.get_kv_tensors()

	def get_layer_kv_with_page_table(
		self,
		layer_idx: int,
		*,
		component_name: Optional[str] = None,
	):
		manager, physical_layer_id = self._manager_for_layer(
			component_name, layer_idx, "get_layer_kv_with_page_table"
		)
		return manager.get_layer_kv_with_page_table(physical_layer_id)

	def update_layer_decode_new_token(
		self,
		k_tensor,
		v_tensor,
		sequence_lengths,
		layer_idx: int,
		batch_slice: Optional[tuple] = None,
		slot_indices=None,
		*,
		component_name: Optional[str] = None,
	) -> None:
		manager, physical_layer_id = self._manager_for_layer(
			component_name, layer_idx, "update_layer_decode_new_token"
		)
		return manager.update_layer_decode_new_token(
			k_tensor=k_tensor,
			v_tensor=v_tensor,
			sequence_lengths=sequence_lengths,
			layer_idx=physical_layer_id,
			batch_slice=batch_slice,
			slot_indices=slot_indices,
		)

	def get_context_kv_page_ptrs(
		self,
		sequence_id: int,
		layer_idx: int,
		context_length: int,
		*,
		component_name: Optional[str] = None,
	):
		component = self._component_or_default(
			component_name, "get_context_kv_page_ptrs"
		)
		return component.manager.get_context_kv_page_ptrs(
			int(sequence_id),
			component.resolve_physical_layer(int(layer_idx)),
			component.token_capacity(int(context_length)),
		)

	def get_sequence_layer_page_pointers(
		self,
		sequence_id: int,
		layer_idx: int,
		*,
		component_name: Optional[str] = None,
	):
		manager, physical_layer_id = self._manager_for_layer(
			component_name, layer_idx, "get_sequence_layer_page_pointers"
		)
		return manager.get_sequence_layer_page_pointers(
			int(sequence_id), physical_layer_id
		)

	def export_layer_page_pointer_table(
		self, *, component_name: Optional[str] = None
	):
		component = self._component_or_default(
			component_name, "export_layer_page_pointer_table"
		)
		return component.manager.export_layer_page_pointer_table()

	def export_active_sequence_page_counts(
		self, *, component_name: Optional[str] = None
	):
		component = self._component_or_default(
			component_name, "export_active_sequence_page_counts"
		)
		return component.manager.export_active_sequence_page_counts()

	def get_padded_3d_page_pointers(
		self, *, component_name: Optional[str] = None
	):
		component = self._component_or_default(
			component_name, "get_padded_3d_page_pointers"
		)
		return component.manager.get_padded_3d_page_pointers()

	def copy_kv_to_tensor(
		self, sequence_id: int, *, component_name: Optional[str] = None
	):
		component = self._component_or_default(component_name, "copy_kv_to_tensor")
		return component.manager.copy_kv_to_tensor(int(sequence_id))

	def copy_tensor_to_kv(
		self,
		sequence_id: int,
		k_tensor,
		*,
		component_name: Optional[str] = None,
	) -> None:
		component = self._component_or_default(component_name, "copy_tensor_to_kv")
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
