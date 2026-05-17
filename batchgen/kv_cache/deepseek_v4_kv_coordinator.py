"""DeepSeek-V4 KV coordinators.

DeepSeek-V4 has multiple logical KV components whose layer sets and token
rates differ. The generic Host/GPU coordinators keep the component registry and
layer mapping metadata; the DSV4 coordinators below add the runtime facade that
the rest of BatchGen expects: allocate/free lifecycle APIs plus component-aware
offload/load/write helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator
from batchgen.kv_cache.host_kv_coordinator import (
	AsyncKVTask,
	HostKVCoordinator,
	wait_kv_tasks,
)


PRIMARY_MLA = "primary_mla"
SWA = "swa"
COMPRESSOR_C4 = "compressor_c4"
COMPRESSOR_C128 = "compressor_c128"
INDEXER_C4 = "indexer_c4"

DEEPSEEK_V4_COMPONENT_ORDER = (
	PRIMARY_MLA,
	SWA,
	COMPRESSOR_C4,
	COMPRESSOR_C128,
	INDEXER_C4,
)

_SUPPORTED_COMPRESS_RATIOS = {0, 4, 128}

TokenCapacityFn = Callable[[int], int]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepSeekV4KVLayout:
	"""Layer and token routing metadata for DeepSeek-V4 KV components."""

	compression_ratios: tuple[int, ...]
	c4_layer_map: tuple[int, ...]
	c128_layer_map: tuple[int, ...]
	sliding_window: Optional[int] = None

	@classmethod
	def from_compression_ratios(
		cls,
		compression_ratios: Sequence[int],
		*,
		num_layers: Optional[int] = None,
		sliding_window: Optional[int] = None,
	) -> "DeepSeekV4KVLayout":
		ratios = [int(ratio) for ratio in compression_ratios]
		if num_layers is None:
			num_layers = len(ratios)
		else:
			num_layers = int(num_layers)
		if num_layers <= 0:
			raise ValueError("DeepSeekV4KVLayout.num_layers must be > 0")
		if len(ratios) > num_layers:
			raise ValueError(
				"compression_ratios has more entries than num_layers: "
				f"{len(ratios)} > {num_layers}"
			)
		if len(ratios) < num_layers:
			ratios.extend([0] * (num_layers - len(ratios)))
		invalid = sorted(set(ratios) - _SUPPORTED_COMPRESS_RATIOS)
		if invalid:
			raise ValueError(
				"Unsupported DeepSeek-V4 compression ratios: "
				f"{invalid}; expected only 0, 4, or 128"
			)
		if sliding_window is not None and int(sliding_window) <= 0:
			raise ValueError("sliding_window must be > 0 when set")
		ratios_tuple = tuple(ratios)
		return cls(
			compression_ratios=ratios_tuple,
			c4_layer_map=_compact_layer_map(ratios_tuple, 4),
			c128_layer_map=_compact_layer_map(ratios_tuple, 128),
			sliding_window=None if sliding_window is None else int(sliding_window),
		)

	@property
	def num_layers(self) -> int:
		return len(self.compression_ratios)

	@property
	def num_c4_layers(self) -> int:
		return _count_physical_layers(self.c4_layer_map)

	@property
	def num_c128_layers(self) -> int:
		return _count_physical_layers(self.c128_layer_map)

	def layer_mapping(self, component_name: str) -> Optional[tuple[int, ...]]:
		if component_name == COMPRESSOR_C4:
			return self.c4_layer_map
		if component_name == COMPRESSOR_C128:
			return self.c128_layer_map
		if component_name == INDEXER_C4:
			return self.c4_layer_map
		if component_name in (PRIMARY_MLA, SWA):
			return None
		raise KeyError(f"Unknown DeepSeek-V4 KV component: {component_name}")

	def physical_layer_count(self, component_name: str) -> int:
		if component_name in (PRIMARY_MLA, SWA):
			return self.num_layers
		mapping = self.layer_mapping(component_name)
		if mapping is None:
			return self.num_layers
		return _count_physical_layers(mapping)

	def token_capacity_fn(self, component_name: str) -> TokenCapacityFn:
		if component_name == COMPRESSOR_C4 or component_name == INDEXER_C4:
			return _ceil_div_fn(4)
		if component_name == COMPRESSOR_C128:
			return _ceil_div_fn(128)
		if component_name == SWA and self.sliding_window is not None:
			window = self.sliding_window
			return lambda num_tokens: min(_check_positive_tokens(num_tokens), window)
		if component_name == PRIMARY_MLA or component_name == SWA:
			return _identity_token_capacity
		raise KeyError(f"Unknown DeepSeek-V4 KV component: {component_name}")

	def token_capacity(self, component_name: str, num_tokens: int) -> int:
		return self.token_capacity_fn(component_name)(num_tokens)


class DeepSeekV4HostKVCoordinator(HostKVCoordinator):
	"""Runtime facade for DeepSeek-V4 host KV worker views.

	Lifecycle/page APIs operate on every registered component by default, because
	a sequence must exist in every DSV4 KV pool. Data movement APIs operate on one
	component at a time and accept ``component_name=...``. When omitted, they use
	the default component so existing single-KV call sites keep working.
	"""

	def __init__(
		self,
		*,
		compression_ratios: Sequence[int],
		primary_mla: Any = None,
		swa: Any = None,
		compressor_c4: Any = None,
		compressor_c128: Any = None,
		indexer_c4: Any = None,
		num_layers: Optional[int] = None,
		sliding_window: Optional[int] = None,
		default_component: Optional[str] = None,
	) -> None:
		super().__init__()
		self.layout = DeepSeekV4KVLayout.from_compression_ratios(
			compression_ratios,
			num_layers=num_layers,
			sliding_window=sliding_window,
		)

		if primary_mla is not None:
			self.register_component(
				PRIMARY_MLA,
				primary_mla,
				token_capacity_fn=self.layout.token_capacity_fn(PRIMARY_MLA),
			)
		self._register_optional_view(SWA, swa)
		self._register_optional_view(COMPRESSOR_C4, compressor_c4)
		self._register_optional_view(COMPRESSOR_C128, compressor_c128)
		self._register_optional_view(INDEXER_C4, indexer_c4)
		if not self.component_names:
			raise RuntimeError("DeepSeekV4HostKVCoordinator requires at least one view")
		self.default_component_name = self._resolve_default_component_name(
			default_component
		)

	def _register_optional_view(self, component_name: str, view: Any) -> None:
		if view is None:
			return
		self._ensure_component_has_layers(component_name)
		mapping = self.layout.layer_mapping(component_name)
		if _view_owns_layer_mapping(view):
			mapping = None
		self.register_component(
			component_name,
			view,
			logical_to_physical_layer=mapping,
			token_capacity_fn=self.layout.token_capacity_fn(component_name),
		)

	def _ensure_component_has_layers(self, component_name: str) -> None:
		if self.layout.physical_layer_count(component_name) <= 0:
			raise ValueError(
				f"DeepSeek-V4 component {component_name!r} has no physical layers"
			)

	def _resolve_default_component_name(
		self, component_name: Optional[str]
	) -> str:
		if component_name is not None:
			self.get_component(component_name)
			return component_name
		if SWA in self.component_names:
			return SWA
		if PRIMARY_MLA in self.component_names:
			return PRIMARY_MLA
		return self.component_names[0]

	def _component_or_default(self, component_name: Optional[str], context: str):
		name = self.default_component_name if component_name is None else component_name
		try:
			return self.get_component(name)
		except KeyError as exc:
			raise KeyError(f"{context}: unknown DSV4 host KV component {name!r}") from exc

	def _view_for_data_op(
		self, component_name: Optional[str], logical_layer_id: int, context: str
	) -> tuple[Any, int]:
		component = self._component_or_default(component_name, context)
		return component.view, component.view_layer_id(int(logical_layer_id))

	def _call_all_views(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
		results: dict[str, Any] = {}
		for component in self.components():
			method = getattr(component.view, method_name)
			results[component.name] = method(*args, **kwargs)
		return results

	# -- Lifecycle --

	def initialize(self, **kwargs: Any) -> dict[str, Any]:
		return self._call_all_views("initialize", **kwargs)

	def shutdown(self) -> dict[str, Any]:
		results: dict[str, Any] = {}
		for component in reversed(list(self.components())):
			results[component.name] = component.view.shutdown()
		return results

	# -- Sequence/page management --

	def register_sequences(self, sequence_ids) -> dict[str, Any]:
		sequence_ids = list(sequence_ids)
		results: dict[str, Any] = {}
		registered: list[Any] = []
		try:
			for component in self.components():
				results[component.name] = component.view.register_sequences(sequence_ids)
				registered.append(component)
		except Exception:
			for component in reversed(registered):
				try:
					component.view.unregister_sequences(sequence_ids)
				except Exception:
					logger.exception(
						"Failed to rollback DSV4 host KV registration for %s on %s",
						sequence_ids[:10],
						component.name,
					)
			raise
		return results

	def unregister_sequence(self, sequence_id: int) -> dict[str, Any]:
		return self._call_all_views("unregister_sequence", int(sequence_id))

	def unregister_sequences(self, sequence_ids) -> dict[str, Any]:
		return self._call_all_views("unregister_sequences", list(sequence_ids))

	def allocate_pages_for_sequences(
		self,
		seq_token_pairs,
		*,
		component_name: Optional[str] = None,
	):
		pairs = _normalize_seq_token_pairs(seq_token_pairs)
		if component_name is not None:
			component = self._component_or_default(
				component_name, "allocate_pages_for_sequences"
			)
			return component.view.allocate_pages_for_sequences(
				component.map_sequence_tokens(pairs)
			)

		sequence_ids = [seq_id for seq_id, _ in pairs]
		results: dict[str, Any] = {}
		allocated: list[Any] = []
		try:
			for component in self.components():
				results[component.name] = component.view.allocate_pages_for_sequences(
					component.map_sequence_tokens(pairs)
				)
				allocated.append(component)
		except Exception:
			for component in reversed(allocated):
				try:
					component.view.release_sequence_pages(sequence_ids)
				except Exception:
					logger.exception(
						"Failed to rollback DSV4 host KV allocation for %s on %s",
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
			return component.view.grow_sequence_pages(int(sequence_id), int(num_pages))
		return self._call_all_views(
			"grow_sequence_pages", int(sequence_id), int(num_pages)
		)

	def grow_pages_for_sequences(
		self,
		seq_page_pairs,
		*,
		component_name: Optional[str] = None,
	):
		pairs = _normalize_seq_token_pairs(seq_page_pairs)
		if component_name is not None:
			component = self._component_or_default(
				component_name, "grow_pages_for_sequences"
			)
			return component.view.grow_pages_for_sequences(pairs)
		return self._call_all_views("grow_pages_for_sequences", pairs)

	def release_sequence_pages(self, sequence_ids) -> dict[str, Any]:
		return self._call_all_views("release_sequence_pages", list(sequence_ids))

	def free_pages_for_sequences(self, sequence_ids) -> dict[str, Any]:
		return self.release_sequence_pages(sequence_ids)

	def build_page_table(self, sequence_ids):
		results = self._call_all_views("build_page_table", list(sequence_ids))
		return results[self.default_component_name]

	# -- Query --

	def get_stats(self):
		return self.get_view(self.default_component_name).get_stats()

	def get_stats_by_component(self) -> dict[str, Any]:
		return self._call_all_views("get_stats")

	def data_base_address(self, *, component_name: Optional[str] = None):
		component = self._component_or_default(component_name, "data_base_address")
		return component.view.data_base_address()

	def read_sequence_kv_to_cpu(
		self, sequence_id: int, *, component_name: Optional[str] = None
	):
		component = self._component_or_default(component_name, "read_sequence_kv_to_cpu")
		return component.view.read_sequence_kv_to_cpu(int(sequence_id))

	def write_sequence_kv_from_cpu(
		self,
		sequence_id: int,
		k_tensor,
		v_tensor=None,
		*,
		component_name: Optional[str] = None,
	):
		component = self._component_or_default(component_name, "write_sequence_kv_from_cpu")
		return component.view.write_sequence_kv_from_cpu(
			int(sequence_id), k_tensor, v_tensor
		)

	def k_page_ptr(
		self,
		layer_idx: int,
		page_idx: int,
		*,
		component_name: Optional[str] = None,
	):
		view, view_layer_id = self._view_for_data_op(
			component_name, layer_idx, "k_page_ptr"
		)
		return view.k_page_ptr(view_layer_id, int(page_idx))

	def v_page_ptr(
		self,
		layer_idx: int,
		page_idx: int,
		*,
		component_name: Optional[str] = None,
	):
		view, view_layer_id = self._view_for_data_op(
			component_name, layer_idx, "v_page_ptr"
		)
		return view.v_page_ptr(view_layer_id, int(page_idx))

	def get_sequence_layer_page_pointers(
		self,
		sequence_id: int,
		layer_idx: int,
		max_tokens=None,
		*,
		component_name: Optional[str] = None,
	):
		component = self._component_or_default(
			component_name, "get_sequence_layer_page_pointers"
		)
		return component.view.get_sequence_layer_page_pointers(
			int(sequence_id),
			component.view_layer_id(int(layer_idx)),
			None if max_tokens is None else component.token_capacity(int(max_tokens)),
		)

	# -- Component-aware host/device data movement --

	def async_offload_layer_kv_to_host(
		self,
		layer_idx: int,
		sequence_ids,
		k_tensor,
		v_tensor=None,
		sequence_lengths=None,
		*,
		component_name: Optional[str] = None,
	):
		if sequence_lengths is None:
			raise TypeError("async_offload_layer_kv_to_host requires sequence_lengths")
		view, view_layer_id = self._view_for_data_op(
			component_name, layer_idx, "async_offload_layer_kv_to_host"
		)
		return view.async_offload_layer_kv_to_host(
			view_layer_id, sequence_ids, k_tensor, v_tensor, sequence_lengths
		)

	def async_append_decode_kv_to_host(
		self,
		layer_idx: int,
		sequence_ids,
		k_tensor,
		v_tensor=None,
		sequence_lengths=None,
		*,
		component_name: Optional[str] = None,
	):
		if sequence_lengths is None:
			raise TypeError("async_append_decode_kv_to_host requires sequence_lengths")
		view, view_layer_id = self._view_for_data_op(
			component_name, layer_idx, "async_append_decode_kv_to_host"
		)
		return view.async_append_decode_kv_to_host(
			view_layer_id, sequence_ids, k_tensor, v_tensor, sequence_lengths
		)

	def async_append_decode_kv_to_host_batched_kernel(
		self,
		entries,
		sequence_ids,
		sequence_lengths,
		*,
		component_name: Optional[str] = None,
	):
		component = self._component_or_default(
			component_name, "async_append_decode_kv_to_host_batched_kernel"
		)
		mapped_entries = [
			(
				component.view_layer_id(int(entry[0])),
				entry[1],
				entry[2],
			)
			for entry in entries
		]
		return component.view.async_append_decode_kv_to_host_batched_kernel(
			mapped_entries, sequence_ids, sequence_lengths
		)

	def async_load_layer_kv_to_device(
		self,
		sequence_ids,
		k_device_ptrs,
		v_device_ptrs=None,
		*,
		component_name: Optional[str] = None,
	):
		component = self._component_or_default(
			component_name, "async_load_layer_kv_to_device"
		)
		return component.view.async_load_layer_kv_to_device(
			sequence_ids, k_device_ptrs, v_device_ptrs
		)

	def async_load_layer_paged_kv_to_device(
		self,
		sequence_ids,
		active_page_counts,
		k_device_ptrs,
		v_device_ptrs=None,
		*,
		component_name: Optional[str] = None,
	):
		component = self._component_or_default(
			component_name, "async_load_layer_paged_kv_to_device"
		)
		return component.view.async_load_layer_paged_kv_to_device(
			sequence_ids, active_page_counts, k_device_ptrs, v_device_ptrs
		)

	def async_load_components_paged_kv_to_device(
		self,
		component_loads: Mapping[str, Mapping[str, Any]],
		*,
		tensors: Any = None,
	) -> AsyncKVTask:
		tasks: dict[str, Any] = {}
		for component_name, kwargs in component_loads.items():
			try:
				tasks[component_name] = self.async_load_layer_paged_kv_to_device(
					component_name=component_name, **dict(kwargs)
				)
			except Exception:
				wait_kv_tasks(tasks, context="DSV4 host KV load")
				raise
		return AsyncKVTask(tasks=tasks, tensors=tensors)

	def async_offload_components_kv_to_host(
		self,
		component_offloads: Mapping[str, Mapping[str, Any]],
		*,
		tensors: Any = None,
	) -> AsyncKVTask:
		tasks: dict[str, Any] = {}
		for component_name, kwargs in component_offloads.items():
			try:
				tasks[component_name] = self.async_offload_layer_kv_to_host(
					component_name=component_name, **dict(kwargs)
				)
			except Exception:
				wait_kv_tasks(tasks, context="DSV4 host KV offload")
				raise
		return AsyncKVTask(tasks=tasks, tensors=tensors)


class DeepSeekV4GPUKVCoordinator(GPUKVCoordinator):
	"""Runtime facade for DeepSeek-V4 GPU paged KV managers."""

	def __init__(
		self,
		*,
		compression_ratios: Sequence[int],
		primary_mla: Any = None,
		swa: Any = None,
		compressor_c4: Any = None,
		compressor_c128: Any = None,
		indexer_c4: Any = None,
		num_layers: Optional[int] = None,
		sliding_window: Optional[int] = None,
		default_component: Optional[str] = None,
	) -> None:
		super().__init__()
		self.layout = DeepSeekV4KVLayout.from_compression_ratios(
			compression_ratios,
			num_layers=num_layers,
			sliding_window=sliding_window,
		)

		if primary_mla is not None:
			self.register_component(
				PRIMARY_MLA,
				primary_mla,
				token_capacity_fn=self.layout.token_capacity_fn(PRIMARY_MLA),
			)
		self._register_optional_manager(SWA, swa)
		self._register_optional_manager(COMPRESSOR_C4, compressor_c4)
		self._register_optional_manager(COMPRESSOR_C128, compressor_c128)
		self._register_optional_manager(INDEXER_C4, indexer_c4)
		if not self.component_names:
			raise RuntimeError("DeepSeekV4GPUKVCoordinator requires at least one manager")
		self.default_component_name = self._resolve_default_component_name(
			default_component
		)

	def _register_optional_manager(self, component_name: str, manager: Any) -> None:
		if manager is None:
			return
		self._ensure_component_has_layers(component_name)
		self.register_component(
			component_name,
			manager,
			logical_to_physical_layer=self.layout.layer_mapping(component_name),
			token_capacity_fn=self.layout.token_capacity_fn(component_name),
		)

	def _ensure_component_has_layers(self, component_name: str) -> None:
		if self.layout.physical_layer_count(component_name) <= 0:
			raise ValueError(
				f"DeepSeek-V4 component {component_name!r} has no physical layers"
			)

	def _resolve_default_component_name(
		self, component_name: Optional[str]
	) -> str:
		if component_name is not None:
			self.get_component(component_name)
			return component_name
		if SWA in self.component_names:
			return SWA
		if PRIMARY_MLA in self.component_names:
			return PRIMARY_MLA
		return self.component_names[0]

	def _component_or_default(self, component_name: Optional[str], context: str):
		name = self.default_component_name if component_name is None else component_name
		try:
			return self.get_component(name)
		except KeyError as exc:
			raise KeyError(f"{context}: unknown DSV4 GPU KV component {name!r}") from exc

	def _manager_for_layer(
		self, component_name: Optional[str], logical_layer_id: int, context: str
	) -> tuple[Any, int]:
		component = self._component_or_default(component_name, context)
		return component.manager, component.resolve_physical_layer(int(logical_layer_id))

	def _call_all_managers(
		self, method_name: str, *args: Any, **kwargs: Any
	) -> dict[str, Any]:
		results: dict[str, Any] = {}
		for component in self.components():
			method = getattr(component.manager, method_name)
			results[component.name] = method(*args, **kwargs)
		return results

	# -- Lifecycle --

	def initialize(self) -> dict[str, Any]:
		return self._call_all_managers("initialize")

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
		return self.get_manager(self.default_component_name).config

	@property
	def device(self):
		return self.get_manager(self.default_component_name).device

	@property
	def _gpu_page_table_manager(self):
		return self.get_manager(self.default_component_name)._gpu_page_table_manager

	@property
	def _sequences(self):
		return self.get_manager(self.default_component_name)._sequences

	# -- Page allocation/free --

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
		return results.get(self.default_component_name, {}).get(int(sequence_id), [])

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
		allocated: list[tuple[Any, Any]] = []
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
						"Failed to rollback DSV4 GPU KV allocation for %s on %s",
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
		return results.get(self.default_component_name, {}).get(int(sequence_id), [])

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
		return self._call_all_managers(
			"grow_pages_for_sequences", sequence_ids, num_pages
		)

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
		return results.get(self.default_component_name, 0)

	def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> dict[str, Any]:
		return self._call_all_managers(
			"free_pages_for_sequences", [int(seq_id) for seq_id in sequence_ids]
		)

	# -- Page table --

	def rebuild_page_table(self, sequence_ids: Sequence[int]):
		results = self._call_all_managers(
			"rebuild_page_table", [int(seq_id) for seq_id in sequence_ids]
		)
		return results[self.default_component_name]

	def clear_page_table(self) -> dict[str, Any]:
		return self._call_all_managers("clear_page_table")

	def get_page_table_version(self) -> int:
		return self.get_manager(self.default_component_name).get_page_table_version()

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

	# -- Query/write helpers --

	def get_stats(self):
		return self.get_manager(self.default_component_name).get_stats()

	def get_stats_by_component(self) -> dict[str, Any]:
		return self._call_all_managers("get_stats")

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


def _compact_layer_map(
	compression_ratios: Sequence[int], target_ratio: int
) -> tuple[int, ...]:
	next_physical_layer = 0
	mapping: list[int] = []
	for ratio in compression_ratios:
		if int(ratio) == target_ratio:
			mapping.append(next_physical_layer)
			next_physical_layer += 1
		else:
			mapping.append(-1)
	return tuple(mapping)


def _count_physical_layers(mapping: Sequence[int]) -> int:
	return sum(1 for physical_layer in mapping if int(physical_layer) >= 0)


def _ceil_div_fn(divisor: int) -> TokenCapacityFn:
	def token_capacity(num_tokens: int) -> int:
		checked = _check_positive_tokens(num_tokens)
		return (checked + divisor - 1) // divisor

	return token_capacity


def _identity_token_capacity(num_tokens: int) -> int:
	return _check_positive_tokens(num_tokens)


def _check_positive_tokens(num_tokens: int) -> int:
	num_tokens = int(num_tokens)
	if num_tokens <= 0:
		raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
	return num_tokens


def _view_owns_layer_mapping(view: Any) -> bool:
	return bool(getattr(view, "uses_logical_layer_mapping", False))


def _normalize_seq_token_pairs(seq_token_pairs) -> list[tuple[int, int]]:
	return [
		(int(sequence_id), _check_positive_tokens(num_tokens))
		for sequence_id, num_tokens in list(seq_token_pairs)
	]


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
				f"Cannot rollback DSV4 GPU KV allocation for seq {seq_id}: "
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
