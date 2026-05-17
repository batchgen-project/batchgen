"""Lightweight registry for host-side heterogeneous KV views."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from batchgen.kv_cache.coordinator_utils import HostKVComponent
logger = logging.getLogger(__name__)


@dataclass
class AsyncKVTask:
	"""Composite async KV task keyed by component name."""

	tasks: Mapping[str, Any]
	tensors: Any = None

	def wait(self) -> None:
		errors = []
		for component_name, task in self.tasks.items():
			if task is None:
				continue
			try:
				task.wait()
			except Exception as exc:
				errors.append((component_name, exc))
		if not errors:
			return
		if len(errors) == 1:
			raise errors[0][1]
		names = ", ".join(name for name, _ in errors)
		raise RuntimeError(
			f"AsyncKVTask wait failed for components: {names}"
		) from errors[0][1]


def wait_kv_tasks(tasks: Mapping[str, Any], *, context: str = "host KV task") -> None:
	"""Best-effort drain for async tasks that were already launched."""

	for component_name, task in tasks.items():
		if task is None:
			continue
		try:
			task.wait()
		except Exception:
			logger.exception(
				"%s launched before failure did not drain cleanly: %s",
				context,
				component_name,
			)


class HostKVCoordinator:
	"""Minimal component registry for host KV.

	The coordinator centralizes component lookup, layer routing, optional
	token-capacity mapping, and generic host/device KV movement helpers.
	"""

	def __init__(self) -> None:
		self._components: Dict[str, HostKVComponent] = {}

	def register_component(
		self,
		component: HostKVComponent | str,
		view: Any = None,
		**kwargs: Any,
	) -> HostKVComponent:
		if isinstance(component, HostKVComponent):
			if view is not None or kwargs:
				raise ValueError(
					"Pass either a HostKVComponent or name/view/kwargs, not both"
				)
			item = component
		else:
			item = HostKVComponent(name=component, view=view, **kwargs)
		if item.name in self._components:
			raise ValueError(f"Host KV component already registered: {item.name}")
		self._components[item.name] = item
		setattr(self, item.name, item.view)
		return item

	@property
	def component_names(self) -> list[str]:
		return list(self._components.keys())

	def components(self) -> Iterator[HostKVComponent]:
		return iter(self._components.values())

	def get_component(self, name: str) -> HostKVComponent:
		try:
			return self._components[name]
		except KeyError as exc:
			raise KeyError(f"Unknown host KV component: {name}") from exc

	def get_view(self, name: str) -> Any:
		return self.get_component(name).view

	def resolve_physical_layer(self, component_name: str, logical_layer_id: int) -> int:
		return self.get_component(component_name).resolve_physical_layer(logical_layer_id)

	def storage_layer_id(self, component_name: str, logical_layer_id: int) -> int:
		return self.get_component(component_name).storage_layer_id(logical_layer_id)

	def map_token_counts(
		self, component_name: str, token_counts: Iterable[int]
	) -> list[int]:
		return self.get_component(component_name).map_token_counts(list(token_counts))

	def call_all(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
		"""Call the same method on every backing view."""

		results: dict[str, Any] = {}
		for component in self.components():
			method = getattr(component.view, method_name)
			results[component.name] = method(*args, **kwargs)
		return results

	def initialize(self, **kwargs: Any) -> dict[str, Any]:
		return self.call_all("initialize", **kwargs)

	def shutdown(self) -> dict[str, Any]:
		results: dict[str, Any] = {}
		for component in reversed(list(self.components())):
			results[component.name] = component.view.shutdown()
		return results

	def register_sequences(self, sequence_ids) -> dict[str, Any]:
		sequence_ids = list(sequence_ids)
		results: dict[str, Any] = {}
		registered: list[HostKVComponent] = []
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
						"Failed to rollback host KV registration for %s on %s",
						sequence_ids[:10],
						component.name,
					)
			raise
		return results

	def unregister_sequence(self, sequence_id: int) -> dict[str, Any]:
		return self.call_all("unregister_sequence", int(sequence_id))

	def unregister_sequences(self, sequence_ids) -> dict[str, Any]:
		return self.call_all("unregister_sequences", list(sequence_ids))

	def allocate_pages_for_sequences(
		self,
		seq_token_pairs,
		*,
		component_name: Optional[str] = None,
	):
		pairs = [
			(int(sequence_id), int(num_tokens))
			for sequence_id, num_tokens in list(seq_token_pairs)
		]
		sequence_ids = [seq_id for seq_id, _ in pairs]
		token_counts = [num_tokens for _, num_tokens in pairs]
		if component_name is not None:
			component = self._component_for_op(
				component_name, "allocate_pages_for_sequences"
			)
			return component.view.allocate_pages_for_sequences(
				list(zip(sequence_ids, component.map_token_counts(token_counts)))
			)

		results: dict[str, Any] = {}
		allocated: list[HostKVComponent] = []
		try:
			for component in self.components():
				results[component.name] = component.view.allocate_pages_for_sequences(
					list(zip(sequence_ids, component.map_token_counts(token_counts)))
				)
				allocated.append(component)
		except Exception:
			for component in reversed(allocated):
				try:
					component.view.release_sequence_pages(sequence_ids)
				except Exception:
					logger.exception(
						"Failed to rollback host KV allocation for %s on %s",
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
			return component.view.grow_sequence_pages(int(sequence_id), int(num_pages))
		return self.call_all("grow_sequence_pages", int(sequence_id), int(num_pages))

	def grow_pages_for_sequences(
		self,
		seq_page_pairs,
		*,
		component_name: Optional[str] = None,
	):
		pairs = [
			(int(sequence_id), int(num_pages))
			for sequence_id, num_pages in list(seq_page_pairs)
		]
		if component_name is not None:
			component = self._component_for_op(
				component_name, "grow_pages_for_sequences"
			)
			return component.view.grow_pages_for_sequences(pairs)
		return self.call_all("grow_pages_for_sequences", pairs)

	def release_sequence_pages(self, sequence_ids) -> dict[str, Any]:
		return self.call_all("release_sequence_pages", list(sequence_ids))

	def free_pages_for_sequences(self, sequence_ids) -> dict[str, Any]:
		return self.release_sequence_pages(sequence_ids)

	def build_page_table(self, sequence_ids):
		return self.call_all("build_page_table", list(sequence_ids))

	def get_stats(self):
		return self.call_all("get_stats")

	def get_stats_by_component(self) -> dict[str, Any]:
		return self.get_stats()

	def data_base_address(self, *, component_name: str):
		component = self._component_for_op(component_name, "data_base_address")
		return component.view.data_base_address()

	def read_sequence_kv_to_cpu(self, sequence_id: int, *, component_name: str):
		component = self._component_for_op(component_name, "read_sequence_kv_to_cpu")
		return component.view.read_sequence_kv_to_cpu(int(sequence_id))

	def write_sequence_kv_from_cpu(
		self,
		sequence_id: int,
		k_tensor,
		v_tensor=None,
		*,
		component_name: str,
	):
		component = self._component_for_op(component_name, "write_sequence_kv_from_cpu")
		return component.view.write_sequence_kv_from_cpu(
			int(sequence_id), k_tensor, v_tensor
		)

	def k_page_ptr(
		self,
		layer_idx: int,
		page_idx: int,
		*,
		component_name: str,
	):
		view, storage_layer_id = self._view_for_data_op(
			component_name, layer_idx, "k_page_ptr"
		)
		return view.k_page_ptr(storage_layer_id, int(page_idx))

	def v_page_ptr(
		self,
		layer_idx: int,
		page_idx: int,
		*,
		component_name: str,
	):
		view, storage_layer_id = self._view_for_data_op(
			component_name, layer_idx, "v_page_ptr"
		)
		return view.v_page_ptr(storage_layer_id, int(page_idx))

	def get_sequence_layer_page_pointers(
		self,
		sequence_id: int,
		layer_idx: int,
		max_tokens=None,
		*,
		component_name: str,
	):
		component = self._component_for_op(
			component_name, "get_sequence_layer_page_pointers"
		)
		return component.view.get_sequence_layer_page_pointers(
			int(sequence_id),
			component.storage_layer_id(int(layer_idx)),
			None if max_tokens is None else component.token_capacity(int(max_tokens)),
		)

	def _component_for_op(self, component_name: str, context: str) -> HostKVComponent:
		if component_name is None:
			raise KeyError(f"{context}: component_name is required")
		try:
			return self.get_component(component_name)
		except KeyError as exc:
			raise KeyError(
				f"{context}: unknown host KV component {component_name!r}"
			) from exc

	def _view_for_data_op(
		self, component_name: str, logical_layer_id: int, context: str
	) -> tuple[Any, int]:
		component = self._component_for_op(component_name, context)
		return component.view, component.storage_layer_id(int(logical_layer_id))

	def async_offload_layer_kv_to_host(
		self,
		layer_idx: int,
		sequence_ids,
		k_tensor,
		v_tensor=None,
		sequence_lengths=None,
		*,
		component_name: str,
	):
		if sequence_lengths is None:
			raise TypeError("async_offload_layer_kv_to_host requires sequence_lengths")
		view, storage_layer_id = self._view_for_data_op(
			component_name, layer_idx, "async_offload_layer_kv_to_host"
		)
		return view.async_offload_layer_kv_to_host(
			storage_layer_id, sequence_ids, k_tensor, v_tensor, sequence_lengths
		)

	def async_append_decode_kv_to_host(
		self,
		layer_idx: int,
		sequence_ids,
		k_tensor,
		v_tensor=None,
		sequence_lengths=None,
		*,
		component_name: str,
	):
		if sequence_lengths is None:
			raise TypeError("async_append_decode_kv_to_host requires sequence_lengths")
		view, storage_layer_id = self._view_for_data_op(
			component_name, layer_idx, "async_append_decode_kv_to_host"
		)
		return view.async_append_decode_kv_to_host(
			storage_layer_id, sequence_ids, k_tensor, v_tensor, sequence_lengths
		)

	def async_append_decode_kv_to_host_batched_kernel(
		self,
		entries,
		sequence_ids,
		sequence_lengths,
		*,
		component_name: str,
	):
		component = self._component_for_op(
			component_name, "async_append_decode_kv_to_host_batched_kernel"
		)
		mapped_entries = [
			(
				component.storage_layer_id(int(entry[0])),
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
		component_name: str,
	):
		component = self._component_for_op(
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
		component_name: str,
	):
		component = self._component_for_op(
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
		context: str = "host KV load",
	) -> AsyncKVTask:
		tasks: dict[str, Any] = {}
		for component_name, kwargs in component_loads.items():
			try:
				tasks[component_name] = self.async_load_layer_paged_kv_to_device(
					component_name=component_name, **dict(kwargs)
				)
			except Exception:
				wait_kv_tasks(tasks, context=context)
				raise
		return AsyncKVTask(tasks=tasks, tensors=tensors)

	def async_offload_components_kv_to_host(
		self,
		component_offloads: Mapping[str, Mapping[str, Any]],
		*,
		tensors: Any = None,
		context: str = "host KV offload",
	) -> AsyncKVTask:
		tasks: dict[str, Any] = {}
		for component_name, kwargs in component_offloads.items():
			try:
				tasks[component_name] = self.async_offload_layer_kv_to_host(
					component_name=component_name, **dict(kwargs)
				)
			except Exception:
				wait_kv_tasks(tasks, context=context)
				raise
		return AsyncKVTask(tasks=tasks, tensors=tensors)
