"""Lightweight registry for host-side heterogeneous KV views."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence


LayerMapping = Mapping[int, int] | Sequence[int]
TokenCapacityFn = Callable[[int], int]
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


@dataclass
class HostKVComponent:
	"""One named host KV component.

	The component keeps only metadata needed to route from a model's logical
	layer id to the compact physical layer stored by its backing view.
	"""

	name: str
	view: Any
	logical_to_physical_layer: Optional[LayerMapping] = None
	token_capacity_scale: float = 1.0
	token_capacity_fn: Optional[TokenCapacityFn] = None

	def __post_init__(self) -> None:
		if not self.name:
			raise ValueError("HostKVComponent.name must be non-empty")
		if self.view is None:
			raise ValueError(f"HostKVComponent({self.name}): view must be set")
		if self.token_capacity_scale <= 0:
			raise ValueError(
				f"HostKVComponent({self.name}): token_capacity_scale must be > 0"
			)

	def token_capacity(self, num_tokens: int) -> int:
		if num_tokens <= 0:
			raise ValueError(
				f"HostKVComponent({self.name}): num_tokens must be > 0, got {num_tokens}"
			)
		if self.token_capacity_fn is not None:
			capacity = int(self.token_capacity_fn(int(num_tokens)))
		else:
			capacity = int(math.ceil(int(num_tokens) * self.token_capacity_scale))
		if capacity <= 0:
			raise ValueError(
				f"HostKVComponent({self.name}): token capacity must be > 0, got {capacity}"
			)
		return capacity

	def resolve_physical_layer(self, logical_layer_id: int) -> int:
		if logical_layer_id < 0:
			raise IndexError(
				f"HostKVComponent({self.name}): logical layer id must be >= 0"
			)
		if self.logical_to_physical_layer is not None:
			return _resolve_from_mapping(
				self.name, self.logical_to_physical_layer, logical_layer_id
			)
		resolver = getattr(self.view, "resolve_physical_layer", None)
		if resolver is not None:
			return int(resolver(logical_layer_id))
		return int(logical_layer_id)

	def view_layer_id(self, logical_layer_id: int) -> int:
		"""Layer id that should be passed to this backing view.

		If the C++ view already owns logical-layer mapping, keep passing the
		logical id into the view. If the mapping is owned by this Python
		component, pass the resolved compact physical layer id.
		"""

		if self.logical_to_physical_layer is None:
			return int(logical_layer_id)
		return self.resolve_physical_layer(logical_layer_id)

	def map_sequence_tokens(
		self, seq_token_pairs: Iterable[tuple[int, int]]
	) -> list[tuple[int, int]]:
		return [
			(int(seq_id), self.token_capacity(int(num_tokens)))
			for seq_id, num_tokens in seq_token_pairs
		]


class HostKVCoordinator:
	"""Minimal component registry for host KV.

	This class deliberately does not mirror the full HostPagedKVWorkerView API.
	Callers should get the backing view with ``get_view(name)`` and use the
	view's existing methods. The coordinator only centralizes component lookup,
	layer routing, and optional token-capacity mapping.
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

	def view_layer_id(self, component_name: str, logical_layer_id: int) -> int:
		return self.get_component(component_name).view_layer_id(logical_layer_id)

	def map_sequence_tokens(
		self, component_name: str, seq_token_pairs: Iterable[tuple[int, int]]
	) -> list[tuple[int, int]]:
		return self.get_component(component_name).map_sequence_tokens(seq_token_pairs)

	def call_all(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
		"""Call the same method on every backing view."""

		results: dict[str, Any] = {}
		for component in self.components():
			method = getattr(component.view, method_name)
			results[component.name] = method(*args, **kwargs)
		return results


def _resolve_from_mapping(
	component_name: str, mapping: LayerMapping, logical_layer_id: int
) -> int:
	if isinstance(mapping, Mapping):
		physical = mapping.get(int(logical_layer_id), -1)
	else:
		if logical_layer_id >= len(mapping):
			physical = -1
		else:
			physical = int(mapping[logical_layer_id])
	if physical < 0:
		raise KeyError(
			f"Host KV component {component_name!r} has no physical layer for "
			f"logical layer {logical_layer_id}"
		)
	return int(physical)
