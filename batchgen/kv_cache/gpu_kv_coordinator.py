"""Lightweight registry for GPU-side heterogeneous KV managers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence


LayerMapping = Mapping[int, int] | Sequence[int]
TokenCapacityFn = Callable[[int], int]


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
		return _resolve_from_mapping(
			self.name, self.logical_to_physical_layer, logical_layer_id
		)

	def map_token_counts(self, token_counts: Sequence[int]) -> list[int]:
		return [self.token_capacity(int(count)) for count in token_counts]


class GPUKVCoordinator:
	"""Minimal component registry for GPU KV.

	This class does not proxy the GPUPagedKVCacheManager API. Callers choose a
	component, get its manager, and call the manager directly.
	"""

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
			f"GPU KV component {component_name!r} has no physical layer for "
			f"logical layer {logical_layer_id}"
		)
	return int(physical)
