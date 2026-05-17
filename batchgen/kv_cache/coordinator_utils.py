"""Shared helpers for heterogeneous KV coordinators."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


LayerMapping = Mapping[int, int] | Sequence[int]
TokenCapacityFn = Callable[[int], int]


@dataclass
class KVComponent:
	"""Common metadata for one heterogeneous KV component."""

	name: str
	storage: Any
	component_kind: str
	logical_to_physical_layer: LayerMapping | None = None
	token_capacity_scale: float = 1.0
	token_capacity_fn: TokenCapacityFn | None = None

	def __post_init__(self) -> None:
		if not self.name:
			raise ValueError(f"{type(self).__name__}.name must be non-empty")
		if self.storage is None:
			raise ValueError(f"{type(self).__name__}({self.name}): storage must be set")
		if self.token_capacity_scale <= 0:
			raise ValueError(
				f"{type(self).__name__}({self.name}): token_capacity_scale must be > 0"
			)

	def token_capacity(self, num_tokens: int) -> int:
		if num_tokens <= 0:
			raise ValueError(
				f"{type(self).__name__}({self.name}): num_tokens must be > 0, got {num_tokens}"
			)
		if self.token_capacity_fn is not None:
			capacity = int(self.token_capacity_fn(int(num_tokens)))
		else:
			capacity = int(math.ceil(int(num_tokens) * self.token_capacity_scale))
		if capacity <= 0:
			raise ValueError(
				f"{type(self).__name__}({self.name}): token capacity must be > 0, got {capacity}"
			)
		return capacity

	def resolve_physical_layer(self, logical_layer_id: int) -> int:
		if logical_layer_id < 0:
			raise IndexError(
				f"{type(self).__name__}({self.name}): logical layer id must be >= 0"
			)
		if self.logical_to_physical_layer is None:
			resolver = getattr(self.storage, "resolve_physical_layer", None)
			if resolver is not None:
				return int(resolver(logical_layer_id))
			return int(logical_layer_id)
		return resolve_from_layer_mapping(
			self.component_kind,
			self.name,
			self.logical_to_physical_layer,
			logical_layer_id,
		)


class HostKVComponent(KVComponent):
	"""One named host KV component backed by a host worker view."""

	def __init__(
		self,
		name: str,
		view: Any,
		logical_to_physical_layer: LayerMapping | None = None,
		token_capacity_scale: float = 1.0,
		token_capacity_fn: TokenCapacityFn | None = None,
	) -> None:
		super().__init__(
			name=name,
			storage=view,
			component_kind="Host KV",
			logical_to_physical_layer=logical_to_physical_layer,
			token_capacity_scale=token_capacity_scale,
			token_capacity_fn=token_capacity_fn,
		)

	@property
	def view(self) -> Any:
		return self.storage

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


class GPUKVComponent(KVComponent):
	"""One named GPU KV component backed by a paged GPU manager."""

	def __init__(
		self,
		name: str,
		manager: Any,
		logical_to_physical_layer: LayerMapping | None = None,
		token_capacity_scale: float = 1.0,
		token_capacity_fn: TokenCapacityFn | None = None,
	) -> None:
		super().__init__(
			name=name,
			storage=manager,
			component_kind="GPU KV",
			logical_to_physical_layer=logical_to_physical_layer,
			token_capacity_scale=token_capacity_scale,
			token_capacity_fn=token_capacity_fn,
		)

	@property
	def manager(self) -> Any:
		return self.storage

	def map_token_counts(self, token_counts: Sequence[int]) -> list[int]:
		return [self.token_capacity(int(count)) for count in token_counts]


def resolve_from_layer_mapping(
	component_kind: str,
	component_name: str,
	mapping: LayerMapping,
	logical_layer_id: int,
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
			f"{component_kind} component {component_name!r} has no physical layer "
			f"for logical layer {logical_layer_id}"
		)
	return int(physical)
