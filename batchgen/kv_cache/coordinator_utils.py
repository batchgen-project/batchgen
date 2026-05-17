"""Shared helpers for heterogeneous KV coordinators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


LayerMapping = Sequence[int]


@dataclass
class KVComponent:
	"""Common registry entry for one heterogeneous KV component."""

	name: str
	storage: Any

	def __post_init__(self) -> None:
		if not self.name:
			raise ValueError(f"{type(self).__name__}.name must be non-empty")
		if self.storage is None:
			raise ValueError(f"{type(self).__name__}({self.name}): storage must be set")


class HostKVComponent(KVComponent):
	"""One named host KV component backed by a host worker view."""

	def __init__(
		self,
		name: str,
		view: Any,
	) -> None:
		super().__init__(
			name=name,
			storage=view,
		)

	@property
	def view(self) -> Any:
		return self.storage


class GPUKVComponent(KVComponent):
	"""One named GPU KV component backed by a paged GPU manager."""

	def __init__(
		self,
		name: str,
		manager: Any,
	) -> None:
		super().__init__(
			name=name,
			storage=manager,
		)

	@property
	def manager(self) -> Any:
		return self.storage


def resolve_from_layer_mapping(
	component_kind: str,
	component_name: str,
	mapping: LayerMapping,
	logical_layer_id: int,
) -> int:
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


def validate_layer_mapping(
	component_kind: str,
	component_name: str,
	mapping: LayerMapping,
) -> None:
	if len(mapping) == 0:
		raise ValueError(
			f"{component_kind} component {component_name!r}: "
			"logical_to_physical_layer must be non-empty"
		)

	has_physical_layer = False
	for logical_layer_id, physical_layer in enumerate(mapping):
		physical_layer = int(physical_layer)
		if physical_layer < -1:
			raise ValueError(
				f"{component_kind} component {component_name!r}: "
				"logical_to_physical_layer entries must be >= -1, got "
				f"{physical_layer} at logical layer {logical_layer_id}"
			)
		if physical_layer >= 0:
			has_physical_layer = True

	if not has_physical_layer:
		raise ValueError(
			f"{component_kind} component {component_name!r} has no physical layers"
		)
