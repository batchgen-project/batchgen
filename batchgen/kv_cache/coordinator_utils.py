"""Shared helpers for heterogeneous KV coordinators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


LayerMapping = Mapping[int, int] | Sequence[int]


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
