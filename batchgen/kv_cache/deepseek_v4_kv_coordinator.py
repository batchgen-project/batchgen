"""DeepSeek-V4 KV coordinators.

DeepSeek-V4 has multiple logical KV components whose layer sets and token
rates differ. The generic Host/GPU coordinators keep the component registry and
layer mapping metadata; the DSV4 coordinators below only register the model's
component layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from batchgen.kv_cache.coordinator_utils import TokenCapacityFn
from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator
from batchgen.kv_cache.host_kv_coordinator import HostKVCoordinator


SWA = "swa"
COMPRESSOR_C4 = "compressor_c4"
COMPRESSOR_C128 = "compressor_c128"
INDEXER_C4 = "indexer_c4"

_SUPPORTED_COMPRESS_RATIOS = {0, 4, 128}


@dataclass
class DeepSeekV4KVLayout:
	"""Layer and token routing metadata for DeepSeek-V4 KV components."""

	compression_ratios: list[int]
	c4_logical_to_physical_layer: list[int]
	c128_logical_to_physical_layer: list[int]
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
		return cls(
			compression_ratios=ratios,
			c4_logical_to_physical_layer=_compact_logical_to_physical_layer(
				ratios, 4
			),
			c128_logical_to_physical_layer=_compact_logical_to_physical_layer(
				ratios, 128
			),
			sliding_window=None if sliding_window is None else int(sliding_window),
		)

	@property
	def num_layers(self) -> int:
		return len(self.compression_ratios)

	@property
	def num_c4_layers(self) -> int:
		return _count_physical_layers(self.c4_logical_to_physical_layer)

	@property
	def num_c128_layers(self) -> int:
		return _count_physical_layers(self.c128_logical_to_physical_layer)

	def logical_to_physical_layer(self, component_name: str) -> Optional[list[int]]:
		if component_name == COMPRESSOR_C4:
			return self.c4_logical_to_physical_layer
		if component_name == COMPRESSOR_C128:
			return self.c128_logical_to_physical_layer
		if component_name == INDEXER_C4:
			return self.c4_logical_to_physical_layer
		if component_name == SWA:
			return None
		raise KeyError(f"Unknown DeepSeek-V4 KV component: {component_name}")

	def physical_layer_count(self, component_name: str) -> int:
		if component_name == SWA:
			return self.num_layers
		mapping = self.logical_to_physical_layer(component_name)
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
			return lambda num_tokens: min(int(num_tokens), window)
		if component_name == SWA:
			return _identity_token_capacity
		raise KeyError(f"Unknown DeepSeek-V4 KV component: {component_name}")

	def token_capacity(self, component_name: str, num_tokens: int) -> int:
		return self.token_capacity_fn(component_name)(num_tokens)


class DeepSeekV4HostKVCoordinator(HostKVCoordinator):
	"""Runtime facade for DeepSeek-V4 host KV worker views.

	The base coordinator provides lifecycle, page, query, and data movement
	for registered components. DSV4 only supplies its component layout.
	"""

	def __init__(
		self,
		*,
		compression_ratios: Sequence[int],
		swa: Any,
		compressor_c4: Any = None,
		compressor_c128: Any = None,
		indexer_c4: Any = None,
		num_layers: Optional[int] = None,
		sliding_window: Optional[int] = None,
	) -> None:
		super().__init__()
		self.layout = DeepSeekV4KVLayout.from_compression_ratios(
			compression_ratios,
			num_layers=num_layers,
			sliding_window=sliding_window,
		)

		self.register_component(
			SWA,
			swa,
			token_capacity_fn=self.layout.token_capacity_fn(SWA),
		)
		for component_name, view in (
			(COMPRESSOR_C4, compressor_c4),
			(COMPRESSOR_C128, compressor_c128),
			(INDEXER_C4, indexer_c4),
		):
			if view is None:
				continue
			self.register_component(
				component_name,
				view,
				logical_to_physical_layer=(
					None
					if _view_owns_layer_mapping(view)
					else self.layout.logical_to_physical_layer(component_name)
				),
				token_capacity_fn=self.layout.token_capacity_fn(component_name),
			)


class DeepSeekV4GPUKVCoordinator(GPUKVCoordinator):
	"""Runtime facade for DeepSeek-V4 GPU paged KV managers."""

	def __init__(
		self,
		*,
		compression_ratios: Sequence[int],
		swa: Any,
		compressor_c4: Any = None,
		compressor_c128: Any = None,
		indexer_c4: Any = None,
		num_layers: Optional[int] = None,
		sliding_window: Optional[int] = None,
	) -> None:
		super().__init__()
		self.layout = DeepSeekV4KVLayout.from_compression_ratios(
			compression_ratios,
			num_layers=num_layers,
			sliding_window=sliding_window,
		)

		self.register_component(
			SWA,
			swa,
			token_capacity_fn=self.layout.token_capacity_fn(SWA),
		)
		for component_name, manager in (
			(COMPRESSOR_C4, compressor_c4),
			(COMPRESSOR_C128, compressor_c128),
			(INDEXER_C4, indexer_c4),
		):
			if manager is None:
				continue
			self.register_component(
				component_name,
				manager,
				logical_to_physical_layer=(
					self.layout.logical_to_physical_layer(component_name)
				),
				token_capacity_fn=self.layout.token_capacity_fn(component_name),
			)


def _compact_logical_to_physical_layer(
	compression_ratios: Sequence[int], target_ratio: int
) -> list[int]:
	next_physical_layer = 0
	mapping: list[int] = []
	for ratio in compression_ratios:
		if int(ratio) == target_ratio:
			mapping.append(next_physical_layer)
			next_physical_layer += 1
		else:
			mapping.append(-1)
	return mapping


def _count_physical_layers(mapping: Sequence[int]) -> int:
	return sum(1 for physical_layer in mapping if int(physical_layer) >= 0)


def _ceil_div_fn(divisor: int) -> TokenCapacityFn:
	def token_capacity(num_tokens: int) -> int:
		num_tokens = int(num_tokens)
		return (num_tokens + divisor - 1) // divisor

	return token_capacity


def _identity_token_capacity(num_tokens: int) -> int:
	return int(num_tokens)


def _view_owns_layer_mapping(view: Any) -> bool:
	return bool(getattr(view, "uses_logical_layer_mapping", False))
