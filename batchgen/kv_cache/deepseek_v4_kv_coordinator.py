"""DeepSeek-V4 KV coordinators.

DeepSeek-V4 has multiple logical KV components whose layer sets and token
rates differ. The generic Host/GPU coordinators keep the component registry and
layer mapping metadata; the DSV4 coordinators below only register the model's
component layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator
from batchgen.kv_cache.host_kv_coordinator import HostKVCoordinator


SWA = "swa"
COMPRESSOR_C4 = "compressor_c4"
COMPRESSOR_C128 = "compressor_c128"
INDEXER_C4 = "indexer_c4"

DEEPSEEK_V4_COMPONENT_ORDER = (
	SWA,
	COMPRESSOR_C4,
	COMPRESSOR_C128,
	INDEXER_C4,
)

_SUPPORTED_COMPRESS_RATIOS = {0, 4, 128}

TokenCapacityFn = Callable[[int], int]


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
		if component_name == SWA:
			return None
		raise KeyError(f"Unknown DeepSeek-V4 KV component: {component_name}")

	def physical_layer_count(self, component_name: str) -> int:
		if component_name == SWA:
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
		default_component: Optional[str] = None,
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
		self._register_optional_view(COMPRESSOR_C4, compressor_c4)
		self._register_optional_view(COMPRESSOR_C128, compressor_c128)
		self._register_optional_view(INDEXER_C4, indexer_c4)
		self.set_default_component(
			SWA if default_component is None else default_component
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
		default_component: Optional[str] = None,
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
		self._register_optional_manager(COMPRESSOR_C4, compressor_c4)
		self._register_optional_manager(COMPRESSOR_C128, compressor_c128)
		self._register_optional_manager(INDEXER_C4, indexer_c4)
		self.set_default_component(
			SWA if default_component is None else default_component
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
