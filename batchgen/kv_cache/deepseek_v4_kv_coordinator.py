"""DeepSeek-V4 KV coordinators.

DeepSeek-V4 has multiple logical KV components whose layer sets and token
rates differ. Layer and allocation policy live in each KV view/manager; the
DSV4 coordinators below only wire the model's concrete components together.
"""

from __future__ import annotations

from typing import Any

SWA = "swa"
COMPRESSOR_C4 = "compressor_c4"
COMPRESSOR_C128 = "compressor_c128"
INDEXER_C4 = "indexer_c4"
COMPRESSOR_C4_STATE = "compressor_c4_state"
COMPRESSOR_C128_STATE = "compressor_c128_state"
INDEXER_C4_STATE = "indexer_c4_state"

_PAGED_COMPONENT_NAMES = (SWA, COMPRESSOR_C4, COMPRESSOR_C128, INDEXER_C4)
_STATE_COMPONENT_NAMES = (
    COMPRESSOR_C4_STATE,
    COMPRESSOR_C128_STATE,
    INDEXER_C4_STATE,
)
_COMPONENT_NAMES = _PAGED_COMPONENT_NAMES + _STATE_COMPONENT_NAMES


class DeepSeekV4HostKVCoordinator:
    """Runtime facade for DeepSeek-V4 host KV components.

    Each component owns its own layout and operation protocol. The coordinator
    only registers names and performs the small amount of model-level lifecycle
    wiring whose signatures differ between paged KV and compressor state.
    """

    def __init__(
        self,
        *,
        swa: Any,
        compressor_c4: Any = None,
        compressor_c128: Any = None,
        indexer_c4: Any = None,
        compressor_c4_state: Any = None,
        compressor_c128_state: Any = None,
        indexer_c4_state: Any = None,
    ) -> None:
        self.swa = swa
        self.compressor_c4 = compressor_c4
        self.compressor_c128 = compressor_c128
        self.indexer_c4 = indexer_c4
        self.compressor_c4_state = compressor_c4_state
        self.compressor_c128_state = compressor_c128_state
        self.indexer_c4_state = indexer_c4_state

    def views_by_group(self) -> dict[int, Any]:
        return {
            group_id: manager
            for group_id, manager in (
                (0, self.swa),
                (1, self.compressor_c4),
                (2, self.compressor_c128),
                (3, self.indexer_c4),
            )
            if manager is not None
        }

    def initialize(
        self, device_index: int, create_region: bool = False
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name in _PAGED_COMPONENT_NAMES:
            manager = getattr(self, component_name, None)
            if manager is not None:
                results[component_name] = manager.initialize(
                    device_index=int(device_index),
                    create_region=create_region,
                )
        for component_name in _STATE_COMPONENT_NAMES:
            manager = getattr(self, component_name, None)
            if manager is not None:
                results[component_name] = manager.initialize(int(device_index))
        return results

    def shutdown(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name in reversed(_STATE_COMPONENT_NAMES):
            manager = getattr(self, component_name, None)
            if manager is not None:
                results[component_name] = manager.shutdown()
        for component_name in reversed(_PAGED_COMPONENT_NAMES):
            manager = getattr(self, component_name, None)
            if manager is not None:
                results[component_name] = manager.shutdown()
        return results


class DeepSeekV4GPUKVCoordinator:
    """Runtime facade for DeepSeek-V4 GPU KV components."""

    def __init__(
        self,
        *,
        swa: Any,
        compressor_c4: Any = None,
        compressor_c128: Any = None,
        indexer_c4: Any = None,
        compressor_c4_state: Any = None,
        compressor_c128_state: Any = None,
        indexer_c4_state: Any = None,
    ) -> None:
        self.swa = swa
        self.compressor_c4 = compressor_c4
        self.compressor_c128 = compressor_c128
        self.indexer_c4 = indexer_c4
        self.compressor_c4_state = compressor_c4_state
        self.compressor_c128_state = compressor_c128_state
        self.indexer_c4_state = indexer_c4_state

    def managers_by_group(self) -> dict[int, Any]:
        return {
            group_id: manager
            for group_id, manager in (
                (0, self.swa),
                (1, self.compressor_c4),
                (2, self.compressor_c128),
                (3, self.indexer_c4),
            )
            if manager is not None
        }

    def initialize(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name in _COMPONENT_NAMES:
            manager = getattr(self, component_name, None)
            if manager is not None:
                results[component_name] = manager.initialize()
        return results

    def destroy(self, *, empty_cuda_cache: bool = False) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name in reversed(_COMPONENT_NAMES):
            manager = getattr(self, component_name, None)
            if manager is not None:
                results[component_name] = manager.destroy(
                    empty_cuda_cache=empty_cuda_cache
                )
        return results
