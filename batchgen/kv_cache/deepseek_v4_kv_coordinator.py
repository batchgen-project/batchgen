"""DeepSeek-V4 KV coordinators.

DeepSeek-V4 has multiple logical KV components whose layer sets and token
rates differ. Layer and allocation policy live in each KV view/manager; the
DSV4 coordinators below only register the model's component names.
"""

from __future__ import annotations

from typing import Any

from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator
from batchgen.kv_cache.host_kv_coordinator import HostKVCoordinator

SWA = "swa"
COMPRESSOR_C4 = "compressor_c4"
COMPRESSOR_C128 = "compressor_c128"
INDEXER_C4 = "indexer_c4"
COMPRESSOR_C4_STATE = "compressor_c4_state"
COMPRESSOR_C128_STATE = "compressor_c128_state"
INDEXER_C4_STATE = "indexer_c4_state"


class DeepSeekV4HostKVCoordinator(HostKVCoordinator):
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
        super().__init__()
        for component_name, view in (
            (SWA, swa),
            (COMPRESSOR_C4, compressor_c4),
            (COMPRESSOR_C128, compressor_c128),
            (INDEXER_C4, indexer_c4),
            (COMPRESSOR_C4_STATE, compressor_c4_state),
            (COMPRESSOR_C128_STATE, compressor_c128_state),
            (INDEXER_C4_STATE, indexer_c4_state),
        ):
            setattr(self, component_name, view)
            if view is None:
                continue
            self.register_component(component_name, view)

    def initialize(
        self, device_index: int, create_region: bool = False
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name in (
            SWA,
            COMPRESSOR_C4,
            COMPRESSOR_C128,
            INDEXER_C4,
        ):
            manager = getattr(self, component_name, None)
            if manager is not None:
                results[component_name] = manager.initialize(
                    device_index=int(device_index),
                    create_region=create_region,
                )
        for component_name, manager in (
            (COMPRESSOR_C4_STATE, self.compressor_c4_state),
            (COMPRESSOR_C128_STATE, self.compressor_c128_state),
            (INDEXER_C4_STATE, self.indexer_c4_state),
        ):
            if manager is not None:
                results[component_name] = manager.initialize(int(device_index))
        return results

    def shutdown(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name, manager in (
            (INDEXER_C4_STATE, self.indexer_c4_state),
            (COMPRESSOR_C128_STATE, self.compressor_c128_state),
            (COMPRESSOR_C4_STATE, self.compressor_c4_state),
        ):
            if manager is not None:
                results[component_name] = manager.shutdown()
        for component_name in (
            INDEXER_C4,
            COMPRESSOR_C128,
            COMPRESSOR_C4,
            SWA,
        ):
            manager = getattr(self, component_name, None)
            if manager is not None:
                results[component_name] = manager.shutdown()
        return results


class DeepSeekV4GPUKVCoordinator(GPUKVCoordinator):
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
        super().__init__()
        for component_name, manager in (
            (SWA, swa),
            (COMPRESSOR_C4, compressor_c4),
            (COMPRESSOR_C128, compressor_c128),
            (INDEXER_C4, indexer_c4),
            (COMPRESSOR_C4_STATE, compressor_c4_state),
            (COMPRESSOR_C128_STATE, compressor_c128_state),
            (INDEXER_C4_STATE, indexer_c4_state),
        ):
            setattr(self, component_name, manager)
            if manager is None:
                continue
            self.register_component(component_name, manager)

    def initialize(self) -> dict[str, Any]:
        return self.call_all("initialize")

    def destroy(self, *, empty_cuda_cache: bool = False) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name, manager in reversed(list(self.components())):
            results[component_name] = manager.destroy(
                empty_cuda_cache=empty_cuda_cache
            )
        return results
