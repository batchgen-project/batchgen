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

_STATE_COMPONENT_NAMES = (
    COMPRESSOR_C4_STATE,
    COMPRESSOR_C128_STATE,
    INDEXER_C4_STATE,
)


def _iter_state_components(coordinator):
    for component_name in _STATE_COMPONENT_NAMES:
        manager = getattr(coordinator, component_name, None)
        if manager is not None:
            yield component_name, manager


def _state_manager_for_op(coordinator, component_name: str, context: str):
    if component_name is None:
        raise KeyError(f"{context}: component_name is required")
    if component_name not in _STATE_COMPONENT_NAMES:
        raise KeyError(
            f"{context}: unknown DeepSeek-V4 state component {component_name!r}"
        )
    manager = getattr(coordinator, component_name, None)
    if manager is None:
        raise KeyError(
            f"{context}: DeepSeek-V4 state component {component_name!r} is not set"
        )
    return manager


class DeepSeekV4HostKVCoordinator(HostKVCoordinator):
    """Runtime facade for DeepSeek-V4 host KV worker views.

    The base coordinator provides lifecycle, page, query, and data movement for
    registered components. Each worker view owns its layer mapping.
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
        ):
            if view is None:
                continue
            self.register_component(component_name, view)

        self.compressor_c4_state = compressor_c4_state
        self.compressor_c128_state = compressor_c128_state
        self.indexer_c4_state = indexer_c4_state

    @property
    def state_component_names(self) -> list[str]:
        return [name for name, _ in _iter_state_components(self)]

    def state_components(self):
        return _iter_state_components(self)

    def get_state_manager(self, name: str) -> Any:
        return _state_manager_for_op(self, name, "get_state_manager")

    def initialize_state_managers(self, device_index: int) -> dict[str, Any]:
        return {
            component_name: manager.initialize(int(device_index))
            for component_name, manager in self.state_components()
        }

    def initialize(
        self, device_index: int, create_region: bool = False
    ) -> dict[str, Any]:
        results = super().initialize(
            device_index=int(device_index),
            create_region=create_region,
        )
        results.update(self.initialize_state_managers(int(device_index)))
        return results

    def shutdown(self) -> dict[str, Any]:
        results = super().shutdown()
        for component_name, manager in reversed(list(self.state_components())):
            results[component_name] = manager.shutdown()
        return results

    def allocate_state_items_for_sequences(
        self, sequence_ids, *, component_name: str
    ):
        manager = _state_manager_for_op(
            self, component_name, "allocate_state_items_for_sequences"
        )
        return manager.allocate_state_items_for_sequences(
            [int(seq_id) for seq_id in sequence_ids]
        )

    def release_sequence_states(
        self, sequence_ids, *, component_name: str | None = None
    ) -> dict[str, Any] | Any:
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        if component_name is not None:
            manager = _state_manager_for_op(
                self, component_name, "release_sequence_states"
            )
            return manager.release_sequence_states(sequence_ids)
        return {
            name: manager.release_sequence_states(sequence_ids)
            for name, manager in self.state_components()
        }

    def async_offload_decode_state_to_host(
        self,
        layer_idx: int,
        sequence_ids,
        state_tensor,
        raw_positions,
        *,
        component_name: str,
    ):
        manager = _state_manager_for_op(
            self, component_name, "async_offload_decode_state_to_host"
        )
        return manager.async_offload_decode_state_to_host(
            int(layer_idx), sequence_ids, state_tensor, raw_positions
        )

    def async_load_decode_state_to_device(
        self,
        layer_idx: int,
        sequence_ids,
        state_tensor,
        raw_positions,
        *,
        component_name: str,
    ):
        manager = _state_manager_for_op(
            self, component_name, "async_load_decode_state_to_device"
        )
        return manager.async_load_decode_state_to_device(
            int(layer_idx), sequence_ids, state_tensor, raw_positions
        )

    def async_append_decode_state_to_host_batched_kernel(
        self,
        entries,
        sequence_ids,
        raw_positions,
        *,
        component_name: str,
    ):
        manager = _state_manager_for_op(
            self,
            component_name,
            "async_append_decode_state_to_host_batched_kernel",
        )
        normalized_entries = [
            (
                int(entry[0]),
                entry[1],
            )
            for entry in entries
        ]
        return manager.async_append_decode_state_to_host_batched_kernel(
            normalized_entries, sequence_ids, raw_positions
        )

    def async_offload_state_items_to_host(
        self, sequence_ids, state_device_ptrs, *, component_name: str
    ):
        manager = _state_manager_for_op(
            self, component_name, "async_offload_state_items_to_host"
        )
        return manager.async_offload_state_items_to_host(
            sequence_ids, state_device_ptrs
        )

    def async_load_state_items_to_device(
        self, sequence_ids, state_device_ptrs, *, component_name: str
    ):
        manager = _state_manager_for_op(
            self, component_name, "async_load_state_items_to_device"
        )
        return manager.async_load_state_items_to_device(
            sequence_ids, state_device_ptrs
        )


class DeepSeekV4GPUKVCoordinator(GPUKVCoordinator):
    """Runtime facade for DeepSeek-V4 GPU paged KV managers."""

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
        ):
            if manager is None:
                continue
            self.register_component(component_name, manager)

        self.compressor_c4_state = compressor_c4_state
        self.compressor_c128_state = compressor_c128_state
        self.indexer_c4_state = indexer_c4_state

    @property
    def state_component_names(self) -> list[str]:
        return [name for name, _ in _iter_state_components(self)]

    def state_components(self):
        return _iter_state_components(self)

    def get_state_manager(self, name: str) -> Any:
        return _state_manager_for_op(self, name, "get_state_manager")

    def initialize(self) -> dict[str, Any]:
        results = super().initialize()
        for component_name, manager in self.state_components():
            results[component_name] = manager.initialize()
        return results

    def destroy(self, *, empty_cuda_cache: bool = False) -> dict[str, Any]:
        results = super().destroy(empty_cuda_cache=empty_cuda_cache)
        for component_name, manager in reversed(list(self.state_components())):
            results[component_name] = manager.destroy(
                empty_cuda_cache=empty_cuda_cache
            )
        return results

    def allocate_state_items_for_sequences(
        self, sequence_ids, *, component_name: str
    ):
        manager = _state_manager_for_op(
            self, component_name, "allocate_state_items_for_sequences"
        )
        return manager.allocate_state_items_for_sequences(
            [int(seq_id) for seq_id in sequence_ids]
        )

    def release_sequence_states(
        self, sequence_ids, *, component_name: str | None = None
    ) -> dict[str, Any] | Any:
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        if component_name is not None:
            manager = _state_manager_for_op(
                self, component_name, "release_sequence_states"
            )
            return manager.release_sequence_states(sequence_ids)
        return {
            name: manager.release_sequence_states(sequence_ids)
            for name, manager in self.state_components()
        }

    def prepare_state_decode_step(
        self, sequence_ids, raw_positions, *, component_name: str
    ) -> None:
        manager = _state_manager_for_op(
            self, component_name, "prepare_state_decode_step"
        )
        return manager.prepare_decode_step(
            [int(seq_id) for seq_id in sequence_ids], raw_positions
        )

    def update_layer_decode_state(
        self,
        state_tensor,
        raw_positions,
        layer_idx: int,
        *,
        component_name: str,
        sequence_ids=None,
        batch_slice=None,
        assume_prepared: bool = False,
    ) -> None:
        manager = _state_manager_for_op(
            self, component_name, "update_layer_decode_state"
        )
        return manager.update_layer_decode_state(
            state_tensor,
            raw_positions,
            int(layer_idx),
            sequence_ids=sequence_ids,
            batch_slice=batch_slice,
            assume_prepared=assume_prepared,
        )

    def export_state_item_pointers(self, sequence_ids, *, component_name: str):
        manager = _state_manager_for_op(
            self, component_name, "export_state_item_pointers"
        )
        return manager.export_state_item_pointers(
            [int(seq_id) for seq_id in sequence_ids]
        )

    def get_sequence_layer_state_item_pointer(
        self, sequence_id: int, layer_idx: int, *, component_name: str
    ) -> int:
        manager = _state_manager_for_op(
            self, component_name, "get_sequence_layer_state_item_pointer"
        )
        return manager.get_sequence_layer_state_item_pointer(
            int(sequence_id), int(layer_idx)
        )
