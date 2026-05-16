"""Generic coordinator for GPU-side heterogeneous paged KV components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager, GPUPagedKVStats


LayerMapping = Mapping[int, int] | Sequence[int]
TokenCapacityFn = Callable[[int], int]


@dataclass
class GPUKVComponent:
    """One GPU KV component backed by one GPUPagedKVCacheManager."""

    name: str
    manager: GPUPagedKVCacheManager
    logical_to_physical_layer: Optional[LayerMapping] = None
    token_capacity_scale: float = 1.0
    token_capacity_fn: Optional[TokenCapacityFn] = None
    is_paged: bool = True

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

    def scaled_tokens(self, token_counts: Sequence[int]) -> List[int]:
        return [self.token_capacity(int(count)) for count in token_counts]

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


class GPUKVCoordinator:
    """Coordinates multiple homogeneous GPU paged KV managers."""

    def __init__(self, primary_component_name: str = "primary") -> None:
        self._components: Dict[str, GPUKVComponent] = {}
        self._primary_component_name = primary_component_name

    @property
    def component_names(self) -> List[str]:
        return list(self._components.keys())

    @property
    def primary_component_name(self) -> str:
        return self._primary_component_name

    @property
    def primary_component(self) -> GPUKVComponent:
        return self.get_component(self._primary_component_name)

    @property
    def primary(self) -> GPUPagedKVCacheManager:
        return self.primary_component.manager

    @property
    def config(self):
        return self.primary.config

    @property
    def device(self):
        return self.primary.device

    @property
    def is_initialized(self) -> bool:
        return all(component.manager.is_initialized for component in self._iter_components())

    @property
    def _gpu_page_table_manager(self):
        return self.primary._gpu_page_table_manager

    @property
    def _sequences(self):
        return self.primary._sequences

    def register_component(
        self,
        component: GPUKVComponent | str,
        manager: Optional[GPUPagedKVCacheManager] = None,
        **kwargs: Any,
    ) -> GPUKVComponent:
        if isinstance(component, GPUKVComponent):
            if manager is not None or kwargs:
                raise ValueError(
                    "Pass either a GPUKVComponent or name/manager/kwargs, not both"
                )
            item = component
        else:
            if manager is None:
                raise ValueError("manager must be provided when registering by name")
            item = GPUKVComponent(name=component, manager=manager, **kwargs)
        if item.name in self._components:
            raise ValueError(f"GPU KV component already registered: {item.name}")
        self._components[item.name] = item
        return item

    def get_component(self, name: str) -> GPUKVComponent:
        try:
            return self._components[name]
        except KeyError as exc:
            raise KeyError(f"Unknown GPU KV component: {name}") from exc

    def get_manager(self, name: str) -> GPUPagedKVCacheManager:
        return self.get_component(name).manager

    def resolve_physical_layer(self, component_name: str, logical_layer_id: int) -> int:
        return self.get_component(component_name).resolve_physical_layer(logical_layer_id)

    def _iter_components(self) -> Iterable[GPUKVComponent]:
        if not self._components:
            raise RuntimeError("GPUKVCoordinator has no registered components")
        return self._components.values()

    def initialize(self) -> None:
        for component in self._iter_components():
            component.manager.initialize()

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        for component in self._iter_components():
            component.manager.destroy(empty_cuda_cache=empty_cuda_cache)

    def allocate_pages(self, sequence_id: int, num_tokens: int) -> List[int]:
        result = self.allocate_pages_for_sequences([sequence_id], [num_tokens])
        return result.get(sequence_id, [])

    def allocate_pages_for_sequences(
        self, sequence_ids: Sequence[int], num_tokens: Sequence[int]
    ) -> Dict[int, List[int]]:
        if len(sequence_ids) != len(num_tokens):
            raise ValueError(
                "allocate_pages_for_sequences: sequence_ids and num_tokens must have the same length"
            )
        self._preflight_allocate(sequence_ids, num_tokens, "allocate_pages_for_sequences")
        allocations_by_component: Dict[str, Dict[int, List[int]]] = {}
        try:
            for component in self._iter_components():
                allocations_by_component[component.name] = (
                    component.manager.allocate_pages_for_sequences(
                        sequence_ids, component.scaled_tokens(num_tokens)
                    )
                )
        except Exception:
            for component in self._iter_components():
                self._rollback_allocations(
                    component.manager, allocations_by_component.get(component.name)
                )
            raise
        return allocations_by_component.get(self._primary_component_name, {})

    def grow_sequence_pages(self, sequence_id: int, num_pages: int) -> List[int]:
        result = self.grow_pages_for_sequences([sequence_id], [num_pages])
        return result.get(sequence_id, [])

    def grow_pages_for_sequences(
        self, sequence_ids: Sequence[int], num_pages: Sequence[int]
    ) -> Dict[int, List[int]]:
        self._preflight_grow(sequence_ids, num_pages, "grow_pages_for_sequences")
        allocations_by_component: Dict[str, Dict[int, List[int]]] = {}
        try:
            for component in self._iter_components():
                allocations_by_component[component.name] = (
                    component.manager.grow_pages_for_sequences(sequence_ids, num_pages)
                )
        except Exception:
            for component in self._iter_components():
                self._rollback_allocations(
                    component.manager, allocations_by_component.get(component.name)
                )
            raise
        return allocations_by_component.get(self._primary_component_name, {})

    def extend_pages_for_sequence(self, sequence_id: int, new_total_tokens: int) -> int:
        added_by_component: Dict[str, int] = {}
        try:
            for component in self._iter_components():
                added_by_component[component.name] = component.manager.extend_pages_for_sequence(
                    sequence_id, component.token_capacity(new_total_tokens)
                )
        except Exception:
            raise
        return int(added_by_component.get(self._primary_component_name, 0))

    def rebuild_page_table(self, sequence_ids: Sequence[int]) -> Any:
        primary_table = None
        for component in self._iter_components():
            table = component.manager.rebuild_page_table(sequence_ids)
            if component.name == self._primary_component_name:
                primary_table = table
        return primary_table

    def clear_page_table(self) -> None:
        for component in self._iter_components():
            component.manager.clear_page_table()

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        for component in self._iter_components():
            component.manager.free_pages_for_sequences(sequence_ids)

    def get_stats(self) -> GPUPagedKVStats:
        return self.primary.get_stats()

    def get_component_stats(self) -> Dict[str, GPUPagedKVStats]:
        return {
            component.name: component.manager.get_stats()
            for component in self._iter_components()
        }

    def get_kv_tensors(self):
        return self.primary.get_kv_tensors()

    def get_layer_kv_with_page_table(
        self,
        component_name_or_layer_idx: str | int,
        logical_layer_id: Optional[int] = None,
    ):
        if logical_layer_id is None:
            component_name = self._primary_component_name
            logical_layer_id = int(component_name_or_layer_idx)
        else:
            component_name = str(component_name_or_layer_idx)
        component = self.get_component(component_name)
        physical_layer = component.resolve_physical_layer(int(logical_layer_id))
        return component.manager.get_layer_kv_with_page_table(physical_layer)

    def update_layer_decode_new_token(self, *args: Any, **kwargs: Any) -> Any:
        component_name = kwargs.pop("component_name", self._primary_component_name)
        if args and isinstance(args[0], str):
            component_name = args[0]
            args = args[1:]
        component = self.get_component(component_name)

        if "layer_idx" in kwargs:
            kwargs["layer_idx"] = component.resolve_physical_layer(int(kwargs["layer_idx"]))
            return component.manager.update_layer_decode_new_token(*args, **kwargs)

        if len(args) < 4:
            raise TypeError(
                "update_layer_decode_new_token requires layer_idx as a keyword or "
                "the fourth positional argument"
            )
        mutable_args = list(args)
        mutable_args[3] = component.resolve_physical_layer(int(mutable_args[3]))
        return component.manager.update_layer_decode_new_token(*mutable_args, **kwargs)

    def get_page_table_version(self) -> int:
        return self.primary.get_page_table_version()

    def get_cuda_graph_page_table(self):
        return self.primary.get_cuda_graph_page_table()

    def get_cuda_graph_page_table_storage(self):
        return self.primary.get_cuda_graph_page_table_storage()

    def ensure_cuda_graph_page_table(self, sequence_ids: Sequence[int]):
        return self.primary.ensure_cuda_graph_page_table(sequence_ids)

    def get_cuda_graph_page_table_state(self):
        return self.primary.get_cuda_graph_page_table_state()

    def copy_kv_to_tensor(self, sequence_id: int):
        return self.primary.copy_kv_to_tensor(sequence_id)

    def copy_tensor_to_kv(self, sequence_id: int, k_tensor: Any) -> None:
        self.primary.copy_tensor_to_kv(sequence_id, k_tensor)

    def get_context_kv_page_ptrs(self, *args: Any, **kwargs: Any):
        return self.primary.get_context_kv_page_ptrs(*args, **kwargs)

    def get_sequence_layer_page_pointers(self, *args: Any, **kwargs: Any):
        return self.primary.get_sequence_layer_page_pointers(*args, **kwargs)

    def export_layer_page_pointer_table(self, *args: Any, **kwargs: Any):
        return self.primary.export_layer_page_pointer_table(*args, **kwargs)

    def export_active_sequence_page_counts(self):
        return self.primary.export_active_sequence_page_counts()

    def get_padded_3d_page_pointers(self, *args: Any, **kwargs: Any):
        return self.primary.get_padded_3d_page_pointers(*args, **kwargs)

    def _preflight_allocate(
        self, sequence_ids: Sequence[int], num_tokens: Sequence[int], op_name: str
    ) -> None:
        for component in self._iter_components():
            manager = component.manager
            manager._ensure_initialized()
            component_tokens = component.scaled_tokens(num_tokens)
            required_pages = manager._geometry.required_pages(component_tokens).tolist()
            missing_total = 0
            for seq_id, required in zip(sequence_ids, required_pages):
                required_int = int(required)
                state = manager._sequences.get(seq_id)
                current = int(state.pages.numel()) if state else 0
                missing_total += max(0, required_int - current)
            if missing_total > manager._free_pages.size:
                raise RuntimeError(
                    f"{op_name}: insufficient free pages for GPU KV component "
                    f"{component.name!r}: need {missing_total}, "
                    f"free {manager._free_pages.size}"
                )

    def _preflight_grow(
        self, sequence_ids: Sequence[int], num_pages: Sequence[int], op_name: str
    ) -> None:
        if len(sequence_ids) != len(num_pages):
            raise ValueError(f"{op_name}: sequence_ids and num_pages must match")
        needed = sum(int(count) for count in num_pages)
        for component in self._iter_components():
            manager = component.manager
            manager._ensure_initialized()
            missing = [seq_id for seq_id in sequence_ids if seq_id not in manager._sequences]
            if missing:
                raise KeyError(
                    f"{op_name}: component {component.name!r} missing sequences: "
                    + ", ".join(str(seq_id) for seq_id in missing[:10])
                )
            if needed > manager._free_pages.size:
                raise RuntimeError(
                    f"{op_name}: insufficient free pages for GPU KV component "
                    f"{component.name!r}: need {needed}, free {manager._free_pages.size}"
                )

    def _rollback_allocations(
        self,
        manager: GPUPagedKVCacheManager,
        allocations: Optional[Mapping[int, Sequence[int]]],
    ) -> None:
        if not allocations:
            return
        reclaimed = []
        for seq_id, pages in allocations.items():
            if not pages:
                continue
            state = manager._sequences.get(seq_id)
            if state is None:
                continue
            count = len(pages)
            tail = state.pages[-count:].tolist()
            if tail != list(pages):
                raise RuntimeError(
                    f"Cannot rollback GPU KV allocation for seq {seq_id}: "
                    f"tail={tail}, allocated={list(pages)}"
                )
            reclaimed.append(state.pages[-count:].clone())
            if state.pages.numel() == count:
                del manager._sequences[seq_id]
            else:
                state.pages = state.pages[:-count].clone()
        if reclaimed:
            import torch

            manager._free_pages.push(torch.cat(reclaimed, dim=0))
            manager._clear_active_page_pointer_tables()


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
