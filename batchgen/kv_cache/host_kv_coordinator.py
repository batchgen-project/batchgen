"""Generic coordinator for host-side heterogeneous KV components.

The coordinator keeps model-specific KV composition out of call sites.  Each
component still owns a normal HostPagedKVWorkerView, so existing C++ storage and
copy paths remain the bottom layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


LayerMapping = Mapping[int, int] | Sequence[int]
TokenCapacityFn = Callable[[int], int]


@dataclass
class HostKVComponent:
    """One logical host KV component backed by one worker view.

    ``logical_to_physical_layer`` is coordinator-owned routing.  Use it when the
    backing view stores only the compact physical layers for this component, for
    example C4 layers ``{2: 0, 5: 1, 8: 2}``.  If it is omitted, layer ids are
    passed to the view unchanged; mapped C++ worker views can then resolve the
    logical layer internally.

    ``token_capacity_scale`` and ``token_capacity_fn`` convert original sequence
    token counts into this component's token capacity during allocation.  The
    function wins over the scale when both are supplied.
    """

    name: str
    view: Any
    logical_to_physical_layer: Optional[LayerMapping] = None
    token_capacity_scale: float = 1.0
    token_capacity_fn: Optional[TokenCapacityFn] = None
    is_paged: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HostKVComponent.name must be non-empty")
        if self.view is None:
            raise ValueError(f"HostKVComponent({self.name}): view must be set")
        if self.token_capacity_scale <= 0:
            raise ValueError(
                f"HostKVComponent({self.name}): token_capacity_scale must be > 0"
            )

    def token_capacity(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            raise ValueError(
                f"HostKVComponent({self.name}): num_tokens must be > 0, got {num_tokens}"
            )
        if self.token_capacity_fn is not None:
            capacity = int(self.token_capacity_fn(int(num_tokens)))
        else:
            capacity = int(math.ceil(int(num_tokens) * self.token_capacity_scale))
        if capacity <= 0:
            raise ValueError(
                f"HostKVComponent({self.name}): token capacity must be > 0, got {capacity}"
            )
        return capacity

    def resolve_physical_layer(self, logical_layer_id: int) -> int:
        if logical_layer_id < 0:
            raise IndexError(
                f"HostKVComponent({self.name}): logical layer id must be >= 0"
            )
        if self.logical_to_physical_layer is not None:
            return _resolve_from_mapping(
                self.name, self.logical_to_physical_layer, logical_layer_id
            )
        resolver = getattr(self.view, "resolve_physical_layer", None)
        if resolver is not None:
            return int(resolver(logical_layer_id))
        return int(logical_layer_id)

    def view_layer_id(self, logical_layer_id: int) -> int:
        """Layer id to pass into the backing view for layer-specific calls."""

        if self.logical_to_physical_layer is not None:
            return self.resolve_physical_layer(logical_layer_id)
        return int(logical_layer_id)

    def scaled_sequence_tokens(
        self, seq_token_pairs: Iterable[tuple[int, int]]
    ) -> List[tuple[int, int]]:
        return [
            (int(seq_id), self.token_capacity(int(num_tokens)))
            for seq_id, num_tokens in seq_token_pairs
        ]


class AsyncKVTaskGroup:
    """Composite task returned by multi-component async operations."""

    def __init__(self, tasks: Mapping[str, Any], tensors: Any = None) -> None:
        if not tasks:
            raise RuntimeError("AsyncKVTaskGroup requires at least one task")
        self.tasks = dict(tasks)
        self.tensors = tensors

    def wait(self) -> None:
        errors: List[tuple[str, Exception]] = []
        for name, task in self.tasks.items():
            try:
                task.wait()
            except Exception as exc:  # pragma: no cover - exercised by C++ task failures
                errors.append((name, exc))
        if not errors:
            return
        if len(errors) == 1:
            raise errors[0][1]
        names = ", ".join(name for name, _ in errors)
        raise RuntimeError(f"AsyncKVTaskGroup wait failed for components: {names}") from errors[0][1]


class HostKVCoordinator:
    """Registers and coordinates multiple HostPagedKVWorkerView components."""

    def __init__(self, primary_component_name: str = "primary") -> None:
        self._components: Dict[str, HostKVComponent] = {}
        self._primary_component_name = primary_component_name

    @property
    def component_names(self) -> List[str]:
        return list(self._components.keys())

    @property
    def primary_component_name(self) -> str:
        return self._primary_component_name

    @property
    def primary_component(self) -> HostKVComponent:
        return self.get_component(self._primary_component_name)

    @property
    def primary_view(self) -> Any:
        return self.primary_component.view

    def register_component(
        self,
        component: HostKVComponent | str,
        view: Any = None,
        **kwargs: Any,
    ) -> HostKVComponent:
        if isinstance(component, HostKVComponent):
            if view is not None or kwargs:
                raise ValueError(
                    "Pass either a HostKVComponent or name/view/kwargs, not both"
                )
            item = component
        else:
            item = HostKVComponent(name=component, view=view, **kwargs)
        if item.name in self._components:
            raise ValueError(f"Host KV component already registered: {item.name}")
        self._components[item.name] = item
        return item

    def get_component(self, name: str) -> HostKVComponent:
        try:
            return self._components[name]
        except KeyError as exc:
            raise KeyError(f"Unknown host KV component: {name}") from exc

    def get_view(self, name: str) -> Any:
        return self.get_component(name).view

    def resolve_physical_layer(self, component_name: str, logical_layer_id: int) -> int:
        return self.get_component(component_name).resolve_physical_layer(logical_layer_id)

    def _iter_components(self) -> Iterable[HostKVComponent]:
        if not self._components:
            raise RuntimeError("HostKVCoordinator has no registered components")
        return self._components.values()

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        for component in self._iter_components():
            component.view.initialize(*args, **kwargs)

    def shutdown(self) -> None:
        for component in self._iter_components():
            shutdown = getattr(component.view, "shutdown", None)
            if shutdown is not None:
                shutdown()

    def register_sequences(self, sequence_ids: Sequence[int]) -> None:
        for component in self._iter_components():
            component.view.register_sequences(sequence_ids)

    def unregister_sequences(self, sequence_ids: Sequence[int]) -> None:
        for component in self._iter_components():
            component.view.unregister_sequences(sequence_ids)

    def release_sequence_pages(self, sequence_ids: Sequence[int]) -> None:
        for component in self._iter_components():
            component.view.release_sequence_pages(sequence_ids)

    def allocate_pages_for_sequences(
        self, seq_token_pairs: Iterable[tuple[int, int]]
    ) -> Any:
        pairs = list(seq_token_pairs)
        primary_result = None
        for component in self._iter_components():
            result = component.view.allocate_pages_for_sequences(
                component.scaled_sequence_tokens(pairs)
            )
            if component.name == self._primary_component_name:
                primary_result = result
        return primary_result

    def grow_pages_for_sequences(
        self, seq_page_pairs: Iterable[tuple[int, int]]
    ) -> Any:
        pairs = [(int(seq_id), int(pages)) for seq_id, pages in seq_page_pairs]
        primary_result = None
        for component in self._iter_components():
            result = component.view.grow_pages_for_sequences(pairs)
            if component.name == self._primary_component_name:
                primary_result = result
        return primary_result

    def build_page_table(
        self, sequence_ids: Sequence[int], component_name: Optional[str] = None
    ) -> Any:
        component = self.get_component(component_name or self._primary_component_name)
        return component.view.build_page_table(sequence_ids)

    def build_component_page_tables(self, sequence_ids: Sequence[int]) -> Dict[str, Any]:
        return {
            component.name: component.view.build_page_table(sequence_ids)
            for component in self._iter_components()
        }

    def get_stats(self) -> Any:
        return self.primary_view.get_stats()

    def get_component_stats(self) -> Dict[str, Any]:
        return {
            component.name: component.view.get_stats()
            for component in self._iter_components()
        }

    def get_sequence_layer_page_pointers(
        self,
        component_name: str,
        sequence_id: int,
        logical_layer_id: int,
        max_tokens: Optional[int] = None,
    ) -> Any:
        component = self.get_component(component_name)
        return component.view.get_sequence_layer_page_pointers(
            sequence_id, component.view_layer_id(logical_layer_id), max_tokens
        )

    def async_offload_component_layer_to_host(
        self,
        component_name: str,
        *,
        layer_idx: int,
        sequence_ids: Sequence[int],
        k_tensor: Any,
        v_tensor: Any = None,
        sequence_lengths: Any,
    ) -> Any:
        component = self.get_component(component_name)
        return component.view.async_offload_layer_kv_to_host(
            component.view_layer_id(layer_idx),
            sequence_ids,
            k_tensor,
            v_tensor,
            sequence_lengths,
        )

    def async_append_decode_component_to_host(
        self,
        component_name: str,
        *,
        layer_idx: int,
        sequence_ids: Sequence[int],
        k_tensor: Any,
        v_tensor: Any = None,
        sequence_lengths: Any,
    ) -> Any:
        component = self.get_component(component_name)
        return component.view.async_append_decode_kv_to_host(
            component.view_layer_id(layer_idx),
            sequence_ids,
            k_tensor,
            v_tensor,
            sequence_lengths,
        )

    def async_load_component_paged_kv_to_device(
        self,
        component_name: str,
        *,
        sequence_ids: Any,
        active_page_counts: Any,
        k_device_ptrs: Any,
        v_device_ptrs: Any = None,
    ) -> Any:
        component = self.get_component(component_name)
        return component.view.async_load_layer_paged_kv_to_device(
            sequence_ids=sequence_ids,
            active_page_counts=active_page_counts,
            k_device_ptrs=k_device_ptrs,
            v_device_ptrs=v_device_ptrs,
        )

    def async_load_components_paged_kv_to_device(
        self,
        loads: Mapping[str, Mapping[str, Any]],
        *,
        tensors: Any = None,
    ) -> AsyncKVTaskGroup:
        tasks: Dict[str, Any] = {}
        for name, kwargs in loads.items():
            tasks[name] = self.async_load_component_paged_kv_to_device(
                name, **dict(kwargs)
            )
        return AsyncKVTaskGroup(tasks, tensors=tensors)


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
            f"Host KV component {component_name!r} has no physical layer for "
            f"logical layer {logical_layer_id}"
        )
    return int(physical)
