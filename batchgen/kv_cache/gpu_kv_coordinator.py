"""Lightweight registry for GPU-side heterogeneous KV managers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


class GPUKVCoordinator:
    """Component-aware facade for GPU-side heterogeneous KV managers."""

    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}

    def register_component(
        self,
        component_name: str,
        manager: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise ValueError(
                "GPUKVCoordinator does not accept component metadata; "
                "configure the backing GPUPagedKVCacheManager instead"
            )
        if not component_name:
            raise ValueError("GPU KV component name must be non-empty")
        if manager is None:
            raise ValueError(
                f"GPU KV component {component_name!r}: manager must be set"
            )
        if component_name in self._components:
            raise ValueError(
                f"GPU KV component already registered: {component_name}"
            )
        self._components[component_name] = manager
        setattr(self, component_name, manager)
        return manager

    @property
    def component_names(self) -> list[str]:
        return list(self._components.keys())

    def components(self) -> Iterator[tuple[str, Any]]:
        return iter(self._components.items())

    def get_component(self, name: str) -> Any:
        try:
            return self._components[name]
        except KeyError as exc:
            raise KeyError(f"Unknown GPU KV component: {name}") from exc

    def get_manager(self, name: str) -> Any:
        return self.get_component(name)

    def call_all(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Call the same method on every backing manager."""

        results: dict[str, Any] = {}
        for component_name, manager in self.components():
            method = getattr(manager, method_name)
            results[component_name] = method(*args, **kwargs)
        return results

    def _component_for_op(self, component_name: str, context: str) -> Any:
        if component_name is None:
            raise KeyError(f"{context}: component_name is required")
        try:
            return self.get_component(component_name)
        except KeyError as exc:
            raise KeyError(
                f"{context}: unknown GPU KV component {component_name!r}"
            ) from exc

    def initialize(self) -> dict[str, Any]:
        return self.call_all("initialize")

    def destroy(self, *, empty_cuda_cache: bool = False) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for component_name, manager in reversed(list(self.components())):
            results[component_name] = manager.destroy(
                empty_cuda_cache=empty_cuda_cache
            )
        return results

    @property
    def is_initialized(self) -> bool:
        return all(
            bool(getattr(manager, "is_initialized", False))
            for _, manager in self.components()
        )

    def allocate_pages(
        self,
        sequence_id: int,
        num_tokens: int,
        *,
        component_name: Optional[str] = None,
    ):
        if component_name is not None:
            manager = self._component_for_op(component_name, "allocate_pages")
            return manager.allocate_pages(int(sequence_id), int(num_tokens))

        return self.allocate_pages_for_sequences([sequence_id], [num_tokens])

    def allocate_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        num_tokens: Sequence[int],
        *,
        component_name: Optional[str] = None,
    ):
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        num_tokens = [int(tokens) for tokens in num_tokens]
        if component_name is not None:
            manager = self._component_for_op(
                component_name, "allocate_pages_for_sequences"
            )
            return manager.allocate_pages_for_sequences(
                sequence_ids, num_tokens
            )

        results: dict[str, Any] = {}
        allocated: list[tuple[str, Any, Any]] = []
        try:
            for component_name, manager in self.components():
                result = manager.allocate_pages_for_sequences(
                    sequence_ids, num_tokens
                )
                results[component_name] = result
                allocated.append((component_name, manager, result))
        except Exception:
            for component_name, manager, allocations in reversed(allocated):
                try:
                    _rollback_gpu_allocations(manager, allocations)
                except Exception:
                    logger.exception(
                        "Failed to rollback GPU KV allocation for %s on %s",
                        sequence_ids[:10],
                        component_name,
                    )
            raise
        return results

    def grow_sequence_pages(
        self,
        sequence_id: int,
        num_pages: int,
        *,
        component_name: Optional[str] = None,
    ):
        if component_name is not None:
            manager = self._component_for_op(
                component_name, "grow_sequence_pages"
            )
            return manager.grow_sequence_pages(int(sequence_id), int(num_pages))
        return self.grow_pages_for_sequences([sequence_id], [num_pages])

    def grow_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        num_pages: Sequence[int],
        *,
        component_name: Optional[str] = None,
    ):
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        num_pages = [int(count) for count in num_pages]
        if component_name is not None:
            manager = self._component_for_op(
                component_name, "grow_pages_for_sequences"
            )
            return manager.grow_pages_for_sequences(sequence_ids, num_pages)
        return self.call_all(
            "grow_pages_for_sequences", sequence_ids, num_pages
        )

    def extend_pages_for_sequence(
        self,
        sequence_id: int,
        new_total_tokens: int,
        *,
        component_name: Optional[str] = None,
    ):
        if component_name is not None:
            manager = self._component_for_op(
                component_name, "extend_pages_for_sequence"
            )
            return manager.extend_pages_for_sequence(
                int(sequence_id), int(new_total_tokens)
            )
        results: dict[str, Any] = {}
        for component_name, manager in self.components():
            results[component_name] = manager.extend_pages_for_sequence(
                int(sequence_id), int(new_total_tokens)
            )
        return results

    def free_pages_for_sequences(
        self, sequence_ids: Sequence[int]
    ) -> dict[str, Any]:
        return self.call_all(
            "free_pages_for_sequences", [int(seq_id) for seq_id in sequence_ids]
        )

    def rebuild_page_table(self, sequence_ids: Sequence[int]):
        return self.call_all(
            "rebuild_page_table", [int(seq_id) for seq_id in sequence_ids]
        )

    def clear_page_table(self) -> dict[str, Any]:
        return self.call_all("clear_page_table")

    def get_page_table_version(self, *, component_name: str) -> int:
        manager = self._component_for_op(
            component_name, "get_page_table_version"
        )
        return manager.get_page_table_version()

    def get_cuda_graph_page_table(self, *, component_name: str):
        manager = self._component_for_op(
            component_name, "get_cuda_graph_page_table"
        )
        return manager.get_cuda_graph_page_table()

    def get_cuda_graph_page_table_storage(self, *, component_name: str):
        manager = self._component_for_op(
            component_name, "get_cuda_graph_page_table_storage"
        )
        return manager.get_cuda_graph_page_table_storage()

    def ensure_cuda_graph_page_table(
        self,
        sequence_ids: Sequence[int],
        *,
        component_name: str,
    ):
        manager = self._component_for_op(
            component_name, "ensure_cuda_graph_page_table"
        )
        return manager.ensure_cuda_graph_page_table(
            [int(seq_id) for seq_id in sequence_ids]
        )

    def prepare_decode_step(
        self,
        sequence_ids: Sequence[int],
        raw_positions,
        *,
        component_name: str,
        **kwargs,
    ):
        manager = self._component_for_op(component_name, "prepare_decode_step")
        prepare = getattr(manager, "prepare_decode_step")
        return prepare(
            [int(seq_id) for seq_id in sequence_ids],
            raw_positions,
            **kwargs,
        )

    def get_cuda_graph_page_table_state(self, *, component_name: str):
        manager = self._component_for_op(
            component_name, "get_cuda_graph_page_table_state"
        )
        return manager.get_cuda_graph_page_table_state()

    def get_stats(self):
        return self.call_all("get_stats")

    def get_stats_by_component(self) -> dict[str, Any]:
        return self.get_stats()

    def get_kv_tensors(self, *, component_name: str):
        manager = self._component_for_op(component_name, "get_kv_tensors")
        return manager.get_kv_tensors()

    def get_layer_kv_with_page_table(
        self,
        layer_idx: int,
        *,
        component_name: str,
    ):
        manager = self._component_for_op(
            component_name, "get_layer_kv_with_page_table"
        )
        return manager.get_layer_kv_with_page_table(int(layer_idx))

    def update_layer_decode_new_token(
        self,
        k_tensor,
        v_tensor,
        sequence_lengths,
        layer_idx: int,
        batch_slice: Optional[tuple] = None,
        slot_indices=None,
        *,
        component_name: str,
        **kwargs,
    ) -> None:
        manager = self._component_for_op(
            component_name, "update_layer_decode_new_token"
        )
        return manager.update_layer_decode_new_token(
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            sequence_lengths=sequence_lengths,
            layer_idx=int(layer_idx),
            batch_slice=batch_slice,
            slot_indices=slot_indices,
            **kwargs,
        )

    def get_context_kv_page_ptrs(
        self,
        sequence_id: int,
        layer_idx: int,
        context_length: int,
        *,
        component_name: str,
    ):
        manager = self._component_for_op(
            component_name, "get_context_kv_page_ptrs"
        )
        return manager.get_context_kv_page_ptrs(
            int(sequence_id),
            int(layer_idx),
            int(context_length),
        )

    def get_sequence_layer_page_pointers(
        self,
        sequence_id: int,
        layer_idx: int,
        *,
        component_name: str,
    ):
        manager = self._component_for_op(
            component_name, "get_sequence_layer_page_pointers"
        )
        return manager.get_sequence_layer_page_pointers(
            int(sequence_id), int(layer_idx)
        )

    def export_layer_page_pointer_table(self, *, component_name: str):
        manager = self._component_for_op(
            component_name, "export_layer_page_pointer_table"
        )
        return manager.export_layer_page_pointer_table()

    def export_active_sequence_page_counts(self, *, component_name: str):
        manager = self._component_for_op(
            component_name, "export_active_sequence_page_counts"
        )
        return manager.export_active_sequence_page_counts()

    def get_padded_3d_page_pointers(self, *, component_name: str):
        manager = self._component_for_op(
            component_name, "get_padded_3d_page_pointers"
        )
        return manager.get_padded_3d_page_pointers()

    def copy_kv_to_tensor(self, sequence_id: int, *, component_name: str):
        manager = self._component_for_op(component_name, "copy_kv_to_tensor")
        return manager.copy_kv_to_tensor(int(sequence_id))

    def copy_tensor_to_kv(
        self,
        sequence_id: int,
        k_tensor,
        *,
        component_name: str,
    ) -> None:
        manager = self._component_for_op(component_name, "copy_tensor_to_kv")
        return manager.copy_tensor_to_kv(int(sequence_id), k_tensor)


def _rollback_gpu_allocations(manager: Any, allocations: Any) -> None:
    if not isinstance(allocations, Mapping):
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
                f"tail={tail}, allocated={pages}"
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
