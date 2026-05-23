from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.kv_cache.coordinator_utils import as_int_list
from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)


class PositionMappedGPUPagedKVCacheManager(GPUPagedKVCacheManager):
    """Base class for GPU KV managers that remap raw token positions.

    Subclasses own the model-specific mapping from raw positions/lengths to
    storage positions. This base class keeps the shared prepared-decode and
    page-table bookkeeping out of each variant.
    """

    manager_name = "PositionMappedGPUPagedKVCacheManager"

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        model_config: Optional[ModelConfig] = None,
        *,
        config: Optional[GPUPagedKVConfig] = None,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        super().__init__(
            engine_config=engine_config,
            model_config=model_config,
            config=config,
            device=device,
        )
        self.page_size_tokens = int(self.config.page_size_tokens)
        if self.page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be > 0")

        self._last_page_table_order: Optional[list[int]] = None
        self._page_table_dirty = False
        self._prepared_decode_page_table: Optional[torch.Tensor] = None
        self._prepared_decode_positions: Optional[torch.Tensor] = None
        self._prepared_decode_position_count = 0

    def initialize(self) -> None:
        super().initialize()
        self._ensure_prepared_decode_position_buffer()

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        self._last_page_table_order = None
        self._page_table_dirty = False
        self._prepared_decode_page_table = None
        self._prepared_decode_positions = None
        self._prepared_decode_position_count = 0
        return super().destroy(empty_cuda_cache=empty_cuda_cache)

    def allocate_pages(self, sequence_id: int, num_tokens: int) -> list[int]:
        allocations = self.allocate_pages_for_sequences(
            [int(sequence_id)], [int(num_tokens)]
        )
        return allocations.get(int(sequence_id), [])

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        super().free_pages_for_sequences(sequence_ids)
        self._after_free_pages_for_sequences(sequence_ids)
        if self._last_page_table_order is not None:
            freed = set(sequence_ids)
            self._last_page_table_order = [
                seq_id
                for seq_id in self._last_page_table_order
                if seq_id not in freed
            ]
        self._page_table_dirty = True
        if self._rebuild_after_free_pages():
            self._rebuild_last_page_table_if_dirty()

    def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        self._last_page_table_order = [int(seq_id) for seq_id in sequence_ids]
        table = super().rebuild_page_table(self._last_page_table_order)
        self._page_table_dirty = False
        return table

    def clear_page_table(self) -> None:
        self._last_page_table_order = None
        self._page_table_dirty = False
        self._prepared_decode_page_table = None
        return super().clear_page_table()

    def ensure_cuda_graph_page_table(
        self, sequence_ids: Sequence[int]
    ) -> torch.Tensor:
        self._last_page_table_order = [int(seq_id) for seq_id in sequence_ids]
        if self._page_table_dirty:
            if self._last_page_table_order:
                super().rebuild_page_table(self._last_page_table_order)
            else:
                super().clear_page_table()
            self._page_table_dirty = False
            return super().get_cuda_graph_page_table()
        return super().ensure_cuda_graph_page_table(self._last_page_table_order)

    def prepare_decode_step(
        self,
        sequence_ids: Sequence[int],
        raw_positions: Sequence[int] | torch.Tensor,
        *,
        refresh_page_table: bool = True,
        use_cuda_graph_page_table: bool = False,
    ) -> None:
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        raw_values = as_int_list(raw_positions)
        if len(sequence_ids) != len(raw_values):
            raise ValueError(
                "prepare_decode_step: sequence_ids and raw_positions must "
                "have the same length"
            )

        storage_positions = [
            self._prepare_decode_storage_position(sequence_id, int(raw_pos))
            for sequence_id, raw_pos in zip(sequence_ids, raw_values)
        ]
        self._write_prepared_decode_positions(storage_positions)

        if refresh_page_table:
            self._prepared_decode_page_table = self._refresh_decode_page_table(
                sequence_ids,
                use_cuda_graph_page_table=use_cuda_graph_page_table,
            )
        else:
            self._prepared_decode_page_table = None

    def get_context_kv_page_ptrs(
        self, sequence_id: int, layer_idx: int, context_length: int
    ):
        storage_tokens = self._context_storage_tokens(
            int(sequence_id), int(context_length)
        )
        self._rebuild_last_page_table_if_dirty()
        return super().get_context_kv_page_ptrs(
            int(sequence_id), int(layer_idx), storage_tokens
        )

    def _resolve_decode_update_inputs(
        self,
        *,
        k_tensor: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor],
        batch_slice: Optional[tuple],
        slot_indices: Optional[torch.Tensor],
        assume_prepared: bool,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        batch_size = int(k_tensor.shape[0])
        resolved_slots = self._resolve_slot_indices(
            batch_size=batch_size,
            batch_slice=batch_slice,
            slot_indices=slot_indices,
        )
        if assume_prepared:
            if sequence_lengths is not None:
                raise TypeError(
                    f"{self.manager_name}: sequence_lengths must be None "
                    "when assume_prepared=True"
                )
            storage_positions = self._resolve_prepared_storage_positions(
                batch_size=batch_size,
                batch_slice=batch_slice,
            )
        else:
            raw_positions = self._resolve_raw_positions(
                batch_size=batch_size,
                batch_slice=batch_slice,
                sequence_lengths=sequence_lengths,
            )
            storage_positions = self._prepare_storage_positions(
                raw_positions=raw_positions,
                slot_indices=resolved_slots,
                like=raw_positions,
            )
            self._rebuild_last_page_table_if_dirty()
        return batch_size, resolved_slots, storage_positions

    def _decode_page_table(self, *, assume_prepared: bool) -> torch.Tensor:
        if assume_prepared and self._prepared_decode_page_table is not None:
            page_table = self._prepared_decode_page_table
        else:
            page_table = self._gpu_page_table_manager.gpu_table
        if page_table is None:
            raise RuntimeError(
                f"{self.manager_name}: GPU page table is not initialized"
            )
        return page_table

    def _prepare_decode_storage_position(
        self, sequence_id: int, raw_position: int
    ) -> int:
        raise NotImplementedError

    def _context_storage_tokens(
        self, sequence_id: int, context_length: int
    ) -> int:
        raise NotImplementedError

    def _invalid_slot_storage_position(self) -> int:
        return -1

    def _after_free_pages_for_sequences(
        self, sequence_ids: Sequence[int]
    ) -> None:
        del sequence_ids

    def _rebuild_after_free_pages(self) -> bool:
        return False

    def _allocate_storage_pages(
        self, sequence_id: int, storage_tokens: int
    ) -> list[int]:
        return GPUPagedKVCacheManager.allocate_pages(
            self, int(sequence_id), int(storage_tokens)
        )

    def _grow_storage_sequence_pages(
        self, sequence_id: int, num_pages: int
    ) -> list[int]:
        return GPUPagedKVCacheManager.grow_sequence_pages(
            self, int(sequence_id), int(num_pages)
        )

    def _resolve_slot_indices(
        self,
        *,
        batch_size: int,
        batch_slice: Optional[tuple],
        slot_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        page_table = self._gpu_page_table_manager.gpu_table
        if page_table is None:
            raise RuntimeError(
                f"{self.manager_name}: GPU page table is not initialized"
            )
        if slot_indices is None:
            slot_indices = self._gpu_page_table_manager._slot_index_tensor
            if slot_indices is None:
                raise RuntimeError(
                    f"{self.manager_name}: slot indices are unavailable"
                )
        else:
            slot_indices = slot_indices.to(
                device=page_table.device, dtype=torch.int32
            )
        if batch_slice is not None and slot_indices.shape[0] != batch_size:
            start_idx, end_idx = batch_slice
            slot_indices = slot_indices[start_idx:end_idx]
        if slot_indices.shape[0] != batch_size:
            raise ValueError(
                f"{self.manager_name}: slot_indices must align with batch "
                f"size, got {slot_indices.shape[0]} vs {batch_size}"
            )
        return slot_indices.contiguous()

    def _resolve_raw_positions(
        self,
        *,
        batch_size: int,
        batch_slice: Optional[tuple],
        sequence_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(sequence_lengths, torch.Tensor):
            raise TypeError("sequence_lengths must be a torch.Tensor")
        raw_positions = sequence_lengths
        if batch_slice is not None and raw_positions.shape[0] != batch_size:
            start_idx, end_idx = batch_slice
            raw_positions = raw_positions[start_idx:end_idx]
        if raw_positions.shape[0] != batch_size:
            raise ValueError(
                f"{self.manager_name}: sequence_lengths must align with "
                f"batch size, got {raw_positions.shape[0]} vs {batch_size}"
            )
        return raw_positions.contiguous()

    def _resolve_prepared_storage_positions(
        self,
        *,
        batch_size: int,
        batch_slice: Optional[tuple],
    ) -> torch.Tensor:
        buffer = self._ensure_prepared_decode_position_buffer()
        prepared_count = int(self._prepared_decode_position_count)
        if batch_slice is not None and prepared_count != batch_size:
            start_idx, end_idx = batch_slice
            if start_idx < 0 or end_idx < start_idx or end_idx > prepared_count:
                raise ValueError(
                    f"{self.manager_name}: batch_slice is outside the "
                    f"prepared decode-position range, slice={batch_slice}, "
                    f"prepared={prepared_count}"
                )
            positions = buffer[start_idx:end_idx]
        else:
            if batch_size > prepared_count:
                raise ValueError(
                    f"{self.manager_name}: requested {batch_size} prepared "
                    f"positions, but only {prepared_count} are available"
                )
            positions = buffer[:batch_size]
        if positions.shape[0] != batch_size:
            raise ValueError(
                f"{self.manager_name}: prepared positions must align with "
                f"batch size, got {positions.shape[0]} vs {batch_size}"
            )
        return positions

    def _prepare_storage_positions(
        self,
        *,
        raw_positions: torch.Tensor,
        slot_indices: torch.Tensor,
        like: torch.Tensor,
    ) -> torch.Tensor:
        slot_values = as_int_list(slot_indices)
        raw_values = as_int_list(raw_positions)
        slot_order = self._gpu_page_table_manager.slot_to_seq_id
        storage_positions: list[int] = []
        for slot, raw_pos in zip(slot_values, raw_values):
            if slot < 0:
                storage_positions.append(self._invalid_slot_storage_position())
                continue
            if slot >= len(slot_order):
                raise IndexError(
                    f"slot index {slot} exceeds active slot order length "
                    f"{len(slot_order)}"
                )
            sequence_id = int(slot_order[slot])
            storage_positions.append(
                self._prepare_decode_storage_position(sequence_id, int(raw_pos))
            )
        return torch.as_tensor(
            storage_positions,
            dtype=like.dtype,
            device=like.device,
        )

    def _ensure_prepared_decode_position_buffer(self) -> torch.Tensor:
        if self._prepared_decode_positions is None:
            max_slots = int(self._gpu_page_table_manager.max_slots)
            self._prepared_decode_positions = torch.full(
                (max_slots,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
        return self._prepared_decode_positions

    def _write_prepared_decode_positions(
        self, positions: Sequence[int]
    ) -> None:
        buffer = self._ensure_prepared_decode_position_buffer()
        count = len(positions)
        if count > buffer.numel():
            raise ValueError(
                "prepare_decode_step: active batch size exceeds the prepared "
                f"decode-position buffer capacity, batch={count}, "
                f"capacity={buffer.numel()}"
            )
        if count:
            buffer[:count].copy_(
                torch.as_tensor(
                    positions,
                    dtype=buffer.dtype,
                    device=buffer.device,
                )
            )
        previous_count = self._prepared_decode_position_count
        if previous_count > count:
            buffer[count:previous_count].fill_(-1)
        self._prepared_decode_position_count = count

    def _rebuild_last_page_table_if_dirty(self) -> None:
        if not self._page_table_dirty:
            return
        order = self._last_page_table_order
        if order is None:
            order = list(self._gpu_page_table_manager.slot_to_seq_id)
        if order:
            self._last_page_table_order = [int(seq_id) for seq_id in order]
            super().rebuild_page_table(self._last_page_table_order)
        else:
            super().clear_page_table()
        self._page_table_dirty = False

    def _refresh_decode_page_table(
        self,
        sequence_ids: Sequence[int],
        *,
        use_cuda_graph_page_table: bool,
    ) -> Optional[torch.Tensor]:
        order = [int(seq_id) for seq_id in sequence_ids]
        if use_cuda_graph_page_table:
            return self.ensure_cuda_graph_page_table(order)

        manager = self._gpu_page_table_manager
        if not order:
            if (
                self._page_table_dirty
                or manager.gpu_table is None
                or manager.slot_to_seq_id
            ):
                self.clear_page_table()
            else:
                self._last_page_table_order = []
            return manager.gpu_table

        if (
            self._page_table_dirty
            or manager.gpu_table is None
            or manager.slot_to_seq_id != order
        ):
            return self.rebuild_page_table(order)

        self._last_page_table_order = order
        return manager.gpu_table


__all__ = ["PositionMappedGPUPagedKVCacheManager"]
