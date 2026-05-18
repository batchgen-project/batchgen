from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.kv_cache.coordinator_utils import as_int_list, ceil_div
from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)


class CompressedRatioGPUPagedKVCacheManager(GPUPagedKVCacheManager):
    """GPU paged KV manager for floor-compressed token streams.

    The base manager still owns storage, page allocation, page tables, and the
    token update kernel. This wrapper only maps raw sequence lengths to
    ``raw_length // compression_ratio`` and filters decode steps that have not
    produced a full compressed KV token yet.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        model_config: Optional[ModelConfig] = None,
        *,
        config: Optional[GPUPagedKVConfig] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        compression_ratio: int,
    ) -> None:
        super().__init__(
            engine_config=engine_config,
            model_config=model_config,
            config=config,
            device=device,
        )
        self.compression_ratio = int(compression_ratio)
        if self.compression_ratio <= 0:
            raise ValueError("compression_ratio must be > 0")
        self.page_size_tokens = int(self.config.page_size_tokens)
        if self.page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be > 0")

        self._last_page_table_order: Optional[list[int]] = None
        self._page_table_dirty = False
        self._prepared_decode_positions: Optional[torch.Tensor] = None
        self._prepared_decode_position_count = 0

    def initialize(self) -> None:
        super().initialize()
        self._ensure_prepared_decode_position_buffer()

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        self._last_page_table_order = None
        self._page_table_dirty = False
        self._prepared_decode_positions = None
        self._prepared_decode_position_count = 0
        return super().destroy(empty_cuda_cache=empty_cuda_cache)

    def allocate_pages(self, sequence_id: int, num_tokens: int) -> list[int]:
        allocations = self.allocate_pages_for_sequences(
            [int(sequence_id)], [int(num_tokens)]
        )
        return allocations.get(int(sequence_id), [])

    def allocate_pages_for_sequences(
        self, sequence_ids: Sequence[int], num_tokens: Sequence[int]
    ) -> dict[int, list[int]]:
        if len(sequence_ids) != len(num_tokens):
            raise ValueError(
                "allocate_pages_for_sequences: sequence_ids and num_tokens "
                "must have the same length"
            )
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        storage_tokens = [
            self._storage_capacity_tokens(self._compressed_tokens(tokens))
            for tokens in num_tokens
        ]
        allocations = super().allocate_pages_for_sequences(
            sequence_ids, storage_tokens
        )
        if allocations:
            self._page_table_dirty = True
        return allocations

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        super().free_pages_for_sequences(sequence_ids)
        if self._last_page_table_order is not None:
            freed = set(sequence_ids)
            self._last_page_table_order = [
                seq_id
                for seq_id in self._last_page_table_order
                if seq_id not in freed
            ]
        self._page_table_dirty = True

    def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        self._last_page_table_order = [int(seq_id) for seq_id in sequence_ids]
        table = super().rebuild_page_table(self._last_page_table_order)
        self._page_table_dirty = False
        return table

    def clear_page_table(self) -> None:
        self._last_page_table_order = None
        self._page_table_dirty = False
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
        """Prepare compressed decode writes before the update kernel.

        ``raw_positions`` are uncompressed decode write positions. Positions
        that do not finish a compression group are recorded as ``-1`` and will
        be skipped by :meth:`update_layer_decode_new_token`.
        """

        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        raw_values = as_int_list(raw_positions)
        if len(sequence_ids) != len(raw_values):
            raise ValueError(
                "prepare_decode_step: sequence_ids and raw_positions must "
                "have the same length"
            )

        storage_positions: list[int] = []
        for sequence_id, raw_pos in zip(sequence_ids, raw_values):
            raw_end = self._checked_raw_end(raw_pos)
            compressed_tokens = raw_end // self.compression_ratio
            self._ensure_compressed_capacity(sequence_id, compressed_tokens)
            if raw_end % self.compression_ratio == 0:
                storage_positions.append(compressed_tokens - 1)
            else:
                storage_positions.append(-1)
        self._write_prepared_decode_positions(storage_positions)

        if refresh_page_table:
            self._refresh_decode_page_table(
                sequence_ids,
                use_cuda_graph_page_table=use_cuda_graph_page_table,
            )

    def get_context_kv_page_ptrs(
        self, sequence_id: int, layer_idx: int, context_length: int
    ):
        compressed_length = self._compressed_tokens(context_length)
        self._ensure_compressed_capacity(int(sequence_id), compressed_length)
        self._rebuild_last_page_table_if_dirty()
        return super().get_context_kv_page_ptrs(
            int(sequence_id),
            int(layer_idx),
            self._storage_capacity_tokens(compressed_length),
        )

    def update_layer_decode_new_token(
        self,
        k_tensor: torch.Tensor,
        v_tensor: Optional[torch.Tensor],
        sequence_lengths: Optional[torch.Tensor],
        layer_idx: int,
        batch_slice: Optional[tuple] = None,
        slot_indices: Optional[torch.Tensor] = None,
        *,
        assume_prepared: bool = False,
    ) -> None:
        """Append only decode rows that produced a compressed KV token."""

        batch_size = int(k_tensor.shape[0])
        resolved_slots = self._resolve_slot_indices(
            batch_size=batch_size,
            batch_slice=batch_slice,
            slot_indices=slot_indices,
        )
        if assume_prepared:
            if sequence_lengths is not None:
                raise TypeError(
                    "CompressedRatioGPUPagedKVCacheManager: sequence_lengths "
                    "must be None when assume_prepared=True"
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

        valid_rows = torch.nonzero(storage_positions >= 0).flatten()
        if int(valid_rows.numel()) == 0:
            return
        valid_rows = valid_rows.to(device=k_tensor.device, dtype=torch.long)
        compact_k = k_tensor.index_select(0, valid_rows).contiguous()
        compact_v = None
        if v_tensor is not None:
            compact_v = v_tensor.index_select(0, valid_rows).contiguous()
        compact_slots = resolved_slots.index_select(0, valid_rows).contiguous()
        compact_positions = storage_positions.index_select(
            0, valid_rows.to(device=storage_positions.device)
        ).contiguous()
        return super().update_layer_decode_new_token(
            k_tensor=compact_k,
            v_tensor=compact_v,
            sequence_lengths=compact_positions,
            layer_idx=int(layer_idx),
            batch_slice=None,
            slot_indices=compact_slots,
        )

    def map_raw_lengths_to_compressed_lengths(
        self,
        raw_lengths: Sequence[int] | torch.Tensor,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        raw_values = as_int_list(raw_lengths)
        output_device = device
        if output_device is None and isinstance(raw_lengths, torch.Tensor):
            output_device = raw_lengths.device
        return torch.as_tensor(
            [
                max(0, int(length)) // self.compression_ratio
                for length in raw_values
            ],
            dtype=dtype,
            device=output_device if output_device is not None else self.device,
        )

    def _compressed_tokens(self, raw_tokens: int) -> int:
        return max(0, int(raw_tokens)) // self.compression_ratio

    @staticmethod
    def _storage_capacity_tokens(exposed_tokens: int) -> int:
        return max(1, int(exposed_tokens))

    @staticmethod
    def _checked_raw_end(raw_position: int) -> int:
        raw_position = int(raw_position)
        if raw_position < 0:
            raise ValueError("raw decode positions must be non-negative")
        return raw_position + 1

    def _ensure_compressed_capacity(
        self, sequence_id: int, exposed_tokens: int
    ) -> None:
        capacity_tokens = self._storage_capacity_tokens(exposed_tokens)
        required_pages = ceil_div(capacity_tokens, self.page_size_tokens)
        state = self._sequences.get(sequence_id)
        current_pages = int(state.pages.numel()) if state is not None else 0
        missing_pages = max(0, required_pages - current_pages)
        if missing_pages == 0:
            return
        if state is None:
            super().allocate_pages(sequence_id, capacity_tokens)
        else:
            super().grow_sequence_pages(sequence_id, missing_pages)
        self._page_table_dirty = True

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
                "CompressedRatioGPUPagedKVCacheManager: GPU page table is not "
                "initialized"
            )
        if slot_indices is None:
            slot_indices = self._gpu_page_table_manager._slot_index_tensor
            if slot_indices is None:
                raise RuntimeError(
                    "CompressedRatioGPUPagedKVCacheManager: slot indices are "
                    "unavailable"
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
                "CompressedRatioGPUPagedKVCacheManager: slot_indices must "
                f"align with batch size, got {slot_indices.shape[0]} vs "
                f"{batch_size}"
            )
        return slot_indices.contiguous()

    @staticmethod
    def _resolve_raw_positions(
        *,
        batch_size: int,
        batch_slice: Optional[tuple],
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(sequence_lengths, torch.Tensor):
            raise TypeError("sequence_lengths must be a torch.Tensor")
        raw_positions = sequence_lengths
        if batch_slice is not None and raw_positions.shape[0] != batch_size:
            start_idx, end_idx = batch_slice
            raw_positions = raw_positions[start_idx:end_idx]
        if raw_positions.shape[0] != batch_size:
            raise ValueError(
                "CompressedRatioGPUPagedKVCacheManager: sequence_lengths must "
                f"align with batch size, got {raw_positions.shape[0]} vs "
                f"{batch_size}"
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
                    "CompressedRatioGPUPagedKVCacheManager: batch_slice is "
                    "outside the prepared decode-position range"
                )
            positions = buffer[start_idx:end_idx]
        else:
            if batch_size > prepared_count:
                raise ValueError(
                    "CompressedRatioGPUPagedKVCacheManager: requested "
                    f"{batch_size} prepared positions, but only "
                    f"{prepared_count} are available"
                )
            positions = buffer[:batch_size]
        if positions.shape[0] != batch_size:
            raise ValueError(
                "CompressedRatioGPUPagedKVCacheManager: prepared positions "
                f"must align with batch size, got {positions.shape[0]} vs "
                f"{batch_size}"
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
                storage_positions.append(-1)
                continue
            if slot >= len(slot_order):
                raise IndexError(
                    f"slot index {slot} exceeds active slot order length "
                    f"{len(slot_order)}"
                )
            sequence_id = int(slot_order[slot])
            raw_end = self._checked_raw_end(raw_pos)
            compressed_tokens = raw_end // self.compression_ratio
            self._ensure_compressed_capacity(sequence_id, compressed_tokens)
            if raw_end % self.compression_ratio == 0:
                storage_positions.append(compressed_tokens - 1)
            else:
                storage_positions.append(-1)
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


__all__ = ["CompressedRatioGPUPagedKVCacheManager"]
