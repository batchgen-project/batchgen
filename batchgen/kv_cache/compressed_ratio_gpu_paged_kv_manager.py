from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.kv_cache.compressed_ratio_gpu_kv_kernels import (
    run_masked_paged_kv_token_update_fused,
)
from batchgen.kv_cache.coordinator_utils import as_int_list, ceil_div
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig
from batchgen.kv_cache.position_mapped_gpu_paged_kv_manager import (
    PositionMappedGPUPagedKVCacheManager,
)


class CompressedRatioGPUPagedKVCacheManager(
    PositionMappedGPUPagedKVCacheManager
):
    """GPU paged KV manager for floor-compressed token streams.

    The base manager still owns storage, page allocation, page tables, and the
    token update kernel. This wrapper only maps raw sequence lengths to
    ``raw_length // compression_ratio`` and filters decode steps that have not
    produced a full compressed KV token yet.
    """

    manager_name = "CompressedRatioGPUPagedKVCacheManager"

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

        layer_idx = self.resolve_physical_layer(layer_idx)
        batch_size = int(k_tensor.shape[0])
        if k_tensor.shape[1] != 1:
            raise ValueError(
                "update_layer_decode_new_token: k_tensor must have sequence "
                f"dimension 1, got {k_tensor.shape[1]}"
            )

        _, resolved_slots, storage_positions = (
            self._resolve_decode_update_inputs(
                k_tensor=k_tensor,
                sequence_lengths=sequence_lengths,
                batch_slice=batch_slice,
                slot_indices=slot_indices,
                assume_prepared=assume_prepared,
            )
        )
        page_table = self._decode_page_table(assume_prepared=assume_prepared)

        if storage_positions.dtype != torch.int32:
            storage_positions = storage_positions.to(dtype=torch.int32)
        storage_positions = storage_positions.to(
            device=page_table.device, dtype=torch.int32
        ).contiguous()

        k_tokens = k_tensor.view(batch_size, -1)
        v_tokens = None
        v_cache_layer = None
        if v_tensor is not None and self._v_cache is not None:
            v_tokens = v_tensor.view(batch_size, -1)
            v_cache_layer = self._v_cache[layer_idx]

        run_masked_paged_kv_token_update_fused(
            k_cache=self._k_cache[layer_idx],
            k_tokens=k_tokens,
            page_table=page_table,
            slot_indices=resolved_slots,
            token_indices=storage_positions,
            page_size_tokens=self.config.page_size_tokens,
            v_cache=v_cache_layer,
            v_tokens=v_tokens,
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
            self._allocate_storage_pages(sequence_id, capacity_tokens)
        else:
            self._grow_storage_sequence_pages(sequence_id, missing_pages)
        self._page_table_dirty = True

    def _prepare_decode_storage_position(
        self, sequence_id: int, raw_position: int
    ) -> int:
        raw_end = self._checked_raw_end(raw_position)
        compressed_tokens = raw_end // self.compression_ratio
        self._ensure_compressed_capacity(sequence_id, compressed_tokens)
        if raw_end % self.compression_ratio != 0:
            return -1
        return compressed_tokens - 1

    def _context_storage_tokens(
        self, sequence_id: int, context_length: int
    ) -> int:
        compressed_length = self._compressed_tokens(context_length)
        self._ensure_compressed_capacity(sequence_id, compressed_length)
        return self._storage_capacity_tokens(compressed_length)


__all__ = ["CompressedRatioGPUPagedKVCacheManager"]
