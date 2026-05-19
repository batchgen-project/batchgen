from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from batchgen.kv_cache.compressed_ratio_gpu_kv_kernels import (
    run_compressed_state_update,
)
from batchgen.kv_cache.coordinator_utils import (
    as_int_list,
    resolve_from_layer_mapping,
)
from batchgen.kv_cache.gpu_paged_kv_manager import (
    _normalize_device,
    _normalize_gpu_layer_mapping,
    _TensorStack,
)


@dataclass(frozen=True)
class CompressedStateGPUConfig:
    num_layers: int
    num_state_items: int
    ring_size: int
    state_dim: int
    state_dtype: torch.dtype
    cuda_graph_max_slots: Optional[int] = None
    logical_to_physical_layer: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class CompressedStateGPUStats:
    num_total_state_items: int
    num_free_state_items: int
    num_used_state_items: int
    num_active_sequences: int


class CompressedStateGPUManager:
    """GPU storage for rolling compressor state.

    This manager stores compressor scratch state, not completed compressed KV.
    Each active sequence owns one fixed-size state item. A raw token position
    only selects the ring slot within that item:

    ``state_item_id * ring_size + raw_position % ring_size``.
    """

    manager_name = "CompressedStateGPUManager"

    def __init__(
        self,
        *,
        config: CompressedStateGPUConfig,
        device: Union[str, int, torch.device],
        ratio: int,
        overlap: bool,
    ) -> None:
        self.config = config
        self.device = _normalize_device(device)
        self.ratio = int(ratio)
        self.overlap = bool(overlap)
        if self.ratio <= 0:
            raise ValueError("ratio must be > 0")
        if self.config.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.config.num_state_items <= 0:
            raise ValueError("num_state_items must be > 0")
        if self.config.ring_size <= 0:
            raise ValueError("ring_size must be > 0")
        if self.config.state_dim <= 0:
            raise ValueError("state_dim must be > 0")
        if self.config.ring_size % self.ratio != 0:
            raise ValueError("ring_size must be divisible by ratio")
        self._logical_to_physical_layer = _normalize_gpu_layer_mapping(
            self.config.logical_to_physical_layer,
            self.config.num_layers,
        )
        self._reset_runtime_state()

    def initialize(self) -> None:
        if self._is_initialized:
            return
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self._state_cache = torch.zeros(
            (
                self.config.num_layers,
                self.config.num_state_items,
                self.config.ring_size,
                self.config.state_dim,
            ),
            dtype=self.config.state_dtype,
            device=self.device,
        )
        self._free_state_items = _TensorStack(self.config.num_state_items)
        self._ensure_prepared_state_slot_buffer()
        self._is_initialized = True

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        if not self._is_initialized:
            return
        self._reset_runtime_state()
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def allocate_state_item(self, sequence_id: int) -> int:
        self._ensure_initialized()
        return self._ensure_state_item(int(sequence_id))

    def allocate_state_items_for_sequences(
        self, sequence_ids: Sequence[int]
    ) -> dict[int, int]:
        self._ensure_initialized()
        return {
            int(seq_id): self._ensure_state_item(int(seq_id))
            for seq_id in sequence_ids
        }

    def release_sequence_states(self, sequence_ids: Sequence[int]) -> None:
        self._ensure_initialized()
        reclaimed: list[int] = []
        for seq_id in [int(seq_id) for seq_id in sequence_ids]:
            state_item_id = self._sequence_state_items.pop(seq_id, None)
            if state_item_id is not None:
                reclaimed.append(state_item_id)
        if reclaimed:
            self._free_state_items.push(reclaimed)

    def prepare_decode_step(
        self,
        sequence_ids: Sequence[int],
        raw_positions: Sequence[int] | torch.Tensor,
    ) -> None:
        self._ensure_initialized()
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        raw_values = as_int_list(raw_positions)
        if len(sequence_ids) != len(raw_values):
            raise ValueError(
                "prepare_decode_step: sequence_ids and raw_positions must "
                "have the same length"
            )
        slots = [
            self._prepare_state_slot(sequence_id, raw_position)
            for sequence_id, raw_position in zip(sequence_ids, raw_values)
        ]
        self._write_prepared_state_slots(slots)

    def update_layer_decode_state(
        self,
        state_tensor: torch.Tensor,
        raw_positions: Optional[Sequence[int] | torch.Tensor],
        layer_idx: int,
        *,
        sequence_ids: Optional[Sequence[int]] = None,
        batch_slice: Optional[tuple[int, int]] = None,
        assume_prepared: bool = False,
    ) -> None:
        self._ensure_initialized()
        physical_layer = self.resolve_physical_layer(layer_idx)
        tokens = self._flatten_state_tensor(state_tensor)
        batch_size = int(tokens.shape[0])
        if assume_prepared:
            if raw_positions is not None:
                raise TypeError(
                    "raw_positions must be None when assume_prepared=True"
                )
            slots = self._resolve_prepared_state_slots(
                batch_size=batch_size,
                batch_slice=batch_slice,
            )
        else:
            if raw_positions is None:
                raise TypeError("raw_positions must be provided")
            if sequence_ids is None:
                raise TypeError(
                    "sequence_ids must be provided when assume_prepared=False"
                )
            slots = self._resolve_state_slots_from_sequences(
                sequence_ids=sequence_ids,
                raw_positions=raw_positions,
                batch_size=batch_size,
                batch_slice=batch_slice,
            )

        cache = self._state_cache[physical_layer].view(
            self.config.num_state_items * self.config.ring_size,
            self.config.state_dim,
        )
        run_compressed_state_update(
            state_cache=cache,
            state_tokens=tokens,
            state_slots=slots.to(device=self.device, dtype=torch.int32),
        )

    def get_layer_state_buffer(self, layer_idx: int) -> torch.Tensor:
        self._ensure_initialized()
        return self._state_cache[self.resolve_physical_layer(layer_idx)]

    def get_state_tensors(self) -> torch.Tensor:
        self._ensure_initialized()
        return self._state_cache

    def get_sequence_layer_state_item_pointer(
        self, sequence_id: int, layer_idx: int
    ) -> int:
        self._ensure_initialized()
        physical_layer = self.resolve_physical_layer(layer_idx)
        state_item_id = self._get_sequence_state_item(int(sequence_id))
        return int(self._state_cache[physical_layer, state_item_id].data_ptr())

    def export_state_item_pointers(
        self, sequence_ids: Sequence[int]
    ) -> torch.Tensor:
        self._ensure_initialized()
        order = [int(seq_id) for seq_id in sequence_ids]
        table = torch.empty(
            (self.config.num_layers, len(order)),
            dtype=torch.int64,
            device="cpu",
        )
        for layer_idx in range(self.config.num_layers):
            for batch_idx, seq_id in enumerate(order):
                state_item_id = self._get_sequence_state_item(seq_id)
                table[layer_idx, batch_idx] = int(
                    self._state_cache[layer_idx, state_item_id].data_ptr()
                )
        return table

    def state_item_ptr(self, layer_idx: int, state_item_id: int) -> int:
        self._ensure_initialized()
        physical_layer = self.resolve_physical_layer(layer_idx)
        state_item_id = int(state_item_id)
        if state_item_id < 0 or state_item_id >= self.config.num_state_items:
            raise IndexError("state item id out of range")
        return int(self._state_cache[physical_layer, state_item_id].data_ptr())

    def resolve_state_slot(self, sequence_id: int, raw_position: int) -> int:
        self._ensure_initialized()
        return self._prepare_state_slot(int(sequence_id), int(raw_position))

    def get_stats(self) -> CompressedStateGPUStats:
        self._ensure_initialized()
        used = self.config.num_state_items - self._free_state_items.size
        return CompressedStateGPUStats(
            num_total_state_items=self.config.num_state_items,
            num_free_state_items=self._free_state_items.size,
            num_used_state_items=used,
            num_active_sequences=len(self._sequence_state_items),
        )

    @property
    def state_cache(self) -> torch.Tensor:
        return self.get_state_tensors()

    @property
    def uses_logical_layer_mapping(self) -> bool:
        return self._logical_to_physical_layer is not None

    def resolve_physical_layer(self, logical_layer_id: int) -> int:
        logical_layer_id = int(logical_layer_id)
        if logical_layer_id < 0:
            raise IndexError("logical layer id must be >= 0")
        if self._logical_to_physical_layer is None:
            if logical_layer_id >= self.config.num_layers:
                raise IndexError(f"layer_idx {logical_layer_id} out of range")
            return logical_layer_id
        return resolve_from_layer_mapping(
            "GPU compressed state",
            "state",
            self._logical_to_physical_layer,
            logical_layer_id,
        )

    def _reset_runtime_state(self) -> None:
        self._is_initialized = False
        self._state_cache: Optional[torch.Tensor] = None
        self._free_state_items: Optional[_TensorStack] = None
        self._sequence_state_items: dict[int, int] = {}
        self._prepared_state_slots: Optional[torch.Tensor] = None
        self._prepared_state_slot_count = 0

    def _ensure_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "CompressedStateGPUManager.initialize must be called before use"
            )

    def _ensure_state_item(self, sequence_id: int) -> int:
        state_item_id = self._sequence_state_items.get(sequence_id)
        if state_item_id is not None:
            return state_item_id
        if self._free_state_items.size <= 0:
            raise RuntimeError("Insufficient free compressed state items")
        state_item = self._free_state_items.pop(1)
        state_item_id = int(state_item[0].item())
        self._sequence_state_items[sequence_id] = state_item_id
        return state_item_id

    def _prepare_state_slot(self, sequence_id: int, raw_position: int) -> int:
        raw_position = int(raw_position)
        if raw_position < 0:
            raise ValueError("raw positions must be non-negative")
        state_item_id = self._ensure_state_item(sequence_id)
        ring_offset = raw_position % self.config.ring_size
        return state_item_id * self.config.ring_size + ring_offset

    def _resolve_state_slots_from_sequences(
        self,
        *,
        sequence_ids: Sequence[int],
        raw_positions: Sequence[int] | torch.Tensor,
        batch_size: int,
        batch_slice: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        sequence_values = [int(seq_id) for seq_id in sequence_ids]
        raw_values = as_int_list(raw_positions)
        if batch_slice is not None and len(sequence_values) != batch_size:
            start_idx, end_idx = batch_slice
            sequence_values = sequence_values[start_idx:end_idx]
        if batch_slice is not None and len(raw_values) != batch_size:
            start_idx, end_idx = batch_slice
            raw_values = raw_values[start_idx:end_idx]
        if len(sequence_values) != batch_size:
            raise ValueError("sequence_ids must align with batch size")
        if len(raw_values) != batch_size:
            raise ValueError("raw_positions must align with batch size")
        slots = [
            self._prepare_state_slot(sequence_id, raw_position)
            for sequence_id, raw_position in zip(sequence_values, raw_values)
        ]
        return torch.as_tensor(slots, dtype=torch.int32, device=self.device)

    def _flatten_state_tensor(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if state_tensor.device != self.device:
            state_tensor = state_tensor.to(self.device)
        if state_tensor.shape[0] <= 0:
            raise ValueError("state_tensor batch dimension must be non-empty")
        tokens = state_tensor.contiguous().view(int(state_tensor.shape[0]), -1)
        if int(tokens.shape[1]) != self.config.state_dim:
            raise ValueError(
                "state tensor flattened dimension must equal state_dim, got "
                f"{int(tokens.shape[1])} vs {self.config.state_dim}"
            )
        return tokens

    def _ensure_prepared_state_slot_buffer(self) -> torch.Tensor:
        if self._prepared_state_slots is None:
            max_slots = self.config.cuda_graph_max_slots
            if max_slots is None:
                max_slots = 1024
            self._prepared_state_slots = torch.full(
                (int(max_slots),), -1, dtype=torch.int32, device=self.device
            )
        return self._prepared_state_slots

    def _write_prepared_state_slots(self, slots: Sequence[int]) -> None:
        buffer = self._ensure_prepared_state_slot_buffer()
        count = len(slots)
        if count > int(buffer.numel()):
            raise ValueError(
                "prepare_decode_step batch exceeds prepared state-slot buffer"
            )
        if count:
            buffer[:count].copy_(
                torch.as_tensor(slots, dtype=torch.int32, device=self.device)
            )
        previous_count = self._prepared_state_slot_count
        if previous_count > count:
            buffer[count:previous_count].fill_(-1)
        self._prepared_state_slot_count = count

    def _resolve_prepared_state_slots(
        self,
        *,
        batch_size: int,
        batch_slice: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        buffer = self._ensure_prepared_state_slot_buffer()
        prepared_count = self._prepared_state_slot_count
        if batch_slice is not None and prepared_count != batch_size:
            start_idx, end_idx = batch_slice
            slots = buffer[start_idx:end_idx]
        else:
            slots = buffer[:batch_size]
        if int(slots.shape[0]) != batch_size:
            raise ValueError("prepared state slots must align with batch size")
        return slots

    def _get_sequence_state_item(self, sequence_id: int) -> int:
        state_item_id = self._sequence_state_items.get(int(sequence_id))
        if state_item_id is None:
            raise KeyError(
                f"Sequence {sequence_id} has no compressed state item"
            )
        return state_item_id


__all__ = [
    "CompressedStateGPUConfig",
    "CompressedStateGPUManager",
    "CompressedStateGPUStats",
]
