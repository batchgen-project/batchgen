from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from batchgen.kv_cache.compressed_ratio_gpu_kv_kernels import (
    run_compressed_state_update,
)
from batchgen.kv_cache.coordinator_utils import (
    as_int_list,
)
from batchgen.kv_cache.gpu_paged_kv_manager import (
    CUDAGraphPageTableState,
    GPUPagedKVStats,
    _GPUPageTableManager,
    _normalize_device,
    _normalize_gpu_layer_mapping,
    _SequenceState,
    _TensorStack,
)


@dataclass(frozen=True)
class CompressedStateGPUConfig:
    num_layers: int
    num_pages: int
    state_page_size_tokens: int
    ring_size: int
    state_dim: int
    state_dtype: torch.dtype
    cuda_graph_max_pages_per_sequence: Optional[int] = None
    cuda_graph_max_slots: Optional[int] = None
    logical_to_physical_layer: Optional[Sequence[int]] = None

    @property
    def page_size_tokens(self) -> int:
        return self.state_page_size_tokens


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
        if self.config.num_pages <= 0:
            raise ValueError("num_pages must be > 0")
        if self.config.state_page_size_tokens <= 0:
            raise ValueError("state_page_size_tokens must be > 0")
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
                self.config.num_pages,
                self.config.ring_size,
                self.config.state_dim,
            ),
            dtype=self.config.state_dtype,
            device=self.device,
        )
        max_pages = self.config.cuda_graph_max_pages_per_sequence
        if max_pages is None:
            max_pages = 1
        max_slots = self.config.cuda_graph_max_slots
        if max_slots is None:
            max_slots = 1024
        self._page_table_manager = _GPUPageTableManager(
            device=self.device,
            max_pages_per_sequence=int(max_pages),
            max_slots=int(max_slots),
        )
        self._free_pages = _TensorStack(self.config.num_pages)
        self._ensure_prepared_state_slot_buffer()
        self._is_initialized = True

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        if not self._is_initialized:
            return
        self._reset_runtime_state()
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def allocate_pages(
        self, sequence_id: int, raw_num_tokens: int
    ) -> list[int]:
        return self.allocate_pages_for_sequences(
            [int(sequence_id)], [int(raw_num_tokens)]
        ).get(int(sequence_id), [])

    def allocate_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        raw_num_tokens: Sequence[int],
    ) -> dict[int, list[int]]:
        self._ensure_initialized()
        if len(sequence_ids) != len(raw_num_tokens):
            raise ValueError(
                "allocate_pages_for_sequences: sequence_ids and "
                "raw_num_tokens must have the same length"
            )
        allocations: dict[int, list[int]] = {}
        for seq_id, raw_tokens in zip(sequence_ids, raw_num_tokens):
            allocations[int(seq_id)] = self._ensure_capacity(
                int(seq_id), int(raw_tokens)
            )
        if any(allocations.values()):
            self._clear_active_page_pointer_tables()
            self._page_table_dirty = True
        return allocations

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        self._ensure_initialized()
        reclaimed: list[torch.Tensor] = []
        for seq_id in [int(seq_id) for seq_id in sequence_ids]:
            state = self._sequences.pop(seq_id, None)
            if state is not None:
                reclaimed.append(state.pages)
        if reclaimed:
            self._free_pages.push(torch.cat(reclaimed, dim=0))
            self._clear_active_page_pointer_tables()
            self._page_table_dirty = True

    def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        self._ensure_initialized()
        order = [int(seq_id) for seq_id in sequence_ids]
        missing = [seq_id for seq_id in order if seq_id not in self._sequences]
        if missing:
            raise KeyError(
                "rebuild_page_table: unallocated sequence ids: "
                + ", ".join(str(seq_id) for seq_id in missing)
            )
        self._last_page_table_order = order
        table = self._page_table_manager.rebuild(order, self._sequences)
        self._page_table_dirty = False
        self._update_active_page_pointer_tables()
        return table

    def clear_page_table(self) -> None:
        self._ensure_initialized()
        manager = self._page_table_manager
        max_pages = max(1, manager.max_pages_per_sequence)
        manager.gpu_table = torch.full(
            (0, max_pages), -1, dtype=torch.int32, device=self.device
        )
        manager._ensure_cuda_graph_table().fill_(-1)
        manager._cuda_graph_table_valid = True
        manager._cuda_graph_slot_count = 0
        manager.seq_id_to_slot = {}
        manager.slot_to_seq_id = []
        manager._slot_index_tensor = torch.empty(
            0, dtype=torch.int32, device=self.device
        )
        manager._slot_to_seq_id_tensor = torch.empty(
            0, dtype=torch.int64, device=self.device
        )
        manager._active_page_indices_cpu = torch.empty(0, dtype=torch.int64)
        manager.rebuild_version += 1
        self._last_page_table_order = []
        self._page_table_dirty = False
        self._clear_active_page_pointer_tables()

    def ensure_cuda_graph_page_table(
        self, sequence_ids: Sequence[int]
    ) -> torch.Tensor:
        self._ensure_initialized()
        order = [int(seq_id) for seq_id in sequence_ids]
        manager = self._page_table_manager
        if manager._cuda_graph_table_valid and manager.slot_to_seq_id == order:
            return manager.get_cuda_graph_table()
        if order:
            self.rebuild_page_table(order)
        else:
            self.clear_page_table()
        return manager.get_cuda_graph_table()

    def get_cuda_graph_page_table_state(self) -> CUDAGraphPageTableState:
        self._ensure_initialized()
        return self._page_table_manager.get_cuda_graph_state()

    def prepare_decode_step(
        self,
        sequence_ids: Sequence[int],
        raw_positions: Sequence[int] | torch.Tensor,
        *,
        refresh_page_table: bool = True,
        use_cuda_graph_page_table: bool = False,
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
        if refresh_page_table:
            if use_cuda_graph_page_table:
                self._prepared_decode_page_table = (
                    self.ensure_cuda_graph_page_table(sequence_ids)
                )
            elif (
                self._page_table_dirty
                or self._page_table_manager.gpu_table is None
                or self._page_table_manager.slot_to_seq_id != sequence_ids
            ):
                self._prepared_decode_page_table = self.rebuild_page_table(
                    sequence_ids
                )
            else:
                self._prepared_decode_page_table = (
                    self._page_table_manager.gpu_table
                )
        else:
            self._prepared_decode_page_table = None

    def update_layer_decode_state(
        self,
        state_tensor: torch.Tensor,
        raw_positions: Optional[torch.Tensor],
        layer_idx: int,
        batch_slice: Optional[tuple] = None,
        slot_indices: Optional[torch.Tensor] = None,
        *,
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
            slots = self._resolve_state_slots_from_batch(
                raw_positions=raw_positions,
                batch_size=batch_size,
                batch_slice=batch_slice,
                slot_indices=slot_indices,
            )

        cache = self._state_cache[physical_layer].view(
            self.config.num_pages * self.config.ring_size,
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

    def get_sequence_layer_state_page_pointers(
        self, sequence_id: int, layer_idx: int
    ) -> list[int]:
        self._ensure_initialized()
        physical_layer = self.resolve_physical_layer(layer_idx)
        state = self._get_sequence_state(int(sequence_id))
        return [
            self._state_cache[physical_layer, int(page)].data_ptr()
            for page in state.pages.tolist()
        ]

    def get_stats(self) -> GPUPagedKVStats:
        self._ensure_initialized()
        used = self.config.num_pages - self._free_pages.size
        return GPUPagedKVStats(
            num_total_pages=self.config.num_pages,
            num_free_pages=self._free_pages.size,
            num_used_pages=used,
            num_total_pages_allocated=used,
        )

    @property
    def page_size_tokens(self) -> int:
        return self.config.state_page_size_tokens

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
        from batchgen.kv_cache.coordinator_utils import (
            resolve_from_layer_mapping,
        )

        return resolve_from_layer_mapping(
            "GPU compressed state",
            "state",
            self._logical_to_physical_layer,
            logical_layer_id,
        )

    def _reset_runtime_state(self) -> None:
        self._is_initialized = False
        self._state_cache: Optional[torch.Tensor] = None
        self._free_pages: Optional[_TensorStack] = None
        self._sequences: dict[int, _SequenceState] = {}
        self._page_table_manager: Optional[_GPUPageTableManager] = None
        self._last_page_table_order: Optional[list[int]] = None
        self._page_table_dirty = False
        self._prepared_decode_page_table: Optional[torch.Tensor] = None
        self._prepared_state_slots: Optional[torch.Tensor] = None
        self._prepared_state_slot_count = 0
        self._active_page_ptr_table: Optional[torch.Tensor] = None

    def _ensure_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "CompressedStateGPUManager.initialize must be called before use"
            )

    def _ensure_capacity(
        self, sequence_id: int, raw_num_tokens: int
    ) -> list[int]:
        required_pages = self._required_pages(raw_num_tokens)
        state = self._sequences.get(sequence_id)
        current = int(state.pages.numel()) if state is not None else 0
        missing = max(0, required_pages - current)
        if missing == 0:
            return []
        if missing > self._free_pages.size:
            raise RuntimeError(
                f"Insufficient free pages: need {missing}, "
                f"have {self._free_pages.size}"
            )
        new_pages = self._free_pages.pop(missing)
        if state is None:
            self._sequences[sequence_id] = _SequenceState(pages=new_pages)
        else:
            state.append_pages(new_pages)
        return new_pages.tolist()

    def _required_pages(self, raw_num_tokens: int) -> int:
        del raw_num_tokens
        return 1

    def _prepare_state_slot(self, sequence_id: int, raw_position: int) -> int:
        raw_position = int(raw_position)
        if raw_position < 0:
            raise ValueError("raw positions must be non-negative")
        self._ensure_capacity(sequence_id, raw_position + 1)
        state = self._get_sequence_state(sequence_id)
        if int(state.pages.numel()) != 1:
            raise RuntimeError("state capacity was not allocated")
        page_id = int(state.pages[0].item())
        ring_offset = raw_position % self.config.ring_size
        return page_id * self.config.ring_size + ring_offset

    def _resolve_state_slots_from_batch(
        self,
        *,
        raw_positions: torch.Tensor,
        batch_size: int,
        batch_slice: Optional[tuple],
        slot_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        page_table = self._page_table_manager.gpu_table
        if page_table is None:
            raise RuntimeError("GPU state page table is not initialized")
        if slot_indices is None:
            slot_indices = self._page_table_manager._slot_index_tensor
            if slot_indices is None:
                raise RuntimeError("slot indices are unavailable")
        else:
            slot_indices = slot_indices.to(
                device=self.device, dtype=torch.int32
            )
        if batch_slice is not None and slot_indices.shape[0] != batch_size:
            start_idx, end_idx = batch_slice
            slot_indices = slot_indices[start_idx:end_idx]
        if raw_positions.shape[0] != batch_size and batch_slice is not None:
            start_idx, end_idx = batch_slice
            raw_positions = raw_positions[start_idx:end_idx]
        if raw_positions.shape[0] != batch_size:
            raise ValueError("raw_positions must align with batch size")
        if slot_indices.shape[0] != batch_size:
            raise ValueError("slot_indices must align with batch size")

        slot_values = as_int_list(slot_indices)
        raw_values = as_int_list(raw_positions)
        order = self._page_table_manager.slot_to_seq_id
        slots = []
        for slot, raw_position in zip(slot_values, raw_values):
            if slot < 0:
                slots.append(-1)
                continue
            if slot >= len(order):
                raise IndexError(f"slot index {slot} out of range")
            slots.append(
                self._prepare_state_slot(int(order[slot]), int(raw_position))
            )
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
            max_slots = int(self._page_table_manager.max_slots)
            self._prepared_state_slots = torch.full(
                (max_slots,), -1, dtype=torch.int32, device=self.device
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
        batch_slice: Optional[tuple],
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

    def _get_sequence_state(self, sequence_id: int) -> _SequenceState:
        state = self._sequences.get(int(sequence_id))
        if state is None:
            raise KeyError(f"Sequence {sequence_id} has no state pages")
        return state

    def _update_active_page_pointer_tables(self) -> None:
        rows: list[list[int]] = []
        max_pages = 0
        order = self._page_table_manager.slot_to_seq_id
        for seq_id in order:
            state = self._sequences.get(int(seq_id))
            pages = [] if state is None else state.pages.tolist()
            max_pages = max(max_pages, len(pages))
            rows.append([int(page) for page in pages])
        if not rows:
            self._active_page_ptr_table = torch.empty(
                (
                    self.config.num_layers,
                    0,
                    0,
                ),
                dtype=torch.int64,
                device="cpu",
            )
            return
        table = torch.zeros(
            (
                self.config.num_layers,
                len(rows),
                max_pages,
            ),
            dtype=torch.int64,
            device="cpu",
        )
        for layer_idx in range(self.config.num_layers):
            for slot, pages in enumerate(rows):
                for page_ordinal, page_id in enumerate(pages):
                    table[layer_idx, slot, page_ordinal] = int(
                        self._state_cache[layer_idx, page_id].data_ptr()
                    )
        self._active_page_ptr_table = table

    def _clear_active_page_pointer_tables(self) -> None:
        self._active_page_ptr_table = None


__all__ = [
    "CompressedStateGPUConfig",
    "CompressedStateGPUManager",
]
