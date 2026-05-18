from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.kv_cache.coordinator_utils import as_int_list, ceil_div
from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)


@dataclass
class _SWASequenceState:
    window_start_page: int = 0
    active_pages: int = 0
    max_seen_raw_pos: int = -1
    has_tokens: bool = False


@dataclass(frozen=True)
class _WindowForRawEnd:
    window_start_page: int
    active_tokens: int
    required_pages: int


class SWAGPUPagedKVCacheManager(GPUPagedKVCacheManager):
    """Page-level sliding-window GPU paged KV manager.

    Storage, page allocation, page-table rebuilds, and update kernels come from
    ``GPUPagedKVCacheManager``. This subclass only converts raw token positions
    into window-local storage positions and releases old prefix pages when the
    page-aligned SWA window moves forward.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        model_config: Optional[ModelConfig] = None,
        *,
        config: Optional[GPUPagedKVConfig] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        window_size_tokens: int,
    ) -> None:
        super().__init__(
            engine_config=engine_config,
            model_config=model_config,
            config=config,
            device=device,
        )
        self.window_size_tokens = int(window_size_tokens)
        self.page_size_tokens = int(self.config.page_size_tokens)
        if self.page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be > 0")
        if self.window_size_tokens <= 0:
            raise ValueError("window_size_tokens must be > 0")
        if self.window_size_tokens % self.page_size_tokens != 0:
            raise ValueError(
                "window_size_tokens must be divisible by page_size_tokens "
                "for page-level SWA"
            )
        self.window_pages = self.window_size_tokens // self.page_size_tokens
        self._states: dict[int, _SWASequenceState] = {}
        self._last_page_table_order: Optional[list[int]] = None
        self._page_table_dirty = False
        self._prepared_decode_positions: Optional[torch.Tensor] = None
        self._prepared_decode_position_count = 0

    def initialize(self) -> None:
        super().initialize()
        self._ensure_prepared_decode_position_buffer()

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        self._states.clear()
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
        raw_tokens = [int(tokens) for tokens in num_tokens]
        active_tokens: list[int] = []
        windows: list[_WindowForRawEnd] = []
        for token_count in raw_tokens:
            window = self._compute_window_for_raw_end(token_count)
            if window.active_tokens <= 0:
                raise ValueError(
                    "allocate_pages_for_sequences: num_tokens entries must "
                    "be > 0"
                )
            active_tokens.append(window.active_tokens)
            windows.append(window)

        allocations = super().allocate_pages_for_sequences(
            sequence_ids, active_tokens
        )
        for seq_id, token_count, window in zip(
            sequence_ids, raw_tokens, windows
        ):
            state = self._states.setdefault(seq_id, _SWASequenceState())
            state.window_start_page = window.window_start_page
            state.active_pages = window.required_pages
            state.max_seen_raw_pos = token_count - 1
            state.has_tokens = True
        self._page_table_dirty = True
        return allocations

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        super().free_pages_for_sequences(sequence_ids)
        for seq_id in sequence_ids:
            self._states.pop(seq_id, None)
        if self._last_page_table_order is not None:
            freed = set(sequence_ids)
            self._last_page_table_order = [
                seq_id
                for seq_id in self._last_page_table_order
                if seq_id not in freed
            ]
        else:
            freed = set(sequence_ids)
            active_order = [
                seq_id
                for seq_id in self._gpu_page_table_manager.slot_to_seq_id
                if seq_id not in freed
            ]
            if active_order != self._gpu_page_table_manager.slot_to_seq_id:
                self._last_page_table_order = active_order
        self._page_table_dirty = True
        self._rebuild_last_page_table_if_dirty()

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
        """Advance SWA metadata before the actual decode KV write.

        ``raw_positions`` are the model-global decode write positions, aligned
        with ``sequence_ids``. This method performs the work that can change
        allocation metadata: page-level window advance, prefix-page release,
        capacity growth, and optional page-table refresh. The window-local write
        positions are written into the manager-owned decode-position buffer and
        consumed by :meth:`update_layer_decode_new_token` when
        ``assume_prepared=True``.
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
            active_tokens, _ = self._update_window_for_raw_end(
                sequence_id, int(raw_pos) + 1
            )
            storage_positions.append(active_tokens - 1)
        self._write_prepared_decode_positions(storage_positions)

        if refresh_page_table:
            self._refresh_decode_page_table(
                sequence_ids,
                use_cuda_graph_page_table=use_cuda_graph_page_table,
            )

    def get_prepared_decode_positions(
        self, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Return the manager-owned prepared decode-position buffer view."""

        buffer = self._ensure_prepared_decode_position_buffer()
        count = (
            self._prepared_decode_position_count
            if batch_size is None
            else int(batch_size)
        )
        if count < 0 or count > self._prepared_decode_position_count:
            raise ValueError(
                "get_prepared_decode_positions: requested "
                f"{count} positions, but only "
                f"{self._prepared_decode_position_count} are prepared"
            )
        return buffer[:count]

    def get_context_kv_page_ptrs(
        self, sequence_id: int, layer_idx: int, context_length: int
    ):
        active_tokens, _ = self._update_window_for_raw_end(
            int(sequence_id), int(context_length)
        )
        self._rebuild_last_page_table_if_dirty()
        return super().get_context_kv_page_ptrs(
            int(sequence_id), int(layer_idx), active_tokens
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
        """Append decode KV using raw token positions.

        ``sequence_lengths`` keeps the old manager's argument name, but for this
        wrapper it is interpreted as raw decode write positions unless
        ``assume_prepared`` is set. With ``assume_prepared=True``, this method
        reads the manager-owned prepared decode-position buffer by default; no
        window advance, allocation, prefix release, or page-table rebuild is
        performed here.
        """

        batch_size = int(k_tensor.shape[0])
        resolved_slots = self._resolve_slot_indices(
            batch_size=batch_size,
            batch_slice=batch_slice,
            slot_indices=slot_indices,
        )
        if assume_prepared:
            storage_positions = self._resolve_prepared_storage_positions(
                batch_size=batch_size,
                batch_slice=batch_slice,
                sequence_lengths=sequence_lengths,
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
                like=sequence_lengths,
            )
            self._rebuild_last_page_table_if_dirty()
        return super().update_layer_decode_new_token(
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            sequence_lengths=storage_positions,
            layer_idx=int(layer_idx),
            batch_slice=None,
            slot_indices=resolved_slots,
        )

    def map_raw_lengths_to_window_local_lengths(
        self,
        sequence_ids: Sequence[int],
        raw_lengths: Sequence[int] | torch.Tensor,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """Return page-level SWA lengths aligned with ``sequence_ids``.

        This is the value that attention should consume as cache length when it
        uses the page table owned by this SWA manager.
        """

        sequence_ids = [int(seq_id) for seq_id in sequence_ids]
        raw_values = as_int_list(raw_lengths)
        if len(sequence_ids) != len(raw_values):
            raise ValueError(
                "map_raw_lengths_to_window_local_lengths: sequence_ids and "
                "raw_lengths must have the same length"
            )
        local_lengths: list[int] = []
        for seq_id, raw_length in zip(sequence_ids, raw_values):
            state = self._states.get(seq_id)
            if state is None or not state.has_tokens:
                window = self._compute_window_for_raw_end(raw_length)
                local_lengths.append(window.active_tokens)
                continue
            window_start_token = state.window_start_page * self.page_size_tokens
            local_lengths.append(max(0, int(raw_length) - window_start_token))
        output_device = device
        if output_device is None and isinstance(raw_lengths, torch.Tensor):
            output_device = raw_lengths.device
        return torch.as_tensor(
            local_lengths,
            dtype=dtype,
            device=output_device if output_device is not None else self.device,
        )

    def window_start_page(self, sequence_id: int) -> int:
        state = self._states.get(int(sequence_id))
        if state is None:
            raise KeyError(
                f"window_start_page: unknown sequence id {sequence_id}"
            )
        return state.window_start_page

    def window_start_pages(self, sequence_ids: Sequence[int]) -> list[int]:
        return [self.window_start_page(seq_id) for seq_id in sequence_ids]

    def _compute_window_for_raw_end(
        self, raw_end_tokens: int
    ) -> _WindowForRawEnd:
        raw_end_tokens = int(raw_end_tokens)
        if raw_end_tokens <= 0:
            return _WindowForRawEnd(0, 0, 0)
        first_needed_token = max(0, raw_end_tokens - self.window_size_tokens)
        window_start_page = first_needed_token // self.page_size_tokens
        window_start_token = window_start_page * self.page_size_tokens
        active_tokens = raw_end_tokens - window_start_token
        required_pages = ceil_div(active_tokens, self.page_size_tokens)
        if required_pages > self.window_pages + 1:
            raise RuntimeError(
                f"SWA active pages {required_pages} exceed window_pages + 1 "
                f"({self.window_pages + 1})"
            )
        return _WindowForRawEnd(
            window_start_page=window_start_page,
            active_tokens=active_tokens,
            required_pages=required_pages,
        )

    def _update_window_for_raw_end(
        self, sequence_id: int, raw_end_tokens: int
    ) -> tuple[int, int]:
        window = self._compute_window_for_raw_end(raw_end_tokens)
        state = self._states.setdefault(sequence_id, _SWASequenceState())
        if (
            state.has_tokens
            and window.window_start_page < state.window_start_page
        ):
            raise ValueError(
                "SWA GPU manager does not support writing tokens older than "
                "the current page-level window"
            )
        if (
            state.has_tokens
            and window.window_start_page > state.window_start_page
        ):
            pages_to_release = (
                window.window_start_page - state.window_start_page
            )
            if pages_to_release > state.active_pages:
                raise RuntimeError(
                    f"sequence {sequence_id} cannot release {pages_to_release} "
                    f"SWA pages with only {state.active_pages} active pages"
                )
            self._release_sequence_prefix_pages(sequence_id, pages_to_release)
            state.active_pages -= pages_to_release
            self._page_table_dirty = True

        state.window_start_page = window.window_start_page
        added_pages = self._ensure_active_capacity(
            sequence_id, window.required_pages, state
        )
        if raw_end_tokens > 0:
            state.max_seen_raw_pos = max(
                state.max_seen_raw_pos, raw_end_tokens - 1
            )
            state.has_tokens = True
        return window.active_tokens, added_pages

    def _ensure_active_capacity(
        self,
        sequence_id: int,
        required_pages: int,
        state: _SWASequenceState,
    ) -> int:
        if required_pages <= 0:
            return 0
        if required_pages > self.window_pages + 1:
            raise ValueError(
                f"sequence {sequence_id} requires {required_pages} active "
                f"pages, exceeding window_pages + 1 ({self.window_pages + 1})"
            )
        current_pages = int(state.active_pages)
        base_state = self._sequences.get(sequence_id)
        if current_pages == 0 and base_state is not None:
            current_pages = int(base_state.pages.numel())
            state.active_pages = current_pages
        missing_pages = max(0, required_pages - current_pages)
        if missing_pages:
            if base_state is None:
                super().allocate_pages(
                    sequence_id, missing_pages * self.page_size_tokens
                )
            else:
                super().grow_sequence_pages(sequence_id, missing_pages)
            state.active_pages += missing_pages
            self._page_table_dirty = True
        return missing_pages

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
                "SWAGPUPagedKVCacheManager: GPU page table is not initialized"
            )
        if slot_indices is None:
            slot_indices = self._gpu_page_table_manager._slot_index_tensor
            if slot_indices is None:
                raise RuntimeError(
                    "SWAGPUPagedKVCacheManager: slot indices are unavailable"
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
                "SWAGPUPagedKVCacheManager: slot_indices must align with "
                f"batch size, got {slot_indices.shape[0]} vs {batch_size}"
            )
        return slot_indices.contiguous()

    def _resolve_raw_positions(
        self,
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
                "SWAGPUPagedKVCacheManager: sequence_lengths must align with "
                f"batch size, got {raw_positions.shape[0]} vs {batch_size}"
            )
        return raw_positions.contiguous()

    def _resolve_prepared_storage_positions(
        self,
        *,
        batch_size: int,
        batch_slice: Optional[tuple],
        sequence_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if sequence_lengths is not None:
            return self._resolve_raw_positions(
                batch_size=batch_size,
                batch_slice=batch_slice,
                sequence_lengths=sequence_lengths,
            )

        buffer = self._ensure_prepared_decode_position_buffer()
        prepared_count = int(self._prepared_decode_position_count)
        if batch_slice is not None and prepared_count != batch_size:
            start_idx, end_idx = batch_slice
            if start_idx < 0 or end_idx < start_idx or end_idx > prepared_count:
                raise ValueError(
                    "SWAGPUPagedKVCacheManager: batch_slice is outside the "
                    f"prepared decode-position range, slice={batch_slice}, "
                    f"prepared={prepared_count}"
                )
            positions = buffer[start_idx:end_idx]
        else:
            if batch_size > prepared_count:
                raise ValueError(
                    "SWAGPUPagedKVCacheManager: requested "
                    f"{batch_size} prepared positions, but only "
                    f"{prepared_count} are available"
                )
            positions = buffer[:batch_size]

        if positions.shape[0] != batch_size:
            raise ValueError(
                "SWAGPUPagedKVCacheManager: prepared positions must align "
                f"with batch size, got {positions.shape[0]} vs {batch_size}"
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
                storage_positions.append(0)
                continue
            if slot >= len(slot_order):
                raise IndexError(
                    f"slot index {slot} exceeds active slot order length "
                    f"{len(slot_order)}"
                )
            sequence_id = int(slot_order[slot])
            active_tokens, _ = self._update_window_for_raw_end(
                sequence_id, int(raw_pos) + 1
            )
            storage_positions.append(active_tokens - 1)
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
                "prepare_decode_step: active batch size exceeds the "
                f"prepared decode-position buffer capacity, batch={count}, "
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


__all__ = ["SWAGPUPagedKVCacheManager"]
