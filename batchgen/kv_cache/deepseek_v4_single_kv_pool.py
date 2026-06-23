from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVStats
from batchgen_kernels.attention.v4_fused_qnorm_rope_kv import (
    HEAD_DIM,
    NOPE_DIM,
    TOKEN_BYTES,
    TOKEN_DATA_SIZE,
    dequantize_nope_from_fp8,
    fused_v4_qnorm_rope_kv_insert,
)

assert TOKEN_BYTES == 584, f"Unexpected DeepSeek-V4 token size: {TOKEN_BYTES}"

_INDEXER_QUANT_BLOCK_SIZE = 128
_INDEXER_FP8_MAX = 448.0
_INDEXER_SCALE_BYTES = 4
_MODEL1_TILE_SIZE = 64
_MODEL1_NUM_TILES = NOPE_DIM // _MODEL1_TILE_SIZE


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return -(-value // divisor)


def _normalize_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    raise TypeError(f"Unsupported device spec: {device!r}")


def _as_int_tensor(values: Iterable[int]) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=torch.int32)


@dataclass(frozen=True)
class DeepSeekV4PoolConfig:
    num_layers: int
    num_pages: int
    page_size_tokens: int
    bytes_per_token: int
    bytes_per_page_padded: int
    store_dtype: torch.dtype


@dataclass
class _SequenceState:
    pages: torch.Tensor

    def append_pages(self, new_pages: torch.Tensor) -> None:
        if self.pages.numel() == 0:
            self.pages = new_pages.clone()
            return
        self.pages = torch.cat((self.pages, new_pages), dim=0)


class _PageStack:
    def __init__(self, capacity: int) -> None:
        self._pages: List[int] = list(range(capacity - 1, -1, -1))

    @property
    def size(self) -> int:
        return len(self._pages)

    def pop(self, count: int) -> torch.Tensor:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > self.size:
            raise RuntimeError(
                f"Insufficient free pages: need {count}, have {self.size}"
            )
        values = self._pages[-count:]
        del self._pages[-count:]
        return _as_int_tensor(values)

    def push(self, pages: torch.Tensor | Iterable[int]) -> None:
        tensor = torch.as_tensor(list(pages), dtype=torch.int32).view(-1)
        self._pages.extend(int(x) for x in tensor.tolist())


class DeepSeekV4SingleKVPool:
    """Raw uint8 paged KV pool for DeepSeek-V4 FlashMLA fp8 K cache.

    FlashMLA MODEL1 fp8 sparse uses a *split* per-page layout (`tests/quant.py`):

    - page bytes `[0 : block_size * 576)` store per-token bodies
      `(448B fp8 NoPE + 128B bf16 RoPE)`.
    - page bytes `[block_size * 576 : block_size * 584)` store per-token scale
      trailers `(7B ue8m0 scales + 1B pad)`.
    - the page stride is padded to a multiple of 576 bytes.

    BatchGen still uses the existing `_insert_into_paged_cache` helper to produce
    logical `[token, 584]` packed rows, but those rows must then be scattered into
    the split page regions above rather than written as contiguous token rows.
    """

    bytes_per_token = TOKEN_BYTES
    qk_nope_head_dim = NOPE_DIM
    qk_rope_head_dim = HEAD_DIM - NOPE_DIM
    store_dtype = torch.uint8
    token_body_bytes = TOKEN_DATA_SIZE
    token_scale_bytes = TOKEN_BYTES - TOKEN_DATA_SIZE

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size_tokens: int,
        device: torch.device | str | int,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if num_pages <= 0:
            raise ValueError(f"num_pages must be > 0, got {num_pages}")
        if page_size_tokens <= 0:
            raise ValueError(
                f"page_size_tokens must be > 0, got {page_size_tokens}"
            )

        self.device = _normalize_device(device)
        self.config = DeepSeekV4PoolConfig(
            num_layers=int(num_layers),
            num_pages=int(num_pages),
            page_size_tokens=int(page_size_tokens),
            bytes_per_token=self.bytes_per_token,
            bytes_per_page_padded=_ceil_div(
                int(page_size_tokens) * self.bytes_per_token, 576
            )
            * 576,
            store_dtype=self.store_dtype,
        )

        self._storage: Optional[torch.Tensor] = None
        self._free_pages: Optional[_PageStack] = None
        self._sequences: Dict[int, _SequenceState] = {}
        self._page_table: Optional[torch.Tensor] = None
        self._active_sequence_ids: Tuple[int, ...] = tuple()
        self._page_table_version = 0
        self.is_initialized = False

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def num_pages(self) -> int:
        return self.config.num_pages

    @property
    def page_size_tokens(self) -> int:
        return self.config.page_size_tokens

    @property
    def bytes_per_page_padded(self) -> int:
        return self.config.bytes_per_page_padded

    def initialize(self) -> None:
        if self.is_initialized:
            return
        self._storage = torch.zeros(
            (
                self.num_layers,
                self.num_pages,
                self.bytes_per_page_padded,
            ),
            dtype=self.store_dtype,
            device=self.device,
        )
        self._free_pages = _PageStack(self.num_pages)
        self._sequences.clear()
        self._page_table = None
        self._active_sequence_ids = tuple()
        self._page_table_version = 0
        self._scale_view_all_layers().zero_()
        self._scale_view_all_layers()[..., :_MODEL1_NUM_TILES] = (
            self._zero_scale_byte()
        )
        self.is_initialized = True

    @staticmethod
    def _zero_scale_byte() -> int:
        zero_scale = torch.pow(
            torch.tensor(2.0, dtype=torch.float32),
            torch.ceil(torch.log2(torch.tensor(1e-4, dtype=torch.float32))),
        )
        return int(zero_scale.to(torch.float8_e8m0fnu).view(torch.uint8).item())

    def _scale_view_all_layers(self) -> torch.Tensor:
        assert self._storage is not None
        body_bytes = self.page_size_tokens * self.token_body_bytes
        scale_bytes = self.page_size_tokens * self.token_scale_bytes
        return self._storage[:, :, body_bytes : body_bytes + scale_bytes].view(
            self.num_layers,
            self.num_pages,
            self.page_size_tokens,
            self.token_scale_bytes,
        )

    def _pack_model1_rows(self, kv_processed: torch.Tensor) -> torch.Tensor:
        if kv_processed.ndim != 2 or kv_processed.shape[-1] != HEAD_DIM:
            raise ValueError(
                f"kv_processed must have shape [N, {HEAD_DIM}], got {tuple(kv_processed.shape)}"
            )
        num_tokens = kv_processed.shape[0]
        packed = torch.empty(
            (num_tokens, self.bytes_per_token),
            dtype=torch.uint8,
            device=self.device,
        )
        packed.zero_()
        packed[:, NOPE_DIM:TOKEN_DATA_SIZE] = (
            kv_processed[:, NOPE_DIM:]
            .contiguous()
            .view(torch.uint8)
            .reshape(num_tokens, -1)
        )

        tiles = (
            kv_processed[:, :NOPE_DIM]
            .float()
            .reshape(num_tokens, _MODEL1_NUM_TILES, _MODEL1_TILE_SIZE)
        )
        scale = torch.pow(
            2.0,
            torch.ceil(
                torch.log2(
                    torch.clamp_min(tiles.abs().amax(dim=-1) / 448.0, 1e-4)
                )
            ),
        )
        packed[:, TOKEN_DATA_SIZE : TOKEN_DATA_SIZE + _MODEL1_NUM_TILES] = (
            scale.to(torch.float8_e8m0fnu).view(torch.uint8)
        )
        packed[:, :NOPE_DIM] = (
            (tiles / scale.unsqueeze(-1))
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .reshape(num_tokens, NOPE_DIM)
        )
        return packed

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        del empty_cuda_cache
        self._storage = None
        self._free_pages = None
        self._sequences.clear()
        self._page_table = None
        self._active_sequence_ids = tuple()
        self._page_table_version = 0
        self.is_initialized = False

    def _ensure_initialized(self) -> None:
        if (
            not self.is_initialized
            or self._storage is None
            or self._free_pages is None
        ):
            raise RuntimeError("DeepSeekV4SingleKVPool is not initialized")

    def _clear_page_table(self) -> None:
        self._page_table = None
        self._active_sequence_ids = tuple()
        self._page_table_version += 1

    def _required_pages(self, token_count: int) -> int:
        if token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {token_count}")
        return _ceil_div(int(token_count), self.page_size_tokens)

    def _get_sequence_state(self, sequence_id: int) -> _SequenceState:
        state = self._sequences.get(sequence_id)
        if state is None:
            raise KeyError(f"Sequence {sequence_id} is not allocated")
        return state

    def _rollback_allocations(self, allocations: Dict[int, List[int]]) -> None:
        if not allocations:
            return
        self._ensure_initialized()
        assert self._free_pages is not None
        reclaimed: List[int] = []
        for seq_id, new_pages in allocations.items():
            if not new_pages:
                continue
            state = self._sequences.get(seq_id)
            if state is None:
                continue
            keep = state.pages.numel() - len(new_pages)
            if keep <= 0:
                self._sequences.pop(seq_id, None)
            else:
                state.pages = state.pages[:keep].clone()
            reclaimed.extend(int(page) for page in new_pages)
        if reclaimed:
            self._free_pages.push(reclaimed)
            self._clear_page_table()

    def allocate_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        num_tokens: Sequence[int],
    ) -> Dict[int, List[int]]:
        self._ensure_initialized()
        if len(sequence_ids) != len(num_tokens):
            raise ValueError(
                "allocate_pages_for_sequences: sequence_ids and num_tokens must match"
            )
        if not sequence_ids:
            return {}

        assert self._free_pages is not None
        allocations: Dict[int, List[int]] = {}
        any_changes = False
        for seq_id, token_count in zip(sequence_ids, num_tokens):
            required_pages = self._required_pages(int(token_count))
            state = self._sequences.get(int(seq_id))
            current_pages = 0 if state is None else int(state.pages.numel())
            missing = max(0, required_pages - current_pages)
            if missing == 0:
                allocations[int(seq_id)] = []
                continue
            new_pages = self._free_pages.pop(missing)
            if state is None:
                self._sequences[int(seq_id)] = _SequenceState(new_pages.clone())
            else:
                state.append_pages(new_pages)
            allocations[int(seq_id)] = new_pages.tolist()
            any_changes = True
        if any_changes:
            self._clear_page_table()
        return allocations

    def extend_pages_for_sequence(
        self, sequence_id: int, new_total_tokens: int
    ) -> int:
        self._ensure_initialized()
        state = self._get_sequence_state(sequence_id)
        required_pages = self._required_pages(new_total_tokens)
        missing = max(0, required_pages - int(state.pages.numel()))
        if missing == 0:
            return 0
        assert self._free_pages is not None
        new_pages = self._free_pages.pop(missing)
        state.append_pages(new_pages)
        self._clear_page_table()
        return missing

    def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        self._ensure_initialized()
        ordered = [int(seq_id) for seq_id in sequence_ids]
        if not ordered:
            raise ValueError(
                "rebuild_page_table: sequence_ids must be non-empty"
            )
        missing = [
            seq_id for seq_id in ordered if seq_id not in self._sequences
        ]
        if missing:
            raise KeyError(
                "rebuild_page_table: unknown sequence ids: "
                + ", ".join(str(seq_id) for seq_id in missing)
            )
        max_pages = max(
            int(self._sequences[seq_id].pages.numel()) for seq_id in ordered
        )
        table = torch.full(
            (len(ordered), max_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        for row, seq_id in enumerate(ordered):
            pages = self._sequences[seq_id].pages.to(device=self.device)
            table[row, : pages.numel()] = pages
        self._page_table = table
        self._active_sequence_ids = tuple(ordered)
        self._page_table_version += 1
        return table

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        self._ensure_initialized()
        if not sequence_ids:
            return
        assert self._free_pages is not None
        reclaimed: List[int] = []
        for seq_id in sequence_ids:
            state = self._sequences.pop(int(seq_id), None)
            if state is None:
                raise KeyError(
                    f"free_pages_for_sequences: unknown sequence {seq_id}"
                )
            reclaimed.extend(int(page) for page in state.pages.tolist())
        if reclaimed:
            self._free_pages.push(reclaimed)
            self._clear_page_table()

    def get_stats(self) -> GPUPagedKVStats:
        self._ensure_initialized()
        assert self._free_pages is not None
        used = self.num_pages - self._free_pages.size
        return GPUPagedKVStats(
            num_total_pages=self.num_pages,
            num_free_pages=self._free_pages.size,
            num_used_pages=used,
            num_total_pages_allocated=used,
        )

    def get_page_table_version(self) -> int:
        self._ensure_initialized()
        return self._page_table_version

    def get_sequence_pages(self, sequence_id: int) -> torch.Tensor:
        self._ensure_initialized()
        return self._get_sequence_state(sequence_id).pages.clone()

    def sequence_token_slots(
        self, sequence_id: int, positions: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        self._ensure_initialized()
        pos = torch.as_tensor(
            list(positions)
            if not isinstance(positions, torch.Tensor)
            else positions
        )
        if pos.ndim != 1:
            raise ValueError(f"positions must be 1D, got {tuple(pos.shape)}")
        if pos.numel() == 0:
            return pos.to(dtype=torch.int64, device=self.device)
        if (pos < 0).any():
            raise ValueError("positions must be non-negative")
        state = self._get_sequence_state(sequence_id)
        page_offsets = torch.div(
            pos.to(torch.int64), self.page_size_tokens, rounding_mode="floor"
        )
        token_offsets = torch.remainder(
            pos.to(torch.int64), self.page_size_tokens
        )
        if int(page_offsets.max().item()) >= state.pages.numel():
            raise ValueError(
                f"Sequence {sequence_id} only has {state.pages.numel()} pages, got positions {pos.tolist()}"
            )
        pages = state.pages.to(
            device=pos.device, dtype=torch.int64
        ).index_select(0, page_offsets.to(torch.int64))
        return pages.to(
            device=self.device
        ) * self.page_size_tokens + token_offsets.to(self.device)

    def _layer_storage(self, layer_idx: int) -> torch.Tensor:
        self._ensure_initialized()
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        assert self._storage is not None
        return self._storage[layer_idx]

    def _token_view(self, layer_idx: int) -> torch.Tensor:
        storage = self._layer_storage(layer_idx)
        return storage[:, : self.page_size_tokens * self.bytes_per_token].view(
            self.num_pages,
            self.page_size_tokens,
            self.bytes_per_token,
        )

    def _body_view(self, layer_idx: int) -> torch.Tensor:
        storage = self._layer_storage(layer_idx)
        body_bytes = self.page_size_tokens * self.token_body_bytes
        return storage[:, :body_bytes].view(
            self.num_pages,
            self.page_size_tokens,
            self.token_body_bytes,
        )

    def _scale_view(self, layer_idx: int) -> torch.Tensor:
        storage = self._layer_storage(layer_idx)
        body_bytes = self.page_size_tokens * self.token_body_bytes
        scale_bytes = self.page_size_tokens * self.token_scale_bytes
        return storage[:, body_bytes : body_bytes + scale_bytes].view(
            self.num_pages,
            self.page_size_tokens,
            self.token_scale_bytes,
        )

    def get_layer_kv_with_page_table(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        self._ensure_initialized()
        if self._page_table is None:
            raise RuntimeError(
                "get_layer_kv_with_page_table: page table is not initialized; call rebuild_page_table first"
            )
        flash_view = (
            self._layer_storage(layer_idx)[
                :, : self.page_size_tokens * self.bytes_per_token
            ]
            .view(torch.float8_e4m3fn)
            .view(
                self.num_pages, self.page_size_tokens, 1, self.bytes_per_token
            )
        )
        return flash_view, None, self._page_table

    def _scatter_packed_rows(
        self,
        layer_idx: int,
        token_slots: torch.Tensor,
        packed_rows: torch.Tensor,
    ) -> None:
        if token_slots.ndim != 1:
            raise ValueError(
                f"token_slots must be 1D, got {tuple(token_slots.shape)}"
            )
        if (
            packed_rows.ndim != 2
            or packed_rows.shape[1] != self.bytes_per_token
        ):
            raise ValueError(
                f"packed_rows must have shape [N, {self.bytes_per_token}], got {tuple(packed_rows.shape)}"
            )
        if token_slots.shape[0] != packed_rows.shape[0]:
            raise ValueError(
                f"token_slots and packed_rows must align, got {token_slots.shape[0]} and {packed_rows.shape[0]}"
            )
        token_slots = token_slots.to(device=self.device, dtype=torch.int64)
        if token_slots.numel() == 0:
            return
        max_slot = self.num_pages * self.page_size_tokens
        if os.environ.get("V4_KV_DEBUG") == "1":
            import sys as _sys

            try:
                torch.cuda.synchronize(self.device)
                _smin = int(token_slots.min().item())
                _smax = int(token_slots.max().item())
                _sys.stderr.write(
                    f"[V4_KV_DEBUG] layer={layer_idx} n={token_slots.numel()} "
                    f"slot_min={_smin} slot_max={_smax} max_slot={max_slot} "
                    f"num_pages={self.num_pages} page_size={self.page_size_tokens} "
                    f"packed_rows={tuple(packed_rows.shape)}\n"
                )
                _sys.stderr.flush()
            except Exception as _e:
                _sys.stderr.write(
                    f"[V4_KV_DEBUG] pre-scatter sync FAILED: {_e}\n"
                )
                _sys.stderr.flush()
                raise
        if (token_slots < 0).any() or (token_slots >= max_slot).any():
            raise ValueError(
                f"token_slots out of range for capacity {max_slot}: {token_slots.tolist()}"
            )
        page_indices = torch.div(
            token_slots, self.page_size_tokens, rounding_mode="floor"
        )
        token_offsets = torch.remainder(token_slots, self.page_size_tokens)
        packed_rows = packed_rows.to(device=self.device, dtype=self.store_dtype)
        body_view = self._body_view(layer_idx)
        scale_view = self._scale_view(layer_idx)
        body_view[page_indices, token_offsets] = packed_rows[
            :, : self.token_body_bytes
        ]
        scale_view[page_indices, token_offsets] = packed_rows[
            :, self.token_body_bytes :
        ]

    def _gather_packed_rows(
        self,
        layer_idx: int,
        token_slots: torch.Tensor,
    ) -> torch.Tensor:
        rows = token_slots.to(device=self.device, dtype=torch.int64)
        if rows.ndim != 1:
            raise ValueError(f"token_slots must be 1D, got {tuple(rows.shape)}")
        if rows.numel() == 0:
            return torch.empty(
                (0, self.bytes_per_token),
                dtype=self.store_dtype,
                device=self.device,
            )
        page_indices = torch.div(
            rows, self.page_size_tokens, rounding_mode="floor"
        )
        token_offsets = torch.remainder(rows, self.page_size_tokens)
        bodies = self._body_view(layer_idx)[page_indices, token_offsets]
        scales = self._scale_view(layer_idx)[page_indices, token_offsets]
        return torch.cat((bodies, scales), dim=-1)

    def store_kv(
        self,
        *,
        layer_idx: int,
        token_slots: torch.Tensor | Sequence[int],
        kv_processed: torch.Tensor,
    ) -> None:
        rows = torch.as_tensor(
            list(token_slots)
            if not isinstance(token_slots, torch.Tensor)
            else token_slots,
            device=self.device,
        )
        if kv_processed.ndim != 2 or kv_processed.shape[-1] != HEAD_DIM:
            raise ValueError(
                f"kv_processed must have shape [N, {HEAD_DIM}], got {tuple(kv_processed.shape)}"
            )
        scratch = self._pack_model1_rows(kv_processed.to(device=self.device))
        self._scatter_packed_rows(layer_idx, rows, scratch)

    def store_qnorm_rope_kv(
        self,
        *,
        layer_idx: int,
        token_slots: torch.Tensor | Sequence[int],
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = torch.as_tensor(
            list(token_slots)
            if not isinstance(token_slots, torch.Tensor)
            else token_slots,
            device=self.device,
        )
        scratch = torch.empty(
            (q.shape[0], self.bytes_per_token),
            dtype=torch.uint8,
            device=self.device,
        )
        q_out, kv_out = fused_v4_qnorm_rope_kv_insert(
            q=q,
            kv=kv,
            kv_weight=kv_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            kv_cache=scratch,
            block_table=torch.arange(q.shape[0], device=self.device),
            eps=eps,
        )
        self._scatter_packed_rows(
            layer_idx,
            rows,
            self._pack_model1_rows(kv_out.to(device=self.device)),
        )
        return q_out, kv_out

    def debug_read_kv(
        self,
        *,
        layer_idx: int,
        token_slots: torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        rows = torch.as_tensor(
            list(token_slots)
            if not isinstance(token_slots, torch.Tensor)
            else token_slots,
            device=self.device,
            dtype=torch.int64,
        )
        if rows.ndim != 1:
            raise ValueError(f"token_slots must be 1D, got {tuple(rows.shape)}")
        if rows.numel() == 0:
            return torch.empty(
                (0, HEAD_DIM), dtype=torch.bfloat16, device=self.device
            )
        packed = self._gather_packed_rows(layer_idx, rows)
        nope_fp8 = packed[:, :NOPE_DIM].view(torch.float8_e4m3fn)
        rope = (
            packed[:, NOPE_DIM:TOKEN_DATA_SIZE]
            .contiguous()
            .view(torch.bfloat16)
        )
        scales = packed[:, TOKEN_DATA_SIZE:TOKEN_BYTES][:, : NOPE_DIM // 64]
        nope = dequantize_nope_from_fp8(nope_fp8, scales)
        return torch.cat((nope.to(torch.bfloat16), rope), dim=-1)


class DeepSeekV4IndexerPool:
    bytes_per_token = 128 + _INDEXER_SCALE_BYTES
    store_dtype = torch.uint8

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size_tokens: int,
        indexer_head_dim: int,
        device: torch.device | str | int,
    ) -> None:
        if indexer_head_dim <= 0:
            raise ValueError(
                f"indexer_head_dim must be > 0, got {indexer_head_dim}"
            )
        if indexer_head_dim % _INDEXER_QUANT_BLOCK_SIZE != 0:
            raise ValueError(
                "indexer_head_dim must be divisible by 128 for scale layout"
            )
        self.indexer_head_dim = int(indexer_head_dim)
        self.device = _normalize_device(device)
        self.config = DeepSeekV4PoolConfig(
            num_layers=int(num_layers),
            num_pages=int(num_pages),
            page_size_tokens=int(page_size_tokens),
            bytes_per_token=self.bytes_per_token,
            bytes_per_page_padded=int(page_size_tokens) * self.bytes_per_token,
            store_dtype=self.store_dtype,
        )
        self._storage: Optional[torch.Tensor] = None
        self._free_pages: Optional[_PageStack] = None
        self._sequences: Dict[int, _SequenceState] = {}
        self._page_table: Optional[torch.Tensor] = None
        self.is_initialized = False

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def num_pages(self) -> int:
        return self.config.num_pages

    @property
    def page_size_tokens(self) -> int:
        return self.config.page_size_tokens

    def initialize(self) -> None:
        if self.is_initialized:
            return
        self._storage = torch.zeros(
            (
                self.num_layers,
                self.num_pages,
                self.config.bytes_per_page_padded,
            ),
            dtype=self.store_dtype,
            device=self.device,
        )
        self._free_pages = _PageStack(self.num_pages)
        self._sequences.clear()
        self._page_table = None
        self.is_initialized = True

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        del empty_cuda_cache
        self._storage = None
        self._free_pages = None
        self._sequences.clear()
        self._page_table = None
        self.is_initialized = False

    def _ensure_initialized(self) -> None:
        if (
            not self.is_initialized
            or self._storage is None
            or self._free_pages is None
        ):
            raise RuntimeError("DeepSeekV4IndexerPool is not initialized")

    def _required_pages(self, token_count: int) -> int:
        if token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {token_count}")
        return _ceil_div(token_count, self.page_size_tokens)

    def _clear_page_table(self) -> None:
        self._page_table = None

    def _token_view(self, layer_idx: int) -> torch.Tensor:
        self._ensure_initialized()
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        assert self._storage is not None
        return self._storage[layer_idx].view(
            self.num_pages, self.page_size_tokens, self.bytes_per_token
        )

    def _rollback_allocations(self, allocations: Dict[int, List[int]]) -> None:
        if not allocations:
            return
        self._ensure_initialized()
        assert self._free_pages is not None
        reclaimed: List[int] = []
        for seq_id, new_pages in allocations.items():
            if not new_pages:
                continue
            state = self._sequences.get(seq_id)
            if state is None:
                continue
            keep = state.pages.numel() - len(new_pages)
            if keep <= 0:
                self._sequences.pop(seq_id, None)
            else:
                state.pages = state.pages[:keep].clone()
            reclaimed.extend(int(page) for page in new_pages)
        if reclaimed:
            self._free_pages.push(reclaimed)
            self._clear_page_table()

    def allocate_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        num_tokens: Sequence[int],
    ) -> Dict[int, List[int]]:
        self._ensure_initialized()
        if len(sequence_ids) != len(num_tokens):
            raise ValueError(
                "allocate_pages_for_sequences: sequence_ids and num_tokens must match"
            )
        assert self._free_pages is not None
        allocations: Dict[int, List[int]] = {}
        changed = False
        for seq_id, token_count in zip(sequence_ids, num_tokens):
            required_pages = self._required_pages(int(token_count))
            state = self._sequences.get(int(seq_id))
            current = 0 if state is None else int(state.pages.numel())
            missing = max(0, required_pages - current)
            if missing == 0:
                allocations[int(seq_id)] = []
                continue
            new_pages = self._free_pages.pop(missing)
            if state is None:
                self._sequences[int(seq_id)] = _SequenceState(new_pages.clone())
            else:
                state.append_pages(new_pages)
            allocations[int(seq_id)] = new_pages.tolist()
            changed = True
        if changed:
            self._clear_page_table()
        return allocations

    def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        self._ensure_initialized()
        ordered = [int(seq_id) for seq_id in sequence_ids]
        if not ordered:
            raise ValueError(
                "rebuild_page_table: sequence_ids must be non-empty"
            )
        missing = [
            seq_id for seq_id in ordered if seq_id not in self._sequences
        ]
        if missing:
            raise KeyError(
                "rebuild_page_table: unknown sequence ids: "
                + ", ".join(str(seq_id) for seq_id in missing)
            )
        max_pages = max(
            int(self._sequences[seq_id].pages.numel()) for seq_id in ordered
        )
        table = torch.full(
            (len(ordered), max_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        for row, seq_id in enumerate(ordered):
            table[row, : self._sequences[seq_id].pages.numel()] = (
                self._sequences[seq_id].pages.to(device=self.device)
            )
        self._page_table = table
        return table

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        self._ensure_initialized()
        assert self._free_pages is not None
        reclaimed: List[int] = []
        for seq_id in sequence_ids:
            state = self._sequences.pop(int(seq_id), None)
            if state is None:
                raise KeyError(
                    f"free_pages_for_sequences: unknown sequence {seq_id}"
                )
            reclaimed.extend(int(page) for page in state.pages.tolist())
        if reclaimed:
            self._free_pages.push(reclaimed)
            self._clear_page_table()

    def get_stats(self) -> GPUPagedKVStats:
        self._ensure_initialized()
        assert self._free_pages is not None
        used = self.num_pages - self._free_pages.size
        return GPUPagedKVStats(
            num_total_pages=self.num_pages,
            num_free_pages=self._free_pages.size,
            num_used_pages=used,
            num_total_pages_allocated=used,
        )

    def get_sequence_pages(self, sequence_id: int) -> torch.Tensor:
        self._ensure_initialized()
        state = self._sequences.get(sequence_id)
        if state is None:
            raise KeyError(f"Sequence {sequence_id} is not allocated")
        return state.pages.clone()

    def sequence_token_slots(
        self, sequence_id: int, positions: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        pos = torch.as_tensor(
            list(positions)
            if not isinstance(positions, torch.Tensor)
            else positions
        )
        if pos.ndim != 1:
            raise ValueError(f"positions must be 1D, got {tuple(pos.shape)}")
        state = self._sequences.get(sequence_id)
        if state is None:
            raise KeyError(f"Sequence {sequence_id} is not allocated")
        page_offsets = torch.div(
            pos.to(torch.int64), self.page_size_tokens, rounding_mode="floor"
        )
        token_offsets = torch.remainder(
            pos.to(torch.int64), self.page_size_tokens
        )
        if (
            pos.numel()
            and int(page_offsets.max().item()) >= state.pages.numel()
        ):
            raise ValueError(
                f"Sequence {sequence_id} only has {state.pages.numel()} pages, got positions {pos.tolist()}"
            )
        pages = state.pages.to(
            device=pos.device, dtype=torch.int64
        ).index_select(0, page_offsets)
        return pages.to(
            device=self.device
        ) * self.page_size_tokens + token_offsets.to(self.device)

    def store_indexer(
        self,
        *,
        layer_idx: int,
        token_slots: torch.Tensor | Sequence[int],
        index_k: torch.Tensor,
    ) -> None:
        if index_k.ndim != 2 or index_k.shape[-1] != self.indexer_head_dim:
            raise ValueError(
                f"index_k must have shape [N, {self.indexer_head_dim}], got {tuple(index_k.shape)}"
            )
        rows = torch.as_tensor(
            list(token_slots)
            if not isinstance(token_slots, torch.Tensor)
            else token_slots,
            device=self.device,
            dtype=torch.int64,
        )
        if rows.shape[0] != index_k.shape[0]:
            raise ValueError("token_slots and index_k must align")
        token_view = self._token_view(layer_idx)
        page_indices = torch.div(
            rows, self.page_size_tokens, rounding_mode="floor"
        )
        token_offsets = torch.remainder(rows, self.page_size_tokens)

        scale = torch.abs(index_k.float()).amax(dim=-1) / _INDEXER_FP8_MAX
        scale = torch.clamp_min(scale, 1e-4)
        quantized = (index_k.float() / scale.unsqueeze(-1)).to(
            torch.float8_e4m3fn
        )

        packed = torch.empty(
            (index_k.shape[0], self.bytes_per_token),
            dtype=torch.uint8,
            device=self.device,
        )
        packed[:, : self.indexer_head_dim] = quantized.view(torch.uint8)
        packed[:, self.indexer_head_dim :] = (
            scale.to(torch.float32)
            .view(torch.uint8)
            .view(index_k.shape[0], _INDEXER_SCALE_BYTES)
        )
        token_view[page_indices, token_offsets] = packed

    def debug_read_indexer(
        self,
        *,
        layer_idx: int,
        token_slots: torch.Tensor | Sequence[int],
    ) -> torch.Tensor:
        rows = torch.as_tensor(
            list(token_slots)
            if not isinstance(token_slots, torch.Tensor)
            else token_slots,
            device=self.device,
            dtype=torch.int64,
        )
        page_indices = torch.div(
            rows, self.page_size_tokens, rounding_mode="floor"
        )
        token_offsets = torch.remainder(rows, self.page_size_tokens)
        packed = self._token_view(layer_idx)[page_indices, token_offsets]
        quantized = packed[:, : self.indexer_head_dim].view(torch.float8_e4m3fn)
        scales = (
            packed[:, self.indexer_head_dim :]
            .contiguous()
            .view(rows.shape[0], _INDEXER_SCALE_BYTES)
            .view(torch.float32)
            .squeeze(-1)
        )
        return (quantized.float() * scales.unsqueeze(-1)).to(torch.bfloat16)


__all__ = [
    "DeepSeekV4IndexerPool",
    "DeepSeekV4PoolConfig",
    "DeepSeekV4SingleKVPool",
]
