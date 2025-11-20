from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from batchgen.config.config import EngineConfig
from batchgen.kv_cache.gpu_kv_kernels import (
    run_paged_kv_token_update,
    run_paged_kv_token_update_fused,
)

# Default initial per-sequence token capacity used when first creating the
# GPU-side page table. 16384 tokens -> with 64-token pages means 256 pages
# per sequence (this is conservative and can be recomputed from config).
DEFAULT_INITIAL_TOKEN_CAPACITY = 16384


def _require_positive(value: Optional[int], field_name: str) -> int:
    """Ensures that ``value`` is a strictly positive integer."""
    if value is None or value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {value}")
    return int(value)


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return -(-value // divisor)


def _as_int_tensor(values: Iterable[int]) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=torch.int32)


@dataclass(frozen=True)
class GPUPagedKVStats:
    num_total_pages: int
    num_free_pages: int
    num_used_pages: int
    num_total_pages_allocated: int


@dataclass(frozen=True)
class GPUPagedKVConfig:
    num_layers: int
    num_pages: int
    page_size_tokens: int
    num_k_heads: int
    k_head_dim: int
    num_v_heads: Optional[int]
    v_head_dim: Optional[int]
    kv_dtype: torch.dtype

    @classmethod
    def from_engine(cls, engine_config) -> "GPUPagedKVConfig":  # noqa: D401
        cfg = engine_config.gpu_kv_config
        num_layers = _require_positive(
            getattr(cfg, "num_layers", None), "num_layers"
        )
        num_pages = _require_positive(
            getattr(cfg, "num_pages", None), "num_pages"
        )
        page_size = _require_positive(
            getattr(cfg, "page_size_tokens", None), "page_size_tokens"
        )
        num_k_heads = _require_positive(
            getattr(cfg, "num_k_heads", None), "num_k_heads"
        )
        k_head_dim = _require_positive(
            getattr(cfg, "k_head_dim", None), "k_head_dim"
        )
        kv_dtype = getattr(cfg, "kv_dtype", None)
        if kv_dtype is None:
            raise ValueError("gpu_kv_config.kv_dtype must be defined")
        num_v_heads = getattr(cfg, "num_v_heads", None)
        v_head_dim = getattr(cfg, "v_head_dim", None)
        if num_v_heads is not None and num_v_heads > 0:
            v_head_dim = _require_positive(v_head_dim, "v_head_dim")
        else:
            num_v_heads = 0
            v_head_dim = 0
        return cls(
            num_layers=num_layers,
            num_pages=num_pages,
            page_size_tokens=page_size,
            num_k_heads=num_k_heads,
            k_head_dim=k_head_dim,
            num_v_heads=num_v_heads,
            v_head_dim=v_head_dim,
            kv_dtype=kv_dtype,
        )

    @property
    def has_v_cache(self) -> bool:
        return bool(self.num_v_heads and self.v_head_dim)


class _TensorStack:
    """A lightweight stack backed by a tensor for deterministic page allocation."""

    def __init__(self, capacity: int):
        self._buffer = torch.empty(capacity, dtype=torch.int32)
        self._size = 0
        if capacity > 0:
            self.push(torch.arange(capacity - 1, -1, -1, dtype=torch.int32))

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._buffer.numel()

    def push(self, values: torch.Tensor | Iterable[int]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.int32)
        count = int(tensor.numel())
        if self._size + count > self.capacity:
            raise RuntimeError("TensorStack overflow")
        self._buffer[self._size : self._size + count] = tensor
        self._size += count

    def pop(self, count: int) -> torch.Tensor:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > self._size:
            raise RuntimeError("Insufficient elements in TensorStack")
        start = self._size - count
        result = self._buffer[start : self._size].clone()
        self._size = start
        return result


@dataclass
class _SequenceState:
    pages: torch.Tensor

    def capacity_tokens(self, page_size_tokens: int) -> int:
        return int(self.pages.numel()) * page_size_tokens

    def append_pages(self, new_pages: torch.Tensor) -> None:
        if self.pages.numel() == 0:
            self.pages = new_pages.clone()
        else:
            self.pages = torch.cat([self.pages, new_pages], dim=0)


class GPUPagedKVGeometry:
    """Utility helpers mirroring the host geometry checks."""

    def __init__(self, config: GPUPagedKVConfig):
        self._config = config

    @property
    def config(self) -> GPUPagedKVConfig:
        return self._config

    def ensure_layer_bounds(self, layer_idx: int, context: str) -> None:
        if layer_idx < 0 or layer_idx >= self._config.num_layers:
            raise IndexError(f"{context}: layer_idx {layer_idx} out of range")

    def ensure_positive_tokens(self, num_tokens: int, context: str) -> None:
        if num_tokens <= 0:
            raise ValueError(
                f"{context}: num_tokens must be > 0, got {num_tokens}"
            )

    def required_pages(
        self,
        num_tokens: Union[int, Sequence[int], torch.Tensor],
    ) -> Union[int, torch.Tensor]:
        """Returns the number of pages required for ``num_tokens``."""

        if isinstance(num_tokens, int):
            self.ensure_positive_tokens(num_tokens, "required_pages")
            return _ceil_div(num_tokens, self._config.page_size_tokens)

        tokens = self._normalize_token_vector(num_tokens, "required_pages")
        page_size = self._config.page_size_tokens
        return torch.div(
            tokens + (page_size - 1), page_size, rounding_mode="floor"
        )

    def _normalize_token_vector(
        self, values: Union[Sequence[int], torch.Tensor], context: str
    ) -> torch.Tensor:
        tokens = torch.as_tensor(values, dtype=torch.long, device="cpu")
        if tokens.dim() != 1:
            raise ValueError(
                f"{context}: expected 1-D tensor of token counts, got shape {tuple(tokens.shape)}"
            )
        if tokens.numel() == 0:
            raise ValueError(f"{context}: token count vector must be non-empty")
        if torch.any(tokens <= 0):
            raise ValueError(f"{context}: all token counts must be positive")
        return tokens


class GPUPagedKVLayout:
    """Computes tensor shapes and byte sizes similar to the host layout."""

    def __init__(self, config: GPUPagedKVConfig):
        self._config = config
        element_size = torch.tensor([], dtype=config.kv_dtype).element_size()
        self._k_page_bytes = (
            config.page_size_tokens
            * config.num_k_heads
            * config.k_head_dim
            * element_size
        )
        if config.has_v_cache:
            self._v_page_bytes = (
                config.page_size_tokens
                * config.num_v_heads
                * config.v_head_dim
                * element_size
            )
        else:
            self._v_page_bytes = 0

    @property
    def k_page_bytes(self) -> int:
        return self._k_page_bytes

    @property
    def v_page_bytes(self) -> int:
        return self._v_page_bytes


class _GPUPageTableManager:
    """Lightweight manager that maintains a 2-D GPU tensor acting as a
    page table. The table layout is [num_slots, max_pages_per_sequence]
    and uses -1 as the padding value for sequences with fewer pages.

    The manager also keeps two small host-side mappings to translate
    sequence ids <-> slot indices: ``seq_id_to_slot`` and
    ``slot_to_seq_id``. The table is rebuilt (on device) on demand when
    `rebuild` is called. The rebuild operation is intentionally simple and
    safe because it is expected to be infrequent.
    """

    def __init__(self, device: torch.device, max_pages_per_sequence: int):
        self.device = device
        self.max_pages_per_sequence = int(max_pages_per_sequence)
        self.seq_id_to_slot: Dict[int, int] = {}
        self.slot_to_seq_id: List[
            Optional[int]
        ] = []  # this is also sequence order
        # GPU tensor of shape [num_slots, max_pages_per_sequence]
        self.gpu_table: Optional[torch.Tensor] = None
        # Small cached 1-D slot_ids,like [0, 1, 2, ...] total length is len(sequence_ids)
        self._slot_index_tensor: Optional[torch.Tensor] = None
        # Cached tensor mirroring slot_to_seq_id on device for downstream consumers.
        self._slot_to_seq_id_tensor: Optional[torch.Tensor] = None

    def rebuild(
        self,
        sequence_ids: Sequence[int],
        sequences: Dict[int, _SequenceState],
    ) -> torch.Tensor:
        """(Re)build the GPU page table for the provided sequence order.

        Optimization: if the requested `sequence_ids` exactly matches the
        current `slot_to_seq_id` prefix and a GPU table already exists,
        return a view into the existing tensor instead of rebuilding.
        """
        wanted_order = list(sequence_ids)

        # Compute the maximum number of pages required by the provided
        # sequences. If any sequence requires more than the current
        # max_pages_per_sequence, expand the table size by at least one
        # extra page (leave an extra slot) to reduce immediate re-resize.
        max_required = 0
        for seq_id in wanted_order:
            state = sequences.get(seq_id)
            if state is None:
                continue
            max_required = max(max_required, int(state.pages.numel()))

        # Ensure at least one extra page margin.
        desired_pages = max_required + 1
        new_max = max(self.max_pages_per_sequence, desired_pages)

        num_slots = len(wanted_order)

        # Fast-path: if an existing GPU table already matches the exact
        # shape we need and the order is identical, return a view into it.
        if (
            self.gpu_table is not None
            and self.gpu_table.shape[0] == num_slots
            and self.gpu_table.shape[1] >= new_max
            and self.slot_to_seq_id == wanted_order
        ):
            self._slot_index_tensor = self._build_slot_index_tensor(num_slots)
            self._slot_to_seq_id_tensor = self._build_slot_to_seq_id_tensor(
                wanted_order
            )
            # If table has more columns than needed, return a view with
            # the requested number of rows and the existing columns.
            return self.gpu_table[:num_slots, :]

        # Build a fresh GPU table with the computed new_max columns.
        table = torch.full(
            (num_slots, new_max), -1, dtype=torch.int32, device=self.device
        )

        # Replace mappings with exactly the requested order. This keeps
        # the slot mapping deterministic and equal in length to num_slots.
        self.seq_id_to_slot = {
            seq_id: idx for idx, seq_id in enumerate(wanted_order)
        }
        self.slot_to_seq_id = list(wanted_order)
        self.max_pages_per_sequence = int(new_max)

        # Fill table rows from sequence state pages.
        for slot, seq_id in enumerate(wanted_order):
            state = sequences.get(seq_id)
            if state is None or state.pages.numel() == 0:
                continue
            pages = state.pages.to(self.device, dtype=torch.int32)
            count = int(pages.numel())
            # pages should fit into new_max because we ensured desired_pages
            if count > new_max:
                # As a last-resort safety: truncate to new_max and log.
                logging.warning(
                    "Sequence %s has %d pages > new_max %d; truncating",
                    seq_id,
                    count,
                    new_max,
                )
                count = new_max
                pages = pages[:count]
            table[slot, :count] = pages[:count]

        self.gpu_table = table
        self._slot_index_tensor = self._build_slot_index_tensor(num_slots)
        self._slot_to_seq_id_tensor = self._build_slot_to_seq_id_tensor(
            self.slot_to_seq_id
        )
        return table

    def get_slot_index_tensor(self) -> torch.Tensor:
        """Returns a cached 1-D tensor mapping logical batch order to slots."""
        if self._slot_index_tensor is None:
            raise RuntimeError(
                "GPU page table slot indices unavailable; call rebuild() first"
            )
        tensor = self._slot_index_tensor
        return tensor

    def get_slot_to_seq_id_tensor(
        self, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Returns cached sequence ids ordered by the current slot mapping."""
        if self._slot_to_seq_id_tensor is None:
            raise RuntimeError(
                "GPU page table seq-id tensor unavailable; call rebuild() first"
            )
        tensor = self._slot_to_seq_id_tensor
        if batch_size is None:
            return tensor
        if batch_size > tensor.shape[0]:
            raise ValueError(
                f"Requested batch_size {batch_size} exceeds cached slot-to-seq tensor length {tensor.shape[0]}"
            )
        return tensor[:batch_size]

    def _build_slot_index_tensor(self, num_slots: int) -> torch.Tensor:
        return torch.arange(num_slots, dtype=torch.int32, device=self.device)

    def _build_slot_to_seq_id_tensor(
        self, sequence_ids: Sequence[int]
    ) -> torch.Tensor:
        if not sequence_ids:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.as_tensor(
            sequence_ids, dtype=torch.int64, device=self.device
        )


class GPUPagedKVCacheManager:
    """Per-GPU paged KV cache manager inspired by the host implementation."""

    def __init__(self, engine_config: EngineConfig):
        self._engine_config = engine_config
        self.config = GPUPagedKVConfig.from_engine(engine_config)
        self._geometry = GPUPagedKVGeometry(self.config)
        self._layout = GPUPagedKVLayout(self.config)
        self.device = torch.device(engine_config.Basic_Config.device)
        self._core_engine = None

        self._k_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None
        self._k_page_ptr_table: Optional[torch.Tensor] = (
            None  # [layer_id, num_pages]
        )
        self._v_page_ptr_table: Optional[torch.Tensor] = None
        self._free_pages = _TensorStack(self.config.num_pages)
        self._sequences: Dict[int, _SequenceState] = {}
        # GPU-side page table manager (lazy updated in allocate_pages_for_sequences)
        max_pages_per_seq = _ceil_div(
            DEFAULT_INITIAL_TOKEN_CAPACITY, self.config.page_size_tokens
        )
        self._gpu_page_table_manager: _GPUPageTableManager = (
            _GPUPageTableManager(
                device=self.device, max_pages_per_sequence=max_pages_per_seq
            )
        )

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def initialize(self, core_engine) -> None:
        """Instantiates GPU tensors and prepares allocator state."""

        self._set_device()
        self._core_engine = core_engine
        shape = (
            self.config.num_layers,
            self.config.num_pages,
            self.config.page_size_tokens,
            self.config.num_k_heads,
            self.config.k_head_dim,
        )
        self._k_cache = torch.empty(
            shape, dtype=self.config.kv_dtype, device=self.device
        )
        logging.debug("Initialized K cache %s", tuple(self._k_cache.shape))

        if self.config.has_v_cache:
            v_shape = (
                self.config.num_layers,
                self.config.num_pages,
                self.config.page_size_tokens,
                self.config.num_v_heads,
                self.config.v_head_dim,
            )
            self._v_cache = torch.empty(
                v_shape, dtype=self.config.kv_dtype, device=self.device
            )
            logging.debug("Initialized V cache %s", tuple(self._v_cache.shape))
        else:
            self._v_cache = None

        self._refresh_page_pointer_tables()

        total_bytes = self._k_cache.element_size() * self._k_cache.numel() + (
            self._v_cache.element_size() * self._v_cache.numel()
            if self._v_cache is not None
            else 0
        )
        logging.info(
            "GPUPagedKVCacheManager ready (device=%s, layers=%d, pages=%d, v_cache=%s, total_bytes=%d (%fGB))",
            self.device,
            self.config.num_layers,
            self.config.num_pages,
            "enabled" if self.config.has_v_cache else "disabled",
            total_bytes,
            total_bytes / (1024**3),
        )

    def allocate_pages(self, sequence_id: int, num_tokens: int) -> List[int]:
        """Allocates enough pages to hold ``num_tokens`` for ``sequence_id``."""

        self._ensure_initialized()
        required_pages = self._geometry.required_pages(num_tokens)
        state = self._sequences.get(sequence_id)
        current_pages = state.pages.numel() if state else 0
        missing = max(0, required_pages - current_pages)
        if missing == 0:
            return []
        if missing > self._free_pages.size:
            raise RuntimeError(
                f"Insufficient free pages: need {missing}, have {self._free_pages.size}"
            )
        new_pages = self._free_pages.pop(missing)
        if state is None:
            state = _SequenceState(pages=new_pages)
            self._sequences[sequence_id] = state
        else:
            state.append_pages(new_pages)
        return new_pages.tolist()

    def allocate_pages_for_sequences(
        self, sequence_ids: Sequence[int], num_tokens: Sequence[int]
    ) -> Dict[int, List[int]]:
        """Allocates enough pages to hold `num_tokens` for each sequence."""

        self._ensure_initialized()
        if len(sequence_ids) != len(num_tokens):
            raise ValueError(
                "allocate_pages_for_sequences: sequence_ids and num_tokens must be the same length"
            )
        if not sequence_ids:
            return {}

        required_pages = self._geometry.required_pages(num_tokens).tolist()
        allocations: Dict[int, List[int]] = {}

        for seq_id, required in zip(sequence_ids, required_pages):
            required_int = int(required)
            if required_int <= 0:
                raise ValueError(
                    f"allocate_pages_for_sequences: required pages must be positive for seq {seq_id}, got {required_int}"
                )

            state = self._sequences.get(seq_id)
            current = state.pages.numel() if state else 0
            missing = max(0, required_int - current)
            if missing == 0:
                allocations[seq_id] = []
                continue

            if missing > self._free_pages.size:
                raise RuntimeError(
                    f"Insufficient free pages for seq {seq_id}: need {missing}, free {self._free_pages.size}"
                )

            new_pages = self._free_pages.pop(missing)
            if state is None:
                self._sequences[seq_id] = _SequenceState(pages=new_pages)
            else:
                state.append_pages(new_pages)
            allocations[seq_id] = new_pages.tolist()
        return allocations

    def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        """Rebuilds the GPU page table following ``sequence_ids`` order.

        Args:
            sequence_ids: Logical sequence ordering to materialize on device.

        Returns:
            torch.Tensor: GPU-resident page table view aligned with ``sequence_ids``.
        """

        self._ensure_initialized()
        ordered_ids = list(sequence_ids)
        if not ordered_ids:
            raise ValueError(
                "rebuild_page_table: sequence_ids must be non-empty"
            )

        missing = [
            seq_id for seq_id in ordered_ids if seq_id not in self._sequences
        ]
        if missing:
            raise KeyError(
                "rebuild_page_table: sequence_ids contain unallocated sequences: "
                + ", ".join(str(seq_id) for seq_id in missing)
            )

        return self._gpu_page_table_manager.rebuild(
            ordered_ids, self._sequences
        )

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        """Batch version of :meth:`free_sequence` for releasing multiple IDs."""

        self._ensure_initialized()
        if not sequence_ids:
            return

        missing = [
            seq_id for seq_id in sequence_ids if seq_id not in self._sequences
        ]
        if missing:
            raise KeyError(
                "free_pages_for_sequences: unknown sequence ids: "
                + ", ".join(str(seq_id) for seq_id in missing)
            )

        reclaimed: List[torch.Tensor] = []
        for seq_id in sequence_ids:
            state = self._sequences.pop(seq_id)
            reclaimed.append(state.pages)

        if reclaimed:
            concatenated = torch.cat(reclaimed, dim=0)
            self._free_pages.push(concatenated)

    def get_context_kv_page_ptrs(
        self, sequence_id: int, layer_idx: int, context_length: int
    ) -> Tuple[List[int], Optional[List[int]]]:
        """Returns contiguous GPU page pointers for context loading."""

        self._ensure_initialized()
        self._geometry.ensure_layer_bounds(
            layer_idx, "get_context_kv_page_ptrs"
        )
        state = self._get_sequence_state(sequence_id)
        required_pages = self._geometry.required_pages(context_length)
        if required_pages > state.pages.numel():
            raise ValueError(
                f"Sequence {sequence_id} requires {required_pages} pages, only {state.pages.numel()} allocated"
            )
        page_indices = state.pages[:required_pages].tolist()
        k_ptrs = [
            self._k_cache[layer_idx, idx].data_ptr() for idx in page_indices
        ]
        v_ptrs = None
        if self._v_cache is not None:
            v_ptrs = [
                self._v_cache[layer_idx, idx].data_ptr() for idx in page_indices
            ]
        return k_ptrs, v_ptrs

    def get_sequence_layer_page_pointers(
        self, sequence_id: int, layer_idx: int
    ) -> Tuple[List[int], Optional[List[int]]]:
        """Returns page pointers for copy operations between host and device."""

        self._ensure_initialized()
        self._geometry.ensure_layer_bounds(
            layer_idx, "get_sequence_layer_page_pointers"
        )
        state = self._get_sequence_state(sequence_id)
        page_indices = state.pages.tolist()
        k_ptrs = [
            self._k_cache[layer_idx, idx].data_ptr() for idx in page_indices
        ]
        if self._v_cache is None:
            return k_ptrs, None
        v_ptrs = [
            self._v_cache[layer_idx, idx].data_ptr() for idx in page_indices
        ]
        return k_ptrs, v_ptrs

    def update_new_token(
        self,
        k_tensor: torch.Tensor,
        v_tensor: Optional[torch.Tensor],
        sequence_lengths: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Writes single-position KV tokens for ``layer_idx`` using the cached
        GPU page table order.

        The provided ``k_tensor`` (and optional ``v_tensor``) must align with
        the sequence ordering used when ``allocate_pages_for_sequences`` last
        triggered a rebuild of the GPU page table.
        """
        # op_name = "update_new_token"
        # self._ensure_initialized()
        # self._geometry.ensure_layer_bounds(layer_idx, op_name)
        # self._validate_token_inputs(k_tensor, v_tensor)
        start = time.time()
        batch_size, seq_len, _, _ = k_tensor.shape
        if seq_len != 1:
            raise ValueError(
                f"update_new_token: k_tensor must have sequence dimension 1, got {seq_len}"
            )

        page_table = self._gpu_page_table_manager.gpu_table
        if page_table is None:
            raise RuntimeError(
                f"update_new_token: GPU page table is not initialized; "
                "call allocate_pages_for_sequences before updating tokens"
            )

        # all the tensors are continuous
        # slot_indices = self._gpu_page_table_manager.get_slot_index_tensor()
        slot_indices = self._gpu_page_table_manager._slot_index_tensor
        token_indices = sequence_lengths

        page_table_view = page_table
        k_tokens = k_tensor.view(batch_size, -1)

        if v_tensor is not None and self._v_cache is not None:
            v_tokens: Optional[torch.Tensor] = v_tensor.view(batch_size, -1)
        else:
            v_tokens = None

        k_cache_layer = self._k_cache[layer_idx]
        v_cache_layer = None
        if self._v_cache is not None:
            v_cache_layer = self._v_cache[layer_idx]

        end = time.time()
        print(f"Preprocessing time: {(end - start) * 1000:.6f} ms")

        run_paged_kv_token_update_fused(
            k_cache=k_cache_layer,
            k_tokens=k_tokens,
            page_table=page_table_view,
            slot_indices=slot_indices,
            token_indices=token_indices,
            page_size_tokens=self.config.page_size_tokens,
            v_cache=v_cache_layer,
            v_tokens=v_tokens,
        )

    def get_layer_kv_with_page_table(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns layer tensors alongside a placeholder for the page table."""
        self._ensure_initialized()
        self._geometry.ensure_layer_bounds(
            layer_idx, "get_layer_kv_with_page_table"
        )
        gpu_table = None
        mgr = self._gpu_page_table_manager
        gpu_table = mgr.gpu_table
        if gpu_table is not None:
            return (
                self._k_cache[layer_idx],
                None if self._v_cache is None else self._v_cache[layer_idx],
                gpu_table,
            )
        else:
            raise RuntimeError(
                "get_layer_kv_with_page_table: GPU page table is not initialized; "
                "call allocate_pages_for_sequences and build_page_table before using this method"
            )

    def get_kv_tensors(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exposes the raw K/V cache tensors."""

        self._ensure_initialized()
        return self._k_cache, self._v_cache

    def get_stats(self) -> GPUPagedKVStats:
        """Returns allocator statistics for observability."""

        num_used = self.config.num_pages - self._free_pages.size
        return GPUPagedKVStats(
            num_total_pages=self.config.num_pages,
            num_free_pages=self._free_pages.size,
            num_used_pages=num_used,
            num_total_pages_allocated=num_used,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    def _ensure_initialized(self) -> None:
        if self._k_cache is None:
            raise RuntimeError(
                "GPUPagedKVCacheManager.initialize must be called before use"
            )

    def _get_sequence_state(self, sequence_id: int) -> _SequenceState:
        state = self._sequences.get(sequence_id)
        if state is None:
            raise KeyError(f"Sequence {sequence_id} not registered on GPU")
        return state

    def _validate_token_inputs(
        self,
        k_tensor: torch.Tensor,
        v_tensor: Optional[torch.Tensor],
    ) -> None:
        if k_tensor.dim() != 4:
            raise ValueError(
                "k_tensor must have shape [batch, seq, heads, dim]"
            )
        batch_size, _, num_k_heads, k_head_dim = k_tensor.shape
        if (
            num_k_heads != self.config.num_k_heads
            or k_head_dim != self.config.k_head_dim
        ):
            raise ValueError("k_tensor head shape mismatch with configuration")
        if k_tensor.device != self.device:
            raise ValueError(f"k_tensor must be on device {self.device}")
        if v_tensor is None:
            if self.config.has_v_cache:
                logging.debug(
                    "V cache enabled but v_tensor missing; skipping V updates"
                )
            return
        if not self.config.has_v_cache:
            raise ValueError("V tensor provided but V cache disabled")
        if v_tensor.dim() != 4:
            raise ValueError(
                "v_tensor must have shape [batch, seq, heads, dim]"
            )
        if v_tensor.shape[:2] != k_tensor.shape[:2]:
            raise ValueError(
                "k_tensor and v_tensor must share batch/seq dimensions"
            )
        _, _, num_v_heads, v_head_dim = v_tensor.shape
        if (
            num_v_heads != self.config.num_v_heads
            or v_head_dim != self.config.v_head_dim
        ):
            raise ValueError("v_tensor head shape mismatch with configuration")
        if v_tensor.device != self.device:
            raise ValueError(f"v_tensor must be on device {self.device}")

    def _prepare_sequence_lengths(
        self,
        *,
        op_name: str,
        batch_size: int,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(sequence_lengths, torch.Tensor):
            raise TypeError(
                f"{op_name}: sequence_lengths must be a torch.Tensor on device {self.device}"
            )
        if sequence_lengths.dim() != 1:
            raise ValueError(
                f"{op_name}: sequence_lengths must be 1-D, got {sequence_lengths.shape}"
            )
        if sequence_lengths.numel() != batch_size:
            raise ValueError(
                f"{op_name}: sequence_lengths must have {batch_size} elements, "
                f"got {sequence_lengths.numel()}"
            )
        if sequence_lengths.device != self.device:
            raise ValueError(
                f"{op_name}: sequence_lengths must be on device {self.device}"
            )
        if sequence_lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"{op_name}: sequence_lengths must be int32/int64, got {sequence_lengths.dtype}"
            )
        return sequence_lengths.contiguous()

    def _resolve_token_location(
        self,
        state: _SequenceState,
        sequence_id: int,
        token_index: int,
        context: str,
    ) -> Tuple[int, int]:
        if state.pages.numel() == 0:
            raise RuntimeError(
                f"{context}: sequence {sequence_id} has no allocated GPU pages"
            )
        page_size = self.config.page_size_tokens
        page_slot = token_index // page_size
        if page_slot >= state.pages.numel():
            raise RuntimeError(
                f"{context}: sequence {sequence_id} token {token_index} exceeds allocated pages {state.pages.numel()}"
            )
        offset = token_index % page_size
        gpu_page = int(state.pages[page_slot].item())
        return gpu_page, offset

    def _build_page_table(
        self,
        sequence_ids: Sequence[int],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Build a page table on GPU.

        Returns:
            page_table: [batch, max_pages_per_sequence] int32 on self.device,
            padded with -1 where a sequence has fewer pages.
        """
        self._ensure_initialized()
        self._geometry.ensure_layer_bounds(layer_idx, "_build_page_table")

        seq_states = [
            self._get_sequence_state(seq_id) for seq_id in sequence_ids
        ]

        page_tensors = [
            state.pages.to(self.device, dtype=torch.int32)
            for state in seq_states
        ]

        page_table = pad_sequence(
            page_tensors,
            batch_first=True,  # -> [batch, max_len]
            padding_value=-1,
        )

        return page_table

    def _refresh_page_pointer_tables(self) -> None:
        self._k_page_ptr_table = self._build_page_pointer_table(
            self._k_cache, self._layout.k_page_bytes
        )
        if self.config.has_v_cache:
            self._v_page_ptr_table = self._build_page_pointer_table(
                self._v_cache, self._layout.v_page_bytes
            )
        else:
            self._v_page_ptr_table = None

    def _build_page_pointer_table(
        self, cache_tensor: torch.Tensor, page_bytes: int
    ) -> torch.Tensor:
        """Builds a page pointer table for ``cache_tensor``.

        The result is a CPU tensor of shape [num_layers, num_pages] containing
        the device pointers (as int64) to each [layer, page, ...] slice.
        """

        if cache_tensor is None:
            raise ValueError("cache_tensor must not be None")

        if cache_tensor.dim() < 2:
            raise ValueError(
                f"cache_tensor must have at least 2 dims, got {cache_tensor.dim()}"
            )
        num_layers, num_pages = cache_tensor.shape[:2]
        base_ptr = cache_tensor.data_ptr()
        elem_size = cache_tensor.element_size()

        # Use strides (in elements) to compute per-(layer,page) offsets.
        stride_layer, stride_page = cache_tensor.stride()[:2]
        dev = cache_tensor.device

        layer_idx = torch.arange(
            num_layers, dtype=torch.int64, device=dev
        ).view(-1, 1)
        page_idx = torch.arange(num_pages, dtype=torch.int64, device=dev).view(
            1, -1
        )

        offset_elems = layer_idx * stride_layer + page_idx * stride_page
        offset_bytes = offset_elems * elem_size

        base_ptr_tensor = torch.tensor(base_ptr, dtype=torch.int64, device=dev)

        pointer_table = (base_ptr_tensor + offset_bytes).to(
            device="cpu", dtype=torch.int64
        )

        return pointer_table

    def export_layer_page_pointer_table(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Materializes layer-major device pointer tensors using current slots.

        The returned tensors are CPU-resident matrices shaped ``[num_layers,
        total_pages]`` where ``total_pages`` is the sum of pages allocated for
        the sequences tracked by ``_gpu_page_table_manager`` in slot order. The
        optional second tensor is present only when the V cache is enabled.
        """

        self._ensure_initialized()
        k_table = self._get_page_ptr_table(is_value=False)
        k_ptrs = k_table

        v_ptrs: Optional[torch.Tensor] = None
        if self.config.has_v_cache:
            v_table = self._get_page_ptr_table(is_value=True)
            v_ptrs = v_table

        return k_ptrs, v_ptrs

    def _get_page_ptr_table(self, *, is_value: bool) -> torch.Tensor:
        """Returns the cached pointer table for the requested cache kind."""

        table = self._v_page_ptr_table if is_value else self._k_page_ptr_table
        if table is None:
            cache_name = "V" if is_value else "K"
            raise RuntimeError(
                f"{cache_name} page pointer table is unavailable; ensure initialize() was called"
            )
        return table
