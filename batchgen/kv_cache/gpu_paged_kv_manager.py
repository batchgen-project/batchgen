from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from batchgen.config.config import EngineConfig
from batchgen.kv_cache.gpu_kv_kernels import run_paged_kv_token_update


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
class LayerPagePointerBatch:
    layer_ids: torch.Tensor
    page_indices: torch.Tensor
    k_ptrs: torch.Tensor
    v_ptrs: Optional[torch.Tensor] = None


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
        self._layer_index_template = torch.arange(
            self.config.num_layers, dtype=torch.int32
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

    def free_sequence(self, sequence_id: int) -> None:
        """Returns all pages associated with ``sequence_id`` to the free list."""

        state = self._sequences.pop(sequence_id, None)
        if state is None:
            raise RuntimeError(f"Sequence {sequence_id} not registered on GPU")
        self._free_pages.push(state.pages)

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
        sequence_ids: Sequence[int],
        sequence_lengths: Sequence[int],
        layer_idx: int,
    ) -> None:
        """Writes single-position KV tokens into the cache for ``layer_idx``."""

        op_name = "update_new_token"
        self._ensure_initialized()
        self._geometry.ensure_layer_bounds(layer_idx, op_name)
        self._validate_token_inputs(k_tensor, v_tensor, sequence_ids)

        batch_size, seq_len, _, _ = k_tensor.shape
        if seq_len != 1:
            raise ValueError(
                f"{op_name}: k_tensor must have sequence dimension 1, got {seq_len}"
            )

        normalized_lengths = sequence_lengths

        page_indices: List[int] = []
        token_offsets: List[int] = []

        for batch_idx in range(batch_size):
            seq_id = sequence_ids[batch_idx]
            token_index = normalized_lengths[batch_idx]
            state = self._get_sequence_state(seq_id)
            gpu_page, offset = self._resolve_token_location(
                state, seq_id, token_index, op_name
            )
            page_indices.append(gpu_page)
            token_offsets.append(offset)

        k_tokens = k_tensor.view(batch_size, -1).contiguous()

        page_indices_t = torch.tensor(
            page_indices, dtype=torch.int32, device=self.device
        )
        token_offsets_t = torch.tensor(
            token_offsets, dtype=torch.int32, device=self.device
        )

        if v_tensor is not None and self._v_cache is not None:
            v_tokens = v_tensor.view(batch_size, -1).contiguous()
        else:
            v_tokens = None

        run_paged_kv_token_update(
            k_cache=self._k_cache,
            k_tokens=k_tokens,
            page_indices=page_indices_t,
            token_offsets=token_offsets_t,
            layer_idx=layer_idx,
            v_cache=self._v_cache,
            v_tokens=v_tokens,
        )

    def get_layer_kv_with_page_table(
        self, layer_idx: int, sequence_ids: Sequence[int]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns layer tensors alongside a placeholder for the page table."""

        self._ensure_initialized()
        self._geometry.ensure_layer_bounds(
            layer_idx, "get_layer_kv_with_page_table"
        )
        page_table: Optional[torch.Tensor]
        try:
            page_table = self._build_page_table(sequence_ids, layer_idx)
        except NotImplementedError:
            page_table = None
        return (
            self._k_cache[layer_idx],
            None if self._v_cache is None else self._v_cache[layer_idx],
            page_table,
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
        sequence_ids: Sequence[int],
    ) -> None:
        if k_tensor.dim() != 4:
            raise ValueError(
                "k_tensor must have shape [batch, seq, heads, dim]"
            )
        batch_size, _, num_k_heads, k_head_dim = k_tensor.shape
        if len(sequence_ids) != batch_size:
            raise ValueError("sequence_ids length must equal batch size")
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

    def build_layer_page_pointer_batch(
        self, page_indices: Sequence[int] | torch.Tensor
    ) -> LayerPagePointerBatch:
        """Returns layer-major tensors of K/V page pointers for ``page_indices``.

        The returned tensors live on CPU and are laid out as ``[num_layers *
        len(page_indices)]`` for direct consumption by CUDA kernels without
        Python-side loops.
        """

        self._ensure_initialized()
        page_tensor = torch.as_tensor(
            page_indices, dtype=torch.int32, device="cpu"
        ).reshape(-1)
        if page_tensor.numel() == 0:
            empty_i32 = torch.empty(0, dtype=torch.int32)
            empty_i64 = torch.empty(0, dtype=torch.int64)
            return LayerPagePointerBatch(empty_i32, empty_i32, empty_i64, None)

        invalid_mask = (page_tensor < 0) | (
            page_tensor >= self.config.num_pages
        )
        if torch.any(invalid_mask):
            raise ValueError(
                "page_indices contain values outside [0, num_pages)"
            )

        index_tensor = page_tensor.to(torch.long)
        layer_ids = self._layer_index_template.repeat_interleave(
            page_tensor.numel()
        )
        repeated_pages = page_tensor.repeat(self.config.num_layers)

        k_ptrs = self._get_page_ptr_table(is_value=False).index_select(
            1, index_tensor
        )
        k_ptrs = k_ptrs.reshape(-1).contiguous()

        v_ptrs: Optional[torch.Tensor] = None
        if self.config.has_v_cache:
            v_table = self._get_page_ptr_table(is_value=True)
            v_ptrs = (
                v_table.index_select(1, index_tensor).reshape(-1).contiguous()
            )

        return LayerPagePointerBatch(layer_ids, repeated_pages, k_ptrs, v_ptrs)

    def _get_page_ptr_table(self, *, is_value: bool) -> torch.Tensor:
        """Returns the cached pointer table for the requested cache kind."""

        table = self._v_page_ptr_table if is_value else self._k_page_ptr_table
        if table is None:
            cache_name = "V" if is_value else "K"
            raise RuntimeError(
                f"{cache_name} page pointer table is unavailable; ensure initialize() was called"
            )
        return table
