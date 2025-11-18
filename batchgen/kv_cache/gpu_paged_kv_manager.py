from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


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
    alignment_bytes: int = 64

    @classmethod
    def from_engine(cls, engine_config) -> "GPUPagedKVConfig":  # noqa: D401
        cfg = engine_config.gpu_kv_config
        num_layers = _require_positive(
            getattr(cfg, "num_layers", None), "num_layers"
        )
        num_pages = _require_positive(
            getattr(cfg, "num_gpu_pages", None), "num_gpu_pages"
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
        kv_dtype = getattr(cfg, "kv_dtype_torch", None)
        if kv_dtype is None:
            raise ValueError("gpu_kv_config.kv_dtype_torch must be defined")
        num_v_heads = getattr(cfg, "num_v_heads", None)
        v_head_dim = getattr(cfg, "v_head_dim", None)
        if num_v_heads is not None and num_v_heads > 0:
            v_head_dim = _require_positive(v_head_dim, "v_head_dim")
        else:
            num_v_heads = 0
            v_head_dim = 0
        alignment = getattr(cfg, "alignment_bytes", 64) or 64
        return cls(
            num_layers=num_layers,
            num_pages=num_pages,
            page_size_tokens=page_size,
            num_k_heads=num_k_heads,
            k_head_dim=k_head_dim,
            num_v_heads=num_v_heads,
            v_head_dim=v_head_dim,
            kv_dtype=kv_dtype,
            alignment_bytes=alignment,
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
    length_tokens: int = 0

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

    def required_pages(self, num_tokens: int) -> int:
        self.ensure_positive_tokens(num_tokens, "required_pages")
        return _ceil_div(num_tokens, self._config.page_size_tokens)


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


class GPUKVCacheManager:
    """Per-GPU paged KV cache manager inspired by the host implementation."""

    def __init__(self, engine_config):
        self._engine_config = engine_config
        self.config = GPUPagedKVConfig.from_engine(engine_config)
        self._geometry = GPUPagedKVGeometry(self.config)
        self._layout = GPUPagedKVLayout(self.config)
        self.device = torch.device(engine_config.basic_config.gpu_device_id)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._core_engine = None

        self._k_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None
        self._free_pages = _TensorStack(self.config.num_pages)
        self._sequences: Dict[int, _SequenceState] = {}

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def initialize(self, core_engine) -> None:
        """Instantiates GPU tensors and prepares allocator state."""

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
        self._logger.info("Initialized K cache %s", tuple(self._k_cache.shape))

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
            self._logger.info(
                "Initialized V cache %s", tuple(self._v_cache.shape)
            )
        else:
            self._v_cache = None
            self._logger.info("Initialized in K-only mode (no V cache)")

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
        self._logger.debug(
            "Allocated %d pages for seq %s -> %s",
            missing,
            sequence_id,
            new_pages.tolist(),
        )
        return new_pages.tolist()

    def allocate_pages_for_sequences(self, sequence_ids: Sequence[int], num_tokens: int) -> Dict[int, List[int]]:
        """Allocates enough pages to hold ``num_tokens`` for each sequence in ``sequence_ids``."""
        self._ensure_initialized()
        required_pages = self._geometry.required_pages(num_tokens)
        allocations: Dict[int, List[int]] = {}

        for sequence_id in sequence_ids:
            state = self._sequences.get(sequence_id)
            current_pages = state.pages.numel() if state else 0
            missing = max(0, required_pages - current_pages)
            if missing == 0:
                allocations[sequence_id] = []
                continue
            if missing > self._free_pages.size:
                raise RuntimeError(
                    f"Insufficient free pages: need {missing} for seq {sequence_id}, have {self._free_pages.size}"
                )
            new_pages = self._free_pages.pop(missing)
            if state is None:
                state = _SequenceState(pages=new_pages)
                self._sequences[sequence_id] = state
            else:
                state.append_pages(new_pages)
            allocations[sequence_id] = new_pages.tolist()
            self._logger.debug(
                "Allocated %d pages for seq %s -> %s",
                missing,
                sequence_id,
                new_pages.tolist(),
            )

        return allocations

    def free_sequence(self, sequence_id: int) -> None:
        """Returns all pages associated with ``sequence_id`` to the free list."""

        state = self._sequences.pop(sequence_id, None)
        if state is None:
            raise RuntimeError(f"Sequence {sequence_id} not registered on GPU")
        self._free_pages.push(state.pages)
        self._logger.debug(
            "Freed %d pages for seq %s", state.pages.numel(), sequence_id
        )

    def load_offloaded_context(
        self, sequence_id: int, context_length: int
    ) -> None:
        """Loads CPU-resident KV context for ``sequence_id`` onto the GPU."""

        self._ensure_initialized()
        state = self._get_sequence_state(sequence_id)
        required_pages = self._geometry.required_pages(context_length)
        if state.pages.numel() < required_pages:
            raise ValueError(
                f"Sequence {sequence_id} has {state.pages.numel()} pages, requires {required_pages}"
            )
        if self._core_engine is None:
            raise RuntimeError("Core engine must be set before loading context")

        for layer_idx in range(self.config.num_layers):
            gpu_k_ptrs, gpu_v_ptrs = self.get_context_kv_page_ptrs(
                sequence_id, layer_idx, context_length
            )
            cpu_k_ptrs, cpu_v_ptrs = self._core_engine.get_context_kv_page_ptrs(
                sequence_id, layer_idx, context_length
            )
            self._core_engine.blocking_h2d_kv_page_copy(
                cpu_k_ptrs, gpu_k_ptrs, self._layout.k_page_bytes
            )
            if gpu_v_ptrs is not None and cpu_v_ptrs is not None:
                self._core_engine.blocking_h2d_kv_page_copy(
                    cpu_v_ptrs, gpu_v_ptrs, self._layout.v_page_bytes
                )

        state.length_tokens = context_length

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
        layer_idx: int,
    ) -> None:
        """Writes new KV tokens into the cache for ``layer_idx``."""

        self._ensure_initialized()
        self._geometry.ensure_layer_bounds(layer_idx, "update_new_token")
        self._validate_token_inputs(k_tensor, v_tensor, sequence_ids)
        batch_size, seq_len, _, _ = k_tensor.shape

        for batch_idx in range(batch_size):
            seq_id = sequence_ids[batch_idx]
            state = self._get_sequence_state(seq_id)
            required_pages = self._geometry.required_pages(
                state.length_tokens + seq_len
            )
            if required_pages > state.pages.numel():
                raise RuntimeError(
                    f"Sequence {seq_id} requires {required_pages} pages but only has {state.pages.numel()}"
                )

            tokens_written = 0
            while tokens_written < seq_len:
                absolute = state.length_tokens + tokens_written
                page_idx = absolute // self.config.page_size_tokens
                offset = absolute % self.config.page_size_tokens
                tokens_this_page = min(
                    self.config.page_size_tokens - offset,
                    seq_len - tokens_written,
                )

                gpu_page = int(state.pages[page_idx].item())
                k_source = k_tensor[
                    batch_idx,
                    tokens_written : tokens_written + tokens_this_page,
                ]
                self._k_cache[
                    layer_idx, gpu_page, offset : offset + tokens_this_page
                ].copy_(k_source)

                if v_tensor is not None and self._v_cache is not None:
                    v_source = v_tensor[
                        batch_idx,
                        tokens_written : tokens_written + tokens_this_page,
                    ]
                    self._v_cache[
                        layer_idx, gpu_page, offset : offset + tokens_this_page
                    ].copy_(v_source)

                tokens_written += tokens_this_page

            state.length_tokens += seq_len
            self._logger.debug(
                "Layer %d sequence %s extended to %d tokens",
                layer_idx,
                seq_id,
                state.length_tokens,
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
            page_table = self._build_page_table(sequence_ids)
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
    def _ensure_initialized(self) -> None:
        if self._k_cache is None:
            raise RuntimeError(
                "GPUKVCacheManager.init must be called before use"
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
                self._logger.debug(
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

    def _build_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        """Placeholder for caller-provided page table implementation."""

        raise NotImplementedError(
            "Page table construction is intentionally left blank. "
            "Provide your own implementation via subclassing or composition."
        )
