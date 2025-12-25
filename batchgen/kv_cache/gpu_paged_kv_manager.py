from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from batchgen.config.config import (
	DevicePagedKVConfig,
	EngineConfig,
	ModelConfig,
)
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


def _positive_or_none(value: Optional[int]) -> Optional[int]:
	if value is None:
		return None
	if value > 0:
		return int(value)
	return None


def _resolve_positive(
	primary: Optional[int], fallback: Optional[int], field_name: str
) -> int:
	for candidate in (primary, fallback):
		if candidate is not None and candidate > 0:
			return int(candidate)
	raise ValueError(f"{field_name} must be > 0")


def _normalize_device(
	device_like: Union[torch.device, str, int],
) -> torch.device:
	if isinstance(device_like, torch.device):
		return device_like
	if isinstance(device_like, str):
		return torch.device(device_like)
	if isinstance(device_like, int):
		return torch.device(f"cuda:{device_like}")
	raise ValueError(f"Unsupported device specification: {device_like}")


class _DtypeResolver:
	_DTYPE_MAP: Dict[str, torch.dtype] = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
		"float8_e4m3fn": torch.float8_e4m3fn,
		"float8_e5m2": torch.float8_e5m2,
	}

	@classmethod
	def from_string(cls, dtype_str: Optional[str]) -> torch.dtype:
		if dtype_str is None:
			return torch.bfloat16
		normalized = dtype_str.lower().replace("torch.", "")
		if normalized not in cls._DTYPE_MAP:
			raise ValueError(f"Unsupported kv dtype '{dtype_str}'")
		return cls._DTYPE_MAP[normalized]


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
	def from_device_config(
		cls,
		*,
		device_config: DevicePagedKVConfig,
		model_config: ModelConfig,
		kv_dtype: Optional[torch.dtype] = None,
	) -> "GPUPagedKVConfig":
		num_layers = _resolve_positive(
			_positive_or_none(device_config.num_layers),
			_positive_or_none(model_config.num_hidden_layers),
			"Device_Paged_KV_Config.num_layers",
		)
		pages_per_layer = _require_positive(
			device_config.num_pages_per_layer,
			"Device_Paged_KV_Config.num_pages_per_layer",
		)
		page_size = _require_positive(
			device_config.page_size, "Device_Paged_KV_Config.page_size"
		)
		model_heads = _positive_or_none(
			model_config.num_key_value_heads
		) or _positive_or_none(model_config.num_attention_heads)
		num_k_heads = _resolve_positive(
			_positive_or_none(device_config.num_k_heads),
			model_heads,
			"Device_Paged_KV_Config.num_k_heads",
		)
		num_kv_dim = _resolve_positive(
			_positive_or_none(device_config.k_head_dim),
			_positive_or_none(model_config.head_dim),
			"Device_Paged_KV_Config.k_head_dim",
		)
		raw_v_heads = device_config.num_v_heads
		if raw_v_heads is None or raw_v_heads < 0:
			raise ValueError("Device_Paged_KV_Config.num_v_heads must be >= 0")
		if raw_v_heads == 0:
			resolved_v_heads = 0
			resolved_v_dim = 0
		else:
			resolved_v_heads = _resolve_positive(
				_positive_or_none(raw_v_heads),
				num_k_heads,
				"Device_Paged_KV_Config.num_v_heads",
			)
			resolved_v_dim = _resolve_positive(
				_positive_or_none(device_config.v_head_dim),
				num_kv_dim,
				"Device_Paged_KV_Config.v_head_dim",
			)

		dtype = kv_dtype
		if dtype is None:
			dtype = _DtypeResolver.from_string(device_config.kv_dtype)

		return cls(
			num_layers=num_layers,
			num_pages=pages_per_layer,
			page_size_tokens=page_size,
			num_k_heads=num_k_heads,
			k_head_dim=num_kv_dim,
			num_v_heads=resolved_v_heads,
			v_head_dim=resolved_v_dim,
			kv_dtype=dtype,
		)

	@classmethod
	def from_engine(
		cls, *, engine_config: EngineConfig, model_config: ModelConfig
	) -> "GPUPagedKVConfig":
		basic = engine_config.Basic_Config
		dtype = getattr(basic, "kv_dtype_torch", None)
		if dtype is None:
			dtype = _DtypeResolver.from_string(getattr(basic, "kv_dtype", None))
		return cls.from_device_config(
			device_config=engine_config.Device_Paged_KV_Config,
			model_config=model_config,
			kv_dtype=dtype,
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
		# Flattened list of active page indices ordered by slot_to_seq_id.
		self._active_page_indices_cpu: Optional[torch.Tensor] = None

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
		# new_max = max(self.max_pages_per_sequence, desired_pages)
		new_max = desired_pages

		num_slots = len(wanted_order)
		flat_pages_cpu = self._build_flat_page_index_tensor(
			wanted_order, sequences
		)

		# Fast-path: if an existing GPU table already matches the exact
		# shape we need and the order is identical, return a view into it.
		old_shape = self.gpu_table.shape if self.gpu_table is not None else None
		old_slot_count = len(self.slot_to_seq_id) if self.slot_to_seq_id else 0
		
		shape_match = self.gpu_table is not None and self.gpu_table.shape[0] == num_slots
		cols_sufficient = self.gpu_table is not None and self.gpu_table.shape[1] >= new_max
		order_match = self.slot_to_seq_id == wanted_order
		
		reuse_existing = (
			self.gpu_table is not None
			and shape_match
			and cols_sufficient
			and order_match
		)
		
		# DEBUG: Log detailed rebuild info
		logging.info(
			f"GPUPageTableManager.rebuild: "
			f"num_slots={num_slots}, new_max={new_max}, "
			f"old_shape={old_shape}, old_slot_count={old_slot_count}, "
			f"shape_match={shape_match}, cols_sufficient={cols_sufficient}, order_match={order_match}, "
			f"reuse_existing={reuse_existing}"
		)

		table = (
			self.gpu_table
			if reuse_existing
			else torch.full(
				(num_slots, new_max), -1, dtype=torch.int32, device=self.device
			)
		)

		# Replace mappings with exactly the requested order. This keeps
		# the slot mapping deterministic and equal in length to num_slots.
		self.seq_id_to_slot = {
			seq_id: idx for idx, seq_id in enumerate(wanted_order)
		}
		self.slot_to_seq_id = list(wanted_order)
		self.max_pages_per_sequence = int(new_max)

		fill_region = table[:num_slots, :new_max]

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
			fill_region[slot, :count] = pages[:count]

		if not reuse_existing:
			self.gpu_table = table
		self._slot_index_tensor = self._build_slot_index_tensor(num_slots)
		self._slot_to_seq_id_tensor = self._build_slot_to_seq_id_tensor(
			self.slot_to_seq_id
		)
		self._active_page_indices_cpu = flat_pages_cpu
		
		# DEBUG: Log final state after rebuild
		final_shape = self.gpu_table.shape if self.gpu_table is not None else None
		return_shape = table[:num_slots, :].shape
		logging.info(
			f"GPUPageTableManager.rebuild DONE: "
			f"self.gpu_table.shape={final_shape}, return_shape={return_shape}, "
			f"slot_to_seq_id_len={len(self.slot_to_seq_id)}"
		)
		
		# If table has more columns than needed, return a view with
		# the requested number of rows and the existing columns.
		return table[:num_slots, :]

	def get_slot_index_tensor(self) -> torch.Tensor:
		"""Returns a cached 1-D tensor mapping logical batch order to slots."""
		if self._slot_index_tensor is None:
			raise RuntimeError(
				"GPU page table slot indices unavailable; call rebuild() first"
			)
		tensor = self._slot_index_tensor
		return tensor

	def get_active_page_indices(self) -> torch.Tensor:
		if self._active_page_indices_cpu is None:
			raise RuntimeError(
				"Active page indices unavailable; call rebuild() first"
			)
		return self._active_page_indices_cpu

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

	def _build_flat_page_index_tensor(
		self, sequence_ids: Sequence[int], sequences: Dict[int, _SequenceState]
	) -> torch.Tensor:
		buffers: List[torch.Tensor] = []
		for seq_id in sequence_ids:
			state = sequences.get(seq_id)
			if state is None or state.pages.numel() == 0:
				continue  
			buffers.append(state.pages.to("cpu", dtype=torch.int64))
		if not buffers:
			return torch.empty(0, dtype=torch.int64)
		return torch.cat(buffers, dim=0)


class GPUPagedKVCacheManager:
	"""Per-GPU paged KV cache manager inspired by the host implementation."""

	def __init__(
		self,
		engine_config: Optional[EngineConfig] = None,
		model_config: Optional[ModelConfig] = None,
		*,
		config: Optional[GPUPagedKVConfig] = None,
		device: Optional[Union[str, int, torch.device]] = None,
	):
		if config is not None and (
			engine_config is not None or model_config is not None
		):
			raise ValueError(
				"Pass either `config` or (`engine_config`, `model_config`), not both"
			)

		if config is None:
			if engine_config is None or model_config is None:
				raise ValueError(
					"engine_config and model_config must be provided when config is absent"
				)
			config = GPUPagedKVConfig.from_engine(
				engine_config=engine_config, model_config=model_config
			)
			device_handle = engine_config.Basic_Config.device
			if device_handle is None:
				raise ValueError(
					"engine_config.Basic_Config.device must be specified for GPU KV manager"
				)
			resolved_device = _normalize_device(device_handle)
		else:
			if device is None:
				raise ValueError(
					"device must be provided when initializing with GPUPagedKVConfig"
				)
			resolved_device = _normalize_device(device)

		self._engine_config = engine_config
		self.config = config
		self._geometry = GPUPagedKVGeometry(self.config)
		self._layout = GPUPagedKVLayout(self.config)
		self.device = resolved_device
		self._reset_runtime_state()

	# ------------------------------------------------------------------
	# Public APIs
	# ------------------------------------------------------------------
	def initialize(self) -> None:
		"""Instantiates GPU tensors and prepares allocator state."""

		if self._is_initialized:
			logging.warning(
				"GPUPagedKVCacheManager.initialize called while already initialized; skipping"
			)
			return

		self._set_device()
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
		self._is_initialized = True

	def destroy(self, *, empty_cuda_cache: bool = False) -> None:
		"""
		Releases GPU buffers and resets the allocator state so it can be reused.

		Args:
			empty_cuda_cache: If ``True``, clears PyTorch’s CUDA caching allocator
				after all tensor references have been dropped, allowing reserved
				memory to be returned to the driver. The default is ``False``.

				This is typically used only when transitioning from the decode
				stage to the prefill stage. During the prefill stage, the GPU will
				continue to rely on the caching allocator’s memory, so clearing the
				cache is usually unnecessary.
		"""

		if not self._is_initialized:
			logging.warning(
				"GPUPagedKVCacheManager.destroy called while uninitialized; skipping"
			)
			return

		self._reset_runtime_state()
		if empty_cuda_cache:
			self._release_cached_cuda_memory()

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
		self._clear_active_page_pointer_tables()
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
		any_changes = False

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
			any_changes = True
		if any_changes:
			self._clear_active_page_pointer_tables()
		return allocations

	def grow_sequence_pages(
		self, sequence_id: int, num_pages: int
	) -> List[int]:
		"""Appends ``num_pages`` new pages to an already allocated sequence."""

		allocations = self.grow_pages_for_sequences([sequence_id], [num_pages])
		return allocations.get(sequence_id, [])

	def grow_pages_for_sequences(
		self, sequence_ids: Sequence[int], num_pages: Sequence[int]
	) -> Dict[int, List[int]]:
		"""Appends explicit page counts for the provided ``sequence_ids``."""

		self._ensure_initialized()
		if len(sequence_ids) != len(num_pages):
			raise ValueError(
				"grow_pages_for_sequences: sequence_ids and num_pages must be the same length"
			)
		if not sequence_ids:
			return {}

		missing = [
			seq_id for seq_id in sequence_ids if seq_id not in self._sequences
		]
		if missing:
			raise KeyError(
				"grow_pages_for_sequences: sequences not allocated: "
				+ ", ".join(str(seq_id) for seq_id in missing)
			)

		available = self._free_pages.size
		normalized_counts: List[int] = []
		for seq_id, count in zip(sequence_ids, num_pages):
			if count <= 0:
				raise ValueError(
					f"grow_pages_for_sequences: num_pages must be positive for seq {seq_id}, got {count}"
				)
			if count > available:
				raise RuntimeError(
					f"grow_pages_for_sequences: insufficient free pages for seq {seq_id}: need {count}, free {available}"
				)
			available -= count
			normalized_counts.append(int(count))

		allocations: Dict[int, List[int]] = {}
		for seq_id, count in zip(sequence_ids, normalized_counts):
			new_pages = self._free_pages.pop(count)
			state = self._sequences[seq_id]
			state.append_pages(new_pages)
			allocations[seq_id] = new_pages.tolist()

		if allocations:
			self._clear_active_page_pointer_tables()
		return allocations

	def clear_page_table(self) -> None:
		"""Clear the GPU page table to empty state (0 sequences).
		
		This should be called when all sequences have been freed and the batch is empty.
		"""
		self._ensure_initialized()
		mgr = self._gpu_page_table_manager
		
		# Clear the GPU table to empty shape [0, max_pages]
		max_pages = mgr.max_pages_per_sequence if mgr.max_pages_per_sequence > 0 else 1
		mgr.gpu_table = torch.full(
			(0, max_pages), -1, dtype=torch.int32, device=mgr.device
		)
		
		# Clear the slot mappings
		mgr.seq_id_to_slot = {}
		mgr.slot_to_seq_id = []
		mgr._slot_index_tensor = None
		mgr._slot_to_seq_id_tensor = None
		mgr._active_page_indices_cpu = None
		
		# Clear active page pointer tables
		self._clear_active_page_pointer_tables()

	def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
		"""Rebuilds the GPU page table following ``sequence_ids`` order.

		Args:
			sequence_ids: Logical sequence ordering to materialize on device.

		Returns:
			torch.Tensor: GPU-resident page table view aligned with ``sequence_ids``.
		"""
		import logging

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

		# DEBUG: Log before rebuild
		old_shape = self._gpu_page_table_manager.gpu_table.shape if self._gpu_page_table_manager.gpu_table is not None else None
		
		table = self._gpu_page_table_manager.rebuild(
			ordered_ids, self._sequences
		)
		
		# DEBUG: Log after rebuild
		new_shape = self._gpu_page_table_manager.gpu_table.shape if self._gpu_page_table_manager.gpu_table is not None else None
		logging.debug(
			f"rebuild_page_table: requested {len(ordered_ids)} seqs, "
			f"old_shape={old_shape}, new_shape={new_shape}, "
			f"returned_shape={table.shape}"
		)
		
		self._update_active_page_pointer_tables()
		return table

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
		if reclaimed:
			self._clear_active_page_pointer_tables()

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

	def update_layer_decode_new_token(
		self,
		k_tensor: torch.Tensor,
		v_tensor: Optional[torch.Tensor],
		sequence_lengths: torch.Tensor,
		layer_idx: int,
		batch_slice: Optional[tuple] = None,  # (start_idx, end_idx) for micro-batching
	) -> None:
		"""Writes single-position KV tokens for ``layer_idx`` using the cached
		GPU page table order.

		The provided ``k_tensor`` (and optional ``v_tensor``) must align with
		the sequence ordering used when ``allocate_pages_for_sequences`` last
		triggered a rebuild of the GPU page table.
		
		Args:
			batch_slice: Optional tuple (start_idx, end_idx) indicating which slice 
				of the full batch this call represents. When provided, the page table 
				and slot_indices will be sliced accordingly.
		"""
		# op_name = "update_layer_decode_new_token"
		# self._ensure_initialized()
		# self._geometry.ensure_layer_bounds(layer_idx, op_name)
		# self._validate_token_inputs(k_tensor, v_tensor)
		batch_size, seq_len, _, _ = k_tensor.shape
		if seq_len != 1:
			raise ValueError(
				"update_layer_decode_new_token: k_tensor must have sequence dimension 1, "
				f"got {seq_len}"
			)

		page_table = self._gpu_page_table_manager.gpu_table
		if page_table is None:
			raise RuntimeError(
				f"update_layer_decode_new_token: GPU page table is not initialized; "
				"call allocate_pages_for_sequences before updating tokens"
			)

		# all the tensors are continuous
		# slot_indices = self._gpu_page_table_manager.get_slot_index_tensor()
		slot_indices = self._gpu_page_table_manager._slot_index_tensor
		token_indices = sequence_lengths

		# Apply batch slice if provided (for micro-batching)
		# NOTE: We keep the FULL page_table but slice slot_indices.
		# slot_indices[i] tells the kernel which row of page_table to use for token i.
		# For micro-batch [start_idx:end_idx], token i corresponds to global slot (start_idx + i),
		# so slot_indices should be [start_idx, start_idx+1, ..., end_idx-1].
		if batch_slice is not None:
			start_idx, end_idx = batch_slice
			# slot_indices[start_idx:end_idx] gives us [start_idx, start_idx+1, ..., end_idx-1]
			# which correctly maps micro-batch tokens to full page table rows
			slot_indices = slot_indices[start_idx:end_idx]
		page_table_view = page_table  # Always use full page table
		k_tokens = k_tensor.view(batch_size, -1)

		if v_tensor is not None and self._v_cache is not None:
			v_tokens: Optional[torch.Tensor] = v_tensor.view(batch_size, -1)
		else:
			v_tokens = None

		k_cache_layer = self._k_cache[layer_idx]
		v_cache_layer = None
		if self._v_cache is not None:
			v_cache_layer = self._v_cache[layer_idx]

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
		if not self._is_initialized:
			raise RuntimeError(
				"GPUPagedKVCacheManager.initialize must be called before use"
			)

	@property
	def is_initialized(self) -> bool:
		return self._is_initialized

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
		self._clear_active_page_pointer_tables()

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

	def _reset_runtime_state(self) -> None:
		self._k_cache = None
		self._v_cache = None
		self._k_page_ptr_table = None
		self._v_page_ptr_table = None
		self._active_page_indices = None
		self._k_active_page_ptr_table = None
		self._v_active_page_ptr_table = None
		self._free_pages = _TensorStack(self.config.num_pages)
		self._sequences: Dict[int, _SequenceState] = {}
		max_pages_per_seq = _ceil_div(
			DEFAULT_INITIAL_TOKEN_CAPACITY, self.config.page_size_tokens
		)
		self._gpu_page_table_manager = _GPUPageTableManager(
			device=self.device, max_pages_per_sequence=max_pages_per_seq
		)
		self._is_initialized = False

	def _release_cached_cuda_memory(self) -> None:
		if self.device.type != "cuda":
			return
		self._set_device()
		torch.cuda.synchronize(self.device)
		torch.cuda.empty_cache()

	def _update_active_page_pointer_tables(self) -> None:
		active_indices = self._gpu_page_table_manager.get_active_page_indices()
		self._active_page_indices = active_indices
		self._k_active_page_ptr_table = self._select_active_page_columns(
			self._k_page_ptr_table, active_indices
		)
		if self.config.has_v_cache:
			self._v_active_page_ptr_table = self._select_active_page_columns(
				self._v_page_ptr_table, active_indices
			)
		else:
			self._v_active_page_ptr_table = None

	def _clear_active_page_pointer_tables(self) -> None:
		self._active_page_indices = None
		self._k_active_page_ptr_table = None
		self._v_active_page_ptr_table = None

	def _select_active_page_columns(
		self,
		base_table: Optional[torch.Tensor],
		page_indices: torch.Tensor,
	) -> torch.Tensor:
		if base_table is None:
			raise RuntimeError(
				"Base page pointer table unavailable; ensure initialize() was called"
			)
		if page_indices.numel() == 0:
			num_layers = base_table.shape[0]
			return base_table.new_empty((num_layers, 0))
		index = page_indices
		if index.device != base_table.device or index.dtype != torch.long:
			index = index.to(device=base_table.device, dtype=torch.long)
		return torch.index_select(base_table, dim=1, index=index)

	def _get_active_page_ptr_table(self, *, is_value: bool) -> torch.Tensor:
		table = (
			self._v_active_page_ptr_table
			if is_value
			else self._k_active_page_ptr_table
		)
		if table is None:
			cache_name = "V" if is_value else "K"
			raise RuntimeError(
				f"{cache_name} page pointer table unavailable; call rebuild_page_table() first"
			)
		return table

	def _materialize_layer_pointer_tensor(
		self,
		*,
		pointer_table: torch.Tensor,
		page_counts: Sequence[int],
	) -> torch.Tensor:
		num_layers, total_pages = pointer_table.shape
		seq_count = len(page_counts)
		max_pages = max(page_counts, default=0)
		if seq_count == 0:
			raise RuntimeError(
				"_materialize_layer_pointer_tensor: no active sequences available"
			)
		result = pointer_table.new_zeros((num_layers, seq_count, max_pages))
		cursor = 0
		for seq_idx, count in enumerate(page_counts):
			if count < 0:
				raise ValueError(
					"_materialize_layer_pointer_tensor: page counts must be non-negative"
				)
			if count == 0:
				continue
			next_cursor = cursor + count
			if next_cursor > total_pages:
				raise RuntimeError(
					"_materialize_layer_pointer_tensor: page count prefix exceeds available columns"
				)
			result[:, seq_idx, :count] = pointer_table[:, cursor:next_cursor]
			cursor = next_cursor
		if cursor != total_pages:
			raise RuntimeError(
				"_materialize_layer_pointer_tensor: page counts do not sum to available columns"
			)
		return result

	def _require_active_slot_order(self, *, op_name: str) -> List[int]:
		if self._active_page_indices is None:
			raise RuntimeError(
				f"{op_name}: active page tables unavailable; call rebuild_page_table() first"
			)
		slot_order = self._gpu_page_table_manager.slot_to_seq_id
		if not slot_order:
			raise RuntimeError(
				f"{op_name}: no active sequences tracked; call rebuild_page_table() with active sequences"
			)
		return slot_order

	def _build_sequence_page_counts(
		self, slot_order: Sequence[int]
	) -> List[int]:
		return [
			int(self._get_sequence_state(seq_id).pages.numel())
			for seq_id in slot_order
		]

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
		k_ptrs = self._get_active_page_ptr_table(is_value=False)

		v_ptrs: Optional[torch.Tensor] = None
		if self.config.has_v_cache:
			v_ptrs = self._get_active_page_ptr_table(is_value=True)

		return k_ptrs, v_ptrs

	def export_active_sequence_page_counts(self) -> torch.Tensor:
		"""Returns active per-sequence page counts in slot order.

		The order matches ``export_layer_page_pointer_table`` by mirroring the
		slot ordering maintained within ``_gpu_page_table_manager``. The result
		is a CPU ``int32`` tensor of shape ``[num_active_sequences]``.
		"""

		self._ensure_initialized()
		slot_order = self._require_active_slot_order(
			op_name="export_active_sequence_page_counts"
		)
		page_counts = self._build_sequence_page_counts(slot_order)
		return torch.tensor(page_counts, dtype=torch.int32, device="cpu")

	def get_padded_3d_page_pointers(
		self,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		"""
		Returns layer-major pointer tensors grouped by sequence and page.

		Returns:
			k_tensor: Shape [num_layers, num_active_sequences, max_pages]
			v_tensor: Shape [num_layers, num_active_sequences, max_pages] (if enabled)
		"""

		self._ensure_initialized()
		slot_order = self._require_active_slot_order(
			op_name="get_padded_3d_page_pointers"
		)
		page_counts = self._build_sequence_page_counts(slot_order)

		k_tensor = self._materialize_layer_pointer_tensor(
			pointer_table=self._get_active_page_ptr_table(is_value=False),
			page_counts=page_counts,
		)

		v_tensor: Optional[torch.Tensor] = None
		if self.config.has_v_cache:
			v_tensor = self._materialize_layer_pointer_tensor(
				pointer_table=self._get_active_page_ptr_table(is_value=True),
				page_counts=page_counts,
			)

		return k_tensor, v_tensor

	# In gpu_paged_kv_manager.py
	def extend_pages_for_sequence(self, sequence_id: int, new_total_tokens: int) -> int:
		self._ensure_initialized()
		
		state = self._sequences.get(sequence_id)
		current_pages = state.pages.numel() if state else 0
		required_pages = self._geometry.required_pages(new_total_tokens)
		additional_pages = max(0, required_pages - current_pages)
		
		if additional_pages <= 0:
			return 0
		
		if additional_pages > self._free_pages.size:
			raise RuntimeError(
				f"Insufficient free pages: need {additional_pages}, have {self._free_pages.size}"
			)
		
		new_pages = self._free_pages.pop(additional_pages)
		
		if state is None:
			self._sequences[sequence_id] = _SequenceState(pages=new_pages)
		else:
			state.append_pages(new_pages)
		
		return additional_pages
