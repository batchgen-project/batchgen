from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

from batchgen.kv_cache.coordinator_utils import (
    resolve_from_layer_mapping,
)
from batchgen.kv_cache.gpu_paged_kv_manager import (
    _normalize_device,
    _normalize_gpu_layer_mapping,
    _TensorStack,
)


@dataclass(frozen=True)
class KDAStateGPUConfig:
    """Geometry for the fixed-size per-sequence KDA state pool.

    Unlike the rolling compressor there is no ring: the fla kernel updates
    each sequence's recurrent + short-conv state *in place*, so a sequence
    owns exactly one slot per state item for its whole lifetime.

    ``conv_width`` is the conv KERNEL width W; the conv pools store the last
    W-1 raw inputs per slot — the ``causal_conv1d.cu`` state contract
    (per-layer view shape ``(num_state_items, conv_dim, W-1)``).
    """

    num_kda_layers: int
    num_state_items: int
    num_heads: int  # HV (number of value heads)
    head_dim: int = 128
    conv_dim: Optional[int] = None  # num_heads * head_dim; derived if None
    conv_width: int = 4  # kernel width W; pools store W-1 entries per slot
    recurrent_dtype: torch.dtype = torch.float32
    conv_dtype: torch.dtype = torch.bfloat16
    cuda_graph_max_slots: Optional[int] = None
    logical_to_physical_layer: Optional[Sequence[int]] = None

    def resolved_conv_dim(self) -> int:
        if self.conv_dim is not None:
            return int(self.conv_dim)
        return int(self.num_heads) * int(self.head_dim)


@dataclass(frozen=True)
class KDAStateGPUStats:
    num_total_state_items: int
    num_free_state_items: int
    num_used_state_items: int
    num_active_sequences: int


class KDAStateGPUManager:
    """GPU storage for per-sequence Kimi Delta Attention recurrent + conv state.

    Each active sequence owns one fixed-size state item (one slot) per KDA
    layer. The fla kernel mutates ``recurrent_state`` and the three short
    causal-conv states in place, so this manager only handles slot
    allocation, view export, free-list bookkeeping, and zero-on-alloc.

    M5.1: this manager is the canonical, CUDA-graph-ready home of the KDA
    state — the recurrent pool, the three conv pools and the persistent
    decode slot-index buffer are each allocated ONCE with a fixed address.
    ``KimiLinearKDAWrapper``'s per-layer pools are views of these tensors
    and its slot facade delegates all alloc/free/zeroing here. Per-layer
    conv views are ``(num_state_items, conv_dim, conv_width-1)`` — the
    ``causal_conv1d.cu`` layout (matching the wrapper).
    """

    manager_name = "KDAStateGPUManager"

    def __init__(
        self,
        *,
        config: KDAStateGPUConfig,
        device: Union[str, int, torch.device],
    ) -> None:
        self.config = config
        self.device = _normalize_device(device)
        if self.config.num_kda_layers <= 0:
            raise ValueError("num_kda_layers must be > 0")
        if self.config.num_state_items <= 0:
            raise ValueError("num_state_items must be > 0")
        if self.config.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.config.head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        if self.config.conv_width < 2:
            raise ValueError(
                "conv_width must be >= 2 (pools store W-1 entries per slot)"
            )
        if self.config.resolved_conv_dim() <= 0:
            raise ValueError("conv_dim must be > 0")
        self._logical_to_physical_layer = _normalize_gpu_layer_mapping(
            self.config.logical_to_physical_layer,
            self.config.num_kda_layers,
        )
        self._reset_runtime_state()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        if self._is_initialized:
            return
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        cfg = self.config
        conv_dim = cfg.resolved_conv_dim()
        self._recurrent_state = torch.zeros(
            (
                cfg.num_kda_layers,
                cfg.num_state_items,
                cfg.num_heads,
                cfg.head_dim,
                cfg.head_dim,
            ),
            dtype=cfg.recurrent_dtype,
            device=self.device,
        )
        # causal_conv1d.cu contract: a slot holds the last W-1 raw inputs,
        # so per-layer views are contiguous (num_state_items, conv_dim, W-1)
        # 3-D tensors satisfying the kernel's dim()==3 && size(2)==W-1 check.
        # The 4-D allocation keeps ONE fixed base address per q/k/v pool
        # (CUDA-graph capture requirement).
        conv_shape = (
            cfg.num_kda_layers,
            cfg.num_state_items,
            conv_dim,
            cfg.conv_width - 1,
        )
        self._conv_q = torch.zeros(
            conv_shape, dtype=cfg.conv_dtype, device=self.device
        )
        self._conv_k = torch.zeros(
            conv_shape, dtype=cfg.conv_dtype, device=self.device
        )
        self._conv_v = torch.zeros(
            conv_shape, dtype=cfg.conv_dtype, device=self.device
        )
        self._free_state_items = _TensorStack(cfg.num_state_items)
        self._ensure_prepared_state_slot_buffer()
        self._is_initialized = True

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        if not self._is_initialized:
            return
        self._reset_runtime_state()
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # slot allocation / free
    # ------------------------------------------------------------------
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

    def reset_state_items(self, state_item_ids: Sequence[int]) -> None:
        """Zero recurrent + conv (+ aux, if present) state for the slots."""
        self._ensure_initialized()
        slots = [int(s) for s in state_item_ids]
        if not slots:
            return
        idx = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        for slot in slots:
            if slot < 0 or slot >= self.config.num_state_items:
                raise IndexError(f"state item id {slot} out of range")
        # index_fill_ over the state-item dim (dim=1) for every layer at once.
        self._recurrent_state.index_fill_(1, idx, 0)
        self._conv_q.index_fill_(1, idx, 0)
        self._conv_k.index_fill_(1, idx, 0)
        self._conv_v.index_fill_(1, idx, 0)
        # Per-slot auxiliary rows (block_reps on branches that carry them)
        # are zeroed too so the F4 zero-on-alloc covers every pool.
        block_reps = getattr(self, "_block_reps", None)
        if block_reps is not None:
            block_reps.index_fill_(0, idx, 0)

    # ------------------------------------------------------------------
    # view export
    # ------------------------------------------------------------------
    def get_layer_recurrent_view(self, logical_layer: int) -> torch.Tensor:
        """Recurrent state view [num_state_items, HV, head_dim, head_dim]."""
        self._ensure_initialized()
        physical = self.resolve_physical_layer(logical_layer)
        return self._recurrent_state[physical]

    def get_layer_conv_views(
        self, logical_layer: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Conv views (q, k, v), each [num_state_items, conv_dim, width-1]."""
        self._ensure_initialized()
        physical = self.resolve_physical_layer(logical_layer)
        return (
            self._conv_q[physical],
            self._conv_k[physical],
            self._conv_v[physical],
        )

    def get_recurrent_tensors(self) -> torch.Tensor:
        self._ensure_initialized()
        return self._recurrent_state

    def get_conv_tensors(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_initialized()
        return (self._conv_q, self._conv_k, self._conv_v)

    # ------------------------------------------------------------------
    # decode-step slot preparation (CUDA graph static buffer)
    # ------------------------------------------------------------------
    def prepare_decode_step(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        """Fill the static slot-index buffer with each sequence's slot.

        Returns the (view onto the) prepared slot buffer for the batch.
        """
        self._ensure_initialized()
        slots = [
            self._get_sequence_state_item(int(seq_id))
            for seq_id in sequence_ids
        ]
        self._write_prepared_state_slots(slots)
        return self._prepared_state_slots[: len(slots)]

    def get_prepared_state_slots(self) -> torch.Tensor:
        self._ensure_initialized()
        return self._prepared_state_slots[: self._prepared_state_slot_count]

    # ------------------------------------------------------------------
    # sequence -> slot lookup
    # ------------------------------------------------------------------
    @property
    def sequence_state_items(self) -> dict[int, int]:
        return dict(self._sequence_state_items)

    def get_sequence_state_item(self, sequence_id: int) -> int:
        self._ensure_initialized()
        return self._get_sequence_state_item(int(sequence_id))

    def has_sequence_state_item(self, sequence_id: int) -> bool:
        self._ensure_initialized()
        return int(sequence_id) in self._sequence_state_items

    def get_stats(self) -> KDAStateGPUStats:
        self._ensure_initialized()
        used = self.config.num_state_items - self._free_state_items.size
        return KDAStateGPUStats(
            num_total_state_items=self.config.num_state_items,
            num_free_state_items=self._free_state_items.size,
            num_used_state_items=used,
            num_active_sequences=len(self._sequence_state_items),
        )

    # ------------------------------------------------------------------
    # layer mapping
    # ------------------------------------------------------------------
    @property
    def uses_logical_layer_mapping(self) -> bool:
        return self._logical_to_physical_layer is not None

    def resolve_physical_layer(self, logical_layer_id: int) -> int:
        logical_layer_id = int(logical_layer_id)
        if logical_layer_id < 0:
            raise IndexError("logical layer id must be >= 0")
        if self._logical_to_physical_layer is None:
            if logical_layer_id >= self.config.num_kda_layers:
                raise IndexError(
                    f"layer_idx {logical_layer_id} out of range"
                )
            return logical_layer_id
        return resolve_from_layer_mapping(
            "GPU KDA state",
            "state",
            self._logical_to_physical_layer,
            logical_layer_id,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _reset_runtime_state(self) -> None:
        self._is_initialized = False
        self._recurrent_state: Optional[torch.Tensor] = None
        self._conv_q: Optional[torch.Tensor] = None
        self._conv_k: Optional[torch.Tensor] = None
        self._conv_v: Optional[torch.Tensor] = None
        self._free_state_items: Optional[_TensorStack] = None
        self._sequence_state_items: dict[int, int] = {}
        self._prepared_state_slots: Optional[torch.Tensor] = None
        self._prepared_state_slot_count = 0

    def _ensure_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "KDAStateGPUManager.initialize must be called before use"
            )

    def _ensure_state_item(self, sequence_id: int) -> int:
        state_item_id = self._sequence_state_items.get(sequence_id)
        if state_item_id is not None:
            return state_item_id
        if self._free_state_items.size <= 0:
            raise RuntimeError("Insufficient free KDA state items")
        state_item = self._free_state_items.pop(1)
        state_item_id = int(state_item[0].item())
        # F4 fix: zero the (possibly recycled) slot across every layer's
        # conv + recurrent pool on FRESH alloc — a recycled slot must not
        # leak the previous sequence's state, and layers > 0 seeing
        # has_initial_state=True for a just-allocated sequence stays
        # harmless (zero state == no state). Idempotent re-allocs return
        # above and never re-zero live state.
        self.reset_state_items([state_item_id])
        self._sequence_state_items[sequence_id] = state_item_id
        return state_item_id

    def _get_sequence_state_item(self, sequence_id: int) -> int:
        state_item_id = self._sequence_state_items.get(int(sequence_id))
        if state_item_id is None:
            raise KeyError(
                f"Sequence {sequence_id} has no KDA state item"
            )
        return state_item_id

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


__all__ = [
    "KDAStateGPUConfig",
    "KDAStateGPUManager",
    "KDAStateGPUStats",
]
