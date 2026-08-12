# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear wrappers for BatchGen execution.

- KimiLinearAttnWrapper: NoPE-MLA attention (prefill offload + paged decode).
- KimiLinearKDAWrapper: KDA linear attention with per-layer conv/recurrent
  state pools and a shared sequence->slot manager.

KDA state lifecycle:
  - prefill: slots allocated for new sequences (prepare_prefill), final
    conv/recurrent states written into the pools.
  - decode: slots looked up from cur_batch, state updated in place.
  - completion/eviction: worker calls KimiLinearKDAWrapper.free_sequences(ids)
    (hook registered next to GPU KV release).
"""

import logging
from typing import Dict, List, Optional

import torch

from batchgen.models.wrappers import AttnWrapperBase
from batchgen.models.wrappers.expert import ExpertWrapperBase


class KimiLinearExpertWrapper(ExpertWrapperBase):
    """BF16 routed/shared expert wrapper.

    Weights are BF16 (no dequant). When ``persistent=False`` the base class
    streams the expert's weights from the host copy-engine buffer on every
    forward (``load_weights`` -> ``apply_weights`` -> compute -> ``free_weights``);
    the MoE layer MUST drive every streamed expert in weight-copy-task order
    (layer-major ascending, including 0-token experts) so the copy engine does
    not stall. ``_forward_impl`` just runs the wrapped KimiMLP (w1/w3 gate+up,
    SiLU, w2 down).
    """

    def dequantize_weights(self, weights_dict):
        return weights_dict  # BF16 already

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.module(hidden_states)


class KDASlotManager:
    """Shared sequence(global id) -> slot allocator for all KDA layers."""

    def __init__(self, num_slots: int, device):
        self.num_slots = num_slots
        self.device = device
        self.seq_to_slot: Dict[int, int] = {}
        self.free_slots: List[int] = list(range(num_slots))

    def alloc(self, seq_id: int) -> int:
        if seq_id in self.seq_to_slot:
            return self.seq_to_slot[seq_id]
        if not self.free_slots:
            raise RuntimeError(
                f"KDA state pool exhausted ({self.num_slots} slots); "
                "increase pool size or implement eviction"
            )
        slot = self.free_slots.pop()
        self.seq_to_slot[seq_id] = slot
        return slot

    def lookup(self, seq_id: int) -> int:
        return self.seq_to_slot[seq_id]

    def free(self, seq_id: int) -> None:
        slot = self.seq_to_slot.pop(seq_id, None)
        if slot is not None:
            self.free_slots.append(slot)

    def contains(self, seq_id: int) -> bool:
        return seq_id in self.seq_to_slot


class KDALayerState:
    """Per-layer view: pools + slot manager + per-step batch context."""

    def __init__(self, slot_manager: KDASlotManager, conv_q, conv_k, conv_v,
                 recurrent_pool):
        self.slot_manager = slot_manager
        self.conv_q = conv_q
        self.conv_k = conv_k
        self.conv_v = conv_v
        self.recurrent_pool = recurrent_pool
        self.cur_batch_ids: List[int] = []
        self.has_initial_state: Optional[torch.Tensor] = None
        self.cur_decode_slots: Optional[torch.Tensor] = None

    def prepare_prefill(self, seq_ids: List[int]) -> torch.Tensor:
        """Allocate slots for new sequences; return (bsz,) int32 slot ids and
        set has_initial_state for resumed ones."""
        device = self.conv_q.device
        has_init = torch.tensor(
            [self.slot_manager.contains(s) for s in seq_ids],
            dtype=torch.bool, device=device,
        )
        slots = [self.slot_manager.alloc(s) for s in seq_ids]
        self.cur_batch_ids = seq_ids
        self.has_initial_state = has_init
        return torch.tensor(slots, dtype=torch.int32, device=device)

    def set_decode_batch(self, seq_ids: List[int]) -> None:
        device = self.conv_q.device
        self.cur_batch_ids = seq_ids
        self.cur_decode_slots = torch.tensor(
            [self.slot_manager.lookup(s) for s in seq_ids],
            dtype=torch.int32, device=device,
        )


class KimiLinearKDAWrapper(AttnWrapperBase):
    """Wrapper for KimiKDAAttention (linear attention, no KV cache)."""

    # Shared across all KDA layer instances (set by the PSM).
    slot_manager: Optional[KDASlotManager] = None
    layer_pools: Dict[int, KDALayerState] = {}

    def __init__(self, module, layer_idx, core_engine, engine_config,
                 model_config, persistent=True):
        super().__init__(module, layer_idx, core_engine, engine_config,
                         model_config, persistent=persistent)
        self.module_key = f"kda_attn_{layer_idx}"

    @classmethod
    def init_state_pools(cls, kda_layer_indices, num_slots, num_heads, head_dim,
                         conv_width, proj_size, device, dtype):
        """Allocate conv/recurrent pools for every KDA layer (called by PSM)."""
        cls.slot_manager = KDASlotManager(num_slots, device)
        cls.layer_pools = {}
        for i in kda_layer_indices:
            conv_q = torch.zeros(num_slots, proj_size, conv_width - 1,
                                 dtype=dtype, device=device)
            conv_k = torch.zeros(num_slots, proj_size, conv_width - 1,
                                 dtype=dtype, device=device)
            conv_v = torch.zeros(num_slots, proj_size, conv_width - 1,
                                 dtype=dtype, device=device)
            recurrent = torch.zeros(num_slots, num_heads, head_dim, head_dim,
                                    dtype=torch.float32, device=device)
            cls.layer_pools[i] = KDALayerState(
                cls.slot_manager, conv_q, conv_k, conv_v, recurrent
            )

    @classmethod
    def free_sequences(cls, seq_ids) -> None:
        """Worker hook: release KDA slots for completed/evicted sequences."""
        if cls.slot_manager is None:
            return
        for s in seq_ids:
            cls.slot_manager.free(int(s))

    @classmethod
    def reset(cls) -> None:
        cls.slot_manager = None
        cls.layer_pools = {}

    def dequantize_weights(self, weights_dict):
        return weights_dict

    def _forward_prefill(self, hidden_states, **kwargs):
        # Prepack (varlen) prefill: hidden_states is [1, total_tokens, H],
        # sequences densely packed with prepack_cu_seqlens.
        if not getattr(AttnWrapperBase, "prepack_mode", False):
            raise RuntimeError(
                "Kimi-Linear serving requires prepack prefill (default). "
                "Standard padded prefill is not supported."
            )
        state = KimiLinearKDAWrapper.layer_pools[self.layer_idx]
        seq_ids = list(AttnWrapperBase.cur_batch or [])
        device = hidden_states.device
        cu_seqlens = AttnWrapperBase.prepack_cu_seqlens.to(device)
        hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])

        slot_ids = state.prepare_prefill(seq_ids)
        out = self.module.kda_prefill_serving(
            hidden_2d, cu_seqlens, slot_ids, state.has_initial_state, state
        )
        return out.unsqueeze(0)

    def _forward_decode(self, hidden_states, **kwargs):
        state = KimiLinearKDAWrapper.layer_pools[self.layer_idx]
        state.set_decode_batch(list(AttnWrapperBase.cur_batch or []))
        return self.module.kda_decode_serving(hidden_states, state)


class KimiLinearAttnWrapper(AttnWrapperBase):
    """Wrapper for KimiMLAAttention (NoPE-MLA, paged KV)."""

    def __init__(self, module, layer_idx, core_engine, engine_config,
                 model_config, persistent=True):
        super().__init__(module, layer_idx, core_engine, engine_config,
                         model_config, persistent=persistent)

    def dequantize_weights(self, weights_dict):
        return weights_dict

    def _forward_prefill(self, hidden_states, **kwargs):
        # Prepack (varlen) prefill: hidden_states is [1, total_tokens, H].
        if not getattr(AttnWrapperBase, "prepack_mode", False):
            raise RuntimeError(
                "Kimi-Linear serving requires prepack prefill (default). "
                "Standard padded prefill is not supported."
            )
        device = hidden_states.device
        hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        cu_seqlens = AttnWrapperBase.prepack_cu_seqlens.to(device)

        attn_output, offload_kv = self.module.mla_prefill_nope_prepacked(
            hidden_2d,
            AttnWrapperBase.position_ids,
            cu_seqlens,
            AttnWrapperBase.prepack_max_seqlen,
            AttnWrapperBase.prepack_num_sequences,
        )
        self._offload_prepacked_kv(offload_kv, cu_seqlens)
        return attn_output.unsqueeze(0)

    def _offload_prepacked_kv(self, offload_kv, cu_seqlens):
        """Offload compressed MLA KV per-sequence to host paged KV."""
        global_sequence_ids = list(AttnWrapperBase.cur_batch or [])
        num_sequences = AttnWrapperBase.prepack_num_sequences
        view = self.core_engine.host_paged_kv_worker_view
        for seq_idx in range(num_sequences):
            start_idx = int(cu_seqlens[seq_idx].item())
            end_idx = int(cu_seqlens[seq_idx + 1].item())
            seq_len = end_idx - start_idx
            if seq_len == 0:
                continue
            seq_kv = offload_kv[start_idx:end_idx].unsqueeze(0).unsqueeze(2)
            view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=[global_sequence_ids[seq_idx]],
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )

    def _forward_decode(self, hidden_states, **kwargs):
        position_ids = AttnWrapperBase.position_ids
        cache_seqlens = AttnWrapperBase.cache_seqlens
        max_seqlen = AttnWrapperBase.max_seqlen
        gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager

        attn_output, k_tensor = self.module.mla_decoding_nope_with_pagekv(
            hidden_states,
            position_ids,
            cache_seqlens,
            max_seqlen,
            None,
            gpu_paged_kv_manager,
            self.layer_idx,
            None,
        )

        if AttnWrapperBase.kv_append_callback is not None:
            AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)
        return attn_output
