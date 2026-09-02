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

KDA state lifecycle (M5.1 unified):
  - storage + slot accounting are owned by ONE KDAStateGPUManager
    (fixed-address recurrent/conv pools + persistent decode slot-index
    buffer — the CUDA-graph-ready canonical home). The per-layer
    KDALayerState pools handed to serving_modules are VIEWS of the
    manager's tensors; the KDASlotManager facade delegates to it.
  - prefill: slots allocated for new sequences (prepare_prefill); the
    manager zeroes a fresh slot across every layer's conv+recurrent pool
    (F4 fix), then final conv/recurrent states are written into the pools.
  - decode: slots staged in place through the manager's persistent slot
    buffer (prepare_decode_step), state updated in place.
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
    """Shared sequence(global id) -> slot facade for all KDA layers.

    M5.1: slot accounting is owned by the authoritative KDAStateGPUManager
    (free list, seq->slot map, and the F4 zero-on-alloc that clears a fresh
    slot across every layer's conv+recurrent pool — the wrapper pools are
    views of the manager's tensors, so one zeroing covers all layers). This
    facade only adapts the wrapper-facing alloc/lookup/free/contains API
    and holds no state of its own.
    """

    def __init__(self, state_manager):
        self.state_manager = state_manager

    @property
    def num_slots(self) -> int:
        return self.state_manager.config.num_state_items

    @property
    def device(self):
        return self.state_manager.device

    def alloc(self, seq_id: int) -> int:
        return self.state_manager.allocate_state_item(int(seq_id))

    def lookup(self, seq_id: int) -> int:
        return self.state_manager.get_sequence_state_item(int(seq_id))

    def free(self, seq_id: int) -> None:
        self.state_manager.release_sequence_states([int(seq_id)])

    def contains(self, seq_id: int) -> bool:
        return self.state_manager.has_sequence_state_item(int(seq_id))


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
        self.cur_batch_ids = seq_ids
        # Staged through the manager's persistent int32 slot buffer: fixed
        # address (CUDA-graph static input), refreshed in place each step;
        # no per-layer tensor allocation.
        self.cur_decode_slots = (
            self.slot_manager.state_manager.prepare_decode_step(seq_ids)
        )


class KimiLinearKDAWrapper(AttnWrapperBase):
    """Wrapper for KimiKDAAttention (linear attention, no KV cache)."""

    # Shared across all KDA layer instances (set by the PSM).
    state_manager = None  # KDAStateGPUManager: pools + slot accounting
    slot_manager: Optional[KDASlotManager] = None
    layer_pools: Dict[int, KDALayerState] = {}

    def __init__(self, module, layer_idx, core_engine, engine_config,
                 model_config, persistent=True):
        super().__init__(module, layer_idx, core_engine, engine_config,
                         model_config, persistent=persistent)
        self.module_key = f"kda_attn_{layer_idx}"
        self._resident_prefill_segment_tokens = None

    @classmethod
    def init_state_pools(cls, kda_layer_indices, num_slots, num_heads, head_dim,
                         conv_width, proj_size, device, dtype):
        """Build the unified KDA state pools for every KDA layer (PSM call).

        M5.1: a single KDAStateGPUManager owns the storage — recurrent pool
        (L, slots, H, K, K) fp32, conv pools (L, slots, proj, W-1) and the
        persistent decode slot buffer — each allocated once with a fixed
        address (CUDA-graph capture requirement). Every KDALayerState
        receives VIEWS of the manager's tensors; slot alloc/free/zeroing is
        delegated to the manager through the KDASlotManager facade.
        """
        from batchgen.kv_cache.kda_state_gpu_manager import (
            KDAStateGPUConfig,
            KDAStateGPUManager,
        )

        cls.state_manager = KDAStateGPUManager(
            config=KDAStateGPUConfig(
                num_kda_layers=len(kda_layer_indices),
                num_state_items=num_slots,
                num_heads=num_heads,
                head_dim=head_dim,
                conv_dim=proj_size,
                conv_width=conv_width,
                conv_dtype=dtype,
                cuda_graph_max_slots=num_slots,
            ),
            device=device,
        )
        cls.state_manager.initialize()
        cls.slot_manager = KDASlotManager(cls.state_manager)
        cls.layer_pools = {}
        for physical, i in enumerate(kda_layer_indices):
            conv_q, conv_k, conv_v = (
                cls.state_manager.get_layer_conv_views(physical)
            )
            recurrent = cls.state_manager.get_layer_recurrent_view(physical)
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
        cls.state_manager = None
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
        profiler = getattr(self.module, "_streamed_sp8_profiler", None)
        if (
            profiler is not None
            and not profiler._prefill_profile_enabled
        ):
            profiler = None
        attention_span = (
            profiler.begin_profile_span() if profiler is not None else None
        )
        state = KimiLinearKDAWrapper.layer_pools[self.layer_idx]
        seq_ids = list(AttnWrapperBase.cur_batch or [])
        device = hidden_states.device
        cu_seqlens = AttnWrapperBase.prepack_cu_seqlens.to(device)
        hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])

        slot_ids = state.prepare_prefill(seq_ids)
        segment_tokens = self._resident_prefill_segment_tokens
        if segment_tokens is None:
            out = self.module.kda_prefill_serving(
                hidden_2d, cu_seqlens, slot_ids, state.has_initial_state, state
            )
        else:
            out = self.module.kda_prefill_serving(
                hidden_2d,
                cu_seqlens,
                slot_ids,
                state.has_initial_state,
                state,
                segment_tokens=segment_tokens,
            )
        if profiler is not None:
            profiler.end_profile_span("kda_attention", attention_span)
        return out.unsqueeze(0)

    def _forward_decode(self, hidden_states, **kwargs):
        # Empty DP rank (bsz==0): skip — 0-row conv1d_update /
        # fused_recurrent grids are undefined. Streamed-expert lockstep is
        # preserved: the MoE drive (moe_forward_serving) runs in the decoder
        # layer's FFN after attention and still load+frees every expert via
        # a 0-row forward.
        if hidden_states.shape[0] == 0:
            return hidden_states
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

        profiler = getattr(self.module, "_streamed_sp8_profiler", None)
        if (
            profiler is not None
            and not profiler._prefill_profile_enabled
        ):
            profiler = None
        attention_span = (
            profiler.begin_profile_span() if profiler is not None else None
        )
        attn_output, offload_kv = self.module.mla_prefill_nope_prepacked(
            hidden_2d,
            AttnWrapperBase.position_ids,
            cu_seqlens,
            AttnWrapperBase.prepack_max_seqlen,
            AttnWrapperBase.prepack_num_sequences,
        )
        if profiler is not None:
            profiler.end_profile_span("mla_attention", attention_span)
        offload_span = (
            profiler.begin_profile_span() if profiler is not None else None
        )
        self._offload_prepacked_kv(offload_kv, cu_seqlens)
        if profiler is not None:
            profiler.end_profile_span("mla_kv_offload", offload_span)
        return attn_output.unsqueeze(0)

    def _offload_prepacked_kv(self, offload_kv, cu_seqlens):
        """Offload compressed MLA KV per-sequence to host paged KV."""
        global_sequence_ids = list(AttnWrapperBase.cur_batch or [])
        num_sequences = AttnWrapperBase.prepack_num_sequences
        view = self.core_engine.host_paged_kv_worker_view
        # The worker publishes the same lengths as a host list next to the
        # device cu_seqlens; reading them here avoids two device syncs per
        # sequence per MLA layer (16 per layer at exact 64K).
        seq_lengths = AttnWrapperBase.prepack_seq_lengths
        if seq_lengths is None or len(seq_lengths) != num_sequences:
            seq_lengths = cu_seqlens.diff().tolist()
        start_idx = 0
        for seq_idx in range(num_sequences):
            seq_len = int(seq_lengths[seq_idx])
            end_idx = start_idx + seq_len
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
            start_idx = end_idx

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

        import os as _os
        _skip_cb = _os.environ.get("BATCHGEN_SKIP_KV_CALLBACK", "0") == "1"
        if (not _skip_cb and AttnWrapperBase.kv_append_callback is not None
                and k_tensor.shape[0] > 0):
            AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)
        return attn_output
