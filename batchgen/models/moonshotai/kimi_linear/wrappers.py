# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
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
    the MoE layer declares its 0-token experts with
    ``skip_idle_streamed_experts`` and runs the rest in weight-copy-task order
    (layer-major ascending). ``_forward_impl`` just runs the wrapped KimiMLP
    (w1/w3 gate+up, SiLU, w2 down).
    """

    def dequantize_weights(self, weights_dict):
        return weights_dict  # BF16 already

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.module(hidden_states)


def kda_state_pool_decision(
    *,
    existing_config,
    existing_device,
    live_sequences: int,
    wanted_config,
    wanted_device,
) -> str:
    """``"reuse"`` or ``"rebuild"`` for the KDA state pools. Pure.

    For a linear-attention model the per-sequence recurrent + conv state IS
    that sequence's KV: it is written by its prefill and read by every decode
    step until the sequence completes or is evicted. The pools therefore
    outlive one phase, exactly like the GPU KV pool, which is "sized exactly
    once per process and every later activation maps that same size".

    ``init_state_pools`` runs inside ``_build_model``, and the prefill->decode
    switch rebuilds the model whenever the phase instances are not persistent
    (``deep_free_model_memory`` then ``configure_decoding``'s
    ``if self.model is None: self._build_model()``). Building a second
    ``KDAStateGPUManager`` there replaces the sequence->slot map with an empty
    one, so every sequence the prefill just wrote state for loses it and the
    first decode step raises ``KeyError: Sequence <global_idx> has no KDA
    state item``. Reusing a manager whose config and device are identical
    keeps that map AND the pools' fixed addresses, which is what CUDA-graph
    capture requires anyway.

    A changed config cannot be reused: the pools' shapes differ. Dropping live
    state to honour it would silently destroy decoded context, so that case is
    refused by name instead; it is only a rebuild when nothing is live.
    """
    if existing_config is None:
        return "rebuild"
    if existing_config == wanted_config and existing_device == wanted_device:
        return "reuse"
    if live_sequences > 0:
        raise RuntimeError(
            "KDA state pools must be rebuilt with a different layout while "
            f"{live_sequences} sequences still hold state in them: their "
            "recurrent + conv state is their KV and cannot be recomputed. "
            f"existing={existing_config} on {existing_device}; "
            f"wanted={wanted_config} on {wanted_device}"
        )
    return "rebuild"


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
    # (ChunkBatch, levels) of the prefix-pool chunk being prefilled. Every KDA
    # layer of a chunk runs the same levels over the same slots, so the context
    # is built at the first of them and reused by the other 68.
    _pool_levels = None

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

        Idempotent for an identical layout: ``_build_model`` runs again on a
        phase switch that does not keep its instances, and a second manager
        there would drop every live sequence's state (see
        ``kda_state_pool_decision``). The pools are then kept and only the
        per-layer views and the facade are re-derived.
        """
        from batchgen.kv_cache.kda_state_gpu_manager import (
            KDAStateGPUConfig,
            KDAStateGPUManager,
        )

        config = KDAStateGPUConfig(
            num_kda_layers=len(kda_layer_indices),
            num_state_items=num_slots,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_dim=proj_size,
            conv_width=conv_width,
            conv_dtype=dtype,
            cuda_graph_max_slots=num_slots,
        )
        wanted_device = torch.device(device)
        existing = cls.state_manager
        live = (
            len(existing.sequence_state_items) if existing is not None else 0
        )
        decision = kda_state_pool_decision(
            existing_config=None if existing is None else existing.config,
            existing_device=None if existing is None else existing.device,
            live_sequences=live,
            wanted_config=config,
            wanted_device=wanted_device,
        )
        if decision == "reuse":
            # A model rebuild (phase switch without persistent instances) must
            # not cost the sequences their state; the views below are re-derived
            # from the SAME pools, so their addresses do not move either.
            logging.info(
                "[KDA] reusing the existing state pools across a model "
                "rebuild: %d slots, %d sequences still holding state",
                num_slots, live,
            )
        else:
            cls.state_manager = KDAStateGPUManager(
                config=config, device=device
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
        cls._pool_levels = None

    @classmethod
    def _pool_level_context(cls, state, seq_ids, slot_ids):
        """The levels of the prefix-pool chunk being prefilled, else None.

        A pooled segment is a prefill row of its own, so a row that continues
        one cannot start from zero state: the chunk runs level by level and a
        child's slot is forked from its parent segment's. The parent is keyed
        by the segment's pseudo sequence id, which holds its slot until the
        wave ends, so it resolves whether the parent ran in this chunk or in an
        earlier one.
        """
        batch = getattr(AttnWrapperBase, "prefill_pool_batch", None)
        if batch is None:
            cls._pool_levels = None
            return None
        if cls._pool_levels is not None and cls._pool_levels[0] is batch:
            return cls._pool_levels[1]
        from batchgen.prefix_reuse.levels import state_levels

        segment_sids = getattr(
            AttnWrapperBase, "prefill_pool_segment_sids", None
        )
        if segment_sids is None:
            raise RuntimeError(
                "a prefill prefix pool chunk reached a KDA layer without the "
                "wave's segment pseudo sequence ids"
            )
        levels = state_levels(
            batch,
            segment_sids=segment_sids,
            row_ids=seq_ids,
            slots=slot_ids,
            has_initial_state=state.has_initial_state,
            slot_of=cls.slot_manager.lookup,
            # A loaded segment left no state to fork; its reader resumes from
            # the slot the restore bound, which state_levels verifies.
            materialized=getattr(
                AttnWrapperBase, "prefill_pool_materialized_segments", ()
            ) or (),
        )
        cls._pool_levels = (batch, levels)
        return levels

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
        pool_levels = self._pool_level_context(state, seq_ids, slot_ids)
        segment_tokens = self._resident_prefill_segment_tokens
        if segment_tokens is None:
            out = self.module.kda_prefill_serving(
                hidden_2d, cu_seqlens, slot_ids, state.has_initial_state, state,
                pool_levels=pool_levels,
            )
        else:
            out = self.module.kda_prefill_serving(
                hidden_2d,
                cu_seqlens,
                slot_ids,
                state.has_initial_state,
                state,
                segment_tokens=segment_tokens,
                pool_levels=pool_levels,
            )
        if profiler is not None:
            profiler.end_profile_span("kda_attention", attention_span)
        return out.unsqueeze(0)

    def _forward_decode(self, hidden_states, **kwargs):
        # Empty DP rank (bsz==0): skip — 0-row conv1d_update /
        # fused_recurrent grids are undefined. The decoder layer's FFN
        # (moe_forward_serving) still runs and declares every streamed expert
        # idle, which releases their slots.
        if hidden_states.shape[0] == 0:
            return hidden_states
        state = KimiLinearKDAWrapper.layer_pools[self.layer_idx]
        state.set_decode_batch(list(AttnWrapperBase.cur_batch or []))
        return self.module.kda_decode_serving(hidden_states, state)


class KimiLinearAttnWrapper(AttnWrapperBase):
    """Wrapper for KimiMLAAttention (NoPE-MLA, paged KV)."""

    #: ``(ChunkBatch, DsaPoolPrefill)`` of the prefix-pool chunk being
    #: prefilled, or None. The addressing is index math over the chunk's page
    #: table, identical for every DSA layer, so it is built once per chunk.
    _pool_sparse_ctx = None

    def __init__(self, module, layer_idx, core_engine, engine_config,
                 model_config, persistent=True):
        super().__init__(module, layer_idx, core_engine, engine_config,
                         model_config, persistent=persistent)

    def dequantize_weights(self, weights_dict):
        return weights_dict

    def _forward_prefill(self, hidden_states, **kwargs):
        # Retire the PREVIOUS MLA layer's async KV offloads before this layer
        # allocates: the d2h memcpy must land before the caching allocator can
        # hand its source storage to this layer's K/V (expandable_segments).
        AttnWrapperBase.retire_pending_prefill_offloads_before_layer(
            self.layer_idx,
            device=hidden_states.device,
        )
        # Prepack (varlen) prefill: hidden_states is [1, total_tokens, H].
        if not getattr(AttnWrapperBase, "prepack_mode", False):
            raise RuntimeError(
                "Kimi-Linear serving requires prepack prefill (default). "
                "Standard padded prefill is not supported."
            )
        indexer = getattr(self.module, "indexer", None)
        # Resolved first, before any projection runs: a pooled DSA wave needs
        # this layer's latent AND index-key pool rows, and a pool that cannot
        # give both is named here rather than after the compute.
        pool_ctx = latent_pool = indexer_pool = None
        if indexer is not None:
            pool_ctx, latent_pool, indexer_pool = self._pool_sparse_rows()
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
        index_keys = None
        if indexer is not None:
            from .dsa_prefill import prefill_index_keys

            # Computed once: scored by the sparse prefill below AND stored for
            # decode, so both read the same keys.
            index_keys = prefill_index_keys(indexer, hidden_2d)
        if index_keys is not None and self._prefill_needs_selection(indexer):
            from .dsa_sparse_prefill import dsa_sparse_prefill

            attn_output, offload_kv = dsa_sparse_prefill(
                self.module,
                hidden_2d,
                index_keys,
                cu_seqlens,
                AttnWrapperBase.prepack_max_seqlen,
                pool=pool_ctx,
                latent_pool=latent_pool,
                indexer_pool=indexer_pool,
            )
        else:
            attn_output, offload_kv = self.module.mla_prefill_nope_prepacked(
                hidden_2d,
                AttnWrapperBase.position_ids,
                cu_seqlens,
                AttnWrapperBase.prepack_max_seqlen,
                AttnWrapperBase.prepack_num_sequences,
                **self._pool_prefill_kwargs(),
            )
        if profiler is not None:
            profiler.end_profile_span("mla_attention", attention_span)
        offload_span = (
            profiler.begin_profile_span() if profiler is not None else None
        )
        if AttnWrapperBase.prefill_pool_batch is not None:
            # One pass over the rows writing BOTH regions: the latent and, on
            # a DSA layer, the index keys of the same tokens at the same Host
            # positions. Two passes would leave a window in which a region
            # holds a row the other does not.
            self._offload_prefill_pool_rows(offload_kv, index_keys)
        else:
            self._offload_prepacked_kv(offload_kv, cu_seqlens)
            if index_keys is not None:
                self._write_prefill_index_keys(index_keys, cu_seqlens)
        if profiler is not None:
            profiler.end_profile_span("mla_kv_offload", offload_span)
        return attn_output.unsqueeze(0)

    @staticmethod
    def _prefill_needs_selection(indexer) -> bool:
        """Whether some query of this packed batch sees more than ``index_topk``
        keys. Kimi prefill attends within each packed row only (the dense path
        hands FlashAttention the same ``cu_seqlens`` for q and k), so a row's
        longest causal context is its own length; when every row fits, the
        selection is the identity and the dense K3 prefill is exact.

        A prefix-pool row is the exception: it continues a prefix that is
        already KV, so its causal context is longer than its own query span
        and the pool is only readable through the sparse path (which the
        selection being the identity makes exact for a short context too).
        """
        if AttnWrapperBase.prefill_pool_batch is not None:
            return True
        return int(AttnWrapperBase.prepack_max_seqlen) > int(indexer.index_topk)

    def _pool_sparse_rows(self):
        """``(pool chunk, latent pool row, indexer pool row)``, or three Nones.

        Outside a pool wave the sparse prefill runs over its own packed keys
        and takes no pool argument. Inside one, every DSA layer shares the
        chunk's addressing -- it is pure index math over the chunk's page
        table -- so it is built once per chunk and cached on the class, the
        same way the KDA levels are.
        """
        pool_batch = AttnWrapperBase.prefill_pool_batch
        if pool_batch is None:
            KimiLinearAttnWrapper._pool_sparse_ctx = None
            return None, None, None
        from .dsa_sparse_prefill import build_dsa_pool_prefill

        # Resolved BEFORE any index math, so a pool this layer cannot score
        # against is refused by name and not by an attribute error halfway
        # through building a chunk's addressing.
        pool = AttnWrapperBase.prefill_pool
        indexer_row = (
            (pool.indexer_row or {}).get(self.layer_idx)
            if pool is not None and pool.indexer is not None
            else None
        )
        if indexer_row is None:
            raise RuntimeError(
                f"Kimi-K3.1 DSA layer {self.layer_idx} has no prefix-pool "
                "indexer row; the pool must keep one index-key row per DSA "
                "layer or a pooled suffix would select over its own keys only"
            )
        latent_row = self._pool_prefill_kwargs()["pool_layer"]
        cached = KimiLinearAttnWrapper._pool_sparse_ctx
        if cached is not None and cached[0] is pool_batch:
            ctx = cached[1]
        else:
            ctx = build_dsa_pool_prefill(
                page_table=pool_batch.page_table,
                cache_seqlens=pool_batch.cache_seqlens,
                cu_seqlens_q=pool_batch.cu_seqlens_q,
                position_ids=AttnWrapperBase.position_ids,
                slots=pool_batch.slots,
                page_tokens=pool.page_tokens,
                max_cache_seqlen=pool_batch.max_cache_seqlen,
            )
            KimiLinearAttnWrapper._pool_sparse_ctx = (pool_batch, ctx)
        return ctx, pool.k[latent_row], pool.indexer[indexer_row]

    def materialize_pool_segment_index_keys(self, *, index_keys, slots):
        """Worker seam: load one slab of cached INDEX KEYS into this layer.

        The dual-region counterpart of ``materialize_pool_segment``. Same
        duck-typed contract: the worker walks its attention layers and calls
        this on the ones that have it, so a model without an indexer is
        skipped. The row comes from the same indexer map the computed path
        uses, so a loaded prefix's keys and a computed one's land in the same
        row at the same slots.
        """

        pool = AttnWrapperBase.prefill_pool
        if pool is None or AttnWrapperBase.prefill_pool_batch is None:
            raise RuntimeError(
                f"Kimi-K3.1 DSA layer {self.layer_idx} was asked to "
                "materialize index keys outside a prefix-pool wave"
            )
        row = (pool.indexer_row or {}).get(self.layer_idx)
        if row is None:
            raise RuntimeError(
                f"Kimi-K3.1 DSA layer {self.layer_idx} has no prefix-pool "
                "indexer row to materialize into"
            )
        return self.module.dsa_materialize_pool_segment_index_keys(
            index_keys, pool=pool, indexer_row=row, slots=slots
        )

    def _write_prefill_index_keys(self, index_keys, cu_seqlens):
        """Kimi-K3.1: leave one index key per prompt token in the aux region.

        Same packed rows, ids and lengths the latent offload just used, so the
        two regions hold the same tokens at the same positions. The PACKED
        form: a prefix-pool wave writes both regions row by row instead
        (``_offload_prefill_pool_rows``), because a pooled row's Host
        destination is its own start, not token 0.
        """
        from .dsa_prefill import offload_packed_index_keys, quantize_index_keys

        seq_lengths = AttnWrapperBase.prepack_seq_lengths
        if seq_lengths is None or len(seq_lengths) != AttnWrapperBase.prepack_num_sequences:
            seq_lengths = cu_seqlens.diff().tolist()
        keys_fp8, scales = quantize_index_keys(index_keys)
        offload_packed_index_keys(
            aux_view=AttnWrapperBase.host_paged_kv_worker_view_aux,
            layer_idx=self.layer_idx,
            keys_fp8=keys_fp8,
            scales=scales,
            sequence_ids=list(AttnWrapperBase.cur_batch or []),
            sequence_lengths=seq_lengths,
            pin=AttnWrapperBase.pin_prefill_offload_tensor,
            track=AttnWrapperBase.track_prefill_offload_task,
        )

    def _pool_prefill_kwargs(self):
        """This layer's prefix-pool arguments, empty outside a pool wave.

        The pool holds the EXPANDED K/V of the MLA layers densely, one row per
        MLA layer, so the row comes from the pool layout's own engine-layer map
        (``prefill_pool_layout``). A Host-KV physical layer is NOT that row: the
        48B profile keeps all 27 layers and resolves to itself, which indexes
        past a 7-row pool, while K3's 93->24 map happens to coincide.
        """
        pool_batch = AttnWrapperBase.prefill_pool_batch
        if pool_batch is None:
            return {}
        layer_map = AttnWrapperBase.prefill_pool_layer_map
        row = int(layer_map[self.layer_idx]) if layer_map else -1
        if row < 0:
            raise RuntimeError(
                f"MLA layer {self.layer_idx} has no prefix-pool row; the pool "
                f"layout covers {len(layer_map) if layer_map else 0} engine layers"
            )
        return {
            "pool": AttnWrapperBase.prefill_pool,
            "pool_batch": pool_batch,
            "pool_layer": row,
        }

    def materialize_pool_segment(self, *, latent, slots):
        """Worker seam (M3 Slice 3): load one slab into THIS layer's pool row.

        Duck-typed: the worker stages a chunk's materialized segments by
        walking its attention layers and calling this on the ones that have
        it, so a model without a materialization contract is simply skipped.
        The pool row comes from the same layout map the computed path uses,
        so a loaded segment and a computed one land in the same row.
        """

        kwargs = self._pool_prefill_kwargs()
        if not kwargs:
            raise RuntimeError(
                f"MLA layer {self.layer_idx} was asked to materialize a "
                "segment outside a prefix-pool wave"
            )
        return self.module.mla_materialize_pool_segment(
            latent,
            pool=kwargs["pool"],
            pool_layer=kwargs["pool_layer"],
            slots=slots,
        )

    def _offload_prepacked_kv(self, offload_kv, cu_seqlens):
        """Offload compressed MLA KV per-sequence to host paged KV."""
        if AttnWrapperBase.prefill_pool_batch is not None:
            self._offload_prefill_pool_rows(offload_kv)
            return
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
            task = view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=[global_sequence_ids[seq_idx]],
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )
            # The offload is a std::async thread issuing cudaMemcpyAsync on a
            # d2h stream. Dropping the future is fire-and-forget: decode's
            # host->GPU load (a different stream, a different thread) then
            # reads pinned host pages the memcpy has not written yet -- and
            # pinned memory is not zeroed. Register the task so the worker's
            # end-of-prefill retire waits on it, and pin the source so the
            # allocator cannot reuse its storage mid-copy (GLM-5 pattern).
            AttnWrapperBase.pin_prefill_offload_tensor(seq_kv, self.layer_idx)
            AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)
            # Complete the copy before issuing the next one. Tracking alone
            # left a residual: with one std::async thread per sequence per MLA
            # layer (768/rank) the worker view's page table is read on the task
            # thread with no synchronisation, and the busiest TP ranks still
            # came back with partially written physical-layer-1 pages (probed
            # at FlashMLA's input: non-finite prompt K on ranks 0-1 only,
            # ranks 2-7 clean, surviving a full barrier around the load).
            # Waiting here is correct by construction and measured no slower
            # (38K-token prefill 15.0 s vs 16.5 s async): the D2H volume is
            # ~75 MB per MLA layer per rank at exact-64K, ~0.1 s in total.
            task.wait()
            start_idx = end_idx

    def _offload_prefill_pool_rows(self, offload_kv, index_keys=None):
        """Offload each prefix-pool row's latent at its own Host start.

        A pooled segment row goes to its segment's pseudo sequence at 0, a tail
        row to its prompt's Host pages at the tail's start — the row's own
        activations, never the pool, so the pool's pages are free again as soon
        as the chunk ends. Every rank of a serve group offloads, exactly as the
        packed path above does: the group replicates the rows and writes the
        same bytes to the same mirrored pages.

        ``index_keys`` are this chunk's bf16 DSA index keys (K3.1), written to
        the AUXILIARY region at the same row starts and token counts: the two
        regions must hold the same tokens at the same positions, because one
        page table addresses both and a prefix is only reusable when every
        group holds it. The aux region is page-split FP8, so the quantization
        happens here, at the offload boundary, exactly as the packed
        (non-pooled) writer does -- the pool's own indexer row stays bf16 for
        this layer's scoring.
        """
        view = self.core_engine.host_paged_kv_worker_view
        aux_view = None
        if index_keys is not None:
            from .dsa_prefill import quantize_index_keys

            aux_view = AttnWrapperBase.host_paged_kv_worker_view_aux
            if aux_view is None:
                raise RuntimeError(
                    f"Kimi-K3.1 DSA layer {self.layer_idx}: no auxiliary Host "
                    "KV worker view is bound, so this pooled chunk's index "
                    "keys would be dropped and every decode step would score "
                    "against stale or zeroed keys"
                )
            if not getattr(aux_view, "is_page_split", False):
                raise RuntimeError(
                    f"Kimi-K3.1 DSA layer {self.layer_idx}: the auxiliary Host "
                    "region is not page-split FP8; K3.1 writes e4m3 keys + "
                    "fp32 scales for the ReLU-gated paged scorer, so the "
                    "kimi_k31_indexer profile must be the one allocated"
                )
            if index_keys.shape[0] != offload_kv.shape[0]:
                raise RuntimeError(
                    f"Kimi-K3.1 DSA layer {self.layer_idx}: "
                    f"{index_keys.shape[0]} index key(s) against "
                    f"{offload_kv.shape[0]} latent token(s) in this pooled "
                    "chunk"
                )
        bounds = AttnWrapperBase.prefill_pool_batch.row_bounds
        starts = AttnWrapperBase.prefill_pool_row_starts
        sequence_ids = list(AttnWrapperBase.cur_batch or [])
        if len(bounds) - 1 != len(sequence_ids) or len(starts) != len(sequence_ids):
            raise RuntimeError("Kimi-K3 prefix-pool row metadata size mismatch")
        # Batch-level batchgen_debug.prefix_pool_offload_check. None unless the
        # worker installed a recorder, and the check precedes every copy and
        # hash below, so the flag being off costs one attribute read per layer
        # per chunk and nothing else.
        recorder = AttnWrapperBase.prefill_pool_offload_recorder
        items = AttnWrapperBase.prefill_pool_batch.items if recorder is not None else ()
        for row, (start, end) in enumerate(zip(bounds, bounds[1:])):
            row_kv = offload_kv[start:end].unsqueeze(0).unsqueeze(2)
            if recorder is not None:
                # Digested before the task is issued, so the recorded bytes are
                # exactly the bytes the D2H is about to read.
                recorder.record_row(
                    view=view,
                    layer_idx=self.layer_idx,
                    row=row,
                    kind=items[row].kind,
                    host_sid=sequence_ids[row],
                    dest_start_tokens=starts[row],
                    token_count=end - start,
                    source=row_kv,
                )
            task = view.async_offload_layer_kv_range_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=[sequence_ids[row]],
                k_tensor=row_kv,
                v_tensor=None,
                raw_start_positions=[starts[row]],
                token_counts=[end - start],
            )
            # Same contract as the packed path: pin the source, register the
            # task, and wait before issuing the next row.
            AttnWrapperBase.pin_prefill_offload_tensor(row_kv, self.layer_idx)
            AttnWrapperBase.track_prefill_offload_task(task, self.layer_idx)
            task.wait()
            if index_keys is None:
                continue
            keys_fp8, scales = quantize_index_keys(index_keys[start:end])
            aux_task = aux_view.async_offload_layer_kv_range_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=[sequence_ids[row]],
                k_tensor=keys_fp8.unsqueeze(0),
                v_tensor=None,
                raw_start_positions=[starts[row]],
                token_counts=[end - start],
                scale_tensor=scales.unsqueeze(0),
            )
            AttnWrapperBase.pin_prefill_offload_tensor(keys_fp8, self.layer_idx)
            AttnWrapperBase.pin_prefill_offload_tensor(scales, self.layer_idx)
            AttnWrapperBase.track_prefill_offload_task(aux_task, self.layer_idx)
            aux_task.wait()

    def _forward_decode(self, hidden_states, **kwargs):
        if getattr(self.module, "indexer", None) is not None:
            return self._forward_decode_dsa(hidden_states)
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

    def _forward_decode_dsa(self, hidden_states):
        """Kimi-K3.1 DSA decode (``dsa_decode.dsa_decode_forward``).

        Host appends mirror the dense path for the latent and add the
        page-split index key + its scale for the auxiliary region, so both
        Host regions keep every decoded token. The dense path's legacy
        ``BATCHGEN_SKIP_KV_CALLBACK`` debug switch is deliberately not read
        here (no new env guards; the decode graph ignores it too): a DSA
        layer always appends both regions.
        """
        from .dsa_decode import dsa_decode_forward

        result = dsa_decode_forward(
            self.module,
            hidden_states,
            AttnWrapperBase.position_ids,
            AttnWrapperBase.cache_seqlens,
            AttnWrapperBase.max_seqlen,
            AttnWrapperBase.gpu_paged_kv_manager,
            AttnWrapperBase.gpu_paged_kv_manager_aux,
            self.layer_idx,
            list(AttnWrapperBase.cur_batch or []),
        )
        if result.k_tensor.shape[0] > 0:
            if AttnWrapperBase.kv_append_callback is not None:
                aux_callback = getattr(AttnWrapperBase, "kv_append_callback_aux", None)
                if aux_callback is None:
                    # Appending the latent but not the index key would leave
                    # the Host aux region without this token; a later evict +
                    # reload would then score it against stale or zeroed bytes.
                    raise RuntimeError(
                        f"Kimi-K3.1 DSA layer {self.layer_idx}: a latent Host "
                        "append callback is bound but no auxiliary one is"
                    )
                AttnWrapperBase.kv_append_callback(
                    self.layer_idx, result.k_tensor, None)
                aux_callback(
                    self.layer_idx, result.index_k_fp8, None, result.index_scales)
        return result.attn_output
