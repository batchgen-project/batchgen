# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear Parallel Strategy Manager (PSM).

BF16-only design (simpler than K2.5):
  - One model build serves both phases; wrappers route prefill/decode.
  - MoE prefill: pure DP — all 256 BF16 experts host-offloaded and streamed
    per rank by the copy engine (no collectives, no INT4/marlin).
  - MoE decode (decode_moe_mode="resident_ep", M4 P0.3): each rank holds its
    EP-8 shard resident (32 experts x 26 MoE layers, stacked BF16, ~11.8 GB)
    and runs all_gather -> fused_moe_bf16 -> comm.all_reduce(SUM) per layer
    (batchgen.moe.fused_moe_bf16_resident seam). "streamed" falls back to
    the prefill-style streaming path.
  - Attention: NoPE-MLA layers use paged KV + FlashMLA decode; KDA layers
    use pooled conv/recurrent state (KimiLinearKDAWrapper pools).
  - Decode CUDA graphs (decode_graph_mode="graph"|"compare", M5.2 Phase A):
    per-layer attention spans plus the K3 resident-MXFP4 MoE segments are
    captured/replayed by KimiLinearDecodeGraph (cuda_graph_segments.py). A
    startup prewarm captures the first admitted bucket before decode;
    "eager" (default) installs the adapter but replays nothing, so
    batchgen_debug can switch modes on a live server; "off" installs nothing.

PSM <-> worker contract (methods, AttnWrapperBase class attrs the worker writes
each step, injected module attributes, comm handoff, weight-storage sharing,
KDA state pools) is documented in:
    batchgen-context/architecture/PSM_WORKER_CONTRACT.md
Read that first when modifying this file or batchgen_worker.py.
"""

import logging
import os
import time
import types

import torch

from batchgen.models.wrappers import AttnWrapperBase

from .block_residual import (
    BlockResidualCarrier,
    decoder_layer_forward_block_residual,
    make_output_block_residual_pre_hook,
)
from .config import require_num_routed_experts
from .serving_modules import (
    fuse_kda_decode_projections,
    kda_decode_serving,
    kda_prefill_serving,
    mla_decoding_nope_with_pagekv,
    mla_prefill_nope,
    mla_prefill_nope_prepacked,
    moe_forward_serving,
)
from .tp_weight_sharding import (
    shard_mla_tensor,
    shard_shared_expert_tensor,
)
from .wrappers import (
    KimiLinearAttnWrapper,
    KimiLinearExpertWrapper,
    KimiLinearKDAWrapper,
)


def _replace_param(root_module, dotted_name, tensor):
    """Replace a leaf nn.Parameter by object (materializes meta params).

    `param.data = tensor` (set_data) cannot convert a meta parameter to a
    concrete tensor, so we swap the Parameter object instead.
    """
    *parent, leaf = dotted_name.split(".")
    mod = root_module.get_submodule(".".join(parent)) if parent else root_module
    mod._parameters[leaf] = torch.nn.Parameter(tensor.detach(), requires_grad=False)


class KimiLinearParallelStrategyManager:
    """Parallel Strategy Manager for Kimi-Linear (KDA + NoPE-MLA hybrid)."""

    def __init__(
        self,
        loaded_model_config,
        engine_config,
        model_config,
        core_engine,
        skeleton_state_dict,
        rank,
        global_rank,
        world_size,
    ):
        self.loaded_model_config = loaded_model_config
        self.engine_config = engine_config
        self.model_config = model_config
        self.core_engine = core_engine
        self.skeleton_state_dict = skeleton_state_dict
        self.rank = rank
        self.global_rank = global_rank
        self.world_size = world_size

        self.model = None
        self.weight_copy_task = {}
        self.state_dict_name_map = {}

        # K3 vs 48B. The ONLY thing this decides in the PSM is the skeleton key
        # translation (`_skeleton_ckpt_key`); it is the same discriminator the
        # rest of the family uses (kimi_initializer.is_k3,
        # kimi_parameter_server._detect_kimi_family).
        self._is_k3 = getattr(loaded_model_config, "model_type", None) == "kimi_k3"
        self._distributed_weight_sharded = bool(
            getattr(
                engine_config.Basic_Config,
                "distributed_weight_sharded",
                False,
            )
        )
        if self._distributed_weight_sharded and (
            not self._is_k3 or world_size not in (16, 32)
        ):
            raise ValueError(
                "distributed host weights require K3 with world_size=16 "
                "(2 nodes) or world_size=32 (4 nodes)"
            )
        # Weight transport (initializer-set from the distributed store config,
        # no env var). "host_rdma" is the validated default in which every rank
        # pulls its own 112-expert shard; "hierarchical_gdr" pulls on eight
        # source ranks and replicates GPU-to-GPU across nodes.
        self._distributed_weight_transport = str(
            getattr(
                engine_config.Basic_Config,
                "distributed_weight_transport",
                "host_rdma",
            )
        )
        if self._distributed_weight_transport not in (
            "host_rdma",
            "hierarchical_gdr",
        ):
            raise ValueError(
                "distributed_weight_transport must be 'host_rdma' or "
                f"'hierarchical_gdr', got "
                f"{self._distributed_weight_transport!r}"
            )
        self._hierarchical_gdr = (
            self._distributed_weight_transport == "hierarchical_gdr"
        )

        # `model_config` is a ModelConfig, which has NO `n_routed_experts`
        # field at all — the old `getattr(..., 256) or 256` here therefore
        # evaluated to 256 unconditionally, for every model. Harmless on the
        # 48B (which really has 256) and a silent 3.5x undercount on K3's 896.
        self.num_experts = require_num_routed_experts(loaded_model_config)
        assert self.num_experts % world_size == 0, (
            f"num_experts {self.num_experts} not divisible by world_size {world_size}"
        )
        self.experts_per_rank = self.num_experts // world_size
        self.local_expert_start = self.global_rank * self.experts_per_rank

        self._kda_pool_slots = int(getattr(
            engine_config.GPU_Buffer_Config, "kda_state_slots", 256
        ))
        # M-PR-6: stream attn / kda_attn / shared_expert too (planner-set,
        # default False so the validated 48B path stays fully resident).
        self._stream_all_modules = bool(getattr(
            engine_config.Basic_Config, "stream_all_modules", False
        ))

        # M2a: head-parallel TP for KDA. G=1 is the validated single-shard
        # path (every derived value collapses to "all heads on this rank", so
        # every KDA seam below is byte-identical to before). G>1 slices
        # kda_num_heads across G contiguous ranks (one attn_tp sub-group per
        # block of G ranks); this rank owns heads [rank%G * Hl : +Hl].
        G = int(getattr(engine_config.Basic_Config, "attention_group_size", 1))
        kda_num_heads = int(loaded_model_config.kda_num_heads)
        assert kda_num_heads % G == 0, (
            f"kda_num_heads {kda_num_heads} not divisible by "
            f"attention_group_size {G}"
        )
        assert world_size % G == 0, (
            f"world_size {world_size} not divisible by "
            f"attention_group_size {G}"
        )
        self._attn_tp_size = G
        self._attn_tp_rank = global_rank % G
        self._attn_tp_group_id = global_rank // G
        self._attn_tp_hl = kda_num_heads // G
        self._attn_tp_head_dim = int(loaded_model_config.kda_head_dim)
        self._attn_tp_group = None  # torch NCCL sub-group, built at configure
        # Weight assembly has its own communicator.  The streamed-SP8
        # prefetcher launches the next layer's six all-gathers from a host
        # thread while the main thread runs the next layer's attention.  A
        # separate process group prevents those collectives from being
        # ordered against attention's all-reduces when ranks reach the
        # prefetch point at slightly different times.
        self._weight_tp_group = None
        # Hierarchical GDR adds a third communicator family: eight cross-node
        # groups, one per local TP slot, each carrying that slot's 112-expert
        # shard from its single source rank to the remaining nodes.
        self._cross_weight_group = None
        self._cross_weight_root = None
        self._cross_weight_source = False
        if self._hierarchical_gdr and not (
            self._distributed_weight_sharded
            and world_size in (16, 32)
            and self._attn_tp_size == 8
        ):
            raise ValueError(
                "hierarchical_gdr weight transport is defined only for the "
                "distributed K3 TP8/world16 or TP8/world32 topology; got "
                f"distributed_weight_sharded={self._distributed_weight_sharded}, "
                f"world_size={world_size}, "
                f"attention_group_size={self._attn_tp_size}"
            )
        if self._hierarchical_gdr:
            self._require_hierarchical_gdr_runtime()
        if self._stream_all_modules and self._attn_tp_size > 1:
            # Streamed KDA feeds full-96-head tensors from the copy-engine
            # ring, which _load_kda_modules never sees, so the head slice
            # below cannot run — the DP x TP token-flow + streamed-shard seam
            # is M2b (core), out of M2a scope. Fail by name here rather than
            # crash on a shape mismatch 93 layers into decode.
            raise NotImplementedError(
                "attention_group_size>1 with stream_all_modules is not wired "
                "(M2a shards the RESIDENT KDA load path; streamed-KDA head "
                "sharding is M2b). Run head-parallel KDA with "
                "stream_all_modules off."
            )
        self._comm = None
        self._resident_ep_built = False
        self._streamed_sp8_buffer = None
        self._prefill_moe_mode = (
            "streamed_sp8" if self._distributed_weight_sharded else "streamed"
        )
        self._decode_graph = None

    # ------------------------------------------------------------------ #
    #  Phase configuration                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_hierarchical_gdr_runtime():
        """Fail closed unless implicit NCCL launch ordering is supported."""
        if os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") != "1":
            raise RuntimeError(
                "hierarchical_gdr requires NCCL_LAUNCH_ORDER_IMPLICIT=1 "
                "before process-group initialization so cross-node weight "
                "broadcasts can overlap TP8 collectives without a "
                "cross-communicator launch-order deadlock"
            )

        nccl_version = torch.cuda.nccl.version()
        if not isinstance(nccl_version, tuple) or len(nccl_version) < 2:
            raise RuntimeError(
                "hierarchical_gdr could not verify the NCCL runtime version; "
                f"torch.cuda.nccl.version() returned {nccl_version!r}"
            )
        nccl_major_minor = tuple(int(value) for value in nccl_version[:2])
        if nccl_major_minor < (2, 26):
            raise RuntimeError(
                "hierarchical_gdr requires NCCL >=2.26 for implicit launch "
                f"ordering, got {nccl_version!r}"
            )

        cuda_version = torch.version.cuda
        try:
            cuda_major_minor = tuple(
                int(value) for value in cuda_version.split(".")[:2]
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "hierarchical_gdr could not verify torch.version.cuda; "
                f"got {cuda_version!r}"
            ) from exc
        if len(cuda_major_minor) < 2 or cuda_major_minor < (12, 3):
            raise RuntimeError(
                "hierarchical_gdr requires CUDA >=12.3 for implicit launch "
                f"ordering, got {cuda_version!r}"
            )

    def set_comm(self, comm):
        """Receive the BatchGen NCCL communicator from the worker. Consumed by
        the resident-EP decode MoE (all_gather + all_reduce per MoE layer);
        the streamed prefill path uses no collectives."""
        self._comm = comm

    def set_prefill_moe_mode(self, mode):
        """Select the K3 prefill MoE control for the next active batch.

        ``streamed`` is the legacy replicated pure-DP path. ``resident_ep`` is
        the R5 control. ``streamed_sp8`` keeps TP8 attention but scatters token
        rows locally across the eight GPUs, runs one grouped all-expert MoE per
        layer, and gathers rows only inside the node.
        """
        value = str(mode or "streamed").strip().lower()
        if value not in {"streamed", "resident_ep", "streamed_sp8"}:
            raise ValueError(
                "k3_prefill_moe_mode must be 'streamed', 'resident_ep' or "
                "'streamed_sp8', "
                f"got {mode!r}"
            )
        if value in {"resident_ep", "streamed_sp8"} and not self._is_k3:
            raise ValueError(
                f"{value} prefill is implemented only for Kimi-K3"
            )
        if (
            getattr(self, "_distributed_weight_sharded", False)
            and value != "streamed_sp8"
        ):
            raise ValueError(
                "distributed K3 host weights require k3_prefill_moe_mode="
                "'streamed_sp8' (the legacy replicated path would request "
                "experts outside this worker's ingress shard)"
            )
        self._prefill_moe_mode = value

    def default_prefill_moe_mode(self):
        """Return the safe batch default for this weight-source topology."""
        return "streamed_sp8" if self._distributed_weight_sharded else "streamed"

    def prefill_uses_resident_ep(self):
        return self._prefill_moe_mode == "resident_ep"

    def prefill_uses_streamed_sp8(self):
        return self._prefill_moe_mode == "streamed_sp8"

    def streamed_sp8_reseeds_h2d_on_reentry(self):
        """Whether each streamed-SP8 prefill starts a fresh local H2D cycle.

        Hierarchical GDR sources read their owned shard directly from the local
        compact store; the other ranks have an empty host-copy task.  No rank
        acquires a remote daemon generation, so carrying a partially filled GPU
        ring across resident decode is unnecessary and can strand a full-batch
        consumer behind stale leases.  Host-RDMA still depends on the daemon's
        acquire/release generations and must preserve its cursor.
        """
        return self.prefill_uses_streamed_sp8() and self._hierarchical_gdr

    def streamed_sp8_requires_global_pass_alignment(self):
        """Whether every node must join each streamed prefill model pass."""
        return self.prefill_uses_streamed_sp8() and self._hierarchical_gdr

    def run_streamed_sp8_transport_only_prefill(self, num_passes):
        """Join hierarchical weight broadcasts for passes with no local rows."""
        num_passes = int(num_passes)
        if num_passes < 0:
            raise ValueError("streamed-SP8 transport-only pass count must be >= 0")
        if num_passes == 0:
            return
        if not self.streamed_sp8_requires_global_pass_alignment():
            raise RuntimeError(
                "streamed-SP8 transport-only prefill is only valid for "
                "hierarchical GDR"
            )
        if self._streamed_sp8_buffer is None:
            raise RuntimeError(
                "streamed-SP8 transport-only prefill requires an initialized "
                "layer buffer"
            )
        for _ in range(num_passes):
            self._streamed_sp8_buffer.participate_empty_prefill_pass()

    def prefill_sequence_limits(self):
        """Return the persistent KDA-state capacity available to prefill.

        The prefill token cap bounds temporary activation/scratch work. KDA
        state is different: one slot is retained for each sequence until it
        completes or is evicted, so the scheduler must not admit more local
        sequences than the fixed GPU pool can retain. With TP8 KDA, each
        sequence is replicated across the eight ranks in one node-local
        attention group; expose the free-slot count as a node limit. The
        manager is absent before the first model build, so the configured
        capacity is the available capacity at initial admission.
        """
        available = self._kda_pool_slots
        state_manager = getattr(KimiLinearKDAWrapper, "state_manager", None)
        if state_manager is not None:
            available = min(
                available,
                state_manager.get_stats().num_free_state_items,
            )
        available = max(0, int(available))
        if self._attn_tp_size > 1:
            return {"max_sequences_per_node": available}
        return {"max_sequences_per_rank": available}

    def _set_prefill_memory_tiling(self, enabled):
        """Bound K3 prefill temporaries for resident-EP and streamed-SP8."""
        if self.model is None:
            return
        tile = 512 if enabled else None
        for module in self.model.modules():
            if hasattr(module, "_resident_prefill_token_tile"):
                module._resident_prefill_token_tile = tile
            if hasattr(module, "_resident_prefill_segment_tokens"):
                module._resident_prefill_segment_tokens = (
                    # The resident decode shards and streamed-SP8 buffers are
                    # co-resident during prefill. A 4K KDA segment releases
                    # another ~1.1 GiB of scratch versus 8K, enough for the
                    # next layer's normalized q/k rows at exact 64K.
                    4096 if enabled else None
                )
        for layer in self.model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            dense = getattr(layer, "mlp", None)
            if dense is not None:
                dense._resident_prefill_token_tile = tile
            shared = getattr(moe, "shared_experts", None) if moe is not None else None
            if shared is not None:
                shared._resident_prefill_token_tile = tile

    def _set_resident_ep_prefill_enabled(self, enabled):
        """Route K3 MoE through resident EP in prefill and compact its scratch."""
        if not enabled:
            from batchgen.moe.fused_moe_mxfp4_resident import (
                ResidentEPMXFP4MoELayer,
            )
            ResidentEPMXFP4MoELayer.release_prefill_output()
        self._set_prefill_memory_tiling(enabled)
        if self.model is None:
            return
        for layer in self.model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is not None:
                moe._resident_ep_prefill_enabled = bool(enabled)
                resident = getattr(moe, "_resident_ep_moe", None)
                if resident is not None:
                    resident.compact_dispatch = bool(enabled)

    def _release_resident_ep_decode(self):
        """Drop phase-inactive resident decode experts before streamed prefill.

        A world16 K3 rank's resident MXFP4 EP shard occupies about 84 GiB. It
        is never read by streamed-SP8 prefill, whose experts arrive through
        the layer ring, so retaining both copies forces exact-64K prompts into
        one-sequence model passes. Decode graphs bake the resident shard's
        addresses and must be released first; configure_decoding rebuilds the
        shard and reinstalls/captures graphs before the next decode forward.
        """
        if not self._resident_ep_built:
            return 0

        if self._decode_graph is not None:
            self._decode_graph.release()
            self._decode_graph = None

        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )

        ResidentEPMXFP4MoELayer.release_prefill_output()
        released_bytes = 0
        released_layers = 0
        for layer in self.model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            resident = (
                getattr(moe, "_resident_ep_moe", None)
                if moe is not None
                else None
            )
            if resident is None:
                continue
            shard = getattr(resident, "shard", None)
            if shard is not None and hasattr(shard, "nbytes"):
                released_bytes += int(shard.nbytes())
            moe._resident_ep_moe = None
            released_layers += 1

        self._resident_ep_built = False
        torch.cuda.empty_cache()
        if self.rank == 0:
            logging.info(
                "[K3_PREFILL_MOE] released resident decode shards: "
                "%d layers, %.2f GiB",
                released_layers,
                released_bytes / (1024 ** 3),
            )
        return released_bytes

    def _set_streamed_sp8_prefill_enabled(self, enabled):
        self._set_prefill_memory_tiling(enabled)
        if self.model is None:
            return
        for layer in self.model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is not None:
                moe._streamed_sp8_prefill_enabled = bool(enabled)

    def _streamed_sp8_attention_modules(self):
        """The UNDERLYING attention module of every layer.

        ``self_attn`` is the streaming wrapper by the time prefill is
        configured, but the serving methods that run the TP all-reduce are
        installed on the module it wraps, so the launch-order callback the
        streamed-SP8 schedule needs has to live there too.
        """
        if self.model is None:
            return
        for layer in self.model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            yield getattr(attn, "module", attn)

    def _release_streamed_sp8_prefill(self):
        try:
            if self._streamed_sp8_buffer is not None:
                self._streamed_sp8_buffer.close()
        finally:
            if self.model is not None:
                for layer in self.model.model.layers:
                    moe = getattr(layer, "block_sparse_moe", None)
                    dense = getattr(layer, "mlp", None)
                    if dense is not None:
                        for name in (
                            "_streamed_sp8_row_group",
                            "_streamed_sp8_profiler",
                        ):
                            if hasattr(dense, name):
                                delattr(dense, name)
                    for norm in (
                        getattr(layer, "self_attention_res_norm", None),
                        getattr(layer, "mlp_res_norm", None),
                    ):
                        if norm is None:
                            continue
                        for name in (
                            "_streamed_sp8_row_group",
                            "_streamed_sp8_order_wait",
                            "_streamed_sp8_profiler",
                            "_streamed_sp8_profile_name",
                            "_streamed_sp8_keep_sharded",
                        ):
                            if hasattr(norm, name):
                                delattr(norm, name)
                    if moe is not None:
                        moe._streamed_sp8_prefill_enabled = False
                        moe._streamed_sp8_moe = None
                        for name in (
                            "_streamed_sp8_sharded_carry",
                            "_streamed_sp8_global_rows",
                        ):
                            if hasattr(moe, name):
                                delattr(moe, name)
                        shared = getattr(moe, "shared_experts", None)
                        if (
                            shared is not None
                            and hasattr(shared, "_streamed_sp8_profiler")
                        ):
                            del shared._streamed_sp8_profiler
                output_norm = getattr(
                    self.model.model, "output_attn_res_norm", None
                )
                if output_norm is not None:
                    for name in (
                        "_streamed_sp8_row_group",
                        "_streamed_sp8_order_wait",
                        "_streamed_sp8_profiler",
                        "_streamed_sp8_profile_name",
                    ):
                        if hasattr(output_norm, name):
                            delattr(output_norm, name)
            # Drop the callback with the buffer it closes over: decode reaches
            # the SAME ``_reduce_mla_tp_output`` helper, and a stale reference
            # there would park its all-reduce on a torn-down handshake. This
            # cleanup is required even when close() surfaces an ingress error.
            for module in self._streamed_sp8_attention_modules():
                for name in (
                    "_streamed_sp8_order_wait",
                    "_streamed_sp8_profiler",
                    "_streamed_sp8_output_row_shard",
                ):
                    if hasattr(module, name):
                        delattr(module, name)
            self._streamed_sp8_buffer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare_resident_ep_prefill_output(self, num_global):
        """Reserve the reusable global FP32 combine output before expert HBM."""
        if not self.prefill_uses_resident_ep():
            return
        hidden_size = self.loaded_model_config.routed_expert_hidden_size
        if hidden_size is None:
            raise RuntimeError(
                "resident-EP prefill output requires LatentMoE hidden size"
            )
        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )
        output = ResidentEPMXFP4MoELayer.prepare_prefill_output(
            num_global,
            hidden_size,
            self.engine_config.Basic_Config.device_torch,
        )
        logging.info(
            "[K3_PREFILL_MOE] preallocated FP32 output shape=%s bytes=%s",
            tuple(output.shape),
            output.numel() * output.element_size(),
        )

    def _build_weight_copy_task(self):
        """Modules the copy engine must stream (host-offloaded), in layer-major
        ascending order.

        Each module TYPE has its own ring, so what matters is that the list for
        a type is in the order the consumer will request it — the producer
        drains its list front to front and the consumer blocks on the head. An
        out-of-order request finds no matching slot and dies on
        ``get_weights``' 2 s ``std::runtime_error``
        (GPU_Weight_Buffer.cpp:196-233); it does not hang and it does not
        return the wrong module.

        Consumer order, per forward pass:
          * one ``attn_{L}`` or ``kda_attn_{L}`` per layer, L ascending, at the
            top of the decoder layer;
          * one ``shared_expert_{L}`` per MoE layer, L ascending;
          * all ``routed_expert_{L}_{E}``, E ascending within L, L ascending —
            ``moe_forward_serving`` drives EVERY expert including 0-token ones
            precisely to keep this true.

        With ``stream_all_modules`` off (the 48B default) the first three lists
        stay empty: those modules are resident and never touch the ring.
        """
        cfg = self.loaded_model_config
        task = {"attn": [], "kda_attn": [], "shared_expert": [],
                "routed_expert": []}
        for layer_idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
            if self._stream_all_modules:
                if cfg.is_kda_layer(layer_idx):
                    task["kda_attn"].append(f"kda_attn_{layer_idx}")
                else:
                    task["attn"].append(f"attn_{layer_idx}")
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None or moe.experts is None:
                continue
            if self._stream_all_modules and getattr(
                moe, "shared_experts", None
            ) is not None:
                task["shared_expert"].append(f"shared_expert_{layer_idx}")
            if (
                self._hierarchical_gdr
                and self.prefill_uses_streamed_sp8()
                and not self._cross_weight_source
            ):
                # Hierarchical GDR: every rank that is not one of the eight
                # sources receives this layer's shard over the cross-node
                # broadcast, so it must request nothing at all from the host
                # store.
                continue
            for e_idx in range(len(moe.experts)):
                if self.prefill_uses_streamed_sp8():
                    start = self._attn_tp_rank * (
                        len(moe.experts) // self._attn_tp_size
                    )
                    end = start + len(moe.experts) // self._attn_tp_size
                    if not (start <= e_idx < end):
                        continue
                task["routed_expert"].append(
                    f"routed_expert_{layer_idx}_{e_idx}"
                )
        return task

    def configure_prefill(self):
        """Build the model and switch to streamed or resident-EP prefill."""
        self.loaded_model_config.phase = "prefill"
        self.loaded_model_config.ep_size = (
            self.world_size if self.prefill_uses_resident_ep() else 1
        )
        if self.model is None:
            self._build_model()
        AttnWrapperBase.phase = "prefill"
        KimiLinearExpertWrapper.phase = "prefill"
        # block_residual never outlives one decoder pass; drop anything a
        # previous (possibly aborted) pass left parked so the first prefill of
        # this phase starts from the seeded-at-layer-0 state.
        BlockResidualCarrier.reset()
        self.weight_copy_task = self._build_weight_copy_task()
        if self.prefill_uses_resident_ep():
            self._release_streamed_sp8_prefill()
            if self._stream_all_modules or self._attn_tp_size <= 1:
                raise RuntimeError(
                    "resident-EP prefill requires K3's resident TP attention "
                    "layout (attention_group_size>1, stream_all_modules=False)"
                )
            self._init_resident_ep_decode()
            self._set_resident_ep_prefill_enabled(True)
            self.weight_copy_task["routed_expert"] = []
            if self.rank == 0:
                logging.info(
                    "[K3_PREFILL_MOE] resident_ep enabled: rank-owned Marlin "
                    "experts, EP32 collectives, compact routed scratch"
                )
        elif self.prefill_uses_streamed_sp8():
            if self._stream_all_modules or self._attn_tp_size != 8:
                raise RuntimeError(
                    "streamed-SP8 prefill requires K3's resident TP8 attention "
                    "layout (attention_group_size=8, stream_all_modules=False)"
                )
            self._set_resident_ep_prefill_enabled(False)
            self._release_resident_ep_decode()
            self._init_streamed_sp8_prefill()
            self._set_streamed_sp8_prefill_enabled(True)
            if self.rank == 0:
                logging.info(
                    "[K3_PREFILL_MOE] streamed_sp8 enabled: transport=%s, "
                    "112 experts/shard, no TP8 weight all-gather, "
                    "node-local latent gather/FP32 reduce-scatter, "
                    "zero cross-node MoE activation collectives",
                    self._distributed_weight_transport,
                )
        else:
            self._release_streamed_sp8_prefill()
            self._set_resident_ep_prefill_enabled(False)
        return self.model, self.weight_copy_task

    def configure_decoding(self, padding_bsz=None, comm=None):
        """Switch to decode phase (model already built).

        decode_moe_mode="resident_ep" (default, set by the planner):
        materialize the stacked EP shard once and return an EMPTY
        routed_expert copy task — the worker then never starts the decode
        H2D streamer (batchgen_worker._load_decode_model gates on a
        non-empty routed_expert list). "streamed": legacy pure-DP streaming,
        identical to prefill.
        """
        if (
            self._distributed_weight_sharded
            and self._decode_moe_mode() != "resident_ep"
        ):
            raise RuntimeError(
                "distributed K3 host weights require resident-EP decode; "
                "streamed decode would replace the preserved streamed-SP8 "
                "prefill schedule"
            )
        if self._stream_all_modules:
            # Decode under stream_all_modules is NOT wired: the worker starts
            # the decode H2D streamer only when the routed_expert task is
            # non-empty (_load_decode_model), and decode_moe_mode="resident_ep"
            # empties it below — so nothing would ever refill the attn /
            # kda_attn / shared_expert rings and every persistent=False
            # wrapper would die on get_weights' 2 s throw. Fail here, loudly,
            # instead of at that timeout 93 layers deep.
            raise NotImplementedError(
                "stream_all_modules is a PREFILL-only path (M-PR-6). Decode "
                "needs a decode-phase H2D streamer for attn/kda_attn/"
                "shared_expert, which does not exist. Run prefill-only "
                "(max_tokens=1 still enters decode today — see PREFILL_PLAN "
                "C4) or turn the flag off."
            )
        self.loaded_model_config.phase = "decode"
        self._release_streamed_sp8_prefill()
        self._set_resident_ep_prefill_enabled(False)
        if comm is not None:
            self._comm = comm
        if self.model is None:
            self._build_model()
        AttnWrapperBase.phase = "decode"
        KimiLinearExpertWrapper.phase = "decode"
        BlockResidualCarrier.reset()
        self.weight_copy_task = self._build_weight_copy_task()
        if self._decode_moe_mode() == "resident_ep":
            self._init_resident_ep_decode()
            # Resident shards serve every routed expert: decode streams
            # nothing, so the copy engine gets no decode expert tasks.
            self.weight_copy_task["routed_expert"] = []
        self._init_decode_graph()
        return self.model, self.weight_copy_task

    def _decode_moe_mode(self):
        """Planner-set decode MoE execution mode (config-driven, no env vars):
        "resident_ep" (M4 P0.3 default) or "streamed" (fallback)."""
        return getattr(self.engine_config.EP_Config, "decode_moe_mode",
                       "resident_ep")

    def _decode_graph_mode(self):
        """Planner-set decode CUDA-graph mode (config-driven, no env vars):
        "eager" (default until M5.5), "graph" or "compare"."""
        return getattr(self.engine_config.Basic_Config, "decode_graph_mode",
                       "eager")

    def _init_decode_graph(self):
        """Install the Phase-A decode CUDA-graph adapter (M5.2), if asked for.

        Per-layer attention spans and, for K3 resident-MXFP4, the companion
        grouped-MoE segments are captured by the adapter. The worker invokes
        ``prewarm_decode_graphs`` after binding the real GPU KV manager so the
        first measured decode forward does not pay capture/setup time. See
        cuda_graph_segments.py for the capture-structure rationale.
        """
        mode = self._decode_graph_mode()
        if mode == "off":
            return
        # Install for "eager" too: the adapter then replays nothing (pure
        # pass-through to the wrapper path) but is present, so a batch-level
        # batchgen_debug.kimi_decode_graph_mode can switch graph/compare/eager
        # on a live server — the project's debug-flag policy (cf.
        # glm5_moe_mode), instead of a cold restart per experiment. Buckets
        # capture lazily, so eager mode costs no capture time or memory.
        # "off" is the rollback escape that installs nothing at all.
        if self._decode_graph is not None:
            self._decode_graph.set_mode(mode)
            return
        from .cuda_graph_segments import KimiLinearDecodeGraph

        basic = self.engine_config.Basic_Config
        self._decode_graph = KimiLinearDecodeGraph(
            self.model,
            self.loaded_model_config,
            device=basic.device_torch,
            buckets=getattr(basic, "decode_graph_buckets", None),
            mode=mode,
            compare_every=getattr(basic, "decode_graph_compare_every", None),
            rank=self.rank,
        )
        self._decode_graph.install()

    def prewarm_decode_graphs(self, gpu_manager=None):
        """Capture the first K3 decode bucket before measured decode starts.

        GPU KV allocation is intentionally worker-owned and happens after this
        manager's ``configure_decoding`` call.  The worker therefore passes
        the initialized manager here so the graph's MLA capture can bind its
        stable page-table/K-cache addresses without duplicating KV lifecycle
        logic in the PSM.
        """
        if self._decode_graph is None:
            return False
        if gpu_manager is not None:
            manager = getattr(gpu_manager, "primary", gpu_manager)
            AttnWrapperBase.gpu_paged_kv_manager = manager
        return self._decode_graph.prewarm_decode_graphs()

    def _init_resident_ep_decode(self):
        """Materialize the stacked EP-8 BF16 shards ONCE (idempotent) and
        attach a ResidentEPMoELayer to every MoE block.

        Source is the host copy-engine weight storage (core_engine.get_tensor,
        the exact tensors the streamed path consumes) — after this one-time
        H2D there is no per-step expert traffic in decode. HBM arithmetic for
        the ~84 GiB/rank K3 world16 shards lives at the allocation site
        (batchgen.moe.fused_moe_bf16_resident.build_layer_shard).
        """
        if self._resident_ep_built:
            return
        assert self._comm is not None, (
            "resident-EP decode needs the NCCL communicator (worker passes "
            "it via configure_decoding(comm=...) or set_comm)"
        )
        cfg = self.loaded_model_config
        device = self.engine_config.Basic_Config.device_torch
        from .k3.mxfp4_expert import is_mxfp4_quantized

        if is_mxfp4_quantized(cfg):
            # K3 MXFP4 LatentMoE (M3.1a): repack-once marlin shards + a resident
            # layer that runs the latent dataflow (down/norm/up seam). The BF16
            # stacked shard cannot represent it (hidden-space, no latent seam).
            from batchgen.moe.fused_moe_mxfp4_resident import (
                build_resident_ep_mxfp4_layers,
            )

            build_resident_ep_mxfp4_layers(
                self.model.model.layers,
                self.core_engine.get_tensor,
                self._comm,
                self.world_size,
                self.global_rank,
                self.local_expert_start,
                self.experts_per_rank,
                device,
            )
        else:
            from batchgen.moe.fused_moe_bf16_resident import (
                build_resident_ep_layers,
            )

            build_resident_ep_layers(
                self.model.model.layers,
                self.core_engine.get_tensor,
                self._comm,
                self.world_size,
                self.global_rank,
                self.local_expert_start,
                self.experts_per_rank,
                cfg.moe_intermediate_size,
                device,
            )
        self._resident_ep_built = True

    def _init_streamed_sp8_prefill(self):
        """Attach one reusable layer-wise all-expert buffer to every K3 MoE."""
        if self._streamed_sp8_buffer is not None:
            return
        cfg = self.loaded_model_config
        from batchgen.moe.streamed_sp8_mxfp4 import (
            StreamedSP8LayerBuffer,
            StreamedSP8MXFP4MoELayer,
        )

        expert_ring_depth = int(
            self.engine_config.GPU_Buffer_Config
            .num_prefill_module_buffer["routed_expert"]
        )
        if expert_ring_depth <= 0:
            raise RuntimeError(
                "streamed-SP8 prefill requires a positive routed-expert "
                "prefill ring depth"
            )
        self._streamed_sp8_buffer = StreamedSP8LayerBuffer(
            core_engine=self.core_engine,
            device=self.engine_config.Basic_Config.device_torch,
            tp_group=self._weight_tp_group,
            tp_rank=self._attn_tp_rank,
            tp_size=self._attn_tp_size,
            num_experts=cfg.n_routed_experts,
            intermediate_size=cfg.moe_intermediate_size,
            latent_size=cfg.routed_expert_hidden_size,
            acquire_batch_size=expert_ring_depth,
            layer_indices=[
                layer_idx
                for layer_idx, layer in enumerate(self.model.model.layers)
                if getattr(layer, "block_sparse_moe", None) is not None
                and getattr(layer.block_sparse_moe, "experts", None) is not None
            ],
            cross_group=self._cross_weight_group,
            cross_root=self._cross_weight_root,
            cross_source=self._cross_weight_source,
        )
        row_group = (
            self._attn_tp_size,
            self._attn_tp_rank,
            self._attn_tp_group,
        )
        order_wait = (
            self._streamed_sp8_buffer.order_tp_collective_after_cross_launch
        )
        for layer_idx, layer in enumerate(self.model.model.layers):
            moe = getattr(layer, "block_sparse_moe", None)
            streamed_moe = (
                moe is not None and getattr(moe, "experts", None) is not None
            )
            dense = getattr(layer, "mlp", None)
            if dense is not None:
                dense._streamed_sp8_row_group = row_group
                dense._streamed_sp8_profiler = StreamedSP8MXFP4MoELayer
            for norm, profile_name in (
                (
                    getattr(layer, "self_attention_res_norm", None),
                    "self_depth_mix",
                ),
                (getattr(layer, "mlp_res_norm", None), "mlp_depth_mix"),
            ):
                if norm is None:
                    continue
                norm._streamed_sp8_row_group = row_group
                norm._streamed_sp8_order_wait = order_wait
                norm._streamed_sp8_profiler = StreamedSP8MXFP4MoELayer
                norm._streamed_sp8_profile_name = profile_name
                if profile_name == "mlp_depth_mix" and streamed_moe:
                    norm._streamed_sp8_keep_sharded = True
            if not streamed_moe:
                continue
            moe._streamed_sp8_sharded_carry = True
            moe._streamed_sp8_moe = StreamedSP8MXFP4MoELayer(
                layer_idx=layer_idx,
                buffer=self._streamed_sp8_buffer,
                down_proj=moe.routed_expert_down_proj,
                norm=(
                    moe.routed_expert_norm
                    if getattr(moe, "latent_moe_use_norm", False)
                    else None
                ),
                up_proj=moe.routed_expert_up_proj,
                chunk_rows=int(getattr(
                    self.engine_config.Module_Batching_Config,
                    "k3_prefill_grouped_chunk_rows",
                    2_048,
                )),
                collective_stripe_threshold_rows=int(getattr(
                    self.engine_config.Module_Batching_Config,
                    "k3_prefill_collective_stripe_threshold_rows",
                    32_768,
                )),
            )
            shared = getattr(moe, "shared_experts", None)
            if shared is not None:
                shared._streamed_sp8_profiler = StreamedSP8MXFP4MoELayer
        output_norm = getattr(
            self.model.model, "output_attn_res_norm", None
        )
        if output_norm is not None:
            output_norm._streamed_sp8_row_group = row_group
            output_norm._streamed_sp8_order_wait = order_wait
            output_norm._streamed_sp8_profiler = StreamedSP8MXFP4MoELayer
            output_norm._streamed_sp8_profile_name = "output_depth_mix"
        # The cross-node broadcast gate opens at the end of each MoE serving
        # branch; whichever TP8 collective follows first (a depth-mix gather
        # or the next attention all-reduce) must observe every cross-node call
        # already host-enqueued.
        for module in self._streamed_sp8_attention_modules():
            module._streamed_sp8_order_wait = order_wait
            module._streamed_sp8_profiler = StreamedSP8MXFP4MoELayer
        for layer in self.model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None or getattr(moe, "experts", None) is None:
                continue
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn = getattr(attn, "module", attn)
                attn._streamed_sp8_output_row_shard = True

    def set_num_tokens_per_rank(self, num_tokens_per_rank):
        """Worker hook (duck-typed by _sync_decode_moe_rank_counts): per-step
        MAX decode rows across ranks. Defines the padded all_gather /
        all_reduce layout of the resident-EP decode MoE — every rank
        (including empty ones) sizes the global buffer from this scalar.

        M2b: under TP-G decode the worker passes the POST-scatter share
        ceil(B_grp/G) (each rank owns 1/G of the group's rows after
        moe_forward_resident_ep_decode's scatter), so this scalar stays the
        per-rank distinct-row count the DP-32 resident layer expects."""
        # Drive BOTH resident classes' per-rank layout scalar. Only one is
        # ever materialized per model (BF16 stacked hidden shard vs MXFP4
        # latent shard), but num_tokens_per_rank is a class attribute so
        # setting the unused one is a harmless no-op — and the MXFP4 EP
        # forward (_forward_ep) asserts it before the collectives.
        from batchgen.moe.fused_moe_bf16_resident import ResidentEPMoELayer
        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )

        ResidentEPMoELayer.set_num_tokens_per_rank(num_tokens_per_rank)
        ResidentEPMXFP4MoELayer.set_num_tokens_per_rank(num_tokens_per_rank)

    def set_rank_token_counts(self, rank_token_counts):
        """Publish whether every global decode rank has a live row.

        The K3 resident-MXFP4 MoE CUDA graph contains global EP collectives. Its
        graph/eager decision must therefore be identical on all ranks; the
        worker already has the synchronized count vector at the batch boundary,
        so pass it to the resident layer once instead of probing it from every
        forward.
        """
        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )
        ResidentEPMXFP4MoELayer.set_rank_token_counts(rank_token_counts)

    @property
    def attn_tp_size(self):
        """G — the head-parallel (TP-KDA) sub-group size. 1 == pure DP-32. The
        worker reads this (getattr, default 1) to size the decode MoE padding
        and to gate the decode DP-group assignment / KDA reshard (M2b)."""
        return self._attn_tp_size

    @staticmethod
    def scatter_rows(x, group_size, group_rank):
        """Intra-group decode-MoE row scatter (M2b). Pure local slice — the
        group's rows are replicated across its G ranks, so no collective is
        needed. See moe_tp_reshard for the contract."""
        from .moe_tp_reshard import scatter_rows

        return scatter_rows(x, group_size, group_rank)

    @staticmethod
    def all_gather_rows(routed_local, num_rows, group_size, group_rank, group):
        """Intra-group decode-MoE row gather (M2b): reassemble the full group
        batch on every rank from each rank's routed slice over attn_tp_group."""
        from .moe_tp_reshard import all_gather_rows

        return all_gather_rows(
            routed_local, num_rows, group_size, group_rank, group
        )

    # ------------------------------------------------------------------ #
    #  Model build                                                        #
    # ------------------------------------------------------------------ #

    def _build_model(self):
        from .model import KimiLinearForCausalLM

        start = time.perf_counter()
        cfg = self.loaded_model_config
        device = self.engine_config.Basic_Config.device_torch

        # Construct on META device: this allocates NO weight storage, so every
        # rank shares the single weight copy held in the parameter server's
        # shm (skeleton_state_dict) and its own GPU module tensors — we never
        # materialize a per-rank CPU copy of the 48B model (in particular the
        # 256 experts/layer are never allocated on the host).
        with torch.device("meta"):
            self.model = KimiLinearForCausalLM(cfg)
        logging.info(f"Rank {self.rank}: model skeleton constructed on meta "
                     f"({time.perf_counter() - start:.1f}s)")

        # 1. skeleton weights (embeddings, norms, gates, lm_head) — assigned
        #    from the shared shm storage (zero-copy across ranks).
        self._load_model_skeleton()

        # 2. attn/kda/shared module weights. Resident by default (they are
        #    small on the 48B); under stream_all_modules these are emptied
        #    instead and streamed from the ring like the routed experts.
        self._load_attn_modules()
        self._load_kda_modules()
        self._load_shared_expert_modules()

        # 2b. head-parallel KDA sub-group (M2a). Collective across ALL ranks;
        #     no-op when attention_group_size==1. Built before _config_kda_
        #     modules stamps it onto the KDA modules.
        self._build_attn_tp_group()

        # 3. serving method injection + wrappers
        self._config_attn_modules()
        self._config_kda_modules()
        self._config_expert_modules()   # wrap routed experts as streamed
        self._config_block_residual()   # K3 Block Attention Residuals
        self._config_lm_head_hook()
        self._wrap_logits_output()

        # 4. KDA state pools
        kda_indices = [
            i for i in range(cfg.num_hidden_layers) if cfg.is_kda_layer(i)
        ]
        # M2a: the pools hold this rank's LOCAL heads (Hl == kda_num_heads for
        # G==1). The conv/recurrent kernels only ever see the local shard.
        KimiLinearKDAWrapper.init_state_pools(
            kda_indices,
            num_slots=self._kda_pool_slots,
            num_heads=self._attn_tp_hl,
            head_dim=cfg.kda_head_dim,
            conv_width=cfg.kda_conv_size,
            proj_size=self._attn_tp_hl * cfg.kda_head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        self.model.eval()
        self.model.to(device)
        if self._is_k3 and torch.cuda.is_available():
            # The service wall starts at request admission, so neither Triton
            # compilation nor the 187 per-stage folded score vectors may be a
            # first-request lazy cost.  Do both before startup completes.
            from .attn_residual_triton import warmup_attn_residual_triton

            warmup_start = time.perf_counter()
            num_score_vectors = warmup_attn_residual_triton(self.model.model)
            if self.rank == 0:
                logging.info(
                    "K3 attention-residual Triton mixer prewarmed: %s score "
                    "vectors in %.1fs",
                    num_score_vectors,
                    time.perf_counter() - warmup_start,
                )
        if self.rank == 0:
            logging.info(
                f"Kimi-Linear model configured in {time.perf_counter() - start:.1f}s "
                f"(EP {self.world_size}, {self.experts_per_rank} experts/rank, "
                f"KDA slots {self._kda_pool_slots})"
            )

    def _skeleton_ckpt_key(self, model_param_name):
        """`model.named_parameters()` name -> the key the parameter server used.

        The C++ parameter server keys `skeleton_state_dict_` by the CHECKPOINT
        name (Parameter_Server.cpp:357-397), and every K3 text tensor carries a
        `language_model.` prefix that `model.named_parameters()` does not
        (k3/tensor_map.py: K3_CKPT_PREFIX). The 48B checkpoint has no prefix, so
        this is the identity there and that path is unchanged.

        Translated in this direction, and in this one place — the single
        skeleton lookup below — deliberately:

          * model-name -> ckpt-name is a total function (prepend a constant).
            The reverse is not: K3's `skeleton_state_dict_` ALSO holds the 168
            unprefixed `vision_tower.` / `mm_projector.` tensors (they are in
            neither the name map nor the skeleton declaration, so the C++ side
            promotes them), and a strip-on-ingest pass would have to guess which
            keys are text before it could rewrite them.
          * one lookup per parameter, never two. A "try prefixed, else bare"
            probe would trade a genuinely missing tensor for a silent wrong-name
            hit somewhere in the 1026-name space, which is exactly the class of
            bug the reconciler exists to prevent.

        Imported INSIDE the K3 branch, like every other `.k3` use in this
        package: `k3/__init__.py` states that nothing is exported eagerly, and a
        module-level import would drag `k3.tensor_map` -> `k3.mxfp4_layout` ->
        `weight_reconciler` into every 48B run that merely imports the PSM.
        """
        if not self._is_k3:
            return model_param_name
        from .k3.tensor_map import k3_skeleton_key

        return k3_skeleton_key(model_param_name)

    def _load_model_skeleton(self):
        """Load non-module weights from the checkpoint state dict."""
        device = self.engine_config.Basic_Config.device_torch
        # Loaded via core_engine instead: self_attn (attn/kda), experts (stacked),
        # shared experts. Everything else (embed, norms, router gate, attn_res,
        # lm_head) comes from the checkpoint dict here.
        skip_fragments = (".self_attn.", ".block_sparse_moe.experts.",
                          ".block_sparse_moe.shared_experts.")
        n_loaded = 0
        missing = []
        for key, param in self.model.named_parameters():
            if any(f in key for f in skip_fragments):
                continue
            ckpt_key = self._skeleton_ckpt_key(key)
            if ckpt_key in self.skeleton_state_dict:
                # Materialize on GPU from the shared shm tensor (native dtype).
                _replace_param(self.model, key,
                               self.skeleton_state_dict[ckpt_key].to(device))
                n_loaded += 1
            elif param.is_meta:
                missing.append((key, ckpt_key))
        if missing:
            # Read the prefix from the constant the lookup actually used —
            # a duplicated literal here would keep printing "language_model."
            # after K3_CKPT_PREFIX changed, i.e. the diagnostic would misname
            # the very key it failed to find (adversarial-review W1).
            prefix = "(none)"
            if self._is_k3:
                from .k3.tensor_map import K3_CKPT_PREFIX

                prefix = K3_CKPT_PREFIX
            raise RuntimeError(
                f"Kimi-Linear skeleton: {len(missing)} params not found in "
                f"weight storage (still on meta). Each was looked up under its "
                f"CHECKPOINT key (prefix {prefix}); there is no bare-name "
                f"fallback, because a second probe would trade a missing tensor "
                f"for a silently wrong one. First failures as "
                f"(param, checkpoint key): {missing[:8]}"
            )
        if self.rank == 0:
            logging.info(f"Skeleton: {n_loaded} tensors loaded from shared storage")

    def _clear_streamed_params(self, module):
        """Empty a module's params so meta is cleared and `model.to(device)` is
        safe; the copy engine fills them from the ring per forward.

        Same contract as `_config_expert_modules` — `p.data =` cannot
        materialize a meta parameter, so the Parameter object is swapped.
        """
        device = self.engine_config.Basic_Config.device_torch
        for name, _ in list(module.named_parameters()):
            _replace_param(module, name, torch.empty(0, device=device))

    def _load_attn_modules(self):
        device = self.engine_config.Basic_Config.device_torch
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            if cfg.is_kda_layer(layer_idx):
                continue
            attn = self.model.model.layers[layer_idx].self_attn
            if self._stream_all_modules:
                self._clear_streamed_params(attn)
                continue
            tensors = self.core_engine.get_tensor(f"attn_{layer_idx}")
            for name, p in list(attn.named_parameters()):
                if name in tensors:
                    tensor = tensors[name]
                    if self._attn_tp_size > 1:
                        tensor = shard_mla_tensor(
                            tensor,
                            name,
                            self._attn_tp_size,
                            self._attn_tp_rank,
                        )
                    _replace_param(attn, name, tensor.to(device=device))
                elif self.rank == 0:
                    logging.warning(f"attn_{layer_idx}: missing tensor {name}")
            if self._attn_tp_size > 1:
                local_heads = attn.num_heads // self._attn_tp_size
                attn.num_heads = local_heads
                attn.num_key_value_heads = local_heads
                attn.num_key_value_groups = 1
                if hasattr(attn, "q_b_proj"):
                    attn.q_b_proj.out_features = (
                        local_heads * attn.q_head_dim
                    )
                if hasattr(attn, "q_proj"):
                    attn.q_proj.out_features = (
                        local_heads * attn.q_head_dim
                    )
                attn.kv_b_proj.out_features = local_heads * (
                    attn.qk_nope_head_dim + attn.v_head_dim
                )
                if hasattr(attn, "g_proj"):
                    attn.g_proj.out_features = (
                        local_heads * attn.v_head_dim
                    )
                attn.o_proj.in_features = local_heads * attn.v_head_dim

    def _build_attn_tp_group(self):
        """Build the head-parallel (KDA TP) NCCL sub-groups (M2a).

        COLLECTIVE: every rank creates every group, in the same order, and
        keeps the one it belongs to as ``self._attn_tp_group``. No-op for
        G==1. The global PyNccl communicator (``self._comm``, EP-32) is
        untouched; this is a separate torch NCCL group, exactly like the
        worker's own ``dist.new_group`` usage.
        """
        if self._attn_tp_size <= 1:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "attention_group_size>1 needs torch.distributed initialized "
                "(head-parallel KDA builds NCCL sub-groups at configure time)."
            )
        G = self._attn_tp_size
        for g in range(self.world_size // G):
            grp = dist.new_group(ranks=list(range(g * G, (g + 1) * G)))
            if g == self._attn_tp_group_id:
                self._attn_tp_group = grp
        # Do not reuse the attention group for streamed weight assembly.  The
        # prefetch thread can otherwise enqueue a weight all-gather after one
        # rank has already entered the next attention all-reduce, which is an
        # invalid NCCL ordering even though both collectives are node-local.
        for g in range(self.world_size // G):
            grp = dist.new_group(ranks=list(range(g * G, (g + 1) * G)))
            if g == self._attn_tp_group_id:
                self._weight_tp_group = grp
        self._build_cross_weight_group()

    def _build_cross_weight_group(self):
        """Build the eight cross-node weight groups (hierarchical GDR only).

        COLLECTIVE, like the two node-local families above: every rank creates
        all eight groups in the same order and keeps the one it belongs to.
        They are created LAST so that the creation order of the attention and
        node-local weight groups is unchanged for host_rdma.

        Group ``g`` is ``[g + n*8 for n in range(num_nodes)]`` — local TP slot
        ``g`` on each node, i.e. exactly the ranks that need experts
        ``[112g, 112(g+1))``. Its source is local slot ``g`` on node
        ``(g * num_nodes) // 8``, so the eight sources spread evenly over the
        nodes (two per node on four nodes, four per node on two nodes) and
        every node drives an egress stream instead of one node fanning out all
        896 experts.
        """
        if not self._hierarchical_gdr:
            return
        import torch.distributed as dist

        G = self._attn_tp_size
        num_nodes = self.world_size // G
        for g in range(G):
            grp = dist.new_group(
                ranks=[g + node * G for node in range(num_nodes)]
            )
            if g == self._attn_tp_rank:
                self._cross_weight_group = grp
                self._cross_weight_root = ((g * num_nodes) // G) * G + g
        self._cross_weight_source = (
            self.global_rank == self._cross_weight_root
        )

    def _head_shard_kda_tensor(self, name, tensor):
        """Slice a KDA weight/param to this rank's head shard (M2a, G>1).

        head block = [rank%G * Hl : +Hl]; projection block = that * head_dim.
          * f_a_proj / g_a_proj / o_norm : REPLICATE (per-head over head_dim
            or a shared low-rank latent) -> no slice.
          * o_proj                       : COLS (dim1) -> row-parallel, summed
            by the all_reduce in serving_modules.
          * A_log                         : per-HEAD, sliced on the FLATTENED
            head axis (the 48B checkpoint ships it as (1, 1, H, 1) -- heads on
            axis 2, NOT axis 0 -- so a dim-0 slice empties every rank but rank 0
            into a null-pointer tensor that the fla gate kernel dereferences;
            G=8 prefill IMA, bug_log 2026-08-14).
          * b_proj                        : per-HEAD rows [lo:hi] ((H, hidden)).
          * everything else (q/k/v_proj, {q,k,v}_conv1d weight+bias, f_b_proj,
            g_proj, g_b_proj, dt_bias): per-(head*head_dim) rows [rlo:rhi].
        """
        hd = self._attn_tp_head_dim
        lo = self._attn_tp_rank * self._attn_tp_hl
        hi = lo + self._attn_tp_hl
        rlo, rhi = lo * hd, hi * hd
        base = name.split(".")[0]
        if base in ("f_a_proj", "g_a_proj", "o_norm"):
            return tensor
        if base == "o_proj":
            return tensor[:, rlo:rhi]
        if base == "A_log":
            # The fla gate kernel reads A_log FLAT as A_log[i_h] (one log-decay
            # scalar per head). Its stored shape is (1, 1, H, 1) on the 48B
            # checkpoint, so slice the flattened head axis -- a dim-0 slice
            # (tensor[lo:hi]) hits the size-1 leading axis and returns 0 rows.
            # A_log LAYOUT differs only in its checkpoint padding: 48B stores
            # one value per head (sometimes as (1,1,H,1)); K3 stores the same
            # per-head vector padded from 96 to 128 entries.  K3's padding is
            # not a per-head-dimension parameter: every rank must receive its
            # own slice of the first 96 values, or all TP ranks would consume
            # rank 0's decay constants.
            from .k3.tensor_map import K3_A_LOG_PADDED_LEN

            a = tensor.reshape(-1)
            n = self._attn_tp_hl * self._attn_tp_size
            if a.numel() == n or (
                getattr(self, "_is_k3", False)
                and a.numel() == K3_A_LOG_PADDED_LEN
                and n < a.numel()
            ):
                return a[lo:hi]
            raise ValueError(
                f"A_log has {a.numel()} elements (shape {tuple(tensor.shape)}); "
                f"expected kda_num_heads={n} or K3's padded length "
                f"{K3_A_LOG_PADDED_LEN}"
            )
        if base == "b_proj":
            return tensor[lo:hi]
        return tensor[rlo:rhi]

    def _load_kda_modules(self):
        device = self.engine_config.Basic_Config.device_torch
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            if not cfg.is_kda_layer(layer_idx):
                continue
            kda = self.model.model.layers[layer_idx].self_attn
            if self._stream_all_modules:
                self._clear_streamed_params(kda)
                continue
            tensors = self.core_engine.get_tensor(f"kda_attn_{layer_idx}")
            for name, p in list(kda.named_parameters()):
                if name in tensors:
                    t = tensors[name]
                    if self._attn_tp_size > 1:
                        t = self._head_shard_kda_tensor(name, t).contiguous()
                    _replace_param(kda, name, t.to(device=device))
                elif self.rank == 0:
                    logging.warning(f"kda_attn_{layer_idx}: missing tensor {name}")
            if self._attn_tp_size > 1:
                # After sharding, this module owns Hl heads: the serving math
                # (reshapes, conv/recurrent grids) reads num_heads/num_k_heads.
                kda.num_heads = kda.num_k_heads = self._attn_tp_hl

    def _load_shared_expert_modules(self):
        device = self.engine_config.Basic_Config.device_torch
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
            if not hasattr(layer, "block_sparse_moe"):
                continue
            shared = layer.block_sparse_moe.shared_experts
            if shared is None:
                continue
            if self._stream_all_modules:
                # Wrapped as a streamed expert in _config_expert_modules;
                # ExpertWrapperBase turns expert_idx=-1 into the module key
                # "shared_expert_{L}" the parameter server already serves.
                self._clear_streamed_params(shared)
                continue
            tensors = self.core_engine.get_tensor(f"shared_expert_{layer_idx}")
            for name, p in list(shared.named_parameters()):
                if name in tensors:
                    tensor = tensors[name]
                    if self._attn_tp_size > 1:
                        tensor = shard_shared_expert_tensor(
                            tensor,
                            name,
                            self._attn_tp_size,
                            self._attn_tp_rank,
                        )
                    _replace_param(shared, name, tensor.to(device=device))
                elif self.rank == 0:
                    logging.warning(f"shared_expert_{layer_idx}: missing {name}")
            if self._attn_tp_size > 1:
                local_intermediate = shared.gate_proj.weight.shape[0]
                shared.intermediate_size = local_intermediate
                shared.gate_proj.out_features = local_intermediate
                shared.up_proj.out_features = local_intermediate
                shared.down_proj.in_features = local_intermediate

    # ------------------------------------------------------------------ #
    #  Serving method injection                                           #
    # ------------------------------------------------------------------ #

    def _config_attn_modules(self):
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            if cfg.is_kda_layer(layer_idx):
                continue
            attn = self.model.model.layers[layer_idx].self_attn
            attn.mla_prefill_nope = types.MethodType(mla_prefill_nope, attn)
            attn.mla_prefill_nope_prepacked = types.MethodType(
                mla_prefill_nope_prepacked, attn
            )
            attn.mla_decoding_nope_with_pagekv = types.MethodType(
                mla_decoding_nope_with_pagekv, attn
            )
            attn.attn_tp_size = self._attn_tp_size
            attn.attn_tp_group = self._attn_tp_group
            self.model.model.layers[layer_idx].self_attn = KimiLinearAttnWrapper(
                attn, layer_idx, self.core_engine, self.engine_config,
                self.model_config, persistent=not self._stream_all_modules,
            )

    def _config_kda_modules(self):
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            if not cfg.is_kda_layer(layer_idx):
                continue
            kda = self.model.model.layers[layer_idx].self_attn
            kda.kda_prefill_serving = types.MethodType(kda_prefill_serving, kda)
            kda.kda_decode_serving = types.MethodType(kda_decode_serving, kda)
            if self._is_k3 and not self._stream_all_modules:
                # K3's full-rank KDA gate is eligible for the decode-only
                # projection fusion.  Prefill continues to use the original
                # calls; the rebinding shares their storage with the fused
                # buffer instead of duplicating HBM.
                fuse_kda_decode_projections(kda)
            # M2a: stamp the head-parallel context the serving methods read
            # (attn_tp_size==1 -> the o_proj all_reduce is skipped, unchanged).
            kda.attn_tp_size = self._attn_tp_size
            kda.attn_tp_rank = self._attn_tp_rank
            kda.attn_tp_group = self._attn_tp_group
            kda.Hl = self._attn_tp_hl
            self.model.model.layers[layer_idx].self_attn = KimiLinearKDAWrapper(
                kda, layer_idx, self.core_engine, self.engine_config,
                self.model_config, persistent=not self._stream_all_modules,
            )

    def _config_expert_modules(self):
        """Wrap routed experts as streamed (host-offloaded) BF16 experts and
        inject the pure-DP MoE forward. Shared experts stay resident.

        Routed-expert weights are NOT resident: KimiLinearExpertWrapper
        (persistent=False) streams each expert from the copy-engine buffer per
        forward. Expert module params are emptied here so meta is cleared and
        `model.to(device)` is safe; the copy engine fills them per step.
        """
        device = self.engine_config.Basic_Config.device_torch
        cfg = self.loaded_model_config
        # MXFP4 (K3) experts declare the checkpoint's packed/scale names and
        # need the validating wrapper; BF16 (48B) experts keep the plain one.
        from .k3.mxfp4_expert import K3MXFP4Expert, KimiK3MXFP4ExpertWrapper

        for layer_idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
            if not hasattr(layer, "block_sparse_moe"):
                continue
            moe = layer.block_sparse_moe
            if moe.experts is not None:
                for e_idx in range(len(moe.experts)):
                    expert = moe.experts[e_idx]
                    if expert is None:
                        continue
                    # Clear meta/params -> empty GPU tensors (streamed later).
                    # (`p.data =` cannot materialize meta params — swap the
                    # Parameter object via _replace_param instead.)
                    for name, _ in list(expert.named_parameters()):
                        _replace_param(expert, name,
                                       torch.empty(0, device=device))
                    wrapper_cls = (
                        KimiK3MXFP4ExpertWrapper
                        if isinstance(expert, K3MXFP4Expert)
                        else KimiLinearExpertWrapper
                    )
                    moe.experts[e_idx] = wrapper_cls(
                        expert, layer_idx, e_idx, self.core_engine,
                        self.engine_config, self.model_config,
                        persistent=False,
                    )
            shared = getattr(moe, "shared_experts", None)
            if self._stream_all_modules and shared is not None:
                # expert_idx=-1 -> module_key "shared_expert_{L}"
                # (ExpertWrapperBase._build_module_key), which is exactly the
                # key the parameter server registers for this module. Params
                # were emptied in _load_shared_expert_modules.
                moe.shared_experts = KimiLinearExpertWrapper(
                    shared, layer_idx, -1, self.core_engine,
                    self.engine_config, self.model_config,
                    persistent=False,
                )
            elif shared is not None and self._attn_tp_size > 1:
                shared._tp_size = self._attn_tp_size
                shared._tp_group = self._attn_tp_group
            # M2b: stamp the head-parallel (TP) context the resident-EP decode
            # forward reads for the intra-group token scatter/gather. G==1 leaves
            # attn_tp_size==1, so moe_forward_resident_ep_decode skips it and the
            # validated pure-DP path is byte-identical.
            moe.attn_tp_size = self._attn_tp_size
            moe.attn_tp_rank = self._attn_tp_rank
            moe.attn_tp_group = self._attn_tp_group
            moe.forward = types.MethodType(moe_forward_serving, moe)

    def _config_block_residual(self):
        """Wire K3's Block Attention Residuals into the serving path.

        No-op unless ``attn_res_block_size`` is set: the 48B keeps the classic
        pre-norm residual body and its ``(hidden_states,)`` 1-tuple return.

        For K3 two things are installed, because the two phases reach the
        decoder stack through different callers:

          * every decoder layer's ``forward`` -> ``decoder_layer_forward_
            block_residual``. DECODE arrives via ``KimiLinearModel.forward``,
            which threads ``block_residual`` explicitly, and is passed straight
            through. PREFILL arrives from the worker's prepack loop, which
            drives the layers itself, passes no ``block_residual`` and keeps
            only ``layer_outputs[0]``; that caller gets the carrier.
          * a pre-hook on ``model.norm`` applying the OUTPUT depth mix before
            the final norm (mixer-then-norm). The worker calls ``norm``
            directly, so this is the only seam where the output stage can run;
            it no-ops for the explicit path, which has already mixed.

        Both are pure wiring — the layer body itself is ``model.py``'s
        ``_forward_attn_residual`` in either case.
        """
        cfg = self.loaded_model_config
        if getattr(cfg, "attn_res_block_size", None) is None:
            return
        BlockResidualCarrier.configure(cfg.num_hidden_layers,
                                       cfg.attn_res_block_size)
        for layer in self.model.model.layers:
            layer.forward = types.MethodType(
                decoder_layer_forward_block_residual, layer
            )
        self.model.model.norm.register_forward_pre_hook(
            make_output_block_residual_pre_hook(self.model.model)
        )
        if self.rank == 0:
            logging.info(
                f"Block Attention Residuals wired: {cfg.num_hidden_layers} "
                f"layers, block size {cfg.attn_res_block_size}"
            )

    def _lm_head_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self):
        self.model.lm_head.register_forward_pre_hook(
            self._lm_head_forward_pre_hook
        )

    def _wrap_logits_output(self):
        """Worker expects HF-style outputs with .logits; our forward returns a
        raw tensor — wrap it."""
        model = self.model
        orig_forward = model.forward

        def forward_with_logits(self, *args, **kwargs):
            # The worker passes attention_mask=None; it would ride **kwargs
            # into the decoder-layer call and collide with the model's own
            # explicit attention_mask= argument (TypeError: multiple values).
            # Drop it here to keep model.py parity-clean.
            kwargs.pop("attention_mask", None)
            logits = orig_forward(*args, **kwargs)
            return types.SimpleNamespace(logits=logits)

        model.forward = types.MethodType(forward_with_logits, model)
