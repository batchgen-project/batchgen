# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
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
from batchgen.phase_lifecycle import (
    AdmitContext,
    ExitMode,
    PhaseEnterContext,
    PhaseKey,
)

from .block_residual import (
    BlockResidualCarrier,
    decoder_layer_forward_block_residual,
    make_output_block_residual_pre_hook,
)
from .config import is_k3_model_type, require_num_routed_experts
from .serving_modules import (
    dsa_materialize_pool_segment_index_keys,
    fuse_kda_decode_projections,
    kda_decode_serving,
    kda_prefill_serving,
    mla_decoding_nope_with_pagekv,
    mla_materialize_pool_segment,
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


# Slots in the KDA state pool reserved for decode-graph scratch rather than
# user sequences. The planner allocates the pool as ``sequence_slots + 1``
# (planner.py kda_state_slots); this is that "+ 1".
_KDA_GRAPH_SCRATCH_SLOTS = 1

# Every prefill MoE mode set_prefill_moe_mode accepts.
PREFILL_MOE_MODES = ("streamed", "resident_ep", "streamed_sp8")


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

        # K3 (and K3.1) vs 48B: the same family test the rest of the family
        # reads (config.is_k3_model_type — kimi_initializer.is_k3,
        # kimi_parameter_server). Skeleton key translation, the resident/
        # streamed prefill modes, distributed weights and the TP-G seams all
        # branch on it.
        self._is_k3 = is_k3_model_type(getattr(loaded_model_config, "model_type", None))
        self._distributed_weight_sharded = bool(
            getattr(
                engine_config.Basic_Config,
                "distributed_weight_sharded",
                False,
            )
        )
        if self._distributed_weight_sharded and not self._is_k3:
            raise ValueError("distributed host weights require K3")
        # Weight transport (initializer-set from the distributed store config,
        # no env var). "host_rdma" is the validated default in which every rank
        # pulls its own expert shard through the node daemon;
        # "hierarchical_gdr" pulls on one source rank per span slot and
        # replicates GPU-to-GPU across spans; "local" runs no daemon, so every
        # expert a rank reads must be one it owns as an EP rank.
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
            "local",
        ):
            raise ValueError(
                "distributed_weight_transport must be 'host_rdma', "
                f"'hierarchical_gdr' or 'local', got "
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
        # Slots the planner added for pooled prefix segments (planner.py
        # _K3_PREFIX_SEGMENT_STATE_SLOTS); never user capacity.
        self._kda_segment_slots = int(getattr(
            engine_config.GPU_Buffer_Config, "kda_prefix_segment_slots", 0
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
        # World-level, so built once per process and reused by every later
        # model rebuild (``_build_attn_tp_group``). The signature is the
        # (world_size, attention TP degree) the live groups were built for; a
        # later build request that disagrees is refused by name there.
        self._attn_tp_groups_built = False
        self._attn_tp_group_signature = None
        # Weight assembly has its own communicator.  The streamed-SP8
        # prefetcher launches the next layer's six all-gathers from a host
        # thread while the main thread runs the next layer's attention.  A
        # separate process group prevents those collectives from being
        # ordered against attention's all-reduces when ranks reach the
        # prefetch point at slightly different times.
        self._weight_tp_group = None
        # Streamed prefill span S (planner ``k3_prefill_stream_span``): the S
        # ranks of one span hold one copy of a layer's routed experts between
        # them, each streaming ``experts / S`` (the experts it owns as a rank
        # of the span), and gather the latent rows of their S / G DP ranks.
        # S == G is the node-local TP8 layout of the 8-GPU hosts; S > G spans
        # several DP ranks (GB300 trays, TP4 on one B300). 0: no streamed
        # layout is planned (G == 1, or the 48B).
        self._stream_span = int(getattr(
            engine_config.Module_Batching_Config, "k3_prefill_stream_span", 0
        ) or 0)
        if self._stream_span:
            if (
                self._stream_span % G
                or world_size % self._stream_span
                or self.num_experts % self._stream_span
            ):
                raise ValueError(
                    f"k3_prefill_stream_span {self._stream_span} must be a "
                    f"multiple of attention_group_size {G}, divide "
                    f"world_size {world_size} and divide the "
                    f"{self.num_experts} routed experts"
                )
            self._stream_span_rank = global_rank % self._stream_span
        else:
            self._stream_span_rank = 0
        self._stream_span_group = None  # torch NCCL group, built at configure
        # Hierarchical GDR adds a third communicator family: one cross-span
        # group per span slot, each carrying that slot's expert shard from its
        # single source rank to the same slot of every other span.
        self._cross_weight_group = None
        self._cross_weight_root = None
        self._cross_weight_source = False
        if self._hierarchical_gdr and not (
            self._distributed_weight_sharded
            and self._stream_span
            and world_size > self._stream_span
        ):
            raise ValueError(
                "hierarchical_gdr weight transport replicates one stream "
                "span's experts to the other spans; it needs distributed host "
                "weights and more than one span; got "
                f"distributed_weight_sharded={self._distributed_weight_sharded}, "
                f"world_size={world_size}, "
                f"k3_prefill_stream_span={self._stream_span}"
            )
        if (
            self._distributed_weight_sharded
            and self._distributed_weight_transport == "local"
            and self._stream_span not in (0, world_size)
        ):
            # A span narrower than the world streams experts another host
            # owns, which only a daemon transport can fetch.
            raise ValueError(
                "distributed_weight_transport 'local' needs the streamed "
                "prefill span to cover the EP world (each rank streams only "
                f"the experts it owns); got k3_prefill_stream_span="
                f"{self._stream_span} at world_size={world_size}"
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
        # Whether the build TP-shards the LatentMoE projections. Fixed by the
        # planner and the topology, so the server prefill mode below is known
        # before any model exists.
        self._latent_tp_planned = self._plans_latent_tp_sharding()
        self._prefill_moe_mode = self.default_prefill_moe_mode()
        self._decode_graph = None
        # Worker phase-instance reuse protocol (default lifecycle). The worker
        # sets ``reuse_phase_instances``; ``_instances_are_reused`` decides
        # whether this manager honours it and publishes the instance tuples.
        self.reuse_phase_instances = False
        self.prefill_instance = None
        self.decode_instance = None
        # "prefill" / "decode" once a phase has been entered on the model.
        self._active_phase = None

    # ------------------------------------------------------------------ #
    #  Phase-instance reuse protocol                                     #
    # ------------------------------------------------------------------ #

    @property
    def persistent_phase_instances(self):
        """Always False: ``--persistent-phase-instances`` is not supported.

        The worker's flag gate admits any manager with ``activate_decoding``
        and then writes True here, so the refusal has to live in the setter
        to stay at startup instead of surfacing at the first phase switch.
        """
        return False

    @persistent_phase_instances.setter
    def persistent_phase_instances(self, value):
        if value:
            raise RuntimeError(
                "Kimi-K3 does not support --persistent-phase-instances; it "
                "uses the default phase-instance reuse lifecycle"
            )

    @property
    def _instances_are_reused(self):
        """Whether the worker's reuse request is honoured for this deployment.

        Honoured only with resident-EP decode. Streamed decode keeps a
        non-empty routed-expert decode task, which the reuse activation
        refuses.
        """
        if not self.reuse_phase_instances:
            return False
        return self._decode_moe_mode() == "resident_ep"

    @property
    def decode_experts_resident(self):
        """Whether the decode instance's routed experts are resident AND active.

        The worker treats a resident decode instance as the active one
        and skips the prefill->decode switch while the instance is active.
        A resident-EP prefill also builds the shard, so the shard alone would
        keep the model in prefill mode through decode.
        """
        return bool(self._resident_ep_built and self._active_phase == "decode")

    def release_decode_routed_experts(self):
        """Release the decode graph and then this rank's resident EP shard."""
        self._release_resident_ep_decode()

    def _publish_prefill_instance(self):
        if self._instances_are_reused:
            self.prefill_instance = (self.model, self.weight_copy_task)
        return self.model, self.weight_copy_task

    def _publish_decode_instance(self):
        if self._instances_are_reused:
            self.decode_instance = (self.model, self.weight_copy_task)
        return self.model, self.weight_copy_task

    def _require_built_model(self, phase):
        if self.model is None:
            raise RuntimeError(
                f"Kimi-K3 activate_{phase} found no model: phase activation "
                "never builds one, so the instance was dropped (for example "
                "by deep_free_model_memory) while the worker still treated "
                "it as reusable"
            )

    def activate_prefill(self):
        """Switch the built model into the selected prefill mode."""
        self._require_built_model("prefill")
        start = time.perf_counter()
        stages = self._enter_prefill()
        self._log_phase_activation("prefill", start, stages)
        return self._publish_prefill_instance()

    def activate_decoding(self):
        """Switch the built model into decode: resident shard and decode graph."""
        self._require_built_model("decoding")
        start = time.perf_counter()
        stages = self._enter_decoding()
        self._log_phase_activation("decoding", start, stages)
        return self._publish_decode_instance()

    def _log_phase_activation(self, phase, start, stages):
        # Every rank: the resident shard is rebuilt on each switch into decode,
        # and a slow node is only visible in its own ranks' timings.
        total = time.perf_counter() - start
        memory = ""
        if torch.cuda.is_available():
            free, device_total = torch.cuda.mem_get_info(
                self.engine_config.Basic_Config.device_torch
            )
            memory = " free=%.2fGiB total=%.2fGiB" % (
                free / (1024 ** 3), device_total / (1024 ** 3)
            )
        logging.info(
            "[K3_PHASE] rank %s activate_%s total=%.3fs %s prefill_moe_mode=%s "
            "resident_ep_built=%s%s",
            self.global_rank,
            phase,
            total,
            " ".join(
                "%s=%.3fs" % (name, seconds)
                for name, seconds in stages.items()
            ),
            self._prefill_moe_mode,
            self._resident_ep_built,
            memory,
        )

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
        """Install the server prefill MoE mode on the built model's manager.

        The worker calls it once, at startup, with the mode
        ``default_prefill_moe_mode`` returned; every prefill wave of the
        server runs that mode. ``streamed`` is the legacy replicated pure-DP
        path. ``resident_ep`` is the R5 control. ``streamed_sp8`` keeps TP8
        attention but scatters token rows locally across the eight GPUs, runs
        one grouped all-expert MoE per layer, and gathers rows only inside
        the node.
        """
        self._prefill_moe_mode = self._validated_prefill_moe_mode(
            mode, resident_shard_built=self._resident_ep_built
        )

    def _validated_prefill_moe_mode(self, mode, *, resident_shard_built):
        """Normalize ``mode`` and refuse one this topology cannot run."""
        value = str(mode or "streamed").strip().lower()
        if value not in PREFILL_MOE_MODES:
            raise ValueError(
                "prefill MoE mode must be 'streamed', 'resident_ep' or "
                "'streamed_sp8', "
                f"got {mode!r}"
            )
        if value in {"resident_ep", "streamed_sp8"} and not self._is_k3:
            raise ValueError(
                f"{value} prefill is implemented only for Kimi-K3"
            )
        if value == "streamed" and self._latent_tp_planned:
            # The streamed path would hand each rank's latent column slice to
            # full-width experts (activation contract violation 93 layers in).
            raise ValueError(
                "prefill MoE mode 'streamed' needs unsharded LatentMoE "
                "projections; with TP-sharded projections use 'streamed_sp8' "
                "or 'resident_ep'"
            )
        if getattr(self, "_distributed_weight_sharded", False) and value == "streamed":
            # A host stores its own ranks' EP experts, so the resident shard
            # and the span-owned streamed shard are local; the legacy
            # replicated path would request every expert by name.
            raise ValueError(
                "distributed K3 host weights require k3_prefill_moe_mode="
                "'streamed_sp8' or 'resident_ep' (the legacy replicated path "
                "would request experts outside this host's store)"
            )
        if value == "streamed_sp8" and not getattr(self, "_stream_span", 0):
            raise ValueError(
                "k3_prefill_moe_mode='streamed_sp8' needs a planned "
                "k3_prefill_stream_span (a K3 DP-attention profile row)"
            )
        return value

    def default_prefill_moe_mode(self):
        """Return the server prefill MoE mode, fixed for the server's lifetime.

        The worker reads it once and every prefill wave runs it; no batch
        changes it. Without ``--prefill-moe-mode`` it follows the weight
        topology: ``streamed`` feeds each rank's full-width latent into
        full-width experts, so it is only valid while the LatentMoE
        projections are unsharded. Once they are TP-sharded (planner
        ``k3_shard_latent_projections``, TP8 K3, single node or not) each rank
        holds a column slice and prefill must go through ``streamed_sp8``,
        which rebuilds the full projections per phase. Distributed host
        weights force ``streamed_sp8`` for the same reason plus ingress.

        ``--prefill-moe-mode`` (``Basic_Config.prefill_moe_mode``) replaces
        that default and is validated here, at construction, so a mode the
        topology cannot run is refused before any model is built. The
        resident EP shard a distributed ``resident_ep`` server needs is
        built by the startup decode configure before the worker installs
        the mode.
        """
        requested = getattr(
            self.engine_config.Basic_Config, "prefill_moe_mode", None
        )
        if requested is None:
            if self._distributed_weight_sharded or self._latent_tp_planned:
                return "streamed_sp8"
            return "streamed"
        mode = self._validated_prefill_moe_mode(
            requested, resident_shard_built=True
        )
        if mode == "resident_ep" and (
            self._stream_all_modules or self._attn_tp_size <= 1
        ):
            raise ValueError(
                "--prefill-moe-mode resident_ep needs K3's resident TP "
                "attention layout (attention_group_size > 1, "
                "stream_all_modules off)"
            )
        return mode

    def prefill_uses_resident_ep(self):
        return self._prefill_moe_mode == "resident_ep"

    def release_decode_graph(self):
        """Drop captured decode graphs; they bake GPU KV-pool addresses."""
        if self._decode_graph is not None:
            self._decode_graph.release()
            self._decode_graph = None

    def prefill_uses_streamed_sp8(self):
        return self._prefill_moe_mode == "streamed_sp8"

    def gather_prefill_last_token_hidden(
        self, hidden_states, last_token_indices, num_global_rows, trace=None
    ):
        """Reassemble only sequence-final rows from the TP8 token shards.

        Streamed-SP8 carries one contiguous token-row shard through the final
        depth mix and norm.  The worker still needs one final hidden vector per
        replicated sequence for its unchanged sampling/writeback contract, but
        gathering every token row here would discard the whole-stack sharding
        win.  Exactly one TP rank owns each requested global row, so a small
        zero-filled all-reduce reconstructs the requested vectors exactly.

        ``trace`` is the batch-level ``prefix_pool_epilogue_trace`` recorder or
        None. It is given the shard, the owned mask and the per-row norms on
        both sides of the all-reduce, because none of that is visible to the
        caller; production behaviour with ``trace=None`` is unchanged. Every
        trace statement here is wrapped: this function runs a collective, so a
        diagnostics failure before the all_reduce would strand every other
        rank inside it.
        """

        from batchgen.prefix_reuse.probe_guard import run_probe

        if not self.prefill_uses_streamed_sp8():
            selected = hidden_states[0, last_token_indices, :]
            if trace is not None:
                from batchgen.prefix_reuse.epilogue_trace import (
                    record_unsharded_gather,
                )

                run_probe(
                    "epilogue_trace.gather_unsharded",
                    lambda: record_unsharded_gather(
                        trace,
                        selected,
                        num_global_rows=int(hidden_states.shape[1]),
                        manager=self,
                    ),
                    rank=self.global_rank,
                )
            return selected

        from .moe_tp_reshard import balanced_row_split

        num_global_rows = int(num_global_rows)
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "streamed-SP8 final hidden state must have shape (1, rows, hidden)"
            )
        if hidden_states.shape[1] == num_global_rows:
            return hidden_states[0, last_token_indices, :]

        start, end = balanced_row_split(
            num_global_rows, self._attn_tp_size
        )[self._attn_tp_rank]
        if hidden_states.shape[1] != end - start:
            raise ValueError(
                "streamed-SP8 final hidden state has the wrong local row count"
            )

        selected = hidden_states.new_zeros(
            (last_token_indices.numel(), hidden_states.shape[-1])
        )
        owned = (last_token_indices >= start) & (last_token_indices < end)
        selected[owned] = hidden_states[
            0, (last_token_indices[owned] - start).long(), :
        ]
        # Canary for the s4/s5 gather bug (an int32 row-offset wrap in the
        # Triton RMSNorm, since fixed): every last-token row this rank owns must
        # carry a real hidden state, so an all-zero owned row means the gather
        # dropped it. Check the owned rows directly and warn only on the anomaly
        # -- the old form sampled a fixed trailing window (range(-8, 0)), which
        # both missed non-tail drops and IndexError'd on <8-sequence batches.
        owned_all_zero = 0
        if owned.any():
            owned_all_zero = int(
                (selected[owned].abs().sum(dim=-1) == 0).sum().item()
            )
            if owned_all_zero:
                logging.warning(
                    "[K3_PREFILL_GATHER] rank %s: %d of %d owned last-token rows "
                    "are all-zero (shard=[%d, %d), global_rows=%d)",
                    self.global_rank, owned_all_zero, int(owned.sum().item()),
                    start, end, num_global_rows,
                )

        output_norm = getattr(
            self.model.model, "output_attn_res_norm", None
        )
        order_wait = getattr(
            output_norm, "_streamed_sp8_order_wait", None
        )
        if order_wait is not None:
            order_wait()

        import torch.distributed as dist

        # Read BEFORE the collective, so it must not be able to skip it.
        pre_norms = (
            run_probe(
                "epilogue_trace.gather_pre_norms",
                lambda: selected.detach().float().norm(dim=-1).tolist(),
                rank=self.global_rank,
            )
            if trace is not None else None
        )
        dist.all_reduce(selected, group=self._attn_tp_group)
        if trace is not None:
            run_probe(
                "epilogue_trace.gather_record",
                lambda: trace.record_gather(
                    attn_tp_rank=self._attn_tp_rank,
                    attn_tp_size=self._attn_tp_size,
                    num_global_rows=num_global_rows,
                    shard_start=start, shard_end=end,
                    owned=owned.detach().cpu().tolist(),
                    pre_norms=pre_norms,
                    post_norms=selected.detach().float().norm(dim=-1).tolist(),
                    canary_zero_rows=(owned_all_zero if owned.any() else 0),
                ),
                rank=self.global_rank,
            )
        return selected

    def streamed_sp8_requires_global_pass_alignment(self):
        """Whether every rank must join each streamed prefill model pass.

        True when a pass's MoE collectives reach beyond one DP rank: the span
        gathers the latent rows of several DP ranks (S > G), or hierarchical
        GDR broadcasts weights across spans.
        """
        return self.prefill_uses_streamed_sp8() and (
            self._hierarchical_gdr or self._stream_span > self._attn_tp_size
        )

    def run_streamed_sp8_transport_only_prefill(self, num_passes):
        """Join the MoE collectives of passes this DP rank has no rows for.

        Each MoE layer runs its streamed forward with zero local rows: it
        loads the layer (ingress, and the hierarchical source broadcast),
        joins the span's latent gather and FP32 reduce-scatter when another
        DP rank of the span has rows, and opens the cross-span launch gate,
        in the same per-layer order as a populated pass. Attention, the
        shared expert and logits stay inside the DP rank and are skipped.
        """
        num_passes = int(num_passes)
        if num_passes < 0:
            raise ValueError("streamed-SP8 transport-only pass count must be >= 0")
        if num_passes == 0:
            return
        if not self.streamed_sp8_requires_global_pass_alignment():
            raise RuntimeError(
                "streamed-SP8 transport-only prefill is only valid when a "
                "pass's MoE collectives reach beyond the DP rank"
            )
        if self._streamed_sp8_buffer is None or self.model is None:
            raise RuntimeError(
                "streamed-SP8 transport-only prefill requires an initialized "
                "layer buffer"
            )
        hidden = int(self.loaded_model_config.hidden_size)
        device = self.engine_config.Basic_Config.device_torch
        empty = torch.empty((0, hidden), dtype=torch.bfloat16, device=device)
        buffer = self._streamed_sp8_buffer
        for _ in range(num_passes):
            if buffer.has_pending():
                raise RuntimeError(
                    "transport-only streamed-SP8 pass started with a pending layer"
                )
            for layer in self.model.model.layers:
                moe = getattr(layer, "block_sparse_moe", None)
                streamed = (
                    getattr(moe, "_streamed_sp8_moe", None)
                    if moe is not None else None
                )
                if streamed is None:
                    continue
                streamed.forward(empty, moe.gate, 0)
                buffer.allow_cross_launch()
            if buffer.has_pending():
                raise RuntimeError(
                    "transport-only streamed-SP8 pass left a pending layer"
                )

    def run_resident_ep_collective_only_prefill(self):
        """Join one resident-EP prefill pass with zero local rows.

        A resident-EP prefill pass issues, per MoE layer and in layer order,
        the EP-world all_gathers and all_reduce inside ``resident.forward``;
        everything else in the pass is node-local. A rank whose admission
        wave has fewer micro-batches than the global maximum (or none) joins
        those collectives here so its resident experts keep serving the other
        ranks' tokens — the caller runs the worker's rank-count sync first,
        exactly as a real pass does before its forward.
        """
        if not self._resident_ep_built or self.model is None:
            raise RuntimeError(
                "resident-EP collective-only prefill needs the resident shard"
            )
        hidden = int(self.loaded_model_config.hidden_size)
        device = self.engine_config.Basic_Config.device_torch
        empty = torch.empty((0, hidden), dtype=torch.bfloat16, device=device)
        for layer in self.model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            resident = getattr(moe, "_resident_ep_moe", None) if moe is not None else None
            if resident is None:
                continue
            # TP-sharded latent projections apply down_proj to the TP group's
            # (here empty) replicated rows; the collective shapes still follow
            # the synced per-rank count, so an empty group is a valid pass.
            resident.forward(
                empty, moe.gate,
                x_group=empty if resident.latent_tp_size > 1 else None,
            )

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
        # The planner sizes the pool as ``sequence_slots + 1`` (planner.py:
        # kda_state_slots), where the extra slot is decode-graph scratch, NOT
        # user capacity. Nothing else reserves it, so reporting the whole pool
        # let admission fill every slot: with a 193-slot pool the scheduler was
        # told "193 free", selected 193 sequences for the node, and prefill died
        # with "Insufficient free KDA state items" because the scratch slot had
        # been consumed. Hold it back here, where capacity is reported.
        available -= _KDA_GRAPH_SCRATCH_SLOTS + self._kda_segment_slots
        available = max(0, int(available))
        if self._attn_tp_size > 1:
            return {"max_sequences_per_node": available}
        return {"max_sequences_per_rank": available}

    def _set_prefill_memory_tiling(self, enabled, token_tile=512):
        """Bound K3 prefill temporaries for resident-EP and streamed-SP8.

        ``token_tile`` is the row tile for the norms and the dense/shared
        FFNs.  Resident-EP keeps 512.  Streamed-SP8 passes 8,192, the FFN's
        validated even tiler: at exact 64K a 512-row tile issued about
        10,000 launches per layer for the shared expert and ran it 3.6x
        slower than the same GEMMs in isolation, to save about 250 MiB of
        scratch.  The KDA segment stays at 4,096 in both modes.
        """
        if self.model is None:
            return
        tile = int(token_tile) if enabled else None
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
                    chunk_rows = int(getattr(
                        self.engine_config.Module_Batching_Config,
                        "k3_resident_prefill_chunk_rows", 0,
                    ) or 0)
                    if chunk_rows > 0:
                        resident.compact_large_pass_rows = chunk_rows

    def _reset_model_attached_state(self):
        """Forget what was attached to a model this instance no longer owns.

        ``_build_model`` runs again whenever the phase instances are rebuilt:
        the worker's ``deep_free_model_memory()`` drops the model, and the next
        ``configure_prefill`` / ``configure_decoding`` finds ``self.model is
        None`` and builds a new one. Everything the configure steps attach
        lives ON that model's modules — the resident-EP shard hangs off
        ``layer.block_sparse_moe._resident_ep_moe``, the streamed-SP8 buffer
        off the MoE blocks and the attention modules, and a captured decode
        graph swaps their ``forward`` methods and bakes their addresses — so a
        freshly built model starts with none of it.

        The flags that record "already attached" are therefore claims about the
        PREVIOUS model and must not outlive it. With ``_resident_ep_built``
        left True across the rebuild, ``configure_decoding``'s
        ``_init_resident_ep_decode()`` returned early, the new model's MoE
        blocks kept no resident layer, and decode fell through to the streamed
        expert wrapper — whose decode weight-copy task is deliberately empty
        under resident EP — so ``core_engine.get_weights`` had nothing staged
        and every rank raised ``GPU_Weight_Buffer get_weights()`` at the
        SECOND prefill->decode switch. The first switch worked because the
        shard was genuinely built there.

        Releasing is not this function's job: the owners run before the
        rebuild (``deep_free_model_memory`` on the worker side,
        ``_release_resident_ep_decode`` / ``_release_streamed_sp8_prefill``
        here). This drops only the bookkeeping that refers to dead modules, so
        each phase instance stages what it needs at its own build.
        """
        self._resident_ep_built = False
        self._streamed_sp8_buffer = None
        # The instance tuples hold the dead model; publishing them again is
        # the next configure's job.
        self.prefill_instance = None
        self.decode_instance = None
        self._active_phase = None
        # Restores the old model's forwards and drops its captured buckets;
        # None-safe, so the first build is a no-op.
        self.release_decode_graph()
        # The FP32 combine output of the previous model's last resident prefill
        # is class-level, so nothing above drops it: a new model's passes
        # reserve their own (``prepare_prefill_output``), and holding the dead
        # pass's buffer across every rebuild is a per-switch leak.
        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )

        ResidentEPMXFP4MoELayer.release_prefill_output()

    def _assert_resident_ep_attached(self):
        """The resident shard must be ON the live model when the flag says so.

        ``_resident_ep_built`` is a claim, and the claim is cheap to check: the
        resident layer is an attribute of every MoE block. Checking it where
        the build is skipped turns a stale flag into a named lifecycle error
        instead of a weight-buffer failure three layers down in the decode MoE.
        """
        if self.model is None:
            raise RuntimeError(
                "resident-EP decode shard is marked built while this strategy "
                "manager holds no model; the flag outlived its model"
            )
        for layer in self.model.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None:
                continue
            if getattr(moe, "_resident_ep_moe", None) is None:
                raise RuntimeError(
                    "resident-EP decode shard is marked built but the live "
                    "model's MoE blocks carry no resident layer: the flag "
                    "survived a model rebuild, so decode would fall through "
                    "to the streamed expert path with no staged weights"
                )
            return

    def _release_resident_ep_decode(self):
        """Drop phase-inactive resident decode experts before streamed prefill.

        A world16 K3 rank's resident MXFP4 EP shard occupies about 84 GiB. It
        is never read by streamed-SP8 prefill, whose experts arrive through
        the layer ring, so retaining both copies forces exact-64K prompts into
        one-sequence model passes. Decode graphs bake the resident shard's
        addresses and must be released first; configure_decoding rebuilds the
        shard and reinstalls/captures graphs before the next decode forward.
        """
        # Class-level, so it belongs to the PASS that allocated it and not to
        # any model instance: it must go back whether or not a shard is
        # currently attached. Keeping this above the flag check is load-bearing
        # — once a rebuild clears the flag (``_reset_model_attached_state``),
        # everything below is skipped, and leaving the FP32 combine
        # output of a finished prefill mapped would be a per-switch leak.
        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )

        ResidentEPMXFP4MoELayer.release_prefill_output()
        if not self._resident_ep_built:
            return 0

        self.release_decode_graph()

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
        self._set_prefill_memory_tiling(enabled, token_tile=8192)
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
            self._drop_streamed_full_latent_projections()
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
                            "_streamed_sp8_layer_idx",
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
                        "_streamed_sp8_keep_sharded",
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
        out-of-order request either throws at once (a module already consumed
        in this pass) or ends in ``get_weights``' stall ``std::runtime_error``;
        it does not hang and it does not return the wrong module.

        Consumer order, per forward pass:
          * one ``attn_{L}`` or ``kda_attn_{L}`` per layer, L ascending, at the
            top of the decoder layer;
          * one ``shared_expert_{L}`` per MoE layer, L ascending;
          * all ``routed_expert_{L}_{E}``, E ascending within L, L ascending —
            ``moe_forward_serving`` runs the experts with tokens in this order
            and declares the 0-token ones skipped first, so the producer never
            waits on a slot that no consumer will request.

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
                    start = self._stream_span_rank * (
                        len(moe.experts) // self._stream_span
                    )
                    end = start + len(moe.experts) // self._stream_span
                    if not (start <= e_idx < end):
                        continue
                task["routed_expert"].append(
                    f"routed_expert_{layer_idx}_{e_idx}"
                )
        return task

    def configure_prefill(self):
        """Build the model if needed and switch to the selected prefill mode."""
        if self.model is None:
            self._build_model()
        self._enter_prefill()
        if self._is_k3:
            self._prewarm_triton_kernels()
        return self._publish_prefill_instance()

    def _prewarm_triton_kernels(self):
        """Compile K3's reachable Triton keys before the first request.

        Startup reaches this from ``prepare_kimi_k3_startup`` on every rank,
        before readiness. It runs after ``_enter_prefill`` because the prefill
        mode sets the KDA segment cap, which is part of the planned shapes.
        Once per process (``triton_prewarm.prewarm_k3_triton_kernels``); later
        phase switches return at once. The node-local attention-TP ranks share
        one Triton cache directory, so they split the compiles and meet at a
        barrier on that group before every rank loads the full set.
        """
        if not torch.cuda.is_available():
            return
        from batchgen.config.config import prefill_token_cap_for_mode

        from .triton_prewarm import prewarm_k3_triton_kernels

        basic = self.engine_config.Basic_Config
        buckets = list(getattr(basic, "decode_graph_buckets", None) or [])
        if not buckets:
            from .cuda_graph_segments import DEFAULT_DECODE_GRAPH_BUCKETS

            buckets = list(DEFAULT_DECODE_GRAPH_BUCKETS)
        part_index, part_count, barrier = 0, 1, None
        group = self._attn_tp_group
        if self._attn_tp_size > 1 and group is not None:
            import torch.distributed as dist

            if dist.is_initialized():
                part_index, part_count = self._attn_tp_rank, self._attn_tp_size
                device = torch.device(basic.device_torch)
                device_ids = (
                    [device.index] if device.index is not None else None
                )

                def _group_barrier():
                    dist.barrier(group=group, device_ids=device_ids)

                barrier = _group_barrier

        start = time.perf_counter()
        if prewarm_k3_triton_kernels(
            self.model,
            self.loaded_model_config,
            decode_buckets=buckets,
            # The micro-batch cap of every wave of this server: its prefill
            # MoE mode is fixed at construction.
            max_prefill_tokens=prefill_token_cap_for_mode(
                self.engine_config.Module_Batching_Config,
                self._prefill_moe_mode,
            ),
            part_index=part_index,
            part_count=part_count,
            barrier=barrier,
            rank=self.global_rank,
        ):
            logging.info(
                "[K3_TRITON_WARMUP] rank=%s total wall=%.1fs",
                self.global_rank,
                time.perf_counter() - start,
            )

    def _enter_prefill(self):
        """Switch the built model to streamed, streamed-SP8 or resident-EP prefill.

        Re-entrant on the same model: a live streamed-SP8 buffer is kept and
        the weight-copy task is rebuilt deterministically, so the worker's
        installed H2D schedule fingerprint still matches. Returns stage
        timings in seconds.
        """
        start = time.perf_counter()
        self._active_phase = "prefill"
        self.loaded_model_config.phase = "prefill"
        self.loaded_model_config.ep_size = (
            self.world_size if self.prefill_uses_resident_ep() else 1
        )
        AttnWrapperBase.phase = "prefill"
        KimiLinearExpertWrapper.phase = "prefill"
        # block_residual never outlives one decoder pass; drop anything a
        # previous (possibly aborted) pass left parked so the first prefill of
        # this phase starts from the seeded-at-layer-0 state.
        BlockResidualCarrier.reset()
        self.weight_copy_task = self._build_weight_copy_task()
        copy_task_done = time.perf_counter()
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
            if (
                self._stream_all_modules
                or self._attn_tp_size <= 1
                or not self._stream_span
            ):
                raise RuntimeError(
                    "streamed-SP8 prefill requires K3's resident TP attention "
                    "layout (attention_group_size>1, stream_all_modules=False) "
                    "and a planned k3_prefill_stream_span"
                )
            self._set_resident_ep_prefill_enabled(False)
            self._release_resident_ep_decode()
            self._init_streamed_sp8_prefill()
            self._set_streamed_sp8_prefill_enabled(True)
            if self.rank == 0:
                logging.info(
                    "[K3_PREFILL_MOE] streamed_sp8 enabled: transport=%s, "
                    "TP%d attention, stream span %d (%d experts/shard), "
                    "latent gather/FP32 reduce-scatter over the span",
                    self._distributed_weight_transport,
                    self._attn_tp_size,
                    self._stream_span,
                    self.num_experts // self._stream_span,
                )
        else:
            self._release_streamed_sp8_prefill()
            self._set_resident_ep_prefill_enabled(False)
        return {
            "copy_task": copy_task_done - start,
            "moe_mode": time.perf_counter() - copy_task_done,
        }

    def configure_decoding(self, padding_bsz=None, comm=None):
        """Build the model if needed and switch to decode phase.

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
            # wrapper would die on get_weights' stall throw. Fail here,
            # loudly, instead of at that timeout 93 layers deep.
            raise NotImplementedError(
                "stream_all_modules is a PREFILL-only path (M-PR-6). Decode "
                "needs a decode-phase H2D streamer for attn/kda_attn/"
                "shared_expert, which does not exist. Run prefill-only "
                "(max_tokens=1 still enters decode today — see PREFILL_PLAN "
                "C4) or turn the flag off."
            )
        if comm is not None:
            self._comm = comm
        build_seconds = 0.0
        if self.model is None:
            # The rebuild drops the streamed-SP8 buffer reference without
            # closing it, so close it first.
            self._release_streamed_sp8_prefill()
            build_start = time.perf_counter()
            self._build_model()
            build_seconds = time.perf_counter() - build_start
        stages = self._enter_decoding()
        # Stage timings on every rank: r29 (2x8 H200) showed node 1 reaching
        # the resident shard build 36 s after entering configure_decoding
        # (node 0: 7 s) with nothing logged in between.
        logging.info(
            "[K3] configure_decoding rank %s: release_streamed=%.1fs "
            "build/copy_task=%.1fs resident_ep=%.1fs decode_graph=%.1fs",
            self.global_rank,
            stages["release_streamed"],
            build_seconds + stages["copy_task"],
            stages["resident_ep"],
            stages["decode_graph"],
        )
        return self._publish_decode_instance()

    def _enter_decoding(self):
        """Switch the built model to decode.

        Releases the streamed-SP8 prefill state, materializes the resident EP
        shard when it is not built, and installs a fresh decode graph adapter
        (the previous one was released with the shard whose addresses it
        bakes). Returns stage timings in seconds.
        """
        self._active_phase = "decode"
        self.loaded_model_config.phase = "decode"
        _t0 = time.perf_counter()
        self._release_streamed_sp8_prefill()
        _t1 = time.perf_counter()
        self._set_resident_ep_prefill_enabled(False)
        AttnWrapperBase.phase = "decode"
        KimiLinearExpertWrapper.phase = "decode"
        BlockResidualCarrier.reset()
        self.weight_copy_task = self._build_weight_copy_task()
        _t2 = time.perf_counter()
        if self._decode_moe_mode() == "resident_ep":
            self._init_resident_ep_decode()
            # Resident shards serve every routed expert: decode streams
            # nothing, so the copy engine gets no decode expert tasks.
            self.weight_copy_task["routed_expert"] = []
        _t3 = time.perf_counter()
        self._init_decode_graph()
        return {
            "release_streamed": _t1 - _t0,
            "copy_task": _t2 - _t1,
            "resident_ep": _t3 - _t2,
            "decode_graph": time.perf_counter() - _t3,
        }

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
            deepep_ll=bool(getattr(
                self.engine_config.Module_Batching_Config, "k3_deepep_ll", False)),
        )
        self._decode_graph.install()

    def preload_runtime_extensions(self) -> tuple:
        """Model-specific native dependencies, resolved before readiness.

        The worker's pre-readiness startup pass calls this after its own
        family-wide preload. Kimi-K3.1's DSA layers reach kernels no Kimi-K3
        layer does (``dsa_runtime``); a dense model has nothing to add.
        Returns the names resolved.
        """
        dsa = getattr(self.loaded_model_config, "dsa", None)
        if dsa is None or dsa() is None:
            return ()
        from .dsa_runtime import preload_dsa_runtime

        resolved = preload_dsa_runtime()
        logging.info(
            "Rank %s: [K3_STARTUP] Loaded DSA runtime %s", self.rank, list(resolved)
        )
        return resolved

    def prewarm_deepep_exchange(self) -> bool:
        """Build the process-lifetime DeepEP low-latency buffer at startup.

        The first decode configure otherwise pays the 16-rank NVSHMEM
        handshake (~5 s of the ~19 s first configure, r38) inside the measured
        wall; the exchange only depends on the graph geometry (max group
        bucket / TP, latent width, expert counts), so it can exist before the
        first admission. ~1.9 GiB of HBM for the prefill phase (98 GiB free).
        """
        if not bool(getattr(self.engine_config.Module_Batching_Config, "k3_deepep_ll", False)):
            return False
        import torch.distributed as dist
        from batchgen.moe.deepep_ll import get_low_latency_exchange
        from .moe_cuda_graph_segments import k3_moe_graph_buckets
        if not dist.is_initialized() or dist.get_world_size() != self.world_size:
            raise RuntimeError("k3_deepep_ll: torch.distributed world != EP world")
        basic = self.engine_config.Basic_Config
        buckets = list(getattr(basic, "decode_graph_buckets", None) or [])
        if not buckets:
            from .cuda_graph_segments import DEFAULT_DECODE_GRAPH_BUCKETS
            buckets = list(DEFAULT_DECODE_GRAPH_BUCKETS)
        tp = int(self._attn_tp_size)
        max_lt = max(k3_moe_graph_buckets(buckets, tp)) // tp
        # the routed latent width (the TP-sharded down_proj only holds 1/tp of it)
        latent = int(getattr(self.loaded_model_config, "routed_expert_hidden_size", 0) or 0)
        if latent <= 0:
            raise RuntimeError("k3_deepep_ll: routed_expert_hidden_size missing from the model config")
        t0 = time.perf_counter()
        ex = get_low_latency_exchange(
            dist.group.WORLD, max_tokens_per_rank=max_lt, hidden=latent,
            num_experts=self.num_experts, num_local_experts=self.experts_per_rank)
        logging.info(
            "[K3] DeepEP low-latency exchange prewarmed at startup: max %d tokens/rank, hidden %d, "
            "%d experts, rdma buffer %.0f MiB (%.1fs)",
            ex.max_tokens_per_rank, ex.hidden, ex.num_experts, ex.rdma_bytes / 2**20,
            time.perf_counter() - t0)
        return True

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
            # Kimi-K3.1: the DSA spans also bake the auxiliary (index-key)
            # cache; a single-region manager binds none.
            AttnWrapperBase.gpu_paged_kv_manager_aux = getattr(
                gpu_manager, "auxiliary", None
            )
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
            self._assert_resident_ep_attached()
            return
        assert self.world_size == 1 or self._comm is not None, (
            "resident-EP decode needs the NCCL communicator (worker passes "
            "it via configure_decoding(comm=...) or set_comm); world_size 1 "
            "runs without collectives and needs none"
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
            if self._latent_projections_sharded():
                for layer in self.model.model.layers:
                    moe = getattr(layer, "block_sparse_moe", None)
                    resident = getattr(moe, "_resident_ep_moe", None) if moe is not None else None
                    if resident is not None:
                        resident.set_latent_tp(
                            self._attn_tp_size, self._attn_tp_rank, self._attn_tp_group
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
        # Both bounds come from the planner (the machine profile row on TP8);
        # a missing value is a planning error, never a default.
        batching = self.engine_config.Module_Batching_Config
        planned_rows = {}
        for name in (
            "k3_prefill_grouped_chunk_rows",
            "k3_prefill_collective_stripe_threshold_rows",
        ):
            planned_rows[name] = int(getattr(batching, name, 0) or 0)
            if planned_rows[name] <= 0:
                raise RuntimeError(
                    f"streamed-SP8 prefill requires a planned positive "
                    f"Module_Batching_Config.{name}; the planner did not set it"
                )
        self._streamed_sp8_buffer = StreamedSP8LayerBuffer(
            core_engine=self.core_engine,
            device=self.engine_config.Basic_Config.device_torch,
            span_group=self._stream_span_group,
            span_rank=self._stream_span_rank,
            span_size=self._stream_span,
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
                norm._streamed_sp8_layer_idx = layer_idx
                if profile_name == "mlp_depth_mix" and streamed_moe:
                    norm._streamed_sp8_keep_sharded = True
            if not streamed_moe:
                continue
            moe._streamed_sp8_sharded_carry = True
            moe._streamed_sp8_moe = StreamedSP8MXFP4MoELayer(
                layer_idx=layer_idx,
                buffer=self._streamed_sp8_buffer,
                down_proj=(
                    self._streamed_full_latent_projection(layer_idx, "routed_expert_down_proj")
                    if self._latent_projections_sharded()
                    else moe.routed_expert_down_proj
                ),
                norm=(
                    moe.routed_expert_norm
                    if getattr(moe, "latent_moe_use_norm", False)
                    else None
                ),
                up_proj=(
                    self._streamed_full_latent_projection(layer_idx, "routed_expert_up_proj")
                    if self._latent_projections_sharded()
                    else moe.routed_expert_up_proj
                ),
                chunk_rows=planned_rows["k3_prefill_grouped_chunk_rows"],
                collective_stripe_threshold_rows=planned_rows[
                    "k3_prefill_collective_stripe_threshold_rows"
                ],
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
            output_norm._streamed_sp8_keep_sharded = True
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

    @property
    def attn_tp_group(self):
        """This rank's attention (serve) TP process group; None for G == 1.

        The worker reduces its prefill prefix-pool budget and checks its plan
        digest over this group, so every rank of a serve group runs the same
        wave."""
        return self._attn_tp_group

    def supports_group_pool_prefill(self):
        """Whether the prefill prefix pool stays group-identical at G > 1.

        A serve group replicates every sequence onto all G of its ranks and
        prefills them in lockstep, so one plan per group suffices: each rank
        pools its own head shard of the same segments and the per-layer
        collectives still pair up chunk for chunk. True for the whole family:
        the 48B at G > 1 runs the same resident head-parallel KDA (M2b) and the
        same NoPE-MLA pool seams, and at G == 1 the worker never asks."""
        return True

    def prefill_pool_layout(self):
        """K3's prefill prefix-pool geometry: the EXPANDED MLA K/V, not the latent.

        Pool attention reads K of ``qk_nope + qk_rope`` and V of ``v_head_dim``
        per LOCAL head (``_load_attn_modules`` already sharded them), while the
        Host KV keeps one 576-wide latent per token, so the pool cannot be
        sized from the Host geometry. ``layer_map`` maps an engine layer to its
        dense pool row, the same way the Host and GPU pools are mapped.

        K3.1 (DSA) is the other case: its pooled prefill is the SPARSE one,
        whose attention is the absorbed ``qv`` form over the 576-wide latent at
        page size 1 (``dsa_sparse_prefill``), and whose selection reads the
        index keys. So the pool holds one latent head and no V, and the index
        keys go in the pool's own indexer rows (the worker's
        ``_prefill_pool_layout`` declares those from the layers' indexers).
        Declaring the expanded heads there instead would size the pool for a
        kernel the DSA path never calls.

        Answerable with or without a resident model: the worker calls this
        while planning a wave, which happens in the decode phase where the
        model may be deep-freed.
        """
        cfg = self.loaded_model_config
        mla_layers = [
            layer_idx for layer_idx in range(cfg.num_hidden_layers)
            if cfg.is_mla_layer(layer_idx)
        ]
        rows = {layer_idx: row for row, layer_idx in enumerate(mla_layers)}
        layer_map = tuple(
            rows.get(layer_idx, -1)
            for layer_idx in range(cfg.num_hidden_layers)
        )
        sparse = bool(cfg.num_dsa_layers)
        if sparse and sorted(rows) != sorted(
            layer_idx for layer_idx in rows if cfg.is_dsa_layer(layer_idx)
        ):
            raise RuntimeError(
                "the prefill prefix pool needs every MLA layer of a DSA "
                "model to be indexer-gated: the pool keeps one latent row "
                "per MLA layer and the sparse prefill selects on every one "
                f"of them ({cfg.num_dsa_layers} DSA of {len(rows)} MLA)"
            )
        # Derived from the config, NOT from the module: the worker plans a
        # pooled wave at reservation time, which can fall in a phase window
        # where the model has been deep-freed and not rebuilt yet
        # (``self.model is None``), and reading a layer there is an
        # ``AttributeError: 'NoneType' object has no attribute 'model'``. Every
        # number below is a model constant that does not move with the phase:
        # the K/V widths are never sharded, and ``_load_attn_modules`` shards
        # the head count by exactly ``_attn_tp_size``, which is fixed for the
        # process. Same contract as the KDA state pools and as the worker's own
        # DSA ``_prefill_pool_layout`` cache: decided once, reused across phases.
        heads = int(cfg.num_attention_heads)
        if self._attn_tp_size > 1:
            heads //= self._attn_tp_size
        if sparse:
            layout = {
                "num_layers": len(mla_layers),
                "layer_map": layer_map,
                "num_kv_heads": 1,
                "k_head_dim": int(cfg.compressed_kv_dim),
                # No V row: the absorbed sparse attention reads the latent's
                # two halves out of the K row itself.
                "v_head_dim": 0,
            }
        else:
            layout = {
                "num_layers": len(mla_layers),
                "layer_map": layer_map,
                "num_kv_heads": heads,
                "k_head_dim": int(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
                "v_head_dim": int(cfg.v_head_dim),
            }
        if self.model is None:
            return layout
        # The module is the ground truth whenever it is resident, so compare
        # instead of trusting the derivation: a future sharding seam that
        # changes a head count or a width must fail here, where the pool is
        # sized, rather than corrupt pool rows it no longer matches.
        attn = self.model.model.layers[mla_layers[0]].self_attn
        attn = getattr(attn, "module", attn)
        if sparse:
            live = {
                "num_kv_heads": 1,
                "k_head_dim": int(attn.kv_lora_rank + attn.qk_rope_head_dim),
                "v_head_dim": 0,
            }
        else:
            live = {
                "num_kv_heads": int(attn.num_heads),
                "k_head_dim": int(
                    attn.qk_nope_head_dim + attn.qk_rope_head_dim
                ),
                "v_head_dim": int(attn.v_head_dim),
            }
        divergent = {
            key: (value, layout[key])
            for key, value in live.items() if value != layout[key]
        }
        if divergent:
            raise RuntimeError(
                "prefill pool layout derived from the config disagrees with "
                f"the resident MLA module (attn_tp_size={self._attn_tp_size}): "
                + ", ".join(
                    f"{key} module={module} config={config}"
                    for key, (module, config) in sorted(divergent.items())
                )
            )
        return layout

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
        # A new model carries none of the previous one's attachments, so the
        # bookkeeping that says otherwise is dropped HERE, before any configure
        # step consults it to decide whether to stage something.
        self._reset_model_attached_state()
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
        # 2c. TP-shard the LatentMoE projections across the attention TP group
        #     (DECODE_CONCURRENCY_PLAN.md): -8.3 GiB/rank on the world16 K3.
        self._shard_latent_projections()
        # 2d. vocab-parallel embed_tokens / lm_head across the same group
        #     (slice 2): -3.84 GiB/rank. Before the lm_head hook is attached.
        self._shard_vocab_parallel()

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
        # K3's Triton kernels (attention-residual mixer included) are
        # prewarmed by configure_prefill, before readiness.
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

    def _latent_projections_sharded(self):
        return bool(getattr(self, "_latent_tp_sharded", False))

    def _plans_latent_tp_sharding(self):
        """Whether ``_shard_latent_projections`` shards this deployment.

        A function of the planner flag ``k3_shard_latent_projections`` and the
        attention TP degree only, so the prefill mode can be chosen before
        the model is built.
        """
        return bool(
            self._is_k3
            and getattr(
                self.engine_config.Module_Batching_Config,
                "k3_shard_latent_projections", False,
            )
            and self._attn_tp_size > 1
        )

    def _shard_vocab_parallel(self):
        """Replace ``embed_tokens`` / ``lm_head`` by vocab-parallel modules that
        keep this rank's ``[c*V/G, (c+1)*V/G)`` rows (planner
        ``k3_shard_vocab_parallel``, TP8 K3 on H200). The lookup all_reduces
        and the head all_gathers over the attention TP group, so callers see
        the same tensors as before (see vocab_parallel.py)."""
        self._vocab_tp_sharded = False
        G = self._attn_tp_size
        want = bool(getattr(
            self.engine_config.Module_Batching_Config,
            "k3_shard_vocab_parallel", False,
        ))
        if not (self._is_k3 and want and G > 1 and self._attn_tp_group is not None):
            return
        from .vocab_parallel import (
            VocabParallelEmbedding, VocabParallelLMHead, vocab_shard_bounds,
        )
        device = self.engine_config.Basic_Config.device_torch
        embed = self.model.model.embed_tokens
        head = self.model.lm_head
        vocab = int(embed.weight.shape[0])
        start, end = vocab_shard_bounds(vocab, G, self._attn_tp_rank)
        hosts = {}
        for key in ("model.embed_tokens.weight", "lm_head.weight"):
            ckpt_key = self._skeleton_ckpt_key(key)
            host = self.skeleton_state_dict.get(ckpt_key)
            if host is None:
                raise RuntimeError(
                    f"{key} is not in the skeleton state dict (looked up {ckpt_key})")
            if int(host.shape[0]) != vocab:
                raise RuntimeError(
                    f"{key} has {host.shape[0]} rows, expected the vocab size {vocab}")
            hosts[key] = host
        freed = (embed.weight.numel() * embed.weight.element_size()
                 + head.weight.numel() * head.weight.element_size())
        self.model.model.embed_tokens = VocabParallelEmbedding(
            hosts["model.embed_tokens.weight"][start:end].contiguous().to(device=device),
            start, self._attn_tp_group, padding_idx=getattr(embed, "padding_idx", None),
        )
        self.model.lm_head = VocabParallelLMHead(
            hosts["lm_head.weight"][start:end].contiguous().to(device=device),
            vocab, G, self._attn_tp_group,
        )
        freed -= (self.model.model.embed_tokens.weight.numel() + self.model.lm_head.weight.numel()) \
            * self.model.lm_head.weight.element_size()
        del embed, head
        self._vocab_tp_sharded = True
        torch.cuda.empty_cache()
        if self.rank == 0:
            logging.info(
                "[K3] embed_tokens/lm_head vocab-parallel over TP%d: rows [%d, %d) of %d, "
                "%.2f GiB/rank released", G, start, end, vocab, freed / (1024 ** 3),
            )

    def _shard_latent_projections(self):
        """Keep only this rank's column slice of ``routed_expert_down_proj``
        (and the matching input rows of ``routed_expert_up_proj``) on the GPU.

        Enabled by the planner (``k3_shard_latent_projections``) on TP8 K3.
        The skeleton loader materialized the full tensors; the shm host
        tensors stay referenced so streamed-SP8 prefill (row-sharded tokens,
        needs the full projections) can build temporary full modules per
        phase. Resident decode / resident-EP prefill consume the slices
        through ``ResidentEPMXFP4MoELayer`` (set_latent_tp).
        """
        self._latent_tp_sharded = False
        self._latent_full_host = {}
        G = self._attn_tp_size
        if not self._latent_tp_planned:
            return
        if self._attn_tp_group is None:
            raise RuntimeError(
                "latent projections are planned TP-sharded, but the attention "
                "TP group was not built; the prefill mode was chosen for "
                "sharded projections"
            )
        device = self.engine_config.Basic_Config.device_torch
        c = self._attn_tp_rank
        n_layers = 0
        freed = 0
        for layer_idx, layer in enumerate(self.model.model.layers):
            moe = getattr(layer, "block_sparse_moe", None)
            down = getattr(moe, "routed_expert_down_proj", None) if moe is not None else None
            up = getattr(moe, "routed_expert_up_proj", None) if moe is not None else None
            if down is None or up is None:
                continue
            k_latent = int(down.weight.shape[0])
            if k_latent % G != 0:
                raise RuntimeError(
                    f"latent width {k_latent} is not divisible by the TP size {G}")
            cols = k_latent // G
            for name, module, slicer in (
                ("routed_expert_down_proj", down,
                 lambda w: w[c * cols:(c + 1) * cols, :]),
                ("routed_expert_up_proj", up,
                 lambda w: w[:, c * cols:(c + 1) * cols]),
            ):
                key = f"model.layers.{layer_idx}.block_sparse_moe.{name}.weight"
                ckpt_key = self._skeleton_ckpt_key(key)
                host = self.skeleton_state_dict.get(ckpt_key)
                if host is None:
                    raise RuntimeError(
                        f"latent projection {key} is not in the skeleton state dict "
                        f"(looked up {ckpt_key})")
                self._latent_full_host[(layer_idx, name)] = host
                freed += module.weight.numel() * module.weight.element_size()
                _replace_param(module, "weight", slicer(host).contiguous().to(device=device))
                freed -= module.weight.numel() * module.weight.element_size()
            down.out_features = cols
            up.in_features = cols
            n_layers += 1
        self._latent_tp_sharded = n_layers > 0
        if self._latent_tp_sharded:
            torch.cuda.empty_cache()
            if self.rank == 0:
                logging.info(
                    "[K3] latent projections TP%d-sharded on %d MoE layers: "
                    "%.2f GiB/rank released", G, n_layers, freed / (1024 ** 3))

    def _streamed_full_latent_projection(self, layer_idx, name):
        """Temporary full-weight module for streamed-SP8 prefill (its rows are
        sharded across the node, so it needs the unsharded projection). Built
        from the shm host tensor per prefill phase, dropped on release."""
        cache = getattr(self, "_streamed_full_latent", None)
        if cache is None:
            cache = self._streamed_full_latent = {}
        mod = cache.get((layer_idx, name))
        if mod is None:
            host = self._latent_full_host[(layer_idx, name)]
            device = self.engine_config.Basic_Config.device_torch
            mod = torch.nn.Linear(host.shape[1], host.shape[0], bias=False, device="meta")
            _replace_param(mod, "weight", host.to(device=device))
            cache[(layer_idx, name)] = mod
        return mod

    def _drop_streamed_full_latent_projections(self):
        cache = getattr(self, "_streamed_full_latent", None)
        if cache:
            cache.clear()
            torch.cuda.empty_cache()

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
        # The topology the live groups were built for. It is immutable for the
        # life of the process today (both values are fixed at construction),
        # so a later call that disagrees is a lifecycle error, not a request
        # to rebuild: say so by name instead of handing back groups that no
        # longer describe the world.
        signature = (int(self.world_size), int(self._attn_tp_size))
        if self._attn_tp_groups_built:
            if signature != self._attn_tp_group_signature:
                raise RuntimeError(
                    "attention TP topology changed in-process after the NCCL "
                    "sub-groups were built: they were created for "
                    "(world_size, attention_group_size)="
                    f"{self._attn_tp_group_signature} and this call asks for "
                    f"{signature}. The groups are world-level and cannot be "
                    "rebuilt here (torch drops a group only at "
                    "destroy_process_group, which no phase switch calls, and "
                    "the live ones are already baked into captured decode "
                    "graphs), so the new topology is refused rather than "
                    "silently served with the old communicators."
                )
            # Build ONCE per process. A NCCL sub-group is a world-level
            # resource: ``dist.new_group`` keeps every group it creates in
            # torch's global registry for the life of the process (only
            # ``destroy_process_group`` drops one, which no phase switch
            # calls), and each communicator's device buffers live OUTSIDE the
            # caching allocator. ``_build_model`` runs again on every rebuild
            # of a phase instance, so rebuilding these groups there added a
            # fresh set of communicators per cycle that nothing held, nothing
            # freed, and no torch counter could see — a driver-level leak that
            # only surfaced as an OOM at the largest later allocation (the KV
            # pool's ``cuMemCreate``). Unlike the model-attached state, these
            # are deliberately NOT reset by ``_reset_model_attached_state``:
            # the world did not change, so the groups are still the right ones.
            #
            # Collective-safe: ``new_group`` must be called by every rank in
            # the same order, and this flag records THIS process's own build —
            # every rank therefore takes the same branch on the same rebuild.
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
        self._build_stream_span_group()
        self._build_cross_weight_group()
        # Every family is up; later rebuilds reuse them (see the guard above),
        # and the signature records the topology they belong to.
        self._attn_tp_groups_built = True
        self._attn_tp_group_signature = signature

    def _build_stream_span_group(self):
        """Build the streamed-prefill span groups.

        COLLECTIVE. With S == G the span is the DP rank, so the node-local
        weight group above serves it and no group is created: the creation
        order of every existing topology is unchanged. With S > G every rank
        creates the ``world / S`` span groups in the same order and keeps
        its own.
        """
        S = self._stream_span
        if not S:
            return
        if S == self._attn_tp_size:
            self._stream_span_group = self._weight_tp_group
            return
        import torch.distributed as dist

        for span in range(self.world_size // S):
            grp = dist.new_group(ranks=list(range(span * S, (span + 1) * S)))
            if span == self.global_rank // S:
                self._stream_span_group = grp

    def _build_cross_weight_group(self):
        """Build the cross-span weight groups (hierarchical GDR only).

        COLLECTIVE, like the families above: every rank creates all ``S``
        groups in the same order and keeps the one it belongs to. They are
        created LAST so that the creation order of the attention and
        node-local weight groups is unchanged for host_rdma.

        Group ``g`` is ``[g + n*S for n in range(num_spans)]`` — span slot
        ``g`` of every span, i.e. exactly the ranks that stream experts
        ``[g*E/S, (g+1)*E/S)``. Its source is slot ``g`` of span
        ``(g * num_spans) // S``, so the sources spread evenly over the spans
        (two per span on four spans, four per span on two spans with S = 8)
        and every span drives an egress stream instead of one fanning out
        every expert.
        """
        if not self._hierarchical_gdr:
            return
        import torch.distributed as dist

        S = self._stream_span
        num_spans = self.world_size // S
        for g in range(S):
            grp = dist.new_group(
                ranks=[g + span * S for span in range(num_spans)]
            )
            if g == self._stream_span_rank:
                self._cross_weight_group = grp
                self._cross_weight_root = ((g * num_spans) // S) * S + g
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
            # M3 Slice 3: expands a committed latent into this layer's pool
            # slots through the SAME kv_b_proj the prefill path uses.
            attn.mla_materialize_pool_segment = types.MethodType(
                mla_materialize_pool_segment, attn
            )
            if getattr(attn, "indexer", None) is not None:
                # K3.1: the auxiliary region's pages ARE the index keys, so
                # the second half of a dual-region prefix load is a write into
                # this layer's indexer pool row.
                attn.dsa_materialize_pool_segment_index_keys = types.MethodType(
                    dsa_materialize_pool_segment_index_keys, attn
                )
            attn.attn_tp_size = self._attn_tp_size
            attn.attn_tp_rank = self._attn_tp_rank
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

    def phase_enter(self, key: PhaseKey, ctx: PhaseEnterContext):
        """Enter ``key.phase`` and return its ``(model, weight_copy_task)``.

        A kept phase instance is activated in place; otherwise the existing
        configure method builds it. Decode context is validated before either
        path so a caller cannot silently build buffers with missing sizing.
        """
        if key.phase == "prefill":
            if self.prefill_instance is not None:
                return self.activate_prefill()
            return self.configure_prefill()
        if key.phase == "decode":
            if ctx.comm is None:
                raise ValueError(
                    "decode phase_enter requires PhaseEnterContext.comm"
                )
            if ctx.max_num_seq is None:
                raise ValueError(
                    "decode phase_enter requires PhaseEnterContext.max_num_seq"
                )
            if self.decode_instance is not None:
                return self.activate_decoding()
            return self.configure_decoding(
                padding_bsz=ctx.max_num_seq, comm=ctx.comm
            )
        raise ValueError(f"phase_enter: unsupported phase {key.phase!r}")

    def admit(self, key: PhaseKey, ctx: AdmitContext) -> None:
        """Prepare one admission wave for the live phase instance.

        Kimi-Linear has no manager-owned per-wave reset or buffer-init helper;
        its streamed prefill buffer is phase-enter state and is initialized by
        ``_enter_prefill``. Decode admission is worker-owned because KV
        allocation and local row placement do not belong to this manager.
        """
        if key.phase == "prefill":
            return
        if key.phase == "decode":
            return
        raise ValueError(f"admit: unsupported phase {key.phase!r}")

    def phase_exit(self, key: PhaseKey, mode: ExitMode) -> None:
        """Leave ``key.phase`` in ``mode`` while retaining reusable state.

        One model serves both phases, so RETIRE of either phase retires that
        model: what the configure steps attached to it (the streamed-SP8
        buffer and its pending prefetch, the resident EP shard and the decode
        graph that bakes its addresses) is released while the model is still
        reachable, and both instance tuples go with it, so no kept instance
        outlives the model the worker's ``deep_free_model_memory`` then frees.
        On decode SUSPEND, only routed experts are released; KDA state pools
        are manager-owned and survive so the next activation keeps recurrent
        and convolution state intact.
        """
        if key.phase not in ("prefill", "decode"):
            raise ValueError(f"phase_exit: unsupported phase {key.phase!r}")
        if mode is ExitMode.RETIRE:
            self._release_streamed_sp8_prefill()
            self._release_resident_ep_decode()
            self.prefill_instance = None
            self.decode_instance = None
            self._active_phase = None
            self.model = None
            self.weight_copy_task = {}
            return
        if mode is not ExitMode.SUSPEND:
            raise ValueError(f"phase_exit: unsupported mode {mode!r}")
        if key.phase == "prefill":
            return
        # Worker guard at ``batchgen/batchgen`` + ``_worker.py:12076`` requires
        # a kept decode instance with resident experts before release.
        if not self.decode_experts_resident:
            raise RuntimeError(
                "decode SUSPEND needs a resident reusable decode instance; "
                "a rank without one must RETIRE"
            )
        self.release_decode_routed_experts()

    def holds_instance(self, phase: str) -> bool:
        """Return whether a resumable instance is retained for ``phase``."""
        if phase == "prefill":
            return self.prefill_instance is not None
        if phase == "decode":
            return self.decode_instance is not None
        raise ValueError(f"holds_instance: unsupported phase {phase!r}")
