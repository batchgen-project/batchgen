"""Decode CUDA-graph state of one ``BatchGenWorker`` rank.

``GraphRuntime`` exclusively owns the 18 pieces of coupled graph state that
used to be loose worker attributes: the manager/adapter handles, the
whole-model capture flags (model-agnostic and GLM-5), the capture-once
counter, and the span-warn rate limiter.

It also owns every ``batchgen.cuda_graph`` import the worker used to make:
the manager/bucketing constructors, the adapter ABC used by the startup type
backstop, the per-step ``BatchState`` snapshot and the model-agnostic
eager-vs-graph compare. The one exception is the decode bucket ladder
(``batchgen.cuda_graph.buckets``: ``decode_graph_bucket_sizes``,
``check_decode_width_within_buckets``, ``DecodeWidthExceedsGraphBucketError``),
which the worker imports directly since the rebase onto main 1c372209;
routing it through this class is a follow-up, not part of the rebase.

What this class is NOT, in this slice: a capture/replay engine. The capture,
eligibility, and replay logic stays in the worker and reads this object's
fields; only the state (and the imports) moved. The deeper move of that logic
behind ``ModelCudaGraphAdapter`` is a later slice -- it is not behavior-neutral
(GPT-OSS ships no adapter, admission reads graph state, and the worker's seam
is wider than the ABC).

Each ``batchgen.cuda_graph`` import below sits inside the method that needs
it, exactly where the worker made it: the imports are late-bound on purpose
(a test monkeypatches ``batchgen.cuda_graph.CUDAGraphManager`` between calls),
and module-level imports here would silently break that binding.

Imports: stdlib + ``torch``-level ``batchgen.cuda_graph`` (function-local).
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

from batchgen.config.graph_span_policy import graph_context_span_cap_tokens
from batchgen.config.model_name_utils import is_kimi_k25_backend_model
from batchgen.cuda_graph.buckets import check_decode_width_within_buckets


class GraphRuntime:
    """The worker's decode-graph state machine, as owned state.

    One asymmetry of today's lifecycle is reproduced here exactly and is
    PINNED, NOT ENDORSED -- fixing it would change behavior and belongs in
    its own slice:

    * the GLM-5 capture-OOM handler nulls the whole manager but adds only the
      failing bucket to ``glm5_whole_model_graph_failed_buckets``, so the
      buckets already captured in that episode are dropped with the manager
      yet stay eligible for the next warmup.

    ``decode_graph_captures`` is the capture-once guard of a reused decode
    instance. It is reset to 0, together with the captured-graph fields, by
    the worker's ``_release_decode_graph_state``, which the deep free calls
    only when the decode instance is NOT held; a held instance keeps all of
    its graph state, the counter included, across the deep free.
    """

    def __init__(self) -> None:
        # --- core manager / adapter state -------------------------------
        self.cuda_graph_manager = None
        # Phase C: cuda-graph adapter (cached from
        # initializer.get_cuda_graph_adapter()) becomes the default and only
        # decode-graph path. When an adapter is present it owns eligibility /
        # replay-inputs / KV staging; the legacy `glm5_*` whole-model code
        # path is unreachable for models that ship an adapter. Phase B's
        # BATCHGEN_DECODE_GRAPH_ADAPTER_DUAL env gate is retired -- adapters
        # are now the production path, not a migration toggle.
        self.cuda_graph_adapter = None

        # --- whole-model graph state (model-agnostic names) -------------
        self.whole_model_graph = False
        # These four were never set in the worker's __init__: every read
        # before the first capture went through a getattr default. The field
        # initializers below ARE those defaults, so a direct read is exactly
        # what the getattr returned.
        self.whole_model_bucketing = None
        self.whole_model_segment = None
        # Absent => "glm5_whole_model" at the single read site, which spells
        # that default itself (`... or "glm5_whole_model"`). This field must
        # stay None here: a getattr-with-default against an initialized field
        # would hand the read a None segment name and break GLM-5 replay.
        self.whole_model_segment_name = None
        self.k25_whole_model_graph = False
        # Whole-model decode graph capture episodes completed by this process.
        # On the reuse lifecycle it must stay at 1: the switch invariants and
        # the capture-once guard in _setup_cuda_graphs read it.
        self.decode_graph_captures = 0
        # Monotone stamp of the last "step exceeds the captured span" WARN.
        self.decode_graph_span_warn_monotonic = None

        # --- GLM-5 whole-model state ------------------------------------
        self.glm5_whole_model_graph = False
        self.glm5_whole_model_graph_failed_buckets = set()
        self.glm5_whole_model_graph_capture_attempted_for_batch = False
        self.glm5_whole_model_graph_state_change_after_capture_logged = False
        self.glm5_whole_model_graph_signature = None
        self.glm5_whole_model_graph_unavailable_reason = None
        # Late-init in the worker too (getattr defaults False / 0 / None).
        self.glm5_whole_model_all_short = False
        self.glm5_whole_model_index_topk = 0
        self.glm5_whole_model_capture_input_ids = None

    # ------------------------------------------------------------------ #
    # cuda_graph-layer constructors
    # ------------------------------------------------------------------ #

    def new_bucketing(self, bucket_sizes):
        """``BatchSizeBucketing`` over ``bucket_sizes``."""
        from batchgen.cuda_graph import BatchSizeBucketing

        return BatchSizeBucketing(bucket_sizes)

    def new_manager(self, bucketing, *, device):
        """A fresh ``CUDAGraphManager`` for ``bucketing`` on ``device``."""
        from batchgen.cuda_graph import CUDAGraphManager

        return CUDAGraphManager(bucketing, device=device)

    # ------------------------------------------------------------------ #
    # adapter handle
    # ------------------------------------------------------------------ #

    def adopt_adapter(self, initializer, *, model_name) -> None:
        """Fetch and type-check the model's cuda-graph adapter, once.

        Pulls the adapter from the initializer (None if the model does not
        implement one) and stores it. ``model_name`` names the model in the
        rejection message; the worker passes its own.
        """
        self.cuda_graph_adapter = getattr(
            initializer, "get_cuda_graph_adapter", lambda: None
        )()
        if self.cuda_graph_adapter is not None:
            # Every model's graph behavior must come through the
            # ModelCudaGraphAdapter ABC. An object that merely happens to carry
            # the right method names would pass every call site here and then
            # fail mid-decode on the first hook it does not implement, so the
            # type is asserted once, at startup, on the fetch itself.
            from batchgen.cuda_graph.adapter import ModelCudaGraphAdapter

            if not isinstance(self.cuda_graph_adapter, ModelCudaGraphAdapter):
                adapter_type = type(self.cuda_graph_adapter)
                raise RuntimeError(
                    "get_cuda_graph_adapter() for model "
                    f"{model_name!r} returned "
                    f"{adapter_type.__module__}.{adapter_type.__qualname__}, "
                    "which is not a ModelCudaGraphAdapter. Every model's "
                    "graph behavior must come through this ABC: subclass "
                    "batchgen.cuda_graph.adapter.ModelCudaGraphAdapter "
                    "instead of duck-typing it."
                )
            logging.info(
                "Cuda-graph adapter discovered: %s (advertises %s)",
                type(self.cuda_graph_adapter).__name__,
                [m.value for m in self.cuda_graph_adapter.advertised_modes()],
            )

    # ------------------------------------------------------------------ #
    # per-step seam objects
    # ------------------------------------------------------------------ #

    def make_batch_state(self, **fields):
        """The per-step ``BatchState`` snapshot the adapter seam consumes."""
        from batchgen.cuda_graph.adapter import BatchState

        return BatchState(**fields)

    def compare_decode_outputs(self, **kwargs):
        """Model-agnostic eager-vs-graph compare (observability only)."""
        from batchgen.cuda_graph.compare import compare_decode_outputs

        return compare_decode_outputs(**kwargs)

    # ------------------------------------------------------------------ #
    # teardown
    # ------------------------------------------------------------------ #

    def drop_refs_for_shutdown(self) -> None:
        """Drop the graph refs that retain model / segment / KV memory.

        The order is load-bearing and is the ordered-shutdown's own: the
        adapter's ``_ctx`` holds model, segment and KV-manager references, so
        it goes first; then the managers and the captured segment. Python's
        attribute destruction order is not an ownership contract, which is why
        the shutdown path spells it out.
        """
        self.cuda_graph_adapter = None
        self.cuda_graph_manager = None
        self.whole_model_segment = None

    # ------------------------------------------------------------------ #
    # capture orchestration (moved verbatim from BatchGenWorker; the
    # worker remains the sole state owner and is rebound as `self`)
    # ------------------------------------------------------------------ #

    def _setup_cuda_graphs(self, worker, gpu_manager):
        """Capture CUDA graphs for decode: full attention block per layer.

        Each graph captures the entire attention block in one shot:
          RMSNorm → QKV proj → split → reshape → RoPE → KV write → FA → O_proj
          → residual add + post-attn RMSNorm

        Dynamic metadata (cache_seqlens) is passed as a static-address input buffer.
        KV cache, page table, and cos/sin tables are at fixed GPU addresses.
        """
        # Moved VERBATIM from batchgen_worker.py (repatriation); rebinding self to
        # the worker keeps every original self. spelling. The engine is stateless.
        self = worker
        from batchgen.models.openai.gpt_oss_120b.cuda_graph_segments import (
            FullAttnSegment, MoESegment, MoEComputeSegment, SharedMoEBufferPool,
            WholeModelSegment,
        )
        from batchgen.models.wrappers.attention import AttnWrapperBase

        # Detect K2.5 model for specialized graph segment
        _is_k25 = is_kimi_k25_backend_model(self.model_name)

        # The ladder covers the per-rank decode width: decode admission is
        # bounded by that width, so every admitted step has a bucket to
        # replay and a step above the top bucket is an invariant violation,
        # not an eager fallback.
        bucket_sizes = self._decode_graph_bucket_sizes()
        decode_width = self._max_decode_rank_bsz()
        check_decode_width_within_buckets(decode_width, bucket_sizes)
        if not getattr(self, "_decode_graph_ladder_logged", False):
            logging.info(
                "CUDA graph bucket sizes: %s (decode_width=%d, "
                "--cuda-graph-max-bucket-size=%d, --cuda-graph-num-buckets=%d)",
                bucket_sizes,
                decode_width,
                int(self.args.cuda_graph_max_bucket_size),
                int(self.args.cuda_graph_num_buckets),
            )
            self._decode_graph_ladder_logged = True
        bucketing = self._graph.new_bucketing(bucket_sizes)
        manager = self._graph.new_manager(bucketing, device=self.torch_device)

        # Use model's max_position_embeddings (not the deployment max_model_len) so the
        # RoPE cos/sin cache captured in the graph covers ALL possible positions.
        max_rope_len = getattr(self.model_config, 'max_position_embeddings', 131072)
        # Phase C: only the whole-model graph remains for GLM-5 (the dsa / moe /
        # dsa_full / layer modes are retired and their branches deleted).
        glm5_whole_graph_enabled = (
            self._glm5_whole_model_graph_requested_for_current_batch()
        )
        if glm5_whole_graph_enabled:
            # The request is the requirement: requested here means required.
            whole_graph_required = glm5_whole_graph_enabled
            local_bsz = int(getattr(self, "_current_decode_local_batch_size", 0) or 0)
            max_bsz = int(getattr(self, "_current_decode_max_rank_batch_size", 0) or 0)
            capture_attempted = bool(
                self._graph.glm5_whole_model_graph_capture_attempted_for_batch
            )
            if max_bsz <= 0:
                logging.info(
                    f"Rank {self.rank}: no global GLM-5 decode rows; skipping whole-model graph capture"
                )
                return
            failed_buckets = self._graph.glm5_whole_model_graph_failed_buckets
            capture_buckets = [
                int(bucket)
                for bucket in bucketing.bucket_sizes
                if int(bucket) not in failed_buckets
            ]
            if not capture_buckets:
                self._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
                if whole_graph_required:
                    raise RuntimeError("GLM-5 whole-model CUDA graph required but all configured buckets failed")
                return
            cur_batch = getattr(AttnWrapperBase, "cur_batch", None) or []
            if len(cur_batch) != local_bsz:
                logging.info(
                    f"Rank {self.rank}: GLM-5 whole-model graph capture deferred until "
                    "decode wrapper state is bound"
                )
                return
            existing_manager = self._graph.cuda_graph_manager
            existing_signature = self._glm5_whole_model_graph_capture_signature()
            if existing_manager is not None and self._graph.glm5_whole_model_graph:
                missing_configured = any(
                    not existing_manager.has_bucket_for_all_segments(bucket)
                    for bucket in capture_buckets
                )
                if (
                    not missing_configured
                    and existing_signature == self._graph.glm5_whole_model_graph_signature
                ):
                    self._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
                    return
                # A recapture armed by a changed criticality probe request is
                # the one deliberate exception: it falls through to the stale
                # graph release below and captures in place.
                if capture_attempted and not getattr(self, "_criticality_probe_recapture", False):
                    logging.info(
                        f"Rank {self.rank}: GLM-5 whole-model CUDA graph state changed after "
                        "configure-time capture; using eager decode instead of recapturing"
                    )
                    return
            self._assert_decode_graph_capture_allowed()

            from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
                Glm5WholeModelSegment,
                make_glm5_whole_model_graph_segment_name,
            )
            from batchgen.models.glm.glm5.cuda_graph_segments import Glm5FullDsaAttnSegment
            from batchgen.models.glm.glm5.layer_cuda_graph_segments import (
                Glm5DecoderLayerGraphSegment,
            )
            from batchgen.models.glm.glm5.model import Glm5MoE, resolve_glm5_3d_mtp
            from batchgen.models.glm.glm5.moe_cuda_graph_segments import (
                Glm5MoEGraphBufferPool,
                Glm5MoEGraphSegment,
            )

            primary_manager = getattr(gpu_manager, "primary", gpu_manager)
            aux_manager = getattr(
                gpu_manager,
                "auxiliary",
                getattr(self.core_engine, "gpu_paged_kv_manager_aux", None),
            )
            if aux_manager is None:
                raise RuntimeError("GLM-5 whole-model CUDA graph requested but auxiliary GPU KV manager is missing")
            active_sequence_ids = list(cur_batch)
            primary_manager.ensure_cuda_graph_page_table(active_sequence_ids)
            aux_manager.ensure_cuda_graph_page_table(active_sequence_ids)
            primary_page_table = primary_manager.get_cuda_graph_page_table_storage()
            aux_page_table = aux_manager.get_cuda_graph_page_table_storage()
            if primary_page_table is None or aux_page_table is None:
                raise RuntimeError(
                    "GLM-5 whole-model CUDA graph requested but GPU page-table storage is not initialized"
                )
            primary_page_size = int(primary_manager.config.page_size_tokens)
            aux_page_size = int(aux_manager.config.page_size_tokens)
            # The span comes from the DEPLOYMENT bound, never from the batch in
            # front of us: the first decode batch used to fix it for the life
            # of the process, so a later, longer turn had no replayable graph.
            # A 1-byte aux element type is the page-split FP8 layout, which
            # decides which scorer owns the context-proportional statics.
            aux_is_page_split = int(aux_manager.config.kv_dtype.itemsize) == 1
            capacity_seqlen = self._glm5_dsa_graph_score_capacity_tokens(
                primary_page_table,
                primary_page_size,
                aux_page_table,
                aux_page_size,
                deployment_context_tokens=self._deployment_context_bound_tokens(),
                span_cap_tokens=graph_context_span_cap_tokens(
                    bucket_sizes=bucketing.bucket_sizes,
                    aux_is_page_split=aux_is_page_split,
                ),
            )
            graph_max_seqlen = int(capacity_seqlen)
            # A batch already past the span is not a capture error: capturing
            # at the span still serves every shorter step, and this step takes
            # the eager path (see the decode-step fallback).
            current_capture_max_seqlen = int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)
            if current_capture_max_seqlen > graph_max_seqlen:
                self._warn_decode_graph_span_exceeded(
                    current_capture_max_seqlen, graph_max_seqlen
                )
            # all_short lifetime: every admissible row's final length fits in
            # index_topk, so the DSA segments attend densely to the resident
            # page cache and skip indexer scoring/top-k for the graph lifetime.
            # index_topk comes from the live indexer module — the value that
            # actually drives scoring; a local-checkpoint model_config may not
            # carry the field at all.
            required_seqlen = self._glm5_dsa_graph_required_tokens(
                active_sequence_ids,
                page_size=primary_page_size,
            )
            _first_indexer = next(
                (
                    getattr(layer.self_attn.module, "indexer", None)
                    for layer in self.model.model.layers
                    if getattr(layer.self_attn.module, "indexer", None) is not None
                ),
                None,
            )
            index_topk_cfg = int(getattr(_first_indexer, "index_topk", 0) or 0)
            # ONE general graph lifetime (POIS 2026-10-13). The all_short lifetime
            # is a second, narrower lifetime: it only serves rows whose final
            # length fits in index_topk, so a batch that grows past that bound has
            # to fall back to eager or force a recapture. That is why the default
            # path recaptured every bucket on every phase switch, and why the
            # persistent path rejected long rows outright. Pinning the selection to
            # the general lifetime makes the decode graphs capture-once and
            # length-agnostic (their span still comes from KV capacity via
            # graph_max_seqlen above), at the cost of always scoring the indexer.
            # The all_short capture code itself is left in place and never selected.
            all_short = False
            self._graph.glm5_whole_model_all_short = all_short
            self._graph.glm5_whole_model_index_topk = index_topk_cfg
            if self.rank == 0:
                logging.info(
                    "Rank %s: GLM-5 whole-model graph attention span "
                    "required_seqlen=%s index_topk=%s all_short=%s "
                    "(general lifetime pinned)",
                    self.rank,
                    required_seqlen,
                    index_topk_cfg,
                    all_short,
                )

            # Model-wrapper class state goes through the worker binder. The
            # engine module does not write the wrapper classes.
            self._bind_graph_capture_kv_managers(primary_manager, aux_manager)
            primary_k_cache, _ = primary_manager.get_kv_tensors()
            aux_k_cache, _ = aux_manager.get_kv_tensors()
            rank_counts = getattr(self, "_current_decode_rank_token_counts", None)
            if rank_counts is None:
                rank_counts = torch.full(
                    (self.world_size,),
                    local_bsz,
                    dtype=torch.int64,
                    device=self.torch_device,
                )
            for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                wrapper = decoder_layer.self_attn
                indexer = getattr(wrapper.module, "indexer", None)
                if indexer is None and layer_idx == 0:
                    # Reuse-topk segments need a producing full layer first.
                    raise RuntimeError(
                        "Layer 0: GLM-5 whole-model graph requires a DSA "
                        "indexer on the first layer")
                if getattr(wrapper, "_fp8_absorb_weights", None) is None:
                    wrapper.initialize_decode_absorb()
                if indexer is not None and (
                    getattr(wrapper, "_fused_wqb_weights", None) is None
                    or getattr(wrapper, "_indexer_cuda_module", None) is None
                ):
                    # Indexer-only kernels; GLM-5.2 skip layers have none and
                    # get Glm5ReuseTopkAttnSegment instead.
                    wrapper.initialize_fused_kernels()
                if getattr(wrapper, "_fp8_absorb_weights", None) is None:
                    raise RuntimeError(f"Layer {layer_idx}: GLM-5 whole-model graph requires FP8 absorb weights")
                if (
                    getattr(wrapper, "_fp8_qkv_a_proj", None) is None
                    or getattr(wrapper, "_fp8_qkv_a_scale", None) is None
                ):
                    raise RuntimeError(
                        f"Layer {layer_idx}: GLM-5 whole-model graph requires "
                        "fused Q-A/KV-A weights"
                    )
                if indexer is not None:
                    if getattr(wrapper, "_fused_wqb_weights", None) is None:
                        raise RuntimeError(f"Layer {layer_idx}: GLM-5 whole-model graph requires fused WQB weights")
                    if getattr(wrapper, "_indexer_cuda_module", None) is None:
                        raise RuntimeError(f"Layer {layer_idx}: GLM-5 whole-model graph requires fused indexer CUDA module")
            moe_not_ready = []
            for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                mlp = getattr(decoder_layer, "mlp", None)
                if hasattr(mlp, "experts_per_rank") and not getattr(mlp, "_fp8_blockwise_ready", False):
                    moe_not_ready.append(layer_idx)
            if moe_not_ready:
                reason = (
                    "GLM-5 whole-model CUDA graph requires all local MoE experts "
                    "to be persistent and stacked for the 3D FP8 path; unavailable "
                    f"for layers {moe_not_ready[:5]}{'...' if len(moe_not_ready) > 5 else ''}. "
                    "Single-node partial-persistent expert configs can run eager/mixed "
                    "decode, but cannot validate the real whole-model graph."
                )
                self._graph.glm5_whole_model_graph_unavailable_reason = reason
                if whole_graph_required:
                    raise RuntimeError(reason)
                logging.warning("%s Using eager decode without whole-model graph compare.", reason)
                return

            if self._graph.glm5_whole_model_graph or self._graph.whole_model_segment is not None:
                logging.info(
                    f"Rank {self.rank}: releasing stale GLM-5 whole-model CUDA graph "
                    "before configure-time capture"
                )
                self._release_glm5_whole_model_graph_state(empty_cuda_cache=True)

            manager = self._graph.new_manager(bucketing, device=self.torch_device)
            moe_layers = [
                layer.mlp for layer in self.model.model.layers
                if isinstance(getattr(layer, "mlp", None), Glm5MoE)
            ]
            moe_pool = None
            if moe_layers:
                first_moe = moe_layers[0]
                moe_pool = Glm5MoEGraphBufferPool(
                    world_size=self.world_size,
                    hidden_size=first_moe.hidden_size,
                    num_experts_per_tok=first_moe.num_experts_per_tok,
                    num_local_experts=first_moe.experts_per_rank,
                    intermediate_size=first_moe.config.moe_intermediate_size,
                    device=self.torch_device,
                    bucket_sizes=bucket_sizes,
                    # Graph decode fan-in is bounded by the largest captured
                    # bucket across all ranks; the pool still takes
                    # max(base_mtp, round_up(world_size x max_bucket)).
                    base_mtp=resolve_glm5_3d_mtp(
                        self.world_size * max(bucket_sizes)
                    ),
                )
            shared_dsa_buffers = {}
            shared_reuse_buffers = {}
            last_full_segment = None
            last_cos_table = None
            last_sin_table = None
            last_index_topk = None
            layer_segments = []
            for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                wrapper = decoder_layer.self_attn
                indexer = wrapper.module.indexer
                primary_blocked_k = primary_k_cache[layer_idx]
                if indexer is None:
                    # GLM-5.2 skip_topk layer: no indexer module. Reuse the
                    # most recent full layer's top-k (eager semantics,
                    # glm5_decode_selector.py:443-461) via the reuse segment.
                    # Rope tables are layer-invariant; borrow the producer's.
                    if last_full_segment is None:
                        raise RuntimeError(
                            "GLM-5 whole-model graph: first layer has no DSA "
                            "indexer; layer 0 must be a full indexer layer"
                        )
                    from batchgen.models.glm.glm5.reuse_topk_segment import (
                        Glm5ReuseTopkAttnSegment,
                    )
                    dsa_segment = Glm5ReuseTopkAttnSegment(
                        wrapper=wrapper,
                        primary_blocked_k=primary_blocked_k,
                        primary_page_table=primary_page_table,
                        absorb_weights=wrapper._fp8_absorb_weights,
                        cos_table=last_cos_table,
                        sin_table=last_sin_table,
                        max_seqlen=graph_max_seqlen,
                        index_topk=last_index_topk,
                        page_size=primary_page_size,
                        topk_source=last_full_segment,
                        all_short=all_short,
                        shared_buffers=shared_reuse_buffers,
                    )
                else:
                    # The aux pool may hold only the indexer layers (GLM-5.2):
                    # resolve the physical slot instead of indexing by engine layer.
                    aux_blocked_k = aux_k_cache[aux_manager.resolve_physical_layer(layer_idx)]
                    dummy = torch.empty(
                        1,
                        1,
                        indexer.rope_head_dim,
                        device=primary_blocked_k.device,
                        dtype=torch.bfloat16,
                    )
                    cos_table, sin_table = indexer.rotary_emb(dummy, seq_len=graph_max_seqlen)
                    dsa_segment = Glm5FullDsaAttnSegment(
                        wrapper=wrapper,
                        primary_blocked_k=primary_blocked_k,
                        aux_blocked_k=aux_blocked_k,
                        primary_page_table=primary_page_table,
                        aux_page_table=aux_page_table,
                        wq_b_weights=wrapper._fused_wqb_weights,
                        absorb_weights=wrapper._fp8_absorb_weights,
                        cuda_module=wrapper._indexer_cuda_module,
                        cos_table=cos_table,
                        sin_table=sin_table,
                        max_seqlen=graph_max_seqlen,
                        index_topk=indexer.index_topk,
                        page_size=primary_page_size,
                        aux_page_size=aux_page_size,
                        all_short=all_short,
                        shared_buffers=shared_dsa_buffers,
                    )
                    last_full_segment = dsa_segment
                    last_cos_table = cos_table
                    last_sin_table = sin_table
                    last_index_topk = indexer.index_topk
                moe_segment = None
                moe = getattr(decoder_layer, "mlp", None)
                if isinstance(moe, Glm5MoE):
                    moe_segment = Glm5MoEGraphSegment(
                        moe,
                        moe_pool,
                        moe.comm,
                        world_size=self.world_size,
                        rank=self.rank,
                        device=self.torch_device,
                    )
                layer_segments.append(
                    Glm5DecoderLayerGraphSegment(
                        layer=decoder_layer,
                        dsa_segment=dsa_segment,
                        moe_segment=moe_segment,
                        device=self.torch_device,
                        world_size=self.world_size,
                        capture_local_bsz=local_bsz,
                        capture_rank_token_counts=rank_counts,
                    )
                )
            vocab_size = getattr(self.model, 'vocab_size', None) or self.model.config.vocab_size
            hidden_size = self.model.config.hidden_size
            compare_probe_layers = self._glm5_whole_model_graph_probe_layers(
                len(self.model.model.layers)
            )
            whole_seg = Glm5WholeModelSegment(
                model=self.model,
                device=self.torch_device,
                world_size=self.world_size,
                max_pages_per_seq=primary_page_table.shape[1],
                max_aux_pages_per_seq=aux_page_table.shape[1],
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                max_bucket_size=bucketing._max_bucket,
                max_seqlen=graph_max_seqlen,
                include_embedding=True,
                include_lm_head=True,
                compare_probe_layers=compare_probe_layers,
                layer_segments=layer_segments,
                # e4m3 for a page-split FP8 indexer pool, bf16 for AoS: the aux
                # staging clone must not requantize the stream.
                aux_kv_staging_dtype=aux_manager.config.kv_dtype,
                stage_kv_for_offload=not self.glm5_disable_decode_host_kv_staging,
            )
            segment_name = make_glm5_whole_model_graph_segment_name()
            manager.register_segment(segment_name, whole_seg)
            first_wrapper = self.model.model.layers[0].self_attn.module
            num_heads = int(getattr(first_wrapper, "num_heads", 64))
            logging.info(
                f"Rank {self.rank}: capturing GLM-5 whole-model CUDA graph "
                f"segment={segment_name} buckets={capture_buckets}, "
                f"context_capacity={graph_max_seqlen}"
            )
            # Criticality probe: installed before warmup and capture, because
            # its spin nodes are issued by the segments' forward. Without a
            # request the adapter is not asked and nothing is constructed.
            self._criticality_probe = None
            probe_request = getattr(self, "_criticality_probe_request", None)
            if probe_request is not None:
                self._criticality_probe = self._graph.cuda_graph_adapter.install_criticality_probe(
                    probe_request,
                    whole_model_segment=whole_seg,
                    device=self.torch_device,
                )
            torch.cuda.synchronize(self.torch_device)
            dist.barrier()
            self._graph.cuda_graph_manager = manager
            self._graph.whole_model_graph = True
            self._graph.glm5_whole_model_graph = True
            self._graph.whole_model_bucketing = bucketing
            self._graph.whole_model_segment = whole_seg
            self._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
            try:
                # Largest-first: the big bucket primes the shared graph pool and
                # smaller buckets reuse its blocks instead of growing the pool.
                for capture_bucket in sorted(capture_buckets, reverse=True):
                    torch.cuda.synchronize(self.torch_device)
                    dist.barrier()
                    whole_seg.set_capture_inputs(
                        **self._make_glm5_whole_model_capture_inputs(
                            bucket=capture_bucket,
                            num_heads=num_heads,
                        )
                    )
                    manager.warmup_and_capture_buckets([capture_bucket])
                    torch.cuda.synchronize(self.torch_device)
                    dist.barrier()
            except torch.OutOfMemoryError as exc:
                manager.drop_bucket(capture_bucket)
                self._graph.glm5_whole_model_graph_failed_buckets.add(capture_bucket)
                self._graph.cuda_graph_manager = None
                self._graph.whole_model_segment = None
                self._graph.glm5_whole_model_all_short = False
                self._graph.glm5_whole_model_index_topk = 0
                self._graph.whole_model_bucketing = None
                self._graph.glm5_whole_model_capture_input_ids = None
                self._graph.whole_model_graph = False
                self._graph.glm5_whole_model_graph = False
                self._criticality_probe = None
                self._criticality_probe_recapture = False
                torch.cuda.empty_cache()
                if whole_graph_required:
                    raise
                logging.error(
                    f"Rank {self.rank}: GLM-5 whole-model CUDA graph configure-time "
                    f"capture for bucket BS={capture_bucket} ran out of memory; using eager decode: {exc}"
                )
                return
            self._graph.glm5_whole_model_graph_signature = self._glm5_whole_model_graph_capture_signature()
            self._record_decode_graph_capture()
            self._criticality_probe_recapture = False
            # Phase B: hand the just-captured segment to the adapter so its
            # eligibility() / prepare_replay_inputs() / stage_post_graph_kv()
            # can run against the same captured graph the legacy path uses.
            # Without this attach, adapter._ctx stays None and eligibility()
            # returns EAGER/adapter_not_built every step (the worker silently
            # falls back to the legacy path).
            if self._graph.cuda_graph_adapter is not None:
                try:
                    _adapter_max_seqlen = int(getattr(whole_seg, "max_seqlen", 0) or 0)
                    _adapter_gpu_mgr = gpu_manager
                    _attach = getattr(self._graph.cuda_graph_adapter, "attach_existing_segment", None)
                    if _attach is not None:
                        _attach(
                            model=self.model,
                            whole_model_segment=whole_seg,
                            bucketing=bucketing,
                            gpu_kv_manager=_adapter_gpu_mgr,
                            device=self.torch_device,
                            max_seqlen_cap=_adapter_max_seqlen,
                        )
                    for _bucket in capture_buckets:
                        _sig = self._graph.cuda_graph_adapter.capture_signature(
                            bucket=int(_bucket),
                            gpu_kv_manager=_adapter_gpu_mgr,
                            max_seqlen=_adapter_max_seqlen,
                        )
                        self._graph.cuda_graph_adapter.record_capture(
                            segment_name=segment_name,
                            bucket=int(_bucket),
                            signature=_sig,
                        )
                except Exception as _exc:
                    logging.warning(
                        "Phase B: failed to attach/record_capture on adapter: %s", _exc,
                    )
            # The decode graphs are captured and active, so the eager MoE
            # buffers these layers share (one set for the whole model) are dead
            # residency competing with the KV pool that
            # _finalize_gpu_kv_pool_after_capture sizes from the post-capture
            # steady usage. The eager fallback paths re-create them on demand
            # at their own entry, so this is a residency switch only.
            for _mlp in moe_layers[:1]:
                _eager_moe_freed = _mlp.release_eager_decode_buffers()
                if _eager_moe_freed and self.rank == 0:
                    logging.info(
                        "Rank %s: released the eager MoE decode buffers "
                        "(%.3f GiB) after graph capture",
                        self.rank,
                        _eager_moe_freed / (1024 ** 3),
                    )
            stats = manager.get_capture_stats()
            logging.info(
                f"Rank {self.rank}: GLM-5 whole-model CUDA graph ready in "
                f"{stats['total_capture_time_ms']:.0f}ms for buckets {capture_buckets}"
            )
            if self._glm5_whole_model_graph_timing_requested_for_current_batch():
                logging.info(
                    "[GLM5_WHOLE_GRAPH_TIMING] rank=%s buckets=%s capture_ms=%.3f",
                    self.rank,
                    capture_buckets,
                    stats["total_capture_time_ms"],
                )
            return

        # Every other decode graph path is under the same capture-once rule.
        self._assert_decode_graph_capture_allowed()

        # GPT-OSS-specific pre-warm and per-layer segment registration
        # K2.5 uses MLA (not GQA) and has its own segment class, skip per-layer setup
        has_moe_graph = False
        moe_pool = None
        if not _is_k25:
            # Pre-warm: initialize sinks and RoPE cache before capture
            for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                wrapper = decoder_layer.self_attn
                # Initialize sinks for persistent mode
                if wrapper.sinks is None and wrapper.persistent and hasattr(wrapper.module, 'sinks'):
                    wrapper.sinks = wrapper.module.sinks.data.to(self.torch_device)
                elif wrapper.sinks is not None:
                    wrapper.sinks = wrapper.sinks.to(self.torch_device)
                # Pre-warm RoPE cos/sin cache to max position embeddings
                dummy = torch.zeros(1, 1, wrapper.num_kv_heads, wrapper.head_dim, device=self.torch_device)
                wrapper.module.rotary_emb(dummy, seq_len=max_rope_len)

            # Register full attention segments
            # Use max possible pages based on max sequence length, not current state.
            # The page_table static buffer column width is baked into the graph —
            # if sequences grow beyond this during decode, FlashAttention reads
            # past the buffer causing illegal memory access.
            page_size_tokens = gpu_manager.config.page_size_tokens
            max_seq_len = self.model.config.max_position_embeddings
            max_pages = (max_seq_len + page_size_tokens - 1) // page_size_tokens
            for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                attn_wrapper = decoder_layer.self_attn
                seg = FullAttnSegment(decoder_layer, attn_wrapper, layer_idx, max_rope_len,
                                      max_pages, page_size_tokens)
                manager.register_segment(f"layer_{layer_idx}_full_attn", seg)

            # Register MoE segments with shared buffer pool (EP mode)
            for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                moe_decode = decoder_layer.mlp
                if (hasattr(moe_decode, 'persistent_expert_indices')
                        and len(moe_decode.persistent_expert_indices) > 0
                        and hasattr(moe_decode, 'comm') and moe_decode.comm is not None):
                    # Create shared pool once from first MoE layer's params
                    if moe_pool is None:
                        moe_pool = SharedMoEBufferPool(
                            world_size=self.world_size,
                            hidden_size=moe_decode.hidden_size,
                            total_experts=moe_decode.total_experts,
                            num_experts_per_tok=moe_decode.num_experts_per_tok,
                            num_local_experts=len(moe_decode.persistent_expert_indices),
                            N_intermediate=moe_decode.gate_weight_ref.shape[0],
                            device=self.torch_device,
                        )
                        moe_pool.setup(bucketing.bucket_sizes)
                    moe_seg = MoESegment(
                        moe_decode, moe_pool, moe_decode.comm,
                        self.world_size, self.rank, self.torch_device,
                    )
                    decoder_layer._moe_segment = moe_seg
                    decoder_layer._moe_bucketing = bucketing
                    # Register compute segment for graph capture.
                    # All_gather is graph-captured; all_reduce remains eager.
                    if not os.environ.get("BATCHGEN_MOE_EAGER"):
                        moe_compute_seg = MoEComputeSegment(
                            moe_decode, moe_pool, moe_decode.comm,
                            self.world_size, self.rank, self.torch_device,
                        )
                        manager.register_segment(f"layer_{layer_idx}_moe", moe_compute_seg)
                        has_moe_graph = True
        else:
            # K2.5: Pre-warm RoPE cache (shared instance)
            rotary_emb = self.model.model._shared_rotary_emb
            dummy = torch.zeros(1, 1, 1, rotary_emb.dim, device=self.torch_device)
            rotary_emb(dummy, seq_len=max_rope_len)
            # Compute max_pages for K2.5
            page_size_tokens = gpu_manager.config.page_size_tokens
            max_seq_len = self.model.config.max_position_embeddings
            max_pages = (max_seq_len + page_size_tokens - 1) // page_size_tokens

        # Set gpu_paged_kv_manager so segments can access it during capture
        # The engine module does not write the wrapper classes.
        self._bind_graph_capture_kv_manager(gpu_manager)

        # K2.5: whole-model graph via the cuda-graph adapter (minimal parallel
        # path). POIS decision: serialize the shared expert inline to reclaim
        # the eager-MoE launch overhead. Self-contained; leaves the GLM-5 /
        # GPT-OSS whole-model paths byte-for-byte untouched. Falls back to the
        # legacy per-layer K2.5 path below if no adapter is present.
        if _is_k25 and self._graph.cuda_graph_adapter is not None:
            from batchgen.models.moonshotai.kimi_k25.cuda_graph_adapter import (
                SEGMENT_NAME_WHOLE_MODEL as _K25_WM_NAME,
            )
            bundle = self._graph.cuda_graph_adapter.build_segments(
                model=self.model,
                bucketing=bucketing,
                gpu_kv_manager=gpu_manager,
                world_size=self.world_size,
                rank=self.rank,
                device=self.torch_device,
                max_seqlen_cap=max_rope_len,
            )
            whole_seg = bundle.whole_model
            manager.register_segment(_K25_WM_NAME, whole_seg)
            if self.rank == 0:
                logging.info(
                    f"CUDA graph capture (K2.5 whole-model): segment={_K25_WM_NAME} "
                    f"× {len(bucketing.bucket_sizes)} buckets {bucketing.bucket_sizes}"
                )
            # NCCL collectives are baked into the graph — all ranks must capture
            # the same buckets simultaneously.
            torch.cuda.synchronize(self.torch_device)
            dist.barrier()
            manager.warmup_and_capture_all()
            for _bucket in bucketing.bucket_sizes:
                _sig = self._graph.cuda_graph_adapter.capture_signature(
                    bucket=_bucket, gpu_kv_manager=gpu_manager, max_seqlen=max_rope_len,
                )
                self._graph.cuda_graph_adapter.record_capture(
                    segment_name=_K25_WM_NAME, bucket=_bucket, signature=_sig,
                )
            self._graph.cuda_graph_manager = manager
            self._graph.whole_model_graph = True
            self._graph.k25_whole_model_graph = True
            self._graph.whole_model_segment_name = _K25_WM_NAME
            self._graph.whole_model_bucketing = bucketing
            self._graph.whole_model_segment = whole_seg
            self._record_decode_graph_capture()
            if self.rank == 0:
                stats = manager.get_capture_stats()
                logging.info(
                    f"CUDA graphs ready (K2.5 whole-model): {stats['total_capture_time_ms']:.0f}ms"
                )
            return

        # Whole-model graph is the default for GPT-OSS.
        # K2.5 without an adapter falls back to per-layer (segmented) mode.
        if _is_k25:
            use_whole_model = False
        else:
            use_whole_model = True

        if use_whole_model:
            # Whole-model mode: single graph for entire decode pass.
            # Discard per-layer segments, register one WholeModelSegment instead.
            manager = self._graph.new_manager(bucketing, device=self.torch_device)

            vocab_size = getattr(self.model, 'vocab_size', None) or self.model.config.vocab_size
            hidden_size = self.model.config.hidden_size

            if _is_k25:
                # K2.5 uses MLA attention + 3D strided MoE — different segment class
                from batchgen.models.moonshotai.kimi_k25.cuda_graph_segments import K25WholeModelSegment
                whole_seg = K25WholeModelSegment(
                    model=self.model,
                    device=self.torch_device,
                    max_pages_per_seq=max_pages,
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    max_bucket_size=bucketing._max_bucket,
                )
            else:
                # GPT-OSS: GQA attention + SharedMoEBufferPool
                # Build MoE segments dict (layer_idx → MoESegment) for WholeModelSegment.
                # Use MoESegment (not MoEComputeSegment) because it includes all_reduce
                # inside the graph — required for single-graph whole-model capture.
                moe_segments = {}
                for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                    moe_decode = decoder_layer.mlp
                    if (hasattr(moe_decode, 'persistent_expert_indices')
                            and len(moe_decode.persistent_expert_indices) > 0
                            and hasattr(moe_decode, 'comm') and moe_decode.comm is not None):
                        moe_segments[layer_idx] = MoESegment(
                            moe_decode, moe_pool, moe_decode.comm,
                            self.world_size, self.rank, self.torch_device,
                        )

                whole_seg = WholeModelSegment(
                    model=self.model,
                    moe_pool=moe_pool,
                    moe_segments=moe_segments,
                    device=self.torch_device,
                    max_pages_per_seq=max_pages,
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    max_bucket_size=bucketing._max_bucket,
                )

            manager.register_segment("whole_model", whole_seg)

            if self.rank == 0:
                logging.info(
                    f"CUDA graph capture: whole-model × "
                    f"{len(bucketing.bucket_sizes)} buckets {bucketing.bucket_sizes}"
                )

            # Sync all ranks — NCCL collectives require simultaneous participation
            torch.cuda.synchronize(self.torch_device)
            dist.barrier()

            manager.warmup_and_capture_all()

            # Reset capture mode flags
            for layer in self.model.model.layers:
                layer._graph_capture_mode = False

            self._graph.cuda_graph_manager = manager
            self._graph.whole_model_graph = True
            self._graph.whole_model_bucketing = bucketing
            self._graph.whole_model_segment = whole_seg
            if self.rank == 0:
                stats = manager.get_capture_stats()
                logging.info(
                    f"CUDA graphs ready (whole-model): {stats['total_capture_time_ms']:.0f}ms"
                )
        else:
            # Per-layer mode: capture attention graph per layer, MoE stays eager.
            self._graph.whole_model_graph = False

            if _is_k25:
                # K2.5: Register K25AttnSegment per layer (MLA attention only, no MoE graph).
                # MoE stays eager to preserve async shared expert overlap.
                # Each rank uses local batch_size for bucket selection (DP-attention, no NCCL).
                from batchgen.models.moonshotai.kimi_k25.cuda_graph_segments import K25AttnSegment

                for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                    attn_wrapper = decoder_layer.self_attn
                    seg = K25AttnSegment(
                        decoder_layer, attn_wrapper, layer_idx,
                        max_seq_len=max_rope_len,
                        max_pages_per_seq=max_pages,
                        page_size_tokens=page_size_tokens,
                    )
                    seg_name = f"layer_{layer_idx}_attn"
                    manager.register_segment(seg_name, seg)

                if self.rank == 0:
                    logging.info(
                        f"CUDA graph capture (K2.5 MLA): {len(self.model.model.layers)} layers (attn only) × "
                        f"{len(bucketing.bucket_sizes)} buckets {bucketing.bucket_sizes}"
                    )

                manager.warmup_and_capture_all()

                # Enable graph mode on each decoder layer
                for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                    decoder_layer.enable_cuda_graph(
                        manager,
                        attn_name=f"layer_{layer_idx}_attn",
                        max_pages_per_seq=max_pages,
                    )
            else:
                # GPT-OSS: per-layer mode (existing behavior)
                if self.rank == 0:
                    num_segs = "attn+moe" if has_moe_graph else "attn"
                    logging.info(
                        f"CUDA graph capture: {len(self.model.model.layers)} layers ({num_segs}) × "
                        f"{len(bucketing.bucket_sizes)} buckets {bucketing.bucket_sizes}"
                    )

                # Sync all ranks before warmup — MoE segments use NCCL collectives
                # which require all ranks to participate simultaneously.
                if has_moe_graph:
                    torch.cuda.synchronize(self.torch_device)
                    dist.barrier()

                manager.warmup_and_capture_all()

                # Enable graph mode on each decoder layer
                for layer_idx, decoder_layer in enumerate(self.model.model.layers):
                    moe_name = f"layer_{layer_idx}_moe" if has_moe_graph else None
                    decoder_layer.enable_cuda_graph(
                        manager,
                        full_attn_name=f"layer_{layer_idx}_full_attn",
                        moe_name=moe_name,
                    )

            self._graph.cuda_graph_manager = manager
            if self.rank == 0:
                stats = manager.get_capture_stats()
                logging.info(
                    f"CUDA graphs ready: {stats['total_capture_time_ms']:.0f}ms"
                )
        self._record_decode_graph_capture()
