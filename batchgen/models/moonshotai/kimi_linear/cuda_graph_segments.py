# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear CUDA-graph decode — Phase A (plan M5, item 4; M5.2).

CAPTURE STRUCTURE (decision + why)
----------------------------------
Every MoE layer's decode forward runs NCCL collectives (all_gather +
all_reduce, `batchgen.moe.fused_moe_bf16_resident.ResidentEPMoELayer.forward`),
which must stay OUTSIDE a Phase-A graph. Kimi-Linear has 27 layers, 26 of them
MoE (`first_k_dense_replace=1` => layer 0 is a dense KimiMLP), so the collective
pattern forces 26 eager sections and therefore at most 27 graph spans per step.

Chosen: **one graph per layer, covering that layer's attention span**

    input_layernorm -> self_attn (KDA or NoPE-MLA) -> +residual
                    -> post_attention_layernorm            => (normed, residual)

with the MoE (or, for the dense layer, its MLP) and the post-FFN residual add
run eagerly between replays. Layer 0 has no collective, so its span additionally
folds the dense MLP + second residual: the whole layer is one graph (`fold_ffn`).

Result: 27 replays/step. Rejected alternative — maximal spans that also cross
the layer boundary (`[post-MoE add of layer k] + [attention span of layer k+1]`)
— would give 26 replays, one fewer (~4% of replay overhead, single-digit us),
at the cost of (a) heterogeneous span signatures, (b) the adapter owning
`KimiLinearModel.forward`'s layer loop (a second copy of model.py's residual
wiring, drift risk), and (c) losing the 1:1 layer<->graph mapping that the
graph-vs-eager compare and the M5.5 gate ladder attribute deltas with. The one
structurally free win (folding the collective-free dense layer) is taken.

WHAT IS AND IS NOT IN THE GRAPH
-------------------------------
In:   layer norms, all projections, 3x causal_conv1d_update + fla
      fused_recurrent_kda_fwd (KDA), paged-KV token write + dense MLA decode
      (MLA), the attention TP all-reduce, o_proj, residual adds, dense MLP of
      layer 0. The dense MLA decode is the capability-selected seam
      (``batchgen.attention.mla.paged_decode``): flash_mla on Hopper,
      flashinfer trtllm-gen on Blackwell; both capture.
K3 resident-MXFP4 MoE layers additionally use a companion graph segment; it
      owns their EP/TP collectives, grouped Marlin path, and shared expert.
Out:  embedding, final norm, lm_head, KV offload callback (fired post-replay
      with a cloned k_tensor), KDA slot alloc/free/zeroing,
      and the per-step static-buffer refresh — all eager, same stream, ordered
      before/after the replay (test_kda_segment_capture.py contract).

STATIC BUFFERS
--------------
* hidden_states [bucket, 1, H] — owned by `CUDAGraphManager` (one per span),
  refreshed by `manager.replay(..., hidden_states=...)`, pad rows zeroed.
* cache_seqlens / token_indices / slot_indices / num_valid_tokens / page_table
  — one shared `_BucketStatics` set per bucket (kimi_k25 adapter refresh
  pattern: the batch's rows of the KV manager's graph-stable page table are
  copied into a fixed [bucket, width] buffer each step; ``width`` is the
  decode seam's ``block_table_width`` of the manager's max_pages, so the
  kernel reads the static table in place).
* KDA slot indices — NOT reallocated here: `KDAStateGPUManager` already owns a
  persistent int32 buffer at a fixed address (M5.1), so the segments bind a
  bucket-sized view of it and `prepare_decode_step` refreshes it in place, once
  per step for all 20 KDA layers (the M5 item-2 eager pre-win).

PADDING ROWS (plan M5.3)
------------------------
Bucket padding rows must not corrupt live state:
  * paged KV write: skipped via `num_valid_tokens` (rows >= bsz are dropped by
    the Triton kernel). It is 0 during warmup, so capture-time warmup never
    writes KV at all.
  * KDA: padded rows point at a dedicated SCRATCH slot owned by this adapter
    (a reserved sequence id in the KDA slot manager), never at -1 — -1 through
    `ssm_state_indices` is an out-of-bounds write BEFORE the fla pool base.
    Several padded rows share the scratch slot; that race is benign precisely
    because the slot is scratch. Warmup uses the scratch slot for every row.
  * The dense MLA decode reads pad rows with cache_seqlens=1 and
    page_table=0 (a valid page); the outputs are discarded by the
    bucket->bsz slice.

Attention buckets remain rank-local. The K3 MoE graph uses a separate
TP-group-sized bucket: the worker's globally synchronized
``num_tokens_per_rank`` is converted to ``tp_size * ntp`` so every EP rank
replays the same collective shapes even when DP groups have different local
batch sizes. If any EP rank is empty, all ranks use eager MoE for that step;
this avoids mixing graph and eager collectives.

KIMI-K3.1 DSA LAYERS
--------------------
A full-attention layer that carries the DSA indexer captures
``dsa_decode.dsa_decode_graph_safe`` instead of the dense NoPE-MLA inline:
the index key write, every row scored, the selection transform and FA3 over
the selected tokens, all against static tables. The auxiliary (index-key)
cache gets its own fixed ``[bucket, max_aux_pages]`` page-table buffer,
refreshed with the latent table's batch order, and its own place in the KV
signature. The step's index keys leave the graph the way the latent does: as
static outputs the driver appends to the auxiliary Host region after replay.

Mode is planner/config-driven (`Basic_Config.decode_graph_mode`, default
"eager") with a batch-level override through
`debug_state.kimi_decode_graph_mode` = "graph" | "eager" | "compare"
(glm5_moe_mode precedent). No env vars, new or old.
"""

import logging
import time
from typing import Dict, List, Optional

import torch

from batchgen.attention.mla.paged_decode import block_table_width
from batchgen.cuda_graph.graph_manager import (
    BatchSizeBucketing,
    CUDAGraphManager,
    TensorSpec,
)
from batchgen.models.wrappers import AttnWrapperBase

from .block_residual import num_block_residual_columns
from .dsa_decode import dsa_decode_forward, dsa_decode_graph_safe
from .dsa_index_rotation import KERNEL_INDEX_HEAD_DIM
from .layer_body import (
    KimiLayerBodyOps,
    run_kimi_block_residual_layer_body,
    run_kimi_layer_body,
)
from .serving_modules import (
    kda_decode_serving,
    mla_decode_nope_core,
    mla_decoding_nope_with_pagekv,
)
from .wrappers import KDALayerState, KimiLinearKDAWrapper

logger = logging.getLogger(__name__)

# Per-rank decode buckets. Kimi-Linear's decode admission cap is
# MoE_decoding_micro_batch_size=16 rows/rank (planner.py), so 16 is the top
# bucket; the planner may override the list.
DEFAULT_DECODE_GRAPH_BUCKETS = [1, 2, 4, 8, 16]

# Compare-mode rate limit (steps between graph-vs-eager comparisons).
DEFAULT_COMPARE_EVERY = 64

# Whole-model replay telemetry: whole-model replays between diagnostic log
# lines. The CUDA event pair is read back only once its end event has already
# completed, so neither the counters nor this log ever synchronize the decode
# stream.
WHOLE_TELEMETRY_LOG_EVERY = 512

# fla's Triton kernels JIT/autotune on the first call per shape; a cold shape at
# capture time compiles host-side inside the capture. 3 warmup iterations per
# (segment, bucket) is the count validated by
# batchgen_kernels/tests/kimi_linear/test_kda_segment_capture.py.
WARMUP_ITERATIONS = 3

# Sequence id that owns the KDA scratch slot (padding + warmup rows). Real
# global sequence ids are non-negative, so this can never collide.
GRAPH_SCRATCH_SEQ_ID = -1_000_001

_VALID_MODES = ("eager", "graph", "compare")


def _debug_dict() -> dict:
    debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
    return debug if isinstance(debug, dict) else {}


def _debug_mode() -> Optional[str]:
    """Batch-level `kimi_decode_graph_mode` override, or None."""
    value = _debug_dict().get("kimi_decode_graph_mode")
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    return mode if mode in _VALID_MODES else None


def _mla_decode_graph_safe(
    attn,
    hidden_states,
    *,
    layer_idx,
    page_table,
    slot_indices,
    token_indices,
    cache_seqlens,
    num_valid_tokens,
    page_size_tokens,
    mla_schedule=None,
):
    """The captured binding of `serving_modules.mla_decode_nope_core`.

    The dense math exists ONCE, in the core; the eager step
    (`serving_modules.mla_decoding_nope_with_pagekv`) runs the same function.
    This binding injects exactly the three differences a capture forces:
      1. the new token is written with `run_paged_kv_token_update_fused` over
         the STATIC page table instead of
         `gpu_paged_kv_manager.update_layer_decode_new_token` (which re-reads
         the manager's active table / slot tensor — both may be reallocated
         between steps, so their addresses cannot be baked into a graph);
      2. the dense MLA decode reads the same STATIC page table instead of
         the manager's active block table;
      3. bucket padding rows are dropped by `num_valid_tokens` on the write.

    Args:
        attn: KimiMLAAttention module (unwrapped).
        hidden_states: (bucket, 1, hidden) post-input_layernorm activations.
        layer_idx: absolute layer index (paged-KV layer key).
        page_table: (bucket, max_pages) int32 static page table.
        slot_indices: (bucket,) int32 page-table row per batch row.
        token_indices: (bucket,) int32 absolute write position per batch row.
        cache_seqlens: (bucket,) int32 context length including the new token.
        num_valid_tokens: (1,) int32 unpadded row count.
        page_size_tokens: tokens per KV page.
        mla_schedule: the step's ``paged_decode.DecodeSchedule`` shared by
            every MLA layer (whole-model graph), or None for a standalone
            per-layer capture, which then records its own inside the seam.

    Returns:
        (attn_output (bucket, 1, hidden), k_tensor (bucket, 1, 1, kv_dim)).
    """
    # The same pure-BF16 seam as the eager path; it never pulls in the legacy
    # FP8/FA3 backend and its DeepGEMM dependency.
    from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused

    def bind_kv_write(k_tensor):
        # k_cache is the manager's per-layer view. Its address is fixed only
        # for the life of ONE KV manager instance: the slice base is
        # `data_ptr + layer * num_pages * page_stride`, so a re-init with a
        # different num_pages moves it even when data_ptr is reused. That is
        # why `_signature` must carry the cache shape/stride, not just the
        # pointer. The graph brings its own static page table, so only the
        # cache tensor is read here; an empty rank captures before it ever
        # builds an active table.
        blocked_k, _ = AttnWrapperBase.gpu_paged_kv_manager.get_layer_kv_cache(
            layer_idx
        )
        run_paged_kv_token_update_fused(
            k_cache=blocked_k,
            k_tokens=k_tensor.view(k_tensor.shape[0], -1),
            page_table=page_table,
            slot_indices=slot_indices,
            token_indices=token_indices,
            page_size_tokens=page_size_tokens,
            num_valid_tokens=num_valid_tokens,
        )
        return blocked_k, page_table, k_tensor

    return mla_decode_nope_core(
        attn,
        hidden_states,
        cache_seqlens=cache_seqlens,
        bind_kv_write=bind_kv_write,
        mla_schedule=mla_schedule,
    )


class _BucketStatics:
    """Fixed-address decode inputs shared by every span at one bucket size.

    Allocated once per bucket (never reallocated — captured graphs bake the
    addresses in) and refreshed in place, eagerly, before the step's replays.
    """

    def __init__(self, bucket: int, max_pages: int, device, kda_slots, *,
                 block_residual_columns: int = 0, hidden_size: int = 0,
                 dtype=None, max_aux_pages: int = 0):
        """``max_pages`` is the static table's column count: the decode
        seam's ``block_table_width`` of the manager's graph table, which may
        exceed the manager's own width (the extra columns stay page 0 and no
        length reaches them)."""
        self.bucket = bucket
        self.cache_seqlens = torch.ones(
            bucket, dtype=torch.int32, device=device
        )
        self.token_indices = torch.zeros(
            bucket, dtype=torch.int32, device=device
        )
        # page_table rows are copied in batch order, so row i IS batch row i.
        self.slot_indices = torch.arange(
            bucket, dtype=torch.int32, device=device
        )
        self.num_valid_tokens = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        self.page_table = torch.zeros(
            (bucket, max_pages), dtype=torch.int32, device=device
        )
        # Kimi-K3.1 DSA: the auxiliary (index-key) cache's rows in the SAME
        # batch order, so ``slot_indices`` addresses both tables. None for a
        # model without indexer layers.
        self.aux_page_table = (
            torch.zeros((bucket, max_aux_pages), dtype=torch.int32, device=device)
            if max_aux_pages > 0 else None
        )
        # View of KDAStateGPUManager's persistent slot buffer (M5.1) — bound,
        # not allocated.
        self.kda_slots = kda_slots
        self.block_residual = None
        if block_residual_columns:
            self.block_residual = torch.empty(
                bucket, block_residual_columns, hidden_size,
                dtype=dtype, device=device,
            )

    def arm_for_capture(self) -> None:
        """Neutral contents for warmup + capture: no KV writes (num_valid=0),
        page 0 / seqlen 1 reads, and KDA rows already pointing at the scratch
        slot (set by the caller through prepare_decode_step)."""
        self.num_valid_tokens.zero_()
        self.cache_seqlens.fill_(1)
        self.token_indices.zero_()
        self.page_table.zero_()
        if self.aux_page_table is not None:
            self.aux_page_table.zero_()
        if self.block_residual is not None:
            self.block_residual.zero_()

    def refresh(self, bsz: int, cache_seqlens, token_indices, page_rows,
                aux_page_rows=None) -> None:
        """Bind this step's batch into the static buffers (rows >= bsz are
        padding: KV-write-skipped, page 0, seqlen 1)."""
        if self.aux_page_table is None and aux_page_rows is not None:
            raise RuntimeError(
                "decode graph statics: this step gave auxiliary rows but the "
                "auxiliary page table is absent"
            )
        if self.aux_page_table is not None and aux_page_rows is None and bsz > 0:
            raise RuntimeError(
                "decode graph statics: the auxiliary page table is allocated "
                "but this step gave no auxiliary rows"
            )
        self.num_valid_tokens.fill_(bsz)
        if bsz < self.bucket:
            self.cache_seqlens[bsz:].fill_(1)
            self.token_indices[bsz:].zero_()
            self.page_table[bsz:].zero_()
            if self.aux_page_table is not None:
                self.aux_page_table[bsz:].zero_()
        self.cache_seqlens[:bsz].copy_(cache_seqlens, non_blocking=True)
        self.token_indices[:bsz].copy_(token_indices, non_blocking=True)
        self.page_table[:bsz, : page_rows.shape[1]].copy_(
            page_rows, non_blocking=True
        )
        if self.aux_page_table is not None and bsz > 0:
            self.aux_page_table[:bsz].copy_(aux_page_rows, non_blocking=True)


class KimiLinearSpanSegment:
    """One decoder layer's capturable decode span (`CapturableSegment`).

    KDA layer: input_layernorm -> kda_decode_serving -> residual body -> post_ln.
    MLA layer: input_layernorm -> graph-safe NoPE-MLA -> residual body -> post_ln.
    Dense layer (no `block_sparse_moe`, i.e. no collectives): the span also
    folds the dense MLP + second residual, so the whole layer is one graph.

    For K3 the residual body is Block Attention Residual, not the classic
    ``hidden + attention`` path. Every bucket owns one fixed-address full
    ``(bucket, num_boundaries, hidden)`` buffer shared by all 93 spans. A layer
    captures its statically-known active-column view; boundary spans write the
    pre-mix prefix into their column before the MLP-depth mix. Thus the graph
    carries the complete K3 state without copying that buffer between spans.
    """

    def __init__(self, layer, layer_idx: int, statics: Dict[int, _BucketStatics],
                 page_size_tokens: int, dtype, *, aux_page_size_tokens: int = 0,
                 max_aux_context: int = 0):
        self.layer = layer
        self.layer_idx = layer_idx
        self.attn = getattr(layer.self_attn, "module", layer.self_attn)
        self.is_kda = bool(layer.is_linear_attn)
        # Kimi-K3.1: a full-attention layer with the DSA indexer captures the
        # sparse step (``_graph_attention``); the dense inline would attend
        # over every cached token and never score the index keys.
        self.is_dsa = (
            not self.is_kda and getattr(self.attn, "indexer", None) is not None
        )
        if self.is_dsa and (aux_page_size_tokens <= 0 or max_aux_context <= 0):
            raise ValueError(
                f"DSA layer {layer_idx} needs the auxiliary cache geometry "
                f"(aux_page_size_tokens={aux_page_size_tokens}, "
                f"max_aux_context={max_aux_context})"
            )
        self.aux_page_size_tokens = int(aux_page_size_tokens)
        self.max_aux_context = int(max_aux_context)
        self.fold_ffn = not hasattr(layer, "block_sparse_moe")
        self.hidden_size = layer.hidden_size
        self.page_size_tokens = page_size_tokens
        self.dtype = dtype
        self._statics = statics
        self._buf: Optional[_BucketStatics] = None
        self.use_block_residual = bool(
            getattr(layer, "use_attn_residuals", False)
        )
        self.block_size = (
            int(layer.attn_res_block_size) if self.use_block_residual else 0
        )
        self.is_block_boundary = bool(
            self.use_block_residual and layer_idx % self.block_size == 0
        )
        self.num_blocks_after = (
            layer_idx // self.block_size + 1 if self.use_block_residual else 0
        )
        self.num_blocks_before = (
            self.num_blocks_after - int(self.is_block_boundary)
        )
        self.kda_state: Optional[KDALayerState] = (
            KimiLinearKDAWrapper.layer_pools[layer_idx] if self.is_kda else None
        )
        self.kv_dim = (
            0 if self.is_kda
            else self.attn.kv_lora_rank + self.attn.qk_rope_head_dim
        )

    # -- CapturableSegment ------------------------------------------------

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), self.dtype
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        hidden = TensorSpec(("batch_size", 1, self.hidden_size), self.dtype)
        specs = {"hidden": hidden} if self.fold_ffn else {
            "normed": hidden, "residual": hidden,
        }
        if not self.is_kda:
            specs["k_tensor"] = TensorSpec(
                ("batch_size", 1, 1, self.kv_dim), self.dtype
            )
        if self.is_dsa:
            specs["index_k"] = TensorSpec(
                ("batch_size", 1, 1, KERNEL_INDEX_HEAD_DIM), torch.float8_e4m3fn
            )
            specs["index_scales"] = TensorSpec(("batch_size",), torch.float32)
        return specs

    def setup_static_buffers(self, bucket_size: int) -> None:
        """Bind this bucket's shared statics (called by CUDAGraphManager right
        before warmup+capture, so the bindings are what the graph bakes in)."""
        self._buf = self._statics[bucket_size]
        if self.is_kda:
            self.kda_state.cur_decode_slots = self._buf.kda_slots

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        mla_schedule=None,
        kda_cu_seqlens=None,
    ) -> Dict[str, torch.Tensor]:
        """The captured span: the ONE shared layer body under the graph
        operator set (graph-safe attention; ``layer.mlp`` on a fold-FFN span,
        else the body returns the halves for the driver's MoE stitch;
        out-of-place prefix merges; the boundary snapshot written in place
        into the bucket-owned static buffer)."""
        attention = self._graph_attention
        if mla_schedule is not None or kda_cu_seqlens is not None:

            def attention(normed):
                return self._graph_attention(
                    normed, mla_schedule, kda_cu_seqlens,
                )

        ffn = self.layer.mlp if self.fold_ffn else None
        if self.use_block_residual:
            buf = self._buf.block_residual

            def append_boundary(flat_prefix, active):
                # The captured path writes the current layer's pre-mix prefix
                # into its statically-known column of the bucket's shared
                # buffer, then widens the active view. Compare mode binds a
                # private torch.cat instead (``_eager_span``), so the eager
                # reference cannot advance this state before the replay.
                buf[:, self.num_blocks_before].copy_(flat_prefix)
                return buf[:, : self.num_blocks_after]

            out = run_kimi_block_residual_layer_body(
                layer=self.layer,
                ops=KimiLayerBodyOps(
                    attn=attention,
                    ffn=ffn,
                    append_boundary=append_boundary,
                ),
                hidden_states=hidden_states,
                block_residual=buf[:, : self.num_blocks_before],
            )
            # The residual bank is bucket-owned static state, not a segment
            # output: the driver reads it through _block_residual_view.
            out.pop("block_residual")
            return out

        return run_kimi_layer_body(
            layer=self.layer,
            ops=KimiLayerBodyOps(attn=attention, ffn=ffn),
            hidden_states=hidden_states,
        )

    def _graph_attention(
        self,
        normed,
        mla_schedule=None,
        kda_cu_seqlens=None,
    ):
        """``(attn_out, kv_outputs)``: the attention output and the cache rows
        this step produced for the Host append (``k_tensor``; plus
        ``index_k`` / ``index_scales`` on a DSA layer), empty for KDA."""
        buf = self._buf
        if self.is_kda:
            attn_out = kda_decode_serving(
                self.attn,
                normed,
                self.kda_state,
                cu_seqlens=kda_cu_seqlens,
            )
            return attn_out, {}
        if self.is_dsa:
            # The dense decode schedule is unused here: the sparse attention
            # runs through the sparse seam over the selected tokens. Both page
            # tables come from the graph's static buffer, so only the cache
            # tensors are read; an empty rank captures before it ever builds
            # an active table.
            blocked_k, _ = AttnWrapperBase.gpu_paged_kv_manager.get_layer_kv_cache(
                self.layer_idx
            )
            aux_blocked_k, _ = (
                AttnWrapperBase.gpu_paged_kv_manager_aux.get_layer_kv_cache(
                    self.layer_idx
                )
            )
            step = dsa_decode_graph_safe(
                self.attn,
                normed,
                layer_idx=self.layer_idx,
                page_table=buf.page_table,
                slot_indices=buf.slot_indices,
                token_indices=buf.token_indices,
                cache_seqlens=buf.cache_seqlens,
                num_valid_tokens=buf.num_valid_tokens,
                page_size_tokens=self.page_size_tokens,
                blocked_k=blocked_k,
                aux_blocked_k=aux_blocked_k,
                aux_page_table=buf.aux_page_table,
                aux_page_size=self.aux_page_size_tokens,
                max_seqlen=self.max_aux_context,
            )
            return step.attn_output, {
                "k_tensor": step.k_tensor,
                "index_k": step.index_k_fp8,
                "index_scales": step.index_scales,
            }
        else:
            attn_out, k_tensor = _mla_decode_graph_safe(
                self.attn,
                normed,
                layer_idx=self.layer_idx,
                page_table=buf.page_table,
                slot_indices=buf.slot_indices,
                token_indices=buf.token_indices,
                cache_seqlens=buf.cache_seqlens,
                num_valid_tokens=buf.num_valid_tokens,
                page_size_tokens=self.page_size_tokens,
                mla_schedule=mla_schedule,
            )
        return attn_out, {"k_tensor": k_tensor}


class KimiLinearDecodeGraph:
    """Phase-A CUDA-graph decode step policy for Kimi-Linear.

    The worker's decode driver (``decode_engine.DecoupledGpuDecode.run``) owns
    the per-step dispatch: it calls ``begin_decode_step`` before the forward,
    replays the whole-model graph through ``replay_whole_model_step`` when the
    step is armed, otherwise runs the plain eager model forward, and always
    calls ``end_decode_step``. The former ``KimiLinearModel.forward`` patch
    that hid this dispatch inside the model is retired. ``install()`` still
    patches each decoder ``layer.forward`` (plus ``_new_block_residual``):
    that is the kept per-layer legacy/compare arm, which replays one span per
    layer from inside the eager model loop while the step is armed.

    Lifecycle: constructed + installed by the PSM at `configure_decoding` when
    the planner asks for it, and released with the decode instance that owns
    it (decode RETIRE, model rebuild) — never at a decode SUSPEND. The
    dedicated KDA scratch slot is reserved during installation, before server
    readiness and sequence admission.

    Capture-once: the first ``prewarm_decode_graphs`` in a captured mode
    captures the adapter's WHOLE ladder (every bucket a step can replay) and
    seals it; nothing else ever captures. A kept decode instance runs that
    prewarm in the worker's startup decode graph capture, before readiness,
    so its graphs are captured once per process; a rebuilt instance runs it
    at its decode configure. After the seal the KV / KDA storage the graphs
    bake must not move (``_require_capture_signature`` raises). Every step
    that cannot be replayed (prefill phase, eager mode, no sealed ladder, bsz
    above the top bucket, a bucket outside the captured ladder, an
    unavailable page table) falls through to the unmodified eager layer
    forward.
    """

    def __init__(
        self,
        model,
        model_config,
        *,
        device,
        buckets: Optional[List[int]] = None,
        mode: str = "graph",
        compare_every: Optional[int] = None,
        rank: int = 0,
        deepep_ll: bool = False,
    ):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.rank = int(rank)
        self.deepep_ll = bool(deepep_ll)
        self.mode = mode if mode in _VALID_MODES else "eager"
        self.compare_every = int(compare_every or DEFAULT_COMPARE_EVERY)
        self.bucketing = BatchSizeBucketing(
            list(buckets or DEFAULT_DECODE_GRAPH_BUCKETS)
        )
        self.manager = CUDAGraphManager(self.bucketing, device=device)
        # fla autotune needs more warmup than the manager's default (2/1).
        self.manager.WARMUP_ITERATIONS = WARMUP_ITERATIONS
        self.manager.WARMUP_ITERATIONS_SUBSEQUENT = WARMUP_ITERATIONS

        self.segments: Dict[int, KimiLinearSpanSegment] = {}
        self.last_compare: Dict[int, float] = {}
        self.step = 0

        self._installed = False
        self._built = False
        self._statics: Dict[int, _BucketStatics] = {}
        self._captured: set = set()
        self._capture_stats: Dict[int, dict] = {}
        self._scratch_slot: Optional[int] = None
        self._kv_signature = None
        # Set once ``prewarm_decode_graphs`` has captured the whole ladder;
        # from then on nothing captures and the KV signature is fixed.
        self._ladder_captured = False
        # Fixed-address arenas the K3 resident MoE shards live in (empty for
        # allocator-backed shards). A MoE graph replay reads them, so it is
        # refused while any is unmapped.
        self._moe_arenas: List[object] = []
        self._max_pages = 0
        self._static_pages = 0
        self._page_size_tokens = 0
        # Kimi-K3.1 DSA: the auxiliary (index-key) cache geometry. Zero when
        # the model has no indexer layer, which keeps every K3 path unchanged.
        self._has_dsa = any(
            getattr(getattr(layer.self_attn, "module", layer.self_attn),
                    "indexer", None) is not None
            for layer in model.model.layers
        )
        self._max_aux_pages = 0
        self._aux_page_size_tokens = 0
        self._orig_new_block_residual = None
        self._orig_layer_forwards: Dict[int, object] = {}
        self._fallbacks_logged: dict = {}
        self._uses_block_residual = bool(
            getattr(model_config, "attn_res_block_size", None) is not None
        )
        self._block_residual_columns = (
            num_block_residual_columns(
                model_config.num_hidden_layers,
                model_config.attn_res_block_size,
            )
            if self._uses_block_residual else 0
        )

        # per-step state
        self._step_active = False
        self._bucket = 0
        self._bsz = 0
        self._compare_step = False
        self._slot_index_long: Optional[torch.Tensor] = None
        self._compare_arange: Optional[torch.Tensor] = None

        # K3 resident-MXFP4 MoE graphs use a separate manager because the
        # attention buckets include values (1, 2, 4) that are not divisible by
        # the TP8 row group. Build this lazily only when the model actually has
        # resident latent-MoE shards; other Kimi variants retain eager MoE.
        self.moe_manager = None
        self.moe_bucketing = None
        self.moe_pool = None
        self.moe_segments = {}
        self._moe_built = False
        self._moe_build_attempted = False
        self._moe_captured = set()
        self._moe_capture_stats = {}
        self._moe_step_active = False
        self._moe_bucket = 0
        self._moe_group_bucket = 0
        self._moe_ntp = 0
        # K3 whole-model graph state.  The existing per-layer managers remain
        # available as the correctness/fallback path; the outer manager owns a
        # separate graph pool so it can capture direct child forwards rather
        # than recursively replaying the per-layer graphs.
        self.whole_manager = None
        self.whole_segment = None
        self._whole_built = False
        self._whole_captured = set()
        self._whole_capture_stats = {}
        self._whole_step_active = False
        self._whole_bucket = 0
        self._whole_name = "kimi_linear/whole_model"
        # Whole-model replay telemetry.  One reusable CUDA event pair brackets
        # the replay; the elapsed time is folded in on a LATER step, once the
        # end event reports complete, so the measured path never waits on it.
        # The KV staging figure is host-side launch cost only, for the same
        # reason: the clone and the callbacks are asynchronous.
        self._whole_replay_events = None
        self._whole_replay_pending = False
        self._whole_replays = 0
        self._whole_replay_samples = 0
        self._whole_replay_ms = 0.0
        self._whole_kv_stage_samples = 0
        self._whole_kv_stage_host_ms = 0.0
        # One per-rank scalar input for the K3 MoE graph.  Its value is
        # refreshed once per decode step; unlike a Python argument it is safe
        # to read inside a captured graph and avoids rebuilding a validity mask
        # from host-visible row counts in every MoE layer.
        self._moe_valid_tokens = torch.empty(
            (1,), dtype=torch.int32, device=self.device
        )

    # ------------------------------------------------------------------ #
    #  Install / release                                                  #
    # ------------------------------------------------------------------ #

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in _VALID_MODES else "eager"

    @property
    def ladder_captured(self) -> bool:
        """Whether this adapter's graph ladder has been captured (sealed)."""
        return self._ladder_captured

    def install(self) -> None:
        """Patch the per-layer decode forwards (idempotent).

        The model-level forward is NOT patched: the worker's decode driver
        invokes this object's step API directly. Only the per-layer legacy
        arm (layer spans replayed from inside the eager model loop) installs
        its layer hooks here.
        """
        if self._installed:
            return
        slot_manager = KimiLinearKDAWrapper.slot_manager
        if slot_manager is None:
            raise RuntimeError(
                "Kimi decode graph installation requires initialized KDA "
                "state pools"
            )
        # This adapter is installed by configure_decoding during K3's
        # pre-readiness startup pass. Reserve padding/warmup state now so
        # prefill admission observes the reduced free-slot count; allocating
        # it lazily on first decode can otherwise exhaust a completely full
        # KDA pool.
        self._scratch_slot = slot_manager.alloc(GRAPH_SCRATCH_SEQ_ID)
        inner = self.model.model
        if self._uses_block_residual:
            self._orig_new_block_residual = inner._new_block_residual
            inner._new_block_residual = self._make_new_block_residual(
                self._orig_new_block_residual
            )
        for layer_idx, layer in enumerate(inner.layers):
            orig = layer.forward
            self._orig_layer_forwards[layer_idx] = orig
            layer.forward = self._make_layer_forward(layer_idx, orig)
        self._installed = True
        if self.rank == 0:
            logger.info(
                "[KIMI_DECODE_GRAPH] installed on %d layers "
                "(mode=%s buckets=%s compare_every=%d)",
                len(inner.layers), self.mode, self.bucketing.bucket_sizes,
                self.compare_every,
            )

    def release(self) -> None:
        """Restore the eager layer forwards and drop every captured bucket."""
        if self._installed:
            inner = self.model.model
            if self._orig_new_block_residual is not None:
                inner._new_block_residual = self._orig_new_block_residual
                self._orig_new_block_residual = None
            for layer_idx, orig in self._orig_layer_forwards.items():
                inner.layers[layer_idx].forward = orig
            self._orig_layer_forwards = {}
            self._installed = False
        self._drop_whole_graphs()
        self._drop_graphs()
        self._drop_moe_graphs()
        if self._scratch_slot is not None:
            if KimiLinearKDAWrapper.slot_manager is not None:
                KimiLinearKDAWrapper.slot_manager.free(GRAPH_SCRATCH_SEQ_ID)
            self._scratch_slot = None

    def get_capture_stats(self) -> dict:
        stats = dict(self.manager.get_capture_stats())
        stats["kimi_capture"] = dict(self._capture_stats)
        if self.whole_manager is not None:
            stats["kimi_whole_capture"] = dict(self._whole_capture_stats)
            stats["kimi_whole_manager"] = self.whole_manager.get_capture_stats()
            stats["kimi_whole_replay"] = self._whole_replay_telemetry()
        if self.moe_manager is not None:
            stats["kimi_moe_capture"] = dict(self._moe_capture_stats)
            stats["kimi_moe_manager"] = self.moe_manager.get_capture_stats()
        return stats

    # ------------------------------------------------------------------ #
    #  Decode-driver step API (the worker's decode loop calls these)      #
    # ------------------------------------------------------------------ #

    def begin_decode_step(self, input_ids) -> bool:
        """Decide this decode step's path; the worker's decode driver calls
        this immediately before the step's forward.

        Runs ``_begin_step`` (the per-step path decision and static-buffer
        refresh) and, when the whole-model replay is armed, validates
        ``input_ids`` exactly as the retired ``KimiLinearModel.forward`` hook
        did. Returns whether the driver must replay the whole-model graph
        through ``replay_whole_model_step``; ``False`` means the driver runs
        the plain eager model forward, during which the patched layer
        forwards replay per-layer spans while the step stays armed. The
        driver must pair every call with ``end_decode_step`` after the
        step's forward, replayed or eager.
        """
        self._begin_step()
        if self._whole_step_active:
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.dim() != 2
                or input_ids.shape[1] != 1
                or input_ids.shape[0] != self._bsz
            ):
                self._fallback(
                    "whole-model graph requires decode input_ids [local_bsz, 1]"
                )
                self._whole_step_active = False
                self._step_active = False
        return self._whole_step_active

    def replay_whole_model_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replay the captured transformer body and apply the LM head.

        The retired model-forward hook returned ``_run_whole_model``'s hidden
        states to the outer CausalLM, which applied ``lm_head``; this driver
        call keeps that exact pair, so the step's logits are unchanged.
        Returns ``[local_bsz, 1, vocab]`` logits.
        """
        hidden_states = self._run_whole_model(input_ids)
        return self.model.lm_head(hidden_states)

    def end_decode_step(self) -> None:
        """Disarm the step (the retired model-forward hook's ``finally``)."""
        self._step_active = False
        self._moe_step_active = False
        self._whole_step_active = False

    # ------------------------------------------------------------------ #
    #  Patched forwards (the kept per-layer legacy/compare arm)           #
    # ------------------------------------------------------------------ #

    def _make_new_block_residual(self, orig_new):
        def new_block_residual(hidden_states):
            if not self._step_active:
                return orig_new(hidden_states)
            expected = (self._bsz, 1, self.model_config.hidden_size)
            if tuple(hidden_states.shape) != expected:
                self._fallback(
                    "block-residual seed shape {} != {}".format(
                        tuple(hidden_states.shape), expected
                    )
                )
                self._step_active = False
                return orig_new(hidden_states)
            return self._block_residual_view(-1)
        return new_block_residual

    def _make_layer_forward(self, layer_idx: int, orig_forward):
        def forward(hidden_states, attention_mask=None, position_ids=None,
                    past_key_values=None, cu_seqlens=None,
                    block_residual=None, **kwargs):
            if self._step_active and hidden_states.shape[0] != self._bsz:
                # The step's metadata (slots, page rows, cache_seqlens) is bound
                # for cur_batch; anything else (e.g. a micro-batched decode
                # slice) must not replay against it.
                self._fallback("hidden rows != cur_batch size")
                self._step_active = False
            if not self._step_active:
                return orig_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cu_seqlens=cu_seqlens,
                    block_residual=block_residual,
                    **kwargs,
                )
            if self._uses_block_residual:
                expected = self._block_residual_view(layer_idx - 1)
                if not self._same_block_residual(block_residual, expected):
                    self._fallback(
                        "layer {} received moved/misshaped block_residual".format(
                            layer_idx
                        )
                    )
                    self._step_active = False
                    return orig_forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        cu_seqlens=cu_seqlens,
                        block_residual=block_residual,
                        **kwargs,
                    )
                return (
                    self._run_layer(layer_idx, hidden_states),
                    self._block_residual_view(layer_idx),
                )
            return (self._run_layer(layer_idx, hidden_states),)
        return forward

    @staticmethod
    def _same_block_residual(actual, expected) -> bool:
        if actual is None:
            return False
        return (
            tuple(actual.shape) == tuple(expected.shape)
            and tuple(actual.stride()) == tuple(expected.stride())
            and actual.dtype == expected.dtype
            and actual.device == expected.device
            and actual.untyped_storage().data_ptr()
            == expected.untyped_storage().data_ptr()
        )

    def _block_residual_view(self, layer_idx: int) -> torch.Tensor:
        statics = self._statics[self._bucket]
        columns = 0 if layer_idx < 0 else self.segments[layer_idx].num_blocks_after
        return statics.block_residual[: self._bsz, :columns]

    # ------------------------------------------------------------------ #
    #  Step driver                                                        #
    # ------------------------------------------------------------------ #

    def _fallback(self, reason: str) -> None:
        # Warn on first occurrence, then re-warn at decade counts. A
        # persistent fallback means graphs are configured but never used —
        # that must stay visible in steady state, not go quiet after one line.
        n = self._fallbacks_logged.get(reason, 0) + 1
        self._fallbacks_logged[reason] = n
        if n == 1 or n in (10, 100, 1000) or n % 10000 == 0:
            logger.warning(
                "[KIMI_DECODE_GRAPH] rank=%s eager fallback (x%d): %s",
                self.rank, n, reason,
            )

    def _moe_all_ranks_nonempty(self) -> bool:
        """Return the boundary-synchronized EP emptiness decision.

        A graph MoE replay contains global EP collectives.  An empty rank
        cannot enter the patched non-empty attention path, so allowing other
        ranks to replay the MoE graph would mix graph and eager collective
        sequences and deadlock.  The worker publishes this boolean only when
        the decode rank-count vector changes; it is not recomputed per token.
        """
        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )

        return bool(
            getattr(ResidentEPMXFP4MoELayer, "decode_all_ranks_nonempty", False)
        )

    def _ensure_moe_built(self) -> bool:
        """Build/register the K3 resident-MoE graphs once.

        The attention graph remains usable when this optional path is absent.
        We only enable the MoE graph when every MoE layer has the same
        resident MXFP4 latent layout and a valid TP group; a partial graph
        would otherwise change collective ordering for just some layers.
        """
        if self._moe_built:
            return True
        if self._moe_build_attempted:
            return False
        self._moe_build_attempted = True

        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )
        from .moe_cuda_graph_segments import (
            K3MoEGraphBufferPool,
            K3MoEGraphSegment,
            k3_moe_graph_buckets,
        )

        layers = self.model.model.layers
        if not any(
            bool(getattr(getattr(layer, "block_sparse_moe", None),
                        "use_latent_moe", False))
            for layer in layers
        ):
            # The 48B Kimi-Linear variant has resident BF16 MoE, not the K3
            # latent MXFP4 layout captured by this companion manager.
            return False
        moe_layers = []
        for layer_idx, layer in enumerate(layers):
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None:
                continue
            resident = getattr(moe, "_resident_ep_moe", None)
            if not isinstance(resident, ResidentEPMXFP4MoELayer):
                self._fallback(
                    f"K3 resident-MXFP4 MoE unavailable at layer {layer_idx}"
                )
                return False
            moe_layers.append((layer_idx, moe, resident))

        if not moe_layers:
            return False
        arenas = {}
        for _, _, resident in moe_layers:
            arena = getattr(resident.shard, "arena", None)
            if arena is not None:
                arenas[id(arena)] = arena
        self._moe_arenas = list(arenas.values())

        tp_size = int(getattr(moe_layers[0][1], "attn_tp_size", 1))
        if tp_size <= 1:
            self._fallback("K3 resident-MoE graph requires TP>1")
            return False
        ep_world_size = int(moe_layers[0][2].world_size)
        if any(
            int(getattr(moe, "attn_tp_size", 1)) != tp_size
            or int(getattr(resident, "world_size", 0)) != ep_world_size
            for _, moe, resident in moe_layers
        ):
            self._fallback("K3 MoE layers have inconsistent graph topology")
            return False

        moe_buckets = k3_moe_graph_buckets(self.bucketing.bucket_sizes, tp_size)
        self.moe_bucketing = BatchSizeBucketing(moe_buckets)
        self.moe_manager = CUDAGraphManager(
            self.moe_bucketing, device=self.device
        )
        self.moe_manager.WARMUP_ITERATIONS = WARMUP_ITERATIONS
        self.moe_manager.WARMUP_ITERATIONS_SUBSEQUENT = WARMUP_ITERATIONS

        _, first_moe, first_resident = moe_layers[0]
        shard = first_resident.shard
        deepep_group = None
        if self.deepep_ll:
            import torch.distributed as dist
            # Slice 3: the LL exchange spans the EP world (= the torch
            # distributed world of the K3 server) and needs the TP-sharded
            # latent (own rows are assembled by a TP all_to_all).
            if int(getattr(first_resident, "latent_tp_size", 1)) != tp_size:
                raise RuntimeError(
                    "k3_deepep_ll requires the TP-sharded latent projections")
            if not dist.is_initialized() or dist.get_world_size() != ep_world_size:
                raise RuntimeError(
                    "k3_deepep_ll: torch.distributed world "
                    f"{dist.get_world_size() if dist.is_initialized() else None} "
                    f"!= EP world {ep_world_size}")
            deepep_group = dist.group.WORLD
        self.moe_pool = K3MoEGraphBufferPool(
            world_size=first_resident.world_size,
            tp_size=tp_size,
            num_local_experts=shard.num_local,
            intermediate_size=shard.N,
            latent_size=shard.K_latent,
            hidden_size=first_moe.hidden_dim,
            top_k=first_moe.top_k,
            expert_buckets=moe_buckets,
            device=self.device,
            deepep_group=deepep_group,
        )
        if deepep_group is not None:
            # Construct the NVSHMEM buffer now (a 16-rank handshake, ~11 s)
            # rather than inside the first bucket's warmup.
            t0 = time.perf_counter()
            free0 = torch.cuda.mem_get_info(self.device)[0]
            self.moe_pool.setup()
            free1 = torch.cuda.mem_get_info(self.device)[0]
            logger.info(
                "[KIMI_DECODE_GRAPH] rank=%s DeepEP buffer + pool took %.2f GiB "
                "(free %.2f -> %.2f GiB)",
                self.rank, (free0 - free1) / 2**30, free0 / 2**30, free1 / 2**30,
            )
            logger.info(
                "[KIMI_DECODE_GRAPH] rank=%s DeepEP low-latency EP exchange ready: "
                "max %d tokens/rank, hidden %d, %d experts, rdma buffer %.0f MiB (%.1fs)",
                self.rank, self.moe_pool.deepep.max_tokens_per_rank,
                self.moe_pool.deepep.hidden, self.moe_pool.deepep.num_experts,
                self.moe_pool.deepep.rdma_bytes / 2**20, time.perf_counter() - t0,
            )

        for layer_idx, moe, resident in moe_layers:
            segment = K3MoEGraphSegment(
                moe, resident, self.moe_pool, device=self.device
            )
            name = self._moe_name(layer_idx)
            self.moe_segments[layer_idx] = segment
            self.moe_manager.register_segment(name, segment)
        if deepep_group is not None:
            # Numeric self-check of the exchange on synthetic rows (the graph
            # warmup only sees zeros): smallest and largest bucket.
            first_segment = self.moe_segments[moe_layers[0][0]]
            first_segment.self_check_deepep(
                (min(moe_buckets), max(moe_buckets)),
                num_experts=ep_world_size * shard.num_local,
            )

        self._moe_built = True
        logger.info(
            "[KIMI_DECODE_GRAPH] rank=%s built K3 resident-MoE graphs: "
            "%d layers, TP%d, EP%d, buckets=%s",
            self.rank, len(moe_layers), tp_size, first_resident.world_size,
            moe_buckets,
        )
        return True

    def _moe_name(self, layer_idx: int) -> str:
        return f"kimi_linear/moe_{layer_idx}"

    def _moe_group_bucket_for_step(self) -> Optional[int]:
        if self.moe_bucketing is None:
            return None
        from batchgen.moe.fused_moe_mxfp4_resident import (
            ResidentEPMXFP4MoELayer,
        )

        ntp = int(
            getattr(ResidentEPMXFP4MoELayer, "num_tokens_per_rank", 0) or 0
        )
        if ntp <= 0:
            return None
        tp_size = int(next(iter(self.moe_segments.values())).tp_size)
        group_size = tp_size * ntp
        try:
            return self.moe_bucketing.get_padded_size(group_size)
        except ValueError:
            self._fallback(
                f"K3 MoE group batch {group_size} exceeds graph buckets"
            )
            return None

    def _whole_model_bucket_for_step(self) -> Optional[int]:
        """Return the globally safe model bucket for inline K3 MoE.

        The resident EP collectives are sized from the worker's synchronized
        post-TP ``num_tokens_per_rank``.  Consequently the outer graph must
        use the corresponding TP-group bucket even when this rank's local
        attention batch is smaller.  Using a local bucket here would capture
        different graph shapes in different DP groups and deadlock at the
        first collective.
        """
        if not self._uses_block_residual or not self.moe_segments:
            return None
        moe_bucket = self._moe_group_bucket_for_step()
        if moe_bucket is None:
            return None
        if moe_bucket not in self.bucketing.bucket_sizes:
            self._fallback(
                f"whole-model K3 bucket {moe_bucket} is not configured "
                f"(buckets={self.bucketing.bucket_sizes})"
            )
            return None
        return int(moe_bucket)

    def _ensure_whole_built(self) -> bool:
        """Build the direct-call transformer-body segment once."""
        if self._whole_built:
            return True
        if not self._ensure_moe_built() or not self.moe_segments:
            return False
        from .whole_model_cuda_graph_segments import KimiLinearWholeModelSegment

        first_moe = next(iter(self.moe_segments.values()))
        tp_size = int(first_moe.tp_size)
        tp_rank = int(first_moe.tp_rank)
        dtype = self.segments[0].dtype
        self.whole_manager = CUDAGraphManager(
            self.bucketing, device=self.device
        )
        self.whole_manager.WARMUP_ITERATIONS = WARMUP_ITERATIONS
        self.whole_manager.WARMUP_ITERATIONS_SUBSEQUENT = WARMUP_ITERATIONS
        self.whole_segment = KimiLinearWholeModelSegment(
            model=self.model,
            layer_segments=list(self.segments.values()),
            moe_segments=self.moe_segments,
            statics=self._statics,
            tp_size=tp_size,
            tp_rank=tp_rank,
            device=self.device,
            hidden_size=self.model_config.hidden_size,
            dtype=dtype,
            max_bucket_size=self.bucketing.bucket_sizes[-1],
        )
        self.whole_manager.register_segment(
            self._whole_name, self.whole_segment
        )
        self._whole_built = True
        if self.rank == 0:
            logger.info(
                "[KIMI_DECODE_GRAPH] built K3 whole-model segment: "
                "layers=%d TP%d/TP-rank%d buckets=%s",
                len(self.segments), tp_size, tp_rank,
                self.bucketing.bucket_sizes,
            )
        return True

    @staticmethod
    def _collective_capture_barrier(device) -> None:
        """Synchronize graph capture boundaries when NCCL is present."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        torch.cuda.synchronize(device)
        dist.barrier()

    def capture_mode(self) -> str:
        """The mode in force: the server debug state's
        ``kimi_decode_graph_mode`` when it is set, else the planner's mode.

        The one answer for the capture, for every replay, and for whether the
        manager's kept decode instance owns graphs at all."""
        return _debug_mode() or self.mode

    def prewarm_decode_graphs(self) -> bool:
        """Capture this adapter's whole decode graph ladder, once.

        The worker calls this after it has allocated/bound the real GPU KV
        manager: for a kept decode instance first in its startup decode graph
        capture, before readiness, then at every decode configure; for a
        rebuilt instance at its decode configure. Capture uses only the
        dedicated KDA scratch state and zero-valued graph inputs, so it cannot
        mutate a live request.

        The first call in a captured mode captures every bucket a later step
        can replay and seals the ladder; every later call only verifies that
        the KV / KDA storage the graphs bake has not moved (a moved pool is a
        lifecycle error, raised, never a silent recapture). Steps never
        capture.

        Collective order: everything here is independent of this rank's local
        batch. Every rank — with rows or without — builds the K3 MoE graphs
        (exchange pool setup + the DeepEP self-check) first and then captures
        the same buckets of the same planner ladder in the same order, so the
        captures' EP-world collectives match across the cohort. Returns
        whether a captured ladder is in place.
        """
        mode = self.capture_mode()
        if mode == "eager":
            return False
        if AttnWrapperBase.phase != "decode":
            self._fallback("decode graph prewarm requested outside decode")
            return False
        kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        if kv_manager is None:
            self._fallback("decode graph prewarm has no gpu_paged_kv_manager")
            return False
        signature = self._signature(kv_manager)
        if self._ladder_captured:
            self._require_capture_signature(signature)
            return True

        self._ensure_moe_built()
        if not self._ensure_built(kv_manager):
            return False
        self._kv_signature = signature
        start = time.perf_counter()
        # K3 whole-model capture is only meaningful in graph mode.  Compare
        # mode deliberately retains the proven per-layer compare path until a
        # whole-stack eager reference is wired; it must not silently compare
        # only a subset of the transformer.
        if mode == "graph" and self._whole_model_ladder_available():
            ladder = {"whole_model": self._whole_model_ladder()}
            for bucket in ladder["whole_model"]:
                self._capture_whole_bucket(bucket)
        else:
            ladder = {"attention": list(self.bucketing.bucket_sizes)}
            for bucket in ladder["attention"]:
                self._capture_bucket(bucket)
            if self._moe_built:
                ladder["moe"] = list(self.moe_bucketing.bucket_sizes)
                for bucket in ladder["moe"]:
                    self._capture_moe_bucket(bucket)
        self._ladder_captured = True
        logger.info(
            "[KIMI_DECODE_GRAPH] rank=%s decode graph ladder captured once "
            "(mode=%s %s) in %.2fs; later decode configures replay it",
            self.rank, mode, ladder, time.perf_counter() - start,
        )
        return True

    def _whole_model_ladder_available(self) -> bool:
        """Whether the K3 whole-model graph serves this model's graph steps."""
        return bool(
            self._uses_block_residual
            and self._moe_built
            and self.moe_segments
            and self._ensure_whole_built()
        )

    def _whole_model_ladder(self) -> List[int]:
        """Every whole-model bucket a graph step can select.

        A step's whole-model bucket is its synchronized TP-group MoE bucket
        (``_whole_model_bucket_for_step``), so the ladder is the MoE bucket
        ladder. Each of those must be a decode graph bucket: the planner keeps
        the per-rank cap a TP-size multiple and a bucket. A MoE bucket outside
        the decode ladder would leave every step padded to it with no graph,
        so it is refused here instead of falling back to eager in steady
        state.
        """
        ladder = [int(b) for b in self.moe_bucketing.bucket_sizes]
        missing = [b for b in ladder if b not in self.bucketing.bucket_sizes]
        if missing:
            raise RuntimeError(
                f"K3 whole-model graph buckets {missing} (TP-group MoE ladder "
                f"{ladder}) are not decode graph buckets "
                f"{list(self.bucketing.bucket_sizes)}; a step padded to them "
                "would have no captured graph"
            )
        return ladder

    def _require_capture_signature(self, signature) -> None:
        """The KV / KDA storage the sealed ladder baked must not have moved."""
        if signature == self._kv_signature:
            return
        raise RuntimeError(
            f"[KIMI_DECODE_GRAPH] rank={self.rank}: the KV/KDA storage the "
            f"captured decode graph ladder bakes changed ({self._kv_signature} "
            f"-> {signature}); the ladder is captured once and only replayed, "
            "so the decode instance's KV pool must be remapped at its reserved "
            "addresses (a moved pool needs the instance retired and rebuilt)"
        )

    def _require_moe_shards_resident(self) -> None:
        """A MoE graph replay reads the resident shards: they must be mapped."""
        for arena in self._moe_arenas:
            if not arena.is_mapped:
                raise RuntimeError(
                    f"[KIMI_DECODE_GRAPH] rank={self.rank}: a K3 MoE graph "
                    "replay was requested while the resident EP shard's arena "
                    "is unmapped (decode SUSPEND); replaying it is an illegal "
                    "access, so the decode activation must map and refill it"
                )

    def _begin_step(self) -> None:
        """Decide the step's path and refresh every static buffer.

        Everything here is eager, on the decode stream, ordered before the
        replays that consume it. Nothing here builds or captures: only the
        buckets of the sealed ladder replay, every other step runs eager.
        """
        self._step_active = False
        self._moe_step_active = False
        self._whole_step_active = False
        if AttnWrapperBase.phase != "decode":
            return
        mode = self.capture_mode()
        if mode == "eager":
            return
        if not self._ladder_captured:
            self._fallback(
                "decode graph ladder not captured (no prewarm in a captured mode)"
            )
            return

        seq_ids = list(AttnWrapperBase.cur_batch or [])
        bsz = len(seq_ids)
        if bsz == 0:
            # Empty DP rank.  In graph mode it replays the peers' whole-model
            # bucket with zero valid rows (every row a scratch KDA slot, page 0,
            # no KV write) so the graph-captured EP collectives stay matched;
            # otherwise the eager path early-returns on 0 rows and still
            # drives the MoE collectives (worker no-skip invariant).
            if mode == "graph":
                self._begin_empty_whole_step()
            return
        try:
            bucket = self.bucketing.get_padded_size(bsz)
        except ValueError:
            self._fallback(f"bsz {bsz} exceeds top bucket "
                           f"{self.bucketing.bucket_sizes[-1]}")
            return

        kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        if kv_manager is None:
            self._fallback("no gpu_paged_kv_manager bound")
            return
        # Graph-stable page table, refreshed by the worker at every rebuild.
        # `ensure_...` is a cheap Python-side order check that rebuilds only if
        # the table no longer matches the batch order (glm5 adapter precedent);
        # both accessors raise when the active batch exceeds the reserved
        # graph capacity, which is an eager step.
        ensure = getattr(kv_manager, "ensure_cuda_graph_page_table", None)
        try:
            page_table = (
                ensure(seq_ids) if ensure is not None
                else kv_manager.get_cuda_graph_page_table()
            )
            aux_page_table = self._aux_graph_page_table(seq_ids)
        except (RuntimeError, KeyError) as exc:
            self._fallback(f"graph page table unavailable ({exc})")
            return

        # A moved page table / K cache invalidates the addresses baked into
        # every captured graph; the sealed ladder is never recaptured.
        self._require_capture_signature(self._signature(kv_manager))

        # The outer graph includes the global resident-EP collectives, so its
        # bucket is derived from the synchronized post-TP count rather than
        # this rank's local attention batch.  Empty peers replay the same
        # bucket through ``_begin_empty_whole_step``, so the whole-model path
        # does not wait for every EP rank to hold rows (r27 tail: 150-230 ms
        # eager steps once one node had drained, vs ~80 ms replays).
        if (
            mode == "graph"
            and self._whole_built
            and self._moe_group_bucket_for_step() is not None
        ):
            whole_bucket = self._whole_model_bucket_for_step()
            if whole_bucket is not None and whole_bucket in self._whole_captured:
                self._require_moe_shards_resident()
                self._refresh_statics(
                    whole_bucket, bsz, seq_ids, page_table, aux_page_table
                )
                # ``_run_whole_model`` is entered from the decode driver's
                # ``replay_whole_model_step``, while ``_bucket`` is also used
                # by the legacy layer hook.  Keep a dedicated whole-bucket
                # field: the
                # outer graph bucket can be larger than this rank's local
                # attention batch when another DP group has more rows.
                self._whole_bucket = whole_bucket
                self._bucket = whole_bucket
                self._bsz = bsz
                self.step += 1
                self._step_active = True
                self._whole_step_active = True
                return
            if whole_bucket is not None:
                self._fallback(
                    f"whole-model bucket {whole_bucket} is not in the captured "
                    "ladder"
                )

        if bucket not in self._captured:
            self._fallback(
                f"attention bucket {bucket} is not in the captured ladder"
            )
            return

        # K3's resident latent MoE uses global EP collectives.  Derive its
        # group-sized replay bucket from the worker-synchronized post-TP
        # ``ntp`` rather than from this rank's local batch size.
        if self._moe_built and self._moe_all_ranks_nonempty():
            moe_bucket = self._moe_group_bucket_for_step()
            if moe_bucket is not None and moe_bucket in self._moe_captured:
                self._require_moe_shards_resident()
                self._moe_ntp = moe_bucket // next(
                    iter(self.moe_segments.values())
                ).tp_size
                self._moe_group_bucket = moe_bucket
                self._moe_bucket = moe_bucket
                self._moe_step_active = True

        self._compare_step = (
            mode == "compare" and self.step % max(1, self.compare_every) == 0
        )
        if self._compare_step:
            self.last_compare = {}
            self._compare_arange = torch.arange(
                bsz, dtype=torch.int32, device=self.device
            )
        self._refresh_statics(bucket, bsz, seq_ids, page_table, aux_page_table)

        self._bucket = bucket
        self._bsz = bsz
        self.step += 1
        self._step_active = True

    def _whole_replay_event_pair(self):
        """The reusable ``(start, end)`` timing events for whole-model replay."""
        if self._whole_replay_events is None:
            self._whole_replay_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._whole_replay_events

    def _collect_whole_replay_sample(self) -> None:
        """Fold in the previous replay's event pair once it has completed.

        ``Event.query`` is a non-blocking poll: a replay still in flight stays
        pending and is read on a later step rather than stalling this one, so
        the timed path itself is never synchronized.
        """
        if not self._whole_replay_pending:
            return
        start, end = self._whole_replay_events
        if not end.query():
            return
        self._whole_replay_ms += start.elapsed_time(end)
        self._whole_replay_samples += 1
        self._whole_replay_pending = False

    def _whole_replay_telemetry(self) -> dict:
        samples = self._whole_replay_samples
        staged = self._whole_kv_stage_samples
        return {
            "replays": self._whole_replays,
            "replay_samples": samples,
            "replay_ms_total": self._whole_replay_ms,
            "replay_ms_avg": (
                self._whole_replay_ms / samples if samples else 0.0
            ),
            "kv_stage_samples": staged,
            "kv_stage_host_ms_total": self._whole_kv_stage_host_ms,
            "kv_stage_host_ms_avg": (
                self._whole_kv_stage_host_ms / staged if staged else 0.0
            ),
        }

    def _maybe_log_whole_telemetry(self) -> None:
        if self._whole_replays % WHOLE_TELEMETRY_LOG_EVERY:
            return
        t = self._whole_replay_telemetry()
        logger.info(
            "[KIMI_DECODE_GRAPH] rank=%s whole-model bucket=%d replays=%d "
            "replay=%.3fms (n=%d) kv_stage_host=%.3fms (n=%d)",
            self.rank, self._whole_bucket, t["replays"],
            t["replay_ms_avg"], t["replay_samples"],
            t["kv_stage_host_ms_avg"], t["kv_stage_samples"],
        )

    def _run_whole_model(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replay the captured transformer body and stage MLA KV once."""
        if self.whole_manager is None or self.whole_segment is None:
            raise RuntimeError("K3 whole-model graph is active but not built")
        statics = self._statics[self._whole_bucket]
        self._collect_whole_replay_sample()
        # Only bracket a replay when the previous pair has been consumed; the
        # events are reused, so re-recording a pending pair would drop it.
        timed = not self._whole_replay_pending
        if timed:
            start_event, end_event = self._whole_replay_event_pair()
            start_event.record()
        graph_out = self.whole_manager.replay(
            self._whole_name,
            self._whole_bucket,
            input_ids=input_ids,
            num_valid_tokens=statics.num_valid_tokens,
        )
        if timed:
            end_event.record()
            self._whole_replay_pending = True
        self._whole_replays += 1

        # The outer CausalLM applies lm_head after this method returns. Keep
        # only real local rows; the graph is globally bucketed so all ranks
        # execute identical collective shapes.
        hidden_states = graph_out["hidden_states"][: self._bsz]
        callback = AttnWrapperBase.kv_append_callback
        kv_buffer = self.whole_segment._kv_key_buffer
        if callback is not None and kv_buffer is not None and self._bsz > 0:
            # One device clone for the compact MLA stack, then one lightweight
            # callback per MLA layer.  The graph buffer is physically dense,
            # while the callback still receives the logical engine layer id.
            stage_start = time.perf_counter()
            staged = kv_buffer[:, : self._bsz].clone()
            for logical_layer_idx in self.whole_segment._primary_kv_layers:
                physical_layer_idx = self.whole_segment._logical_to_physical_kv[
                    logical_layer_idx
                ]
                callback(logical_layer_idx, staged[physical_layer_idx], None)
            self._append_whole_index_keys()
            # Host cost of issuing the staging work, not its device time: the
            # clone and the callbacks are asynchronous and must stay that way.
            self._whole_kv_stage_host_ms += (
                time.perf_counter() - stage_start
            ) * 1e3
            self._whole_kv_stage_samples += 1
        self._maybe_log_whole_telemetry()
        return hidden_states

    def _run_layer(self, layer_idx: int, hidden_states: torch.Tensor):
        segment = self.segments[layer_idx]
        eager_ref = None
        if self._compare_step:
            # Reference FIRST, on cloned KDA rows, so the graph replay still
            # sees the pre-step state (the fla/conv update is in place).
            eager_ref = self._eager_span(segment, hidden_states)
        outputs = self.manager.replay(
            self._name(layer_idx), self._bsz, hidden_states=hidden_states
        )
        if eager_ref is not None:
            self._record_compare(segment, outputs, eager_ref)

        k_tensor = outputs.get("k_tensor")
        if k_tensor is not None and AttnWrapperBase.kv_append_callback is not None:
            # Cloned: the worker defers the D2H offload, the static output
            # buffer is overwritten by this segment's next replay.
            AttnWrapperBase.kv_append_callback(layer_idx, k_tensor.clone(), None)
        if segment.is_dsa:
            self._append_index_keys(
                layer_idx, outputs["index_k"], outputs["index_scales"]
            )

        if segment.fold_ffn:
            return outputs["hidden"]
        if self._moe_step_active and layer_idx in self.moe_segments:
            # The decoder's TP-G ranks hold the same group rows.  The resident
            # EP contract needs distinct rows per global rank, so scatter the
            # group batch with the same balanced split as the eager path.  The
            # graph input keeps BOTH the full group rows (for the shared expert)
            # and this rank's local slice (for router/down/EP work).
            moe_input = outputs["normed"].reshape(-1, segment.hidden_size)
            moe_segment = self.moe_segments[layer_idx]
            local = self._scatter_moe_rows(moe_input, moe_segment.tp_size,
                                           moe_segment.tp_rank)
            self._moe_valid_tokens.fill_(local.shape[0])
            moe_out = self.moe_manager.replay(
                self._moe_name(layer_idx),
                self._moe_group_bucket,
                padded=moe_input,
                local=local,
                num_valid_tokens=self._moe_valid_tokens,
            )["moe_output"]
            # NCCL's all-gather is necessarily rank-major.  For a group batch
            # not divisible by TP, remove each rank's deterministic pad span;
            # a divisible batch is already in the original row order and takes
            # the allocation-free fast path.
            moe_out = self._reassemble_moe_rows(
                moe_out, self._bsz, moe_segment.tp_size
            )
            return outputs["residual"] + moe_out.view(
                self._bsz, 1, segment.hidden_size
            )

        # Unsupported variants, empty-rank steps, and explicit eager mode
        # retain the production resident-EP implementation.
        ffn_out = segment.layer._run_ffn(outputs["normed"])
        return outputs["residual"] + ffn_out

    @staticmethod
    def _scatter_moe_rows(hidden_states, group_size, group_rank):
        from .moe_tp_reshard import scatter_rows

        return scatter_rows(hidden_states, group_size, group_rank)

    @staticmethod
    def _reassemble_moe_rows(gathered, num_rows, group_size):
        if int(num_rows) % int(group_size) == 0:
            return gathered[:num_rows]
        from .moe_tp_reshard import reassemble_rows

        ntp = (int(num_rows) + int(group_size) - 1) // int(group_size)
        return reassemble_rows(
            gathered[: int(group_size) * ntp].view(
                int(group_size), ntp, gathered.shape[-1]
            ),
            int(num_rows),
            int(group_size),
        )

    # ------------------------------------------------------------------ #
    #  Build / capture                                                    #
    # ------------------------------------------------------------------ #

    def _name(self, layer_idx: int) -> str:
        return f"kimi_linear/span_{layer_idx}"

    def _ensure_built(self, kv_manager) -> bool:
        """Build segments + reserve the KDA scratch slot (once)."""
        if self._built:
            return True
        if KimiLinearKDAWrapper.slot_manager is None:
            self._fallback("KDA state pools not initialized")
            return False
        table = kv_manager.get_cuda_graph_page_table_storage()
        self._max_pages = int(table.shape[1])
        self._page_size_tokens = int(kv_manager.config.page_size_tokens)
        # The statics hold the width the dense decode reads in place.
        self._static_pages = block_table_width(
            self._max_pages, self._page_size_tokens
        )
        if self._has_dsa:
            aux_manager = self._require_aux_manager()
            aux_table = aux_manager.get_cuda_graph_page_table_storage()
            self._max_aux_pages = int(aux_table.shape[1])
            self._aux_page_size_tokens = int(aux_manager.config.page_size_tokens)
        if self._scratch_slot is None:
            raise RuntimeError(
                "Kimi decode graph KDA scratch slot was not reserved during "
                "installation"
            )
        layers = self.model.model.layers
        dtype = layers[0].input_layernorm.weight.dtype
        for layer_idx, layer in enumerate(layers):
            segment = KimiLinearSpanSegment(
                layer, layer_idx, self._statics, self._page_size_tokens, dtype,
                aux_page_size_tokens=self._aux_page_size_tokens,
                max_aux_context=self._max_aux_pages * self._aux_page_size_tokens,
            )
            self.segments[layer_idx] = segment
            self.manager.register_segment(self._name(layer_idx), segment)
        self._built = True
        if self.rank == 0:
            n_kda = sum(1 for s in self.segments.values() if s.is_kda)
            logger.info(
                "[KIMI_DECODE_GRAPH] rank=%s built %d spans "
                "(%d KDA, %d MLA, %d whole-layer dense), kda_scratch_slot=%d, "
                "page_size=%d max_pages=%d static_pages=%d",
                self.rank, len(self.segments), n_kda,
                len(self.segments) - n_kda,
                sum(1 for s in self.segments.values() if s.fold_ffn),
                self._scratch_slot, self._page_size_tokens, self._max_pages,
                self._static_pages,
            )
        return True

    def _bucket_statics(self, bucket: int) -> _BucketStatics:
        statics = self._statics.get(bucket)
        if statics is None:
            slots = KimiLinearKDAWrapper.state_manager.prepare_decode_step(
                [GRAPH_SCRATCH_SEQ_ID] * bucket
            )
            statics = _BucketStatics(
                bucket,
                self._static_pages,
                self.device,
                slots,
                block_residual_columns=self._block_residual_columns,
                hidden_size=self.model_config.hidden_size,
                dtype=(self.segments[0].dtype
                       if self._uses_block_residual else None),
                max_aux_pages=self._max_aux_pages,
            )
            self._statics[bucket] = statics
        return statics

    def _capture_bucket(self, bucket: int) -> None:
        """Warm up + capture all spans for one bucket (first use of it)."""
        statics = self._bucket_statics(bucket)
        # Point every row at the scratch slot and disable the KV write, so the
        # warmup iterations cannot touch a live sequence's state.
        KimiLinearKDAWrapper.state_manager.prepare_decode_step(
            [GRAPH_SCRATCH_SEQ_ID] * bucket
        )
        statics.arm_for_capture()

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        free_before, _ = torch.cuda.mem_get_info(self.device)
        self.manager.warmup_and_capture_buckets([bucket])
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        free_after, _ = torch.cuda.mem_get_info(self.device)
        used_mib = (free_before - free_after) / (1024 ** 2)
        self._captured.add(bucket)
        self._capture_stats[bucket] = {
            "seconds": elapsed, "mib": used_mib, "segments": len(self.segments),
        }
        logger.warning(
            "[KIMI_DECODE_GRAPH] rank=%s captured bucket=%d: %d spans in "
            "%.2fs, +%.1f MiB",
            self.rank, bucket, len(self.segments), elapsed, used_mib,
        )

    def _capture_moe_bucket(self, bucket: int) -> None:
        """Warm up and capture every K3 MoE layer for one global bucket."""
        if self.moe_manager is None:
            raise RuntimeError("K3 MoE graph manager is not initialized")
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.moe_manager.warmup_and_capture_buckets([bucket])
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        self._moe_captured.add(bucket)
        self._moe_capture_stats[bucket] = {
            "seconds": elapsed,
            "segments": len(self.moe_segments),
        }
        logger.warning(
            "[KIMI_DECODE_GRAPH] rank=%s captured K3 MoE bucket=%d: "
            "%d layers in %.2fs",
            self.rank, bucket, len(self.moe_segments), elapsed,
        )

    def _capture_whole_bucket(self, bucket: int) -> None:
        """Capture one globally-shaped K3 transformer-body graph."""
        if self.whole_manager is None or self.whole_segment is None:
            raise RuntimeError("K3 whole-model graph manager is not initialized")
        statics = self._bucket_statics(bucket)
        KimiLinearKDAWrapper.state_manager.prepare_decode_step(
            [GRAPH_SCRATCH_SEQ_ID] * bucket
        )
        statics.arm_for_capture()
        self._collective_capture_barrier(self.device)
        start = time.perf_counter()
        free_before, _ = torch.cuda.mem_get_info(self.device)
        # Buckets capture lazily one at a time, so the manager's "first bucket"
        # warmup (3 eager forwards, ~1 s: fla autotune + JIT) would repeat for
        # every bucket; after the first whole-model capture one eager forward
        # is enough (r35b: 6 in-wall captures at ~2.4 s each).
        self.whole_manager.WARMUP_ITERATIONS = (
            WARMUP_ITERATIONS if not self._whole_captured else 1
        )
        self.whole_manager.warmup_and_capture_buckets([bucket])
        torch.cuda.synchronize(self.device)
        free_after, _ = torch.cuda.mem_get_info(self.device)
        self._collective_capture_barrier(self.device)
        elapsed = time.perf_counter() - start
        used_mib = (free_before - free_after) / (1024 ** 2)
        self._whole_captured.add(bucket)
        self._whole_capture_stats[bucket] = {
            "seconds": elapsed,
            "mib": used_mib,
            "layers": len(self.segments),
        }
        logger.warning(
            "[KIMI_DECODE_GRAPH] rank=%s captured K3 whole-model bucket=%d: "
            "%d layers in %.2fs, +%.1f MiB",
            self.rank, bucket, len(self.segments), elapsed, used_mib,
        )

    def _drop_graphs(self) -> None:
        """Release every captured bucket and the geometry derived from the KV
        manager (adapter release only). Segments are rebuilt (and
        re-registered on a fresh CUDAGraphManager) by the next prewarm."""
        self._ladder_captured = False
        self._kv_signature = None
        self._drop_whole_graphs()
        for bucket in list(self._captured):
            self.manager.drop_bucket(bucket)
        self._captured = set()
        self._capture_stats = {}
        self._statics = {}
        self.segments = {}
        self._built = False
        self.manager = CUDAGraphManager(self.bucketing, device=self.device)
        self.manager.WARMUP_ITERATIONS = WARMUP_ITERATIONS
        self.manager.WARMUP_ITERATIONS_SUBSEQUENT = WARMUP_ITERATIONS

    def _drop_whole_graphs(self) -> None:
        """Release whole-model graphs before child span state is rebuilt."""
        if self.whole_manager is not None:
            for bucket in list(self._whole_captured):
                self.whole_manager.drop_bucket(bucket)
        self.whole_manager = None
        self.whole_segment = None
        self._whole_built = False
        self._whole_captured = set()
        self._whole_capture_stats = {}
        self._whole_step_active = False
        # The in-flight event pair belongs to a graph that no longer exists;
        # drop the sample rather than attributing it to the rebuilt graph.
        self._whole_replay_pending = False

    def _drop_moe_graphs(self) -> None:
        """Release K3 MoE graphs and their shared static pool."""
        if self.moe_manager is not None:
            for bucket in list(self._moe_captured):
                self.moe_manager.drop_bucket(bucket)
        if self.moe_pool is not None:
            self.moe_pool.release()
        self.moe_manager = None
        self.moe_bucketing = None
        self.moe_pool = None
        self.moe_segments = {}
        self._moe_built = False
        self._moe_build_attempted = False
        self._moe_captured = set()
        self._moe_capture_stats = {}
        self._moe_step_active = False

    @staticmethod
    def _tensor_signature(tensor) -> tuple:
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def _signature(self, kv_manager) -> tuple:
        """Everything external whose storage is baked into the graphs: page
        table, K cache, KDA recurrent/conv pools, and KDA slot-index buffer.

        The MLA spans bake `_k_cache[physical_layer]` — a per-layer SLICE whose
        base address is `data_ptr + p * num_pages * page_stride`, a function of
        the FULL cache shape and not just its base pointer. The worker
        re-creates the KV manager per batch job and sizes it from that rank's
        free HBM, so num_pages varies between jobs. A re-init that reuses the
        same base address with a different page count leaves data_ptr unchanged
        while every slice above physical layer 0 moves — by an amount
        proportional to p, which is why the corruption grew with layer depth.
        Shape/stride MUST be in every tensor signature. Otherwise an allocator
        may reuse the same base pointer for changed geometry while per-layer
        views move, and graphs replay against valid but wrong storage (task
        #12's K-cache failure mode applies equally to KDA views).
        """
        storage = kv_manager.get_cuda_graph_page_table_storage()
        k_cache, _ = kv_manager.get_kv_tensors()
        state_manager = KimiLinearKDAWrapper.state_manager
        kda_state = ()
        if state_manager is not None:
            recurrent = state_manager.get_recurrent_tensors()
            conv = state_manager.get_conv_tensors()
            prepared = state_manager._prepared_state_slots
            kda_state = tuple(
                self._tensor_signature(t)
                for t in (recurrent, *conv, prepared)
            )
        aux_state = ()
        if self._has_dsa:
            # The DSA spans bake the auxiliary cache's per-layer views too.
            aux_manager = self._require_aux_manager()
            aux_k_cache, _ = aux_manager.get_kv_tensors()
            aux_state = (
                self._tensor_signature(
                    aux_manager.get_cuda_graph_page_table_storage()
                ),
                self._tensor_signature(aux_k_cache),
                int(aux_manager.config.page_size_tokens),
            )
        return (
            self._tensor_signature(storage),
            self._tensor_signature(k_cache),
            kda_state,
            int(kv_manager.config.page_size_tokens),
            aux_state,
        )

    # ------------------------------------------------------------------ #
    #  Kimi-K3.1 DSA: the auxiliary (index-key) cache                     #
    # ------------------------------------------------------------------ #

    def _require_aux_manager(self):
        aux_manager = AttnWrapperBase.gpu_paged_kv_manager_aux
        if aux_manager is None:
            raise RuntimeError(
                "Kimi-K3.1 decode graph: the model has DSA indexer layers but no "
                "auxiliary (index-key) GPU KV manager is bound"
            )
        return aux_manager

    def _aux_graph_page_table(self, seq_ids):
        """The auxiliary cache's graph-stable table in ``seq_ids`` order, or
        None for a model without indexer layers. Same accessor contract as
        the latent table: rebuilt only when the order no longer matches,
        raising (an eager step) when the batch exceeds the graph capacity."""
        if not self._has_dsa:
            return None
        aux_manager = self._require_aux_manager()
        ensure = getattr(aux_manager, "ensure_cuda_graph_page_table", None)
        return (
            ensure(seq_ids) if ensure is not None
            else aux_manager.get_cuda_graph_page_table()
        )

    def _append_index_keys(self, layer_idx, index_k, index_scales) -> None:
        """Hand one DSA layer's index keys to the auxiliary Host append.

        Cloned, like the latent: the worker defers the D2H copy and the static
        outputs are overwritten by the next replay. Without an auxiliary
        callback while the latent one is bound, the Host would keep a latent
        token with no index key for decode to score — refused.
        """
        callback = AttnWrapperBase.kv_append_callback_aux
        if callback is None:
            if AttnWrapperBase.kv_append_callback is not None:
                raise RuntimeError(
                    f"Kimi-K3.1 DSA layer {layer_idx}: a latent Host append "
                    "callback is bound but no auxiliary one is"
                )
            return
        if self._bsz <= 0:
            return
        callback(
            layer_idx,
            index_k[: self._bsz].clone(),
            None,
            index_scales[: self._bsz].clone(),
        )

    def _append_whole_index_keys(self) -> None:
        """The whole-model graph's staged index keys, one aux append per DSA
        layer (logical engine layer ids, like the latent)."""
        segment = self.whole_segment
        if segment is None or not segment.index_key_layers:
            return
        for position, layer_idx in enumerate(segment.index_key_layers):
            self._append_index_keys(
                layer_idx,
                segment.index_key_buffer[position],
                segment.index_scale_buffer[position],
            )

    # ------------------------------------------------------------------ #
    #  Per-step static refresh                                            #
    # ------------------------------------------------------------------ #

    def _begin_empty_whole_step(self) -> None:
        """Arm a zero-row whole-model replay on an empty DP rank.

        Mirrors the non-empty branch of ``_begin_step`` minus the local page
        table: the bucket comes from the worker-synchronized post-TP count,
        it must be in the sealed ladder (every rank captured the same one),
        and the statics are refreshed with ``bsz=0`` (== ``arm_for_capture``).
        A rank that cannot take this path returns silently to eager exactly
        as before; its peers then fall back through the same synced decision.
        """
        whole_bucket = self._prepare_empty_whole_bucket()
        if whole_bucket is None:
            return
        self._whole_bucket = whole_bucket
        self._bucket = whole_bucket
        self._bsz = 0
        self.step += 1
        self._step_active = True
        self._whole_step_active = True

    def _prepare_empty_whole_bucket(self) -> Optional[int]:
        """The collective half of an empty DP rank's decode step.

        Selects the peers' synchronized whole-model bucket from the sealed
        ladder and refreshes its statics with zero valid rows. The rowed
        ranks' ``_begin_step`` takes the same decision from the same synced
        count and the same captured set (the ladder capture is identical on
        every rank), so an empty rank replays exactly when they do. Returns
        the bucket, or ``None`` when the whole-model path is off for this
        step (the peers' synced decision then keeps them off it as well).
        """
        kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        if kv_manager is None or not self._whole_built:
            return None
        self._require_capture_signature(self._signature(kv_manager))
        whole_bucket = self._whole_model_bucket_for_step()
        if whole_bucket is None or whole_bucket not in self._whole_captured:
            return None
        self._require_moe_shards_resident()
        self._refresh_statics(whole_bucket, 0, [], None)
        return whole_bucket

    def _refresh_statics(self, bucket, bsz, seq_ids, page_table,
                         aux_page_table=None) -> None:
        state_manager = KimiLinearKDAWrapper.state_manager
        statics = self._bucket_statics(bucket)

        # ONE slot staging for all 20 KDA layers (plan M5 item 2); padded rows
        # take the scratch slot, never -1.
        padded_ids = list(seq_ids)
        if bucket > bsz:
            padded_ids += [GRAPH_SCRATCH_SEQ_ID] * (bucket - bsz)
        state_manager.prepare_decode_step(padded_ids)

        if bsz == 0:
            # Empty DP rank: no batch tensors on the wrapper; every row is
            # padding (KV-write-skipped, page 0, seqlen 1).
            empty = torch.zeros((0,), dtype=torch.int32, device=self.device)
            statics.refresh(0, empty, empty, statics.page_table[:0])
            return
        cache_seqlens = AttnWrapperBase.cache_seqlens[:bsz].to(torch.int32)
        token_indices = (
            AttnWrapperBase.position_ids[:bsz].reshape(-1).to(torch.int32)
        )
        # kimi_k25 refresh pattern: batch-ordered rows of the KV manager's
        # graph-stable table copied into the fixed [bucket, max_pages] buffer
        # (page-table row i == batch row i; the worker rebuilds the table when
        # the batch order changes).
        statics.refresh(
            bsz, cache_seqlens, token_indices, page_table[:bsz, : self._max_pages],
            None if aux_page_table is None
            else aux_page_table[:bsz, : self._max_aux_pages],
        )

        if self._compare_step:
            self._slot_index_long = torch.tensor(
                [state_manager.get_sequence_state_item(s) for s in seq_ids],
                dtype=torch.long, device=self.device,
            )

    # ------------------------------------------------------------------ #
    #  Compare mode (graph vs eager)                                      #
    # ------------------------------------------------------------------ #

    def _clone_kda_state(self, state: KDALayerState) -> KDALayerState:
        """Row-wise copy of this batch's KDA state so the eager reference can
        run without advancing the live pools."""
        idx = self._slot_index_long
        clone = KDALayerState(
            None,
            state.conv_q.index_select(0, idx),
            state.conv_k.index_select(0, idx),
            state.conv_v.index_select(0, idx),
            state.recurrent_pool.index_select(0, idx),
        )
        clone.cur_decode_slots = self._compare_arange
        return clone

    def _eager_span(self, segment, hidden_states) -> Dict[str, torch.Tensor]:
        """Same span, production eager kernels, no live-state mutation.

        KDA runs on cloned state rows. MLA runs the real
        `mla_decoding_nope_with_pagekv`: its paged-KV write is idempotent (same
        value, same slot/position as the graph's), and the KV offload callback
        is deliberately NOT fired here.
        """
        layer = segment.layer

        def eager_attention(normed):
            if segment.is_kda:
                return (
                    kda_decode_serving(
                        segment.attn,
                        normed,
                        self._clone_kda_state(segment.kda_state),
                    ),
                    {},
                )
            if segment.is_dsa:
                # The production eager DSA step; both cache writes are
                # idempotent (same rows, same values as the graph's), and no
                # Host append is fired here.
                result = dsa_decode_forward(
                    segment.attn,
                    normed,
                    AttnWrapperBase.position_ids,
                    AttnWrapperBase.cache_seqlens,
                    AttnWrapperBase.max_seqlen,
                    AttnWrapperBase.gpu_paged_kv_manager,
                    AttnWrapperBase.gpu_paged_kv_manager_aux,
                    segment.layer_idx,
                    list(AttnWrapperBase.cur_batch or []),
                )
                return result.attn_output, {}
            attn_out, k_tensor = mla_decoding_nope_with_pagekv(
                segment.attn,
                normed,
                AttnWrapperBase.position_ids,
                AttnWrapperBase.cache_seqlens,
                AttnWrapperBase.max_seqlen,
                None,
                AttnWrapperBase.gpu_paged_kv_manager,
                segment.layer_idx,
                None,
            )
            return attn_out, {}

        ffn = layer.mlp if segment.fold_ffn else None
        if segment.use_block_residual:

            def append_boundary(flat_prefix, active):
                # Compare mode materializes the one new column privately, so
                # the eager reference cannot advance the bucket's static
                # residual bank before the graph replay.
                return torch.cat(
                    (active, flat_prefix.unsqueeze(1)), dim=1
                )

            out = run_kimi_block_residual_layer_body(
                layer=layer,
                ops=KimiLayerBodyOps(
                    attn=eager_attention,
                    ffn=ffn,
                    append_boundary=append_boundary,
                ),
                hidden_states=hidden_states,
                block_residual=segment._buf.block_residual[
                    : self._bsz, : segment.num_blocks_before
                ],
            )
            out.pop("block_residual")
            return out

        return run_kimi_layer_body(
            layer=layer,
            ops=KimiLayerBodyOps(attn=eager_attention, ffn=ffn),
            hidden_states=hidden_states,
        )

    def _record_compare(self, segment, graph_out, eager_out) -> None:
        deltas = {}
        for key, ref in eager_out.items():
            got = graph_out[key]
            deltas[key] = (got.float() - ref.float()).abs().max().item()
        worst = max(deltas.values()) if deltas else 0.0
        self.last_compare[segment.layer_idx] = worst
        logger.warning(
            "[KIMI_DECODE_GRAPH] compare rank=%s step=%d layer=%d kind=%s "
            "bsz=%d bucket=%d max|d|=%.3e (%s)",
            self.rank, self.step - 1, segment.layer_idx,
            "kda" if segment.is_kda else "mla", self._bsz, self._bucket, worst,
            " ".join(f"{k}={v:.3e}" for k, v in sorted(deltas.items())),
        )


__all__ = [
    "DEFAULT_COMPARE_EVERY",
    "DEFAULT_DECODE_GRAPH_BUCKETS",
    "GRAPH_SCRATCH_SEQ_ID",
    "KimiLinearDecodeGraph",
    "KimiLinearSpanSegment",
]
