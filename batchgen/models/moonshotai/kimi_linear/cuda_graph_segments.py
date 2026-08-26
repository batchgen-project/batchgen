# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
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
      fused_recurrent_kda_fwd (KDA), paged-KV token write + FlashMLA (MLA),
      the attention TP all-reduce, o_proj, residual adds, dense MLP of layer 0.
Out:  MoE (collectives), embedding, final norm, lm_head, KV offload callback
      (fired post-replay with a cloned k_tensor), KDA slot alloc/free/zeroing,
      and the per-step static-buffer refresh — all eager, same stream, ordered
      before/after the replay (test_kda_segment_capture.py contract).

STATIC BUFFERS
--------------
* hidden_states [bucket, 1, H] — owned by `CUDAGraphManager` (one per span),
  refreshed by `manager.replay(..., hidden_states=...)`, pad rows zeroed.
* cache_seqlens / token_indices / slot_indices / num_valid_tokens / page_table
  — one shared `_BucketStatics` set per bucket (kimi_k25 adapter refresh
  pattern: the batch's rows of the KV manager's graph-stable page table are
  copied into a fixed [bucket, max_pages] buffer each step).
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
  * FlashMLA reads pad rows with cache_seqlens=1 and page_table=0 (a valid
    page); the outputs are discarded by the bucket->bsz slice.

Buckets are rank-local: the MoE all_gather/all_reduce sees the unpadded local
rows (graph outputs are sliced back to bsz before the eager FFN), so no
cross-rank bucket lockstep is required in Phase A. Phase B (whole-model graph
with in-graph all_reduce) is out of scope here.

Mode is planner/config-driven (`Basic_Config.decode_graph_mode`, default
"eager") with a batch-level override through
`batchgen_debug.kimi_decode_graph_mode` = "graph" | "eager" | "compare"
(glm5_moe_mode precedent). No env vars, new or old.
"""

import logging
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from batchgen.cuda_graph.graph_manager import (
    BatchSizeBucketing,
    CUDAGraphManager,
    TensorSpec,
)
from batchgen.models.wrappers import AttnWrapperBase

from .block_residual import apply_attn_res, num_block_residual_columns
from .serving_modules import (
    _reduce_mla_tp_output,
    kda_decode_serving,
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
):
    """Graph-safe inline of `serving_modules.mla_decoding_nope_with_pagekv`.

    Identical math; exactly three differences, all forced by capture:
      1. the new token is written with `run_paged_kv_token_update_fused` over
         the STATIC page table instead of
         `gpu_paged_kv_manager.update_layer_decode_new_token` (which re-reads
         the manager's active table / slot tensor — both may be reallocated
         between steps, so their addresses cannot be baked into a graph);
      2. FlashMLA reads the same STATIC page table instead of the manager's
         active block table;
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

    Returns:
        (attn_output (bucket, 1, hidden), k_tensor (bucket, 1, 1, kv_dim)).
    """
    # K3's graph path uses the same pure-BF16 FlashMLA entry points as the
    # eager path.  Import them directly so graph capture does not pull in the
    # unused legacy FP8/FA3 backend and its DeepGEMM dependency.
    from flash_mla import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused

    bsz = hidden_states.shape[0]
    hidden_2d = hidden_states.squeeze(1)

    if attn.q_lora_rank is not None:
        q = F.linear(hidden_2d, attn.q_a_proj.weight)
        q = attn.q_a_layernorm(q)
        q = F.linear(q, attn.q_b_proj.weight)
    else:
        q = F.linear(hidden_2d, attn.q_proj.weight)
    new_compressed_kv = F.linear(
        hidden_2d, attn.kv_a_proj_with_mqa.weight
    ).view(bsz, 1, -1)

    q = q.view(bsz, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1
    )

    # NoPE: k_pe is carried unrotated.
    kv, k_pe = torch.split(
        new_compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
    )
    normed_kv = attn.kv_a_layernorm(kv)
    offload_kv = torch.cat([normed_kv, k_pe], dim=-1)
    k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1))

    # k_cache is the manager's per-layer view. Its address is fixed only for
    # the life of ONE KV manager instance: the slice base is
    # `data_ptr + layer * num_pages * page_stride`, so a re-init with a
    # different num_pages moves it even when data_ptr is reused. That is why
    # `_signature` must carry the cache shape/stride, not just the pointer.
    blocked_k, _, _ = (
        AttnWrapperBase.gpu_paged_kv_manager.get_layer_kv_with_page_table(
            layer_idx=layer_idx
        )
    )
    run_paged_kv_token_update_fused(
        k_cache=blocked_k,
        k_tokens=k_tensor.view(bsz, -1),
        page_table=page_table,
        slot_indices=slot_indices,
        token_indices=token_indices,
        page_size_tokens=page_size_tokens,
        num_valid_tokens=num_valid_tokens,
    )

    kv_b_proj = attn.kv_b_proj.weight.data.view(
        attn.num_heads, -1, attn.kv_lora_rank
    )
    q_absorb = kv_b_proj[:, : attn.qk_nope_head_dim, :]
    out_absorb = kv_b_proj[:, attn.qk_nope_head_dim :, :]

    qk_head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    query_states = torch.empty(
        bsz, attn.num_heads, 1, qk_head_dim,
        dtype=blocked_k.dtype, device=hidden_states.device,
    )
    q_nope = q_nope.squeeze(2)
    query_states[:, :, :, : attn.kv_lora_rank] = torch.einsum(
        "bhd,hdc->bhc", q_nope, q_absorb
    ).view(bsz, attn.num_heads, 1, attn.kv_lora_rank)
    query_states[:, :, :, attn.kv_lora_rank :] = q_pe
    query_states = query_states.view(bsz, 1, attn.num_heads, qk_head_dim)

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, attn.num_heads, 1
    )
    attn_out, _ = flash_mla_with_kvcache(
        query_states,
        blocked_k,
        page_table,
        cache_seqlens,
        attn.kv_lora_rank,
        tile_scheduler_metadata,
        num_splits,
        attn.scaling,
        True,
    )

    attn_output = torch.einsum("bqhc,hdc->bhqd", attn_out, out_absorb)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, attn.num_heads * attn.v_head_dim)
    if attn.use_output_gate:
        gate = F.linear(hidden_2d, attn.g_proj.weight).sigmoid()
        attn_output = attn_output * gate
    attn_output = F.linear(attn_output, attn.o_proj.weight)
    attn_output = _reduce_mla_tp_output(attn, attn_output)
    return attn_output.view(bsz, 1, -1), k_tensor


class _BucketStatics:
    """Fixed-address decode inputs shared by every span at one bucket size.

    Allocated once per bucket (never reallocated — captured graphs bake the
    addresses in) and refreshed in place, eagerly, before the step's replays.
    """

    def __init__(self, bucket: int, max_pages: int, device, kda_slots, *,
                 block_residual_columns: int = 0, hidden_size: int = 0,
                 dtype=None):
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
        if self.block_residual is not None:
            self.block_residual.zero_()

    def refresh(self, bsz: int, cache_seqlens, token_indices, page_rows) -> None:
        """Bind this step's batch into the static buffers (rows >= bsz are
        padding: KV-write-skipped, page 0, seqlen 1)."""
        self.num_valid_tokens.fill_(bsz)
        if bsz < self.bucket:
            self.cache_seqlens[bsz:].fill_(1)
            self.token_indices[bsz:].zero_()
            self.page_table[bsz:].zero_()
        self.cache_seqlens[:bsz].copy_(cache_seqlens, non_blocking=True)
        self.token_indices[:bsz].copy_(token_indices, non_blocking=True)
        self.page_table[:bsz].copy_(page_rows, non_blocking=True)


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
                 page_size_tokens: int, dtype):
        self.layer = layer
        self.layer_idx = layer_idx
        self.attn = getattr(layer.self_attn, "module", layer.self_attn)
        self.is_kda = bool(layer.is_linear_attn)
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
        return specs

    def setup_static_buffers(self, bucket_size: int) -> None:
        """Bind this bucket's shared statics (called by CUDAGraphManager right
        before warmup+capture, so the bindings are what the graph bakes in)."""
        self._buf = self._statics[bucket_size]
        if self.is_kda:
            self.kda_state.cur_decode_slots = self._buf.kda_slots

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_block_residual:
            return self._forward_block_residual(
                hidden_states,
                self._graph_attention,
                self._buf.block_residual,
                write_boundary=True,
            )

        layer = self.layer
        normed = layer.input_layernorm(hidden_states)
        attn_out, k_tensor = self._graph_attention(normed)
        residual = hidden_states + attn_out
        normed_out = layer.post_attention_layernorm(residual)
        if self.fold_ffn:
            out = {"hidden": residual + layer.mlp(normed_out)}
        else:
            out = {"normed": normed_out, "residual": residual}
        if k_tensor is not None:
            out["k_tensor"] = k_tensor
        return out

    def _graph_attention(self, normed):
        buf = self._buf
        if self.is_kda:
            attn_out = kda_decode_serving(self.attn, normed, self.kda_state)
            k_tensor = None
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
            )
        return attn_out, k_tensor

    def _forward_block_residual(
        self,
        hidden_states: torch.Tensor,
        attention,
        block_residual: torch.Tensor,
        *,
        write_boundary: bool,
    ) -> Dict[str, torch.Tensor]:
        """K3's exact per-layer transition with graph-stable residual state.

        ``write_boundary=True`` is the captured path and writes the current
        layer's pre-mix prefix into the bucket-owned static buffer. Compare mode
        passes ``False`` and materializes the one new column privately, so the
        eager reference cannot advance state before the graph replay.
        """
        layer = self.layer
        hidden_size = self.hidden_size
        flat_prefix = hidden_states.reshape(-1, hidden_size)
        active = block_residual[:, : self.num_blocks_before]

        attn_input = hidden_states
        if self.num_blocks_before:
            attn_input = apply_attn_res(
                flat_prefix,
                active,
                layer.self_attention_res_proj,
                layer.self_attention_res_norm,
            ).view_as(hidden_states)

        prefix_sum = hidden_states
        if self.is_block_boundary:
            if write_boundary:
                block_residual[:, self.num_blocks_before].copy_(flat_prefix)
                active = block_residual[:, : self.num_blocks_after]
            else:
                active = torch.cat(
                    (active, flat_prefix.unsqueeze(1)), dim=1
                )
            prefix_sum = None

        normed = layer.input_layernorm(attn_input)
        attn_out, k_tensor = attention(normed)
        prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

        ffn_input = apply_attn_res(
            prefix_sum.reshape(-1, hidden_size),
            active,
            layer.mlp_res_proj,
            layer.mlp_res_norm,
        ).view_as(hidden_states)
        normed_out = layer.post_attention_layernorm(ffn_input)
        if self.fold_ffn:
            out = {"hidden": prefix_sum + layer.mlp(normed_out)}
        else:
            out = {"normed": normed_out, "residual": prefix_sum}
        if k_tensor is not None:
            out["k_tensor"] = k_tensor
        return out


class KimiLinearDecodeGraph:
    """Phase-A CUDA-graph decode driver for Kimi-Linear.

    Installs itself on the model (one patched forward per decoder layer plus a
    step hook on `KimiLinearModel.forward`) instead of going through the
    worker's `ModelCudaGraphAdapter` allowlist, which would need a core edit
    (`batchgen_worker.py`); the model-side contract is otherwise the same.

    Lifecycle: constructed + installed by the PSM at `configure_decoding` when
    the planner asks for it. Graphs are captured lazily, on the first decode
    step that uses a bucket. Every step that cannot be replayed (prefill phase,
    empty rank, bsz above the top bucket, invalid/moved page table, eager mode)
    falls through to the unmodified eager layer forward.
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
    ):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.rank = int(rank)
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
        self._max_pages = 0
        self._page_size_tokens = 0
        self._orig_model_forward = None
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

    # ------------------------------------------------------------------ #
    #  Install / release                                                  #
    # ------------------------------------------------------------------ #

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in _VALID_MODES else "eager"

    def install(self) -> None:
        """Patch the decode forwards (idempotent)."""
        if self._installed:
            return
        inner = self.model.model
        self._orig_model_forward = inner.forward
        inner.forward = self._make_model_forward(self._orig_model_forward)
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
        """Restore the eager forwards and drop every captured bucket."""
        if self._installed:
            inner = self.model.model
            inner.forward = self._orig_model_forward
            if self._orig_new_block_residual is not None:
                inner._new_block_residual = self._orig_new_block_residual
                self._orig_new_block_residual = None
            for layer_idx, orig in self._orig_layer_forwards.items():
                inner.layers[layer_idx].forward = orig
            self._orig_layer_forwards = {}
            self._installed = False
        self._drop_graphs()
        if self._scratch_slot is not None:
            if KimiLinearKDAWrapper.slot_manager is not None:
                KimiLinearKDAWrapper.slot_manager.free(GRAPH_SCRATCH_SEQ_ID)
            self._scratch_slot = None

    def get_capture_stats(self) -> dict:
        stats = dict(self.manager.get_capture_stats())
        stats["kimi_capture"] = dict(self._capture_stats)
        return stats

    # ------------------------------------------------------------------ #
    #  Patched forwards                                                   #
    # ------------------------------------------------------------------ #

    def _make_model_forward(self, orig_forward):
        def forward(*args, **kwargs):
            self._begin_step()
            try:
                return orig_forward(*args, **kwargs)
            finally:
                self._step_active = False
        return forward

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

    def _begin_step(self) -> None:
        """Decide the step's path and refresh every static buffer.

        Everything here is eager, on the decode stream, ordered before the
        replays that consume it.
        """
        self._step_active = False
        if AttnWrapperBase.phase != "decode":
            return
        mode = _debug_mode() or self.mode
        if mode == "eager":
            return

        seq_ids = list(AttnWrapperBase.cur_batch or [])
        bsz = len(seq_ids)
        if bsz == 0:
            # Empty DP rank: the eager path early-returns on 0 rows and still
            # drives the MoE collectives (worker no-skip invariant).
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
        except (RuntimeError, KeyError) as exc:
            self._fallback(f"graph page table unavailable ({exc})")
            return

        # Signature first: a moved page table / K cache invalidates the
        # addresses baked into every graph, so drop before (re)building.
        signature = self._signature(kv_manager)
        if signature != self._kv_signature:
            if self._kv_signature is not None:
                logger.warning(
                    "[KIMI_DECODE_GRAPH] rank=%s KV storage changed "
                    "(%s -> %s); dropping captured graphs",
                    self.rank, self._kv_signature, signature,
                )
                self._drop_graphs()
            self._kv_signature = signature

        if not self._ensure_built(kv_manager):
            return
        if bucket not in self._captured:
            self._capture_bucket(bucket)

        self._compare_step = (
            mode == "compare" and self.step % max(1, self.compare_every) == 0
        )
        if self._compare_step:
            self.last_compare = {}
            self._compare_arange = torch.arange(
                bsz, dtype=torch.int32, device=self.device
            )
        self._refresh_statics(bucket, bsz, seq_ids, page_table)

        self._bucket = bucket
        self._bsz = bsz
        self.step += 1
        self._step_active = True

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

        if segment.fold_ffn:
            return outputs["hidden"]
        # MoE (NCCL collectives) stays eager, outside the graph.
        ffn_out = segment.layer._run_ffn(outputs["normed"])
        return outputs["residual"] + ffn_out

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
        # One reserved slot for padding + warmup rows (plan M5.3): never -1,
        # which the fla kernel would turn into an OOB write before the pool.
        self._scratch_slot = KimiLinearKDAWrapper.slot_manager.alloc(
            GRAPH_SCRATCH_SEQ_ID
        )
        layers = self.model.model.layers
        dtype = layers[0].input_layernorm.weight.dtype
        for layer_idx, layer in enumerate(layers):
            segment = KimiLinearSpanSegment(
                layer, layer_idx, self._statics, self._page_size_tokens, dtype,
            )
            self.segments[layer_idx] = segment
            self.manager.register_segment(self._name(layer_idx), segment)
        self._built = True
        if self.rank == 0:
            n_kda = sum(1 for s in self.segments.values() if s.is_kda)
            logger.info(
                "[KIMI_DECODE_GRAPH] rank=%s built %d spans "
                "(%d KDA, %d MLA, %d whole-layer dense), kda_scratch_slot=%d, "
                "page_size=%d max_pages=%d",
                self.rank, len(self.segments), n_kda,
                len(self.segments) - n_kda,
                sum(1 for s in self.segments.values() if s.fold_ffn),
                self._scratch_slot, self._page_size_tokens, self._max_pages,
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
                self._max_pages,
                self.device,
                slots,
                block_residual_columns=self._block_residual_columns,
                hidden_size=self.model_config.hidden_size,
                dtype=(self.segments[0].dtype
                       if self._uses_block_residual else None),
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

    def _drop_graphs(self) -> None:
        """Release every captured bucket and the geometry derived from the KV
        manager. Segments are rebuilt (and re-registered on a fresh
        CUDAGraphManager) on the next eligible step."""
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
        return (
            self._tensor_signature(storage),
            self._tensor_signature(k_cache),
            kda_state,
            int(kv_manager.config.page_size_tokens),
        )

    # ------------------------------------------------------------------ #
    #  Per-step static refresh                                            #
    # ------------------------------------------------------------------ #

    def _refresh_statics(self, bucket, bsz, seq_ids, page_table) -> None:
        state_manager = KimiLinearKDAWrapper.state_manager
        statics = self._bucket_statics(bucket)

        # ONE slot staging for all 20 KDA layers (plan M5 item 2); padded rows
        # take the scratch slot, never -1.
        padded_ids = list(seq_ids)
        if bucket > bsz:
            padded_ids += [GRAPH_SCRATCH_SEQ_ID] * (bucket - bsz)
        state_manager.prepare_decode_step(padded_ids)

        cache_seqlens = AttnWrapperBase.cache_seqlens[:bsz].to(torch.int32)
        token_indices = (
            AttnWrapperBase.position_ids[:bsz].reshape(-1).to(torch.int32)
        )
        # kimi_k25 refresh pattern: batch-ordered rows of the KV manager's
        # graph-stable table copied into the fixed [bucket, max_pages] buffer
        # (page-table row i == batch row i; the worker rebuilds the table when
        # the batch order changes).
        statics.refresh(
            bsz, cache_seqlens, token_indices, page_table[:bsz, : self._max_pages]
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
                    None,
                )
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
            return attn_out, k_tensor

        if segment.use_block_residual:
            return segment._forward_block_residual(
                hidden_states,
                eager_attention,
                segment._buf.block_residual[: self._bsz],
                write_boundary=False,
            )

        normed = layer.input_layernorm(hidden_states)
        attn_out, _ = eager_attention(normed)
        residual = hidden_states + attn_out
        normed_out = layer.post_attention_layernorm(residual)
        if segment.fold_ffn:
            return {"hidden": residual + layer.mlp(normed_out)}
        return {"normed": normed_out, "residual": residual}

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
