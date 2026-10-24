# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Graph-capturable GLM-5.2 attention segment for skip_topk layers.

Runtime-qualified on exact ``h200-instance-3`` (2026-09-17): all 16 default
whole-model graph buckets through 128 were captured, and a 10,240-token prompt
replayed the whole graph through decode completion. Focused contract tests
cover shared-layer setup and output wiring. A dedicated graph-vs-eager numeric
comparison remains part of the final release-candidate validation.

GLM-5.2 places a DSA indexer on 21 of 78 layers; the other 57 ``skip_topk``
layers REUSE the top-k selected by the most recent indexer layer
(glm5_decode_selector.py:443-461: skip layers read
``GLM5AttnWrapper._dsa_prev_topk_indices``; full layers with
``next_skip_topk`` write it). Under whole-model capture that Python handoff
becomes a static-tensor read: this segment consumes the per-bucket
``top_k_indices`` buffer of the PRECEDING ``Glm5FullDsaAttnSegment``
(capture order == layer order, so the producer's buffers exist first).

Relative to the full segment this variant drops every indexer stage
(wk proj / k-norm / rope-hadamard / aux-KV write / head gates / wq_b proj /
score+topk) and keeps the MLA spine: q_a/q_b + kv_a projections, fused
rmsnorm-rope, primary-KV update, the selected-page-table transform over the
BORROWED top-k, FA3, out-absorb, o_proj. It inherits the full segment's two
attention branches, so an ``all_short`` graph skips the transform entirely
and FA3 reads the resident cache.
"""

import logging

from dataclasses import fields
from typing import Dict, Optional

import torch

from batchgen_kernels.attention.dsa.bf16_absorb import (
    BF16AbsorbWeights,
    absorb_out_out,
    absorb_q_out,
)
from batchgen.attention.mla.fa3_backend import act_quant
from batchgen.models.glm.glm5.cuda_graph_segments import (
    Glm5FullDsaAttnSegment,
    _Glm5FullDsaSegmentBuffers,
    _fa3_with_kvcache,
    transform_selected_positions_out,
    w8a8_deepgemm,
    _fused_rmsnorm_rope,
    fused_rmsnorm,
)

# Max-bucket view split for the SKIP-layer buffer set (see the full segment's
# split in cuda_graph_segments.py). Indexer-only fields are 1-element
# placeholders here and pass through from the base unsliced; top_k_indices is
# borrowed per bucket from the producing full segment.
_GLM5_REUSE_VIEW_FIELDS = frozenset({
    "valid_mask",
    "aux_valid_mask",
    "row_indices",
    "valid_rows_bf16",
    "valid_rows_ones",
    "valid_rows_zeros",
    "safe_slot_zeros",
    "skip_slot_neg_ones",
    "safe_seqlen_zeros",
    "kv_primary_slot_indices",
    "kv_aux_slot_indices",
    "safe_primary_slot_indices",
    "safe_aux_slot_indices",
    "safe_cache_seqlens",
    "token_indices",
    "qkv_a",
    "q_a",
    "q_a_normed",
    "q_flat",
    "q_rope_4d",
    "new_compressed_kv",
    "selected_token_ids",
    "selected_lengths",
    "absorbed_q",
    "attn_heads",
})
_GLM5_REUSE_PLACEHOLDER_FIELDS = frozenset({
    "indexer_k_raw",
    "indexer_k_x_fp8",
    "indexer_k_x_scale",
    "indexer_k_tma_desc",
    "q_x_fp8",
    "q_x_scale",
    "q_tma_desc",
    "q_flat_indexer",
    "q_index",
    "head_gates",
    "positions_expanded",
    # Both scorers' statics: a skip layer never scores, so neither the bf16
    # aggregate nor the DeepGEMM paged-MQA inputs are ever read here.
    "agg_scores",
    "q_index_fp8",
    "aux_block_table",
    "aux_context_lens",
    "aux_deep_gemm_context_lens",
})
# Rebuilt per bucket: top_k_indices is borrowed from the producing full
# segment's buffer set FOR THIS BUCKET, never sliced from our own base.
_GLM5_REUSE_SPECIAL_FIELDS = frozenset({"top_k_indices"})
# Left at its None default: the DeepGEMM paged-MQA schedule belongs to the
# indexer layers' prologue, and a skip layer never scores.
_GLM5_REUSE_PER_STEP_FIELDS = frozenset({"aux_schedule_meta"})

_split_delta = (
    set(f.name for f in fields(_Glm5FullDsaSegmentBuffers))
    ^ (_GLM5_REUSE_VIEW_FIELDS | _GLM5_REUSE_PLACEHOLDER_FIELDS
       | _GLM5_REUSE_SPECIAL_FIELDS | _GLM5_REUSE_PER_STEP_FIELDS)
)
if _split_delta:
    raise AssertionError(
        f"_Glm5FullDsaSegmentBuffers fields not covered by (or extraneous in) "
        f"the reuse-segment view split: {sorted(_split_delta)}"
    )


class Glm5ReuseTopkAttnSegment(Glm5FullDsaAttnSegment):
    """Skip-layer segment: MLA spine with a borrowed top-k buffer."""

    # A skip layer never reads or writes the aux cache, so its row-validity
    # mask deliberately does NOT require a valid aux slot -- unlike the full
    # segment's. Distinct variant => distinct hoisted prologue, and the
    # whole-model driver rejects a buffer set claimed by both variants.
    ROW_VALIDITY_VARIANT = "reuse"

    # This segment does NOT run the indexer: it reads the shared
    # ``top_k_indices`` the preceding indexer layer wrote. A run of these
    # layers therefore transforms the SAME selection, which is what lets a
    # driver hoist the transform to the first layer of each run.
    SELECTS_TOP_K = False

    # A skip layer never writes the aux cache, so it takes only the primary
    # host-KV staging slice and keeps its 1-element aux placeholder.
    WRITES_AUX_HOST_KV = False

    def __init__(
        self,
        *,
        wrapper,
        primary_blocked_k: torch.Tensor,
        primary_page_table: torch.Tensor,
        absorb_weights,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        max_seqlen: int,
        index_topk: int,
        page_size: int,
        topk_source: Glm5FullDsaAttnSegment,
        all_short: bool = False,
        shared_buffers: Optional[dict] = None,
    ) -> None:
        # Deliberately do NOT call super().__init__ — it derefs indexer-only
        # weights. Mirror every field the kept spine + inherited helpers read.
        self.wrapper = wrapper
        self.attn = wrapper.module
        self.layer_idx = int(wrapper.layer_idx)
        self.primary_blocked_k = primary_blocked_k
        self.primary_page_table = primary_page_table
        self.absorb_weights = absorb_weights
        self.cos_table = cos_table.contiguous()
        self.sin_table = sin_table.contiguous()
        self.max_seqlen = int(max_seqlen)
        self.index_topk = int(index_topk)
        self.page_size = int(page_size)
        # Same graph-lifetime specialization as the full segment: skip the
        # transform and let FA3 read the resident page-size-N cache.
        self.all_short = bool(all_short)
        self.topk_source = topk_source
        if _fa3_with_kvcache is None:
            raise RuntimeError(
                "GLM-5 DSA graph requires flash_attn_interface "
                "(FlashAttention-3): flash_attn_with_kvcache is unavailable"
            )
        # NOTE: full segments share ONE _buffers dict across all indexer
        # layers (that is what the worker's shared_dsa_buffers is). Skip
        # segments share their OWN dict — passing the full segments' dict
        # here would let our placeholder entries clobber theirs. Because the
        # full dict is shared, topk_source may be ANY full segment.
        self._uses_shared_buffers = shared_buffers is not None
        self._buffers: Dict[int, _Glm5FullDsaSegmentBuffers] = (
            shared_buffers if shared_buffers is not None else {}
        )
        self._outputs: Dict[int, object] = {}
        # Primary-only host-KV staging slice for this layer; see
        # Glm5FullDsaAttnSegment.set_host_kv_staging.
        self._host_kv_staging: Optional[Dict[str, Optional[torch.Tensor]]] = None

    # -- capture-plumbing overrides ------------------------------------------
    # No super() here: the parent's specs/output methods deref aux_blocked_k /
    # attn.indexer, which skip layers do not have.
    def get_static_input_specs(self, bucket_size: int):
        # Parent body is deref-safe for skip layers (literals + attn dims);
        # only the aux slot input is dropped.
        specs = dict(Glm5FullDsaAttnSegment.get_static_input_specs(self, bucket_size))
        specs.pop("aux_slot_indices", None)
        return specs

    def get_static_output_specs(self, bucket_size: int):
        from batchgen.models.glm.glm5.cuda_graph_segments import TensorSpec
        return {
            "attn_output": TensorSpec(
                ("batch_size", 1, self.attn.hidden_size),
                torch.bfloat16,
            ),
            "primary_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.primary_blocked_k.shape[3]),
                torch.bfloat16,
            ),
        }

    def _setup_static_output_buffers(self, bucket_size: int) -> None:
        if bucket_size in self._outputs:
            return
        from batchgen.models.glm.glm5.cuda_graph_segments import (
            _Glm5FullDsaSegmentOutputs,
        )
        device = self.primary_blocked_k.device
        attn = self.attn
        kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
        staging = self._host_kv_staging
        base = (
            self._outputs[max(self._outputs)]
            if self._outputs and bucket_size < max(self._outputs)
            else None
        )
        if base is not None:
            attn_output = base.attn_output[:bucket_size]
            # 1-element placeholder passes through unsliced.
            indexer_k_tensor = base.indexer_k_tensor
        else:
            attn_output = torch.empty(
                bucket_size, attn.hidden_size, dtype=torch.bfloat16, device=device,
            )
            # Never written on this path; 1-element placeholder satisfies the
            # dataclass without allocating per-bucket aux storage.
            indexer_k_tensor = torch.empty(1, dtype=torch.bfloat16, device=device)
        if staging is not None:
            # Staged: the fused norm+rope launch writes the host-append source
            # straight into the composing segment's slice for this layer.
            primary_k_tensor = staging["key"][:bucket_size]
        elif base is not None:
            primary_k_tensor = base.primary_k_tensor[:bucket_size]
        else:
            primary_k_tensor = torch.empty(
                bucket_size, 1, 1, kv_dim, dtype=torch.bfloat16, device=device,
            )
        self._outputs[bucket_size] = _Glm5FullDsaSegmentOutputs(
            primary_k_tensor=primary_k_tensor,
            indexer_k_tensor=indexer_k_tensor,
            attn_output=attn_output,
        )

    def _make_bucket_view_buffers(self, bucket_size: int) -> _Glm5FullDsaSegmentBuffers:
        base_bucket = self._base_bucket()
        source_buffers = self.topk_source._buffers.get(bucket_size)
        if source_buffers is None:
            raise RuntimeError(
                "Glm5ReuseTopkAttnSegment: topk_source has no buffers for "
                f"bucket {bucket_size}; producing indexer-layer segment must "
                "be set up first"
            )
        base = self._buffers[base_bucket]
        view_kwargs = {
            name: getattr(base, name)[:bucket_size]
            for name in _GLM5_REUSE_VIEW_FIELDS
        }
        placeholder_kwargs = {
            name: getattr(base, name) for name in _GLM5_REUSE_PLACEHOLDER_FIELDS
        }
        return _Glm5FullDsaSegmentBuffers(
            top_k_indices=source_buffers.top_k_indices,
            **view_kwargs,
            **placeholder_kwargs,
        )

    def setup_static_buffers(self, bucket_size: int) -> None:
        if bucket_size in self._buffers:
            self._setup_static_output_buffers(bucket_size)
            return
        if self._buffers and bucket_size < max(self._buffers):
            self._buffers[bucket_size] = self._make_bucket_view_buffers(bucket_size)
            self._setup_static_output_buffers(bucket_size)
            return
        if self._buffers:
            logging.warning(
                "GLM-5 reuse-topk statics: bucket %d requested after smaller "
                "base %d; allocating a full set (largest-first capture "
                "restores the max-bucket-view footprint)",
                bucket_size,
                max(self._buffers),
            )
        device = self.primary_blocked_k.device
        attn = self.attn
        kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim

        source_buffers = self.topk_source._buffers.get(bucket_size)
        if source_buffers is None:
            # Producer captures before consumers (layer order); if we get here
            # the whole-model builder wired the wrong source. Fail loudly.
            raise RuntimeError(
                "Glm5ReuseTopkAttnSegment: topk_source has no buffers for "
                f"bucket {bucket_size}; producing indexer-layer segment must "
                "be set up first"
            )

        selected_token_ids = torch.empty(
            bucket_size, self.index_topk,
            dtype=torch.int32, device=device,
        )
        selected_lengths = torch.empty(bucket_size, dtype=torch.int32, device=device)
        selected_lengths.fill_(self._padding_selected_length())
        one_i32 = torch.ones(1, dtype=torch.int32, device=device)
        one_bf16 = torch.ones(1, dtype=torch.bfloat16, device=device)
        qkv_a = torch.empty(
            bucket_size,
            attn.q_lora_rank + kv_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self._buffers[bucket_size] = _Glm5FullDsaSegmentBuffers(
            valid_mask=torch.empty(bucket_size, dtype=torch.bool, device=device),
            aux_valid_mask=torch.empty(bucket_size, dtype=torch.bool, device=device),
            row_indices=torch.arange(bucket_size, dtype=torch.int32, device=device),
            valid_rows_bf16=torch.empty(bucket_size, dtype=torch.bfloat16, device=device),
            valid_rows_ones=torch.ones(bucket_size, dtype=torch.bfloat16, device=device),
            valid_rows_zeros=torch.zeros(bucket_size, dtype=torch.bfloat16, device=device),
            safe_slot_zeros=torch.zeros(bucket_size, dtype=torch.int32, device=device),
            skip_slot_neg_ones=torch.full((bucket_size,), -1, dtype=torch.int32, device=device),
            safe_seqlen_zeros=torch.zeros(bucket_size, dtype=torch.int32, device=device),
            kv_primary_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            kv_aux_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            safe_primary_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            safe_aux_slot_indices=torch.zeros(bucket_size, dtype=torch.int32, device=device),
            safe_cache_seqlens=torch.empty(bucket_size, dtype=torch.int32, device=device),
            token_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            qkv_a=qkv_a,
            q_a=qkv_a[:, : attn.q_lora_rank],
            q_a_normed=torch.empty(
                bucket_size,
                attn.q_lora_rank,
                dtype=torch.bfloat16,
                device=device,
            ),
            q_flat=torch.empty(
                bucket_size, attn.num_heads * attn.q_head_dim,
                dtype=torch.bfloat16, device=device,
            ),
            q_rope_4d=torch.empty(
                bucket_size, attn.num_heads, 1, attn.qk_rope_head_dim,
                dtype=torch.bfloat16, device=device,
            ),
            new_compressed_kv=qkv_a[:, attn.q_lora_rank :].view(
                bucket_size, 1, kv_dim
            ),
            # Indexer-only fields: 1-element placeholders (never read on this
            # path; dataclass requires them).
            indexer_k_raw=one_bf16,
            indexer_k_x_fp8=one_bf16,
            indexer_k_x_scale=one_bf16,
            indexer_k_tma_desc=one_i32,
            q_x_fp8=one_bf16,
            q_x_scale=one_bf16,
            q_tma_desc=one_i32,
            q_flat_indexer=one_bf16,
            q_index=one_bf16,
            head_gates=one_bf16,
            positions_expanded=one_i32,
            agg_scores=one_bf16,
            q_index_fp8=one_bf16,
            aux_block_table=one_i32,
            aux_context_lens=one_i32,
            aux_deep_gemm_context_lens=one_i32,
            # THE handoff: the producing full segment's static top-k buffer.
            top_k_indices=source_buffers.top_k_indices,
            selected_token_ids=selected_token_ids,
            selected_lengths=selected_lengths,
            absorbed_q=torch.empty(
                bucket_size, attn.num_heads, attn.kv_lora_rank,
                dtype=torch.bfloat16, device=device,
            ),
            attn_heads=torch.empty(
                bucket_size, 1, attn.num_heads, attn.v_head_dim,
                dtype=torch.bfloat16, device=device,
            ),
        )
        self._setup_static_output_buffers(bucket_size)

    def compute_row_validity(
        self,
        buffers: _Glm5FullDsaSegmentBuffers,
        *,
        cache_seqlens: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        aux_slot_indices: Optional[torch.Tensor] = None,  # accepted, ignored
        num_valid_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        """Row-validity statics for a skip layer: no aux slot, no aux buffers.

        Deliberately NOT the parent's block. This layer never reads or writes
        the aux cache, so an invalid aux slot must not invalidate the row, and
        ``kv_aux_slot_indices`` / ``safe_aux_slot_indices`` are never read on
        this path. Hoisting is otherwise identical: the inputs are the same
        step-level statics and the outputs are the skip layers' own shared
        buffer set, so the whole-model driver runs this once per step.
        """

        if num_valid_tokens is None:
            torch.ge(primary_slot_indices, 0, out=buffers.valid_mask)
        else:
            torch.lt(buffers.row_indices, num_valid_tokens, out=buffers.valid_mask)
            torch.ge(primary_slot_indices, 0, out=buffers.aux_valid_mask)
            torch.logical_and(buffers.valid_mask, buffers.aux_valid_mask, out=buffers.valid_mask)
        torch.where(
            buffers.valid_mask, primary_slot_indices, buffers.skip_slot_neg_ones,
            out=buffers.kv_primary_slot_indices,
        )
        torch.where(
            buffers.valid_mask, primary_slot_indices, buffers.safe_slot_zeros,
            out=buffers.safe_primary_slot_indices,
        )
        torch.where(
            buffers.valid_mask, cache_seqlens, buffers.safe_seqlen_zeros,
            out=buffers.safe_cache_seqlens,
        )
        torch.where(
            buffers.valid_mask, buffers.valid_rows_ones, buffers.valid_rows_zeros,
            out=buffers.valid_rows_bf16,
        )

    def compute_step_statics(
        self,
        buffers: _Glm5FullDsaSegmentBuffers,
        *,
        position_ids: torch.Tensor,
    ) -> None:
        """The step's position tensors for a skip layer: token indices only.

        Deliberately NOT the parent's block: ``positions_expanded`` feeds the
        indexer Q RoPE, which this layer does not run, so its slot in this
        buffer set is a 1-element placeholder and must stay unwritten.
        Hoisting is otherwise identical -- one step-level input, one shared
        output, so the whole-model driver runs this once per step.
        """

        batch_size = int(buffers.token_indices.shape[0])
        buffers.token_indices.copy_(position_ids.view(batch_size))

    def paged_mqa_prep_key(self, bucket_size: int) -> None:
        """No paged-MQA preparation: a skip layer never scores the aux cache."""
        return None

    def run_paged_mqa_prep(self, bucket_size: int, **kwargs) -> None:
        """Never callable: the key above is ``None``, so a driver skips this.

        Overridden to name the mistake instead of inheriting the parent's
        body, which would dereference an aux page table this segment has no
        field for.
        """
        raise RuntimeError(
            f"Layer {self.layer_idx}: a GLM-5.2 skip layer has no aux "
            "indexer cache and never scores, so it has no paged-MQA "
            "preparation to run; paged_mqa_prep_key returns None and the "
            "driver must skip it"
        )

    def selected_positions_key(self, bucket_size: int) -> tuple:
        """Identity of every tensor this layer's selected-position transform touches.

        Two consecutive skip layers with the same key, and no selecting layer
        between them, transform identical inputs into the same buffers, so the
        second one's launch is pure repetition. Any difference -- a different
        borrowed top-k, a different buffer set, a different page table --
        changes the key and the transform runs again.
        """

        buffers = self._buffers.get(bucket_size)
        if buffers is None:
            self.setup_static_buffers(bucket_size)
            buffers = self._buffers[bucket_size]
        return (
            id(self.primary_page_table),
            id(buffers.top_k_indices),
            id(buffers.safe_cache_seqlens),
            id(buffers.safe_primary_slot_indices),
            id(buffers.selected_token_ids),
            id(buffers.selected_lengths),
            int(self.page_size),
            int(self.index_topk),
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        num_valid_tokens: Optional[torch.Tensor] = None,
        aux_slot_indices: Optional[torch.Tensor] = None,  # accepted, ignored
        row_validity_ready: bool = False,
        paged_mqa_prep_ready: bool = False,  # accepted, ignored: never scores
        selected_positions_ready: bool = False,
        attn_output_mask_deferred: bool = False,
    ) -> Dict[str, torch.Tensor]:
        attn = self.attn
        batch_size = hidden_states.shape[0]
        buffers = self._buffers.get(batch_size)
        if buffers is None:
            self.setup_static_buffers(batch_size)
            buffers = self._buffers[batch_size]
        outputs = self._outputs.get(batch_size)
        if outputs is None:
            self._setup_static_output_buffers(batch_size)
            outputs = self._outputs[batch_size]

        # Layer-invariant: the whole-model driver ran them once for this step
        # over the skip layers' shared buffer set (see compute_row_validity
        # and compute_step_statics).
        if not row_validity_ready:
            self.compute_row_validity(
                buffers,
                cache_seqlens=cache_seqlens,
                primary_slot_indices=primary_slot_indices,
                aux_slot_indices=aux_slot_indices,
                num_valid_tokens=num_valid_tokens,
            )
            self.compute_step_statics(buffers, position_ids=position_ids)

        hidden_flat = hidden_states.view(batch_size, attn.hidden_size).contiguous()
        hidden_fp8, hidden_scale = act_quant(
            hidden_flat,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=num_valid_tokens is not None,
        )
        fused_qkv_a_weight = getattr(self.wrapper, "_fp8_qkv_a_proj", None)
        fused_qkv_a_scale = getattr(self.wrapper, "_fp8_qkv_a_scale", None)
        if fused_qkv_a_weight is None or fused_qkv_a_scale is None:
            raise RuntimeError(
                f"Layer {self.layer_idx}: GLM-5 graph requires fused Q-A/KV-A weights"
            )
        w8a8_deepgemm(
            hidden_fp8, hidden_scale, fused_qkv_a_weight, fused_qkv_a_scale,
            out=buffers.qkv_a, num_valid_tokens=num_valid_tokens, expected_m=batch_size,
        )
        # Masked DeepGEMM leaves rows >= num_valid_tokens unwritten, so this
        # buffer carries stale rows. They are not masked here: every consumer
        # of a masked-GEMM output is row-local (fused_rmsnorm, act_quant, RoPE,
        # the absorb GEMVs, FA3) and every write to state shared across rows is
        # already gated on the same row validity -- see the invariant note on
        # `attn_output`'s mask below, which is the one mask that must stay.
        q_a_normed = fused_rmsnorm(
            buffers.q_a,
            attn.q_a_layernorm.weight,
            attn.q_a_layernorm.eps,
            out=buffers.q_a_normed,
        )
        q_a_fp8, q_a_scale = act_quant(
            q_a_normed,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=num_valid_tokens is not None,
        )
        w8a8_deepgemm(
            q_a_fp8, q_a_scale, attn.q_b_proj.weight,
            self.wrapper.weight_dequant_scale["q_b_proj.weight_scale_inv"],
            out=buffers.q_flat, num_valid_tokens=num_valid_tokens, expected_m=batch_size,
        )
        # NoPE/RoPE split folded into the consumers -- see the matching comment
        # in cuda_graph_segments.py. Both halves are read out of the single
        # q_b_proj output with strides, so neither gather nor its device copy
        # is launched here either; the 57 skip layers and the 21 indexer layers
        # must stay on the same stride contract or they diverge.
        q_heads = buffers.q_flat.view(batch_size, attn.num_heads, attn.q_head_dim)
        q_nope_strided = q_heads[..., : attn.qk_nope_head_dim]
        q_rope_strided = q_heads[..., attn.qk_nope_head_dim :].unsqueeze(2)

        # Gate folded into the producing kernel -- see the matching comment in
        # cuda_graph_segments.py. Still a SELECT (exact zeros for invalid rows),
        # never a multiply: bf16 NaN * 0 is NaN.
        # The paged-cache store is FOLDED INTO the producing kernel (the
        # separate `run_paged_kv_token_update_fused` launch is gone): lane 0
        # stores the row it just computed straight to the page-table
        # destination, gated on the already-gated slot sentinel (the -1 rows
        # of `kv_primary_slot_indices` carry both the num_valid_tokens cutoff
        # and the row mask) and on `row_valid_mask`.
        _fused_rmsnorm_rope(
            buffers.new_compressed_kv,
            buffers.q_rope_4d,
            self.cos_table,
            self.sin_table,
            position_ids,
            attn.kv_a_layernorm.weight,
            attn.kv_lora_rank,
            attn.qk_rope_head_dim,
            eps=attn.kv_a_layernorm.eps,
            out=outputs.primary_k_tensor.view(batch_size, 1, -1),
            row_valid_mask=buffers.valid_mask,
            q_pe_src=q_rope_strided,
            cache_tensor=self.primary_blocked_k,
            page_table=self.primary_page_table,
            slot_indices=buffers.kv_primary_slot_indices,
            token_indices=buffers.token_indices,
            page_size_tokens=self.page_size,
        )

        # num_valid_tokens is forwarded but the BF16 container IGNORES it:
        # the Triton kernel's mask existed only so a Triton program never took
        # a dynamic early return inside a captured graph, and a bmm has no such
        # branch. Safe here because every consumer between this point and
        # act_quant is row-local (see the invariant note above), act_quant
        # never loads rows >= num_valid_tokens, and the one load-bearing mask
        # on `attn_output` below is independent of the absorb. Full argument in
        # BF16AbsorbWeights' docstring.
        absorb_q_out(
            q_nope_strided, self.absorb_weights, buffers.absorbed_q,
            num_valid_tokens=num_valid_tokens,
        )
        if self.all_short:
            attn_out = self._run_all_short_fa3(buffers)
        else:
            # The producer selected the top-k; this layer only re-runs the
            # (cheap, B*index_topk int32 writes) logical->physical transform.
            # Row mode (dense vs top-k) is decided inside the transform from
            # ``cache_seqlens > index_topk``, so no row_modes buffer is needed.
            #
            # RUN-level, not layer-level: every input below is step-level
            # except ``top_k_indices``, which only a selecting layer writes,
            # and the outputs are this variant's shared buffers. So for a run
            # of consecutive skip layers the transform produces the same bytes
            # every time, and the driver runs it on the run's FIRST layer only
            # (``selected_positions_ready`` on the rest). It is NOT shareable
            # with the preceding indexer layer's transform: that one is
            # computed from the FULL variant's row-validity mask, which
            # differs on the aux slot.
            if not selected_positions_ready:
                transform_selected_positions_out(
                    self.primary_page_table,
                    buffers.safe_cache_seqlens,
                    buffers.top_k_indices,           # borrowed from the producer
                    buffers.selected_token_ids,
                    buffers.selected_lengths,
                    page_size=self.page_size,
                    primary_slot_indices=buffers.safe_primary_slot_indices,
                    num_valid_tokens=num_valid_tokens,
                )
            attn_out = self._run_selected_fa3(buffers)
        # Same as the q absorb above: forwarded, ignored by the BF16 path.
        # `attn_heads`' invalid rows are never read -- act_quant below returns
        # early for them without loading.
        absorb_out_out(
            attn_out, self.absorb_weights, buffers.attn_heads,
            num_valid_tokens=num_valid_tokens,
        )
        attn_heads_flat = buffers.attn_heads.reshape(batch_size, attn.num_heads * attn.v_head_dim)
        attn_output_fp8, attn_output_scale = act_quant(
            attn_heads_flat,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=num_valid_tokens is not None,
        )
        w8a8_deepgemm(
            attn_output_fp8, attn_output_scale, attn.o_proj.weight,
            self.wrapper.weight_dequant_scale["o_proj.weight_scale_inv"],
            out=outputs.attn_output,
            num_valid_tokens=num_valid_tokens, expected_m=batch_size,
        )
        # The one load-bearing row mask in this segment. Everything above is
        # row-local or already row-gated, but `attn_output` leaves the segment
        # into the decoder layer's residual add and from there into the MoE
        # router, whose gate top-k and ragged dispatch DO mix rows: an invalid
        # row's stale activations would produce stale expert choices and inflate
        # the per-expert counts. It also covers the rows that are inside
        # num_valid_tokens but have no cache slot -- the masked GEMMs compute
        # those rows, so no num_valid_tokens gate zeroes them, and `valid_mask`
        # is what makes them inert.
        #
        # Deferred when the caller folds it into the residual add that consumes
        # `attn_output`; see cuda_graph_segments.py for the full note.
        if not attn_output_mask_deferred:
            outputs.attn_output.mul_(buffers.valid_rows_bf16.view(batch_size, 1))

        return {
            "attn_output": outputs.attn_output.view(batch_size, 1, attn.hidden_size),
            "primary_k_tensor": outputs.primary_k_tensor,
        }
