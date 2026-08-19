# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""Graph-capturable GLM-5.2 attention segment for skip_topk layers.

DRAFT / UNVALIDATED (2026-08-13, perf plan glm52-h200-perf-optimization P3):
compile-checked only; has NOT been captured, replayed, or compared against
eager. Validate with the graph-vs-eager compare + the standard gauntlet
before any use.

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
rmsnorm-rope, primary-KV update, selected-page-table transform, FA3,
out-absorb, and o_proj.
"""

from typing import Dict, Optional

import torch

from batchgen.attention.mla.fa3_backend import act_quant
from batchgen.models.glm.glm5.cuda_graph_segments import (
    Glm5FullDsaAttnSegment,
    _Glm5FullDsaSegmentBuffers,
    _assert_dsa_buffer_field_coverage,
    fp8_q_absorb_out,
    fp8_out_absorb_out,
    run_paged_kv_token_update_fused,
    transform_selected_positions_out,
    w8a8_deepgemm,
    _fused_rmsnorm_rope,
    fused_rmsnorm,
)


# Per-bucket buffer VIEWS, skip-layer flavour. Same contract as the full
# segment (allocate once at the largest bucket, slice for the rest — see
# cuda_graph_segments._FULL_DSA_BUCKET_DIM_FIELDS), but this dict holds the
# skip layers' own placeholder-carrying sets, so the split differs.
#
# dim 0 == bucket size; every initializer here is slice-invariant
# (torch.empty / arange / ones / zeros / full / fill_).
_REUSE_BUCKET_DIM_FIELDS = (
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
    "qkv_a",
    "q_a",
    "q_a_normed",
    "q_flat",
    "q_nope",
    "q_rope_4d",
    "new_compressed_kv",
    "selected_token_ids",
    "selected_lengths",
    "absorbed_q",
    "attn_heads",
)

# Indexer-only fields: 1-element placeholders, never read on this path. They
# are NOT bucket-shaped, so a view bucket reuses the base's placeholder tensor
# objects unchanged instead of slicing them.
_REUSE_PLACEHOLDER_FIELDS = (
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
    "agg_scores",
)

# Rebuilt per bucket: top_k_indices is borrowed from the producing full
# segment's buffer set FOR THIS BUCKET (never sliced from our own base).
_REUSE_REBUILT_FIELDS = (
    "top_k_indices",
)

_assert_dsa_buffer_field_coverage(
    _REUSE_BUCKET_DIM_FIELDS + _REUSE_PLACEHOLDER_FIELDS + _REUSE_REBUILT_FIELDS,
    "reuse_topk_segment",
)


class Glm5ReuseTopkAttnSegment(Glm5FullDsaAttnSegment):
    """Skip-layer segment: MLA spine with a borrowed top-k buffer."""

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
        self.topk_source = topk_source
        self.all_short = bool(getattr(topk_source, "all_short", False))
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
        self._flashmla_metadata_specs: Dict[int, object] = {}

    # -- capture-plumbing overrides ------------------------------------------
    # No super() here: the parent's specs/output methods deref aux_blocked_k /
    # attn.indexer, which skip layers do not have.
    def get_static_input_specs(self, bucket_size: int):
        # Parent body is deref-safe for skip layers (literals + attn dims +
        # _flashmla_tensor_metadata_specs); only the aux slot input is dropped.
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
        self._outputs[bucket_size] = _Glm5FullDsaSegmentOutputs(
            primary_k_tensor=torch.empty(
                bucket_size, 1, 1, kv_dim, dtype=torch.bfloat16, device=device,
            ),
            # Never written on this path; 1-element placeholder satisfies the
            # dataclass without allocating per-bucket aux storage.
            indexer_k_tensor=torch.empty(1, dtype=torch.bfloat16, device=device),
            attn_output=torch.empty(
                bucket_size, attn.hidden_size, dtype=torch.bfloat16, device=device,
            ),
        )

    # setup_static_buffers itself is INHERITED from Glm5FullDsaAttnSegment: the
    # dispatcher only touches self._buffers, the two hooks below, and
    # _setup_static_output_buffers (all overridden here or deref-safe), so the
    # allocate-once-at-the-largest-bucket policy stays in one place.
    def _topk_source_buffers(self, bucket_size: int):
        source_buffers = self.topk_source._buffers.get(bucket_size)
        if source_buffers is None:
            # Producer captures before consumers (layer order); if we get here
            # the whole-model builder wired the wrong source. Fail loudly.
            raise RuntimeError(
                "Glm5ReuseTopkAttnSegment: topk_source has no buffers for "
                f"bucket {bucket_size}; producing indexer-layer segment must "
                "be set up first"
            )
        return source_buffers

    def _view_static_buffers(
        self,
        base: _Glm5FullDsaSegmentBuffers,
        bucket_size: int,
    ) -> _Glm5FullDsaSegmentBuffers:
        buffers = {
            name: getattr(base, name)[:bucket_size]
            for name in _REUSE_BUCKET_DIM_FIELDS
        }
        buffers.update(
            {name: getattr(base, name) for name in _REUSE_PLACEHOLDER_FIELDS}
        )
        # THE handoff, per bucket: the producing full segment's static top-k
        # buffer for THIS bucket (itself a view of the producer's base).
        buffers["top_k_indices"] = self._topk_source_buffers(bucket_size).top_k_indices
        return _Glm5FullDsaSegmentBuffers(**buffers)

    def _allocate_static_buffers(self, bucket_size: int) -> _Glm5FullDsaSegmentBuffers:
        device = self.primary_blocked_k.device
        attn = self.attn
        kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim

        source_buffers = self._topk_source_buffers(bucket_size)

        qkv_a = torch.empty(
            bucket_size,
            attn.q_lora_rank + kv_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        selected_token_ids = torch.empty(
            bucket_size, self.index_topk,
            dtype=torch.int32, device=device,
        )
        selected_lengths = torch.empty(bucket_size, dtype=torch.int32, device=device)
        selected_lengths.fill_(self._padding_selected_length())
        one_i32 = torch.ones(1, dtype=torch.int32, device=device)
        one_bf16 = torch.ones(1, dtype=torch.bfloat16, device=device)
        return _Glm5FullDsaSegmentBuffers(
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
            q_nope=torch.empty(
                bucket_size, attn.num_heads, attn.qk_nope_head_dim,
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

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        flashmla_tile_scheduler_metadata: torch.Tensor,
        flashmla_num_splits: torch.Tensor,
        num_valid_tokens: Optional[torch.Tensor] = None,
        aux_slot_indices: Optional[torch.Tensor] = None,  # accepted, ignored
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
        valid_rows_bf16_4d = buffers.valid_rows_bf16.view(batch_size, 1, 1, 1)

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
        buffers.qkv_a.mul_(buffers.valid_rows_bf16.view(batch_size, 1))
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
        buffers.q_flat.mul_(buffers.valid_rows_bf16.view(batch_size, 1))
        q_view = buffers.q_flat.view(batch_size, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
        buffers.q_nope.copy_(q_view[..., : attn.qk_nope_head_dim].squeeze(2).contiguous())
        buffers.q_rope_4d.copy_(q_view[..., attn.qk_nope_head_dim :].contiguous())

        offload_kv = _fused_rmsnorm_rope(
            buffers.new_compressed_kv,
            buffers.q_rope_4d,
            self.cos_table,
            self.sin_table,
            position_ids,
            attn.kv_a_layernorm.weight,
            attn.kv_lora_rank,
            attn.qk_rope_head_dim,
            eps=attn.kv_a_layernorm.eps,
        )
        outputs.primary_k_tensor.copy_(offload_kv.view(batch_size, 1, 1, -1))
        token_indices = position_ids.view(batch_size).to(dtype=torch.int32)
        run_paged_kv_token_update_fused(
            k_cache=self.primary_blocked_k,
            k_tokens=outputs.primary_k_tensor.view(batch_size, -1),
            page_table=self.primary_page_table,
            slot_indices=buffers.kv_primary_slot_indices,
            token_indices=token_indices,
            page_size_tokens=self.page_size,
            num_valid_tokens=num_valid_tokens,
        )

        fp8_q_absorb_out(
            buffers.q_nope, self.absorb_weights, buffers.absorbed_q,
            num_valid_tokens=num_valid_tokens,
        )
        if self.all_short:
            attn_out = self._run_all_short_fa3(buffers)
        else:
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
        fp8_out_absorb_out(
            attn_out, self.absorb_weights, buffers.attn_heads,
            num_valid_tokens=num_valid_tokens,
        )
        buffers.attn_heads.mul_(valid_rows_bf16_4d)
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
        outputs.attn_output.mul_(buffers.valid_rows_bf16.view(batch_size, 1))

        return {
            "attn_output": outputs.attn_output.view(batch_size, 1, attn.hidden_size),
            "primary_k_tensor": outputs.primary_k_tensor,
        }
