"""GLM-5 DSA eager decode attention segment.

This module owns the full eager selector segment for GLM-5 DSA decode:
decode hidden states and metadata enter here, the GLM-5 DSA kernels are
invoked, sparse MLA attention runs on FlashAttention-3 directly over the
resident paged KV, and the return value carries the attention output ready
for out-absorb / o_proj.

The eager path shares its attention backend with the GLM-5 whole-model graph
(``Glm5FullDsaAttnSegment._run_selected_fa3`` / ``_run_all_short_fa3``): the
selected KV is never gathered into a padded buffer. Instead the logical
top-k positions are transformed into physical token IDs that act as an FA3
page table over a page-size-1 view of the cache.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch

try:
    from flash_attn_interface import flash_attn_with_kvcache as _fa3_with_kvcache
except ImportError:  # pragma: no cover - resolved by the node's FA3 install
    _fa3_with_kvcache = None

from batchgen.attention.mla.fa3_backend import act_quant
from batchgen.attention.mla.fused_rmsnorm_rope import (
    fused_rmsnorm_rope_with_q_native as _fused_rmsnorm_rope,
)
from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
from batchgen.models.glm.glm5.decode_utils import (
    build_batch_slot_indices,
    build_clamped_dense_token_indices,
    reorder_block_table_to_batch_slots,
)
from batchgen.models.wrappers import AttnWrapperBase
# Phase C: _dsa_short_count moved from AttnWrapperBase to GLM5AttnWrapper
# (audit §A finding #8).
from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper
from batchgen.timing import get_decode_timer
from batchgen_kernels.attention.dsa.selected_page_table import (
    transform_selected_positions_out,
)


def _slot_indices_override(
    attr_name: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    # Phase C: glm5_decode_*_slot_indices moved to GLM5AttnWrapper
    # (audit §A finding #8). The legacy AttnWrapperBase read is kept as
    # a fallback for graph backends that may still bind via the old name.
    override = getattr(GLM5AttnWrapper, attr_name, None)
    if override is None:
        override = getattr(AttnWrapperBase, attr_name, None)
    if override is None:
        return None
    if override.shape[0] < batch_size:
        raise RuntimeError(
            f"GLM-5 decode slot override {attr_name} has too few rows: "
            f"{override.shape[0]} < {batch_size}"
        )
    return override[:batch_size].to(device=device, dtype=torch.int32)


@dataclass(frozen=True)
class Glm5DsaFlashMlaInputs:
    """Outputs of the GLM-5 DSA eager attention segment.

    ``attn_out`` is the FlashAttention-3 sparse MLA output, ``[B, 1, H,
    kv_lora_rank]``, i.e. what the consumer feeds straight into out-absorb.
    The gather-era fields are gone with the gather: there is no padded
    ``selected_mla_kv`` slab, no packed FlashMLA ``query_states`` and no
    ``PreparedSparseFlashMlaDecode``.
    """

    attn_out: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    selected_lengths: torch.Tensor
    # Physical page-size-1 token IDs handed to FA3 as its ``page_table``,
    # exposed for debug only. ``None`` on the all-short fast path, which
    # attends over the real page table and therefore builds no selection.
    selected_token_ids: torch.Tensor | None
    row_modes: torch.Tensor
    primary_k_tensor: torch.Tensor
    indexer_k_tensor: torch.Tensor | None
    branch_label: str


@dataclass(frozen=True)
class Glm5DsaGraphSegmentInputs:
    """Inputs for `Glm5DsaAttnSegment` plus KV tensors for host callbacks."""

    q_a: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    head_gates: torch.Tensor
    cache_seqlens: torch.Tensor
    positions_expanded: torch.Tensor
    primary_slot_indices: torch.Tensor
    aux_slot_indices: torch.Tensor
    primary_k_tensor: torch.Tensor
    indexer_k_tensor: torch.Tensor


def build_glm5_dsa_graph_segment_inputs(
    wrapper,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    gpu_paged_kv_manager,
    gpu_paged_kv_manager_aux,
    *,
    write_kv: bool = True,
) -> Glm5DsaGraphSegmentInputs:
    """Build graph-segment inputs and update primary/aux KV caches.

    This is the graph-route prefix for GLM-5 DSA decode. It intentionally stops
    before scoring/selection/FlashMLA so those operations can be replayed by
    `Glm5DsaAttnSegment` using static graph buffers and the persistent paged
    primary/aux KV tensors.
    """

    weight_scale = wrapper.weight_dequant_scale
    attn = wrapper.module
    indexer = attn.indexer
    bsz = hidden_states.shape[0]
    dt = get_decode_timer()
    li = wrapper.layer_idx

    if bsz == 0:
        raise ValueError("empty GLM-5 DSA batches must be handled before graph prep")

    with (dt.timed("act_quant", li) if dt else nullcontext()):
        hidden_flat = hidden_states.squeeze(1)
        hidden_fp8, hidden_scale = act_quant(hidden_flat)

    with (dt.timed("q_proj", li) if dt else nullcontext()):
        q_a = w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            attn.q_a_proj.weight,
            weight_scale["q_a_proj.weight_scale_inv"],
        )
        q_a_normed = attn.q_a_layernorm(q_a).contiguous()
        q_a_fp8, q_a_scale = act_quant(q_a_normed)
        q = w8a8_deepgemm(
            q_a_fp8,
            q_a_scale,
            attn.q_b_proj.weight,
            weight_scale["q_b_proj.weight_scale_inv"],
        )
        q = q.view(bsz, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1,
        )
        q_nope = q_nope.squeeze(2).contiguous()
        q_pe = q_pe.contiguous()

    with (dt.timed("kv_proj", li) if dt else nullcontext()):
        new_compressed_kv = w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            attn.kv_a_proj_with_mqa.weight,
            weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
        ).view(bsz, 1, -1)
        cos, sin = attn.rotary_emb(q_pe, seq_len=max_seqlen)
        offload_kv = _fused_rmsnorm_rope(
            new_compressed_kv,
            q_pe,
            cos,
            sin,
            position_ids,
            attn.kv_a_layernorm.weight,
            attn.kv_lora_rank,
            attn.qk_rope_head_dim,
            eps=attn.kv_a_layernorm.eps,
        )
        q_rope = q_pe.squeeze(2).contiguous()

    new_token_pos = position_ids.squeeze(-1)
    manager_device = gpu_paged_kv_manager.device
    seq_lengths_i32 = new_token_pos.to(dtype=torch.int32, device=manager_device)
    aux_device = gpu_paged_kv_manager_aux.device
    primary_slot_indices = _slot_indices_override(
        "glm5_decode_primary_slot_indices",
        bsz,
        manager_device,
    )
    aux_slot_indices = _slot_indices_override(
        "glm5_decode_aux_slot_indices",
        bsz,
        aux_device,
    )
    if primary_slot_indices is None or aux_slot_indices is None:
        current_batch = list(AttnWrapperBase.cur_batch) if AttnWrapperBase.cur_batch else []
        if primary_slot_indices is None:
            primary_slot_indices = build_batch_slot_indices(
                current_batch,
                gpu_paged_kv_manager._gpu_page_table_manager.seq_id_to_slot,
                bsz,
                manager_device,
            )
        if aux_slot_indices is None:
            aux_slot_indices = build_batch_slot_indices(
                current_batch,
                gpu_paged_kv_manager_aux._gpu_page_table_manager.seq_id_to_slot,
                bsz,
                aux_device,
            )

    k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1))
    if k_tensor.device != manager_device:
        k_tensor = k_tensor.to(manager_device)
    if write_kv:
        with (dt.timed("kv_write", li) if dt else nullcontext()):
            gpu_paged_kv_manager.update_layer_decode_new_token(
                k_tensor=k_tensor,
                v_tensor=None,
                sequence_lengths=seq_lengths_i32,
                layer_idx=li,
                slot_indices=primary_slot_indices,
            )

    with (dt.timed("indexer_k", li) if dt else nullcontext()):
        if wrapper._indexer_cuda_weights is None:
            raise RuntimeError(
                f"[layer {wrapper.layer_idx}] GLM-5 DSA graph route requires WP2 "
                "fused indexer KV projection; PyTorch fallback is disabled"
            )
        from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
            cuda_wk_proj_gemm_only,
        )

        k_raw = cuda_wk_proj_gemm_only(
            hidden_flat,
            wrapper._indexer_cuda_weights,
            wrapper._indexer_cuda_module,
        )
        k_normed = indexer.k_norm(k_raw)
        indexer_k_tensor = indexer._fused_rope_hadamard_or_fallback(
            k_normed.unsqueeze(1), new_token_pos, max_seqlen=max_seqlen,
        ).unsqueeze(2)
        seq_lengths_i32_aux = (
            seq_lengths_i32
            if aux_device == manager_device
            else new_token_pos.to(dtype=torch.int32, device=aux_device)
        )
        if write_kv:
            gpu_paged_kv_manager_aux.update_layer_decode_new_token(
                k_tensor=indexer_k_tensor,
                v_tensor=None,
                sequence_lengths=seq_lengths_i32_aux,
                layer_idx=li,
                slot_indices=aux_slot_indices,
            )


    with (dt.timed("indexer_score", li) if dt else nullcontext()):
        from batchgen_kernels.attention.dsa.fused_indexer_score import compute_head_gates

        head_gates = compute_head_gates(
            hidden_flat,
            indexer.weights_proj.weight.data,
            indexer.index_n_heads,
            indexer.index_head_dim,
        )
        positions_expanded = new_token_pos[:, None].expand(
            bsz,
            indexer.index_n_heads,
        ).contiguous()

    return Glm5DsaGraphSegmentInputs(
        q_a=q_a_normed,
        q_nope=q_nope,
        q_rope=q_rope,
        head_gates=head_gates,
        cache_seqlens=cache_seqlens.to(dtype=torch.int32, device=manager_device),
        positions_expanded=positions_expanded,
        primary_slot_indices=primary_slot_indices.to(dtype=torch.int32, device=manager_device),
        aux_slot_indices=aux_slot_indices.to(dtype=torch.int32, device=manager_device),
        primary_k_tensor=k_tensor,
        indexer_k_tensor=indexer_k_tensor,
    )


def build_glm5_dsa_flashmla_inputs(
    wrapper,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    gpu_paged_kv_manager,
    gpu_paged_kv_manager_aux,
    *,
    return_selected_indices: bool = False,
) -> Glm5DsaFlashMlaInputs:
    """Run the GLM-5 BF16 DSA eager decode segment through sparse attention."""

    weight_scale = wrapper.weight_dequant_scale
    attn = wrapper.module
    indexer = attn.indexer
    bsz = hidden_states.shape[0]
    dt = get_decode_timer()
    li = wrapper.layer_idx

    if bsz == 0:
        raise ValueError("empty GLM-5 DSA batches must be handled before FlashMLA prep")

    with (dt.timed("act_quant", li) if dt else nullcontext()):
        hidden_flat = hidden_states.squeeze(1)
        hidden_fp8, hidden_scale = act_quant(hidden_flat)

    with (dt.timed("q_proj", li) if dt else nullcontext()):
        q_a = w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            attn.q_a_proj.weight,
            weight_scale["q_a_proj.weight_scale_inv"],
        )
        q_a_normed = attn.q_a_layernorm(q_a)
        q_a_fp8, q_a_scale = act_quant(q_a_normed)
        q = w8a8_deepgemm(
            q_a_fp8,
            q_a_scale,
            attn.q_b_proj.weight,
            weight_scale["q_b_proj.weight_scale_inv"],
        )
        q = q.view(bsz, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1,
        )
        q_pe = q_pe.contiguous()

    with (dt.timed("kv_proj", li) if dt else nullcontext()):
        new_compressed_kv = w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            attn.kv_a_proj_with_mqa.weight,
            weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
        ).view(bsz, 1, -1)
        cos, sin = attn.rotary_emb(q_pe, seq_len=max_seqlen)
        offload_kv = _fused_rmsnorm_rope(
            new_compressed_kv,
            q_pe,
            cos,
            sin,
            position_ids,
            attn.kv_a_layernorm.weight,
            attn.kv_lora_rank,
            attn.qk_rope_head_dim,
            eps=attn.kv_a_layernorm.eps,
        )

    new_token_pos = position_ids.squeeze(-1)
    manager_device = gpu_paged_kv_manager.device
    seq_lengths_i32 = new_token_pos.to(dtype=torch.int32, device=manager_device)
    aux_device = gpu_paged_kv_manager_aux.device
    primary_slot_indices = _slot_indices_override(
        "glm5_decode_primary_slot_indices",
        bsz,
        manager_device,
    )
    aux_slot_indices = _slot_indices_override(
        "glm5_decode_aux_slot_indices",
        bsz,
        aux_device,
    )
    slot_override_active = primary_slot_indices is not None
    if primary_slot_indices is None or aux_slot_indices is None:
        current_batch = list(AttnWrapperBase.cur_batch) if AttnWrapperBase.cur_batch else []
        if primary_slot_indices is None:
            primary_slot_indices = build_batch_slot_indices(
                current_batch,
                gpu_paged_kv_manager._gpu_page_table_manager.seq_id_to_slot,
                bsz,
                manager_device,
            )
        if aux_slot_indices is None:
            aux_slot_indices = build_batch_slot_indices(
                current_batch,
                gpu_paged_kv_manager_aux._gpu_page_table_manager.seq_id_to_slot,
                bsz,
                aux_device,
            )

    verify_indices = os.environ.get("BATCHGEN_GLM5_VERIFY_INDICES", "0") == "1"
    if verify_indices and wrapper.layer_idx <= 4:
        _log_dsa_bounds(
            wrapper,
            bsz,
            cache_seqlens,
            max_seqlen,
            new_token_pos,
            primary_slot_indices,
            aux_slot_indices,
            gpu_paged_kv_manager,
            gpu_paged_kv_manager_aux,
        )

    with (dt.timed("kv_write", li) if dt else nullcontext()):
        k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1))
        if k_tensor.device != manager_device:
            k_tensor = k_tensor.to(manager_device)
        gpu_paged_kv_manager.update_layer_decode_new_token(
            k_tensor=k_tensor,
            v_tensor=None,
            sequence_lengths=seq_lengths_i32,
            layer_idx=li,
            slot_indices=primary_slot_indices,
        )
        if AttnWrapperBase.kv_append_callback is not None:
            AttnWrapperBase.kv_append_callback(li, k_tensor, None)

    indexer_k_tensor = None
    if wrapper.module.indexer is not None:
        with (dt.timed("indexer_k", li) if dt else nullcontext()):
            if wrapper._indexer_cuda_weights is not None:
                from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
                    cuda_wk_proj_gemm_only,
                )

                k_raw = cuda_wk_proj_gemm_only(
                    hidden_flat,
                    wrapper._indexer_cuda_weights,
                    wrapper._indexer_cuda_module,
                )
                k_normed = indexer.k_norm(k_raw)
                indexer_kv = indexer._fused_rope_hadamard_or_fallback(
                    k_normed.unsqueeze(1), new_token_pos, max_seqlen=max_seqlen,
                ).unsqueeze(2)
            else:
                raise RuntimeError(
                    f"[layer {wrapper.layer_idx}] GLM-5 DSA selector requires WP2 "
                    "fused indexer KV projection; PyTorch fallback is disabled"
                )
            indexer_k_tensor = indexer_kv
            seq_lengths_i32_aux = (
                seq_lengths_i32
                if aux_device == manager_device
                else new_token_pos.to(dtype=torch.int32, device=aux_device)
            )
            gpu_paged_kv_manager_aux.update_layer_decode_new_token(
                k_tensor=indexer_k_tensor,
                v_tensor=None,
                sequence_lengths=seq_lengths_i32_aux,
                layer_idx=li,
                slot_indices=aux_slot_indices,
            )
            if AttnWrapperBase.kv_append_callback_aux is not None:
                AttnWrapperBase.kv_append_callback_aux(li, indexer_k_tensor, None)

    with (dt.timed("indexer_score", li) if dt else nullcontext()):
        if wrapper.module.skip_topk:
            top_k_indices = type(wrapper)._dsa_prev_topk_indices
            assert top_k_indices is not None, "shared DSA layer has no carried top-k; layer 0 must be full"
            index_topk = top_k_indices.shape[-1]
            row_modes = (cache_seqlens > index_topk).to(torch.int32)
            branch_label = "reuse-shared"
        else:
            # `_dsa_prev_topk_indices` is the ONLY remaining consumer of the
            # dense-short-circuit index build: with the gather gone, the
            # all-short fast path below reads the real page table and never
            # looks at logical top-k positions. So the dense build is skipped
            # exactly when this layer does not have to carry top-k to a
            # top-k-reusing layer. (The transform kernel does ignore carried
            # indices for rows with `seqlen <= index_topk` and re-derives
            # 0..len-1 itself, so an all-short carry is formally redundant —
            # but the reuse layer also reads `top_k_indices.shape[-1]` for
            # `index_topk`, and nothing else in the eager path guarantees the
            # reuse family stays all-short, so the carry is still produced.)
            top_k_indices, branch_label, row_modes = _select_glm5_dsa_indices(
                wrapper,
                hidden_states,
                q_a_normed,
                cache_seqlens,
                max_seqlen,
                new_token_pos,
                gpu_paged_kv_manager_aux,
                aux_slot_indices,
                need_dense_indices=bool(wrapper.module.next_skip_topk),
            )
            index_topk = wrapper.module.indexer.index_topk
            if wrapper.module.next_skip_topk:
                type(wrapper)._dsa_prev_topk_indices = top_k_indices

    with (dt.timed("q_absorb", li) if dt else nullcontext()):
        absorbed_q = _absorb_q_nope(wrapper, q_nope)

    mla_blocked_k, _, mla_block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(li)
    mla_page_size = gpu_paged_kv_manager.config.page_size_tokens
    # Slot semantics, identical to the retired gather: with an active slot
    # override the storage-ordered table is indexed through
    # `primary_slot_indices`; otherwise the table is reordered into batch
    # order up front and no slot indirection remains.
    primary_selector_slots = None
    if slot_override_active:
        primary_selector_slots = primary_slot_indices
    else:
        mla_block_table = reorder_block_table_to_batch_slots(
            mla_block_table, primary_slot_indices,
        )
    safe_cache_seqlens = cache_seqlens.to(
        dtype=torch.int32, device=mla_block_table.device,
    )

    if branch_label == "dense-short-circuit":
        # Batch-level all-short fast path: every row's context fits inside
        # index_topk, so attention is dense over the whole context. Read the
        # resident page-`mla_page_size` cache directly — no transform, no
        # selection table, no copy. This is the eager twin of
        # `Glm5FullDsaAttnSegment._run_all_short_fa3`.
        selected_token_ids = None
        selected_lengths = safe_cache_seqlens
        if verify_indices and wrapper.layer_idx <= 4:
            _log_selected_token_bounds(
                wrapper,
                bsz,
                None,
                selected_lengths,
                mla_block_table,
                mla_blocked_k,
                mla_page_size,
                branch_label,
            )
        with (dt.timed("sparse_attn", li) if dt else nullcontext()):
            attn_out = _run_all_short_fa3(
                attn,
                q_pe,
                absorbed_q,
                mla_blocked_k,
                page_table=mla_block_table,
                cache_batch_idx=primary_selector_slots,
                cache_seqlens=safe_cache_seqlens,
            )
    else:
        selected_token_ids = torch.empty(
            bsz,
            index_topk,
            dtype=torch.int32,
            device=mla_block_table.device,
        )
        selected_lengths = torch.empty(
            bsz, dtype=torch.int32, device=mla_block_table.device,
        )
        transform_selected_positions_out(
            mla_block_table,
            safe_cache_seqlens,
            top_k_indices,
            selected_token_ids,
            selected_lengths,
            page_size=mla_page_size,
            primary_slot_indices=primary_selector_slots,
        )
        if verify_indices and wrapper.layer_idx <= 4:
            _log_selected_token_bounds(
                wrapper,
                bsz,
                selected_token_ids,
                selected_lengths,
                mla_block_table,
                mla_blocked_k,
                mla_page_size,
                branch_label,
            )
        with (dt.timed("sparse_attn", li) if dt else nullcontext()):
            attn_out = _run_selected_fa3(
                attn,
                q_pe,
                absorbed_q,
                mla_blocked_k,
                selected_token_ids,
                selected_lengths,
            )

    return Glm5DsaFlashMlaInputs(
        attn_out=attn_out,
        q_nope=q_nope.squeeze(2).contiguous(),
        q_rope=q_pe.squeeze(2).contiguous(),
        selected_lengths=selected_lengths,
        selected_token_ids=(
            selected_token_ids
            if (verify_indices or return_selected_indices)
            else None
        ),
        row_modes=row_modes,
        primary_k_tensor=k_tensor,
        indexer_k_tensor=indexer_k_tensor,
        branch_label=branch_label,
    )


def _absorb_q_nope(wrapper, q_nope: torch.Tensor) -> torch.Tensor:
    """FP8 WGMMA q absorb: ``[B, H, 1, qk_nope]`` → ``[B, H, kv_lora_rank]``.

    Lifted out of the retired `_build_query_states` pack: FA3 consumes the
    absorbed q as its separate ``qv`` input, so nothing concatenates it with
    q_rope any more.
    """
    if wrapper._fp8_absorb_weights is None:
        raise RuntimeError(
            f"[layer {wrapper.layer_idx}] GLM-5 DSA selector requires WP5 FP8 "
            "q_absorb; PyTorch/BF16 fallback is disabled"
        )

    from batchgen_kernels.attention.dsa.fp8_absorb import fp8_q_absorb

    return fp8_q_absorb(q_nope.squeeze(2), wrapper._fp8_absorb_weights)


def _require_fa3():
    if _fa3_with_kvcache is None:
        raise RuntimeError(
            "GLM-5 DSA eager decode requires flash_attn_interface "
            "(FlashAttention-3): flash_attn_with_kvcache is unavailable"
        )
    return _fa3_with_kvcache


def _run_all_short_fa3(
    attn,
    q_pe: torch.Tensor,
    absorbed_q: torch.Tensor,
    mla_blocked_k: torch.Tensor,
    *,
    page_table: torch.Tensor,
    cache_batch_idx: torch.Tensor | None,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Dense MLA decode straight over the resident page-`page_size` KV."""
    fa3 = _require_fa3()
    return fa3(
        q=q_pe.squeeze(2).unsqueeze(1),
        k_cache=mla_blocked_k[..., attn.kv_lora_rank :],
        v_cache=mla_blocked_k[..., : attn.kv_lora_rank],
        qv=absorbed_q.unsqueeze(1),
        page_table=page_table,
        cache_batch_idx=cache_batch_idx,
        cache_seqlens=cache_seqlens,
        softmax_scale=float(attn.softmax_scale),
        causal=True,
        num_splits=0,
        return_softmax_lse=False,
    )


def _run_selected_fa3(
    attn,
    q_pe: torch.Tensor,
    absorbed_q: torch.Tensor,
    mla_blocked_k: torch.Tensor,
    selected_token_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
) -> torch.Tensor:
    """Sparse MLA decode over the selected PHYSICAL token IDs.

    The KV cache is re-viewed at page size 1 so the per-row selected token
    IDs act directly as the FA3 page table: the selected KV is never
    materialized. ``cache_batch_idx`` is intentionally absent — the token IDs
    are already absolute, so there is no slot indirection.
    """
    fa3 = _require_fa3()
    flat_kv = mla_blocked_k.view(-1, 1, 1, mla_blocked_k.shape[-1])
    return fa3(
        q=q_pe.squeeze(2).unsqueeze(1),
        k_cache=flat_kv[..., attn.kv_lora_rank :],
        v_cache=flat_kv[..., : attn.kv_lora_rank],
        qv=absorbed_q.unsqueeze(1),
        page_table=selected_token_ids,
        cache_seqlens=selected_lengths,
        softmax_scale=float(attn.softmax_scale),
        causal=True,
        num_splits=0,
        return_softmax_lse=False,
    )


def _select_glm5_dsa_indices(
    wrapper,
    hidden_states: torch.Tensor,
    q_a_normed: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    new_token_pos: torch.Tensor,
    gpu_paged_kv_manager_aux,
    aux_slot_indices: torch.Tensor,
    *,
    need_dense_indices: bool = True,
) -> tuple[torch.Tensor | None, str, torch.Tensor]:
    """Select DSA top-k positions for this decode step.

    ``need_dense_indices=False`` lets the ``dense-short-circuit`` branch
    return ``None`` instead of materializing dense token indices that nothing
    consumes: the eager all-short FA3 fast path attends over the real page
    table. The default stays ``True`` so legacy callers and unit tests keep
    the historical return contract.
    """
    indexer = wrapper.module.indexer
    index_topk = indexer.index_topk
    row_modes = (cache_seqlens > index_topk).to(torch.int32)
    short_mask = cache_seqlens <= index_topk
    batch_size = int(short_mask.shape[0])

    # This hint is computed once per decode step in the worker. Falling back to
    # a local reduction keeps unit tests and legacy callers functional.
    short_count = GLM5AttnWrapper._dsa_short_count
    if short_count is None:
        short_count = int(short_mask.sum().item())

    any_short = short_count > 0
    any_long = short_count < batch_size
    device = hidden_states.device

    if not any_long:
        # Historical eager short-circuit: short rows never run indexer scoring.
        top_k_indices = None
        if need_dense_indices:
            top_k_indices = build_clamped_dense_token_indices(
                cache_seqlens,
                index_topk,
                device,
            )
        return top_k_indices, "dense-short-circuit", row_modes

    indexer_blocked_k, _, idx_block_table = (
        gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(wrapper.layer_idx)
    )
    aux_page_size = gpu_paged_kv_manager_aux.config.page_size_tokens

    if not any_short:
        idx_block_table = reorder_block_table_to_batch_slots(
            idx_block_table, aux_slot_indices,
        )
        top_k_indices = indexer.score_and_select_paged(
            q_a_normed.unsqueeze(1),
            hidden_states,
            indexer_blocked_k,
            idx_block_table,
            cache_seqlens,
            gpu_paged_kv_manager_aux,
            aux_page_size,
            positions=new_token_pos,
            max_seqlen=max_seqlen,
        )
        return top_k_indices, "full-indexer", row_modes

    # Mixed batch: preserve the short-row dense path and score only long rows.
    long_mask = ~short_mask
    top_k_indices = torch.empty(
        batch_size,
        index_topk,
        dtype=torch.long,
        device=device,
    )
    top_k_indices[short_mask] = build_clamped_dense_token_indices(
        cache_seqlens[short_mask],
        index_topk,
        device,
    )

    long_cache_seqlens = cache_seqlens[long_mask]
    long_max_seqlen = int(long_cache_seqlens.max().item())
    long_mask_aux = long_mask.to(aux_slot_indices.device)
    idx_block_table_long = reorder_block_table_to_batch_slots(
        idx_block_table,
        aux_slot_indices[long_mask_aux],
    )
    long_top_k = indexer.score_and_select_paged(
        q_a_normed[long_mask].unsqueeze(1),
        hidden_states[long_mask],
        indexer_blocked_k,
        idx_block_table_long,
        long_cache_seqlens,
        gpu_paged_kv_manager_aux,
        aux_page_size,
        positions=new_token_pos[long_mask],
        max_seqlen=long_max_seqlen,
    )
    top_k_indices[long_mask] = long_top_k
    return top_k_indices, "mixed", row_modes

def _log_dsa_bounds(
    wrapper,
    bsz: int,
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    new_token_pos: torch.Tensor,
    primary_slot_indices: torch.Tensor,
    aux_slot_indices: torch.Tensor,
    gpu_paged_kv_manager,
    gpu_paged_kv_manager_aux,
) -> None:
    rk = AttnWrapperBase.get_rank_safe()
    prim_pt = gpu_paged_kv_manager._gpu_page_table_manager.gpu_table
    aux_pt = gpu_paged_kv_manager_aux._gpu_page_table_manager.gpu_table
    prim_rows = 0 if prim_pt is None else int(prim_pt.shape[0])
    aux_rows = 0 if aux_pt is None else int(aux_pt.shape[0])
    prim_cols = 0 if prim_pt is None else int(prim_pt.shape[1])
    aux_cols = 0 if aux_pt is None else int(aux_pt.shape[1])
    prim_pages = int(gpu_paged_kv_manager.config.num_pages)
    aux_pages = int(gpu_paged_kv_manager_aux.config.num_pages)
    prim_psz = int(gpu_paged_kv_manager.config.page_size_tokens)
    aux_psz = int(gpu_paged_kv_manager_aux.config.page_size_tokens)
    logging.warning(
        f"[VERIFY-DSA rank={rk} L{wrapper.layer_idx} bsz={bsz}] "
        f"primary_slot_shape={tuple(primary_slot_indices.shape)} "
        f"max_rows={prim_rows} cols={prim_cols} "
        f"num_pages={prim_pages} page_sz={prim_psz} | "
        f"aux_slot_shape={tuple(aux_slot_indices.shape)} "
        f"max_rows={aux_rows} cols={aux_cols} "
        f"num_pages={aux_pages} page_sz={aux_psz} | "
        f"cache_seq_shape={tuple(cache_seqlens.shape)} "
        f"pos_shape={tuple(new_token_pos.shape)} "
        f"max_seqlen={max_seqlen}"
    )
    expected_prim_pages = (max_seqlen + prim_psz - 1) // prim_psz
    expected_aux_pages = (max_seqlen + aux_psz - 1) // aux_psz
    if expected_prim_pages > prim_cols:
        logging.warning(
            f"[VERIFY-DSA rank={rk} L{wrapper.layer_idx}] "
            f"max_seqlen={max_seqlen} needs {expected_prim_pages} primary pages "
            f"but page_table only has {prim_cols} cols — gather will wrap"
        )
    if expected_aux_pages > aux_cols:
        logging.warning(
            f"[VERIFY-DSA rank={rk} L{wrapper.layer_idx}] "
            f"max_seqlen={max_seqlen} needs {expected_aux_pages} aux pages "
            f"but aux page_table only has {aux_cols} cols — gather will wrap"
        )


def _log_selected_token_bounds(
    wrapper,
    bsz: int,
    selected_token_ids: torch.Tensor | None,
    selected_lengths: torch.Tensor,
    mla_block_table: torch.Tensor,
    mla_blocked_k: torch.Tensor,
    mla_page_size: int,
    branch_label: str,
) -> None:
    """Validate the FA3 page table that replaced the gather.

    Successor of the retired `_log_gather_bounds`. On the transform path it
    checks the PHYSICAL token IDs against the page-size-1 cache extent and
    reports how many slots carry the ``-1`` padding sentinel; on the
    all-short fast path there is no selection table, so it checks the page
    table's extent against the longest context instead. Both run only under
    ``BATCHGEN_GLM5_VERIFY_INDICES=1`` for the first few layers: the
    reductions below synchronize.
    """
    rk = AttnWrapperBase.get_rank_safe()
    bt_shape = tuple(mla_block_table.shape)
    bk_shape = tuple(mla_blocked_k.shape)
    total_tokens = bk_shape[0] * mla_page_size
    max_len = int(selected_lengths.max().item())
    if selected_token_ids is None:
        pages_needed = (max_len + mla_page_size - 1) // mla_page_size
        logging.warning(
            f"[VERIFY-FA3 rank={rk} L{wrapper.layer_idx} bsz={bsz}] "
            f"branch={branch_label} all-short page-{mla_page_size} fast path | "
            f"mla_block_table.shape={bt_shape} mla_blocked_k.shape={bk_shape} "
            f"total_tokens={total_tokens} max_cache_seqlen={max_len} "
            f"pages_needed={pages_needed} page_table_cols={bt_shape[1]}"
        )
        if pages_needed > bt_shape[1]:
            logging.warning(
                f"[VERIFY-FA3 rank={rk} L{wrapper.layer_idx}] "
                f"max_cache_seqlen={max_len} needs {pages_needed} pages but the "
                f"page table only has {bt_shape[1]} cols — FA3 will read past it"
            )
        return

    padded = int((selected_token_ids < 0).sum().item())
    max_id = int(selected_token_ids.max().item())
    logging.warning(
        f"[VERIFY-FA3 rank={rk} L{wrapper.layer_idx} bsz={bsz}] "
        f"branch={branch_label} "
        f"selected_token_ids.shape={tuple(selected_token_ids.shape)} | "
        f"mla_block_table.shape={bt_shape} mla_blocked_k.shape={bk_shape} "
        f"page_size={mla_page_size} total_tokens={total_tokens} "
        f"max_physical_id={max_id} neg_one_slots={padded} "
        f"max_selected_len={max_len}"
    )
    if max_id >= total_tokens:
        logging.warning(
            f"[VERIFY-FA3 rank={rk} L{wrapper.layer_idx}] "
            f"max_physical_id={max_id} >= total_tokens={total_tokens} — the "
            "transform produced an out-of-range FA3 page-table entry"
        )
