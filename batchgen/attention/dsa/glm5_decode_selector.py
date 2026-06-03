"""GLM-5 DSA pre-FlashMLA decode input builder.

This module owns the full eager selector segment for GLM-5 DSA decode:
decode hidden states and metadata enter here, the GLM-5 DSA kernels are
invoked, and the return value is ready for the raw FlashMLA call.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from batchgen.attention.dsa.sparse_decode_mla import (
    PreparedSparseFlashMlaDecode,
    prepare_sparse_flash_mla_decode_inputs,
)
from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16
from batchgen.attention.mla.fa3_backend import act_quant
from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization
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


def _write_indexer_k_fp8_paged(
    gpu_paged_kv_manager_aux,
    layer_idx: int,
    k_normed_bf16: torch.Tensor,
    new_token_pos: torch.Tensor,
    aux_slot_indices: torch.Tensor,
) -> None:
    """Quantize indexer K to FP8 and scatter it into the page-split aux cache.

    Bypasses ``update_layer_decode_new_token`` (which writes an INTERLEAVED bf16
    layout) and instead writes the page-split FP8 layout deep_gemm expects:
    per page ``[page_size*128 e4m3 K | page_size*4 fp32 scale]``.

    Physical-slot mapping reproduces ``update_layer_decode_new_token`` /
    ``run_paged_kv_token_update_fused``: for batch row i,
        slot = aux_slot_indices[i]              (row of the page table)
        pos  = new_token_pos[i]                 (position of the new token)
        physical_page = page_table[slot, pos // page_size]
        offset        = pos %  page_size
        loc           = physical_page * page_size + offset
    ``loc`` is the absolute PHYSICAL token slot consumed by ``split_write_fp8``.

    k_normed_bf16 : [B, 128] bf16, post k_norm + RoPE + Hadamard.
    """
    from batchgen.attention.dsa.indexer_fp8 import fused_indexer_k_write_fp8

    physical_layer = gpu_paged_kv_manager_aux.resolve_physical_layer(layer_idx)
    k_cache = gpu_paged_kv_manager_aux._k_cache  # [L, num_pages, page_size, 1, 132] uint8
    page_size = int(gpu_paged_kv_manager_aux.config.page_size_tokens)
    head_dim = k_normed_bf16.shape[-1]
    page_table = gpu_paged_kv_manager_aux._gpu_page_table_manager.gpu_table
    if page_table is None:
        raise RuntimeError(
            "GLM-5 DSA FP8 aux write: GPU page table is not initialized; "
            "call allocate_pages_for_sequences before scoring"
        )

    num_pages = k_cache.shape[1]
    page_bytes = page_size * (head_dim + 4)  # head_dim e4m3 + 4 fp32 scale bytes
    buf_u8 = k_cache[physical_layer].view(num_pages, page_bytes)

    slots = aux_slot_indices.to(device=page_table.device, dtype=torch.int64)
    pos = new_token_pos.to(device=page_table.device, dtype=torch.int64)
    page_col = pos // page_size
    offset = pos % page_size
    physical_page = page_table[slots, page_col].to(torch.int64)
    loc = (physical_page * page_size + offset).to(torch.int32)

    fused_indexer_k_write_fp8(buf_u8, loc, k_normed_bf16.to(buf_u8.device), page_size=page_size)


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
    """Outputs of the GLM-5 DSA selector segment before FlashMLA invocation."""

    flashmla: PreparedSparseFlashMlaDecode
    query_states: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    selected_mla_kv: torch.Tensor
    selected_lengths: torch.Tensor
    selected_indices: torch.Tensor | None
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
        # Post k_norm + RoPE + Hadamard indexer K. Keep [B,128] for FP8 quant and
        # [B,1,1,128] for host callbacks / downstream consumers.
        indexer_k_bf16 = indexer._fused_rope_hadamard_or_fallback(
            k_normed.unsqueeze(1), new_token_pos, max_seqlen=max_seqlen,
        ).squeeze(1)  # [B, 128]
        indexer_k_tensor = indexer_k_bf16.view(bsz, 1, 1, indexer_k_bf16.shape[-1])
        if write_kv:
            # FP8 page-split write (bypass interleaved update_layer_decode_new_token).
            _write_indexer_k_fp8_paged(
                gpu_paged_kv_manager_aux,
                li,
                indexer_k_bf16,
                new_token_pos,
                aux_slot_indices,
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
    """Build GLM-5 BF16 DSA inputs up to the FlashMLA invocation boundary."""

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
            # Post k_norm + RoPE + Hadamard indexer K, [B, 128] bf16.
            indexer_k_bf16 = indexer._fused_rope_hadamard_or_fallback(
                k_normed.unsqueeze(1), new_token_pos, max_seqlen=max_seqlen,
            ).squeeze(1)
        else:
            raise RuntimeError(
                f"[layer {wrapper.layer_idx}] GLM-5 DSA selector requires WP2 "
                "fused indexer KV projection; PyTorch fallback is disabled"
            )
        # indexer_k_tensor kept as [B,1,1,128] bf16 for host callbacks below; the
        # GPU aux cache is now FP8 page-split, written via _write_indexer_k_fp8_paged.
        indexer_k_tensor = indexer_k_bf16.view(bsz, 1, 1, indexer_k_bf16.shape[-1])
        _write_indexer_k_fp8_paged(
            gpu_paged_kv_manager_aux,
            li,
            indexer_k_bf16,
            new_token_pos,
            aux_slot_indices,
        )
        if AttnWrapperBase.kv_append_callback_aux is not None:
            AttnWrapperBase.kv_append_callback_aux(li, indexer_k_tensor, None)

    with (dt.timed("indexer_score", li) if dt else nullcontext()):
        top_k_indices, branch_label, row_modes = _select_glm5_dsa_indices(
            wrapper,
            hidden_states,
            q_a_normed,
            cache_seqlens,
            max_seqlen,
            new_token_pos,
            gpu_paged_kv_manager_aux,
            aux_slot_indices,
        )

    with (dt.timed("sparse_gather", li) if dt else nullcontext()):
        mla_blocked_k, _, mla_block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(li)
        primary_selector_slots = None
        if slot_override_active:
            primary_selector_slots = primary_slot_indices
        else:
            mla_block_table = reorder_block_table_to_batch_slots(
                mla_block_table, primary_slot_indices,
            )
        mla_page_size = gpu_paged_kv_manager.config.page_size_tokens
        if verify_indices and wrapper.layer_idx <= 4:
            _log_gather_bounds(
                wrapper,
                bsz,
                top_k_indices,
                mla_block_table,
                mla_blocked_k,
                mla_page_size,
                branch_label,
            )
        selected_mla_kv, selected_lengths, selected_indices, row_modes = (
            select_mla_kv_for_flashmla_bf16(
                mla_blocked_k,
                mla_block_table,
                cache_seqlens,
                top_k_indices,
                index_topk=wrapper.module.indexer.index_topk,
                page_size=mla_page_size,
                return_indices=verify_indices or return_selected_indices,
                primary_slot_indices=primary_selector_slots,
            )
        )

    with (dt.timed("q_absorb", li) if dt else nullcontext()):
        query_states = _build_query_states(wrapper, q_nope, q_pe, selected_mla_kv)

    flashmla = prepare_sparse_flash_mla_decode_inputs(
        query_states,
        selected_mla_kv,
        selected_lengths,
        attn.num_heads,
        attn.softmax_scale,
        head_dim_v=attn.kv_lora_rank,
        page_size=mla_page_size,
    )

    return Glm5DsaFlashMlaInputs(
        flashmla=flashmla,
        query_states=query_states,
        q_nope=q_nope.squeeze(2).contiguous(),
        q_rope=q_pe.squeeze(2).contiguous(),
        selected_mla_kv=selected_mla_kv,
        selected_lengths=selected_lengths,
        selected_indices=selected_indices,
        row_modes=row_modes,
        primary_k_tensor=k_tensor,
        indexer_k_tensor=indexer_k_tensor,
        branch_label=branch_label,
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
) -> tuple[torch.Tensor, str, torch.Tensor]:
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

def _build_query_states(
    wrapper,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    selected_mla_kv: torch.Tensor,
) -> torch.Tensor:
    attn = wrapper.module
    weight_scale = wrapper.weight_dequant_scale
    bsz = q_nope.shape[0]

    if wrapper._cached_q_absorb is not None:
        q_absorb = wrapper._cached_q_absorb
        out_absorb = wrapper._cached_out_absorb
    else:
        kv_b_proj = deepseek_v3_dequantization(
            attn.kv_b_proj.weight.data,
            weight_scale["kv_b_proj.weight_scale_inv"],
        ).view(attn.num_heads, -1, attn.kv_lora_rank)
        q_absorb = kv_b_proj[:, :attn.qk_nope_head_dim, :].contiguous()
        out_absorb = kv_b_proj[:, attn.qk_nope_head_dim:, :].contiguous()
        if wrapper.w_kc is None:
            wrapper.w_kc = q_absorb.transpose(1, 2).contiguous().transpose(1, 2)
            wrapper.w_vc = out_absorb.contiguous().transpose(1, 2)

    qk_head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    query_states = torch.empty(
        bsz,
        attn.num_heads,
        1,
        qk_head_dim,
        dtype=selected_mla_kv.dtype,
        device=selected_mla_kv.device,
    )
    q_nope_squeezed = q_nope.squeeze(2)

    if wrapper._fp8_absorb_weights is not None:
        from batchgen_kernels.attention.dsa.fp8_absorb import fp8_q_absorb

        absorbed_q = fp8_q_absorb(q_nope_squeezed, wrapper._fp8_absorb_weights)
        query_states[:, :, :, :attn.kv_lora_rank] = absorbed_q.view(
            bsz, attn.num_heads, 1, attn.kv_lora_rank,
        )
    else:
        raise RuntimeError(
            f"[layer {wrapper.layer_idx}] GLM-5 DSA selector requires WP5 FP8 "
            "q_absorb; PyTorch/BF16 fallback is disabled"
        )

    query_states[:, :, :, attn.kv_lora_rank:] = q_pe
    return query_states.view(bsz, 1, attn.num_heads, qk_head_dim)


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


def _log_gather_bounds(
    wrapper,
    bsz: int,
    top_k_indices: torch.Tensor,
    mla_block_table: torch.Tensor,
    mla_blocked_k: torch.Tensor,
    mla_page_size: int,
    branch_label: str,
) -> None:
    rk = AttnWrapperBase.get_rank_safe()
    tk_shape = tuple(top_k_indices.shape)
    bt_shape = tuple(mla_block_table.shape)
    bk_shape = tuple(mla_blocked_k.shape)
    num_pages_loaded = bk_shape[0]
    max_flat_idx = num_pages_loaded * mla_page_size
    expected_max_valid_tok = bt_shape[1] * mla_page_size
    logging.warning(
        f"[VERIFY-GATHER rank={rk} L{wrapper.layer_idx} bsz={bsz}] "
        f"top_k_shape={tk_shape} | "
        f"mla_block_table.shape={bt_shape} mla_blocked_k.shape={bk_shape} "
        f"page_size={mla_page_size} max_flat_idx_if_clean={max_flat_idx} "
        f"max_tok_pos_in_bt_cols={expected_max_valid_tok} | "
        f"which branch: {branch_label}"
    )
