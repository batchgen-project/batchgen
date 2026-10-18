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

Both primary MLA KV formats are served here, and -- exactly as in the captured
graph (`batchgen/models/glm/glm5/cuda_graph_segments.py:1005-1038`) -- the
format is read off the ALLOCATED cache, never derived from a model name:

* bf16 AoS (GLM-5, 5.1, 5.3), ``kv_lora_rank + qk_rope_head_dim`` elements per
  token: FA3, split ``q``/``qv`` pair, dense over the real page table for an
  all-short batch and over the page-size-1 re-view for a selecting one.
* FlashMLA packed FP8 (GLM-5.2), 656 BYTES per token: sparse FP8 FlashMLA for
  BOTH batch shapes, because the pin has no dense FP8 MLA decode kernel. The
  all-short batch is expressed as "select every token of the row", which the
  transform kernel already produces for a row inside the selection budget.

The writer follows the same discriminator: what is stored into the cache, and
what is handed to the host append, is the cache's own RECORD, produced exactly
once by ``prepare_primary_mla_kv``.
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

from batchgen.attention.dsa.aux_indexer_k_writer import write_aux_indexer_k
from batchgen.attention.mla.fa3_backend import act_quant
from batchgen.attention.mla.fused_rmsnorm_rope import (
    fused_rmsnorm_rope_with_q_native as _fused_rmsnorm_rope,
)
from batchgen.attention.mla.packed_fp8_sparse_decode import (
    SparseFp8MlaMetadata,
    guard_sparse_indices_against_empty_rows,
    prepare_sparse_fp8_mla_metadata,
    run_sparse_fp8_mla_decode,
)
from batchgen.attention.mla.primary_kv_record import (
    dequantize_mla_kv_packed,
    prepare_primary_mla_kv,
    primary_cache_is_packed_fp8,
    primary_record_dim,
    primary_record_dtype,
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


def _build_primary_kv_record(
    *,
    attn,
    latent_kv: torch.Tensor,
    primary_blocked_k: torch.Tensor,
    bsz: int,
) -> torch.Tensor:
    """Produce the primary cache's RECORD for this decode step's tokens.

    ``latent_kv`` is the bf16 compressed KV straight out of the fused
    RMSNorm+RoPE kernel, already viewed as ``[B, 1, 1, source_dim]`` and
    already resident on the cache's device. The return value is what BOTH the
    GPU paged store and the host append receive, so the record is produced
    exactly once and every later hop is a byte copy -- the invariant
    `batchgen/attention/mla/primary_kv_record.py:10-22` exists to hold.

    The format is discriminated off ``primary_blocked_k``, the cache the KV
    manager allocated: a bf16 AoS cache's record IS the latent (returned
    unchanged, so that path keeps its single store and allocates nothing), a
    packed-FP8 cache's record is 656 quantized bytes. A mis-sized cache is
    named by ``primary_cache_is_packed_fp8`` rather than written through.
    """

    source_dim = int(attn.kv_lora_rank + attn.qk_rope_head_dim)
    if not primary_cache_is_packed_fp8(primary_blocked_k, source_dim):
        return latent_kv

    # One destination, handed to the store and to the host append. Allocated
    # from the CACHE's dtype/record dim rather than from constants, so this
    # cannot disagree with what the paged store's dtype-equality guard accepts
    # (`primary_kv_record.py:53-63`).
    record = torch.empty(
        (bsz, 1, 1, primary_record_dim(primary_blocked_k)),
        dtype=primary_record_dtype(primary_blocked_k),
        device=primary_blocked_k.device,
    )
    prepare_primary_mla_kv(
        primary_blocked_k=primary_blocked_k,
        compressed_kv=latent_kv.view(bsz, -1),
        out=record,
        source_dim=source_dim,
    )
    return record


# Sparse FP8 FlashMLA statics, memoized per (device, batch, topk, q heads).
#
# In sparse mode FlashMLA's metadata kernel substitutes ``topk`` for every
# row's length and never reads ``cache_seqlens``
# (``csrc/smxx/get_mla_metadata.cu:30``, quoted in
# `batchgen/attention/mla/packed_fp8_sparse_decode.py:13-21`), so its output
# depends only on these keys plus arch-fixed quantities: it is step-invariant.
# Eager decode has no capture to hang it off, so without a cache every layer of
# every step would pay a metadata launch plus two allocations to rebuild a
# tensor pair that cannot have changed -- pure overhead.
#
# Module level rather than per wrapper, because every layer of a model shares
# the same head count: one entry serves all layers instead of each building its
# own identical copy. Sharing is safe because both tensors are READ-ONLY to the
# consumers -- the kernel only reads the metadata, and
# ``guard_sparse_indices_against_empty_rows`` writes into ``indices`` while its
# ``zero_column`` operand is only an operand
# (`packed_fp8_sparse_decode.py:200`). The device is part of the key so a
# multi-GPU process cannot hand one rank another rank's buffers.
_SPARSE_FP8_DECODE_STATICS: dict[
    tuple[str, int, int, int, int], tuple[SparseFp8MlaMetadata, torch.Tensor]
] = {}


def _sparse_fp8_decode_statics(
    *,
    batch_size: int,
    topk: int,
    num_heads_q: int,
    device: torch.device,
) -> tuple[SparseFp8MlaMetadata, torch.Tensor]:
    """This step's sparse metadata and empty-row guard column, built once."""

    key = (
        device.type,
        -1 if device.index is None else int(device.index),
        int(batch_size),
        int(topk),
        int(num_heads_q),
    )
    statics = _SPARSE_FP8_DECODE_STATICS.get(key)
    if statics is None:
        statics = (
            prepare_sparse_fp8_mla_metadata(
                batch_size=int(batch_size),
                topk=int(topk),
                num_heads_q=int(num_heads_q),
                num_heads_k=1,
                num_q_tokens_per_q_seq=1,
                device=device,
            ),
            # Column 0 of a padding row is clamped up to this, so the row
            # attends to physical token 0 and returns a finite, meaningless
            # value instead of NaN (packed_fp8_sparse_decode note 3). The eager
            # path has no row mask, so a NaN row would propagate -- which is
            # why the guard runs here too.
            torch.zeros(int(batch_size), dtype=torch.int32, device=device),
        )
        _SPARSE_FP8_DECODE_STATICS[key] = statics
    return statics


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
    # What the host append must receive for the primary cache: the cache's own
    # RECORD, i.e. the bf16 latent for an AoS cache and the 656-byte packed
    # record for a GLM-5.2 one -- the very bytes the GPU store was given, so
    # the host page and the cache row are byte-identical. With ``write_kv``
    # False (the graph/eager compare prologue) no cache is written, nothing
    # quantizes, and the bf16 source is reported instead; that is the same
    # deliberate asymmetry the aux tensor documents below.
    primary_k_tensor: torch.Tensor
    # The aux tensors the host append must receive: the bf16 keys for an AoS
    # aux cache, the quantized e4m3 keys plus their fp32 per-token scales for a
    # page-split one (the host region requires the scales and rejects them in
    # the other format).
    indexer_k_tensor: torch.Tensor
    indexer_k_scales: torch.Tensor | None = None


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
            # The ALLOCATED cache is the format discriminator, so it is fetched
            # before the write. `get_layer_kv_with_page_table` returns exactly
            # the tensor `update_layer_decode_new_token` stores into -- both
            # resolve the physical layer and index the same `_k_cache` list
            # (`batchgen/kv_cache/gpu_paged_kv_manager.py:1756` and `:1845`) --
            # and only its dtype and record dim are read here, which are
            # allocation-time invariants. Reading them before the write is
            # therefore not an observation of stale cache CONTENT.
            primary_blocked_k, _, _ = (
                gpu_paged_kv_manager.get_layer_kv_with_page_table(li)
            )
            # On a packed cache this is the 656-byte record; on a bf16 AoS
            # cache it is the latent itself, unchanged. The same tensor is
            # stored and returned for the host append, so nothing quantizes
            # twice.
            k_tensor = _build_primary_kv_record(
                attn=attn,
                latent_kv=k_tensor,
                primary_blocked_k=primary_blocked_k,
                bsz=bsz,
            )
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
        # write_kv=False is the graph/eager compare prologue: no cache is
        # written, so nothing quantizes, and a page-split aux cache leaves the
        # bf16 source here (that debug comparison then faces the graph's e4m3
        # output; the production path above always quantizes).
        indexer_k_host = indexer_k_tensor
        indexer_k_scales = None
        if write_kv:
            aux_blocked_k, _, aux_page_table = (
                gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(li)
            )
            indexer_k_host, indexer_k_scales = write_aux_indexer_k(
                aux_blocked_k=aux_blocked_k,
                indexer_k=indexer_k_tensor,
                page_table=aux_page_table,
                slot_indices=aux_slot_indices.to(
                    device=aux_page_table.device, dtype=torch.int32,
                ),
                token_indices=seq_lengths_i32_aux,
                page_size_tokens=gpu_paged_kv_manager_aux.config.page_size_tokens,
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
        indexer_k_tensor=indexer_k_host,
        indexer_k_scales=indexer_k_scales,
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

    # Fetched BEFORE the write, because the ALLOCATED cache is what decides
    # what a record is and which decode reader may touch it. This used to sit
    # just above the attention branch; nothing between here and there mutates
    # the primary manager (the indexer and selection stages talk only to the
    # aux manager), so the handles are the same tensors either way, and the
    # writer below now needs the discriminator.
    mla_blocked_k, _, mla_block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(li)
    mla_page_size = gpu_paged_kv_manager.config.page_size_tokens
    primary_is_packed = primary_cache_is_packed_fp8(
        mla_blocked_k, int(attn.kv_lora_rank + attn.qk_rope_head_dim)
    )

    with (dt.timed("kv_write", li) if dt else nullcontext()):
        latent_kv = offload_kv.view(bsz, 1, 1, offload_kv.size(-1))
        if latent_kv.device != manager_device:
            latent_kv = latent_kv.to(manager_device)
        # The record, not the latent: on a packed cache the store and the host
        # append must both receive the 656 quantized bytes, and they must
        # receive the SAME tensor so the row is quantized once. On a bf16 AoS
        # cache this returns the latent unchanged.
        k_tensor = _build_primary_kv_record(
            attn=attn,
            latent_kv=latent_kv,
            primary_blocked_k=mla_blocked_k,
            bsz=bsz,
        )
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
    # What the host append received: the bf16 keys for an AoS aux cache, the
    # quantized e4m3 keys for a page-split one. Reported instead of the bf16
    # source so the graph/eager compare checks the bytes both paths stored.
    indexer_k_host = None
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
            # The aux cache's own format decides how the keys are written: a
            # GLM-5.2 page-split FP8 cache is quantized and page-split in one
            # fused launch that also returns the (k_fp8, scales) pair the host
            # region needs; a bf16 AoS cache keeps the plain paged update and
            # gets no scales.
            aux_blocked_k, _, aux_page_table = (
                gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(li)
            )
            indexer_k_host, host_scales = write_aux_indexer_k(
                aux_blocked_k=aux_blocked_k,
                indexer_k=indexer_k_tensor,
                page_table=aux_page_table,
                slot_indices=aux_slot_indices.to(
                    device=aux_page_table.device, dtype=torch.int32,
                ),
                token_indices=seq_lengths_i32_aux,
                page_size_tokens=gpu_paged_kv_manager_aux.config.page_size_tokens,
            )
            if AttnWrapperBase.kv_append_callback_aux is not None:
                AttnWrapperBase.kv_append_callback_aux(
                    li, indexer_k_host, None, host_scales,
                )

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
        # One contiguous copy serves both the absorb kernel (which would
        # otherwise make its own) and the debug dataclass field below.
        q_nope_flat = q_nope.squeeze(2).contiguous()
        absorbed_q = _absorb_q_nope(wrapper, q_nope_flat)

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

    if branch_label == "dense-short-circuit" and not primary_is_packed:
        # Batch-level all-short fast path: every row's context fits inside
        # index_topk, so attention is dense over the whole context. Read the
        # resident page-`mla_page_size` cache directly — no transform, no
        # selection table, no copy. This is the eager twin of
        # `Glm5FullDsaAttnSegment._run_all_short_fa3`.
        #
        # bf16 AoS only. A packed-FP8 cache has no dense FP8 MLA decode kernel
        # in the pin at all (SM90 refuses dense FP8, SM100 refuses anything
        # that is not FP8+sparse -- `csrc/pybind.cpp:331-357`), so it takes the
        # selection path below and expresses "attend to the whole short
        # context" as "select every token of it". Same cut as the captured
        # graph (`cuda_graph_segments.py:1874-1906`).
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
                primary_is_packed=primary_is_packed,
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
        with (dt.timed("sparse_select", li) if dt else nullcontext()):
            if top_k_indices is None:
                # Reachable only on a PACKED all-short step: that branch skips
                # the dense index build when no top-k carry needs it
                # (`_select_glm5_dsa_indices`), but the packed reader runs the
                # transform on both batch shapes and the transform derives
                # (batch, index_topk) from this tensor's own shape
                # (`selected_page_table.py:107`). These are also exactly the
                # values the kernel re-derives for a row inside the budget --
                # it substitutes identity offsets whenever
                # `seqlen <= index_topk` (`selected_page_table.py:52`) -- so
                # this is the honest input, not a placeholder.
                top_k_indices = build_clamped_dense_token_indices(
                    safe_cache_seqlens, index_topk, mla_block_table.device,
                )
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
                primary_is_packed=primary_is_packed,
            )
        # One timer, two backends: the selection above is format-independent
        # (physical token IDs are physical token IDs), only the reader differs.
        with (dt.timed("sparse_attn", li) if dt else nullcontext()):
            if primary_is_packed:
                attn_out = _run_sparse_fp8(
                    attn,
                    q_pe,
                    absorbed_q,
                    mla_blocked_k,
                    selected_token_ids,
                    selected_lengths,
                    block_table=mla_block_table[:bsz],
                )
            else:
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
        q_nope=q_nope_flat,
        q_rope=q_pe.squeeze(2).contiguous(),
        selected_lengths=selected_lengths,
        selected_token_ids=(
            selected_token_ids
            if (verify_indices or return_selected_indices)
            else None
        ),
        row_modes=row_modes,
        primary_k_tensor=k_tensor,
        indexer_k_tensor=indexer_k_host,
        branch_label=branch_label,
    )


def _absorb_q_nope(wrapper, q_nope_flat: torch.Tensor) -> torch.Tensor:
    """FP8 WGMMA q absorb: contiguous ``[B, H, qk_nope]`` → ``[B, H, kv_lora_rank]``.

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

    return fp8_q_absorb(q_nope_flat, wrapper._fp8_absorb_weights)


def _run_sparse_fp8(
    attn,
    q_pe: torch.Tensor,
    absorbed_q: torch.Tensor,
    mla_blocked_k: torch.Tensor,
    selected_token_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    *,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Sparse FP8 FlashMLA decode over the packed cache's physical token IDs.

    The ONE reader of a packed-FP8 primary cache in the eager path, used for
    both batch shapes. Differences from the FA3 twin, all forced by the kernel:

    * the cache is passed exactly as allocated, NOT re-viewed at page size 1 --
      each index already encodes ``physical_page * 64 + offset``, and the pin
      hard-requires a 64-token page with the whole 656-byte record contiguous
      (`csrc/pybind.cpp:236,249-256`);
    * ONE 576-wide query replaces FA3's split ``q``/``qv`` pair. The
      concatenation order IS the record's own order -- 512 absorbed-NoPE dims
      then 64 RoPE dims -- so `absorbed_q` must come first;
    * ``block_table`` and ``cache_seqlens`` are shape-checked placeholders the
      sparse kernel never reads (packed_fp8_sparse_decode note 2). Real
      tensors are passed anyway, so a pin bump that starts reading them cannot
      read garbage. ``block_table`` is sliced to the batch because the pin
      checks ``CHECK_SHAPE(block_table, batch_size, max_pages)``
      (`csrc/pybind.cpp:258`) and the caller's table has one row per page-table
      SLOT whenever a slot override is active -- so, exactly as in the captured
      twin (`cuda_graph_segments.py:1209-1214`), those rows are slot-major
      rather than batch-major, and a pin bump that starts reading them must
      re-derive this argument instead of trusting the slice.

    No row masking is added: the eager path relies on the -1 slot sentinel, and
    the transform already fills a row whose slot is -1 with all -1. The one
    thing that cannot be left to the sentinel is the softmax denominator, hence
    the empty-row guard below.
    """

    # Every key is read off the tensors actually passed to the kernel, never
    # from config: the metadata fixes how many index blocks are swept and how
    # the scheduler splits the heads (`packed_fp8_sparse_decode.py:146-153`),
    # so a key that disagreed with the query or the indices would attend to the
    # wrong window instead of failing.
    metadata, zero_column = _sparse_fp8_decode_statics(
        batch_size=int(selected_token_ids.shape[0]),
        topk=int(selected_token_ids.shape[1]),
        num_heads_q=int(absorbed_q.shape[1]),
        device=selected_token_ids.device,
    )
    # A row of only -1 makes the kernel divide 0 by 0 and return NaN
    # (packed_fp8_sparse_decode note 3). Unlike the captured graph there is no
    # row mask downstream to swallow it, so this is the only thing standing
    # between a slot-less row and a NaN that poisons the step.
    guard_sparse_indices_against_empty_rows(selected_token_ids, zero_column)
    # [B, H, 512] ++ [B, H, 64] -> [B, H, 576], then [B, 1, H, 576]. A single
    # concat, unlike the graph's two copies into a static buffer: eager decode
    # has no capture, so the kernel-output strides that forced the graph to
    # pre-allocate (`cuda_graph_segments.py:403-409`) are not a constraint
    # here.
    query = torch.cat((absorbed_q, q_pe.squeeze(2)), dim=-1)
    return run_sparse_fp8_mla_decode(
        query=query.unsqueeze(1),
        packed_blocked_k=mla_blocked_k,
        indices=selected_token_ids.unsqueeze(1),
        block_table=block_table,
        cache_seqlens=selected_lengths,
        metadata=metadata,
        softmax_scale=float(attn.softmax_scale),
        head_dim_v=int(attn.kv_lora_rank),
    )


def _require_fa3():
    # Reached only on a bf16 AoS cache: a packed-FP8 cache is read by
    # `_run_sparse_fp8` and never enters either FA3 runner, so FA3's absence
    # must not be reported as a GLM-5.2 decode failure. That is the eager twin
    # of the per-format requirement at `cuda_graph_segments.py:1033-1042`.
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
    *,
    primary_is_packed: bool,
) -> None:
    """Validate the selection table that replaced the gather.

    Successor of the retired `_log_gather_bounds`. On the transform path it
    checks the PHYSICAL token IDs against the cache's token extent and reports
    how many slots carry the ``-1`` padding sentinel; on the all-short fast
    path there is no selection table, so it checks the page table's extent
    against the longest context instead. Both run only under
    ``BATCHGEN_GLM5_VERIFY_INDICES=1`` for the first few layers: the reductions
    below synchronize.

    Format-independent by construction. A physical token ID means
    ``physical_page * page_size + offset`` for both the FA3 page-size-1 re-view
    and the sparse FP8 kernel, and the extent it is checked against is
    ``pages * page_size`` -- derived from the cache's shape, so it is right
    whether a record is ``source_dim`` bf16 elements or 656 bytes. The record
    width is the one number that differs between the formats, and nothing here
    divides by it; it is reported so a log line says which format it describes.
    On a packed cache the probe below additionally compares VALUES, which needs
    the debug-only dequantizer.
    """
    rk = AttnWrapperBase.get_rank_safe()
    bt_shape = tuple(mla_block_table.shape)
    bk_shape = tuple(mla_blocked_k.shape)
    total_tokens = bk_shape[0] * mla_page_size
    record = (
        f"{bk_shape[-1]}B packed-fp8" if primary_is_packed
        else f"{bk_shape[-1]}x{mla_blocked_k.dtype} bf16-aos"
    )
    max_len = int(selected_lengths.max().item())
    if selected_token_ids is None:
        pages_needed = (max_len + mla_page_size - 1) // mla_page_size
        logging.warning(
            f"[VERIFY-FA3 rank={rk} L{wrapper.layer_idx} bsz={bsz}] "
            f"branch={branch_label} all-short page-{mla_page_size} fast path | "
            f"mla_block_table.shape={bt_shape} mla_blocked_k.shape={bk_shape} "
            f"record={record} "
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
        f"record={record} "
        f"page_size={mla_page_size} total_tokens={total_tokens} "
        f"max_physical_id={max_id} neg_one_slots={padded} "
        f"max_selected_len={max_len}"
    )
    if max_id >= total_tokens:
        logging.warning(
            f"[VERIFY-FA3 rank={rk} L{wrapper.layer_idx}] "
            f"max_physical_id={max_id} >= total_tokens={total_tokens} — the "
            "transform produced an out-of-range selection entry"
        )
    if not primary_is_packed:
        return

    # Packed cache only: an in-range index set can still be the WRONG index
    # set, and the bounds above cannot see that. Dequantizing the records row 0
    # actually selected separates the two failures that matter:
    #
    #   all_zero_records > 0 -> the indices point at never-written slots (a
    #       freshly allocated page is 0x00 bytes with 0.0f scales, which
    #       dequantizes to exactly 0 -- `mla_kv_packed.py:199-210`), i.e. a
    #       page-table or slot bug the extent check passes;
    #   nonfinite > 0 -> the stored bytes themselves are bad, i.e. the write
    #       path, not the selection.
    #
    # DEBUG ONLY, and only under the env gate: `dequantize_mla_kv_packed` is
    # not called by any decode or prefill path -- the kernel dequantizes
    # internally -- and this probe synchronizes, like the reductions above.
    flat_records = mla_blocked_k.reshape(-1, bk_shape[-1])
    row0 = selected_token_ids[0]
    valid_row0 = row0[row0 >= 0]
    if valid_row0.numel() == 0:
        logging.warning(
            f"[VERIFY-FA3 rank={rk} L{wrapper.layer_idx}] packed probe skipped: "
            "row 0 selected no valid token (the empty-row guard will clamp it "
            "to physical token 0 to keep the kernel's denominator non-zero)"
        )
        return
    probe_ids = valid_row0[: min(8, int(valid_row0.numel()))].to(torch.long)
    probed = dequantize_mla_kv_packed(flat_records.index_select(0, probe_ids))
    logging.warning(
        f"[VERIFY-FA3 rank={rk} L{wrapper.layer_idx}] packed probe: "
        f"records={int(probe_ids.numel())} "
        f"first_ids={probe_ids[:4].tolist()} "
        f"absmax={float(probed.abs().amax().item()):.4f} "
        f"all_zero_records={int((probed == 0).all(dim=-1).sum().item())} "
        f"nonfinite={int((~torch.isfinite(probed)).sum().item())}"
    )
