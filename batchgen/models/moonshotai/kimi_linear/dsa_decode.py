# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 DSA decode: one sparse full-attention layer step.

Two entry points share every statement below: ``dsa_decode_forward`` is the
eager step (the wrapper's), ``dsa_decode_graph_safe`` the one a captured
decode span runs (``cuda_graph_segments``); its docstring lists the only
differences a capture forces.

Route B: the index keys live in DeepGEMM's page-split FP8 layout and are scored
by the ReLU-gated ``fp8_paged_mqa_logits``; attention over the selected keys
runs through ``batchgen.attention.mla.sparse_dispatch`` (FlashAttention-3
against a page-size-1 view on Hopper, flashinfer trtllm-gen sparse MLA on
Blackwell) over the resident latent cache, so the selected KV is never
gathered. These are the production
page-split DSA kernels; this module drives them for Kimi's NoPE gated MLA.

Per step, in order:

1. **Index key of the new token** — ``fused_indexer_k_norm_rope_write_fp8``
   (LayerNorm + identity RoPE + no Hadamard + e4m3 quantize + page write, one
   launch), BEFORE any scoring, so the new token can select itself. Always
   written, even when nothing is scored: the aux cache must stay complete for
   later steps.
2. **Mode.** A query that sees at most ``index_topk`` keys has nothing to
   select (the reference oracle proves sparse == dense there). When EVERY row
   of the batch is such a row, the layer runs the unchanged dense K3 decode
   (``serving_modules.mla_decoding_nope_with_pagekv``) — bit-identical to
   Kimi-K3, not merely equivalent.
3. **Selection** for rows with more than ``index_topk`` keys: Q =
   ``wq_b(q_a)`` (no rotation on this route), head gates =
   ``weights_proj(h) * H^-0.5 * D^-0.5``, ``paged_mqa_score_and_topk``. Since
   ``relu(c x) = c relu(x)`` for ``c > 0``, the kernel's
   ``sum_h gate_h relu(q_h . k)`` orders keys exactly as the reference
   ``sum_h (w_h H^-0.5) relu(D^-0.5 q_h . k)`` does, up to FP8 rounding.
4. **Sparse attention**: ``transform_selected_positions_out`` maps each row's
   logical selection (all keys for a short row, its top-k for a long one) to
   physical token ids, the sparse seam's token rows, with the absorbed
   query: ``softmax(scale (q_pe . k_pe + q_abs . c)) c`` — the same MLA
   attention FlashMLA computes over the 576-wide latent.
5. Out-absorb, sigmoid output gate, ``o_proj``, TP reduce — the dense path's
   own statements.

Every kernel is reached through a module-level seam resolved on first use, so
the CPU tests run this exact orchestration against reference kernels.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from batchgen.attention.mla.sparse_dispatch import sparse_mla_decode

from .dsa_index_rotation import (
    KERNEL_INDEX_HEAD_DIM,
    check_index_head_dim,
    decode_writer_rotation_args,
    index_q_rows,
)
from .serving_modules import _reduce_mla_tp_output, mla_decoding_nope_with_pagekv

# --------------------------------------------------------------------------- #
#  Kernel seams                                                               #
# --------------------------------------------------------------------------- #


def _fused_index_k_writer():
    from batchgen_kernels.attention.dsa.indexer.indexer_k_norm_rope_write_fp8 import (
        fused_indexer_k_norm_rope_write_fp8,
    )

    return fused_indexer_k_norm_rope_write_fp8


def _paged_scorer():
    from batchgen_kernels.attention.dsa.paged_mqa_scorer import paged_mqa_score_and_topk

    return paged_mqa_score_and_topk


def _head_gates():
    from batchgen_kernels.attention.dsa.fused_indexer_score import compute_head_gates

    return compute_head_gates


def _selected_positions():
    from batchgen_kernels.attention.dsa.selected_page_table import (
        transform_selected_positions_out,
    )

    return transform_selected_positions_out


def _paged_scorer_out():
    """Graph-capturable scorer: slot indirection + ``num_valid_tokens``, every
    cross-step buffer the caller's."""
    from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
        paged_mqa_score_and_topk_out,
    )

    return paged_mqa_score_and_topk_out


def _latent_token_writer():
    """The graph-safe paged latent write the dense K3 graph span uses."""
    from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused

    return run_paged_kv_token_update_fused


# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #


def batch_slot_indices(current_batch: Sequence[int], seq_id_to_slot,
                       batch_size: int, device) -> torch.Tensor:
    """Page-table slot of each decode row, int32, in batch order."""
    if len(current_batch) != batch_size:
        raise RuntimeError(
            "Kimi-K3.1 DSA decode batch/slot mismatch: "
            f"batch_size={batch_size}, current_batch={len(current_batch)}"
        )
    missing = [int(s) for s in current_batch if int(s) not in seq_id_to_slot]
    if missing:
        raise RuntimeError(
            f"Kimi-K3.1 DSA decode: no aux page-table slot for {missing[:8]}"
        )
    return torch.tensor(
        [int(seq_id_to_slot[int(s)]) for s in current_batch],
        dtype=torch.int32, device=device,
    )


def rows_in_batch_order(table: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
    """``table[slot]`` per decode row: the batch-row-ordered table DeepGEMM
    reads (it has no slot indirection). Contiguous int32."""
    rows = table.index_select(0, slot_indices.to(device=table.device, dtype=torch.long))
    return rows.to(torch.int32).contiguous()


def clamp_to_last_token(indices: torch.Tensor, cache_seqlens: torch.Tensor) -> torch.Tensor:
    """Cap each row's selected positions at its last valid token."""
    cap = (cache_seqlens.to(device=indices.device, dtype=torch.long) - 1).clamp(min=0)
    return torch.minimum(indices.to(torch.long), cap.unsqueeze(-1))


# --------------------------------------------------------------------------- #
#  Steps                                                                      #
# --------------------------------------------------------------------------- #


def write_decode_index_keys(
    module,
    hidden_flat: torch.Tensor,
    aux_manager,
    layer_idx: int,
    aux_slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize + page-write this step's index keys; return the host pair.

    ``(k_fp8 [B, 1, 1, 128] e4m3, scales [B] fp32)`` — exactly what the aux
    Host-append callback takes for a page-split region.
    """
    aux_blocked_k, _, aux_page_table = aux_manager.get_layer_kv_with_page_table(layer_idx)
    return write_index_keys_into(
        module,
        hidden_flat,
        aux_blocked_k=aux_blocked_k,
        aux_page_table=aux_page_table,
        slot_indices=aux_slot_indices.to(device=aux_page_table.device, dtype=torch.int32),
        token_indices=token_indices.to(device=aux_page_table.device, dtype=torch.int32),
        page_size=int(aux_manager.config.page_size_tokens),
    )


def write_index_keys_into(
    module,
    hidden_flat: torch.Tensor,
    *,
    aux_blocked_k: torch.Tensor,
    aux_page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size: int,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The fused writer launch over an explicit aux table — the eager step's
    live table, or a captured graph's static one. Rows at or beyond
    ``num_valid_tokens`` write nothing into the cache (graph padding); the
    returned host-append sources cover every row."""
    indexer = module.indexer
    check_index_head_dim(indexer.index_head_dim)
    bsz = hidden_flat.shape[0]
    k_raw = F.linear(hidden_flat, indexer.wk.weight).to(torch.bfloat16).contiguous()
    k_fp8 = torch.empty(bsz, 1, 1, KERNEL_INDEX_HEAD_DIM,
                        dtype=torch.float8_e4m3fn, device=k_raw.device)
    scales = torch.empty(bsz, dtype=torch.float32, device=k_raw.device)
    rotation = decode_writer_rotation_args(bsz, k_raw.device)
    _fused_index_k_writer()(
        aux_cache=aux_blocked_k,
        k_raw=k_raw,
        ln_weight=indexer.k_norm.weight.data.contiguous(),
        ln_bias=indexer.k_norm.bias.data.contiguous(),
        ln_eps=float(indexer.k_norm.eps),
        cos_cache=rotation["cos_cache"],
        sin_cache=rotation["sin_cache"],
        positions=rotation["positions"],
        page_table=aux_page_table,
        slot_indices=slot_indices,
        token_indices=token_indices,
        page_size_tokens=int(page_size),
        apply_hadamard=bool(rotation["apply_hadamard"]),
        k_fp8_out=k_fp8,
        scales_out=scales,
        num_valid_tokens=num_valid_tokens,
    )
    return k_fp8, scales


def select_long_rows(
    module,
    hidden_flat: torch.Tensor,
    q_a_normed: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    aux_manager,
    layer_idx: int,
    aux_slot_indices: torch.Tensor,
    long_mask: torch.Tensor,
) -> torch.Tensor:
    """``[B, index_topk]`` int64 logical positions: the top-k of each LONG row
    (clamped to its last token); short rows hold ``-1`` and are never read —
    the selection transform takes every key of a short row by itself."""
    indexer = module.indexer
    index_topk = int(indexer.index_topk)
    bsz = hidden_flat.shape[0]
    top_k = torch.full((bsz, index_topk), -1, dtype=torch.long, device=hidden_flat.device)

    long_rows = long_mask.nonzero(as_tuple=True)[0]
    q_idx = F.linear(q_a_normed[long_rows], indexer.wq_b.weight).view(
        long_rows.numel(), indexer.index_n_heads, indexer.index_head_dim)
    q_idx = index_q_rows(q_idx).to(torch.bfloat16).contiguous()
    gates = _head_gates()(
        hidden_flat[long_rows],
        indexer.weights_proj.weight.data,
        indexer.index_n_heads,
        indexer.index_head_dim,
    )
    aux_blocked_k, _, aux_page_table = aux_manager.get_layer_kv_with_page_table(layer_idx)
    long_seqlens = cache_seqlens[long_rows].to(torch.int32).contiguous()
    selected = _paged_scorer()(
        q=q_idx,
        aux_blocked_k=aux_blocked_k,
        block_table=rows_in_batch_order(
            aux_page_table, aux_slot_indices[long_rows.to(aux_slot_indices.device)]),
        head_gates=gates,
        cache_seqlens=long_seqlens,
        topk=min(index_topk, int(max_seqlen)),
        max_seqlen=int(max_seqlen),
        index_head_dim=int(indexer.index_head_dim),
        page_size=int(aux_manager.config.page_size_tokens),
    )
    top_k[long_rows] = clamp_to_last_token(selected, long_seqlens)
    return top_k


def sparse_mla_attention(
    module,
    q_pe: torch.Tensor,
    absorbed_q: torch.Tensor,
    mla_blocked_k: torch.Tensor,
    primary_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    top_k: torch.Tensor,
    page_size: int,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse MLA over the selected PHYSICAL tokens; ``[B, 1, H, kv_lora_rank]``
    (``batchgen.attention.mla.sparse_dispatch``).

    ``primary_page_table`` rows are batch rows, the convention the dense K3
    decode hands FlashMLA, so no slot indirection is passed. Rows at or beyond
    ``num_valid_tokens`` (graph padding) select nothing: length 0.
    """
    bsz, index_topk = top_k.shape
    device = primary_page_table.device
    physical = torch.empty(bsz, index_topk, dtype=torch.int32, device=device)
    lengths = torch.empty(bsz, dtype=torch.int32, device=device)
    _selected_positions()(
        primary_page_table,
        cache_seqlens.to(device=device, dtype=torch.int32),
        top_k.to(device),
        physical,
        lengths,
        page_size=int(page_size),
        num_valid_tokens=num_valid_tokens,
    )
    return sparse_mla_decode(
        q_pe, absorbed_q, mla_blocked_k, physical, lengths, float(module.scaling))


# --------------------------------------------------------------------------- #
#  The layer step                                                             #
# --------------------------------------------------------------------------- #


class KimiDsaDecodeResult(NamedTuple):
    attn_output: torch.Tensor          # [B, 1, hidden]
    k_tensor: torch.Tensor             # [B, 1, 1, 576] latent written to pages
    index_k_fp8: Optional[torch.Tensor]  # [B, 1, 1, 128] e4m3 (host append)
    index_scales: Optional[torch.Tensor]  # [B] fp32 (host append)
    mode: str                          # "dense" | "sparse" | "empty"


def dsa_decode_forward(
    module,
    hidden_states: torch.Tensor,
    q_position_ids: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    gpu_paged_kv_manager,
    gpu_paged_kv_manager_aux,
    layer_idx: int,
    current_batch: Sequence[int],
) -> KimiDsaDecodeResult:
    """One Kimi-K3.1 DSA layer decode step. See the module docstring."""
    if gpu_paged_kv_manager is None or gpu_paged_kv_manager_aux is None:
        raise RuntimeError(
            f"Kimi-K3.1 DSA layer {layer_idx} needs both the primary (latent) and "
            "the auxiliary (index-key) GPU KV managers"
        )
    bsz, q_len, _ = hidden_states.shape
    if bsz == 0:
        attn_output, k_tensor = mla_decoding_nope_with_pagekv(
            module, hidden_states, q_position_ids, cache_seqlens, max_seqlen,
            None, gpu_paged_kv_manager, layer_idx, None,
        )
        return KimiDsaDecodeResult(attn_output, k_tensor, None, None, "empty")
    if q_len != 1:
        raise ValueError("decode path only supports q_len=1")

    hidden_flat = hidden_states.squeeze(1)
    token_indices = q_position_ids.squeeze(-1)
    aux_slots = batch_slot_indices(
        list(current_batch),
        gpu_paged_kv_manager_aux._gpu_page_table_manager.seq_id_to_slot,
        bsz,
        gpu_paged_kv_manager_aux.device,
    )
    index_k_fp8, index_scales = write_decode_index_keys(
        module, hidden_flat, gpu_paged_kv_manager_aux, layer_idx, aux_slots,
        token_indices,
    )

    long_mask = cache_seqlens.to(hidden_flat.device) > int(module.indexer.index_topk)
    if not bool(long_mask.any()):
        attn_output, k_tensor = mla_decoding_nope_with_pagekv(
            module, hidden_states, q_position_ids, cache_seqlens, max_seqlen,
            None, gpu_paged_kv_manager, layer_idx, None,
        )
        return KimiDsaDecodeResult(
            attn_output, k_tensor, index_k_fp8, index_scales, "dense")

    q_a, q_nope, q_pe, offload_kv = _project(module, hidden_flat)
    manager_device = gpu_paged_kv_manager.device
    k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1)).to(manager_device)
    gpu_paged_kv_manager.update_layer_decode_new_token(
        k_tensor=k_tensor,
        v_tensor=None,
        sequence_lengths=token_indices.to(dtype=torch.int32, device=manager_device),
        layer_idx=layer_idx,
        batch_slice=None,
    )
    blocked_k, _, block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(
        layer_idx=layer_idx)
    absorbed_q, out_absorb = _absorb(module, q_nope)

    top_k = select_long_rows(
        module, hidden_flat, q_a, cache_seqlens, max_seqlen,
        gpu_paged_kv_manager_aux, layer_idx, aux_slots, long_mask,
    )
    attn_out = sparse_mla_attention(
        module,
        q_pe.unsqueeze(1).to(blocked_k.dtype).contiguous(),
        absorbed_q.unsqueeze(1).to(blocked_k.dtype).contiguous(),
        blocked_k,
        block_table,
        cache_seqlens,
        top_k,
        int(gpu_paged_kv_manager.config.page_size_tokens),
    )
    attn_output = _finish(module, attn_out, out_absorb, hidden_flat)
    return KimiDsaDecodeResult(
        attn_output.view(bsz, 1, -1), k_tensor, index_k_fp8, index_scales, "sparse")


# --------------------------------------------------------------------------- #
#  The dense path's own statements, shared by the eager and graph steps        #
# --------------------------------------------------------------------------- #


def _project(module, hidden_flat: torch.Tensor):
    """``(q_a, q_nope [B, H, nope], q_pe [B, H, rope], offload_kv [B, 1, 576])``
    — ``mla_decoding_nope_with_pagekv``'s projections, statement for statement
    (NoPE: ``k_pe`` is carried unrotated)."""
    bsz = hidden_flat.shape[0]
    q_a = F.linear(hidden_flat, module.q_a_proj.weight)
    q_a = module.q_a_layernorm(q_a)
    q = F.linear(q_a, module.q_b_proj.weight)
    new_compressed_kv = F.linear(
        hidden_flat, module.kv_a_proj_with_mqa.weight).view(bsz, 1, -1)
    q = q.view(bsz, 1, module.num_heads, module.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
    kv, k_pe = torch.split(
        new_compressed_kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
    normed_kv = module.kv_a_layernorm(kv)
    offload_kv = torch.cat([normed_kv, k_pe], dim=-1)
    return q_a, q_nope.squeeze(2), q_pe.squeeze(2), offload_kv


def _absorb(module, q_nope: torch.Tensor):
    """``(absorbed_q [B, H, kv_lora_rank], out_absorb [H, v, kv_lora_rank])``."""
    kv_b_proj = module.kv_b_proj.weight.data.view(module.num_heads, -1, module.kv_lora_rank)
    q_absorb = kv_b_proj[:, : module.qk_nope_head_dim, :]
    out_absorb = kv_b_proj[:, module.qk_nope_head_dim:, :]
    return torch.einsum("bhd,hdc->bhc", q_nope, q_absorb), out_absorb


def _finish(module, attn_out: torch.Tensor, out_absorb: torch.Tensor,
            hidden_flat: torch.Tensor) -> torch.Tensor:
    """Out-absorb, sigmoid output gate, ``o_proj``, TP reduce: ``[B, hidden]``."""
    bsz = hidden_flat.shape[0]
    attn_output = torch.einsum("bqhc,hdc->bhqd", attn_out, out_absorb)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, module.num_heads * module.v_head_dim)
    if module.use_output_gate:
        gate = F.linear(hidden_flat, module.g_proj.weight).sigmoid()
        attn_output = attn_output * gate
    attn_output = F.linear(attn_output, module.o_proj.weight)
    return _reduce_mla_tp_output(module, attn_output)


# --------------------------------------------------------------------------- #
#  The graph-safe step                                                         #
# --------------------------------------------------------------------------- #


class KimiDsaGraphStep(NamedTuple):
    attn_output: torch.Tensor   # [bucket, 1, hidden]
    k_tensor: torch.Tensor      # [bucket, 1, 1, 576] latent (host append)
    index_k_fp8: torch.Tensor   # [bucket, 1, 1, 128] e4m3 (aux host append)
    index_scales: torch.Tensor  # [bucket] fp32 (aux host append)


def graph_score_width(index_topk: int, max_seqlen: int) -> int:
    """Top-k width a captured step scores at: ``index_topk``, unless the
    graph's whole context capacity is smaller — then every row is short and
    the selection transform takes each row's keys by itself."""
    if int(max_seqlen) <= 0:
        raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")
    return min(int(index_topk), int(max_seqlen))


def dsa_decode_graph_safe(
    module,
    hidden_states: torch.Tensor,
    *,
    layer_idx: int,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    page_size_tokens: int,
    blocked_k: torch.Tensor,
    aux_blocked_k: torch.Tensor,
    aux_page_table: torch.Tensor,
    aux_page_size: int,
    max_seqlen: int,
) -> KimiDsaGraphStep:
    """One Kimi-K3.1 DSA layer decode step with no host synchronization.

    The eager step's math with exactly the differences a capture forces:

    1. one path for every batch: each row is scored, and the selection
       transform takes a short row's keys by itself (its top-k is never read),
       so there is no data-dependent dense/sparse switch;
    2. every table is the caller's STATIC one, in batch-row order
       (``slot_indices`` is the identity row map), never the managers' live
       tables whose addresses may move between steps;
    3. bucket padding rows (``>= num_valid_tokens``) write nothing to either
       cache and select nothing (length 0);
    4. the scorer's context width is the graph's static capacity
       ``max_seqlen``, not the batch maximum.

    ``blocked_k`` / ``aux_blocked_k`` are this layer's cache views, bound by
    the caller once per graph geometry.
    """
    bsz = hidden_states.shape[0]
    hidden_flat = hidden_states.squeeze(1)
    indexer = module.indexer

    # 1. this step's index keys, before anything scores the cache
    index_k_fp8, index_scales = write_index_keys_into(
        module,
        hidden_flat,
        aux_blocked_k=aux_blocked_k,
        aux_page_table=aux_page_table,
        slot_indices=slot_indices,
        token_indices=token_indices,
        page_size=aux_page_size,
        num_valid_tokens=num_valid_tokens,
    )

    # 2. projections + the new latent token over the static table
    q_a, q_nope, q_pe, offload_kv = _project(module, hidden_flat)
    k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1))
    _latent_token_writer()(
        k_cache=blocked_k,
        k_tokens=k_tensor.view(bsz, -1),
        page_table=page_table,
        slot_indices=slot_indices,
        token_indices=token_indices,
        page_size_tokens=int(page_size_tokens),
        num_valid_tokens=num_valid_tokens,
    )
    absorbed_q, out_absorb = _absorb(module, q_nope)

    # 3. score every row
    width = graph_score_width(indexer.index_topk, max_seqlen)
    q_idx = F.linear(q_a, indexer.wq_b.weight).view(
        bsz, indexer.index_n_heads, indexer.index_head_dim)
    q_idx = index_q_rows(q_idx).to(torch.bfloat16).contiguous()
    gates = _head_gates()(
        hidden_flat,
        indexer.weights_proj.weight.data,
        indexer.index_n_heads,
        indexer.index_head_dim,
    )
    device = q_idx.device
    top_k = torch.empty(bsz, width, dtype=torch.int32, device=device)
    _paged_scorer_out()(
        q=q_idx,
        aux_blocked_k=aux_blocked_k,
        aux_page_table=aux_page_table,
        aux_slot_indices=slot_indices,
        head_gates=gates,
        cache_seqlens=cache_seqlens,
        q_fp8=torch.empty(bsz, 1, indexer.index_n_heads, indexer.index_head_dim,
                          dtype=torch.float8_e4m3fn, device=device),
        block_table=torch.empty(bsz, aux_page_table.shape[1],
                                dtype=torch.int32, device=device),
        context_lens=torch.empty(bsz, 1, dtype=torch.int32, device=device),
        top_k_indices=top_k,
        topk=width,
        max_seqlen=int(max_seqlen),
        index_head_dim=int(indexer.index_head_dim),
        page_size=int(aux_page_size),
        num_valid_tokens=num_valid_tokens,
    )

    # 4. attention over the selection
    attn_out = sparse_mla_attention(
        module,
        q_pe.unsqueeze(1).to(blocked_k.dtype).contiguous(),
        absorbed_q.unsqueeze(1).to(blocked_k.dtype).contiguous(),
        blocked_k,
        page_table,
        cache_seqlens,
        clamp_to_last_token(top_k, cache_seqlens),
        int(page_size_tokens),
        num_valid_tokens=num_valid_tokens,
    )
    attn_output = _finish(module, attn_out, out_absorb, hidden_flat)
    return KimiDsaGraphStep(
        attn_output.view(bsz, 1, -1), k_tensor, index_k_fp8, index_scales)
