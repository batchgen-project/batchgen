# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 DSA prefill attention: each prompt token attends to its top-k.

A query token whose causal context holds at most ``index_topk`` keys has
nothing to select, and the reference oracle proves sparse == dense there, so
a packed batch whose longest row fits in ``index_topk`` keeps the unchanged
dense K3 prefill. Once a row is longer, every query beyond the first
``index_topk`` positions must attend only to its indexer-selected keys — dense
prefill there is a different model. This module is that sparse prefill.

Selection mirrors the production page-split DSA prefill: the causal
ReLU-gated FP8 scorer ``deep_gemm.fp8_mqa_logits`` over packed-global
``[start, end)`` key ranges, then ``fast_topk_2048_out`` (dense prefix for a
short row, exact top-2048 for a long one) and an offset to packed-global key
indices. Kimi-K3.1's indexer weights are bf16, so the activations are
quantized with ``act_quant`` instead of reusing FP8 weight scales.

Attention runs on the SAME kernel as Kimi-K3.1's sparse decode: FlashAttention-3
with ``qv`` (the absorbed query) over a page-size-1 view of this chunk's
latent, every query token its own row and its packed-global selection its
page table. That keeps one sparse-attention kernel and one set of plumbing
for prefill and decode, and imposes no head-count constraint — K3 runs 12
query heads per rank under TP8.

Everything else is the dense prefill's own statement: the projections, the
NoPE latent written to the Host cache, the absorbed query and out-absorb, the
sigmoid output gate, ``o_proj`` and the TP reduce.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .dsa_decode import _fa3_with_kvcache, _head_gates, _sparse_fa3_num_splits
from .dsa_index_rotation import check_index_head_dim, index_q_rows
from .serving_modules import _reduce_mla_tp_output

#: ``fast_topk_2048_out`` is the only top-k the production causal scorer has.
PREFILL_TOPK = 2048


def packed_causal_ranges(cu_seqlens: torch.Tensor,
                         total_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Packed-global ``[start, end)`` key range of every query token.

    Token ``i`` of the packed batch belongs to the row starting at
    ``cu_seqlens[r]`` and may see keys ``[cu_seqlens[r], i]``: ``end = i + 1``.
    int32, contiguous.
    """
    cu = cu_seqlens.to(torch.int64)
    if cu.dim() != 1 or cu.numel() < 2 or int(cu[0]) != 0 or int(cu[-1]) != total_tokens:
        raise ValueError(
            f"cu_seqlens must run 0..{total_tokens} over the packed tokens, got "
            f"{cu.tolist() if cu.numel() < 16 else tuple(cu.shape)}"
        )
    lengths = cu[1:] - cu[:-1]
    if bool((lengths <= 0).any()):
        raise ValueError("every packed row must hold at least one token")
    starts = torch.repeat_interleave(cu[:-1], lengths, output_size=total_tokens)
    ends = torch.arange(1, total_tokens + 1, device=cu.device, dtype=torch.int64)
    return starts.to(torch.int32).contiguous(), ends.to(torch.int32).contiguous()


def _prefill_index_topk():
    """Production causal scorer + top-k (GPU). Returns the callable."""
    return _score_and_select_prefill


def _score_and_select_prefill(
    *,
    q_idx: torch.Tensor,
    index_keys: torch.Tensor,
    head_gates: torch.Tensor,
    causal_starts: torch.Tensor,
    causal_ends: torch.Tensor,
    max_seqlen: int,
    topk: int,
) -> torch.Tensor:
    """``[T, topk]`` int32 packed-global key indices, ``-1`` padded.

    ``deep_gemm.fp8_mqa_logits`` computes ``sum_h w_h relu(q_h . k)`` with the
    per-token K scale folded in; with ``w = head_gates * q_scale`` that is the
    reference ``sum_h (gate_h) relu(D^-0.5 q_h . k)`` (``head_gates`` already
    carries ``H^-0.5 * D^-0.5``). Chunked over query rows so the fp32 logits
    stay bounded, exactly as the production prefill scorer chunks.
    """
    if int(topk) != PREFILL_TOPK:
        raise NotImplementedError(
            f"the causal prefill top-k is fast_topk_2048 only; index_topk={topk}"
        )
    import deep_gemm

    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen_kernels.attention.dsa.fast_topk_cuda import fast_topk_2048_out

    total = q_idx.shape[0]
    k_fp8, k_scale = act_quant(index_keys.contiguous())
    k_fp8_with_scale = (k_fp8, k_scale.squeeze(-1).contiguous())
    lengths = (causal_ends - causal_starts).contiguous()
    output = torch.empty(total, PREFILL_TOPK, dtype=torch.int32, device=q_idx.device)
    score_width = ((int(max_seqlen) + 255) // 256) * 256
    rows = max(4, min(8192, (2 * 1024 ** 3) // max(score_width * 4, 1)) // 4 * 4)
    for start in range(0, total, rows):
        end = min(start + rows, total)
        q_fp8, q_scale = act_quant(q_idx[start:end].contiguous())
        weights = (head_gates[start:end].unsqueeze(-1) * q_scale).squeeze(-1).contiguous()
        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            k_fp8_with_scale,
            weights,
            causal_starts[start:end],
            causal_ends[start:end],
            max_seqlen_k=score_width,
            clean_logits=False,
        )
        fast_topk_2048_out(logits, lengths[start:end], output[start:end])
    return offset_to_packed_global(output, causal_starts)


def offset_to_packed_global(indices: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    """Row-relative top-k indices -> packed-global key indices; ``-1`` kept."""
    return torch.where(
        indices >= 0, indices + starts.to(indices.dtype).unsqueeze(-1), indices
    ).contiguous()


def dsa_sparse_prefill(
    module,
    hidden_2d: torch.Tensor,
    index_keys: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA prefill of a packed batch: ``(attn_output [T, hidden],
    offload_kv [T, kv_lora_rank + qk_rope_head_dim])``.

    ``index_keys`` are this chunk's ``[T, 128]`` bf16 index keys
    (``dsa_prefill.prefill_index_keys``), the same keys the Host write stores.
    """
    indexer = module.indexer
    check_index_head_dim(indexer.index_head_dim)
    total = hidden_2d.shape[0]
    if index_keys.shape != (total, indexer.index_head_dim):
        raise ValueError(
            f"index_keys must be [{total}, {indexer.index_head_dim}], got "
            f"{tuple(index_keys.shape)}"
        )
    heads, lora = module.num_heads, module.kv_lora_rank

    # ---- the dense prefill's projections ----
    q_a = module.q_a_layernorm(module.q_a_proj(hidden_2d))
    q = module.q_b_proj(q_a).view(total, heads, module.q_head_dim)
    q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)
    compressed_kv = module.kv_a_proj_with_mqa(hidden_2d)
    compressed_kv, k_pe = torch.split(compressed_kv, [lora, module.qk_rope_head_dim], dim=-1)
    normed_kv = module.kv_a_layernorm(compressed_kv)
    offload_kv = torch.cat([normed_kv, k_pe], dim=-1)

    # ---- selection ----
    starts, ends = packed_causal_ranges(cu_seqlens, total)
    q_idx = F.linear(q_a, indexer.wq_b.weight).view(
        total, indexer.index_n_heads, indexer.index_head_dim)
    q_idx = index_q_rows(q_idx).to(torch.bfloat16).contiguous()
    gates = _head_gates()(
        hidden_2d, indexer.weights_proj.weight.data,
        indexer.index_n_heads, indexer.index_head_dim)
    topk = int(indexer.index_topk)
    indices = _prefill_index_topk()(
        q_idx=q_idx,
        index_keys=index_keys.to(torch.bfloat16),
        head_gates=gates,
        causal_starts=starts,
        causal_ends=ends,
        max_seqlen=int(max_seqlen),
        topk=topk,
    )
    selected = (ends - starts).clamp(max=topk).to(torch.int32).contiguous()

    # ---- attention: FA3 qv over this chunk's latent, one row per query ----
    kv_b = module.kv_b_proj.weight.view(heads, -1, lora)
    q_absorb = kv_b[:, : module.qk_nope_head_dim, :]
    out_absorb = kv_b[:, module.qk_nope_head_dim:, :]
    q_latent = torch.einsum("thd,hdc->thc", q_nope, q_absorb)
    flat_kv = offload_kv.view(total, 1, 1, offload_kv.shape[-1])
    attn_latent = _fa3_with_kvcache()(
        q=q_pe.reshape(total, 1, heads, module.qk_rope_head_dim).contiguous(),
        k_cache=flat_kv[..., lora:],
        v_cache=flat_kv[..., :lora],
        qv=q_latent.reshape(total, 1, heads, lora).to(q_pe.dtype).contiguous(),
        page_table=indices.to(torch.int32),
        cache_seqlens=selected,
        softmax_scale=float(module.scaling),
        causal=True,
        num_splits=_sparse_fa3_num_splits(),
        return_softmax_lse=False,
    )
    attn_heads = torch.einsum("thc,hdc->thd", attn_latent[:, 0], out_absorb)

    attn_output = attn_heads.reshape(total, heads * module.v_head_dim).contiguous()
    if module.use_output_gate:
        attn_output = attn_output * module.g_proj(hidden_2d).sigmoid()
    attn_output = module.o_proj(attn_output)
    attn_output = _reduce_mla_tp_output(module, attn_output)
    return attn_output, offload_kv
