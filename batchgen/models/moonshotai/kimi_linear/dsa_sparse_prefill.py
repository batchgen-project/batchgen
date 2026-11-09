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

Attention runs through the same seam as Kimi-K3.1's sparse decode
(``batchgen.attention.mla.sparse_dispatch``) with the absorbed query over this
chunk's latent rows, every query token its own row and its packed-global
selection its token rows: FlashAttention-3 ``qv`` on Hopper (no head-count
constraint; K3 runs 12 query heads per rank under TP8), flash_mla's sparse
prefill on Blackwell (heads zero-padded to its 64-head tile).

Everything else is the dense prefill's own statement: the projections, the
NoPE latent written to the Host cache, the absorbed query and out-absorb, the
sigmoid output gate, ``o_proj`` and the TP reduce.

This is also the pooled form. Under the in-wave prefill prefix pool a row
continues a prefix that is already KV in the pool, so its keys are not only
the ones this chunk computes: ``build_dsa_pool_prefill`` turns the chunk's
page table into a per-row key list over pool slots, the chunk's own keys are
written into the pool before the scorer gathers that list, and the selected
indices are mapped back to pool slots for attention. Both regions take part --
the latent pool row carries attention's KV, the indexer pool row carries the
keys selection scores -- which is why a DSA pool wave is refused unless both
rows are handed in. Prefill then stays the sparse path for every pooled wave,
short rows included, exactly as it does for a long packed batch: the selection
is the identity whenever a row sees at most ``index_topk`` keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from batchgen.attention.mla.sparse_dispatch import sparse_mla_prefill

from .dsa_decode import _head_gates
from .dsa_index_rotation import check_index_head_dim, index_q_rows
from .serving_modules import _reduce_mla_tp_output

#: ``fast_topk_2048_out`` is the only top-k the production causal scorer has.
PREFILL_TOPK = 2048


@dataclass(frozen=True)
class DsaPoolPrefill:
    """One prefix-pool chunk's addressing, shared by every DSA layer.

    A pooled row continues a prefix that is already KV in the pool -- computed
    by an earlier row of this wave or LOADED from the prefix cache -- so its
    query tokens must select over the cached keys as well as their own.

    ``key_slots`` lists every row's keys ``[0, cache_len)`` as pool slots, rows
    concatenated; a query token's causal range is its row's span in that list
    up to and including its own absolute position, which is exactly the
    packed-row range of the no-pool form shifted by the prefix length. Top-k
    indices over that list are mapped back to pool slots, so both the scorer
    and attention read the pool directly and nothing re-materializes a key.
    """

    slots: torch.Tensor  # int64 [T]: pool slot of each query token
    key_slots: torch.Tensor  # int64 [K]
    causal_starts: torch.Tensor  # int32 [T]
    causal_ends: torch.Tensor  # int32 [T]
    max_cache_seqlen: int


def build_dsa_pool_prefill(
    *,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    position_ids: torch.Tensor,
    slots: torch.Tensor,
    page_tokens: int,
    max_cache_seqlen: int,
) -> DsaPoolPrefill:
    """Address a pool chunk whose rows hold ancestor pages then their own.

    Pooled segments and tails start on page boundaries, so a row's key at
    absolute position ``t`` sits in slot ``t % page_tokens`` of its
    ``t // page_tokens``-th page.
    """

    device = page_table.device
    rows = cache_seqlens.numel()
    cache = cache_seqlens.to(torch.int64)
    key_offsets = torch.zeros(rows + 1, dtype=torch.int64, device=device)
    key_offsets[1:] = torch.cumsum(cache, 0)
    total_keys = int(key_offsets[-1])
    key_rows = torch.repeat_interleave(
        torch.arange(rows, device=device), cache, output_size=total_keys
    )
    t = torch.arange(total_keys, device=device) - key_offsets[:-1][key_rows]
    key_slots = (
        page_table.to(torch.int64)[key_rows, t // page_tokens] * page_tokens
        + t % page_tokens
    )
    q_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
    q_rows = torch.repeat_interleave(
        torch.arange(rows, device=device), q_lengths,
        output_size=position_ids.numel(),
    )
    starts = key_offsets[:-1][q_rows]
    ends = starts + position_ids.to(torch.int64) + 1
    return DsaPoolPrefill(
        slots=slots.to(torch.int64),
        key_slots=key_slots.contiguous(),
        causal_starts=starts.to(torch.int32).contiguous(),
        causal_ends=ends.to(torch.int32).contiguous(),
        max_cache_seqlen=int(max_cache_seqlen),
    )


def map_topk_to_pool_slots(
    topk_indices: torch.Tensor, key_slots: torch.Tensor
) -> torch.Tensor:
    """Key-list offsets -> pool slots; ``-1`` padding stays ``-1``."""

    mapped = key_slots[topk_indices.clamp(min=0).to(torch.int64)].to(
        topk_indices.dtype
    )
    return torch.where(topk_indices >= 0, mapped, topk_indices).contiguous()


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

    from batchgen.quantization.act_quant import act_quant
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
    pool: DsaPoolPrefill | None = None,
    latent_pool: torch.Tensor | None = None,
    indexer_pool: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA prefill of a packed batch: ``(attn_output [T, hidden],
    offload_kv [T, kv_lora_rank + qk_rope_head_dim])``.

    ``index_keys`` are this chunk's ``[T, 128]`` bf16 index keys
    (``dsa_prefill.prefill_index_keys``), the same keys the Host write stores.

    With ``pool`` the rows continue prefixes already resident in the in-wave
    prefix pool. This layer's latent and index keys are written to their pool
    slots first; the scorer then gathers each row's WHOLE key list -- cached
    prefix keys and this chunk's own, in one packed tensor -- so a suffix query
    selects over cached and new keys together, and attention reads the latent
    pool (``latent_pool``, this layer's row) through the selected slots. The
    no-pool form is the same statement with every row's prefix empty.
    """
    indexer = module.indexer
    check_index_head_dim(indexer.index_head_dim)
    total = hidden_2d.shape[0]
    if index_keys.shape != (total, indexer.index_head_dim):
        raise ValueError(
            f"index_keys must be [{total}, {indexer.index_head_dim}], got "
            f"{tuple(index_keys.shape)}"
        )
    if pool is not None:
        if latent_pool is None or indexer_pool is None:
            raise RuntimeError(
                "a Kimi-K3.1 DSA pool-prefill layer needs both this layer's "
                "latent pool row and its indexer pool row; without the "
                "indexer row the suffix would select over its own keys only "
                "and silently ignore the cached prefix"
            )
        if int(pool.slots.numel()) != total:
            raise ValueError(
                f"the pool chunk names {int(pool.slots.numel())} slot(s) for "
                f"{total} query token(s)"
            )
    elif latent_pool is not None or indexer_pool is not None:
        raise RuntimeError(
            "pool rows were given without a pool chunk to address them"
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
    if pool is None:
        starts, ends = packed_causal_ranges(cu_seqlens, total)
        score_keys = index_keys.to(torch.bfloat16)
        score_width = int(max_seqlen)
    else:
        # The chunk's own keys land in the pool first, so the gather below
        # returns cached prefix keys and new keys in one causal order.
        starts, ends = pool.causal_starts, pool.causal_ends
        indexer_flat = indexer_pool.view(-1, index_keys.shape[-1])
        indexer_flat[pool.slots] = index_keys.to(indexer_flat.dtype)
        score_keys = indexer_flat[pool.key_slots].to(torch.bfloat16)
        score_width = pool.max_cache_seqlen
    q_idx = F.linear(q_a, indexer.wq_b.weight).view(
        total, indexer.index_n_heads, indexer.index_head_dim)
    q_idx = index_q_rows(q_idx).to(torch.bfloat16).contiguous()
    gates = _head_gates()(
        hidden_2d, indexer.weights_proj.weight.data,
        indexer.index_n_heads, indexer.index_head_dim)
    topk = int(indexer.index_topk)
    indices = _prefill_index_topk()(
        q_idx=q_idx,
        index_keys=score_keys,
        head_gates=gates,
        causal_starts=starts,
        causal_ends=ends,
        max_seqlen=int(score_width),
        topk=topk,
    )
    selected = (ends - starts).clamp(max=topk).to(torch.int32).contiguous()

    # ---- attention: the sparse seam over the latent, one row per query ----
    kv_b = module.kv_b_proj.weight.view(heads, -1, lora)
    q_absorb = kv_b[:, : module.qk_nope_head_dim, :]
    out_absorb = kv_b[:, module.qk_nope_head_dim:, :]
    q_latent = torch.einsum("thd,hdc->thc", q_nope, q_absorb)
    if pool is None:
        kv_rows = offload_kv.view(total, offload_kv.shape[-1])
    else:
        latent_flat = latent_pool.view(-1, offload_kv.shape[-1])
        latent_flat[pool.slots] = offload_kv.to(latent_flat.dtype)
        # Pool slots ARE the token rows of the selection, so the kernel's KV
        # source is the whole pool row.
        kv_rows = latent_flat
        indices = map_topk_to_pool_slots(indices, pool.key_slots)
    attn_latent = sparse_mla_prefill(
        q_pe.reshape(total, 1, heads, module.qk_rope_head_dim).contiguous(),
        q_latent.reshape(total, 1, heads, lora).to(q_pe.dtype).contiguous(),
        kv_rows,
        indices.to(torch.int32),
        selected,
        float(module.scaling),
    )
    attn_heads = torch.einsum("thc,hdc->thd", attn_latent[:, 0], out_absorb)

    attn_output = attn_heads.reshape(total, heads * module.v_head_dim).contiguous()
    if module.use_output_gate:
        attn_output = attn_output * module.g_proj(hidden_2d).sigmoid()
    attn_output = module.o_proj(attn_output)
    attn_output = _reduce_mla_tp_output(module, attn_output)
    return attn_output, offload_kv
