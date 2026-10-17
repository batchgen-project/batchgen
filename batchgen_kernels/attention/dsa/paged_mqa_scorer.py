"""Production DSA indexer scorer: ``deep_gemm.fp8_paged_mqa_logits`` + top-k.

One decode step, one indexer layer, in four launches:

1. an explicitly saturating cast (clamp to +-448, then cast; see
   ``paged_mqa_logits.saturate_q_to_e4m3_range``) of the post-RoPE/Hadamard
   ``q`` into the static e4m3 buffer DeepGEMM reads;
2. one Triton pass that gathers the aux page table's rows into batch order
   (DeepGEMM has no slot indirection; rows that read nothing get a defined
   all-page-0 row) and derives the 2-D ``context_lens`` twice: the TRUE
   lengths for the top-k, and the guarded copy (row 0 ``>= 1``) DeepGEMM is
   handed, because its sm90 metadata scheduler faults on all-zero lengths at
   a 32-aligned batch (``paged_mqa_logits.guard_deep_gemm_context_lens``);
3. ``fp8_paged_mqa_logits`` straight over the page-split FP8 aux cache -- no
   dense gather, no dequantize, and the ReLU-gated GLM-5 aggregation;
4. the existing production top-k, which owns the masking by the TRUE lengths
   (see ``paged_mqa_logits.topk_from_logits_reference``).

The contract, the row-stride trap and the torch references live in the
torch-only module ``paged_mqa_logits``; this module is the GPU half.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from batchgen_kernels.attention.dsa.fast_topk_cuda import fast_topk_2048_out
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    _topk_from_scores_kernel,
)
from batchgen_kernels.attention.dsa.paged_mqa_logits import (
    PAGED_MQA_BLOCK_KV,
    PAGED_MQA_NEXT_N,
    PagedMqaLogitsPlan,
    align_context_len,
    check_logits_layout,
    guard_deep_gemm_context_lens,
    plan_paged_mqa_logits,
    require_page_split_aux_cache,
    saturate_q_to_e4m3_range,
)

__all__ = [
    "deep_gemm_paged_mqa_logits",
    "paged_mqa_score_and_topk",
    "paged_mqa_score_and_topk_out",
    "prepare_paged_mqa_inputs_out",
    "warm_paged_mqa_logits",
]

_FAST_TOPK_WIDTH = 2048
# Template instantiations already compiled in this process. The metadata kernel
# is templated on align(batch, 32) and the logits kernel on (next_n, heads,
# head_dim), so GLM-5.2's 21 indexer layers share one instantiation per bucket:
# without this, warming would re-launch the same two kernels 21 times per
# bucket for no gain.
_WARMED: set[tuple[int, int, int]] = set()


def _deep_gemm():
    """The DeepGEMM module, or a named error.

    There is no fallback: a page-split FP8 aux cache is unreadable by the
    retired bf16 Triton scorer, so a missing DeepGEMM must stop decode instead
    of silently selecting candidates from a misread cache.
    """

    try:
        import deep_gemm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "GLM-5.2 DSA decode scores the page-split FP8 indexer cache with "
            "deep_gemm.fp8_paged_mqa_logits, but deep_gemm is not importable"
        ) from exc
    for name in ("fp8_paged_mqa_logits", "get_paged_mqa_logits_metadata", "get_num_sms"):
        if not hasattr(deep_gemm, name):
            raise RuntimeError(
                f"the installed deep_gemm does not expose {name}; GLM-5.2 DSA "
                "decode requires the paged MQA logits API"
            )
    return deep_gemm


# --------------------------------------------------------------------------- #
# Per-step input preparation
# --------------------------------------------------------------------------- #

@triton.jit
def _prepare_paged_mqa_inputs_kernel(
    PAGE_TABLE_ptr,        # [num_slots, max_pages] int32
    SLOT_ptr,              # [batch] int32
    SEQLENS_ptr,           # [batch] int32
    NUM_VALID_TOKENS_ptr,  # [1] int32, read only when HAS_VALID_TOKENS
    BLOCK_TABLE_ptr,       # [batch, max_pages] int32 (out)
    CONTEXT_LENS_ptr,      # [batch, 1] int32 (out): TRUE lengths, for the top-k
    DG_CONTEXT_LENS_ptr,   # [batch, 1] int32 (out): guarded lengths, for DeepGEMM
    page_table_stride,
    block_table_stride,
    max_pages,
    HAS_VALID_TOKENS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Gather page-table rows into batch order and derive both ``context_lens``.

    Device-side form of ``prepare_paged_mqa_inputs_reference`` followed by
    ``guard_deep_gemm_context_lens``, fused so the guard costs no extra launch.
    """

    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    slot = tl.load(SLOT_ptr + pid_b).to(tl.int32)
    live = slot >= 0
    if HAS_VALID_TOKENS:
        num_valid = tl.load(NUM_VALID_TOKENS_ptr).to(tl.int32)
        live = live & (pid_b < num_valid)
    seqlen = tl.load(SEQLENS_ptr + pid_b).to(tl.int32)
    context_len = tl.where(live, seqlen, 0)
    # A row that reads nothing never dereferences its slot (a padded row may
    # carry any slot value) and gets a DEFINED all-page-0 table row: the guard
    # below may make DeepGEMM read one token of row 0 even when its true
    # length is 0, and that read must land on a page the aux pool owns.
    reads = context_len > 0
    safe_slot = tl.where(reads, slot, 0).to(tl.int64)

    offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = offs < max_pages
    # For a reading row, `-1` entries are passed through: DeepGEMM never visits
    # a page past the row's context_len, and an out-of-range page index makes
    # its TMA load zero-fill -- the same value the retired Triton scorer
    # produced for an unmapped page.
    pages = tl.load(
        PAGE_TABLE_ptr + safe_slot * page_table_stride + offs,
        mask=mask & reads,
        other=0,
    )
    tl.store(BLOCK_TABLE_ptr + pid_b * block_table_stride + offs, pages, mask=mask)

    if pid_p == 0:
        tl.store(CONTEXT_LENS_ptr + pid_b, context_len)
        # guard_deep_gemm_context_lens: row 0 >= 1, every other row unchanged,
        # so the lengths DeepGEMM schedules over are never all zero.
        dg_len = tl.where(pid_b == 0, tl.maximum(context_len, 1), context_len)
        tl.store(DG_CONTEXT_LENS_ptr + pid_b, dg_len)


def prepare_paged_mqa_inputs_out(
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table_out: torch.Tensor,
    context_lens_out: torch.Tensor,
    num_valid_tokens: Optional[torch.Tensor] = None,
    *,
    deep_gemm_context_lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-launch, allocation-free form of ``prepare_paged_mqa_inputs_reference``.

    Also writes ``deep_gemm_context_lens_out =
    guard_deep_gemm_context_lens(context_lens_out)`` in the same launch.
    ``context_lens_out`` keeps the TRUE lengths (the top-k's bound);
    ``deep_gemm_context_lens_out`` is the only form DeepGEMM may be handed.
    Returns ``(block_table_out, context_lens_out, deep_gemm_context_lens_out)``.
    """

    if page_table.dim() != 2:
        raise ValueError(f"page_table must be 2-D, got {tuple(page_table.shape)}")
    if page_table.dtype != torch.int32:
        raise TypeError(f"page_table must be int32, got {page_table.dtype}")
    if page_table.stride(1) != 1:
        raise ValueError(
            f"page_table rows must be dense, got strides {page_table.stride()}"
        )
    batch = int(slot_indices.shape[0])
    max_pages = int(page_table.shape[1])
    if slot_indices.dim() != 1 or slot_indices.dtype != torch.int32:
        raise TypeError(
            "slot_indices must be a 1-D int32 tensor, got "
            f"{tuple(slot_indices.shape)} {slot_indices.dtype}"
        )
    if tuple(cache_seqlens.shape) != (batch,) or cache_seqlens.dtype != torch.int32:
        raise TypeError(
            f"cache_seqlens must be int32 with shape {(batch,)}, got "
            f"{tuple(cache_seqlens.shape)} {cache_seqlens.dtype}"
        )
    if tuple(block_table_out.shape) != (batch, max_pages):
        raise ValueError(
            f"block_table_out must have shape {(batch, max_pages)}, "
            f"got {tuple(block_table_out.shape)}"
        )
    if block_table_out.dtype != torch.int32 or block_table_out.stride(1) != 1:
        raise TypeError(
            "block_table_out must be int32 with dense rows, got "
            f"{block_table_out.dtype} strides {block_table_out.stride()}"
        )
    for name, lens in (
        ("context_lens_out", context_lens_out),
        ("deep_gemm_context_lens_out", deep_gemm_context_lens_out),
    ):
        if tuple(lens.shape) != (batch, PAGED_MQA_NEXT_N):
            raise ValueError(
                f"{name} must have shape {(batch, PAGED_MQA_NEXT_N)}, "
                f"got {tuple(lens.shape)}"
            )
        if lens.dtype != torch.int32 or not lens.is_contiguous():
            raise TypeError(
                f"{name} must be a contiguous int32 tensor, got "
                f"{lens.dtype} strides {lens.stride()}"
            )
    if deep_gemm_context_lens_out.data_ptr() == context_lens_out.data_ptr():
        raise ValueError(
            "deep_gemm_context_lens_out must not alias context_lens_out: the "
            "top-k masks by the TRUE lengths, DeepGEMM reads the guarded ones"
        )
    if num_valid_tokens is not None:
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(
                f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}"
            )
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                "num_valid_tokens must contain one element, got "
                f"{tuple(num_valid_tokens.shape)}"
            )

    block_p = min(256, triton.next_power_of_2(max_pages))
    grid = (batch, triton.cdiv(max_pages, block_p))
    _prepare_paged_mqa_inputs_kernel[grid](
        page_table,
        slot_indices,
        cache_seqlens,
        num_valid_tokens if num_valid_tokens is not None else cache_seqlens,
        block_table_out,
        context_lens_out,
        deep_gemm_context_lens_out,
        page_table.stride(0),
        block_table_out.stride(0),
        max_pages,
        HAS_VALID_TOKENS=num_valid_tokens is not None,
        BLOCK_P=block_p,
    )
    return block_table_out, context_lens_out, deep_gemm_context_lens_out


# --------------------------------------------------------------------------- #
# The DeepGEMM call
# --------------------------------------------------------------------------- #

def deep_gemm_paged_mqa_logits(
    q_fp8: torch.Tensor,
    aux_blocked_k: torch.Tensor,
    head_gates: torch.Tensor,
    deep_gemm_context_lens: torch.Tensor,
    block_table: torch.Tensor,
    plan: PagedMqaLogitsPlan,
) -> torch.Tensor:
    """``logits[b, t] = sum_h head_gates[b, h] * relu(q[b, h] . k[b, t])``.

    ``deep_gemm_context_lens`` is the 2-D ``[batch, next_n]`` form: the
    item-2a node probe found the 1-D form hard-asserts on the serving build,
    and ``clean_logits=True`` is unreachable alongside the 2-D form, so the
    logits come back UNMASKED outside each row's length and the top-k must
    bound itself by the TRUE lengths, which are elementwise ``<=`` these.

    It must be the GUARDED form (row 0 ``>= 1``): either the preparation
    pass's ``deep_gemm_context_lens_out`` or
    ``guard_deep_gemm_context_lens(...)``. Never pass the true lengths here.

    Both the ``schedule_meta`` and the ``logits`` tensors are allocated by
    DeepGEMM (it has no ``out=`` form). Under capture they come from the
    graph-private pool at fixed addresses and the logits block is freed once
    the top-k has consumed it, so every indexer layer reuses the same block.
    """

    deep_gemm = _deep_gemm()
    # UPSTREAM DEFECT -- why the lengths are guarded: sgl deep_gemm
    # 0.1.5.post3, sm90_fp8_mqa_logits.hpp:188 (the sm90 non-varlen paged MQA
    # metadata scheduler) reads its shared-memory prefix sum out of bounds when
    # EVERY context_len is 0 and batch % 32 == 0 -> CUDA_ERROR_ILLEGAL_ADDRESS
    # (item-2a node repro 2026-10-17: batch 32/96/128 fail, 127 passes, one
    # non-zero row passes). The caller's guard keeps row 0 >= 1 so the total is
    # never zero. Re-test that repro before dropping the guard on a pin bump.
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        deep_gemm_context_lens,
        PAGED_MQA_BLOCK_KV,
        deep_gemm.get_num_sms(),
    )
    logits = deep_gemm.fp8_paged_mqa_logits(
        q_fp8,
        # DeepGEMM asserts a kByte cache and re-views it as (e4m3 values, fp32
        # scales); the KV manager allocates the same bytes as float8_e4m3fn.
        aux_blocked_k.view(torch.uint8),
        head_gates,
        deep_gemm_context_lens,
        block_table,
        schedule_meta,
        plan.max_context_len,
        False,
    )
    check_logits_layout(logits, plan)
    return logits


def _topk_from_logits(
    logits: torch.Tensor,
    context_lens: torch.Tensor,
    top_k_indices: torch.Tensor,
    plan: PagedMqaLogitsPlan,
    num_valid_tokens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run the production top-k over UNMASKED logits.

    ``context_lens`` must be the TRUE lengths, not the guarded copy DeepGEMM
    was handed: a row 0 whose true length is 0 was forced to a one-token read
    and must still select nothing. ``true <= guarded`` elementwise, so every
    column read here was written by DeepGEMM.

    ``fast_topk_2048_out`` reads a row's first ``length`` columns and nothing
    else, so the uninitialized tail DeepGEMM leaves behind is structurally
    unreachable. The generic Triton fallback masks explicitly instead, through
    the same lengths.
    """

    lengths = context_lens.view(plan.batch)
    if plan.topk == _FAST_TOPK_WIDTH and top_k_indices.dtype == torch.int32:
        fast_topk_2048_out(
            logits,
            lengths,
            top_k_indices,
            num_valid_tokens=num_valid_tokens,
        )
        return top_k_indices
    _topk_from_scores_kernel[(plan.batch,)](
        logits,
        top_k_indices,
        num_valid_tokens if num_valid_tokens is not None else lengths,
        lengths,
        max_seqlen=plan.max_context_len,
        topk=plan.topk,
        HAS_VALID_TOKENS=num_valid_tokens is not None,
        MASK_BY_SEQLEN=True,
        BLOCK_N=triton.next_power_of_2(plan.max_context_len),
    )
    return top_k_indices


def paged_mqa_score_and_topk_out(
    *,
    q: torch.Tensor,
    aux_blocked_k: torch.Tensor,
    aux_page_table: torch.Tensor,
    aux_slot_indices: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    q_fp8: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    top_k_indices: torch.Tensor,
    topk: int,
    max_seqlen: int,
    index_head_dim: int,
    page_size: int,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Graph-capturable scorer: every buffer read across steps is the caller's.

    ``context_lens`` receives the TRUE per-row lengths and bounds the top-k.
    The guarded copy DeepGEMM reads is derived from the same inputs inside
    this call by the same launch, so it carries no state between replays.
    """

    plan = plan_paged_mqa_logits(
        q=q,
        aux_blocked_k=aux_blocked_k,
        head_gates=head_gates,
        cache_seqlens=cache_seqlens,
        max_pages_per_seq=int(aux_page_table.shape[1]),
        topk=topk,
        max_seqlen=max_seqlen,
        index_head_dim=index_head_dim,
        page_size=page_size,
    )
    if tuple(q_fp8.shape) != plan.q_shape:
        raise ValueError(
            f"q_fp8 must have shape {plan.q_shape}, got {tuple(q_fp8.shape)}"
        )
    if q_fp8.dtype != torch.float8_e4m3fn or not q_fp8.is_contiguous():
        raise TypeError(
            "q_fp8 must be a contiguous float8_e4m3fn tensor, got "
            f"{q_fp8.dtype} strides {q_fp8.stride()}"
        )
    if tuple(top_k_indices.shape) != (plan.batch, plan.topk):
        raise ValueError(
            f"top_k_indices must have shape {(plan.batch, plan.topk)}, "
            f"got {tuple(top_k_indices.shape)}"
        )
    if top_k_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"top_k_indices must be int32 or int64, got {top_k_indices.dtype}"
        )

    q_fp8.view(plan.batch, plan.num_heads, plan.head_dim).copy_(
        saturate_q_to_e4m3_range(q)
    )
    # The guarded lengths are produced and consumed inside this call (and so
    # inside the captured graph), exactly like DeepGEMM's own schedule_meta and
    # logits, so a graph-private allocation is replay-safe here: nothing
    # outside the graph writes or reads it.
    deep_gemm_context_lens = torch.empty_like(context_lens)
    prepare_paged_mqa_inputs_out(
        aux_page_table,
        aux_slot_indices,
        cache_seqlens,
        block_table,
        context_lens,
        num_valid_tokens=num_valid_tokens,
        deep_gemm_context_lens_out=deep_gemm_context_lens,
    )
    logits = deep_gemm_paged_mqa_logits(
        q_fp8,
        aux_blocked_k,
        head_gates,
        deep_gemm_context_lens,
        block_table,
        plan,
    )
    return _topk_from_logits(
        logits, context_lens, top_k_indices, plan, num_valid_tokens
    )


def paged_mqa_score_and_topk(
    *,
    q: torch.Tensor,
    aux_blocked_k: torch.Tensor,
    block_table: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    topk: int,
    max_seqlen: int,
    index_head_dim: int,
    page_size: int,
) -> torch.Tensor:
    """Eager scorer, for a block table that is ALREADY in batch-row order.

    The eager decode selector reorders the aux page table to batch slots before
    it calls the indexer, so there is no slot indirection left to resolve and
    the Triton preparation pass is not needed.
    """

    plan = plan_paged_mqa_logits(
        q=q,
        aux_blocked_k=aux_blocked_k,
        head_gates=head_gates,
        cache_seqlens=cache_seqlens,
        max_pages_per_seq=int(block_table.shape[1]),
        topk=topk,
        max_seqlen=max_seqlen,
        index_head_dim=index_head_dim,
        page_size=page_size,
    )
    if tuple(block_table.shape) != (plan.batch, plan.max_pages_per_seq):
        raise ValueError(
            "eager block_table must be in batch-row order with shape "
            f"{(plan.batch, plan.max_pages_per_seq)}, got {tuple(block_table.shape)}"
        )
    if block_table.dtype != torch.int32:
        raise TypeError(
            f"block_table must be int32 (the GPU page table's dtype), got "
            f"{block_table.dtype}"
        )
    if block_table.stride(1) != 1:
        raise ValueError(
            f"block_table rows must be dense, got strides {block_table.stride()}"
        )

    q_fp8 = (
        saturate_q_to_e4m3_range(q)
        .to(torch.float8_e4m3fn)
        .reshape(plan.q_shape)
        .contiguous()
    )
    context_lens = cache_seqlens.reshape(plan.batch, PAGED_MQA_NEXT_N).contiguous()
    # Same never-all-zero guarantee as the graph path. The eager block table is
    # the caller's (possibly the live page table itself), so it is not
    # rewritten: a row 0 with true length 0 -- unreachable here, the eager
    # selector routes rows with seqlen <= topk to its dense path -- would have
    # its forced one-token read land on block_table[0, 0], a real page or the
    # `-1` sentinel DeepGEMM's TMA zero-fills, and the top-k discards it.
    deep_gemm_context_lens = guard_deep_gemm_context_lens(context_lens)
    top_k_indices = torch.empty(
        plan.batch, plan.topk, dtype=torch.int32, device=q.device
    )
    logits = deep_gemm_paged_mqa_logits(
        q_fp8, aux_blocked_k, head_gates, deep_gemm_context_lens, block_table, plan
    )
    return _topk_from_logits(logits, context_lens, top_k_indices, plan, None)


# --------------------------------------------------------------------------- #
# JIT warm-up
# --------------------------------------------------------------------------- #

def warm_paged_mqa_logits(
    *,
    aux_blocked_k: torch.Tensor,
    batch: int,
    num_heads: int,
    index_head_dim: int,
    max_pages_per_seq: int,
    max_seqlen: int,
    page_size: int,
) -> None:
    """Compile both DeepGEMM kernels before a CUDA graph capture can need them.

    ``fp8_paged_mqa_logits`` and ``get_paged_mqa_logits_metadata`` are JIT
    kernels: the first call for a given template instantiation compiles and
    loads a module. The metadata kernel's template includes ``align(batch,
    32)``, so every captured bucket needs its own warm call; the logits kernel
    itself is keyed on (next_n, heads, head_dim, block_kv, stages) and compiles
    once.

    Warming runs on its OWN temporaries -- the template, not the pointer, is
    what gets cached, and this keeps the segment's static buffers untouched.
    Reading page 0 for one page of a zeroed cache is enough to launch.
    """

    require_page_split_aux_cache(aux_blocked_k, index_head_dim)
    device = aux_blocked_k.device
    batch = int(batch)
    # align(batch, 32) is the metadata kernel's own template parameter, so
    # buckets that round to the same value share one instantiation.
    key = ((batch + 31) // 32 * 32, int(num_heads), int(index_head_dim))
    if key in _WARMED:
        return
    q_fp8 = torch.zeros(
        batch,
        PAGED_MQA_NEXT_N,
        int(num_heads),
        int(index_head_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    head_gates = torch.zeros(batch, int(num_heads), dtype=torch.float32, device=device)
    block_table = torch.zeros(
        batch, int(max_pages_per_seq), dtype=torch.int32, device=device
    )
    # Every row is >= 1 (page_size and max_seqlen are positive), so these
    # lengths already satisfy guard_deep_gemm_context_lens's contract.
    context_lens = torch.full(
        (batch, PAGED_MQA_NEXT_N),
        min(int(page_size), int(max_seqlen)),
        dtype=torch.int32,
        device=device,
    )
    plan = PagedMqaLogitsPlan(
        batch=batch,
        next_n=PAGED_MQA_NEXT_N,
        num_heads=int(num_heads),
        head_dim=int(index_head_dim),
        num_pages=int(aux_blocked_k.shape[0]),
        page_size=int(page_size),
        max_pages_per_seq=int(max_pages_per_seq),
        topk=1,
        max_seqlen=int(max_seqlen),
        max_context_len=align_context_len(int(max_seqlen)),
    )
    deep_gemm_paged_mqa_logits(
        q_fp8, aux_blocked_k, head_gates, context_lens, block_table, plan
    )
    _WARMED.add(key)
