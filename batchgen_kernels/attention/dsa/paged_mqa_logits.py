"""Contract for scoring the DSA indexer with ``deep_gemm.fp8_paged_mqa_logits``.

The kernel computes the GLM-5 ground-truth indexer aggregation straight out of
the page-split FP8 aux cache::

    logits[b, t] = sum_h  weights[b, h] * relu(q[b, h] . k[b, t])

``relu`` is hardcoded in DeepGEMM's epilogue, so swapping the bf16 Triton
scorer for this kernel is a SEMANTICS change (the Triton scorer summed signed
per-head scores). ``weights`` is BatchGen's ``head_gates`` unchanged: the gates
already carry ``n_heads**-0.5 * index_head_dim**-0.5``, and
``relu(s * x) == s * relu(x)`` for ``s > 0``, so folding the positive scale
into the weights instead of into ``q . k`` is exact.

This module deliberately imports nothing but torch. It holds the shape/dtype/
stride contract, the row-stride trap, the masking contract and the torch
references the GPU kernels are measured against, so all of it stays importable
and unit-testable without Triton, DeepGEMM or a GPU.

Three contract points are easy to get wrong and are therefore checked here:

1. **Row stride.** DeepGEMM allocates ``[batch * next_n, align(mcl, 256)]``
   and returns a SLICE to ``max_context_len``, so the row stride is the
   *aligned* value. ``align_context_len`` is what the integration passes as
   ``max_context_len``, which makes the slice the whole allocation and the
   realized stride equal to the number of columns. ``check_logits_layout``
   re-asserts that on the returned tensor instead of trusting the arithmetic.
2. **Masking.** ``clean_logits=True`` is unreachable for the 2-D
   ``context_lens`` form this integration must use, so DeepGEMM leaves
   positions outside ``[0, context_len)`` UNINITIALIZED. Masking is the top-k's
   job: ``topk_from_logits_reference`` is the semantics the production CUDA
   top-k implements (it reads a row's first ``length`` columns and nothing
   else), and the unit tests use it to prove that arbitrary bytes past
   ``length`` cannot change a selection.
3. **q quantization.** DeepGEMM takes ``q`` as plain e4m3 with no scale
   channel, so the conversion is a saturating cast -- the same form DeepGEMM's
   own reference test and the item-2a node probe used.
4. **Never all-zero lengths.** DeepGEMM's sm90 metadata scheduler reads past
   its shared-memory prefix sum when EVERY ``context_len`` is 0 and the batch
   is a multiple of 32 (``guard_deep_gemm_context_lens``). DeepGEMM is
   therefore handed a guarded COPY whose row 0 is at least 1, while the top-k
   keeps masking by the TRUE lengths; ``true <= guarded`` elementwise, so the
   top-k still never reads a column DeepGEMM did not write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from batchgen_kernels.attention.dsa.indexer_k_quant import (
    FP8_MAX,
    SCALE_BYTES_PER_TOKEN,
    cache_data_dim,
    cache_is_page_split,
)

__all__ = [
    "PAGED_MQA_BLOCK_KV",
    "PAGED_MQA_LOGITS_ALIGN",
    "PAGED_MQA_NEXT_N",
    "PagedMqaLogitsPlan",
    "align_context_len",
    "check_logits_layout",
    "guard_deep_gemm_context_lens",
    "plan_paged_mqa_logits",
    "prepare_paged_mqa_inputs_reference",
    "quantize_q_e4m3_reference",
    "require_page_split_aux_cache",
    "saturate_q_to_e4m3_range",
    "topk_from_logits_reference",
]

# Hard-asserted by DeepGEMM's host side: the aux cache page must hold exactly
# 64 tokens.
PAGED_MQA_BLOCK_KV = 64
# DeepGEMM's `split_kv`: the logits allocation is aligned to this and the
# kernel asserts `logits_stride % split_kv == 0`.
PAGED_MQA_LOGITS_ALIGN = 256
# One decode token per row. DeepGEMM also accepts 2, which BatchGen's decode
# does not produce on this path.
PAGED_MQA_NEXT_N = 1


def align_context_len(max_seqlen: int) -> int:
    """``max_context_len`` to pass DeepGEMM so the logits rows are contiguous.

    Rounding UP is not an optimization: the eager mixed-batch path derives its
    ``max_seqlen`` from the longest row in the batch, which is page-aligned at
    best, so demanding ``max_seqlen % 256 == 0`` would reject most real
    batches. Passing the aligned value instead makes DeepGEMM's returned slice
    the whole allocation (stride == columns, which the top-k assumes) and the
    extra columns are never read: every reader is bounded by the row's own
    ``context_len <= max_seqlen``.
    """

    if max_seqlen <= 0:
        raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")
    align = PAGED_MQA_LOGITS_ALIGN
    return (int(max_seqlen) + align - 1) // align * align


@dataclass(frozen=True)
class PagedMqaLogitsPlan:
    """Everything the launch needs, derived without touching device memory."""

    batch: int
    next_n: int
    num_heads: int
    head_dim: int
    num_pages: int
    page_size: int
    max_pages_per_seq: int
    topk: int
    max_seqlen: int
    max_context_len: int

    @property
    def logits_shape(self) -> Tuple[int, int]:
        return (self.batch * self.next_n, self.max_context_len)

    @property
    def q_shape(self) -> Tuple[int, int, int, int]:
        return (self.batch, self.next_n, self.num_heads, self.head_dim)


def require_page_split_aux_cache(
    aux_blocked_k: torch.Tensor,
    index_head_dim: int,
) -> None:
    """Fail loud unless the aux cache is in DeepGEMM's page-split FP8 layout.

    The selector gates on the CACHE FORMAT, never on a model name or an SM
    version: H20 and H200 are both ``arch_major == 9``, so there is no hardware
    axis to dispatch on, and the one property that decides whether DeepGEMM can
    read the cache is whether the KV manager allocated the page-split carrier.
    """

    if not cache_is_page_split(aux_blocked_k, index_head_dim):
        raise RuntimeError(
            "deep_gemm.fp8_paged_mqa_logits requires the page-split FP8 aux "
            "indexer cache (per token: "
            f"{index_head_dim} e4m3 bytes + {SCALE_BYTES_PER_TOKEN} fp32 scale "
            f"bytes); got a cache of shape {tuple(aux_blocked_k.shape)} and "
            f"dtype {aux_blocked_k.dtype}"
        )


def plan_paged_mqa_logits(
    *,
    q: torch.Tensor,
    aux_blocked_k: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_pages_per_seq: int,
    topk: int,
    max_seqlen: int,
    index_head_dim: int,
    page_size: int,
) -> PagedMqaLogitsPlan:
    """Validate the whole call against DeepGEMM's host asserts, up front.

    Every check is on shapes, dtypes and strides, so this never reads device
    memory and never synchronizes. Raising here instead of inside DeepGEMM
    turns an opaque ``DG_HOST_ASSERT`` abort into a named BatchGen error.
    """

    require_page_split_aux_cache(aux_blocked_k, index_head_dim)

    if q.dim() != 3:
        raise ValueError(
            f"q must be [batch, num_heads, head_dim], got {tuple(q.shape)}"
        )
    batch, num_heads, head_dim = (int(x) for x in q.shape)
    if head_dim != index_head_dim:
        raise ValueError(
            f"q head_dim {head_dim} != indexer head_dim {index_head_dim}"
        )
    if head_dim not in (32, 64, 128):
        raise ValueError(
            f"deep_gemm.fp8_paged_mqa_logits supports head_dim 32/64/128, got {head_dim}"
        )
    if 128 % num_heads != 0:
        raise ValueError(
            "deep_gemm.fp8_paged_mqa_logits requires 128 % num_heads == 0, "
            f"got num_heads={num_heads}"
        )

    if aux_blocked_k.dim() != 4:
        raise ValueError(
            "aux_blocked_k must be [num_pages, page_size, 1, carrier], "
            f"got {tuple(aux_blocked_k.shape)}"
        )
    num_pages, cache_page_size, num_kv_heads, carrier = (
        int(x) for x in aux_blocked_k.shape
    )
    if num_kv_heads != 1:
        raise ValueError(
            f"aux_blocked_k must have exactly one KV head, got {num_kv_heads}"
        )
    if cache_page_size != int(page_size):
        raise ValueError(
            f"aux cache page size {cache_page_size} != page_size {page_size}"
        )
    if cache_page_size != PAGED_MQA_BLOCK_KV:
        raise ValueError(
            "deep_gemm.fp8_paged_mqa_logits hard-asserts block_kv == "
            f"{PAGED_MQA_BLOCK_KV} tokens per page, got {cache_page_size}"
        )
    if carrier != index_head_dim + SCALE_BYTES_PER_TOKEN:
        raise ValueError(
            f"page-split carrier must be {index_head_dim + SCALE_BYTES_PER_TOKEN} "
            f"bytes per token, got {carrier}"
        )
    # DeepGEMM re-views the one cache tensor as (e4m3 values, fp32 scales) with
    # strides derived from stride(0); the two inner strides must be the carrier
    # and the last must be dense.
    if aux_blocked_k.stride(1) != carrier or aux_blocked_k.stride(2) != carrier:
        raise ValueError(
            "aux cache strides(1,2) must both equal the per-token carrier "
            f"{carrier}, got {aux_blocked_k.stride()}"
        )
    if aux_blocked_k.stride(3) != 1:
        raise ValueError(
            f"aux cache last dim must be dense, got stride {aux_blocked_k.stride(3)}"
        )
    if aux_blocked_k.stride(0) % 4 != 0:
        raise ValueError(
            "aux cache page stride must be 4-byte aligned (DeepGEMM re-views "
            f"the scale region as fp32), got {aux_blocked_k.stride(0)}"
        )
    if int(cache_data_dim(aux_blocked_k)) != index_head_dim:
        raise ValueError(
            f"aux cache data dim {int(cache_data_dim(aux_blocked_k))} != "
            f"indexer head dim {index_head_dim}"
        )

    if tuple(head_gates.shape) != (batch, num_heads):
        raise ValueError(
            f"head_gates must have shape {(batch, num_heads)}, "
            f"got {tuple(head_gates.shape)}"
        )
    if head_gates.dtype != torch.float32:
        raise TypeError(f"head_gates must be float32, got {head_gates.dtype}")
    if not head_gates.is_contiguous():
        raise ValueError("head_gates must be contiguous")

    if tuple(cache_seqlens.shape) != (batch,):
        raise ValueError(
            f"cache_seqlens must have shape {(batch,)}, "
            f"got {tuple(cache_seqlens.shape)}"
        )
    if cache_seqlens.dtype != torch.int32:
        raise TypeError(f"cache_seqlens must be int32, got {cache_seqlens.dtype}")

    if int(max_pages_per_seq) <= 0:
        raise ValueError(f"max_pages_per_seq must be positive, got {max_pages_per_seq}")
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if int(topk) > int(max_seqlen):
        raise ValueError(f"topk={topk} exceeds max_seqlen={max_seqlen}")
    pages_for_max_seqlen = (int(max_seqlen) + cache_page_size - 1) // cache_page_size
    if pages_for_max_seqlen > int(max_pages_per_seq):
        raise ValueError(
            f"max_seqlen={max_seqlen} needs {pages_for_max_seqlen} pages per "
            f"sequence but the page table has only {max_pages_per_seq} columns"
        )

    return PagedMqaLogitsPlan(
        batch=batch,
        next_n=PAGED_MQA_NEXT_N,
        num_heads=num_heads,
        head_dim=head_dim,
        num_pages=num_pages,
        page_size=cache_page_size,
        max_pages_per_seq=int(max_pages_per_seq),
        topk=int(topk),
        max_seqlen=int(max_seqlen),
        max_context_len=align_context_len(int(max_seqlen)),
    )


def check_logits_layout(logits: torch.Tensor, plan: PagedMqaLogitsPlan) -> None:
    """Re-assert the row stride the top-k assumes, on the real tensor.

    DeepGEMM returns a slice of a 256-aligned allocation, so a wrong
    ``max_context_len`` would silently shift every row rather than fail. The
    top-k reads row ``b`` at ``stride(0) * b``, so a mismatch here is the one
    error that must never pass.
    """

    if tuple(logits.shape) != plan.logits_shape:
        raise RuntimeError(
            f"deep_gemm returned logits of shape {tuple(logits.shape)}, "
            f"expected {plan.logits_shape}"
        )
    if logits.dtype != torch.float32:
        raise RuntimeError(f"logits must be float32, got {logits.dtype}")
    if logits.stride(1) != 1 or logits.stride(0) != plan.max_context_len:
        raise RuntimeError(
            "logits row stride must equal its column count "
            f"({plan.max_context_len}); got strides {logits.stride()}. "
            "max_context_len was not 256-aligned."
        )


def saturate_q_to_e4m3_range(q: torch.Tensor) -> torch.Tensor:
    """Clamp ``q`` to the finite e4m3 range ``[-448, 448]`` before the cast.

    The saturation is EXPLICIT because ``Tensor.to(float8_e4m3fn)`` does not
    define it across torch versions: torch 2.9's CPU cast returns NaN for
    ``|x| > 448`` (node diagnostic, item-2a N1) where newer builds saturate.
    Clamping first makes the cast in-range on every build and matches the
    page-split K writer, whose ``.cu`` conversion is ``__NV_SATFINITE``. bf16
    represents 448 exactly, so the clamp itself is exact. NaN inputs pass
    through as NaN (a NaN q is an upstream bug, not a range question).
    """

    return q.clamp(min=-FP8_MAX, max=FP8_MAX)


def quantize_q_e4m3_reference(q: torch.Tensor) -> torch.Tensor:
    """``[batch, num_heads, head_dim]`` bf16 q -> ``[batch, 1, H, D]`` e4m3.

    A saturating cast with no scale: ``fp8_paged_mqa_logits`` reads ``q`` as
    plain e4m3 and has no q-scale channel (the per-token fp32 scales in the
    cache cover K only). In range this is byte-for-byte what DeepGEMM's own
    ``test_paged_mqa_logits`` and the item-2a node probe fed the kernel; out of
    range the saturation is explicit (``saturate_q_to_e4m3_range``), and the
    production q-cast in ``paged_mqa_scorer`` uses the same helper.

    A positive per-row rescale would be order-exact (``relu(c * x) ==
    c * relu(x)`` and the top-k is per row), so it stays available if node
    numerics ever need it; a CPU study at the production q magnitude measured
    no calc_diff improvement from it, so the plain cast is the production form.
    """

    if q.dim() != 3:
        raise ValueError(f"q must be [batch, num_heads, head_dim], got {tuple(q.shape)}")
    batch, num_heads, head_dim = q.shape
    return saturate_q_to_e4m3_range(q).to(torch.float8_e4m3fn).reshape(
        batch, PAGED_MQA_NEXT_N, num_heads, head_dim
    )


def prepare_paged_mqa_inputs_reference(
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for the two per-step inputs DeepGEMM needs.

    BatchGen's page table is indexed by cache SLOT; DeepGEMM indexes it by
    BATCH ROW and has no slot indirection, so the rows must be gathered. The
    same pass derives the 2-D ``context_lens``:

    * ``context_lens[b, 0] = cache_seqlens[b]`` for a live row and ``0`` for a
      row with no cache slot or past ``num_valid_tokens``. A zero-length row
      makes DeepGEMM read nothing and the top-k emit ``-1`` padding. These are
      the TRUE lengths; DeepGEMM itself is handed
      ``guard_deep_gemm_context_lens(context_lens)``.
    * ``block_table[b] = page_table[slot[b]]`` for a row that reads
      (``context_lens[b] > 0``); ``-1`` page entries are passed through
      unchanged. They are only reachable past a row's allocated pages, which
      DeepGEMM never visits, and if one ever appeared in range its TMA load is
      out of bounds and zero-fills, matching the ``other=0.0`` the retired
      Triton scorer produced.
    * ``block_table[b] = 0`` (every column is page 0, which the aux pool always
      owns) for a row that reads nothing. Its slot is never dereferenced -- a
      padded row past ``num_valid_tokens`` may carry any slot value -- and the
      row is DEFINED, because the guard can force DeepGEMM to read one token
      of row 0 even when row 0's true length is 0; that read must land on a
      real page, never on another sequence's stale table row or garbage.

    Returns ``(block_table [batch, max_pages] int32, context_lens [batch, 1]
    int32)``.
    """

    if page_table.dim() != 2:
        raise ValueError(f"page_table must be 2-D, got {tuple(page_table.shape)}")
    if page_table.dtype != torch.int32:
        raise TypeError(f"page_table must be int32, got {page_table.dtype}")
    if slot_indices.dim() != 1:
        raise ValueError(f"slot_indices must be 1-D, got {tuple(slot_indices.shape)}")
    if cache_seqlens.shape != slot_indices.shape:
        raise ValueError(
            "cache_seqlens and slot_indices must have the same shape, got "
            f"{tuple(cache_seqlens.shape)} and {tuple(slot_indices.shape)}"
        )
    if cache_seqlens.dtype != torch.int32:
        raise TypeError(f"cache_seqlens must be int32, got {cache_seqlens.dtype}")

    batch = int(slot_indices.shape[0])
    live = slot_indices >= 0
    if num_valid_tokens is not None:
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                "num_valid_tokens must contain one element, got "
                f"{tuple(num_valid_tokens.shape)}"
            )
        rows = torch.arange(batch, device=slot_indices.device, dtype=torch.int32)
        live = live & (rows < num_valid_tokens.reshape(()).to(torch.int32))

    true_lens = torch.where(live, cache_seqlens, torch.zeros_like(cache_seqlens))
    reads = true_lens > 0
    safe_slots = torch.where(
        reads, slot_indices, torch.zeros_like(slot_indices)
    ).to(torch.int64)
    gathered = page_table.index_select(0, safe_slots)
    block_table = torch.where(
        reads.reshape(batch, 1), gathered, torch.zeros_like(gathered)
    ).contiguous()
    context_lens = true_lens.reshape(batch, 1).contiguous()
    return block_table, context_lens


def guard_deep_gemm_context_lens(context_lens: torch.Tensor) -> torch.Tensor:
    """The ``context_lens`` DeepGEMM may be handed: row 0 forced to ``>= 1``.

    Upstream defect (sgl deep_gemm 0.1.5.post3,
    ``csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp:188``; node repro
    2026-10-17): the sm90 non-varlen ``get_paged_mqa_logits_metadata``
    scheduler binary-searches each SM's segment start over the prefix sum of
    per-row segment counts. When the total is 0 every search ends at
    ``q_idx == batch_size`` and the kernel then reads ``prefix_sum[q_idx]``;
    the shared buffer holds ``align(batch, 32)`` entries, so the read is out
    of bounds exactly when ``batch % 32 == 0`` (CUDA_ERROR_ILLEGAL_ADDRESS).
    It fires at whole-model capture warmup (all-zero inputs) and would fire at
    replay on any rank with ``num_valid_tokens == 0`` at buckets 32/64/96/128.

    Forcing row 0 to ``max(len0, 1)`` makes the total segment count >= 1, so
    every search ends inside the batch. When row 0 genuinely reads nothing its
    forced one-token read hits ``block_table[0, 0] == page 0`` (the
    preparation pass defines that row) and the result is never selected: the
    top-k masks by the TRUE ``context_lens``, not by this copy.

    Pure tensor ops, no host read: graph-capturable, and replay recomputes it
    from whatever the static ``context_lens`` holds that step. Returns a new
    tensor; the input is not modified.
    """

    if context_lens.dim() != 2 or context_lens.dtype != torch.int32:
        raise TypeError(
            "context_lens must be a 2-D [batch, next_n] int32 tensor, got "
            f"{tuple(context_lens.shape)} {context_lens.dtype}"
        )
    guarded = context_lens.clone(memory_format=torch.contiguous_format)
    guarded[:1].clamp_(min=1)
    return guarded


def topk_from_logits_reference(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The masking contract the production CUDA top-k implements.

    This is where the responsibility for masking invalid positions lives:
    DeepGEMM does not mask (``clean_logits=True`` is unreachable for the 2-D
    ``context_lens`` form), so the top-k must never look past a row's own
    ``length``. Row semantics, mirroring ``fast_topk_2048_out``:

    * row past ``num_valid_tokens`` -> all ``-1``;
    * ``length <= topk`` -> the dense prefix ``0 .. length - 1``, then ``-1``
      padding (no score is read at all);
    * otherwise -> the exact top-``topk`` over columns ``[0, length)`` only.

    Returns ``[batch, topk]`` int32.
    """

    if logits.dim() != 2:
        raise ValueError(f"logits must be [batch, N], got {tuple(logits.shape)}")
    batch = int(logits.shape[0])
    if tuple(lengths.shape) != (batch,):
        raise ValueError(
            f"lengths must have shape {(batch,)}, got {tuple(lengths.shape)}"
        )
    topk = int(topk)
    out = torch.full((batch, topk), -1, dtype=torch.int32, device=logits.device)
    valid_rows = batch
    if num_valid_tokens is not None:
        valid_rows = int(num_valid_tokens.reshape(()).item())
    for b in range(min(batch, valid_rows)):
        length = int(lengths[b].item())
        if length <= 0:
            continue
        if length <= topk:
            out[b, :length] = torch.arange(
                length, dtype=torch.int32, device=logits.device
            )
            continue
        row = logits[b, :length]
        picked = torch.topk(row, topk, dim=-1, sorted=True).indices
        out[b] = picked.to(torch.int32)
    return out
