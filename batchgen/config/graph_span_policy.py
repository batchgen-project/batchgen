"""Captured context span policy for whole-model decode graphs.

The span (``graph_max_seqlen``, logged as ``context_capacity``) is derived in
two places that must agree: the worker, which captures the graph, and
``triton_prewarm``, which JIT-compiles the span-dependent kernels BEFORE that
capture so the compile cannot happen inside it. The policy therefore lives
here, in a module with no torch, Triton or DeepGEMM import, so both can use it
and the CPU tests can exercise it directly.

The span itself is the DEPLOYMENT context bound -- what admission clamps every
sequence to -- and never a property of whichever decode batch arrives first.
This module owns only the one term that is a resource decision: how much span
the graph's context-proportional statics can afford.
"""

from __future__ import annotations

from typing import Sequence


__all__ = [
    "GRAPH_CONTEXT_STATICS_BUDGET_BYTES",
    "GRAPH_CONTEXT_STATICS_BYTES_PER_TOKEN_PER_ROW",
    "GRAPH_CONTEXT_SPAN_ALIGN",
    "GRAPH_CONTEXT_SPAN_FLOOR_TOKENS",
    "graph_context_span_cap_tokens",
]


# Budget for the decode graph's CONTEXT-PROPORTIONAL statics, i.e. the ones
# whose bytes grow with the span rather than only with the bucket. There are
# two cost classes:
#
#   4 B per PAGE per row -- the primary/aux graph page tables
#     (2 * max_slots * span/page_size * 4 B) and, on the page-split scorer,
#     `aux_block_table` (sum(buckets) * span/aux_page_size * 4 B). At a 1M span
#     and 64-token pages that is tens to a couple of hundred MB, depending on
#     how many slots the planner asks for.
#   4 B per TOKEN per row -- the DSA scorer's fp32 logits, 64x the above at a
#     64-token page, which is what actually bounds an affordable span:
#       page-split (DeepGEMM): the block comes from the graph-private pool and
#         is freed after each layer's top-k, so ONE top-bucket block covers
#         every indexer layer AND every bucket -> 4 B * max_bucket * span
#       bf16 AoS (Triton): `agg_scores` is a per-bucket static, so the rows add
#         up across the bucket set            -> 4 B * sum(buckets) * span
#
# Only the second class is charged against this budget. The capture runs before
# `_finalize_gpu_kv_pool_after_capture`, so these bytes are already accounted
# for when the KV pool is sized -- a wider span is paid for in resident KV
# capacity, not in an OOM.
#
# Measured against this budget (span 1,048,576, 64-token pages, 9 buckets):
#   top bucket 128, page-split : 512 MiB of logits, cap = 1,048,576 (no trim)
#   top bucket 256, page-split : cap = 524,288 (the trim engages)
#   top bucket 128, bf16 AoS   : rows sum to 276, so 1104 MiB unbudgeted and
#                                cap = 486,144
# A trimmed span is not a correctness limit: a step past it runs eager decode.
# DIAGNOSTIC (span A/B): clamps the captured span to 131,072 tokens at top
# bucket 128 to isolate the DeepGEMM logits-extent cost; never merge.
GRAPH_CONTEXT_STATICS_BUDGET_BYTES = 64 * 1024 * 1024
# The scorer logits are fp32.
GRAPH_CONTEXT_STATICS_BYTES_PER_TOKEN_PER_ROW = 4
# DeepGEMM's logits row stride is 256-aligned (`align_context_len`), so a span
# that is not a multiple of 256 buys nothing.
GRAPH_CONTEXT_SPAN_ALIGN = 256
# A capture narrower than this serves so few steps that eager decode is the
# better path; the budget never trims below it.
GRAPH_CONTEXT_SPAN_FLOOR_TOKENS = 16384


def graph_context_span_cap_tokens(
    *,
    bucket_sizes: Sequence[int],
    aux_is_page_split: bool,
    budget_bytes: int = GRAPH_CONTEXT_STATICS_BUDGET_BYTES,
) -> int:
    """Largest captured span whose context-proportional statics fit a budget.

    See `GRAPH_CONTEXT_STATICS_BUDGET_BYTES` for the two row counts: the
    page-split scorer reuses one graph-pool block across layers and buckets, so
    its rows are the top bucket; the bf16 AoS scorer holds a per-bucket
    `agg_scores` static, so its rows add up over the bucket set.
    """
    buckets = [int(b) for b in bucket_sizes if int(b) > 0]
    if not buckets:
        raise ValueError(
            "the decode graph span cap requires at least one positive bucket size"
        )
    rows = max(buckets) if aux_is_page_split else sum(buckets)
    bytes_per_token = rows * GRAPH_CONTEXT_STATICS_BYTES_PER_TOKEN_PER_ROW
    span = int(budget_bytes) // bytes_per_token
    span -= span % GRAPH_CONTEXT_SPAN_ALIGN
    return max(span, GRAPH_CONTEXT_SPAN_FLOOR_TOKENS)
