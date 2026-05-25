"""Builders for prefill attention forward metadata."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from batchgen.attention.forward_metadata import (
    ForwardBatchMetadata,
    KVCacheMetadata,
    PrefillAttentionMetadata,
)
from batchgen.batch_order import PrefillSequenceSpan
from batchgen.prefill.prepack import PrepackMetadata
from batchgen.prefill.prefix_reuse import PrefixReusePrefillPlan


def build_prefill_forward_metadata(
    *,
    prepack_metadata: PrepackMetadata,
    batch_spans: Sequence[PrefillSequenceSpan],
    seq_start: int,
    seq_end: int,
    position_ids: torch.Tensor,
    device: torch.device,
    prefix_reuse_plan: Optional[PrefixReusePrefillPlan] = None,
    kv_cache_metadata: Optional[KVCacheMetadata] = None,
) -> ForwardBatchMetadata:
    """Build first-class metadata for one prepacked prefill micro-batch."""

    if seq_start < 0 or seq_end < seq_start:
        raise ValueError(f"Invalid sequence range [{seq_start}, {seq_end})")
    q_seq_lens = [
        int(length)
        for length in prepack_metadata.original_seq_lengths[seq_start:seq_end]
    ]
    if len(q_seq_lens) != len(batch_spans):
        raise ValueError(
            f"batch_spans length must match micro-batch sequence count: "
            f"{len(batch_spans)} != {len(q_seq_lens)}"
        )
    span_seq_lens = [int(span.seq_len) for span in batch_spans]
    if span_seq_lens != q_seq_lens:
        raise ValueError(
            f"batch span sequence lengths do not match prepack lengths: "
            f"{span_seq_lens} != {q_seq_lens}"
        )

    global_sequence_ids = [int(span.global_seq_id) for span in batch_spans]
    total_query_tokens = sum(q_seq_lens)
    if position_ids.numel() != total_query_tokens:
        raise ValueError(
            f"position_ids length must match micro-batch query tokens: "
            f"{position_ids.numel()} != {total_query_tokens}"
        )
    position_ids = position_ids.to(device=device)
    cu_seqlens_q = _build_cu_seqlens(q_seq_lens, device=device)

    if prefix_reuse_plan is None:
        kv_seq_lens = list(q_seq_lens)
    else:
        kv_seq_lens = _build_prefix_reuse_kv_seq_lens(
            plan=prefix_reuse_plan,
            seq_start=seq_start,
            seq_end=seq_end,
            q_seq_lens=q_seq_lens,
            global_sequence_ids=global_sequence_ids,
        )
    cu_seqlens_k = _build_cu_seqlens(kv_seq_lens, device=device)

    return ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=global_sequence_ids,
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(q_seq_lens, default=0),
            max_seqlen_k=max(kv_seq_lens, default=0),
            q_seq_lens=q_seq_lens,
            kv_seq_lens=kv_seq_lens,
            position_ids=position_ids,
        ),
        kv_cache=kv_cache_metadata,
    )


def _build_prefix_reuse_kv_seq_lens(
    *,
    plan: PrefixReusePrefillPlan,
    seq_start: int,
    seq_end: int,
    q_seq_lens: Sequence[int],
    global_sequence_ids: Sequence[int],
) -> list[int]:
    sequence_plans = plan.sequences[seq_start:seq_end]
    if len(sequence_plans) != len(q_seq_lens):
        raise ValueError(
            f"prefix reuse plan slice length mismatch: "
            f"{len(sequence_plans)} != {len(q_seq_lens)}"
        )

    suffix_lens: list[int] = []
    kv_seq_lens: list[int] = []
    plan_sequence_ids: list[int] = []
    for item in sequence_plans:
        suffix_lens.append(int(item.suffix_length))
        kv_seq_lens.append(int(item.full_logical_context_length))
        plan_sequence_ids.append(int(item.sequence_id))

    if suffix_lens != [int(length) for length in q_seq_lens]:
        raise ValueError(
            f"prefix reuse suffix lengths do not match query lengths: "
            f"{suffix_lens} != {list(q_seq_lens)}"
        )
    if plan_sequence_ids != [int(seq_id) for seq_id in global_sequence_ids]:
        raise ValueError(
            f"prefix reuse sequence ids do not match batch spans: "
            f"{plan_sequence_ids} != {list(global_sequence_ids)}"
        )
    return kv_seq_lens


def _build_cu_seqlens(
    seq_lens: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    values = [0]
    running = 0
    for length in seq_lens:
        running += int(length)
        values.append(running)
    return torch.tensor(values, dtype=torch.int32, device=device)
