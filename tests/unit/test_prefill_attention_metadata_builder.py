from __future__ import annotations

import pytest
import torch

from batchgen.batch_order import PrefillSequenceSpan
from batchgen.prefill.attention_metadata_builder import (
    build_prefill_forward_metadata,
)
from batchgen.prefill.prepack import PrepackMetadata, prepack_sequences
from batchgen.prefill.prefix_reuse import (
    PrefixReusePrefillPlan,
    PrefixReuseSequencePlan,
)


def _span(
    row_index: int, global_seq_id: int, seq_len: int
) -> PrefillSequenceSpan:
    return PrefillSequenceSpan(
        row_index=row_index,
        local_idx=10 + row_index,
        uuid=f"uuid-{row_index}",
        global_seq_id=global_seq_id,
        seq_len=seq_len,
        start=0,
        end=seq_len,
    )


def _spans(
    global_ids: list[int], seq_lens: list[int]
) -> list[PrefillSequenceSpan]:
    cursor = 0
    spans = []
    for row_index, (global_seq_id, seq_len) in enumerate(
        zip(global_ids, seq_lens)
    ):
        spans.append(
            PrefillSequenceSpan(
                row_index=row_index,
                local_idx=10 + row_index,
                uuid=f"uuid-{row_index}",
                global_seq_id=global_seq_id,
                seq_len=seq_len,
                start=cursor,
                end=cursor + seq_len,
            )
        )
        cursor += seq_len
    return spans


def _prepack_metadata(seq_lens: list[int]) -> PrepackMetadata:
    return PrepackMetadata(
        packed_input_ids=torch.empty((0,), dtype=torch.long),
        packed_attention_mask=torch.empty((0,), dtype=torch.long),
        packed_position_ids=torch.empty((0,), dtype=torch.long),
        sequence_ids=torch.empty((0,), dtype=torch.long),
        cu_seqlens_per_row=[],
        max_seqlen_per_row=[],
        original_seq_lengths=seq_lens,
        num_original_sequences=len(seq_lens),
        num_packed_rows=0,
        row_length=max(seq_lens, default=0),
        pack_assignment=[],
    )


def _prefix_plan(
    global_ids: list[int],
    prefix_lens: list[int],
    suffix_lens: list[int],
    raw_prefix_lens: list[int] | None = None,
) -> PrefixReusePrefillPlan:
    sequences = []
    suffix_input_ids = []
    suffix_position_ids = []
    if raw_prefix_lens is None:
        raw_prefix_lens = list(prefix_lens)
    for local_idx, (
        global_id,
        prefix_len,
        suffix_len,
        raw_prefix_len,
    ) in enumerate(zip(global_ids, prefix_lens, suffix_lens, raw_prefix_lens)):
        prompt_length = prefix_len + suffix_len
        sequences.append(
            PrefixReuseSequencePlan(
                local_idx=local_idx,
                sequence_id=global_id,
                prompt_length=prompt_length,
                raw_prefix_shared_tokens=raw_prefix_len,
                prefix_shared_tokens=prefix_len,
                suffix_start_pos=prefix_len,
                suffix_length=suffix_len,
                full_logical_context_length=prompt_length,
                is_full_hit=(raw_prefix_len == prompt_length),
            )
        )
        suffix_input_ids.append(torch.arange(suffix_len, dtype=torch.long))
        suffix_position_ids.append(
            torch.arange(prefix_len, prompt_length, dtype=torch.long)
        )

    return PrefixReusePrefillPlan(
        sequences=sequences,
        suffix_input_ids=suffix_input_ids,
        suffix_position_ids=suffix_position_ids,
        cache_seqlens=torch.tensor(prefix_lens, dtype=torch.int32),
        total_prompt_tokens=sum(
            prefix + suffix for prefix, suffix in zip(prefix_lens, suffix_lens)
        ),
        total_suffix_tokens=sum(suffix_lens),
        saved_prefill_tokens=sum(prefix_lens),
    )


def test_build_prefill_forward_metadata_without_prefix_reuse():
    prepack = prepack_sequences(
        [
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[4, 5]], dtype=torch.long),
        ],
        [
            torch.tensor([[1, 1, 1]], dtype=torch.long),
            torch.tensor([[1, 1]], dtype=torch.long),
        ],
        device=torch.device("cpu"),
    )

    metadata = build_prefill_forward_metadata(
        prepack_metadata=prepack,
        batch_spans=_spans([100, 101], [3, 2]),
        seq_start=0,
        seq_end=2,
        position_ids=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert metadata.phase == "prefill"
    assert metadata.global_sequence_ids == [100, 101]
    assert metadata.prefill.q_seq_lens == [3, 2]
    assert metadata.prefill.kv_seq_lens == [3, 2]
    assert metadata.prefill.cu_seqlens_q.tolist() == [0, 3, 5]
    assert metadata.prefill.cu_seqlens_k.tolist() == [0, 3, 5]
    assert metadata.prefill.prefix_reuse is None


def test_build_prefill_forward_metadata_with_prefix_reuse_slice():
    prepack = _prepack_metadata([99, 2, 1, 88])
    plan = _prefix_plan(
        global_ids=[90, 100, 101, 91],
        prefix_lens=[0, 3, 0, 0],
        suffix_lens=[99, 2, 1, 88],
    )

    metadata = build_prefill_forward_metadata(
        prepack_metadata=prepack,
        batch_spans=_spans([100, 101], [2, 1]),
        seq_start=1,
        seq_end=3,
        position_ids=torch.tensor([3, 4, 0], dtype=torch.long),
        device=torch.device("cpu"),
        prefix_reuse_plan=plan,
    )

    prefix_reuse = metadata.prefill.prefix_reuse
    assert metadata.prefill.q_seq_lens == [2, 1]
    assert metadata.prefill.kv_seq_lens == [5, 1]
    assert metadata.prefill.cu_seqlens_q.tolist() == [0, 2, 3]
    assert metadata.prefill.cu_seqlens_k.tolist() == [0, 5, 6]
    assert prefix_reuse.prefix_lens.tolist() == [3, 0]
    assert prefix_reuse.suffix_lens.tolist() == [2, 1]
    assert prefix_reuse.full_seq_lens.tolist() == [5, 1]
    assert prefix_reuse.saved_tokens == 3
    assert prefix_reuse.is_full_hit.tolist() == [False, False]


def test_build_prefill_forward_metadata_with_mixed_hit_miss_and_full_hit():
    prepack = _prepack_metadata([2, 1, 1])
    plan = _prefix_plan(
        global_ids=[100, 101, 102],
        prefix_lens=[3, 0, 3],
        suffix_lens=[2, 1, 1],
        raw_prefix_lens=[3, 0, 4],
    )

    metadata = build_prefill_forward_metadata(
        prepack_metadata=prepack,
        batch_spans=_spans([100, 101, 102], [2, 1, 1]),
        seq_start=0,
        seq_end=3,
        position_ids=torch.tensor([3, 4, 0, 3], dtype=torch.long),
        device=torch.device("cpu"),
        prefix_reuse_plan=plan,
    )

    prefix_reuse = metadata.prefill.prefix_reuse
    assert metadata.prefill.q_seq_lens == [2, 1, 1]
    assert metadata.prefill.kv_seq_lens == [5, 1, 4]
    assert metadata.prefill.cu_seqlens_q.tolist() == [0, 2, 3, 4]
    assert metadata.prefill.cu_seqlens_k.tolist() == [0, 5, 6, 10]
    assert prefix_reuse.prefix_lens.tolist() == [3, 0, 3]
    assert prefix_reuse.suffix_lens.tolist() == [2, 1, 1]
    assert prefix_reuse.is_full_hit.tolist() == [False, False, True]


def test_build_prefill_forward_metadata_rejects_suffix_length_mismatch():
    prepack = _prepack_metadata([3])
    plan = _prefix_plan(global_ids=[100], prefix_lens=[2], suffix_lens=[1])

    with pytest.raises(ValueError, match="suffix lengths"):
        build_prefill_forward_metadata(
            prepack_metadata=prepack,
            batch_spans=[_span(0, 100, 3)],
            seq_start=0,
            seq_end=1,
            position_ids=torch.tensor([2, 3, 4], dtype=torch.long),
            device=torch.device("cpu"),
            prefix_reuse_plan=plan,
        )


def test_build_prefill_forward_metadata_rejects_sequence_id_mismatch():
    prepack = _prepack_metadata([1])
    plan = _prefix_plan(global_ids=[200], prefix_lens=[0], suffix_lens=[1])

    with pytest.raises(ValueError, match="sequence ids"):
        build_prefill_forward_metadata(
            prepack_metadata=prepack,
            batch_spans=[_span(0, 100, 1)],
            seq_start=0,
            seq_end=1,
            position_ids=torch.tensor([0], dtype=torch.long),
            device=torch.device("cpu"),
            prefix_reuse_plan=plan,
        )
