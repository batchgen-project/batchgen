import pytest
import torch

from batchgen.attention.forward_metadata import (
    DecodeAttentionMetadata,
    ForwardBatchMetadata,
    KVCacheMetadata,
    PrefillAttentionMetadata,
    PrefixReuseMetadata,
)


def _prefix_reuse_metadata(prefix_lens):
    prefix = torch.tensor(prefix_lens, dtype=torch.int32)
    suffix = torch.tensor([2, 4, 0], dtype=torch.int32)
    full = prefix + suffix
    return PrefixReuseMetadata(
        prefix_lens=prefix,
        suffix_lens=suffix,
        full_seq_lens=full,
        saved_tokens=int(prefix.sum().item()),
        is_full_hit=suffix == 0,
        global_sequence_ids=[100, 101, 102],
    )


def test_prefill_metadata_validates_no_reuse():
    metadata = PrefillAttentionMetadata(
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3, 7], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_k=4,
        q_seq_lens=[3, 4],
        kv_seq_lens=[3, 4],
        position_ids=torch.arange(7, dtype=torch.int64),
    )
    batch = ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[10, 11],
        prefill=metadata,
        kv_cache=KVCacheMetadata(),
    )

    batch.validate()


def test_prefill_metadata_validates_partial_hit_miss_and_full_hit():
    prefix = _prefix_reuse_metadata([4, 0, 5])
    metadata = PrefillAttentionMetadata(
        cu_seqlens_q=torch.tensor([0, 2, 6, 6], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 6, 10, 15], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_k=6,
        q_seq_lens=[2, 4, 0],
        kv_seq_lens=[6, 4, 5],
        position_ids=torch.tensor([4, 5, 0, 1, 2, 3], dtype=torch.int64),
        prefix_reuse=prefix,
    )
    batch = ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[100, 101, 102],
        prefill=metadata,
    )

    batch.validate()


def test_prefix_reuse_metadata_rejects_inconsistent_full_length():
    metadata = PrefixReuseMetadata(
        prefix_lens=torch.tensor([2], dtype=torch.int32),
        suffix_lens=torch.tensor([3], dtype=torch.int32),
        full_seq_lens=torch.tensor([4], dtype=torch.int32),
        saved_tokens=2,
        is_full_hit=torch.tensor([False]),
        global_sequence_ids=[1],
    )

    with pytest.raises(ValueError, match="prefix_lens \\+ suffix_lens"):
        metadata.validate()


def test_prefix_reuse_metadata_rejects_full_hit_with_suffix():
    metadata = PrefixReuseMetadata(
        prefix_lens=torch.tensor([2], dtype=torch.int32),
        suffix_lens=torch.tensor([1], dtype=torch.int32),
        full_seq_lens=torch.tensor([3], dtype=torch.int32),
        saved_tokens=2,
        is_full_hit=torch.tensor([True]),
        global_sequence_ids=[1],
    )

    with pytest.raises(ValueError, match="full-hit sequence"):
        metadata.validate()


def test_prefill_metadata_rejects_bad_cu_seqlens():
    metadata = PrefillAttentionMetadata(
        cu_seqlens_q=torch.tensor([0, 2, 7], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3, 7], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_k=4,
        q_seq_lens=[3, 4],
        kv_seq_lens=[3, 4],
        position_ids=torch.arange(7, dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="cu_seqlens_q"):
        metadata.validate()


def test_prefill_metadata_rejects_query_longer_than_kv():
    metadata = PrefillAttentionMetadata(
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 4], dtype=torch.int32),
        max_seqlen_q=5,
        max_seqlen_k=4,
        q_seq_lens=[5],
        kv_seq_lens=[4],
        position_ids=torch.arange(5, dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="q_seq_lens cannot exceed"):
        metadata.validate()


def test_decode_metadata_validates_page_table_and_slots():
    metadata = DecodeAttentionMetadata(
        cache_seqlens=torch.tensor([5, 7], dtype=torch.int32),
        max_seqlen=7,
        page_table=torch.zeros((2, 2), dtype=torch.int32),
        slot_indices=torch.tensor([4, 6], dtype=torch.int64),
    )
    batch = ForwardBatchMetadata(
        phase="decode",
        global_sequence_ids=[10, 11],
        decode=metadata,
    )

    batch.validate()


def test_forward_metadata_rejects_phase_mismatch():
    batch = ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[1],
        decode=DecodeAttentionMetadata(
            cache_seqlens=torch.tensor([1], dtype=torch.int32),
            max_seqlen=1,
        ),
    )

    with pytest.raises(ValueError, match="prefill metadata is required"):
        batch.validate()

