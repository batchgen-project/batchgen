from __future__ import annotations

import pytest
import torch

from batchgen.attention.forward_metadata import (
    DecodeAttentionMetadata,
    ForwardBatchMetadata,
    KVCacheMetadata,
    PrefillAttentionMetadata,
)
from batchgen.attention.forward_metadata_context import (
    _LEGACY_ATTENTION_FIELDS,
    bind_forward_batch_metadata,
    get_current_forward_batch_metadata,
)
from batchgen.models.wrappers.attention import AttnWrapperBase


@pytest.fixture(autouse=True)
def restore_legacy_attention_fields():
    previous_values = {
        field: getattr(AttnWrapperBase, field, None)
        for field in _LEGACY_ATTENTION_FIELDS
    }
    yield
    for field, value in previous_values.items():
        setattr(AttnWrapperBase, field, value)


def _prefill_metadata(prefix_reuse: bool = True) -> ForwardBatchMetadata:
    q_seq_lens = [2, 1, 1]
    kv_seq_lens = [5, 1, 4] if prefix_reuse else list(q_seq_lens)
    cu_seqlens_k = (
        torch.tensor([0, 5, 6, 10], dtype=torch.int32)
        if prefix_reuse
        else torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    )

    return ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[11, 12, 13],
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=torch.tensor([0, 2, 3, 4], dtype=torch.int32),
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=2,
            max_seqlen_k=max(kv_seq_lens),
            q_seq_lens=q_seq_lens,
            kv_seq_lens=kv_seq_lens,
            position_ids=torch.tensor([3, 4, 0, 3], dtype=torch.int64),
        ),
        kv_cache=KVCacheMetadata(
            gpu_paged_kv_manager=object(),
            host_worker_view=object(),
            aux_gpu_paged_kv_manager=object(),
            aux_host_worker_view=object(),
        ),
    )


def _decode_metadata() -> ForwardBatchMetadata:
    return ForwardBatchMetadata(
        phase="decode",
        global_sequence_ids=[21, 22],
        decode=DecodeAttentionMetadata(
            cache_seqlens=torch.tensor([5, 7], dtype=torch.int32),
            max_seqlen=7,
            page_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            slot_indices=torch.tensor([4, 6], dtype=torch.int64),
        ),
    )


def _partial_reuse_prefill_metadata() -> ForwardBatchMetadata:
    return ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[31, 32],
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=torch.tensor([0, 2, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 5, 6], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=5,
            q_seq_lens=[2, 1],
            kv_seq_lens=[5, 1],
            position_ids=torch.tensor([3, 4, 0], dtype=torch.int64),
        ),
    )


def test_get_required_raises_when_unbound():
    assert get_current_forward_batch_metadata() is None
    with pytest.raises(RuntimeError, match="required"):
        get_current_forward_batch_metadata(required=True)


def test_bind_forward_batch_metadata_sets_and_restores_legacy_fields():
    AttnWrapperBase.phase = "decode"
    AttnWrapperBase.cur_batch = [99]
    AttnWrapperBase.prepack_mode = False
    AttnWrapperBase.prepack_cu_seqlens = None
    AttnWrapperBase.prepack_max_seqlen = None
    AttnWrapperBase.prepack_num_sequences = None
    AttnWrapperBase.prepack_seq_lengths = None
    AttnWrapperBase.prepack_prefix_reuse_mode = False
    AttnWrapperBase.prepack_prefix_shared_tokens = None
    AttnWrapperBase.prepack_full_seq_lengths = None

    metadata = _prefill_metadata()
    with bind_forward_batch_metadata(metadata) as bound:
        assert bound is metadata
        assert get_current_forward_batch_metadata(required=True) is metadata
        assert AttnWrapperBase.phase == "prefill"
        assert AttnWrapperBase.cur_batch == [11, 12, 13]
        assert AttnWrapperBase.prepack_mode is True
        assert (
            AttnWrapperBase.prepack_cu_seqlens is metadata.prefill.cu_seqlens_q
        )
        assert AttnWrapperBase.prepack_max_seqlen == 2
        assert AttnWrapperBase.prepack_num_sequences == 3
        assert AttnWrapperBase.prepack_seq_lengths == [2, 1, 1]
        assert AttnWrapperBase.prepack_prefix_reuse_mode is True
        assert AttnWrapperBase.prepack_prefix_shared_tokens == [3, 0, 3]
        assert AttnWrapperBase.prepack_full_seq_lengths == [5, 1, 4]
        assert AttnWrapperBase.position_ids is metadata.prefill.position_ids
        assert AttnWrapperBase.cache_seqlens is None
        assert AttnWrapperBase.max_seqlen is None
        assert (
            AttnWrapperBase.gpu_paged_kv_manager
            is metadata.kv_cache.gpu_paged_kv_manager
        )
        assert (
            AttnWrapperBase.host_paged_kv_worker_view
            is metadata.kv_cache.host_worker_view
        )
        assert (
            AttnWrapperBase.gpu_paged_kv_manager_aux
            is metadata.kv_cache.aux_gpu_paged_kv_manager
        )
        assert (
            AttnWrapperBase.host_paged_kv_worker_view_aux
            is metadata.kv_cache.aux_host_worker_view
        )

    assert get_current_forward_batch_metadata() is None
    assert AttnWrapperBase.phase == "decode"
    assert AttnWrapperBase.cur_batch == [99]
    assert AttnWrapperBase.prepack_mode is False
    assert AttnWrapperBase.prepack_cu_seqlens is None
    assert AttnWrapperBase.prepack_prefix_shared_tokens is None


def test_bind_forward_batch_metadata_restores_on_exception():
    AttnWrapperBase.phase = "decode"
    metadata = _prefill_metadata()

    with pytest.raises(ValueError, match="boom"):
        with bind_forward_batch_metadata(metadata):
            assert AttnWrapperBase.phase == "prefill"
            raise ValueError("boom")

    assert get_current_forward_batch_metadata() is None
    assert AttnWrapperBase.phase == "decode"


def test_bind_forward_batch_metadata_supports_nested_contexts():
    outer = _prefill_metadata(prefix_reuse=False)
    inner = _decode_metadata()

    with bind_forward_batch_metadata(outer):
        assert get_current_forward_batch_metadata(required=True) is outer
        assert AttnWrapperBase.phase == "prefill"
        assert AttnWrapperBase.prepack_prefix_reuse_mode is False
        with bind_forward_batch_metadata(inner):
            assert get_current_forward_batch_metadata(required=True) is inner
            assert AttnWrapperBase.phase == "decode"
            assert AttnWrapperBase.prepack_mode is False
            assert AttnWrapperBase.cache_seqlens is inner.decode.cache_seqlens
            assert AttnWrapperBase.max_seqlen == 7

        assert get_current_forward_batch_metadata(required=True) is outer
        assert AttnWrapperBase.phase == "prefill"
        assert AttnWrapperBase.prepack_mode is True
        assert AttnWrapperBase.cache_seqlens is None


def test_legacy_fields_do_not_leak_across_batches():
    prefix_batch = _prefill_metadata(prefix_reuse=True)
    plain_batch = _prefill_metadata(prefix_reuse=False)

    with bind_forward_batch_metadata(prefix_batch):
        assert AttnWrapperBase.prepack_prefix_reuse_mode is True
        assert AttnWrapperBase.prepack_prefix_shared_tokens == [3, 0, 3]

    with bind_forward_batch_metadata(plain_batch):
        assert AttnWrapperBase.prepack_prefix_reuse_mode is False
        assert AttnWrapperBase.prepack_prefix_shared_tokens is None
        assert AttnWrapperBase.prepack_full_seq_lengths is None


def test_prefix_cache_metadata_prefers_bound_forward_metadata():
    class WrapperWithBadLegacyState(AttnWrapperBase):
        prepack_cu_seqlens = None
        prepack_max_seqlen = None
        prepack_num_sequences = None
        prepack_seq_lengths = None
        cur_batch = None

    wrapper = object.__new__(WrapperWithBadLegacyState)
    metadata = _prefill_metadata()

    with bind_forward_batch_metadata(metadata):
        prefix_metadata = wrapper.prefix_cache_metadata()

    assert prefix_metadata.global_sequence_ids == [11, 12, 13]
    assert prefix_metadata.seq_lengths == [2, 1, 1]
    assert prefix_metadata.prefix_shared_tokens == [3, 0, 3]
    assert prefix_metadata.full_seq_lengths == [5, 1, 4]
    assert prefix_metadata.prefix_reuse_mode is True


def test_prefix_cache_metadata_rejects_bound_decode_metadata():
    wrapper = object.__new__(AttnWrapperBase)

    with bind_forward_batch_metadata(_decode_metadata()):
        with pytest.raises(RuntimeError, match="prefill metadata"):
            wrapper.prefix_cache_metadata()


def test_prefix_cache_metadata_explicit_matches_legacy_fields():
    from batchgen.models.wrappers.prefix_cache import (
        PrefixCachePrepackMetadata,
        ensure_prefix_cache_prepack_metadata,
    )

    metadata = _partial_reuse_prefill_metadata()
    wrapper_metadata = PrefixCachePrepackMetadata.from_prefill_metadata(
        metadata.prefill,
        global_sequence_ids=metadata.global_sequence_ids,
    )
    wrapper = object.__new__(AttnWrapperBase)

    with bind_forward_batch_metadata(metadata):
        explicit_metadata = wrapper.prefix_cache_metadata()

    assert (
        explicit_metadata.cu_seqlens_list()
        == wrapper_metadata.cu_seqlens_list()
    )
    assert explicit_metadata.max_seqlen == wrapper_metadata.max_seqlen
    assert explicit_metadata.num_sequences == wrapper_metadata.num_sequences
    assert explicit_metadata.seq_lengths == wrapper_metadata.seq_lengths
    assert (
        explicit_metadata.global_sequence_ids
        == wrapper_metadata.global_sequence_ids
    )
    assert (
        explicit_metadata.prefix_reuse_mode
        == wrapper_metadata.prefix_reuse_mode
    )
    assert (
        explicit_metadata.prefix_shared_tokens
        == wrapper_metadata.prefix_shared_tokens
    )
    assert (
        explicit_metadata.full_seq_lengths == wrapper_metadata.full_seq_lengths
    )
    assert (
        ensure_prefix_cache_prepack_metadata(metadata).global_sequence_ids
        == metadata.global_sequence_ids
    )
    with pytest.raises(RuntimeError, match="global sequence ids"):
        ensure_prefix_cache_prepack_metadata(metadata.prefill)
