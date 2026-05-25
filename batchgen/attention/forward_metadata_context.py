"""Context binding for first-class attention forward metadata.

This module is the compatibility bridge between explicit
``ForwardBatchMetadata`` and the legacy ``AttnWrapperBase`` class variables.
The metadata object remains the source of truth; legacy fields are only
populated for the dynamic extent of a single forward call.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from batchgen.attention.forward_metadata import (
    DecodeAttentionMetadata,
    ForwardBatchMetadata,
    KVCacheMetadata,
    PrefillAttentionMetadata,
)


_CURRENT_FORWARD_BATCH_METADATA: ContextVar[Optional[ForwardBatchMetadata]] = (
    ContextVar("current_forward_batch_metadata", default=None)
)

_LEGACY_ATTENTION_FIELDS = (
    "phase",
    "cur_batch",
    "position_ids",
    "prepack_mode",
    "prepack_cu_seqlens",
    "prepack_max_seqlen",
    "prepack_num_sequences",
    "prepack_seq_lengths",
    "prepack_prefix_reuse_mode",
    "prepack_prefix_shared_tokens",
    "prepack_full_seq_lengths",
    "prepack_full_hit_mode",
    "cache_seqlens",
    "max_seqlen",
    "gpu_paged_kv_manager",
    "host_paged_kv_worker_view",
    "prefill_prefix_materialization",
    "gpu_paged_kv_manager_aux",
    "host_paged_kv_worker_view_aux",
)


def get_current_forward_batch_metadata(
    required: bool = False,
) -> Optional[ForwardBatchMetadata]:
    """Return the metadata bound to the current execution context."""

    metadata = _CURRENT_FORWARD_BATCH_METADATA.get()
    if metadata is None and required:
        raise RuntimeError("ForwardBatchMetadata is required but is not bound")
    return metadata


@contextmanager
def bind_forward_batch_metadata(
    metadata: ForwardBatchMetadata,
) -> Iterator[ForwardBatchMetadata]:
    """Bind metadata for one forward and mirror it into legacy wrapper fields."""

    if not isinstance(metadata, ForwardBatchMetadata):
        raise TypeError("metadata must be a ForwardBatchMetadata instance")

    # Import lazily so metadata users can be unit-tested without importing model
    # wrappers unless the compatibility bridge is actually used.
    from batchgen.models.wrappers.attention import AttnWrapperBase

    previous_values = {
        field: getattr(AttnWrapperBase, field, None)
        for field in _LEGACY_ATTENTION_FIELDS
    }
    token = _CURRENT_FORWARD_BATCH_METADATA.set(metadata)
    try:
        _sync_legacy_attention_wrapper(AttnWrapperBase, metadata)
        yield metadata
    finally:
        _CURRENT_FORWARD_BATCH_METADATA.reset(token)
        for field, value in previous_values.items():
            setattr(AttnWrapperBase, field, value)


def _sync_legacy_attention_wrapper(
    wrapper_cls: type,
    metadata: ForwardBatchMetadata,
) -> None:
    wrapper_cls.phase = metadata.phase
    wrapper_cls.cur_batch = list(metadata.global_sequence_ids)

    if metadata.phase == "prefill":
        assert metadata.prefill is not None
        _sync_prefill_fields(wrapper_cls, metadata.prefill)
    else:
        assert metadata.decode is not None
        _sync_decode_fields(wrapper_cls, metadata.decode)

    if metadata.kv_cache is not None:
        _sync_kv_cache_fields(wrapper_cls, metadata.kv_cache)


def _sync_prefill_fields(
    wrapper_cls: type,
    prefill: PrefillAttentionMetadata,
) -> None:
    wrapper_cls.position_ids = prefill.position_ids
    wrapper_cls.prepack_mode = True
    wrapper_cls.prepack_cu_seqlens = prefill.cu_seqlens_q
    wrapper_cls.prepack_max_seqlen = int(prefill.max_seqlen_q)
    wrapper_cls.prepack_num_sequences = prefill.batch_size
    wrapper_cls.prepack_seq_lengths = list(prefill.q_seq_lens)
    wrapper_cls.cache_seqlens = None
    wrapper_cls.max_seqlen = None

    if prefill.prefix_reuse is None:
        wrapper_cls.prepack_prefix_reuse_mode = False
        wrapper_cls.prepack_prefix_shared_tokens = None
        wrapper_cls.prepack_full_seq_lengths = None
        wrapper_cls.prepack_full_hit_mode = False
        return

    _sync_prefix_reuse_fields(wrapper_cls, prefill)


def _sync_prefix_reuse_fields(
    wrapper_cls: type,
    prefill: PrefillAttentionMetadata,
) -> None:
    prefix_lens = [
        int(kv_len) - int(q_len)
        for q_len, kv_len in zip(prefill.q_seq_lens, prefill.kv_seq_lens)
    ]
    full_seq_lens = [int(length) for length in prefill.kv_seq_lens]
    wrapper_cls.prepack_prefix_reuse_mode = any(
        length > 0 for length in prefix_lens
    )
    wrapper_cls.prepack_prefix_shared_tokens = prefix_lens
    wrapper_cls.prepack_full_seq_lengths = full_seq_lens
    wrapper_cls.prepack_full_hit_mode = False


def _sync_decode_fields(
    wrapper_cls: type, decode: DecodeAttentionMetadata
) -> None:
    wrapper_cls.position_ids = None
    wrapper_cls.prepack_mode = False
    wrapper_cls.prepack_cu_seqlens = None
    wrapper_cls.prepack_max_seqlen = None
    wrapper_cls.prepack_num_sequences = None
    wrapper_cls.prepack_seq_lengths = None
    wrapper_cls.prepack_prefix_reuse_mode = False
    wrapper_cls.prepack_prefix_shared_tokens = None
    wrapper_cls.prepack_full_seq_lengths = None
    wrapper_cls.prepack_full_hit_mode = False
    wrapper_cls.cache_seqlens = decode.cache_seqlens
    wrapper_cls.max_seqlen = int(decode.max_seqlen)


def _sync_kv_cache_fields(wrapper_cls: type, kv_cache: KVCacheMetadata) -> None:
    wrapper_cls.gpu_paged_kv_manager = kv_cache.gpu_paged_kv_manager
    wrapper_cls.host_paged_kv_worker_view = kv_cache.host_worker_view
    wrapper_cls.prefill_prefix_materialization = (
        kv_cache.prefill_prefix_materialization
    )
    wrapper_cls.gpu_paged_kv_manager_aux = kv_cache.aux_gpu_paged_kv_manager
    wrapper_cls.host_paged_kv_worker_view_aux = kv_cache.aux_host_worker_view
