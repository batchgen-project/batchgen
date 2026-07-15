"""Prefix-cache metadata compatibility helpers.

The source of truth for prefill execution metadata is
``ForwardBatchMetadata``. This module only provides the legacy conversion path
for wrappers that still receive state through ``AttnWrapperBase`` class fields.
"""

from __future__ import annotations

from typing import Sequence

import torch

from batchgen.attention.forward_metadata import (
    ForwardBatchMetadata,
    PrefillAttentionMetadata,
)
from batchgen.attention.forward_metadata_context import (
    get_current_forward_batch_metadata,
)


def ensure_prefix_cache_forward_metadata(metadata) -> ForwardBatchMetadata:
    """Return validated prefill ``ForwardBatchMetadata``.

    Prefix-cache-aware compute/offload paths should consume
    ``ForwardBatchMetadata`` directly. Passing ``PrefillAttentionMetadata`` is
    intentionally rejected because it lacks global sequence ids.
    """

    if isinstance(metadata, ForwardBatchMetadata):
        _require_prefill(metadata)
        _validate_forward_metadata(metadata)
        return metadata
    if isinstance(metadata, PrefillAttentionMetadata):
        raise RuntimeError(
            "PrefillAttentionMetadata does not carry global sequence ids; "
            "pass ForwardBatchMetadata or use AttnWrapperBase-bound fields"
        )
    raise TypeError("metadata must be ForwardBatchMetadata")


def current_or_legacy_prefix_cache_metadata(
    wrapper_cls: type,
) -> ForwardBatchMetadata:
    """Prefer bound metadata, otherwise build it from legacy wrapper fields."""

    metadata = get_current_forward_batch_metadata()
    if metadata is not None:
        return ensure_prefix_cache_forward_metadata(metadata)
    return build_prefix_cache_forward_metadata_from_wrapper_cls(wrapper_cls)


def build_prefix_cache_forward_metadata_from_wrapper_cls(
    wrapper_cls: type,
) -> ForwardBatchMetadata:
    """Build ``ForwardBatchMetadata`` from legacy prepack class variables."""

    if getattr(wrapper_cls, "phase", None) not in (None, "prefill"):
        raise RuntimeError("Prefix cache prepack metadata requires prefill metadata")

    cu_seqlens = _require_attr(wrapper_cls, "prepack_cu_seqlens")
    max_seqlen = int(_require_attr(wrapper_cls, "prepack_max_seqlen"))
    num_sequences = int(_require_attr(wrapper_cls, "prepack_num_sequences"))
    seq_lengths = _int_list(_require_attr(wrapper_cls, "prepack_seq_lengths"))
    global_sequence_ids = _int_list(_require_attr(wrapper_cls, "cur_batch"))
    append_seq_lengths = _optional_int_list(
        getattr(wrapper_cls, "prepack_append_seq_lengths", None)
    )
    if append_seq_lengths is None:
        append_seq_lengths = list(seq_lengths)

    if len(seq_lengths) != num_sequences:
        raise RuntimeError(
            "Prefix cache seq_lengths length does not match num_sequences: "
            f"{len(seq_lengths)} != {num_sequences}"
        )
    if len(global_sequence_ids) != num_sequences:
        raise RuntimeError(
            "Prefix cache cur_batch length does not match num_sequences: "
            f"{len(global_sequence_ids)} != {num_sequences}"
        )
    if len(append_seq_lengths) != num_sequences:
        raise RuntimeError(
            "Prefix cache append_seq_lengths length does not match "
            f"num_sequences: {len(append_seq_lengths)} != {num_sequences}"
        )
    _validate_append_lengths(
        append_seq_lengths=append_seq_lengths,
        query_seq_lengths=seq_lengths,
    )

    prefix_reuse_mode = bool(
        getattr(wrapper_cls, "prepack_prefix_reuse_mode", False)
    )
    if prefix_reuse_mode:
        prefix_shared_tokens = _int_list(
            _require_attr(wrapper_cls, "prepack_prefix_shared_tokens")
        )
        full_seq_lengths = _int_list(
            _require_attr(wrapper_cls, "prepack_full_seq_lengths")
        )
        if len(prefix_shared_tokens) != num_sequences:
            raise RuntimeError(
                "Prefix shared token count length does not match batch: "
                f"{len(prefix_shared_tokens)} != {num_sequences}"
            )
        if len(full_seq_lengths) != num_sequences:
            raise RuntimeError(
                "Full sequence length metadata length does not match batch: "
                f"{len(full_seq_lengths)} != {num_sequences}"
            )
        for idx, (append_len, prefix_tokens, full_length) in enumerate(
            zip(append_seq_lengths, prefix_shared_tokens, full_seq_lengths)
        ):
            expected = int(append_len) + int(prefix_tokens)
            if expected != int(full_length):
                raise RuntimeError(
                    "Prefix cache full length mismatch at sequence "
                    f"{idx}: append={append_len}, prefix={prefix_tokens}, "
                    f"full={full_length}"
                )
        kv_seq_lengths = full_seq_lengths
    else:
        kv_seq_lengths = list(seq_lengths)

    position_ids = getattr(wrapper_cls, "position_ids", None)
    return ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=global_sequence_ids,
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=_build_cu_seqlens_like(
                kv_seq_lengths,
                reference=cu_seqlens,
            ),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max(kv_seq_lengths, default=0),
            q_seq_lens=seq_lengths,
            kv_seq_lens=kv_seq_lengths,
            position_ids=position_ids,
            append_seq_lens=append_seq_lengths,
        ),
    )


def _require_prefill(metadata: ForwardBatchMetadata) -> None:
    if metadata.phase != "prefill" or metadata.prefill is None:
        raise RuntimeError("Prefix cache prepack metadata requires prefill metadata")


def _validate_forward_metadata(metadata: ForwardBatchMetadata) -> None:
    prefill = metadata.require_prefill()
    num_sequences = int(prefill.batch_size)
    if len(metadata.global_sequence_ids) != num_sequences:
        raise RuntimeError(
            "Prefix cache global sequence id count does not match batch: "
            f"{len(metadata.global_sequence_ids)} != {num_sequences}"
        )
    if len(prefill.kv_seq_lens) != num_sequences:
        raise RuntimeError(
            "Prefix cache metadata KV length count does not match batch: "
            f"{len(prefill.kv_seq_lens)} != {num_sequences}"
        )
    if len(metadata.append_seq_lengths) != num_sequences:
        raise RuntimeError(
            "Prefix cache append length count does not match batch: "
            f"{len(metadata.append_seq_lengths)} != {num_sequences}"
        )
    if len(prefill.cu_seqlens_q) != num_sequences + 1:
        raise RuntimeError(
            "Prefix cache cu_seqlens length does not match batch: "
            f"{len(prefill.cu_seqlens_q)} != {num_sequences + 1}"
        )
    _validate_append_lengths(
        append_seq_lengths=metadata.append_seq_lengths,
        query_seq_lengths=prefill.q_seq_lens,
    )
    prefix_tokens = [
        int(kv_len) - int(append_len)
        for kv_len, append_len in zip(
            prefill.kv_seq_lens,
            metadata.append_seq_lengths,
        )
    ]
    if any(tokens < 0 for tokens in prefix_tokens):
        raise RuntimeError("Prefix cache metadata requires kv lengths >= append lengths")


def _require_attr(wrapper_cls: type, name: str):
    value = getattr(wrapper_cls, name, None)
    if value is None:
        raise RuntimeError(f"Prefix cache prepack metadata requires {name}")
    return value


def _int_list(values: Sequence[int]) -> list[int]:
    return [int(value) for value in values]


def _optional_int_list(values: Sequence[int] | None) -> list[int] | None:
    if values is None:
        return None
    return _int_list(values)


def _validate_append_lengths(
    *,
    append_seq_lengths: Sequence[int],
    query_seq_lengths: Sequence[int],
) -> None:
    for idx, (append_len, query_len) in enumerate(
        zip(append_seq_lengths, query_seq_lengths)
    ):
        if int(append_len) < 0 or int(append_len) > int(query_len):
            raise RuntimeError(
                "Prefix cache append length must be within query length at "
                f"sequence {idx}: append={append_len}, query={query_len}"
            )


def _build_cu_seqlens_like(
    seq_lengths: Sequence[int],
    *,
    reference,
):
    values = [0]
    running = 0
    for length in seq_lengths:
        running += int(length)
        values.append(running)

    if hasattr(reference, "new_tensor"):
        return reference.new_tensor(values)
    try:
        return torch.tensor(values, dtype=torch.int32)
    except AttributeError:
        return values
