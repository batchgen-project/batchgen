"""Common prefix-cache metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


def _build_cu_seqlens_values(seq_lengths: Sequence[int]) -> List[int]:
    values = [0]
    running = 0
    for length in seq_lengths:
        running += int(length)
        values.append(running)
    return values


def ensure_prefix_cache_prepack_metadata(
    metadata,
) -> "PrefixCachePrepackMetadata":
    """Normalize explicit or legacy-compatible prefix metadata."""

    if isinstance(metadata, PrefixCachePrepackMetadata):
        return metadata
    if getattr(metadata, "phase", None) is not None:
        return PrefixCachePrepackMetadata.from_forward_metadata(metadata)
    if getattr(metadata, "cu_seqlens_q", None) is not None:
        raise RuntimeError(
            "PrefillAttentionMetadata does not carry global sequence ids; "
            "pass ForwardBatchMetadata or use AttnWrapperBase-bound fields"
        )
    raise TypeError(
        "metadata must be PrefixCachePrepackMetadata, PrefillAttentionMetadata, "
        "or ForwardBatchMetadata"
    )


@dataclass(frozen=True)
class PrefixCachePrepackMetadata:
    """Validated prepack metadata needed by prefix-cache-aware wrappers."""

    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: List[int]
    max_seqlen: int
    num_sequences: int
    seq_lengths: List[int]
    append_seq_lengths: List[int]
    global_sequence_ids: List[int]
    prefix_reuse_mode: bool
    prefix_shared_tokens: Optional[List[int]]
    full_seq_lengths: Optional[List[int]]

    @classmethod
    def from_prefill_metadata(
        cls,
        prefill_metadata,
        *,
        global_sequence_ids: Sequence[int],
    ) -> "PrefixCachePrepackMetadata":
        """Build wrapper-compatible metadata from explicit prefill metadata."""

        prefix_shared_tokens = None
        full_seq_lengths = None
        seq_lengths = [int(length) for length in prefill_metadata.q_seq_lens]
        append_seq_lengths = _prefill_append_seq_lengths(prefill_metadata)
        kv_seq_lengths = [
            int(length) for length in prefill_metadata.kv_seq_lens
        ]
        if len(append_seq_lengths) != len(seq_lengths):
            raise RuntimeError(
                "Prefix cache metadata append length count does not match "
                f"query length count: {len(append_seq_lengths)} != "
                f"{len(seq_lengths)}"
            )
        if len(kv_seq_lengths) != len(seq_lengths):
            raise RuntimeError(
                "Prefix cache metadata KV length count does not match "
                f"query length count: {len(kv_seq_lengths)} != "
                f"{len(seq_lengths)}"
            )
        prefix_tokens = [
            int(kv_len) - int(append_len)
            for append_len, kv_len in zip(append_seq_lengths, kv_seq_lengths)
        ]
        if any(tokens < 0 for tokens in prefix_tokens):
            raise RuntimeError(
                "Prefix cache metadata requires kv lengths >= append lengths"
            )
        for idx, (append_len, query_len) in enumerate(
            zip(append_seq_lengths, seq_lengths)
        ):
            if append_len < 0 or append_len > query_len:
                raise RuntimeError(
                    "Prefix cache metadata requires append lengths within "
                    f"query lengths at sequence {idx}: append={append_len}, "
                    f"query={query_len}"
                )
        prefix_reuse_mode = any(tokens > 0 for tokens in prefix_tokens)
        if prefix_reuse_mode:
            prefix_shared_tokens = prefix_tokens
            full_seq_lengths = kv_seq_lengths

        metadata = cls(
            cu_seqlens=prefill_metadata.cu_seqlens_q,
            cu_seqlens_cpu=_build_cu_seqlens_values(seq_lengths),
            max_seqlen=int(prefill_metadata.max_seqlen_q),
            num_sequences=int(prefill_metadata.batch_size),
            seq_lengths=seq_lengths,
            append_seq_lengths=append_seq_lengths,
            global_sequence_ids=[int(seq_id) for seq_id in global_sequence_ids],
            prefix_reuse_mode=prefix_reuse_mode,
            prefix_shared_tokens=prefix_shared_tokens,
            full_seq_lengths=full_seq_lengths,
        )
        return metadata

    @classmethod
    def from_forward_metadata(
        cls,
        forward_metadata,
    ) -> "PrefixCachePrepackMetadata":
        """Build wrapper-compatible metadata from a bound forward metadata object."""

        if (
            forward_metadata.phase != "prefill"
            or forward_metadata.prefill is None
        ):
            raise RuntimeError(
                "Prefix cache prepack metadata requires bound prefill metadata"
            )
        return cls.from_prefill_metadata(
            forward_metadata.prefill,
            global_sequence_ids=forward_metadata.global_sequence_ids,
        )

    @classmethod
    def from_wrapper_cls(
        cls, wrapper_cls: type
    ) -> "PrefixCachePrepackMetadata":
        """Build metadata from legacy wrapper class variables."""

        cu_seqlens = getattr(wrapper_cls, "prepack_cu_seqlens", None)
        max_seqlen = getattr(wrapper_cls, "prepack_max_seqlen", None)
        num_sequences = getattr(wrapper_cls, "prepack_num_sequences", None)
        seq_lengths = getattr(wrapper_cls, "prepack_seq_lengths", None)
        append_seq_lengths = getattr(
            wrapper_cls, "prepack_append_seq_lengths", None
        )
        global_sequence_ids = getattr(wrapper_cls, "cur_batch", None)
        prefix_reuse_mode = bool(
            getattr(wrapper_cls, "prepack_prefix_reuse_mode", False)
        )
        prefix_shared_tokens = getattr(
            wrapper_cls, "prepack_prefix_shared_tokens", None
        )
        full_seq_lengths = getattr(
            wrapper_cls, "prepack_full_seq_lengths", None
        )

        if cu_seqlens is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires cu_seqlens"
            )
        if max_seqlen is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires max_seqlen"
            )
        if num_sequences is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires num_sequences"
            )
        if seq_lengths is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires seq_lengths"
            )
        if global_sequence_ids is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires cur_batch"
            )

        seq_lengths = [int(length) for length in seq_lengths]
        if append_seq_lengths is None:
            append_seq_lengths = list(seq_lengths)
        else:
            append_seq_lengths = [int(length) for length in append_seq_lengths]
        global_sequence_ids = [int(seq_id) for seq_id in global_sequence_ids]
        num_sequences = int(num_sequences)
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
        for idx, (append_len, query_len) in enumerate(
            zip(append_seq_lengths, seq_lengths)
        ):
            if append_len < 0 or append_len > query_len:
                raise RuntimeError(
                    "Prefix cache append length must be within query length at "
                    f"sequence {idx}: append={append_len}, query={query_len}"
                )
        if len(cu_seqlens) != num_sequences + 1:
            raise RuntimeError(
                "Prefix cache cu_seqlens length does not match num_sequences: "
                f"{len(cu_seqlens)} != {num_sequences + 1}"
            )

        needs_prefix_metadata = prefix_reuse_mode
        if needs_prefix_metadata:
            if prefix_shared_tokens is None:
                raise RuntimeError(
                    "Prefix cache mode requires prepack_prefix_shared_tokens"
                )
            if full_seq_lengths is None:
                raise RuntimeError(
                    "Prefix cache mode requires prepack_full_seq_lengths"
                )
            prefix_shared_tokens = [
                int(tokens) for tokens in prefix_shared_tokens
            ]
            full_seq_lengths = [int(length) for length in full_seq_lengths]
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
                expected_full_length = int(append_len) + int(prefix_tokens)
                if expected_full_length != int(full_length):
                    raise RuntimeError(
                        "Prefix cache full length mismatch at sequence "
                        f"{idx}: append={append_len}, prefix={prefix_tokens}, "
                        f"full={full_length}"
                    )

        metadata = cls(
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=_build_cu_seqlens_values(seq_lengths),
            max_seqlen=int(max_seqlen),
            num_sequences=num_sequences,
            seq_lengths=seq_lengths,
            append_seq_lengths=append_seq_lengths,
            global_sequence_ids=global_sequence_ids,
            prefix_reuse_mode=prefix_reuse_mode,
            prefix_shared_tokens=prefix_shared_tokens,
            full_seq_lengths=full_seq_lengths,
        )
        return metadata

    def cu_seqlens_list(self) -> List[int]:
        return list(self.cu_seqlens_cpu)

    def append_seq_lengths_list(self) -> List[int]:
        return list(self.append_seq_lengths)


def _prefill_append_seq_lengths(prefill_metadata) -> List[int]:
    append_seq_lens = getattr(prefill_metadata, "append_seq_lens", None)
    if append_seq_lens is None:
        return [int(length) for length in prefill_metadata.q_seq_lens]
    return [int(length) for length in append_seq_lens]
