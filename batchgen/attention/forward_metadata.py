"""First-class forward metadata for attention execution.

These dataclasses describe the logical forward batch without depending on
legacy wrapper class variables. They intentionally do not mutate runtime state;
builders are responsible for constructing them from already-validated static
planning inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

ForwardPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class PrefillAttentionMetadata:
    """Attention metadata for prefill or suffix-only prefill.

    Prefix reuse is represented by q/kv length divergence:
    ``kv_seq_lens[i] - q_seq_lens[i]`` is the cached prefix length for sequence
    ``i``. The legacy wrapper context mirrors these derived values into
    ``AttnWrapperBase.prepack_prefix_*`` for model wrappers.
    """

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    q_seq_lens: list[int]
    kv_seq_lens: list[int]
    position_ids: torch.Tensor
    append_seq_lens: Optional[list[int]] = None

    @property
    def batch_size(self) -> int:
        return len(self.q_seq_lens)


@dataclass(frozen=True)
class DecodeAttentionMetadata:
    """Attention metadata for decode forward batches."""

    cache_seqlens: torch.Tensor
    max_seqlen: int
    page_table: Optional[torch.Tensor] = None
    slot_indices: Optional[torch.Tensor] = None
    batch_slice: Optional[slice] = None

    @property
    def batch_size(self) -> int:
        return int(self.cache_seqlens.numel())


@dataclass(frozen=True)
class KVCacheMetadata:
    """KV cache handles associated with a forward batch."""

    gpu_paged_kv_manager: Optional[object] = None
    host_worker_view: Optional[object] = None
    aux_gpu_paged_kv_manager: Optional[object] = None
    aux_host_worker_view: Optional[object] = None
    prefill_prefix_materialization: Optional[object] = None


@dataclass(frozen=True)
class ForwardBatchMetadata:
    """Top-level metadata object for one model forward batch."""

    phase: ForwardPhase
    global_sequence_ids: list[int]
    prefill: Optional[PrefillAttentionMetadata] = None
    decode: Optional[DecodeAttentionMetadata] = None
    kv_cache: Optional[KVCacheMetadata] = None

    def require_prefill(self) -> PrefillAttentionMetadata:
        if self.phase != "prefill" or self.prefill is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires prefill metadata"
            )
        return self.prefill

    @property
    def cu_seqlens(self) -> torch.Tensor:
        return self.require_prefill().cu_seqlens_q

    @property
    def cu_seqlens_cpu(self) -> list[int]:
        return _build_cu_seqlens_values(self.seq_lengths)

    @property
    def max_seqlen(self) -> int:
        return int(self.require_prefill().max_seqlen_q)

    @property
    def num_sequences(self) -> int:
        return int(self.require_prefill().batch_size)

    @property
    def seq_lengths(self) -> list[int]:
        return [int(length) for length in self.require_prefill().q_seq_lens]

    @property
    def append_seq_lengths(self) -> list[int]:
        prefill = self.require_prefill()
        if prefill.append_seq_lens is None:
            return [int(length) for length in prefill.q_seq_lens]
        return [int(length) for length in prefill.append_seq_lens]

    @property
    def prefix_shared_tokens(self) -> Optional[list[int]]:
        tokens = [
            int(kv_len) - int(append_len)
            for kv_len, append_len in zip(
                self.require_prefill().kv_seq_lens,
                self.append_seq_lengths,
            )
        ]
        if any(token < 0 for token in tokens):
            raise RuntimeError(
                "Prefix cache metadata requires kv lengths >= append lengths"
            )
        return tokens if any(token > 0 for token in tokens) else None

    @property
    def prefix_reuse_mode(self) -> bool:
        tokens = self.prefix_shared_tokens
        return tokens is not None and any(token > 0 for token in tokens)

    @property
    def full_seq_lengths(self) -> Optional[list[int]]:
        if not self.prefix_reuse_mode:
            return None
        return [int(length) for length in self.require_prefill().kv_seq_lens]

    def cu_seqlens_list(self) -> list[int]:
        return list(self.cu_seqlens_cpu)

    def append_seq_lengths_list(self) -> list[int]:
        return list(self.append_seq_lengths)


def _build_cu_seqlens_values(seq_lengths: list[int]) -> list[int]:
    values = [0]
    running = 0
    for length in seq_lengths:
        running += int(length)
        values.append(running)
    return values
