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
class PrefixReuseMetadata:
    """Prefix reuse information for a prefill forward batch."""

    prefix_lens: torch.Tensor
    suffix_lens: torch.Tensor
    full_seq_lens: torch.Tensor
    saved_tokens: int
    is_full_hit: torch.Tensor
    global_sequence_ids: list[int]


@dataclass(frozen=True)
class PrefillAttentionMetadata:
    """Attention metadata for prefill or suffix-only prefill."""

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    q_seq_lens: list[int]
    kv_seq_lens: list[int]
    position_ids: torch.Tensor
    prefix_reuse: Optional[PrefixReuseMetadata] = None

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
