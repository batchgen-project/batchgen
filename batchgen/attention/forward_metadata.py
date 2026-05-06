"""First-class forward metadata for attention execution.

These dataclasses describe the logical forward batch without depending on
legacy wrapper class variables. They intentionally do not mutate runtime state;
callers should validate them before binding or passing them to wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch


ForwardPhase = Literal["prefill", "decode"]


def _to_int_list(values: Sequence[int], name: str) -> list[int]:
    try:
        result = [int(value) for value in values]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc
    return result


def _require_1d_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(tensor.shape)}")


def _require_integer_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64 dtype, got {tensor.dtype}")


def _require_bool_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must use bool dtype, got {tensor.dtype}")


def _tensor_values(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _validate_non_negative(values: Sequence[int], name: str) -> None:
    for idx, value in enumerate(values):
        if int(value) < 0:
            raise ValueError(f"{name}[{idx}] must be non-negative, got {value}")


def _validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    seq_lens: Sequence[int],
    name: str,
) -> None:
    _require_1d_tensor(cu_seqlens, name)
    _require_integer_tensor(cu_seqlens, name)
    if cu_seqlens.numel() != len(seq_lens) + 1:
        raise ValueError(
            f"{name} length must be batch_size + 1: "
            f"{cu_seqlens.numel()} != {len(seq_lens) + 1}"
        )
    values = _tensor_values(cu_seqlens)
    if not values or values[0] != 0:
        raise ValueError(f"{name} must start with 0")
    expected = [0]
    running = 0
    for length in seq_lens:
        running += int(length)
        expected.append(running)
    if values != expected:
        raise ValueError(f"{name} does not match sequence lengths: {values} != {expected}")


@dataclass(frozen=True)
class PrefixReuseMetadata:
    """Prefix reuse information for a prefill forward batch."""

    prefix_lens: torch.Tensor
    suffix_lens: torch.Tensor
    full_seq_lens: torch.Tensor
    saved_tokens: int
    is_full_hit: torch.Tensor
    global_sequence_ids: list[int]

    def validate(self) -> None:
        _require_1d_tensor(self.prefix_lens, "prefix_lens")
        _require_1d_tensor(self.suffix_lens, "suffix_lens")
        _require_1d_tensor(self.full_seq_lens, "full_seq_lens")
        _require_1d_tensor(self.is_full_hit, "is_full_hit")
        _require_integer_tensor(self.prefix_lens, "prefix_lens")
        _require_integer_tensor(self.suffix_lens, "suffix_lens")
        _require_integer_tensor(self.full_seq_lens, "full_seq_lens")
        _require_bool_tensor(self.is_full_hit, "is_full_hit")

        batch_size = len(self.global_sequence_ids)
        for name, tensor in (
            ("prefix_lens", self.prefix_lens),
            ("suffix_lens", self.suffix_lens),
            ("full_seq_lens", self.full_seq_lens),
            ("is_full_hit", self.is_full_hit),
        ):
            if tensor.numel() != batch_size:
                raise ValueError(
                    f"{name} length must match global_sequence_ids: "
                    f"{tensor.numel()} != {batch_size}"
                )

        prefix = _tensor_values(self.prefix_lens)
        suffix = _tensor_values(self.suffix_lens)
        full = _tensor_values(self.full_seq_lens)
        full_hit = [bool(value) for value in self.is_full_hit.detach().cpu().tolist()]
        _validate_non_negative(prefix, "prefix_lens")
        _validate_non_negative(suffix, "suffix_lens")
        _validate_non_negative(full, "full_seq_lens")

        for idx, (prefix_len, suffix_len, full_len, is_full) in enumerate(
            zip(prefix, suffix, full, full_hit)
        ):
            if prefix_len + suffix_len != full_len:
                raise ValueError(
                    "prefix_lens + suffix_lens must equal full_seq_lens: "
                    f"idx={idx}, {prefix_len} + {suffix_len} != {full_len}"
                )
            if is_full and suffix_len != 0:
                raise ValueError(
                    f"full-hit sequence must have zero suffix length: idx={idx}, "
                    f"suffix_len={suffix_len}"
                )
            if (suffix_len == 0) != is_full:
                raise ValueError(
                    f"is_full_hit must match suffix_lens == 0: idx={idx}, "
                    f"is_full_hit={is_full}, suffix_len={suffix_len}"
                )

        if int(self.saved_tokens) != sum(prefix):
            raise ValueError(
                f"saved_tokens must equal sum(prefix_lens): "
                f"{int(self.saved_tokens)} != {sum(prefix)}"
            )


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

    def validate(self) -> None:
        q_seq_lens = _to_int_list(self.q_seq_lens, "q_seq_lens")
        kv_seq_lens = _to_int_list(self.kv_seq_lens, "kv_seq_lens")
        if len(q_seq_lens) != len(kv_seq_lens):
            raise ValueError(
                f"q_seq_lens and kv_seq_lens must have the same length: "
                f"{len(q_seq_lens)} != {len(kv_seq_lens)}"
            )
        _validate_non_negative(q_seq_lens, "q_seq_lens")
        _validate_non_negative(kv_seq_lens, "kv_seq_lens")
        for idx, (q_len, kv_len) in enumerate(zip(q_seq_lens, kv_seq_lens)):
            if q_len > kv_len:
                raise ValueError(
                    f"q_seq_lens cannot exceed kv_seq_lens: idx={idx}, "
                    f"{q_len} > {kv_len}"
                )

        _validate_cu_seqlens(self.cu_seqlens_q, q_seq_lens, "cu_seqlens_q")
        _validate_cu_seqlens(self.cu_seqlens_k, kv_seq_lens, "cu_seqlens_k")
        _require_1d_tensor(self.position_ids, "position_ids")
        _require_integer_tensor(self.position_ids, "position_ids")

        total_q = sum(q_seq_lens)
        if self.position_ids.numel() != total_q:
            raise ValueError(
                f"position_ids length must match total query tokens: "
                f"{self.position_ids.numel()} != {total_q}"
            )
        expected_max_q = max(q_seq_lens, default=0)
        expected_max_k = max(kv_seq_lens, default=0)
        if int(self.max_seqlen_q) != expected_max_q:
            raise ValueError(
                f"max_seqlen_q mismatch: {int(self.max_seqlen_q)} != {expected_max_q}"
            )
        if int(self.max_seqlen_k) != expected_max_k:
            raise ValueError(
                f"max_seqlen_k mismatch: {int(self.max_seqlen_k)} != {expected_max_k}"
            )

        if self.prefix_reuse is not None:
            self.prefix_reuse.validate()
            if len(self.prefix_reuse.global_sequence_ids) != len(q_seq_lens):
                raise ValueError(
                    "prefix_reuse batch size must match prefill metadata batch size"
                )
            suffix_lens = _tensor_values(self.prefix_reuse.suffix_lens)
            full_seq_lens = _tensor_values(self.prefix_reuse.full_seq_lens)
            if suffix_lens != q_seq_lens:
                raise ValueError(
                    f"prefix_reuse suffix_lens must match q_seq_lens: "
                    f"{suffix_lens} != {q_seq_lens}"
                )
            if full_seq_lens != kv_seq_lens:
                raise ValueError(
                    f"prefix_reuse full_seq_lens must match kv_seq_lens: "
                    f"{full_seq_lens} != {kv_seq_lens}"
                )


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

    def validate(self) -> None:
        _require_1d_tensor(self.cache_seqlens, "cache_seqlens")
        _require_integer_tensor(self.cache_seqlens, "cache_seqlens")
        values = _tensor_values(self.cache_seqlens)
        _validate_non_negative(values, "cache_seqlens")
        expected_max = max(values, default=0)
        if int(self.max_seqlen) != expected_max:
            raise ValueError(
                f"max_seqlen mismatch: {int(self.max_seqlen)} != {expected_max}"
            )

        if self.page_table is not None:
            if not isinstance(self.page_table, torch.Tensor):
                raise TypeError("page_table must be a torch.Tensor")
            if self.page_table.ndim != 2:
                raise ValueError(
                    f"page_table must be 2D, got shape={tuple(self.page_table.shape)}"
                )
            if self.page_table.shape[0] != self.batch_size:
                raise ValueError(
                    f"page_table batch dimension mismatch: "
                    f"{self.page_table.shape[0]} != {self.batch_size}"
                )

        if self.slot_indices is not None:
            _require_1d_tensor(self.slot_indices, "slot_indices")
            _require_integer_tensor(self.slot_indices, "slot_indices")
            if self.slot_indices.numel() != self.batch_size:
                raise ValueError(
                    f"slot_indices length must match batch size: "
                    f"{self.slot_indices.numel()} != {self.batch_size}"
                )


@dataclass(frozen=True)
class KVCacheMetadata:
    """KV cache handles associated with a forward batch."""

    gpu_paged_kv_manager: Optional[object] = None
    host_worker_view: Optional[object] = None
    aux_gpu_paged_kv_manager: Optional[object] = None
    aux_host_worker_view: Optional[object] = None

    def validate(self) -> None:
        # Handles are intentionally opaque. Validation only asserts the object is
        # structurally a metadata container and leaves capability checks to users.
        return None


@dataclass(frozen=True)
class ForwardBatchMetadata:
    """Top-level metadata object for one model forward batch."""

    phase: ForwardPhase
    global_sequence_ids: list[int]
    prefill: Optional[PrefillAttentionMetadata] = None
    decode: Optional[DecodeAttentionMetadata] = None
    kv_cache: Optional[KVCacheMetadata] = None

    def validate(self) -> None:
        if self.phase not in ("prefill", "decode"):
            raise ValueError(f"Unsupported forward phase: {self.phase!r}")
        global_sequence_ids = _to_int_list(
            self.global_sequence_ids, "global_sequence_ids"
        )
        if self.phase == "prefill":
            if self.prefill is None:
                raise ValueError("prefill metadata is required for prefill phase")
            if self.decode is not None:
                raise ValueError("decode metadata must be None for prefill phase")
            self.prefill.validate()
            if len(global_sequence_ids) != self.prefill.batch_size:
                raise ValueError(
                    f"global_sequence_ids length must match prefill batch size: "
                    f"{len(global_sequence_ids)} != {self.prefill.batch_size}"
                )
            if self.prefill.prefix_reuse is not None:
                prefix_ids = _to_int_list(
                    self.prefill.prefix_reuse.global_sequence_ids,
                    "prefix_reuse.global_sequence_ids",
                )
                if prefix_ids != global_sequence_ids:
                    raise ValueError(
                        "prefix_reuse global_sequence_ids must match forward batch: "
                        f"{prefix_ids} != {global_sequence_ids}"
                    )
        else:
            if self.decode is None:
                raise ValueError("decode metadata is required for decode phase")
            if self.prefill is not None:
                raise ValueError("prefill metadata must be None for decode phase")
            self.decode.validate()
            if len(global_sequence_ids) != self.decode.batch_size:
                raise ValueError(
                    f"global_sequence_ids length must match decode batch size: "
                    f"{len(global_sequence_ids)} != {self.decode.batch_size}"
                )

        if self.kv_cache is not None:
            self.kv_cache.validate()
