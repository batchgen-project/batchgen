"""Sparse FP8 FlashMLA decode over the packed primary MLA cache.

This is the ONLY decode reader of a packed-FP8 primary cache, for both the
selected (top-k) path and the all-short path. There is no dense FP8 MLA decode
kernel in the pin -- SM90 refuses dense FP8 outright and SM100 refuses anything
that is not FP8+sparse (`csrc/pybind.cpp:331-357`) -- so "all short rows" is
served by the same sparse kernel with the row's own tokens as its indices,
which is what makes the reader cut a hard cut rather than a dual-format one.

Facts about the pinned kernel that this module's shape depends on, all read
out of `references/FlashMLA-1408756`:

1. **The sparse metadata is step-invariant.** `get_mla_metadata_kernel` uses
   ``cur_s_k = params.topk == -1 ? seqlens_k[i] : params.topk``
   (`csrc/smxx/get_mla_metadata.cu:30`): in SPARSE mode it ignores
   ``cache_seqlens`` entirely and treats every row as exactly ``topk`` long.
   The metadata therefore depends only on (batch_size, topk) and on
   arch-fixed quantities (SM count, head counts), so it can be computed ONCE
   per capture bucket and replayed unchanged forever. That removes the
   metadata kernel from the captured region and removes the open question of
   whether its output shape is replay-stable.
2. **``block_table`` and ``cache_seqlens`` are unused in sparse mode.** The
   sparse FP8 kernel reads only ``params.topk`` and ``params.indices_ptr``
   (`csrc/sm90/decode/sparse_fp8/splitkv_mla.cu` -- no reference to
   ``block_table`` or ``seqlens_k_ptr``); the page index is already encoded in
   each index. Both are still shape- and dtype-checked at the pybind boundary,
   so they are passed as correctly-shaped placeholders and their CONTENT is
   never load-bearing. We pass the real tensors anyway, so a future pin that
   starts reading them cannot read garbage.
3. **An index of -1 is memory-safe but a row of ONLY -1 produces NaN.**
   The producer clamps a -1 into page 0 slot 63 deliberately
   (`splitkv_mla.cu:468`) and the consumer masks those lanes to -inf
   (`:91-93`), so they contribute nothing. But the row's softmax denominator
   ``rL`` then stays 0 and the epilogue divides by it
   (`components/epilogue.h:34`), so a row with zero valid indices returns
   NaN -- the same class of degenerate-input hazard as the DeepGEMM
   all-zero-lens case. Padding rows in a capture bucket have exactly that
   shape (``transform_selected_positions_out`` fills a row whose slot is -1
   or whose index exceeds ``num_valid_tokens`` with all -1), so
   ``guard_sparse_indices_against_empty_rows`` below gives every row one valid
   index. It is a tensor-level clamp on a single column, graph-capturable, and
   it cannot change a live row: a live row's first index is already >= 0.
4. **``topk`` must be a multiple of 64** (``FLASH_ASSERT(params.topk %
   TOPK_BLOCK_SIZE == 0)``, ``TOPK_BLOCK_SIZE = 64``), and the kernel always
   sweeps ``ceil(topk / 64)`` index blocks for every row regardless of how
   many are valid. Sparse decode cost is therefore set by the WIDTH of the
   indices tensor, not by the row's length -- which is why
   ``sparse_indices_width`` exists: an all-short bucket sizes its indices to
   the bucket's own KV bound instead of paying the full 2048 every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "TOPK_BLOCK_SIZE",
    "SparseFp8MlaMetadata",
    "sparse_indices_width",
    "prepare_sparse_fp8_mla_metadata",
    "guard_sparse_indices_against_empty_rows",
    "run_sparse_fp8_mla_decode",
]

# csrc/sm90/decode/sparse_fp8/components/config.h:15
TOPK_BLOCK_SIZE = 64


def _flash_mla_ops():
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    return flash_mla_with_kvcache, get_mla_metadata


@dataclass(frozen=True)
class SparseFp8MlaMetadata:
    """Per-bucket FlashMLA sparse metadata, computed once (see note 1)."""

    tile_scheduler_metadata: torch.Tensor
    num_splits: torch.Tensor
    batch_size: int
    topk: int
    num_heads_q: int


def sparse_indices_width(max_kv_tokens: int, index_topk: int) -> int:
    """Width of the indices tensor for a bucket, rounded to the kernel's block.

    ``max_kv_tokens`` is the largest number of KV tokens any row of this
    capture can ever attend to; ``index_topk`` is the selection budget. The
    sparse kernel sweeps the full width every step (note 4), so a bucket whose
    rows can never exceed, say, 512 tokens must not be given a 2048-wide
    indices tensor -- it would do 4x the KV work of the dense path it replaces.
    """

    if index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {index_topk}")
    if index_topk % TOPK_BLOCK_SIZE != 0:
        raise ValueError(
            f"FlashMLA requires topk % {TOPK_BLOCK_SIZE} == 0 "
            f"(FLASH_ASSERT in splitkv_mla.cu), got index_topk={index_topk}"
        )
    if max_kv_tokens <= 0:
        raise ValueError(
            f"max_kv_tokens must be positive, got {max_kv_tokens}"
        )
    width = min(int(index_topk), int(max_kv_tokens))
    width = -(-width // TOPK_BLOCK_SIZE) * TOPK_BLOCK_SIZE
    return min(width, int(index_topk))


def prepare_sparse_fp8_mla_metadata(
    *,
    batch_size: int,
    topk: int,
    num_heads_q: int,
    num_heads_k: int = 1,
    num_q_tokens_per_q_seq: int = 1,
    device: torch.device,
) -> SparseFp8MlaMetadata:
    """Build the sparse FP8 metadata for one capture bucket.

    ``cache_seqlens`` is a required argument of ``get_mla_metadata`` but is
    ignored whenever ``topk`` is given (note 1), so this passes a ``topk``-
    filled vector: it is the value the kernel substitutes anyway, which keeps
    the call honest instead of relying on a zero vector being ignored. That
    also sidesteps the degenerate all-zero-lengths input entirely -- the
    lesson from the DeepGEMM metadata path.
    """

    # Validated before the kernel is resolved, for the same reason as in
    # `run_sparse_fp8_mla_decode`.
    if topk % TOPK_BLOCK_SIZE != 0:
        raise ValueError(
            f"FlashMLA requires topk % {TOPK_BLOCK_SIZE} == 0, got {topk}"
        )
    if num_heads_k != 1:
        raise ValueError(
            "FlashMLA's FP8 KV cache requires num_heads_k == 1 "
            f"(csrc/pybind.cpp:253), got {num_heads_k}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_heads_q <= 0 or num_heads_q % num_heads_k != 0:
        raise ValueError(
            f"num_heads_q ({num_heads_q}) must be a positive multiple of "
            f"num_heads_k ({num_heads_k}) -- csrc/pybind.cpp:65"
        )

    _, get_mla_metadata = _flash_mla_ops()
    cache_seqlens = torch.full(
        (batch_size,), int(topk), dtype=torch.int32, device=device
    )
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens,
        int(num_q_tokens_per_q_seq) * int(num_heads_q) // int(num_heads_k),
        int(num_heads_k),
        num_heads_q=int(num_heads_q),
        is_fp8_kvcache=True,
        topk=int(topk),
    )
    if not isinstance(tile_scheduler_metadata, torch.Tensor) or not isinstance(
        num_splits, torch.Tensor
    ):
        raise TypeError(
            "sparse FP8 FlashMLA decode requires the tensor metadata API; the "
            "installed flash_mla returned non-tensor metadata"
        )
    return SparseFp8MlaMetadata(
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        batch_size=int(batch_size),
        topk=int(topk),
        num_heads_q=int(num_heads_q),
    )


def guard_sparse_indices_against_empty_rows(
    indices: torch.Tensor, zero_column: torch.Tensor
) -> None:
    """Give every row at least one valid index, in place (see note 3).

    ``indices`` is ``[batch, topk]`` int32. A row of only -1 would make the
    kernel divide 0 by 0 and emit NaN for that row; padding rows have exactly
    that content. Clamping column 0 up to 0 costs one elementwise launch, is
    capturable, and is a no-op on any row that has a real first index, because
    ``transform_selected_positions_out`` fills a live row's column 0 first and
    only ever writes a non-negative physical token id there.

    The guarded row then attends to physical token 0 and produces a finite,
    meaningless result -- which is correct, because the segment's row mask
    zeroes that row's contribution before it leaves the layer.
    """

    if indices.ndim != 2:
        raise ValueError(
            f"sparse indices must be [batch, topk], got {tuple(indices.shape)}"
        )
    if indices.dtype != torch.int32:
        raise TypeError(
            f"FlashMLA requires int32 indices, got {indices.dtype}"
        )
    if zero_column.shape != (indices.shape[0],):
        raise ValueError(
            f"zero_column must have shape {(indices.shape[0],)}, got "
            f"{tuple(zero_column.shape)}"
        )
    torch.maximum(indices[:, 0], zero_column, out=indices[:, 0])


def run_sparse_fp8_mla_decode(
    *,
    query: torch.Tensor,
    packed_blocked_k: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    metadata: "SparseFp8MlaMetadata",
    softmax_scale: float,
    head_dim_v: int,
) -> torch.Tensor:
    """Attend over the packed cache at the given physical token indices.

    Args:
        query: bf16 ``[batch, 1, num_heads_q, 576]``. The first 512 dims are
            the ABSORBED NoPE query (q_nope @ absorb) and the last 64 the RoPE
            query, matching the record's own NoPE-then-RoPE order -- the
            kernel takes one 576-wide query, not FA3's split
            ``q``/``qv``/``k_cache``/``v_cache`` quartet.
        packed_blocked_k: uint8 ``[num_pages, 64, 1, 656]``, the paged cache
            exactly as allocated. Unlike the FA3 selected path this is NOT
            re-viewed at page size 1: the page index is encoded in each index.
        indices: int32 ``[batch, 1, topk]``, ``physical_page * 64 + offset``,
            -1 for invalid, last dim contiguous. Every row must already have a
            valid entry (``guard_sparse_indices_against_empty_rows``).
        block_table, cache_seqlens: shape-checked placeholders (note 2).
        metadata: this bucket's step-invariant metadata (note 1).

    Returns:
        bf16 ``[batch, 1, num_heads_q, head_dim_v]``.
    """

    # Validated BEFORE the kernel is resolved, so a shape disagreement is
    # reported as itself rather than as whatever the kernel does with it, and
    # so these guards are exercisable without Triton or a GPU.
    # The metadata is built once per (batch, topk, heads) and reused, so these
    # three are the whole agreement between it and this call. A mismatch is
    # not caught downstream: the kernel reads the tile schedule the metadata
    # baked and the indices this call passes, so a disagreement attends to the
    # wrong window or the wrong number of rows WITHOUT failing.
    if indices.shape[-1] != metadata.topk:
        raise ValueError(
            f"indices width {indices.shape[-1]} does not match the metadata's "
            f"topk {metadata.topk}: the metadata fixes how many index blocks "
            "the kernel sweeps, so a mismatch silently attends to the wrong "
            "window"
        )
    if query.shape[0] != metadata.batch_size:
        raise ValueError(
            f"query batch {query.shape[0]} does not match the metadata's "
            f"batch_size {metadata.batch_size}: num_splits is sized "
            "batch_size + 1 and the tile schedule is built for that many rows"
        )
    if query.shape[-2] != metadata.num_heads_q:
        raise ValueError(
            f"query has {query.shape[-2]} heads but the metadata was built "
            f"for {metadata.num_heads_q}: the head count sets the tile "
            "schedule's work split, so a mismatch mis-partitions the kernel"
        )
    if indices.shape[0] != query.shape[0]:
        raise ValueError(
            f"indices batch {indices.shape[0]} does not match query batch "
            f"{query.shape[0]}"
        )

    flash_mla_with_kvcache, _ = _flash_mla_ops()
    attn_out, _ = flash_mla_with_kvcache(
        query,
        packed_blocked_k,
        block_table,
        cache_seqlens,
        int(head_dim_v),
        metadata.tile_scheduler_metadata,
        metadata.num_splits,
        softmax_scale=float(softmax_scale),
        # The pin asserts causal is False whenever indices are given: top-k
        # attention has nothing to do with a causal mask, and the selection
        # itself already excludes anything the row may not see.
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
    )
    return attn_out
