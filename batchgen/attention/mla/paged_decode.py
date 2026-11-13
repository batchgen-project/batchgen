"""Dense paged MLA decode: one semantic op, capability-selected kernel.

Contract (all architectures):
    query        [B, 1, H, D]  BF16, D = head_dim_ckv + head_dim_kpe (576)
    blocked_k    [P, S, 1, D]  BF16 packed paged latent cache (page size S)
    block_table  [B, max_pages] int32
    cache_seqlens[B]           int32, context length incl. the new token
    schedule     DecodeSchedule from ``decode_schedule`` (optional): one per
                 step, shared by every MLA layer of that step
    returns      [B, 1, H, head_dim_v] BF16, head_dim_v = head_dim_ckv (512)

Every backend launches from device tensors and shape-derived integers only, so
a call (and a ``decode_schedule`` call) can be captured in a CUDA graph and
replayed against refreshed ``cache_seqlens`` / ``block_table`` contents.

Backends (resolved once per process from the device capability):
    9.x  -> flash_mla dense decode (Hopper kernels; the current upstream
            wheel keeps them on its legacy side behind use_legacy=True). The
            schedule is flash_mla's tile-scheduler metadata, computed on device
            from ``cache_seqlens``.
    10.x -> flashinfer ``trtllm_batch_decode_with_kv_cache_mla`` on the
            trtllm-gen backend (``trtllm_gen_mla``, the launch the sparse seam
            shares): device-side ``seq_lens``, a process-lifetime workspace and
            ``max_seq_len`` = the block-table capacity, a static bound. The
            kernel schedules on device, so the schedule is empty. The kernel
            reads the block table in 128-token groups: its column count must
            be a multiple of ``128 // S`` (``block_table_width``); a narrower
            table is padded with -1 columns that no length reaches.
Anything else fails loud.
"""

from __future__ import annotations

import inspect
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

from batchgen.attention.mla import trtllm_gen_mla

_BACKEND: Optional[str] = None
_FLASHMLA_DENSE_KWARGS: Optional[dict] = None
#: trtllm-gen reads page ids in groups covering this many tokens.
_TRTLLM_GEN_TOKENS_PER_TABLE_GROUP = 128


class DecodeSchedule(NamedTuple):
    """The per-step dense decode schedule (flash_mla's scheduler metadata on
    Hopper; empty on Blackwell, whose kernel schedules on device)."""

    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None


def _backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        major = torch.cuda.get_device_capability()[0]
        if major == 9:
            _BACKEND = "flashmla"
        elif major == 10:
            _BACKEND = "flashinfer-trtllm-gen"
        else:
            raise RuntimeError(
                f"no dense paged MLA decode backend for compute capability {major}.x"
            )
    return _BACKEND


def graph_capturable() -> bool:
    """Whether the selected decode kernel can be captured in a CUDA graph.

    Both backends launch from device tensors and static integers: flash_mla
    from its on-device scheduler metadata, trtllm-gen from device lengths, a
    fixed workspace and a shape-derived ``max_seq_len``.
    """
    return _backend() in ("flashmla", "flashinfer-trtllm-gen")


def flashmla_dense_kwargs() -> dict:
    """Extra kwargs a flash_mla dense-decode call needs for the installed package."""
    global _FLASHMLA_DENSE_KWARGS
    if _FLASHMLA_DENSE_KWARGS is None:
        from flash_mla import flash_mla_with_kvcache

        parameters = inspect.signature(flash_mla_with_kvcache).parameters
        _FLASHMLA_DENSE_KWARGS = {"use_legacy": True} if "use_legacy" in parameters else {}
    return _FLASHMLA_DENSE_KWARGS


def block_table_width(max_pages: int, page_size: int) -> int:
    """Column count a block table of ``max_pages`` pages needs for the kernel.

    A caller that owns a fixed-address table (CUDA-graph statics) allocates
    this many columns so the decode reads it in place; any narrower table is
    padded per call.
    """
    if _backend() == "flashmla":
        return int(max_pages)
    group = _TRTLLM_GEN_TOKENS_PER_TABLE_GROUP
    if page_size <= 0 or (group % page_size and page_size % group):
        raise trtllm_gen_mla.MlaShapeError(
            f"trtllm-gen MLA decode needs a page size that divides or is a "
            f"multiple of {group} tokens, got {page_size}"
        )
    multiple = max(1, group // page_size)
    return -(-int(max_pages) // multiple) * multiple


def decode_schedule(cache_seqlens: torch.Tensor, num_heads: int) -> DecodeSchedule:
    """The step's schedule for ``cache_seqlens``; shareable by every MLA layer
    of the step (same lengths, same local head count)."""
    if _backend() == "flashmla":
        from flash_mla import get_mla_metadata

        metadata, num_splits = get_mla_metadata(
            cache_seqlens, num_heads, 1, **flashmla_dense_kwargs()
        )
        return DecodeSchedule(metadata, num_splits)
    return DecodeSchedule()


def _trtllm_gen_dense(query, blocked_k, block_table, cache_seqlens, head_dim_v,
                      softmax_scale):
    page_size = blocked_k.shape[1]
    width = block_table_width(block_table.shape[1], page_size)
    pad = width - block_table.shape[1]
    table = F.pad(block_table, (0, pad), value=-1) if pad else block_table
    return trtllm_gen_mla.trtllm_gen_mla_decode(
        query,
        blocked_k,
        table,
        cache_seqlens,
        kv_lora_rank=head_dim_v,
        # The table capacity bounds every row's length without reading the
        # device lengths back.
        max_seq_len=width * page_size,
        softmax_scale=softmax_scale,
        sparse_top_k=0,
    )


def mla_paged_decode(
    query: torch.Tensor,
    blocked_k: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    schedule: Optional[DecodeSchedule] = None,
) -> torch.Tensor:
    batch, q_len, num_heads, head_dim = query.shape
    assert q_len == 1, "dense paged MLA decode takes one query token per sequence"
    if schedule is None:
        schedule = decode_schedule(cache_seqlens, num_heads)

    if _backend() == "flashmla":
        from flash_mla import flash_mla_with_kvcache

        out, _ = flash_mla_with_kvcache(
            query, blocked_k, block_table, cache_seqlens, head_dim_v,
            schedule.tile_scheduler_metadata, schedule.num_splits, softmax_scale,
            True, **flashmla_dense_kwargs(),
        )
        return out

    return _trtllm_gen_dense(
        query, blocked_k, block_table, cache_seqlens, head_dim_v, softmax_scale
    )
