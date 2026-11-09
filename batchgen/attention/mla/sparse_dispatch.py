"""Sparse MLA attention over indexer-selected tokens: one semantic op per phase,
capability-selected kernel.

The op (decode and prefill alike), per query row ``n`` and head ``h``::

    out[n, h] = sum_j softmax_j(scale * (q_latent[n, h] . c_j + q_pe[n, h] . k_pe_j)) c_j

over the token rows ``j in token_rows[n, :lengths[n]]`` of a latent KV whose
row is ``[c (kv_lora_rank) | k_pe (rope)]`` (576 = 512 + 64 for Kimi-K3.1).
Entries at or beyond ``lengths[n]`` are never read. A row of length 0 selects
nothing; its output is unspecified and must not be consumed (graph padding).

Contract (all architectures):
    q_pe        [N, 1, H, rope]  BF16
    q_latent    [N, 1, H, lora]  BF16, the absorbed query
    token_rows  [N, topk]        int32 KV row ids (decode: page * S + offset)
    lengths     [N]              int32, active prefix of ``token_rows``
    returns     [N, 1, H, lora]  BF16
Decode reads the paged latent cache ``blocked_k [P, S, 1, rope + lora]``;
prefill reads flat latent rows ``kv_rows [R, rope + lora]``.

Backends (resolved once per process from the device capability):
    9.x  -> FlashAttention-3 ``qv`` over a page-size-1 view of the latent
            (decode and prefill; the pre-seam calls, unchanged);
    10.x -> decode: flashinfer ``trtllm_batch_decode_with_kv_cache_mla``,
            trtllm-gen sparse MLA (token-row page table, ``seq_lens`` = the
            active top-k length, BF16 KV unchanged);
            prefill: ``flash_mla.flash_mla_sparse_fwd`` on its legacy SM10x
            kernels (``d_qk`` 576), query heads zero-padded to the kernel's
            64/128 head tile, ``topk_length`` = the active length.
Anything else fails loud.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

_BACKEND: Optional[str] = None
_FI_WORKSPACES: dict = {}
#: trtllm-gen MLA scratch per device; zero-initialized once (semaphore state).
_FI_WORKSPACE_BYTES = 128 * 1024 * 1024
#: flash_mla sparse prefill runs only 64- or 128-head query tiles.
_FLASHMLA_PREFILL_HEAD_TILES = (64, 128)
#: The legacy SM10x sparse prefill consumes the top-k in blocks of this width.
_FLASHMLA_PREFILL_TOPK_BLOCK = 128
#: Query rows per prefill launch: bounds the padded-head query and output
#: buffers (rows are independent, so chunking does not change the result).
_FLASHMLA_PREFILL_ROW_CHUNK = 8192

#: Third-party callables each backend reaches (startup preload checks them).
SITE_SYMBOLS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "fa3": {"flash_attn_interface": ("flash_attn_with_kvcache",)},
    "blackwell": {
        "flashinfer.mla": ("trtllm_batch_decode_with_kv_cache_mla",),
        "flash_mla": ("flash_mla_sparse_fwd",),
    },
}


class SparseMlaShapeError(ValueError):
    """An input the selected sparse MLA kernel cannot run."""


def _backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        major = torch.cuda.get_device_capability()[0]
        if major == 9:
            _BACKEND = "fa3"
        elif major == 10:
            _BACKEND = "blackwell"
        else:
            raise RuntimeError(
                f"no sparse MLA attention backend for compute capability {major}.x"
            )
    return _BACKEND


def site_symbols() -> Dict[str, Tuple[str, ...]]:
    """``{module: callables}`` the selected backend calls."""
    return SITE_SYMBOLS[_backend()]


def graph_capturable() -> bool:
    """Whether the selected sparse DECODE kernel can be captured in a CUDA graph.

    FA3 captures. The flashinfer trtllm-gen sparse call is one functional
    launch over a process-lifetime workspace with device-side lengths and a
    static ``max_seq_len`` (the top-k width): no per-call plan and no host
    synchronization. The Kimi decode span on Blackwell nevertheless runs eager
    while the dense seam (``paged_decode.graph_capturable``) is not
    capturable; the planner requires both.
    """
    return _backend() in ("fa3", "blackwell")


# --------------------------------------------------------------------------- #
#  Hopper: FlashAttention-3 qv over a page-size-1 latent view                  #
# --------------------------------------------------------------------------- #


def _fa3_with_kvcache():
    from flash_attn_interface import flash_attn_with_kvcache

    return flash_attn_with_kvcache


def _fa3_num_splits() -> int:
    from batchgen.attention.dsa.fa3_split_policy import GLM5_SPARSE_FA3_NUM_SPLITS

    return GLM5_SPARSE_FA3_NUM_SPLITS


def _fa3_sparse(q_pe, q_latent, flat_kv, token_rows, lengths, softmax_scale):
    lora = q_latent.shape[-1]
    return _fa3_with_kvcache()(
        q=q_pe,
        k_cache=flat_kv[..., lora:],
        v_cache=flat_kv[..., :lora],
        qv=q_latent,
        page_table=token_rows,
        cache_seqlens=lengths,
        softmax_scale=float(softmax_scale),
        causal=True,
        num_splits=_fa3_num_splits(),
        return_softmax_lse=False,
    )


# --------------------------------------------------------------------------- #
#  Blackwell                                                                   #
# --------------------------------------------------------------------------- #


def _fi_workspace(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    workspace = _FI_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.zeros(_FI_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _FI_WORKSPACES[key] = workspace
    return workspace


def _trtllm_sparse_decode(q_pe, q_latent, blocked_k, token_rows, lengths, softmax_scale):
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    batch, _, num_heads, lora = q_latent.shape
    rope = q_pe.shape[-1]
    pages, page_size, kv_heads, width = blocked_k.shape
    if kv_heads != 1 or width != lora + rope:
        raise SparseMlaShapeError(
            f"blocked_k must be [P, S, 1, {lora + rope}], got {tuple(blocked_k.shape)}"
        )
    query = torch.cat([q_latent, q_pe], dim=-1)
    out = trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=blocked_k.view(pages, page_size, width),
        workspace_buffer=_fi_workspace(query.device),
        # A runner-cache key only (flashinfer marks it for removal); the
        # absorbed latent query has no separate nope part.
        qk_nope_head_dim=lora,
        kv_lora_rank=lora,
        qk_rope_head_dim=rope,
        block_tables=token_rows.unsqueeze(1),
        # A zero-length row yields NaN; length 1 over its -1 slot yields 0.
        # Such rows are graph padding whose output nothing reads.
        seq_lens=lengths.clamp(min=1),
        max_seq_len=int(token_rows.shape[1]),
        sparse_mla_top_k=int(token_rows.shape[1]),
        bmm1_scale=float(softmax_scale),
        bmm2_scale=1.0,
        backend="trtllm-gen",
    )
    return out.view(batch, 1, num_heads, lora)


def _flashmla_sparse_prefill(q_pe, q_latent, kv_rows, token_rows, lengths, softmax_scale):
    from flash_mla import flash_mla_sparse_fwd

    total, _, num_heads, lora = q_latent.shape
    rope = q_pe.shape[-1]
    width = lora + rope
    if kv_rows.dim() != 2 or kv_rows.shape[-1] != width:
        raise SparseMlaShapeError(
            f"kv_rows must be [R, {width}], got {tuple(kv_rows.shape)}"
        )
    head_tile = next((t for t in _FLASHMLA_PREFILL_HEAD_TILES if num_heads <= t), None)
    if head_tile is None:
        raise SparseMlaShapeError(
            f"flash_mla sparse prefill runs at most {_FLASHMLA_PREFILL_HEAD_TILES[-1]} "
            f"query heads, got {num_heads}"
        )
    topk = int(token_rows.shape[1])
    if topk % _FLASHMLA_PREFILL_TOPK_BLOCK != 0:
        raise SparseMlaShapeError(
            f"flash_mla sparse prefill needs a top-k width that is a multiple of "
            f"{_FLASHMLA_PREFILL_TOPK_BLOCK}, got {topk}"
        )
    kv = kv_rows.view(-1, 1, width)
    out = q_latent.new_empty((total, 1, num_heads, lora))
    for start in range(0, total, _FLASHMLA_PREFILL_ROW_CHUNK):
        end = min(start + _FLASHMLA_PREFILL_ROW_CHUNK, total)
        rows = end - start
        query = q_latent.new_zeros((rows, head_tile, width))
        query[:, :num_heads, :lora] = q_latent[start:end, 0]
        query[:, :num_heads, lora:] = q_pe[start:end, 0]
        chunk = flash_mla_sparse_fwd(
            query,
            kv,
            token_rows[start:end].to(torch.int32).contiguous().view(rows, 1, topk),
            float(softmax_scale),
            d_v=lora,
            topk_length=lengths[start:end].to(torch.int32).contiguous(),
            use_legacy=True,
            return_log2_lse=False,
        )[0]
        out[start:end, 0] = chunk[:, :num_heads]
    return out


# --------------------------------------------------------------------------- #
#  The op                                                                      #
# --------------------------------------------------------------------------- #


def sparse_mla_decode(
    q_pe: torch.Tensor,
    q_latent: torch.Tensor,
    blocked_k: torch.Tensor,
    token_rows: torch.Tensor,
    lengths: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Decode: one query token per row over the paged latent cache."""
    if _backend() == "fa3":
        flat_kv = blocked_k.view(-1, 1, 1, blocked_k.shape[-1])
        return _fa3_sparse(q_pe, q_latent, flat_kv, token_rows, lengths, softmax_scale)
    return _trtllm_sparse_decode(q_pe, q_latent, blocked_k, token_rows, lengths, softmax_scale)


def sparse_mla_prefill(
    q_pe: torch.Tensor,
    q_latent: torch.Tensor,
    kv_rows: torch.Tensor,
    token_rows: torch.Tensor,
    lengths: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Prefill: every query token its own row over flat latent rows."""
    if _backend() == "fa3":
        flat_kv = kv_rows.view(-1, 1, 1, kv_rows.shape[-1])
        return _fa3_sparse(q_pe, q_latent, flat_kv, token_rows, lengths, softmax_scale)
    return _flashmla_sparse_prefill(q_pe, q_latent, kv_rows, token_rows, lengths, softmax_scale)
