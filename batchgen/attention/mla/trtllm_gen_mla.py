"""The Blackwell MLA decode launch shared by the dense and the sparse MLA seams.

Both ``paged_decode`` (dense, block-table rows) and ``sparse_dispatch``
(sparse, token-row selection) run flashinfer's functional
``trtllm_batch_decode_with_kv_cache_mla`` on the trtllm-gen backend. This
module owns what the two have in common, so there is one copy of it:

* the per-device workspace: zero-initialized once (the multi-CTA reduction
  keeps semaphore state in it) and reused by every call for the life of the
  process, so a captured graph bakes one fixed address;
* the length contract: ``seq_lens`` is the device-side active length of each
  row. A row of length 0 (graph padding) would produce NaN, so it runs as
  length 1 over its first block-table entry; its output is unspecified and
  must not be consumed;
* the shape contract of the latent cache ``blocked_k [P, S, 1, lora + rope]``
  and of the absorbed query ``[B, 1, H, lora + rope]``.

The launch takes only device tensors, shape-derived integers and the softmax
scale: no per-call plan and no host synchronization, which is what makes it
safe to capture in a CUDA graph.
"""

from __future__ import annotations

import torch

#: Per-device trtllm-gen MLA scratch (semaphores + multi-CTA partials).
_FI_WORKSPACES: dict = {}
_FI_WORKSPACE_BYTES = 128 * 1024 * 1024

#: Third-party callable the launch reaches (startup preload checks it).
SITE_SYMBOLS = {"flashinfer.mla": ("trtllm_batch_decode_with_kv_cache_mla",)}


class MlaShapeError(ValueError):
    """An input the selected MLA kernel cannot run."""


def workspace(device: torch.device) -> torch.Tensor:
    """The process-lifetime trtllm-gen workspace of ``device``."""
    key = (device.type, device.index)
    buffer = _FI_WORKSPACES.get(key)
    if buffer is None:
        buffer = torch.zeros(_FI_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _FI_WORKSPACES[key] = buffer
    return buffer


def trtllm_gen_mla_decode(
    query: torch.Tensor,
    blocked_k: torch.Tensor,
    block_tables: torch.Tensor,
    lengths: torch.Tensor,
    *,
    kv_lora_rank: int,
    max_seq_len: int,
    softmax_scale: float,
    sparse_top_k: int,
) -> torch.Tensor:
    """One trtllm-gen MLA decode launch; returns ``[B, 1, H, kv_lora_rank]``.

    ``block_tables`` is ``[B, pages]`` page ids (dense, ``sparse_top_k == 0``)
    or ``[B, 1, top_k]`` token-row ids (sparse). ``max_seq_len`` must be a
    static upper bound of ``lengths`` (a shape, never a device read).
    """
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    batch, q_len, num_heads, width = query.shape
    pages, page_size, kv_heads, kv_width = blocked_k.shape
    if kv_heads != 1 or kv_width != width:
        raise MlaShapeError(
            f"blocked_k must be [P, S, 1, {width}], got {tuple(blocked_k.shape)}"
        )
    out = trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=blocked_k.view(pages, page_size, kv_width),
        workspace_buffer=workspace(query.device),
        # A runner-cache key only (flashinfer marks it for removal); the
        # absorbed latent query has no separate nope part.
        qk_nope_head_dim=kv_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=width - kv_lora_rank,
        block_tables=block_tables,
        # A zero-length row yields NaN; length 1 over its first entry yields a
        # finite value. Such rows are graph padding whose output nothing reads.
        seq_lens=lengths.clamp(min=1),
        max_seq_len=int(max_seq_len),
        sparse_mla_top_k=int(sparse_top_k),
        bmm1_scale=float(softmax_scale),
        bmm2_scale=1.0,
        backend="trtllm-gen",
    )
    return out.view(batch, q_len, num_heads, kv_lora_rank)
