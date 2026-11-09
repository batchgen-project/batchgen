"""Dense paged MLA decode: one semantic op, capability-selected kernel.

Contract (all architectures):
    query        [B, 1, H, D]  BF16, D = head_dim_ckv + head_dim_kpe (576)
    blocked_k    [P, S, 1, D]  BF16 packed paged latent cache (page size S)
    block_table  [B, max_pages] int32
    cache_seqlens[B]           int32, context length incl. the new token
    returns      [B, 1, H, head_dim_v] BF16, head_dim_v = head_dim_ckv (512)

Backends (resolved once per process from the device capability):
    9.x  -> flash_mla dense decode (Hopper kernels; the current upstream
            wheel keeps them on its legacy side behind use_legacy=True);
    10.x -> flashinfer BatchMLAPagedAttentionWrapper, trtllm-gen backend
            (Blackwell-native; packed query/KV views, no copies). The
            current flash_mla wheel has no dense decode for 10.x.
Anything else fails loud.
"""

from __future__ import annotations

import inspect
from typing import Optional

import torch

_BACKEND: Optional[str] = None
_FLASHMLA_DENSE_KWARGS: Optional[dict] = None
_FI_WRAPPERS: dict = {}
_FI_WORKSPACE_BYTES = 256 * 1024 * 1024


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

    flash_mla decode takes precomputed scheduler metadata and captures; the
    flashinfer trtllm-gen path plans per call and has no graph-safe variant
    in BatchGen yet, so decode on Blackwell runs eager.
    """
    return _backend() == "flashmla"


def flashmla_dense_kwargs() -> dict:
    """Extra kwargs a flash_mla dense-decode call needs for the installed package."""
    global _FLASHMLA_DENSE_KWARGS
    if _FLASHMLA_DENSE_KWARGS is None:
        from flash_mla import flash_mla_with_kvcache

        parameters = inspect.signature(flash_mla_with_kvcache).parameters
        _FLASHMLA_DENSE_KWARGS = {"use_legacy": True} if "use_legacy" in parameters else {}
    return _FLASHMLA_DENSE_KWARGS


def _flashinfer_wrapper(device: torch.device):
    key = (device.type, device.index)
    wrapper = _FI_WRAPPERS.get(key)
    if wrapper is None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        workspace = torch.empty(_FI_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="trtllm-gen")
        _FI_WRAPPERS[key] = wrapper
    return wrapper


def mla_paged_decode(
    query: torch.Tensor,
    blocked_k: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
) -> torch.Tensor:
    batch, q_len, num_heads, head_dim = query.shape
    assert q_len == 1, "dense paged MLA decode takes one query token per sequence"
    backend = _backend()

    if backend == "flashmla":
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        extra = flashmla_dense_kwargs()
        metadata, num_splits = get_mla_metadata(cache_seqlens, num_heads, 1, **extra)
        out, _ = flash_mla_with_kvcache(
            query, blocked_k, block_table, cache_seqlens, head_dim_v,
            metadata, num_splits, softmax_scale, True, **extra,
        )
        return out

    wrapper = _flashinfer_wrapper(query.device)
    page_size = blocked_k.shape[1]
    wrapper.plan(
        num_heads=num_heads,
        head_dim_ckv=head_dim_v,
        head_dim_kpe=head_dim - head_dim_v,
        page_size=page_size,
        causal=True,
        sm_scale=softmax_scale,
        q_data_type=query.dtype,
        kv_data_type=blocked_k.dtype,
        cum_seq_lens_q=torch.arange(batch + 1, dtype=torch.int32, device=query.device),
        block_tables=block_table,
        seq_lens=cache_seqlens,
        max_q_len=1,
        query_layout="packed",
        kv_cache_layout="packed",
    )
    out = query.new_empty((batch, num_heads, head_dim_v))
    wrapper.run(query=query[:, 0], kv_cache=blocked_k[:, :, 0], out=out)
    return out.unsqueeze(1)
