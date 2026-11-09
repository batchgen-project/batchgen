"""Capability-selected variable-length prefill attention."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

_BACKEND_CACHE: Optional[Callable[..., torch.Tensor]] = None
_BACKEND_CAPABILITY: Optional[tuple[int, int]] = None
_SDPA_FALLBACK_WARNED = False


def _detect_capability() -> tuple[int, int]:
    """CUDA capability of the active device; (9, 0) when CUDA is absent.

    Mirrors runtime_preflight's snapshot without importing it: attention is a
    lower layer than the preflight gate (no reverse edge), and both must agree
    because they read the same torch API.
    """
    if not torch.cuda.is_available():
        return (9, 0)
    return tuple(torch.cuda.get_device_capability())


def _warn_sdpa_fallback() -> None:
    global _SDPA_FALLBACK_WARNED
    if not _SDPA_FALLBACK_WARNED:
        logger.warning(
            "Variable-length prefill backend unavailable; using the SDPA "
            "reference fallback"
        )
        _SDPA_FALLBACK_WARNED = True


def _sdpa_varlen_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float],
    causal: bool,
) -> torch.Tensor:
    sdpa = getattr(F, "scaled_dot_product_attention", None)
    if sdpa is None:
        raise RuntimeError("PyTorch SDPA is unavailable for varlen attention")

    del max_seqlen_q, max_seqlen_k
    outputs = []
    q_bounds = cu_seqlens_q.tolist()
    k_bounds = cu_seqlens_k.tolist()
    for q_start, q_end, k_start, k_end in zip(
        q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]
    ):
        q_seq = q[q_start:q_end]
        k_seq = k[k_start:k_end]
        v_seq = v[k_start:k_end]
        if q_seq.shape[0] == 0:
            outputs.append(q_seq.new_empty((0, q.shape[1], v.shape[2])))
            continue

        q_len, k_len = q_seq.shape[0], k_seq.shape[0]
        attn_mask = None
        if causal:
            attn_mask = torch.ones(
                (q_len, k_len), dtype=torch.bool, device=q.device
            ).tril(diagonal=k_len - q_len)
        sdpa_kwargs = {"attn_mask": attn_mask}
        if softmax_scale is not None:
            sdpa_kwargs["scale"] = softmax_scale
        out = sdpa(
            q_seq.transpose(0, 1).unsqueeze(0),
            k_seq.transpose(0, 1).unsqueeze(0),
            v_seq.transpose(0, 1).unsqueeze(0),
            **sdpa_kwargs,
        )
        outputs.append(out.squeeze(0).transpose(0, 1).contiguous())

    if not outputs:
        return q.new_empty((0, q.shape[1], v.shape[2]))
    return torch.cat(outputs, dim=0)


def _flashinfer_varlen_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float],
    causal: bool,
) -> torch.Tensor:
    import flashinfer
    from flashinfer.prefill import single_prefill_with_kv_cache

    del max_seqlen_q, max_seqlen_k, flashinfer
    outputs = []
    q_bounds = cu_seqlens_q.tolist()
    k_bounds = cu_seqlens_k.tolist()
    for q_start, q_end, k_start, k_end in zip(
        q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]
    ):
        q_seq = q[q_start:q_end]
        if q_seq.shape[0] == 0:
            outputs.append(q_seq.new_empty((0, q.shape[1], v.shape[2])))
            continue
        kwargs = {"causal": causal}
        if softmax_scale is not None:
            kwargs["sm_scale"] = softmax_scale
        out = single_prefill_with_kv_cache(
            q_seq,
            k[k_start:k_end],
            v[k_start:k_end],
            **kwargs,
        )
        outputs.append(out[0] if isinstance(out, tuple) else out)

    if not outputs:
        return q.new_empty((0, q.shape[1], v.shape[2]))
    return torch.cat(outputs, dim=0)


def _select_backend(capability: tuple[int, int]) -> Callable[..., torch.Tensor]:
    if capability[0] == 9:
        try:
            from flash_attn_interface import flash_attn_varlen_func

            return flash_attn_varlen_func
        except (ImportError, AttributeError):
            _warn_sdpa_fallback()
            return _sdpa_varlen_prefill
    if capability[0] == 10:
        return _flashinfer_varlen_prefill
    _warn_sdpa_fallback()
    return _sdpa_varlen_prefill


def _backend_for_capability(
    capability: tuple[int, int],
) -> Callable[..., torch.Tensor]:
    global _BACKEND_CACHE, _BACKEND_CAPABILITY
    if _BACKEND_CACHE is None or _BACKEND_CAPABILITY != capability:
        _BACKEND_CACHE = _select_backend(capability)
        _BACKEND_CAPABILITY = capability
    return _BACKEND_CACHE


def varlen_attention_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Run the K3 prepacked varlen attention operation."""
    global _BACKEND_CACHE, _BACKEND_CAPABILITY
    capability = _detect_capability()
    backend = _backend_for_capability(capability)
    try:
        output = backend(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    except ImportError:
        if capability[0] != 10 or backend is not _flashinfer_varlen_prefill:
            raise
        _warn_sdpa_fallback()
        _BACKEND_CACHE = _sdpa_varlen_prefill
        _BACKEND_CAPABILITY = capability
        output = _BACKEND_CACHE(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    return output[0] if isinstance(output, tuple) else output
