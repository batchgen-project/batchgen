# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash model definition.

This file is intentionally self-contained.  It mirrors the V4 tensor names from
``assets/inference/model.py`` while exposing the BatchGen worker contract:
``ForCausalLM.model``, ``ForCausalLM.lm_head``, ``model.embed_tokens``,
``model.layers``, ``model.norm``, ``layer.self_attn`` and ``layer.mlp``.

The structure is DP-attention + EP-MoE oriented: attention modules hold full
head projections, and MoE layers expose global expert slots that the parallel
strategy manager assigns to per-rank expert-parallel ranges.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _v4_diag(message: str) -> None:
    if os.environ.get("BATCHGEN_V4_DIAG", "0") != "1":
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(f"[V4-DIAG rank={rank}] {message}", flush=True)


def _v4_timing_enabled() -> bool:
    return os.environ.get("BATCHGEN_V4_TIMING", "0") == "1"


def _v4_timing(message: str) -> None:
    if not _v4_timing_enabled():
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(f"[V4-TIMING rank={rank}] {message}", flush=True)


def _v4_sync_time(device: torch.device | None = None) -> float:
    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()
    return time.perf_counter()


_FP4_E2M1_TABLE_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


@dataclass
class _CausalLMOutput:
    """Minimal output container with ``.logits`` for BatchGen workers."""

    logits: torch.Tensor


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _cfg_rope(config: Any, name: str, default: Any) -> Any:
    rope_scaling = _cfg(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and name in rope_scaling:
        return rope_scaling[name]
    return _cfg(config, name, default)


def _linear_ramp_factor(start: int, stop: int, dim: int, device: torch.device) -> torch.Tensor:
    if start == stop:
        stop += 1
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - start) / (stop - start)
    return torch.clamp(ramp, 0, 1)


def _find_yarn_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    max_seq_len: int,
) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_yarn_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_seq_len: int,
) -> Tuple[int, int]:
    low = math.floor(_find_yarn_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_yarn_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _build_rope_freqs_cis(
    positions: torch.Tensor,
    dim: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    original_seq_len: int,
) -> torch.Tensor:
    device = positions.device
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    if original_seq_len > 0:
        low, high = _find_yarn_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - _linear_ramp_factor(low, high, dim // 2, device)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    phase = positions.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.polar(torch.ones_like(phase), phase)


def _apply_rotary_emb_inplace(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> None:
    rope_dim = x.shape[-1]
    if inverse:
        freqs_cis = freqs_cis.conj()
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if x.dim() == 4:
        if freqs_cis.dim() == 2:
            freqs_view = freqs_cis.view(1, freqs_cis.size(0), 1, freqs_cis.size(1))
        else:
            freqs_view = freqs_cis.unsqueeze(2)
    elif x.dim() == 3:
        if freqs_cis.dim() == 2:
            freqs_view = freqs_cis.view(1, freqs_cis.size(0), freqs_cis.size(1))
        else:
            freqs_view = freqs_cis
    else:
        raise RuntimeError(f"Unsupported V4 rotary tensor rank: {x.dim()}")
    rotated = torch.view_as_real(x_complex * freqs_view).flatten(-2)
    x.copy_(rotated.to(dtype=x.dtype).view(*x.shape[:-1], rope_dim))


# ---------------------------------------------------------------------------
# QAT (quantization-aware-training) simulation helpers
# ---------------------------------------------------------------------------
#
# The DeepSeek-V4-Flash reference implementation at
# `assets/inference/model.py` applies these three operations on K, V, and the
# indexer's q at every cache write — the comment at line 505 of that file
# makes the intent explicit:
#
#   # FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for
#   # positional precision
#
# The model was trained with these QAT simulations applied. Skipping them
# at inference produces drift from the trained activation distribution that
# accumulates over decode steps and shows up as "model emits 2-5 valid
# tokens then degenerates". See the call sites in DeepSeekV4FlashAttention,
# DeepSeekV4FlashCompressor (rotate=True / rotate=False), and
# DeepSeekV4FlashIndexer below.
#
# Numerical spec is taken from `assets/inference/kernel.py:105-201`. We
# match: per-block amax, scale = amax / fp_max (with FP4 scales rounded to
# power-of-2), in-place round-trip dequant via the matching torch FP8 dtype.
# PyTorch lacks a native FP4 dtype so we enumerate the 15 E2M1FN values.

# matches assets/inference/kernel.py and the call sites at model.py:506, 372
_QAT_FP8_BLOCK_SIZE = 64
# matches `fp4_block_size = 32` at assets/inference/model.py:18
_QAT_FP4_BLOCK_SIZE = 32

# FP4 E2M1FN representable magnitudes; sign added separately. Values are the
# fixed point set the reference's tilelang fp4_quant_kernel rounds to.
_QAT_FP4_VALUES = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
                   0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _qat_fp8_act_quant_inplace(
    x: torch.Tensor, block_size: int = _QAT_FP8_BLOCK_SIZE
) -> None:
    """Per-block FP8 E4M3 round-trip in place, matching the reference's
    ``act_quant(x, block_size, ..., inplace=True)`` at kernel.py:105-126.

    Computes per-block amax (last dim, blocks of ``block_size``), scale =
    amax / 448, casts to ``float8_e4m3fn`` (round-to-nearest at FP8
    boundaries), dequantises by multiplying by scale, casts back to the
    input dtype, and writes the result back into ``x``. Equivalent to the
    QAT noise applied during V4-Flash training.
    """
    if x.numel() == 0:
        return
    last = x.shape[-1]
    if last % block_size != 0:
        raise ValueError(
            f"_qat_fp8_act_quant_inplace: last dim {last} must be a multiple "
            f"of block_size {block_size}"
        )
    orig_dtype = x.dtype
    *prefix, _ = x.shape
    blocks = x.view(*prefix, last // block_size, block_size).float()
    # Floor at 1e-12 (denormal-style); keeps tiny-norm blocks at full
    # precision instead of squashing to zero. The reference kernel uses 1e-4
    # for tilelang numerical stability, not as a QAT spec.
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    fp8_max = 448.0
    # Clamp the scaled value into the FP8 representable range *before* the
    # cast. PyTorch's cast to float8_e4m3fn on an out-of-range FP32 value
    # can saturate to inf/NaN, silently corrupting cache writes.
    quantised = (blocks / scale).clamp_(-fp8_max, fp8_max)
    quantised = quantised.to(torch.float8_e4m3fn).to(torch.float32)
    dequantised = (quantised * scale).to(orig_dtype)
    x.copy_(dequantised.view_as(x))


def _qat_fp4_act_quant_inplace(
    x: torch.Tensor, block_size: int = _QAT_FP4_BLOCK_SIZE
) -> None:
    """Per-block FP4 E2M1FN round-trip in place, matching the reference's
    ``fp4_act_quant(x, block_size, inplace=True)`` at kernel.py:186-201.

    Per-block amax / 6 → scale, scale rounded *up* to the next power-of-2,
    block elements rounded to the nearest of the 15 FP4 magnitudes (with
    sign), then multiplied back by scale. PyTorch has no native FP4 dtype,
    so the round step enumerates representable values explicitly.
    """
    if x.numel() == 0:
        return
    last = x.shape[-1]
    if last % block_size != 0:
        raise ValueError(
            f"_qat_fp4_act_quant_inplace: last dim {last} must be a multiple "
            f"of block_size {block_size}"
        )
    orig_dtype = x.dtype
    *prefix, _ = x.shape
    blocks = x.view(*prefix, last // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    raw_scale = amax / 6.0
    # Round up to the next power-of-2; matches kernel.py:164 fast_round_scale.
    pow2_scale = torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))
    scaled = (blocks / pow2_scale).clamp_(-6.0, 6.0)
    fp4_values = torch.tensor(_QAT_FP4_VALUES, dtype=torch.float32, device=x.device)
    diffs = (scaled.unsqueeze(-1) - fp4_values).abs()
    rounded = fp4_values[diffs.argmin(dim=-1)]
    dequantised = (rounded * pow2_scale).to(orig_dtype)
    x.copy_(dequantised.view_as(x))


def _qat_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Hadamard rotation matching the reference's ``rotate_activation`` at
    assets/inference/model.py:247-251. The reference imports
    ``fast_hadamard_transform.hadamard_transform``; we prefer that module
    when available (CPU+GPU, exact same kernel), then fall back to
    BatchGen's CUDA wrapper (GPU-only), then to a pure-PyTorch Walsh-
    Hadamard butterfly (CPU) for parity tests.
    """
    if x.dtype != torch.bfloat16:
        raise RuntimeError(
            f"_qat_hadamard_rotate expects bf16; got {x.dtype}"
        )
    scale = x.size(-1) ** -0.5
    try:
        from fast_hadamard_transform import hadamard_transform as _ext_hadamard

        return _ext_hadamard(x, scale=scale)
    except ImportError:
        pass
    if x.is_cuda:
        from batchgen.other_kernels.hadamard_transform import (
            hadamard_transform as _bg_hadamard,
        )

        return _bg_hadamard(x, scale=scale)
    return _qat_hadamard_python_cpu(x, scale)


def _qat_hadamard_python_cpu(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Pure-PyTorch Walsh-Hadamard along the last dim, multiplied by ``scale``.

    Used only as a CPU fallback for parity tests when neither
    ``fast_hadamard_transform`` nor BatchGen's CUDA Hadamard kernel is
    available. Last dim must be a power of two.
    """
    last = x.shape[-1]
    if last & (last - 1):
        raise ValueError(
            f"_qat_hadamard_python_cpu requires a power-of-2 last dim, got {last}"
        )
    y = x.float().contiguous()
    h = 1
    while h < last:
        # Reshape so the transform pair lives along dim -2.
        groups = last // (2 * h)
        view = y.view(*y.shape[:-1], groups, 2, h)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        y = view.reshape(*y.shape[:-1], last)
        h *= 2
    return (y * scale).to(x.dtype)


def _linear_from_weight(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linear helper for V4 slots.

    FP8 checkpoint tensors are block scaled.  This fallback dequantizes them in
    PyTorch for correctness-oriented bring-up; optimized wrappers/kernels replace
    this path in production.
    """

    raw_weight_shape = tuple(weight.shape)
    weight = _dequant_weight(weight, scale, x.dtype)
    if x.shape[-1] != weight.shape[-1]:
        scale_shape = None if scale is None else tuple(scale.shape)
        raise RuntimeError(
            "DeepSeek-V4 linear shape mismatch: "
            f"input={tuple(x.shape)}, weight={tuple(weight.shape)}, "
            f"raw_weight={raw_weight_shape}, scale={scale_shape}"
        )
    return F.linear(x, weight, bias)


def _window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    device: torch.device,
) -> torch.Tensor:
    if start_pos >= window_size - 1:
        pos = start_pos % window_size
        matrix = torch.cat(
            [
                torch.arange(pos + 1, window_size, device=device),
                torch.arange(0, pos + 1, device=device),
            ],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1, device=device),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        matrix = (
            (base - window_size + 1).clamp(0)
            + torch.arange(min(seqlen, window_size), device=device)
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _compress_topk_idxs(
    ratio: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio, device=device) + offset
    else:
        matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _sparse_attn_from_topk(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if topk_idxs.numel() == 0 or topk_idxs.size(-1) == 0:
        return torch.zeros_like(q)
    bsz, seqlen, _, head_dim = q.shape
    valid = topk_idxs >= 0
    clamped = topk_idxs.clamp_min(0).long()
    gather_idx = clamped.unsqueeze(-1).expand(bsz, seqlen, topk_idxs.size(-1), head_dim)
    kv_rows = torch.gather(
        kv.unsqueeze(1).expand(-1, seqlen, -1, -1),
        2,
        gather_idx,
    )
    scores = torch.einsum("bshd,bskd->bhsk", q.float(), kv_rows.float()) * softmax_scale
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~valid[:, None, :, :], neg_inf)
    sink = attn_sink.float().view(1, q.size(2), 1, 1).to(scores.device)
    scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(scores - scores_max).masked_fill(~valid[:, None, :, :], 0)
    denom = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max)
    probs = (exp_scores / denom).to(q.dtype)
    return torch.einsum("bhsk,bskd->bshd", probs, kv_rows.to(q.dtype))


def _dequant_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if _is_fp4_e2m1_weight(weight, scale):
        return _dequant_fp4_e2m1_weight(weight, scale, dtype)
    if scale is not None and scale.ndim == 2 and weight.ndim == 2:
        row_block = max(weight.shape[0] // scale.shape[0], 1)
        col_block = max(weight.shape[1] // scale.shape[1], 1)
        expanded_scale = scale.to(torch.float32).repeat_interleave(
            row_block, dim=0
        ).repeat_interleave(col_block, dim=1)
        expanded_scale = expanded_scale[: weight.shape[0], : weight.shape[1]]
        return (weight.to(torch.float32) * expanded_scale).to(dtype)
    return weight.to(dtype)


def _is_fp4_e2m1_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
) -> bool:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None and weight.dtype == fp4_dtype:
        return True
    if weight.dtype in (torch.int8, torch.uint8):
        return True
    return (
        scale is not None
        and weight.ndim == 2
        and scale.ndim == 2
        and weight.shape[0] == scale.shape[0]
        and weight.shape[1] == scale.shape[1] * 16
    )


def _fp4_packed_bytes(weight: torch.Tensor) -> torch.Tensor:
    if weight.element_size() == 1:
        return weight.contiguous().view(torch.uint8)
    return weight.contiguous().to(torch.uint8)


def _dequant_fp4_e2m1_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if scale is None:
        raise RuntimeError("DeepSeek-V4 FP4 weight is missing its E8M0 scale tensor.")
    if (
        weight.is_cuda
        and scale.is_cuda
        and os.environ.get("BATCHGEN_V4_DISABLE_CUTE_FP4_DEQUANT", "0") != "1"
    ):
        return _dequant_fp4_e2m1_weight_cute(weight, scale, dtype)
    packed = _fp4_packed_bytes(weight)
    table = torch.tensor(
        _FP4_E2M1_TABLE_VALUES,
        dtype=torch.float32,
        device=packed.device,
    )
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    unpacked = torch.empty(unpacked_shape, dtype=torch.float32, device=packed.device)
    unpacked[..., 0::2] = table[low.long()]
    unpacked[..., 1::2] = table[high.long()]

    expanded_scale = scale.to(torch.float32).unsqueeze(-1).expand(
        *scale.shape, 32
    ).reshape(*scale.shape[:-1], scale.shape[-1] * 32)
    expanded_scale = expanded_scale[..., : unpacked.shape[-1]]
    return (unpacked * expanded_scale).to(dtype)


def _dequant_fp4_e2m1_weight_cute(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    from batchgen.moe.cute_mxfp4_dequant import mxfp4_dequant_single_expert_cute

    packed = _fp4_packed_bytes(weight).contiguous()
    if scale.element_size() != 1:
        raise RuntimeError(
            "DeepSeek-V4 CUTE FP4 dequant expects one-byte E8M0 scales, "
            f"got dtype={scale.dtype} element_size={scale.element_size()}"
        )
    scale_kmajor = scale.contiguous().view(torch.uint8).t().contiguous()
    output = torch.empty(
        packed.shape[0],
        packed.shape[1] * 2,
        dtype=torch.bfloat16,
        device=packed.device,
    )
    mxfp4_dequant_single_expert_cute(packed, scale_kmajor, output)
    return output.to(dtype) if output.dtype != dtype else output


class DeepSeekV4FlashLinearSlot(nn.Module):
    """Runtime-loaded linear slot.

    Attention and expert bundle tensors are owned by the BatchGen parameter
    server/wrappers, not by the skeleton state dict.  The slot records shape
    metadata and receives tensors at wrapper execution time.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor], prefix: str) -> None:
        self.weight = tensors.get(f"{prefix}.weight")
        self.scale = tensors.get(f"{prefix}.scale")

    def clear_runtime_tensors(self) -> None:
        self.weight = None
        self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError(
                f"DeepSeek-V4 linear slot ({self.out_features}, {self.in_features}) "
                "has no runtime weight loaded."
            )
        return _linear_from_weight(x, self.weight, self.scale, self.bias)


class DeepSeekV4FlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.float()).to(dtype)


class DeepSeekV4FlashCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        eps: float,
        overlap: bool = False,
        rotate: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        # When rotate=True (only the indexer's compressor sets this), the
        # reference applies a Hadamard + FP4 round-trip on the compressed
        # kv before writing to cache (assets/inference/model.py:368-370,
        # via Compressor(rotate=True)). When False, it applies an FP8
        # round-trip on the non-rope dims (line 372). See _qat_*_inplace
        # helpers near the top of this file.
        self.rotate = rotate
        coeff = 2 if overlap else 1
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coeff * head_dim, dtype=torch.float32)
        )
        self.wkv = DeepSeekV4FlashLinearSlot(hidden_size, coeff * head_dim)
        self.wgate = DeepSeekV4FlashLinearSlot(hidden_size, coeff * head_dim)
        self.norm = DeepSeekV4FlashRMSNorm(head_dim, eps)
        self.kv_cache: Optional[torch.Tensor] = None
        self.kv_state: Optional[torch.Tensor] = None
        self.score_state: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor], prefix: str) -> None:
        self.wkv.set_runtime_tensors(tensors, f"{prefix}.wkv")
        self.wgate.set_runtime_tensors(tensors, f"{prefix}.wgate")
        if f"{prefix}.norm.weight" in tensors:
            self.norm.weight.data = tensors[f"{prefix}.norm.weight"].to(
                self.norm.weight.device
            )
        if f"{prefix}.ape" in tensors:
            self.ape.data = tensors[f"{prefix}.ape"].to(self.ape.device)

    def clear_runtime_tensors(self) -> None:
        self.wkv.clear_runtime_tensors()
        self.wgate.clear_runtime_tensors()

    def _ensure_state(self, bsz: int, device: torch.device) -> None:
        coeff = 2 if self.overlap else 1
        shape = (bsz, coeff * self.compress_ratio, coeff * self.head_dim)
        if (
            self.kv_state is None
            or self.kv_state.shape != shape
            or self.kv_state.device != device
        ):
            self.kv_state = torch.zeros(shape, dtype=torch.float32, device=device)
            self.score_state = torch.full(
                shape,
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )

    def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
        if self.kv_cache is None:
            raise RuntimeError("DeepSeek-V4 compressor KV cache is not initialized.")
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        coeff = 2 if self.overlap else 1
        d = self.head_dim
        rd = self.rope_head_dim
        dtype = x.dtype
        self._ensure_state(bsz, x.device)

        kv = self.wkv(x).float()
        score = self.wgate(x).float()
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if self.overlap else 0
            if self.overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape.float()
                )
            if remainder > 0:
                kv, kv_remainder = kv.split([cutoff, remainder], dim=1)
                self.kv_state[:bsz, offset : offset + remainder] = kv_remainder
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder].float()
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape.float()
            if self.overlap:
                kv = self._overlap_transform(kv, 0)
                score = self._overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            score = score + self.ape[start_pos % ratio].float()
            if self.overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1,
                        keepdim=True,
                    )
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)
        if not should_compress:
            return None

        kv = self.norm(kv.to(dtype))
        if self.freqs_cis is None:
            if start_pos == 0:
                positions = torch.arange(0, cutoff, ratio, device=x.device)
            else:
                positions = torch.tensor([start_pos + 1 - ratio], device=x.device)
            freqs_cis = _build_rope_freqs_cis(
                positions,
                rd,
                40000.0,
                1.0,
                32,
                1,
                0,
            )
        elif start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        _apply_rotary_emb_inplace(kv[..., -rd:], freqs_cis)
        # QAT simulation, matching reference assets/inference/model.py:367-372.
        # rotate=True branch is the indexer's compressor; rotate=False is the
        # main attention's compressor. See _qat_*_inplace helpers.
        if self.rotate:
            kv = _qat_hadamard_rotate(kv)
            _qat_fp4_act_quant_inplace(kv)
        else:
            _qat_fp8_act_quant_inplace(kv[..., :-rd])
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv

    def _overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        bsz, seqlen, _, _ = tensor.shape
        ratio = self.compress_ratio
        d = self.head_dim
        new_tensor = tensor.new_full((bsz, seqlen, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor


class DeepSeekV4FlashIndexer(nn.Module):
    def __init__(self, config: Any, compress_ratio: int):
        super().__init__()
        hidden_size = int(_cfg(config, "hidden_size", _cfg(config, "dim", 4096)))
        q_lora_rank = int(_cfg(config, "q_lora_rank", 1024))
        head_dim = int(_cfg(config, "index_head_dim", 128))
        n_heads = int(_cfg(config, "index_n_heads", 64))
        rope_head_dim = int(_cfg(config, "qk_rope_head_dim", _cfg(config, "rope_head_dim", 64)))
        eps = float(_cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6)))

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = int(_cfg(config, "index_topk", 512))
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim ** -0.5
        self.wq_b = DeepSeekV4FlashLinearSlot(q_lora_rank, n_heads * head_dim)
        self.weights_proj = DeepSeekV4FlashLinearSlot(hidden_size, n_heads)
        self.compressor = DeepSeekV4FlashCompressor(
            hidden_size,
            head_dim,
            rope_head_dim,
            compress_ratio,
            eps,
            overlap=True,
            # The indexer's compressor uses Hadamard + FP4 simulation
            # (assets/inference/model.py:398 passes True for rotate).
            rotate=True,
        )
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor], prefix: str) -> None:
        self.wq_b.set_runtime_tensors(tensors, f"{prefix}.wq_b")
        self.weights_proj.set_runtime_tensors(tensors, f"{prefix}.weights_proj")
        self.compressor.set_runtime_tensors(tensors, f"{prefix}.compressor")

    def clear_runtime_tensors(self) -> None:
        self.wq_b.clear_runtime_tensors()
        self.weights_proj.clear_runtime_tensors()
        self.compressor.clear_runtime_tensors()

    def _ensure_cache(self, bsz: int, end_pos: int, device: torch.device, dtype: torch.dtype) -> None:
        cache_len = max(end_pos // self.compress_ratio, 1)
        if (
            self.kv_cache is None
            or self.kv_cache.size(0) < bsz
            or self.kv_cache.size(1) < cache_len
            or self.kv_cache.device != device
            or self.kv_cache.dtype != dtype
        ):
            old_cache = self.kv_cache
            new_cache = torch.zeros(
                bsz,
                cache_len,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            # Preserve previously written compressed KV. Without this copy
            # the indexer's cache is wiped every time cache_len grows (every
            # compress_ratio decode steps), so top-k scoring sees zeros and
            # the model degenerates after a handful of tokens.
            if old_cache is not None:
                copy_bsz = min(bsz, old_cache.size(0))
                copy_len = min(cache_len, old_cache.size(1))
                if (
                    copy_bsz > 0
                    and copy_len > 0
                    and old_cache.device == device
                    and old_cache.dtype == dtype
                ):
                    new_cache[:copy_bsz, :copy_len] = old_cache[
                        :copy_bsz,
                        :copy_len,
                    ]
            self.kv_cache = new_cache
        self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen
        ratio = self.compress_ratio
        self._ensure_cache(bsz, end_pos, x.device, x.dtype)

        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.head_dim)
        if self.freqs_cis is None:
            positions = torch.arange(start_pos, end_pos, device=x.device)
            freqs_cis = _build_rope_freqs_cis(
                positions,
                min(self.head_dim, _cfg(self.compressor, "rope_head_dim", self.head_dim)),
                40000.0,
                1.0,
                32,
                1,
                0,
            )
        else:
            freqs_cis = self.freqs_cis[start_pos:end_pos]
        rd = min(self.compressor.rope_head_dim, self.head_dim)
        _apply_rotary_emb_inplace(q[..., -rd:], freqs_cis)
        # QAT: Hadamard + FP4 round-trip on q, matching reference
        # assets/inference/model.py:414-416. Mirrors the indexer's
        # compressor's rotate=True branch so q · kv stays consistent under
        # quant noise (the einsum below would otherwise mix un-rotated q
        # with rotated kv_cache).
        q = _qat_hadamard_rotate(q)
        _qat_fp4_act_quant_inplace(q)

        self.compressor(x, start_pos)
        cache_len = end_pos // ratio
        if cache_len == 0:
            return x.new_empty(bsz, seqlen, 0, dtype=torch.long)
        weights = self.weights_proj(x).float() * (
            self.softmax_scale * self.n_heads ** -0.5
        )
        index_score = torch.einsum(
            "bshd,btd->bsht",
            q.float(),
            self.kv_cache[:bsz, :cache_len].float(),
        )
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(cache_len, device=x.device).repeat(seqlen, 1)
            mask = mask >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score = index_score + torch.where(mask, float("-inf"), 0).unsqueeze(0)
        k = min(self.index_topk, cache_len)
        topk_idxs = index_score.topk(k, dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=x.device).view(1, seqlen, 1) // ratio
            return torch.where(mask, -1, topk_idxs + offset)
        return topk_idxs + offset


class DeepSeekV4FlashAttention(nn.Module):
    """DP attention surface for V4.

    All V4 projection tensors use their checkpoint names as attributes.  The
    optimized sparse/compressed attention implementation is attached by the V4
    attention wrapper; this module also carries a small PyTorch fallback for
    early smoke tests on short prompts.
    """

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(_cfg(config, "hidden_size", _cfg(config, "dim", 4096)))
        self.n_heads = int(_cfg(config, "num_attention_heads", _cfg(config, "n_heads", 64)))
        self.head_dim = int(_cfg(config, "head_dim", 512))
        self.q_lora_rank = int(_cfg(config, "q_lora_rank", 1024))
        self.o_groups = int(_cfg(config, "o_groups", 8))
        self.o_lora_rank = int(_cfg(config, "o_lora_rank", 1024))
        self.rope_head_dim = int(_cfg(config, "qk_rope_head_dim", _cfg(config, "rope_head_dim", 64)))
        self.eps = float(_cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6)))
        self.softmax_scale = self.head_dim ** -0.5
        self.window_size = int(_cfg(config, "sliding_window", _cfg(config, "window_size", 128)))

        ratios = list(_cfg(config, "compress_ratios", []))
        self.compress_ratio = int(ratios[layer_idx]) if layer_idx < len(ratios) else 0
        self.rope_theta = float(
            _cfg(config, "compress_rope_theta", 40000.0)
            if self.compress_ratio
            else _cfg(config, "rope_theta", 10000.0)
        )
        self.rope_factor = float(_cfg_rope(config, "factor", 1.0))
        self.beta_fast = float(_cfg_rope(config, "beta_fast", 32))
        self.beta_slow = float(_cfg_rope(config, "beta_slow", 1))
        self.original_seq_len = (
            int(_cfg_rope(config, "original_max_position_embeddings", 0))
            if self.compress_ratio
            else 0
        )

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = DeepSeekV4FlashLinearSlot(self.hidden_size, self.q_lora_rank)
        self.q_norm = DeepSeekV4FlashRMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = DeepSeekV4FlashLinearSlot(
            self.q_lora_rank, self.n_heads * self.head_dim
        )
        self.wkv = DeepSeekV4FlashLinearSlot(self.hidden_size, self.head_dim)
        self.kv_norm = DeepSeekV4FlashRMSNorm(self.head_dim, self.eps)
        self.wo_a = DeepSeekV4FlashLinearSlot(
            self.n_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
        )
        self.wo_b = DeepSeekV4FlashLinearSlot(
            self.o_groups * self.o_lora_rank, self.hidden_size
        )

        if self.compress_ratio:
            rope_head_dim = int(_cfg(config, "qk_rope_head_dim", _cfg(config, "rope_head_dim", 64)))
            self.compressor = DeepSeekV4FlashCompressor(
                self.hidden_size,
                self.head_dim,
                rope_head_dim,
                self.compress_ratio,
                self.eps,
                overlap=self.compress_ratio == 4,
            )
            self.indexer = (
                DeepSeekV4FlashIndexer(config, self.compress_ratio)
                if self.compress_ratio == 4
                else None
            )
            self._compress_kv_cache: Optional[torch.Tensor] = None
            self._compress_freqs_cis: Optional[torch.Tensor] = None
        else:
            self.compressor = None
            self.indexer = None
            self._compress_kv_cache = None
            self._compress_freqs_cis = None

    def _positions_for_rope(
        self,
        *,
        bsz: int,
        q_len: int,
        position_ids: Optional[torch.Tensor],
        cache_seqlens: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if position_ids is not None:
            positions = position_ids.to(device=device, dtype=torch.long)
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(bsz, -1)
            return positions[:, -q_len:]
        if cache_seqlens is not None and q_len == 1:
            return (cache_seqlens.to(device=device, dtype=torch.long) - 1).clamp_min(0).view(bsz, 1)
        return torch.arange(q_len, device=device, dtype=torch.long).view(1, q_len).expand(bsz, q_len)

    def _apply_rope(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor],
        cache_seqlens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.rope_head_dim <= 0:
            return None
        positions = self._positions_for_rope(
            bsz=q.size(0),
            q_len=q.size(1),
            position_ids=position_ids,
            cache_seqlens=cache_seqlens,
            device=q.device,
        )
        flat_positions = positions.reshape(-1)
        freqs = _build_rope_freqs_cis(
            flat_positions,
            self.rope_head_dim,
            self.rope_theta,
            self.rope_factor,
            self.beta_fast,
            self.beta_slow,
            self.original_seq_len,
        ).view(*positions.shape, self.rope_head_dim // 2)
        if positions.size(0) == 1 and q.size(0) != 1:
            freqs = freqs.expand(q.size(0), -1, -1)
        _apply_rotary_emb_inplace(q[..., -self.rope_head_dim:], freqs)
        _apply_rotary_emb_inplace(kv[..., -self.rope_head_dim:], freqs)
        return freqs

    def _softmax_with_sink(self, attn_scores: torch.Tensor) -> torch.Tensor:
        scores = attn_scores.float()
        sink = self.attn_sink.float().view(1, self.n_heads, 1, 1).to(scores.device)
        scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - scores_max)
        sink_exp = torch.exp(sink - scores_max)
        denom = exp_scores.sum(dim=-1, keepdim=True) + sink_exp
        return (exp_scores / denom).to(attn_scores.dtype)

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            getattr(self, name).set_runtime_tensors(tensors, name)
        if "attn_sink" in tensors:
            self.attn_sink.data = tensors["attn_sink"].to(self.attn_sink.device)
        if "q_norm.weight" in tensors:
            self.q_norm.weight.data = tensors["q_norm.weight"].to(
                self.q_norm.weight.device
            )
        if "kv_norm.weight" in tensors:
            self.kv_norm.weight.data = tensors["kv_norm.weight"].to(
                self.kv_norm.weight.device
            )
        if self.compressor is not None:
            self.compressor.set_runtime_tensors(tensors, "compressor")
        if self.indexer is not None:
            self.indexer.set_runtime_tensors(tensors, "indexer")

    def clear_runtime_tensors(self) -> None:
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            getattr(self, name).clear_runtime_tensors()
        if self.compressor is not None:
            self.compressor.clear_runtime_tensors()
        if self.indexer is not None:
            self.indexer.clear_runtime_tensors()

    def _ensure_compress_prefill_state(
        self,
        bsz: int,
        seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self.compressor is None:
            return
        cache_len = max(seqlen // self.compress_ratio, 1)
        self._ensure_compress_cache_capacity(bsz, cache_len, device, dtype)
        positions = torch.arange(max(seqlen, 1), dtype=torch.long, device=device)
        self._ensure_compress_freqs(positions)
        self.compressor.kv_cache = self._compress_kv_cache
        self.compressor.freqs_cis = self._compress_freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self._compress_freqs_cis

    def _ensure_compress_cache_capacity(
        self,
        bsz: int,
        cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            self._compress_kv_cache is None
            or self._compress_kv_cache.size(0) < bsz
            or self._compress_kv_cache.size(1) < cache_len
            or self._compress_kv_cache.device != device
            or self._compress_kv_cache.dtype != dtype
        ):
            old_cache = self._compress_kv_cache
            new_cache = torch.zeros(
                bsz,
                cache_len,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            if old_cache is not None:
                copy_bsz = min(bsz, old_cache.size(0))
                copy_len = min(cache_len, old_cache.size(1))
                if (
                    copy_bsz > 0
                    and copy_len > 0
                    and old_cache.device == device
                    and old_cache.dtype == dtype
                ):
                    new_cache[:copy_bsz, :copy_len] = old_cache[
                        :copy_bsz,
                        :copy_len,
                    ]
            self._compress_kv_cache = new_cache

    def _ensure_compress_freqs(self, positions: torch.Tensor) -> None:
        if positions.numel() == 0:
            return
        seqlen = int(positions.max().item()) + 1
        if (
            self._compress_freqs_cis is None
            or self._compress_freqs_cis.size(0) < max(seqlen, 1)
            or self._compress_freqs_cis.device != positions.device
        ):
            all_positions = torch.arange(
                max(seqlen, 1),
                dtype=torch.long,
                device=positions.device,
            )
            self._compress_freqs_cis = _build_rope_freqs_cis(
                all_positions,
                self.rope_head_dim,
                self.rope_theta,
                self.rope_factor,
                self.beta_fast,
                self.beta_slow,
                self.original_seq_len,
            )

    def _window_cache_from_past(
        self,
        past_kv: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        bsz, max_len, head_dim = past_kv.shape
        window_cache = past_kv.new_zeros(bsz, self.window_size, head_dim)
        lengths = cache_seqlens.to(device=past_kv.device, dtype=torch.long).clamp(
            min=0,
            max=max_len,
        )
        for batch_idx, valid_len_tensor in enumerate(lengths):
            valid_len = int(valid_len_tensor.item())
            if valid_len <= 0:
                continue
            if valid_len <= self.window_size:
                window_cache[batch_idx, :valid_len] = past_kv[batch_idx, :valid_len]
                continue
            cutoff = valid_len % self.window_size
            recent = past_kv[batch_idx, valid_len - self.window_size : valid_len]
            split = self.window_size - cutoff
            window_cache[batch_idx, cutoff:] = recent[:split]
            if cutoff:
                window_cache[batch_idx, :cutoff] = recent[split:]
        return window_cache

    def _prefill_sparse_attention(
        self,
        hidden_states: torch.Tensor,
        q_low: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        kv_for_attn = kv
        topk_idxs = _window_topk_idxs(
            self.window_size,
            bsz,
            seqlen,
            start_pos,
            hidden_states.device,
        )
        if self.compressor is not None:
            self._ensure_compress_prefill_state(
                bsz,
                start_pos + seqlen,
                hidden_states.device,
                kv.dtype,
            )
            offset = kv.size(1) if start_pos == 0 else self.window_size
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(hidden_states, q_low, start_pos, offset)
            else:
                compress_topk_idxs = _compress_topk_idxs(
                    self.compress_ratio,
                    bsz,
                    seqlen,
                    start_pos,
                    offset,
                    hidden_states.device,
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
            kv_compress = self.compressor(hidden_states, start_pos)
            if kv_compress is not None:
                kv_for_attn = torch.cat([kv, kv_compress], dim=1)
        return _sparse_attn_from_topk(
            q,
            kv_for_attn,
            self.attn_sink,
            topk_idxs.int(),
            self.softmax_scale,
        )

    def _decode_sparse_attention(
        self,
        hidden_states: torch.Tensor,
        q_low: torch.Tensor,
        q: torch.Tensor,
        past_kv: torch.Tensor,
        cache_seqlens: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        if q_len != 1:
            raise RuntimeError(
                "DeepSeek-V4 sparse decode fallback currently expects q_len=1, "
                f"got {q_len}."
            )
        kv_for_attn = self._window_cache_from_past(past_kv, cache_seqlens)
        topk_idxs = _window_topk_idxs(
            self.window_size,
            bsz,
            q_len,
            start_pos,
            hidden_states.device,
        )
        if self.compressor is not None:
            cache_len = max((start_pos + q_len) // self.compress_ratio, 1)
            self._ensure_compress_cache_capacity(
                bsz,
                cache_len,
                hidden_states.device,
                past_kv.dtype,
            )
            positions = torch.arange(
                start_pos,
                start_pos + q_len,
                dtype=torch.long,
                device=hidden_states.device,
            )
            self._ensure_compress_freqs(positions)
            self.compressor.kv_cache = self._compress_kv_cache
            self.compressor.freqs_cis = self._compress_freqs_cis
            offset = self.window_size
            if self.indexer is not None:
                self.indexer.freqs_cis = self._compress_freqs_cis
                compress_topk_idxs = self.indexer(hidden_states, q_low, start_pos, offset)
            else:
                compress_topk_idxs = _compress_topk_idxs(
                    self.compress_ratio,
                    bsz,
                    q_len,
                    start_pos,
                    offset,
                    hidden_states.device,
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
            self.compressor(hidden_states, start_pos)
            kv_for_attn = torch.cat(
                [kv_for_attn, self._compress_kv_cache[:bsz]],
                dim=1,
            )
        return _sparse_attn_from_topk(
            q,
            kv_for_attn,
            self.attn_sink,
            topk_idxs.int(),
            self.softmax_scale,
        )

    def empty_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        bsz, q_len, _ = hidden_states.shape
        attn_output = hidden_states.new_empty(bsz, q_len, self.hidden_size)
        kv = hidden_states.new_empty(bsz, q_len, self.head_dim)
        return attn_output, None, kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        del use_cache
        bsz, q_len, _ = hidden_states.shape
        is_decode = past_key_value is not None
        if is_decode:
            _v4_diag(f"attn L{self.layer_idx} start shape={tuple(hidden_states.shape)}")
        q_low = self.q_norm(self.wq_a(hidden_states))
        if is_decode:
            _v4_diag(f"attn L{self.layer_idx} wq_a done")
        q = self.wq_b(q_low).view(bsz, q_len, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + self.eps)
        if is_decode:
            _v4_diag(f"attn L{self.layer_idx} q done")

        kv = self.kv_norm(self.wkv(hidden_states))
        freqs_cis = self._apply_rope(
            q,
            kv,
            position_ids=position_ids,
            cache_seqlens=cache_seqlens,
        )
        # QAT: FP8 round-trip on the non-rope dims of kv before it is fed
        # to attention and (for the decode path) written to the window
        # cache. Matches reference assets/inference/model.py:506. Rope dims
        # stay bf16 for positional precision.
        if self.rope_head_dim < self.head_dim:
            _qat_fp8_act_quant_inplace(kv[..., :-self.rope_head_dim])
        if is_decode:
            _v4_diag(f"attn L{self.layer_idx} wkv done")
        kv_for_attn = kv
        start_pos = 0
        if position_ids is not None:
            pos = position_ids.to(device=hidden_states.device, dtype=torch.long)
            if pos.dim() == 1:
                pos = pos.view(1, -1)
            start_pos = int(pos[0, 0].item())
        if past_key_value is None and attention_mask is None:
            attn_output = self._prefill_sparse_attention(
                hidden_states,
                q_low,
                q,
                kv,
                start_pos,
            )
        else:
            attn_output = None
            if past_key_value is not None:
                kv_for_attn = self._normalize_past_kv(past_key_value)
                if q_len == 1 and cache_seqlens is not None:
                    self._write_current_kv(kv_for_attn, kv, cache_seqlens)
                    attn_output = self._decode_sparse_attention(
                        hidden_states,
                        q_low,
                        q,
                        kv_for_attn,
                        cache_seqlens,
                        start_pos,
                    )
            if attn_output is None:
                k = kv_for_attn.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
                v = k
                attn_scores = torch.einsum("bshd,bthd->bhst", q, k) * self.softmax_scale
                attn_scores = self._apply_fallback_masks(
                    attn_scores,
                    attention_mask,
                    cache_seqlens,
                    q_len,
                    kv_for_attn.size(1),
                    past_key_value is not None,
                )
                attn_weights = self._softmax_with_sink(attn_scores).to(q.dtype)
                attn_output = torch.einsum("bhst,bthd->bshd", attn_weights, v)
        if freqs_cis is not None:
            _apply_rotary_emb_inplace(
                attn_output[..., -self.rope_head_dim:],
                freqs_cis,
                inverse=True,
            )
        if is_decode:
            _v4_diag(f"attn L{self.layer_idx} fallback attention done")

        attn_output = attn_output.reshape(
            bsz, q_len, self.o_groups, self.n_heads // self.o_groups * self.head_dim
        )
        wo_a_weight = self.wo_a.weight
        if wo_a_weight is None:
            raise RuntimeError("DeepSeek-V4 attention wo_a weight is not loaded.")
        wo_a_weight = _dequant_weight(
            wo_a_weight,
            self.wo_a.scale,
            hidden_states.dtype,
        )
        wo_a = wo_a_weight.view(
            self.o_groups, self.o_lora_rank, self.n_heads // self.o_groups * self.head_dim
        )
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
        attn_output = self.wo_b(attn_output.flatten(2))
        if is_decode:
            _v4_diag(f"attn L{self.layer_idx} output done")
        return attn_output, None, kv

    @staticmethod
    def _normalize_past_kv(past_key_value: torch.Tensor) -> torch.Tensor:
        if past_key_value.dim() == 4 and past_key_value.size(2) == 1:
            return past_key_value.squeeze(2)
        if past_key_value.dim() == 3:
            return past_key_value
        raise RuntimeError(
            "DeepSeek-V4 fallback attention expected past KV with shape "
            f"[B, T, D] or [B, T, 1, D], got {tuple(past_key_value.shape)}"
        )

    @staticmethod
    def _write_current_kv(
        past_kv: torch.Tensor,
        current_kv: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        positions = (cache_seqlens.to(current_kv.device).long() - 1).clamp_min(0)
        batch_idx = torch.arange(current_kv.size(0), device=current_kv.device)
        valid = positions < past_kv.size(1)
        if valid.any():
            past_kv[batch_idx[valid], positions[valid]] = current_kv[valid, 0]

    @staticmethod
    def _apply_fallback_masks(
        attn_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_seqlens: Optional[torch.Tensor],
        q_len: int,
        kv_len: int,
        using_past: bool,
    ) -> torch.Tensor:
        neg_inf = torch.finfo(attn_scores.dtype).min
        device = attn_scores.device
        if cache_seqlens is not None:
            valid_lens = cache_seqlens.to(device).long().clamp(max=kv_len)
            key_pos = torch.arange(kv_len, device=device).unsqueeze(0)
            mask = key_pos >= valid_lens.unsqueeze(1)
            return attn_scores.masked_fill(mask[:, None, None, :], neg_inf)

        if attention_mask is not None and attention_mask.dim() == 2:
            key_mask = attention_mask[:, -kv_len:].to(device) == 0
            attn_scores = attn_scores.masked_fill(key_mask[:, None, None, :], neg_inf)
        elif attention_mask is not None and attention_mask.dim() == 4:
            attn_scores = attn_scores + attention_mask.to(device)

        if not using_past and q_len > 1:
            causal = torch.triu(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal[None, None, :, :], neg_inf)
        return attn_scores


class DeepSeekV4FlashGate(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(_cfg(config, "hidden_size", _cfg(config, "dim", 4096)))
        self.num_experts = int(_cfg(config, "n_routed_experts", _cfg(config, "num_local_experts", 256)))
        self.topk = int(_cfg(config, "num_experts_per_tok", _cfg(config, "n_activated_experts", 6)))
        self.score_func = str(_cfg(config, "scoring_func", _cfg(config, "score_func", "sqrtsoftplus")))
        self.route_scale = float(_cfg(config, "routed_scaling_factor", _cfg(config, "route_scale", 1.5)))
        self.norm_topk_prob = bool(_cfg(config, "norm_topk_prob", True))
        self.is_hash_layer = layer_idx < int(_cfg(config, "num_hash_layers", _cfg(config, "n_hash_layers", 3)))

        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        if self.is_hash_layer:
            vocab_size = int(_cfg(config, "vocab_size", 129280))
            self.tid2eid = nn.Parameter(
                torch.empty(vocab_size, self.topk, dtype=torch.long),
                requires_grad=False,
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(self.num_experts, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(hidden_states.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        elif self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        else:
            raise ValueError(f"Unsupported V4 gate score function: {self.score_func}")

        raw_scores = scores
        if self.is_hash_layer:
            if input_ids is None:
                topk_indices = torch.topk(scores, k=self.topk, dim=-1)[1]
            else:
                topk_indices = self.tid2eid[input_ids].long()
        else:
            select_scores = scores + self.bias.float().unsqueeze(0)
            topk_indices = torch.topk(select_scores, k=self.topk, dim=-1)[1]

        topk_weights = raw_scores.gather(-1, topk_indices)
        if self.score_func != "softmax" and self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weights * self.route_scale, topk_indices


class DeepSeekV4FlashExpertPlaceholder(nn.Module):
    """Lightweight expert slot replaced/configured by V4 expert wrappers."""

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        self.runtime_weights: Optional[Dict[str, torch.Tensor]] = None

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        self.runtime_weights = tensors

    def clear_runtime_tensors(self) -> None:
        self.runtime_weights = None

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if self.runtime_weights is None:
            raise RuntimeError("DeepSeek-V4 expert weights are not loaded.")
        return _linear_from_weight(
            x,
            self.runtime_weights[f"{name}.weight"],
            self.runtime_weights.get(f"{name}.scale"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate = self._linear(hidden_states, "w1").float()
        up = self._linear(hidden_states, "w3").float()
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        hidden_states = F.silu(gate) * up
        if weights is not None:
            hidden_states = hidden_states * weights
        return self._linear(hidden_states.to(weights.dtype if weights is not None else gate.dtype), "w2")


class DeepSeekV4FlashMoE(nn.Module):
    """V4 EP-MoE surface with global expert slots."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = int(_cfg(config, "hidden_size", _cfg(config, "dim", 4096)))
        self.intermediate_size = int(_cfg(config, "moe_intermediate_size", _cfg(config, "moe_inter_dim", 2048)))
        self.total_experts = int(_cfg(config, "n_routed_experts", _cfg(config, "num_local_experts", 256)))
        self.num_experts_per_tok = int(_cfg(config, "num_experts_per_tok", _cfg(config, "n_activated_experts", 6)))
        self.swiglu_limit = float(_cfg(config, "swiglu_limit", 10.0))
        self.gate = DeepSeekV4FlashGate(config, layer_idx)
        self.experts = nn.ModuleList(
            [
                DeepSeekV4FlashExpertPlaceholder(
                    self.hidden_size, self.intermediate_size, self.swiglu_limit
                )
                for _ in range(self.total_experts)
            ]
        )
        self.shared_experts = DeepSeekV4FlashExpertPlaceholder(
            self.hidden_size, self.intermediate_size, 0.0
        )
        self.comm = None
        self.routed_expert_start_idx = 0
        self.routed_expert_end_idx = self.total_experts
        self.experts_per_rank = self.total_experts
        self.enable_ep_offloading = False

    def configure_ep(self, rank: int, world_size: int, comm=None) -> None:
        self.comm = comm
        self.experts_per_rank = math.ceil(self.total_experts / world_size)
        self.routed_expert_start_idx = min(rank * self.experts_per_rank, self.total_experts)
        self.routed_expert_end_idx = min(
            (rank + 1) * self.experts_per_rank, self.total_experts
        )
        self.enable_ep_offloading = world_size > 1

    @staticmethod
    def _gather_token_rows(rows: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _v4_diag(f"moe gather rows enter local={rows.shape[0]} dtype={rows.dtype}")
        local_tokens = torch.tensor(
            [rows.shape[0]],
            dtype=torch.int64,
            device=rows.device,
        )
        all_tokens = torch.empty(
            dist.get_world_size(),
            dtype=torch.int64,
            device=rows.device,
        )
        dist.all_gather_into_tensor(all_tokens, local_tokens)
        max_tokens = int(all_tokens.max().item())
        _v4_diag(
            f"moe gather counts all={all_tokens.detach().cpu().tolist()} max={max_tokens}"
        )
        if max_tokens == 0:
            return rows, 0, 0

        padded_shape = (max_tokens, *rows.shape[1:])
        if rows.shape[0] == max_tokens:
            padded = rows
        else:
            padded = rows.new_zeros(padded_shape)
            if rows.shape[0] > 0:
                padded[: rows.shape[0]] = rows

        gathered_flat = rows.new_empty(
            (dist.get_world_size() * max_tokens, *rows.shape[1:])
        )
        dist.all_gather_into_tensor(gathered_flat, padded.contiguous())
        _v4_diag("moe gather payload done")
        gathered = gathered_flat.view(
            dist.get_world_size(), max_tokens, *rows.shape[1:]
        )
        valid_chunks = [
            gathered[rank, : int(all_tokens[rank].item())]
            for rank in range(dist.get_world_size())
            if int(all_tokens[rank].item()) > 0
        ]
        if valid_chunks:
            gathered_rows = torch.cat(valid_chunks, dim=0)
        else:
            gathered_rows = rows.new_empty((0, *rows.shape[1:]))
        local_start = int(all_tokens[: dist.get_rank()].sum().item())
        return gathered_rows, local_start, int(local_tokens.item())

    @staticmethod
    def _gather_token_ids(token_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if token_ids is None:
            return None
        gathered_ids, _, _ = DeepSeekV4FlashMoE._gather_token_rows(
            token_ids.reshape(-1, 1)
        )
        return gathered_ids.reshape(-1).long()

    def _active_local_experts(self, topk_indices: torch.Tensor) -> list[int]:
        if topk_indices.numel() == 0:
            return []
        active = torch.unique(topk_indices.reshape(-1))
        active = active[
            (active >= self.routed_expert_start_idx)
            & (active < self.routed_expert_end_idx)
        ]
        return [int(expert_idx) for expert_idx in active.detach().cpu().tolist()]

    def _forward_ep(
        self,
        flat_states: torch.Tensor,
        flat_ids: Optional[torch.Tensor],
        output_shape: torch.Size,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        _v4_diag(f"moe L{self.layer_idx} ep start local={flat_states.shape[0]}")
        timing = _v4_timing_enabled()
        last_time = _v4_sync_time(flat_states.device) if timing else 0.0

        def mark(label: str) -> None:
            nonlocal last_time
            if not timing:
                return
            now = _v4_sync_time(flat_states.device)
            _v4_timing(f"moe L{self.layer_idx} {label} {(now - last_time) * 1000:.2f}ms")
            last_time = now

        global_states, local_start, local_tokens = self._gather_token_rows(flat_states)
        mark("gather_states")
        _v4_diag(
            f"moe L{self.layer_idx} states gathered global={global_states.shape[0]} "
            f"local_start={local_start} local={local_tokens}"
        )
        global_ids = self._gather_token_ids(flat_ids)
        mark("gather_ids")
        _v4_diag(f"moe L{self.layer_idx} ids gathered")
        routed_global = torch.zeros_like(global_states, dtype=torch.float32)

        if global_states.shape[0] > 0:
            topk_weights, topk_indices = self.gate(global_states, global_ids)
            mark("gate")
            _v4_diag(f"moe L{self.layer_idx} gate done")
            active_experts = self._active_local_experts(topk_indices)
            mark(f"active_local count={len(active_experts)}")
            for expert_idx in active_experts:
                token_idx, topk_pos = torch.where(topk_indices == expert_idx)
                expert_out = self.experts[expert_idx](
                    global_states[token_idx],
                    topk_weights[token_idx, topk_pos].unsqueeze(-1),
                )
                routed_global[token_idx] += expert_out.float()
            mark("routed_experts")

        _v4_diag(f"moe L{self.layer_idx} all_reduce enter")
        dist.all_reduce(routed_global)
        mark("all_reduce")
        _v4_diag(f"moe L{self.layer_idx} all_reduce done")
        local_routed = routed_global[local_start : local_start + local_tokens]
        if flat_states.shape[0] == 0:
            shared = torch.zeros_like(flat_states, dtype=torch.float32)
        else:
            _v4_diag(f"moe L{self.layer_idx} shared enter")
            shared = self.shared_experts(flat_states).float()
            mark("shared")
            _v4_diag(f"moe L{self.layer_idx} shared done")
        output = (local_routed + shared).to(output_dtype).view(output_shape)
        mark("finish")
        return output

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        flat_ids = input_ids.reshape(-1) if input_ids is not None else None
        if self.enable_ep_offloading and dist.is_initialized():
            return self._forward_ep(flat_states, flat_ids, shape, hidden_states.dtype)

        topk_weights, topk_indices = self.gate(flat_states, flat_ids)

        routed = torch.zeros_like(flat_states, dtype=torch.float32)
        for expert_idx in self._active_local_experts(topk_indices):
            token_idx, topk_pos = torch.where(topk_indices == expert_idx)
            expert_out = self.experts[expert_idx](
                flat_states[token_idx],
                topk_weights[token_idx, topk_pos].unsqueeze(-1),
            )
            routed[token_idx] += expert_out.float()

        shared = self.shared_experts(flat_states).float()
        return (routed + shared).to(hidden_states.dtype).view(shape)


def _hc_split(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.sigmoid(
        mixes[..., :hc_mult] * scale[0] + base[:hc_mult]
    ) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * scale[1]
        + base[hc_mult : 2 * hc_mult]
    )
    comb_base = base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb = torch.softmax(comb * scale[2] + comb_base, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(int(sinkhorn_iters) - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class DeepSeekV4FlashDecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(_cfg(config, "hidden_size", _cfg(config, "dim", 4096)))
        self.hc_mult = int(_cfg(config, "hc_mult", 4))
        self.hc_eps = float(_cfg(config, "hc_eps", 1e-6))
        self.hc_sinkhorn_iters = int(_cfg(config, "hc_sinkhorn_iters", 20))
        self.rms_norm_eps = float(_cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6)))
        hc_dim = self.hc_mult * self.hidden_size
        mix_hc = (2 + self.hc_mult) * self.hc_mult

        self.self_attn = DeepSeekV4FlashAttention(config, layer_idx)
        self.attn = self.self_attn
        self.mlp = DeepSeekV4FlashMoE(config, layer_idx)
        self.ffn = self.mlp
        self.attn_norm = DeepSeekV4FlashRMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = DeepSeekV4FlashRMSNorm(self.hidden_size, self.rms_norm_eps)
        self.input_layernorm = self.attn_norm
        self.post_attention_layernorm = self.ffn_norm

        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _hc_pre(
        self,
        hidden_states: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = hidden_states.shape
        flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        mixes = F.linear(flat, fn) * rsqrt
        pre, post, comb = _hc_split(
            mixes,
            scale,
            base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        reduced = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
        return reduced.to(hidden_states.dtype), post, comb

    def _hc_post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return (
            post.unsqueeze(-1) * hidden_states.unsqueeze(-2)
            + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        ).to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        del output_attentions, kwargs
        is_decode = past_key_value is not None
        timing = is_decode and _v4_timing_enabled()
        last_time = _v4_sync_time(hidden_states.device) if timing else 0.0

        def mark(label: str) -> None:
            nonlocal last_time
            if not timing:
                return
            now = _v4_sync_time(hidden_states.device)
            _v4_timing(
                f"layer {self.layer_idx} {label} {(now - last_time) * 1000:.2f}ms"
            )
            last_time = now

        if is_decode:
            _v4_diag(f"layer {self.layer_idx} start hidden={tuple(hidden_states.shape)}")
        collapse_hc_state = hidden_states.dim() == 3
        if collapse_hc_state:
            hidden_states = hidden_states.unsqueeze(2).expand(
                -1, -1, self.hc_mult, -1
            ).contiguous()

        residual = hidden_states
        attn_input, post, comb = self._hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        attn_input = self.attn_norm(attn_input)
        mark("hc_attn")
        if is_decode:
            _v4_diag(f"layer {self.layer_idx} attn enter")
        attn_out, attn_weights, present = self.self_attn(
            attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_seqlens=cache_seqlens,
            use_cache=use_cache,
        )
        if is_decode:
            _v4_diag(f"layer {self.layer_idx} attn done")
        mark("attn")
        hidden_states = self._hc_post(attn_out, residual, post, comb)
        mark("post_attn")

        residual = hidden_states
        mlp_input, post, comb = self._hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        mlp_input = self.ffn_norm(mlp_input)
        mark("hc_moe")
        if is_decode:
            _v4_diag(f"layer {self.layer_idx} moe enter")
        mlp_out = self.mlp(mlp_input, input_ids)
        if is_decode:
            _v4_diag(f"layer {self.layer_idx} moe done")
        mark("moe")
        hidden_states = self._hc_post(mlp_out, residual, post, comb)
        mark("post_moe")
        if collapse_hc_state:
            hidden_states = hidden_states.mean(dim=2)
        if is_decode:
            _v4_diag(f"layer {self.layer_idx} done")
        mark("done")
        return hidden_states, attn_weights, present


class DeepSeekV4FlashModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_size = int(_cfg(config, "hidden_size", _cfg(config, "dim", 4096)))
        self.vocab_size = int(_cfg(config, "vocab_size", 129280))
        self.hc_mult = int(_cfg(config, "hc_mult", 4))
        self.hc_eps = float(_cfg(config, "hc_eps", 1e-6))
        self.rms_norm_eps = float(_cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6)))
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            int(_cfg(config, "pad_token_id", 1)),
        )
        self.embed = self.embed_tokens
        self.layers = nn.ModuleList(
            [
                DeepSeekV4FlashDecoderLayer(config, layer_idx)
                for layer_idx in range(int(_cfg(config, "num_hidden_layers", _cfg(config, "n_layers", 43))))
            ]
        )
        self.norm = DeepSeekV4FlashRMSNorm(self.hidden_size, self.rms_norm_eps)
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def _hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        mixes = F.linear(flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        return torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2).to(
            hidden_states.dtype
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        del return_dict, kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)

        is_decode = past_key_values is not None
        if is_decode:
            _v4_diag(
                f"model forward decode start inputs={tuple(inputs_embeds.shape)} "
                f"layers={len(self.layers)}"
            )
        hidden_states = inputs_embeds.unsqueeze(2).expand(
            -1, -1, self.hc_mult, -1
        ).contiguous()
        presents = []
        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None
            if is_decode:
                _v4_diag(f"model layer {idx} enter")
            hidden_states, _, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_ids=input_ids,
                past_key_value=past_kv,
                output_attentions=bool(output_attentions),
                use_cache=bool(use_cache),
            )
            if is_decode:
                _v4_diag(f"model layer {idx} exit")
            if use_cache:
                presents.append(present)
        hidden_states = self.norm(self._hc_head(hidden_states))
        if is_decode:
            _v4_diag("model forward decode done")
        if use_cache:
            return hidden_states, tuple(presents)
        return (hidden_states,)


class DeepSeekV4FlashForCausalLM(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.model = DeepSeekV4FlashModel(config)
        self.vocab_size = int(_cfg(config, "vocab_size", 129280))
        hidden_size = int(_cfg(config, "hidden_size", _cfg(config, "dim", 4096)))
        self.lm_head = nn.Linear(hidden_size, self.vocab_size, bias=False)
        self.head = self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> _CausalLMOutput:
        del kwargs
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return _CausalLMOutput(logits=logits)
