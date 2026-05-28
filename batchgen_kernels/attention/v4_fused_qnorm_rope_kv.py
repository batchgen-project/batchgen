# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

import torch

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_BLOCK_SIZE = 64
SCALE_DIM = NOPE_DIM // QUANT_BLOCK_SIZE + 1
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
ROPE_BYTES = ROPE_DIM * torch.tensor([], dtype=torch.bfloat16).element_size()
TOKEN_DATA_SIZE = NOPE_DIM + ROPE_BYTES
TOKEN_BYTES = TOKEN_DATA_SIZE + SCALE_DIM

try:
    from batchgen_kernels import load_extension

    _C = load_extension("batchgen_kernels.attention._C_v4_attn")
    _cuda_available = True
except (ImportError, Exception):
    _C = None
    _cuda_available = False


def _rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    rrms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rrms).to(x.dtype)


def _rmsnorm_with_weight(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    rrms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rrms * weight.float()).to(x.dtype)


def _apply_gptj_rope_last_64_dims(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return x.clone()
    out = x.clone()
    rope = out[:, -ROPE_DIM:].float().view(-1, ROPE_DIM // 2, 2)
    cache = cos_sin_cache.index_select(0, positions.long())
    cos = cache[:, 0::2, 0]
    sin = cache[:, 0::2, 1]
    even = rope[..., 0]
    odd = rope[..., 1]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    out[:, -ROPE_DIM:] = (
        torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(x.dtype)
    )
    return out


def encode_ue8m0_scale(absmax: torch.Tensor) -> torch.Tensor:
    absmax_fp32 = absmax.float()
    nonzero = absmax_fp32 > 0
    safe_absmax = torch.where(
        nonzero, absmax_fp32, torch.ones_like(absmax_fp32)
    )
    exponent = torch.ceil(torch.log2(safe_absmax / FP8_MAX))
    encoded = torch.where(nonzero, exponent + 127.0, torch.zeros_like(exponent))
    return encoded.clamp_(0.0, 255.0).to(torch.uint8)


def decode_ue8m0_scale(encoded: torch.Tensor) -> torch.Tensor:
    encoded_fp32 = encoded.float()
    exponent = encoded_fp32 - 127.0
    scale = torch.exp2(exponent)
    return torch.where(encoded == 0, torch.zeros_like(scale), scale)


def quantize_nope_to_fp8(
    nope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nope_blocks = nope.float().reshape(nope.shape[0], -1, QUANT_BLOCK_SIZE)
    absmax = nope_blocks.abs().amax(dim=-1)
    encoded = encode_ue8m0_scale(absmax)
    scale = torch.where(
        encoded == 0,
        torch.ones_like(absmax),
        torch.exp2(encoded.float() - 127.0),
    )
    quantized = torch.clamp(
        nope_blocks / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX
    ).to(FP8_DTYPE)
    return quantized.reshape_as(nope), encoded, scale


def dequantize_nope_from_fp8(
    nope_fp8: torch.Tensor,
    encoded_scale: torch.Tensor,
) -> torch.Tensor:
    expanded = decode_ue8m0_scale(encoded_scale).unsqueeze(-1)
    return (
        nope_fp8.float().view(nope_fp8.shape[0], -1, QUANT_BLOCK_SIZE)
        * expanded
    ).reshape(nope_fp8.shape[0], NOPE_DIM)


def _insert_into_paged_cache(
    kv_processed: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> None:
    if kv_processed.shape[0] == 0:
        return
    if kv_cache.ndim != 2:
        raise ValueError(f"kv_cache must be 2D, got {tuple(kv_cache.shape)}")
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if kv_cache.shape[1] < TOKEN_BYTES:
        raise ValueError(
            f"kv_cache last dim must be >= {TOKEN_BYTES}, got {kv_cache.shape[1]}"
        )

    nope = kv_processed[:, :NOPE_DIM].contiguous()
    rope = kv_processed[:, NOPE_DIM:].contiguous()
    nope_fp8, encoded_scale, _ = quantize_nope_to_fp8(nope)
    pages = block_table.long().contiguous()
    if pages.numel() != kv_processed.shape[0]:
        raise ValueError(
            f"block_table shape mismatch: expected {kv_processed.shape[0]}, got {pages.numel()}"
        )
    if pages.numel() and (
        (pages < 0).any() or (pages >= kv_cache.shape[0]).any()
    ):
        raise ValueError("block_table contains out-of-range page indices")

    kv_cache[pages, :NOPE_DIM] = nope_fp8.view(torch.uint8).reshape(
        -1, NOPE_DIM
    )
    kv_cache[pages, NOPE_DIM:TOKEN_DATA_SIZE] = rope.view(torch.uint8).reshape(
        -1, ROPE_BYTES
    )
    kv_cache[pages, TOKEN_DATA_SIZE:TOKEN_BYTES] = torch.cat(
        (
            encoded_scale,
            torch.zeros(
                encoded_scale.shape[0],
                TOKEN_BYTES - TOKEN_DATA_SIZE - encoded_scale.shape[1],
                device=encoded_scale.device,
                dtype=torch.uint8,
            ),
        ),
        dim=-1,
    )


def _fused_v4_qnorm_rope_kv_insert_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = _rmsnorm_no_weight(q, eps)
    kv_out = _rmsnorm_with_weight(kv, kv_weight, eps)
    q_out = _apply_gptj_rope_last_64_dims(q_out, positions, cos_sin_cache)
    kv_out = _apply_gptj_rope_last_64_dims(kv_out, positions, cos_sin_cache)
    _insert_into_paged_cache(kv_out, kv_cache, block_table)
    return q_out, kv_out


def fused_v4_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 2 or q.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"q must have shape [T, {HEAD_DIM}], got {tuple(q.shape)}"
        )
    if kv.ndim != 2 or kv.shape != q.shape:
        raise ValueError(
            f"kv must have shape {tuple(q.shape)}, got {tuple(kv.shape)}"
        )
    if kv_weight.ndim != 1 or kv_weight.shape[0] != HEAD_DIM:
        raise ValueError(
            f"kv_weight must have shape [{HEAD_DIM}], got {tuple(kv_weight.shape)}"
        )
    if cos_sin_cache.ndim != 3 or cos_sin_cache.shape[1:] != (ROPE_DIM, 2):
        raise ValueError(
            "cos_sin_cache must have shape [max_pos, 64, 2], "
            f"got {tuple(cos_sin_cache.shape)}"
        )
    if positions.ndim != 1 or positions.shape[0] != q.shape[0]:
        raise ValueError(
            f"positions must have shape [{q.shape[0]}], got {tuple(positions.shape)}"
        )
    if block_table.ndim != 1 or block_table.shape[0] != q.shape[0]:
        raise ValueError(
            f"block_table must have shape [{q.shape[0]}], got {tuple(block_table.shape)}"
        )
    if positions.numel() and (
        positions.min() < 0 or positions.max() >= cos_sin_cache.shape[0]
    ):
        raise ValueError("positions are out of range for cos_sin_cache")

    if (
        _cuda_available
        and q.is_cuda
        and kv.is_cuda
        and hasattr(_C, "fused_v4_qnorm_rope_kv_insert")
    ):
        try:
            return _C.fused_v4_qnorm_rope_kv_insert(
                q,
                kv,
                kv_weight,
                cos_sin_cache,
                positions,
                kv_cache,
                block_table,
                eps,
            )
        except Exception:
            pass

    return _fused_v4_qnorm_rope_kv_insert_fallback(
        q,
        kv,
        kv_weight,
        cos_sin_cache,
        positions,
        kv_cache,
        block_table,
        eps,
    )


__all__ = [
    "FP8_MAX",
    "HEAD_DIM",
    "NOPE_DIM",
    "QUANT_BLOCK_SIZE",
    "ROPE_DIM",
    "SCALE_DIM",
    "TOKEN_BYTES",
    "TOKEN_DATA_SIZE",
    "decode_ue8m0_scale",
    "dequantize_nope_from_fp8",
    "encode_ue8m0_scale",
    "fused_v4_qnorm_rope_kv_insert",
    "quantize_nope_to_fp8",
]
