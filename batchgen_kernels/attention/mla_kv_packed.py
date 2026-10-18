"""FlashMLA packed FP8 MLA KV record: geometry, reference quantizer, dequant.

One token of the packed cache is 656 contiguous bytes, exactly as
`references/FlashMLA-1408756/csrc/pybind.cpp:249-256` hard-checks and
`README.md:117-121` documents:

    [   0, 512)  512 x float8_e4m3  quantized NoPE (the compressed KV latent)
    [ 512, 528)    4 x float32      one scale per 128-element NoPE block
    [ 528, 656)   64 x bfloat16     RoPE, NOT quantized

and the kernel dequantizes with ``value = fp8_byte * scale`` in bf16
(`csrc/sm90/decode/sparse_fp8/components/dequant.h:21-28`). The page block size
is hard-required to be 64 (`pybind.cpp:236`) with the whole block contiguous
(``kcache.stride(1) == 656``), so the record is token-major inside the page:
the packed cache is an ordinary contiguous-AoS cache whose token happens to be
656 BYTES instead of 576 bf16 elements.

This module deliberately imports nothing but torch. The packed bytes are the
byte-exactness contract between the decode/prefill writers, the staged host
pages and the FlashMLA kernel, so the reference implementation of that contract
stays importable -- and testable -- without Triton or a GPU.

Numerics: this is a port of SGLang's shipped quantizer,
`references/sglang/python/sglang/srt/layers/attention/nsa/quant_k_cache.py`
(`_quantize_k_cache_fast` / `_quantize_k_cache_fast_kernel`), which is the path
that produced its FP8-stored production shape. It is byte-identical to that
kernel -- fp32 amax, ``scale = amax / 448``, reciprocal multiply, clamp to
[-448, 448] -- with ONE deliberate divergence, documented at
``_quantize_block`` below: a block whose amax is 0.

Note that SGLang's own two implementations disagree on non-degenerate input
too: `_quantize_k_cache_ref` reduces amax in bf16 and divides, while
`_quantize_k_cache_fast` reduces in fp32 and multiplies by the reciprocal --
which is why SGLang's self-test compares them at atol/rtol 0.2 rather than
byte-wise. We follow the fast (shipped) kernel; `tests/` pins that choice by
reproducing the ref's bytes when, and only when, the bf16 reduction is put
back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

__all__ = [
    "NOPE_DIM",
    "ROPE_DIM",
    "SOURCE_DIM",
    "QUANT_TILE",
    "NUM_SCALES",
    "SCALE_BYTES",
    "ROPE_BYTES",
    "TOKEN_BYTES",
    "PAGE_SIZE",
    "FP8_MAX",
    "PackedRecordGeometry",
    "packed_record_geometry",
    "cache_is_packed_fp8",
    "quantize_mla_kv_packed_reference",
    "dequantize_mla_kv_packed",
    "validate_packed_quantize_operands",
]

NOPE_DIM = 512
ROPE_DIM = 64
SOURCE_DIM = NOPE_DIM + ROPE_DIM
QUANT_TILE = 128
NUM_SCALES = NOPE_DIM // QUANT_TILE
SCALE_BYTES = NUM_SCALES * 4
ROPE_BYTES = ROPE_DIM * 2
TOKEN_BYTES = NOPE_DIM + SCALE_BYTES + ROPE_BYTES
PAGE_SIZE = 64

# torch.finfo(torch.float8_e4m3fn).max. Spelled as a literal so this module
# stays a plain description of the on-disk record.
FP8_MAX = 448.0


@dataclass(frozen=True)
class PackedRecordGeometry:
    """Byte offsets of the three regions of ONE packed token record."""

    nope_start: int = 0
    nope_end: int = NOPE_DIM
    scale_start: int = NOPE_DIM
    scale_end: int = NOPE_DIM + SCALE_BYTES
    rope_start: int = NOPE_DIM + SCALE_BYTES
    rope_end: int = TOKEN_BYTES


def packed_record_geometry() -> PackedRecordGeometry:
    return PackedRecordGeometry()


def cache_is_packed_fp8(blocked_kv: torch.Tensor, source_dim: int) -> bool:
    """True when this paged primary cache stores packed FP8 records.

    ``source_dim`` is the UNPACKED compressed-KV dim (``kv_lora_rank +
    qk_rope_head_dim``, 576 for GLM-5). A packed cache has a 1-byte element
    type and carries ``TOKEN_BYTES`` bytes per token; a bf16 AoS cache carries
    ``source_dim`` elements. Anything else is a mis-sized cache and is named
    here instead of being written through or attended over -- the same
    discriminate-on-the-allocated-cache rule the aux cache uses, so one decode
    path serves every GLM-5 variant without re-deriving the format from a
    model name.
    """

    carrier = int(blocked_kv.shape[-1])
    element_size = int(blocked_kv.element_size())
    source_dim = int(source_dim)
    if element_size == 1:
        if source_dim != SOURCE_DIM:
            raise ValueError(
                "the packed FP8 MLA record encodes a "
                f"{SOURCE_DIM}-element latent ({NOPE_DIM} NoPE + {ROPE_DIM} "
                f"RoPE), so it cannot carry a {source_dim}-element one"
            )
        if carrier != TOKEN_BYTES:
            raise ValueError(
                "a 1-byte primary MLA cache must hold FlashMLA's packed FP8 "
                f"record, i.e. {TOKEN_BYTES} bytes per token ({NOPE_DIM} e4m3 "
                f"+ {SCALE_BYTES} scale + {ROPE_BYTES} bf16 RoPE), got a "
                f"trailing dim of {carrier}"
            )
        return True
    if carrier != source_dim:
        raise ValueError(
            f"a bf16 AoS primary MLA cache must carry {source_dim} elements "
            f"per token, got a trailing dim of {carrier}"
        )
    return False


def validate_packed_quantize_operands(
    kv: torch.Tensor, out: torch.Tensor
) -> int:
    """Every precondition of a packed quantize, launch-free.

    Split out so the Triton launcher and this reference enforce one contract,
    and so a rejection happens before any kernel is enqueued.
    """

    if kv.ndim != 2:
        raise ValueError(
            "packed MLA quantize takes a flattened [tokens, "
            f"{SOURCE_DIM}] source, got {tuple(kv.shape)}"
        )
    if kv.shape[1] != SOURCE_DIM:
        raise ValueError(
            f"packed MLA quantize requires {SOURCE_DIM} source elements per "
            f"token ({NOPE_DIM} NoPE + {ROPE_DIM} RoPE), got {kv.shape[1]}"
        )
    if kv.dtype != torch.bfloat16:
        raise ValueError(
            "the packed record stores its RoPE half as RAW bf16 bytes, so the "
            f"source must be bfloat16, got {kv.dtype}"
        )
    if kv.stride(1) != 1:
        raise ValueError(
            "packed MLA quantize requires a contiguous last source dimension"
        )
    if out.ndim != 2 or out.shape != (kv.shape[0], TOKEN_BYTES):
        raise ValueError(
            f"packed MLA quantize writes [{kv.shape[0]}, {TOKEN_BYTES}], got "
            f"{tuple(out.shape)}"
        )
    if out.dtype != torch.uint8:
        raise ValueError(
            "the packed record mixes e4m3, fp32 and bf16 fields, so its "
            f"destination must be a uint8 byte carrier, got {out.dtype}"
        )
    if not out.is_contiguous():
        raise ValueError(
            "packed MLA quantize requires a contiguous destination: FlashMLA "
            "hard-checks that the whole 656-byte record is contiguous "
            "(csrc/pybind.cpp:255)"
        )
    if out.device != kv.device:
        raise ValueError(
            f"source and destination must share a device, got {kv.device} and "
            f"{out.device}"
        )
    return int(kv.shape[0])


def _quantize_block(block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize one 128-element NoPE block per token.

    ``block`` is ``[tokens, QUANT_TILE]`` bf16; returns ``(fp8_values,
    fp32_scales)``.

    DELIBERATE DIVERGENCE from both SGLang implementations: a block whose amax
    is 0 stores ``scale = 0.0f`` and all-zero e4m3 bytes, so it dequantizes
    back to exactly 0. SGLang leaves this case undefined and its two
    implementations disagree on it -- the torch reference computes ``0 / 0``
    and stores an e4m3 NaN (0x7f) beside a 0.0f scale, so FlashMLA reads back
    ``NaN * 0 = NaN``; the Triton kernel multiplies by ``1 / 0 = inf``, gets
    ``0 * inf = NaN``, and `tl.clamp`'s default non-propagating min/max
    silently turns that into a +-448 byte beside the same 0.0f scale. Neither
    reconstructs the value that was stored, and a NaN in the KV cache poisons
    every later step of that sequence, so the zero block is defined here
    instead: it is the ONE input for which the exact answer is free.

    This also matches the allocation invariant the packed pool relies on --
    ``torch.zeros`` gives every untouched token scale bits 0.0f and data bytes
    0x00, which dequantizes to 0 -- so a quantized all-zero token and a
    never-written token are the same bytes.
    """

    values = block.to(torch.float32)
    amax = values.abs().amax(dim=-1, keepdim=True)
    scale = amax / FP8_MAX
    is_zero = amax == 0.0
    # Guard the reciprocal, then select: computing 1/0 first and masking after
    # would still feed inf through the multiply on the masked lane.
    safe_scale = torch.where(is_zero, torch.ones_like(scale), scale)
    quantized = torch.clamp(values * (1.0 / safe_scale), -FP8_MAX, FP8_MAX)
    quantized = torch.where(is_zero, torch.zeros_like(quantized), quantized)
    return quantized.to(torch.float8_e4m3fn), torch.where(
        is_zero, torch.zeros_like(scale), scale
    )


def quantize_mla_kv_packed_reference(
    kv: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Pack ``[tokens, 576]`` bf16 compressed KV into ``[tokens, 656]`` uint8.

    The byte-exactness reference for the Triton kernel of the same name. It is
    device-agnostic torch, so it runs on CPU -- which is what makes the
    RECORD's contract (geometry, field layout, dequant semantics, the
    zero-block divergence) a dev-measurable test rather than a node-only one.

    The kernel comparison it anchors is SAME-DEVICE, though: byte-exactness
    holds against this reference evaluated on the kernel's own device, not
    against a host evaluation of it. ``amax / FP8_MAX`` rounds differently on
    host and device and a 1-ULP scale shift flips payload bytes at e4m3
    rounding boundaries, so a CPU-vs-GPU byte comparison measures the division,
    not the kernel. The dev-side tests of this module compare CPU against CPU
    and are unaffected.
    """

    if out is None:
        out = torch.empty(
            (kv.shape[0], TOKEN_BYTES), dtype=torch.uint8, device=kv.device
        )
    num_tokens = validate_packed_quantize_operands(kv, out)
    if num_tokens == 0:
        return out

    geom = packed_record_geometry()
    nope_bytes = out[:, geom.nope_start : geom.nope_end]
    scale_bytes = out[:, geom.scale_start : geom.scale_end]
    rope_bytes = out[:, geom.rope_start : geom.rope_end]

    scales = scale_bytes.view(torch.float32)
    for tile in range(NUM_SCALES):
        lo = tile * QUANT_TILE
        hi = lo + QUANT_TILE
        quantized, scale = _quantize_block(kv[:, lo:hi])
        nope_bytes[:, lo:hi] = quantized.view(torch.uint8)
        scales[:, tile] = scale.squeeze(-1)

    # RoPE is stored unquantized: a raw byte copy of the bf16 half.
    rope_bytes.copy_(kv[:, NOPE_DIM:].contiguous().view(torch.uint8))
    return out


def dequantize_mla_kv_packed(packed: torch.Tensor) -> torch.Tensor:
    """``[..., 656]`` uint8 -> ``[..., 576]`` bf16, as FlashMLA reads it.

    DEBUG / VERIFICATION ONLY -- no decode or prefill path calls this. The
    kernel dequantizes internally; this exists so a test or an indices/bounds
    probe can compare VALUES against the bf16 baseline, and so the packed
    bytes a writer produced can be checked against the activations it was
    given. Port of SGLang's `dequant_k_cache._dequantize_k_cache_ref`.
    """

    if packed.dtype != torch.uint8:
        raise ValueError(
            f"packed MLA dequantize takes uint8 bytes, got {packed.dtype}"
        )
    if packed.shape[-1] != TOKEN_BYTES:
        raise ValueError(
            f"packed MLA dequantize takes {TOKEN_BYTES}-byte records, got a "
            f"trailing dim of {packed.shape[-1]}"
        )
    flat = packed.reshape(-1, TOKEN_BYTES)
    if not flat.is_contiguous():
        flat = flat.contiguous()

    geom = packed_record_geometry()
    nope = flat[:, geom.nope_start : geom.nope_end].view(torch.float8_e4m3fn)
    scales = flat[:, geom.scale_start : geom.scale_end].view(torch.float32)
    rope = flat[:, geom.rope_start : geom.rope_end].view(torch.bfloat16)

    out = torch.empty(
        (flat.shape[0], SOURCE_DIM), dtype=torch.bfloat16, device=packed.device
    )
    for tile in range(NUM_SCALES):
        lo = tile * QUANT_TILE
        hi = lo + QUANT_TILE
        block = nope[:, lo:hi].to(torch.float32) * scales[:, tile : tile + 1]
        out[:, lo:hi] = block.to(torch.bfloat16)
    out[:, NOPE_DIM:] = rope
    return out.view(*packed.shape[:-1], SOURCE_DIM)
