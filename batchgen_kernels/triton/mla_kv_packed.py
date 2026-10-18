"""The one quantizer of the packed FP8 MLA KV record.

``quantize_mla_kv_packed`` turns ``[tokens, 576]`` bf16 compressed KV into
``[tokens, 656]`` uint8 bytes in FlashMLA's packed FP8 layout. It is called
EXACTLY ONCE per token, at the writer -- the decode graph segment, the eager
decode selector, and the prefill host-offload paths -- and every hop after it
(the GPU paged-cache store, the staging buffer, the D2H host append, the H2D
decode-admission load, a prefix-reuse span) is a byte copy of those 656 bytes.
That single quantization point is what makes "the GPU cache row equals the
staged host row equals the shm page" an assertable invariant instead of a
hope, and it is why the cache store stays the existing dtype-agnostic bare-copy
kernel rather than gaining a second quantize.

The byte layout, the numerics, and the one deliberate divergence from SGLang's
shipped quantizer all live in ``batchgen_kernels.attention.mla_kv_packed``,
which is torch-only and is the byte-exactness reference this kernel is tested
against.

That byte contract is SAME-DEVICE: this kernel is pinned byte-for-byte to the
reference evaluated on the SAME GPU, not to the reference evaluated on the
host. ``scale = fp32_amax / 448`` rounds differently on host and device, and a
1-ULP scale difference flips payload bytes wherever a rescaled element lands
on an e4m3 rounding boundary -- measured on the node, kernel vs GPU reference
is 0 differing bytes across 1..1024 tokens while CPU vs GPU reference differs
in up to 2495 bytes at 1024 tokens, most of them scale bytes. So the CPU
reference is the dev-measurable description of the RECORD (geometry, layout,
dequant semantics, the zero-block divergence -- all of
``tests/test_glm52_packed_mla_kv_record.py``, which compares CPU against CPU),
and it is NOT a cross-device byte oracle for this kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from batchgen_kernels.attention.mla_kv_packed import (
    FP8_MAX,
    NOPE_DIM,
    NUM_SCALES,
    QUANT_TILE,
    ROPE_DIM,
    SOURCE_DIM,
    TOKEN_BYTES,
    validate_packed_quantize_operands,
)

__all__ = ["quantize_mla_kv_packed"]


@triton.jit
def _quantize_mla_kv_packed_kernel(
    nope_q_ptr,  # float8_e4m3 view of out[:, 0:512]
    scale_ptr,  # float32 view of out[:, 512:528]
    rope_ptr,  # bfloat16 view of out[:, 528:656]
    kv_ptr,  # bfloat16 [tokens, 576]
    nope_q_stride_0,
    scale_stride_0,
    rope_stride_0,
    kv_stride_0,
    num_tokens,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN_VAL: tl.constexpr,
    FP8_MAX_VAL: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)
    if token_id >= num_tokens:
        return

    if raw_block_id < NUM_NOPE_BLOCKS:
        # One 128-element NoPE block: amax -> fp32 scale -> e4m3 bytes.
        offs = raw_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        y = tl.load(kv_ptr + token_id * kv_stride_0 + offs, mask=mask, other=0.0)
        y = y.to(tl.float32)

        y_s = tl.max(tl.abs(y)) / FP8_MAX_VAL
        # An all-zero block is the one input where 1/y_s is not finite. Select
        # a unit divisor for the reciprocal and then force the result to zero,
        # so the stored record is (scale 0.0f, data 0x00) and dequantizes back
        # to exactly 0 -- see the DELIBERATE DIVERGENCE note on
        # ``mla_kv_packed._quantize_block``. Masking only the output would
        # still evaluate 0 * inf = NaN on this lane first.
        is_zero = y_s == 0.0
        safe_s = tl.where(is_zero, 1.0, y_s)
        y_q = tl.clamp(y * (1.0 / safe_s), FP8_MIN_VAL, FP8_MAX_VAL)
        y_q = tl.where(is_zero, 0.0, y_q)

        tl.store(
            nope_q_ptr + token_id * nope_q_stride_0 + offs,
            y_q.to(nope_q_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            scale_ptr + token_id * scale_stride_0 + raw_block_id,
            tl.where(is_zero, 0.0, y_s),
        )
    else:
        # RoPE is stored unquantized: copy the bf16 half verbatim.
        offs = (raw_block_id - NUM_NOPE_BLOCKS) * GROUP_SIZE + tl.arange(
            0, GROUP_SIZE
        )
        mask = offs < DIM_ROPE
        data = tl.load(
            kv_ptr + token_id * kv_stride_0 + DIM_NOPE + offs,
            mask=mask,
            other=0.0,
        )
        tl.store(rope_ptr + token_id * rope_stride_0 + offs, data, mask=mask)


def quantize_mla_kv_packed(
    kv: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Quantize ``[tokens, 576]`` bf16 compressed KV into packed FP8 bytes.

    Args:
        kv: bf16 ``[tokens, 576]``, contiguous last dim. ``[:, :512]`` is the
            NoPE latent that gets quantized, ``[:, 512:]`` the RoPE half that
            is copied verbatim.
        out: optional uint8 ``[tokens, 656]`` destination, contiguous. Passing
            the destination is the normal case: the decode writer quantizes
            straight into the buffer the host append stages, so the packed
            bytes exist exactly once.

    Returns:
        ``out`` (allocated when not supplied).
    """

    if out is None:
        out = torch.empty(
            (kv.shape[0], TOKEN_BYTES), dtype=torch.uint8, device=kv.device
        )
    num_tokens = validate_packed_quantize_operands(kv, out)
    if num_tokens == 0:
        return out

    # Typed views over the three byte regions of every record. `out` is
    # contiguous [tokens, 656], so each region's row stride is 656 bytes
    # expressed in that region's element size.
    nope_q = out[:, :NOPE_DIM].view(torch.float8_e4m3fn)
    scales = out[:, NOPE_DIM : NOPE_DIM + NUM_SCALES * 4].view(torch.float32)
    rope = out[:, NOPE_DIM + NUM_SCALES * 4 :].view(torch.bfloat16)

    num_blocks_per_token = triton.cdiv(SOURCE_DIM, QUANT_TILE)

    _quantize_mla_kv_packed_kernel[(num_tokens, num_blocks_per_token)](
        nope_q,
        scales,
        rope,
        kv,
        nope_q.stride(0),
        scales.stride(0),
        rope.stride(0),
        kv.stride(0),
        num_tokens,
        NUM_NOPE_BLOCKS=NOPE_DIM // QUANT_TILE,
        GROUP_SIZE=QUANT_TILE,
        DIM_NOPE=NOPE_DIM,
        DIM_ROPE=ROPE_DIM,
        FP8_MIN_VAL=-FP8_MAX,
        FP8_MAX_VAL=FP8_MAX,
    )
    return out
