"""Expert-major masked SiLU*up + per-token-group FP8 quantization.

The activation between the two masked m-grouped GEMMs of a DeepEP low-latency
MoE layer. Both GEMMs run on the uniform-stride receive layout

    [E_local, stride, dim]      stride = num_ranks * tokens_per_rank

and take their live row count per expert from the exchange's device-side
``recv_count``, so this kernel has to do the same: rows at or beyond
``masked_m[e]`` are neither read nor written, and the grid is derived from the
PADDED extent only. That is what makes the whole chain CUDA-graph safe — the
launch geometry is a function of the captured shapes, and the only per-step
input is the contents of a device tensor.

Layouts
-------
``gate`` / ``up``   [E, S, I]  bf16, last dim contiguous
``out``             [E, S, I]  float8_e4m3fn, last dim contiguous
``out_scale``       [E, S, I // group_size] f32, ANY strides — the caller
                    passes the MN-major (``stride(-2) == 1``) view the masked
                    GEMM wants for its A-scales, which is a transposed view of
                    a contiguous [E, I // group_size, S] allocation.
``masked_m``        [E] int32, device

Numerics match the reference masked activation of the DeepGEMM MoE runner this
path mirrors: the SiLU is evaluated in fp32 and cast back to the input dtype
BEFORE the multiply by ``up``, so the product and its absmax are bf16; the
group scale is ``absmax / 448`` in fp32 and the quantized value is clamped to
the e4m3 range. It is deliberately not our own ragged activation's fp32
product — the two GEMMs it feeds are DeepGEMM's, and this keeps the whole
R1 -> act -> R2 chain numerically equal to the reference implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP8_DTYPE = torch.float8_e4m3fn


@triton.jit
def _silu_mul_masked_group_quant_kernel(
    gate_ptr,
    gate_stride_e,
    gate_stride_s,
    up_ptr,
    up_stride_e,
    up_stride_s,
    out_ptr,
    out_stride_e,
    out_stride_s,
    scale_ptr,
    scale_stride_e,
    scale_stride_s,
    scale_stride_g,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    expert_id = tl.program_id(2)
    row_block = tl.program_id(1)
    group_block = tl.program_id(0)
    rows_per_launch = tl.num_programs(1)

    live_rows = tl.load(masked_m_ptr + expert_id)

    gate_stride_e = tl.cast(gate_stride_e, dtype=tl.int64)
    gate_stride_s = tl.cast(gate_stride_s, dtype=tl.int64)
    up_stride_e = tl.cast(up_stride_e, dtype=tl.int64)
    up_stride_s = tl.cast(up_stride_s, dtype=tl.int64)
    out_stride_e = tl.cast(out_stride_e, dtype=tl.int64)
    out_stride_s = tl.cast(out_stride_s, dtype=tl.int64)
    scale_stride_e = tl.cast(scale_stride_e, dtype=tl.int64)
    scale_stride_s = tl.cast(scale_stride_s, dtype=tl.int64)

    offs_n = group_block * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = offs_n < size_n
    gate_base = gate_ptr + expert_id * gate_stride_e + offs_n
    up_base = up_ptr + expert_id * up_stride_e + offs_n
    out_base = out_ptr + expert_id * out_stride_e + offs_n
    scale_base = (
        scale_ptr + expert_id * scale_stride_e + group_block * scale_stride_g
    )

    for row in tl.range(
        row_block, live_rows, rows_per_launch, num_stages=NUM_STAGES
    ):
        gate = tl.load(
            gate_base + row * gate_stride_s, mask=col_mask, other=0.0
        ).to(tl.float32)
        up = tl.load(up_base + row * up_stride_s, mask=col_mask, other=0.0)
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(gate_ptr.dtype.element_ty)
        gate_up = up * gate
        absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        group_scale = absmax / fp8_max
        quantized = tl.clamp(gate_up / group_scale, fp8_min, fp8_max).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_base + row * out_stride_s, quantized, mask=col_mask)
        tl.store(scale_base + row * scale_stride_s, group_scale)


def silu_mul_masked_group_quant(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
    out_scale: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    group_size: int = 128,
    rows_per_launch: int = 0,
    num_stages: int = 6,
) -> None:
    """``out, out_scale = quant(silu(gate) * up)`` over the live rows only.

    Every buffer is caller-owned: the kernel allocates nothing and performs no
    host-device synchronization, so it captures. ``rows_per_launch`` is the
    number of row-blocks per expert (the reference runner uses 64 below four
    experts and 32 otherwise); 0 selects that rule.
    """
    if gate.dim() != 3 or up.dim() != 3 or out.dim() != 3:
        raise ValueError(
            "masked act/quant takes expert-major [E, S, I] gate, up and out; "
            f"got {tuple(gate.shape)} / {tuple(up.shape)} / {tuple(out.shape)}"
        )
    if gate.shape != up.shape or gate.shape != out.shape:
        raise ValueError(
            "masked act/quant needs gate, up and out on the same [E, S, I] "
            f"extent; got {tuple(gate.shape)} / {tuple(up.shape)} / "
            f"{tuple(out.shape)}"
        )
    if gate.dtype != up.dtype:
        raise ValueError(
            f"masked act/quant needs one input dtype; got {gate.dtype} gate "
            f"and {up.dtype} up"
        )
    if out.dtype != _FP8_DTYPE:
        raise ValueError(
            f"masked act/quant writes {_FP8_DTYPE}, not {out.dtype}"
        )
    if out_scale.dtype != torch.float32:
        raise ValueError(
            f"masked act/quant writes f32 group scales, not {out_scale.dtype}"
        )
    num_experts, _rows, size_n = gate.shape
    if size_n % group_size != 0:
        raise ValueError(
            f"masked act/quant needs the intermediate size {size_n} to be a "
            f"multiple of the quant group {group_size}"
        )
    if out_scale.shape != (num_experts, gate.shape[1], size_n // group_size):
        raise ValueError(
            "masked act/quant needs out_scale on the logical "
            f"[E, S, I // {group_size}] extent "
            f"{(num_experts, gate.shape[1], size_n // group_size)}; got "
            f"{tuple(out_scale.shape)}"
        )
    if masked_m.shape != (num_experts,):
        raise ValueError(
            f"masked act/quant needs masked_m [{num_experts}]; got "
            f"{tuple(masked_m.shape)}"
        )
    if masked_m.dtype != torch.int32:
        raise ValueError(
            f"masked act/quant needs an int32 masked_m; got {masked_m.dtype}"
        )
    for name, tensor in (("gate", gate), ("up", up), ("out", out)):
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"masked act/quant needs a contiguous last dim on {name}; got "
                f"strides {tuple(tensor.stride())}"
            )

    if rows_per_launch <= 0:
        rows_per_launch = 64 if num_experts < 4 else 32
    grid = (triton.cdiv(size_n, group_size), rows_per_launch, num_experts)
    finfo = torch.finfo(_FP8_DTYPE)

    _silu_mul_masked_group_quant_kernel[grid](
        gate,
        gate.stride(0),
        gate.stride(1),
        up,
        up.stride(0),
        up.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        out_scale,
        out_scale.stride(0),
        out_scale.stride(1),
        out_scale.stride(2),
        masked_m,
        size_n,
        finfo.max,
        -finfo.max,
        BLOCK_N=group_size,
        NUM_STAGES=num_stages,
        num_warps=1,
    )


@triton.jit
def _silu_mul_ragged_group_quant_kernel(
    gate_ptr,
    gate_stride_s,
    up_ptr,
    up_stride_s,
    out_ptr,
    out_stride_s,
    scale_ptr,
    scale_stride_s,
    counts_ptr,
    cu_seqlens_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    expert_id = tl.program_id(2)
    row_block = tl.program_id(1)
    group_block = tl.program_id(0)
    rows_per_launch = tl.num_programs(1)

    seg_start = tl.load(cu_seqlens_ptr + expert_id).to(tl.int64)
    live_rows = tl.load(counts_ptr + expert_id)

    gate_stride_s = tl.cast(gate_stride_s, dtype=tl.int64)
    up_stride_s = tl.cast(up_stride_s, dtype=tl.int64)
    out_stride_s = tl.cast(out_stride_s, dtype=tl.int64)
    scale_stride_s = tl.cast(scale_stride_s, dtype=tl.int64)

    offs_n = group_block * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = offs_n < size_n
    gate_base = gate_ptr + seg_start * gate_stride_s + offs_n
    up_base = up_ptr + seg_start * up_stride_s + offs_n
    out_base = out_ptr + seg_start * out_stride_s + offs_n
    scale_base = scale_ptr + seg_start * scale_stride_s + group_block

    for row in tl.range(
        row_block, live_rows, rows_per_launch, num_stages=NUM_STAGES
    ):
        gate = tl.load(
            gate_base + row * gate_stride_s, mask=col_mask, other=0.0
        ).to(tl.float32)
        up = tl.load(up_base + row * up_stride_s, mask=col_mask, other=0.0)
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(gate_ptr.dtype.element_ty)
        gate_up = up * gate
        absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        group_scale = absmax / fp8_max
        quantized = tl.clamp(gate_up / group_scale, fp8_min, fp8_max).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_base + row * out_stride_s, quantized, mask=col_mask)
        tl.store(scale_base + row * scale_stride_s, group_scale)


def silu_mul_ragged_group_quant(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
    out_scale: torch.Tensor,
    counts: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    group_size: int = 128,
    rows_per_launch: int = 0,
    num_stages: int = 6,
) -> None:
    """``out, out_scale = quant(silu(gate) * up)`` over the live ragged rows.

    The flat-row sibling of :func:`silu_mul_masked_group_quant` for the
    contiguous m-grouped prefill layout: expert ``e``'s live rows sit at
    ``[cu_seqlens[e], cu_seqlens[e] + counts[e])`` of 2-D ``[m, I]`` buffers,
    and the alignment holes between segments are neither read nor written
    (the grouped GEMM skips them by its ``-1`` layout rows, and the caller's
    zero-filled scale buffer keeps their scales finite). ``out_scale`` is the
    ROW-MAJOR fp32 ``[m, I // group_size]`` matrix the contiguous DeepGEMM
    API takes as SFA directly, so no transpose shim runs behind this kernel.

    Same numerics as the masked sibling (the reference DeepGEMM chain): SiLU
    in fp32 cast back to the input dtype before the multiply, bf16 absmax,
    ``absmax / 448`` fp32 group scale, clamped e4m3 store. Every buffer is
    caller-owned, nothing allocates, nothing synchronizes; the segment
    bounds are trusted device tensors exactly like ``masked_m``.
    """
    if gate.dim() != 2 or up.dim() != 2 or out.dim() != 2:
        raise ValueError(
            "ragged act/quant takes flat [m, I] gate, up and out; got "
            f"{tuple(gate.shape)} / {tuple(up.shape)} / {tuple(out.shape)}"
        )
    if gate.shape != up.shape or gate.shape != out.shape:
        raise ValueError(
            "ragged act/quant needs gate, up and out on the same [m, I] "
            f"extent; got {tuple(gate.shape)} / {tuple(up.shape)} / "
            f"{tuple(out.shape)}"
        )
    if gate.dtype != up.dtype:
        raise ValueError(
            f"ragged act/quant needs one input dtype; got {gate.dtype} gate "
            f"and {up.dtype} up"
        )
    if out.dtype != _FP8_DTYPE:
        raise ValueError(
            f"ragged act/quant writes {_FP8_DTYPE}, not {out.dtype}"
        )
    if out_scale.dtype != torch.float32:
        raise ValueError(
            f"ragged act/quant writes f32 group scales, not {out_scale.dtype}"
        )
    rows, size_n = gate.shape
    if size_n % group_size != 0:
        raise ValueError(
            f"ragged act/quant needs the intermediate size {size_n} to be a "
            f"multiple of the quant group {group_size}"
        )
    if out_scale.shape != (rows, size_n // group_size):
        raise ValueError(
            "ragged act/quant needs a row-major out_scale on the "
            f"[m, I // {group_size}] extent {(rows, size_n // group_size)}; "
            f"got {tuple(out_scale.shape)}"
        )
    if out_scale.stride(-1) != 1:
        raise ValueError(
            "ragged act/quant writes row-major scales; got strides "
            f"{tuple(out_scale.stride())}"
        )
    if counts.ndim != 1 or cu_seqlens.shape != (counts.numel() + 1,):
        raise ValueError(
            "ragged act/quant needs counts [E] and cu_seqlens [E + 1]; got "
            f"{tuple(counts.shape)} / {tuple(cu_seqlens.shape)}"
        )
    if counts.dtype != torch.int32 or cu_seqlens.dtype != torch.int32:
        raise ValueError(
            "ragged act/quant needs int32 counts and cu_seqlens; got "
            f"{counts.dtype} / {cu_seqlens.dtype}"
        )
    for name, tensor in (("gate", gate), ("up", up), ("out", out)):
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"ragged act/quant needs a contiguous last dim on {name}; "
                f"got strides {tuple(tensor.stride())}"
            )

    num_experts = counts.numel()
    if rows_per_launch <= 0:
        rows_per_launch = 64 if num_experts < 4 else 32
    grid = (triton.cdiv(size_n, group_size), rows_per_launch, num_experts)
    finfo = torch.finfo(_FP8_DTYPE)

    _silu_mul_ragged_group_quant_kernel[grid](
        gate,
        gate.stride(0),
        up,
        up.stride(0),
        out,
        out.stride(0),
        out_scale,
        out_scale.stride(0),
        counts,
        cu_seqlens,
        size_n,
        finfo.max,
        -finfo.max,
        BLOCK_N=group_size,
        NUM_STAGES=num_stages,
        num_warps=1,
    )
