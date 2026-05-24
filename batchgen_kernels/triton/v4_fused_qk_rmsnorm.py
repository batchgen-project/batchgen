# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_rmsnorm_kernel(
    qr_ptr,
    qr_out_ptr,
    qr_stride_t,
    qr_stride_h,
    qr_out_stride_t,
    qr_out_stride_h,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_stride_t,
    kv_out_stride_t,
    eps,
    N_HEADS: tl.constexpr,
    Q_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)
    offs = tl.arange(0, BLOCK_SIZE)

    if pid_task < N_HEADS:
        size = Q_DIM
        mask = offs < size
        row_in = (
            qr_ptr
            + pid_token * qr_stride_t
            + pid_task.to(tl.int64) * qr_stride_h
        )
        row_out = (
            qr_out_ptr
            + pid_token * qr_out_stride_t
            + pid_task.to(tl.int64) * qr_out_stride_h
        )
        x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / size
        rrms = tl.rsqrt(variance + eps)
        y = x * rrms
        tl.store(row_out + offs, y.to(row_out.dtype.element_ty), mask=mask)
    else:
        size = KV_DIM
        mask = offs < size
        row_in = kv_ptr + pid_token * kv_stride_t
        row_out = kv_out_ptr + pid_token * kv_out_stride_t
        x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(kv_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / size
        rrms = tl.rsqrt(variance + eps)
        y = x * rrms * w
        tl.store(row_out + offs, y.to(row_out.dtype.element_ty), mask=mask)


def fused_qk_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm on Q (no weight) + global RMSNorm on KV (with weight).

    Shapes:
        qr:        [T, n_heads, head_dim] bfloat16, last dim contiguous
        kv:        [T, kv_dim]            bfloat16, last dim contiguous
        kv_weight: [kv_dim]               float32,  contiguous

    Semantics (math in fp32, output cast to bf16):
        qr_out[t, h, :] = qr[t, h, :] * rsqrt(mean(qr[t, h, :]**2) + eps)
        kv_out[t, :]    = kv[t, :]    * rsqrt(mean(kv[t, :]**2)   + eps) * kv_weight

    Grid is (num_tokens, n_heads + 1): the first n_heads programs per token
    handle one Q head each (per-head reduction); the last program handles the
    KV row (global reduction + weight).
    """
    assert (
        qr.ndim == 3
    ), f"qr must be [T, n_heads, head_dim], got {tuple(qr.shape)}"
    assert kv.ndim == 2, f"kv must be [T, kv_dim], got {tuple(kv.shape)}"
    assert (
        qr.shape[0] == kv.shape[0]
    ), f"token dim mismatch: qr={tuple(qr.shape)}, kv={tuple(kv.shape)}"
    assert qr.is_cuda and kv.is_cuda and kv_weight.is_cuda
    assert qr.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert kv_weight.dtype == torch.float32
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert kv_weight.is_contiguous()
    assert (
        kv.shape[1] == kv_weight.shape[0]
    ), f"weight dim mismatch: kv={tuple(kv.shape)}, kv_weight={tuple(kv_weight.shape)}"

    num_tokens, n_heads, head_dim = qr.shape
    kv_dim = kv.shape[1]
    assert n_heads >= 1, "n_heads must be >= 1"

    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    block_size = triton.next_power_of_2(max(head_dim, kv_dim))
    assert (
        block_size <= 8192
    ), f"head_dim/kv_dim too large for single-pass kernel: {head_dim}/{kv_dim}"

    _fused_qk_rmsnorm_kernel[(num_tokens, n_heads + 1)](
        qr,
        qr_out,
        qr.stride(0),
        qr.stride(1),
        qr_out.stride(0),
        qr_out.stride(1),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        eps,
        N_HEADS=n_heads,
        Q_DIM=head_dim,
        KV_DIM=kv_dim,
        BLOCK_SIZE=block_size,
    )
    return qr_out, kv_out
