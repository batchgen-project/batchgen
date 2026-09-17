"""INT4 (W4A16) GEMM Triton kernels for SM100 (Blackwell).

Pure-Triton port of the Hopper WGMMA INT4 grouped MoE kernels (K2.5 decode)
whose `.cu` uses `wgmma`/TMA intrinsics not built for sm_100a.

Weight / scale layout (matches the SM90a `.cu`, e.g.
`single_expert_int4_wgmma.cu::load_decode_rhs_int4_swizzled`):
  * `w_packed` : [N, K // 2] uint8 — output-channel major, two INT4 values per
    byte along K. The **low** nibble `(byte & 0x0F)` is the even K index, the
    **high** nibble `((byte >> 4) & 0x0F)` is the odd K index.
  * `scale`    : [N, K // group_size] bf16 — one scale per group of
    `group_size` (=32) contiguous K values, shared across the group.
  * dequant    : `w[n, k] = (nibble - 8) * scale[n, k // group_size]`
  * GEMM       : `out[m, n] = sum_k x[m, k] * w[n, k]`  (i.e. `x @ dequant(W).T`).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_KP': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_KP': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_KP': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_KP': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'K', 'N', 'GROUP_SIZE'],
)
@triton.jit
def _int4_gemm_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,      # w_packed: [N, K//2] uint8
    stride_sn, stride_sk,      # scale:    [N, K//group_size] bf16
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KP: tl.constexpr,    # packed-K (bytes) per iteration; real-K = 2*BLOCK_KP
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kp = tl.arange(0, BLOCK_KP)

    Kp = K // 2
    # packed-K bytes per scale group (group_size K-values -> group_size//2 bytes)
    GROUP_BYTES = GROUP_SIZE // 2

    m_mask = offs_m[:, None] < M
    n_mask_out = offs_n[None, :] < N      # [1, BLOCK_N] for output/x-side
    n_mask_w = offs_n[:, None] < N        # [BLOCK_N, 1] for weight/scale rows

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kp_start in range(0, Kp, BLOCK_KP):
        kp = kp_start + offs_kp                      # [BLOCK_KP] packed indices
        kp_mask = kp < Kp

        # x even / odd columns: x[:, 2*kp] and x[:, 2*kp+1]
        x_even = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (2 * kp[None, :]) * stride_xk,
            mask=m_mask & kp_mask[None, :], other=0.0,
        )
        x_odd = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (2 * kp[None, :] + 1) * stride_xk,
            mask=m_mask & kp_mask[None, :], other=0.0,
        )

        # packed weight byte [BLOCK_N, BLOCK_KP]
        wb = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + kp[None, :] * stride_wk,
            mask=n_mask_w & kp_mask[None, :], other=0,
        )
        lo = ((wb & 0x0F).to(tl.int32) - 8).to(tl.bfloat16)
        hi = (((wb >> 4) & 0x0F).to(tl.int32) - 8).to(tl.bfloat16)

        # scale [BLOCK_N, BLOCK_KP]: group index = (2*kp) // group_size = kp // GROUP_BYTES
        sg = kp // GROUP_BYTES
        sc = tl.load(
            scale_ptr + offs_n[:, None] * stride_sn + sg[None, :] * stride_sk,
            mask=n_mask_w & kp_mask[None, :], other=0.0,
        )
        w_lo = (lo * sc)          # [BLOCK_N, BLOCK_KP] bf16
        w_hi = (hi * sc)

        acc += tl.dot(x_even, tl.trans(w_lo), out_dtype=tl.float32)
        acc += tl.dot(x_odd, tl.trans(w_hi), out_dtype=tl.float32)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask & n_mask_out)


def int4_grouped_gemm(
    x: torch.Tensor,          # [M, K] BF16
    w_packed: torch.Tensor,   # [N, K // 2] uint8 (INT4 packed, output-major)
    scales: torch.Tensor,     # [N, K // group_size] BF16
    group_size: int = 32,
) -> torch.Tensor:
    """Single-expert INT4×BF16 GEMM with per-group dequant. Returns [M, N] BF16.

    Computes `out = x @ dequant(w_packed, scales).T` where
    `dequant[n, k] = ((nibble(n,k) - 8) * scales[n, k // group_size])`.
    """
    assert x.dim() == 2, f"x must be 2D, got {x.shape}"
    M, K = x.shape
    if w_packed.dtype == torch.int32:
        w_packed = w_packed.view(torch.uint8)
    assert w_packed.dtype == torch.uint8, f"w_packed must be uint8, got {w_packed.dtype}"
    N, Kp = w_packed.shape
    assert Kp == K // 2, f"w_packed K mismatch: x K={K} -> expected {K // 2}, got {Kp}"
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"

    x = x.contiguous()
    w_packed = w_packed.contiguous()
    scales = scales.contiguous()

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    _int4_gemm_kernel[grid](
        x, w_packed, scales, out,
        M, K, N,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=group_size,
    )
    return out


def int4_moe_grouped_gemm(
    x: torch.Tensor,                  # [total_M, K] BF16 (sorted by expert)
    w_packed_list,                    # list[Tensor [N, K//2] uint8] per expert
    scales_list,                      # list[Tensor [N, K//group_size] bf16] per expert
    m_offsets,                        # [num_experts + 1] int (row offsets into x)
    group_size: int = 32,
) -> torch.Tensor:
    """Batched INT4 MoE GEMM across experts (decode: small M per expert).

    Tokens are assumed pre-sorted by expert; `m_offsets[e]:m_offsets[e+1]` is the
    contiguous row range for expert `e`. Returns [total_M, N] BF16.
    """
    total_M, K = x.shape
    N = w_packed_list[0].shape[0]
    out = torch.empty((total_M, N), dtype=torch.bfloat16, device=x.device)
    num_experts = len(w_packed_list)
    offs = m_offsets.tolist() if torch.is_tensor(m_offsets) else list(m_offsets)
    for e in range(num_experts):
        lo, hi = offs[e], offs[e + 1]
        if hi <= lo:
            continue
        out[lo:hi] = int4_grouped_gemm(
            x[lo:hi], w_packed_list[e], scales_list[e], group_size=group_size,
        )
    return out
