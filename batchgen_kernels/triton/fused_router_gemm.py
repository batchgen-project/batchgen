"""Router GEMM + bias Triton kernel for SM100 (Blackwell).

Pure-Triton replacement for the WGMMA half of the fused-gate kernel
(`fused_gate_forward`) whose Hopper-only `.cu` is not built on sm_100a.
Computes the router logits

    logits[N, E] = hidden[N, K_dim] @ weight[E, K_dim].T  (+ bias[E])

in FP32 (FP32 tensor-core accumulation) to match the SM90a kernel's output
dtype and precision. The subsequent TopK+Softmax stage is a generic CUDA
kernel (`gate_topk_softmax`) already compiled for sm100.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['N', 'K_dim', 'E'],
)
@triton.jit
def _router_gemm_bias_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, K_dim, E,
    stride_xn, stride_xk,
    stride_wk, stride_we,
    stride_on, stride_oe,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_we)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K_dim, BLOCK_K):
        k_mask = offs_k[None, :] < (K_dim - k_start)
        x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < N) & k_mask, other=0.0)
        w_tile = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < (K_dim - k_start)) & (offs_n[None, :] < E),
            other=0.0,
        )
        acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < E, other=0.0)
        acc += bias[None, :].to(tl.float32)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_on + offs_n[None, :] * stride_oe)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < N) & (offs_n[None, :] < E))


def router_gemm_bias(
    hidden: torch.Tensor,        # [N, K_dim] BF16
    weight: torch.Tensor,        # [E, K_dim] BF16 (nn.Linear weight, row-major)
    bias: torch.Tensor = None,   # [E] BF16/FP32 or None
) -> torch.Tensor:
    """Router GEMM + optional bias. Returns logits [N, E] FP32.

    Args:
        hidden: Input activations [N, K_dim] BF16.
        weight: Router weight [E, K_dim] BF16 (nn.Linear layout).
        bias: Optional router bias [E].

    Returns:
        logits [N, E] FP32 (FP32 accumulation, matches SM90a output dtype).
    """
    assert hidden.dim() == 2, f"hidden must be 2D, got {hidden.shape}"
    N, K_dim = hidden.shape
    E = weight.shape[0]
    assert weight.shape[1] == K_dim, (
        f"weight K mismatch: hidden K={K_dim}, weight K={weight.shape[1]}"
    )

    hidden = hidden.contiguous()
    # [E, K_dim] -> [K_dim, E] so the inner GEMM dim is contiguous-friendly.
    w_t = weight.t().contiguous()

    has_bias = bias is not None
    bias_arg = bias if has_bias else hidden  # dummy ptr when no bias
    if has_bias and bias.dtype != torch.float32 and bias.dtype != torch.bfloat16:
        bias_arg = bias.to(torch.float32)

    logits = torch.empty((N, E), dtype=torch.float32, device=hidden.device)

    grid = lambda meta: (
        triton.cdiv(N, meta['BLOCK_M']),
        triton.cdiv(E, meta['BLOCK_N']),
    )
    _router_gemm_bias_kernel[grid](
        hidden, w_t, bias_arg, logits,
        N, K_dim, E,
        hidden.stride(0), hidden.stride(1),
        w_t.stride(0), w_t.stride(1),
        logits.stride(0), logits.stride(1),
        HAS_BIAS=has_bias,
    )
    return logits
