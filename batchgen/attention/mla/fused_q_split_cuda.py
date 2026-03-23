"""CUDA kernel: fused q_b output split into q_nope + q_pe (contiguous).

Replaces: q.view(bsz,1,H,q_head_dim).transpose(1,2) → split([nope,rope]) → q_pe.contiguous()

The q_b_proj output is [bsz, H*q_head_dim] = [bsz, 12288] where q_head_dim=192.
Each head's 192 dims split into q_nope[128] + q_pe[64].

This kernel reads the flat q_b output and writes:
  - q_nope: [bsz, H, nope_dim] contiguous  (for einsum in q_absorb)
  - q_pe:   [bsz, H, 1, rope_dim] contiguous (for RoPE kernel)

Grid: (H, bsz) — one block per (head, batch)
Block: max(nope_dim, rope_dim) threads
"""

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void fused_q_split_kernel(
    const __nv_bfloat16* __restrict__ q_flat,  // [B, H * q_head_dim]
    __nv_bfloat16* __restrict__ q_nope,        // [B, H, nope_dim]
    __nv_bfloat16* __restrict__ q_pe,          // [B, H, 1, rope_dim]
    int B, int H, int q_head_dim, int nope_dim, int rope_dim
) {
    int head = blockIdx.x;
    int batch = blockIdx.y;
    if (batch >= B) return;

    int tid = threadIdx.x;

    // Source offset: q_flat[batch, head * q_head_dim + ...]
    int src_base = batch * H * q_head_dim + head * q_head_dim;

    // Copy nope part: q_flat[..., 0:nope_dim] → q_nope[batch, head, :]
    if (tid < nope_dim) {
        int dst_idx = batch * H * nope_dim + head * nope_dim + tid;
        q_nope[dst_idx] = q_flat[src_base + tid];
    }

    // Copy rope part: q_flat[..., nope_dim:nope_dim+rope_dim] → q_pe[batch, head, 0, :]
    if (tid < rope_dim) {
        int dst_idx = batch * H * rope_dim + head * rope_dim + tid;
        q_pe[dst_idx] = q_flat[src_base + nope_dim + tid];
    }
}

void fused_q_split_forward(
    torch::Tensor q_flat,    // [B, H * q_head_dim]
    torch::Tensor q_nope,    // [B, H, nope_dim]
    torch::Tensor q_pe       // [B, H, 1, rope_dim]
) {
    int B = q_flat.size(0);
    int total_dim = q_flat.size(1);
    int nope_dim = q_nope.size(2);
    int rope_dim = q_pe.size(3);
    int H = total_dim / (nope_dim + rope_dim);

    dim3 grid(H, B);
    int threads = (nope_dim > rope_dim) ? nope_dim : rope_dim;
    threads = ((threads + 31) / 32) * 32;

    fused_q_split_kernel<<<grid, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        B, H, nope_dim + rope_dim, nope_dim, rope_dim
    );
}
"""

_CPP_SRC = r"""
void fused_q_split_forward(torch::Tensor q_flat, torch::Tensor q_nope, torch::Tensor q_pe);
"""

_module = None


def _load():
    global _module
    if _module is not None:
        return _module
    _module = load_inline(
        name="fused_q_split_cuda",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_q_split_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )
    return _module


def fused_q_split(
    q_flat: torch.Tensor,       # [bsz, H * q_head_dim] — q_b_proj output
    num_heads: int = 64,
    nope_dim: int = 128,
    rope_dim: int = 64,
    q_nope: torch.Tensor = None,  # [bsz, H, nope_dim] pre-allocated
    q_pe: torch.Tensor = None,    # [bsz, H, 1, rope_dim] pre-allocated
):
    """Fused split of q_b output into q_nope + q_pe (both contiguous).

    Replaces:
        q = q_flat.view(bsz, 1, H, q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [nope_dim, rope_dim], dim=-1)
        q_pe = q_pe.contiguous()

    Returns:
        q_nope: [bsz, H, nope_dim] contiguous
        q_pe: [bsz, H, 1, rope_dim] contiguous
    """
    bsz = q_flat.shape[0]

    if q_nope is None:
        q_nope = torch.empty(bsz, num_heads, nope_dim, dtype=q_flat.dtype, device=q_flat.device)
    if q_pe is None:
        q_pe = torch.empty(bsz, num_heads, 1, rope_dim, dtype=q_flat.dtype, device=q_flat.device)

    mod = _load()
    mod.fused_q_split_forward(q_flat.contiguous(), q_nope, q_pe)

    return q_nope, q_pe
