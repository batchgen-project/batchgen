"""CUDA implementation of fused q_absorb + query_states construction.

Replaces the Triton prototype with a CUDA kernel using warp-level reduction.

Grid: (H=64, B) — one block per (head, batch) pair.
Each block has 512 threads, one per output C-dim element.
Each thread reduces across D=128 input elements.
Then 64 threads copy q_pe values.
"""

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// fused_q_absorb_kernel:
//   For each (head, batch):
//     output[b, 0, h, 0:C] = q_nope[b, h, :] @ q_absorb[h, :, :]  (GEMV: [D] x [D, C] -> [C])
//     output[b, 0, h, C:C+R] = q_pe[b, h, 0, :]                    (copy R values)
//
// Grid: (H, B)
// Block: max(C, R) threads (at least 512 for C=512)

__global__ void fused_q_absorb_kernel(
    const __nv_bfloat16* __restrict__ q_nope,      // [B, H, D]
    const __nv_bfloat16* __restrict__ q_absorb,     // [H, D, C]
    const __nv_bfloat16* __restrict__ q_pe,         // [B, H, 1, R]
    __nv_bfloat16* __restrict__ output,             // [B, 1, H, C+R]
    int B, int H, int D, int C, int R
) {
    int head = blockIdx.x;
    int batch = blockIdx.y;
    if (batch >= B) return;

    extern __shared__ __nv_bfloat16 smem[];  // [D] BF16

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int CR = C + R;

    // Load q_nope[b, h, :] into shared memory (D=128, coalesced)
    const __nv_bfloat16* qn = q_nope + (batch * H + head) * D;
    for (int i = tid; i < D; i += nthreads) {
        smem[i] = qn[i];
    }
    __syncthreads();

    // Phase 1: GEMV — each thread computes one output element
    // absorbed[tid] = sum_d smem[d] * q_absorb[h, d, tid]
    if (tid < C) {
        const __nv_bfloat16* qa = q_absorb + head * D * C;
        float acc = 0.0f;

        // Unrolled reduction across D=128
        int d = 0;
        for (; d + 7 < D; d += 8) {
            float s0 = __bfloat162float(smem[d]);
            float s1 = __bfloat162float(smem[d+1]);
            float s2 = __bfloat162float(smem[d+2]);
            float s3 = __bfloat162float(smem[d+3]);
            float s4 = __bfloat162float(smem[d+4]);
            float s5 = __bfloat162float(smem[d+5]);
            float s6 = __bfloat162float(smem[d+6]);
            float s7 = __bfloat162float(smem[d+7]);

            acc += s0 * __bfloat162float(qa[(d  ) * C + tid]);
            acc += s1 * __bfloat162float(qa[(d+1) * C + tid]);
            acc += s2 * __bfloat162float(qa[(d+2) * C + tid]);
            acc += s3 * __bfloat162float(qa[(d+3) * C + tid]);
            acc += s4 * __bfloat162float(qa[(d+4) * C + tid]);
            acc += s5 * __bfloat162float(qa[(d+5) * C + tid]);
            acc += s6 * __bfloat162float(qa[(d+6) * C + tid]);
            acc += s7 * __bfloat162float(qa[(d+7) * C + tid]);
        }
        for (; d < D; d++) {
            acc += __bfloat162float(smem[d]) * __bfloat162float(qa[d * C + tid]);
        }

        int out_idx = batch * H * CR + head * CR + tid;
        output[out_idx] = __float2bfloat16(acc);
    }

    // Phase 2: Copy q_pe to output[b, 0, h, C:]
    if (tid < R) {
        int pe_idx = (batch * H + head) * R + tid;
        int out_idx = batch * H * CR + head * CR + C + tid;
        output[out_idx] = q_pe[pe_idx];
    }
}

torch::Tensor fused_q_absorb_forward(
    torch::Tensor q_nope,       // [B, H, D]
    torch::Tensor q_absorb,     // [H, D, C]
    torch::Tensor q_pe,         // [B, H, 1, R]
    torch::Tensor output        // [B, 1, H, C+R] pre-allocated
) {
    int B = q_nope.size(0);
    int H = q_nope.size(1);
    int D = q_nope.size(2);
    int C = q_absorb.size(2);
    int R = q_pe.size(3);

    dim3 grid(H, B);
    int threads = (C > R) ? C : R;
    // Round up to warp multiple
    threads = ((threads + 31) / 32) * 32;
    int smem_bytes = D * sizeof(__nv_bfloat16);

    fused_q_absorb_kernel<<<grid, threads, smem_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_absorb.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        B, H, D, C, R
    );

    return output;
}
"""

_CPP_SRC = r"""
torch::Tensor fused_q_absorb_forward(
    torch::Tensor q_nope,
    torch::Tensor q_absorb,
    torch::Tensor q_pe,
    torch::Tensor output
);
"""

_module = None


def _load():
    global _module
    if _module is not None:
        return _module
    _module = load_inline(
        name="fused_q_absorb_cuda",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_q_absorb_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )
    return _module


def fused_q_absorb_query_states_cuda(
    q_nope: torch.Tensor,      # [bsz, H, D] squeezed
    q_absorb: torch.Tensor,    # [H, D, C]
    q_pe: torch.Tensor,        # [bsz, H, 1, R] contiguous
    output: torch.Tensor = None,
) -> torch.Tensor:
    """CUDA fused q_absorb + query_states construction.

    Returns: [bsz, 1, H, C+R] in flash_mla input layout.
    """
    B, H, D = q_nope.shape
    C = q_absorb.shape[2]
    R = q_pe.shape[-1]

    if output is None:
        output = torch.empty(B, 1, H, C + R, dtype=q_nope.dtype, device=q_nope.device)

    # Ensure contiguous
    q_nope = q_nope.contiguous()
    q_absorb = q_absorb.contiguous()
    q_pe = q_pe.contiguous()

    mod = _load()
    # The kernel writes to output with layout [B, H, C+R] (treating as [B, 1, H, C+R])
    mod.fused_q_absorb_forward(q_nope, q_absorb, q_pe, output.view(B, H, C + R))

    return output
