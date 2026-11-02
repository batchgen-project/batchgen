#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#define FP8_MAX_VAL 448.0f
#define QUANT_EPS 1e-12f
#define BLOCK_SIZE_QUANT 128

// ============================================================================
// act_quant_3d — FP8 quantization with transposed scale output (v2 layout).
// Input:  x [E, mtp, K] BF16
// Output: y [E, mtp, K] uint8 (FP8), scale_t [num_k_blocks, E*mtp] F32
//
// v2: grid = (E, mtp), one CTA per (expert, token). Padded slots
// (token >= tokens_per_expert[expert]) early-exit. Same numerics as v1
// but parallelizes the token axis instead of serializing the
// `for m in 0..valid_tokens` loop, raising occupancy from ~12%
// (v1: 16 CTAs at GLM-5 EP=16 on 132 SMs) to fully populating the SMs.
// Bit-exact match against v1 — see tests/kernels/test_act_quant_3d_v2.py.
// ============================================================================
__global__ void act_quant_3d_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ y,
    float* __restrict__ scale_t,      // [num_k_blocks, E * mtp] — transposed
    const int32_t* __restrict__ tokens_per_expert,
    int E, int mtp, int K, int num_k_blocks
) {
    const int expert = blockIdx.x;
    const int token  = blockIdx.y;
    const int valid_tokens = tokens_per_expert[expert];
    if (token >= valid_tokens) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const int64_t m_global = (int64_t)expert * mtp + token;
    const int64_t total_m = (int64_t)E * mtp;
    const __nv_bfloat16* x_row = x + m_global * K;
    uint8_t* y_row = y + m_global * K;

    for (int kb = warp_id; kb < num_k_blocks; kb += num_warps) {
        int col_base = kb * BLOCK_SIZE_QUANT;

        float vals[4];
        float local_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col = col_base + lane_id * 4 + i;
            if (col < K) {
                vals[i] = __bfloat162float(x_row[col]);
            } else {
                vals[i] = 0.0f;
            }
            local_max = fmaxf(local_max, fabsf(vals[i]));
        }

        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, other);
        }

        float s = fmaxf(local_max, QUANT_EPS) / FP8_MAX_VAL;
        float inv_s = 1.0f / s;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col = col_base + lane_id * 4 + i;
            if (col < K) {
                float scaled = vals[i] * inv_s;
                scaled = fmaxf(fminf(scaled, FP8_MAX_VAL), -FP8_MAX_VAL);
                y_row[col] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
            }
        }

        // Store scale in transposed layout: scale_t[kb, m_global]
        if (lane_id == 0) {
            scale_t[(int64_t)kb * total_m + m_global] = s;
        }
    }
}

// ============================================================================
// fused_silu_quant_3d — SiLU×gate + FP8 quant with transposed scale output
// Input:  gate [E, mtp, N] BF16, up [E, mtp, N] BF16
// Output: y [E, mtp, N] uint8 (FP8), scale_t [num_n_blocks, E*mtp] F32
// ============================================================================
__global__ void fused_silu_quant_3d_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t* __restrict__ y,
    float* __restrict__ scale_t,      // [num_n_blocks, E * mtp] — transposed
    const int32_t* __restrict__ tokens_per_expert,
    int E, int mtp, int N, int num_n_blocks
) {
    const int expert = blockIdx.x;
    const int valid_tokens = tokens_per_expert[expert];
    if (valid_tokens == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const int64_t expert_offset = (int64_t)expert * mtp * N;
    const int64_t total_m = (int64_t)E * mtp;

    for (int m = 0; m < valid_tokens; m++) {
        const int64_t row_offset = expert_offset + (int64_t)m * N;
        const int64_t m_global = (int64_t)expert * mtp + m;

        for (int kb = warp_id; kb < num_n_blocks; kb += num_warps) {
            int col_base = kb * BLOCK_SIZE_QUANT;

            float vals[4];
            float local_max = 0.0f;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col_base + lane_id * 4 + i;
                if (col < N) {
                    float g = __bfloat162float(gate[row_offset + col]);
                    float u = __bfloat162float(up[row_offset + col]);
                    float silu_g = g / (1.0f + expf(-g));
                    vals[i] = silu_g * u;
                } else {
                    vals[i] = 0.0f;
                }
                local_max = fmaxf(local_max, fabsf(vals[i]));
            }

            #pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                float other = __shfl_xor_sync(0xffffffff, local_max, offset);
                local_max = fmaxf(local_max, other);
            }

            float s = fmaxf(local_max, QUANT_EPS) / FP8_MAX_VAL;
            float inv_s = 1.0f / s;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col_base + lane_id * 4 + i;
                if (col < N) {
                    float scaled = vals[i] * inv_s;
                    scaled = fmaxf(fminf(scaled, FP8_MAX_VAL), -FP8_MAX_VAL);
                    y[row_offset + col] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
                }
            }

            // Store scale in transposed layout: scale_t[kb, m_global]
            if (lane_id == 0) {
                scale_t[(int64_t)kb * total_m + m_global] = s;
            }
        }
    }
}

// ============================================================================
// C++ wrappers
// ============================================================================
void act_quant_3d(
    torch::Tensor x,
    torch::Tensor tokens_per_expert,
    torch::Tensor out_fp8,
    torch::Tensor out_scale_t
) {
    TORCH_CHECK(x.dim() == 3, "x must be 3D [E, mtp, K]");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int E = x.size(0);
    int mtp = x.size(1);
    int K = x.size(2);
    int num_k_blocks = (K + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // v2: grid = (E, mtp) — one CTA per (expert, token).
    dim3 grid_act(E, mtp);
    act_quant_3d_kernel<<<grid_act, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        out_fp8.data_ptr<uint8_t>(),
        out_scale_t.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(),
        E, mtp, K, num_k_blocks);
}

void fused_silu_quant_3d(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor tokens_per_expert,
    torch::Tensor out_fp8,
    torch::Tensor out_scale_t
) {
    TORCH_CHECK(gate.dim() == 3 && up.dim() == 3, "Inputs must be 3D");
    TORCH_CHECK(gate.sizes() == up.sizes(), "Shape mismatch");

    int E = gate.size(0);
    int mtp = gate.size(1);
    int N = gate.size(2);
    int num_n_blocks = (N + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_silu_quant_3d_kernel<<<E, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        out_fp8.data_ptr<uint8_t>(),
        out_scale_t.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(),
        E, mtp, N, num_n_blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("act_quant_3d", torch::wrap_pybind_function(act_quant_3d),
          "act_quant_3d");
    m.def("fused_silu_quant_3d", torch::wrap_pybind_function(fused_silu_quant_3d),
          "fused_silu_quant_3d");
}
