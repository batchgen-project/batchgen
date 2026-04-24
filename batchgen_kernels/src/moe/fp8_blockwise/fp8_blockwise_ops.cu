// BatchGen — FP8 Blockwise MoE Pipeline Operations
// CUDA kernels for 3D-sparse FP8 quantization and fused SiLU activation.
// All kernels operate on [E, mtp, dim] layout with tokens_per_expert masking.
//
// Kernels:
// 1. act_quant_3d       — BF16→FP8 blockwise quantization (128-element blocks)
// 2. silu_mul_3d        — SiLU(gate) × up (vectorized BF16)
// 3. fused_silu_quant_3d — SiLU(gate) × up + FP8 quantization (fuses S1 epilogue + S3 input)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

#define FP8_MAX_VAL 448.0f
#define QUANT_EPS 1e-12f
#define BLOCK_SIZE_QUANT 128

// ============================================================================
// Kernel 1: act_quant_3d — 3D-sparse FP8 blockwise quantization
// Grid: (E,)   Block: 128 threads (4 warps)
// Each CTA processes one expert, loops over valid tokens only.
// Each warp handles one K-block (128 elements) at a time, warps stride.
// ============================================================================
// v2: grid = (E, mtp), one CTA per (expert, token).
// v1 (grid=(E,) with serial outer token loop) was launch-limited at
// E=16 → only 16 CTAs on H20's ~132 SMs, ~12% occupancy, ~594 µs per
// call at K=6144. v2 saturates the GPU with E×mtp=2048 CTAs and
// runs in ~22 µs (27× faster, validated in
// tests/kernels/test_act_quant_3d_v2.py bit-exact vs v1).
__global__ void act_quant_3d_kernel(
    const __nv_bfloat16* __restrict__ x,   // [E, mtp, K]
    uint8_t* __restrict__ y,                // [E, mtp, K] FP8
    float* __restrict__ scale,              // [E, mtp, num_k_blocks]
    const int32_t* __restrict__ tokens_per_expert,  // [E]
    int mtp, int K, int num_k_blocks
) {
    const int expert = blockIdx.x;
    const int token  = blockIdx.y;
    const int valid_tokens = tokens_per_expert[expert];
    if (token >= valid_tokens) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const __nv_bfloat16* x_row =
        x + ((int64_t)expert * mtp + token) * K;
    uint8_t* y_row =
        y + ((int64_t)expert * mtp + token) * K;
    float* scale_row =
        scale + ((int64_t)expert * mtp + token) * num_k_blocks;

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

        // Match SGLang's FP32 op sequence (`scale = amax * (1/448)`,
        // `scaled = x / scale`) so the CUDA kernel produces byte-exact
        // FP8 vs SGLang's Triton `_act_quant_kernel`. Previously we did
        //     s = fmax(local_max, eps) / FP8_MAX_VAL;  // FP32 division
        //     inv_s = 1.0f / s;                        // FP32 reciprocal
        //     scaled = vals[i] * inv_s;                // FP32 mul
        // which introduces an extra FP32 rounding step and flips the
        // final FP8 cast on ~0.1% of elements (see
        // tests/test_glm5_act_quant_triton_vs_triton.py).
        constexpr float FP8_MAX_VAL_INV = 1.0f / FP8_MAX_VAL;
        float s = fmaxf(local_max, QUANT_EPS) * FP8_MAX_VAL_INV;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col = col_base + lane_id * 4 + i;
            if (col < K) {
                float scaled = vals[i] / s;
                scaled = fmaxf(fminf(scaled, FP8_MAX_VAL), -FP8_MAX_VAL);
                y_row[col] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
            }
        }

        if (lane_id == 0) {
            scale_row[kb] = s;
        }
    }
}

// ============================================================================
// Kernel 2: silu_mul_3d — 3D-sparse SiLU×gate
// Grid: (E,)   Block: 256 threads
// ============================================================================
__global__ void silu_mul_3d_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ tokens_per_expert,
    int mtp, int N
) {
    const int expert = blockIdx.x;
    const int valid_tokens = tokens_per_expert[expert];
    if (valid_tokens == 0) return;

    const int tid = threadIdx.x;
    const int64_t expert_offset = (int64_t)expert * mtp * N;

    for (int m = 0; m < valid_tokens; m++) {
        const int64_t row_offset = expert_offset + (int64_t)m * N;

        for (int i = tid * 8; i < N; i += blockDim.x * 8) {
            if (i + 7 < N) {
                uint4 g_vec = *reinterpret_cast<const uint4*>(&gate[row_offset + i]);
                uint4 u_vec = *reinterpret_cast<const uint4*>(&up[row_offset + i]);

                __nv_bfloat16* g_arr = reinterpret_cast<__nv_bfloat16*>(&g_vec);
                __nv_bfloat16* u_arr = reinterpret_cast<__nv_bfloat16*>(&u_vec);

                uint4 out_vec;
                __nv_bfloat16* o_arr = reinterpret_cast<__nv_bfloat16*>(&out_vec);

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float g = __bfloat162float(g_arr[j]);
                    float u = __bfloat162float(u_arr[j]);
                    float silu_g = g / (1.0f + expf(-g));
                    o_arr[j] = __float2bfloat16(silu_g * u);
                }

                *reinterpret_cast<uint4*>(&output[row_offset + i]) = out_vec;
            } else {
                for (int j = 0; j < 8 && (i + j) < N; j++) {
                    float g = __bfloat162float(gate[row_offset + i + j]);
                    float u = __bfloat162float(up[row_offset + i + j]);
                    float silu_g = g / (1.0f + expf(-g));
                    output[row_offset + i + j] = __float2bfloat16(silu_g * u);
                }
            }
        }
    }
}

// ============================================================================
// Kernel 3: fused_silu_quant_3d — SiLU×gate + FP8 quantization
// Grid: (E,)   Block: 128 threads (4 warps)
// Fuses silu_mul + act_quant: eliminates intermediate BF16 buffer.
// ============================================================================
__global__ void fused_silu_quant_3d_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t* __restrict__ y,
    float* __restrict__ scale,
    const int32_t* __restrict__ tokens_per_expert,
    int mtp, int N, int num_n_blocks
) {
    const int expert = blockIdx.x;
    const int valid_tokens = tokens_per_expert[expert];
    if (valid_tokens == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const int64_t expert_offset = (int64_t)expert * mtp * N;
    float* scale_base = scale + (int64_t)expert * mtp * num_n_blocks;

    for (int m = 0; m < valid_tokens; m++) {
        const int64_t row_offset = expert_offset + (int64_t)m * N;
        float* scale_row = scale_base + (int64_t)m * num_n_blocks;

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

            if (lane_id == 0) {
                scale_row[kb] = s;
            }
        }
    }
}

// ============================================================================
// C++ Wrappers
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> act_quant_3d(
    torch::Tensor x,
    torch::Tensor tokens_per_expert
) {
    TORCH_CHECK(x.dim() == 3, "x must be 3D [E, mtp, K]");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int E = x.size(0);
    int mtp = x.size(1);
    int K = x.size(2);
    int num_k_blocks = (K + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;

    auto y = torch::empty({E, mtp, K}, torch::dtype(torch::kUInt8).device(x.device()));
    auto scale = torch::empty({E, mtp, num_k_blocks}, torch::dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 2D grid: one CTA per (expert, token). Padded tokens early-return
    // inside the kernel so no compute wasted on expert_count < mtp.
    dim3 grid(E, mtp);
    act_quant_3d_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        y.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(),
        mtp, K, num_k_blocks);

    return std::make_tuple(y, scale);
}

torch::Tensor silu_mul_3d(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor tokens_per_expert
) {
    TORCH_CHECK(gate.dim() == 3 && up.dim() == 3, "Inputs must be 3D");
    TORCH_CHECK(gate.sizes() == up.sizes(), "Shape mismatch");
    TORCH_CHECK(gate.dtype() == torch::kBFloat16, "Must be BF16");

    int E = gate.size(0);
    int mtp = gate.size(1);
    int N = gate.size(2);

    auto output = torch::empty_like(gate);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    silu_mul_3d_kernel<<<E, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        tokens_per_expert.data_ptr<int32_t>(),
        mtp, N);

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> fused_silu_quant_3d(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor tokens_per_expert
) {
    TORCH_CHECK(gate.dim() == 3 && up.dim() == 3, "Inputs must be 3D");
    TORCH_CHECK(gate.sizes() == up.sizes(), "Shape mismatch");

    int E = gate.size(0);
    int mtp = gate.size(1);
    int N = gate.size(2);
    int num_n_blocks = (N + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;

    auto y = torch::empty({E, mtp, N}, torch::dtype(torch::kUInt8).device(gate.device()));
    auto scale = torch::empty({E, mtp, num_n_blocks}, torch::dtype(torch::kFloat32).device(gate.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_silu_quant_3d_kernel<<<E, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        y.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(),
        mtp, N, num_n_blocks);

    return std::make_tuple(y, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("act_quant_3d", &act_quant_3d,
          "FP8 blockwise quantization on 3D [E, mtp, K] layout");
    m.def("silu_mul_3d", &silu_mul_3d,
          "SiLU(gate) * up on 3D [E, mtp, N] layout");
    m.def("fused_silu_quant_3d", &fused_silu_quant_3d,
          "Fused SiLU(gate) * up + FP8 quantization on 3D layout");
}
