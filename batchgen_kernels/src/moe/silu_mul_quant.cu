// Fused SiLU(gate) * up + per-token FP8 E4M3 quantization kernel.
// Ported from sglang's silu_and_mul_masked_post_quant.cuh (contig path),
// simplified for the batchgen API: separate gate/up inputs, per-token scale.
//
// Algorithm (per token row):
//   1. Each thread loads 8 bf16 elements from gate and up.
//   2. Compute SiLU(g) * u in float32 for each pair.
//   3. Block-level reduction to find per-token absmax.
//   4. scale = absmax / FP8_E4M3_MAX; quantize and store.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr float kFP8E4M3Max = 448.0f;

// Block-wide max reduction: warp-level butterfly + cross-warp via smem.
__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    // Intra-warp butterfly reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }

    if (lane == 0) smem[warp] = val;
    __syncthreads();

    // First warp reduces across warps
    if (warp == 0) {
        val = lane < num_warps ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
        }
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

// Grid:  T blocks (one per token row)
// Block: D/8 threads (each thread handles 8 contiguous elements)
__global__ __launch_bounds__(1024, 2)
void silu_mul_quant_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t D
) {
    const int64_t token_id = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* gate_row = gate + token_id * D;
    const __nv_bfloat16* up_row   = up  + token_id * D;
    uint8_t* out_row = output + token_id * D;

    // --- Pass 1: SiLU * mul, find local absmax ---
    float results[8];
    float local_max = 0.0f;
    const int base = tid * 8;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float g = __bfloat162float(gate_row[base + i]);
        float u = __bfloat162float(up_row[base + i]);
        float silu_g = g / (1.0f + expf(-g));
        float val = silu_g * u;
        results[i] = val;
        local_max = fmaxf(local_max, fabsf(val));
    }

    // --- Block reduction for per-token absmax ---
    float absmax = fmaxf(block_reduce_max(local_max), 1e-10f);
    float scale = absmax / kFP8E4M3Max;
    float inv_scale = 1.0f / scale;

    if (tid == 0) {
        scales[token_id] = scale;
    }

    // --- Quantize and store FP8 ---
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float scaled_val = results[i] * inv_scale;
        out_row[base + i] = __nv_cvt_float_to_fp8(
            scaled_val, __NV_SATFINITE, __NV_E4M3);
    }
}

}  // namespace

// Host wrapper called from Python via load_inline.
void silu_mul_quant_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor output,
    torch::Tensor scales
) {
    TORCH_CHECK(gate.is_cuda(),  "gate must be CUDA");
    TORCH_CHECK(up.is_cuda(),    "up must be CUDA");
    TORCH_CHECK(output.is_cuda(),"output must be CUDA");
    TORCH_CHECK(scales.is_cuda(),"scales must be CUDA");
    TORCH_CHECK(gate.dtype() == torch::kBFloat16, "gate must be bfloat16");
    TORCH_CHECK(up.dtype()   == torch::kBFloat16, "up must be bfloat16");
    TORCH_CHECK(gate.dim() == 2 && gate.is_contiguous(),
                "gate must be contiguous [T, D]");
    TORCH_CHECK(up.dim() == 2 && up.is_contiguous(),
                "up must be contiguous [T, D]");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate/up shape mismatch");
    TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");

    const int64_t T = gate.size(0);
    const int64_t D = gate.size(1);

    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8, got ", D);
    TORCH_CHECK(D / 8 <= 1024, "D/8 must be <= 1024, got ", D / 8);
    TORCH_CHECK(output.dim() == 2 && output.size(0) == T && output.size(1) == D,
                "output shape must be [T, D]");
    TORCH_CHECK(scales.dim() == 1 && scales.size(0) == T,
                "scales shape must be [T]");

    c10::cuda::CUDAGuard device_guard(gate.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const int threads = static_cast<int>(D / 8);
    silu_mul_quant_kernel<<<static_cast<int>(T), threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        static_cast<uint8_t*>(output.data_ptr()),
        scales.data_ptr<float>(),
        D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
