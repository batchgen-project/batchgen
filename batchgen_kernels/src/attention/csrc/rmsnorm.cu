#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include "attention_ops.h"

// ============================================================================
// Warp-level reduction
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using warp shuffles (no shared memory for reduction)
// Requires blockDim.x <= 256 (8 warps)
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[8]; // max 8 warps for 256 threads

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    // Warp-level reduce
    val = warp_reduce_sum(val);

    // Write warp results
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val; // Only thread 0 has the final result
}

// ============================================================================
// BF16 hidden-size-6144 vector specialization
// ============================================================================

constexpr int kVectorHiddenSize = 6144;
constexpr int kVectorWidth = 8;
constexpr int kVectorThreads = kVectorHiddenSize / kVectorWidth;
constexpr int kVectorWarps = kVectorThreads / 32;

struct alignas(16) BFloat16x8 {
    __nv_bfloat16 values[kVectorWidth];
};

static_assert(sizeof(BFloat16x8) == 16, "BFloat16x8 must be 16 bytes");
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "at::BFloat16 and __nv_bfloat16 must have the same size");

__device__ __forceinline__ float block_reduce_sum_24(
    float val,
    float* warp_sums)
{
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < kVectorWarps) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void rmsnorm_bf16_6144_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float eps,
    const int* __restrict__ num_valid_ptr)
{
    int row = blockIdx.x;
    if (num_valid_ptr != nullptr && row >= *num_valid_ptr) return;

    __shared__ float warp_sums[kVectorWarps];
    __shared__ float s_inv_rms;

    const auto* input_vectors = reinterpret_cast<const BFloat16x8*>(input);
    const auto* weight_vectors = reinterpret_cast<const BFloat16x8*>(weight);
    auto* output_vectors = reinterpret_cast<BFloat16x8*>(output);
    int vector_index = row * kVectorThreads + threadIdx.x;

    BFloat16x8 input_vector = input_vectors[vector_index];
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < kVectorWidth; ++i) {
        float val = __bfloat162float(input_vector.values[i]);
        sum_sq = fmaf(val, val, sum_sq);
    }

    float total = block_reduce_sum_24(sum_sq, warp_sums);
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(total / kVectorHiddenSize + eps);
    }
    __syncthreads();

    BFloat16x8 weight_vector = weight_vectors[threadIdx.x];
    BFloat16x8 output_vector;
    #pragma unroll
    for (int i = 0; i < kVectorWidth; ++i) {
        float val = __bfloat162float(input_vector.values[i]);
        float scale = __bfloat162float(weight_vector.values[i]);
        output_vector.values[i] = __float2bfloat16_rn(val * s_inv_rms * scale);
    }
    output_vectors[vector_index] = output_vector;
}

__global__ void add_rmsnorm_bf16_6144_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ normed_out,
    float eps,
    const int* __restrict__ num_valid_ptr)
{
    int row = blockIdx.x;
    if (num_valid_ptr != nullptr && row >= *num_valid_ptr) return;

    __shared__ float warp_sums[kVectorWarps];
    __shared__ float s_inv_rms;

    auto* residual_vectors = reinterpret_cast<BFloat16x8*>(residual);
    const auto* hidden_vectors = reinterpret_cast<const BFloat16x8*>(hidden);
    const auto* weight_vectors = reinterpret_cast<const BFloat16x8*>(weight);
    auto* output_vectors = reinterpret_cast<BFloat16x8*>(normed_out);
    int vector_index = row * kVectorThreads + threadIdx.x;

    BFloat16x8 residual_vector = residual_vectors[vector_index];
    BFloat16x8 hidden_vector = hidden_vectors[vector_index];
    // Matches the scalar fallback: the reduction uses the unrounded float sum
    // while the normalized numerator re-reads the BF16-rounded residual.
    BFloat16x8 rounded_sum;
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < kVectorWidth; ++i) {
        float sum = __bfloat162float(residual_vector.values[i])
                  + __bfloat162float(hidden_vector.values[i]);
        rounded_sum.values[i] = __float2bfloat16_rn(sum);
        sum_sq = fmaf(sum, sum, sum_sq);
    }
    residual_vectors[vector_index] = rounded_sum;

    float total = block_reduce_sum_24(sum_sq, warp_sums);
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(total / kVectorHiddenSize + eps);
    }
    __syncthreads();

    BFloat16x8 weight_vector = weight_vectors[threadIdx.x];
    BFloat16x8 output_vector;
    #pragma unroll
    for (int i = 0; i < kVectorWidth; ++i) {
        float val = __bfloat162float(rounded_sum.values[i]);
        float scale = __bfloat162float(weight_vector.values[i]);
        output_vector.values[i] = __float2bfloat16_rn(val * s_inv_rms * scale);
    }
    output_vectors[vector_index] = output_vector;
}

inline bool is_aligned_16(const void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0;
}

// ============================================================================
// RMSNorm kernel
// ============================================================================

template <typename T>
__global__ void rmsnorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int hidden_size,
    float eps,
    const int* __restrict__ num_valid_ptr)
{
    int row = blockIdx.x;
    // Skip padding rows when num_valid_ptr is provided (CUDA graph path)
    if (num_valid_ptr != nullptr && row >= *num_valid_ptr) return;

    const T* x_row = input + row * hidden_size;
    T* o_row = output + row * hidden_size;

    // Phase 1: Compute sum of squares with vectorized loads
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(x_row[i]);
        sum_sq += val * val;
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq);

    // Broadcast inv_rms
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Phase 2: Normalize and write output
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(x_row[i]);
        float w = static_cast<float>(weight[i]);
        o_row[i] = static_cast<T>(val * inv_rms * w);
    }
}

// ============================================================================
// Fused Add + RMSNorm kernel
// ============================================================================

template <typename T>
__global__ void add_rmsnorm_kernel(
    T* __restrict__ residual,       // in-place updated
    const T* __restrict__ hidden,
    const T* __restrict__ weight,
    T* __restrict__ normed_out,
    int hidden_size,
    float eps,
    const int* __restrict__ num_valid_ptr)
{
    int row = blockIdx.x;
    // Skip padding rows when num_valid_ptr is provided (CUDA graph path)
    if (num_valid_ptr != nullptr && row >= *num_valid_ptr) return;

    T* r_row = residual + row * hidden_size;
    const T* h_row = hidden + row * hidden_size;
    T* o_row = normed_out + row * hidden_size;

    // Phase 1: Add and compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = static_cast<float>(r_row[i]);
        float h = static_cast<float>(h_row[i]);
        float s = r + h;
        // Store updated residual
        r_row[i] = static_cast<T>(s);
        sum_sq += s * s;
    }

    // Reduce
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Phase 2: Normalize (re-read residual which now has the sum)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(r_row[i]);
        float w = static_cast<float>(weight[i]);
        o_row[i] = static_cast<T>(val * inv_rms * w);
    }
}

// ============================================================================
// Host functions
// ============================================================================

torch::Tensor rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float eps,
    c10::optional<torch::Tensor> num_valid_tokens)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");

    input = input.contiguous();
    weight = weight.contiguous();

    auto sizes = input.sizes();
    int hidden_size = sizes[sizes.size() - 1];
    int num_rows = input.numel() / hidden_size;

    auto output = torch::empty_like(input);

    if (num_rows == 0) return output;

    const int* num_valid_ptr = nullptr;
    if (num_valid_tokens.has_value() && num_valid_tokens->defined()) {
        num_valid_ptr = num_valid_tokens->data_ptr<int>();
    }

    const int threads = 256;
    const int blocks = num_rows;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_SWITCH(input.scalar_type(),
        "rmsnorm_forward",
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] {
                bool vector_aligned = is_aligned_16(input.data_ptr<at::BFloat16>())
                                   && is_aligned_16(weight.data_ptr<at::BFloat16>())
                                   && is_aligned_16(output.data_ptr<at::BFloat16>());
                if (hidden_size == kVectorHiddenSize && vector_aligned) {
                    rmsnorm_bf16_6144_kernel<<<blocks, kVectorThreads, 0, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(
                            input.data_ptr<at::BFloat16>()),
                        reinterpret_cast<const __nv_bfloat16*>(
                            weight.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__nv_bfloat16*>(
                            output.data_ptr<at::BFloat16>()),
                        eps, num_valid_ptr);
                } else {
                    if (hidden_size == kVectorHiddenSize && !vector_aligned) {
                        TORCH_WARN_ONCE(
                            "BF16 RMSNorm hidden-size-6144 vector alignment check failed; "
                            "falling back to the scalar kernel");
                    }
                    rmsnorm_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
                        input.data_ptr<at::BFloat16>(),
                        weight.data_ptr<at::BFloat16>(),
                        output.data_ptr<at::BFloat16>(),
                        hidden_size, eps, num_valid_ptr);
                }
            })
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] { rmsnorm_kernel<at::Half><<<blocks, threads, 0, stream>>>(
                input.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                hidden_size, eps, num_valid_ptr); })
        AT_DISPATCH_CASE(at::ScalarType::Float,
            [&] { rmsnorm_kernel<float><<<blocks, threads, 0, stream>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                hidden_size, eps, num_valid_ptr); })
    );

    return output;
}

std::vector<torch::Tensor> add_rmsnorm_forward(
    torch::Tensor residual,
    torch::Tensor hidden,
    torch::Tensor weight,
    float eps,
    c10::optional<torch::Tensor> num_valid_tokens)
{
    TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");

    residual = residual.contiguous();
    hidden = hidden.contiguous();
    weight = weight.contiguous();

    auto sizes = residual.sizes();
    int hidden_size = sizes[sizes.size() - 1];
    int num_rows = residual.numel() / hidden_size;

    auto normed_out = torch::empty_like(residual);

    if (num_rows == 0) return {normed_out, residual};

    const int* num_valid_ptr = nullptr;
    if (num_valid_tokens.has_value() && num_valid_tokens->defined()) {
        num_valid_ptr = num_valid_tokens->data_ptr<int>();
    }

    const int threads = 256;
    const int blocks = num_rows;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_SWITCH(residual.scalar_type(),
        "add_rmsnorm_forward",
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] {
                bool vector_aligned = is_aligned_16(residual.data_ptr<at::BFloat16>())
                                   && is_aligned_16(hidden.data_ptr<at::BFloat16>())
                                   && is_aligned_16(weight.data_ptr<at::BFloat16>())
                                   && is_aligned_16(normed_out.data_ptr<at::BFloat16>());
                if (hidden_size == kVectorHiddenSize && vector_aligned) {
                    add_rmsnorm_bf16_6144_kernel<<<
                        blocks, kVectorThreads, 0, stream>>>(
                        reinterpret_cast<__nv_bfloat16*>(
                            residual.data_ptr<at::BFloat16>()),
                        reinterpret_cast<const __nv_bfloat16*>(
                            hidden.data_ptr<at::BFloat16>()),
                        reinterpret_cast<const __nv_bfloat16*>(
                            weight.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__nv_bfloat16*>(
                            normed_out.data_ptr<at::BFloat16>()),
                        eps, num_valid_ptr);
                } else {
                    if (hidden_size == kVectorHiddenSize && !vector_aligned) {
                        TORCH_WARN_ONCE(
                            "BF16 Add+RMSNorm hidden-size-6144 vector alignment check failed; "
                            "falling back to the scalar kernel");
                    }
                    add_rmsnorm_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
                        residual.data_ptr<at::BFloat16>(),
                        hidden.data_ptr<at::BFloat16>(),
                        weight.data_ptr<at::BFloat16>(),
                        normed_out.data_ptr<at::BFloat16>(),
                        hidden_size, eps, num_valid_ptr);
                }
            })
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] { add_rmsnorm_kernel<at::Half><<<blocks, threads, 0, stream>>>(
                residual.data_ptr<at::Half>(),
                hidden.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                normed_out.data_ptr<at::Half>(),
                hidden_size, eps, num_valid_ptr); })
        AT_DISPATCH_CASE(at::ScalarType::Float,
            [&] { add_rmsnorm_kernel<float><<<blocks, threads, 0, stream>>>(
                residual.data_ptr<float>(),
                hidden.data_ptr<float>(),
                weight.data_ptr<float>(),
                normed_out.data_ptr<float>(),
                hidden_size, eps, num_valid_ptr); })
    );

    return {normed_out, residual};
}
