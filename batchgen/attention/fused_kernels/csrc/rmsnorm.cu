#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
// RMSNorm kernel
// ============================================================================

template <typename T>
__global__ void rmsnorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int hidden_size,
    float eps)
{
    int row = blockIdx.x;
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
    float eps)
{
    int row = blockIdx.x;
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
    float eps)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");

    input = input.contiguous();
    weight = weight.contiguous();

    auto sizes = input.sizes();
    int hidden_size = sizes[sizes.size() - 1];
    int num_rows = input.numel() / hidden_size;

    auto output = torch::empty_like(input);

    // 256 threads: each handles hidden_size/256 = 16 elements for H=4096
    const int threads = 256;
    const int blocks = num_rows;

    AT_DISPATCH_SWITCH(input.scalar_type(),
        "rmsnorm_forward",
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] { rmsnorm_kernel<at::BFloat16><<<blocks, threads>>>(
                input.data_ptr<at::BFloat16>(),
                weight.data_ptr<at::BFloat16>(),
                output.data_ptr<at::BFloat16>(),
                hidden_size, eps); })
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] { rmsnorm_kernel<at::Half><<<blocks, threads>>>(
                input.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                output.data_ptr<at::Half>(),
                hidden_size, eps); })
        AT_DISPATCH_CASE(at::ScalarType::Float,
            [&] { rmsnorm_kernel<float><<<blocks, threads>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                hidden_size, eps); })
    );

    return output;
}

std::vector<torch::Tensor> add_rmsnorm_forward(
    torch::Tensor residual,
    torch::Tensor hidden,
    torch::Tensor weight,
    float eps)
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

    const int threads = 256;
    const int blocks = num_rows;

    AT_DISPATCH_SWITCH(residual.scalar_type(),
        "add_rmsnorm_forward",
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] { add_rmsnorm_kernel<at::BFloat16><<<blocks, threads>>>(
                residual.data_ptr<at::BFloat16>(),
                hidden.data_ptr<at::BFloat16>(),
                weight.data_ptr<at::BFloat16>(),
                normed_out.data_ptr<at::BFloat16>(),
                hidden_size, eps); })
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] { add_rmsnorm_kernel<at::Half><<<blocks, threads>>>(
                residual.data_ptr<at::Half>(),
                hidden.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                normed_out.data_ptr<at::Half>(),
                hidden_size, eps); })
        AT_DISPATCH_CASE(at::ScalarType::Float,
            [&] { add_rmsnorm_kernel<float><<<blocks, threads>>>(
                residual.data_ptr<float>(),
                hidden.data_ptr<float>(),
                weight.data_ptr<float>(),
                normed_out.data_ptr<float>(),
                hidden_size, eps); })
    );

    return {normed_out, residual};
}
