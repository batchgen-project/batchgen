#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

// Simple warp reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Simple RMSNorm kernel - optimized for fast compilation
template <typename T>
__global__ void simple_rmsnorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int batch_size,
    int hidden_size,
    float eps) {
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const T* input_row = input + batch_idx * hidden_size;
    T* output_row = output + batch_idx * hidden_size;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(input_row[i]);
        sum_sq += val * val;
    }
    
    // Reduce across threads in block
    __shared__ float shared_sum[1024];
    shared_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    
    // Simple reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Compute normalization factor
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(shared_sum[0] / hidden_size + eps);
    }
    __syncthreads();
    
    // Apply normalization
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(input_row[i]);
        float w = static_cast<float>(weight[i]);
        output_row[i] = static_cast<T>(val * inv_rms * w);
    }
}

// Host function
torch::Tensor simple_rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    
    // Make tensors contiguous if they aren't already
    input = input.contiguous();
    weight = weight.contiguous();
    
    auto input_shape = input.sizes();
    int hidden_size = input_shape[input_shape.size() - 1];
    int batch_size = input.numel() / hidden_size;
    
    auto output = torch::empty_like(input);
    
    // Simple kernel launch - use 256 threads per block
    const int threads = min(256, hidden_size);
    const int blocks = batch_size;
    
    // Dispatch based on type - now includes BF16 support
    if (input.scalar_type() == torch::kFloat32) {
        simple_rmsnorm_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, hidden_size, eps
        );
    } else if (input.scalar_type() == torch::kFloat16) {
        simple_rmsnorm_kernel<at::Half><<<blocks, threads>>>(
            input.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            batch_size, hidden_size, eps
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        simple_rmsnorm_kernel<at::BFloat16><<<blocks, threads>>>(
            input.data_ptr<at::BFloat16>(),
            weight.data_ptr<at::BFloat16>(),
            output.data_ptr<at::BFloat16>(),
            batch_size, hidden_size, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only float32, float16, and bfloat16 are supported.");
    }
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error occurred");
    }
    
    return output;
}