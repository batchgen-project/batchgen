#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <mma.h>

// Vectorized data types for efficient memory access
struct float4_accessor {
    union {
        float4 vec;
        float arr[4];
    };
    __device__ float4_accessor() {}
    __device__ float4_accessor(float4 v) : vec(v) {}
};

struct half8_accessor {
    union {
        float4 vec;
        __half arr[8];
    };
    __device__ half8_accessor() {}
    __device__ half8_accessor(float4 v) : vec(v) {}
};

// Optimized warp reduction with better instruction utilization
template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fast block reduction using CUB
template<typename T, int BLOCK_SIZE>
__device__ __forceinline__ T blockReduceSum(T val) {
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T result = BlockReduce(temp_storage).Sum(val);
    return result;
}

// Optimized RMSNorm kernel for hidden_size <= 1024 (single warp per element)
template <typename T, int HIDDEN_SIZE>
__global__ void __launch_bounds__(32, 32) 
rmsnorm_warp_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int batch_size,
    float eps) {
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const T* input_row = input + batch_idx * HIDDEN_SIZE;
    T* output_row = output + batch_idx * HIDDEN_SIZE;
    
    // Phase 1: Compute sum of squares with vectorized access
    float sum_sq = 0.0f;
    
    if constexpr (HIDDEN_SIZE <= 32) {
        // Single iteration per thread
        if (threadIdx.x < HIDDEN_SIZE) {
            float val = static_cast<float>(input_row[threadIdx.x]);
            sum_sq = val * val;
        }
    } else {
        // Multiple iterations with vectorized access when possible
        constexpr int VEC_SIZE = (std::is_same_v<T, float>) ? 4 : 8;
        constexpr int VECTORIZED_SIZE = (HIDDEN_SIZE / VEC_SIZE) * VEC_SIZE;
        
        // Vectorized portion
        for (int i = threadIdx.x * VEC_SIZE; i < VECTORIZED_SIZE; i += 32 * VEC_SIZE) {
            if constexpr (std::is_same_v<T, float>) {
                float4 vals = reinterpret_cast<const float4*>(input_row)[i / 4];
                sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
            } else {
                // For half precision, load as float4 but interpret as 8 halves
                float4 vals = reinterpret_cast<const float4*>(input_row)[i / 8];
                half8_accessor acc(vals);
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float val = static_cast<float>(acc.arr[j]);
                    sum_sq += val * val;
                }
            }
        }
        
        // Handle remaining elements
        for (int i = VECTORIZED_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += 32) {
            float val = static_cast<float>(input_row[i]);
            sum_sq += val * val;
        }
    }
    
    // Warp reduction
    sum_sq = warpReduceSum(sum_sq);
    
    // Broadcast to all threads in warp
    float inv_rms = rsqrtf(__shfl_sync(0xffffffff, sum_sq, 0) / HIDDEN_SIZE + eps);
    
    // Phase 2: Apply normalization with vectorized writes
    if constexpr (HIDDEN_SIZE <= 32) {
        if (threadIdx.x < HIDDEN_SIZE) {
            float val = static_cast<float>(input_row[threadIdx.x]);
            float w = static_cast<float>(weight[threadIdx.x]);
            output_row[threadIdx.x] = static_cast<T>(val * inv_rms * w);
        }
    } else {
        constexpr int VEC_SIZE = (std::is_same_v<T, float>) ? 4 : 8;
        constexpr int VECTORIZED_SIZE = (HIDDEN_SIZE / VEC_SIZE) * VEC_SIZE;
        
        // Vectorized portion
        for (int i = threadIdx.x * VEC_SIZE; i < VECTORIZED_SIZE; i += 32 * VEC_SIZE) {
            if constexpr (std::is_same_v<T, float>) {
                float4 vals = reinterpret_cast<const float4*>(input_row)[i / 4];
                float4 weights = reinterpret_cast<const float4*>(weight)[i / 4];
                
                float4 output_vals;
                output_vals.x = vals.x * inv_rms * weights.x;
                output_vals.y = vals.y * inv_rms * weights.y;
                output_vals.z = vals.z * inv_rms * weights.z;
                output_vals.w = vals.w * inv_rms * weights.w;
                
                reinterpret_cast<float4*>(output_row)[i / 4] = output_vals;
            } else {
                float4 vals = reinterpret_cast<const float4*>(input_row)[i / 8];
                float4 weights = reinterpret_cast<const float4*>(weight)[i / 8];
                
                half8_accessor val_acc(vals);
                half8_accessor weight_acc(weights);
                half8_accessor output_acc;
                
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float val = static_cast<float>(val_acc.arr[j]);
                    float w = static_cast<float>(weight_acc.arr[j]);
                    output_acc.arr[j] = static_cast<__half>(val * inv_rms * w);
                }
                
                reinterpret_cast<float4*>(output_row)[i / 8] = output_acc.vec;
            }
        }
        
        // Handle remaining elements
        for (int i = VECTORIZED_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += 32) {
            float val = static_cast<float>(input_row[i]);
            float w = static_cast<float>(weight[i]);
            output_row[i] = static_cast<T>(val * inv_rms * w);
        }
    }
}

// High-performance kernel for large hidden sizes (> 1024)
template <typename T, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
rmsnorm_block_kernel(
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
    
    // Phase 1: Compute sum of squares with better memory coalescing
    float sum_sq = 0.0f;
    
    // Vectorized memory access
    constexpr int VEC_SIZE = (std::is_same_v<T, float>) ? 4 : 8;
    const int vectorized_size = (hidden_size / VEC_SIZE) * VEC_SIZE;
    
    // Process vectorized portion
    for (int i = threadIdx.x * VEC_SIZE; i < vectorized_size; i += BLOCK_SIZE * VEC_SIZE) {
        if constexpr (std::is_same_v<T, float>) {
            float4 vals = reinterpret_cast<const float4*>(input_row)[i / 4];
            sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
        } else {
            float4 vals = reinterpret_cast<const float4*>(input_row)[i / 8];
            half8_accessor acc(vals);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float val = static_cast<float>(acc.arr[j]);
                sum_sq += val * val;
            }
        }
    }
    
    // Process remaining elements
    for (int i = vectorized_size + threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = static_cast<float>(input_row[i]);
        sum_sq += val * val;
    }
    
    // Block reduction using CUB
    sum_sq = blockReduceSum<float, BLOCK_SIZE>(sum_sq);
    
    // Broadcast result
    __shared__ float shared_inv_rms;
    if (threadIdx.x == 0) {
        shared_inv_rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    
    float inv_rms = shared_inv_rms;
    
    // Phase 2: Apply normalization with vectorized writes
    for (int i = threadIdx.x * VEC_SIZE; i < vectorized_size; i += BLOCK_SIZE * VEC_SIZE) {
        if constexpr (std::is_same_v<T, float>) {
            float4 vals = reinterpret_cast<const float4*>(input_row)[i / 4];
            float4 weights = reinterpret_cast<const float4*>(weight)[i / 4];
            
            float4 output_vals;
            output_vals.x = vals.x * inv_rms * weights.x;
            output_vals.y = vals.y * inv_rms * weights.y;
            output_vals.z = vals.z * inv_rms * weights.z;
            output_vals.w = vals.w * inv_rms * weights.w;
            
            reinterpret_cast<float4*>(output_row)[i / 4] = output_vals;
        } else {
            float4 vals = reinterpret_cast<const float4*>(input_row)[i / 8];
            float4 weights = reinterpret_cast<const float4*>(weight)[i / 8];
            
            half8_accessor val_acc(vals);
            half8_accessor weight_acc(weights);
            half8_accessor output_acc;
            
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float val = static_cast<float>(val_acc.arr[j]);
                float w = static_cast<float>(weight_acc.arr[j]);
                output_acc.arr[j] = static_cast<__half>(val * inv_rms * w);
            }
            
            reinterpret_cast<float4*>(output_row)[i / 8] = output_acc.vec;
        }
    }
    
    // Handle remaining elements
    for (int i = vectorized_size + threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = static_cast<float>(input_row[i]);
        float w = static_cast<float>(weight[i]);
        output_row[i] = static_cast<T>(val * inv_rms * w);
    }
}

// Template dispatch function with compile-time optimization
template <typename T>
void launch_optimized_rmsnorm(
    const T* input,
    const T* weight,
    T* output,
    int batch_size,
    int hidden_size,
    float eps) {
    
    // Compile-time dispatch for common hidden sizes
    switch (hidden_size) {
        case 64:
            rmsnorm_warp_kernel<T, 64><<<batch_size, 32>>>(
                input, weight, output, batch_size, eps);
            break;
        case 128:
            rmsnorm_warp_kernel<T, 128><<<batch_size, 32>>>(
                input, weight, output, batch_size, eps);
            break;
        case 256:
            rmsnorm_warp_kernel<T, 256><<<batch_size, 32>>>(
                input, weight, output, batch_size, eps);
            break;
        case 512:
            rmsnorm_warp_kernel<T, 512><<<batch_size, 32>>>(
                input, weight, output, batch_size, eps);
            break;
        case 768:
            rmsnorm_warp_kernel<T, 768><<<batch_size, 32>>>(
                input, weight, output, batch_size, eps);
            break;
        case 1024:
            rmsnorm_warp_kernel<T, 1024><<<batch_size, 32>>>(
                input, weight, output, batch_size, eps);
            break;
        default:
            if (hidden_size <= 1024) {
                // Dynamic dispatch for other small sizes
                if (hidden_size <= 256) {
                    rmsnorm_warp_kernel<T, 256><<<batch_size, 32>>>(
                        input, weight, output, batch_size, eps);
                } else if (hidden_size <= 512) {
                    rmsnorm_warp_kernel<T, 512><<<batch_size, 32>>>(
                        input, weight, output, batch_size, eps);
                } else {
                    rmsnorm_warp_kernel<T, 1024><<<batch_size, 32>>>(
                        input, weight, output, batch_size, eps);
                }
            } else {
                // Large hidden sizes
                if (hidden_size <= 4096) {
                    rmsnorm_block_kernel<T, 256><<<batch_size, 256>>>(
                        input, weight, output, batch_size, hidden_size, eps);
                } else if (hidden_size <= 8192) {
                    rmsnorm_block_kernel<T, 512><<<batch_size, 512>>>(
                        input, weight, output, batch_size, hidden_size, eps);
                } else {
                    rmsnorm_block_kernel<T, 1024><<<batch_size, 1024>>>(
                        input, weight, output, batch_size, hidden_size, eps);
                }
            }
    }
}

// Main host function with improved error handling
torch::Tensor fused_rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float eps = 1e-6) {
    
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");
    
    auto input_shape = input.sizes();
    int hidden_size = input_shape[input_shape.size() - 1];
    int batch_size = input.numel() / hidden_size;
    
    TORCH_CHECK(weight.numel() == hidden_size, 
                "Weight size must match input's last dimension");
    TORCH_CHECK(eps > 0, "Epsilon must be positive");
    
    // Create output tensor with same properties as input
    auto output = torch::empty_like(input);
    
    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "fused_rmsnorm", [&] {
        launch_optimized_rmsnorm<scalar_t>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            eps
        );
    });
    
    // Check for CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}

// Backward pass kernel (optional, for training)
template <typename T>
__global__ void rmsnorm_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ grad_input,
    T* __restrict__ grad_weight,
    int batch_size,
    int hidden_size,
    float eps) {
    
    // Implementation for backward pass
    // This would include gradient computation for both input and weight
    // Left as an exercise or can be implemented if needed
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_rmsnorm_forward, "Fused RMSNorm forward pass");
    m.def("rmsnorm_forward", &fused_rmsnorm_forward, "Fused RMSNorm forward pass (alias)");
}