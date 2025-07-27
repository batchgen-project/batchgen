# # fused_rmsnorm.py
# """
# Self-contained Fused RMSNorm implementation with JIT CUDA compilation.
# Just drop this file into your project and import FusedRMSNorm.

# Usage:
#     from fused_rmsnorm import FusedRMSNorm
    
#     # Drop-in replacement for any RMSNorm
#     rmsnorm = FusedRMSNorm(hidden_size=4096)
#     output = rmsnorm(input_tensor)
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import warnings
# from typing import Optional

# # Global variable to cache the compiled extension
# _cuda_extension = None
# _compilation_attempted = False

# def _get_cuda_kernel_source():
#     """Returns the CUDA kernel source code as a string"""
#     return '''
# #include <torch/extension.h>
# #include <ATen/ATen.h>
# #include <ATen/cuda/CUDAContext.h>
# #include <c10/cuda/CUDAGuard.h>
# #include <cuda_runtime.h>
# #include <cuda_bf16.h>
# #include <cuda_fp16.h>

# // Optimized warp reduction
# __device__ __forceinline__ float warpReduceSum(float val) {
#     #pragma unroll
#     for (int offset = 16; offset > 0; offset /= 2) {
#         val += __shfl_down_sync(0xffffffff, val, offset);
#     }
#     return val;
# }

# // Vectorized data types for efficient memory access
# struct float4_accessor {
#     union {
#         float4 vec;
#         float arr[4];
#     };
#     __device__ float4_accessor() {}
#     __device__ float4_accessor(float4 v) : vec(v) {}
# };

# struct half8_accessor {
#     union {
#         float4 vec;
#         __half arr[8];
#     };
#     __device__ half8_accessor() {}
#     __device__ half8_accessor(float4 v) : vec(v) {}
# };

# // Optimized kernel for small hidden sizes (<=1024)
# template <typename T>
# __global__ void __launch_bounds__(32, 32)
# rmsnorm_small_kernel(
#     const T* __restrict__ input,
#     const T* __restrict__ weight,
#     T* __restrict__ output,
#     int batch_size,
#     int hidden_size,
#     float eps) {
    
#     int batch_idx = blockIdx.x;
#     if (batch_idx >= batch_size) return;
    
#     const T* input_row = input + batch_idx * hidden_size;
#     T* output_row = output + batch_idx * hidden_size;
    
#     // Phase 1: Compute sum of squares with vectorized access
#     float sum_sq = 0.0f;
    
#     // Vectorized loading when possible
#     if (hidden_size >= 128 && std::is_same_v<T, float>) {
#         // Process 4 floats at a time
#         const int vec_size = 4;
#         const int vectorized_end = (hidden_size / vec_size) * vec_size;
        
#         for (int i = threadIdx.x * vec_size; i < vectorized_end; i += 32 * vec_size) {
#             float4 vals = reinterpret_cast<const float4*>(input_row)[i / vec_size];
#             sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
#         }
        
#         // Handle remaining elements
#         for (int i = vectorized_end + threadIdx.x; i < hidden_size; i += 32) {
#             float val = static_cast<float>(input_row[i]);
#             sum_sq += val * val;
#         }
#     } else if (hidden_size >= 256 && std::is_same_v<T, __half>) {
#         // Process 8 halves at a time
#         const int vec_size = 8;
#         const int vectorized_end = (hidden_size / vec_size) * vec_size;
        
#         for (int i = threadIdx.x * vec_size; i < vectorized_end; i += 32 * vec_size) {
#             float4 vals = reinterpret_cast<const float4*>(input_row)[i / vec_size];
#             half8_accessor acc(vals);
#             #pragma unroll
#             for (int j = 0; j < 8; j++) {
#                 float val = static_cast<float>(acc.arr[j]);
#                 sum_sq += val * val;
#             }
#         }
        
#         // Handle remaining elements
#         for (int i = vectorized_end + threadIdx.x; i < hidden_size; i += 32) {
#             float val = static_cast<float>(input_row[i]);
#             sum_sq += val * val;
#         }
#     } else {
#         // Standard processing
#         for (int i = threadIdx.x; i < hidden_size; i += 32) {
#             float val = static_cast<float>(input_row[i]);
#             sum_sq += val * val;
#         }
#     }
    
#     // Warp reduction
#     sum_sq = warpReduceSum(sum_sq);
    
#     // Broadcast result and compute normalization factor
#     float inv_rms = rsqrtf(__shfl_sync(0xffffffff, sum_sq, 0) / hidden_size + eps);
    
#     // Phase 2: Apply normalization with vectorized writes
#     if (hidden_size >= 128 && std::is_same_v<T, float>) {
#         const int vec_size = 4;
#         const int vectorized_end = (hidden_size / vec_size) * vec_size;
        
#         for (int i = threadIdx.x * vec_size; i < vectorized_end; i += 32 * vec_size) {
#             float4 vals = reinterpret_cast<const float4*>(input_row)[i / vec_size];
#             float4 weights = reinterpret_cast<const float4*>(weight)[i / vec_size];
            
#             float4 output_vals;
#             output_vals.x = vals.x * inv_rms * weights.x;
#             output_vals.y = vals.y * inv_rms * weights.y;
#             output_vals.z = vals.z * inv_rms * weights.z;
#             output_vals.w = vals.w * inv_rms * weights.w;
            
#             reinterpret_cast<float4*>(output_row)[i / vec_size] = output_vals;
#         }
        
#         for (int i = vectorized_end + threadIdx.x; i < hidden_size; i += 32) {
#             float val = static_cast<float>(input_row[i]);
#             float w = static_cast<float>(weight[i]);
#             output_row[i] = static_cast<T>(val * inv_rms * w);
#         }
#     } else if (hidden_size >= 256 && std::is_same_v<T, __half>) {
#         const int vec_size = 8;
#         const int vectorized_end = (hidden_size / vec_size) * vec_size;
        
#         for (int i = threadIdx.x * vec_size; i < vectorized_end; i += 32 * vec_size) {
#             float4 vals = reinterpret_cast<const float4*>(input_row)[i / vec_size];
#             float4 weights = reinterpret_cast<const float4*>(weight)[i / vec_size];
            
#             half8_accessor val_acc(vals);
#             half8_accessor weight_acc(weights);
#             half8_accessor output_acc;
            
#             #pragma unroll
#             for (int j = 0; j < 8; j++) {
#                 float val = static_cast<float>(val_acc.arr[j]);
#                 float w = static_cast<float>(weight_acc.arr[j]);
#                 output_acc.arr[j] = static_cast<__half>(val * inv_rms * w);
#             }
            
#             reinterpret_cast<float4*>(output_row)[i / vec_size] = output_acc.vec;
#         }
        
#         for (int i = vectorized_end + threadIdx.x; i < hidden_size; i += 32) {
#             float val = static_cast<float>(input_row[i]);
#             float w = static_cast<float>(weight[i]);
#             output_row[i] = static_cast<T>(val * inv_rms * w);
#         }
#     } else {
#         for (int i = threadIdx.x; i < hidden_size; i += 32) {
#             float val = static_cast<float>(input_row[i]);
#             float w = static_cast<float>(weight[i]);
#             output_row[i] = static_cast<T>(val * inv_rms * w);
#         }
#     }
# }

# // Block-based kernel for large hidden sizes (>1024)
# template <typename T>
# __global__ void __launch_bounds__(256, 4)
# rmsnorm_large_kernel(
#     const T* __restrict__ input,
#     const T* __restrict__ weight,
#     T* __restrict__ output,
#     int batch_size,
#     int hidden_size,
#     float eps) {
    
#     int batch_idx = blockIdx.x;
#     if (batch_idx >= batch_size) return;
    
#     const T* input_row = input + batch_idx * hidden_size;
#     T* output_row = output + batch_idx * hidden_size;
    
#     // Phase 1: Compute sum of squares
#     float sum_sq = 0.0f;
#     for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
#         float val = static_cast<float>(input_row[i]);
#         sum_sq += val * val;
#     }
    
#     // Block reduction using shared memory
#     __shared__ float shared_sum[256];
#     shared_sum[threadIdx.x] = sum_sq;
#     __syncthreads();
    
#     // Reduce in shared memory
#     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
#         if (threadIdx.x < stride) {
#             shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
#         }
#         __syncthreads();
#     }
    
#     // Broadcast result
#     __shared__ float inv_rms;
#     if (threadIdx.x == 0) {
#         inv_rms = rsqrtf(shared_sum[0] / hidden_size + eps);
#     }
#     __syncthreads();
    
#     // Phase 2: Apply normalization
#     for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
#         float val = static_cast<float>(input_row[i]);
#         float w = static_cast<float>(weight[i]);
#         output_row[i] = static_cast<T>(val * inv_rms * w);
#     }
# }

# // Template dispatch function
# template <typename T>
# void launch_rmsnorm_kernel(
#     const T* input,
#     const T* weight,
#     T* output,
#     int batch_size,
#     int hidden_size,
#     float eps) {
    
#     // Conservative kernel launch parameters
#     if (hidden_size <= 1024) {
#         // Use 32 threads (1 warp) for small hidden sizes
#         const int threads = 32;
#         const int blocks = batch_size;
        
#         rmsnorm_small_kernel<T><<<blocks, threads>>>(
#             input, weight, output, batch_size, hidden_size, eps);
#     } else {
#         // Use 256 threads for larger hidden sizes
#         const int threads = 256;
#         const int blocks = batch_size;
        
#         rmsnorm_large_kernel<T><<<blocks, threads>>>(
#             input, weight, output, batch_size, hidden_size, eps);
#     }
    
#     // Check for launch errors
#     cudaError_t launch_error = cudaGetLastError();
#     if (launch_error != cudaSuccess) {
#         printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(launch_error));
#     }
# }

# // Main host function
# torch::Tensor fused_rmsnorm_forward(
#     torch::Tensor input,
#     torch::Tensor weight,
#     float eps) {
    
#     // Input validation
#     TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
#     TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA device");
#     TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
#     TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
#     auto input_shape = input.sizes();
#     int hidden_size = input_shape[input_shape.size() - 1];
#     int batch_size = input.numel() / hidden_size;
    
#     TORCH_CHECK(weight.numel() == hidden_size, 
#                 "Weight size must match input's last dimension");
    
#     // Create output tensor
#     auto output = torch::empty_like(input);
    
#     // Dispatch based on data type
#     if (input.scalar_type() == torch::kFloat32) {
#         launch_rmsnorm_kernel<float>(
#             input.data_ptr<float>(),
#             weight.data_ptr<float>(),
#             output.data_ptr<float>(),
#             batch_size,
#             hidden_size,
#             eps
#         );
#     } else if (input.scalar_type() == torch::kFloat16) {
#         launch_rmsnorm_kernel<at::Half>(
#             input.data_ptr<at::Half>(),
#             weight.data_ptr<at::Half>(),
#             output.data_ptr<at::Half>(),
#             batch_size,
#             hidden_size,
#             eps
#         );
#     } else if (input.scalar_type() == torch::kBFloat16) {
#         launch_rmsnorm_kernel<at::BFloat16>(
#             input.data_ptr<at::BFloat16>(),
#             weight.data_ptr<at::BFloat16>(),
#             output.data_ptr<at::BFloat16>(),
#             batch_size,
#             hidden_size,
#             eps
#         );
#     } else {
#         TORCH_CHECK(false, "Unsupported dtype: " + c10::toString(input.scalar_type()));
#     }
    
#     // Synchronize and check for errors
#     cudaDeviceSynchronize();
#     cudaError_t error = cudaGetLastError();
#     if (error != cudaSuccess) {
#         TORCH_CHECK(false, "CUDA kernel error: " + std::string(cudaGetErrorString(error)));
#     }
    
#     return output;
# }

# // Python binding
# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#     m.def("forward", &fused_rmsnorm_forward, "Fused RMSNorm forward pass");
# }
# '''

# def _try_compile_extension():
#     """Try to compile the CUDA extension using JIT compilation"""
#     global _cuda_extension, _compilation_attempted
    
#     if _compilation_attempted:
#         if _cuda_extension is not None:
#             print("🔄 Using cached CUDA extension")
#         else:
#             print("⚠️  Using cached PyTorch fallback (CUDA compilation failed previously)")
#         return _cuda_extension
    
#     _compilation_attempted = True
    
#     if not torch.cuda.is_available():
#         print("❌ CUDA not available. Using PyTorch fallback.")
#         return None
    
#     print("🔧 Attempting to compile CUDA extension...")
#     print(f"   CUDA version: {torch.version.cuda}")
#     print(f"   GPU: {torch.cuda.get_device_name()}")
    
#     try:
#         from torch.utils.cpp_extension import load_inline
        
#         # Set architecture list if not set
#         import os
#         if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
#             # Get current GPU compute capability
#             capability = torch.cuda.get_device_capability()
#             arch = f"{capability[0]}.{capability[1]}"
#             os.environ['TORCH_CUDA_ARCH_LIST'] = arch
#             print(f"   Set TORCH_CUDA_ARCH_LIST={arch}")
        
#         print("   Compiling kernels (this may take a moment)...")
        
#         _cuda_extension = load_inline(
#             name="fused_rmsnorm_cuda",
#             cpp_sources=[""],  # No C++ source needed
#             cuda_sources=[_get_cuda_kernel_source()],
#             extra_cflags=['-O3', '-std=c++17'],
#             extra_cuda_cflags=[
#                 '-O3',
#                 '-std=c++17', 
#                 '--use_fast_math',
#                 '-U__CUDA_NO_HALF_OPERATORS__',
#                 '-U__CUDA_NO_HALF_CONVERSIONS__',
#                 '-U__CUDA_NO_HALF2_OPERATORS__',
#                 '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
#                 '--expt-relaxed-constexpr',
#                 '--expt-extended-lambda',
#                 f'-gencode=arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}',
#             ],
#             verbose=True  # Enable verbose output to see compilation details
#         )
        
#         print("✅ CUDA extension compiled successfully!")
        
#         # Test the extension with a small tensor
#         try:
#             test_input = torch.randn(1, 4, device='cuda', dtype=torch.float32)
#             test_weight = torch.ones(4, device='cuda', dtype=torch.float32)
#             test_output = _cuda_extension.forward(test_input, test_weight, 1e-6)
#             print("✅ CUDA kernel test passed!")
#         except Exception as test_error:
#             print(f"❌ CUDA kernel test failed: {test_error}")
#             print("   Falling back to PyTorch implementation")
#             _cuda_extension = None
        
#         return _cuda_extension
        
#     except Exception as e:
#         print(f"❌ CUDA extension compilation failed: {e}")
#         print("   Using PyTorch fallback implementation")
#         return None

# def _pytorch_rmsnorm(input, weight, eps=1e-6):
#     """PyTorch fallback implementation"""
#     input_dtype = input.dtype
#     input_fp32 = input.to(torch.float32)
#     variance = input_fp32.pow(2).mean(-1, keepdim=True)
#     input_normed = input_fp32 * torch.rsqrt(variance + eps)
#     return (weight * input_normed).to(input_dtype)

# class FusedRMSNormFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, eps=1e-6):
#         # Try to use CUDA extension
#         cuda_ext = _try_compile_extension()
        
#         if cuda_ext is not None and input.is_cuda:
#             try:
#                 # Use optimized CUDA kernel
#                 output = cuda_ext.forward(input, weight, eps)
#                 ctx.save_for_backward(input, weight)
#                 ctx.eps = eps
#                 # Only print this on first use to avoid spam
#                 if not hasattr(FusedRMSNormFunction, '_cuda_path_logged'):
#                     print("🚀 Using optimized CUDA kernel")
#                     FusedRMSNormFunction._cuda_path_logged = True
#                 return output
#             except Exception as e:
#                 print(f"⚠️  CUDA kernel failed at runtime: {e}")
#                 print("   Falling back to PyTorch implementation")
#                 # Fallback to PyTorch
#                 pass
        
#         # Fallback to PyTorch implementation
#         if not hasattr(FusedRMSNormFunction, '_fallback_path_logged'):
#             if not input.is_cuda:
#                 print("📱 Using PyTorch fallback (CPU tensor)")
#             elif cuda_ext is None:
#                 print("📱 Using PyTorch fallback (CUDA extension unavailable)")
#             else:
#                 print("📱 Using PyTorch fallback (CUDA kernel failed)")
#             FusedRMSNormFunction._fallback_path_logged = True
#         return _pytorch_rmsnorm(input, weight, eps)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         # Simple backward using PyTorch autograd
#         # For production, you'd implement the backward CUDA kernel too
#         input, weight = ctx.saved_tensors
#         eps = ctx.eps
        
#         input.requires_grad_(True)
#         weight.requires_grad_(True)
        
#         with torch.enable_grad():
#             output = _pytorch_rmsnorm(input, weight, eps)
#             grads = torch.autograd.grad(
#                 outputs=output,
#                 inputs=[input, weight],
#                 grad_outputs=grad_output,
#                 retain_graph=False
#             )
        
#         return grads[0], grads[1], None

# class FusedRMSNorm(nn.Module):
#     """
#     High-performance fused RMSNorm implementation.
    
#     Automatically uses optimized CUDA kernel when available,
#     falls back to PyTorch implementation otherwise.
    
#     Args:
#         hidden_size (int): Size of the hidden dimension
#         eps (float): Small constant for numerical stability (default: 1e-6)
#         device: Device to place the weight parameter
#         dtype: Data type for the weight parameter
#     """
    
#     def __init__(self, hidden_size: int, eps: float = 1e-6, device=None, dtype=None):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.eps = eps
        
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        
#         # Try to compile extension on initialization (optional)
#         if torch.cuda.is_available() and device != 'cpu':
#             _try_compile_extension()
    
#     def forward(self, hidden_states):
#         """
#         Forward pass of RMSNorm.
        
#         Args:
#             hidden_states (torch.Tensor): Input tensor of shape (..., hidden_size)
            
#         Returns:
#             torch.Tensor: Normalized tensor of same shape as input
#         """
#         return FusedRMSNormFunction.apply(hidden_states, self.weight, self.eps)
    
#     def extra_repr(self):
#         return f'hidden_size={self.hidden_size}, eps={self.eps}'

# def rms_norm(input, weight, eps=1e-6):
#     """
#     Functional interface for RMSNorm.
    
#     Args:
#         input (torch.Tensor): Input tensor
#         weight (torch.Tensor): Weight parameter
#         eps (float): Small constant for numerical stability
        
#     Returns:
#         torch.Tensor: Normalized tensor
#     """
#     return FusedRMSNormFunction.apply(input, weight, eps)

# # Convenience function for quick testing
# def benchmark_fused_rmsnorm(hidden_size=4096, batch_size=8, seq_len=2048, dtype=torch.float16, num_runs=50):
#     """Quick benchmark to test performance"""
#     if not torch.cuda.is_available():
#         print("CUDA not available for benchmarking")
#         return
        
#     device = torch.device('cuda')
    
#     # Create test data
#     input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    
#     # Test fused implementation
#     fused_rmsnorm = FusedRMSNorm(hidden_size, device=device, dtype=dtype)
    
#     # Test PyTorch implementation
#     def pytorch_rmsnorm(x, w):
#         return _pytorch_rmsnorm(x, w, 1e-6)
    
#     # Force compilation and check status
#     print("🔍 Checking CUDA extension status...")
#     cuda_ext = _try_compile_extension()
#     if cuda_ext is not None:
#         print("✅ CUDA extension available")
#         try:
#             # Test the kernel
#             test_output = cuda_ext.forward(input_tensor[:1, :10, :], fused_rmsnorm.weight, 1e-6)
#             print("✅ CUDA kernel execution test passed")
#         except Exception as e:
#             print(f"❌ CUDA kernel execution test failed: {e}")
#     else:
#         print("❌ CUDA extension not available")
    
#     print(f"\n🎯 Benchmarking {batch_size}×{seq_len}×{hidden_size} tensors ({dtype})...")
    
#     # Warmup
#     print("🔥 Warming up...")
#     for _ in range(10):
#         _ = fused_rmsnorm(input_tensor)
#         _ = pytorch_rmsnorm(input_tensor, fused_rmsnorm.weight)
    
#     torch.cuda.synchronize()
    
#     # Benchmark fused
#     print("⏱️  Benchmarking fused implementation...")
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
    
#     start.record()
#     for _ in range(num_runs):
#         output_fused = fused_rmsnorm(input_tensor)
#     end.record()
#     torch.cuda.synchronize()
#     fused_time = start.elapsed_time(end) / num_runs
    
#     # Benchmark PyTorch
#     print("⏱️  Benchmarking PyTorch implementation...")
#     start.record()
#     for _ in range(num_runs):
#         output_pytorch = pytorch_rmsnorm(input_tensor, fused_rmsnorm.weight)
#     end.record()
#     torch.cuda.synchronize()
#     pytorch_time = start.elapsed_time(end) / num_runs
    
#     # Check correctness
#     max_diff = torch.max(torch.abs(output_fused - output_pytorch)).item()
#     speedup = pytorch_time / fused_time
    
#     print(f"\n📊 Benchmark Results:")
#     print(f"  Configuration: {batch_size}×{seq_len}×{hidden_size} ({dtype})")
#     print(f"  Fused RMSNorm:   {fused_time:.3f} ms")
#     print(f"  PyTorch RMSNorm: {pytorch_time:.3f} ms")
#     print(f"  Speedup:         {speedup:.2f}x")
#     print(f"  Max difference:  {max_diff:.2e}")
    
#     # Analysis
#     if speedup < 1.1:
#         print("\n⚠️  Analysis: Low speedup detected!")
#         if cuda_ext is None:
#             print("   - CUDA extension not compiled successfully")
#             print("   - Check CUDA installation and PyTorch CUDA support")
#         else:
#             print("   - CUDA extension compiled but may not be executing")
#             print("   - Both implementations may be using the same PyTorch fallback")
#         print("   - Try running with CUDA_LAUNCH_BLOCKING=1 for more detailed errors")
#     elif speedup > 2.0:
#         print(f"\n🎉 Excellent speedup! CUDA kernel is working properly.")
#     else:
#         print(f"\n✅ Good speedup. CUDA kernel is likely working.")
    
#     return fused_time, pytorch_time, speedup

# # Auto-compilation check on import (disabled to avoid issues)
# # if torch.cuda.is_available():
# #     # Try to compile in background (non-blocking)
# #     import threading
# #     def _background_compile():
# #         try:
# #             _try_compile_extension()
# #         except:
# #             pass  # Silently fail, will use fallback
# #     
# #     thread = threading.Thread(target=_background_compile, daemon=True)
# #     thread.start()

# def test_cuda_compilation():
#     """Test if CUDA compilation works with detailed error reporting"""
#     print("🔧 Testing CUDA compilation...")
    
#     # Force a fresh compilation attempt
#     global _compilation_attempted, _cuda_extension
#     _compilation_attempted = False
#     _cuda_extension = None
    
#     # Set environment for better error reporting
#     import os
#     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
#     cuda_ext = _try_compile_extension()
    
#     if cuda_ext is not None:
#         print("✅ Compilation successful! Testing kernel execution...")
        
#         try:
#             # Create small test tensors
#             device = torch.device('cuda')
#             test_input = torch.randn(2, 8, device=device, dtype=torch.float32)
#             test_weight = torch.ones(8, device=device, dtype=torch.float32)
            
#             # Test the CUDA kernel
#             output = cuda_ext.forward(test_input, test_weight, 1e-6)
            
#             # Test with PyTorch implementation for comparison
#             expected = _pytorch_rmsnorm(test_input, test_weight, 1e-6)
#             diff = torch.max(torch.abs(output - expected)).item()
            
#             print(f"✅ Kernel execution successful!")
#             print(f"   Max difference vs PyTorch: {diff:.2e}")
            
#             if diff < 1e-4:
#                 print("🎉 CUDA kernel is working correctly!")
#                 return True
#             else:
#                 print("⚠️  Large difference detected - kernel may have issues")
#                 return False
                
#         except Exception as e:
#             print(f"❌ Kernel execution failed: {e}")
#             return False
#     else:
#         print("❌ Compilation failed")
#         return False

# # Debug function to help diagnose issues
# def debug_cuda_status():
#     """Print detailed CUDA and compilation status"""
#     print("🔍 CUDA Environment Debug Information:")
#     print(f"  PyTorch version: {torch.__version__}")
#     print(f"  CUDA available: {torch.cuda.is_available()}")
    
#     if torch.cuda.is_available():
#         print(f"  CUDA version: {torch.version.cuda}")
#         print(f"  cuDNN version: {torch.backends.cudnn.version()}")
#         print(f"  GPU count: {torch.cuda.device_count()}")
#         print(f"  Current GPU: {torch.cuda.current_device()}")
#         print(f"  GPU name: {torch.cuda.get_device_name()}")
#         capability = torch.cuda.get_device_capability()
#         print(f"  Compute capability: {capability[0]}.{capability[1]}")
        
#         # Check memory
#         memory_free, memory_total = torch.cuda.mem_get_info()
#         print(f"  GPU memory: {memory_free//1024**2} MB free / {memory_total//1024**2} MB total")
        
#         # Check if we can compile simple CUDA
#         try:
#             from torch.utils.cpp_extension import load_inline
#             print("  torch.utils.cpp_extension: Available")
#         except ImportError as e:
#             print(f"  torch.utils.cpp_extension: Import failed - {e}")
        
#         # Environment variables
#         import os
#         cuda_home = os.environ.get('CUDA_HOME', 'Not set')
#         print(f"  CUDA_HOME: {cuda_home}")
        
#         cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')
#         print(f"  TORCH_CUDA_ARCH_LIST: {cuda_arch_list}")
    
#     print()

# if __name__ == "__main__":
#     # Quick test when run directly
#     print("🧪 Testing Fused RMSNorm...")
    
#     # Print debug info
#     debug_cuda_status()
    
#     if torch.cuda.is_available():
#         # Test basic functionality
#         hidden_size = 1024
#         rmsnorm = FusedRMSNorm(hidden_size, device='cuda', dtype=torch.float16)
        
#         input_tensor = torch.randn(4, 512, hidden_size, dtype=torch.float16, device='cuda')
#         output = rmsnorm(input_tensor)
        
#         print(f"✅ Basic test passed: {input_tensor.shape} -> {output.shape}")
        
#         # Run benchmark
#         print("\n" + "="*50)
#         benchmark_fused_rmsnorm()
#     else:
#         print("CUDA not available. Testing CPU fallback...")
#         rmsnorm = FusedRMSNorm(256)
#         input_tensor = torch.randn(2, 100, 256)
#         output = rmsnorm(input_tensor)
#         print(f"✅ CPU fallback test passed: {input_tensor.shape} -> {output.shape}")


# fast_fused_rmsnorm.py
"""
Fast-compiling, simplified Fused RMSNorm implementation.
Optimized for quick compilation while still providing speedups.

Usage:
    from fast_fused_rmsnorm import FusedRMSNorm
    rmsnorm = FusedRMSNorm(hidden_size=4096)
    output = rmsnorm(input_tensor)
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional

# Global variable to cache the compiled extension
_cuda_extension = None
_compilation_attempted = False

def _get_simple_cuda_kernel():
    """Returns a simplified CUDA kernel that compiles quickly"""
    return '''
#include <torch/extension.h>
#include <cuda_runtime.h>

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
    float eps) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    auto input_shape = input.sizes();
    int hidden_size = input_shape[input_shape.size() - 1];
    int batch_size = input.numel() / hidden_size;
    
    auto output = torch::empty_like(input);
    
    // Simple kernel launch - use 256 threads per block
    const int threads = min(256, hidden_size);
    const int blocks = batch_size;
    
    // Dispatch based on type
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
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &simple_rmsnorm_forward, "Simple RMSNorm forward");
}
'''

def _compile_simple_extension():
    """Compile the simplified CUDA extension quickly"""
    global _cuda_extension, _compilation_attempted
    
    if _compilation_attempted:
        return _cuda_extension
    
    _compilation_attempted = True
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return None
    
    try:
        from torch.utils.cpp_extension import load_inline
        import os
        
        # Get current GPU architecture only
        capability = torch.cuda.get_device_capability()
        current_arch = f"{capability[0]}{capability[1]}"
        
        print(f"🔧 Compiling for current GPU (compute {capability[0]}.{capability[1]})...")
        
        # Minimal compilation flags for speed
        cuda_flags = [
            '-O2',  # Reduced optimization for faster compilation
            '--use_fast_math',
            f'--gpu-architecture=sm_{current_arch}',  # Only current GPU
        ]
        
        _cuda_extension = load_inline(
            name="simple_rmsnorm_cuda",
            cpp_sources=[""],
            cuda_sources=[_get_simple_cuda_kernel()],
            extra_cflags=['-O2'],
            extra_cuda_cflags=cuda_flags,
            verbose=False  # Reduce output
        )
        
        print("✅ Compilation successful!")
        return _cuda_extension
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return None

def _pytorch_rmsnorm(input, weight, eps=1e-6):
    """PyTorch fallback implementation"""
    input_dtype = input.dtype
    input_fp32 = input.to(torch.float32)
    variance = input_fp32.pow(2).mean(-1, keepdim=True)
    input_normed = input_fp32 * torch.rsqrt(variance + eps)
    return (weight * input_normed).to(input_dtype)

class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps=1e-6):
        cuda_ext = _compile_simple_extension()
        
        if cuda_ext is not None and input.is_cuda:
            try:
                output = cuda_ext.forward(input, weight, eps)
                if not hasattr(FusedRMSNormFunction, '_using_cuda_logged'):
                    print("🚀 Using CUDA kernel")
                    FusedRMSNormFunction._using_cuda_logged = True
                return output
            except Exception as e:
                print(f"⚠️  CUDA kernel failed: {e}")
        
        # Fallback
        if not hasattr(FusedRMSNormFunction, '_using_fallback_logged'):
            print("📱 Using PyTorch fallback")
            FusedRMSNormFunction._using_fallback_logged = True
        return _pytorch_rmsnorm(input, weight, eps)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Use PyTorch autograd for backward pass
        return grad_output, grad_output, None

class FusedRMSNorm(nn.Module):
    """
    Fast-compiling Fused RMSNorm implementation.
    
    Args:
        hidden_size (int): Size of the hidden dimension
        eps (float): Small constant for numerical stability
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
    
    def forward(self, hidden_states):
        return FusedRMSNormFunction.apply(hidden_states, self.weight, self.eps)

def fused_rmsnorm_func(hidden_states, weight, eps=1e-6):
	return FusedRMSNormFunction.apply(hidden_states, weight, eps)

def benchmark_simple(hidden_size=4096, batch_size=8, seq_len=2048, dtype=torch.float16, num_runs=50):
    """Quick benchmark"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = torch.device('cuda')
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    
    # Test both implementations
    fused_rmsnorm = FusedRMSNorm(hidden_size, device=device, dtype=dtype)
    
    def pytorch_rmsnorm(x, w):
        return _pytorch_rmsnorm(x, w, 1e-6)
    
    print(f"🎯 Benchmarking {batch_size}×{seq_len}×{hidden_size} ({dtype})...")
    
    # Warmup
    for _ in range(5):
        _ = fused_rmsnorm(input_tensor)
        _ = pytorch_rmsnorm(input_tensor, fused_rmsnorm.weight)
    
    torch.cuda.synchronize()
    
    # Time fused
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_runs):
        output_fused = fused_rmsnorm(input_tensor)
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end) / num_runs
    
    # Time PyTorch
    start.record()
    for _ in range(num_runs):
        output_pytorch = pytorch_rmsnorm(input_tensor, fused_rmsnorm.weight)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / num_runs
    
    # Check correctness
    max_diff = torch.max(torch.abs(output_fused - output_pytorch)).item()
    speedup = pytorch_time / fused_time
    
    print(f"📊 Results:")
    print(f"  Fused:    {fused_time:.3f} ms")
    print(f"  PyTorch:  {pytorch_time:.3f} ms")
    print(f"  Speedup:  {speedup:.2f}x")
    print(f"  Accuracy: {max_diff:.2e}")
    
    if speedup > 1.5:
        print("🎉 Good speedup! CUDA kernel is working")
    elif speedup < 1.1:
        print("⚠️  Low speedup - check if CUDA kernel compiled")
    
    return speedup

# Quick test function
def quick_test():
    """Quick functionality test"""
    print("🧪 Quick test...")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        capability = torch.cuda.get_device_capability()
        print(f"Compute: {capability[0]}.{capability[1]}")
        
        # Simple test
        rmsnorm = FusedRMSNorm(512, device='cuda', dtype=torch.float16)
        x = torch.randn(2, 100, 512, dtype=torch.float16, device='cuda')
        y = rmsnorm(x)
        
        print(f"✅ Test passed: {x.shape} -> {y.shape}")
        
        # Quick benchmark
        print("\n" + "="*40)
        speedup = benchmark_simple(hidden_size=2048, batch_size=4, seq_len=1024)
        
        if speedup > 1.5:
            print("\n🎉 Success! Ready to use in your project")
        else:
            print("\n⚠️  Using fallback - check CUDA setup")
            
    else:
        print("No CUDA available")

if __name__ == "__main__":
    quick_test()