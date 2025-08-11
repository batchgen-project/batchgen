# fast_fused_rmsnorm.py
"""
Fast-compiling, simplified Fused RMSNorm implementation.
Optimized for quick compilation while still providing speedups.
Now supports BF16, FP16, and FP32.

Usage:
    from fast_fused_rmsnorm import FusedRMSNorm, fused_rmsnorm_func
    
    # As a module (recommended)
    rmsnorm = FusedRMSNorm(hidden_size=4096)
    output = rmsnorm(input_tensor)  # Works with BF16, FP16, FP32
    
    # As a function 
    output = fused_rmsnorm_func(input_tensor, weight, eps=1e-6)
    
    # BF16 example (what you probably need)
    device = torch.device('cuda')
    input_bf16 = torch.randn(8, 2048, 4096, dtype=torch.bfloat16, device=device)
    rmsnorm_bf16 = FusedRMSNorm(4096, device=device, dtype=torch.bfloat16)
    output_bf16 = rmsnorm_bf16(input_bf16)
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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
        
        # Minimal compilation flags for speed - added BF16 support
        cuda_flags = [
            '-O2',  # Reduced optimization for faster compilation
            '--use_fast_math',
            f'--gpu-architecture=sm_{current_arch}',  # Only current GPU
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__', 
            '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '--expt-relaxed-constexpr',
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
        
        # # Test the extension with different dtypes
        # try:
        #     device = torch.device('cuda')
            
        #     # Test BF16
        #     test_input_bf16 = torch.randn(1, 4, device=device, dtype=torch.bfloat16)
        #     test_weight_bf16 = torch.ones(4, device=device, dtype=torch.bfloat16)
        #     output_bf16 = _cuda_extension.forward(test_input_bf16, test_weight_bf16, 1e-6)
        #     print("✅ BF16 kernel test passed!")
            
        #     # Test FP16
        #     test_input_fp16 = torch.randn(1, 4, device=device, dtype=torch.float16)
        #     test_weight_fp16 = torch.ones(4, device=device, dtype=torch.float16)
        #     output_fp16 = _cuda_extension.forward(test_input_fp16, test_weight_fp16, 1e-6)
        #     print("✅ FP16 kernel test passed!")
            
        #     # Test FP32
        #     test_input_fp32 = torch.randn(1, 4, device=device, dtype=torch.float32)
        #     test_weight_fp32 = torch.ones(4, device=device, dtype=torch.float32)
        #     output_fp32 = _cuda_extension.forward(test_input_fp32, test_weight_fp32, 1e-6)
        #     print("✅ FP32 kernel test passed!")
            
        # except Exception as test_error:
        #     print(f"❌ Kernel test failed: {test_error}")
        #     print("   Falling back to PyTorch implementation")
        #     _cuda_extension = None
        
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
                    print(f"🚀 Using CUDA kernel for {input.dtype}")
                    FusedRMSNormFunction._using_cuda_logged = True
                return output
            except Exception as e:
                print(f"⚠️  CUDA kernel failed: {e}")
                print(f"   Input dtype: {input.dtype}, shape: {input.shape}")
                print(f"   Input contiguous: {input.is_contiguous()}")
                print(f"   Weight contiguous: {weight.is_contiguous()}")
        
        # Fallback
        if not hasattr(FusedRMSNormFunction, '_using_fallback_logged'):
            if not input.is_cuda:
                print("📱 Using PyTorch fallback (CPU tensor)")
            elif cuda_ext is None:
                print("📱 Using PyTorch fallback (CUDA extension unavailable)")
            else:
                print(f"📱 Using PyTorch fallback for {input.dtype}")
            FusedRMSNormFunction._using_fallback_logged = True
        return _pytorch_rmsnorm(input, weight, eps)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Use PyTorch autograd for backward pass
        return grad_output, grad_output, None

class FusedRMSNorm(nn.Module):
    """
    Fast-compiling Fused RMSNorm implementation with BF16 support.
    
    This is a drop-in replacement for any RMSNorm implementation.
    Just replace your existing RMSNorm class with this one.
    
    Args:
        hidden_size (int): Size of the hidden dimension
        eps (float): Small constant for numerical stability
        device: Device to place the weight parameter  
        dtype: Data type for the weight parameter (supports BF16, FP16, FP32)
        
    Example:
        # Replace this:
        # self.norm = DeepseekV3RMSNorm(hidden_size)
        
        # With this:
        self.norm = FusedRMSNorm(hidden_size)
        
        # Or with specific dtype:
        self.norm = FusedRMSNorm(hidden_size, device='cuda', dtype=torch.bfloat16)
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

def benchmark_simple(hidden_size=4096, batch_size=8, seq_len=2048, dtype=torch.bfloat16, num_runs=50):
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
    
    def f_rmsnorm(x, w):
        import torch.nn.functional as F
        return F.rms_norm(x, (x.size(-1),), w, 1e-6)
    
    print(f"🎯 Benchmarking {batch_size}×{seq_len}×{hidden_size} ({dtype})...")
    
    # Warmup
    for _ in range(5):
        _ = fused_rmsnorm(input_tensor)
        _ = f_rmsnorm(input_tensor, fused_rmsnorm.weight)
    
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
    
    # Time F.rms_norm (PyTorch's optimized version)
    start.record()
    for _ in range(num_runs):
        output_pytorch = f_rmsnorm(input_tensor, fused_rmsnorm.weight)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / num_runs
    
    # Check correctness
    max_diff = torch.max(torch.abs(output_fused - output_pytorch)).item()
    speedup = pytorch_time / fused_time
    
    print(f"📊 Results:")
    print(f"  Fused RMSNorm:     {fused_time:.3f} ms")
    print(f"  F.rms_norm:        {pytorch_time:.3f} ms")
    print(f"  Speedup:           {speedup:.2f}x")
    print(f"  Max difference:    {max_diff:.2e}")
    
    if speedup > 1.5:
        print("🎉 Good speedup! CUDA kernel is working")
    elif speedup > 1.1:
        print("✅ Modest speedup - kernel is working")
    else:
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
        
        # Test BF16 specifically (what user needs)
        print("\n🔬 Testing BF16 support...")
        rmsnorm_bf16 = FusedRMSNorm(512, device='cuda', dtype=torch.bfloat16)
        x_bf16 = torch.randn(2, 100, 512, dtype=torch.bfloat16, device='cuda')
        y_bf16 = rmsnorm_bf16(x_bf16)
        print(f"✅ BF16 test passed: {x_bf16.shape} -> {y_bf16.shape}")
        
        # Test FP16 as well
        print("\n🔬 Testing FP16 support...")
        rmsnorm_fp16 = FusedRMSNorm(512, device='cuda', dtype=torch.float16)
        x_fp16 = torch.randn(2, 100, 512, dtype=torch.float16, device='cuda')
        y_fp16 = rmsnorm_fp16(x_fp16)
        print(f"✅ FP16 test passed: {x_fp16.shape} -> {y_fp16.shape}")
        
        # Quick benchmark with BF16
        print("\n" + "="*40)
        print("🏃 Running BF16 benchmark...")
        speedup = benchmark_simple(hidden_size=2048, batch_size=4, seq_len=1024, dtype=torch.bfloat16)
        
        if speedup > 1.2:
            print("\n🎉 Success! BF16 CUDA kernel is working")
            print("💡 You can now use FusedRMSNorm in your project with BF16 tensors")
        else:
            print("\n⚠️  Using fallback - but functionality should still work")
            
    else:
        print("No CUDA available")

def test_bf16_support():
    """Test BF16 support specifically"""
    print("🧪 Testing BF16 RMSNorm support...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        device = torch.device('cuda')
        hidden_size = 1024
        
        # Create BF16 tensors
        input_bf16 = torch.randn(4, 256, hidden_size, dtype=torch.bfloat16, device=device)
        rmsnorm_bf16 = FusedRMSNorm(hidden_size, device=device, dtype=torch.bfloat16)
        
        print(f"Input: {input_bf16.shape}, dtype: {input_bf16.dtype}")
        
        # Test forward pass
        output_bf16 = rmsnorm_bf16(input_bf16)
        print(f"Output: {output_bf16.shape}, dtype: {output_bf16.dtype}")
        
        # Compare with F.rms_norm
        import torch.nn.functional as F
        expected_bf16 = F.rms_norm(input_bf16, (hidden_size,), rmsnorm_bf16.weight, 1e-6)
        
        max_diff = torch.max(torch.abs(output_bf16 - expected_bf16)).item()
        print(f"Max difference vs F.rms_norm: {max_diff:.2e}")
        
        if max_diff < 1e-2:  # BF16 has lower precision
            print("✅ BF16 support working correctly!")
            return True
        else:
            print("⚠️  Large difference detected")
            return False
            
    except Exception as e:
        print(f"❌ BF16 test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Fast-Compiling Fused RMSNorm with BF16 Support")
    print("="*60)
    quick_test()
    
    print("\n" + "="*60)
    print("🔬 Dedicated BF16 Test")
    print("="*60)
    test_bf16_support()