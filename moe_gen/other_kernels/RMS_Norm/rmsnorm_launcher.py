# fused_rmsnorm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import warnings

# Try to load the compiled extension
_fused_rmsnorm_cuda = None

def _load_cuda_extension():
    global _fused_rmsnorm_cuda
    if _fused_rmsnorm_cuda is None:
        try:
            # Try to import pre-compiled extension
            import fused_rmsnorm_cuda
            _fused_rmsnorm_cuda = fused_rmsnorm_cuda
        except ImportError:
            try:
                # JIT compile if not available
                current_dir = os.path.dirname(os.path.abspath(__file__))
                cuda_file = os.path.join(current_dir, "fused_rmsnorm_kernel.cu")
                
                _fused_rmsnorm_cuda = load(
                    name="fused_rmsnorm_cuda",
                    sources=[cuda_file],
                    extra_cflags=['-O3', '-std=c++17'],
                    extra_cuda_cflags=[
                        '-O3', 
                        '-std=c++17',
                        '--use_fast_math',
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '-U__CUDA_NO_HALF2_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                        '--expt-relaxed-constexpr',
                        '--expt-extended-lambda',
                        '-gencode=arch=compute_70,code=sm_70',  # V100
                        '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 20xx
                        '-gencode=arch=compute_80,code=sm_80',  # A100
                        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
                        '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
                        '-gencode=arch=compute_90,code=sm_90',  # H100
                    ],
                    verbose=True
                )
            except Exception as e:
                warnings.warn(f"Failed to load CUDA extension: {e}. Falling back to PyTorch implementation.")
                _fused_rmsnorm_cuda = None
    
    return _fused_rmsnorm_cuda


class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps=1e-6):
        cuda_ext = _load_cuda_extension()
        
        if cuda_ext is not None and input.is_cuda:
            # Use optimized CUDA kernel
            output = cuda_ext.forward(input, weight, eps)
            
            # Save for backward
            ctx.save_for_backward(input, weight)
            ctx.eps = eps
            return output
        else:
            # Fallback to PyTorch implementation
            return _pytorch_rmsnorm_forward(input, weight, eps)
    
    @staticmethod
    def backward(ctx, grad_output):
        # For now, use PyTorch autograd for backward pass
        # In production, you'd implement the backward CUDA kernel too
        input, weight = ctx.saved_tensors
        eps = ctx.eps
        
        # Enable gradients for fallback computation
        input.requires_grad_(True)
        weight.requires_grad_(True)
        
        with torch.enable_grad():
            output = _pytorch_rmsnorm_forward(input, weight, eps)
            grads = torch.autograd.grad(
                outputs=output,
                inputs=[input, weight],
                grad_outputs=grad_output,
                retain_graph=False
            )
        
        return grads[0], grads[1], None


def _pytorch_rmsnorm_forward(input, weight, eps=1e-6):
    """PyTorch fallback implementation"""
    input_dtype = input.dtype
    input_fp32 = input.to(torch.float32)
    variance = input_fp32.pow(2).mean(-1, keepdim=True)
    input_normed = input_fp32 * torch.rsqrt(variance + eps)
    return (weight * input_normed).to(input_dtype)


class FusedRMSNorm(nn.Module):
    """
    Fused RMSNorm implementation that automatically chooses between
    optimized CUDA kernel and PyTorch fallback.
    
    Args:
        hidden_size (int): Size of the hidden dimension
        eps (float): Small constant for numerical stability
        device (torch.device): Device to place the weight parameter
        dtype (torch.dtype): Data type for the weight parameter
    """
    
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Initialize weight parameter
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        
        # Try to load CUDA extension on initialization
        _load_cuda_extension()
    
    def forward(self, hidden_states):
        """
        Forward pass of RMSNorm.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (..., hidden_size)
            
        Returns:
            torch.Tensor: Normalized tensor of same shape as input
        """
        return FusedRMSNormFunction.apply(hidden_states, self.weight, self.eps)
    
    def extra_repr(self):
        return f'hidden_size={self.hidden_size}, eps={self.eps}'


def rms_norm(input, weight, eps=1e-6):
    """
    Functional interface for RMSNorm.
    
    Args:
        input (torch.Tensor): Input tensor
        weight (torch.Tensor): Weight parameter
        eps (float): Small constant for numerical stability
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    return FusedRMSNormFunction.apply(input, weight, eps)


# Convenience function for replacing existing RMSNorm implementations
def replace_rmsnorm_with_fused(model, eps=1e-6):
    """
    Replace all RMSNorm layers in a model with FusedRMSNorm.
    
    Args:
        model (nn.Module): Model to modify
        eps (float): Epsilon value for new layers
        
    Returns:
        nn.Module: Modified model
    """
    for name, module in model.named_children():
        if hasattr(module, 'weight') and hasattr(module, 'variance_epsilon'):
            # Assuming this is an RMSNorm layer
            hidden_size = module.weight.size(0)
            device = module.weight.device
            dtype = module.weight.dtype
            
            # Create new fused layer
            new_layer = FusedRMSNorm(hidden_size, eps, device, dtype)
            new_layer.weight.data.copy_(module.weight.data)
            
            # Replace the layer
            setattr(model, name, new_layer)
        else:
            # Recursively replace in child modules
            replace_rmsnorm_with_fused(module, eps)
    
    return model


# Performance testing utilities
def benchmark_rmsnorm(batch_size=32, seq_len=2048, hidden_size=4096, dtype=torch.float16, num_runs=100):
    """
    Benchmark the fused RMSNorm against PyTorch implementation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    
    # Initialize layers
    fused_rmsnorm = FusedRMSNorm(hidden_size).to(device=device, dtype=dtype)
    fused_rmsnorm.weight.data.copy_(weight)
    
    def pytorch_rmsnorm(x, w, eps=1e-6):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (w * x).to(input_dtype)
    
    # Warmup
    for _ in range(10):
        _ = fused_rmsnorm(input_tensor)
        _ = pytorch_rmsnorm(input_tensor, weight)
    
    torch.cuda.synchronize()
    
    # Benchmark fused implementation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        output_fused = fused_rmsnorm(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    fused_time = start_event.elapsed_time(end_event) / num_runs
    
    # Benchmark PyTorch implementation
    start_event.record()
    for _ in range(num_runs):
        output_pytorch = pytorch_rmsnorm(input_tensor, weight)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event) / num_runs
    
    # Check correctness
    max_diff = torch.max(torch.abs(output_fused - output_pytorch)).item()
    
    print(f"Benchmark Results (batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}, dtype={dtype}):")
    print(f"  Fused RMSNorm:   {fused_time:.3f} ms")
    print(f"  PyTorch RMSNorm: {pytorch_time:.3f} ms")
    print(f"  Speedup:         {pytorch_time / fused_time:.2f}x")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Memory usage:    {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    return fused_time, pytorch_time


if __name__ == "__main__":
    # Example usage
    print("Testing Fused RMSNorm...")
    
    # Test basic functionality
    hidden_size = 4096
    batch_size = 8
    seq_len = 2048
    
    # Create test input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)
    
    # Create RMSNorm layer
    rmsnorm = FusedRMSNorm(hidden_size).to(device=device, dtype=torch.float16)
    
    # Forward pass
    output = rmsnorm(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Run benchmark
    if torch.cuda.is_available():
        print("\nRunning benchmark...")
        benchmark_rmsnorm()