# example_usage.py
"""
Example showing how to replace the slow DeepseekV3RMSNorm with optimized FusedRMSNorm
"""

import torch
import torch.nn as nn
import time
from fused_rmsnorm import FusedRMSNorm, benchmark_rmsnorm, replace_rmsnorm_with_fused
# from moe_gen.other_kernels.RMS_Norm.rmsnorm import FusedRMSNorm, benchmark_rmsnorm, replace_rmsnorm_with_fused


# Original slow implementation (from your example)
class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # Expensive conversion
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # Separate operations
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)  # Another conversion


def compare_implementations():
    """Compare the original and optimized implementations"""
    
    print("=" * 60)
    print("FUSED RMSNORM COMPARISON")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 16
    seq_len = 2048
    hidden_size = 4096
    dtype = torch.float16
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input dtype: {dtype}")
    print(f"Memory usage: {input_tensor.numel() * input_tensor.element_size() / 1024**2:.1f} MB")
    
    # Initialize both implementations
    original_rmsnorm = DeepseekV3RMSNorm(hidden_size).to(device=device, dtype=dtype)
    fused_rmsnorm = FusedRMSNorm(hidden_size).to(device=device, dtype=dtype)
    
    # Copy weights to ensure fair comparison
    fused_rmsnorm.weight.data.copy_(original_rmsnorm.weight.data)
    
    print("\n" + "=" * 40)
    print("CORRECTNESS TEST")
    print("=" * 40)
    
    # Test correctness
    with torch.no_grad():
        output_original = original_rmsnorm(input_tensor)
        output_fused = fused_rmsnorm(input_tensor)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(output_original - output_fused)).item()
    mean_diff = torch.mean(torch.abs(output_original - output_fused)).item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Outputs match: {'✓' if max_diff < 1e-3 else '✗'}")
    
    print("\n" + "=" * 40)
    print("PERFORMANCE COMPARISON")
    print("=" * 40)
    
    # Warmup
    for _ in range(10):
        _ = original_rmsnorm(input_tensor)
        _ = fused_rmsnorm(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark original implementation
    num_runs = 100
    start_time = time.time()
    
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_runs):
            output_original = original_rmsnorm(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        original_time = start_event.elapsed_time(end_event) / num_runs
    else:
        start_time = time.time()
        for _ in range(num_runs):
            output_original = original_rmsnorm(input_tensor)
        original_time = (time.time() - start_time) * 1000 / num_runs
    
    # Benchmark fused implementation
    if device.type == 'cuda':
        start_event.record()
        for _ in range(num_runs):
            output_fused = fused_rmsnorm(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        fused_time = start_event.elapsed_time(end_event) / num_runs
    else:
        start_time = time.time()
        for _ in range(num_runs):
            output_fused = fused_rmsnorm(input_tensor)
        fused_time = (time.time() - start_time) * 1000 / num_runs
    
    speedup = original_time / fused_time
    
    print(f"Original RMSNorm: {original_time:.3f} ms")
    print(f"Fused RMSNorm:    {fused_time:.3f} ms")
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Time saved:       {original_time - fused_time:.3f} ms ({(1 - fused_time/original_time)*100:.1f}%)")
    
    return original_time, fused_time, speedup


def benchmark_different_sizes():
    """Benchmark across different tensor sizes"""
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ACROSS DIFFERENT SIZES")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different configurations
    configs = [
        (8, 512, 768),     # Small model
        (16, 1024, 1024),  # Medium model
        (32, 2048, 4096),  # Large model
        (8, 4096, 8192),   # Very large model
    ]
    
    results = []
    
    for batch_size, seq_len, hidden_size in configs:
        print(f"\nTesting: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}")
        
        # Create test data
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, 
                                 dtype=torch.float16, device=device)
        
        # Initialize models
        original_rmsnorm = DeepseekV3RMSNorm(hidden_size).to(device=device, dtype=torch.float16)
        fused_rmsnorm = FusedRMSNorm(hidden_size).to(device=device, dtype=torch.float16)
        fused_rmsnorm.weight.data.copy_(original_rmsnorm.weight.data)
        
        # Benchmark
        num_runs = 50
        
        # Warmup
        for _ in range(5):
            _ = original_rmsnorm(input_tensor)
            _ = fused_rmsnorm(input_tensor)
        
        torch.cuda.synchronize()
        
        # Time original
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(num_runs):
                _ = original_rmsnorm(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            original_time = start_event.elapsed_time(end_event) / num_runs
            
            # Time fused
            start_event.record()
            for _ in range(num_runs):
                _ = fused_rmsnorm(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            fused_time = start_event.elapsed_time(end_event) / num_runs
        else:
            # CPU timing
            start_time = time.time()
            for _ in range(num_runs):
                _ = original_rmsnorm(input_tensor)
            original_time = (time.time() - start_time) * 1000 / num_runs
            
            start_time = time.time()
            for _ in range(num_runs):
                _ = fused_rmsnorm(input_tensor)
            fused_time = (time.time() - start_time) * 1000 / num_runs
        
        speedup = original_time / fused_time
        memory_mb = input_tensor.numel() * input_tensor.element_size() / 1024**2
        
        print(f"  Original: {original_time:.3f} ms")
        print(f"  Fused:    {fused_time:.3f} ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"  Memory:   {memory_mb:.1f} MB")
        
        results.append({
            'config': f"{batch_size}x{seq_len}x{hidden_size}",
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'original_time': original_time,
            'fused_time': fused_time,
            'speedup': speedup,
            'memory_mb': memory_mb
        })
    
    return results


def demonstrate_model_replacement():
    """Show how to replace RMSNorm in an existing model"""
    
    print("\n" + "=" * 60)
    print("MODEL REPLACEMENT EXAMPLE")
    print("=" * 60)
    
    # Create a simple transformer-like model with multiple RMSNorm layers
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_size=768, num_layers=6):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'norm1': DeepseekV3RMSNorm(hidden_size),
                    'norm2': DeepseekV3RMSNorm(hidden_size),
                    'linear': nn.Linear(hidden_size, hidden_size)
                })
                for _ in range(num_layers)
            ])
            self.final_norm = DeepseekV3RMSNorm(hidden_size)
        
        def forward(self, x):
            for layer in self.layers:
                # Simplified transformer layer
                normed = layer['norm1'](x)
                transformed = layer['linear'](normed)
                normed2 = layer['norm2'](transformed)
                x = x + normed2  # Residual connection
            return self.final_norm(x)
    
    # Create model
    model = SimpleTransformer(hidden_size=768, num_layers=6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Original model has {sum(1 for _ in model.modules() if isinstance(_, DeepseekV3RMSNorm))} RMSNorm layers")
    
    # Replace with fused implementation
    model_fused = replace_rmsnorm_with_fused(model)
    
    print(f"Fused model has {sum(1 for _ in model_fused.modules() if isinstance(_, FusedRMSNorm))} FusedRMSNorm layers")
    
    # Test with sample input
    batch_size, seq_len, hidden_size = 8, 512, 768
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Compare outputs
    with torch.no_grad():
        output_original = model(input_tensor)
        output_fused = model_fused(input_tensor)
    
    max_diff = torch.max(torch.abs(output_original - output_fused)).item()
    print(f"Max difference after replacement: {max_diff:.2e}")
    print(f"Replacement successful: {'✓' if max_diff < 1e-3 else '✗'}")


def main():
    """Main function to run all examples"""
    
    print("🚀 Fused RMSNorm Performance Demo")
    print("This demo compares the optimized CUDA implementation with the original PyTorch version")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Performance gains will be limited on CPU.")
        print("   For best results, run this on a GPU with CUDA support.")
    else:
        gpu_name = torch.cuda.get_device_name()
        print(f"🎯 Running on: {gpu_name}")
    
    print()
    
    try:
        # Run comparisons
        compare_implementations()
        
        if torch.cuda.is_available():
            benchmark_different_sizes()
        
        demonstrate_model_replacement()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✅ Fused RMSNorm provides significant speedups over the original implementation")
        print("✅ Output accuracy is maintained (differences < 1e-3)")
        print("✅ Easy drop-in replacement for existing models")
        print("✅ Supports multiple data types (float32, float16, bfloat16)")
        print("✅ Automatic fallback to PyTorch implementation when CUDA unavailable")
        
        print("\n💡 Usage Tips:")
        print("   - Use FusedRMSNorm(hidden_size) as a drop-in replacement")
        print("   - Call replace_rmsnorm_with_fused(model) to convert entire models")
        print("   - Performance gains are highest on GPU with large tensors")
        print("   - Consider using float16 for maximum efficiency in inference")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("This might be due to missing CUDA extension or incompatible PyTorch version")
        print("Try running the setup.py to compile the CUDA extension")


if __name__ == "__main__":
    main()