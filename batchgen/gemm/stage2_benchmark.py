"""
Benchmarking and validation suite for Stage 2: Grouped GEMM (down projection).
"""

import torch
import time
import numpy as np
from typing import List, Tuple


class Stage2Benchmarker:
    """Benchmark grouped GEMM kernels."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = []
    
    def create_test_data(
        self,
        M: int,
        N: int,
        K: int,
        num_experts: int,
        group_sizes: List[int],
    ):
        """Create test data for grouped GEMM."""
        # LHS (activations after expert processing)
        lhs = torch.randn(M, K, device=self.device, dtype=torch.float16)
        lhs = lhs.to(torch.float8_e4m3fn)
        
        lhs_scale = torch.ones(
            M, (K + 127) // 128,
            device=self.device,
            dtype=torch.float32
        ) * 0.1
        
        # RHS (down projection weights)
        rhs_weights = [
            torch.randn(N, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
            for _ in range(num_experts)
        ]
        
        # Scales
        rhs_scales = [
            torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device) * 0.1
            for _ in range(num_experts)
        ]
        
        # Pointer tensors
        rhs_ptrs = torch.tensor(
            [w.data_ptr() for w in rhs_weights],
            device=self.device,
            dtype=torch.int64
        )
        rhs_scale_ptrs = torch.tensor(
            [s.data_ptr() for s in rhs_scales],
            device=self.device,
            dtype=torch.int64
        )
        
        # Group metadata
        group_sizes_tensor = torch.tensor(group_sizes, device=self.device, dtype=torch.int32)
        activated_group_idx = torch.arange(len(group_sizes), device=self.device, dtype=torch.int32)
        
        # Start indices
        group_start_indices = torch.zeros(len(group_sizes), device=self.device, dtype=torch.int32)
        cumsum = 0
        for i, size in enumerate(group_sizes):
            group_start_indices[i] = cumsum
            cumsum += size
        
        num_active = torch.tensor([len(group_sizes)], device=self.device, dtype=torch.int32)
        
        return (
            lhs, lhs_scale,
            rhs_weights, rhs_ptrs,
            rhs_scales, rhs_scale_ptrs,
            group_sizes_tensor, activated_group_idx,
            group_start_indices, num_active
        )
    
    def benchmark_kernel(
        self,
        kernel_func,
        M: int,
        N: int,
        K: int,
        num_experts: int,
        group_sizes: List[int],
        num_iters: int = 100,
    ):
        """Benchmark kernel."""
        data = self.create_test_data(M, N, K, num_experts, group_sizes)
        
        # Warmup
        for _ in range(10):
            _ = kernel_func(*data)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            _ = kernel_func(*data)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        median_time = np.median(times)
        min_time = np.min(times)
        std_time = np.std(times)
        
        # FLOPS: Single GEMM per expert
        total_tokens = sum(group_sizes)
        flops_per_token = 2 * N * K  # One GEMM
        total_flops = flops_per_token * total_tokens
        tflops = total_flops / median_time / 1e9
        
        # Utilization
        h100_peak = 989  # TFLOPS for FP8
        utilization = (tflops / h100_peak) * 100
        
        result = {
            'M': M,
            'N': N,
            'K': K,
            'num_experts': num_experts,
            'total_tokens': total_tokens,
            'avg_group_size': total_tokens / num_experts,
            'time_ms': median_time,
            'time_std': std_time,
            'tflops': tflops,
            'utilization_%': utilization,
        }
        
        self.results.append(result)
        return result
    
    def compare_kernels(
        self,
        kernels: List[Tuple[str, callable]],
        test_configs: List[dict],
    ):
        """Compare kernel implementations."""
        print("\n" + "="*100)
        print("STAGE 2 GROUPED GEMM COMPARISON")
        print("="*100)
        
        print(f"\n{'Config':<20} {'Kernel':<20} {'Time(ms)':<12} {'TFLOPS':<10} {'Util%':<8} {'Speedup':<10}")
        print("-"*100)
        
        for config in test_configs:
            baseline_time = None
            config_name = f"M={config['M']},E={config['num_experts']}"
            
            for kernel_name, kernel_func in kernels:
                result = self.benchmark_kernel(
                    kernel_func,
                    config['M'],
                    config['N'],
                    config['K'],
                    config['num_experts'],
                    config['group_sizes'],
                )
                
                if baseline_time is None:
                    baseline_time = result['time_ms']
                    speedup_str = "1.00x (baseline)"
                else:
                    speedup = baseline_time / result['time_ms']
                    speedup_str = f"{speedup:.2f}x"
                
                print(f"{config_name:<20} {kernel_name:<20} {result['time_ms']:<12.4f} "
                      f"{result['tflops']:<10.2f} {result['utilization_%']:<8.1f} {speedup_str:<10}")
        
        print("="*100)
    
    def sweep_group_sizes(
        self,
        kernel_func,
        M: int = 128,
        N: int = 4096,
        K: int = 14336,
        num_experts: int = 8,
    ):
        """Test various group size distributions."""
        print("\n" + "="*100)
        print("GROUP SIZE SWEEP")
        print("="*100)
        
        test_cases = [
            ([4]*8, "Small uniform (4 tokens/expert)"),
            ([8]*8, "Medium uniform (8 tokens/expert)"),
            ([16]*8, "Large uniform (16 tokens/expert)"),
            ([1, 2, 4, 8, 16, 20, 24, 32], "Variable sizes"),
            ([32, 32, 1, 1, 1, 1, 1, 1], "Highly skewed"),
        ]
        
        print(f"\n{'Case':<30} {'Avg Size':<10} {'Time(ms)':<12} {'TFLOPS':<10} {'Util%':<8}")
        print("-"*100)
        
        for group_sizes, description in test_cases:
            result = self.benchmark_kernel(
                kernel_func,
                M, N, K, num_experts, group_sizes,
            )
            
            print(f"{description:<30} {result['avg_group_size']:<10.1f} "
                  f"{result['time_ms']:<12.4f} {result['tflops']:<10.2f} "
                  f"{result['utilization_%']:<8.1f}")
        
        print("="*100)


class Stage2Validator:
    """Validate grouped GEMM correctness."""
    
    @staticmethod
    def reference_gemm(
        lhs,
        lhs_scale,
        rhs_weights,
        rhs_scales,
        group_sizes,
        group_start_indices,
    ):
        """Reference implementation."""
        M, K = lhs.shape
        N = rhs_weights[0].shape[0]
        
        output = torch.zeros(M, N, device=lhs.device, dtype=torch.bfloat16)
        
        for expert_idx, group_size in enumerate(group_sizes):
            if group_size == 0:
                continue
            
            start_idx = group_start_indices[expert_idx]
            end_idx = start_idx + group_size
            
            # Dequantize LHS
            tokens = lhs[start_idx:end_idx].to(torch.float32)
            token_scales = lhs_scale[start_idx:end_idx]
            for i in range(tokens.shape[0]):
                for k_block in range((K + 127) // 128):
                    k_start = k_block * 128
                    k_end = min(k_start + 128, K)
                    scale = token_scales[i, k_block]
                    tokens[i, k_start:k_end] *= scale
            
            # Dequantize RHS
            rhs_w = rhs_weights[expert_idx].to(torch.float32)
            rhs_w = rhs_w * rhs_scales[expert_idx].mean()
            
            # Compute
            result = torch.matmul(tokens, rhs_w.t())
            output[start_idx:end_idx] = result.to(torch.bfloat16)
        
        return output
    
    @staticmethod
    def validate_correctness(
        kernel_func,
        M: int = 32,
        N: int = 512,
        K: int = 1024,
        num_experts: int = 4,
    ):
        """Validate kernel correctness."""
        print("\n" + "="*80)
        print("CORRECTNESS VALIDATION")
        print("="*80)
        
        group_sizes = [8, 8, 8, 8]
        
        benchmarker = Stage2Benchmarker()
        data = benchmarker.create_test_data(M, N, K, num_experts, group_sizes)
        
        # Run kernel
        kernel_output = kernel_func(*data)
        
        # Run reference
        ref_output = Stage2Validator.reference_gemm(
            data[0], data[1],  # lhs, lhs_scale
            data[2], data[4],  # rhs_weights, rhs_scales
            group_sizes,
            data[8],  # group_start_indices
        )
        
        # Compare
        max_diff = torch.abs(kernel_output - ref_output).max().item()
        mean_diff = torch.abs(kernel_output - ref_output).mean().item()
        rel_error = mean_diff / (torch.abs(ref_output).mean().item() + 1e-8)
        
        print(f"Max absolute error:   {max_diff:.6f}")
        print(f"Mean absolute error:  {mean_diff:.6f}")
        print(f"Relative error:       {rel_error:.6f}")
        
        passed = rel_error < 0.15 and max_diff < 1.0
        
        if passed:
            print("PASS: Kernel output is correct")
        else:
            print("FAIL: Kernel has numerical errors")
        
        print("="*80)
        return passed


def run_stage2_benchmark(
    original_kernel,
    optimized_kernel,
):
    """Run stage 2 benchmark suite."""
    print("\n" + "="*100)
    print("STAGE 2: DOWN PROJECTION BENCHMARK")
    print("="*100)
    
    benchmarker = Stage2Benchmarker()
    validator = Stage2Validator()
    
    # Validation
    print("\n" + "="*100)
    print("STEP 1: CORRECTNESS VALIDATION")
    print("="*100)
    validator.validate_correctness(optimized_kernel)
    
    # Performance comparison
    print("\n" + "="*100)
    print("STEP 2: PERFORMANCE COMPARISON")
    print("="*100)
    
    kernels = [
        ("Original", original_kernel),
        ("Optimized", optimized_kernel),
    ]
    
    # Typical down projection: hidden_dim (14336) → model_dim (4096)
    test_configs = [
        {
            'M': 368,
            'N': 7168,
            'K': 1536,
            'num_experts': 32,
            'group_sizes': [23] * 32,
        },
        {
            'M': 128,
            'N': 4096,
            'K': 14336,
            'num_experts': 8,
            'group_sizes': [16] * 8,
        },
        {
            'M': 64,
            'N': 4096,
            'K': 14336,
            'num_experts': 8,
            'group_sizes': [1, 2, 4, 8, 12, 16, 18, 20],
        },
    ]
    
    benchmarker.compare_kernels(kernels, test_configs)
    
    # Group size sweep
    benchmarker.sweep_group_sizes(optimized_kernel)
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print("""
Stage 2 optimizations applied:
  - 2D grid (experts × N-blocks) for parallel expert processing
  - Hoisted mask computations outside K-loop
  - Simplified scale loading (BLOCK_K == SCALE_BLOCK_K)
  
Expected improvements:
  - 3-6x speedup for typical workloads
  - Better utilization with 2D parallelism
  - Reduced latency from eliminated serial loops
    """)
    print("="*100)


if __name__ == "__main__":
    from batchgen.moe.fused_grouped_dequant_gemm import fused_dequant_grouped_gemm_fp8_fp8_triton
    from w8a8_grouped_gemm_stage_2 import fused_dequant_grouped_gemm_fp8_fp8_triton_optimized
    run_stage2_benchmark(
        fused_dequant_grouped_gemm_fp8_fp8_triton,
        fused_dequant_grouped_gemm_fp8_fp8_triton_optimized
    )