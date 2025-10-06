"""
🔬 W8A8 GEMM BENCHMARKING & VALIDATION SUITE

Comprehensive testing, benchmarking, and performance analysis.
"""

import torch
import triton
import time
from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass
# import matplotlib.pyplot as plt


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    shape: Tuple[int, int, int]  # M, N, K
    time_ms: float
    tflops: float
    bandwidth_gb_s: float
    kernel_type: str
    config: Dict


class W8A8Benchmarker:
    """
    Comprehensive benchmarking suite for W8A8 GEMM kernels.
    """
    
    def __init__(self, device='cuda', warmup_iters=10, bench_iters=100):
        self.device = device
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.results = []
    
    def benchmark_kernel(
        self,
        gemm_func,
        M: int,
        N: int,
        K: int,
        a_block_size: int = 128,
        w_block_size: int = 128,
    ) -> BenchmarkResult:
        """
        Benchmark a single kernel configuration.
        """
        # Create test data
        a = torch.randn(M, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
        w = torch.randn(N, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
        
        a_scale = torch.ones(M, (K + 127) // 128, device=self.device, dtype=torch.float32)
        w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device, dtype=torch.float32)
        
        # Warmup
        for _ in range(self.warmup_iters):
            _ = gemm_func(a, a_scale, w, w_scale, a_block_size, w_block_size, w_block_size)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.bench_iters):
            _ = gemm_func(a, a_scale, w, w_scale, a_block_size, w_block_size, w_block_size)
        end_event.record()
        
        torch.cuda.synchronize()
        
        # Calculate metrics
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / self.bench_iters
        
        # FLOPS calculation: 2*M*N*K (multiply-add)
        flops = 2 * M * N * K
        tflops = (flops / avg_time_ms / 1e9)  # TFLOPS
        
        # Bandwidth calculation
        # Read: M*K (activation) + N*K (weights) + scales
        # Write: M*N (output)
        bytes_read = M * K + N * K + M * ((K + 127) // 128) * 4 + ((N + 127) // 128) * ((K + 127) // 128) * 4
        bytes_write = M * N * 2  # BF16 output
        total_bytes = bytes_read + bytes_write
        bandwidth_gb_s = (total_bytes / avg_time_ms / 1e6)  # GB/s
        
        result = BenchmarkResult(
            shape=(M, N, K),
            time_ms=avg_time_ms,
            tflops=tflops,
            bandwidth_gb_s=bandwidth_gb_s,
            kernel_type="custom",
            config={},
        )
        
        self.results.append(result)
        return result
    
    def sweep_shapes(
        self,
        gemm_func,
        test_shapes: List[Tuple[int, int, int]],
    ) -> List[BenchmarkResult]:
        """
        Sweep through multiple problem sizes.
        """
        results = []
        
        print("=" * 80)
        print("PERFORMANCE SWEEP")
        print("=" * 80)
        print(f"{'M':>6} {'N':>6} {'K':>6} {'Time(ms)':>10} {'TFLOPS':>10} {'BW(GB/s)':>10}")
        print("-" * 80)
        
        for M, N, K in test_shapes:
            result = self.benchmark_kernel(gemm_func, M, N, K)
            results.append(result)
            
            print(f"{M:>6} {N:>6} {K:>6} {result.time_ms:>10.4f} {result.tflops:>10.2f} {result.bandwidth_gb_s:>10.2f}")
        
        print("=" * 80)
        return results
    
    def compare_with_torch(
        self,
        gemm_func,
        M: int,
        N: int,
        K: int,
    ):
        """
        Compare custom kernel with torch.matmul baseline.
        """
        print(f"\n🔥 COMPARISON: Custom vs PyTorch (M={M}, N={N}, K={K})")
        print("=" * 80)
        
        # Create test data
        a_fp16 = torch.randn(M, K, device=self.device, dtype=torch.float16)
        w_fp16 = torch.randn(N, K, device=self.device, dtype=torch.float16)
        
        a_fp8 = a_fp16.to(torch.float8_e4m3fn)
        w_fp8 = w_fp16.to(torch.float8_e4m3fn)
        
        a_scale = torch.ones(M, (K + 127) // 128, device=self.device, dtype=torch.float32)
        w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device, dtype=torch.float32)
        
        # Benchmark PyTorch (FP16)
        torch.cuda.synchronize()
        torch_times = []
        for _ in range(self.bench_iters):
            start = time.perf_counter()
            _ = torch.matmul(a_fp16, w_fp16.t())
            torch.cuda.synchronize()
            torch_times.append((time.perf_counter() - start) * 1000)
        torch_time_ms = np.median(torch_times)
        
        # Benchmark custom kernel (FP8)
        result = self.benchmark_kernel(gemm_func, M, N, K)
        
        # Calculate speedup
        speedup = torch_time_ms / result.time_ms
        
        print(f"PyTorch FP16:     {torch_time_ms:.4f} ms")
        print(f"Custom FP8:       {result.time_ms:.4f} ms")
        print(f"Speedup:          {speedup:.2f}x")
        print(f"TFLOPS:           {result.tflops:.2f}")
        print("=" * 80)
        
        return speedup
    
    def profile_jit_overhead(self, gemm_func, M: int, N: int, K: int, num_calls: int = 50):
        """
        Measure JIT compilation overhead.
        """
        print(f"\n⚡ JIT OVERHEAD ANALYSIS (M={M}, N={N}, K={K})")
        print("=" * 80)
        
        # Create test data
        a = torch.randn(M, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
        w = torch.randn(N, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
        a_scale = torch.ones(M, (K + 127) // 128, device=self.device, dtype=torch.float32)
        w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device, dtype=torch.float32)
        
        times = []
        
        for i in range(num_calls):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = gemm_func(a, a_scale, w, w_scale, 128, 128, 128)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        times = np.array(times)
        
        print(f"First call (with JIT):  {times[0]:.4f} ms")
        print(f"Median (after warmup):  {np.median(times[5:]):.4f} ms")
        print(f"Min:                    {np.min(times[5:]):.4f} ms")
        print(f"Max:                    {np.max(times[5:]):.4f} ms")
        print(f"Std:                    {np.std(times[5:]):.4f} ms")
        print(f"JIT overhead:           {times[0] - np.median(times[5:]):.4f} ms")
        print("=" * 80)
        
        return times


class W8A8Validator:
    """
    Numerical validation suite.
    """
    
    @staticmethod
    def validate_correctness(
        gemm_func,
        M: int = 16,
        N: int = 128,
        K: int = 256,
        rtol: float = 1e-1,
        atol: float = 1e-2,
    ) -> bool:
        """
        Validate kernel correctness against reference implementation.
        """
        print(f"\n✅ CORRECTNESS VALIDATION (M={M}, N={N}, K={K})")
        print("=" * 80)
        
        device = 'cuda'
        
        # Create test data with known values
        torch.manual_seed(42)
        a_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
        w_fp16 = torch.randn(N, K, device=device, dtype=torch.float16)
        
        # Convert to FP8
        a_fp8 = a_fp16.to(torch.float8_e4m3fn)
        w_fp8 = w_fp16.to(torch.float8_e4m3fn)
        
        # Simple per-tensor scales for testing
        a_scale = torch.ones(M, (K + 127) // 128, device=device, dtype=torch.float32) * 0.5
        w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32) * 0.5
        
        # Reference: Manual FP16 matmul with scaling
        a_dequant = a_fp8.to(torch.float16)
        w_dequant = w_fp8.to(torch.float16)
        
        # Apply scales (simplified for validation)
        reference = torch.matmul(a_dequant, w_dequant.t()) * 0.25  # 0.5 * 0.5
        
        # Custom kernel
        custom = gemm_func(a_fp8, a_scale, w_fp8, w_scale, 128, 128, 128)
        
        # Compare
        max_diff = torch.max(torch.abs(reference.to(torch.bfloat16) - custom)).item()
        mean_diff = torch.mean(torch.abs(reference.to(torch.bfloat16) - custom)).item()
        rel_error = mean_diff / (torch.mean(torch.abs(reference)).item() + 1e-8)
        
        print(f"Max absolute error:   {max_diff:.6f}")
        print(f"Mean absolute error:  {mean_diff:.6f}")
        print(f"Relative error:       {rel_error:.6f}")
        
        passed = rel_error < rtol and max_diff < atol
        
        if passed:
            print("✅ PASSED: Kernel is numerically correct!")
        else:
            print("❌ FAILED: Numerical errors exceed tolerance!")
        
        print("=" * 80)
        return passed
    
    @staticmethod
    def validate_shape_boundaries(gemm_func):
        """
        Test edge cases and boundary conditions.
        """
        print("\n🔍 BOUNDARY CONDITION TESTING")
        print("=" * 80)
        
        test_cases = [
            (1, 128, 256, "Single row (M=1)"),
            (7, 128, 256, "Odd M"),
            (16, 117, 256, "Non-aligned N"),
            (16, 128, 255, "Non-aligned K"),
            (16, 128, 128, "Minimal square"),
            (32, 8192, 11008, "Large N, K"),
        ]
        
        all_passed = True
        
        for M, N, K, desc in test_cases:
            try:
                a = torch.randn(M, K, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
                w = torch.randn(N, K, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
                a_scale = torch.ones(M, (K + 127) // 128, device='cuda', dtype=torch.float32)
                w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device='cuda', dtype=torch.float32)
                
                result = gemm_func(a, a_scale, w, w_scale, 128, 128, 128)
                
                assert result.shape == (M, N), f"Shape mismatch: expected {(M, N)}, got {result.shape}"
                assert not torch.isnan(result).any(), "NaN values detected"
                assert not torch.isinf(result).any(), "Inf values detected"
                
                print(f"✅ PASSED: {desc:30s} ({M}x{N}x{K})")
            
            except Exception as e:
                print(f"❌ FAILED: {desc:30s} ({M}x{N}x{K}) - {str(e)}")
                all_passed = False
        
        print("=" * 80)
        return all_passed


# ==================== COMPREHENSIVE TEST SUITE ====================

def run_full_benchmark_suite(gemm_func):
    """
    Run complete benchmarking and validation suite.
    """
    print("\n" + "=" * 80)
    print("🚀 W8A8 GEMM COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    
    # 1. Validation
    validator = W8A8Validator()
    validator.validate_correctness(gemm_func)
    validator.validate_shape_boundaries(gemm_func)
    
    # 2. Benchmarking
    benchmarker = W8A8Benchmarker(warmup_iters=10, bench_iters=100)
    
    # Common LLM shapes (focused on small M)
    llm_shapes = [
        # Token generation (M=1)
        (1, 4096, 4096),
        (1, 4096, 11008),
        (1, 11008, 4096),
        
        # Small batch inference (M=8-16)
        (8, 4096, 4096),
        (8, 4096, 11008),
        (16, 4096, 4096),
        (16, 4096, 11008),
        
        # Medium batch (M=32)
        (32, 4096, 4096),
        (32, 4096, 11008),
        
        # Larger batches
        (64, 4096, 4096),
        (128, 4096, 11008),
    ]
    
    results = benchmarker.sweep_shapes(gemm_func, llm_shapes)
    
    # 3. PyTorch comparison
    for M in [1, 8, 16, 32]:
        benchmarker.compare_with_torch(gemm_func, M, 4096, 11008)
    
    # 4. JIT overhead analysis
    benchmarker.profile_jit_overhead(gemm_func, 16, 4096, 11008)
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    best_tflops = max(results, key=lambda r: r.tflops)
    print(f"Peak TFLOPS:        {best_tflops.tflops:.2f} at shape {best_tflops.shape}")
    
    best_bw = max(results, key=lambda r: r.bandwidth_gb_s)
    print(f"Peak Bandwidth:     {best_bw.bandwidth_gb_s:.2f} GB/s at shape {best_bw.shape}")
    
    # Find best small-M performance
    small_m_results = [r for r in results if r.shape[0] <= 16]
    if small_m_results:
        best_small_m = max(small_m_results, key=lambda r: r.tflops)
        print(f"Best Small-M:       {best_small_m.tflops:.2f} TFLOPS at shape {best_small_m.shape}")
    
    print("=" * 80)


# ==================== ROOFLINE ANALYSIS ====================

def roofline_analysis(results: List[BenchmarkResult], peak_tflops: float, peak_bw_gb_s: float):
    """
    Generate roofline plot to visualize performance vs. hardware limits.
    """
    print("\n📈 ROOFLINE ANALYSIS")
    
    arithmetic_intensities = []
    achieved_tflops = []
    
    for result in results:
        M, N, K = result.shape
        
        # Arithmetic intensity = FLOPS / Bytes
        flops = 2 * M * N * K
        bytes_accessed = M * K + N * K + M * N * 2  # Simplified
        intensity = flops / bytes_accessed
        
        arithmetic_intensities.append(intensity)
        achieved_tflops.append(result.tflops)
    
    # Sort by intensity
    sorted_pairs = sorted(zip(arithmetic_intensities, achieved_tflops))
    intensities, tflops = zip(*sorted_pairs)
    
    # Roofline
    x = np.logspace(-1, 3, 100)
    roofline = np.minimum(peak_tflops, peak_bw_gb_s * x / 1000)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(x, roofline, 'k--', label='Roofline', linewidth=2)
    plt.loglog(intensities, tflops, 'ro-', label='Achieved', markersize=8)
    plt.axhline(y=peak_tflops, color='g', linestyle=':', label=f'Peak Compute ({peak_tflops} TFLOPS)')
    plt.xlabel('Arithmetic Intensity (FLOPS/Byte)')
    plt.ylabel('Performance (TFLOPS)')
    plt.title('W8A8 GEMM Roofline Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('w8a8_roofline.png', dpi=150)
    print("✅ Saved roofline plot to 'w8a8_roofline.png'")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Import your optimized GEMM function
    # from extreme_w8a8_gemm import w8a8_gemm_dispatch
	from w8a8_gemm_no_persistent_cta import w8a8_gemm_dispatch
    
    # # For demonstration, use a placeholder
    # def dummy_gemm_func(a, a_scale, w, w_scale, *args):
    #     # Placeholder - replace with actual kernel
    #     return torch.matmul(a.to(torch.float16), w.t().to(torch.float16)).to(torch.bfloat16)
    
    # Run comprehensive benchmark
	run_full_benchmark_suite(w8a8_gemm_dispatch)
    
    # print("""
    # ╔════════════════════════════════════════════════════════════════╗
    # ║  🔥 W8A8 GEMM BENCHMARKING SUITE READY                         ║
    # ║                                                                ║
    # ║  Usage:                                                        ║
    # ║    from benchmark_validation import run_full_benchmark_suite  ║
    # ║    run_full_benchmark_suite(your_gemm_function)               ║
    # ║                                                                ║
    # ║  Features:                                                     ║
    # ║    ✓ Numerical validation                                     ║
    # ║    ✓ Performance benchmarking                                 ║
    # ║    ✓ PyTorch comparison                                       ║
    # ║    ✓ JIT overhead analysis                                    ║
    # ║    ✓ Boundary testing                                         ║
    # ║    ✓ Roofline analysis                                        ║
    # ╚════════════════════════════════════════════════════════════════╝
    # """)