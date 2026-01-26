"""Benchmark script for Decoupled MXFP4 MoE.

Compares three approaches:
1. Fused MXFP4 Grouped GEMM (existing) - inline dequantization
2. Decoupled Dequant + BF16 Grouped GEMM (new) - separate kernels
3. Unfused Baseline - dequantize once + torch.mm per expert (reference)

Usage:
    # Quick benchmark with default config
    python bench_decoupled_mxfp4_moe.py --quick --tokens 4

    # Full benchmark with multiple token counts
    python bench_decoupled_mxfp4_moe.py --tokens 1 4 8 16 32

    # Detailed component timing
    python bench_decoupled_mxfp4_moe.py --quick --tokens 4 --detailed
"""

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch

# Add batchgen to path
sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")

# GPT-OSS-120B dimensions
DEFAULT_CONFIG = {
    "num_experts": 128,
    "hidden_size": 5120,
    "intermediate_size": 13824,
    "num_experts_per_tok": 8,
}


def create_test_tensors(
    num_experts: int,
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Create test tensors for benchmarking.

    Returns:
        hidden_3d: [num_experts, tokens_per_expert, hidden_size] BF16
        weights: List of [N, K//2] uint8 (one per expert)
        scales: List of [N, K//32] uint8 (one per expert)
    """
    K = hidden_size
    N = intermediate_size

    hidden_3d = torch.randn(
        num_experts, tokens_per_expert, K,
        dtype=torch.bfloat16, device=device
    )

    weights = [
        torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        for _ in range(num_experts)
    ]
    scales = [
        torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
        for _ in range(num_experts)
    ]

    return hidden_3d, weights, scales


def benchmark_fused_mxfp4_gemm(
    hidden_3d: torch.Tensor,
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark existing fused MXFP4 grouped GEMM.

    Returns time in milliseconds.
    """
    try:
        from batchgen.moe.mxfp4_grouped_gemm import (
            setup_expert_weight_pointers,
            grouped_mxfp4_gemm_3d,
        )
    except ImportError as e:
        print(f"  Fused MXFP4 GEMM not available: {e}")
        return float('inf')

    num_experts = hidden_3d.shape[0]
    tokens_per_expert = hidden_3d.shape[1]

    # Setup pointer arrays
    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=hidden_3d.device
    )

    try:
        # Warmup
        for _ in range(warmup_iters):
            _ = grouped_mxfp4_gemm_3d(
                hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
                N, weights[0], scales[0]
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = grouped_mxfp4_gemm_3d(
                hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
                N, weights[0], scales[0]
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        return elapsed
    except Exception as e:
        print(f"  Fused MXFP4 GEMM failed: {e}")
        return float('inf')


def benchmark_dequant_only(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    K: int,
    num_experts: int,
    device: str = "cuda",
    warmup_iters: int = 3,
    bench_iters: int = 10,
    version: str = "v2_e2m1",
) -> float:
    """Benchmark ONLY the batch dequant kernel (isolated test).

    Args:
        weights: List of packed FP4 weights [N, K//2]
        scales: List of scales [N, K//32]
        N: Output dimension
        K: Input dimension
        num_experts: Number of experts
        device: CUDA device
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations
        version: FP4 decode version (v1_sequential, v2_e2m1, v3_binary_tree, v4_branchless)

    Returns time in milliseconds, or float('inf') on failure.
    """
    try:
        from batchgen.moe.decoupled_mxfp4_moe import batch_mxfp4_dequant
    except ImportError as e:
        print(f"  batch_mxfp4_dequant not available: {e}")
        return float('inf')

    # Setup pointer arrays
    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weights], dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scales], dtype=torch.int64, device=device
    )

    # Allocate BF16 buffer
    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    try:
        print(f"    Testing dequant kernel ({version}) in isolation...")

        # Warmup with explicit sync and error checking
        for i in range(warmup_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scales[0], version=version)
            torch.cuda.synchronize()
            print(f"      Warmup {i+1}/{warmup_iters} OK")

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scales[0], version=version)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        print(f"    Dequant kernel ({version}) passed! Time: {elapsed:.3f} ms")
        return elapsed
    except Exception as e:
        print(f"    Dequant kernel ({version}) FAILED: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


def benchmark_all_dequant_versions(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    K: int,
    num_experts: int,
    device: str = "cuda",
    warmup_iters: int = 3,
    bench_iters: int = 10,
    print_results: bool = True,
) -> Dict[str, float]:
    """Benchmark all FP4 decode versions for comparison.

    Args:
        weights: List of packed FP4 weights
        scales: List of scales
        N: Output dimension
        K: Input dimension
        num_experts: Number of experts
        device: CUDA device
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations
        print_results: Whether to print comparison table

    Returns:
        Dictionary mapping version name to time in milliseconds.
    """
    try:
        from batchgen.moe.decoupled_mxfp4_moe import (
            batch_mxfp4_dequant,
            FP4_DECODE_VERSIONS,
        )
    except ImportError as e:
        print(f"  Cannot import FP4_DECODE_VERSIONS: {e}")
        return {}

    # Setup pointer arrays
    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weights], dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scales], dtype=torch.int64, device=device
    )

    # Allocate BF16 buffer
    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    results = {}
    version_notes = {
        "v1_sequential": "16 tl.where() - baseline",
        "v2_e2m1": "E2M1 arithmetic (5-6 where)",
        "v3_binary_tree": "Binary tree (4 where)",
        "v4_branchless": "IEEE bitcast (2 where)",
    }

    print(f"\n{'='*70}")
    print("FP4 DECODE VERSION COMPARISON")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {num_experts} experts, N={N}, K={K}")
    print(f"Output size: {num_experts * N * K * 2 / 1e9:.2f} GB BF16")
    print(f"{'='*70}")

    for version in FP4_DECODE_VERSIONS:
        try:
            # Warmup (compile kernel)
            for _ in range(warmup_iters):
                batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scales[0], version=version)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(bench_iters):
                batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scales[0], version=version)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / bench_iters * 1000

            results[version] = elapsed
            print(f"  {version}: {elapsed:.3f} ms")
        except Exception as e:
            print(f"  {version}: FAILED - {e}")
            results[version] = float('inf')

    # Print comparison table
    if print_results and results:
        # Find best
        valid_results = {k: v for k, v in results.items() if v != float('inf')}
        if valid_results:
            best_version = min(valid_results, key=valid_results.get)
            best_time = valid_results[best_version]
            baseline_time = results.get("v1_sequential", results[best_version])

            print(f"\n{'Version':<20} {'Time (ms)':<12} {'vs Baseline':<12} {'Notes'}")
            print(f"{'-'*70}")

            for version in FP4_DECODE_VERSIONS:
                time_ms = results.get(version, float('inf'))
                if time_ms != float('inf'):
                    speedup = baseline_time / time_ms if baseline_time != float('inf') else 1.0
                    marker = " <-- BEST" if version == best_version else ""
                    notes = version_notes.get(version, "")
                    print(f"{version:<20} {time_ms:<12.3f} {speedup:<12.2f}x {notes}{marker}")
                else:
                    print(f"{version:<20} {'FAILED':<12} {'-':<12} {version_notes.get(version, '')}")

            print(f"{'='*70}")
            print(f"Best: {best_version} at {best_time:.3f} ms")
            if baseline_time != float('inf'):
                print(f"Speedup over baseline (v1_sequential): {baseline_time / best_time:.2f}x")
            print(f"{'='*70}")

    return results


def benchmark_cuda_dequant_only(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    K: int,
    num_experts: int,
    device: str = "cuda",
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> Dict[str, float]:
    """Benchmark CUDA dequant kernels with shared memory LUT.

    Returns dict with timing for each kernel version, or empty dict on failure.
    """
    try:
        from batchgen.moe.cuda_mxfp4_dequant import batch_mxfp4_dequant_cuda
    except ImportError as e:
        print(f"  CUDA batch_mxfp4_dequant not available: {e}")
        return {}

    # Setup pointer arrays
    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weights], dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scales], dtype=torch.int64, device=device
    )

    # Allocate BF16 buffer
    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    results = {}
    kernel_names = ['basic', 'vec4', 'coalesced']

    print("    Testing CUDA dequant kernels...")

    for version, name in enumerate(kernel_names):
        try:
            # Warmup
            for i in range(warmup_iters):
                batch_mxfp4_dequant_cuda(
                    weight_ptrs, scale_ptrs, bf16_buffer,
                    weights[0], scales[0], kernel_version=version
                )
                torch.cuda.synchronize()

            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(bench_iters):
                batch_mxfp4_dequant_cuda(
                    weight_ptrs, scale_ptrs, bf16_buffer,
                    weights[0], scales[0], kernel_version=version
                )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / bench_iters * 1000

            results[name] = elapsed
            print(f"      CUDA {name}: {elapsed:.3f} ms")
        except Exception as e:
            print(f"      CUDA {name} FAILED: {e}")
            results[name] = float('inf')

    return results


def benchmark_gemm_only(
    hidden_3d: torch.Tensor,
    N: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark ONLY the BF16 grouped GEMM kernel (isolated test).

    Uses random BF16 buffer instead of dequantized weights.
    Returns time in milliseconds, or float('inf') on failure.
    """
    try:
        from batchgen.moe.decoupled_mxfp4_moe import bf16_grouped_gemm_3d
    except ImportError as e:
        print(f"  bf16_grouped_gemm_3d not available: {e}")
        return float('inf')

    num_experts = hidden_3d.shape[0]
    tokens_per_expert = hidden_3d.shape[1]
    K = hidden_3d.shape[2]
    device = hidden_3d.device

    # Create random BF16 buffer (skip dequant)
    bf16_buffer = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=device
    )

    try:
        print("    Testing GEMM kernel in isolation...")

        # Warmup with explicit sync and error checking
        for i in range(warmup_iters):
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
            torch.cuda.synchronize()
            print(f"      Warmup {i+1}/{warmup_iters} OK")

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        print(f"    GEMM kernel passed! Time: {elapsed:.3f} ms")
        return elapsed
    except Exception as e:
        print(f"    GEMM kernel FAILED: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


def benchmark_decoupled(
    hidden_3d: torch.Tensor,
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
    detailed: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Benchmark decoupled dequant + BF16 grouped GEMM.

    Returns:
        total_time_ms: Total time in milliseconds
        component_times: Dict with "dequant" and "gemm" times (if detailed)
    """
    try:
        from batchgen.moe.decoupled_mxfp4_moe import (
            batch_mxfp4_dequant,
            bf16_grouped_gemm_3d,
        )
    except ImportError as e:
        print(f"  Decoupled MoE not available: {e}")
        return float('inf'), {}

    num_experts = hidden_3d.shape[0]
    tokens_per_expert = hidden_3d.shape[1]
    K = hidden_3d.shape[2]
    device = hidden_3d.device

    # Setup pointer arrays
    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weights], dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scales], dtype=torch.int64, device=device
    )

    # Allocate BF16 buffer
    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=device
    )

    component_times = {}

    try:
        # Warmup
        for _ in range(warmup_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scales[0])
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
        torch.cuda.synchronize()

        if detailed:
            # Benchmark dequant separately
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(bench_iters):
                batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scales[0])
            torch.cuda.synchronize()
            component_times["dequant"] = (time.perf_counter() - start) / bench_iters * 1000

            # Benchmark GEMM separately
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(bench_iters):
                _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
            torch.cuda.synchronize()
            component_times["gemm"] = (time.perf_counter() - start) / bench_iters * 1000

        # Benchmark total
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scales[0])
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / bench_iters * 1000

        return total_time, component_times
    except Exception as e:
        print(f"  Decoupled benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), {}


def benchmark_unfused_baseline(
    hidden_3d: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    N: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark unfused baseline: dequantize + torch.mm per expert.

    NOTE: This uses the SAME weight for all experts (for fair memory comparison).
    In real MoE, each expert has different weights.

    Returns time in milliseconds.
    """
    try:
        from batchgen.quantization.mxfp4 import mxfp4_dequantize
    except ImportError as e:
        print(f"  mxfp4_dequantize not available: {e}")
        return float('inf')

    num_experts = hidden_3d.shape[0]

    try:
        # Warmup
        for _ in range(warmup_iters):
            weight_bf16 = mxfp4_dequantize(weight, scale, dtype=torch.bfloat16)
            for e in range(num_experts):
                _ = torch.mm(hidden_3d[e], weight_bf16.T)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            weight_bf16 = mxfp4_dequantize(weight, scale, dtype=torch.bfloat16)
            for e in range(num_experts):
                _ = torch.mm(hidden_3d[e], weight_bf16.T)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        return elapsed
    except Exception as e:
        print(f"  Unfused baseline failed: {e}")
        return float('inf')


def benchmark_unfused_per_expert(
    hidden_3d: torch.Tensor,
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark unfused with DIFFERENT weights per expert (fair comparison).

    This is the true cost of per-expert dequantization + GEMM.
    Returns time in milliseconds.
    """
    try:
        from batchgen.quantization.mxfp4 import mxfp4_dequantize
    except ImportError as e:
        print(f"  mxfp4_dequantize not available: {e}")
        return float('inf')

    num_experts = hidden_3d.shape[0]

    try:
        # Warmup
        for _ in range(warmup_iters):
            for e in range(num_experts):
                weight_bf16 = mxfp4_dequantize(weights[e], scales[e], dtype=torch.bfloat16)
                _ = torch.mm(hidden_3d[e], weight_bf16.T)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            for e in range(num_experts):
                weight_bf16 = mxfp4_dequantize(weights[e], scales[e], dtype=torch.bfloat16)
                _ = torch.mm(hidden_3d[e], weight_bf16.T)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        return elapsed
    except Exception as e:
        print(f"  Unfused per-expert failed: {e}")
        return float('inf')


def compute_metrics(
    time_ms: float,
    num_experts: int,
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
) -> Dict[str, float]:
    """Compute performance metrics."""
    M = tokens_per_expert
    N = intermediate_size
    K = hidden_size

    # FLOPs for one GEMM: 2 * M * N * K
    flops_per_expert = 2 * M * N * K
    total_flops = flops_per_expert * num_experts

    # Data movement
    lhs_bytes = num_experts * M * K * 2  # BF16
    rhs_bytes = num_experts * N * K * 2  # Dequantized BF16
    output_bytes = num_experts * M * N * 2
    total_bytes = lhs_bytes + rhs_bytes + output_bytes

    time_s = time_ms / 1000.0
    tflops = total_flops / time_s / 1e12
    bandwidth_tb_s = total_bytes / time_s / 1e12

    return {
        "tflops": tflops,
        "bandwidth_tb_s": bandwidth_tb_s,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
    }


def run_benchmark(
    tokens_per_expert: int,
    config: Dict,
    detailed: bool = False,
    isolate_kernels: bool = True,
) -> Dict[str, float]:
    """Run full benchmark comparison.

    Args:
        tokens_per_expert: Number of tokens per expert
        config: Model configuration dict
        detailed: If True, show component timing breakdown
        isolate_kernels: If True, test each kernel in isolation first

    Returns dict with timing results.
    """
    device = "cuda"

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {tokens_per_expert} tokens/expert")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {config['num_experts']} experts, "
          f"hidden={config['hidden_size']}, intermediate={config['intermediate_size']}")

    # Create test data
    hidden_3d, weights, scales = create_test_tensors(
        config["num_experts"],
        tokens_per_expert,
        config["hidden_size"],
        config["intermediate_size"],
        device,
    )

    results = {}

    # Benchmark fused MXFP4 GEMM
    print("\nBenchmarking fused MXFP4 grouped GEMM...")
    fused_time = benchmark_fused_mxfp4_gemm(
        hidden_3d, weights, scales, config["intermediate_size"]
    )
    results["fused_mxfp4"] = fused_time
    print(f"  Time: {fused_time:.3f} ms")

    # Isolated kernel tests (helps debug which kernel is failing)
    if isolate_kernels:
        print("\n--- ISOLATED KERNEL TESTS ---")

        # Test dequant kernel alone (Triton E2M1)
        print("\n[1/3] Testing Triton E2M1 dequant kernel in isolation...")
        dequant_isolated = benchmark_dequant_only(
            weights, scales, config["intermediate_size"], config["hidden_size"],
            config["num_experts"], device
        )
        results["dequant_triton"] = dequant_isolated

        # Test CUDA dequant kernels with shared memory LUT
        print("\n[2/3] Testing CUDA shared memory LUT dequant kernels...")
        cuda_results = benchmark_cuda_dequant_only(
            weights, scales, config["intermediate_size"], config["hidden_size"],
            config["num_experts"], device
        )
        results["dequant_cuda_basic"] = cuda_results.get("basic", float('inf'))
        results["dequant_cuda_vec4"] = cuda_results.get("vec4", float('inf'))
        results["dequant_cuda_coalesced"] = cuda_results.get("coalesced", float('inf'))
        if cuda_results:
            results["dequant_cuda_best"] = min(cuda_results.values())
        else:
            results["dequant_cuda_best"] = float('inf')

        # Test GEMM kernel alone (with random BF16, no dequant)
        print("\n[3/3] Testing BF16 GEMM kernel in isolation...")
        gemm_isolated = benchmark_gemm_only(
            hidden_3d, config["intermediate_size"]
        )
        results["gemm_isolated"] = gemm_isolated

        if dequant_isolated == float('inf') or gemm_isolated == float('inf'):
            print("\n*** ISOLATED KERNEL TEST FAILED - Skipping combined benchmark ***")
            results["decoupled"] = float('inf')
            return results

        print("\n--- ISOLATED TESTS PASSED ---")

    # Benchmark decoupled (combined)
    print("\nBenchmarking decoupled (dequant + BF16 GEMM) combined...")
    decoupled_time, component_times = benchmark_decoupled(
        hidden_3d, weights, scales, config["intermediate_size"], detailed=detailed
    )
    results["decoupled"] = decoupled_time
    if component_times:
        results["decoupled_dequant"] = component_times.get("dequant", 0)
        results["decoupled_gemm"] = component_times.get("gemm", 0)
        print(f"  Dequant: {component_times['dequant']:.3f} ms")
        print(f"  GEMM:    {component_times['gemm']:.3f} ms")
    print(f"  Total:   {decoupled_time:.3f} ms")

    # Benchmark unfused baseline (single weight - unfair but fast)
    print("\nBenchmarking unfused baseline (single weight reused)...")
    unfused_single_time = benchmark_unfused_baseline(
        hidden_3d, weights[0], scales[0], config["intermediate_size"]
    )
    results["unfused_single"] = unfused_single_time
    print(f"  Time: {unfused_single_time:.3f} ms (NOTE: reuses single weight)")

    # Benchmark unfused per-expert (true cost)
    print("\nBenchmarking unfused per-expert (fair comparison)...")
    unfused_per_expert_time = benchmark_unfused_per_expert(
        hidden_3d, weights, scales, config["intermediate_size"]
    )
    results["unfused_per_expert"] = unfused_per_expert_time
    print(f"  Time: {unfused_per_expert_time:.3f} ms")

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Approach':<40} {'Time (ms)':>12} {'Speedup':>12}")
    print("-" * 70)

    baseline = results["fused_mxfp4"]
    for name, time_ms in [
        ("Fused MXFP4 Grouped GEMM", results["fused_mxfp4"]),
        ("Decoupled (dequant + BF16 GEMM)", results["decoupled"]),
        ("Unfused per-expert (fair)", results["unfused_per_expert"]),
        ("Unfused single weight*", results["unfused_single"]),
    ]:
        if time_ms != float('inf'):
            speedup = baseline / time_ms if time_ms > 0 else 0
            print(f"{name:<40} {time_ms:>12.3f} {speedup:>11.2f}x")
        else:
            print(f"{name:<40} {'N/A':>12} {'N/A':>12}")

    print("-" * 70)
    print("* Single weight reused for all experts - not a fair comparison")

    # Print dequant kernel comparison
    if isolate_kernels and "dequant_triton" in results:
        print(f"\n{'='*70}")
        print("DEQUANT KERNEL COMPARISON")
        print(f"{'='*70}")
        print(f"{'Kernel':<40} {'Time (ms)':>12} {'vs Triton':>12}")
        print("-" * 70)

        triton_time = results.get("dequant_triton", float('inf'))
        for name, key in [
            ("Triton E2M1 (5-6 tl.where)", "dequant_triton"),
            ("CUDA basic (shared mem LUT)", "dequant_cuda_basic"),
            ("CUDA vec4 (4 bytes/thread)", "dequant_cuda_vec4"),
            ("CUDA coalesced (best)", "dequant_cuda_coalesced"),
        ]:
            time_ms = results.get(key, float('inf'))
            if time_ms != float('inf') and triton_time != float('inf'):
                speedup = triton_time / time_ms if time_ms > 0 else 0
                print(f"{name:<40} {time_ms:>12.3f} {speedup:>11.2f}x")
            else:
                print(f"{name:<40} {'N/A':>12} {'N/A':>12}")

        print("-" * 70)
        cuda_best = results.get("dequant_cuda_best", float('inf'))
        gemm_time = results.get("gemm_isolated", float('inf'))
        if cuda_best != float('inf') and gemm_time != float('inf'):
            projected_total = cuda_best + gemm_time
            projected_speedup = baseline / projected_total if projected_total > 0 else 0
            print(f"\nProjected total with CUDA dequant: {projected_total:.3f} ms ({projected_speedup:.2f}x speedup)")

    # Compute metrics for decoupled
    if results["decoupled"] != float('inf'):
        metrics = compute_metrics(
            results["decoupled"],
            config["num_experts"],
            tokens_per_expert,
            config["hidden_size"],
            config["intermediate_size"],
        )
        print(f"\n--- Performance Metrics (Decoupled) ---")
        print(f"Effective TFLOPS:        {metrics['tflops']:>8.2f}")
        print(f"Effective bandwidth:     {metrics['bandwidth_tb_s']:>8.3f} TB/s")

    print(f"{'='*70}")

    return results


def test_gemm_only_minimal():
    """Minimal test of GEMM kernel only (no dequant) - for debugging."""
    print("\n" + "=" * 70)
    print("MINIMAL GEMM KERNEL TEST (no dequant)")
    print("=" * 70)

    try:
        from batchgen.moe.decoupled_mxfp4_moe import bf16_grouped_gemm_3d
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

    # Use smaller dimensions for minimal test
    num_experts = 4  # Start small
    tokens_per_expert = 4
    K = 128  # Must be divisible by BLOCK_K=32
    N = 128  # Must be divisible by BLOCK_N=64

    device = "cuda"
    print(f"Config: {num_experts} experts, {tokens_per_expert} tokens, K={K}, N={N}")

    # Create test tensors
    hidden_3d = torch.randn(num_experts, tokens_per_expert, K, dtype=torch.bfloat16, device=device)
    weight_buffer = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device)
    expert_counts = torch.full((num_experts,), tokens_per_expert, dtype=torch.int32, device=device)

    print("Testing GEMM kernel...")
    try:
        output = bf16_grouped_gemm_3d(hidden_3d, weight_buffer, expert_counts, N)
        torch.cuda.synchronize()
        print(f"SUCCESS! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Decoupled MXFP4 MoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick benchmark
    python bench_decoupled_mxfp4_moe.py --quick --tokens 4

    # Multiple token counts
    python bench_decoupled_mxfp4_moe.py --tokens 1 4 8 16 32

    # Detailed component timing
    python bench_decoupled_mxfp4_moe.py --quick --tokens 4 --detailed

    # Test GEMM kernel only (minimal, for debugging)
    python bench_decoupled_mxfp4_moe.py --test-gemm
"""
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick benchmark with single token count"
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[4],
        help="Tokens per expert to test (default: 4)"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Show detailed component timing (dequant vs GEMM)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=DEFAULT_CONFIG["num_experts"],
        help=f"Number of experts (default: {DEFAULT_CONFIG['num_experts']})"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=DEFAULT_CONFIG["hidden_size"],
        help=f"Hidden size (default: {DEFAULT_CONFIG['hidden_size']})"
    )
    parser.add_argument(
        "--intermediate-size", type=int, default=DEFAULT_CONFIG["intermediate_size"],
        help=f"Intermediate size (default: {DEFAULT_CONFIG['intermediate_size']})"
    )
    parser.add_argument(
        "--no-isolate", action="store_true",
        help="Skip isolated kernel tests (test dequant/GEMM separately)"
    )
    parser.add_argument(
        "--test-gemm", action="store_true",
        help="Run minimal GEMM kernel test only (for debugging)"
    )
    parser.add_argument(
        "--compare-versions", action="store_true",
        help="Compare all FP4 decode versions (v1_sequential, v2_e2m1, v3_binary_tree, v4_branchless)"
    )

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    # Minimal GEMM test mode
    if args.test_gemm:
        success = test_gemm_only_minimal()
        sys.exit(0 if success else 1)

    # Compare all FP4 decode versions
    if args.compare_versions:
        print(f"\n{'#'*70}")
        print("# FP4 DECODE VERSION COMPARISON")
        print(f"{'#'*70}")

        # Create test data
        device = "cuda"
        N = args.intermediate_size
        K = args.hidden_size
        num_experts = args.num_experts

        print(f"\nCreating test data: {num_experts} experts, N={N}, K={K}...")
        weights = [
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
            for _ in range(num_experts)
        ]
        scales = [
            torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
            for _ in range(num_experts)
        ]

        # Run version comparison
        results = benchmark_all_dequant_versions(
            weights, scales, N, K, num_experts, device
        )

        # Also include CUDA kernels in comparison
        print("\n--- CUDA Shared Memory LUT (for reference) ---")
        cuda_results = benchmark_cuda_dequant_only(
            weights, scales, N, K, num_experts, device
        )

        # Final summary
        print(f"\n{'='*70}")
        print("COMPREHENSIVE DEQUANT COMPARISON")
        print(f"{'='*70}")
        print(f"{'Approach':<35} {'Time (ms)':<12} {'Notes'}")
        print("-" * 70)

        # Triton versions
        for version, time_ms in results.items():
            if time_ms != float('inf'):
                notes = {
                    "v1_sequential": "16 tl.where()",
                    "v2_e2m1": "E2M1 arithmetic",
                    "v3_binary_tree": "Binary tree lookup",
                    "v4_branchless": "IEEE bitcast",
                }.get(version, "")
                print(f"Triton {version:<26} {time_ms:<12.3f} {notes}")

        # CUDA versions
        for name, time_ms in cuda_results.items():
            if time_ms != float('inf'):
                print(f"CUDA {name:<29} {time_ms:<12.3f} Shared memory LUT")

        # Find overall best
        all_results = {f"triton_{k}": v for k, v in results.items()}
        all_results.update({f"cuda_{k}": v for k, v in cuda_results.items()})
        valid = {k: v for k, v in all_results.items() if v != float('inf')}

        if valid:
            best = min(valid, key=valid.get)
            print("-" * 70)
            print(f"BEST: {best} at {valid[best]:.3f} ms")

        print(f"{'='*70}")
        sys.exit(0)

    config = {
        "num_experts": args.num_experts,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts_per_tok": DEFAULT_CONFIG["num_experts_per_tok"],
    }

    print(f"\n{'#'*70}")
    print("# DECOUPLED MXFP4 MOE BENCHMARK")
    print(f"{'#'*70}")

    all_results = {}
    for tpe in args.tokens:
        results = run_benchmark(tpe, config, detailed=args.detailed, isolate_kernels=not args.no_isolate)
        all_results[tpe] = results

    # Final summary if multiple token counts
    if len(args.tokens) > 1:
        print(f"\n{'='*70}")
        print("FINAL SUMMARY (all token counts)")
        print(f"{'='*70}")
        print(f"{'Tokens':<10} {'Fused':>12} {'Decoupled':>12} {'Speedup':>12}")
        print("-" * 70)

        for tpe, results in all_results.items():
            fused = results.get("fused_mxfp4", float('inf'))
            decoupled = results.get("decoupled", float('inf'))
            if fused != float('inf') and decoupled != float('inf'):
                speedup = fused / decoupled
                print(f"{tpe:<10} {fused:>12.3f} {decoupled:>12.3f} {speedup:>11.2f}x")
            else:
                print(f"{tpe:<10} {'N/A':>12} {'N/A':>12} {'N/A':>12}")


if __name__ == "__main__":
    main()
