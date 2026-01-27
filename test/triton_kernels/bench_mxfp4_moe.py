"""Unified MXFP4 MoE GEMM Benchmark.

Compares ALL MXFP4 MoE GEMM approaches in one script:
1. Fused MXFP4 Grouped GEMM - inline dequantization during GEMM
2. Decoupled Dequant + BF16 GEMM - separate kernels
3. FP4 decode versions (v1-v6) - dequant kernel variants
4. Unfused baselines - reference implementations

This script consolidates:
- bench_mxfp4_grouped_gemm.py (fused GEMM tuning)
- bench_decoupled_mxfp4_moe.py (decoupled approach, FP4 versions)

Usage:
    # Quick A/B: Fused vs Decoupled comparison
    python bench_mxfp4_moe.py --quick --tokens 4

    # Compare all approaches (default)
    python bench_mxfp4_moe.py --compare-all --tokens 4

    # FP4 decode version comparison only
    python bench_mxfp4_moe.py --compare-fp4

    # GEMM hyperparameter tuning (grid search)
    python bench_mxfp4_moe.py --tune-gemm --tokens 4

    # Numerical validation only
    python bench_mxfp4_moe.py --validate

    # Export results to CSV
    python bench_mxfp4_moe.py --compare-all --output results.csv
"""

import argparse
import csv
import itertools
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add batchgen to path
sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")


# =============================================================================
# Configuration
# =============================================================================

# GPT-OSS-120B dimensions (default)
DEFAULT_CONFIG = {
    "num_experts": 128,
    "hidden_size": 5120,         # K dimension
    "intermediate_size": 13824,  # N dimension
    "num_experts_per_tok": 8,
}

# GEMM tuning search space
GEMM_SEARCH_SPACE = {
    "BLOCK_M": [32, 64, 128, 256],
    "BLOCK_N": [32, 64, 128, 256],
    "BLOCK_K": [32, 64],  # 64 processes 2 scale blocks per K-tile (default)
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2, 3, 4],
}


# =============================================================================
# Shared Infrastructure
# =============================================================================

def create_test_tensors(
    num_experts: int,
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda",
    per_expert_weights: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Create test tensors for benchmarking.

    Args:
        num_experts: Number of experts (128 for GPT-OSS-120B)
        tokens_per_expert: Tokens routed to each expert
        hidden_size: Model hidden dimension (K=5120)
        intermediate_size: MLP intermediate dimension (N=13824)
        device: CUDA device
        per_expert_weights: If True, create different weights per expert (realistic).
                           If False, share single weight across experts (for tuning).

    Returns:
        hidden_3d: [num_experts, tokens_per_expert, hidden_size] BF16
        weights: List of [N, K//2] uint8 packed FP4 (one per expert if per_expert_weights)
        scales: List of [N, K//32] uint8 (one per expert if per_expert_weights)
        expert_counts: [num_experts] int32
    """
    K = hidden_size
    N = intermediate_size

    # Input: [num_experts, tokens_per_expert, hidden_size]
    hidden_3d = torch.randn(
        num_experts, tokens_per_expert, K,
        dtype=torch.bfloat16, device=device
    )

    if per_expert_weights:
        # Different weights per expert (realistic for MoE)
        weights = [
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
            for _ in range(num_experts)
        ]
        scales = [
            torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
            for _ in range(num_experts)
        ]
    else:
        # Single shared weight (for tuning to reduce memory)
        single_weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        single_scale = torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
        weights = [single_weight] * num_experts
        scales = [single_scale] * num_experts

    # Expert token counts (uniform distribution for benchmarking)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert,
        dtype=torch.int32, device=device
    )

    return hidden_3d, weights, scales, expert_counts


def setup_pointer_arrays(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create pointer arrays for grouped GEMM kernels."""
    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weights], dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scales], dtype=torch.int64, device=device
    )
    return weight_ptrs, scale_ptrs


def compute_metrics(
    time_ms: float,
    num_experts: int,
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    include_dequant_output: bool = False,
) -> Dict[str, float]:
    """Compute performance metrics.

    Args:
        time_ms: Execution time in milliseconds
        num_experts: Number of experts
        tokens_per_expert: Tokens per expert (M dimension)
        hidden_size: K dimension
        intermediate_size: N dimension
        include_dequant_output: If True, include BF16 output write in data movement
                               (for decoupled approach)

    Returns:
        Dictionary with TFLOPS, bandwidth, HBM utilization
    """
    M = tokens_per_expert
    N = intermediate_size
    K = hidden_size

    # FLOPs for GEMM: 2 * M * N * K per expert
    flops_per_expert = 2 * M * N * K
    total_flops = flops_per_expert * num_experts

    # Data movement (bytes)
    # Input: LHS activation [E, M, K] BF16
    lhs_bytes = num_experts * M * K * 2

    if include_dequant_output:
        # Decoupled: read packed FP4 + scales, write BF16 weights, then GEMM
        # Packed FP4: E * N * K/2
        packed_bytes = num_experts * N * (K // 2)
        # Scales: E * N * K/32
        scale_bytes = num_experts * N * (K // 32)
        # BF16 output: E * N * K * 2 (written by dequant, read by GEMM)
        bf16_weights_bytes = num_experts * N * K * 2 * 2  # write + read
        rhs_bytes = packed_bytes + scale_bytes + bf16_weights_bytes
    else:
        # Fused: read packed FP4 + scales directly
        rhs_bytes = num_experts * N * (K // 2) + num_experts * N * (K // 32)

    # Output: E * M * N BF16
    output_bytes = num_experts * M * N * 2
    total_bytes = lhs_bytes + rhs_bytes + output_bytes

    time_s = time_ms / 1000.0
    tflops = total_flops / time_s / 1e12
    bandwidth_gb_s = total_bytes / time_s / 1e9

    # GPU-specific peak bandwidth
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" in gpu_name:
        peak_bw_gb_s = 3350  # H100 SXM: 3.35 TB/s
    elif "H20" in gpu_name:
        peak_bw_gb_s = 4000  # H20: 4.0 TB/s
    elif "A100" in gpu_name:
        peak_bw_gb_s = 2039  # A100 SXM: 2.0 TB/s
    else:
        peak_bw_gb_s = 2000  # Default assumption

    hbm_util = bandwidth_gb_s / peak_bw_gb_s * 100

    return {
        "tflops": tflops,
        "bandwidth_gb_s": bandwidth_gb_s,
        "hbm_util_pct": hbm_util,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "peak_bw_gb_s": peak_bw_gb_s,
    }


def compute_dequant_metrics(
    time_ms: float,
    num_experts: int,
    N: int,
    K: int,
) -> Dict[str, float]:
    """Compute metrics for dequantization kernel only.

    Args:
        time_ms: Execution time in milliseconds
        num_experts: Number of experts
        N: Output rows (intermediate_size)
        K: Output columns (hidden_size)

    Returns:
        Dictionary with bandwidth and HBM utilization
    """
    # Data movement for dequant:
    # Input: packed FP4 (E * N * K/2) + scales (E * N * K/32)
    # Output: BF16 (E * N * K * 2)
    packed_bytes = num_experts * N * (K // 2)
    scale_bytes = num_experts * N * (K // 32)
    output_bytes = num_experts * N * K * 2
    total_bytes = packed_bytes + scale_bytes + output_bytes
    total_gb = total_bytes / 1e9

    time_s = time_ms / 1000.0
    bandwidth_gb_s = total_bytes / time_s / 1e9

    # GPU-specific peak bandwidth
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" in gpu_name:
        peak_bw_gb_s = 3350
    elif "H20" in gpu_name:
        peak_bw_gb_s = 4000
    elif "A100" in gpu_name:
        peak_bw_gb_s = 2039
    else:
        peak_bw_gb_s = 2000

    hbm_util = bandwidth_gb_s / peak_bw_gb_s * 100

    return {
        "bandwidth_gb_s": bandwidth_gb_s,
        "hbm_util_pct": hbm_util,
        "total_gb": total_gb,
        "packed_gb": packed_bytes / 1e9,
        "scale_gb": scale_bytes / 1e9,
        "output_gb": output_bytes / 1e9,
        "peak_bw_gb_s": peak_bw_gb_s,
    }


# =============================================================================
# Numerical Validation
# =============================================================================

def reference_mxfp4_dequant(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for MXFP4 dequantization."""
    from batchgen.quantization.mxfp4 import mxfp4_dequantize
    return mxfp4_dequantize(packed, scales, dtype=torch.bfloat16)


def validate_dequant_versions(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    num_experts: int,
    N: int,
    K: int,
    device: str = "cuda",
    num_experts_to_check: int = 3,
    tolerance: float = 0.01,
) -> bool:
    """Validate all dequant kernel versions produce correct output.

    Returns True if all versions pass validation, False otherwise.
    """
    try:
        from batchgen.moe.decoupled_mxfp4_moe import (
            batch_mxfp4_dequant,
            FP4_DECODE_VERSIONS,
        )
    except ImportError as e:
        print(f"  Cannot import batch_mxfp4_dequant: {e}")
        return False

    print(f"\n{'='*70}")
    print("NUMERICAL VALIDATION")
    print(f"{'='*70}")
    print(f"Checking {num_experts_to_check} experts against PyTorch reference")
    print(f"Tolerance: {tolerance*100:.1f}% relative error")
    print(f"{'='*70}")

    # Generate reference outputs
    print("\nGenerating reference outputs...")
    ref_outputs = []
    for i in range(min(num_experts_to_check, num_experts)):
        ref = reference_mxfp4_dequant(weights[i], scales[i])
        ref_outputs.append(ref)
    print(f"  Reference shape: {ref_outputs[0].shape}")

    # Setup pointer arrays
    weight_ptrs, scale_ptrs = setup_pointer_arrays(weights, scales, device)

    # For v6_scale_transpose, create transposed scales
    scales_transposed = [s.t().contiguous() for s in scales]
    scale_ptrs_transposed = torch.tensor(
        [s.data_ptr() for s in scales_transposed], dtype=torch.int64, device=device
    )

    # Allocate BF16 buffer
    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    all_pass = True
    results_summary = []

    for version in FP4_DECODE_VERSIONS:
        print(f"\nValidating {version}...")
        try:
            # Use transposed scales for v6, v7, v8 (they use K-major scale layout)
            if version in ("v6_scale_transpose", "v7_fast_scale", "v8_ieee_pow2"):
                curr_scale_ptrs = scale_ptrs_transposed
                curr_scale_ref = scales_transposed[0]
            else:
                curr_scale_ptrs = scale_ptrs
                curr_scale_ref = scales[0]

            # Run kernel
            batch_mxfp4_dequant(
                weight_ptrs, curr_scale_ptrs, bf16_buffer,
                weights[0], curr_scale_ref, version=version
            )
            torch.cuda.synchronize()

            # Check each expert
            version_pass = True
            max_rel_errors = []
            for i in range(min(num_experts_to_check, num_experts)):
                kernel_out = bf16_buffer[i]
                ref_out = ref_outputs[i]

                diff = (kernel_out.float() - ref_out.float()).abs()
                rel_diff = (diff / ref_out.float().abs().clamp(min=1e-6)).max().item()
                max_rel_errors.append(rel_diff)

                if rel_diff > tolerance:
                    print(f"  FAIL expert {i}: max_rel_error={rel_diff:.4f} > {tolerance*100:.1f}%")
                    version_pass = False
                else:
                    print(f"  PASS expert {i}: max_rel_error={rel_diff:.6f}")

            avg_rel_error = sum(max_rel_errors) / len(max_rel_errors)
            status = "PASS" if version_pass else "FAIL"
            results_summary.append((version, status, avg_rel_error))

            if not version_pass:
                all_pass = False

        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append((version, "ERROR", float('nan')))
            all_pass = False

    # Print summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Version':<20} {'Status':<10} {'Avg Rel Error':<15}")
    print("-" * 70)
    for version, status, avg_err in results_summary:
        if status == "ERROR":
            print(f"{version:<20} {status:<10} {'N/A':<15}")
        else:
            print(f"{version:<20} {status:<10} {avg_err:.6f} ({avg_err*100:.4f}%)")
    print(f"{'='*70}")

    if all_pass:
        print("ALL VERSIONS PASSED VALIDATION")
    else:
        print("SOME VERSIONS FAILED VALIDATION")

    return all_pass


def validate_cute_dequant_versions(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    num_experts: int,
    N: int,
    K: int,
    device: str = "cuda",
    num_experts_to_check: int = 3,
    tolerance: float = 0.01,
) -> bool:
    """Validate CuTe CUDA dequant kernel versions produce correct output.

    Returns True if all versions pass validation, False otherwise.
    """
    try:
        from batchgen.moe.cute_mxfp4_dequant import batch_mxfp4_dequant_cute
    except ImportError as e:
        print(f"  Cannot import CuTe kernel: {e}")
        return False

    print(f"\n{'='*70}")
    print("CUTE CUDA NUMERICAL VALIDATION")
    print(f"{'='*70}")
    print(f"Checking {num_experts_to_check} experts against PyTorch reference")
    print(f"Tolerance: {tolerance*100:.1f}% relative error")
    print(f"{'='*70}")

    # Generate reference outputs
    print("\nGenerating reference outputs...")
    ref_outputs = []
    for i in range(min(num_experts_to_check, num_experts)):
        ref = reference_mxfp4_dequant(weights[i], scales[i])
        ref_outputs.append(ref)
    print(f"  Reference shape: {ref_outputs[0].shape}")

    # Setup pointer arrays
    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weights], dtype=torch.int64, device=device
    )

    # CuTe uses K-major scales: [K//32, N] (transposed)
    scales_transposed = [s.t().contiguous() for s in scales]
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scales_transposed], dtype=torch.int64, device=device
    )

    # Allocate BF16 buffer
    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    all_pass = True
    results_summary = []

    cute_versions = [
        ("cute_simple", 0),
        ("cute_swizzle", 1),
    ]

    for version_name, kernel_ver in cute_versions:
        print(f"\nValidating {version_name}...")
        try:
            # Run kernel
            batch_mxfp4_dequant_cute(
                weight_ptrs, scale_ptrs, bf16_buffer,
                weights[0], scales_transposed[0], kernel_version=kernel_ver
            )
            torch.cuda.synchronize()

            # Check each expert
            version_pass = True
            max_rel_errors = []
            for i in range(min(num_experts_to_check, num_experts)):
                kernel_out = bf16_buffer[i]
                ref_out = ref_outputs[i]

                diff = (kernel_out.float() - ref_out.float()).abs()
                rel_diff = (diff / ref_out.float().abs().clamp(min=1e-6)).max().item()
                max_rel_errors.append(rel_diff)

                if rel_diff > tolerance:
                    print(f"  FAIL expert {i}: max_rel_error={rel_diff:.4f} > {tolerance*100:.1f}%")
                    version_pass = False
                else:
                    print(f"  PASS expert {i}: max_rel_error={rel_diff:.6f}")

            avg_rel_error = sum(max_rel_errors) / len(max_rel_errors)
            status = "PASS" if version_pass else "FAIL"
            results_summary.append((version_name, status, avg_rel_error))

            if not version_pass:
                all_pass = False

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append((version_name, "ERROR", float('nan')))
            all_pass = False

    # Print summary
    print(f"\n{'='*70}")
    print("CUTE VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Version':<20} {'Status':<10} {'Avg Rel Error':<15}")
    print("-" * 70)
    for version, status, avg_err in results_summary:
        if status == "ERROR":
            print(f"{version:<20} {status:<10} {'N/A':<15}")
        else:
            print(f"{version:<20} {status:<10} {avg_err:.6f} ({avg_err*100:.4f}%)")
    print(f"{'='*70}")

    if all_pass:
        print("ALL CUTE VERSIONS PASSED VALIDATION")
    else:
        print("SOME CUTE VERSIONS FAILED VALIDATION")

    return all_pass


# =============================================================================
# Benchmark Functions: Fused MXFP4 Grouped GEMM
# =============================================================================

def benchmark_fused_mxfp4_gemm(
    hidden_3d: torch.Tensor,
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> float:
    """Benchmark fused MXFP4 grouped GEMM (inline dequantization).

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


def benchmark_cute_fused_mxfp4_gemm(
    hidden_3d: torch.Tensor,
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    kernel_version: int = 1,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> float:
    """Benchmark CuTe fused MXFP4 grouped GEMM (inline dequantization).

    Args:
        kernel_version: 0=simple scalar, 1=WMMA tensor cores

    Returns time in milliseconds.
    """
    try:
        from batchgen.moe.cute_fused_mxfp4_gemm import cute_grouped_mxfp4_gemm_3d
        from batchgen.moe.mxfp4_grouped_gemm import setup_expert_weight_pointers
    except ImportError as e:
        print(f"  CuTe fused MXFP4 GEMM not available: {e}")
        return float('inf')

    num_experts = hidden_3d.shape[0]
    tokens_per_expert = hidden_3d.shape[1]

    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=hidden_3d.device
    )

    try:
        # Warmup
        for _ in range(warmup_iters):
            _ = cute_grouped_mxfp4_gemm_3d(
                hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
                N, weights[0], scales[0], kernel_version=kernel_version
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = cute_grouped_mxfp4_gemm_3d(
                hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
                N, weights[0], scales[0], kernel_version=kernel_version
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        return elapsed
    except Exception as e:
        print(f"  CuTe fused MXFP4 GEMM failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


def benchmark_fused_gemm_config(
    hidden_3d: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    expert_counts: torch.Tensor,
    N: int,
    weight_ref: torch.Tensor,
    scale_ref: torch.Tensor,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    num_warps: int,
    num_stages: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> Tuple[float, bool]:
    """Benchmark a single GEMM configuration for tuning.

    Returns (time_ms, success).
    """
    try:
        from batchgen.moe.mxfp4_grouped_gemm import grouped_mxfp4_gemm_3d_tunable
    except ImportError:
        return float('inf'), False

    try:
        # Warmup
        for _ in range(warmup_iters):
            _ = grouped_mxfp4_gemm_3d_tunable(
                hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
                N, weight_ref, scale_ref,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=num_warps, num_stages=num_stages,
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = grouped_mxfp4_gemm_3d_tunable(
                hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
                N, weight_ref, scale_ref,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=num_warps, num_stages=num_stages,
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        return elapsed, True
    except Exception as e:
        return float('inf'), False


def run_gemm_tuning(
    tokens_per_expert: int,
    config: Dict,
    device: str = "cuda",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run hyperparameter grid search for GEMM kernel.

    Returns list of result dictionaries sorted by time.
    """
    try:
        from batchgen.moe.mxfp4_grouped_gemm import setup_expert_weight_pointers
    except ImportError as e:
        print(f"Cannot import GEMM functions: {e}")
        return []

    if verbose:
        print(f"\n{'='*60}")
        print(f"GEMM TUNING: {tokens_per_expert} tokens/expert")
        print(f"{'='*60}")

    # Create test tensors (shared weight for tuning to save memory)
    hidden_3d, weights, scales, expert_counts = create_test_tensors(
        config['num_experts'], tokens_per_expert,
        config['hidden_size'], config['intermediate_size'],
        device, per_expert_weights=False
    )

    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)
    N = config['intermediate_size']

    results = []
    configs = list(itertools.product(
        GEMM_SEARCH_SPACE["BLOCK_M"],
        GEMM_SEARCH_SPACE["BLOCK_N"],
        GEMM_SEARCH_SPACE["BLOCK_K"],
        GEMM_SEARCH_SPACE["num_warps"],
        GEMM_SEARCH_SPACE["num_stages"],
    ))

    if verbose:
        print(f"Testing {len(configs)} configurations...")

    for i, (bm, bn, bk, nw, ns) in enumerate(configs):
        time_ms, success = benchmark_fused_gemm_config(
            hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
            N, weights[0], scales[0],
            bm, bn, bk, nw, ns,
        )

        results.append({
            'tokens_per_expert': tokens_per_expert,
            'BLOCK_M': bm,
            'BLOCK_N': bn,
            'BLOCK_K': bk,
            'num_warps': nw,
            'num_stages': ns,
            'time_ms': time_ms,
            'success': success,
        })

        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(configs)}")

    results.sort(key=lambda x: x['time_ms'])

    if verbose:
        print(f"\nTop 5 configurations:")
        print("-" * 70)
        for r in results[:5]:
            if r['success']:
                print(f"  BLOCK_M={r['BLOCK_M']:3d}, BLOCK_N={r['BLOCK_N']:3d}, "
                      f"warps={r['num_warps']}, stages={r['num_stages']}: "
                      f"{r['time_ms']:.3f} ms")

    return results


# =============================================================================
# Benchmark Functions: Decoupled Dequant + GEMM
# =============================================================================

def benchmark_dequant_kernel(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    K: int,
    num_experts: int,
    device: str = "cuda",
    version: str = "v2_e2m1",
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark dequantization kernel only.

    Returns time in milliseconds.
    """
    try:
        from batchgen.moe.decoupled_mxfp4_moe import batch_mxfp4_dequant
    except ImportError as e:
        print(f"  batch_mxfp4_dequant not available: {e}")
        return float('inf')

    weight_ptrs, scale_ptrs = setup_pointer_arrays(weights, scales, device)

    # For v6, v7, v8, use transposed scales (K-major layout)
    if version in ("v6_scale_transpose", "v7_fast_scale", "v8_ieee_pow2"):
        scales_transposed = [s.t().contiguous() for s in scales]
        scale_ptrs = torch.tensor(
            [s.data_ptr() for s in scales_transposed], dtype=torch.int64, device=device
        )
        scale_ref = scales_transposed[0]
    else:
        scale_ref = scales[0]

    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    try:
        # Warmup
        for _ in range(warmup_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scale_ref, version=version)
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scale_ref, version=version)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        return elapsed
    except Exception as e:
        print(f"  Dequant kernel ({version}) failed: {e}")
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
) -> Dict[str, Dict[str, float]]:
    """Benchmark all FP4 decode versions.

    Returns dict mapping version name to {time_ms, bandwidth_gb_s, hbm_util_pct}.
    """
    try:
        from batchgen.moe.decoupled_mxfp4_moe import FP4_DECODE_VERSIONS
    except ImportError:
        return {}

    results = {}
    version_notes = {
        "v1_sequential": "16 tl.where() - baseline",
        "v2_e2m1": "E2M1 arithmetic (5-6 where)",
        "v3_binary_tree": "Binary tree (4 where)",
        "v4_branchless": "IEEE bitcast (2 where)",
        "v5_memopt": "BLOCK_K=64, 2x fewer blocks",
        "v6_scale_transpose": "K-major scales, coalesced loads",
        "v7_fast_scale": "tl.exp2 (SFU) instead of _ldexp (ALU)",
        "v8_ieee_pow2": "Direct IEEE pow2 (2 int ops + 1 mul)",
    }

    for version in FP4_DECODE_VERSIONS:
        time_ms = benchmark_dequant_kernel(
            weights, scales, N, K, num_experts, device, version, warmup_iters, bench_iters
        )

        if time_ms != float('inf'):
            metrics = compute_dequant_metrics(time_ms, num_experts, N, K)
            results[version] = {
                "time_ms": time_ms,
                "bandwidth_gb_s": metrics["bandwidth_gb_s"],
                "hbm_util_pct": metrics["hbm_util_pct"],
                "notes": version_notes.get(version, ""),
            }
        else:
            results[version] = {
                "time_ms": float('inf'),
                "bandwidth_gb_s": 0,
                "hbm_util_pct": 0,
                "notes": "FAILED",
            }

    return results


def benchmark_cute_dequant_kernel(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    K: int,
    num_experts: int,
    device: str = "cuda",
    kernel_version: int = 0,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark CuTe-style CUDA dequantization kernel.

    Args:
        kernel_version: 0=simple vectorized, 1=swizzled smem

    Returns time in milliseconds.
    """
    try:
        from batchgen.moe.cute_mxfp4_dequant import batch_mxfp4_dequant_cute
    except ImportError as e:
        print(f"  CuTe kernel not available: {e}")
        return float('inf')

    # Setup pointer arrays
    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weights], dtype=torch.int64, device=device
    )

    # CuTe uses K-major scales: [K//32, N]
    scales_transposed = [s.t().contiguous() for s in scales]
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scales_transposed], dtype=torch.int64, device=device
    )

    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    try:
        # Warmup
        for _ in range(warmup_iters):
            batch_mxfp4_dequant_cute(
                weight_ptrs, scale_ptrs, bf16_buffer,
                weights[0], scales_transposed[0], kernel_version=kernel_version
            )
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant_cute(
                weight_ptrs, scale_ptrs, bf16_buffer,
                weights[0], scales_transposed[0], kernel_version=kernel_version
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        return elapsed
    except Exception as e:
        print(f"  CuTe kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


def benchmark_all_cuda_versions(
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    K: int,
    num_experts: int,
    device: str = "cuda",
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Benchmark all CUDA dequant kernel versions (CuTe-style).

    Returns dict mapping version name to {time_ms, bandwidth_gb_s, hbm_util_pct}.
    """
    results = {}
    version_notes = {
        "cute_simple": "CuTe-style: vectorized 128-bit loads/stores, smem LUT",
        "cute_swizzle": "CuTe-style: + swizzled smem for 0 bank conflicts",
    }

    versions = [
        ("cute_simple", 0),
        ("cute_swizzle", 1),
    ]

    for name, kernel_ver in versions:
        time_ms = benchmark_cute_dequant_kernel(
            weights, scales, N, K, num_experts, device, kernel_ver, warmup_iters, bench_iters
        )

        if time_ms != float('inf'):
            metrics = compute_dequant_metrics(time_ms, num_experts, N, K)
            results[name] = {
                "time_ms": time_ms,
                "bandwidth_gb_s": metrics["bandwidth_gb_s"],
                "hbm_util_pct": metrics["hbm_util_pct"],
                "notes": version_notes.get(name, ""),
            }
        else:
            results[name] = {
                "time_ms": float('inf'),
                "bandwidth_gb_s": 0,
                "hbm_util_pct": 0,
                "notes": "FAILED",
            }

    return results


def benchmark_bf16_gemm_kernel(
    hidden_3d: torch.Tensor,
    N: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark BF16 grouped GEMM kernel only (no dequant).

    Returns time in milliseconds.
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

    # Random BF16 weights (skip dequant)
    bf16_buffer = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=device
    )

    try:
        # Warmup
        for _ in range(warmup_iters):
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        return elapsed
    except Exception as e:
        print(f"  BF16 GEMM kernel failed: {e}")
        return float('inf')


def benchmark_decoupled_combined(
    hidden_3d: torch.Tensor,
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    dequant_version: str = "v6_scale_transpose",
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> Tuple[float, Dict[str, float]]:
    """Benchmark decoupled approach: dequant + BF16 GEMM.

    Returns (total_time_ms, {dequant_ms, gemm_ms}).
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

    weight_ptrs, scale_ptrs = setup_pointer_arrays(weights, scales, device)

    # For v6, v7, v8, use transposed scales (K-major layout)
    if dequant_version in ("v6_scale_transpose", "v7_fast_scale", "v8_ieee_pow2"):
        scales_transposed = [s.t().contiguous() for s in scales]
        scale_ptrs = torch.tensor(
            [s.data_ptr() for s in scales_transposed], dtype=torch.int64, device=device
        )
        scale_ref = scales_transposed[0]
    else:
        scale_ref = scales[0]

    bf16_buffer = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=device
    )

    component_times = {}

    try:
        # Warmup combined
        for _ in range(warmup_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scale_ref, version=dequant_version)
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
        torch.cuda.synchronize()

        # Benchmark dequant
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scale_ref, version=dequant_version)
        torch.cuda.synchronize()
        component_times["dequant_ms"] = (time.perf_counter() - start) / bench_iters * 1000

        # Benchmark GEMM
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
        torch.cuda.synchronize()
        component_times["gemm_ms"] = (time.perf_counter() - start) / bench_iters * 1000

        # Benchmark total
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, bf16_buffer, weights[0], scale_ref, version=dequant_version)
            _ = bf16_grouped_gemm_3d(hidden_3d, bf16_buffer, expert_counts, N)
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / bench_iters * 1000

        return total_time, component_times
    except Exception as e:
        print(f"  Decoupled benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), {}


# =============================================================================
# Benchmark Functions: Baselines
# =============================================================================

def benchmark_unfused_single_weight(
    hidden_3d: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    N: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark unfused: dequant once + torch.mm per expert (same weight).

    NOTE: This reuses the SAME weight for all experts (unrealistic but fast reference).
    """
    try:
        from batchgen.quantization.mxfp4 import mxfp4_dequantize
    except ImportError:
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
        return float('inf')


def benchmark_unfused_per_expert(
    hidden_3d: torch.Tensor,
    weights: List[torch.Tensor],
    scales: List[torch.Tensor],
    N: int,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark unfused with DIFFERENT weights per expert (fair comparison)."""
    try:
        from batchgen.quantization.mxfp4 import mxfp4_dequantize
    except ImportError:
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
        return float('inf')


# =============================================================================
# Main Benchmark Runners
# =============================================================================

def run_quick_benchmark(
    tokens_per_expert: int,
    config: Dict,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Quick A/B comparison: Fused vs Decoupled."""
    print(f"\n{'='*70}")
    print("MXFP4 MoE GEMM BENCHMARK")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {config['num_experts']} experts, {tokens_per_expert} tokens/expert")
    print(f"Dimensions: K={config['hidden_size']}, N={config['intermediate_size']}")
    print(f"{'='*70}")

    hidden_3d, weights, scales, expert_counts = create_test_tensors(
        config["num_experts"], tokens_per_expert,
        config["hidden_size"], config["intermediate_size"],
        device, per_expert_weights=True
    )

    N = config["intermediate_size"]
    K = config["hidden_size"]

    results = {}

    # Fused MXFP4 GEMM
    print("\nBenchmarking Fused MXFP4 Grouped GEMM...")
    fused_time = benchmark_fused_mxfp4_gemm(hidden_3d, weights, scales, N)
    results["fused_mxfp4"] = fused_time
    print(f"  Time: {fused_time:.3f} ms")

    # Decoupled (best version: v6_scale_transpose)
    print("\nBenchmarking Decoupled (v6_scale_transpose)...")
    decoupled_time, components = benchmark_decoupled_combined(
        hidden_3d, weights, scales, N, dequant_version="v6_scale_transpose"
    )
    results["decoupled"] = decoupled_time
    results["decoupled_dequant"] = components.get("dequant_ms", 0)
    results["decoupled_gemm"] = components.get("gemm_ms", 0)
    print(f"  Dequant: {components.get('dequant_ms', 0):.3f} ms")
    print(f"  GEMM:    {components.get('gemm_ms', 0):.3f} ms")
    print(f"  Total:   {decoupled_time:.3f} ms")

    # Unfused baseline (single weight - not fair but fast)
    print("\nBenchmarking Unfused baseline (single weight)...")
    unfused_time = benchmark_unfused_single_weight(hidden_3d, weights[0], scales[0], N)
    results["unfused_single"] = unfused_time
    print(f"  Time: {unfused_time:.3f} ms (NOTE: reuses single weight)")

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Approach':<40} {'Time (ms)':>12} {'vs Fused':>12}")
    print("-" * 70)

    baseline = fused_time if fused_time != float('inf') else 1.0
    for name, time_ms in [
        ("Fused MXFP4 Grouped GEMM", fused_time),
        ("Decoupled (dequant + BF16 GEMM)", decoupled_time),
        ("  - Dequant (v6):", results.get("decoupled_dequant", 0)),
        ("  - BF16 GEMM:", results.get("decoupled_gemm", 0)),
        ("Unfused single weight*", unfused_time),
    ]:
        if time_ms != float('inf') and time_ms > 0:
            ratio = baseline / time_ms if name.startswith("  -") else time_ms / baseline
            if name.startswith("  -"):
                print(f"{name:<40} {time_ms:>12.3f} {'':>12}")
            else:
                print(f"{name:<40} {time_ms:>12.3f} {ratio:>11.2f}x")
        else:
            print(f"{name:<40} {'N/A':>12} {'N/A':>12}")

    print("-" * 70)
    print("* Single weight reused - not fair comparison")
    print(f"{'='*70}")

    return results


def run_full_comparison(
    tokens_per_expert: int,
    config: Dict,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Full comparison of all approaches."""
    print(f"\n{'='*70}")
    print("MXFP4 MoE GEMM BENCHMARK - FULL COMPARISON")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {config['num_experts']} experts, {tokens_per_expert} tokens/expert")
    print(f"Dimensions: K={config['hidden_size']}, N={config['intermediate_size']}")

    # Calculate data movement
    num_experts = config['num_experts']
    N = config['intermediate_size']
    K = config['hidden_size']
    packed_gb = num_experts * N * (K // 2) / 1e9
    scale_gb = num_experts * N * (K // 32) / 1e9
    output_gb = num_experts * N * K * 2 / 1e9
    total_gb = packed_gb + scale_gb + output_gb
    print(f"Data movement: {packed_gb:.2f} GB packed + {scale_gb:.2f} GB scales + {output_gb:.2f} GB output = {total_gb:.2f} GB")
    print(f"{'='*70}")

    hidden_3d, weights, scales, expert_counts = create_test_tensors(
        num_experts, tokens_per_expert, K, N, device, per_expert_weights=True
    )

    results = {}

    # 1. Fused MXFP4 GEMM
    print("\n[1/5] Fused MXFP4 Grouped GEMM...")
    fused_time = benchmark_fused_mxfp4_gemm(hidden_3d, weights, scales, N)
    results["fused"] = {"time_ms": fused_time}
    if fused_time != float('inf'):
        metrics = compute_metrics(fused_time, num_experts, tokens_per_expert, K, N, include_dequant_output=False)
        results["fused"].update(metrics)
        print(f"  Time: {fused_time:.3f} ms | {metrics['tflops']:.2f} TFLOPS | {metrics['hbm_util_pct']:.1f}% HBM")

    # 2. Decoupled (v6_scale_transpose)
    print("\n[2/5] Decoupled (v6_scale_transpose)...")
    decoupled_time, components = benchmark_decoupled_combined(
        hidden_3d, weights, scales, N, dequant_version="v6_scale_transpose"
    )
    results["decoupled_v6"] = {
        "time_ms": decoupled_time,
        "dequant_ms": components.get("dequant_ms", 0),
        "gemm_ms": components.get("gemm_ms", 0),
    }
    if decoupled_time != float('inf'):
        metrics = compute_metrics(decoupled_time, num_experts, tokens_per_expert, K, N, include_dequant_output=True)
        results["decoupled_v6"].update(metrics)
        print(f"  Dequant: {components.get('dequant_ms', 0):.3f} ms")
        print(f"  GEMM:    {components.get('gemm_ms', 0):.3f} ms")
        print(f"  Total: {decoupled_time:.3f} ms | {metrics['tflops']:.2f} TFLOPS")

    # 3. FP4 decode version comparison (dequant only)
    print("\n[3/5] FP4 Decode Versions (dequant kernel only)...")
    dequant_results = benchmark_all_dequant_versions(
        weights, scales, N, K, num_experts, device
    )
    results["dequant_versions"] = dequant_results

    # 4. Unfused per-expert (fair comparison)
    print("\n[4/5] Unfused per-expert (fair)...")
    unfused_fair = benchmark_unfused_per_expert(hidden_3d, weights, scales, N)
    results["unfused_per_expert"] = {"time_ms": unfused_fair}
    print(f"  Time: {unfused_fair:.3f} ms")

    # 5. Unfused single weight (reference)
    print("\n[5/5] Unfused single weight (reference)...")
    unfused_single = benchmark_unfused_single_weight(hidden_3d, weights[0], scales[0], N)
    results["unfused_single"] = {"time_ms": unfused_single}
    print(f"  Time: {unfused_single:.3f} ms (NOTE: reuses single weight)")

    # Print comprehensive summary
    print(f"\n{'='*70}")
    print("APPROACH COMPARISON")
    print(f"{'='*70}")
    print(f"{'Approach':<35} {'Time (ms)':>10} {'TFLOPS':>8} {'BW (GB/s)':>10} {'HBM %':>8}")
    print("-" * 70)

    approach_data = [
        ("Fused MXFP4 Grouped GEMM", results.get("fused", {})),
        ("Decoupled (v6_scale_transpose)", results.get("decoupled_v6", {})),
        ("Unfused per-expert (fair)", results.get("unfused_per_expert", {})),
        ("Unfused single weight*", results.get("unfused_single", {})),
    ]

    for name, data in approach_data:
        time_ms = data.get("time_ms", float('inf'))
        tflops = data.get("tflops", 0)
        bw = data.get("bandwidth_gb_s", 0)
        hbm = data.get("hbm_util_pct", 0)
        if time_ms != float('inf'):
            print(f"{name:<35} {time_ms:>10.3f} {tflops:>8.2f} {bw:>10.0f} {hbm:>7.1f}%")
        else:
            print(f"{name:<35} {'N/A':>10} {'N/A':>8} {'N/A':>10} {'N/A':>8}")

    print("-" * 70)

    # FP4 decode version comparison
    print(f"\nFP4 DECODE VERSION COMPARISON (dequant kernel only):")
    print("-" * 70)
    print(f"{'Version':<20} {'Time (ms)':>10} {'BW (GB/s)':>10} {'HBM %':>8} {'vs v1':>8}")
    print("-" * 70)

    baseline_time = dequant_results.get("v1_sequential", {}).get("time_ms", 1.0)
    best_version = None
    best_time = float('inf')

    for version, data in dequant_results.items():
        time_ms = data.get("time_ms", float('inf'))
        bw = data.get("bandwidth_gb_s", 0)
        hbm = data.get("hbm_util_pct", 0)

        if time_ms < best_time:
            best_time = time_ms
            best_version = version

        if time_ms != float('inf'):
            speedup = baseline_time / time_ms if baseline_time != float('inf') else 1.0
            marker = " <--" if time_ms == best_time else ""
            print(f"{version:<20} {time_ms:>10.3f} {bw:>10.0f} {hbm:>7.1f}% {speedup:>7.2f}x{marker}")
        else:
            print(f"{version:<20} {'FAILED':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8}")

    print(f"{'='*70}")
    if best_version:
        print(f"Best dequant: {best_version} at {best_time:.3f} ms")
    print(f"{'='*70}")

    return results


def run_fp4_comparison(
    config: Dict,
    device: str = "cuda",
    skip_validation: bool = False,
    include_cuda: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run FP4 decode version comparison only."""
    print(f"\n{'='*70}")
    print("FP4 DECODE VERSION COMPARISON")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {config['num_experts']} experts")
    print(f"Dimensions: K={config['hidden_size']}, N={config['intermediate_size']}")

    N = config['intermediate_size']
    K = config['hidden_size']
    num_experts = config['num_experts']

    # Create test data
    print("\nCreating test data...")
    weights = [
        torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        for _ in range(num_experts)
    ]
    scales = [
        torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
        for _ in range(num_experts)
    ]

    # Validate first
    if not skip_validation:
        print("\nRunning numerical validation...")
        triton_pass = validate_dequant_versions(weights, scales, num_experts, N, K, device)
        if not triton_pass:
            print("ERROR: Triton validation failed!")
            return {}

        if include_cuda:
            cute_pass = validate_cute_dequant_versions(weights, scales, num_experts, N, K, device)
            if not cute_pass:
                print("WARNING: CuTe validation failed! Continuing with benchmark...")
        print("Validation completed.\n")

    # Benchmark Triton versions
    print("\nBenchmarking Triton versions...")
    results = benchmark_all_dequant_versions(
        weights, scales, N, K, num_experts, device
    )

    # Benchmark CuTe CUDA versions
    if include_cuda:
        print("\nBenchmarking CuTe CUDA versions...")
        cuda_results = benchmark_all_cuda_versions(
            weights, scales, N, K, num_experts, device
        )
        results.update(cuda_results)

    # Print results table
    print(f"\n{'='*100}")
    print("FP4 DECODE VERSION RESULTS")
    print(f"{'='*100}")
    print(f"{'Version':<20} {'Time (ms)':>10} {'BW (GB/s)':>10} {'HBM %':>8} {'vs v1':>10} {'Notes'}")
    print("-" * 100)

    baseline_time = results.get("v1_sequential", {}).get("time_ms", 1.0)
    best_version = None
    best_time = float('inf')

    # Sort: Triton versions first, then CuTe
    triton_versions = [k for k in results.keys() if not k.startswith("cute")]
    cuda_versions = [k for k in results.keys() if k.startswith("cute")]

    for version in triton_versions + cuda_versions:
        data = results[version]
        time_ms = data.get("time_ms", float('inf'))
        bw = data.get("bandwidth_gb_s", 0)
        hbm = data.get("hbm_util_pct", 0)
        notes = data.get("notes", "")

        if time_ms < best_time:
            best_time = time_ms
            best_version = version

        if time_ms != float('inf'):
            speedup = baseline_time / time_ms if baseline_time != float('inf') else 1.0
            marker = " <-- BEST" if version == best_version else ""
            print(f"{version:<20} {time_ms:>10.3f} {bw:>10.0f} {hbm:>7.1f}% {speedup:>9.2f}x {notes}{marker}")
        else:
            print(f"{version:<20} {'FAILED':>10} {'N/A':>10} {'N/A':>8} {'N/A':>10} {notes}")

        # Add separator between Triton and CuTe
        if version == triton_versions[-1] if triton_versions else None and cuda_versions:
            print("-" * 100)
            print("CuTe CUDA kernels:")
            print("-" * 100)

    print(f"{'='*100}")
    if best_version:
        best_data = results[best_version]
        print(f"BEST: {best_version} at {best_time:.3f} ms ({best_data['bandwidth_gb_s']:.0f} GB/s = {best_data['hbm_util_pct']:.1f}% HBM)")
    print(f"{'='*100}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified MXFP4 MoE GEMM Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick A/B: Fused vs Decoupled
    python bench_mxfp4_moe.py --quick --tokens 4

    # Full comparison of all approaches
    python bench_mxfp4_moe.py --compare-all --tokens 4

    # FP4 decode version comparison only
    python bench_mxfp4_moe.py --compare-fp4

    # GEMM hyperparameter tuning
    python bench_mxfp4_moe.py --tune-gemm --tokens 4

    # Numerical validation only
    python bench_mxfp4_moe.py --validate

    # Export results to CSV
    python bench_mxfp4_moe.py --compare-all --tokens 4 --output results.csv
"""
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick", action="store_true",
        help="Quick A/B comparison: Fused vs Decoupled (default)"
    )
    mode_group.add_argument(
        "--compare-all", action="store_true",
        help="Full comparison of all approaches"
    )
    mode_group.add_argument(
        "--compare-fp4", action="store_true",
        help="FP4 decode version comparison only (dequant kernel)"
    )
    mode_group.add_argument(
        "--tune-gemm", action="store_true",
        help="GEMM hyperparameter grid search"
    )
    mode_group.add_argument(
        "--validate", action="store_true",
        help="Numerical validation only (no timing)"
    )
    mode_group.add_argument(
        "--compare-cuda", action="store_true",
        help="CuTe CUDA kernel comparison only (dequant)"
    )
    mode_group.add_argument(
        "--compare-cute-fused", action="store_true",
        help="CuTe fused MXFP4 GEMM vs Triton comparison"
    )

    # Common options
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[4],
        help="Tokens per expert to test (default: 4)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=DEFAULT_CONFIG["num_experts"],
        help=f"Number of experts (default: {DEFAULT_CONFIG['num_experts']})"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=DEFAULT_CONFIG["hidden_size"],
        help=f"Hidden size K (default: {DEFAULT_CONFIG['hidden_size']})"
    )
    parser.add_argument(
        "--intermediate-size", type=int, default=DEFAULT_CONFIG["intermediate_size"],
        help=f"Intermediate size N (default: {DEFAULT_CONFIG['intermediate_size']})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip numerical validation (faster)"
    )

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    config = {
        "num_experts": args.num_experts,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
    }

    # Default to --quick if no mode specified
    if not any([args.quick, args.compare_all, args.compare_fp4, args.tune_gemm, args.validate, args.compare_cuda, args.compare_cute_fused]):
        args.quick = True

    # Validation only mode
    if args.validate:
        device = "cuda"
        N = config['intermediate_size']
        K = config['hidden_size']
        num_experts = config['num_experts']

        print("\nCreating test data...")
        weights = [
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
            for _ in range(num_experts)
        ]
        scales = [
            torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
            for _ in range(num_experts)
        ]

        # Validate Triton kernels
        triton_success = validate_dequant_versions(weights, scales, num_experts, N, K, device)

        # Validate CuTe CUDA kernels
        cute_success = validate_cute_dequant_versions(weights, scales, num_experts, N, K, device)

        # Overall summary
        print(f"\n{'='*70}")
        print("OVERALL VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Triton kernels: {'PASS' if triton_success else 'FAIL'}")
        print(f"CuTe CUDA kernels: {'PASS' if cute_success else 'FAIL'}")
        print(f"{'='*70}")

        success = triton_success and cute_success
        sys.exit(0 if success else 1)

    # FP4 comparison mode
    if args.compare_fp4:
        results = run_fp4_comparison(config, skip_validation=args.skip_validation, include_cuda=True)
        sys.exit(0)

    # CuTe CUDA comparison mode
    if args.compare_cuda:
        print(f"\n{'='*70}")
        print("CUTE CUDA KERNEL COMPARISON")
        print(f"{'='*70}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Config: {config['num_experts']} experts")
        print(f"Dimensions: K={config['hidden_size']}, N={config['intermediate_size']}")

        N = config['intermediate_size']
        K = config['hidden_size']
        num_experts = config['num_experts']

        # Create test data
        print("\nCreating test data...")
        weights = [
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
            for _ in range(num_experts)
        ]
        scales = [
            torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device="cuda")
            for _ in range(num_experts)
        ]

        # Validate CuTe kernels first
        if not args.skip_validation:
            print("\nRunning CuTe numerical validation...")
            if not validate_cute_dequant_versions(weights, scales, num_experts, N, K, "cuda"):
                print("ERROR: CuTe validation failed!")
                sys.exit(1)
            print("CuTe validation passed.\n")

        # Benchmark CuTe CUDA versions
        print("\nBenchmarking CuTe CUDA versions...")
        cuda_results = benchmark_all_cuda_versions(
            weights, scales, N, K, num_experts, "cuda"
        )

        # Also get best Triton for comparison
        print("\nBenchmarking Triton v7 for comparison...")
        triton_time = benchmark_dequant_kernel(
            weights, scales, N, K, num_experts, "cuda", "v7_fast_scale"
        )
        if triton_time != float('inf'):
            triton_metrics = compute_dequant_metrics(triton_time, num_experts, N, K)
            cuda_results["triton_v7_ref"] = {
                "time_ms": triton_time,
                "bandwidth_gb_s": triton_metrics["bandwidth_gb_s"],
                "hbm_util_pct": triton_metrics["hbm_util_pct"],
                "notes": "Triton v7 (best Triton) for reference",
            }

        # Print results
        print(f"\n{'='*100}")
        print("CUTE CUDA vs TRITON COMPARISON")
        print(f"{'='*100}")
        print(f"{'Version':<20} {'Time (ms)':>10} {'BW (GB/s)':>10} {'HBM %':>8} {'Notes'}")
        print("-" * 100)

        best_version = None
        best_time = float('inf')

        for version, data in cuda_results.items():
            time_ms = data.get("time_ms", float('inf'))
            bw = data.get("bandwidth_gb_s", 0)
            hbm = data.get("hbm_util_pct", 0)
            notes = data.get("notes", "")

            if time_ms < best_time:
                best_time = time_ms
                best_version = version

            if time_ms != float('inf'):
                marker = " <-- BEST" if version == best_version else ""
                print(f"{version:<20} {time_ms:>10.3f} {bw:>10.0f} {hbm:>7.1f}% {notes}{marker}")
            else:
                print(f"{version:<20} {'FAILED':>10} {'N/A':>10} {'N/A':>8} {notes}")

        print(f"{'='*100}")
        if best_version:
            best_data = cuda_results[best_version]
            print(f"BEST: {best_version} at {best_time:.3f} ms ({best_data['bandwidth_gb_s']:.0f} GB/s = {best_data['hbm_util_pct']:.1f}% HBM)")
        print(f"{'='*100}")
        sys.exit(0)

    # CuTe fused GEMM comparison mode
    if args.compare_cute_fused:
        print(f"\n{'='*70}")
        print("CUTE FUSED MXFP4 GEMM vs TRITON COMPARISON")
        print(f"{'='*70}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Config: {config['num_experts']} experts")
        print(f"Dimensions: K={config['hidden_size']}, N={config['intermediate_size']}")

        N = config['intermediate_size']
        K = config['hidden_size']
        num_experts = config['num_experts']

        results = {}

        for tpe in args.tokens:
            print(f"\n{'='*70}")
            print(f"TOKENS PER EXPERT: {tpe}")
            print(f"{'='*70}")

            # Create test data
            print("Creating test data...")
            hidden_3d, weights, scales, expert_counts = create_test_tensors(
                num_experts, tpe, K, N, "cuda", per_expert_weights=True
            )

            # Benchmark Triton fused GEMM
            print("\nBenchmarking Triton fused MXFP4 GEMM...")
            triton_time = benchmark_fused_mxfp4_gemm(hidden_3d, weights, scales, N)

            # Benchmark CuTe fused GEMM (simple version)
            print("Benchmarking CuTe fused MXFP4 GEMM (simple)...")
            cute_simple_time = benchmark_cute_fused_mxfp4_gemm(
                hidden_3d, weights, scales, N, kernel_version=0
            )

            # Benchmark CuTe fused GEMM (WMMA version)
            print("Benchmarking CuTe fused MXFP4 GEMM (WMMA)...")
            cute_wmma_time = benchmark_cute_fused_mxfp4_gemm(
                hidden_3d, weights, scales, N, kernel_version=1
            )

            # Results
            print(f"\n{'='*70}")
            print(f"RESULTS (tokens_per_expert={tpe})")
            print(f"{'='*70}")
            print(f"{'Kernel':<30} {'Time (ms)':>12} {'vs Triton':>12}")
            print("-" * 70)

            if triton_time != float('inf'):
                print(f"{'Triton fused MXFP4 GEMM':<30} {triton_time:>12.3f} {'(baseline)':>12}")
            else:
                print(f"{'Triton fused MXFP4 GEMM':<30} {'FAILED':>12}")

            if cute_simple_time != float('inf') and triton_time != float('inf'):
                speedup = triton_time / cute_simple_time
                print(f"{'CuTe fused MXFP4 (simple)':<30} {cute_simple_time:>12.3f} {speedup:>11.2f}x")
            elif cute_simple_time != float('inf'):
                print(f"{'CuTe fused MXFP4 (simple)':<30} {cute_simple_time:>12.3f}")
            else:
                print(f"{'CuTe fused MXFP4 (simple)':<30} {'FAILED':>12}")

            if cute_wmma_time != float('inf') and triton_time != float('inf'):
                speedup = triton_time / cute_wmma_time
                print(f"{'CuTe fused MXFP4 (WMMA)':<30} {cute_wmma_time:>12.3f} {speedup:>11.2f}x")
            elif cute_wmma_time != float('inf'):
                print(f"{'CuTe fused MXFP4 (WMMA)':<30} {cute_wmma_time:>12.3f}")
            else:
                print(f"{'CuTe fused MXFP4 (WMMA)':<30} {'FAILED':>12}")

            print(f"{'='*70}")

            results[tpe] = {
                "triton": triton_time,
                "cute_simple": cute_simple_time,
                "cute_wmma": cute_wmma_time,
            }

        sys.exit(0)

    # GEMM tuning mode
    if args.tune_gemm:
        all_results = []
        for tpe in args.tokens:
            results = run_gemm_tuning(tpe, config)
            all_results.extend(results)

        if args.output and all_results:
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\nResults saved to {args.output}")
        sys.exit(0)

    # Quick or full comparison mode
    all_results = {}
    for tpe in args.tokens:
        if args.compare_all:
            results = run_full_comparison(tpe, config)
        else:  # --quick
            results = run_quick_benchmark(tpe, config)
        all_results[tpe] = results

    # Summary for multiple token counts
    if len(args.tokens) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS TOKEN COUNTS")
        print(f"{'='*70}")
        print(f"{'Tokens':<10} {'Fused':>12} {'Decoupled':>12} {'Speedup':>12}")
        print("-" * 70)

        for tpe, results in all_results.items():
            if args.compare_all:
                fused = results.get("fused", {}).get("time_ms", float('inf'))
                decoupled = results.get("decoupled_v6", {}).get("time_ms", float('inf'))
            else:
                fused = results.get("fused_mxfp4", float('inf'))
                decoupled = results.get("decoupled", float('inf'))

            if fused != float('inf') and decoupled != float('inf'):
                speedup = fused / decoupled
                print(f"{tpe:<10} {fused:>12.3f} {decoupled:>12.3f} {speedup:>11.2f}x")
            else:
                print(f"{tpe:<10} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

        print(f"{'='*70}")

    # Export to CSV if requested
    if args.output:
        # Flatten results for CSV export
        csv_rows = []
        for tpe, results in all_results.items():
            row = {"tokens_per_expert": tpe}
            if args.compare_all:
                row["fused_ms"] = results.get("fused", {}).get("time_ms", "")
                row["decoupled_ms"] = results.get("decoupled_v6", {}).get("time_ms", "")
                row["dequant_ms"] = results.get("decoupled_v6", {}).get("dequant_ms", "")
                row["gemm_ms"] = results.get("decoupled_v6", {}).get("gemm_ms", "")
                row["unfused_per_expert_ms"] = results.get("unfused_per_expert", {}).get("time_ms", "")
            else:
                row["fused_ms"] = results.get("fused_mxfp4", "")
                row["decoupled_ms"] = results.get("decoupled", "")
                row["dequant_ms"] = results.get("decoupled_dequant", "")
                row["gemm_ms"] = results.get("decoupled_gemm", "")
            csv_rows.append(row)

        if csv_rows:
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
