#!/usr/bin/env python3
"""NCU Profiling Script for CuTe Fused MXFP4 GEMM V6.

This script runs the V6 kernel in a way suitable for NVIDIA Nsight Compute profiling.

Usage:
    # Profile with default metrics (recommended first)
    ncu --set full -o v6_profile python profile_v6_ncu.py

    # Profile specific metrics
    ncu --metrics \
        sm__warps_active.avg.pct_of_peak_sustained_active,\
        sm__cycles_active.avg,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        l2__throughput.avg.pct_of_peak_sustained_elapsed,\
        gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
        python profile_v6_ncu.py

    # Detailed memory analysis
    ncu --section MemoryWorkloadAnalysis \
        --section ComputeWorkloadAnalysis \
        --section Occupancy \
        --section WarpStateStatistics \
        -o v6_detailed \
        python profile_v6_ncu.py

Key metrics to examine in NCU:
    1. Occupancy (sm__warps_active.avg.pct_of_peak_sustained_active)
       - V6 target: >50% with 27KB smem

    2. Memory Throughput (dram__throughput.avg.pct_of_peak_sustained_elapsed)
       - Target: >50% for memory-bound kernel
       - V6 should be limited by DRAM bandwidth

    3. Compute Throughput (sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active)
       - Shows WMMA/tensor core utilization

    4. Stall Reasons (smsp__warp_issue_stalled_*)
       - barrier_sync: __syncthreads() overhead
       - long_scoreboard: Memory latency
       - wait: Instruction dependencies

    5. L2 Hit Rate (lts__t_sector_hit_rate.pct)
       - High hit rate = good data reuse
"""

import sys
import torch

sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")

from batchgen.moe.cute_fused_mxfp4_gemm_v6 import cute_grouped_mxfp4_gemm_3d_v6
from batchgen.moe.mxfp4_grouped_gemm import setup_expert_weight_pointers


def main():
    # GPT-OSS-120B dimensions
    num_experts = 128
    hidden_size = 5120      # K
    intermediate_size = 13824  # N
    tokens_per_expert = 4

    print(f"NCU Profiling: CuTe Fused MXFP4 GEMM V6")
    print(f"  Experts: {num_experts}")
    print(f"  Tokens/Expert: {tokens_per_expert}")
    print(f"  K={hidden_size}, N={intermediate_size}")
    print()

    # Create test tensors
    K = hidden_size
    N = intermediate_size

    hidden_3d = torch.randn(
        num_experts, tokens_per_expert, K,
        dtype=torch.bfloat16, device="cuda"
    )

    # Create per-expert weights (packed FP4) and scales
    weights = []
    scales = []
    for _ in range(num_experts):
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        s = torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device="cuda")
        weights.append(w)
        scales.append(s)

    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device="cuda"
    )

    print("Warming up (3 iterations)...")
    for i in range(3):
        _ = cute_grouped_mxfp4_gemm_3d_v6(
            hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
            N, weights[0], scales[0]
        )
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/3 complete")

    print()
    print("Running profiling iteration...")
    print("=" * 60)

    # Single iteration for NCU profiling
    # NCU will capture this kernel launch
    torch.cuda.synchronize()
    output = cute_grouped_mxfp4_gemm_3d_v6(
        hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
        N, weights[0], scales[0]
    )
    torch.cuda.synchronize()

    print("=" * 60)
    print("Profiling complete!")
    print()
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Print expected kernel statistics
    print()
    print("Expected V6 kernel characteristics:")
    print("  - Grid: (total_m_blocks, cdiv(N, 64)) = (~512, 216)")
    print("  - Block: 256 threads (8 warps)")
    print("  - Shared memory: ~27KB per block")
    print("  - Syncs per block: 160 (2 per K-iteration × 80 K-blocks)")
    print()
    print("Key bottleneck hypotheses:")
    print("  1. __syncthreads() overhead (160 syncs/block)")
    print("  2. WMMA tensor core underutilization")
    print("  3. Memory coalescing issues in FP4 decode")
    print("  4. Register pressure from WMMA fragments")


if __name__ == "__main__":
    main()
