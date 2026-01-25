"""Hyperparameter search for MXFP4 grouped GEMM kernel tile sizes.

This script searches for optimal BLOCK_M, BLOCK_N, num_warps, and num_stages
configurations for the MXFP4 grouped GEMM kernel used in GPT-OSS-120B MoE layers.

Note: BLOCK_K=32 is fixed due to MXFP4 scale block size constraint.

Usage:
    # Quick search with default batch sizes
    python bench_mxfp4_grouped_gemm.py --tokens 1 8 32

    # Full search
    python bench_mxfp4_grouped_gemm.py --tokens 1 4 8 16 32 64 --output full_search.csv
"""

import torch
import triton
import itertools
import time
import argparse
import sys
from typing import List, Dict, Tuple, Any

# Problem sizes matching GPT-OSS-120B
DEFAULT_CONFIGS = {
    "hidden_size": 5120,
    "intermediate_size": 13824,
    "num_experts": 128,
}

# Search space
BLOCK_M_VALUES = [32, 64, 128, 256]
BLOCK_N_VALUES = [32, 64, 128, 256]
BLOCK_K_VALUES = [32]  # Fixed for MXFP4 (must match scale block size)
NUM_WARPS_VALUES = [2, 4, 8]
NUM_STAGES_VALUES = [1, 2, 3, 4]


def create_test_tensors(
    num_experts: int,
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test tensors matching GPT-OSS-120B dimensions.

    Args:
        num_experts: Number of experts (128 for GPT-OSS-120B)
        tokens_per_expert: Tokens routed to each expert
        hidden_size: Model hidden dimension (5120)
        intermediate_size: MLP intermediate dimension (13824)
        device: Device to create tensors on

    Returns:
        hidden_3d: [num_experts, tokens_per_expert, hidden_size] BF16
        weight: [intermediate_size, hidden_size//2] uint8 (packed FP4)
        scale: [intermediate_size, hidden_size//32] uint8
        expert_counts: [num_experts] int32
    """
    # Input: [num_experts, tokens_per_expert, hidden_size]
    hidden_3d = torch.randn(
        num_experts, tokens_per_expert, hidden_size,
        dtype=torch.bfloat16, device=device
    )

    # MXFP4 weights: [N, K//2] packed (2 FP4 values per byte)
    weight = torch.randint(
        0, 256, (intermediate_size, hidden_size // 2),
        dtype=torch.uint8, device=device
    )

    # Scales: [N, K//32] (one scale per 32 K values)
    scale = torch.randint(
        120, 134, (intermediate_size, hidden_size // 32),
        dtype=torch.uint8, device=device
    )

    # Expert token counts (uniform distribution for benchmarking)
    expert_counts = torch.full(
        (num_experts,), tokens_per_expert,
        dtype=torch.int32, device=device
    )

    return hidden_3d, weight, scale, expert_counts


def benchmark_config(
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
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> Tuple[float, bool]:
    """Benchmark a single configuration.

    Args:
        hidden_3d: Input tensor [E, M_max, K]
        weight_ptrs: Pointer array to weight tensors
        scale_ptrs: Pointer array to scale tensors
        expert_counts: Token counts per expert
        N: Output dimension
        weight_ref: Reference weight for strides
        scale_ref: Reference scale for strides
        BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes
        num_warps: Number of warps per block
        num_stages: Number of pipeline stages
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations

    Returns:
        (time_ms, success): Average time in ms and success flag
    """
    from batchgen.moe.mxfp4_grouped_gemm import grouped_mxfp4_gemm_3d_tunable

    try:
        # Warmup - also catches compilation errors
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
        elapsed = (time.perf_counter() - start) / bench_iters * 1000  # ms

        return elapsed, True

    except Exception as e:
        print(f"  Config failed: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, "
              f"warps={num_warps}, stages={num_stages}: {e}")
        return float('inf'), False


def run_search(
    tokens_per_expert: int,
    device: str = "cuda",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run hyperparameter search for a given batch size.

    Args:
        tokens_per_expert: Number of tokens per expert
        device: CUDA device
        verbose: Print progress

    Returns:
        List of result dictionaries
    """
    cfg = DEFAULT_CONFIGS

    if verbose:
        print(f"\n{'='*60}")
        print(f"Tokens per expert: {tokens_per_expert}")
        print(f"Hidden: {cfg['hidden_size']}, Intermediate: {cfg['intermediate_size']}")
        print(f"Num experts: {cfg['num_experts']}")
        print(f"{'='*60}")

    # Create test tensors
    hidden_3d, weight, scale, expert_counts = create_test_tensors(
        cfg['num_experts'], tokens_per_expert,
        cfg['hidden_size'], cfg['intermediate_size'], device
    )

    # Setup weight pointers (simulate 128 experts with same weight for benchmark)
    from batchgen.moe.mxfp4_grouped_gemm import setup_expert_weight_pointers
    weight_list = [weight] * cfg['num_experts']
    scale_list = [scale] * cfg['num_experts']
    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weight_list, scale_list)

    N = cfg['intermediate_size']

    results = []

    # Grid search over all configurations
    configs = list(itertools.product(
        BLOCK_M_VALUES, BLOCK_N_VALUES, BLOCK_K_VALUES,
        NUM_WARPS_VALUES, NUM_STAGES_VALUES
    ))

    if verbose:
        print(f"Testing {len(configs)} configurations...")

    for i, (bm, bn, bk, nw, ns) in enumerate(configs):
        time_ms, success = benchmark_config(
            hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
            N, weight, scale,
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

    # Sort by time
    results.sort(key=lambda x: x['time_ms'])

    # Print top 5
    if verbose:
        print(f"\nTop 5 configurations for tokens_per_expert={tokens_per_expert}:")
        print("-" * 70)
        for r in results[:5]:
            if r['success']:
                print(f"  BLOCK_M={r['BLOCK_M']:3d}, BLOCK_N={r['BLOCK_N']:3d}, "
                      f"warps={r['num_warps']}, stages={r['num_stages']}: "
                      f"{r['time_ms']:.3f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MXFP4 Grouped GEMM Hyperparameter Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick search
    python bench_mxfp4_grouped_gemm.py --tokens 1 8 32

    # Full search with all batch sizes
    python bench_mxfp4_grouped_gemm.py --tokens 1 4 8 16 32 64 --output full.csv

    # Single batch size, quick test
    python bench_mxfp4_grouped_gemm.py --tokens 8
"""
    )
    parser.add_argument(
        "--tokens", type=int, nargs='+', default=[1, 8, 32],
        help="Tokens per expert to test (default: 1 8 32)"
    )
    parser.add_argument(
        "--output", type=str, default="mxfp4_gemm_tuning.csv",
        help="Output CSV file (default: mxfp4_gemm_tuning.csv)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Search space: BLOCK_M={BLOCK_M_VALUES}, BLOCK_N={BLOCK_N_VALUES}")
    print(f"              num_warps={NUM_WARPS_VALUES}, num_stages={NUM_STAGES_VALUES}")
    print(f"Total configs per batch size: {len(BLOCK_M_VALUES) * len(BLOCK_N_VALUES) * len(NUM_WARPS_VALUES) * len(NUM_STAGES_VALUES)}")

    all_results = []

    for tpe in args.tokens:
        results = run_search(tpe, verbose=not args.quiet)
        all_results.extend(results)

    # Save results to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    except ImportError:
        # Fallback: save as simple CSV without pandas
        import csv
        with open(args.output, 'w', newline='') as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        print(f"\nResults saved to {args.output}")

    # Print best config per batch size
    print("\n" + "=" * 60)
    print("BEST CONFIGURATIONS SUMMARY")
    print("=" * 60)

    for tpe in args.tokens:
        subset = [r for r in all_results if r['tokens_per_expert'] == tpe and r['success']]
        if subset:
            best = min(subset, key=lambda x: x['time_ms'])
            print(f"tokens={tpe:3d}: BLOCK_M={best['BLOCK_M']:3d}, "
                  f"BLOCK_N={best['BLOCK_N']:3d}, "
                  f"warps={best['num_warps']}, stages={best['num_stages']}, "
                  f"time={best['time_ms']:.3f}ms")
        else:
            print(f"tokens={tpe:3d}: No successful configurations")

    print("=" * 60)


if __name__ == "__main__":
    main()
