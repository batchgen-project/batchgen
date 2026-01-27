#!/usr/bin/env python3
"""NCU profiling script for cute_simple kernel.

Run with:
    ncu --set full -o cute_simple_profile python profile_cute_simple.py

Or for quick metrics:
    ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,dram__throughput.avg_pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_elapsed python profile_cute_simple.py
"""

import sys
import torch

sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")

from batchgen.moe.cute_mxfp4_dequant import batch_mxfp4_dequant_cute

def main():
    print("CuTe cute_simple NCU Profiling")
    print("=" * 60)

    # GPT-OSS-120B dimensions
    num_experts = 128
    N = 13824
    K = 5120
    device = "cuda"

    print(f"Config: {num_experts} experts, N={N}, K={K}")

    # Create test data
    print("Creating test tensors...")
    weights = [
        torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        for _ in range(num_experts)
    ]
    # K-major scales: [K//32, N]
    scales = [
        torch.randint(120, 134, (K // 32, N), dtype=torch.uint8, device=device)
        for _ in range(num_experts)
    ]

    # Pointer arrays
    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weights], dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scales], dtype=torch.int64, device=device
    )

    # Output buffer
    output = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    # Warmup (compile and cache)
    print("Warming up...")
    for _ in range(3):
        batch_mxfp4_dequant_cute(
            weight_ptrs, scale_ptrs, output,
            weights[0], scales[0], kernel_version=0
        )
    torch.cuda.synchronize()

    # Profile run
    print("Running kernel for profiling...")
    batch_mxfp4_dequant_cute(
        weight_ptrs, scale_ptrs, output,
        weights[0], scales[0], kernel_version=0
    )
    torch.cuda.synchronize()

    print("Done. Check NCU output for metrics.")

    # Print data movement stats
    packed_bytes = num_experts * N * (K // 2)
    scale_bytes = num_experts * N * (K // 32)
    output_bytes = num_experts * N * K * 2
    total_bytes = packed_bytes + scale_bytes + output_bytes
    print(f"\nData movement:")
    print(f"  Packed FP4: {packed_bytes / 1e9:.2f} GB")
    print(f"  Scales: {scale_bytes / 1e9:.2f} GB")
    print(f"  Output BF16: {output_bytes / 1e9:.2f} GB")
    print(f"  Total: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
