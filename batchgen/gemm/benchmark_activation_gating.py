import torch
import triton
import time
from typing import Tuple, List
import numpy as np
import triton
import triton.language as tl

@triton.jit
def activation_gating_kernel(
    gate_acc_ptr,
    up_acc_ptr,
    output_ptr,
    M, N: tl.constexpr,
    stride_gate_m, stride_gate_n,
    stride_up_m, stride_up_n,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for SiLU activation and gating, output in bfloat16.
    
    Operations:
    1. gate_activated = silu(gate_acc) where silu(x) = x / (1 + exp(-x))
    2. intermediate = gate_activated * up_acc
    3. Convert to bfloat16
    """
    # Program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load gate_acc and up_acc (float32)
    gate_ptrs = gate_acc_ptr + (offs_m[:, None] * stride_gate_m + offs_n[None, :] * stride_gate_n)
    up_ptrs = up_acc_ptr + (offs_m[:, None] * stride_up_m + offs_n[None, :] * stride_up_n)
    
    gate_acc = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up_acc = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # SiLU activation: silu(x) = x / (1 + exp(-x))
    gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
    
    # Gating: element-wise multiplication
    intermediate = gate_activated * up_acc
    
    # Convert to bfloat16
    output_bf16 = intermediate.to(tl.bfloat16)
    
    # Store output
    output_ptrs = output_ptr + (offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n)
    tl.store(output_ptrs, output_bf16, mask=mask)


@torch.inference_mode()
def activation_gating(
    gate_acc: torch.Tensor,
    up_acc: torch.Tensor,
    block_size_m: int = 32,
    block_size_n: int = 32,
    num_warps: int = 4,
):
    """
    Apply SiLU activation and gating, returning bfloat16 output.
    
    Equivalent to:
        gate_activated = torch.nn.functional.silu(gate_acc)
        intermediate = gate_activated * up_acc
        intermediate = intermediate.to(torch.bfloat16)
    
    Args:
        gate_acc: (M, N) float32 tensor - gate projection accumulator
        up_acc: (M, N) float32 tensor - up projection accumulator
        block_size_m: Block size for M dimension
        block_size_n: Block size for N dimension
        num_warps: Number of warps for kernel execution
    
    Returns:
        intermediate: (M, N) bfloat16 tensor - activated and gated result
    """
    M, N = gate_acc.shape
    assert up_acc.shape == (M, N), "gate_acc and up_acc must have same shape"
    # assert gate_acc.dtype == torch.float32, "gate_acc must be float32"
    # assert up_acc.dtype == torch.float32, "up_acc must be float32"
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=gate_acc.device)
    
    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))
    
    activation_gating_kernel[grid](
        gate_acc, up_acc, output,
        M, N,
        gate_acc.stride(0), gate_acc.stride(1),
        up_acc.stride(0), up_acc.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        num_warps=num_warps,
    )
    
    return output

def benchmark_activation_gating(
    gate_acc: torch.Tensor,
    up_acc: torch.Tensor,
    block_size_m: int,
    block_size_n: int,
    num_warps: int = 4,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Tuple[float, float]:
    """
    Benchmark the activation gating kernel.
    
    Returns:
        (avg_time_ms, bandwidth_gb_s): Average time in ms and bandwidth in GB/s
    """
    # Warmup
    for _ in range(warmup_iterations):
        _ = activation_gating(gate_acc, up_acc, block_size_m, block_size_n, num_warps)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        _ = activation_gating(gate_acc, up_acc, block_size_m, block_size_n, num_warps)
    end_event.record()
    
    torch.cuda.synchronize()
    
    avg_time_ms = start_event.elapsed_time(end_event) / num_iterations
    
    # Calculate bandwidth (GB/s)
    # Read: 2 float32 tensors (gate_acc, up_acc) = 2 * M * N * 4 bytes
    # Write: 1 bfloat16 tensor (output) = M * N * 2 bytes
    M, N = gate_acc.shape
    bytes_transferred = (2 * M * N * 4 + M * N * 2)  # bytes
    bandwidth_gb_s = (bytes_transferred / 1e9) / (avg_time_ms / 1000)
    
    return avg_time_ms, bandwidth_gb_s


def run_tile_size_sweep(
    M_sizes: List[int],
    N_sizes: List[int],
    tile_configs: List[Tuple[int, int]],
    num_warps_list: List[int] = [4],
    device: str = "cuda"
):
    """
    Run comprehensive tile size benchmark sweep.
    
    Args:
        M_sizes: List of M dimensions to test
        N_sizes: List of N dimensions to test
        tile_configs: List of (BLOCK_SIZE_M, BLOCK_SIZE_N) tuples
        num_warps_list: List of num_warps to test
        device: Device to run on
    """
    print("=" * 100)
    print("ACTIVATION GATING KERNEL - TILE SIZE BENCHMARK")
    print("=" * 100)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Testing {len(M_sizes) * len(N_sizes)} problem sizes with {len(tile_configs)} tile configurations")
    print("=" * 100)
    
    for M in M_sizes:
        for N in N_sizes:
            print(f"\n{'=' * 100}")
            print(f"Problem Size: M={M}, N={N}")
            print(f"{'=' * 100}")
            
            # Create input tensors
            gate_acc = torch.randn((M, N), dtype=torch.float32, device=device)
            up_acc = torch.randn((M, N), dtype=torch.float32, device=device)
            
            # Calculate theoretical peak bandwidth for reference
            # A100: ~1555 GB/s, H100: ~3350 GB/s
            peak_bandwidth_gb_s = 1555.0  # Adjust for your GPU
            
            results = []
            
            for num_warps in num_warps_list:
                for block_m, block_n in tile_configs:
                    try:
                        avg_time_ms, bandwidth_gb_s = benchmark_activation_gating(
                            gate_acc, up_acc, block_m, block_n, num_warps
                        )
                        
                        bandwidth_util = (bandwidth_gb_s / peak_bandwidth_gb_s) * 100
                        
                        results.append({
                            'block_m': block_m,
                            'block_n': block_n,
                            'num_warps': num_warps,
                            'time_ms': avg_time_ms,
                            'bandwidth_gb_s': bandwidth_gb_s,
                            'bandwidth_util': bandwidth_util
                        })
                    except Exception as e:
                        print(f"Failed: BLOCK_M={block_m}, BLOCK_N={block_n}, warps={num_warps}: {e}")
            
            # Sort by time (best first)
            results.sort(key=lambda x: x['time_ms'])
            
            # Print results table
            print(f"\n{'Tile Config':<20} {'Warps':<8} {'Time (ms)':<12} {'BW (GB/s)':<12} {'BW Util %':<12} {'Speedup':<10}")
            print("-" * 100)
            
            baseline_time = results[0]['time_ms']
            
            for r in results:
                tile_str = f"({r['block_m']}, {r['block_n']})"
                speedup = baseline_time / r['time_ms']
                
                marker = "⭐" if speedup == 1.0 else ""
                
                print(f"{tile_str:<20} {r['num_warps']:<8} {r['time_ms']:<12.4f} "
                      f"{r['bandwidth_gb_s']:<12.2f} {r['bandwidth_util']:<12.1f} "
                      f"{speedup:<10.2f}x {marker}")
            
            # Print best configuration
            best = results[0]
            print(f"\n{'🏆 Best Config':<20} ({best['block_m']}, {best['block_n']}) with {best['num_warps']} warps: "
                  f"{best['time_ms']:.4f} ms, {best['bandwidth_gb_s']:.2f} GB/s ({best['bandwidth_util']:.1f}% util)")


def run_quick_benchmark():
    """Quick benchmark with common configurations."""
    
    # Common tile sizes to test
    tile_configs = [
        (32, 32),
        (32, 64),
        (32, 128),
        (32, 256),
        (64, 32),
        (64, 64),
        (64, 128),
        (64, 256),
        (128, 32),
        (128, 64),
        (128, 128),
        (128, 256),
        (256, 64),
        (256, 128),
    ]
    
    # Problem sizes (typical for MoE)
    M_sizes = [368, 512, 1024, 2048]
    N_sizes = [7168, 14336]
    
    # Warp counts to test
    num_warps_list = [4, 8]
    
    run_tile_size_sweep(M_sizes, N_sizes, tile_configs, num_warps_list)


def run_detailed_benchmark():
    """Detailed benchmark with more configurations."""
    
    # Extensive tile configurations
    tile_configs = []
    for m in [16, 32, 64, 128, 256]:
        for n in [32, 64, 128, 256, 512]:
            tile_configs.append((m, n))
    
    # Various problem sizes
    M_sizes = [128, 256, 368, 512, 1024, 2048, 4096]
    N_sizes = [4096, 7168, 14336, 28672]
    
    # Multiple warp configurations
    num_warps_list = [2, 4, 8]
    
    run_tile_size_sweep(M_sizes, N_sizes, tile_configs, num_warps_list)


def analyze_tile_aspect_ratio():
    """Analyze the impact of tile aspect ratio."""
    print("\n" + "=" * 100)
    print("TILE ASPECT RATIO ANALYSIS")
    print("=" * 100)
    
    M, N = 2048, 14336
    gate_acc = torch.randn((M, N), dtype=torch.float32, device="cuda")
    up_acc = torch.randn((M, N), dtype=torch.float32, device="cuda")
    
    # Test different aspect ratios with same area
    configs = [
        (32, 128, "1:4"),    # Area = 4096
        (64, 64, "1:1"),     # Area = 4096
        (128, 32, "4:1"),    # Area = 4096
        
        (32, 256, "1:8"),    # Area = 8192
        (64, 128, "1:2"),    # Area = 8192
        (128, 64, "2:1"),    # Area = 8192
        (256, 32, "8:1"),    # Area = 8192
    ]
    
    print(f"\nProblem Size: M={M}, N={N}")
    print(f"\n{'Tile Config':<20} {'Aspect':<10} {'Area':<10} {'Time (ms)':<12} {'BW (GB/s)':<12} {'Relative':<10}")
    print("-" * 100)
    
    results = []
    for block_m, block_n, aspect in configs:
        avg_time_ms, bandwidth_gb_s = benchmark_activation_gating(
            gate_acc, up_acc, block_m, block_n
        )
        results.append((block_m, block_n, aspect, avg_time_ms, bandwidth_gb_s))
    
    baseline = results[0][3]
    for block_m, block_n, aspect, time_ms, bw in results:
        area = block_m * block_n
        relative = time_ms / baseline
        print(f"({block_m}, {block_n}){'':<11} {aspect:<10} {area:<10} "
              f"{time_ms:<12.4f} {bw:<12.2f} {relative:<10.2f}x")


def profile_register_pressure():
    """Analyze register usage vs performance."""
    print("\n" + "=" * 100)
    print("REGISTER PRESSURE vs PERFORMANCE")
    print("=" * 100)
    
    M, N = 1024, 14336
    gate_acc = torch.randn((M, N), dtype=torch.float32, device="cuda")
    up_acc = torch.randn((M, N), dtype=torch.float32, device="cuda")
    
    # Configs with increasing register pressure
    configs = [
        (32, 32, "Low"),
        (64, 64, "Medium-Low"),
        (128, 128, "Medium"),
        (128, 256, "High"),
        (256, 256, "Very High"),
    ]
    
    print(f"\nProblem Size: M={M}, N={N}")
    print(f"\n{'Tile Config':<20} {'Pressure':<15} {'Time (ms)':<12} {'BW (GB/s)':<12} {'Speedup':<10}")
    print("-" * 100)
    
    results = []
    for block_m, block_n, pressure in configs:
        try:
            avg_time_ms, bandwidth_gb_s = benchmark_activation_gating(
                gate_acc, up_acc, block_m, block_n
            )
            results.append((block_m, block_n, pressure, avg_time_ms, bandwidth_gb_s))
        except Exception as e:
            print(f"({block_m}, {block_n}){'':<11} {pressure:<15} FAILED: {e}")
    
    if results:
        baseline = results[0][3]
        for block_m, block_n, pressure, time_ms, bw in results:
            speedup = baseline / time_ms
            marker = "⭐" if speedup == max(r[3] / baseline for r in results) else ""
            print(f"({block_m}, {block_n}){'':<11} {pressure:<15} "
                  f"{time_ms:<12.4f} {bw:<12.2f} {speedup:<10.2f}x {marker}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark activation gating kernel tile sizes")
    parser.add_argument("--mode", choices=["quick", "detailed", "aspect", "register", "all"], 
                        default="quick", help="Benchmark mode")
    
    args = parser.parse_args()
    
    if args.mode == "quick" or args.mode == "all":
        run_quick_benchmark()
    
    if args.mode == "detailed":
        run_detailed_benchmark()
    
    if args.mode == "aspect" or args.mode == "all":
        analyze_tile_aspect_ratio()
    
    if args.mode == "register" or args.mode == "all":
        profile_register_pressure()
    
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)