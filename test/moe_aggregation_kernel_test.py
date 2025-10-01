import torch
import triton
import triton.language as tl


@triton.jit
def moe_fp32_accum_kernel(
    outs_ptr,  # [total_tokens * topk, hidden_dim]
    inv_idxs_ptr,  # [total_tokens * topk] - inverse permutation
    topk_weights_ptr,  # [total_tokens, topk]
    output_ptr,  # [total_tokens, hidden_dim]
    total_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for MoE expert output accumulation with fp32 precision.
    
    Now using inverse index: inv_idxs[j] tells us which position in outs
    corresponds to position j in the conceptual new_x array.
    
    For each token:
    1. Gather its topk expert outputs using inv_idxs
    2. Cast to fp32
    3. Multiply by fp32 weights
    4. Sum across topk dimension
    5. Cast back to bf16
    """
    # Each program handles one token and a chunk of hidden_dim
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    # Calculate the range of hidden dimensions this block handles
    h_start = block_id * BLOCK_SIZE
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < hidden_dim
    
    # Accumulator in fp32
    accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over topk experts for this token
    for k in range(topk):
        # Position in the conceptual new_x array
        new_x_idx = token_id * topk + k
        
        # Get which position in outs this corresponds to
        outs_idx = tl.load(inv_idxs_ptr + new_x_idx)
        
        # Load the weight for this expert
        weight_offset = token_id * topk + k
        weight = tl.load(topk_weights_ptr + weight_offset)
        weight_fp32 = weight.to(tl.float32)
        
        # Load the expert output from outs
        outs_offsets = outs_idx * hidden_dim + h_offsets
        expert_out = tl.load(outs_ptr + outs_offsets, mask=h_mask, other=0.0)
        
        # Cast to fp32, multiply by weight, and accumulate
        expert_out_fp32 = expert_out.to(tl.float32)
        weighted_out = expert_out_fp32 * weight_fp32
        accum += weighted_out
    
    # Cast back to bf16 and store
    output_offsets = token_id * hidden_dim + h_offsets
    output_bf16 = accum.to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + output_offsets, output_bf16, mask=h_mask)


def moe_fp32_accum_triton(
    outs: torch.Tensor,  # [total_tokens * topk, hidden_dim]
    idxs: torch.Tensor,  # [total_tokens * topk]
    topk_weights: torch.Tensor,  # [total_tokens, topk]
) -> torch.Tensor:
    """
    Optimized MoE accumulation that avoids intermediate tensor creation.
    
    The original code does: new_x[idxs] = outs, which means idxs defines
    where each element of outs goes. We need the inverse mapping to find
    which outs element corresponds to each position in new_x.
    
    Args:
        outs: Expert outputs in bf16, shape [total_tokens * topk, hidden_dim]
        idxs: Permutation indices - idxs[i] tells where outs[i] should go
        topk_weights: Weights for each expert per token, shape [total_tokens, topk]
    
    Returns:
        Accumulated output in bf16, shape [total_tokens, hidden_dim]
    """
    total_tokens, topk = topk_weights.shape
    hidden_dim = outs.shape[1]
    
    # Create inverse index: inv_idxs[j] tells which outs position maps to new_x[j]
    # This is the key insight - we need to invert the scatter operation
    inv_idxs = torch.empty_like(idxs)
    inv_idxs[idxs] = torch.arange(len(idxs), device=idxs.device, dtype=idxs.dtype)
    
    # Output tensor
    output = torch.empty(
        (total_tokens, hidden_dim),
        device=outs.device,
        dtype=outs.dtype
    )
    
    # Launch parameters
    BLOCK_SIZE = 128
    grid = lambda META: (
        total_tokens,
        triton.cdiv(hidden_dim, META['BLOCK_SIZE'])
    )
    
    moe_fp32_accum_kernel[grid](
        outs,
        inv_idxs,  # Use inverse index instead
        topk_weights,
        output,
        total_tokens=total_tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# ============================================================================
# Alternative: More optimized version with better memory coalescing
# ============================================================================

@triton.jit
def moe_fp32_accum_kernel_v2(
    outs_ptr,
    inv_idxs_ptr,
    topk_weights_ptr,
    output_ptr,
    total_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized version with better memory access patterns.
    Each block processes multiple tokens to improve memory coalescing.
    """
    # Program handles a block of tokens and a chunk of hidden dims
    token_block_id = tl.program_id(0)
    h_block_id = tl.program_id(1)
    
    # Token range for this block
    TOKENS_PER_BLOCK: tl.constexpr = 4
    token_start = token_block_id * TOKENS_PER_BLOCK
    
    # Hidden dim range
    h_start = h_block_id * BLOCK_SIZE
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < hidden_dim
    
    # Process each token in this block
    for t_idx in range(TOKENS_PER_BLOCK):
        token_id = token_start + t_idx
        
        # Check if this token is valid
        if token_id < total_tokens:
            # Accumulator for this token
            accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            # Accumulate over topk experts
            for k in range(topk):
                new_x_idx = token_id * topk + k
                outs_idx = tl.load(inv_idxs_ptr + new_x_idx)
                
                # topk_weights is [total_tokens, topk], need 2D indexing
                weight_offset = token_id * topk + k
                weight = tl.load(topk_weights_ptr + weight_offset).to(tl.float32)
                
                outs_offsets = outs_idx * hidden_dim + h_offsets
                expert_out = tl.load(outs_ptr + outs_offsets, mask=h_mask, other=0.0)
                accum += expert_out.to(tl.float32) * weight
            
            # Store result
            output_offsets = token_id * hidden_dim + h_offsets
            tl.store(output_ptr + output_offsets, accum.to(output_ptr.dtype.element_ty), mask=h_mask)


def moe_fp32_accum_triton_v2(
    outs: torch.Tensor,
    idxs: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Version 2 with better memory coalescing."""
    total_tokens, topk = topk_weights.shape
    hidden_dim = outs.shape[1]
    
    # Create inverse index
    inv_idxs = torch.empty_like(idxs)
    inv_idxs[idxs] = torch.arange(len(idxs), device=idxs.device, dtype=idxs.dtype)
    
    output = torch.empty((total_tokens, hidden_dim), device=outs.device, dtype=outs.dtype)
    
    BLOCK_SIZE = 128
    TOKENS_PER_BLOCK = 4
    
    grid = lambda META: (
        triton.cdiv(total_tokens, TOKENS_PER_BLOCK),
        triton.cdiv(hidden_dim, META['BLOCK_SIZE'])
    )
    
    moe_fp32_accum_kernel_v2[grid](
        outs, inv_idxs, topk_weights, output,
        total_tokens=total_tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# ============================================================================
# Benchmark and verification code
# ============================================================================

def reference_implementation(outs, idxs, topk_weights):
    """Original implementation for correctness testing."""
    total_tokens, topk = topk_weights.shape
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    
    final_out = (
        new_x.view(total_tokens, topk, -1)
        .type(topk_weights.dtype)
        .mul_(topk_weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out


def test_correctness():
    """Test that the Triton kernel matches the reference implementation."""
    torch.manual_seed(42)
    
    total_tokens = 128
    topk = 4
    hidden_dim = 512
    
    # Create test data
    outs = torch.randn(total_tokens * topk, hidden_dim, device='cuda', dtype=torch.bfloat16)
    idxs = torch.randperm(total_tokens * topk, device='cuda')[:total_tokens * topk]
    topk_weights = torch.randn(total_tokens, topk, device='cuda', dtype=torch.bfloat16)
    topk_weights = torch.softmax(topk_weights.float(), dim=-1).bfloat16()
    
    # Reference
    ref_out = reference_implementation(outs, idxs, topk_weights)
    
    # Triton v1
    triton_out = moe_fp32_accum_triton(outs, idxs, topk_weights)
    
    # Triton v2
    triton_out_v2 = moe_fp32_accum_triton_v2(outs, idxs, topk_weights)
    
    # Check correctness
    print("Max absolute difference (v1):", (ref_out - triton_out).abs().max().item())
    print("Max absolute difference (v2):", (ref_out - triton_out_v2).abs().max().item())
    print("Mean absolute difference (v1):", (ref_out - triton_out).abs().mean().item())
    print("Mean absolute difference (v2):", (ref_out - triton_out_v2).abs().mean().item())
    
    # They should be very close (within bf16 precision)
    assert torch.allclose(ref_out, triton_out, rtol=1e-2, atol=1e-2), "V1 outputs don't match!"
    assert torch.allclose(ref_out, triton_out_v2, rtol=1e-2, atol=1e-2), "V2 outputs don't match!"
    print("✓ All tests passed!")


def benchmark_kernels():
    """Benchmark all implementations with varying problem sizes."""
    import time
    
    print("=" * 80)
    print("BENCHMARK: MoE FP32 Accumulation Kernels")
    print("=" * 80)
    
    # Test configurations
    configs = [
        {"total_tokens": 1024, "topk": 2, "hidden_dim": 4096, "name": "Small (1K tokens)"},
        {"total_tokens": 10240, "topk": 4, "hidden_dim": 4096, "name": "Medium (10K tokens)"},
        {"total_tokens": 102400, "topk": 8, "hidden_dim": 4096, "name": "Large (100K tokens)"},
        {"total_tokens": 102400, "topk": 8, "hidden_dim": 8192, "name": "Large + Wide (100K tokens, 8K hidden)"},
    ]
    
    num_warmup = 10
    num_iters = 100
    
    for config in configs:
        total_tokens = config["total_tokens"]
        topk = config["topk"]
        hidden_dim = config["hidden_dim"]
        
        print(f"\n{'='*80}")
        print(f"Config: {config['name']}")
        print(f"  total_tokens={total_tokens}, topk={topk}, hidden_dim={hidden_dim}")
        print(f"  Total elements: {total_tokens * topk * hidden_dim:,}")
        print(f"{'='*80}")
        
        # Create test data
        torch.manual_seed(42)
        outs = torch.randn(total_tokens * topk, hidden_dim, device='cuda', dtype=torch.bfloat16)
        idxs = torch.randperm(total_tokens * topk, device='cuda')[:total_tokens * topk]
        topk_weights = torch.randn(total_tokens, topk, device='cuda', dtype=torch.bfloat16)
        topk_weights = torch.softmax(topk_weights.float(), dim=-1).bfloat16()
        
        # Benchmark reference implementation
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            ref_out = reference_implementation(outs, idxs, topk_weights)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iters):
            ref_out = reference_implementation(outs, idxs, topk_weights)
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - start) / num_iters * 1000  # ms
        
        # Benchmark Triton V1
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            triton_out = moe_fp32_accum_triton(outs, idxs, topk_weights)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iters):
            triton_out = moe_fp32_accum_triton(outs, idxs, topk_weights)
        torch.cuda.synchronize()
        triton_v1_time = (time.perf_counter() - start) / num_iters * 1000  # ms
        
        # Benchmark Triton V2
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            triton_out_v2 = moe_fp32_accum_triton_v2(outs, idxs, topk_weights)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iters):
            triton_out_v2 = moe_fp32_accum_triton_v2(outs, idxs, topk_weights)
        torch.cuda.synchronize()
        triton_v2_time = (time.perf_counter() - start) / num_iters * 1000  # ms
        
        # Calculate speedups
        v1_speedup = ref_time / triton_v1_time
        v2_speedup = ref_time / triton_v2_time
        
        # Calculate bandwidth
        # Data moved: read outs (bf16), read weights (bf16), write output (bf16)
        # Also read idxs (int32) once for inverse index creation
        bytes_per_iter = (
            total_tokens * topk * hidden_dim * 2 +  # outs (bf16)
            total_tokens * topk * 2 +  # weights (bf16)
            total_tokens * hidden_dim * 2 +  # output (bf16)
            total_tokens * topk * 4  # idxs (int32)
        )
        gb_per_iter = bytes_per_iter / 1e9
        
        ref_bandwidth = gb_per_iter / (ref_time / 1000)
        v1_bandwidth = gb_per_iter / (triton_v1_time / 1000)
        v2_bandwidth = gb_per_iter / (triton_v2_time / 1000)
        
        # Print results
        print(f"\nResults (averaged over {num_iters} iterations):")
        print(f"  Reference:   {ref_time:8.3f} ms  ({ref_bandwidth:6.2f} GB/s)")
        print(f"  Triton V1:   {triton_v1_time:8.3f} ms  ({v1_bandwidth:6.2f} GB/s)  [{v1_speedup:5.2f}x speedup]")
        print(f"  Triton V2:   {triton_v2_time:8.3f} ms  ({v2_bandwidth:6.2f} GB/s)  [{v2_speedup:5.2f}x speedup]")
        
        # Verify correctness
        max_diff_v1 = (ref_out - triton_out).abs().max().item()
        max_diff_v2 = (ref_out - triton_out_v2).abs().max().item()
        print(f"\nCorrectness (max absolute difference):")
        print(f"  V1: {max_diff_v1:.6f}")
        print(f"  V2: {max_diff_v2:.6f}")
        
        if max_diff_v1 > 0.1 or max_diff_v2 > 0.1:
            print("  ⚠️  WARNING: Large numerical differences detected!")
        else:
            print("  ✓ All kernels match reference within tolerance")


def profile_kernel(func_name="moe_fp32_accum_triton", total_tokens=102400, topk=8, hidden_dim=4096):
    """Profile a specific kernel with torch profiler."""
    import torch.profiler as profiler
    
    print(f"\n{'='*80}")
    print(f"PROFILING: {func_name}")
    print(f"  total_tokens={total_tokens}, topk={topk}, hidden_dim={hidden_dim}")
    print(f"{'='*80}\n")
    
    # Create test data
    torch.manual_seed(42)
    outs = torch.randn(total_tokens * topk, hidden_dim, device='cuda', dtype=torch.bfloat16)
    idxs = torch.randperm(total_tokens * topk, device='cuda')[:total_tokens * topk]
    topk_weights = torch.randn(total_tokens, topk, device='cuda', dtype=torch.bfloat16)
    topk_weights = torch.softmax(topk_weights.float(), dim=-1).bfloat16()
    
    # Select function
    if func_name == "reference":
        func = reference_implementation
    elif func_name == "moe_fp32_accum_triton":
        func = moe_fp32_accum_triton
    elif func_name == "moe_fp32_accum_triton_v2":
        func = moe_fp32_accum_triton_v2
    else:
        raise ValueError(f"Unknown function: {func_name}")
    
    # Profile
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            result = func(outs, idxs, topk_weights)
            torch.cuda.synchronize()
    
    # Print results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Save trace for visualization
    trace_file = f"trace_{func_name}.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace saved to: {trace_file}")
    print("View with: chrome://tracing")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_kernels()
    elif len(sys.argv) > 1 and sys.argv[1] == "profile":
        func_name = sys.argv[2] if len(sys.argv) > 2 else "moe_fp32_accum_triton"
        profile_kernel(func_name)
    else:
        test_correctness()