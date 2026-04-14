import torch
import triton
import triton.language as tl


@triton.jit
def scatter_weight_reduce_optimized_kernel(
    # Input pointers
    res_ptr,                    # [nnz, hidden_size]
    nnz_indices_ptr,            # [num_tokens, num_experts_per_tok] - mapping to nnz indices (-1 if empty)
    topk_weight_ptr,            # [num_tokens, num_experts_per_tok]
    # Output pointer
    output_ptr,                 # [num_tokens, hidden_size]
    # Dimensions
    num_tokens,
    num_experts_per_tok,
    hidden_size,
    nnz,                        # Total number of non-zero entries (for bounds checking)
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Optimized version that uses pre-computed inverse mapping.
    This avoids scanning all nnz entries for each token.
    """
    token_idx = tl.program_id(0)
    
    if token_idx >= num_tokens:
        return
    
    h_offset = tl.program_id(1) * BLOCK_SIZE_H
    h_indices = h_offset + tl.arange(0, BLOCK_SIZE_H)
    h_mask = h_indices < hidden_size
    
    accumulator = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    
    # Only loop over experts for this specific token
    for k in range(num_experts_per_tok):
        # Get the nnz index for this token's k-th expert
        mapping_offset = token_idx * num_experts_per_tok + k
        nnz_idx = tl.load(nnz_indices_ptr + mapping_offset)
        
        # Create mask for valid entries (use mask instead of if statement)
        is_valid = (nnz_idx >= 0) & (nnz_idx < nnz)
        
        # Load weight (masked)
        weight = tl.load(topk_weight_ptr + mapping_offset)
        
        # Load result values with proper masking
        # Use tl.where to handle invalid indices safely
        safe_nnz_idx = tl.where(is_valid, nnz_idx, 0)  # Use 0 as safe fallback
        res_offset = safe_nnz_idx * hidden_size + h_indices
        
        # Load with combined mask: valid entry AND within hidden_size bounds
        load_mask = h_mask & is_valid
        res_vals = tl.load(res_ptr + res_offset, mask=load_mask, other=0.0)
        
        # Convert to FP32 and accumulate
        res_vals_fp32 = res_vals.to(tl.float32)
        
        # Only accumulate if valid (weight is already 0 for invalid entries conceptually)
        weighted = tl.where(is_valid, res_vals_fp32 * weight, 0.0)
        accumulator += weighted
    
    # Write result
    output_offset = token_idx * hidden_size + h_indices
    tl.store(output_ptr + output_offset, accumulator, mask=h_mask)


def build_inverse_mapping(
    global_indices: torch.Tensor,     # [nnz]
    token_topk_pos: torch.Tensor,     # [nnz]
    num_tokens: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Build inverse mapping: [num_tokens, num_experts_per_tok] -> nnz_idx"""
    # Use int64 for better compatibility with Triton indexing
    mapping = torch.full((num_tokens, num_experts_per_tok), -1, 
                         dtype=torch.int64, device=global_indices.device)
    
    # Ensure indices are within bounds
    assert global_indices.max() < num_tokens, "global_indices out of bounds"
    assert token_topk_pos.max() < num_experts_per_tok, "token_topk_pos out of bounds"
    
    mapping[global_indices, token_topk_pos] = torch.arange(
        len(global_indices), dtype=torch.int64, device=global_indices.device
    )
    return mapping


def scatter_weight_reduce_optimized(
    res: torch.Tensor,
    global_indices: torch.Tensor,
    token_topk_pos: torch.Tensor,
    topk_weight: torch.Tensor,
    num_tokens: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Optimized version using inverse mapping."""
    assert topk_weight.dtype == torch.float32, "topk_weight must be float32"
    assert topk_weight.shape == (num_tokens, num_experts_per_tok), "topk_weight shape mismatch"
    
    nnz, hidden_size = res.shape
    
    # Build inverse mapping (can be cached if indices don't change)
    nnz_indices = build_inverse_mapping(
        global_indices, token_topk_pos, num_tokens, num_experts_per_tok
    )
    
    output = torch.zeros((num_tokens, hidden_size), device=res.device, dtype=torch.float32)
    
    # Adaptive block size
    BLOCK_SIZE_H = min(triton.next_power_of_2(hidden_size), 256)
    grid = (num_tokens, triton.cdiv(hidden_size, BLOCK_SIZE_H))
    
    scatter_weight_reduce_optimized_kernel[grid](
        res, nnz_indices, topk_weight,
        output,
        num_tokens, num_experts_per_tok, hidden_size, nnz,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return output

import torch
import time


def torch_reference_impl(
    res: torch.Tensor,
    global_indices: torch.Tensor,
    token_topk_pos: torch.Tensor,
    topk_weight: torch.Tensor,
    num_tokens: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Reference implementation matching your original code."""
    hidden_size = res.shape[1]
    device = res.device
    
    global_results = torch.zeros(
        (num_tokens, num_experts_per_tok, hidden_size),
        device=device, 
        dtype=res.dtype
    )
    
    # Scatter
    global_results[global_indices, token_topk_pos, :] = res
    
    # FP32 weighting and reduction
    assert topk_weight.dtype == torch.float32
    weighted_output = global_results.to(torch.float32) * topk_weight.unsqueeze(-1)
    output = weighted_output.sum(dim=1)
    
    return output


def test_correctness():
    """Test correctness with various scenarios."""
    print("=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    test_cases = [
        # (num_tokens, num_experts_per_tok, hidden_size, nnz)
        (128, 2, 256, 256),
        (512, 4, 1024, 2048),
        (1024, 8, 2048, 8192),
        (64, 2, 128, 128),  # Exact coverage
        (100, 3, 512, 150), # Sparse
    ]
    
    for num_tokens, num_experts_per_tok, hidden_size, nnz in test_cases:
        print(f"\nTest: tokens={num_tokens}, K={num_experts_per_tok}, "
              f"hidden={hidden_size}, nnz={nnz}")
        
        # Generate data
        res = torch.randn(nnz, hidden_size, device=device, dtype=torch.bfloat16)
        
        # Generate valid indices
        global_indices = torch.randint(0, num_tokens, (nnz,), device=device)
        token_topk_pos = torch.randint(0, num_experts_per_tok, (nnz,), device=device)
        
        # Weights
        topk_weight = torch.randn(num_tokens, num_experts_per_tok, 
                                  device=device, dtype=torch.float32)
        
        # Reference
        torch_output = torch_reference_impl(
            res, global_indices, token_topk_pos, topk_weight,
            num_tokens, num_experts_per_tok
        )
        
        # Triton
        triton_output = scatter_weight_reduce_optimized(
            res, global_indices, token_topk_pos, topk_weight,
            num_tokens, num_experts_per_tok
        )
        
        # Check
        max_diff = (torch_output - triton_output).abs().max().item()
        mean_diff = (torch_output - triton_output).abs().mean().item()
        rel_error = ((torch_output - triton_output).abs() / 
                     (torch_output.abs() + 1e-6)).mean().item()
        
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Relative error: {rel_error:.6f}")
        
        # BF16 tolerance
        tolerance = 1e-2
        if max_diff < tolerance:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED - max_diff={max_diff} exceeds tolerance={tolerance}")
            
            # Debug info
            diff_mask = (torch_output - triton_output).abs() > tolerance
            if diff_mask.any():
                print(f"  Number of mismatches: {diff_mask.sum().item()}")
                print(f"  Sample torch values: {torch_output[diff_mask][:5]}")
                print(f"  Sample triton values: {triton_output[diff_mask][:5]}")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Duplicate assignments to same (token, expert) slot
    print("\nTest 1: Duplicate assignments (last write wins)")
    num_tokens = 50
    num_experts_per_tok = 2
    hidden_size = 128
    
    res = torch.randn(10, hidden_size, device=device, dtype=torch.bfloat16)
    global_indices = torch.tensor([0, 0, 1, 1, 2], device=device).repeat(2)  # Duplicates
    token_topk_pos = torch.tensor([0, 0, 0, 1, 1], device=device).repeat(2)
    topk_weight = torch.randn(num_tokens, num_experts_per_tok, device=device, dtype=torch.float32)
    
    torch_out = torch_reference_impl(res, global_indices, token_topk_pos, 
                                    topk_weight, num_tokens, num_experts_per_tok)
    triton_out = scatter_weight_reduce_optimized(res, global_indices, token_topk_pos,
                                                topk_weight, num_tokens, num_experts_per_tok)
    
    max_diff = (torch_out - triton_out).abs().max().item()
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  {'✓ PASSED' if max_diff < 1e-2 else '✗ FAILED'}")
    
    # Test 2: Very sparse (most tokens empty)
    print("\nTest 2: Very sparse assignments")
    num_tokens = 1000
    nnz = 20
    
    res = torch.randn(nnz, hidden_size, device=device, dtype=torch.bfloat16)
    global_indices = torch.randint(0, num_tokens, (nnz,), device=device)
    token_topk_pos = torch.randint(0, num_experts_per_tok, (nnz,), device=device)
    topk_weight = torch.randn(num_tokens, num_experts_per_tok, device=device, dtype=torch.float32)
    
    torch_out = torch_reference_impl(res, global_indices, token_topk_pos,
                                    topk_weight, num_tokens, num_experts_per_tok)
    triton_out = scatter_weight_reduce_optimized(res, global_indices, token_topk_pos,
                                                topk_weight, num_tokens, num_experts_per_tok)
    
    max_diff = (torch_out - triton_out).abs().max().item()
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  {'✓ PASSED' if max_diff < 1e-2 else '✗ FAILED'}")
    
    # Test 3: All experts filled for each token
    print("\nTest 3: Dense coverage")
    num_tokens = 100
    num_experts_per_tok = 4
    nnz = num_tokens * num_experts_per_tok
    
    res = torch.randn(nnz, hidden_size, device=device, dtype=torch.bfloat16)
    global_indices = torch.arange(num_tokens, device=device).repeat_interleave(num_experts_per_tok)
    token_topk_pos = torch.arange(num_experts_per_tok, device=device).repeat(num_tokens)
    topk_weight = torch.randn(num_tokens, num_experts_per_tok, device=device, dtype=torch.float32)
    
    torch_out = torch_reference_impl(res, global_indices, token_topk_pos,
                                    topk_weight, num_tokens, num_experts_per_tok)
    triton_out = scatter_weight_reduce_optimized(res, global_indices, token_topk_pos,
                                                topk_weight, num_tokens, num_experts_per_tok)
    
    max_diff = (torch_out - triton_out).abs().max().item()
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  {'✓ PASSED' if max_diff < 1e-2 else '✗ FAILED'}")


def benchmark():
    """Performance benchmark."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    device = torch.device('cuda')
    
    configs = [
        (512, 2, 1024, 1024),
        (1024, 4, 2048, 4096),
        (2048, 8, 4096, 16384),
    ]
    
    warmup = 10
    runs = 100
    
    for num_tokens, num_experts_per_tok, hidden_size, nnz in configs:
        print(f"\nConfig: tokens={num_tokens}, K={num_experts_per_tok}, "
              f"hidden={hidden_size}, nnz={nnz}")
        
        res = torch.randn(nnz, hidden_size, device=device, dtype=torch.bfloat16)
        global_indices = torch.randint(0, num_tokens, (nnz,), device=device)
        token_topk_pos = torch.randint(0, num_experts_per_tok, (nnz,), device=device)
        topk_weight = torch.randn(num_tokens, num_experts_per_tok, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(warmup):
            _ = torch_reference_impl(res, global_indices, token_topk_pos, 
                                    topk_weight, num_tokens, num_experts_per_tok)
            _ = scatter_weight_reduce_optimized(res, global_indices, token_topk_pos,
                                               topk_weight, num_tokens, num_experts_per_tok)
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(runs):
            _ = torch_reference_impl(res, global_indices, token_topk_pos,
                                    topk_weight, num_tokens, num_experts_per_tok)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / runs * 1000
        
        # Benchmark Triton
        start = time.perf_counter()
        for _ in range(runs):
            _ = scatter_weight_reduce_optimized(res, global_indices, token_topk_pos,
                                               topk_weight, num_tokens, num_experts_per_tok)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / runs * 1000
        
        speedup = torch_time / triton_time
        print(f"  PyTorch: {torch_time:.4f} ms")
        print(f"  Triton:  {triton_time:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_correctness()
    test_edge_cases()
    
    if torch.cuda.is_available():
        benchmark()
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)