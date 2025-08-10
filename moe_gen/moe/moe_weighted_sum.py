import torch
import triton
import triton.language as tl


@triton.jit
def moe_weighted_sum_kernel(
    global_results_ptr,
    topk_weight_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    """
    Fused kernel for: weighted_output = global_results * topk_weight.unsqueeze(-1)
                     output = weighted_output.sum(dim=1)
    
    Args:
        global_results: [num_tokens, topk, hidden_size]
        topk_weight: [num_tokens, topk]
        output: [num_tokens, hidden_size]
    """
    # Program IDs
    pid_token = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    # Token range for this CTA
    token_start = pid_token * BLOCK_SIZE_TOKENS
    
    # Hidden dimension range
    hidden_start = pid_hidden * BLOCK_SIZE_HIDDEN
    
    # Create offset arrays
    token_offsets = token_start + tl.arange(0, BLOCK_SIZE_TOKENS)
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE_HIDDEN)
    
    # Create masks
    token_mask = token_offsets < num_tokens
    hidden_mask = hidden_offsets < hidden_size
    
    # Initialize output accumulator for all tokens in this block
    # Shape: [BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN]
    output_acc = tl.zeros((BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN), dtype=tl.float32)
    
    # Sum across all experts
    for expert_idx in range(topk):
        # Load weights for all tokens in this block
        # Shape: [BLOCK_SIZE_TOKENS]
        weight_indices = token_offsets * topk + expert_idx
        weights = tl.load(
            topk_weight_ptr + weight_indices,
            mask=token_mask,
            other=0.0
        )
        
        # Load global_results for all tokens and current expert
        # We need to load [BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN]
        for tok_idx in range(BLOCK_SIZE_TOKENS):
            token_id = token_start + tok_idx
            if token_id < num_tokens:
                # Base index for this token and expert
                base_idx = token_id * topk * hidden_size + expert_idx * hidden_size
                
                # Load values for this token-expert pair
                values = tl.load(
                    global_results_ptr + base_idx + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0
                )
                
                # Get weight for this token (scalar)
                weight = weights[tok_idx] if tok_idx < BLOCK_SIZE_TOKENS else 0.0
                
                # Multiply by weight and accumulate
                weighted_values = values * weight
                output_acc[tok_idx, :] += weighted_values
    
    # Store results for all tokens in this block
    for tok_idx in range(BLOCK_SIZE_TOKENS):
        token_id = token_start + tok_idx
        if token_id < num_tokens:
            output_base = token_id * hidden_size + hidden_start
            
            tl.store(
                output_ptr + output_base + tl.arange(0, BLOCK_SIZE_HIDDEN),
                output_acc[tok_idx, :],
                mask=hidden_mask
            )


@triton.jit
def moe_weighted_sum_kernel_optimized(
    global_results_ptr,
    topk_weight_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    """
    More optimized version that processes multiple tokens vectorially.
    """
    # Program IDs
    pid_token = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    # Token and hidden dimension ranges
    token_start = pid_token * BLOCK_SIZE_TOKENS
    hidden_start = pid_hidden * BLOCK_SIZE_HIDDEN
    
    # Create offset arrays
    token_offsets = token_start + tl.arange(0, BLOCK_SIZE_TOKENS)
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE_HIDDEN)
    
    # Create masks
    token_mask = token_offsets < num_tokens
    hidden_mask = hidden_offsets < hidden_size
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN), dtype=tl.float32)
    
    # Process each expert
    for expert_idx in range(topk):
        # Load weights for current expert across all tokens in block
        weight_ptrs = token_offsets[:, None] * topk + expert_idx
        weights = tl.load(
            topk_weight_ptr + weight_ptrs,
            mask=token_mask[:, None],
            other=0.0
        )  # Shape: [BLOCK_SIZE_TOKENS, 1]
        
        # Load global results for current expert
        # Calculate base pointers for each token
        token_base_ptrs = (token_offsets[:, None] * topk + expert_idx) * hidden_size
        result_ptrs = token_base_ptrs + hidden_offsets[None, :]
        
        # Load values with proper masking
        mask_2d = token_mask[:, None] & hidden_mask[None, :]
        values = tl.load(
            global_results_ptr + result_ptrs,
            mask=mask_2d,
            other=0.0
        )  # Shape: [BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN]
        
        # Multiply by weights and accumulate
        weighted_values = values * weights
        acc += weighted_values
    
    # Store results
    output_ptrs = token_offsets[:, None] * hidden_size + hidden_offsets[None, :]
    mask_2d = token_mask[:, None] & hidden_mask[None, :]
    
    tl.store(
        output_ptr + output_ptrs,
        acc,
        mask=mask_2d
    )


def moe_weighted_sum_triton(global_results, topk_weight, use_optimized=True):
    """
    Triton implementation of fused multiply and sum operation.
    
    Args:
        global_results: torch.Tensor of shape [num_tokens, topk, hidden_size]
        topk_weight: torch.Tensor of shape [num_tokens, topk]
        use_optimized: bool, whether to use the optimized kernel version
    
    Returns:
        output: torch.Tensor of shape [num_tokens, hidden_size]
    """
    num_tokens, topk_val, hidden_size = global_results.shape
    assert topk_weight.shape == (num_tokens, topk_val), f"Shape mismatch: {topk_weight.shape} vs {(num_tokens, topk_val)}"
    
    # Create output tensor
    output = torch.empty(
        (num_tokens, hidden_size),
        device=global_results.device,
        dtype=global_results.dtype
    )
    
    # Tunable block sizes
    BLOCK_SIZE_TOKENS = min(32, triton.next_power_of_2(num_tokens)) if num_tokens < 32 else 32
    BLOCK_SIZE_HIDDEN = min(256, triton.next_power_of_2(hidden_size))
    
    # Ensure minimum block sizes for efficiency
    BLOCK_SIZE_TOKENS = max(4, BLOCK_SIZE_TOKENS)
    BLOCK_SIZE_HIDDEN = max(32, BLOCK_SIZE_HIDDEN)
    
    # Calculate grid dimensions
    grid_tokens = triton.cdiv(num_tokens, BLOCK_SIZE_TOKENS)
    grid_hidden = triton.cdiv(hidden_size, BLOCK_SIZE_HIDDEN)
    
    # Choose kernel
    kernel = moe_weighted_sum_kernel_optimized if use_optimized else moe_weighted_sum_kernel
    
    # Launch kernel
    kernel[(grid_tokens, grid_hidden)](
        global_results,
        topk_weight,
        output,
        num_tokens,
        topk_val,
        hidden_size,
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        BLOCK_SIZE_HIDDEN=BLOCK_SIZE_HIDDEN,
    )
    
    return output


# Alternative version with different blocking strategy for very large hidden_size
@triton.jit
def moe_weighted_sum_kernel_v2(
    global_results_ptr,
    topk_weight_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    """
    Alternative version that processes one token per CTA but tiles across hidden dimension.
    Better for very large hidden_size.
    """
    pid_token = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    if pid_token >= num_tokens:
        return
        
    # Hidden dimension tile
    hidden_start = pid_hidden * BLOCK_SIZE_HIDDEN
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE_HIDDEN)
    hidden_mask = hidden_offsets < hidden_size
    
    # Accumulator for this token's output slice
    acc = tl.zeros((BLOCK_SIZE_HIDDEN,), dtype=tl.float32)
    
    # Sum across all experts
    for expert_idx in range(topk):
        # Load weight (scalar)
        weight_idx = pid_token * topk + expert_idx
        weight = tl.load(topk_weight_ptr + weight_idx)
        
        # Load corresponding slice of global_results
        base_idx = pid_token * topk * hidden_size + expert_idx * hidden_size
        values = tl.load(
            global_results_ptr + base_idx + hidden_offsets,
            mask=hidden_mask,
            other=0.0
        )
        
        # Weighted accumulation
        acc += values * weight
    
    # Store result
    output_base = pid_token * hidden_size
    tl.store(
        output_ptr + output_base + hidden_offsets,
        acc,
        mask=hidden_mask
    )


def moe_weighted_sum_triton_v2(global_results, topk_weight):
    """Alternative implementation - one token per CTA, tiled across hidden dimension."""
    num_tokens, topk, hidden_size = global_results.shape
    
    output = torch.empty(
        (num_tokens, hidden_size),
        device=global_results.device,
        dtype=global_results.dtype
    )
    
    BLOCK_SIZE_HIDDEN = 256
    grid_tokens = num_tokens
    grid_hidden = triton.cdiv(hidden_size, BLOCK_SIZE_HIDDEN)
    
    moe_weighted_sum_kernel_v2[(grid_tokens, grid_hidden)](
        global_results,
        topk_weight,
        output,
        num_tokens,
        topk,
        hidden_size,
        BLOCK_SIZE_TOKENS=1,  # Not used in v2
        BLOCK_SIZE_HIDDEN=BLOCK_SIZE_HIDDEN,
    )
    
    return output


# Example usage and benchmark
if __name__ == "__main__":
    # Test parameters
    num_tokens = 1024
    topk = 8
    hidden_size = 4096
    device = "cuda"
    dtype = torch.bfloat16
    
    # Create test tensors
    global_results = torch.randn(num_tokens, topk, hidden_size, device=device, dtype=dtype)
    topk_weight = torch.randn(num_tokens, topk, device=device, dtype=dtype)
    
    # PyTorch reference
    torch_result = (global_results * topk_weight.unsqueeze(-1)).sum(dim=1)
    
    # Triton versions
    triton_result_v1 = moe_weighted_sum_triton(global_results, topk_weight)
    triton_result_v2 = moe_weighted_sum_triton_v2(global_results, topk_weight)
    
    # Check correctness
    print(f"Max diff v1: {torch.max(torch.abs(torch_result - triton_result_v1))}")
    print(f"Max diff v2: {torch.max(torch.abs(torch_result - triton_result_v2))}")
    
    # Simple benchmark
    import time
    
    def benchmark(func, *args, num_runs=100):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            func(*args)
        torch.cuda.synchronize()
        return (time.time() - start) / num_runs
    
    torch_time = benchmark(lambda: (global_results * topk_weight.unsqueeze(-1)).sum(dim=1))
    triton_v1_time = benchmark(moe_weighted_sum_triton, global_results, topk_weight)
    triton_v2_time = benchmark(moe_weighted_sum_triton_v2, global_results, topk_weight)
    
    print(f"PyTorch time: {torch_time*1000:.3f} ms")
    print(f"Triton v1 time: {triton_v1_time*1000:.3f} ms")
    print(f"Triton v2 time: {triton_v2_time*1000:.3f} ms")
    print(f"Speedup v1: {torch_time/triton_v1_time:.2f}x")
    print(f"Speedup v2: {torch_time/triton_v2_time:.2f}x")