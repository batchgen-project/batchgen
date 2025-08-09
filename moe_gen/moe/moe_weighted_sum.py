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
                     global_results = weighted_output.sum(dim=1)
    
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
    token_end = tl.minimum(token_start + BLOCK_SIZE_TOKENS, num_tokens)
    
    # Hidden dimension range
    hidden_start = pid_hidden * BLOCK_SIZE_HIDDEN
    hidden_end = tl.minimum(hidden_start + BLOCK_SIZE_HIDDEN, hidden_size)
    
    # Create masks
    token_mask = (token_start + tl.arange(0, BLOCK_SIZE_TOKENS)) < num_tokens
    hidden_mask = (hidden_start + tl.arange(0, BLOCK_SIZE_HIDDEN)) < hidden_size
    
    # Process each token in this block
    for token_offset in range(BLOCK_SIZE_TOKENS):
        token_idx = token_start + token_offset
        if token_idx >= num_tokens:
            break
            
        # Initialize accumulator for this token
        acc = tl.zeros((BLOCK_SIZE_HIDDEN,), dtype=tl.float32)
        
        # Sum across all experts for this token
        for expert_idx in range(topk):
            # Load weight for this token-expert pair (scalar)
            weight_idx = token_idx * topk + expert_idx
            weight = tl.load(topk_weight_ptr + weight_idx)
            
            # Load global_results slice for this token-expert pair
            base_idx = token_idx * topk * hidden_size + expert_idx * hidden_size
            hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE_HIDDEN)
            
            # Load with bounds checking
            load_mask = hidden_mask
            values = tl.load(
                global_results_ptr + base_idx + hidden_offsets,
                mask=load_mask,
                other=0.0
            )
            
            # Multiply by weight and accumulate
            weighted_values = values * weight
            acc += weighted_values
        
        # Store the accumulated result
        output_base = token_idx * hidden_size + hidden_start
        hidden_offsets = tl.arange(0, BLOCK_SIZE_HIDDEN)
        store_mask = hidden_mask
        
        tl.store(
            output_ptr + output_base + hidden_offsets,
            acc,
            mask=store_mask
        )


def moe_weighted_sum_triton(global_results, topk_weight):
    """
    Triton implementation of fused multiply and sum operation.
    
    Args:
        global_results: torch.Tensor of shape [num_tokens, topk, hidden_size]
        topk_weight: torch.Tensor of shape [num_tokens, topk]
    
    Returns:
        output: torch.Tensor of shape [num_tokens, hidden_size]
    """
    num_tokens, topk, hidden_size = global_results.shape
    
    # Create output tensor
    output = torch.empty(
        (num_tokens, hidden_size),
        device=global_results.device,
        dtype=global_results.dtype
    )
    
    # Tunable block sizes - these should be tuned for your specific shapes
    BLOCK_SIZE_TOKENS = 4  # Process 4 tokens per CTA
    BLOCK_SIZE_HIDDEN = min(256, triton.next_power_of_2(hidden_size))
    
    # Calculate grid dimensions
    grid_tokens = triton.cdiv(num_tokens, BLOCK_SIZE_TOKENS)
    grid_hidden = triton.cdiv(hidden_size, BLOCK_SIZE_HIDDEN)
    
    # Launch kernel
    moe_weighted_sum_kernel[(grid_tokens, grid_hidden)](
        global_results,
        topk_weight,
        output,
        num_tokens,
        topk,
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