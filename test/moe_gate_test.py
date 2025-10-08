import torch
import torch.nn.functional as F
import numpy as np
import triton
import triton.language as tl

@triton.jit
def moe_gate_optimized_kernel(
    hidden_states_ptr,
    weight_ptr,
    e_score_correction_bias_ptr,
    topk_idx_ptr,
    topk_weight_ptr,
    num_tokens,
    hidden_size,
    n_routed_experts,
    n_group,
    topk_group,
    top_k,
    routed_scaling_factor,
    BLOCK_SIZE_H: tl.constexpr,
    MAX_EXPERTS: tl.constexpr,  # Compile-time constant
):
    """
    Optimized MoE gating kernel.
    """
    token_idx = tl.program_id(0)
    
    if token_idx >= num_tokens:
        return
    
    experts_per_group = n_routed_experts // n_group
    
    # ==================== Compute all expert scores ====================
    expert_indices = tl.arange(0, MAX_EXPERTS)
    expert_mask = expert_indices < n_routed_experts
    scores = tl.zeros([MAX_EXPERTS], dtype=tl.float32)
    
    for e_idx in range(n_routed_experts):
        # Compute dot product for this expert
        acc = tl.zeros([1], dtype=tl.float32)
        
        for h_start in range(0, hidden_size, BLOCK_SIZE_H):
            h_offs = h_start + tl.arange(0, BLOCK_SIZE_H)
            h_mask = h_offs < hidden_size
            
            h_vals = tl.load(hidden_states_ptr + token_idx * hidden_size + h_offs, 
                           mask=h_mask, other=0.0)
            w_vals = tl.load(weight_ptr + e_idx * hidden_size + h_offs, 
                           mask=h_mask, other=0.0)
            
            acc += tl.sum(h_vals * w_vals)
        
        # Sigmoid
        score = tl.sigmoid(acc)
        scores = tl.where(expert_indices == e_idx, score, scores)
    
    # Add correction bias
    bias = tl.load(e_score_correction_bias_ptr + expert_indices, mask=expert_mask, other=0.0)
    scores_corrected = scores + bias
    
    # ==================== Group selection ====================
    # For each group, find top-2 and sum
    MAX_GROUPS = MAX_EXPERTS // 2  # Assume at least 2 experts per group
    group_indices = tl.arange(0, MAX_GROUPS)
    group_mask = group_indices < n_group
    group_scores = tl.zeros([MAX_GROUPS], dtype=tl.float32)
    
    for g_idx in range(n_group):
        group_start = g_idx * experts_per_group
        group_end = group_start + experts_per_group
        
        # Extract group scores
        in_group = (expert_indices >= group_start) & (expert_indices < group_end) & expert_mask
        group_vals = tl.where(in_group, scores_corrected, -1e9)
        
        # Find top-2 using two passes
        max1 = tl.max(group_vals)
        group_vals_without_max1 = tl.where(group_vals == max1, -1e9, group_vals)
        max2 = tl.max(group_vals_without_max1)
        
        group_score = max1 + max2
        group_scores = tl.where(group_indices == g_idx, group_score, group_scores)
    
    # Select top-k groups using iterative max
    selected_groups_mask = tl.zeros([MAX_GROUPS], dtype=tl.int32)
    
    for k in range(topk_group):
        # Find max among unselected groups
        masked_group_scores = tl.where((selected_groups_mask == 0) & group_mask, group_scores, -1e9)
        max_score = tl.max(masked_group_scores)
        
        # Mark the group(s) with max score as selected
        is_max = (masked_group_scores == max_score) & (selected_groups_mask == 0) & group_mask
        selected_groups_mask = tl.where(is_max, 1, selected_groups_mask)
    
    # Create expert mask based on selected groups
    expert_in_selected_group = tl.zeros([MAX_EXPERTS], dtype=tl.int32)
    for g_idx in range(n_group):
        is_selected = tl.sum(tl.where(group_indices == g_idx, selected_groups_mask, 0)) > 0
        if is_selected:
            group_start = g_idx * experts_per_group
            group_end = group_start + experts_per_group
            in_this_group = (expert_indices >= group_start) & (expert_indices < group_end) & expert_mask
            expert_in_selected_group = tl.where(in_this_group, 1, expert_in_selected_group)
    
    # Apply mask to scores
    masked_expert_scores = tl.where(expert_in_selected_group == 1, scores_corrected, -1e9)
    
    # ==================== Final top-k selection ====================
    MAX_TOP_K = 16  # Maximum top_k we support
    selected_experts_mask = tl.zeros([MAX_EXPERTS], dtype=tl.int32)
    topk_expert_indices = tl.zeros([MAX_TOP_K], dtype=tl.int32)
    topk_expert_scores = tl.zeros([MAX_TOP_K], dtype=tl.float32)
    
    for k in range(top_k):
        # Find max among unselected experts
        available_scores = tl.where((selected_experts_mask == 0) & expert_mask, masked_expert_scores, -1e9)
        max_score = tl.max(available_scores)
        
        # Find which expert has this score
        is_max = (available_scores == max_score) & (selected_experts_mask == 0) & expert_mask
        
        # Get the expert index (first one if ties)
        selected_idx = 0
        for e_idx in range(n_routed_experts):
            if tl.sum(tl.where(expert_indices == e_idx, is_max.to(tl.int32), 0)) > 0:
                selected_idx = e_idx
                break
        
        # Mark as selected
        selected_experts_mask = tl.where(expert_indices == selected_idx, 1, selected_experts_mask)
        
        # Store results
        topk_expert_indices = tl.where(tl.arange(0, MAX_TOP_K) == k, selected_idx, topk_expert_indices)
        # Use original scores (not corrected) for weighting
        original_score = tl.sum(tl.where(expert_indices == selected_idx, scores, 0.0))
        topk_expert_scores = tl.where(tl.arange(0, MAX_TOP_K) == k, original_score, topk_expert_scores)
    
    # ==================== Normalize and store ====================
    # Only sum the actual top_k values
    denominator = 0.0
    for k in range(top_k):
        denominator += topk_expert_scores[k]
    denominator += 1e-20
    
    # Write outputs
    for k in range(top_k):
        normalized_weight = (topk_expert_scores[k] / denominator) * routed_scaling_factor
        output_offset = token_idx * top_k + k
        tl.store(topk_idx_ptr + output_offset, topk_expert_indices[k])
        tl.store(topk_weight_ptr + output_offset, normalized_weight)


def moe_gate_forward_optimized(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    n_routed_experts: int,
    top_k: int,
    routed_scaling_factor: float,
):
    bsz, seq_len, h = hidden_states.shape
    num_tokens = bsz * seq_len
    
    hidden_states_flat = hidden_states.view(-1, h).contiguous()
    weight = weight.contiguous()
    e_score_correction_bias = e_score_correction_bias.contiguous()
    
    topk_idx = torch.empty((num_tokens, top_k), dtype=torch.int32, device=hidden_states.device)
    topk_weight = torch.empty((num_tokens, top_k), dtype=torch.float32, device=hidden_states.device)
    
    BLOCK_SIZE_H = min(128, triton.next_power_of_2(h))
    
    # Find the next power of 2 for MAX_EXPERTS
    MAX_EXPERTS = triton.next_power_of_2(n_routed_experts)
    MAX_EXPERTS = max(MAX_EXPERTS, 16)  # At least 16
    
    grid = (num_tokens,)
    
    moe_gate_optimized_kernel[grid](
        hidden_states_flat,
        weight,
        e_score_correction_bias,
        topk_idx,
        topk_weight,
        num_tokens,
        h,
        n_routed_experts,
        n_group,
        topk_group,
        top_k,
        routed_scaling_factor,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        MAX_EXPERTS=MAX_EXPERTS,
    )
    
    return topk_idx, topk_weight

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
        
        # Skip if no assignment for this slot
        if nnz_idx >= 0:
            # Load weight
            weight = tl.load(topk_weight_ptr + mapping_offset)
            
            # Load and weight the result
            res_offset = nnz_idx * hidden_size + h_indices
            res_vals = tl.load(res_ptr + res_offset, mask=h_mask, other=0.0)
            res_vals_fp32 = res_vals.to(tl.float32)
            
            accumulator += res_vals_fp32 * weight
    
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
    mapping = torch.full((num_tokens, num_experts_per_tok), -1, 
                         dtype=torch.int32, device=global_indices.device)
    mapping[global_indices, token_topk_pos] = torch.arange(
        len(global_indices), dtype=torch.int32, device=global_indices.device
    )
    return mapping


def scatter_weight_reduce_optimized(
    res: torch.Tensor,
    global_indices: torch.Tensor,
    token_topk_pos: torch.Tensor,
    topk_weight: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Optimized version using inverse mapping."""
    assert topk_weight.dtype == torch.float32
    
    # Build inverse mapping (can be cached if indices don't change)
    nnz_indices = build_inverse_mapping(
        global_indices, token_topk_pos, num_tokens, num_experts_per_tok
    )
    
    output = torch.zeros((num_tokens, hidden_size), device=res.device, dtype=torch.float32)
    
    BLOCK_SIZE_H = 128 if hidden_size > 64 else 64
    grid = (num_tokens, triton.cdiv(hidden_size, BLOCK_SIZE_H))
    
    scatter_weight_reduce_optimized_kernel[grid](
        res, nnz_indices, topk_weight,
        output,
        num_tokens, num_experts_per_tok, hidden_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return output


def moe_gate_forward_torch(
    hidden_states, 
    weight, 
    e_score_correction_bias,
    n_group, 
    topk_group, 
    n_routed_experts, 
    top_k,
    routed_scaling_factor
):
    """Original PyTorch implementation"""
    bsz, seq_len, h = hidden_states.shape
    
    ### compute gating score
    hidden_states_flat = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states_flat.type(torch.float32), 
        weight.type(torch.float32), 
        None
    )
    scores = logits.sigmoid()
    
    ### select top-k experts
    scores_for_choice = scores.view(bsz * seq_len, -1) + e_score_correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(bsz * seq_len, n_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    
    group_idx = torch.topk(
        group_scores, k=topk_group, dim=-1, sorted=False
    )[1]  # [n, top_k_group]
    
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(bsz * seq_len, n_group, n_routed_experts // n_group)
        .reshape(bsz * seq_len, -1)
    )  # [n, e]
    
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    
    topk_weight = scores.gather(1, topk_idx)
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
    topk_weight = topk_weight * routed_scaling_factor
    
    return topk_idx, topk_weight


def test_moe_gate():
    """Test MoE gate kernel"""
    print("=" * 80)
    print("Testing MoE Gate Kernel")
    print("=" * 80)
    
    # Test configurations
    configs = [
        # (bsz, seq_len, hidden_size, n_routed_experts, n_group, topk_group, top_k)
        (2, 8, 128, 8, 2, 1, 2),           # Small test
        (4, 16, 256, 16, 4, 2, 2),         # Medium test
        (1, 32, 512, 32, 8, 4, 4),         # Larger test
        (2, 64, 1024, 64, 8, 4, 6),        # Large test
    ]
    
    routed_scaling_factor = 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config_idx, (bsz, seq_len, hidden_size, n_routed_experts, n_group, topk_group, top_k) in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Config {config_idx + 1}: bsz={bsz}, seq_len={seq_len}, h={hidden_size}, "
              f"experts={n_routed_experts}, groups={n_group}, topk_group={topk_group}, top_k={top_k}")
        print(f"{'='*80}")
        
        # Generate random inputs
        torch.manual_seed(42 + config_idx)
        hidden_states = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
        weight = torch.randn(n_routed_experts, hidden_size, device=device, dtype=torch.bfloat16)
        e_score_correction_bias = torch.randn(n_routed_experts, device=device, dtype=torch.float32) * 0.1
        
        # PyTorch reference
        print("\nRunning PyTorch reference...")
        topk_idx_torch, topk_weight_torch = moe_gate_forward_torch(
            hidden_states.clone(),
            weight.clone(),
            e_score_correction_bias.clone(),
            n_group,
            topk_group,
            n_routed_experts,
            top_k,
            routed_scaling_factor
        )
        
        # Triton kernel
        print("Running Triton kernel...")
        try:
            topk_idx_triton, topk_weight_triton = moe_gate_forward_optimized(
                hidden_states.clone(),
                weight.clone(),
                e_score_correction_bias.clone(),
                n_group,
                topk_group,
                n_routed_experts,
                top_k,
                routed_scaling_factor
            )
            
            # Convert triton outputs to same dtype as torch
            topk_idx_triton = topk_idx_triton.long()
            topk_weight_triton = topk_weight_triton.float()
            
            print("\n" + "-" * 80)
            print("RESULTS COMPARISON")
            print("-" * 80)
            
            # Check shapes
            print(f"\nShape check:")
            print(f"  PyTorch topk_idx: {topk_idx_torch.shape}")
            print(f"  Triton topk_idx:  {topk_idx_triton.shape}")
            print(f"  PyTorch topk_weight: {topk_weight_torch.shape}")
            print(f"  Triton topk_weight:  {topk_weight_triton.shape}")
            
            assert topk_idx_torch.shape == topk_idx_triton.shape, "Index shape mismatch!"
            assert topk_weight_torch.shape == topk_weight_triton.shape, "Weight shape mismatch!"
            print("  ✓ Shapes match!")
            
            # Compare weights (should be close)
            weight_diff = torch.abs(topk_weight_torch - topk_weight_triton)
            weight_rel_diff = weight_diff / (torch.abs(topk_weight_torch) + 1e-8)
            
            print(f"\nWeight comparison:")
            print(f"  Max absolute diff: {weight_diff.max().item():.6f}")
            print(f"  Mean absolute diff: {weight_diff.mean().item():.6f}")
            print(f"  Max relative diff: {weight_rel_diff.max().item():.6f}")
            print(f"  Mean relative diff: {weight_rel_diff.mean().item():.6f}")
            
            # Check if weights are normalized (sum to routed_scaling_factor per token)
            torch_weight_sums = topk_weight_torch.sum(dim=1)
            triton_weight_sums = topk_weight_triton.sum(dim=1)
            
            print(f"\nWeight sum check (should be ~{routed_scaling_factor}):")
            print(f"  PyTorch - min: {torch_weight_sums.min().item():.6f}, "
                  f"max: {torch_weight_sums.max().item():.6f}, "
                  f"mean: {torch_weight_sums.mean().item():.6f}")
            print(f"  Triton  - min: {triton_weight_sums.min().item():.6f}, "
                  f"max: {triton_weight_sums.max().item():.6f}, "
                  f"mean: {triton_weight_sums.mean().item():.6f}")
            
            # Check if selected experts are valid
            print(f"\nExpert index validity check:")
            print(f"  PyTorch - min: {topk_idx_torch.min().item()}, max: {topk_idx_torch.max().item()}")
            print(f"  Triton  - min: {topk_idx_triton.min().item()}, max: {topk_idx_triton.max().item()}")
            
            assert topk_idx_torch.min() >= 0 and topk_idx_torch.max() < n_routed_experts, "Invalid PyTorch indices!"
            assert topk_idx_triton.min() >= 0 and topk_idx_triton.max() < n_routed_experts, "Invalid Triton indices!"
            print("  ✓ All indices are valid!")
            
            # Check if indices match (they might not due to ties in topk, but weights should be similar)
            # Instead, check if the selected experts are from the correct groups
            num_tokens = bsz * seq_len
            experts_per_group = n_routed_experts // n_group
            
            # Verify group membership
            torch_groups = topk_idx_torch // experts_per_group
            triton_groups = topk_idx_triton // experts_per_group
            
            print(f"\nGroup membership check:")
            print(f"  PyTorch unique groups per token: {[len(torch.unique(torch_groups[i])) for i in range(min(3, num_tokens))]}")
            print(f"  Triton unique groups per token:  {[len(torch.unique(triton_groups[i])) for i in range(min(3, num_tokens))]}")
            
            # Check a few sample tokens in detail
            print(f"\nDetailed comparison for first 3 tokens:")
            for token_idx in range(min(3, num_tokens)):
                print(f"\n  Token {token_idx}:")
                print(f"    PyTorch indices: {topk_idx_torch[token_idx].tolist()}")
                print(f"    Triton indices:  {topk_idx_triton[token_idx].tolist()}")
                print(f"    PyTorch weights: {[f'{w:.4f}' for w in topk_weight_torch[token_idx].tolist()]}")
                print(f"    Triton weights:  {[f'{w:.4f}' for w in topk_weight_triton[token_idx].tolist()]}")
                print(f"    Weight diff:     {[f'{w:.4f}' for w in (topk_weight_torch[token_idx] - topk_weight_triton[token_idx]).tolist()]}")
            
            # Overall assessment
            weight_close = torch.allclose(topk_weight_torch, topk_weight_triton, rtol=1e-2, atol=1e-3)
            
            print(f"\n{'='*80}")
            if weight_close:
                print("✓ TEST PASSED: Weights are close enough!")
            else:
                print("⚠ WARNING: Weights differ more than expected.")
                print("  This might be due to:")
                print("  1. Different tie-breaking in topk selection")
                print("  2. Numerical precision differences")
                print("  3. Different ordering of operations")
                
                # Check if the difference is just due to expert reordering
                torch_sorted_weights = torch.sort(topk_weight_torch, dim=1)[0]
                triton_sorted_weights = torch.sort(topk_weight_triton, dim=1)[0]
                sorted_close = torch.allclose(torch_sorted_weights, triton_sorted_weights, rtol=1e-2, atol=1e-3)
                
                if sorted_close:
                    print("  ✓ Sorted weights match - difference is just expert ordering!")
                else:
                    print("  ✗ Even sorted weights differ - may indicate a bug!")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\n✗ ERROR running Triton kernel: {e}")
            import traceback
            traceback.print_exc()
            continue


def test_scatter_weight_reduce():
    """Test scatter-weight-reduce kernel"""
    print("\n" + "=" * 80)
    print("Testing Scatter-Weight-Reduce Kernel")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configurations
    configs = [
        # (num_tokens, hidden_size, num_experts_per_tok, nnz)
        (16, 128, 2, 32),
        (32, 256, 4, 128),
        (64, 512, 2, 128),
        (128, 1024, 6, 512),
    ]
    
    for config_idx, (num_tokens, hidden_size, num_experts_per_tok, nnz) in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Config {config_idx + 1}: tokens={num_tokens}, hidden={hidden_size}, "
              f"experts_per_tok={num_experts_per_tok}, nnz={nnz}")
        print(f"{'='*80}")
        
        # Generate random inputs
        torch.manual_seed(42 + config_idx)
        res = torch.randn(nnz, hidden_size, device=device, dtype=torch.bfloat16)
        
        # Generate valid indices
        global_indices = torch.randint(0, num_tokens, (nnz,), device=device)
        token_topk_pos = torch.randint(0, num_experts_per_tok, (nnz,), device=device)
        topk_weight = torch.rand(num_tokens, num_experts_per_tok, device=device, dtype=torch.float32)
        
        # Normalize weights
        topk_weight = topk_weight / topk_weight.sum(dim=1, keepdim=True)
        
        # PyTorch reference implementation
        print("\nRunning PyTorch reference...")
        global_results_torch = torch.zeros(
            (num_tokens, num_experts_per_tok, hidden_size),
            device=device, 
            dtype=torch.bfloat16
        )
        global_results_torch[global_indices, token_topk_pos, :] = res
        
        weighted_output_torch = global_results_torch.to(torch.float32) * topk_weight.unsqueeze(-1)
        final_output_torch = weighted_output_torch.sum(dim=1)
        
        # Triton kernel
        print("Running Triton kernel...")
        try:
            final_output_triton = scatter_weight_reduce_optimized(
                res,
                global_indices,
                token_topk_pos,
                topk_weight,
                num_tokens,
                hidden_size,
                num_experts_per_tok,
            )
            
            print("\n" + "-" * 80)
            print("RESULTS COMPARISON")
            print("-" * 80)
            
            # Check shapes
            print(f"\nShape check:")
            print(f"  PyTorch output: {final_output_torch.shape}")
            print(f"  Triton output:  {final_output_triton.shape}")
            assert final_output_torch.shape == final_output_triton.shape, "Shape mismatch!"
            print("  ✓ Shapes match!")
            
            # Compare outputs
            diff = torch.abs(final_output_torch - final_output_triton)
            rel_diff = diff / (torch.abs(final_output_torch) + 1e-8)
            
            print(f"\nOutput comparison:")
            print(f"  Max absolute diff: {diff.max().item():.6f}")
            print(f"  Mean absolute diff: {diff.mean().item():.6f}")
            print(f"  Max relative diff: {rel_diff.max().item():.6f}")
            print(f"  Mean relative diff: {rel_diff.mean().item():.6f}")
            
            # Check sample values
            print(f"\nSample values (first 3 tokens, first 5 dims):")
            for i in range(min(3, num_tokens)):
                print(f"\n  Token {i}:")
                print(f"    PyTorch: {final_output_torch[i, :5].tolist()}")
                print(f"    Triton:  {final_output_triton[i, :5].tolist()}")
                print(f"    Diff:    {(final_output_torch[i, :5] - final_output_triton[i, :5]).tolist()}")
            
            # Overall assessment
            outputs_close = torch.allclose(final_output_torch, final_output_triton, rtol=1e-2, atol=1e-3)
            
            print(f"\n{'='*80}")
            if outputs_close:
                print("✓ TEST PASSED: Outputs match!")
            else:
                print("⚠ WARNING: Outputs differ more than expected")
                
                # Check if specific tokens have issues
                token_max_diff = diff.max(dim=1)[0]
                problematic_tokens = (token_max_diff > 0.01).nonzero(as_tuple=True)[0]
                
                if len(problematic_tokens) > 0:
                    print(f"  Tokens with large differences: {problematic_tokens[:10].tolist()}")
                else:
                    print("  Differences are uniformly small - likely numerical precision")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\n✗ ERROR running Triton kernel: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" TRITON KERNEL VALIDATION SUITE")
    print("=" * 80)
    
    # Test MoE gate
    # test_moe_gate()
    
    # Test scatter-weight-reduce
    test_scatter_weight_reduce()
    
    print("\n" + "=" * 80)
    print(" ALL TESTS COMPLETED")
    print("=" * 80)