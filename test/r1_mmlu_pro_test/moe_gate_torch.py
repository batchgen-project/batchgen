import torch
import torch.nn.functional as F
import time

def moe_gate_forward_original(hidden_states, weight, e_score_correction_bias, 
                              n_group, topk_group, n_routed_experts, top_k, 
                              routed_scaling_factor):
    """Original PyTorch implementation"""
    bsz, seq_len, h = hidden_states.shape
    
    hidden_states_flat = hidden_states.view(-1, h)
    logits = F.linear(hidden_states_flat.float(), weight.float(), None)
    scores = logits.sigmoid()
    
    scores_for_choice = scores.view(bsz * seq_len, -1) + e_score_correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(bsz * seq_len, n_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(bsz * seq_len, n_group, n_routed_experts // n_group)
        .reshape(bsz * seq_len, -1)
    )
    
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    
    topk_weight = scores.gather(1, topk_idx)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor
    
    return topk_idx, topk_weight


def moe_gate_forward_optimized(hidden_states, weight, e_score_correction_bias,
                               n_group, topk_group, n_routed_experts, top_k,
                               routed_scaling_factor):
    """Optimized: avoid unnecessary tensor operations"""
    bsz, seq_len, h = hidden_states.shape
    n = bsz * seq_len
    experts_per_group = n_routed_experts // n_group
    
    # Compute logits and scores
    hidden_states_flat = hidden_states.view(n, h)
    logits = F.linear(hidden_states_flat.float(), weight.float(), None)
    scores = torch.sigmoid(logits)
    scores_for_choice = scores + e_score_correction_bias
    
    # Compute group scores
    scores_grouped = scores_for_choice.view(n, n_group, experts_per_group)
    group_scores = scores_grouped.topk(2, dim=-1)[0].sum(dim=-1)
    
    # Select top groups and create mask more efficiently
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    
    # Direct mask construction without zeros + scatter
    # Create mask by comparing each position to selected indices
    score_mask = torch.zeros(n, n_routed_experts, dtype=torch.bool, device=scores.device)
    for i in range(topk_group):
        group_id = group_idx[:, i:i+1]  # [n, 1]
        start_idx = group_id * experts_per_group
        for j in range(experts_per_group):
            score_mask.scatter_(1, start_idx + j, True)
    
    # Masked topk
    masked_scores = scores_for_choice.masked_fill(~score_mask, float('-inf'))
    topk_idx = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)[1]
    
    # Normalize
    topk_weight = scores.gather(1, topk_idx)
    topk_weight = topk_weight * (routed_scaling_factor / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20))
    
    return topk_idx, topk_weight


def moe_gate_forward_vectorized(hidden_states, weight, e_score_correction_bias,
                                n_group, topk_group, n_routed_experts, top_k,
                                routed_scaling_factor):
    """Fully vectorized version - no loops"""
    bsz, seq_len, h = hidden_states.shape
    n = bsz * seq_len
    experts_per_group = n_routed_experts // n_group
    
    # Compute scores
    hidden_states_flat = hidden_states.view(n, h)
    scores = torch.sigmoid(F.linear(hidden_states_flat.float(), weight.float()))
    scores_for_choice = scores + e_score_correction_bias
    
    # Group scores
    group_scores = (scores_for_choice
                   .view(n, n_group, experts_per_group)
                   .topk(2, dim=-1)[0]
                   .sum(dim=-1))
    
    # Top groups
    group_idx = group_scores.topk(topk_group, dim=-1, sorted=False)[1]
    
    # Vectorized mask creation using advanced indexing
    # Expand group indices to expert indices
    expert_offset = torch.arange(experts_per_group, device=scores.device)
    expert_indices = (group_idx.unsqueeze(-1) * experts_per_group + expert_offset).view(n, -1)
    
    # Create mask efficiently
    score_mask = torch.zeros(n, n_routed_experts, dtype=torch.bool, device=scores.device)
    score_mask.scatter_(1, expert_indices, True)
    
    # Final topk and normalization
    masked_scores = scores_for_choice.masked_fill(~score_mask, float('-inf'))
    topk_idx = masked_scores.topk(top_k, dim=-1, sorted=False)[1]
    
    topk_weight = scores.gather(1, topk_idx)
    topk_weight = topk_weight * (routed_scaling_factor / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20))
    
    return topk_idx, topk_weight


def benchmark_all():
    """Compare all versions"""
    print("="*70)
    print("MoE Gate Optimization Comparison")
    print("="*70)
    
    device = torch.device('cuda')
    configs = [
        ("Small", 4, 512, 512, 32, 8, 4, 4),
        ("Medium", 8, 1024, 1024, 64, 8, 4, 4),
        ("Large", 16, 2048, 2048, 64, 8, 4, 6),
    ]
    
    results = []
    for name, bsz, seq_len, h, n_experts, n_group, topk_group, top_k in configs:
        print(f"\n{name} Config: bsz={bsz}, seq={seq_len}, h={h}, experts={n_experts}")
        
        hidden_states = torch.randn(bsz, seq_len, h, device=device)
        weight = torch.randn(n_experts, h, device=device)
        e_score_bias = torch.randn(n_experts, device=device) * 0.1
        
        versions = [
            ("Original", moe_gate_forward_original),
            ("Optimized", moe_gate_forward_optimized),
            ("Vectorized", moe_gate_forward_vectorized),
        ]
        
        times = {}
        for version_name, func in versions:
            # Warmup
            for _ in range(20):
                _ = func(hidden_states, weight, e_score_bias,
                        n_group, topk_group, n_experts, top_k, 1.5)
            torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                _ = func(hidden_states, weight, e_score_bias,
                        n_group, topk_group, n_experts, top_k, 1.5)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 100 * 1000
            times[version_name] = elapsed
            print(f"  {version_name:12s}: {elapsed:6.3f} ms")
        
        results.append((name, times))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<10} {'Original':<12} {'Optimized':<12} {'Vectorized':<12}")
    print("-"*70)
    for name, times in results:
        print(f"{name:<10} {times['Original']:>10.3f}ms {times['Optimized']:>10.3f}ms {times['Vectorized']:>10.3f}ms")
    print("="*70)


if __name__ == "__main__":
    benchmark_all()