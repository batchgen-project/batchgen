import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time

# Compile the kernel
print("Compiling parallel CUDA kernel...")
parallel_moe = load(
    name="parallel_moe_gate",
    sources=["/data2/tairan/workspace/BatchGen/test/fused_moe_gate.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True
)

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

def moe_gate_forward_hybrid(hidden_states, weight, e_score_correction_bias,
                           n_group, topk_group, n_routed_experts, top_k,
                           routed_scaling_factor):
    """Hybrid: PyTorch matmul + sigmoid, then custom kernel"""
    bsz, seq_len, h = hidden_states.shape
    
    # PyTorch handles heavy lifting
    hidden_states_flat = hidden_states.view(-1, h)
    logits = F.linear(hidden_states_flat.float(), weight.float(), None)
    scores = torch.sigmoid(logits)
    
    # Custom kernel handles MoE routing
    topk_idx, topk_weight = parallel_moe.forward(
        scores,
        e_score_correction_bias,
        n_group,
        topk_group,
        n_routed_experts,
        top_k,
        routed_scaling_factor
    )
    
    return topk_idx, topk_weight

def test_correctness(bsz=4, seq_len=128, h=512, n_experts=32, n_group=8,
                     topk_group=4, top_k=4, routed_scaling_factor=1.5):
    """Test correctness"""
    print(f"\n{'='*70}")
    print(f"Testing Correctness")
    print(f"{'='*70}")
    print(f"Config: bsz={bsz}, seq_len={seq_len}, h={h}, experts={n_experts}")
    print(f"        n_group={n_group}, topk_group={topk_group}, top_k={top_k}")
    
    device = torch.device('cuda')
    
    torch.manual_seed(42)
    hidden_states = torch.randn(bsz, seq_len, h, device=device)
    weight = torch.randn(n_experts, h, device=device)
    e_score_bias = torch.randn(n_experts, device=device) * 0.1
    
    # Original
    with torch.no_grad():
        orig_idx, orig_weight = moe_gate_forward_original(
            hidden_states.clone(), weight.clone(), e_score_bias.clone(),
            n_group, topk_group, n_experts, top_k, routed_scaling_factor
        )
    
    # Hybrid
    with torch.no_grad():
        hybrid_idx, hybrid_weight = moe_gate_forward_hybrid(
            hidden_states.clone(), weight.clone(), e_score_bias.clone(),
            n_group, topk_group, n_experts, top_k, routed_scaling_factor
        )
    
    print(f"\nShapes: {orig_idx.shape}, {hybrid_idx.shape}")
    
    # Check indices (sorted)
    orig_idx_sorted = torch.sort(orig_idx, dim=-1)[0]
    hybrid_idx_sorted = torch.sort(hybrid_idx, dim=-1)[0]
    
    idx_match = torch.allclose(orig_idx_sorted.float(), hybrid_idx_sorted.float(), atol=0)
    print(f"✓ Indices match: {idx_match}")
    
    if not idx_match:
        diff_mask = (orig_idx_sorted != hybrid_idx_sorted)
        num_diff = diff_mask.sum().item()
        print(f"  Mismatches: {num_diff} / {diff_mask.numel()}")
        if num_diff > 0:
            print(f"  Sample mismatches:")
            for i in range(min(3, diff_mask.size(0))):
                if diff_mask[i].any():
                    print(f"    Token {i}: Orig={orig_idx_sorted[i]}, Hybrid={hybrid_idx_sorted[i]}")
    
    # Check weights
    orig_weight_sorted = torch.gather(orig_weight, 1, torch.argsort(orig_idx, dim=-1))
    hybrid_weight_sorted = torch.gather(hybrid_weight, 1, torch.argsort(hybrid_idx, dim=-1))
    
    weight_close = torch.allclose(orig_weight_sorted, hybrid_weight_sorted, rtol=1e-3, atol=1e-4)
    print(f"✓ Weights close: {weight_close}")
    
    if not weight_close:
        max_diff = (orig_weight_sorted - hybrid_weight_sorted).abs().max()
        mean_diff = (orig_weight_sorted - hybrid_weight_sorted).abs().mean()
        print(f"  Max diff: {max_diff.item():.6f}, Mean diff: {mean_diff.item():.6f}")
    
    # Check sums
    orig_sum = orig_weight.sum(dim=-1)
    hybrid_sum = hybrid_weight.sum(dim=-1)
    sum_close = torch.allclose(orig_sum, hybrid_sum, rtol=1e-3, atol=1e-4)
    print(f"✓ Weight sums match: {sum_close}")
    
    return idx_match and weight_close and sum_close

def benchmark(bsz=8, seq_len=1024, h=1024, n_experts=64, n_group=8,
              topk_group=4, top_k=4, routed_scaling_factor=1.5,
              num_warmup=20, num_iters=100):
    """Benchmark performance"""
    print(f"\n{'='*70}")
    print(f"Benchmarking")
    print(f"{'='*70}")
    print(f"Config: bsz={bsz}, seq_len={seq_len}, h={h}, experts={n_experts}")
    print(f"Iterations: {num_iters} (warmup: {num_warmup})")
    
    device = torch.device('cuda')
    
    hidden_states = torch.randn(bsz, seq_len, h, device=device)
    weight = torch.randn(n_experts, h, device=device)
    e_score_bias = torch.randn(n_experts, device=device) * 0.1
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = moe_gate_forward_original(hidden_states, weight, e_score_bias,
                                         n_group, topk_group, n_experts, top_k, 
                                         routed_scaling_factor)
            _ = moe_gate_forward_hybrid(hidden_states, weight, e_score_bias,
                                       n_group, topk_group, n_experts, top_k,
                                       routed_scaling_factor)
    torch.cuda.synchronize()
    
    # Benchmark original
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = moe_gate_forward_original(hidden_states, weight, e_score_bias,
                                         n_group, topk_group, n_experts, top_k,
                                         routed_scaling_factor)
    torch.cuda.synchronize()
    orig_time = (time.perf_counter() - start) / num_iters * 1000
    
    # Benchmark hybrid
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = moe_gate_forward_hybrid(hidden_states, weight, e_score_bias,
                                       n_group, topk_group, n_experts, top_k,
                                       routed_scaling_factor)
    torch.cuda.synchronize()
    hybrid_time = (time.perf_counter() - start) / num_iters * 1000
    
    print(f"\n{'─'*70}")
    print(f"Results:")
    print(f"  Original PyTorch: {orig_time:>8.3f} ms")
    print(f"  Hybrid:           {hybrid_time:>8.3f} ms")
    print(f"{'─'*70}")
    print(f"  Speedup:          {orig_time/hybrid_time:>8.2f}x")
    print(f"{'─'*70}")
    
    return orig_time, hybrid_time

def main():
    print("="*70)
    print("Parallel MoE Gate Kernel: Test Suite")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    
    # Correctness tests
    print("\n" + "="*70)
    print("CORRECTNESS TESTS")
    print("="*70)
    
    configs = [
        ("Small", 2, 64, 256, 16, 4, 2, 2),
        ("Medium", 4, 128, 512, 32, 8, 4, 4),
        ("Large", 8, 256, 1024, 64, 8, 4, 6),
    ]
    
    all_passed = True
    for name, bsz, seq_len, h, n_experts, n_group, topk_group, top_k in configs:
        print(f"\n{name} Configuration:")
        passed = test_correctness(bsz, seq_len, h, n_experts, n_group, topk_group, top_k)
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}")
        all_passed = all_passed and passed
    
    if not all_passed:
        print("\n⚠ Some tests failed. Fix before benchmarking.")
        return
    
    # Benchmarks
    print("\n" + "="*70)
    print("BENCHMARK SUITE")
    print("="*70)
    
    bench_configs = [
        ("Small", 4, 512, 512, 32, 8, 4, 4),
        ("Medium", 8, 1024, 1024, 64, 8, 4, 4),
        ("Large", 16, 2048, 2048, 64, 8, 4, 6),
        ("XLarge", 32, 2048, 2048, 128, 16, 4, 8),
    ]
    
    results = []
    for name, bsz, seq_len, h, n_experts, n_group, topk_group, top_k in bench_configs:
        print(f"\n{name}:")
        try:
            orig_t, hybrid_t = benchmark(bsz, seq_len, h, n_experts, n_group, 
                                        topk_group, top_k, num_warmup=20, num_iters=100)
            results.append((name, orig_t, hybrid_t, orig_t/hybrid_t))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((name, 0, 0, 0))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<12} {'Original':<12} {'Hybrid':<12} {'Speedup':<10}")
    print("─"*70)
    for name, orig_t, hybrid_t, speedup in results:
        if speedup > 0:
            print(f"{name:<12} {orig_t:>10.3f}ms {hybrid_t:>10.3f}ms {speedup:>8.2f}x")
        else:
            print(f"{name:<12} {'ERROR'}")
    print("="*70)

if __name__ == "__main__":
    main()