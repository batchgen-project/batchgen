import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time
import numpy as np

# Compile the CUDA kernel
print("Compiling CUDA kernel...")
fused_moe = load(
    name="fused_moe_gate",
    sources=["/data2/tairan/workspace/BatchGen/test/fused_moe_gate.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True
)

def moe_gate_forward_original(hidden_states, weight, e_score_correction_bias, 
                              n_group, topk_group, n_routed_experts, top_k, 
                              routed_scaling_factor):
    """Original PyTorch implementation"""
    bsz, seq_len, h = hidden_states.shape
    
    # Compute gating score
    hidden_states_flat = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states_flat.type(torch.float32), 
        weight.type(torch.float32), 
        None
    )
    scores = logits.sigmoid()
    
    # Select top-k experts
    scores_for_choice = scores.view(bsz * seq_len, -1) + e_score_correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(bsz * seq_len, n_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    
    group_idx = torch.topk(
        group_scores, k=topk_group, dim=-1, sorted=False
    )[1]
    
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
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
    topk_weight = topk_weight * routed_scaling_factor
    
    return topk_idx, topk_weight

def test_correctness(bsz=4, seq_len=128, h=512, n_experts=32, n_group=8, 
                     topk_group=4, top_k=4, routed_scaling_factor=1.5):
    """Test that fused kernel matches original implementation"""
    print(f"\n{'='*60}")
    print(f"Testing Correctness")
    print(f"{'='*60}")
    print(f"Config: bsz={bsz}, seq_len={seq_len}, h={h}, experts={n_experts}")
    print(f"        n_group={n_group}, topk_group={topk_group}, top_k={top_k}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(bsz, seq_len, h, device=device)
    weight = torch.randn(n_experts, h, device=device)
    e_score_bias = torch.randn(n_experts, device=device) * 0.1
    
    # Original implementation
    with torch.no_grad():
        orig_idx, orig_weight = moe_gate_forward_original(
            hidden_states.clone(),
            weight.clone(),
            e_score_bias.clone(),
            n_group, topk_group, n_experts, top_k,
            routed_scaling_factor
        )
    
    # Fused kernel implementation
    with torch.no_grad():
        fused_idx, fused_weight = fused_moe.forward(
            hidden_states.clone(),
            weight.clone(),
            e_score_bias.clone(),
            n_group, topk_group, n_experts, top_k,
            routed_scaling_factor
        )
    
    print(f"\nOriginal indices shape: {orig_idx.shape}")
    print(f"Fused indices shape: {fused_idx.shape}")
    print(f"Original weights shape: {orig_weight.shape}")
    print(f"Fused weights shape: {fused_weight.shape}")
    
    # Check indices
    # Note: topk can return different orderings, so we sort both
    orig_idx_sorted = torch.sort(orig_idx, dim=-1)[0]
    fused_idx_sorted = torch.sort(fused_idx, dim=-1)[0]
    
    idx_match = torch.allclose(orig_idx_sorted.float(), fused_idx_sorted.float(), atol=0)
    print(f"\n✓ Indices match: {idx_match}")
    if not idx_match:
        diff_mask = (orig_idx_sorted != fused_idx_sorted)
        print(f"  Number of mismatches: {diff_mask.sum().item()}")
        print(f"  First few mismatches:")
        print(f"    Original: {orig_idx_sorted[diff_mask][:5]}")
        print(f"    Fused:    {fused_idx_sorted[diff_mask][:5]}")
    
    # Check weights - gather in same order
    orig_weight_sorted = torch.gather(orig_weight, 1, 
                                      torch.argsort(orig_idx, dim=-1))
    fused_weight_sorted = torch.gather(fused_weight, 1,
                                       torch.argsort(fused_idx, dim=-1))
    
    weight_close = torch.allclose(orig_weight_sorted, fused_weight_sorted, 
                                  rtol=1e-3, atol=1e-4)
    print(f"✓ Weights close: {weight_close}")
    
    if not weight_close:
        max_diff = (orig_weight_sorted - fused_weight_sorted).abs().max()
        mean_diff = (orig_weight_sorted - fused_weight_sorted).abs().mean()
        print(f"  Max difference: {max_diff.item():.6f}")
        print(f"  Mean difference: {mean_diff.item():.6f}")
    
    # Check that weights sum correctly (accounting for scaling)
    orig_sum = orig_weight.sum(dim=-1)
    fused_sum = fused_weight.sum(dim=-1)
    sum_close = torch.allclose(orig_sum, fused_sum, rtol=1e-3, atol=1e-4)
    print(f"✓ Weight sums match: {sum_close}")
    print(f"  Original sum range: [{orig_sum.min().item():.4f}, {orig_sum.max().item():.4f}]")
    print(f"  Fused sum range: [{fused_sum.min().item():.4f}, {fused_sum.max().item():.4f}]")
    
    return idx_match and weight_close and sum_close

def benchmark(bsz=8, seq_len=1024, h=1024, n_experts=64, n_group=8,
              topk_group=4, top_k=4, routed_scaling_factor=1.5, 
              num_warmup=10, num_iters=100):
    """Benchmark both implementations"""
    print(f"\n{'='*60}")
    print(f"Benchmarking Performance")
    print(f"{'='*60}")
    print(f"Config: bsz={bsz}, seq_len={seq_len}, h={h}, experts={n_experts}")
    print(f"        n_group={n_group}, topk_group={topk_group}, top_k={top_k}")
    print(f"Iterations: {num_iters} (after {num_warmup} warmup)")
    
    device = torch.device('cuda')
    
    # Create inputs
    hidden_states = torch.randn(bsz, seq_len, h, device=device)
    weight = torch.randn(n_experts, h, device=device)
    e_score_bias = torch.randn(n_experts, device=device) * 0.1
    
    # Warmup
    print("\nWarming up...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = moe_gate_forward_original(
                hidden_states, weight, e_score_bias,
                n_group, topk_group, n_experts, top_k, routed_scaling_factor
            )
            _ = fused_moe.forward(
                hidden_states.float(), weight.float(), e_score_bias.float(),
                n_group, topk_group, n_experts, top_k, routed_scaling_factor
            )[0]
    torch.cuda.synchronize()
    
    # Benchmark original
    print("\nBenchmarking original PyTorch implementation...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = moe_gate_forward_original(
                hidden_states, weight, e_score_bias,
                n_group, topk_group, n_experts, top_k, routed_scaling_factor
            )
    torch.cuda.synchronize()
    orig_time = (time.perf_counter() - start) / num_iters * 1000
    
    # Benchmark fused
    print("Benchmarking fused CUDA kernel...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = fused_moe.forward(
                hidden_states.float(), weight.float(), e_score_bias.float(),
                n_group, topk_group, n_experts, top_k, routed_scaling_factor
            )[0]
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_iters * 1000
    
    # Results
    print(f"\n{'─'*60}")
    print(f"Results:")
    print(f"{'─'*60}")
    print(f"Original PyTorch: {orig_time:.3f} ms")
    print(f"Fused CUDA:       {fused_time:.3f} ms")
    print(f"{'─'*60}")
    print(f"Speedup:          {orig_time/fused_time:.2f}x")
    print(f"{'─'*60}")
    
    return orig_time, fused_time

def main():
    print("="*60)
    print("Fused MoE Gate: Validation & Benchmark Suite")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available. Tests will fail.")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    
    # Test 1: Small config for correctness
    print("\n" + "="*60)
    print("TEST 1: Small Configuration")
    print("="*60)
    passed = test_correctness(
        bsz=2, seq_len=64, h=256, n_experts=16,
        n_group=4, topk_group=2, top_k=2
    )
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    
    # Test 2: Medium config
    print("\n" + "="*60)
    print("TEST 2: Medium Configuration")
    print("="*60)
    passed = test_correctness(
        bsz=4, seq_len=128, h=512, n_experts=32,
        n_group=8, topk_group=4, top_k=4
    )
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    
    # Test 3: Large config
    print("\n" + "="*60)
    print("TEST 3: Large Configuration")
    print("="*60)
    passed = test_correctness(
        bsz=8, seq_len=256, h=1024, n_experts=64,
        n_group=8, topk_group=4, top_k=6
    )
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    
    # Benchmark Suite
    print("\n" + "="*60)
    print("BENCHMARK SUITE")
    print("="*60)
    
    configs = [
        ("Small", 4, 512, 512, 32, 8, 4, 4),
        ("Medium", 8, 1024, 1024, 64, 8, 4, 4),
        ("Large", 16, 2048, 2048, 64, 8, 4, 6),
    ]
    
    results = []
    for name, bsz, seq_len, h, n_experts, n_group, topk_group, top_k in configs:
        print(f"\n{'─'*60}")
        print(f"Benchmark: {name}")
        orig_t, fused_t = benchmark(
            bsz, seq_len, h, n_experts, n_group, topk_group, top_k,
            num_warmup=20, num_iters=100
        )
        results.append((name, orig_t, fused_t, orig_t/fused_t))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Config':<10} {'Original (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10}")
    print("─"*60)
    for name, orig_t, fused_t, speedup in results:
        print(f"{name:<10} {orig_t:>13.3f}   {fused_t:>13.3f}   {speedup:>8.2f}x")
    print("="*60)

if __name__ == "__main__":
    main()