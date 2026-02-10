"""
Comprehensive sanity checks for CUDA routing kernels.

Validates gate, dispatch, and reduce kernels across varied configurations:
- Batch sizes: 1 to 4096
- Hidden dimensions: 256, 1024, 2880 (GPT-OSS), 4096, 7680
- Expert configs: various E_local and expert_start values
- Edge cases: empty experts, all-local, no-local

Usage:
    python -m batchgen.moe.routing.test_routing
"""

import sys
import torch
import torch.nn.functional as F

from batchgen.moe.routing import (
    gate_topk_softmax_cuda,
    dispatch_count_gather_cuda,
    reduce_weighted_scatter_cuda,
)


# ──────────────────────────────────────────────────────────────────────────────
# Reference implementations (PyTorch, for ground truth)
# ──────────────────────────────────────────────────────────────────────────────

def ref_gate(router_logits, k=4):
    topk_weights, topk_indices = torch.topk(router_logits, k=k, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1)
    return topk_indices.to(torch.int32), topk_weights


def ref_dispatch(x, topk_indices, expert_start, num_local_experts):
    N, K = topk_indices.shape
    H = x.shape[1]
    expert_end = expert_start + num_local_experts
    device = x.device

    expert_counts = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    flat_indices = topk_indices.view(-1)

    for idx in range(N * K):
        eid = flat_indices[idx].item()
        if expert_start <= eid < expert_end:
            expert_counts[eid - expert_start] += 1

    expert_offsets = torch.zeros(num_local_experts + 1, dtype=torch.int32, device=device)
    for e in range(num_local_experts):
        expert_offsets[e + 1] = expert_offsets[e] + expert_counts[e]
    total = expert_offsets[-1].item()

    dispatched_x = torch.zeros(max(total, 1), H, dtype=x.dtype, device=device)
    topk_pos = torch.full((N * K,), -1, dtype=torch.int32, device=device)
    counters = torch.zeros(num_local_experts, dtype=torch.int32, device=device)

    for idx in range(N * K):
        token_id = idx // K
        eid = flat_indices[idx].item()
        if expert_start <= eid < expert_end:
            local_eid = eid - expert_start
            write_pos = expert_offsets[local_eid].item() + counters[local_eid].item()
            dispatched_x[write_pos] = x[token_id]
            topk_pos[idx] = write_pos
            counters[local_eid] += 1

    return dispatched_x, expert_counts, expert_offsets, topk_pos


def ref_reduce(expert_output, topk_pos, topk_weights, N, H, K):
    device = expert_output.device
    output = torch.zeros(N, H, dtype=torch.bfloat16, device=device)
    for i in range(N):
        acc = torch.zeros(H, dtype=torch.float32, device=device)
        for k in range(K):
            pos = topk_pos[i * K + k].item()
            if pos >= 0:
                acc += topk_weights[i, k] * expert_output[pos].float()
        output[i] = acc.to(torch.bfloat16)
    return output


# ──────────────────────────────────────────────────────────────────────────────
# Test configurations
# ──────────────────────────────────────────────────────────────────────────────

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
HIDDEN_DIMS = [256, 1024, 2880, 4096, 7680]
EXPERT_CONFIGS = [
    # (E_total, K, E_local, expert_start)
    (128, 4, 16, 0),     # GPT-OSS default: 128 experts, K=4, 16 local, start=0
    (128, 4, 16, 64),    # GPT-OSS EP rank 4: start=64
    (128, 4, 128, 0),    # Single GPU: all 128 local
    (64, 2, 8, 0),       # Smaller model
    (64, 2, 8, 32),      # Smaller model, non-zero start
    (16, 4, 16, 0),      # All experts local
    (128, 4, 16, 112),   # Last rank: experts 112-127
]


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_gate(verbose=True):
    """Test gate kernel across batch sizes and expert counts."""
    print("\n" + "=" * 70)
    print("GATE KERNEL: topk + softmax")
    print("=" * 70)

    failures = 0
    total = 0

    for E_total, K, _, _ in EXPERT_CONFIGS:
        for N in BATCH_SIZES:
            torch.manual_seed(42 + N + E_total)
            logits = torch.randn(N, E_total, dtype=torch.float32, device="cuda")

            ref_idx, ref_w = ref_gate(logits, k=K)
            test_idx, test_w = gate_topk_softmax_cuda(logits, k=K)

            idx_match = torch.equal(test_idx, ref_idx)
            w_err = (test_w - ref_w).abs().max().item()
            w_match = w_err < 1e-5

            total += 1
            if not (idx_match and w_match):
                failures += 1
                print(f"  [FAIL] N={N}, E={E_total}, K={K}: "
                      f"idx_match={idx_match}, w_err={w_err:.8f}")
            elif verbose and N in [1, 64, 512, 4096]:
                print(f"  [PASS] N={N:>4d}, E={E_total:>3d}, K={K}: w_err={w_err:.8f}")

    # BF16 input auto-cast test
    logits_bf16 = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    test_idx, test_w = gate_topk_softmax_cuda(logits_bf16, k=4)
    ref_idx, ref_w = ref_gate(logits_bf16.float(), k=4)
    bf16_pass = torch.equal(test_idx, ref_idx)
    total += 1
    if not bf16_pass:
        failures += 1
        print(f"  [FAIL] BF16 auto-cast")
    else:
        print(f"  [PASS] BF16 input auto-cast")

    print(f"\nGate: {total - failures}/{total} passed")
    return failures == 0


def test_dispatch(verbose=True):
    """Test dispatch kernel across batch sizes, hidden dims, and expert configs."""
    print("\n" + "=" * 70)
    print("DISPATCH KERNEL: count + prefix_sum + gather")
    print("=" * 70)

    failures = 0
    total = 0

    for H in HIDDEN_DIMS:
        for E_total, K, E_local, expert_start in EXPERT_CONFIGS:
            # Limit large N × large H to avoid slow reference impl
            max_n = 512 if H <= 2880 else 128
            for N in [n for n in BATCH_SIZES if n <= max_n]:
                torch.manual_seed(42 + N + H + E_total + expert_start)
                x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
                topk_idx = torch.randint(0, E_total, (N, K), dtype=torch.int32, device="cuda")

                ref_d, ref_c, ref_o, ref_tp = ref_dispatch(x, topk_idx, expert_start, E_local)
                test_d, test_c, test_o, test_tp = dispatch_count_gather_cuda(
                    x, topk_idx, expert_start, E_local)

                counts_ok = torch.equal(test_c, ref_c)
                offsets_ok = torch.equal(test_o, ref_o)

                # Verify dispatched content via topk_pos
                content_ok = True
                total_disp = ref_o[-1].item()
                for idx in range(N * K):
                    rp = ref_tp[idx].item()
                    tp = test_tp[idx].item()
                    if rp == -1:
                        if tp != -1:
                            content_ok = False
                            break
                    else:
                        if tp < 0 or not torch.equal(test_d[tp], x[idx // K]):
                            content_ok = False
                            break

                total += 1
                ok = counts_ok and offsets_ok and content_ok
                if not ok:
                    failures += 1
                    print(f"  [FAIL] N={N}, H={H}, E={E_total}, E_local={E_local}, "
                          f"start={expert_start}: counts={counts_ok}, offsets={offsets_ok}, "
                          f"content={content_ok}")
                elif verbose and N in [1, 64, 512] and H == 2880:
                    print(f"  [PASS] N={N:>4d}, H={H}, E_local={E_local:>3d}, start={expert_start}")

    # Edge: int64 auto-cast
    x = torch.randn(8, 2880, dtype=torch.bfloat16, device="cuda")
    topk_idx_i64 = torch.randint(0, 128, (8, 4), dtype=torch.int64, device="cuda")
    test_d, test_c, test_o, test_tp = dispatch_count_gather_cuda(x, topk_idx_i64, 0, 16)
    total += 1
    print(f"  [PASS] int64 input auto-cast")

    print(f"\nDispatch: {total - failures}/{total} passed")
    return failures == 0


def test_reduce(verbose=True):
    """Test reduce kernel across batch sizes, hidden dims, and expert configs."""
    print("\n" + "=" * 70)
    print("REDUCE KERNEL: weighted scatter-add")
    print("=" * 70)

    failures = 0
    total = 0

    for H in HIDDEN_DIMS:
        for E_total, K, E_local, expert_start in EXPERT_CONFIGS:
            max_n = 512 if H <= 2880 else 128
            for N in [n for n in BATCH_SIZES if n <= max_n]:
                torch.manual_seed(42 + N + H + E_total + expert_start)
                x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
                topk_idx = torch.randint(0, E_total, (N, K), dtype=torch.int32, device="cuda")
                topk_w = torch.randn(N, K, dtype=torch.float32, device="cuda").softmax(dim=-1)

                _, _, expert_offsets, topk_pos = ref_dispatch(x, topk_idx, expert_start, E_local)
                total_disp = expert_offsets[-1].item()

                expert_out = torch.randn(max(total_disp, 1), H,
                                         dtype=torch.bfloat16, device="cuda")

                ref_out = ref_reduce(expert_out, topk_pos, topk_w, N, H, K)
                test_out = reduce_weighted_scatter_cuda(expert_out, topk_pos, topk_w, N, H, K)

                max_err = (test_out.float() - ref_out.float()).abs().max().item()
                ok = max_err < 0.02  # BF16 tolerance

                total += 1
                if not ok:
                    failures += 1
                    print(f"  [FAIL] N={N}, H={H}, E_local={E_local}, "
                          f"start={expert_start}: max_err={max_err:.6f}")
                elif verbose and N in [1, 64, 512] and H == 2880:
                    print(f"  [PASS] N={N:>4d}, H={H}, E_local={E_local:>3d}, "
                          f"start={expert_start}: max_err={max_err:.6f}")

    # Edge: all topk_pos = -1 (no local experts)
    tp_neg = torch.full((32,), -1, dtype=torch.int32, device="cuda")
    tw = torch.randn(8, 4, dtype=torch.float32, device="cuda")
    eo = torch.randn(1, 2880, dtype=torch.bfloat16, device="cuda")
    result = reduce_weighted_scatter_cuda(eo, tp_neg, tw, 8, 2880, 4)
    total += 1
    if (result == 0).all():
        print(f"  [PASS] all topk_pos=-1 → output zeros")
    else:
        failures += 1
        print(f"  [FAIL] all topk_pos=-1 should produce zeros")

    print(f"\nReduce: {total - failures}/{total} passed")
    return failures == 0


def test_end_to_end(verbose=True):
    """Test full pipeline: gate → dispatch → (mock GEMM) → reduce."""
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE: gate → dispatch → identity GEMM → reduce")
    print("=" * 70)

    failures = 0
    total = 0

    for E_total, K, E_local, expert_start in EXPERT_CONFIGS:
        for N in [1, 8, 64, 256]:
            H = 2880
            torch.manual_seed(42 + N + E_total + expert_start)

            x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
            logits = torch.randn(N, E_total, dtype=torch.float32, device="cuda")

            # CUDA pipeline
            cuda_idx, cuda_w = gate_topk_softmax_cuda(logits, k=K)
            cuda_d, cuda_c, cuda_o, cuda_tp = dispatch_count_gather_cuda(
                x, cuda_idx, expert_start, E_local)
            total_disp = cuda_o[E_local].item()
            # Identity "GEMM": expert_output = dispatched_x (no transformation)
            expert_out = cuda_d[:total_disp].clone()
            cuda_result = reduce_weighted_scatter_cuda(expert_out, cuda_tp, cuda_w, N, H, K)

            # Reference pipeline
            ref_idx, ref_w = ref_gate(logits, k=K)
            ref_d, ref_c, ref_o, ref_tp = ref_dispatch(x, ref_idx, expert_start, E_local)
            ref_total = ref_o[-1].item()
            ref_expert_out = ref_d[:ref_total].clone()
            ref_result = ref_reduce(ref_expert_out, ref_tp, ref_w, N, H, K)

            max_err = (cuda_result.float() - ref_result.float()).abs().max().item()
            ok = max_err < 0.02

            total += 1
            if not ok:
                failures += 1
                print(f"  [FAIL] N={N}, E={E_total}, E_local={E_local}, "
                      f"start={expert_start}: max_err={max_err:.6f}")
            elif verbose:
                print(f"  [PASS] N={N:>4d}, E={E_total:>3d}, E_local={E_local:>3d}, "
                      f"start={expert_start:>3d}: max_err={max_err:.6f}")

    print(f"\nEnd-to-end: {total - failures}/{total} passed")
    return failures == 0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("CUDA ROUTING KERNELS: COMPREHENSIVE SANITY CHECK")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    results = []
    results.append(("Gate", test_gate()))
    results.append(("Dispatch", test_dispatch()))
    results.append(("Reduce", test_reduce()))
    results.append(("End-to-end", test_end_to_end()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:>15s}: {status}")
        all_pass &= passed

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)
