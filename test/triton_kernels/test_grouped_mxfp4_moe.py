#!/usr/bin/env python3
"""Sanity check for grouped MXFP4 MoE functions.

Tests:
1. moe_token_dispatch: Token sorting and expert offsets
2. mxfp4_mlp_forward: Single expert MLP vs reference
3. grouped_mxfp4_moe_forward: Full MoE vs per-expert loop reference

Usage:
    python test/triton_kernels/test_grouped_mxfp4_moe.py
"""

import sys
sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")

import time
import torch
import torch.nn.functional as F
from batchgen.quantization.mxfp4 import mxfp4_dequantize

try:
    from batchgen.moe.mxfp4_grouped_gemm import (
        moe_token_dispatch,
        mxfp4_mlp_forward,
        grouped_mxfp4_moe_forward,
        grouped_mxfp4_moe_forward_3d,  # New true grouped implementation
        setup_expert_weight_pointers,   # Pointer array setup
        fused_mxfp4_single_gemm,
        HAS_TRITON_KERNELS,
    )
except ImportError as e:
    print(f"Import error: {e}")
    HAS_TRITON_KERNELS = False


def reference_mlp_forward(x, gate_packed, gate_scales, gate_bias,
                          up_packed, up_scales, up_bias,
                          down_packed, down_scales, down_bias,
                          alpha=1.702, limit=7.0):
    """Reference MLP using unfused dequant + matmul."""
    gate_bf16 = mxfp4_dequantize(gate_packed, gate_scales, torch.bfloat16)
    up_bf16 = mxfp4_dequantize(up_packed, up_scales, torch.bfloat16)
    down_bf16 = mxfp4_dequantize(down_packed, down_scales, torch.bfloat16)

    gate_out = torch.mm(x, gate_bf16.T)
    if gate_bias is not None:
        gate_out = gate_out + gate_bias

    up_out = torch.mm(x, up_bf16.T)
    if up_bias is not None:
        up_out = up_out + up_bias

    gate_clamped = gate_out.clamp(max=limit)
    up_clamped = up_out.clamp(min=-limit, max=limit)
    glu = gate_clamped * torch.sigmoid(alpha * gate_clamped)
    intermediate = glu * (up_clamped + 1)

    output = torch.mm(intermediate, down_bf16.T)
    if down_bias is not None:
        output = output + down_bias

    return output


def reference_moe_forward(hidden_states, topk_indices, topk_weights,
                          gate_weights, gate_scales, gate_biases,
                          up_weights, up_scales, up_biases,
                          down_weights, down_scales, down_biases):
    """Reference MoE using per-expert loop."""
    num_tokens, hidden = hidden_states.shape
    num_experts = len(gate_weights)
    output = torch.zeros_like(hidden_states)

    for i in range(num_experts):
        expert_mask = (topk_indices == i).any(dim=-1)
        if expert_mask.any():
            expert_weights = torch.where(
                topk_indices == i, topk_weights, torch.zeros_like(topk_weights)
            ).sum(dim=-1)

            expert_input = hidden_states[expert_mask]
            expert_output = reference_mlp_forward(
                expert_input,
                gate_weights[i], gate_scales[i], gate_biases[i] if gate_biases else None,
                up_weights[i], up_scales[i], up_biases[i] if up_biases else None,
                down_weights[i], down_scales[i], down_biases[i] if down_biases else None,
            )
            output[expert_mask] += expert_output * expert_weights[expert_mask].unsqueeze(-1)

    return output


def test_token_dispatch():
    """Test moe_token_dispatch correctness."""
    print("=" * 60)
    print("TEST 1: moe_token_dispatch")
    print("=" * 60)

    device = "cuda"
    num_tokens = 16
    hidden = 64
    num_experts = 8
    k = 2

    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)
    topk_weights = F.softmax(torch.randn(num_tokens, k, device=device), dim=-1).to(torch.bfloat16)

    sorted_hidden, expert_offsets, orig_indices, orig_k, routing_weights = moe_token_dispatch(
        hidden_states, topk_indices, topk_weights, num_experts
    )

    # Verify total count
    total_routed = num_tokens * k
    assert sorted_hidden.shape[0] == total_routed, f"Expected {total_routed}, got {sorted_hidden.shape[0]}"

    # Verify offsets are monotonic
    assert (expert_offsets[1:] >= expert_offsets[:-1]).all(), "Offsets not monotonic"

    # Verify offset bounds
    assert expert_offsets[-1] == total_routed, f"Final offset {expert_offsets[-1]} != {total_routed}"

    print(f"Sorted hidden shape: {sorted_hidden.shape}")
    print(f"Expert offsets: {expert_offsets.tolist()}")
    print("PASSED")
    return True


def test_mlp_forward():
    """Test mxfp4_mlp_forward vs reference."""
    print("\n" + "=" * 60)
    print("TEST 2: mxfp4_mlp_forward")
    print("=" * 60)

    if not HAS_TRITON_KERNELS:
        print("SKIPPED: triton_kernels not installed")
        return None

    device = "cuda"
    M, hidden, intermediate = 8, 128, 256

    x = torch.randn(M, hidden, dtype=torch.bfloat16, device=device)
    gate_packed = torch.randint(0, 256, (intermediate, hidden // 2), dtype=torch.uint8, device=device)
    gate_scales = torch.randint(120, 134, (intermediate, hidden // 32), dtype=torch.uint8, device=device)
    gate_bias = torch.randn(intermediate, dtype=torch.bfloat16, device=device)
    up_packed = torch.randint(0, 256, (intermediate, hidden // 2), dtype=torch.uint8, device=device)
    up_scales = torch.randint(120, 134, (intermediate, hidden // 32), dtype=torch.uint8, device=device)
    up_bias = torch.randn(intermediate, dtype=torch.bfloat16, device=device)
    down_packed = torch.randint(0, 256, (hidden, intermediate // 2), dtype=torch.uint8, device=device)
    down_scales = torch.randint(120, 134, (hidden, intermediate // 32), dtype=torch.uint8, device=device)
    down_bias = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    ref_output = reference_mlp_forward(
        x, gate_packed, gate_scales, gate_bias,
        up_packed, up_scales, up_bias,
        down_packed, down_scales, down_bias,
    )

    test_output = mxfp4_mlp_forward(
        x, gate_packed, gate_scales, gate_bias,
        up_packed, up_scales, up_bias,
        down_packed, down_scales, down_bias,
    )

    diff = (ref_output.float() - test_output.float()).abs()
    rel_error = (diff / ref_output.abs().clamp(min=1e-6)).max().item()

    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    if rel_error < 0.02:
        print("PASSED")
        return True
    else:
        print("FAILED")
        return False


def test_grouped_moe_forward():
    """Test grouped_mxfp4_moe_forward vs reference."""
    print("\n" + "=" * 60)
    print("TEST 3: grouped_mxfp4_moe_forward")
    print("=" * 60)

    if not HAS_TRITON_KERNELS:
        print("SKIPPED: triton_kernels not installed")
        return None

    device = "cuda"
    num_tokens = 32
    hidden = 128
    intermediate = 256
    num_experts = 8
    k = 2

    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_indices = torch.topk(router_logits, k=k, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1).to(torch.bfloat16)

    # Create random expert weights
    gate_weights = [torch.randint(0, 256, (intermediate, hidden // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    gate_scales = [torch.randint(120, 134, (intermediate, hidden // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    gate_biases = [torch.randn(intermediate, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    up_weights = [torch.randint(0, 256, (intermediate, hidden // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    up_scales = [torch.randint(120, 134, (intermediate, hidden // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    up_biases = [torch.randn(intermediate, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    down_weights = [torch.randint(0, 256, (hidden, intermediate // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    down_scales = [torch.randint(120, 134, (hidden, intermediate // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    down_biases = [torch.randn(hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

    ref_output = reference_moe_forward(
        hidden_states, topk_indices, topk_weights,
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        down_weights, down_scales, down_biases,
    )

    test_output = grouped_mxfp4_moe_forward(
        hidden_states, topk_indices, topk_weights,
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        down_weights, down_scales, down_biases,
    )

    diff = (ref_output.float() - test_output.float()).abs()
    rel_error = (diff / ref_output.abs().clamp(min=1e-6)).max().item()

    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    if rel_error < 0.02:
        print("PASSED")
        return True
    else:
        print("FAILED")
        return False


def test_grouped_moe_forward_3d():
    """Test grouped_mxfp4_moe_forward_3d (true grouped GEMM) vs reference."""
    print("\n" + "=" * 60)
    print("TEST 4: grouped_mxfp4_moe_forward_3d (True Grouped GEMM)")
    print("=" * 60)

    if not HAS_TRITON_KERNELS:
        print("SKIPPED: triton_kernels not installed")
        return None

    device = "cuda"
    num_tokens = 32
    hidden = 128
    intermediate = 256
    num_experts = 8
    k = 2

    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_indices = torch.topk(router_logits, k=k, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1).to(torch.bfloat16)

    # Create random expert weights
    gate_weights = [torch.randint(0, 256, (intermediate, hidden // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    gate_scales = [torch.randint(120, 134, (intermediate, hidden // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    gate_biases = [torch.randn(intermediate, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    up_weights = [torch.randint(0, 256, (intermediate, hidden // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    up_scales = [torch.randint(120, 134, (intermediate, hidden // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    up_biases = [torch.randn(intermediate, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    down_weights = [torch.randint(0, 256, (hidden, intermediate // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    down_scales = [torch.randint(120, 134, (hidden, intermediate // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    down_biases = [torch.randn(hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

    # Reference output
    ref_output = reference_moe_forward(
        hidden_states, topk_indices, topk_weights,
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        down_weights, down_scales, down_biases,
    )

    # Setup pointer arrays (one-time at model init)
    gate_ptrs, gate_scale_ptrs = setup_expert_weight_pointers(gate_weights, gate_scales)
    up_ptrs, up_scale_ptrs = setup_expert_weight_pointers(up_weights, up_scales)
    down_ptrs, down_scale_ptrs = setup_expert_weight_pointers(down_weights, down_scales)

    # Stack biases for broadcasting [E, N]
    gate_biases_stacked = torch.stack(gate_biases, dim=0)  # [E, intermediate]
    up_biases_stacked = torch.stack(up_biases, dim=0)
    down_biases_stacked = torch.stack(down_biases, dim=0)  # [E, hidden]

    # Test the new 3D grouped implementation
    test_output = grouped_mxfp4_moe_forward_3d(
        hidden_states, topk_indices, topk_weights,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        down_ptrs, down_scale_ptrs,
        gate_weights[0], gate_scales[0],  # Reference weights for strides
        up_weights[0], up_scales[0],
        down_weights[0], down_scales[0],
        gate_biases=gate_biases_stacked,
        up_biases=up_biases_stacked,
        down_biases=down_biases_stacked,
        num_experts=num_experts,
    )

    diff = (ref_output.float() - test_output.float()).abs()
    rel_error = (diff / ref_output.abs().clamp(min=1e-6)).max().item()

    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    if rel_error < 0.05:  # 5% tolerance for grouped kernel (may have more numerical diff)
        print("PASSED")
        return True
    else:
        print("FAILED")
        return False


def test_performance_benchmark():
    """Benchmark all implementations: reference, per-expert triton_kernels, true grouped 3D."""
    print("\n" + "=" * 60)
    print("TEST 5: Performance Benchmark (GPT-OSS-120B dimensions)")
    print("=" * 60)

    if not HAS_TRITON_KERNELS:
        print("SKIPPED: triton_kernels not installed")
        return None

    device = "cuda"

    # GPT-OSS-120B realistic dimensions
    num_tokens = 32          # Typical decode batch
    hidden = 2880            # GPT-OSS hidden size
    intermediate = 5760      # GPT-OSS intermediate (2x hidden)
    num_experts = 128        # GPT-OSS num experts
    k = 4                    # GPT-OSS top-k

    print(f"Config: {num_tokens} tokens, {num_experts} experts, top-{k}")
    print(f"Dimensions: hidden={hidden}, intermediate={intermediate}")

    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_indices = torch.topk(router_logits, k=k, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1).to(torch.bfloat16)

    # Create expert weights
    print(f"\nAllocating {num_experts} expert weights...")
    gate_weights = [torch.randint(0, 256, (intermediate, hidden // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    gate_scales = [torch.randint(120, 134, (intermediate, hidden // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    gate_biases = [torch.randn(intermediate, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    up_weights = [torch.randint(0, 256, (intermediate, hidden // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    up_scales = [torch.randint(120, 134, (intermediate, hidden // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    up_biases = [torch.randn(intermediate, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    down_weights = [torch.randint(0, 256, (hidden, intermediate // 2), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    down_scales = [torch.randint(120, 134, (hidden, intermediate // 32), dtype=torch.uint8, device=device) for _ in range(num_experts)]
    down_biases = [torch.randn(hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

    # Setup pointer arrays for 3D implementation (one-time at model init)
    print("Setting up pointer arrays for true grouped GEMM...")
    gate_ptrs, gate_scale_ptrs = setup_expert_weight_pointers(gate_weights, gate_scales)
    up_ptrs, up_scale_ptrs = setup_expert_weight_pointers(up_weights, up_scales)
    down_ptrs, down_scale_ptrs = setup_expert_weight_pointers(down_weights, down_scales)

    # Stack biases for 3D implementation
    gate_biases_stacked = torch.stack(gate_biases, dim=0)
    up_biases_stacked = torch.stack(up_biases, dim=0)
    down_biases_stacked = torch.stack(down_biases, dim=0)

    n_warmup = 3
    n_iters = 10

    # ========== Benchmark: Per-expert loop (reference dequant+matmul) ==========
    print(f"\nWarming up reference per-expert loop ({n_warmup} iters)...")
    for _ in range(n_warmup):
        _ = reference_moe_forward(
            hidden_states, topk_indices, topk_weights,
            gate_weights, gate_scales, gate_biases,
            up_weights, up_scales, up_biases,
            down_weights, down_scales, down_biases,
        )
    torch.cuda.synchronize()

    print(f"Timing reference per-expert loop ({n_iters} iters)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = reference_moe_forward(
            hidden_states, topk_indices, topk_weights,
            gate_weights, gate_scales, gate_biases,
            up_weights, up_scales, up_biases,
            down_weights, down_scales, down_biases,
        )
    torch.cuda.synchronize()
    ref_time_ms = (time.perf_counter() - start) / n_iters * 1000

    # ========== Benchmark: Per-expert loop with triton_kernels (old grouped) ==========
    print(f"\nWarming up per-expert triton_kernels ({n_warmup} iters)...")
    for _ in range(n_warmup):
        _ = grouped_mxfp4_moe_forward(
            hidden_states, topk_indices, topk_weights,
            gate_weights, gate_scales, gate_biases,
            up_weights, up_scales, up_biases,
            down_weights, down_scales, down_biases,
        )
    torch.cuda.synchronize()

    print(f"Timing per-expert triton_kernels ({n_iters} iters)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = grouped_mxfp4_moe_forward(
            hidden_states, topk_indices, topk_weights,
            gate_weights, gate_scales, gate_biases,
            up_weights, up_scales, up_biases,
            down_weights, down_scales, down_biases,
        )
    torch.cuda.synchronize()
    old_grouped_time_ms = (time.perf_counter() - start) / n_iters * 1000

    # ========== Benchmark: True Grouped 3D (new implementation) ==========
    print(f"\nWarming up true grouped 3D GEMM ({n_warmup} iters)...")
    for _ in range(n_warmup):
        _ = grouped_mxfp4_moe_forward_3d(
            hidden_states, topk_indices, topk_weights,
            gate_ptrs, gate_scale_ptrs,
            up_ptrs, up_scale_ptrs,
            down_ptrs, down_scale_ptrs,
            gate_weights[0], gate_scales[0],
            up_weights[0], up_scales[0],
            down_weights[0], down_scales[0],
            gate_biases=gate_biases_stacked,
            up_biases=up_biases_stacked,
            down_biases=down_biases_stacked,
            num_experts=num_experts,
        )
    torch.cuda.synchronize()

    print(f"Timing true grouped 3D GEMM ({n_iters} iters)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = grouped_mxfp4_moe_forward_3d(
            hidden_states, topk_indices, topk_weights,
            gate_ptrs, gate_scale_ptrs,
            up_ptrs, up_scale_ptrs,
            down_ptrs, down_scale_ptrs,
            gate_weights[0], gate_scales[0],
            up_weights[0], up_scales[0],
            down_weights[0], down_scales[0],
            gate_biases=gate_biases_stacked,
            up_biases=up_biases_stacked,
            down_biases=down_biases_stacked,
            num_experts=num_experts,
        )
    torch.cuda.synchronize()
    grouped_3d_time_ms = (time.perf_counter() - start) / n_iters * 1000

    # ========== Results ==========
    print(f"\n{'Method':<35} {'Time (ms)':<12} {'Speedup':<10} {'Kernel Launches'}")
    print("-" * 75)
    print(f"{'Reference (dequant+matmul)':<35} {ref_time_ms:<12.2f} {'1.00x':<10} {num_experts}*3 = {num_experts*3}")
    print(f"{'Per-expert triton_kernels':<35} {old_grouped_time_ms:<12.2f} {ref_time_ms/old_grouped_time_ms:.2f}x{'':<5} {num_experts}*3 = {num_experts*3}")
    print(f"{'True Grouped 3D (NEW)':<35} {grouped_3d_time_ms:<12.2f} {ref_time_ms/grouped_3d_time_ms:.2f}x{'':<5} 3")

    print(f"\n*** Speedup of True Grouped 3D vs Per-expert triton_kernels: {old_grouped_time_ms/grouped_3d_time_ms:.2f}x ***")

    print("PASSED (informational)")
    return True


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return False

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"triton_kernels available: {HAS_TRITON_KERNELS}")
    print()

    results = [
        ("Token dispatch", test_token_dispatch()),
        ("MLP forward", test_mlp_forward()),
        ("Grouped MoE forward (per-expert)", test_grouped_moe_forward()),
        ("Grouped MoE forward 3D (true grouped)", test_grouped_moe_forward_3d()),
        ("Performance benchmark", test_performance_benchmark()),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        status = "PASSED" if result is True else "FAILED" if result is False else "SKIPPED"
        print(f"  {name}: {status}")

    print()
    if failed == 0 and passed > 0:
        print(f"ALL {passed} TESTS PASSED")
        return True
    else:
        print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
        return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
