#!/usr/bin/env python3
"""Sanity check: Compare grouped_mxfp4_gemm_3d against torch reference.

This script verifies the raw GEMM kernel correctness by comparing against
a reference implementation that uses dequantize + torch.mm.

Usage:
    python test/triton_kernels/sanity_check_gemm.py
"""

import sys
sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")

import torch
from batchgen.quantization.mxfp4 import mxfp4_dequantize
from batchgen.moe.mxfp4_grouped_gemm import (
    grouped_mxfp4_gemm_3d,
    setup_expert_weight_pointers,
)


def validate_triton_gemm_directly():
    """
    Validate Triton grouped_mxfp4_gemm_3d using the EXACT same pattern
    as validate_cute_fused_gemm in bench_mxfp4_moe.py (which passes).
    """
    print("=" * 70)
    print("TRITON GROUPED MXFP4 GEMM DIRECT VALIDATION")
    print("Using same pattern as CuTe validation (which passed with 0% error)")
    print("=" * 70)

    # Diagnostic: show which kernel is being used
    import batchgen.moe.mxfp4_grouped_gemm as gemm_module
    print(f"\nKernel module: {gemm_module.__file__}")
    print(f"grouped_mxfp4_gemm_3d: {gemm_module.grouped_mxfp4_gemm_3d}")

    device = "cuda"
    num_experts = 4
    tokens_per_expert = 2
    K = 128      # hidden size (small for testing)
    N = 256      # output size (intermediate)

    print(f"\nConfig: {num_experts} experts, {tokens_per_expert} tokens/expert")
    print(f"Dimensions: K={K}, N={N}")

    # Create input hidden states [E, M_max, K]
    hidden_3d = torch.randn(num_experts, tokens_per_expert, K,
                            dtype=torch.bfloat16, device=device)

    # Create weights [N, K//2] and scales [N, K//32] - SAME PATTERN as bench
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    scales = [torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    print(f"\nWeight shape: {weights[0].shape}, Scale shape: {scales[0].shape}")
    print(f"Weight strides: {weights[0].stride()}, Scale strides: {scales[0].stride()}")

    # Setup pointer arrays - SAME as bench
    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)

    # Expert counts - SAME as bench
    expert_counts = torch.full((num_experts,), tokens_per_expert,
                               dtype=torch.int32, device=device)

    # Generate reference outputs - EXACT same code as bench_mxfp4_moe.py lines 447-452
    print("\nGenerating reference outputs (PyTorch dequant + matmul)...")
    ref_outputs = []
    for i in range(num_experts):
        # Dequantize weight
        weight_bf16 = mxfp4_dequantize(weights[i], scales[i], dtype=torch.bfloat16)
        # Compute matmul: [M, K] @ [K, N] -> [M, N]
        ref_out = torch.mm(hidden_3d[i], weight_bf16.T)
        ref_outputs.append(ref_out)
    print(f"  Reference shape: {ref_outputs[0].shape}")

    # Run Triton kernel
    print("\nRunning Triton grouped_mxfp4_gemm_3d...")
    try:
        output = grouped_mxfp4_gemm_3d(
            hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
            N, weights[0], scales[0]
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  ERROR: Kernel failed with: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  Kernel output shape: {output.shape}")

    # Compare - 1% tolerance for BF16
    tolerance = 0.01  # 1% tolerance for BF16 precision
    all_pass = True
    max_rel_errors = []

    print(f"\nValidation (tolerance: {tolerance*100:.1f}%):")
    for i in range(num_experts):
        kernel_out = output[i]
        ref_out = ref_outputs[i]

        # Compute relative error - SAME formula as bench
        diff = (kernel_out.float() - ref_out.float()).abs()
        ref_abs = ref_out.float().abs().clamp(min=1e-6)
        rel_diff = (diff / ref_abs).max().item()
        abs_diff = diff.max().item()
        max_rel_errors.append(rel_diff)

        if rel_diff > tolerance:
            print(f"  FAIL expert {i}: max_rel_error={rel_diff:.4f} ({rel_diff*100:.2f}%), "
                  f"abs_diff={abs_diff:.6f}")
            all_pass = False
        else:
            print(f"  PASS expert {i}: max_rel_error={rel_diff:.6f} ({rel_diff*100:.4f}%), "
                  f"abs_diff={abs_diff:.6f}")

    avg_rel_error = sum(max_rel_errors) / len(max_rel_errors)
    print(f"\n  Average relative error: {avg_rel_error:.6f} ({avg_rel_error*100:.4f}%)")

    # Show sample values
    print(f"\nSample values (expert 0, token 0, first 4 outputs):")
    print(f"  Reference: {[f'{v:.4f}' for v in ref_outputs[0][0, :4].tolist()]}")
    print(f"  Kernel:    {[f'{v:.4f}' for v in output[0, 0, :4].tolist()]}")

    if all_pass:
        print("\n✓ TRITON GEMM PASSED VALIDATION")
    else:
        print("\n✗ TRITON GEMM FAILED VALIDATION")

        # Additional diagnostics if failed
        print("\n" + "=" * 70)
        print("FAILURE DIAGNOSTICS")
        print("=" * 70)

        # Check if dequantization works correctly
        print("\n1. Testing mxfp4_dequantize directly...")
        test_weight = torch.full((4, 2), 0x22, dtype=torch.uint8, device=device)  # FP4=1.0
        test_scale = torch.full((4, 1), 127, dtype=torch.uint8, device=device)  # exp=0
        dequant_result = mxfp4_dequantize(test_weight, test_scale, dtype=torch.bfloat16)
        print(f"   Input: FP4=1.0 everywhere, scale=127 (exp=0)")
        print(f"   Expected: all 1.0")
        print(f"   Actual: {dequant_result[0, :8].tolist()}")
        dequant_ok = torch.allclose(dequant_result.float(), torch.ones_like(dequant_result).float(), rtol=0.01)
        print(f"   Dequantize: {'PASS' if dequant_ok else 'FAIL'}")

        # Check hidden_3d and output shapes/strides
        print(f"\n2. Tensor shapes and strides:")
        print(f"   hidden_3d: shape={hidden_3d.shape}, stride={hidden_3d.stride()}, contiguous={hidden_3d.is_contiguous()}")
        print(f"   output:    shape={output.shape}, stride={output.stride()}, contiguous={output.is_contiguous()}")
        print(f"   weights[0]: shape={weights[0].shape}, stride={weights[0].stride()}, contiguous={weights[0].is_contiguous()}")
        print(f"   scales[0]:  shape={scales[0].shape}, stride={scales[0].stride()}, contiguous={scales[0].is_contiguous()}")

        # Check expert_counts
        print(f"\n3. Expert counts: {expert_counts.tolist()}")

        # Check pointer values
        print(f"\n4. Pointer arrays:")
        print(f"   weight_ptrs: {weight_ptrs.tolist()}")
        print(f"   scale_ptrs: {scale_ptrs.tolist()}")

    return all_pass


def reference_gemm(hidden_3d, weights, scales, expert_counts, N):
    """Reference: dequantize each expert's weight + torch.matmul.

    Args:
        hidden_3d: [E, M_max, K] BF16 input
        weights: List of [N, K//2] uint8 packed FP4 weights per expert
        scales: List of [N, K//32] uint8 scales per expert
        expert_counts: [E] int32 token counts
        N: Output dimension

    Returns:
        output: [E, M_max, N] BF16
    """
    E, M_max, K = hidden_3d.shape
    output = torch.zeros(E, M_max, N, dtype=torch.bfloat16, device=hidden_3d.device)

    for e in range(E):
        num_tokens = expert_counts[e].item()
        if num_tokens == 0:
            continue

        # Dequantize weight for this expert: [N, K//2] + [N, K//32] -> [N, K] BF16
        weight_bf16 = mxfp4_dequantize(weights[e], scales[e], torch.bfloat16)

        # hidden: [M, K] @ weight.T [K, N] -> [M, N]
        x = hidden_3d[e, :num_tokens, :]  # [M, K]
        output[e, :num_tokens, :] = torch.mm(x, weight_bf16.T)

    return output


def sanity_check_single_element():
    """Minimal test with 1 expert, 1 token, to diagnose the issue."""
    print("=" * 60)
    print("DIAGNOSTIC: Single element test (1 expert, 1 token, K=64, N=64)")
    print("=" * 60)

    device = "cuda"

    # Minimal dimensions
    num_experts = 1
    M_max = 1
    K = 64
    N = 64

    print(f"Config: {num_experts} experts, M_max={M_max}, K={K}, N={N}")

    # Simple input: all ones
    hidden_3d = torch.ones(num_experts, M_max, K, dtype=torch.bfloat16, device=device)

    # Simple weights: all zeros (FP4 index 0 = 0.0)
    # This should give output of all zeros
    weights = [torch.zeros((N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    # Neutral scale (127 -> exponent 0 -> 2^0 = 1)
    scales = [torch.full((N, K // 32), 127, dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    expert_counts = torch.full((num_experts,), M_max, dtype=torch.int32, device=device)
    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)

    # Test 1: All zeros weight -> output should be all zeros
    print("\nTest 1: Zero weights (FP4 index 0 = 0.0)")
    ref_output = reference_gemm(hidden_3d, weights, scales, expert_counts, N)
    kernel_output = grouped_mxfp4_gemm_3d(
        hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
        N, weights[0], scales[0]
    )
    print(f"  Reference sum: {ref_output.sum().item():.6f} (should be ~0)")
    print(f"  Kernel sum:    {kernel_output.sum().item():.6f} (should be ~0)")

    # Test 2: Weight with FP4 index 2 = 1.0 in all positions
    # Packed: low nibble=2, high nibble=2 -> byte = 0x22
    print("\nTest 2: Weight with FP4=1.0 everywhere (index 2)")
    weights_ones = [torch.full((N, K // 2), 0x22, dtype=torch.uint8, device=device)
                    for _ in range(num_experts)]
    weight_ptrs_ones, _ = setup_expert_weight_pointers(weights_ones, scales)

    # Dequant reference: each K position = 1.0, so row sum = K = 64
    # Output = input @ weight.T = [1,1,1,...] @ [1,1,1,...].T = K = 64
    ref_output2 = reference_gemm(hidden_3d, weights_ones, scales, expert_counts, N)
    kernel_output2 = grouped_mxfp4_gemm_3d(
        hidden_3d, weight_ptrs_ones, scale_ptrs, expert_counts,
        N, weights_ones[0], scales[0]
    )
    print(f"  Expected output: all 64.0 (K * 1.0)")
    print(f"  Reference [0,0,0]: {ref_output2[0, 0, 0].item():.2f}")
    print(f"  Kernel [0,0,0]:    {kernel_output2[0, 0, 0].item():.2f}")
    print(f"  Reference sum: {ref_output2.sum().item():.2f} (should be {N * 64})")
    print(f"  Kernel sum:    {kernel_output2.sum().item():.2f}")

    # Test 3: Check dequantization directly
    print("\nTest 3: Direct dequantization check")
    weight_bf16 = mxfp4_dequantize(weights_ones[0], scales[0], torch.bfloat16)
    print(f"  Dequantized weight shape: {weight_bf16.shape}")
    print(f"  Dequantized weight [0, :8]: {weight_bf16[0, :8].tolist()}")
    print(f"  Expected: all 1.0")

    return True


def sanity_check_small():
    """Test with small dimensions for easy debugging."""
    print("\n" + "=" * 60)
    print("SANITY CHECK: Small dimensions (4 experts, 2 tokens)")
    print("=" * 60)

    device = "cuda"

    # Small dimensions for debugging
    num_experts = 4
    M_max = 2  # tokens per expert
    K = 128    # hidden size
    N = 256    # output size

    print(f"Config: {num_experts} experts, M_max={M_max}, K={K}, N={N}")

    # Create random input
    hidden_3d = torch.randn(num_experts, M_max, K, dtype=torch.bfloat16, device=device)

    # Create MXFP4 weights and scales for each expert
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    scales = [torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    # All experts have M_max tokens
    expert_counts = torch.full((num_experts,), M_max, dtype=torch.int32, device=device)

    # Setup pointer arrays
    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)

    # Reference output
    print("Computing reference (dequant + torch.mm)...")
    ref_output = reference_gemm(hidden_3d, weights, scales, expert_counts, N)

    # Kernel output
    print("Computing kernel output (grouped_mxfp4_gemm_3d)...")
    kernel_output = grouped_mxfp4_gemm_3d(
        hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
        N, weights[0], scales[0]
    )

    # Compare
    diff = (ref_output.float() - kernel_output.float()).abs()
    max_abs_error = diff.max().item()
    rel_error = (diff / ref_output.abs().clamp(min=1e-6)).max().item()

    print(f"\nMax absolute error: {max_abs_error:.6f}")
    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    # Check specific values
    print(f"\nSample values (expert 0, token 0, first 4 outputs):")
    print(f"  Reference: {ref_output[0, 0, :4].tolist()}")
    print(f"  Kernel:    {kernel_output[0, 0, :4].tolist()}")

    if rel_error < 0.01:  # 1% tolerance for BF16
        print("\n✓ PASSED: Kernel matches reference within 1% tolerance")
        return True
    else:
        print(f"\n✗ FAILED: Relative error {rel_error*100:.2f}% exceeds 1% tolerance")

        # Debug: check each expert separately
        print("\nPer-expert analysis:")
        for e in range(num_experts):
            e_diff = (ref_output[e].float() - kernel_output[e].float()).abs()
            e_rel = (e_diff / ref_output[e].abs().clamp(min=1e-6)).max().item()
            print(f"  Expert {e}: max rel error = {e_rel*100:.2f}%")

        return False


def sanity_check_production():
    """Test with GPT-OSS-120B production dimensions."""
    print("\n" + "=" * 60)
    print("SANITY CHECK: Production dimensions (128 experts, 4 tokens)")
    print("=" * 60)

    device = "cuda"

    # GPT-OSS-120B dimensions
    num_experts = 128
    M_max = 4       # tokens per expert
    K = 5120        # hidden size
    N = 13824       # intermediate size

    print(f"Config: {num_experts} experts, M_max={M_max}, K={K}, N={N}")

    # Create random input
    hidden_3d = torch.randn(num_experts, M_max, K, dtype=torch.bfloat16, device=device)

    # Create MXFP4 weights and scales for each expert
    print("Allocating weights...")
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    scales = [torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    # All experts have M_max tokens
    expert_counts = torch.full((num_experts,), M_max, dtype=torch.int32, device=device)

    # Setup pointer arrays
    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)

    # Reference output (only compute first 2 experts for speed)
    print("Computing reference for first 2 experts...")
    ref_expert_counts = expert_counts.clone()
    ref_expert_counts[2:] = 0  # Only compute first 2 experts for speed
    ref_output = reference_gemm(hidden_3d, weights, scales, ref_expert_counts, N)

    # Kernel output
    print("Computing kernel output...")
    kernel_output = grouped_mxfp4_gemm_3d(
        hidden_3d, weight_ptrs, scale_ptrs, ref_expert_counts,
        N, weights[0], scales[0]
    )

    # Compare only first 2 experts
    diff = (ref_output[:2].float() - kernel_output[:2].float()).abs()
    max_abs_error = diff.max().item()
    rel_error = (diff / ref_output[:2].abs().clamp(min=1e-6)).max().item()

    print(f"\nMax absolute error: {max_abs_error:.6f}")
    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    # Check specific values
    print(f"\nSample values (expert 0, token 0, first 4 outputs):")
    print(f"  Reference: {ref_output[0, 0, :4].tolist()}")
    print(f"  Kernel:    {kernel_output[0, 0, :4].tolist()}")

    if rel_error < 0.01:  # 1% tolerance for BF16
        print("\n✓ PASSED: Kernel matches reference within 1% tolerance")
        return True
    else:
        print(f"\n✗ FAILED: Relative error {rel_error*100:.2f}% exceeds 1% tolerance")

        # Debug: check each expert separately
        print("\nPer-expert analysis (first 2 experts):")
        for e in range(2):
            e_diff = (ref_output[e].float() - kernel_output[e].float()).abs()
            e_rel = (e_diff / ref_output[e].abs().clamp(min=1e-6)).max().item()
            print(f"  Expert {e}: max rel error = {e_rel*100:.2f}%")

        return False


def sanity_check_sparse():
    """Test with sparse routing (only few experts have tokens)."""
    print("\n" + "=" * 60)
    print("SANITY CHECK: Sparse routing (4 of 128 experts active)")
    print("=" * 60)

    device = "cuda"

    # Production dimensions with sparse routing
    num_experts = 128
    M_max = 4       # max tokens per expert
    K = 5120        # hidden size
    N = 13824       # intermediate size

    # Only 4 experts have tokens (typical decode scenario)
    active_experts = [3, 17, 45, 99]

    print(f"Config: {num_experts} experts, {len(active_experts)} active, M_max={M_max}, K={K}, N={N}")

    # Create random input (only active experts have data)
    hidden_3d = torch.zeros(num_experts, M_max, K, dtype=torch.bfloat16, device=device)
    for e in active_experts:
        hidden_3d[e] = torch.randn(M_max, K, dtype=torch.bfloat16, device=device)

    # Create MXFP4 weights and scales for all experts
    print("Allocating weights...")
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    scales = [torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    # Sparse expert counts
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in active_experts:
        expert_counts[e] = M_max

    # Setup pointer arrays
    weight_ptrs, scale_ptrs = setup_expert_weight_pointers(weights, scales)

    # Reference output
    print("Computing reference...")
    ref_output = reference_gemm(hidden_3d, weights, scales, expert_counts, N)

    # Kernel output
    print("Computing kernel output...")
    kernel_output = grouped_mxfp4_gemm_3d(
        hidden_3d, weight_ptrs, scale_ptrs, expert_counts,
        N, weights[0], scales[0]
    )

    # Compare only active experts
    max_rel_error = 0.0
    print("\nPer-expert analysis:")
    for e in active_experts:
        e_diff = (ref_output[e].float() - kernel_output[e].float()).abs()
        e_rel = (e_diff / ref_output[e].abs().clamp(min=1e-6)).max().item()
        max_rel_error = max(max_rel_error, e_rel)
        print(f"  Expert {e}: max rel error = {e_rel*100:.4f}%")

    print(f"\nOverall max relative error: {max_rel_error*100:.4f}%")

    if max_rel_error < 0.01:  # 1% tolerance for BF16
        print("\n✓ PASSED: Kernel matches reference within 1% tolerance")
        return True
    else:
        print(f"\n✗ FAILED: Relative error {max_rel_error*100:.2f}% exceeds 1% tolerance")
        return False


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return False

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    results = []

    # First: Run the direct validation that matches the passing CuTe pattern
    results.append(("Triton direct validation", validate_triton_gemm_directly()))

    # Then run diagnostic test
    print("\n")
    sanity_check_single_element()

    # Run all sanity checks
    results.append(("Small dimensions", sanity_check_small()))
    results.append(("Production dimensions", sanity_check_production()))
    results.append(("Sparse routing", sanity_check_sparse()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print()
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED")
        return True
    else:
        print(f"RESULTS: {passed} passed, {failed} failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
