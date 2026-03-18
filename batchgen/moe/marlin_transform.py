#!/usr/bin/env python3
"""Marlin ↔ WGMMA weight layout transform (GPU + CPU reference).

Marlin layout: [K, N//8] int32, nibbles permuted per m16n8k16 MMA fragment pattern.
WGMMA layout: [N, K//2] uint8, raw sequential nibbles (2 per byte), used as int32 [N, K//8].

The transform is a fixed nibble permutation — pure memory shuffle, no arithmetic.
GPU kernel achieves ~4 μs per projection at K=7168, N=2048 on H20.

Usage:
    from batchgen.moe.marlin_transform import marlin_to_wgmma_cpu, marlin_to_wgmma_gpu
"""

import numpy as np
import torch

from batchgen.moe.marlin_weight_prep import (
    get_weight_perm,
    _get_scale_perms,
    GPTQ_MARLIN_TILE,
    INT4_GROUP_SIZE,
)


def _get_inverse_weight_perm(num_bits: int = 4) -> torch.Tensor:
    """Compute inverse of the Marlin weight permutation.

    perm maps raw_index → marlin_index.
    inv_perm maps marlin_index → raw_index.
    """
    perm = get_weight_perm(num_bits)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm))
    return inv_perm


def _get_inverse_scale_perm() -> list:
    """Compute inverse of the Marlin scale permutation."""
    scale_perm, _ = _get_scale_perms()
    inv_scale_perm = [0] * len(scale_perm)
    for i, p in enumerate(scale_perm):
        inv_scale_perm[p] = i
    return inv_scale_perm


def marlin_to_wgmma_cpu(
    marlin_qw: torch.Tensor,
    marlin_s: torch.Tensor,
    K: int, N: int,
) -> tuple:
    """CPU reference: Marlin [K, N//8] int32 → K2.5 raw [N, K//8] int32.

    Exact inverse of repack_int4_to_marlin_gs32().

    Returns:
        raw_packed: [N, K//8] int32 (K2.5 format, 8 nibbles per int32)
        raw_scales: [N, K//32] BF16 (K2.5 format)
    """
    device = marlin_qw.device

    # marlin_qw shape: [K//16, N*2] int32 (from _marlin_pack_weights)
    # Each row = one K-tile (16 K values), packed: N*16 nibbles → N*2 int32s (8 nibbles each)
    assert marlin_qw.shape == (K // GPTQ_MARLIN_TILE, N * 2), \
        f"Expected [{K // GPTQ_MARLIN_TILE}, {N * 2}], got {marlin_qw.shape}"

    # Step 1: Unpack int32 → nibbles [K//16, N*16]
    flat = marlin_qw.reshape(-1)  # flatten
    n_int32 = flat.numel()
    unpacked = torch.empty(n_int32, 8, dtype=torch.int32, device=device)
    for i in range(8):
        unpacked[:, i] = (flat >> (i * 4)) & 0xF
    q_marlin = unpacked.view(K // GPTQ_MARLIN_TILE, N * GPTQ_MARLIN_TILE)
    # Now [K//16, N*16] — nibbles in Marlin-permuted tile order

    # Step 2: Inverse nibble-level permutation within tiles
    inv_perm = _get_inverse_weight_perm(4).to(device)
    q_tiled = q_marlin.reshape((-1, inv_perm.numel()))[:, inv_perm].reshape(q_marlin.shape)

    # Step 3: Undo tile transpose
    # Forward was: [K//16, 16, N//16, 16] → permute(0,2,1,3) → [K//16, N//16, 16, 16]
    # Inverse: [K//16, N//16, 16, 16] → permute(0,2,1,3) → [K//16, 16, N//16, 16]
    q_tiled = q_tiled.reshape((K // GPTQ_MARLIN_TILE, N // GPTQ_MARLIN_TILE,
                                GPTQ_MARLIN_TILE, GPTQ_MARLIN_TILE))
    q_raw_kn = q_tiled.permute((0, 2, 1, 3)).reshape(K, N)  # [K, N] raw nibble order

    # Step 3: Transpose [K, N] → [N, K]
    q_raw_nk = q_raw_kn.t().contiguous()  # [N, K]

    # Step 4: Pack to [N, K//8] int32 (K2.5 format)
    raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
    for i in range(8):
        raw_packed |= (q_raw_nk[:, i::8] & 0xF) << (i * 4)

    # Step 5: Inverse scale permutation
    # Marlin scales: [K//32, N] permuted → raw: [K//32, N] → transpose → [N, K//32]
    inv_scale_perm = _get_inverse_scale_perm()
    s_inv = marlin_s.to(torch.float16).reshape((-1, len(inv_scale_perm)))[:, inv_scale_perm]
    raw_scales = s_inv.reshape((-1, N)).t().contiguous().to(marlin_s.dtype)  # [N, K//32]

    return raw_packed, raw_scales


_transform_module = None


def _load_transform_module():
    """Compile the fused Marlin→WGMMA transform CUDA kernel."""
    global _transform_module
    if _transform_module is not None:
        return _transform_module

    from pathlib import Path
    cu_path = Path(__file__).parent / "marlin_transform_kernel.cu"
    cuda_src = cu_path.read_text()

    launcher_code = r"""
#include <torch/extension.h>
void marlin_to_wgmma_transform(
    torch::Tensor marlin_qw, torch::Tensor raw_qw, torch::Tensor perm, int K, int N);
void marlin_to_wgmma_scale_transform(
    torch::Tensor marlin_s, torch::Tensor raw_s, torch::Tensor scale_perm, int K_groups, int N);
"""

    from torch.utils.cpp_extension import load_inline
    _transform_module = load_inline(
        name="marlin_transform",
        cpp_sources=[launcher_code],
        cuda_sources=[cuda_src],
        functions=["marlin_to_wgmma_transform", "marlin_to_wgmma_scale_transform"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90a", "--use_fast_math"],
        verbose=False,
    )
    return _transform_module


def marlin_to_wgmma_fused_gpu(
    marlin_qw: torch.Tensor,
    marlin_s: torch.Tensor,
    K: int, N: int,
) -> tuple:
    """Fused GPU transform via custom CUDA kernel. Single launch per tensor.

    Returns:
        raw_packed: [N, K//8] int32
        raw_scales: [N, K//32] BF16
    """
    device = marlin_qw.device
    mod = _load_transform_module()

    # Weight transform — need INVERSE perm (maps raw position → marlin position)
    inv_perm = _get_inverse_weight_perm(4).to(device=device, dtype=torch.int32)
    raw_packed = torch.empty(N, K // 8, dtype=torch.int32, device=device)
    mod.marlin_to_wgmma_transform(marlin_qw, raw_packed, inv_perm, K, N)

    # Scale transform — use CPU reference (scales are tiny, ~1 MB, negligible overhead)
    # The scale permutation has complex reshape interactions with group_size < K.
    # CPU version is validated via round-trip test.
    inv_scale_perm = _get_inverse_scale_perm()
    inv_scale_perm_t = torch.tensor(inv_scale_perm, device=device, dtype=torch.long)
    s_inv = marlin_s.to(torch.float16).reshape(-1, len(inv_scale_perm))
    s_inv = s_inv[:, inv_scale_perm_t]
    raw_scales = s_inv.reshape(-1, N).t().contiguous().to(marlin_s.dtype)

    return raw_packed, raw_scales


def marlin_to_wgmma_gpu(
    marlin_qw: torch.Tensor,
    marlin_s: torch.Tensor,
    K: int, N: int,
) -> tuple:
    """GPU transform: Marlin [K//16, N*2] int32 → K2.5 raw [N, K//8] int32.

    Uses native PyTorch GPU ops (no custom CUDA kernel).
    Pure memory shuffle — all ops are index permutation / reshape / transpose.

    Returns:
        raw_packed: [N, K//8] int32 (K2.5/WGMMA format)
        raw_scales: [N, K//32] BF16
    """
    device = marlin_qw.device
    assert marlin_qw.is_cuda, "marlin_to_wgmma_gpu requires CUDA tensors"

    TILE = GPTQ_MARLIN_TILE  # 16

    # Step 1: Unpack int32 → nibbles [K//16, N*16]
    # Each int32 contains 8 nibbles. Shape [K//16, N*2] → flatten → unpack → reshape
    flat = marlin_qw.reshape(-1)
    # Vectorized unpack: shift each int32 by 0,4,8,...,28 bits, mask to 4 bits
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4  # [8]
    nibbles = ((flat.unsqueeze(1) >> shifts) & 0xF)  # [n_int32, 8]
    q_marlin = nibbles.reshape(K // TILE, N * TILE)  # [K//16, N*16]

    # Step 2: Inverse nibble permutation within tiles
    inv_perm = _get_inverse_weight_perm(4).to(device)
    q_tiled = q_marlin.reshape(-1, inv_perm.numel())  # [n_tile_rows, 1024]
    q_tiled = q_tiled[:, inv_perm]  # apply inverse permutation
    q_tiled = q_tiled.reshape(K // TILE, N * TILE)

    # Step 3: Undo tile transpose [K//16, N//16, 16, 16] → [K//16, 16, N//16, 16] → [K, N]
    q_tiled = q_tiled.reshape(K // TILE, N // TILE, TILE, TILE)
    q_raw_kn = q_tiled.permute(0, 2, 1, 3).reshape(K, N)

    # Step 4: Transpose [K, N] → [N, K]
    q_raw_nk = q_raw_kn.t().contiguous()

    # Step 5: Pack nibbles → int32 [N, K//8] (vectorized)
    q_groups = q_raw_nk.reshape(N, K // 8, 8)  # [N, K//8, 8]
    shifts_pack = torch.arange(8, device=device, dtype=torch.int32) * 4
    raw_packed = (q_groups << shifts_pack).sum(dim=-1).to(torch.int32)

    # Step 6: Inverse scale permutation
    inv_scale_perm = _get_inverse_scale_perm()
    inv_scale_perm_t = torch.tensor(inv_scale_perm, device=device, dtype=torch.long)
    s_inv = marlin_s.to(torch.float16).reshape(-1, len(inv_scale_perm))
    s_inv = s_inv[:, inv_scale_perm_t]
    raw_scales = s_inv.reshape(-1, N).t().contiguous().to(marlin_s.dtype)

    return raw_packed, raw_scales


def test_roundtrip():
    """Verify: K2.5 raw → Marlin → WGMMA raw is bit-identical."""
    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32

    for K, N in [(7168, 2048), (2048, 7168)]:
        print(f"\nRound-trip test K={K} N={N}:")

        # Create K2.5 raw weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        w = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1
        n_groups = K // 32
        w_grouped = w.view(N, n_groups, 32)
        max_val = w_grouped.max(dim=-1, keepdim=True).values
        min_val = w_grouped.min(dim=-1, keepdim=True).values
        scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)
        q = torch.round(w_grouped / scales).int() + 8
        q = torch.clamp(q, 0, 15)
        q_flat = q.view(N, K).int()

        # Pack as K2.5 int32
        raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
        for i in range(8):
            raw_packed |= (q_flat[:, i::8] & 0xF) << (i * 4)
        raw_scales = scales.squeeze(-1).to(torch.bfloat16)

        # Forward: K2.5 → Marlin
        marlin_qw, marlin_s = repack_int4_to_marlin_gs32(raw_packed, raw_scales, K, N)

        # Reverse: Marlin → K2.5
        recovered_packed, recovered_scales = marlin_to_wgmma_cpu(marlin_qw, marlin_s, K, N)

        # Compare
        weight_match = torch.equal(raw_packed, recovered_packed)
        scale_match = torch.equal(raw_scales, recovered_scales)

        print(f"  Weights bit-identical: {weight_match}")
        print(f"  Scales bit-identical:  {scale_match}")
        if not weight_match:
            diff = (raw_packed != recovered_packed).sum().item()
            print(f"  Weight mismatches: {diff} / {raw_packed.numel()}")
        if not scale_match:
            diff = (raw_scales != recovered_scales).sum().item()
            print(f"  Scale mismatches: {diff} / {raw_scales.numel()}")

        status = "PASS" if weight_match and scale_match else "FAIL"
        print(f"  [{status}]")


def test_gpu_vs_cpu():
    """Verify GPU transform matches CPU reference (bit-identical)."""
    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32

    for K, N in [(7168, 2048), (2048, 7168)]:
        print(f"\nGPU vs CPU test K={K} N={N}:")
        device = "cuda"

        # Create K2.5 raw weights
        w = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1
        n_groups = K // 32
        w_grouped = w.view(N, n_groups, 32)
        max_val = w_grouped.max(dim=-1, keepdim=True).values
        min_val = w_grouped.min(dim=-1, keepdim=True).values
        scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)
        q = torch.round(w_grouped / scales).int() + 8
        q = torch.clamp(q, 0, 15)
        q_flat = q.view(N, K).int()
        raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
        for i in range(8):
            raw_packed |= (q_flat[:, i::8] & 0xF) << (i * 4)
        raw_scales = scales.squeeze(-1).to(torch.bfloat16)

        # Forward: K2.5 → Marlin
        marlin_qw, marlin_s = repack_int4_to_marlin_gs32(raw_packed, raw_scales, K, N)

        # CPU transform
        cpu_packed, cpu_scales = marlin_to_wgmma_cpu(marlin_qw, marlin_s, K, N)

        # GPU transform
        gpu_packed, gpu_scales = marlin_to_wgmma_gpu(marlin_qw, marlin_s, K, N)

        w_match = torch.equal(cpu_packed, gpu_packed)
        s_match = torch.equal(cpu_scales, gpu_scales)
        rt_match = torch.equal(raw_packed, gpu_packed)

        print(f"  GPU vs CPU weights: {'MATCH' if w_match else 'MISMATCH'}")
        print(f"  GPU vs CPU scales:  {'MATCH' if s_match else 'MISMATCH'}")
        print(f"  GPU round-trip:     {'MATCH' if rt_match else 'MISMATCH'}")
        status = "PASS" if (w_match and s_match and rt_match) else "FAIL"
        print(f"  [{status}]")


def bench_gpu_transform():
    """Benchmark GPU transform kernel timing."""
    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32

    device = "cuda"
    iters = 100

    print(f"\nGPU Transform Benchmark (iters={iters}):")
    print(f"Device: {torch.cuda.get_device_name()}")

    for K, N, label in [(7168, 2048, "gate/up"), (2048, 7168, "down")]:
        # Create Marlin weights
        w = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1
        n_groups = K // 32
        w_grouped = w.view(N, n_groups, 32)
        max_val = w_grouped.max(dim=-1, keepdim=True).values
        min_val = w_grouped.min(dim=-1, keepdim=True).values
        scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)
        q = torch.round(w_grouped / scales).int() + 8
        q = torch.clamp(q, 0, 15)
        q_flat = q.view(N, K).int()
        raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
        for i in range(8):
            raw_packed |= (q_flat[:, i::8] & 0xF) << (i * 4)
        raw_scales = scales.squeeze(-1).to(torch.bfloat16)
        marlin_qw, marlin_s = repack_int4_to_marlin_gs32(raw_packed, raw_scales, K, N)

        # Warmup
        for _ in range(10):
            marlin_to_wgmma_gpu(marlin_qw, marlin_s, K, N)

        # Benchmark
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            marlin_to_wgmma_gpu(marlin_qw, marlin_s, K, N)
        e.record()
        torch.cuda.synchronize()
        us_per = s.elapsed_time(e) / iters * 1000

        data_mb = K * N / 2 / 1e6  # INT4 packed size
        bw_tb = data_mb * 2 / (us_per * 1e-6) / 1e6  # read + write

        print(f"  {label:8s} K={K:5d} N={N:5d}: {us_per:8.1f} us | {data_mb:.1f} MB | {bw_tb:.2f} TB/s")

    # Per-expert total (3 projections)
    print(f"\n  Per-expert (3 projections): ~{3 * us_per:.0f} us")
    print(f"  Per-layer (24 experts):     ~{24 * 3 * us_per / 1000:.1f} ms")
    print(f"  vs WGMMA compute per layer: ~1200 us → transform overhead shown above")


def test_fused_gpu():
    """Verify fused CUDA transform matches CPU reference (bit-identical)."""
    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32

    for K, N in [(7168, 2048), (2048, 7168)]:
        print(f"\nFused GPU test K={K} N={N}:")
        device = "cuda"

        w = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1
        n_groups = K // 32
        w_grouped = w.view(N, n_groups, 32)
        max_val = w_grouped.max(dim=-1, keepdim=True).values
        min_val = w_grouped.min(dim=-1, keepdim=True).values
        scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)
        q = torch.round(w_grouped / scales).int() + 8
        q = torch.clamp(q, 0, 15)
        q_flat = q.view(N, K).int()
        raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
        for i in range(8):
            raw_packed |= (q_flat[:, i::8] & 0xF) << (i * 4)
        raw_scales = scales.squeeze(-1).to(torch.bfloat16)

        marlin_qw, marlin_s = repack_int4_to_marlin_gs32(raw_packed, raw_scales, K, N)

        # Fused GPU transform
        fused_packed, fused_scales = marlin_to_wgmma_fused_gpu(marlin_qw, marlin_s, K, N)
        torch.cuda.synchronize()

        w_match = torch.equal(raw_packed, fused_packed)
        s_match = torch.equal(raw_scales, fused_scales)

        print(f"  Weights round-trip: {'MATCH' if w_match else 'MISMATCH'}")
        print(f"  Scales round-trip:  {'MATCH' if s_match else 'MISMATCH'}")
        if not w_match:
            diff = (raw_packed != fused_packed).sum().item()
            print(f"  Weight mismatches: {diff} / {raw_packed.numel()}")
        status = "PASS" if (w_match and s_match) else "FAIL"
        print(f"  [{status}]")


def bench_fused_gpu_transform():
    """Benchmark fused CUDA transform kernel timing."""
    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32

    device = "cuda"
    iters = 1000

    print(f"\nFused GPU Transform Benchmark (iters={iters}):")
    print(f"Device: {torch.cuda.get_device_name()}")

    for K, N, label in [(7168, 2048, "gate/up"), (2048, 7168, "down")]:
        w = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1
        n_groups = K // 32
        w_grouped = w.view(N, n_groups, 32)
        max_val = w_grouped.max(dim=-1, keepdim=True).values
        min_val = w_grouped.min(dim=-1, keepdim=True).values
        scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)
        q = torch.round(w_grouped / scales).int() + 8
        q = torch.clamp(q, 0, 15)
        q_flat = q.view(N, K).int()
        raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
        for i in range(8):
            raw_packed |= (q_flat[:, i::8] & 0xF) << (i * 4)
        raw_scales = scales.squeeze(-1).to(torch.bfloat16)
        marlin_qw, marlin_s = repack_int4_to_marlin_gs32(raw_packed, raw_scales, K, N)

        # Pre-allocate output
        perm_t = _get_inverse_weight_perm(4).to(device=device, dtype=torch.int32)
        out_packed = torch.empty(N, K // 8, dtype=torch.int32, device=device)
        K_groups = K // 32
        out_scales = torch.empty(N, K_groups, dtype=torch.bfloat16, device=device)
        mod = _load_transform_module()

        inv_sp = torch.tensor(_get_inverse_scale_perm(), device=device, dtype=torch.int32)

        # Warmup
        for _ in range(10):
            mod.marlin_to_wgmma_transform(marlin_qw, out_packed, perm_t, K, N)
            mod.marlin_to_wgmma_scale_transform(marlin_s, out_scales, inv_sp, K_groups, N)

        # Benchmark weights only
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            mod.marlin_to_wgmma_transform(marlin_qw, out_packed, perm_t, K, N)
        e.record()
        torch.cuda.synchronize()
        w_us = s.elapsed_time(e) / iters * 1000

        # Benchmark scales only
        s.record()
        for _ in range(iters):
            mod.marlin_to_wgmma_scale_transform(marlin_s, out_scales, inv_sp, K_groups, N)
        e.record()
        torch.cuda.synchronize()
        s_us = s.elapsed_time(e) / iters * 1000

        data_mb = K * N / 2 / 1e6
        bw_tb = data_mb * 2 / (w_us * 1e-6) / 1e6

        print(f"  {label:8s} K={K:5d} N={N:5d}: weights={w_us:7.1f} us, scales={s_us:5.1f} us | {data_mb:.1f} MB | {bw_tb:.2f} TB/s")

    print(f"\n  vs PyTorch-based GPU transform: ~890 us (220× faster target)")


if __name__ == "__main__":
    import sys
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"
    if test_name == "all":
        test_roundtrip()
        test_gpu_vs_cpu()
        test_fused_gpu()
        bench_fused_gpu_transform()
    elif test_name == "roundtrip":
        test_roundtrip()
    elif test_name == "gpu":
        test_gpu_vs_cpu()
    elif test_name == "fused":
        test_fused_gpu()
        bench_fused_gpu_transform()
    elif test_name == "bench":
        bench_fused_gpu_transform()
    else:
        print(f"Unknown: {test_name}. Use: all, roundtrip, gpu, fused, bench")
