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

    # Step 1: Unpack Marlin int32 → nibbles [K, N]
    # Marlin packs 8 nibbles per int32 at positions 0,4,8,...,28 bits
    unpacked = torch.empty(K, N // 8, 8, dtype=torch.int32, device=device)
    for i in range(8):
        unpacked[:, :, i] = (marlin_qw >> (i * 4)) & 0xF
    q_marlin = unpacked.view(K, N)  # [K, N] nibbles in Marlin-permuted order

    # Step 2: Inverse Marlin tile permutation
    # _marlin_permute_weights: reshape [K//16, 16, N//16, 16] → permute(0,2,1,3) → perm
    # Inverse: inv_perm → permute(0,2,1,3) → reshape [K, N]
    inv_perm = _get_inverse_weight_perm(4).to(device)

    # Undo nibble-level permutation within tiles
    q_tiled = q_marlin.reshape((K // GPTQ_MARLIN_TILE, N * GPTQ_MARLIN_TILE))
    q_tiled = q_tiled.reshape((-1, inv_perm.numel()))[:, inv_perm].reshape(q_tiled.shape)

    # Undo tile transpose: [K//16, N//16, 16, 16] → permute(0,2,1,3) → [K//16, 16, N//16, 16]
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


if __name__ == "__main__":
    test_roundtrip()
