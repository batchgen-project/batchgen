#!/usr/bin/env python3
"""Standalone single-expert dequant parity test for the Kimi-K2.5 TP-MoE path.

Locks the #1 correctness risk: that the int4 uint8 layout + scales we hand to
SGLang's ``fused_experts`` (``use_int4_w4a16``) decode to EXACTLY the same BF16
weights as BatchGen's marlin INT4 reference, for this rank's TP (1/world_size)
intermediate slice.

It validates, for one expert, three things:
  (1) NIBBLE/BYTE ORDER (integer, bit-exact): unpacking ``raw_packed`` the
      BatchGen way (8 nibbles/int32) vs the SGLang kernel way
      (``.view(torch.uint8)`` little-endian, low nibble = even k) yields the
      same per-element int4 codes.
  (2) FULL DEQUANT (allclose): ``_dequantize_k25_int4`` vs the SGLang int4
      decode ``(nibble - 8) * scale[:, k // group_size]``.
  (3) END-TO-END TP TRANSFORM: starting from realistic marlin tensors
      (raw → marlin → raw round-trip), apply the exact slice/cat/view pipeline
      from ``_load_tp_moe_experts`` for gate|up (w13) and down (w2), then compare
      the SGLang dequant of the uint8 slabs to the BatchGen dequant of the
      corresponding raw slices.

GPU-only (the marlin transform kernel runs on CUDA). Run on a GPU host:
    python test/tp_moe_repack_parity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from batchgen.moe.marlin_weight_prep import _dequantize_k25_int4
from batchgen.moe.marlin_transform import (
    raw_to_marlin_fused_gpu,
    marlin_to_wgmma_fused_gpu,
)

# Kimi-K2.5 constants
H = 7168              # hidden dim
N_INTER = 2048        # moe_intermediate
GROUP_SIZE = 32       # INT4 group size (gs=32)
WORLD_SIZE = 16       # tp16
INTER_PR = N_INTER // WORLD_SIZE  # 128


# --------------------------------------------------------------------------- #
# Builders / SGLang-side reference decode
# --------------------------------------------------------------------------- #
def build_raw_int4(N, K, device, seed):
    """Build a random K2.5 raw INT4 expert projection.

    Returns raw_packed [N, K//8] int32 (8 nibbles/int32), raw_scales [N, K//32] bf16.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device, generator=g)
    raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
    for i in range(8):
        raw_packed |= (q[:, i::8] & 0xF) << (i * 4)
    raw_scales = (torch.rand(N, K // 32, device=device, generator=g) * 0.1 + 0.01).to(torch.bfloat16)
    return raw_packed, raw_scales


def batchgen_codes(raw_packed, K, N):
    """BatchGen unpack: [N, K] int4 codes (nibble - 8), nibble i in int32 bits [i*4:(i+1)*4]."""
    unpacked = torch.empty(N, K // 8, 8, dtype=torch.int32, device=raw_packed.device)
    for i in range(8):
        unpacked[:, :, i] = ((raw_packed >> (i * 4)) & 0xF) - 8
    return unpacked.view(N, K)


def sglang_codes(raw_packed, K, N):
    """SGLang kernel unpack of raw_packed.view(uint8): [N, K] codes (nibble - 8).

    byte j (uint8) holds: low nibble (bits 0-3) = K-element 2j, high nibble = 2j+1.
    """
    u8 = raw_packed.view(torch.uint8).to(torch.int32)  # [N, K//2], little-endian
    low = (u8 & 0xF) - 8                                # even k
    high = ((u8 >> 4) & 0xF) - 8                        # odd k
    return torch.stack([low, high], dim=2).reshape(N, K)


def sglang_int4_dequant(w_u8, scale, K, group_size=GROUP_SIZE):
    """Replicate SGLang fused_moe int4_w4a16 weight decode.

    w_u8: [Nout, K//2] uint8, scale: [Nout, K//group_size] (bf16).
    Returns [K, Nout] float32 — the (input_dim, output_dim) weight the GEMM sees.
    """
    Nout = w_u8.shape[0]
    u8 = w_u8.to(torch.int32)
    low = (u8 & 0xF)
    high = ((u8 >> 4) & 0xF)
    nib = torch.stack([low, high], dim=2).reshape(Nout, K)  # k even = low nibble
    vals = nib.float() - 8.0
    ng = K // group_size
    vals = (vals.view(Nout, ng, group_size) * scale.float().unsqueeze(-1)).view(Nout, K)
    return vals.t().contiguous()  # [K, Nout]


def _check(name, ref, got, atol, rtol):
    ref = ref.float()
    got = got.float()
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    max_abs = (ref - got).abs().max().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max_abs_diff={max_abs:.3e}")
    return ok


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_nibble_order(device):
    """Level 1: integer-exact nibble/byte order lock (the #1 risk)."""
    print("Level 1 — nibble/byte order (integer, bit-exact):")
    ok = True
    for name, (N, K, seed) in {
        "gate/up": (N_INTER, H, 1),
        "down": (H, N_INTER, 2),
    }.items():
        raw, _ = build_raw_int4(N, K, device, seed)
        bg = batchgen_codes(raw, K, N)
        sg = sglang_codes(raw, K, N)
        exact = torch.equal(bg, sg)
        print(f"  [{'PASS' if exact else 'FAIL'}] {name} int4 codes equal: {exact}")
        ok = ok and exact
    return ok


def test_full_dequant(device):
    """Level 2: full dequant parity (BatchGen ref vs SGLang decode)."""
    print("Level 2 — full dequant parity (no slicing):")
    raw, scales = build_raw_int4(N_INTER, H, device, seed=3)
    w_bg = _dequantize_k25_int4(raw, scales, H, N_INTER)        # [H, N_INTER] fp16
    w_sg = sglang_int4_dequant(raw.view(torch.uint8), scales, H)  # [H, N_INTER] f32
    return _check("gate dequant", w_bg, w_sg, atol=1e-2, rtol=1e-2)


def test_tp_transform(device, rank=3):
    """Level 3: end-to-end TP slab transform via marlin round-trip + slicing.

    Mirrors _load_tp_moe_experts exactly for one expert at the given rank.
    """
    print(f"Level 3 — end-to-end TP transform (rank={rank}):")
    r0, r1 = rank * INTER_PR, (rank + 1) * INTER_PR
    dcol0 = rank * (INTER_PR // 8)
    scol0 = rank * (INTER_PR // GROUP_SIZE)
    ok = True

    # gate|up (w13): raw -> marlin (sim. checkpoint) -> marlin_to_wgmma -> slice rows
    def proj_through_marlin(N, K, seed):
        raw0, s0 = build_raw_int4(N, K, device, seed)
        mqw, ms = raw_to_marlin_fused_gpu(raw0, s0, K, N)
        raw, s = marlin_to_wgmma_fused_gpu(mqw, ms, K, N)
        # round-trip sanity: marlin transform must be loss-free on the packed codes
        ok_rt = torch.equal(raw, raw0)
        print(f"  [{'PASS' if ok_rt else 'FAIL'}] marlin round-trip (N={N},K={K}) packed identical: {ok_rt}")
        return raw, s, ok_rt

    raw_g, raw_gs, ok_g = proj_through_marlin(N_INTER, H, seed=11)
    raw_u, raw_us, ok_u = proj_through_marlin(N_INTER, H, seed=12)
    raw_d, raw_ds, ok_d = proj_through_marlin(H, N_INTER, seed=13)
    ok = ok and ok_g and ok_u and ok_d

    # --- w13 (gate|up) ---
    w13_u8 = torch.cat([raw_g[r0:r1], raw_u[r0:r1]], dim=0).contiguous().view(torch.uint8)
    w13_scale = torch.cat([raw_gs[r0:r1], raw_us[r0:r1]], dim=0)
    w13_sg = sglang_int4_dequant(w13_u8, w13_scale, H)  # [H, 2*INTER_PR]

    gate_ref = _dequantize_k25_int4(raw_g, raw_gs, H, N_INTER)[:, r0:r1]  # [H, INTER_PR]
    up_ref = _dequantize_k25_int4(raw_u, raw_us, H, N_INTER)[:, r0:r1]
    w13_ref = torch.cat([gate_ref, up_ref], dim=1)  # [H, 2*INTER_PR]
    ok = _check("w13 (gate|up) slab", w13_ref, w13_sg, atol=1e-2, rtol=1e-2) and ok

    # --- w2 (down) ---
    w2_u8 = raw_d[:, dcol0:dcol0 + INTER_PR // 8].contiguous().view(torch.uint8)
    w2_scale = raw_ds[:, scol0:scol0 + INTER_PR // GROUP_SIZE]
    w2_sg = sglang_int4_dequant(w2_u8, w2_scale, INTER_PR)  # [INTER_PR, H]

    w2_ref = _dequantize_k25_int4(raw_d, raw_ds, N_INTER, H)[r0:r1, :]  # [INTER_PR, H]
    ok = _check("w2 (down) slab", w2_ref, w2_sg, atol=1e-2, rtol=1e-2) and ok
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required (marlin transform kernel is GPU-only).")
        return 0
    device = torch.device("cuda")
    print(f"TP-MoE repack parity test (H={H}, inter={N_INTER}, tp={WORLD_SIZE}, inter_pr={INTER_PR})\n")
    results = [
        test_nibble_order(device),
        test_full_dequant(device),
        test_tp_transform(device, rank=3),
    ]
    passed = all(results)
    print(f"\n{'ALL PASS' if passed else 'FAILURES PRESENT'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
