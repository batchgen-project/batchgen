#!/usr/bin/env python3
"""STAGED GPU parity ladder for the K3 Marlin-MXFP4 kernels (task #34).

NOT run by the CPU workflow. Staged for <gpu-host>, **GPU 1** (GPU 0
belongs to the model workstream). No benchmarks here — parity only.

How to run (<gpu-host>):
    ssh <gpu-host>
    docker exec -it <batchgen container> bash
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate batchgen
    cd <BatchGen repo root>          # sync via git pull — never scp into the repo

    # Build + register-tier gate (runs `setup.py build_ext --inplace` with
    # `-Xptxas -v` captured, greps per-kernel register counts, and FAILS if an
    # E2M1 instantiation jumps more than 16 regs past its U4B8 counterpart —
    # the INT4 M16 tier is ~130 regs / 12.5%% occupancy):
    CUDA_VISIBLE_DEVICES=1 python tests/moe/gpu_parity_mxfp4_marlin.py --build

    CUDA_VISIBLE_DEVICES=1 python tests/moe/gpu_parity_mxfp4_marlin.py --smoke
    CUDA_VISIBLE_DEVICES=1 python tests/moe/gpu_parity_mxfp4_marlin.py
    # v2 kernels only (optionally a subset of variants):
    CUDA_VISIBLE_DEVICES=1 python tests/moe/gpu_parity_mxfp4_marlin.py --v2 \
        --v2-variants 0,2
    # v3 kernels only (optionally a subset of variants):
    CUDA_VISIBLE_DEVICES=1 python tests/moe/gpu_parity_mxfp4_marlin.py --v3 \
        --v3-variants 0

Ladder (tolerance gate per KERNEL_WORKUNIT.md: tol = 1e-5 + 1.6e-2*|ref|,
PASS iff finite AND fail_frac < 1e-4, plus max relative error < 1.6e-2 on the
well-conditioned subset |ref| > 0.1*rms(ref). fail_frac == 0.0 is NOT
asserted — fp32 summation-order noise on cancelled outputs):

  T0 SMOKE      single expert, t=16, K3 shapes, finite output.
  T1 DECODE     bit-exact in-kernel E2M1+E8M0 decode via one-hot activations
                (no accumulation => exactness is legitimate). +-0.5-heavy
                codes specifically pin the bf16-subnormal rebias path of
                dequant_e2m1 (eem=1 -> 0x0040 * 2^126). Zeros are
                CANONICALIZED before the bit compare: nibble 0x8 decodes to
                -0.0 in the reference, but the kernel's fp32 accumulator
                starts at +0.0 and (+0.0) + (1.0 * -0.0) = +0.0 under IEEE
                RN, so a CORRECT kernel emits +0.0 there. Bits stay strict
                for every nonzero output.
  T2 M16        grouped_marlin_gemm_m16_mxfp4 vs oracle-dequant + matmul at
                both K3 shapes, M sweep {1,16,63,64,65,512,4096}.
  T3 S1+SiTU    grouped_marlin_gemm_m16_s1_mxfp4_situ vs eager fp32 SiTU
                reference (modeling_kimi_linear.py:75-82 semantics), M sweep.
  T4 GROUPED    E=32 dense grid with >half zero-token experts (caller does
                NOT filter empties — C-EMPTYGRP). Uses the PRODUCTION fused
                layout: per-expert stacked [2, K//16, N*2] blobs from
                repack_mxfp4_w13_to_marlin_gs32, with up pointers derived by
                byte arithmetic off the gate pointers (storage adjacency).
                The all-empty launch must write NOTHING (asserted via a NaN
                canary in the output buffer).
  T5 MUTATIONS  each deliberately-broken arm must FAIL the gate:
                  m1 INT4 (u4b8+SiLU) entry consuming E2M1 tensors
                  m2 gate/up pointer swap (silent at kernel level otherwise)
                  m3 off-by-one scale group (rolled marlin scales)
                  m4 SiLU-instead-of-SiTU reference vs the SiTU kernel
                Reports the catch count — must be 4/4.
  T6 REGRESSION K2.5 INT4 M16 through the templated <U4B8> instantiation
                still passes its own parity (templating did not disturb the
                production kernels).
  T6b REGRESSION K2.5 fused S1 <U4B8, SILU> — the decode DEFAULT and the
                kernel that got the heavier template surgery (epilogue
                swapped to act_gate_mul<ACT>) — vs SiLU(gate)*up reference.
  T7 NEGATIVES  every wrapper hard-fail check (L2/L3/L4/L5 + activation
                contract) and the raw-pybind TORCH_CHECK seam must RAISE.
                Reports the catch count — must be N/N.

Version 2 ladder (grouped_marlin_gemm_m16{,_s1}_mxfp4{,_situ}_v2), run for
each variant 0..2 = (TM, TN) (1,16) (2,16) (2,8):

  T1v2 DECODE   T1 through the v2 entry (64 one-hot rows => several
                super-tiles per CTA). Scale bytes span 100..128 so the
                folded-scale upper bound (byte 128) is exercised.
  T2v2 M16      oracle gate at M in {1,4,9,16,17,24,32,33,63,65}, both K3
                shapes, two experts through pointer arrays with nonzero
                expert_starts, on the tight AND a production-like max_m_tiles
                grid (outputs must be bit-identical: v2 ignores the value);
                NaN canary rows past each expert's count; v1-vs-v2 agreement
                (fraction differing by > 1 bf16 ulp of |v1| must be < 1e-3;
                max abs diff and bit-identical fraction reported).
  T3v2 S1+SiTU  same as T2v2 for the fused S1 entry on the production
                stacked w13 layout (up pointers by byte arithmetic).
  T4v2 GROUPED  E=28 with skewed counts incl. 0,1,9,16,17,33,40,65 for the
                fused S1 and the S3 grouped entry, NaN canaries, and an
                all-empty launch that must write nothing.
  T5v2 MUTATIONS gate/up swap and rolled scale groups per variant must FAIL
                the gate (unmutated control must PASS); a scale byte > 128
                must be rejected by check_mxfp4_scales_v2_compatible (byte
                128 accepted); an out-of-range variant must raise at the
                wrapper and the raw-pybind seam.

Version 3 ladder (grouped_marlin_gemm_m16{,_s1}_mxfp4{,_situ}_v3, swap-AB
wgmma, sm_90a), run for each variant 0..2 = (JW, NCAP) (2,64) (2,48) (4,32):

  T1v3 DECODE   one-hot bit-exact at 64 and 72 rows (one and two row chunks
                for NCAP=64) on 512x512, uniform and half-heavy codes. Pins
                the A-fragment layout, the swizzled B descriptor and the
                epilogue mapping at once.
  T2v3 M16      oracle gate at M in {1,4,8,9,16,17,24,32,33,40,64,65} (every
                N width 8/16/32/48/64 and chunk boundary), both K3 shapes, two
                experts with nonzero expert_starts, tight AND production-like
                max_m_tiles grids (bit-identical), NaN canaries, v1 agreement.
  T3v3 S1+SiTU  same as T2v3 for the fused S1 entry on the stacked w13 layout.
  T4v3 GROUPED  E=28 skewed counts incl. 0, 1, 8, 9, 64, 65 and 70 rows, fused
                S1 and S3, with and without expert_order (bit-identical), NaN
                canaries, all-empty launch writes nothing.
  T5v3 MUTATIONS gate/up swap and rolled scale groups must FAIL the gate
                (control must PASS); variant 3 must raise at the wrapper and
                the raw-pybind seam.
  T6v3 GRAPH    expert order + S1 + S3 captured in one CUDA graph, replayed
                twice; both replays bit-identical to the eager launches.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

from batchgen.moe import marlin_grouped_moe as mgm
from batchgen.moe import marlin_weight_prep as mwp
from batchgen.moe import mxfp4_oracle_vector as oracle

DEV = "cuda"
K3_K, K3_N = 3584, 3072          # w1/w3 branch: prob_k, prob_n
M_SWEEP = [1, 16, 63, 64, 65, 512, 4096]
SCALE_LO, SCALE_HI = 112, 122    # observed K3 range (frozen verdict)

_results = []


def report(name, ok, detail=""):
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def gate(out, ref, name, verbose=True):
    """Project numerical gate. Returns True iff the gate PASSES. With
    verbose=False only a failing gate prints."""
    out = out.float()
    ref = ref.float()
    finite = bool(torch.isfinite(out).all())
    err = (out - ref).abs()
    tol = 1e-5 + 1.6e-2 * ref.abs()
    fail_frac = float((err > tol).float().mean())
    rms = float(ref.pow(2).mean().sqrt())
    mask = ref.abs() > 0.1 * rms
    max_rel = float((err[mask] / ref.abs()[mask]).max()) if mask.any() else 0.0
    passed = finite and fail_frac < 1e-4 and max_rel < 1.6e-2
    tag = "gate-PASS" if passed else "gate-FAIL"
    if verbose or not passed:
        print(f"    {name}: {tag} fail_frac={fail_frac:.2e} max_rel={max_rel:.2e} "
              f"finite={finite}")
    return passed


def rand_expert(K, N, seed, half_heavy=False, scale_lo=SCALE_LO, scale_hi=SCALE_HI):
    """Random MXFP4 expert weight: packed [N, K//2] u8 + scale [N, K//32] u8."""
    g = torch.Generator().manual_seed(seed)
    if half_heavy:
        # mostly +-0.5 codes (nibbles 0x1/0x9) to hammer the subnormal path
        nib = torch.where(torch.rand(N, K, generator=g) < 0.8,
                          torch.where(torch.rand(N, K, generator=g) < 0.5,
                                      torch.tensor(0x1), torch.tensor(0x9)),
                          torch.randint(0, 16, (N, K), generator=g))
        packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    else:
        packed = torch.randint(0, 256, (N, K // 2), generator=g,
                               dtype=torch.int16).to(torch.uint8)
    scale = torch.randint(scale_lo, scale_hi + 1, (N, K // 32), generator=g,
                          dtype=torch.int16).to(torch.uint8)
    return packed, scale


def marlinize(packed, scale, K, N):
    """CPU repack -> CUDA Marlin tensors with native uint8 E8M0 scales."""
    qw, s = mwp.repack_mxfp4_to_marlin_gs32(packed, scale, K, N,
                                             emit_scale="e8m0")
    return qw.to(DEV), s.to(DEV)


def situ_ref_fp32(g, u):
    """Eager SiTU (modeling_kimi_linear.py:75-82; beta=4, linear_beta=25)."""
    g = g.float()
    u = u.float()
    a = 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)
    return a * (25.0 * torch.tanh(u / 25.0))


def silu_ref_fp32(g, u):
    g = g.float()
    u = u.float()
    return g * torch.sigmoid(g) * u


def dense_expert_bf16(packed, scale):
    """Oracle-dequant reference weight [N, K] bf16 (exact)."""
    return oracle.mxfp4_dequantize_oracle(packed, scale).to(DEV)


# ---------------------------------------------------------------------------

def t0_smoke():
    torch.manual_seed(0)
    w = {}
    for name, (K, N) in (("w1", (K3_K, K3_N)), ("w3", (K3_K, K3_N)),
                         ("w2", (K3_N, K3_K))):
        p, s = rand_expert(K, N, seed=hash(name) % 2**31)
        w[name] = (marlinize(p, s, K, N), (p, s))
    x = torch.randn(16, K3_K, dtype=torch.bfloat16, device=DEV)
    out = mgm.single_expert_marlin_mxfp4_decode(
        x,
        *w["w1"][0], *w["w3"][0], *w["w2"][0],
        N=K3_N, K=K3_K)
    report("T0 smoke", bool(torch.isfinite(out).all()),
           f"out {tuple(out.shape)} finite")


def t1_decode_bitexact():
    """One-hot activations => out row m = dequant(W)[:, k_m] with a single
    nonzero product per accumulator: bit-exact vs the oracle is legitimate."""
    K, N = 512, 512
    ok = True
    for tag, heavy, lo, hi in (("uniform", False, 100, 140),
                               ("half-heavy", True, SCALE_LO, SCALE_HI)):
        p, s = rand_expert(K, N, seed=101, half_heavy=heavy,
                           scale_lo=lo, scale_hi=hi)
        qw, ms = marlinize(p, s, K, N)
        w_ref = dense_expert_bf16(p, s)  # [N, K]
        k_idx = torch.randperm(K)[:64]
        A = torch.zeros(64, K, dtype=torch.bfloat16, device=DEV)
        A[torch.arange(64), k_idx] = 1.0

        mod = mgm._load_module()
        C = torch.empty(64, N, dtype=torch.bfloat16, device=DEV)
        starts = torch.zeros(1, dtype=torch.int32, device=DEV)
        counts = torch.tensor([64], dtype=torch.int32, device=DEV)
        ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)

        def ptr(t):
            return torch.tensor([t.data_ptr()], dtype=torch.int64, device=DEV)

        mod.grouped_marlin_gemm_m16_mxfp4(
            A, ptr(qw), ptr(C), ptr(ms), starts, counts,
            1, N, K, ws, 1, N // 256, 4)
        torch.cuda.synchronize()

        ref = w_ref[:, k_idx].t().contiguous()  # [64, N] bf16 exact
        # Canonicalize +-0.0 before the bit compare (see ladder docstring):
        # nibble 0x8 puts -0.0 (0x8000) in ref, the kernel's +0-seeded fp32
        # accumulator legitimately yields +0.0 (0x0000). Nonzero bits strict.
        Cc, refc = C.cpu(), ref.cpu()
        Cc = torch.where(Cc == 0, Cc.abs(), Cc)
        refc = torch.where(refc == 0, refc.abs(), refc)
        exact = torch.equal(Cc.view(torch.int16), refc.view(torch.int16))
        ok = ok and exact
        print(f"    T1[{tag}]: bit-exact={exact}")
    report("T1 in-kernel E2M1 decode bit-exact (incl. +-0.5 subnormal path)", ok)


def t2_m16_parity():
    ok = True
    for shape_tag, (K, N) in (("w13-branch", (K3_K, K3_N)), ("w2", (K3_N, K3_K))):
        p, s = rand_expert(K, N, seed=202)
        qw, ms = marlinize(p, s, K, N)
        w_ref = dense_expert_bf16(p, s)
        mod = mgm._load_module()
        for M in M_SWEEP:
            A = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
            C = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
            starts = torch.zeros(1, dtype=torch.int32, device=DEV)
            counts = torch.tensor([M], dtype=torch.int32, device=DEV)
            ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
            ptr = lambda t: torch.tensor([t.data_ptr()], dtype=torch.int64, device=DEV)
            mod.grouped_marlin_gemm_m16_mxfp4(
                A, ptr(qw), ptr(C), ptr(ms), starts, counts,
                1, N, K, ws, 1, N // 256, (M + 15) // 16)
            torch.cuda.synchronize()
            ref = A.float() @ w_ref.float().t()
            ok = gate(C, ref, f"T2[{shape_tag}] M={M}") and ok
    report("T2 M16 MXFP4 GEMM parity (K3 shapes, M sweep)", ok)


def _run_s1(x, gate_qw, gate_s, up_qw, up_s, N, K, entry):
    mod = mgm._load_module()
    t = x.shape[0]
    inter = torch.empty(t, N, dtype=torch.bfloat16, device=DEV)
    starts = torch.zeros(1, dtype=torch.int32, device=DEV)
    counts = torch.tensor([t], dtype=torch.int32, device=DEV)
    ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
    ptr = lambda tt: torch.tensor([tt.data_ptr()], dtype=torch.int64, device=DEV)
    getattr(mod, entry)(
        x, ptr(gate_qw), ptr(up_qw), ptr(inter), ptr(gate_s), ptr(up_s),
        starts, counts, 1, N, K, ws, N // 256, (t + 15) // 16)
    torch.cuda.synchronize()
    return inter


def t3_s1_situ_parity():
    K, N = K3_K, K3_N
    p1, s1 = rand_expert(K, N, seed=301)
    p3, s3 = rand_expert(K, N, seed=302, half_heavy=True)
    g_qw, g_s = marlinize(p1, s1, K, N)
    u_qw, u_s = marlinize(p3, s3, K, N)
    w1_ref = dense_expert_bf16(p1, s1)
    w3_ref = dense_expert_bf16(p3, s3)
    ok = True
    for M in M_SWEEP:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        out = _run_s1(x, g_qw, g_s, u_qw, u_s, N, K,
                      "grouped_marlin_gemm_m16_s1_mxfp4_situ")
        # eager reference: fp32 GEMMs -> bf16 (kernel stores bf16 pass results
        # in SMEM before the epilogue) -> fp32 SiTU
        g_ref = (x.float() @ w1_ref.float().t()).to(torch.bfloat16)
        u_ref = (x.float() @ w3_ref.float().t()).to(torch.bfloat16)
        ref = situ_ref_fp32(g_ref, u_ref)
        ok = gate(out, ref, f"T3 M={M}") and ok
    report("T3 fused S1 SiTU parity (M sweep)", ok)
    return (g_qw, g_s, u_qw, u_s, w1_ref, w3_ref)


def t4_grouped_zero_token():
    K, N = K3_K, K3_N
    E, mtp = 32, 64
    torch.manual_seed(404)
    # PRODUCTION fused layout: per-expert stacked [2, K//16, N*2] blob from
    # repack_mxfp4_w13_to_marlin_gs32; up pointers derived by BYTE ARITHMETIC
    # off the gate pointers, exactly as the model side must (storage
    # adjacency — a wide [K, 2N] slice would carry the wrong b_gl_stride).
    weights = []   # keep python refs alive for the pointer arrays
    for e in range(E):
        p1, s1 = rand_expert(K, N, seed=1000 + e)
        p3, s3 = rand_expert(K, N, seed=2000 + e)
        qw, sc = mwp.repack_mxfp4_w13_to_marlin_gs32(p1, s1, p3, s3, K, N,
                                                     emit_scale="e8m0")
        weights.append((qw.to(DEV), sc.to(DEV),
                        dense_expert_bf16(p1, s1), dense_expert_bf16(p3, s3)))
    qw_branch_bytes = weights[0][0][0].numel() * weights[0][0].element_size()
    s_branch_bytes = weights[0][1][0].numel() * weights[0][1].element_size()
    for qw, sc, _, _ in weights:  # pin the adjacency arithmetic per expert
        assert qw.data_ptr() + qw_branch_bytes == qw[1].data_ptr()
        assert sc.data_ptr() + s_branch_bytes == sc[1].data_ptr()
    counts_host = torch.zeros(E, dtype=torch.int32)
    for e in range(0, E, 3):           # >half of the experts stay empty
        counts_host[e] = int(torch.randint(1, mtp + 1, (1,)))
    A = torch.randn(E * mtp, K, dtype=torch.bfloat16, device=DEV)
    inter = torch.full((E * mtp, N), float("nan"), dtype=torch.bfloat16, device=DEV)
    starts = (torch.arange(E, dtype=torch.int32) * mtp).to(DEV)
    counts = counts_host.to(DEV)
    gate_B = torch.tensor([w[0].data_ptr() for w in weights], dtype=torch.int64, device=DEV)
    gate_S = torch.tensor([w[1].data_ptr() for w in weights], dtype=torch.int64, device=DEV)
    up_B = gate_B + qw_branch_bytes    # storage-adjacency derivation
    up_S = gate_S + s_branch_bytes
    C_ptrs = torch.tensor([inter.data_ptr() + e * mtp * N * 2 for e in range(E)],
                          dtype=torch.int64, device=DEV)
    ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)

    mgm.marlin_grouped_stage1_fused_mxfp4_situ(
        A, inter, counts, starts, gate_B, gate_S, up_B, up_S, C_ptrs,
        N, K, ws, max_m_tiles=(mtp + 15) // 16, mtp=mtp, num_experts=E,
        total_rows=int(counts_host.sum()))
    torch.cuda.synchronize()

    ok = True
    for e in range(E):
        c = int(counts_host[e])
        if c == 0:
            continue
        x = A[e * mtp:e * mtp + c]
        g_ref = (x.float() @ weights[e][2].float().t()).to(torch.bfloat16)
        u_ref = (x.float() @ weights[e][3].float().t()).to(torch.bfloat16)
        ref = situ_ref_fp32(g_ref, u_ref)
        ok = gate(inter[e * mtp:e * mtp + c], ref, f"T4 expert {e} c={c}") and ok

    # all-empty launch must be safe AND write nothing (C-EMPTYGRP):
    # NaN canary — every output byte must survive the launch untouched.
    inter.fill_(float("nan"))
    zero = torch.zeros(E, dtype=torch.int32, device=DEV)
    mgm.marlin_grouped_stage1_fused_mxfp4_situ(
        A, inter, zero, starts, gate_B, gate_S, up_B, up_S, C_ptrs,
        N, K, ws, max_m_tiles=(mtp + 15) // 16, mtp=mtp, num_experts=E,
        total_rows=0)
    torch.cuda.synchronize()
    all_nan = bool(torch.isnan(inter).all())
    print(f"    T4 all-empty launch wrote nothing: {all_nan}")
    report("T4 grouped fused S1 with zero-token experts (dense grid, no filtering)",
           ok and all_nan)


def t5_mutations(t3_tensors):
    g_qw, g_s, u_qw, u_s, w1_ref, w3_ref = t3_tensors
    K, N = K3_K, K3_N
    M = 64
    torch.manual_seed(505)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    g_ref = (x.float() @ w1_ref.float().t()).to(torch.bfloat16)
    u_ref = (x.float() @ w3_ref.float().t()).to(torch.bfloat16)
    ref = situ_ref_fp32(g_ref, u_ref)

    caught = 0
    # m1: INT4 entry consuming E2M1 codes — finite plausible garbage, must FAIL
    out = _run_s1(x, g_qw, g_s, u_qw, u_s, N, K, "grouped_marlin_gemm_m16_s1")
    if not gate(out, ref, "T5.m1 int4-entry-on-e2m1"):
        caught += 1
    # m2: gate/up swap — silent at kernel level, must FAIL vs the ordered ref
    out = _run_s1(x, u_qw, u_s, g_qw, g_s, N, K,
                  "grouped_marlin_gemm_m16_s1_mxfp4_situ")
    if not gate(out, ref, "T5.m2 gate/up-swap"):
        caught += 1
    # m3: off-by-one scale group
    g_s_mut = torch.roll(g_s, 1, dims=0).contiguous()
    out = _run_s1(x, g_qw, g_s_mut, u_qw, u_s, N, K,
                  "grouped_marlin_gemm_m16_s1_mxfp4_situ")
    if not gate(out, ref, "T5.m3 off-by-one-scale"):
        caught += 1
    # m4: SiLU-instead-of-SiTU reference vs the SiTU kernel
    out = _run_s1(x, g_qw, g_s, u_qw, u_s, N, K,
                  "grouped_marlin_gemm_m16_s1_mxfp4_situ")
    silu_ref = silu_ref_fp32(g_ref, u_ref)
    if not gate(out, silu_ref, "T5.m4 silu-vs-situ"):
        caught += 1

    report("T5 mutation arms caught", caught == 4, f"catch count {caught}/4")


def _int4_expert(K, N, seed):
    """Random K2.5-style INT4 expert -> (marlin qw, marlin scales, dense ref)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (N, K), generator=g, dtype=torch.int32)
    raw = torch.zeros(N, K // 8, dtype=torch.int32)
    for i in range(8):
        raw |= (q[:, i::8] & 0xF) << (i * 4)
    scales = (torch.rand(N, K // 32, generator=g) * 0.02 + 0.001).to(torch.bfloat16)
    qw, ms = mwp.repack_int4_to_marlin_gs32(raw, scales, K, N)
    w_ref = ((q - 8).float().view(N, K // 32, 32)
             * scales.float().unsqueeze(-1)).view(N, K).to(torch.bfloat16).to(DEV)
    return qw.to(DEV), ms.to(DEV), w_ref


def t6_int4_regression():
    """K2.5 INT4 M16 through the now-templated <U4B8> kernel must still pass."""
    K, N = 1024, 1024
    qw, ms, w_ref = _int4_expert(K, N, seed=606)

    mod = mgm._load_module()
    ok = True
    for M in (1, 64, 512):
        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        C = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
        starts = torch.zeros(1, dtype=torch.int32, device=DEV)
        counts = torch.tensor([M], dtype=torch.int32, device=DEV)
        ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
        ptr = lambda t: torch.tensor([t.data_ptr()], dtype=torch.int64, device=DEV)
        mod.grouped_marlin_gemm_m16(
            A, ptr(qw), ptr(C), ptr(ms), starts, counts,
            1, N, K, ws, 1, N // 256, (M + 15) // 16)
        torch.cuda.synchronize()
        ref = A.float() @ w_ref.float().t()
        ok = gate(C, ref, f"T6 M={M}") and ok
    report("T6 INT4 <U4B8> M16 regression (templating did not disturb production)", ok)


def t6b_int4_s1_regression():
    """K2.5 fused S1 <U4B8, SILU> — the decode DEFAULT — must still pass its
    own parity. This kernel got the heavier template surgery (epilogue
    expression swapped to act_gate_mul<ACT>); T5.m1 alone cannot certify it
    (a must-diverge arm passes even if SiLU output is subtly wrong)."""
    K, N = 1024, 1024
    g_qw, g_s, g_w = _int4_expert(K, N, seed=616)
    u_qw, u_s, u_w = _int4_expert(K, N, seed=617)
    ok = True
    for M in (1, 64, 512):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        out = _run_s1(x, g_qw, g_s, u_qw, u_s, N, K, "grouped_marlin_gemm_m16_s1")
        g_ref = (x.float() @ g_w.float().t()).to(torch.bfloat16)
        u_ref = (x.float() @ u_w.float().t()).to(torch.bfloat16)
        ok = gate(out, silu_ref_fp32(g_ref, u_ref), f"T6b M={M}") and ok
    report("T6b INT4 fused S1 <U4B8,SILU> regression (K2.5 decode default)", ok)


def t7_wrapper_hardfail_negatives():
    """Every hard-fail seam must RAISE: python wrapper checks (ValueError) and
    the raw-pybind TORCH_CHECK seam in the C++ entries (RuntimeError). Uses
    zero counts + real allocations so a REGRESSED (non-raising) arm degrades
    to an empty launch and a clean FAIL, never a wild pointer deref."""
    K, N = K3_K, K3_N
    E, mtp = 4, 16
    t = 8
    x = torch.randn(t, K, dtype=torch.bfloat16, device=DEV)
    p, s = rand_expert(K, N, seed=701)
    qw, ms = marlinize(p, s, K, N)
    qw_bf16, ms_bf16 = mwp.repack_mxfp4_to_marlin_gs32(
        p, s, K, N, emit_scale="bf16")
    qw_bf16, ms_bf16 = qw_bf16.to(DEV), ms_bf16.to(DEV)
    pd, sd = rand_expert(N, K, seed=702)
    dqw, dms = marlinize(pd, sd, N, K)

    A = torch.randn(E * mtp, K, dtype=torch.bfloat16, device=DEV)
    inter = torch.empty(E * mtp, N, dtype=torch.bfloat16, device=DEV)
    counts = torch.zeros(E, dtype=torch.int32, device=DEV)       # empty: safe
    starts = (torch.arange(E, dtype=torch.int32) * mtp).to(DEV)
    bp = torch.tensor([qw.data_ptr()] * E, dtype=torch.int64, device=DEV)
    sp = torch.tensor([ms.data_ptr()] * E, dtype=torch.int64, device=DEV)
    cp = torch.tensor([inter.data_ptr() + e * mtp * N * 2 for e in range(E)],
                      dtype=torch.int64, device=DEV)
    ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
    mod = mgm._load_module()

    def fused(**kw):
        args = dict(dispatched_x_3d=A, intermediate_3d=inter,
                    expert_counts=counts, expert_starts=starts,
                    gate_B_ptrs=bp, gate_scales_ptrs=sp,
                    up_B_ptrs=bp, up_scales_ptrs=sp, C_ptrs=cp,
                    N=N, K=K, workspace=ws, max_m_tiles=1, mtp=mtp,
                    num_experts=E, total_rows=0)
        args.update(kw)
        mgm.marlin_grouped_stage1_fused_mxfp4_situ(**args)

    arms = [
        ("L2 bf16-scale-at-native-e8m0-kernel", lambda: mgm.single_expert_marlin_mxfp4_decode(
            x, qw_bf16, ms_bf16, qw, ms, dqw, dms, N=K3_N, K=K3_K)),
        ("L2 wrong-marlin-shape", lambda: mgm.single_expert_marlin_mxfp4_decode(
            x, dqw, dms, qw, ms, dqw, dms, N=K3_N, K=K3_K)),
        ("L3 ptr-array-int32", lambda: fused(gate_B_ptrs=bp.to(torch.int32))),
        ("L3 counts-wrong-length", lambda: fused(expert_counts=counts[:-1])),
        # intermediate resized to match so the ACT check passes and L4 itself
        # (prob_n % 256) is the check that fires
        ("L4 N-not-256-multiple", lambda: fused(
            N=N - 64, intermediate_3d=torch.empty(
                E * mtp, N - 64, dtype=torch.bfloat16, device=DEV))),
        ("L5 m-tile-bound-too-small", lambda: fused(mtp=64, total_rows=64,
                                                    max_m_tiles=1)),
        ("ACT x-fp16", lambda: fused(dispatched_x_3d=A.half())),
        ("ACT single-expert-x-fp16", lambda: mgm.single_expert_marlin_mxfp4_decode(
            x.half(), qw, ms, qw, ms, dqw, dms, N=K3_N, K=K3_K)),
        # raw-pybind seam: TORCH_CHECKs inside the C++ entries must hard-fail
        # even when the python wrappers are bypassed (K2.5 integration pattern)
        ("RAW n_tiles-mismatch", lambda: mod.grouped_marlin_gemm_m16_mxfp4(
            A, bp, cp, sp, starts, counts, E, N, K, ws, E, N // 256 - 1, 1)),
        ("RAW ptr-array-int32", lambda: mod.grouped_marlin_gemm_m16_s1_mxfp4_situ(
            A, bp.to(torch.int32), bp, cp, sp, sp, starts, counts,
            E, N, K, ws, N // 256, 1)),
    ]
    caught = 0
    for name, fn in arms:
        try:
            fn()
            torch.cuda.synchronize()
            print(f"    T7[{name}]: NOT caught")
        except (ValueError, RuntimeError) as e:
            caught += 1
            print(f"    T7[{name}]: caught ({str(e).splitlines()[0][:64]})")
    report("T7 hard-fail negative arms", caught == len(arms),
           f"catch count {caught}/{len(arms)}")


# ---------------------------------------------------------------------------
# Version 2 kernels
# ---------------------------------------------------------------------------

V2_VARIANTS = sorted(mgm.MXFP4_V2_VARIANTS)
V2_M_SWEEP = [1, 4, 9, 16, 17, 24, 32, 33, 63, 65]
V2_BIG_MAX_M_TILES = 64   # production-like worst-case grid argument (1024 rows)
V2_MTP = 72               # per-expert row stride; rows >= count are NaN canaries
NAN = float("nan")


def _ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=DEV)


def _row_ptrs(buf, E, stride_rows):
    row_bytes = buf.shape[-1] * buf.element_size()
    return torch.tensor([buf.data_ptr() + e * stride_rows * row_bytes
                         for e in range(E)], dtype=torch.int64, device=DEV)


def _expert_rows(buf, counts_host, stride):
    return torch.cat([buf[e * stride:e * stride + c]
                      for e, c in enumerate(counts_host) if c])


def _canary_ok(buf, counts_host, stride):
    """Every row at or beyond an expert's count must still hold the NaN fill."""
    return all(bool(torch.isnan(buf[e * stride + c:(e + 1) * stride]).all())
               for e, c in enumerate(counts_host))


def _canonical_bits(x):
    """bf16 bits with -0.0 folded to +0.0 (see the T1 docstring)."""
    x = x.cpu()
    return torch.where(x == 0, x.abs(), x).view(torch.int16)


def _bf16_ulp(x):
    """One bf16 ulp of |x| (fp32 tensor); zeros get the smallest normal ulp."""
    _, e = torch.frexp(x.abs().clamp_min(2.0 ** -126))
    return torch.exp2((e - 8).float())


def _agree(v1, v2):
    """v1-vs-v2 agreement on rows both produce. Returns (ok, summary); ok iff
    finite and the fraction differing by > 1 bf16 ulp of |v1| is < 1e-3."""
    a, b = v1.float(), v2.float()
    d = (a - b).abs()
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    beyond = float((d > _bf16_ulp(a)).float().mean())
    same = float((_canonical_bits(v1) == _canonical_bits(v2)).float().mean())
    ok = finite and beyond < 1e-3
    return ok, (f"v1: max_abs={float(d.max()):.3e} >1ulp_frac={beyond:.2e} "
                f"bit_identical={same:.6f}")


def _gate_rows(buf, refs, counts_host, stride, tag, verbose=True):
    """Gate all non-empty experts' rows at once; on failure localise it to
    the failing experts. refs[e] is expert e's reference (None if empty)."""
    ok = gate(_expert_rows(buf, counts_host, stride),
              torch.cat([r for r in refs if r is not None]), tag, verbose=verbose)
    if not ok:
        for e, (c, r) in enumerate(zip(counts_host, refs)):
            if c:
                gate(buf[e * stride:e * stride + c], r,
                     f"{tag} expert {e} c={c}", verbose=False)
    return ok


def _w13_experts(E, K, N, seed0):
    """PRODUCTION fused S1 layout (see T4): per-expert stacked [2, K//16, N*2]
    blob, up pointers derived by byte arithmetic off the gate pointers.
    Returns (weights, gate_B, gate_S, up_B, up_S); weights[e] =
    (qw, scales, w1_ref, w3_ref)."""
    weights = []
    for e in range(E):
        p1, s1 = rand_expert(K, N, seed=seed0 + e)
        p3, s3 = rand_expert(K, N, seed=seed0 + 500 + e, half_heavy=bool(e % 2))
        qw, sc = mwp.repack_mxfp4_w13_to_marlin_gs32(p1, s1, p3, s3, K, N,
                                                     emit_scale="e8m0")
        qw, sc = qw.to(DEV), sc.to(DEV)
        mgm.check_mxfp4_scales_v2_compatible(sc)
        weights.append((qw, sc, dense_expert_bf16(p1, s1), dense_expert_bf16(p3, s3)))
    qw_branch_bytes = weights[0][0][0].numel() * weights[0][0].element_size()
    s_branch_bytes = weights[0][1][0].numel() * weights[0][1].element_size()
    for qw, sc, _, _ in weights:
        assert qw.data_ptr() + qw_branch_bytes == qw[1].data_ptr()
        assert sc.data_ptr() + s_branch_bytes == sc[1].data_ptr()
    gate_B = _ptrs([w[0] for w in weights])
    gate_S = _ptrs([w[1] for w in weights])
    return weights, gate_B, gate_S, gate_B + qw_branch_bytes, gate_S + s_branch_bytes


def _s1_ref(x, w1_ref, w3_ref):
    g_ref = (x.float() @ w1_ref.float().t()).to(torch.bfloat16)
    u_ref = (x.float() @ w3_ref.float().t()).to(torch.bfloat16)
    return situ_ref_fp32(g_ref, u_ref)


def _v2_case_line(tag, gate_ok, canary_ok, grids_identical, agree):
    agree_ok, agree_str = agree
    print(f"    {tag}: gate={'ok' if gate_ok else 'FAIL'} "
          f"canary={'ok' if canary_ok else 'FAIL'} "
          f"grids={'identical' if grids_identical else 'DIFFER'} "
          f"{'agree' if agree_ok else 'DISAGREE'} {agree_str}")
    return gate_ok and canary_ok and grids_identical and agree_ok


def print_v2_launch_configs(variants):
    mod = mgm._load_module()
    for v in variants:
        tm, tn = mgm.MXFP4_V2_VARIANTS[v]
        s1_stages, s1_smem = mod.marlin_mxfp4_v2_launch_config(v, True)
        g_stages, g_smem = mod.marlin_mxfp4_v2_launch_config(v, False)
        print(f"    v2 variant {v} (TM={tm}, TN={tn}): S1 stages={s1_stages} "
              f"smem={s1_smem} B; grouped stages={g_stages} smem={g_smem} B")


def t1v2_decode_bitexact(variants):
    K, N = 512, 512
    torch.manual_seed(111)
    mod = mgm._load_module()
    ok = True
    for tag, heavy, lo, hi in (("uniform", False, 100, mgm.MXFP4_V2_MAX_SCALE_BYTE),
                               ("half-heavy", True, SCALE_LO, SCALE_HI)):
        p, s = rand_expert(K, N, seed=111, half_heavy=heavy,
                           scale_lo=lo, scale_hi=hi)
        qw, ms = marlinize(p, s, K, N)
        mgm.check_mxfp4_scales_v2_compatible(ms)
        w_ref = dense_expert_bf16(p, s)
        k_idx = torch.randperm(K)[:64]
        A = torch.zeros(64, K, dtype=torch.bfloat16, device=DEV)
        A[torch.arange(64), k_idx] = 1.0
        ref_bits = _canonical_bits(w_ref[:, k_idx].t().contiguous())
        starts = torch.zeros(1, dtype=torch.int32, device=DEV)
        counts = torch.tensor([64], dtype=torch.int32, device=DEV)
        ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
        for v in variants:
            C = torch.full((64, N), NAN, dtype=torch.bfloat16, device=DEV)
            mod.grouped_marlin_gemm_m16_mxfp4_v2(
                A, _ptrs([qw]), _ptrs([C]), _ptrs([ms]), starts, counts,
                1, N, K, ws, 1, N // 256, 4, v)
            torch.cuda.synchronize()
            exact = torch.equal(_canonical_bits(C), ref_bits)
            ok = ok and exact
            print(f"    T1v2[{tag}] v{v}: bit-exact={exact}")
    report("T1v2 in-kernel E2M1 decode bit-exact, folded scales", ok)


def t2v2_m16_parity(variants):
    ok = True
    E = 2
    for shape_tag, (K, N) in (("w13-branch", (K3_K, K3_N)), ("w2", (K3_N, K3_K))):
        raw = [rand_expert(K, N, seed=222 + 7 * e) for e in range(E)]
        mar = [marlinize(p, s, K, N) for p, s in raw]
        for _, ms in mar:
            mgm.check_mxfp4_scales_v2_compatible(ms)
        w_ref = [dense_expert_bf16(p, s) for p, s in raw]
        B, S = _ptrs([q for q, _ in mar]), _ptrs([s for _, s in mar])
        starts = (torch.arange(E, dtype=torch.int32) * V2_MTP).to(DEV)
        C = torch.empty(E * V2_MTP, N, dtype=torch.bfloat16, device=DEV)
        Cp = _row_ptrs(C, E, V2_MTP)
        ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
        for M in V2_M_SWEEP:
            torch.manual_seed(2200 + M)
            A = torch.randn(E * V2_MTP, K, dtype=torch.bfloat16, device=DEV)
            counts_host = [M] * E
            counts = torch.tensor(counts_host, dtype=torch.int32, device=DEV)
            refs = [A[e * V2_MTP:e * V2_MTP + M].float() @ w_ref[e].float().t()
                    for e in range(E)]
            tight = (M + 15) // 16
            C.fill_(NAN)
            mgm.marlin_grouped_m16_mxfp4(
                A, B, Cp, S, starts, counts, E, N, K, ws, E, N // 256, tight)
            torch.cuda.synchronize()
            v1_rows = _expert_rows(C, counts_host, V2_MTP).clone()
            for v in variants:
                outs, canary = [], True
                for grid in (tight, V2_BIG_MAX_M_TILES):
                    C.fill_(NAN)
                    mgm.marlin_grouped_m16_mxfp4_v2(
                        A, B, Cp, S, starts, counts, E, N, K, ws, E, N // 256,
                        grid, variant=v)
                    torch.cuda.synchronize()
                    canary = _canary_ok(C, counts_host, V2_MTP) and canary
                    outs.append(C.clone())
                tag = f"T2v2[{shape_tag}] v{v} M={M}"
                ok = _v2_case_line(
                    tag,
                    _gate_rows(outs[0], refs, counts_host, V2_MTP, tag, verbose=False),
                    canary,
                    torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)),
                    _agree(v1_rows, _expert_rows(outs[0], counts_host, V2_MTP))) and ok
    report("T2v2 M16 MXFP4 v2 parity (K3 shapes, M sweep, both grids, v1 agreement)", ok)


def t3v2_s1_situ_parity(variants):
    K, N = K3_K, K3_N
    E = 2
    weights, gB, gS, uB, uS = _w13_experts(E, K, N, seed0=3100)
    starts = (torch.arange(E, dtype=torch.int32) * V2_MTP).to(DEV)
    inter = torch.empty(E * V2_MTP, N, dtype=torch.bfloat16, device=DEV)
    Cp = _row_ptrs(inter, E, V2_MTP)
    ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
    ok = True
    for M in V2_M_SWEEP:
        torch.manual_seed(3300 + M)
        A = torch.randn(E * V2_MTP, K, dtype=torch.bfloat16, device=DEV)
        counts_host = [M] * E
        counts = torch.tensor(counts_host, dtype=torch.int32, device=DEV)
        refs = [_s1_ref(A[e * V2_MTP:e * V2_MTP + M], weights[e][2], weights[e][3])
                for e in range(E)]
        tight = (M + 15) // 16
        common = dict(N=N, K=K, workspace=ws, mtp=M, num_experts=E,
                      total_rows=E * M)
        inter.fill_(NAN)
        mgm.marlin_grouped_stage1_fused_mxfp4_situ(
            A, inter, counts, starts, gB, gS, uB, uS, Cp,
            max_m_tiles=tight, **common)
        torch.cuda.synchronize()
        v1_rows = _expert_rows(inter, counts_host, V2_MTP).clone()
        for v in variants:
            outs, canary = [], True
            for grid in (tight, V2_BIG_MAX_M_TILES):
                inter.fill_(NAN)
                mgm.marlin_grouped_stage1_fused_mxfp4_situ_v2(
                    A, inter, counts, starts, gB, gS, uB, uS, Cp,
                    max_m_tiles=grid, variant=v, **common)
                torch.cuda.synchronize()
                canary = _canary_ok(inter, counts_host, V2_MTP) and canary
                outs.append(inter.clone())
            tag = f"T3v2 v{v} M={M}"
            ok = _v2_case_line(
                tag,
                _gate_rows(outs[0], refs, counts_host, V2_MTP, tag, verbose=False),
                canary,
                torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)),
                _agree(v1_rows, _expert_rows(outs[0], counts_host, V2_MTP))) and ok
    report("T3v2 fused S1 SiTU v2 parity (M sweep, both grids, v1 agreement)", ok)


def t4v2_grouped_skewed(variants):
    K, N = K3_K, K3_N
    E, mtp = 28, V2_MTP
    g = torch.Generator().manual_seed(4040)
    fixed = [0, 1, 9, 16, 17, 33, 40, 65]
    rest = torch.randint(0, 66, (E - len(fixed),), generator=g)
    rest[::4] = 0                                   # more empty experts
    counts_all = fixed + [int(c) for c in rest]
    counts_host = [counts_all[i] for i in torch.randperm(E, generator=g).tolist()]
    total = sum(counts_host)
    print(f"    T4v2 counts: {counts_host}")

    weights, gB, gS, uB, uS = _w13_experts(E, K, N, seed0=4100)
    down = []                                       # S3: prob_n=K, prob_k=N
    for e in range(E):
        p, s = rand_expert(N, K, seed=4300 + e)
        q, sc = marlinize(p, s, N, K)
        mgm.check_mxfp4_scales_v2_compatible(sc)
        down.append((q, sc, dense_expert_bf16(p, s)))
    dB, dS = _ptrs([d[0] for d in down]), _ptrs([d[1] for d in down])

    torch.manual_seed(4444)
    A = torch.randn(E * mtp, K, dtype=torch.bfloat16, device=DEV)
    H = torch.randn(E * mtp, N, dtype=torch.bfloat16, device=DEV)  # S3 input
    inter = torch.empty(E * mtp, N, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(E * mtp, K, dtype=torch.bfloat16, device=DEV)
    starts = (torch.arange(E, dtype=torch.int32) * mtp).to(DEV)
    counts = torch.tensor(counts_host, dtype=torch.int32, device=DEV)
    zero = torch.zeros(E, dtype=torch.int32, device=DEV)
    Ip, Op = _row_ptrs(inter, E, mtp), _row_ptrs(out, E, mtp)
    ws1 = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
    ws3 = torch.zeros(K // 256 + 17, dtype=torch.int32, device=DEV)
    rows = lambda x, e: x[e * mtp:e * mtp + counts_host[e]]
    s1_refs = [_s1_ref(rows(A, e), weights[e][2], weights[e][3]) if counts_host[e]
               else None for e in range(E)]
    s3_refs = [rows(H, e).float() @ down[e][2].float().t() if counts_host[e]
               else None for e in range(E)]
    mmt = (mtp + 15) // 16

    def s1(cnt, total_rows, v):
        inter.fill_(NAN)
        mgm.marlin_grouped_stage1_fused_mxfp4_situ_v2(
            A, inter, cnt, starts, gB, gS, uB, uS, Ip, N, K, ws1,
            max_m_tiles=mmt, mtp=mtp, num_experts=E, total_rows=total_rows,
            variant=v)
        torch.cuda.synchronize()

    def s3(cnt, v):
        out.fill_(NAN)
        mgm.marlin_grouped_m16_mxfp4_v2(
            H, dB, Op, dS, starts, cnt, E, K, N, ws3, E, K // 256, mmt, variant=v)
        torch.cuda.synchronize()

    ok = True
    for v in variants:
        s1(counts, total, v)
        g1 = _gate_rows(inter, s1_refs, counts_host, mtp, f"T4v2 S1 v{v}")
        c1 = _canary_ok(inter, counts_host, mtp)
        s3(counts, v)
        g3 = _gate_rows(out, s3_refs, counts_host, mtp, f"T4v2 S3 v{v}")
        c3 = _canary_ok(out, counts_host, mtp)
        s1(zero, 0, v)
        e1 = bool(torch.isnan(inter).all())
        s3(zero, v)
        e3 = bool(torch.isnan(out).all())
        print(f"    T4v2 v{v}: canary S1={c1} S3={c3}; all-empty wrote nothing "
              f"S1={e1} S3={e3}")
        ok = ok and g1 and c1 and g3 and c3 and e1 and e3
    report("T4v2 grouped v2 with skewed and zero-token experts (E=28)", ok)


def t5v2_mutations(variants):
    K, N = K3_K, K3_N
    M = 24                       # two m-blocks for TM=2, partial second block
    weights, gB, gS, uB, uS = _w13_experts(1, K, N, seed0=5100)
    qw, sc, w1_ref, w3_ref = weights[0]
    torch.manual_seed(5050)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    ref = _s1_ref(x, w1_ref, w3_ref)
    inter = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    starts = torch.zeros(1, dtype=torch.int32, device=DEV)
    counts = torch.tensor([M], dtype=torch.int32, device=DEV)
    Cp = _ptrs([inter])
    ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
    g_s_rolled = torch.roll(sc[0], 1, dims=0).contiguous()   # off-by-one group
    gS_rolled = _ptrs([g_s_rolled])

    def run(gb, gs, ub, us, v):
        inter.fill_(NAN)
        mgm.marlin_grouped_stage1_fused_mxfp4_situ_v2(
            x, inter, counts, starts, gb, gs, ub, us, Cp, N, K, ws,
            max_m_tiles=(M + 15) // 16, mtp=M, num_experts=1, total_rows=M,
            variant=v)
        torch.cuda.synchronize()
        return inter

    caught, arms, control_ok = 0, 0, True
    for v in variants:
        control_ok = gate(run(gB, gS, uB, uS, v), ref, f"T5v2.control v{v}") and control_ok
        arms += 2
        if not gate(run(uB, uS, gB, gS, v), ref, f"T5v2.m2 v{v} gate/up-swap"):
            caught += 1
        if not gate(run(gB, gS_rolled, uB, uS, v), ref, f"T5v2.m3 v{v} off-by-one-scale"):
            caught += 1

    def must_raise(name, fn, exc):
        nonlocal caught, arms
        arms += 1
        try:
            fn()
            torch.cuda.synchronize()
            print(f"    T5v2[{name}]: NOT caught")
        except exc as e:
            caught += 1
            print(f"    T5v2[{name}]: caught ({str(e).splitlines()[0][:64]})")

    bad = sc.clone()
    bad.view(-1)[123] = mgm.MXFP4_V2_MAX_SCALE_BYTE + 1
    must_raise("m5 scale-byte>128", lambda: mgm.check_mxfp4_scales_v2_compatible(bad),
               ValueError)
    edge = sc.clone()
    edge.view(-1)[123] = mgm.MXFP4_V2_MAX_SCALE_BYTE
    try:
        mgm.check_mxfp4_scales_v2_compatible(edge)
        edge_ok = True
    except ValueError:
        edge_ok = False
    print(f"    T5v2 scale byte 128 accepted: {edge_ok}")
    must_raise("m6 wrapper variant=4",
               lambda: run(gB, gS, uB, uS, 4), ValueError)
    mod = mgm._load_module()
    must_raise("m6 raw variant=4", lambda: mod.grouped_marlin_gemm_m16_s1_mxfp4_situ_v2(
        x, gB, uB, Cp, gS, uS, starts, counts, 1, N, K, ws, N // 256, 2, 4),
        RuntimeError)
    report("T5v2 mutation arms caught (controls pass)",
           caught == arms and control_ok and edge_ok,
           f"catch count {caught}/{arms}")


# ---------------------------------------------------------------------------
# Version 3 ladder
# ---------------------------------------------------------------------------

V3_VARIANTS = sorted(mgm.MXFP4_V3_VARIANTS)
V3_M_SWEEP = [1, 4, 8, 9, 16, 17, 24, 32, 33, 40, 64, 65]
V3_SMEM_LIMIT = 113 * 1024   # two CTAs per SM


def print_v3_launch_configs(variants):
    mod = mgm._load_module()
    for v in variants:
        jw, ncap = mgm.MXFP4_V3_VARIANTS[v]
        s1 = mod.marlin_mxfp4_v3_launch_config(v, True)
        gr = mod.marlin_mxfp4_v3_launch_config(v, False)
        print(f"    v3 variant {v} (JW={jw}, NCAP={ncap}): S1 stages={s1[0]} "
              f"smem={s1[1]} B cols={s1[2]}; grouped stages={gr[0]} "
              f"smem={gr[1]} B cols={gr[2]}")


def t1v3_decode_bitexact(variants):
    K, N = 512, 512
    mod = mgm._load_module()
    ok = True
    for tag, heavy, lo, hi in (("uniform", False, 100, mgm.MXFP4_V2_MAX_SCALE_BYTE),
                               ("half-heavy", True, SCALE_LO, SCALE_HI)):
        p, s = rand_expert(K, N, seed=131, half_heavy=heavy,
                           scale_lo=lo, scale_hi=hi)
        qw, ms = marlinize(p, s, K, N)
        mgm.check_mxfp4_scales_v2_compatible(ms)
        w_ref = dense_expert_bf16(p, s)
        starts = torch.zeros(1, dtype=torch.int32, device=DEV)
        ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
        for rows in (64, 72):
            g = torch.Generator().manual_seed(1310 + rows)
            k_idx = torch.randperm(K, generator=g)[:rows]
            A = torch.zeros(rows, K, dtype=torch.bfloat16, device=DEV)
            A[torch.arange(rows), k_idx] = 1.0
            ref_bits = _canonical_bits(w_ref[:, k_idx].t().contiguous())
            counts = torch.tensor([rows], dtype=torch.int32, device=DEV)
            for v in variants:
                C = torch.full((rows, N), NAN, dtype=torch.bfloat16, device=DEV)
                mod.grouped_marlin_gemm_m16_mxfp4_v3(
                    A, _ptrs([qw]), _ptrs([C]), _ptrs([ms]), starts, counts,
                    1, N, K, ws, 1, N // 256, (rows + 15) // 16, v, None)
                torch.cuda.synchronize()
                exact = torch.equal(_canonical_bits(C), ref_bits)
                ok = ok and exact
                print(f"    T1v3[{tag}] v{v} rows={rows}: bit-exact={exact}")
    report("T1v3 in-kernel E2M1 decode bit-exact, swap-AB wgmma", ok)


def t2v3_m16_parity(variants):
    ok = True
    E = 2
    for shape_tag, (K, N) in (("w13-branch", (K3_K, K3_N)), ("w2", (K3_N, K3_K))):
        raw = [rand_expert(K, N, seed=232 + 7 * e) for e in range(E)]
        mar = [marlinize(p, s, K, N) for p, s in raw]
        for _, ms in mar:
            mgm.check_mxfp4_scales_v2_compatible(ms)
        w_ref = [dense_expert_bf16(p, s) for p, s in raw]
        B, S = _ptrs([q for q, _ in mar]), _ptrs([s for _, s in mar])
        starts = (torch.arange(E, dtype=torch.int32) * V2_MTP).to(DEV)
        C = torch.empty(E * V2_MTP, N, dtype=torch.bfloat16, device=DEV)
        Cp = _row_ptrs(C, E, V2_MTP)
        ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
        for M in V3_M_SWEEP:
            torch.manual_seed(2300 + M)
            A = torch.randn(E * V2_MTP, K, dtype=torch.bfloat16, device=DEV)
            counts_host = [M] * E
            counts = torch.tensor(counts_host, dtype=torch.int32, device=DEV)
            refs = [A[e * V2_MTP:e * V2_MTP + M].float() @ w_ref[e].float().t()
                    for e in range(E)]
            tight = (M + 15) // 16
            C.fill_(NAN)
            mgm.marlin_grouped_m16_mxfp4(
                A, B, Cp, S, starts, counts, E, N, K, ws, E, N // 256, tight)
            torch.cuda.synchronize()
            v1_rows = _expert_rows(C, counts_host, V2_MTP).clone()
            for v in variants:
                outs, canary = [], True
                for grid in (tight, V2_BIG_MAX_M_TILES):
                    C.fill_(NAN)
                    mgm.marlin_grouped_m16_mxfp4_v3(
                        A, B, Cp, S, starts, counts, E, N, K, ws, E, N // 256,
                        grid, variant=v)
                    torch.cuda.synchronize()
                    canary = _canary_ok(C, counts_host, V2_MTP) and canary
                    outs.append(C.clone())
                tag = f"T2v3[{shape_tag}] v{v} M={M}"
                ok = _v2_case_line(
                    tag,
                    _gate_rows(outs[0], refs, counts_host, V2_MTP, tag, verbose=False),
                    canary,
                    torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)),
                    _agree(v1_rows, _expert_rows(outs[0], counts_host, V2_MTP))) and ok
    report("T2v3 M16 MXFP4 v3 parity (K3 shapes, M sweep, both grids, v1 agreement)", ok)


def t3v3_s1_situ_parity(variants):
    K, N = K3_K, K3_N
    E = 2
    weights, gB, gS, uB, uS = _w13_experts(E, K, N, seed0=3500)
    starts = (torch.arange(E, dtype=torch.int32) * V2_MTP).to(DEV)
    inter = torch.empty(E * V2_MTP, N, dtype=torch.bfloat16, device=DEV)
    Cp = _row_ptrs(inter, E, V2_MTP)
    ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
    ok = True
    for M in V3_M_SWEEP:
        torch.manual_seed(3700 + M)
        A = torch.randn(E * V2_MTP, K, dtype=torch.bfloat16, device=DEV)
        counts_host = [M] * E
        counts = torch.tensor(counts_host, dtype=torch.int32, device=DEV)
        refs = [_s1_ref(A[e * V2_MTP:e * V2_MTP + M], weights[e][2], weights[e][3])
                for e in range(E)]
        tight = (M + 15) // 16
        common = dict(N=N, K=K, workspace=ws, mtp=M, num_experts=E,
                      total_rows=E * M)
        inter.fill_(NAN)
        mgm.marlin_grouped_stage1_fused_mxfp4_situ(
            A, inter, counts, starts, gB, gS, uB, uS, Cp,
            max_m_tiles=tight, **common)
        torch.cuda.synchronize()
        v1_rows = _expert_rows(inter, counts_host, V2_MTP).clone()
        for v in variants:
            outs, canary = [], True
            for grid in (tight, V2_BIG_MAX_M_TILES):
                inter.fill_(NAN)
                mgm.marlin_grouped_stage1_fused_mxfp4_situ_v3(
                    A, inter, counts, starts, gB, gS, uB, uS, Cp,
                    max_m_tiles=grid, variant=v, **common)
                torch.cuda.synchronize()
                canary = _canary_ok(inter, counts_host, V2_MTP) and canary
                outs.append(inter.clone())
            tag = f"T3v3 v{v} M={M}"
            ok = _v2_case_line(
                tag,
                _gate_rows(outs[0], refs, counts_host, V2_MTP, tag, verbose=False),
                canary,
                torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)),
                _agree(v1_rows, _expert_rows(outs[0], counts_host, V2_MTP))) and ok
    report("T3v3 fused S1 SiTU v3 parity (M sweep, both grids, v1 agreement)", ok)


def _v3_skewed_problem(E, seed):
    """Skewed grouped problem shared by T4v3 and T6v3: fused-S1 w13 experts,
    S3 down experts, inputs, outputs, row pointers and per-expert references."""
    K, N, mtp = K3_K, K3_N, V2_MTP
    g = torch.Generator().manual_seed(seed)
    fixed = [0, 1, 8, 9, 16, 17, 33, 40, 64, 65, 70][:E]
    rest = torch.randint(0, mtp + 1, (max(E - len(fixed), 0),), generator=g)
    rest[::4] = 0
    counts_all = fixed + [int(c) for c in rest]
    counts_host = [counts_all[i] for i in torch.randperm(E, generator=g).tolist()]

    weights, gB, gS, uB, uS = _w13_experts(E, K, N, seed0=seed + 100)
    down = []
    for e in range(E):
        p, s = rand_expert(N, K, seed=seed + 300 + e)
        q, sc = marlinize(p, s, N, K)
        mgm.check_mxfp4_scales_v2_compatible(sc)
        down.append((q, sc, dense_expert_bf16(p, s)))
    torch.manual_seed(seed + 7)
    A = torch.randn(E * mtp, K, dtype=torch.bfloat16, device=DEV)
    H = torch.randn(E * mtp, N, dtype=torch.bfloat16, device=DEV)
    inter = torch.empty(E * mtp, N, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(E * mtp, K, dtype=torch.bfloat16, device=DEV)
    rows = lambda x, e: x[e * mtp:e * mtp + counts_host[e]]
    return dict(
        K=K, N=N, mtp=mtp, E=E, counts_host=counts_host,
        counts=torch.tensor(counts_host, dtype=torch.int32, device=DEV),
        starts=(torch.arange(E, dtype=torch.int32) * mtp).to(DEV),
        gB=gB, gS=gS, uB=uB, uS=uS,
        dB=_ptrs([d[0] for d in down]), dS=_ptrs([d[1] for d in down]),
        # The pointer arrays above do not own their tensors.
        owners=(weights, down),
        A=A, H=H, inter=inter, out=out,
        Ip=_row_ptrs(inter, E, mtp), Op=_row_ptrs(out, E, mtp),
        ws1=torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV),
        ws3=torch.zeros(K // 256 + 17, dtype=torch.int32, device=DEV),
        mmt=(mtp + 15) // 16,
        s1_refs=[_s1_ref(rows(A, e), weights[e][2], weights[e][3])
                 if counts_host[e] else None for e in range(E)],
        s3_refs=[rows(H, e).float() @ down[e][2].float().t()
                 if counts_host[e] else None for e in range(E)],
    )


def _v3_s1(P, cnt, total_rows, v, order=None, s1_input=None):
    mgm.marlin_grouped_stage1_fused_mxfp4_situ_v3(
        P["A"] if s1_input is None else s1_input, P["inter"], cnt, P["starts"],
        P["gB"], P["gS"], P["uB"], P["uS"], P["Ip"], P["N"], P["K"], P["ws1"],
        max_m_tiles=P["mmt"], mtp=P["mtp"], num_experts=P["E"],
        total_rows=total_rows, variant=v, expert_order=order)


def _v3_s3(P, cnt, v, order=None, s3_input=None):
    mgm.marlin_grouped_m16_mxfp4_v3(
        P["H"] if s3_input is None else s3_input, P["dB"], P["Op"], P["dS"],
        P["starts"], cnt, P["E"], P["K"], P["N"], P["ws3"], P["E"],
        P["K"] // 256, P["mmt"], variant=v, expert_order=order)


def t4v3_grouped_skewed(variants):
    P = _v3_skewed_problem(28, seed=4800)
    counts_host, mtp = P["counts_host"], P["mtp"]
    total = sum(counts_host)
    print(f"    T4v3 counts: {counts_host}")
    order = mgm.compute_mxfp4_v3_expert_order(P["counts"])
    order_host = order.tolist()
    order_ok = order_host == sorted(range(P["E"]), key=lambda e: (-counts_host[e], e))
    print(f"    T4v3 expert order matches host sort: {order_ok}")
    zero = torch.zeros(P["E"], dtype=torch.int32, device=DEV)
    zero_order = mgm.compute_mxfp4_v3_expert_order(zero)
    ok = order_ok
    for v in variants:
        res = {}
        for tag, o in (("identity", None), ("ordered", order)):
            P["inter"].fill_(NAN)
            _v3_s1(P, P["counts"], total, v, o)
            P["out"].fill_(NAN)
            _v3_s3(P, P["counts"], v, o)
            torch.cuda.synchronize()
            res[tag] = (P["inter"].clone(), P["out"].clone())
        inter, out = res["identity"]
        g1 = _gate_rows(inter, P["s1_refs"], counts_host, mtp, f"T4v3 S1 v{v}")
        c1 = _canary_ok(inter, counts_host, mtp)
        g3 = _gate_rows(out, P["s3_refs"], counts_host, mtp, f"T4v3 S3 v{v}")
        c3 = _canary_ok(out, counts_host, mtp)
        same = all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                   for a, b in zip(res["identity"], res["ordered"]))
        P["inter"].fill_(NAN)
        _v3_s1(P, zero, 0, v, zero_order)
        P["out"].fill_(NAN)
        _v3_s3(P, zero, v, zero_order)
        torch.cuda.synchronize()
        e1 = bool(torch.isnan(P["inter"]).all())
        e3 = bool(torch.isnan(P["out"]).all())
        print(f"    T4v3 v{v}: canary S1={c1} S3={c3}; order bit-identical={same}; "
              f"all-empty wrote nothing S1={e1} S3={e3}")
        ok = ok and g1 and c1 and g3 and c3 and same and e1 and e3
    report("T4v3 grouped v3 with skewed and zero-token experts (E=28), expert order", ok)


def t5v3_mutations(variants):
    K, N = K3_K, K3_N
    M = 40                       # N width 48 for NCAP >= 48, two chunks for NCAP=32
    weights, gB, gS, uB, uS = _w13_experts(1, K, N, seed0=5500)
    qw, sc, w1_ref, w3_ref = weights[0]
    torch.manual_seed(5550)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    ref = _s1_ref(x, w1_ref, w3_ref)
    inter = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    starts = torch.zeros(1, dtype=torch.int32, device=DEV)
    counts = torch.tensor([M], dtype=torch.int32, device=DEV)
    Cp = _ptrs([inter])
    ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=DEV)
    g_s_rolled = torch.roll(sc[0], 1, dims=0).contiguous()
    gS_rolled = _ptrs([g_s_rolled])

    def run(gb, gs, ub, us, v):
        inter.fill_(NAN)
        mgm.marlin_grouped_stage1_fused_mxfp4_situ_v3(
            x, inter, counts, starts, gb, gs, ub, us, Cp, N, K, ws,
            max_m_tiles=(M + 15) // 16, mtp=M, num_experts=1, total_rows=M,
            variant=v)
        torch.cuda.synchronize()
        return inter

    caught, arms, control_ok = 0, 0, True
    for v in variants:
        control_ok = gate(run(gB, gS, uB, uS, v), ref, f"T5v3.control v{v}") and control_ok
        arms += 2
        if not gate(run(uB, uS, gB, gS, v), ref, f"T5v3.m2 v{v} gate/up-swap"):
            caught += 1
        if not gate(run(gB, gS_rolled, uB, uS, v), ref, f"T5v3.m3 v{v} off-by-one-scale"):
            caught += 1

    def must_raise(name, fn, exc):
        nonlocal caught, arms
        arms += 1
        try:
            fn()
            torch.cuda.synchronize()
            print(f"    T5v3[{name}]: NOT caught")
        except exc as e:
            caught += 1
            print(f"    T5v3[{name}]: caught ({str(e).splitlines()[0][:64]})")

    bad_variant = max(V3_VARIANTS) + 1
    must_raise(f"m6 wrapper variant={bad_variant}",
               lambda: run(gB, gS, uB, uS, bad_variant), ValueError)
    mod = mgm._load_module()
    must_raise(f"m6 raw variant={bad_variant}",
               lambda: mod.grouped_marlin_gemm_m16_s1_mxfp4_situ_v3(
                   x, gB, uB, Cp, gS, uS, starts, counts, 1, N, K, ws, N // 256,
                   (M + 15) // 16, bad_variant, None),
               RuntimeError)
    must_raise("m7 wrapper expert_order wrong length",
               lambda: mgm.marlin_grouped_stage1_fused_mxfp4_situ_v3(
                   x, inter, counts, starts, gB, gS, uB, uS, Cp, N, K, ws,
                   max_m_tiles=(M + 15) // 16, mtp=M, num_experts=1,
                   total_rows=M, variant=V3_VARIANTS[0],
                   expert_order=torch.zeros(2, dtype=torch.int32, device=DEV)),
               ValueError)
    report("T5v3 mutation arms caught (controls pass)",
           caught == arms and control_ok, f"catch count {caught}/{arms}")


def t6v3_graph_replay(variants):
    """Expert order + fused S1 + S3 (consuming the S1 output) in one CUDA
    graph; two replays must reproduce the eager launches bit for bit."""
    P = _v3_skewed_problem(12, seed=6600)
    total = sum(P["counts_host"])
    order = torch.empty(P["E"], dtype=torch.int32, device=DEV)
    ok = True
    for v in variants:
        def forward():
            mgm.compute_mxfp4_v3_expert_order(P["counts"], order)
            _v3_s1(P, P["counts"], total, v, order)
            _v3_s3(P, P["counts"], v, order, s3_input=P["inter"])

        P["inter"].fill_(NAN)
        P["out"].fill_(NAN)
        forward()
        torch.cuda.synchronize()
        eager = (P["inter"].clone(), P["out"].clone())

        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            forward()                       # warm-up on the capture stream
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            forward()

        same = True
        for _ in range(2):
            P["inter"].fill_(NAN)
            P["out"].fill_(NAN)
            graph.replay()
            torch.cuda.synchronize()
            same = same and all(
                torch.equal(_canonical_bits(a), _canonical_bits(b))
                for a, b in zip(eager, (P["inter"], P["out"])))
        canary = (_canary_ok(P["inter"], P["counts_host"], P["mtp"])
                  and _canary_ok(P["out"], P["counts_host"], P["mtp"]))
        print(f"    T6v3 v{v}: replays bit-identical to eager={same} canary={canary}")
        ok = ok and same and canary
        del graph
    report("T6v3 CUDA graph capture of order + S1 + S3, replay bit-identical", ok)


def do_build():
    """Rebuild the marlin TU with `-Xptxas -v` captured; print per-kernel
    register counts and FAIL on an E2M1-vs-U4B8 register-tier jump (>16)."""
    repo = Path(__file__).resolve().parents[2]
    kdir = repo / "batchgen_kernels"
    src = kdir / "src" / "moe" / "marlin_grouped_gemm.cu"
    src.touch()  # force recompilation of just this TU (incremental build)
    env = os.environ.copy()
    env["NVCC_APPEND_FLAGS"] = (env.get("NVCC_APPEND_FLAGS", "")
                                + " -Xptxas -v").strip()
    proc = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=kdir, env=env, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    log_path = kdir / "build_ptxas_marlin.log"
    log_path.write_text(log)
    print(f"build log: {log_path}")
    if proc.returncode != 0:
        print(log[-4000:])
        print("FATAL: build failed")
        sys.exit(2)

    # pair each 'Compiling entry function' with the next 'Used N registers'
    # and the spill line ptxas prints between them
    entries, cur, spill = [], None, None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur, spill = m.group(1), None
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur is not None:
            spill = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m and cur is not None:
            entries.append((cur, int(m.group(1)), spill))
            cur = None

    def demangle(n):
        try:
            out = subprocess.run(["c++filt", n], capture_output=True,
                                 text=True).stdout.strip()
            return out or n
        except OSError:
            return n

    marlin = [(demangle(n), n, r, sp) for n, r, sp in entries
              if "Marlin" in n or "marlin_v2" in n or "marlin_v3" in n]
    if not marlin:
        print("FATAL: no Marlin ptxas lines captured — object was not "
              "recompiled? Delete the build dir / .so and rerun --build.")
        sys.exit(2)
    regs = {}
    for dem, mangled, r, sp in marlin:
        print(f"    {r:4d} regs  spill(st,ld)={sp}  {dem}")
        for fam in ("MarlinGrouped_M16_S1", "MarlinGrouped_M16"):
            if fam + "I" in mangled or fam + "<" in dem:
                codec = ("E2M1" if ("L6WCodec1E" in mangled or "WCodec)1" in dem)
                         else "U4B8" if ("L6WCodec0E" in mangled or "WCodec)0" in dem)
                         else "?")
                regs[(fam, codec)] = r
                break
    ok = True
    for fam in ("MarlinGrouped_M16_S1", "MarlinGrouped_M16"):
        u, e = regs.get((fam, "U4B8")), regs.get((fam, "E2M1"))
        if u is not None and e is not None:
            tier_ok = e <= u + 16
            print(f"    {fam}: U4B8={u} E2M1={e} regs "
                  f"({'same tier' if tier_ok else 'TIER JUMP — investigate'})")
            ok = ok and tier_ok
        else:
            print(f"    {fam}: could not classify both codecs (U4B8={u}, "
                  f"E2M1={e}) — inspect {log_path} manually")
            ok = False

    # v2 gate: two CTAs per SM need <= 128 registers at 256 threads, and the
    # instantiations must not spill. 4 variants x {GroupedS1, Grouped}.
    v2 = [(dem, r, sp) for dem, mangled, r, sp in marlin if "marlin_v2" in mangled]
    if len(v2) != 2 * len(V2_VARIANTS):
        print(f"    marlin_v2: expected {2 * len(V2_VARIANTS)} instantiations, "
              f"found {len(v2)} — inspect {log_path} manually")
        ok = False
    for dem, r, sp in v2:
        v2_ok = r <= 128 and sp == (0, 0)
        print(f"    marlin_v2 {'OK  ' if v2_ok else 'FAIL'} regs={r} "
              f"spill(st,ld)={sp}  {dem}")
        ok = ok and v2_ok

    # v3 gate: same register and spill bounds; the wgmma kernels are one
    # symbol per (variant, S1/grouped), all N widths inside.
    v3 = [(dem, r, sp) for dem, mangled, r, sp in marlin
          if "marlin_v3" in mangled and "Grouped" in mangled]
    if len(v3) != 2 * len(V3_VARIANTS):
        print(f"    marlin_v3: expected {2 * len(V3_VARIANTS)} instantiations, "
              f"found {len(v3)} — inspect {log_path} manually")
        ok = False
    for dem, r, sp in v3:
        v3_ok = r <= 128 and sp == (0, 0)
        print(f"    marlin_v3 {'OK  ' if v3_ok else 'FAIL'} regs={r} "
              f"spill(st,ld)={sp}  {dem}")
        ok = ok and v3_ok

    # Dynamic shared memory is not in the ptxas report; ask the freshly built
    # extension (a subprocess, because this process imported the old one).
    probe = ("from batchgen_kernels import load_extension as L\n"
             "m = L('batchgen_kernels.moe._C_marlin_grouped_gemm')\n"
             "for v in range(4):\n"
             "    for f in (True, False):\n"
             "        s, b = m.marlin_mxfp4_v2_launch_config(v, f)\n"
             "        print(f'    marlin_v2 variant {v} {\"S1\" if f else \"grouped\"}: '\n"
             "              f'stages={s} dynamic smem={b} B')\n"
             f"for v in {V3_VARIANTS!r}:\n"
             "    for f in (True, False):\n"
             "        s, b, c = m.marlin_mxfp4_v3_launch_config(v, f)\n"
             f"        tag = 'OK  ' if b <= {V3_SMEM_LIMIT} else 'FAIL'\n"
             "        print(f'    marlin_v3 {tag} variant {v} {\"S1\" if f else \"grouped\"}: '\n"
             "              f'stages={s} dynamic smem={b} B cols={c}')\n"
             f"        assert b <= {V3_SMEM_LIMIT}\n")
    smem = subprocess.run([sys.executable, "-c", probe], cwd=repo,
                          capture_output=True, text=True)
    print(smem.stdout.rstrip() or smem.stderr[-2000:])
    if smem.returncode != 0:
        if smem.stdout:
            print(smem.stderr[-2000:])
        ok = False
    if not ok:
        sys.exit(2)
    print("build + register-tier check OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run T0 only")
    ap.add_argument("--build", action="store_true",
                    help="rebuild with ptxas -v and gate the register tier")
    ap.add_argument("--v2", action="store_true",
                    help="run T0 and the v2 ladder only")
    ap.add_argument("--v2-variants", default=",".join(map(str, V2_VARIANTS)),
                    help="comma-separated v2 variants to test (default: all)")
    ap.add_argument("--v3", action="store_true",
                    help="run T0 and the v3 ladder only")
    ap.add_argument("--v3-variants", default=",".join(map(str, V3_VARIANTS)),
                    help="comma-separated v3 variants to test (default: all)")
    args = ap.parse_args()
    variants = [int(v) for v in args.v2_variants.split(",")]
    unknown = [v for v in variants if v not in V2_VARIANTS]
    if unknown:
        ap.error(f"unknown v2 variants {unknown}; valid: {V2_VARIANTS}")
    v3_variants = [int(v) for v in args.v3_variants.split(",")]
    unknown = [v for v in v3_variants if v not in V3_VARIANTS]
    if unknown:
        ap.error(f"unknown v3 variants {unknown}; valid: {V3_VARIANTS}")

    if args.build:
        do_build()
        return

    if not mgm.is_marlin_mxfp4_available():
        print("FATAL: marlin MXFP4 kernel entries missing — rebuild "
              "batchgen_kernels (this is the L1 hard-fail).")
        sys.exit(2)

    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name()}")

    t0_smoke()
    if not args.smoke:
        run_v1 = not (args.v2 or args.v3)
        run_v2 = run_v1 or args.v2
        run_v3 = run_v1 or args.v3
        if run_v1:
            t1_decode_bitexact()
            t2_m16_parity()
            t3_tensors = t3_s1_situ_parity()
            t4_grouped_zero_token()
            t5_mutations(t3_tensors)
            t6_int4_regression()
            t6b_int4_s1_regression()
            t7_wrapper_hardfail_negatives()
        if run_v2:
            print_v2_launch_configs(variants)
            t1v2_decode_bitexact(variants)
            t2v2_m16_parity(variants)
            t3v2_s1_situ_parity(variants)
            t4v2_grouped_skewed(variants)
            t5v2_mutations(variants)
        if run_v3:
            print_v3_launch_configs(v3_variants)
            t1v3_decode_bitexact(v3_variants)
            t2v3_m16_parity(v3_variants)
            t3v3_s1_situ_parity(v3_variants)
            t4v3_grouped_skewed(v3_variants)
            t5v3_mutations(v3_variants)
            t6v3_graph_replay(v3_variants)

    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} ladder stages passed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
