#!/usr/bin/env python3
"""STAGED GPU parity ladder for the K3 Marlin-MXFP4 kernels (task #34).

NOT run by the CPU workflow. Staged for h20-instance-1, **GPU 1** (GPU 0
belongs to the model workstream). No benchmarks here — parity only.

How to run (instance-1):
    ssh h20-instance-1
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


def gate(out, ref, name):
    """Project numerical gate. Returns True iff the gate PASSES."""
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
    entries, cur = [], None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m and cur is not None:
            entries.append((cur, int(m.group(1))))
            cur = None

    def demangle(n):
        try:
            out = subprocess.run(["c++filt", n], capture_output=True,
                                 text=True).stdout.strip()
            return out or n
        except OSError:
            return n

    marlin = [(demangle(n), n, r) for n, r in entries if "Marlin" in n]
    if not marlin:
        print("FATAL: no Marlin ptxas lines captured — object was not "
              "recompiled? Delete the build dir / .so and rerun --build.")
        sys.exit(2)
    regs = {}
    for dem, mangled, r in marlin:
        print(f"    {r:4d} regs  {dem}")
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
    if not ok:
        sys.exit(2)
    print("build + register-tier check OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run T0 only")
    ap.add_argument("--build", action="store_true",
                    help="rebuild with ptxas -v and gate the register tier")
    args = ap.parse_args()

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
        t1_decode_bitexact()
        t2_m16_parity()
        t3_tensors = t3_s1_situ_parity()
        t4_grouped_zero_token()
        t5_mutations(t3_tensors)
        t6_int4_regression()
        t6b_int4_s1_regression()
        t7_wrapper_hardfail_negatives()

    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} ladder stages passed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
