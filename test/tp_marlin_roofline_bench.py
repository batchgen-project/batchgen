#!/usr/bin/env python3
"""Standalone TP-MoE Marlin roofline + BatchGen-vs-SGLang benchmark (Kimi-K2.5).

Question this answers: BatchGen's grouped Marlin GEMM (grouped_marlin_gemm_m16 +
silu_mul_split) was TUNED FOR EP-MoE (24 experts/rank, gate/up N=2048, down K=2048).
At the *TP-MoE* shapes introduced on tairan/rt-peel-marlin-tp (commit adce12a0,
world_size=16 -> 384 resident experts/rank), gate|up runs as ONE GEMM over
w13=concat (N=2*inter_pr=256) and down has K=inter_pr=128. Is the EP-tuned kernel
still near the HBM roofline at TP, or do tiny-K / single-N-tile / thin-M leave it
off-roofline?

It times, per batch size, the EXACT production decode call sequence
(_forward_decode_tp_batchgen, model.py adce12a0):
    (a) grouped_marlin_gemm_m16(w13, prob_n=2*inter_pr=256, prob_k=H)   -> gate|up
    (b) silu_mul_split([.,2*inter_pr] -> [.,inter_pr])
    (c) grouped_marlin_gemm_m16(w2,  prob_n=H, prob_k=inter_pr=128)     -> down
and (d) the same logical MoE via SGLang's fused_marlin_moe on byte-identical
weights. Reports per-stage us, achieved GB/s, % of the 3.35 TB/s HBM roofline, the
HBM-floor us and the compute-bound floor us, plus BatchGen-(a+b+c) vs SGLang-(d).
CAVEAT: (a+b+c) is GEMM-ONLY. SGLang's fused_marlin_moe ALSO runs
moe_align_block_size (sort/permute) + gather + topk-weighted scatter-combine, which
production wraps in dispatch_scatter_3d / reduce_weighted_scatter (NOT timed here).
So the BG-vs-SGL column is GEMM-only vs full-MoE — it overstates BG and is NOT a
fair speedup. Trust the per-stage roofline numbers, not that ratio.

Decode at these shapes is HBM-bound (weights read once/step, M-independent for
M<=16): the floor is per-rank-weight-bytes / HBM_BW. We compute that floor and the
compute floor and report % achieved.

GPU-only (Marlin GEMM + transform kernels + sgl_kernel are CUDA). Needs a
batchgen_kernels REBUILD on the remote first (silu_mul_split was ADDED in adce12a0;
the cached .so lacks the symbol):
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a python test/tp_marlin_roofline_bench.py
"""

import os
import sys

# append (not insert(0)): keep the freshly-built site-packages batchgen_kernels.so
# ahead of the workspace SOURCE tree (which has a stale/no compiled .so).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu

# Reuse parity scaffolding (weight-gen + SGLang loader) from the test dir.
from tp_moe_repack_parity import build_raw_int4              # noqa: E402
from tp_marlin_moe_parity import _load_fused_marlin_moe      # noqa: E402

# ---- Kimi-K2.5 TP-MoE constants -------------------------------------------- #
H = 7168                          # hidden dim (stage1 K, down N)
N_INTER = 2048                    # moe_intermediate (full, pre-TP)
GROUP_SIZE = 32                   # int4 group size (gs=32, K2.5 native)
WORLD_SIZE = 16                   # tp16 (production)
INTER_PR = N_INTER // WORLD_SIZE  # 128 (per-rank intermediate)
E = 384                           # routed experts (all resident per rank in TP-MoE)
TOP_K = 8                         # top-8 routing

HBM_BW = 3.35e12                  # H20 HBM3 achievable B/s
COMPUTE_TFLOPS = 148e12           # H20 BF16 dense tensor (Marlin dequants int4->bf16)
ITERS = 100
WARMUP = 10
BATCH_SIZES = [1, 4, 8, 16, 32]   # global bsz = bs*world_size; M/expert = bs*128/384


# --------------------------------------------------------------------------- #
# Roofline byte accounting (int4 weight = 0.5 B, bf16 scale/act = 2 B).
# Fixed per-rank weight+scale bytes (read ~once per decode step). max_m_tiles =
# min(num_global,256)/16 (up to 16 at bs>=16), but the extra m-tiles early-exit
# (marlin_grouped_gemm.cu:958) BEFORE the weight-load prologue (:1071), so each
# weight strip is still read ~once — only experts with count>16 (a few % at bs=32)
# re-read — keeping this byte floor accurate to ~few %:
#   stage1 w13 : int4 E*H*(2*INTER_PR)*0.5 + scales E*(H/gs)*(2*INTER_PR)*2
#   down   w2  : int4 E*INTER_PR*H*0.5     + scales E*(INTER_PR/gs)*H*2
# TP and EP hold the SAME total per-rank bytes (resharded model) -> same floor.
# --------------------------------------------------------------------------- #
S1_W_BYTES = E * H * (2 * INTER_PR) * 0.5 + E * (H // GROUP_SIZE) * (2 * INTER_PR) * 2
S3_W_BYTES = E * INTER_PR * H * 0.5 + E * (INTER_PR // GROUP_SIZE) * H * 2


def stage_bytes(routed):
    """Bytes moved per stage for `routed` total expert-rows (= sum expert_counts)."""
    s1 = S1_W_BYTES + routed * H * 2 + routed * (2 * INTER_PR) * 2      # W + A_in + gateup_out
    s2 = routed * (2 * INTER_PR) * 2 + routed * INTER_PR * 2            # gateup_in + inter_out
    s3 = S3_W_BYTES + routed * INTER_PR * 2 + routed * H * 2            # W + inter_in + expert_out
    return s1, s2, s3


def floor_us(nbytes):
    return nbytes / HBM_BW * 1e6


def compute_floor_us(routed):
    """BF16 tensor-core floor. Per routed row: gate+up+down = 6*INTER_PR*H FLOP."""
    flop = routed * 6 * INTER_PR * H
    return flop / COMPUTE_TFLOPS * 1e6


# --------------------------------------------------------------------------- #
# Marlin extension loader with an informative rebuild guard.
# --------------------------------------------------------------------------- #
def load_marlin_or_die():
    """Import the batchgen_kernels marlin extension and verify the TP symbols.

    silu_mul_split was added in adce12a0; a stale .so will either fail the import
    or lack the symbol. Both cases print a clear 'rebuild batchgen_kernels' hint.
    """
    rebuild = ("\n  -> REBUILD batchgen_kernels on the remote (silu_mul_split was "
               "added in commit adce12a0; the cached .so is stale).\n"
               "     e.g.  pip install -e . --no-build-isolation   (or your op_builder path)\n"
               "     verify: python -c \"import batchgen_kernels.moe._C_marlin_grouped_gemm as m; "
               "m.silu_mul_split\"")
    try:
        import batchgen_kernels.moe._C_marlin_grouped_gemm as mod
    except ImportError as ex:
        print(f"ERROR: cannot import the marlin grouped-GEMM extension: {ex}{rebuild}")
        return None
    for sym in ("grouped_marlin_gemm_m16", "silu_mul_split"):
        if not hasattr(mod, sym):
            print(f"ERROR: marlin extension is missing symbol '{sym}'.{rebuild}")
            return None
    return mod


# --------------------------------------------------------------------------- #
# Weight construction (per-rank TP-MoE Marlin slabs; byte-identical to model.py
# _init_tp_batchgen_buffers / SGLang fused_marlin_moe inputs).
# --------------------------------------------------------------------------- #
def build_tp_weights(device):
    """Return contiguous [E, ...] Marlin tensors for the rank's intermediate slice.

      w13       [E, H//16,         4*INTER_PR] int32  (gate|up concat on marlin cols)
      w13_scale [E, H//GROUP_SIZE, 2*INTER_PR] bf16
      w2        [E, INTER_PR//16,  2*H]        int32  (down)
      w2_scale  [E, INTER_PR//GROUP_SIZE, H]   bf16
    """
    w13 = torch.empty(E, H // 16, 4 * INTER_PR, dtype=torch.int32, device=device)
    w13_s = torch.empty(E, H // GROUP_SIZE, 2 * INTER_PR, dtype=torch.bfloat16, device=device)
    w2 = torch.empty(E, INTER_PR // 16, 2 * H, dtype=torch.int32, device=device)
    w2_s = torch.empty(E, INTER_PR // GROUP_SIZE, H, dtype=torch.bfloat16, device=device)

    for e in range(E):
        # gate/up: per-rank output width INTER_PR, contraction K=H.
        g_raw, g_rs = build_raw_int4(INTER_PR, H, device, seed=e * 3 + 0)
        u_raw, u_rs = build_raw_int4(INTER_PR, H, device, seed=e * 3 + 1)
        g_mw, g_ms = raw_to_marlin_fused_gpu(g_raw, g_rs, H, INTER_PR)
        u_mw, u_ms = raw_to_marlin_fused_gpu(u_raw, u_rs, H, INTER_PR)
        # marlin(concat) != concat(marlin): marlinize separately, concat marlin cols.
        w13[e] = torch.cat([g_mw, u_mw], dim=1)
        w13_s[e] = torch.cat([g_ms, u_ms], dim=1)
        # down: per-rank contraction K=INTER_PR, output N=H.
        d_raw, d_rs = build_raw_int4(H, INTER_PR, device, seed=e * 3 + 2)
        w2[e], w2_s[e] = raw_to_marlin_fused_gpu(d_raw, d_rs, INTER_PR, H)

    return w13, w13_s, w2, w2_s


def make_routing(num_global, device):
    """Random distinct top-8 routing over E experts (decode-realistic, uniform prior)."""
    g = torch.Generator(device=device).manual_seed(1234 + num_global)
    hidden = (torch.randn(num_global, H, device=device, generator=g) * 0.1).to(torch.bfloat16)
    topk_ids = torch.empty(num_global, TOP_K, dtype=torch.int32, device=device)
    for t in range(num_global):
        topk_ids[t] = torch.randperm(E, device=device, generator=g)[:TOP_K].to(torch.int32)
    topk_weights = (torch.rand(num_global, TOP_K, device=device, generator=g) * 0.5 + 0.25).float()
    return hidden, topk_ids, topk_weights


# --------------------------------------------------------------------------- #
def cuda_time(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS * 1000.0  # us


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required (Marlin GEMM/transform + sgl_kernel are GPU-only).")
        return 0
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    mod = load_marlin_or_die()
    if mod is None:
        return 2

    print(f"TP-MoE Marlin roofline bench  H={H} inter={N_INTER} ws={WORLD_SIZE} "
          f"inter_pr={INTER_PR} E={E} top_k={TOP_K}")
    print(f"Device: {torch.cuda.get_device_name()}  HBM_BW={HBM_BW/1e12:.2f} TB/s  "
          f"BF16={COMPUTE_TFLOPS/1e12:.0f} TFLOP/s  ridge_AI={COMPUTE_TFLOPS/HBM_BW:.1f} FLOP/B  "
          f"iters={ITERS}")
    print(f"Per-rank weight-read floor: S1(w13)={S1_W_BYTES/1e6:.1f} MB ({floor_us(S1_W_BYTES):.1f} us)  "
          f"S3(w2)={S3_W_BYTES/1e6:.1f} MB ({floor_us(S3_W_BYTES):.1f} us)  "
          f"total={floor_us(S1_W_BYTES + S3_W_BYTES):.1f} us")
    print()

    print("Building TP-MoE Marlin weights (384 experts)...", end=" ", flush=True)
    w13, w13_s, w2, w2_s = build_tp_weights(device)
    print("done")

    # SGLang fused_marlin_moe + workspace (optional baseline).
    sgl_moe = None
    sgl_ws = None
    try:
        sgl_moe = _load_fused_marlin_moe()
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
        sgl_ws = marlin_make_workspace(device, max_blocks_per_sm=4)
    except Exception as ex:  # pragma: no cover
        print(f"WARN: SGLang fused_marlin_moe unavailable ({ex}); BatchGen-only.")

    n_tiles_s1 = (2 * INTER_PR) // 256   # ws<=16 -> >=1 (TP16 -> 1)
    n_tiles_s3 = H // 256                # 28

    # Per-expert weight/scale pointer arrays (stable across the sweep).
    s1_B = torch.tensor([w13[e].data_ptr() for e in range(E)], dtype=torch.int64, device=device)
    s1_S = torch.tensor([w13_s[e].data_ptr() for e in range(E)], dtype=torch.int64, device=device)
    s3_B = torch.tensor([w2[e].data_ptr() for e in range(E)], dtype=torch.int64, device=device)
    s3_S = torch.tensor([w2_s[e].data_ptr() for e in range(E)], dtype=torch.int64, device=device)
    expert_starts_cache = {}

    hdr = (f"{'bs':>3} {'M/e':>4} {'routed':>6} | "
           f"{'S1 us':>7} {'S2 us':>6} {'S3 us':>7} {'BG tot':>7} | "
           f"{'GB/s':>6} {'%roof':>6} | {'HBMfl':>6} {'cmpfl':>6} | {'SGL tot':>7} {'SGLf/BGg':>10}")
    print(hdr)
    print("-" * len(hdr))

    for bs in BATCH_SIZES:
        num_global = bs * WORLD_SIZE
        hidden, topk_ids, topk_weights = make_routing(num_global, device)

        counts = torch.bincount(topk_ids.reshape(-1), minlength=E).int()
        routed = int(counts.sum().item())                 # = num_global * TOP_K
        m_per_e = routed / E
        # Production fixes mtp = tpbuf.max_tokens_padded = BATCHGEN_KIMI_TP_MOE_MTP
        # (default 256), a CONSTANT buffer stride — NOT counts.max(). max_m_tiles is
        # then driven by min(num_global, 256), growing 1/4/8/16/16 across the bs sweep
        # (matches _forward_decode_tp_batchgen, model.py:1447,1469-1470). counts.max()
        # gave a 16x-too-small grid and deleted the launch overhead doc #3 quantifies.
        mtp = int(os.getenv("BATCHGEN_KIMI_TP_MOE_MTP", "256"))
        max_m_tiles = (min(num_global, mtp) + 15) // 16

        if mtp not in expert_starts_cache:
            expert_starts_cache[mtp] = torch.arange(E, dtype=torch.int32, device=device) * mtp
        expert_starts = expert_starts_cache[mtp]
        expert_counts = counts.clamp(max=mtp)

        # 3D grouped buffers (values are timing-irrelevant; parity is a separate test).
        dispatched_x = (torch.randn(E * mtp, H, dtype=torch.bfloat16, device=device) * 0.1)
        gateup = torch.empty(E * mtp, 2 * INTER_PR, dtype=torch.bfloat16, device=device)
        intermediate = torch.empty(E * mtp, INTER_PR, dtype=torch.bfloat16, device=device)
        expert_out = torch.empty(E * mtp, H, dtype=torch.bfloat16, device=device)

        gu_row = 2 * INTER_PR * 2     # bf16 bytes per gateup row
        eo_row = H * 2
        s1_C = torch.tensor([gateup.data_ptr() + e * mtp * gu_row for e in range(E)],
                            dtype=torch.int64, device=device)
        s3_C = torch.tensor([expert_out.data_ptr() + e * mtp * eo_row for e in range(E)],
                            dtype=torch.int64, device=device)
        s1_ws = torch.zeros(E * (n_tiles_s1 + 17), dtype=torch.int32, device=device)
        s3_ws = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=device)

        def bg_s1():
            mod.grouped_marlin_gemm_m16(
                dispatched_x, s1_B, s1_C, s1_S, expert_starts, expert_counts,
                E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, max_m_tiles)

        def bg_s2():
            mod.silu_mul_split(gateup, intermediate, expert_counts, E, mtp, INTER_PR)

        def bg_s3():
            mod.grouped_marlin_gemm_m16(
                intermediate, s3_B, s3_C, s3_S, expert_starts, expert_counts,
                E, H, INTER_PR, s3_ws, E, n_tiles_s3, max_m_tiles)

        def bg_all():
            bg_s1(); bg_s2(); bg_s3()

        def sgl_all():
            return sgl_moe(
                hidden_states=hidden, w1=w13, w2=w2, w1_scale=w13_s, w2_scale=w2_s,
                gating_output=topk_weights, topk_weights=topk_weights, topk_ids=topk_ids,
                global_num_experts=E, expert_map=None,
                g_idx1=None, g_idx2=None, sort_indices1=None, sort_indices2=None,
                w1_zeros=None, w2_zeros=None, workspace=sgl_ws,
                num_bits=4, is_k_full=True, inplace=False, routed_scaling_factor=None)

        for _ in range(WARMUP):
            bg_all()
            if sgl_moe is not None:
                sgl_all()
        torch.cuda.synchronize()

        s1_us = cuda_time(bg_s1)
        s2_us = cuda_time(bg_s2)
        s3_us = cuda_time(bg_s3)
        bg_us = cuda_time(bg_all)
        sgl_us = cuda_time(sgl_all) if sgl_moe is not None else float('nan')

        b1, b2, b3 = stage_bytes(routed)
        tot_bytes = b1 + b2 + b3
        gbps = tot_bytes / (bg_us * 1e-6) / 1e9
        roof = floor_us(tot_bytes)
        cmp_fl = compute_floor_us(routed)
        pct = roof / bg_us * 100.0
        if sgl_moe is not None and sgl_us == sgl_us:  # not NaN
            # SGL(full MoE) / BG(GEMM-only) — apples-to-oranges, NOT a fair speedup
            cmp = f"{sgl_us/bg_us:.2f}x"
        else:
            cmp = "n/a"

        print(f"{bs:3d} {m_per_e:4.1f} {routed:6d} | "
              f"{s1_us:7.1f} {s2_us:6.1f} {s3_us:7.1f} {bg_us:7.1f} | "
              f"{gbps:6.0f} {pct:5.1f}% | {roof:6.1f} {cmp_fl:6.1f} | {sgl_us:7.1f} {cmp:>10}")

    print()
    print("S1=grouped_marlin_gemm_m16(w13,N=2*inter_pr=256,K=H)  S2=silu_mul_split  "
          "S3=grouped_marlin_gemm_m16(w2,N=H,K=inter_pr=128)")
    print("%roof = HBM-weight-read floor / measured (decode is weight-read bound). "
          "HBMfl/cmpfl = HBM / BF16-compute floor us.")
    print("SGLf/BGg = SGL fused_marlin_moe (full MoE: sort+gather+GEMMs+combine) / "
          "BG (S1+S2+S3 GEMMs ONLY). >1 does NOT mean BG faster end-to-end — BG "
          "excludes dispatch_scatter_3d + reduce_weighted_scatter; apples-to-oranges.")
    print("Read: S1 (K=7168 -> 56 k-tiles) should sit near its floor; S3 (K=128 -> 1 "
          "k-tile, STAGES=4 pipeline degenerate) is the off-roofline suspect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
