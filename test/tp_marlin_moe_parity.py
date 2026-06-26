#!/usr/bin/env python3
"""GPU parity test for the Kimi-K2.5 TP-MoE GPTQ-Marlin decode path.

Validates that BatchGen's marlin packing (raw -> marlin via
``raw_to_marlin_fused_gpu``) feeds SGLang's ``fused_marlin_moe`` and produces the
same MoE FFN output as a dequant fp32 reference. Two levels:

  L1 (full expert, no TP): build random K2.5 raw INT4 experts, re-marlinize into
     w1=concat(gate,up) / w2=down GPTQ-Marlin tensors, run fused_marlin_moe with
     all-symmetric args, and compare to the dequant reference
     silu(x@Wg)*(x@Wu) then @Wd.

  L2 (TP shard, world_size=16, rank=3): simulate the per-expert marlin checkpoint,
     run the loader pipeline marlin->raw->slice->raw_to_marlin for ONE rank's
     intermediate slice, run fused_marlin_moe on the sharded weights, and compare
     to the correspondingly sliced reference (partial sum over the rank's slice).
     Also checks that the 16 per-rank reference partials sum to the full L1
     reference (validates the AllReduce(SUM) intermediate partition).

GPU-only (the marlin transform kernel + sgl_kernel are CUDA). Run on a GPU host:
    python test/tp_marlin_moe_parity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from batchgen.moe.marlin_weight_prep import _dequantize_k25_int4
from batchgen.moe.marlin_transform import (
    raw_to_marlin_fused_gpu,
    marlin_to_wgmma_fused_gpu,
)

# Reuse the raw-INT4 generator from the existing repack parity test.
from tp_moe_repack_parity import build_raw_int4  # noqa: E402

# Kimi-K2.5 constants
H = 7168              # hidden dim
N_INTER = 2048        # moe_intermediate
GROUP_SIZE = 32       # INT4 group size (gs=32)
WORLD_SIZE = 16       # tp16
INTER_PR = N_INTER // WORLD_SIZE  # 128

E = 8                 # small expert count (bounds memory)
M = 8                 # tokens
TOP_K = 1             # top-1 routing for a clean reference


def _load_fused_marlin_moe():
    """Import SGLang's fused_marlin_moe, seeding global server args if needed
    (mirrors model._load_fused_marlin_moe / _load_fused_experts_impl)."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    return fused_marlin_moe


def _deq(raw, s, K, N):
    """Dequant a K2.5 raw INT4 projection to fp32 [K, N] (input-major weight)."""
    return _dequantize_k25_int4(raw, s, K, N).float()


def _build_experts(device, seed0):
    """Build E random raw INT4 experts: gate/up [N_INTER,H], down [H,N_INTER]."""
    experts = []
    for e in range(E):
        gate_raw, gate_s = build_raw_int4(N_INTER, H, device, seed0 + e * 3 + 0)
        up_raw, up_s = build_raw_int4(N_INTER, H, device, seed0 + e * 3 + 1)
        down_raw, down_s = build_raw_int4(H, N_INTER, device, seed0 + e * 3 + 2)
        experts.append((gate_raw, gate_s, up_raw, up_s, down_raw, down_s))
    return experts


def _full_weights_cache(experts):
    """Lazy fp32 dequant cache per expert: e -> (Wg[H,N_INTER], Wu, Wd[N_INTER,H])."""
    cache = {}

    def get(e):
        if e not in cache:
            g, gs, u, us, d, ds = experts[e]
            cache[e] = (
                _deq(g, gs, H, N_INTER),
                _deq(u, us, H, N_INTER),
                _deq(d, ds, N_INTER, H),
            )
        return cache[e]

    return get


def _ref_ffn(hidden, topk_ids, topk_weights, get_full, col0=None, col1=None):
    """fp32 reference MoE FFN. If col0:col1 given, restrict the intermediate to
    that output slice (gate/up output cols, down input rows) → partial output."""
    out = torch.zeros(M, H, device=hidden.device, dtype=torch.float32)
    for t in range(M):
        e = int(topk_ids[t, 0])
        Wg, Wu, Wd = get_full(e)
        if col0 is not None:
            Wg = Wg[:, col0:col1]
            Wu = Wu[:, col0:col1]
            Wd = Wd[col0:col1, :]
        x = hidden[t].float()
        act = F.silu(x @ Wg) * (x @ Wu)
        out[t] = float(topk_weights[t, 0]) * (act @ Wd)
    return out


def _routing(device):
    g = torch.Generator(device=device).manual_seed(99)
    hidden = (torch.randn(M, H, device=device, generator=g) * 0.1).to(torch.bfloat16)
    if os.environ.get("DIAG_DEGENERATE") == "1":
        # isolate routing/tw: all tokens -> expert 0, weight 1.0
        topk_ids = torch.zeros(M, TOP_K, dtype=torch.int32, device=device)
        topk_weights = torch.ones(M, TOP_K, device=device).float()
        return hidden, topk_ids, topk_weights
    topk_ids = torch.randint(0, E, (M, TOP_K), dtype=torch.int32, device=device, generator=g)
    topk_weights = (torch.rand(M, TOP_K, device=device, generator=g) * 0.5 + 0.25).float()
    return hidden, topk_ids, topk_weights


def _report(name, ref, got, atol=1e-2, rtol=1e-2):
    ref = ref.float()
    got = got.float()
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    max_abs = (ref - got).abs().max().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max_abs_diff={max_abs:.3e}")
    return ok


# --------------------------------------------------------------------------- #
# L1 — full expert, no TP
# --------------------------------------------------------------------------- #
def test_l1_full(device, fused_marlin_moe, workspace):
    print("L1 — full-expert marlin MoE parity (no TP):")
    experts = _build_experts(device, seed0=1000)
    get_full = _full_weights_cache(experts)
    hidden, topk_ids, topk_weights = _routing(device)

    # Build kernel weights: w1 = re-marlin(concat(gate,up) on output dim), w2 = re-marlin(down).
    w13 = torch.empty(E, H // 16, 4 * N_INTER, dtype=torch.int32, device=device)        # [E,448,8192]
    w13_s = torch.empty(E, H // GROUP_SIZE, 2 * N_INTER, dtype=torch.bfloat16, device=device)  # [E,224,4096]
    w2 = torch.empty(E, N_INTER // 16, 2 * H, dtype=torch.int32, device=device)          # [E,128,14336]
    w2_s = torch.empty(E, N_INTER // GROUP_SIZE, H, dtype=torch.bfloat16, device=device)  # [E,64,7168]
    for e in range(E):
        gate_raw, gate_s, up_raw, up_s, down_raw, down_s = experts[e]
        # marlinize gate/up SEPARATELY then concat marlin cols (marlin(concat)!=concat(marlin)).
        g_mw, g_ms = raw_to_marlin_fused_gpu(gate_raw, gate_s, H, N_INTER)
        u_mw, u_ms = raw_to_marlin_fused_gpu(up_raw, up_s, H, N_INTER)
        w13[e] = torch.cat([g_mw, u_mw], dim=1)
        w13_s[e] = torch.cat([g_ms, u_ms], dim=1)
        w2[e], w2_s[e] = raw_to_marlin_fused_gpu(down_raw, down_s, N_INTER, H)

    out_k = fused_marlin_moe(
        hidden_states=hidden, w1=w13, w2=w2, w1_scale=w13_s, w2_scale=w2_s,
        gating_output=topk_weights, topk_weights=topk_weights, topk_ids=topk_ids,
        global_num_experts=E, expert_map=None,
        g_idx1=None, g_idx2=None, sort_indices1=None, sort_indices2=None,
        w1_zeros=None, w2_zeros=None, workspace=workspace,
        num_bits=4, is_k_full=True, inplace=False, routed_scaling_factor=None,
    )

    out_ref = _ref_ffn(hidden, topk_ids, topk_weights, get_full)
    return _report("L1 full MoE", out_ref, out_k), (experts, get_full, hidden, topk_ids, topk_weights)


# --------------------------------------------------------------------------- #
# L2 — TP shard (one rank's intermediate slice)
# --------------------------------------------------------------------------- #
def test_l2_tp(device, fused_marlin_moe, workspace, shared, rank=3):
    print(f"L2 — TP-shard marlin MoE parity (world_size={WORLD_SIZE}, rank={rank}):")
    experts, get_full, hidden, topk_ids, topk_weights = shared
    r0, r1 = rank * INTER_PR, (rank + 1) * INTER_PR
    dcol0 = rank * (INTER_PR // 8)
    scol0 = rank * (INTER_PR // GROUP_SIZE)

    w13 = torch.empty(E, H // 16, 4 * INTER_PR, dtype=torch.int32, device=device)        # [E,448,512]
    w13_s = torch.empty(E, H // GROUP_SIZE, 2 * INTER_PR, dtype=torch.bfloat16, device=device)  # [E,224,256]
    w2 = torch.empty(E, INTER_PR // 16, 2 * H, dtype=torch.int32, device=device)          # [E,8,14336]
    w2_s = torch.empty(E, INTER_PR // GROUP_SIZE, H, dtype=torch.bfloat16, device=device)  # [E,4,7168]
    for e in range(E):
        gate_raw, gate_s, up_raw, up_s, down_raw, down_s = experts[e]
        # Simulate the per-expert marlin checkpoint (FULL intermediate).
        g_m, g_ms = raw_to_marlin_fused_gpu(gate_raw, gate_s, H, N_INTER)   # [448,4096]/[224,2048]
        u_m, u_ms = raw_to_marlin_fused_gpu(up_raw, up_s, H, N_INTER)
        d_m, d_ms = raw_to_marlin_fused_gpu(down_raw, down_s, N_INTER, H)   # [128,14336]/[64,7168]
        # Loader: marlin -> raw.
        rg, rgs = marlin_to_wgmma_fused_gpu(g_m, g_ms, H, N_INTER)          # [2048,896]/[2048,224]
        ru, rus = marlin_to_wgmma_fused_gpu(u_m, u_ms, H, N_INTER)
        rd, rds = marlin_to_wgmma_fused_gpu(d_m, d_ms, N_INTER, H)          # [7168,256]/[7168,64]
        # gate|up: slice OUTPUT rows, marlinize each separately, concat marlin cols.
        g_mw, g_ms = raw_to_marlin_fused_gpu(rg[r0:r1].contiguous(), rgs[r0:r1].contiguous(), H, INTER_PR)
        u_mw, u_ms = raw_to_marlin_fused_gpu(ru[r0:r1].contiguous(), rus[r0:r1].contiguous(), H, INTER_PR)
        w13[e] = torch.cat([g_mw, u_mw], dim=1)
        w13_s[e] = torch.cat([g_ms, u_ms], dim=1)
        # down: slice INPUT(inter) packed columns + scale columns, re-marlinize.
        d_raw = rd[:, dcol0:dcol0 + INTER_PR // 8].contiguous()            # [7168,16]
        d_raw_s = rds[:, scol0:scol0 + INTER_PR // GROUP_SIZE].contiguous()  # [7168,4]
        w2[e], w2_s[e] = raw_to_marlin_fused_gpu(d_raw, d_raw_s, INTER_PR, H)

    out_k = fused_marlin_moe(
        hidden_states=hidden, w1=w13, w2=w2, w1_scale=w13_s, w2_scale=w2_s,
        gating_output=topk_weights, topk_weights=topk_weights, topk_ids=topk_ids,
        global_num_experts=E, expert_map=None,
        g_idx1=None, g_idx2=None, sort_indices1=None, sort_indices2=None,
        w1_zeros=None, w2_zeros=None, workspace=workspace,
        num_bits=4, is_k_full=True, inplace=False, routed_scaling_factor=None,
    )

    out_ref_tp = _ref_ffn(hidden, topk_ids, topk_weights, get_full, col0=r0, col1=r1)
    ok = _report(f"L2 rank-{rank} partial", out_ref_tp, out_k)

    # Recombine: the 16 per-rank reference partials must sum to the full L1 reference.
    out_recomb = torch.zeros(M, H, device=device, dtype=torch.float32)
    for rr in range(WORLD_SIZE):
        a0, a1 = rr * INTER_PR, (rr + 1) * INTER_PR
        out_recomb += _ref_ffn(hidden, topk_ids, topk_weights, get_full, col0=a0, col1=a1)
    out_full = _ref_ffn(hidden, topk_ids, topk_weights, get_full)
    ok = _report("L2 AllReduce(SUM) recombine == full", out_full, out_recomb) and ok
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required (marlin transform kernel + sgl_kernel are GPU-only).")
        return 0
    device = torch.device("cuda")
    print(f"TP-MoE MARLIN parity test (H={H}, inter={N_INTER}, E={E}, M={M}, "
          f"tp={WORLD_SIZE}, inter_pr={INTER_PR})\n")

    fused_marlin_moe = _load_fused_marlin_moe()
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    workspace = marlin_make_workspace(device, max_blocks_per_sm=4)

    l1_ok, shared = test_l1_full(device, fused_marlin_moe, workspace)
    l2_ok = test_l2_tp(device, fused_marlin_moe, workspace, shared, rank=3)

    passed = l1_ok and l2_ok
    print(f"\n{'ALL PASS' if passed else 'FAILURES PRESENT'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
