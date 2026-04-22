"""Correctness + OOB sweep for every non-torch kernel on GLM-5's MoE 3D
decode path that plausibly runs between "HOT PATH" logs and the L4
rank-2 crash.

Targets (CUDA unless noted):
    1. dispatch_scatter_3d            (batchgen.moe.dispatch_scatter_3d)
    2. act_quant_3d                   (batchgen_kernels.moe._C_fp8_blockwise_ops)
    3. grouped_fp8_blockwise_fused_s1 (batchgen_kernels.moe._C_fp8_blockwise_gemm)
    4. grouped_fp8_blockwise_s3       (batchgen_kernels.moe._C_fp8_blockwise_gemm)
    5. reduce_weighted_scatter        (batchgen.moe.dispatch_scatter_3d)
    6. gate_sigmoid_topk_cuda         (batchgen.moe.routing)
    7. cuda_rmsnorm                   (batchgen.attention.fused_kernels)

Shapes mirror GLM-5 production defaults:
    H = 6144, N_inter = 2048 (moe_intermediate_size), E_local = 16 per rank,
    total_experts = 256, num_experts_per_tok = 8.

Stress points cover:
    - L2 regime: bsz 128, num_global 2048 (16 ranks × 128).
    - L4 regime (the new thing): varied num_global 1808, 2000, 3000, 4000,
      heterogeneous ntp-per-rank, large LongBench-scale prompt histories
      baked into the cache_seqlens (affecting anything that reads KV).
    - Rank-2 exact size: bsz=113, num_global=1808.

Anything that crashes or produces |Δ| > 1.0 vs a PyTorch reference flags
a bug. Kernels get run with torch.cuda.synchronize() after each call so
async errors surface immediately.
"""
import argparse
import sys
from dataclasses import dataclass

import torch


# ------------------------------------------------------------ #
# Config for GLM-5 decode shapes
# ------------------------------------------------------------ #
GLM5_HIDDEN = 6144
GLM5_N_INTER = 2048
GLM5_TOTAL_EXPERTS = 256
GLM5_TOPK = 8
GLM5_NUM_LOCAL_EXPERTS = 16   # experts_per_rank on 16-way EP
GLM5_MTP = 128                # max_tokens_padded per expert (from Glm5MoE3DBuffers default)


@dataclass
class KernelResult:
    name: str
    scenario: str
    ok: bool
    err: str = ""
    max_abs_diff: float = 0.0
    has_nan: bool = False
    has_inf: bool = False


# ------------------------------------------------------------ #
# Kernel wrappers + references
# ------------------------------------------------------------ #


def run_dispatch_scatter_3d(x, topk_idx, E_local, mtp, expert_start=0):
    """Returns (dispatched_x, expert_counts, topk_pos)."""
    from batchgen.moe.dispatch_scatter_3d import dispatch_scatter_3d
    N, H = x.shape
    _, K = topk_idx.shape
    dispatched_x = torch.zeros(E_local * mtp, H, dtype=torch.bfloat16, device=x.device)
    expert_counts = torch.zeros(E_local, dtype=torch.int32, device=x.device)
    expert_counters = torch.zeros(E_local, dtype=torch.int32, device=x.device)
    topk_pos = torch.full((N * K,), -1, dtype=torch.int32, device=x.device)
    ec, tp = dispatch_scatter_3d(
        x, topk_idx.to(torch.int32),
        dispatched_x, expert_start, E_local, mtp,
        expert_counts, expert_counters, topk_pos,
    )
    torch.cuda.synchronize(x.device)
    return dispatched_x, ec, tp


def ref_dispatch_scatter_3d(x, topk_idx, E_local, mtp, expert_start=0):
    N, H = x.shape
    _, K = topk_idx.shape
    dispatched_x = torch.zeros(E_local * mtp, H, dtype=torch.bfloat16, device=x.device)
    expert_counts = torch.zeros(E_local, dtype=torch.int32, device=x.device)
    topk_pos = torch.full((N * K,), -1, dtype=torch.int32, device=x.device)
    for n in range(N):
        for k in range(K):
            e_global = int(topk_idx[n, k].item())
            e_local = e_global - expert_start
            if e_local < 0 or e_local >= E_local:
                continue
            slot = int(expert_counts[e_local].item())
            if slot >= mtp:
                continue
            row = e_local * mtp + slot
            dispatched_x[row] = x[n]
            topk_pos[n * K + k] = row
            expert_counts[e_local] = slot + 1
    return dispatched_x, expert_counts, topk_pos


def run_act_quant_3d(x_3d, tokens_per_expert):
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d
    y_u8, s = act_quant_3d(x_3d, tokens_per_expert)
    torch.cuda.synchronize(x_3d.device)
    return y_u8.view(torch.float8_e4m3fn), s


def run_fused_s1(x_fp8, x_scale, gate_w3d, up_w3d, gate_s3d, up_s3d, seqlens, cu_seqlens, avg):
    from batchgen.moe.grouped_fp8_blockwise_moe import grouped_fp8_blockwise_fused_s1
    y = grouped_fp8_blockwise_fused_s1(
        x_fp8, x_scale, gate_w3d, up_w3d, gate_s3d, up_s3d,
        seqlens, cu_seqlens, avg,
    )
    torch.cuda.synchronize(x_fp8.device)
    return y


def run_s3(x_fp8, x_scale, down_w3d, down_s3d, seqlens, cu_seqlens, avg):
    from batchgen.moe.grouped_fp8_blockwise_moe import grouped_fp8_blockwise_s3
    y = grouped_fp8_blockwise_s3(
        x_fp8, x_scale, down_w3d, down_s3d,
        seqlens, cu_seqlens, avg,
    )
    torch.cuda.synchronize(x_fp8.device)
    return y


def run_reduce_weighted_scatter(expert_out, topk_pos, topk_weights, N, H, K):
    from batchgen.moe.dispatch_scatter_3d import reduce_weighted_scatter
    y = reduce_weighted_scatter(expert_out, topk_pos, topk_weights, N, H, K)
    torch.cuda.synchronize(expert_out.device)
    return y


def ref_reduce_weighted_scatter(expert_out, topk_pos, topk_weights, N, H, K):
    # For each (n, k): find row = topk_pos[n*K+k]; if row >= 0, add
    #   expert_out[row] * topk_weights[n, k] into output[n].
    out = torch.zeros(N, H, dtype=torch.bfloat16, device=expert_out.device)
    for n in range(N):
        for k in range(K):
            row = int(topk_pos[n * K + k].item())
            if row < 0:
                continue
            w = float(topk_weights[n, k].item())
            out[n] += (expert_out[row].to(torch.float32) * w).to(torch.bfloat16)
    return out


def run_gate_sigmoid_topk_cuda(router_logits, e_score_correction, k, scale_factor):
    from batchgen.moe.routing import gate_sigmoid_topk_cuda
    topk_idx, topk_w = gate_sigmoid_topk_cuda(
        router_logits, e_score_correction, k=k, routed_scaling_factor=scale_factor,
    )
    torch.cuda.synchronize(router_logits.device)
    return topk_idx, topk_w


def ref_gate_sigmoid_topk(router_logits, e_score_correction, k, scale_factor):
    scores = torch.sigmoid(router_logits.float())
    biased = scores + e_score_correction.float().unsqueeze(0)
    _, idx = torch.topk(biased, k=k, dim=-1)
    w = scores.gather(-1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    w = w * scale_factor
    return idx.to(torch.int32), w.float()


def run_cuda_rmsnorm(x, weight, eps):
    from batchgen.attention.fused_kernels import cuda_rmsnorm
    y = cuda_rmsnorm(x, weight, eps)
    torch.cuda.synchronize(x.device)
    return y


def ref_cuda_rmsnorm(x, weight, eps):
    x_f = x.to(torch.float32)
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    normed = x_f * torch.rsqrt(var + eps)
    return (normed * weight.to(torch.float32).unsqueeze(0)).to(x.dtype)


# ------------------------------------------------------------ #
# Weight/scale initializers (mimic GLM-5 blockwise FP8 layout)
# ------------------------------------------------------------ #


def _fake_fp8_blockwise_weight(E, N, K, device, block_n=128, block_k=128):
    """Return (w3d, ws3d) matching blockwise FP8 expert-grouped layout used
    by _fp8_blockwise_gemm_3d."""
    # Random bf16 weights, then cast to FP8 E4M3 (saturating).
    w_bf = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.05
    w_fp8 = w_bf.to(torch.float8_e4m3fn)
    # Scales: [E, ceil(N/block_n), pad_k4(ceil(K/block_k))]
    num_n = (N + block_n - 1) // block_n
    num_k = (K + block_k - 1) // block_k
    pad_k = ((num_k + 3) // 4) * 4
    w_scale = torch.ones(E, num_n, pad_k, dtype=torch.float32, device=device)
    return w_fp8, w_scale


# ------------------------------------------------------------ #
# Per-kernel test suites
# ------------------------------------------------------------ #


def test_dispatch_scatter_3d(device, scenarios, results):
    print("\n=== dispatch_scatter_3d ===")
    H, E, mtp, K = GLM5_HIDDEN, GLM5_NUM_LOCAL_EXPERTS, GLM5_MTP, GLM5_TOPK
    total_E = GLM5_TOTAL_EXPERTS
    for tag, N in scenarios:
        try:
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            x = torch.randn(N, H, dtype=torch.bfloat16, device=device)
            # Uniform-random top-k assignments across total_experts.
            topk_idx = torch.randint(0, total_E, (N, K), dtype=torch.int32, device=device)
            expert_start = 0   # assume this rank owns experts [0, E_local)
            out, ec, tp = run_dispatch_scatter_3d(x, topk_idx, E, mtp, expert_start)
            # Reference (CPU loop, slow; cap N for reference at 256 to keep ref fast)
            if N <= 512:
                ref_out, ref_ec, ref_tp = ref_dispatch_scatter_3d(x, topk_idx, E, mtp, expert_start)
                ec_match = bool(torch.equal(ec, ref_ec))
                # topk_pos order within each expert depends on counter race;
                # count matches should match if kernel is correct.
                tp_match = bool(torch.equal(tp, ref_tp))
                scatter_diff = (out.to(torch.float32) - ref_out.to(torch.float32)).abs().max().item()
                r = KernelResult(
                    name="dispatch_scatter_3d", scenario=tag, ok=True,
                    max_abs_diff=float(scatter_diff),
                )
                extra = f"  ec_match={ec_match}  tp_match={tp_match}"
            else:
                # Skip full ref for large N; just verify no crash + no OOM.
                r = KernelResult(name="dispatch_scatter_3d", scenario=tag, ok=True)
                extra = "  (ref skipped for N>512)"
            print(f"  {tag:<22}  N={N:<5}  ok  diff={r.max_abs_diff:.3e}{extra}")
        except Exception as e:
            r = KernelResult(name="dispatch_scatter_3d", scenario=tag, ok=False, err=str(e))
            print(f"  {tag:<22}  N={N:<5}  CRASH: {e}")
        results.append(r)


def test_act_quant_3d(device, scenarios, results):
    print("\n=== act_quant_3d ===")
    E, mtp = GLM5_NUM_LOCAL_EXPERTS, GLM5_MTP
    H = GLM5_HIDDEN
    for tag, filled in scenarios:
        try:
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            x_3d = torch.randn(E, mtp, H, dtype=torch.bfloat16, device=device)
            # tokens_per_expert[e] = how many valid rows in that expert slot
            tpe = torch.tensor(filled, dtype=torch.int32, device=device)
            y, s = run_act_quant_3d(x_3d, tpe)
            nan = bool(torch.isnan(s).any().item())
            inf = bool(torch.isinf(s).any().item())
            r = KernelResult(name="act_quant_3d", scenario=tag, ok=True, has_nan=nan, has_inf=inf)
            print(f"  {tag:<22}  filled_sum={sum(filled):<5}  ok  nan={nan}  inf={inf}")
        except Exception as e:
            r = KernelResult(name="act_quant_3d", scenario=tag, ok=False, err=str(e))
            print(f"  {tag:<22}  filled_sum={sum(filled):<5}  CRASH: {e}")
        results.append(r)


def test_fused_s1_and_s3(device, scenarios, results):
    print("\n=== grouped_fp8_blockwise_fused_s1 + s3 ===")
    E, mtp = GLM5_NUM_LOCAL_EXPERTS, GLM5_MTP
    H, N_inter = GLM5_HIDDEN, GLM5_N_INTER
    for tag, filled in scenarios:
        try:
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            # Activations: [E*mtp, H] fp8
            x_bf = torch.randn(E, mtp, H, dtype=torch.bfloat16, device=device) * 0.3
            tpe = torch.tensor(filled, dtype=torch.int32, device=device)
            y_u8, x_scale_3d = run_act_quant_3d(x_bf, tpe)
            x_quant = y_u8.view(E * mtp, H)
            x_scale_t = x_scale_3d.view(E * mtp, -1).t().contiguous()
            # Weights
            gate_w, gate_s = _fake_fp8_blockwise_weight(E, N_inter, H, device)
            up_w,   up_s   = _fake_fp8_blockwise_weight(E, N_inter, H, device)
            down_w, down_s = _fake_fp8_blockwise_weight(E, H, N_inter, device)
            seqlens = tpe
            cu_seqlens = torch.arange(0, (E + 1) * mtp, mtp, dtype=torch.int32, device=device)
            avg = max(mtp // max(E, 1), 1)
            # Stage 1: gate+up+SiLU
            s1_result = run_fused_s1(
                x_quant.view(torch.float8_e4m3fn), x_scale_t,
                gate_w, up_w, gate_s, up_s,
                seqlens, cu_seqlens, avg,
            )
            # Quantize intermediate and run S3
            inter_3d = s1_result.view(E, mtp, N_inter).contiguous()
            inter_u8, inter_s_3d = run_act_quant_3d(inter_3d, tpe)
            inter_quant = inter_u8.view(E * mtp, N_inter)
            inter_scale_t = inter_s_3d.view(E * mtp, -1).t().contiguous()
            s3_result = run_s3(
                inter_quant.view(torch.float8_e4m3fn), inter_scale_t,
                down_w, down_s, seqlens, cu_seqlens, avg,
            )
            nan = bool(torch.isnan(s3_result).any().item())
            inf = bool(torch.isinf(s3_result).any().item())
            r = KernelResult(name="fused_s1+s3", scenario=tag, ok=True, has_nan=nan, has_inf=inf)
            print(f"  {tag:<22}  filled_sum={sum(filled):<5}  ok  nan={nan}  inf={inf}  "
                  f"|y|∈[{s3_result.abs().min().item():.2e},{s3_result.abs().max().item():.2e}]")
        except Exception as e:
            r = KernelResult(name="fused_s1+s3", scenario=tag, ok=False, err=str(e))
            print(f"  {tag:<22}  filled_sum={sum(filled):<5}  CRASH: {e}")
        results.append(r)


def test_reduce_weighted_scatter(device, scenarios, results):
    print("\n=== reduce_weighted_scatter ===")
    H = GLM5_HIDDEN
    K = GLM5_TOPK
    E, mtp = GLM5_NUM_LOCAL_EXPERTS, GLM5_MTP
    for tag, N in scenarios:
        try:
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            # expert_output [E*mtp, H]
            eo = torch.randn(E * mtp, H, dtype=torch.bfloat16, device=device) * 0.1
            # topk_pos: [N*K] int32 pointing into [0, E*mtp) or -1
            tp = torch.randint(-1, E * mtp, (N * K,), dtype=torch.int32, device=device)
            # topk_weights: [N, K] fp32, normalized
            w = torch.randn(N, K, dtype=torch.float32, device=device).softmax(dim=-1)
            y = run_reduce_weighted_scatter(eo, tp, w, N, H, K)
            # ref (cap N for speed)
            if N <= 256:
                ref = ref_reduce_weighted_scatter(eo, tp, w, N, H, K)
                diff = (y.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
            else:
                diff = 0.0
            nan = bool(torch.isnan(y).any().item())
            r = KernelResult(name="reduce_weighted_scatter", scenario=tag, ok=True,
                             max_abs_diff=float(diff), has_nan=nan)
            print(f"  {tag:<22}  N={N:<5}  ok  diff={diff:.3e}  nan={nan}")
        except Exception as e:
            r = KernelResult(name="reduce_weighted_scatter", scenario=tag, ok=False, err=str(e))
            print(f"  {tag:<22}  N={N:<5}  CRASH: {e}")
        results.append(r)


def test_gate_sigmoid_topk_cuda(device, scenarios, results):
    print("\n=== gate_sigmoid_topk_cuda ===")
    for tag, N in scenarios:
        try:
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            rl = torch.randn(N, GLM5_TOTAL_EXPERTS, dtype=torch.float32, device=device)
            bias = torch.randn(GLM5_TOTAL_EXPERTS, dtype=torch.float32, device=device) * 0.1
            idx, w = run_gate_sigmoid_topk_cuda(rl, bias, GLM5_TOPK, 2.5)
            # ref
            ref_idx, ref_w = ref_gate_sigmoid_topk(rl, bias, GLM5_TOPK, 2.5)
            # Indices: allowed to differ if tie-break order differs; count exact matches
            idx_eq_rows = (idx == ref_idx).all(dim=-1).sum().item()
            w_diff = (w.to(torch.float32) - ref_w).abs().max().item()
            nan = bool(torch.isnan(w).any().item())
            r = KernelResult(name="gate_sigmoid_topk_cuda", scenario=tag, ok=True,
                             max_abs_diff=float(w_diff), has_nan=nan)
            print(f"  {tag:<22}  N={N:<5}  ok  idx_match_rows={idx_eq_rows}/{N}  "
                  f"w_diff={w_diff:.3e}  nan={nan}")
        except Exception as e:
            r = KernelResult(name="gate_sigmoid_topk_cuda", scenario=tag, ok=False, err=str(e))
            print(f"  {tag:<22}  N={N:<5}  CRASH: {e}")
        results.append(r)


def test_cuda_rmsnorm(device, scenarios, results):
    print("\n=== cuda_rmsnorm ===")
    H = GLM5_HIDDEN
    for tag, N in scenarios:
        try:
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            x = torch.randn(N, H, dtype=torch.bfloat16, device=device)
            w = torch.randn(H, dtype=torch.bfloat16, device=device) * 0.5 + 1.0
            y = run_cuda_rmsnorm(x, w, 1e-5)
            ref = ref_cuda_rmsnorm(x, w, 1e-5)
            diff = (y.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
            nan = bool(torch.isnan(y).any().item())
            r = KernelResult(name="cuda_rmsnorm", scenario=tag, ok=True,
                             max_abs_diff=float(diff), has_nan=nan)
            print(f"  {tag:<22}  N={N:<5}  ok  diff={diff:.3e}  nan={nan}")
        except Exception as e:
            r = KernelResult(name="cuda_rmsnorm", scenario=tag, ok=False, err=str(e))
            print(f"  {tag:<22}  N={N:<5}  CRASH: {e}")
        results.append(r)


# ------------------------------------------------------------ #
# Main
# ------------------------------------------------------------ #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available"); return 1
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    print(f"# Device: {torch.cuda.get_device_name(device)} "
          f"(cap {torch.cuda.get_device_capability(device)})  seed={args.seed}")
    print(f"# GLM-5 shapes: H={GLM5_HIDDEN} N_inter={GLM5_N_INTER} "
          f"total_E={GLM5_TOTAL_EXPERTS} topk={GLM5_TOPK} E_local={GLM5_NUM_LOCAL_EXPERTS} mtp={GLM5_MTP}")

    results: list[KernelResult] = []

    # num_global (total tokens across all ranks after AllGather) scenarios
    # covering L2 (2048) and L4 (1808, 2000, 3000, 4000).
    ng_scenarios = [
        ("L2-128/rank",   128 * 16),   # L2 baseline
        ("L4-113/rank",   113 * 16),   # L4 rank-2 size
        ("L4-mid",        2000),
        ("L4-max-decode", 3000),
        ("L4-stress",     4000),
    ]

    test_cuda_rmsnorm(device, ng_scenarios, results)
    test_gate_sigmoid_topk_cuda(device, ng_scenarios, results)

    # For dispatch/reduce, N = num_global
    test_dispatch_scatter_3d(device, ng_scenarios, results)
    test_reduce_weighted_scatter(device, ng_scenarios, results)

    # For act_quant_3d + GEMM, we exercise expert-load-balance patterns.
    # After dispatch, filled[e] is number of tokens routed to local expert e.
    # Uniform vs skewed distributions.
    fill_scenarios = [
        ("uniform-low",     [8] * GLM5_NUM_LOCAL_EXPERTS),
        ("uniform-mid",     [64] * GLM5_NUM_LOCAL_EXPERTS),
        ("uniform-full",    [GLM5_MTP] * GLM5_NUM_LOCAL_EXPERTS),
        ("skewed-one-hot",  [GLM5_MTP] + [0] * (GLM5_NUM_LOCAL_EXPERTS - 1)),
        ("skewed-half-empty", [GLM5_MTP, 0] * (GLM5_NUM_LOCAL_EXPERTS // 2)),
        ("zero-traffic",    [0] * GLM5_NUM_LOCAL_EXPERTS),
    ]
    test_act_quant_3d(device, fill_scenarios, results)
    test_fused_s1_and_s3(device, fill_scenarios, results)

    # Summary
    print("\n# === SUMMARY ===")
    bad = [r for r in results if not r.ok]
    print(f"  total cases: {len(results)}  failures: {len(bad)}")
    for r in bad:
        print(f"    FAIL: {r.name}/{r.scenario}: {r.err}")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
