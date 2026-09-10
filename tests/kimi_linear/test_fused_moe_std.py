"""Standard-gate unit test: batchgen_kernels.triton.fused_moe_bf16 at
Kimi-Linear MoE shapes (E=256, H=2304, I=1024, top-8).

D2 resolution (EXECUTION_PLAN M4, POIS 2026-07-30): the standard BF16
envelope gate (kernel-validation.md) replaces the earlier ad-hoc 1.5e-2
max-abs tolerance:

    tol      = 1e-5 + 1.6e-2 * |ref|          (per element)
    PASS iff no NaN/Inf in the kernel output AND
             (0 elements exceed tol, or fail_fraction < 1e-4)

Reference: eager per-expert BF16 tensor-core matmuls (fp32 accumulation
inside the TC, TF32 disabled), fp32 SiLU and fp32 weighted top-k reduction —
the same data flow as the fused kernel's stages.

Cases: decode M in {64, 128, 256, 512}; prefill M in {8192, 16384, 32768}.
Inputs: realistic magnitudes (randn * 0.1); routing from a random sigmoid
gate with top-k renormalization (KimiMoEGate semantics, self-contained —
no batchgen model imports).

Run (GPU): python batchgen_kernels/tests/kimi_linear/test_fused_moe_std.py
"""

import sys

import torch
import torch.nn.functional as F

from batchgen_kernels.triton.fused_moe_bf16 import fused_moe_bf16

# BF16-TC reference discipline: no TF32 anywhere.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DTYPE = torch.bfloat16
DEVICE = "cuda"

# Kimi-Linear MoE shapes
E = 256
H = 2304
I = 1024
TOP_K = 8

# Standard BF16 envelope gate (kernel-validation.md / D2)
ATOL = 1e-5
RTOL = 1.6e-2
OUTLIER_FRAC = 1e-4

PASS = True


def report(name, ok, detail=""):
    global PASS
    PASS = PASS and ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def check_std_bf16(name, got, ref):
    """Standard BF16 envelope: tol = 1e-5 + 1.6e-2*|ref| per element;
    PASS = finite AND (0 fails or fail_frac < 1e-4)."""
    got32, ref32 = got.float(), ref.float()
    finite = bool(torch.isfinite(got32).all())
    diff = (got32 - ref32).abs()
    tol = ATOL + RTOL * ref32.abs()
    n_fail = int((diff > tol).sum())
    n_total = diff.numel()
    frac = n_fail / max(n_total, 1)
    ok = finite and (n_fail == 0 or frac < OUTLIER_FRAC)
    report(
        name, ok,
        f"max|Δ|={diff.max().item():.2e} fails={n_fail}/{n_total} "
        f"({frac:.1e})" + ("" if finite else " NON-FINITE OUTPUT"),
    )
    return ok


def make_weights(seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w13 = torch.randn(E, 2 * I, H, generator=g, device=DEVICE,
                      dtype=torch.float32).mul_(0.05).to(DTYPE)
    w2 = torch.randn(E, H, I, generator=g, device=DEVICE,
                     dtype=torch.float32).mul_(0.05).to(DTYPE)
    gate_w = torch.randn(E, H, generator=g, device=DEVICE,
                         dtype=torch.float32).mul_(0.25)
    return w13, w2, gate_w


def route(x, gate_w):
    """Sigmoid top-k routing with renormalization (KimiMoEGate semantics,
    routed_scaling_factor = 1)."""
    scores = (x.float() @ gate_w.t()).sigmoid()
    topk_weight, topk_idx = torch.topk(scores, TOP_K, dim=-1, sorted=False)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx, topk_weight


def ref_moe_bf16_tc(x, w13, w2, weights, ids):
    """Eager BF16-TC reference (fp32 accum inside the tensor core, fp32 SiLU,
    fp32 weighted top-k reduction) — mirrors the fused kernel's data flow:
    stage1 out stored bf16, silu_and_mul fp32-internal stored bf16, stage2
    stored bf16, weighted sum in fp32."""
    M = x.shape[0]
    out = torch.zeros(M, H, dtype=torch.float32, device=x.device)
    for e in range(E):
        mask = ids == e                      # (M, top_k)
        tok_sel = mask.any(dim=1)
        if not bool(tok_sel.any()):
            continue
        tok = tok_sel.nonzero(as_tuple=False).squeeze(-1)
        xe = x.index_select(0, tok)          # (n, H) bf16
        h = xe @ w13[e].t()                  # (n, 2I) bf16 TC
        gate, up = h.float().chunk(2, dim=-1)
        y = (F.silu(gate) * up).to(x.dtype) @ w2[e].t()  # (n, H) bf16 TC
        w = (weights * mask).sum(dim=1).index_select(0, tok)
        out.index_add_(0, tok, y.float() * w.float().unsqueeze(-1))
    return out.to(x.dtype)


def run_case(name, M, w13, w2, gate_w, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(M, H, generator=g, device=DEVICE,
                    dtype=torch.float32).mul_(0.1).to(DTYPE)
    topk_idx, topk_weight = route(x, gate_w)

    got = fused_moe_bf16(x, w13, w2, topk_weight, topk_idx)
    ref = ref_moe_bf16_tc(x, w13, w2, topk_weight, topk_idx)
    assert torch.isfinite(ref.float()).all(), f"{name}: reference not finite"
    check_std_bf16(name, got, ref)


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    w13, w2, gate_w = make_weights(seed=7)

    for M in (64, 128, 256, 512):
        run_case(f"decode M={M} E={E} H={H} I={I} top{TOP_K}",
                 M, w13, w2, gate_w, seed=100 + M)
    for M in (8192, 16384, 32768):
        run_case(f"prefill M={M} E={E} H={H} I={I} top{TOP_K}",
                 M, w13, w2, gate_w, seed=200 + M)

    print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
