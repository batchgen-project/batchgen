"""Standard-gate unit test: batchgen_kernels.conv1d at Kimi-Linear KDA shapes
(dim=4096, kernel width W=4, pooled state layout (num_slots, dim, W-1)).

D2 resolution (EXECUTION_PLAN M4, POIS 2026-07-30): the standard BF16
envelope gate (kernel-validation.md) replaces the earlier ad-hoc 1.2e-2
max-abs tolerance:

    tol      = 1e-5 + 1.6e-2 * |ref|          (per element)
    PASS iff no NaN/Inf in the kernel output AND
             (0 elements exceed tol, or fail_fraction < 1e-4)

Reference: naive causal conv accumulated in fp32 over the BF16 inputs and
weights, SiLU in fp32, single cast to BF16 at store — the same data flow as
the CUDA kernel. TF32 disabled (hygiene; no matmuls involved).

Cases (bias=None, SiLU on — the Kimi-Linear KDA configuration):
  1. varlen prefill (packed cu_seqlens, seq lens incl. < W) + pooled
     final-state write at scattered cache_indices
  2. chunked-prefill continuation (has_initial_state) vs single-shot oracle
  3. 8-step causal_conv1d_update decode chain over the pooled state,
     continuing case-1's sequences (prefill -> decode state carry)
Inputs: realistic magnitudes (randn * 0.1).

Run (GPU): python batchgen_kernels/tests/kimi_linear/test_conv1d_std.py
(BATCHGEN_KERNELS_DEV=1 needed only when running from the source tree
without the rebuilt AOT wheel.)
"""

import sys

import torch
import torch.nn.functional as F

from batchgen_kernels.conv1d import causal_conv1d_fwd, causal_conv1d_update

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DTYPE = torch.bfloat16
DEVICE = "cuda"

# Kimi-Linear KDA conv shapes
DIM = 4096
W = 4  # kernel width; conv state width = W - 1

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


# ── fp32-accumulation reference (BF16 in / BF16 out, kernel data flow) ───────
def ref_conv_seq(x, weight, initial_state=None):
    """One sequence: x (T, DIM) bf16 -> (y (T, DIM) bf16, final (DIM, W-1)
    bf16). fp32 accumulation over the W taps, fp32 SiLU, cast at store."""
    T = x.shape[0]
    x32 = x.float()
    w32 = weight.float()  # (DIM, W)
    if initial_state is not None:
        pad = initial_state.float().t()  # (W-1, DIM)
    else:
        pad = torch.zeros(W - 1, DIM, dtype=torch.float32, device=x.device)
    x_ext = torch.cat([pad, x32], dim=0)  # (T + W - 1, DIM)
    y = torch.zeros(T, DIM, dtype=torch.float32, device=x.device)
    for w in range(W):
        y += x_ext[w:w + T] * w32[:, w]
    y = F.silu(y)
    final = x_ext[-(W - 1):].t()  # (DIM, W-1) — last W-1 raw inputs
    return y.to(DTYPE), final.to(DTYPE)


def make_inputs(seed, lens):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    xs = [
        torch.randn(t, DIM, generator=g, device=DEVICE,
                    dtype=torch.float32).mul_(0.1).to(DTYPE)
        for t in lens
    ]
    weight = torch.randn(DIM, W, generator=g, device=DEVICE,
                         dtype=torch.float32).mul_(0.1).to(DTYPE)
    return xs, weight


def cu_seqlens_of(lens):
    cu = [0]
    for t in lens:
        cu.append(cu[-1] + t)
    return torch.tensor(cu, dtype=torch.int32, device=DEVICE)


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    # ── case 1: varlen prefill + pooled state write ──────────────────────────
    lens = [37, 512, 1, 3]  # incl. sequences shorter than W
    slots = torch.tensor([5, 2, 7, 0], dtype=torch.int32, device=DEVICE)
    xs, weight = make_inputs(seed=11, lens=lens)
    pool = torch.zeros(8, DIM, W - 1, dtype=DTYPE, device=DEVICE)
    cu = cu_seqlens_of(lens)

    y = causal_conv1d_fwd(
        torch.cat(xs, dim=0), weight, bias=None,
        conv_states=pool, query_start_loc=cu, cache_indices=slots,
        has_initial_state=None,
    )
    refs = [ref_conv_seq(x, weight) for x in xs]
    for i, t in enumerate(lens):
        y_got = y[cu[i]:cu[i + 1]]
        check_std_bf16(f"varlen prefill seq{i} (T={t}) y", y_got, refs[i][0])
        check_std_bf16(f"varlen prefill seq{i} (T={t}) final state",
                       pool[slots[i]], refs[i][1])

    # ── case 2: chunked-prefill continuation vs single-shot oracle ───────────
    full_len = 200
    split = 77
    (x_full,), weight2 = make_inputs(seed=23, lens=[full_len])
    pool2 = torch.zeros(4, DIM, W - 1, dtype=DTYPE, device=DEVICE)
    slot2 = torch.tensor([1], dtype=torch.int32, device=DEVICE)

    y1 = causal_conv1d_fwd(
        x_full[:split], weight2, bias=None,
        conv_states=pool2, query_start_loc=cu_seqlens_of([split]),
        cache_indices=slot2,
        has_initial_state=torch.tensor([False], device=DEVICE),
    )
    y2 = causal_conv1d_fwd(
        x_full[split:], weight2, bias=None,
        conv_states=pool2, query_start_loc=cu_seqlens_of([full_len - split]),
        cache_indices=slot2,
        has_initial_state=torch.tensor([True], device=DEVICE),
    )
    y_ref, final_ref = ref_conv_seq(x_full, weight2)
    check_std_bf16("chunked prefill chunk1 y", y1, y_ref[:split])
    check_std_bf16("chunked prefill chunk2 y (state carry)",
                   y2, y_ref[split:])
    check_std_bf16("chunked prefill final state", pool2[1], final_ref)

    # ── case 3: decode update chain over the pooled state ────────────────────
    # Continue case-1's 4 sequences for 8 single-token steps.
    n_steps = 8
    n_seqs = len(lens)
    g = torch.Generator(device=DEVICE).manual_seed(31)
    # Reference tails: last W-1 raw inputs per sequence (bf16, like the pool).
    tails = torch.stack([refs[i][1] for i in range(n_seqs)])  # (N, DIM, W-1)
    for step in range(n_steps):
        x_t = torch.randn(n_seqs, DIM, generator=g, device=DEVICE,
                          dtype=torch.float32).mul_(0.1).to(DTYPE)
        y_t = causal_conv1d_update(
            x_t, pool, weight, bias=None, conv_state_indices=slots,
        )
        window = torch.cat(
            [tails.float(), x_t.float().unsqueeze(-1)], dim=-1
        )  # (N, DIM, W)
        y_ref_t = F.silu((window * weight.float()).sum(dim=-1)).to(DTYPE)
        check_std_bf16(f"decode step{step} y (batch of {n_seqs})", y_t, y_ref_t)
        tails = window[:, :, 1:].to(DTYPE)  # roll the state windows
    check_std_bf16("decode final pooled states", pool[slots.long()], tails)

    print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
