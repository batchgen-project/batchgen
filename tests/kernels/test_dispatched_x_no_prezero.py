"""Unit test: confirm `dispatched_x.zero_()` before `dispatch_scatter_3d` is
redundant on the GLM-5 3D-MoE decode path.

The pipeline is: dispatch_scatter_3d → act_quant_3d → grouped_fp8_blockwise_*.
Both `dispatch_scatter_3d` (C++ kernel, only writes [0, count) per expert) and
`act_quant_3d` (CUDA kernel at batchgen_kernels/src/moe/fp8_blockwise/
fp8_blockwise_ops.cu:44, `if (token >= valid_tokens) return;`) respect per-
expert seqlens. Padded rows in `dispatched_x` are never read or written by
the hot path.

This test proves that — filling `dispatched_x` with a garbage sentinel before
dispatch does not change:
    (a) the rows [0, count) of dispatched_x written by scatter, and
    (b) the FP8 quant output at rows [0, count) from act_quant_3d.

Run (on H20): `pytest tests/kernels/test_dispatched_x_no_prezero.py -x -s`
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires Hopper (sm_90a)",
)


def _setup():
    """Small deterministic problem matching the GLM-5 3D-MoE layout."""
    from batchgen.moe.dispatch_scatter_3d import dispatch_scatter_3d
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d

    torch.manual_seed(0)
    device = torch.device("cuda")

    E_local = 4         # local experts on this rank
    mtp = 64            # max_tokens_padded (must be multiple of act_quant BLOCK)
    H = 1024            # hidden; multiple of 128 for block scale
    topk = 4
    G = 96              # total tokens routed globally

    all_tokens = (torch.randn(G, H, dtype=torch.bfloat16, device=device) * 0.5).contiguous()

    # Routing: each token picks `topk` experts in [0, 2*E_local); half are non-local.
    topk_idx = torch.randint(
        0, 2 * E_local, (G, topk), dtype=torch.int32, device=device)

    expert_counts = torch.empty(E_local, dtype=torch.int32, device=device)
    expert_counters = torch.empty(E_local, dtype=torch.int32, device=device)
    topk_pos = torch.empty(G * topk, dtype=torch.int32, device=device)

    return dict(
        device=device, E_local=E_local, mtp=mtp, H=H, topk=topk, G=G,
        all_tokens=all_tokens, topk_idx=topk_idx,
        expert_counts=expert_counts, expert_counters=expert_counters,
        topk_pos=topk_pos,
        dispatch_scatter_3d=dispatch_scatter_3d, act_quant_3d=act_quant_3d,
    )


def _run_dispatch(ctx, dispatched_x):
    return ctx["dispatch_scatter_3d"](
        ctx["all_tokens"], ctx["topk_idx"], dispatched_x,
        0, ctx["E_local"], ctx["mtp"],
        ctx["expert_counts"], ctx["expert_counters"],
        ctx["topk_pos"],
    )


def test_dispatch_scatter_only_writes_valid_rows():
    """dispatch_scatter_3d must not touch rows [count, mtp) per expert."""
    ctx = _setup()
    E, mtp, H = ctx["E_local"], ctx["mtp"], ctx["H"]

    SENTINEL = -1234.5
    dispatched_x = torch.full(
        (E * mtp, H), SENTINEL, dtype=torch.bfloat16, device=ctx["device"])
    counts, _ = _run_dispatch(ctx, dispatched_x)
    torch.cuda.synchronize()

    counts_cpu = counts.cpu().tolist()
    dx_3d = dispatched_x.view(E, mtp, H)
    for e in range(E):
        c = counts_cpu[e]
        if c < mtp:
            tail = dx_3d[e, c:mtp]
            assert torch.all(tail == torch.tensor(SENTINEL, dtype=torch.bfloat16, device=ctx["device"])), (
                f"expert {e}: scatter wrote past count={c} into tail (sentinel clobbered)")


def test_act_quant_3d_ignores_padded_rows():
    """act_quant_3d output at rows [0, count) must match regardless of
    what garbage lives in the padded rows."""
    ctx = _setup()
    E, mtp, H = ctx["E_local"], ctx["mtp"], ctx["H"]

    # Reference: zero-init dispatched_x, then scatter → act_quant_3d.
    dx_ref = torch.zeros(E * mtp, H, dtype=torch.bfloat16, device=ctx["device"])
    counts_ref, _ = _run_dispatch(ctx, dx_ref)
    torch.cuda.synchronize()
    x_ref_3d = dx_ref.view(E, mtp, H)
    q_ref, s_ref = ctx["act_quant_3d"](x_ref_3d, counts_ref)

    # Under test: garbage-init dispatched_x, scatter writes only valid rows,
    # padded rows remain garbage; act_quant_3d runs over the whole tensor.
    dx_gar = torch.full((E * mtp, H), 12345.0, dtype=torch.bfloat16, device=ctx["device"])
    # Include NaN / Inf in the garbage to provoke propagation bugs.
    dx_gar.view(E, mtp, H)[:, mtp // 2 + 1, :] = float('nan')
    dx_gar.view(E, mtp, H)[:, mtp // 2 + 2, :] = float('inf')
    counts_gar, _ = _run_dispatch(ctx, dx_gar)
    torch.cuda.synchronize()
    assert torch.equal(counts_ref, counts_gar), "routing non-determinism — test setup bug"
    x_gar_3d = dx_gar.view(E, mtp, H)
    q_gar, s_gar = ctx["act_quant_3d"](x_gar_3d, counts_gar)
    torch.cuda.synchronize()

    counts_cpu = counts_ref.cpu().tolist()
    for e in range(E):
        c = counts_cpu[e]
        if c == 0:
            continue
        # FP8 quant output bit-exact on valid rows.
        assert torch.equal(q_ref[e, :c], q_gar[e, :c]), \
            f"expert {e}: FP8 quant differs on valid rows (count={c}) — padded garbage leaked in"
        assert torch.equal(s_ref[e, :c], s_gar[e, :c]), \
            f"expert {e}: FP8 scale differs on valid rows (count={c})"
