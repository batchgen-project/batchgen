# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""M2a — isolated bit-exact parity for head-parallel (TP) KDA decode.

Single process, GPU only (fla's KDA kernels are CUDA). This validates the
DECOMPOSITION that PSM head-sharding + the o_proj all_reduce implement:

  reference : the full 96-head ``kda_decode_serving`` -> o_full (post o_proj),
              and the pre-o_proj per-head output o_pre_full (the o_norm output,
              (bsz, 96*128)).
  candidate : for each shard r in range(G), slice EVERY KDA weight to heads
              [r*Hl:(r+1)*Hl] (o_proj by COLUMNS), set num_heads=Hl, and run
              ``kda_decode_serving`` with the head-sliced conv/recurrent state
              (SKIP the all_reduce -- the module is unstamped so attn_tp_size
              defaults to 1) -> partial_r (post o_proj) and o_pre_r
              ((bsz, Hl*128)).

Two invariants are measured for G in {2,4,8}:
  max_pre  = max| concat_r(o_pre_r) - o_pre_full |   (head-independence: the
             projection / conv / recurrence / o_norm sharding is bit-exact
             per head-slice -- P0.6). MUST be 0.
  max_post = max| sum_r partial_r - o_full |          (the o_proj partial-sum
             the all_reduce performs in production).

MEASURED (h20-instance-1, GPU 0, 2026-08-13): max_pre == 0 for every G and
both gate ranks (head-independence is bit-exact); max_post == 3.906e-03 ==
2**-8 uniformly, i.e. EXACTLY one bf16 ULP at O(1) output magnitude. Summing G
independently-bf16-rounded o_proj partials differs from a single fused bf16
matmul by the minimal bf16 quantum -- the standard, unavoidable rounding of a
bf16 row-parallel reduction, NOT a sharding error (proven by max_pre == 0).
The spec's literal ``max_post == 0`` is therefore not achievable post-o_proj;
the gate below is max_pre == 0 (STRICT bit-exact) + max_post <= one-bf16-ULP
band. A real head-mapping bug lands O(1) here (see non-vacuity, ~1.05), 100x+
the band, so the tolerance still catches it.

Plus a NON-VACUITY control: rebuilding the shards with the WRONG o_proj column
block (rotated by one shard) must make max_post > 0, i.e. the test is actually
sensitive to head assignment.

Launch (from repo root ON h20-instance-1):
    K3_GPU_STAGE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest \
        tests/gpu/test_kimi_k3_kda_head_parallel_parity.py -x -q -rA -s
"""

from __future__ import annotations

import copy
import os
import types

import pytest
import torch

if os.environ.get("K3_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError(
        "K3_GPU_STAGE=1 but CUDA is unavailable — this staged run must not "
        "silently skip. Check CUDA_VISIBLE_DEVICES / the driver on instance-1.")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="staged for h20-instance-1 GPU 0 (fla KDA kernels are CUDA)")

KDA_NUM_HEADS = 96
HEAD_DIM = 128
HIDDEN = 512
CONV_SIZE = 4          # W; conv pools hold W-1 raw inputs per slot
BSZ = 4
RMS_EPS = 1e-5


def _cfg(full_rank_gate: bool):
    # KimiKDAAttention.__init__ only reads these attributes.
    return types.SimpleNamespace(
        hidden_size=HIDDEN,
        kda_conv_size=CONV_SIZE,
        kda_head_dim=HEAD_DIM,
        kda_num_heads=KDA_NUM_HEADS,
        kda_use_full_rank_gate=full_rank_gate,
        kda_gate_lower_bound=-5.0,
        rms_norm_eps=RMS_EPS,
    )


def _bind_serving(mod):
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        kda_decode_serving,
    )
    mod.kda_decode_serving = types.MethodType(kda_decode_serving, mod)
    return mod


def _dtype_fix(mod):
    """bf16 for every projection / conv / norm weight (the serving dtype);
    fp32 for A_log and dt_bias (the kernel's gate params)."""
    for name, p in list(mod.named_parameters()):
        if name in ("A_log", "dt_bias"):
            p.data = p.data.float()
        else:
            p.data = p.data.to(torch.bfloat16)
    return mod


def _build_full(full_rank_gate):
    from batchgen.models.moonshotai.kimi_linear.model import KimiKDAAttention

    cfg = _cfg(full_rank_gate)
    mod = KimiKDAAttention(cfg, 0)
    with torch.no_grad():
        # dt_bias is torch.empty in __init__ (uninitialized): fill it so no
        # NaN poisons the recurrence (NaN != NaN would break bit-equality).
        mod.dt_bias.copy_(torch.randn_like(mod.dt_bias) * 0.1)
    mod = mod.cuda()
    _dtype_fix(mod)
    return _bind_serving(mod), cfg


def _shard_slice(name, src, r, Hl, G, head_dim, o_col_block=None):
    lo, hi = r * Hl, (r + 1) * Hl
    rlo, rhi = lo * head_dim, hi * head_dim
    ocol = r if o_col_block is None else o_col_block
    oclo, ochi = ocol * Hl * head_dim, (ocol + 1) * Hl * head_dim
    base = name.split(".")[0]
    if base in ("f_a_proj", "g_a_proj", "o_norm"):
        return src
    if base == "o_proj":
        return src[:, oclo:ochi]
    if base in ("A_log", "b_proj"):
        return src[lo:hi]
    return src[rlo:rhi]


def _build_shard(full, cfg_full, r, G, o_col_block=None):
    from batchgen.models.moonshotai.kimi_linear.model import KimiKDAAttention

    Hl = cfg_full.kda_num_heads // G
    cfg_s = copy.copy(cfg_full)
    cfg_s.kda_num_heads = Hl
    smod = KimiKDAAttention(cfg_s, 0).cuda()
    src = dict(full.named_parameters())
    with torch.no_grad():
        for name, p in smod.named_parameters():
            sl = _shard_slice(name, src[name], r, Hl, G, cfg_full.kda_head_dim,
                              o_col_block=o_col_block)
            p.data = sl.detach().clone().contiguous()
    smod.num_heads = smod.num_k_heads = Hl
    return _bind_serving(smod)


class _State:
    """Minimal KDALayerState stand-in (kda_decode_serving reads exactly these)."""

    def __init__(self, conv_q, conv_k, conv_v, recurrent_pool, slots):
        self.conv_q = conv_q
        self.conv_k = conv_k
        self.conv_v = conv_v
        self.recurrent_pool = recurrent_pool
        self.cur_decode_slots = slots


def _capture_hook(store):
    def hook(_m, inp):
        store["pre"] = inp[0].detach().clone()
    return hook


def _run(mod, hidden, conv_q, conv_k, conv_v, recurrent, slots):
    # Clone state: causal_conv1d_update / fused_recurrent_kda_fwd mutate the
    # pools in place, so each run must start from the shared random base.
    store = {}
    h = mod.o_proj.register_forward_pre_hook(_capture_hook(store))
    try:
        state = _State(conv_q.clone(), conv_k.clone(), conv_v.clone(),
                       recurrent.clone(), slots)
        out = mod.kda_decode_serving(hidden, state)   # (bsz, 1, hidden)
    finally:
        h.remove()
    return out, store["pre"]


def test_head_parallel_kda_bit_exact():
    torch.manual_seed(0)
    device = "cuda"
    slots = torch.arange(BSZ, dtype=torch.int32, device=device)

    results = []          # (full_rank_gate, G, max_pre, max_post)
    nonvac = None

    for full_rank_gate in (False, True):
        full, cfg = _build_full(full_rank_gate)
        hidden = torch.randn(BSZ, 1, HIDDEN, device=device, dtype=torch.bfloat16)

        # Shared random decode state (whole-model, 96 heads). Slices of this
        # feed each shard so every head sees an identical initial state.
        proj = KDA_NUM_HEADS * HEAD_DIM
        rec = torch.randn(BSZ, KDA_NUM_HEADS, HEAD_DIM, HEAD_DIM,
                          device=device, dtype=torch.float32) * 0.1
        cq = torch.randn(BSZ, proj, CONV_SIZE - 1, device=device,
                         dtype=torch.bfloat16) * 0.1
        ck = torch.randn_like(cq) * 1.0
        cv = torch.randn_like(cq) * 1.0

        o_full, pre_full = _run(full, hidden, cq, ck, cv, rec, slots)

        for G in (2, 4, 8):
            Hl = KDA_NUM_HEADS // G
            partials, pre_parts = [], []
            for r in range(G):
                sh = _build_shard(full, cfg, r, G)
                lo, hi = r * Hl, (r + 1) * Hl
                rlo, rhi = lo * HEAD_DIM, hi * HEAD_DIM
                p_r, pre_r = _run(
                    sh, hidden,
                    cq[:, rlo:rhi].contiguous(),
                    ck[:, rlo:rhi].contiguous(),
                    cv[:, rlo:rhi].contiguous(),
                    rec[:, lo:hi].contiguous(),
                    slots,
                )
                partials.append(p_r)
                pre_parts.append(pre_r)
            o_tp = torch.stack(partials, 0).sum(0)
            pre_cat = torch.cat(pre_parts, dim=-1)
            max_pre = (pre_cat.float() - pre_full.float()).abs().max().item()
            max_post = (o_tp.float() - o_full.float()).abs().max().item()
            results.append((full_rank_gate, G, max_pre, max_post))

        # NON-VACUITY (once): rotate each shard's o_proj columns by one block.
        if nonvac is None:
            G = 2
            Hl = KDA_NUM_HEADS // G
            wrong = []
            for r in range(G):
                sh = _build_shard(full, cfg, r, G, o_col_block=(r + 1) % G)
                lo, hi = r * Hl, (r + 1) * Hl
                rlo, rhi = lo * HEAD_DIM, hi * HEAD_DIM
                p_r, _ = _run(
                    sh, hidden,
                    cq[:, rlo:rhi].contiguous(),
                    ck[:, rlo:rhi].contiguous(),
                    cv[:, rlo:rhi].contiguous(),
                    rec[:, lo:hi].contiguous(),
                    slots,
                )
                wrong.append(p_r)
            o_wrong = torch.stack(wrong, 0).sum(0)
            nonvac = (o_wrong.float() - o_full.float()).abs().max().item()

    print("\n=== HEAD-PARALLEL KDA PARITY (decode, isolated single-process) ===")
    for fr, G, mpre, mpost in results:
        print(f"  full_rank_gate={fr!s:5} G={G}: "
              f"max_pre={mpre:.3e}  max_post={mpost:.3e}")
    print(f"  NON-VACUITY (wrong o_proj col block, G=2): "
          f"max|d|={nonvac:.3e}  (must be > 0)")
    print("=" * 66)

    # STRICT: the head-parallel decomposition through o_norm is bit-exact.
    for fr, G, mpre, _ in results:
        assert mpre == 0.0, (
            f"pre-o_proj head-slice NOT bit-exact (full_rank_gate={fr}, G={G}): "
            f"max_pre={mpre} — head-independence violated")
    assert nonvac > 0.0, (
        f"NON-VACUITY FAILED: wrong o_proj column block still matched "
        f"(max|d|={nonvac}); the test is not sensitive to head assignment")
    # o_proj row-parallel reduction: one bf16 ULP at O(1) (measured 2**-8).
    # Band = 2**-6 (a few ULP headroom); a real head-mapping bug is O(1) here.
    BF16_OPROJ_BAND = 2.0 ** -6
    for fr, G, _, mpost in results:
        assert mpost <= BF16_OPROJ_BAND, (
            f"o_tp vs o_full exceeds the bf16 o_proj-reduction band "
            f"(full_rank_gate={fr}, G={G}): max_post={mpost} > {BF16_OPROJ_BAND}")
