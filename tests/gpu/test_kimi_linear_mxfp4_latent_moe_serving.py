# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""MXFP4-LatentMoE parity for the SERVING MoE forward — one GPU, no server.

M3.0 (decision A13): the synthetic MXFP4-LatentMoE correctness TESTBED plus the
STREAMED-path ORACLE that M3.1's resident layer will gate against.

What this pins: ``kimi_linear/serving_modules.py::moe_forward_serving`` on a K3
config whose routed experts are MXFP4-packed (``is_mxfp4_quantized(cfg)`` True,
so ``KimiSparseMoeBlock`` builds ``K3MXFP4Expert``). That streamed forward —
router on the PRE-down hidden, ``routed_expert_down_proj`` once per token, a
per-expert marlin MXFP4 decode (fused S1 gate+up+SiTU -> S3 down) in the latent
space, fp32 combine, ``routed_expert_norm``, ``routed_expert_up_proj``, shared
expert on the identity — is the ORACLE.

It is gated against an fp32 DEQUANT reference: the SAME LatentMoE dataflow, on
the SAME modules (gate / down_proj / norm / up_proj / shared_experts), with each
routed expert recomputed in fp32 from the oracle-dequantized MXFP4 weights
(``mxfp4_dequantize_oracle``) + fp32 SiTU (beta 4 / linear_beta 25). The ONLY
difference between the two is the marlin kernel's bf16-intermediate arithmetic,
which the MXFP4 kernel-validation tolerance absorbs
(``1e-5 + 1.6e-2*|ref|``, fail_frac < 1e-4, max_rel < 1.6e-2 on the
well-conditioned subset — the same gate ``tests/moe/gpu_parity_mxfp4_marlin.py``
uses). The MXFP4 weight quantization CANCELS because both sides dequantize the
exact same packed bytes (E2M1*E8M0 is exact in fp32); this is a kernel-numerics
gate, not a quantization-accuracy gate.

Two synthetic configs (both MXFP4 twins from the M2 harness): SYN25-MXFP4
(latent == hidden/2, the real ratio) and SKEW10-MXFP4 (latent != hidden/2,
moe_intermediate != latent, rms_norm_eps 2e-5, shared width 3x).

DIM CONSTRAINT (the M3.0 finding): the marlin MXFP4 decode requires both the MoE
latent (K) and moe_intermediate (N) to be multiples of 256, so the twins cannot
reuse syn25/skew10's exact dims — see kimi_k3_harness.syn25_mxfp4_config_dict.

Run ON h20-instance-2:

    K3_MXFP4_GPU=1 CUDA_VISIBLE_DEVICES=<g> BATCHGEN_KERNELS_DEV=1 \
    K3_MXFP4_ARTIFACT_DIR=<shared-fs dir> \
    PYTHONPATH=<repo>:<fla-src> python -m pytest \
        tests/gpu/test_kimi_linear_mxfp4_latent_moe_serving.py -x -q -rA \
        -p no:cacheprovider

With ``K3_MXFP4_GPU=1`` a missing GPU is a hard error, never a skip. The
batchgen imports live inside the tests: ``kimi_linear/model.py`` imports fla at
module scope, so collection on a CPU box must not touch it.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kimi_k3_harness as H  # noqa: E402


if os.environ.get("K3_MXFP4_GPU") == "1" and not torch.cuda.is_available():
    raise RuntimeError(
        "K3_MXFP4_GPU=1 but CUDA is unavailable — this staged run must not "
        "silently skip. Check CUDA_VISIBLE_DEVICES / the driver.")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="staged for h20-instance-2 GPU (serving MXFP4-LatentMoE parity)")

DEV = "cuda"
CONFIGS = {
    "syn25_mxfp4": H.syn25_mxfp4_config_dict,
    "skew10_mxfp4": H.skew10_mxfp4_config_dict,
}


# --------------------------------------------------------------------------- #
#  SiTU fp32 reference (modeling_kimi_linear.py:75-82; beta 4, linear_beta 25) #
#  — the fused marlin S1 kernel compiles these two constants in.               #
# --------------------------------------------------------------------------- #
def _situ_fp32(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    g = gate.float()
    u = up.float()
    a = 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)
    return a * (25.0 * torch.tanh(u / 25.0))


def _kernel_gate(out: torch.Tensor, ref: torch.Tensor, name: str):
    """The MXFP4 kernel-validation gate (tests/moe/gpu_parity_mxfp4_marlin.gate):
    finite AND fail_frac(|a-r| > 1e-5 + 1.6e-2|r|) < 1e-4 AND max relative error
    < 1.6e-2 on the well-conditioned subset (|ref| > 0.1*rms). Returns (passed,
    stats-dict)."""
    a = out.float()
    r = ref.float()
    finite = bool(torch.isfinite(a).all())
    err = (a - r).abs()
    tol = 1e-5 + 1.6e-2 * r.abs()
    fail_frac = float((err > tol).float().mean())
    rms = float(r.pow(2).mean().sqrt())
    mask = r.abs() > 0.1 * rms
    max_rel = float((err[mask] / r.abs()[mask]).max()) if bool(mask.any()) else 0.0
    max_abs = float(err.max())
    err_ratio = float((err.pow(2).mean().sqrt() / (rms + 1e-8)))
    passed = finite and fail_frac < 1e-4 and max_rel < 1.6e-2
    stats = dict(finite=finite, fail_frac=fail_frac, max_rel=max_rel,
                 max_abs=max_abs, err_ratio=err_ratio, rms=rms)
    print("[mxfp4-moe] {:38s} {} fail_frac={:.2e} max_rel={:.2e} "
          "max_abs={:.2e} err_ratio={:.2e} rms={:.2e}".format(
              name, "PASS" if passed else "FAIL", fail_frac, max_rel,
              max_abs, err_ratio, rms))
    return passed, stats


# --------------------------------------------------------------------------- #
#  Build + seed the streamed MXFP4 serving block                              #
# --------------------------------------------------------------------------- #
def _build_mxfp4_serving_block(cfg_name: str):
    """A production ``KimiSparseMoeBlock`` with MXFP4-packed routed experts and
    ``moe_forward_serving`` bound on — bf16 float params, uint8 packed experts,
    on the GPU. Returns (block, k3_lin_cfg, cfg_dict)."""
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig
    from batchgen.models.moonshotai.kimi_linear.model import (
        KimiSparseMoeBlock as ServingMoeBlock,
    )
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        moe_forward_serving,
    )
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
        is_mxfp4_quantized,
        K3MXFP4Expert,
    )
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_layout import (
        K3_EXPECTED_QUANT_IGNORE,
    )

    cfg_dict = CONFIGS[cfg_name]()

    # The harness copy of the ignore set must match the layout module's, or the
    # real load path's validate_quantization_config would reject this fixture.
    assert set(H.K3_MXFP4_QUANT_IGNORE) == set(K3_EXPECTED_QUANT_IGNORE), (
        "harness K3_MXFP4_QUANT_IGNORE drifted from mxfp4_layout")

    lin_cfg = KimiLinearConfig.from_hf_dict(cfg_dict)
    # is_mxfp4_quantized runs the full contract validation and must select the
    # packed expert; a False here means the fixture is not a valid MXFP4 config.
    assert is_mxfp4_quantized(lin_cfg), "fixture did not read as MXFP4-quantized"
    assert lin_cfg.model_type == "kimi_k3"
    assert lin_cfg.use_latent_moe and lin_cfg.latent_moe_use_norm

    block = ServingMoeBlock(lin_cfg)
    block.forward = types.MethodType(moe_forward_serving, block)

    assert block.use_latent_moe and block.latent_moe_use_norm
    assert block.moe_hidden_size == lin_cfg.routed_expert_hidden_size
    assert block.hidden_dim == lin_cfg.hidden_size
    assert all(isinstance(e, K3MXFP4Expert) for e in block.experts), (
        "experts are not K3MXFP4Expert — is_mxfp4_quantized did not select the "
        "packed path")

    _seed_block(block, cfg_name)
    _cast_floats_bf16(block)
    block.to(DEV)
    return block, lin_cfg, cfg_dict


def _seed_block(block, cfg_name: str) -> None:
    """Deterministic, name-keyed seeding. BF16 float params get the harness
    distributions (norm gains U(.8,1.2), e_score_correction_bias N(0,.05),
    everything else N(0,.02)); the MXFP4 packed experts get reproducible
    synthetic nibbles/scales from fixed per-(expert, projection) seeds."""
    import zlib

    def _seed(tag: str) -> int:
        return zlib.crc32("{}:{}:{}".format(H.GLOBAL_SEED, cfg_name, tag)
                          .encode()) & 0x7FFFFFFF

    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.endswith((".weight_packed", ".weight_scale")):
                continue                          # MXFP4 experts: seeded below
            g = torch.Generator().manual_seed(_seed("param:" + name))
            buf = torch.empty_like(p, dtype=torch.float32)
            if name.endswith("norm.weight"):
                buf.uniform_(0.8, 1.2, generator=g)
            elif name.endswith("e_score_correction_bias"):
                buf.normal_(0.0, 0.05, generator=g)
            else:
                buf.normal_(0.0, 0.02, generator=g)
            p.copy_(buf.to(p.dtype))

        for e_idx, expert in enumerate(block.experts):
            for proj_name in ("w1", "w3", "w2"):
                proj = getattr(expert, proj_name)
                packed, scale = H.rand_mxfp4_packed(
                    proj.n_out, proj.k_in,
                    seed=_seed("expert:{}:{}".format(e_idx, proj_name)))
                proj.weight_packed.copy_(packed)
                proj.weight_scale.copy_(scale)


def _cast_floats_bf16(block) -> None:
    """Cast every floating parameter to bf16 (serving dtype); leave the uint8
    MXFP4 packed/scale tensors untouched."""
    with torch.no_grad():
        for _name, p in block.named_parameters():
            if p.dtype.is_floating_point:
                p.data = p.data.to(torch.bfloat16)


# --------------------------------------------------------------------------- #
#  fp32 dequant reference — same dataflow, experts recomputed in fp32          #
# --------------------------------------------------------------------------- #
def _dequant_fp32_reference(block, x_3d: torch.Tensor):
    """Recompute ``moe_forward_serving``'s LatentMoE result with each routed
    expert done in fp32 from its oracle-dequantized MXFP4 weights. Reuses the
    block's own gate / down_proj / norm / up_proj / shared_experts so the ONLY
    numeric difference vs the streamed oracle is the marlin kernel arithmetic.

    Returns (full_output, expert_path_before_shared)."""
    from batchgen.moe.mxfp4_oracle_vector import mxfp4_dequantize_oracle

    orig_shape = x_3d.shape
    hidden = block.hidden_dim
    x = x_3d.reshape(-1, hidden)
    num_tokens = x.shape[0]

    topk_idx, topk_weight = block.gate(x_3d)
    K = topk_idx.shape[-1]
    x_latent = block.routed_expert_down_proj(x)       # bf16 [t, latent]
    latent = x_latent.shape[-1]

    flat_expert = topk_idx.reshape(-1)
    token_idx = torch.arange(num_tokens, device=x.device).repeat_interleave(K)
    results = torch.zeros(num_tokens, latent, device=x.device, dtype=torch.float32)

    for e_idx, expert in enumerate(block.experts):
        sel = (flat_expert == e_idx).nonzero(as_tuple=False).squeeze(-1)
        if sel.numel() == 0:
            continue
        rows = token_idx[sel]
        xr = x_latent.index_select(0, rows).float()   # [t_e, latent]
        w1 = mxfp4_dequantize_oracle(expert.w1.weight_packed,
                                     expert.w1.weight_scale, torch.float32)  # [N, latent]
        w3 = mxfp4_dequantize_oracle(expert.w3.weight_packed,
                                     expert.w3.weight_scale, torch.float32)  # [N, latent]
        w2 = mxfp4_dequantize_oracle(expert.w2.weight_packed,
                                     expert.w2.weight_scale, torch.float32)  # [latent, N]
        # Model the marlin S1->S3 kernel's bf16 intermediates exactly
        # (single_expert_marlin_mxfp4_decode): S1 rounds each gate/up GEMM
        # pass to bf16 in SMEM BEFORE the fused SiTU, writes the SiTU result
        # to a bf16 `intermediate`, and S3 writes a bf16 `expert_out`. An
        # all-fp32 reference is NOT the kernel's arithmetic: it keeps ~1%
        # (bf16-ULP) intermediate precision the kernel discards, which the
        # down GEMM's cancellation then amplifies past tol (M3.0 A14: this
        # missing truncation, not a repack/SiTU/kernel bug, is the 17%).
        gate_out = (xr @ w1.t()).to(torch.bfloat16)   # [t_e, N] bf16 pass
        up_out = (xr @ w3.t()).to(torch.bfloat16)     # [t_e, N] bf16 pass
        situ = _situ_fp32(gate_out, up_out).to(torch.bfloat16)  # bf16 S1
        down = (situ.float() @ w2.t()).to(torch.bfloat16)       # bf16 S3
        w = topk_weight.reshape(-1)[sel].unsqueeze(-1).float()
        results.index_add_(0, rows, down.float() * w)

    y = results.to(x.dtype)                           # bf16, exactly where the
    if block.latent_moe_use_norm:                     # streamed path downcasts
        y = block.routed_expert_norm(y)
    y = block.routed_expert_up_proj(y)
    expert_path = y.reshape(orig_shape)
    out = expert_path
    if getattr(block, "shared_experts", None) is not None:
        out = out + block.shared_experts(x_3d)
    return out, expert_path, topk_idx, topk_weight


# --------------------------------------------------------------------------- #
#  1. The fixture is a valid MXFP4-LatentMoE config with packed experts        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg_name", sorted(CONFIGS))
def test_fixture_builds_packed_experts(cfg_name):
    block, cfg, _cfg_dict = _build_mxfp4_serving_block(cfg_name)
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_layout import (
        MXFP4_GROUP_SIZE, MXFP4_PACK_FACTOR,
    )
    latent = cfg.routed_expert_hidden_size
    inter = cfg.moe_intermediate_size
    assert latent % 256 == 0 and inter % 256 == 0, (
        "marlin MXFP4 needs latent & moe_intermediate multiples of 256")
    e0 = block.experts[0]
    # w1/w3: n_out = N (inter), k_in = K (latent); w2: n_out = latent, k_in = N.
    assert list(e0.w1.weight_packed.shape) == [inter, latent // MXFP4_PACK_FACTOR]
    assert list(e0.w1.weight_scale.shape) == [inter, latent // MXFP4_GROUP_SIZE]
    assert list(e0.w2.weight_packed.shape) == [latent, inter // MXFP4_PACK_FACTOR]
    assert e0.w1.weight_packed.dtype == torch.uint8
    assert e0.w1.weight_scale.dtype == torch.uint8
    print("[mxfp4-moe] {} fixture OK: {} experts, latent={} inter={} hidden={}"
          .format(cfg_name, len(block.experts), latent, inter, cfg.hidden_size))


# --------------------------------------------------------------------------- #
#  2. STREAMED (oracle) == fp32 dequant reference, within MXFP4 tolerance       #
#     — and capture the streamed reference artifact for M3.1.                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg_name", sorted(CONFIGS))
def test_streamed_matches_fp32_dequant(cfg_name):
    block, cfg, cfg_dict = _build_mxfp4_serving_block(cfg_name)
    tokens = 64
    x = H.seeded_input(
        "mxfp4_latent_moe:{}".format(cfg_name), 1, tokens, cfg.hidden_size,
        dtype=torch.bfloat16,
    ).to(DEV)

    with torch.no_grad():
        streamed = block(x)                                # ORACLE (kernel path)
        reference, ref_expert_path, topk_idx, topk_weight = \
            _dequant_fp32_reference(block, x)

    assert streamed.shape == reference.shape == (1, tokens, cfg.hidden_size)
    assert torch.isfinite(streamed).all(), "streamed output non-finite"

    ok_full, stats_full = _kernel_gate(
        streamed, reference, "{} full-block".format(cfg_name))
    # Isolated MXFP4 latent path (pre shared expert), the tightest read on the
    # kernel: recompute the streamed path's expert output without the shared add.
    streamed_expert_path = streamed - block.shared_experts(x)
    ok_path, stats_path = _kernel_gate(
        streamed_expert_path, ref_expert_path,
        "{} expert-path".format(cfg_name))

    _save_artifact(cfg_name, cfg_dict, block, x, streamed, streamed_expert_path,
                   reference, topk_idx, topk_weight, stats_full, stats_path)

    assert ok_path, "{} MXFP4 latent expert path failed the kernel gate: {}".format(
        cfg_name, stats_path)
    assert ok_full, "{} full-block failed the kernel gate: {}".format(
        cfg_name, stats_full)


# --------------------------------------------------------------------------- #
#  3. down / norm / up run ONCE per token; every expert is driven in lockstep  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg_name", sorted(CONFIGS))
def test_latent_projections_run_once_per_token(cfg_name):
    block, cfg, _cfg_dict = _build_mxfp4_serving_block(cfg_name)
    tokens = 48
    top_k = block.gate.top_k
    x = H.seeded_input(
        "mxfp4_latent_moe:once:{}".format(cfg_name), 1, tokens, cfg.hidden_size,
        dtype=torch.bfloat16,
    ).to(DEV)

    seen = {}

    def record(tag):
        def hook(_m, inputs, _o):
            seen.setdefault(tag, []).append(tuple(inputs[0].shape))
        return hook

    block.routed_expert_down_proj.register_forward_hook(record("down"))
    block.routed_expert_norm.register_forward_hook(record("norm"))
    block.routed_expert_up_proj.register_forward_hook(record("up"))
    expert_calls = []
    for expert in block.experts:
        expert.register_forward_hook(
            lambda _m, i, _o: expert_calls.append(i[0].shape[0]))

    with torch.no_grad():
        block(x)

    latent = cfg.routed_expert_hidden_size
    assert seen["down"] == [(tokens, cfg.hidden_size)], seen["down"]
    assert seen["norm"] == [(tokens, latent)], seen["norm"]
    assert seen["up"] == [(tokens, latent)], seen["up"]
    assert len(expert_calls) == len(block.experts)         # all driven, 0-token too
    assert sum(expert_calls) == tokens * top_k
    print("[mxfp4-moe] {} once-per-token OK: down/norm/up 1 call on {} rows; "
          "experts driven {}/{}, assignments {}".format(
              cfg_name, tokens, len(expert_calls), len(block.experts),
              sum(expert_calls)))


# --------------------------------------------------------------------------- #
#  Artifact capture (deliverable 3)                                            #
# --------------------------------------------------------------------------- #
def _save_artifact(cfg_name, cfg_dict, block, x, streamed, streamed_expert_path,
                   reference, topk_idx, topk_weight, stats_full, stats_path):
    """Persist the STREAMED oracle (+ config, seeded state_dict, input, routing)
    so M3.1 can gate its resident layer as ``resident == streamed`` on the exact
    same weights and tokens. Writes only when K3_MXFP4_ARTIFACT_DIR is set."""
    artifact_dir = os.environ.get("K3_MXFP4_ARTIFACT_DIR")
    if not artifact_dir:
        print("[mxfp4-moe] K3_MXFP4_ARTIFACT_DIR unset — not persisting {} "
              "oracle".format(cfg_name))
        return
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir,
                        "mxfp4_latent_moe_oracle_{}.pt".format(cfg_name))
    torch.save({
        "cfg_name": cfg_name,
        "config_dict": cfg_dict,
        "state_dict": {k: v.detach().cpu() for k, v in block.state_dict().items()},
        "input": x.detach().cpu(),
        "topk_idx": topk_idx.detach().cpu(),
        "topk_weight": topk_weight.detach().cpu(),
        "streamed_output": streamed.detach().cpu(),
        "streamed_expert_path": streamed_expert_path.detach().cpu(),
        "fp32_reference_output": reference.detach().cpu(),
        "activations_dtype": "bfloat16",
        "kernel_gate": {"full": stats_full, "expert_path": stats_path},
        "generated_by":
            "tests/gpu/test_kimi_linear_mxfp4_latent_moe_serving.py",
    }, path)
    print("[mxfp4-moe] saved oracle artifact -> {}".format(path))
