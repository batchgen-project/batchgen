# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""LatentMoE parity for the SERVING MoE forward — one GPU, no server.

What this pins: ``kimi_linear/serving_modules.py::moe_forward_serving`` (the
function the PSM binds onto every ``KimiSparseMoeBlock``) against the M2 eager
``kimi_k3/model.py::KimiSparseMoeBlock`` — the implementation that is bit-exact
to the HF oracle — on IDENTICAL weights and input.

The property under test is the LatentMoE seam, which the serving forward used
to ignore entirely:

  * router on the PRE-down-proj hidden (not the latent);
  * ``routed_expert_down_proj`` (H -> latent) applied ONCE PER TOKEN before
    dispatch — not once per (token, expert);
  * routed experts computed in the latent space;
  * FP32 combine;
  * ``routed_expert_norm`` ONCE post-combine, pre-up;
  * ``routed_expert_up_proj`` (latent -> H);
  * shared expert on the identity path, in HIDDEN space.

Two synthetic configs are used (both from the M2 harness): K3-SYN-25 (latent =
hidden/2, the real ratio) and K3-SKEW-10 (latent != hidden/2, shared width 3x,
rms_norm_eps 2e-5) — the second kills "derived the dim from the wrong source"
bugs that a hidden/2 ratio would hide.

Run on a CUDA GPU (keep it off the GPU the KDA stage is using):

    K3_LATENT_MOE_GPU=1 CUDA_VISIBLE_DEVICES=1 \
    PYTHONPATH=<repo>:<fla-src> python -m pytest \
        tests/gpu/test_kimi_linear_latent_moe_serving.py -x -q -rA \
        -p no:cacheprovider

With ``K3_LATENT_MOE_GPU=1`` a missing GPU is a hard error, never a skip.
The batchgen imports live inside the tests: ``kimi_linear/model.py`` imports
fla at module scope, so collection on a CPU box must not touch it.
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


if os.environ.get("K3_LATENT_MOE_GPU") == "1" and not torch.cuda.is_available():
    raise RuntimeError(
        "K3_LATENT_MOE_GPU=1 but CUDA is unavailable — this staged run must "
        "not silently skip. Check CUDA_VISIBLE_DEVICES / the driver.")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="staged GPU validation (serving LatentMoE parity)")

DEV = "cuda"
CONFIGS = {
    "syn25": H.syn25_config_dict,
    "skew10": H.skew10_config_dict,
}


# --------------------------------------------------------------------------- #
#  Build the two stacks on identical weights                                   #
# --------------------------------------------------------------------------- #
def _build_pair(cfg_name: str, dtype: torch.dtype):
    """(eager M2 block, serving block) — same seeded weights, same dtype policy.

    The serving block is the REAL production object: the ``kimi_linear``
    ``KimiSparseMoeBlock`` with ``moe_forward_serving`` bound onto it exactly
    as ``KimiLinearParallelStrategyManager._config_expert_modules`` does. Its
    experts are left as plain modules (not ``KimiLinearExpertWrapper``): the
    wrapper only adds the host->GPU weight streaming, which needs a core
    engine and changes no math.
    """
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig
    from batchgen.models.moonshotai.kimi_linear.model import (
        KimiSparseMoeBlock as ServingMoeBlock,
    )
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        moe_forward_serving,
    )

    cfg_dict = CONFIGS[cfg_name]()
    ours = H.load_our_modules()

    k3_cfg = H.build_our_config(cfg_dict)
    eager = ours.model.KimiSparseMoeBlock(k3_cfg)

    lin_cfg = KimiLinearConfig.from_hf_dict(cfg_dict)
    serving = ServingMoeBlock(lin_cfg)
    serving.forward = types.MethodType(moe_forward_serving, serving)

    # Sanity on the config bridge before any numbers are compared.
    assert serving.use_latent_moe, "serving block did not take the LatentMoE path"
    assert serving.latent_moe_use_norm
    assert serving.moe_hidden_size == k3_cfg.routed_expert_hidden_size
    assert serving.hidden_dim == k3_cfg.hidden_size
    assert len(serving.experts) == len(eager.experts)
    assert serving.gate.top_k == eager.gate.top_k

    # Seeds OUR (eager) module, mirrors into the serving module, applies the
    # shared dtype policy (bf16 everywhere except the checkpoint's FP32 set,
    # i.e. gate.e_score_correction_bias).
    H.load_pair(eager, serving, k3_cfg.kda_num_heads, dtype)
    eager.to(DEV)
    serving.to(DEV)
    return eager, serving, k3_cfg


def _err_ratio(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a, r = actual.float(), ref.float()
    return ((a - r).pow(2).mean().sqrt() / (r.pow(2).mean().sqrt() + 1e-8)).item()


# --------------------------------------------------------------------------- #
#  1. Numeric parity vs the M2 eager block                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg_name", sorted(CONFIGS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_serving_latent_moe_matches_eager(cfg_name, dtype):
    eager, serving, cfg = _build_pair(cfg_name, dtype)
    tokens = 64
    x = H.seeded_input(
        "latent_moe:{}".format(cfg_name), 1, tokens, cfg.hidden_size, dtype=dtype
    ).to(DEV)

    with torch.no_grad():
        ref = eager(x)
        got = serving(x)

    assert got.shape == ref.shape == (1, tokens, cfg.hidden_size)
    what = "serving LatentMoE {} {}".format(cfg_name, str(dtype).split(".")[-1])
    ratio = _err_ratio(got, ref)
    max_abs = (got.float() - ref.float()).abs().max().item()
    print("[latent-moe] {:34s} err_ratio={:.3e} max_abs={:.3e} ref_rms={:.3e}"
          .format(what, ratio, max_abs, ref.float().pow(2).mean().sqrt().item()))

    # The only legitimate difference is FP32 combine ORDER (the eager block
    # sums over the top-k axis; the serving block index_add_s in expert order),
    # so the bar is tight — a real seam error (down-proj per (token, expert),
    # norm per expert, router on the latent) lands orders of magnitude above.
    assert torch.isfinite(got).all(), what + ": non-finite output"
    H.assert_bf16_gate(got, ref, what)
    assert ratio < (1e-3 if dtype == torch.bfloat16 else 1e-5), (
        "{}: err_ratio={:.3e}".format(what, ratio))


# --------------------------------------------------------------------------- #
#  2. The router sees the PRE-down hidden — identical index sets              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg_name", sorted(CONFIGS))
def test_router_index_sets_agree(cfg_name):
    eager, serving, cfg = _build_pair(cfg_name, torch.bfloat16)
    tokens = 64
    x = H.seeded_input(
        "latent_moe:{}".format(cfg_name), 1, tokens, cfg.hidden_size,
        dtype=torch.bfloat16,
    ).to(DEV)

    with torch.no_grad():
        ref_idx, ref_w = eager.gate(x)
        got_idx, got_w = serving.gate(x)

    ref_sets = H.topk_index_sets(ref_idx)
    got_sets = H.topk_index_sets(got_idx)
    disagree = sum(int(a != b) for a, b in zip(ref_sets, got_sets))
    print("[latent-moe] router {:28s} index-set agreement {}/{} tokens "
          "(top_k={}, experts={})".format(
              cfg_name, tokens - disagree, tokens, eager.gate.top_k,
              eager.gate.num_experts))
    assert disagree == 0
    # PRE-bias gather: weights must match too, not just the selection.
    assert _err_ratio(got_w.sort(dim=-1).values, ref_w.sort(dim=-1).values) < 1e-6


# --------------------------------------------------------------------------- #
#  3. down_proj / norm / up_proj run ONCE per token, not once per (token, k)   #
# --------------------------------------------------------------------------- #
def test_latent_projections_run_once_per_token():
    eager, serving, cfg = _build_pair("syn25", torch.bfloat16)
    tokens = 48
    top_k = serving.gate.top_k
    x = H.seeded_input(
        "latent_moe:once", 1, tokens, cfg.hidden_size, dtype=torch.bfloat16
    ).to(DEV)

    seen = {}

    def record(tag):
        def hook(_module, inputs, _output):
            seen.setdefault(tag, []).append(tuple(inputs[0].shape))
        return hook

    serving.routed_expert_down_proj.register_forward_hook(record("down"))
    serving.routed_expert_norm.register_forward_hook(record("norm"))
    serving.routed_expert_up_proj.register_forward_hook(record("up"))
    expert_calls = []
    for expert in serving.experts:
        expert.register_forward_hook(
            lambda _m, i, _o: expert_calls.append(i[0].shape[0]))

    with torch.no_grad():
        serving(x)

    latent = cfg.routed_expert_hidden_size
    assert seen["down"] == [(tokens, cfg.hidden_size)], seen["down"]
    assert seen["norm"] == [(tokens, latent)], seen["norm"]
    assert seen["up"] == [(tokens, latent)], seen["up"]
    # Streamed-expert lockstep: EVERY expert is driven, 0-token ones included.
    assert len(expert_calls) == len(serving.experts)
    assert sum(expert_calls) == tokens * top_k
    print("[latent-moe] once-per-token OK: down/norm/up 1 call each on "
          "{} rows; experts driven {}/{} , assignments {}".format(
              tokens, len(expert_calls), len(serving.experts), sum(expert_calls)))


# --------------------------------------------------------------------------- #
#  4. Empty DP rank (0 tokens) still drives every expert and returns (…,0,H)   #
# --------------------------------------------------------------------------- #
def test_empty_rank_drives_every_expert():
    _eager, serving, cfg = _build_pair("syn25", torch.bfloat16)
    x = torch.empty(1, 0, cfg.hidden_size, dtype=torch.bfloat16, device=DEV)
    expert_calls = []
    for expert in serving.experts:
        expert.register_forward_hook(
            lambda _m, i, _o: expert_calls.append(i[0].shape[0]))
    with torch.no_grad():
        out = serving(x)
    assert out.shape == (1, 0, cfg.hidden_size)
    assert len(expert_calls) == len(serving.experts)
    assert sum(expert_calls) == 0


# --------------------------------------------------------------------------- #
#  5. A K3 config MUST NOT reach a hidden-space branch (no silent fallback)    #
# --------------------------------------------------------------------------- #
def test_k3_config_without_latent_moe_hard_fails():
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig
    from batchgen.models.moonshotai.kimi_linear.model import (
        KimiSparseMoeBlock as ServingMoeBlock,
    )
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        moe_forward_serving,
    )

    cfg_dict = H.syn25_config_dict()
    cfg_dict["text_config"].pop("routed_expert_hidden_size")   # the K3 tell
    lin_cfg = KimiLinearConfig.from_hf_dict(cfg_dict)
    assert lin_cfg.model_type == "kimi_k3" and not lin_cfg.use_latent_moe

    with torch.device("meta"):
        block = ServingMoeBlock(lin_cfg)
    block.forward = types.MethodType(moe_forward_serving, block)
    with pytest.raises(RuntimeError, match="non-latent"):
        block(torch.empty(1, 4, lin_cfg.hidden_size, device="meta"))

    # Same for the norm being switched off under a K3 config.
    cfg_dict = H.syn25_config_dict()
    cfg_dict["text_config"]["latent_moe_use_norm"] = False
    lin_cfg = KimiLinearConfig.from_hf_dict(cfg_dict)
    with torch.device("meta"):
        block = ServingMoeBlock(lin_cfg)
    block.forward = types.MethodType(moe_forward_serving, block)
    with pytest.raises(RuntimeError, match="routed_expert_norm"):
        block(torch.empty(1, 4, lin_cfg.hidden_size, device="meta"))


# --------------------------------------------------------------------------- #
#  6. The 48B (non-latent) path is untouched                                   #
# --------------------------------------------------------------------------- #
def test_non_latent_config_still_runs_in_hidden_space():
    """A kimi_linear (48B-shaped) config keeps the hidden-space MoE: no
    down/up projection exists and the guard must not fire."""
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig
    from batchgen.models.moonshotai.kimi_linear.model import (
        KimiSparseMoeBlock as ServingMoeBlock,
    )
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        moe_forward_serving,
    )

    lin_cfg = KimiLinearConfig(
        model_type="kimi_linear",
        hidden_size=128,
        intermediate_size=256,
        n_routed_experts=8,
        num_local_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_shared_experts=1,
        hidden_act="silu",
        routed_expert_hidden_size=None,
    )
    block = ServingMoeBlock(lin_cfg)
    assert not block.use_latent_moe
    assert not hasattr(block, "routed_expert_down_proj")
    torch.manual_seed(0)
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.02)
    block = block.to(DEV).to(torch.bfloat16).eval()

    x = H.seeded_input("non_latent", 1, 16, 128, dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        ref = ServingMoeBlock.forward(block, x)           # eager class method
        block.forward = types.MethodType(moe_forward_serving, block)
        got = block(x)
    ratio = _err_ratio(got, ref)
    print("[latent-moe] 48B-shaped (non-latent) serving vs eager "
          "err_ratio={:.3e}".format(ratio))
    assert ratio < 1e-3
