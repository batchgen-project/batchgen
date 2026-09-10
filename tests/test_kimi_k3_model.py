# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""CPU parity + contract suite for the Kimi-K3 M2 model (prefill-only, eager).

Runs anywhere with torch + transformers>=4.56 + einops: modules under test are
loaded by file path (never ``import batchgen``), the HF oracle is the vendored
checkpoint reference under ``tests/kimi_k3_oracle_assets/`` with the fla CPU
shim installed.  Environment gaps HARD-FAIL — a skip would silently green the
M2 gate.

Mutation discipline: every detector here is proven to fail on a deliberately
broken variant by ``python tests/mutation_check_kimi_k3.py`` (registry in
``tests/kimi_k3_harness.py``).  Do not weaken an assertion without re-running
the mutation check.

What CPU parity does NOT cover (closed by the staged GPU test
tests/gpu/test_kimi_k3_kda_fla_parity.py on h20-instance-1):
  * the fla `chunk_kda` kernel interior (both stacks share the vendored torch
    core on CPU, so it cancels here);
  * real fla ShortConvolution / FusedRMSNormGated vs the pure-torch ports;
  * flash-attention-2 vs eager MLA numerics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kimi_k3_harness as H  # noqa: E402


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #
CFGS = {
    "syn25": H.syn25_config_dict,
    "skew10": H.skew10_config_dict,
}


@pytest.fixture(params=["syn25", "skew10"])
def cfg_name(request):
    return request.param


def cfg_dict_of(name: str) -> dict:
    return CFGS[name]()


# --------------------------------------------------------------------------- #
#  T0 — vendored oracle byte pins                                              #
# --------------------------------------------------------------------------- #
def test_oracle_md5_pins():
    H.check_oracle_pins()
    # the vendored configuration must be byte-identical to the production-
    # vendored copy in kimi_k3/assets/ (single source of truth, two pins)
    assets_copy = H.K3_PKG_DIR / "assets" / "configuration_kimi_k3.py"
    oracle_copy = H.ORACLE_ASSETS_DIR / "configuration_kimi_k3.py"
    assert assets_copy.read_bytes() == oracle_copy.read_bytes(), (
        "tests/kimi_k3_oracle_assets/configuration_kimi_k3.py drifted from "
        "batchgen/models/moonshotai/kimi_k3/assets/configuration_kimi_k3.py")


# --------------------------------------------------------------------------- #
#  T1 — state-dict key/shape parity vs the oracle                              #
# --------------------------------------------------------------------------- #
def test_state_dict_key_parity(cfg_name):
    cfg_dict = cfg_dict_of(cfg_name)
    ours = H.load_our_modules()
    om = H.load_oracle_modules()
    cfg = H.build_our_config(cfg_dict)
    our_model = ours.model.KimiK3ForCausalLM(cfg, kda_backend="reference")
    ocfg = H.build_oracle_config(cfg_dict)
    oracle_model = om.modeling.KimiLinearForCausalLM(ocfg)

    our_shapes = H.named_shapes_of(our_model)
    oracle_shapes = H.named_shapes_of(oracle_model)
    only_ours = sorted(set(our_shapes) - set(oracle_shapes))
    only_oracle = sorted(set(oracle_shapes) - set(our_shapes))
    assert not only_ours and not only_oracle, (
        "state-dict key sets diverge.\n  only ours: {}\n  only oracle: {}"
        .format(only_ours[:10], only_oracle[:10]))

    H_kda = cfg.kda_num_heads
    mismatched = []
    for name, shape in our_shapes.items():
        expected = oracle_shapes[name]
        if name.endswith("A_log"):
            # THE one documented delta: checkpoint ships F32[128] zero-padded;
            # the oracle class allocates [num_heads] (flag B).
            assert shape == (cfg.a_log_padded_len,), name
            assert expected == (H_kda,), name
            continue
        if shape != expected:
            mismatched.append((name, shape, expected))
    assert not mismatched, "shape mismatches: {}".format(mismatched[:10])


def _expected_real_k3_keys():
    """Independent expansion of the 45 model-side templates (TENSOR_NAME_MAP
    minus the `language_model.` prefix, MXFP4 expert packs replaced by the 3
    dense wN.weight the model exposes)."""
    full_attn_1based = set(range(4, 93, 4)) | {93}
    keys = {
        "model.embed_tokens.weight", "model.norm.weight",
        "model.output_attn_res_norm.weight", "model.output_attn_res_proj.weight",
        "lm_head.weight",
    }
    for L in range(93):
        p = "model.layers.{}.".format(L)
        keys |= {
            p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
            p + "self_attention_res_norm.weight", p + "self_attention_res_proj.weight",
            p + "mlp_res_norm.weight", p + "mlp_res_proj.weight",
        }
        if (L + 1) in full_attn_1based:   # MLA
            keys |= {p + "self_attn." + s for s in (
                "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
                "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
                "kv_b_proj.weight", "g_proj.weight", "o_proj.weight")}
        else:                              # KDA
            keys |= {p + "self_attn." + s for s in (
                "q_proj.weight", "k_proj.weight", "v_proj.weight",
                "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
                "A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight",
                "b_proj.weight", "g_proj.weight", "o_norm.weight", "o_proj.weight")}
        if L == 0:
            keys |= {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                     p + "mlp.down_proj.weight"}
        else:
            m = p + "block_sparse_moe."
            keys |= {m + s for s in (
                "gate.weight", "gate.e_score_correction_bias",
                "routed_expert_down_proj.weight", "routed_expert_up_proj.weight",
                "routed_expert_norm.weight", "shared_experts.gate_proj.weight",
                "shared_experts.up_proj.weight", "shared_experts.down_proj.weight")}
            for e in range(896):
                keys |= {m + "experts.{}.w{}.weight".format(e, w) for w in (1, 2, 3)}
    return keys


def test_state_dict_real_config_meta():
    """Meta-device construction at the REAL 93-layer config: full key set,
    section-E shapes for one layer of each kind, and the FP32/BF16 dtype
    policy — zero weights allocated."""
    ours = H.load_our_modules()
    cfg = H.build_our_config(H.real_config_dict())
    with torch.device("meta"):
        model = ours.model.KimiK3ForCausalLM(cfg, kda_backend="reference")
    ours.model.cast_model_to_inference_dtype(model, torch.bfloat16)

    got = dict(model.named_parameters())
    expected = _expected_real_k3_keys()
    only_got = sorted(set(got) - expected)
    only_exp = sorted(expected - set(got))
    assert not only_got and not only_exp, (
        "only model: {} ...\nonly expected: {} ...".format(only_got[:8], only_exp[:8]))
    assert len(got) == 249756

    def check(name, shape, dtype):
        p = got[name]
        assert tuple(p.shape) == shape, (name, tuple(p.shape), shape)
        assert p.dtype == dtype, (name, p.dtype, dtype)

    bf16, f32 = torch.bfloat16, torch.float32
    check("model.embed_tokens.weight", (163840, 7168), bf16)
    check("lm_head.weight", (163840, 7168), bf16)
    check("model.output_attn_res_proj.weight", (1, 7168), bf16)
    # KDA layer 0
    check("model.layers.0.self_attn.q_proj.weight", (12288, 7168), bf16)
    check("model.layers.0.self_attn.q_conv1d.weight", (12288, 1, 4), f32)
    check("model.layers.0.self_attn.A_log", (128,), f32)
    check("model.layers.0.self_attn.dt_bias", (12288,), f32)
    check("model.layers.0.self_attn.f_a_proj.weight", (128, 7168), bf16)
    check("model.layers.0.self_attn.f_b_proj.weight", (12288, 128), bf16)
    check("model.layers.0.self_attn.b_proj.weight", (96, 7168), bf16)
    check("model.layers.0.self_attn.g_proj.weight", (12288, 7168), bf16)
    check("model.layers.0.self_attn.o_norm.weight", (128,), f32)
    check("model.layers.0.self_attn.o_proj.weight", (7168, 12288), bf16)
    check("model.layers.0.mlp.gate_proj.weight", (33792, 7168), bf16)
    check("model.layers.0.mlp.down_proj.weight", (7168, 33792), bf16)
    # MLA layer 3
    check("model.layers.3.self_attn.q_a_proj.weight", (1536, 7168), bf16)
    check("model.layers.3.self_attn.q_a_layernorm.weight", (1536,), bf16)
    check("model.layers.3.self_attn.q_b_proj.weight", (18432, 1536), bf16)
    check("model.layers.3.self_attn.kv_a_proj_with_mqa.weight", (576, 7168), bf16)
    check("model.layers.3.self_attn.kv_a_layernorm.weight", (512,), bf16)
    check("model.layers.3.self_attn.kv_b_proj.weight", (24576, 512), bf16)
    check("model.layers.3.self_attn.g_proj.weight", (12288, 7168), bf16)
    check("model.layers.3.self_attn.o_proj.weight", (7168, 12288), bf16)
    # MoE layer 1
    check("model.layers.1.block_sparse_moe.gate.weight", (896, 7168), bf16)
    check("model.layers.1.block_sparse_moe.gate.e_score_correction_bias", (896,), f32)
    check("model.layers.1.block_sparse_moe.routed_expert_down_proj.weight", (3584, 7168), bf16)
    check("model.layers.1.block_sparse_moe.routed_expert_up_proj.weight", (7168, 3584), bf16)
    check("model.layers.1.block_sparse_moe.routed_expert_norm.weight", (3584,), bf16)
    check("model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight", (6144, 7168), bf16)
    check("model.layers.1.block_sparse_moe.shared_experts.down_proj.weight", (7168, 6144), bf16)
    check("model.layers.1.block_sparse_moe.experts.0.w1.weight", (3072, 3584), bf16)
    check("model.layers.1.block_sparse_moe.experts.895.w2.weight", (3584, 3072), bf16)
    check("model.layers.1.block_sparse_moe.experts.895.w3.weight", (3072, 3584), bf16)
    # the MLA q_a/kv_a eps trap and the config eps
    layer3 = model.model.layers[3]
    assert layer3.self_attn.q_a_layernorm.variance_epsilon == 1e-6
    assert layer3.self_attn.kv_a_layernorm.variance_epsilon == 1e-6
    assert layer3.input_layernorm.variance_epsilon == 1e-5


# --------------------------------------------------------------------------- #
#  T2 — hybrid layer map (1-BASED lists)                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("which", ["syn25", "skew10", "real"])
def test_layer_type_map(which):
    cfg_dict = H.real_config_dict() if which == "real" else cfg_dict_of(which)
    cfg = H.build_our_config(cfg_dict)
    ocfg = H.build_oracle_config(cfg_dict)
    ours_map = {i: cfg.is_kda_layer(i) for i in range(cfg.num_hidden_layers)}
    oracle_map = {i: bool(ocfg.is_kda_layer(i)) for i in range(cfg.num_hidden_layers)}
    assert ours_map == oracle_map
    assert ours_map[0] is True, "layer 0 must be KDA (dense MLP layer)"
    if which == "real":
        mla = {i for i, is_kda in ours_map.items() if not is_kda}
        assert mla == (set(range(3, 92, 4)) | {92}), "double-MLA tail {91,92} expected"
        assert sum(ours_map.values()) == 69


# --------------------------------------------------------------------------- #
#  T3 — MLA module parity                                                      #
# --------------------------------------------------------------------------- #
def _mla_inputs(cfg, dtype, name):
    hidden = H.seeded_input("mla:" + name, 2, 48, cfg.hidden_size, dtype=dtype)
    mask = H.build_causal_mask(48, dtype)
    return hidden, mask


def test_mla_module_fp32(cfg_name):
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "mla", cfg_dict_of(cfg_name), torch.float32, layer_idx=3)
    hidden, mask = _mla_inputs(cfg, torch.float32, cfg_name)
    # preflight: the q_a pre-norm mean-square must sit where the 1e-6 vs 1e-5
    # eps flip is visible above the 1e-6 fp32 gate (discriminative-power check
    # for the eps mutations; the harness seeds q_a_proj smaller on purpose)
    with torch.no_grad():
        q_a = our_mod.q_a_proj(hidden)
        ms = q_a.float().pow(2).mean().item()
    assert 1e-4 < ms < 0.05, (
        "q_a mean-square {} outside the eps-discriminative band: the "
        "q_a_eps mutation would be invisible to the 1e-6 gate".format(ms))
    with torch.no_grad():
        ours = our_mod(hidden, attention_mask=mask)
        ref = oracle_mod(hidden_states=hidden, attention_mask=mask)
    H.assert_fp32_tight(ours, ref, "MLA fp32 [{}]".format(cfg_name))


def test_mla_module_bf16(cfg_name):
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "mla", cfg_dict_of(cfg_name), torch.bfloat16, layer_idx=3)
    hidden, mask = _mla_inputs(cfg, torch.bfloat16, cfg_name)
    with torch.no_grad():
        ours = our_mod(hidden, attention_mask=mask)
        ref = oracle_mod(hidden_states=hidden, attention_mask=mask)
    H.assert_bf16_gate(ours, ref, "MLA bf16 [{}]".format(cfg_name))


def test_mla_module_bf16_bitwise():
    """Ours and the oracle run the identical op sequence in the identical
    order, so BF16 outputs are bit-equal on the same device.  This is the
    detector for interior-dtype mutations (e.g. an fp32 output-gate sigmoid —
    the reference computes it in bf16, ML:470-472) that the relative gate
    cannot see.  If a legitimate future op reordering breaks this, the
    recorded fallback decision is a <=2-ulp gate — do not silently relax."""
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "mla", H.syn25_config_dict(), torch.bfloat16, layer_idx=3)
    hidden, mask = _mla_inputs(cfg, torch.bfloat16, "syn25")
    with torch.no_grad():
        ours = our_mod(hidden, attention_mask=mask)
        ref = oracle_mod(hidden_states=hidden, attention_mask=mask)
    assert torch.equal(ours, ref)


def test_mla_module_hard_fails():
    ours = H.load_our_modules()
    cfg = H.build_our_config(H.syn25_config_dict())
    mod = ours.model.KimiK3MLAAttention(cfg, layer_idx=3)
    hidden = H.seeded_input("mla:hf", 1, 4, cfg.hidden_size)
    with pytest.raises(NotImplementedError, match="M3"):
        mod(hidden, past_key_values=object())


# --------------------------------------------------------------------------- #
#  T4 — KDA module parity (CPU: kernel interior cancels — wiring is under test) #
# --------------------------------------------------------------------------- #
def test_kda_module_fp32(cfg_name):
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "kda", cfg_dict_of(cfg_name), torch.float32, layer_idx=0)
    hidden = H.seeded_input("kda:" + cfg_name, 2, 33, cfg.hidden_size)  # odd T
    with torch.no_grad():
        ours = our_mod(hidden)
        ref = oracle_mod(hidden_states=hidden, attention_mask=None)
    H.assert_fp32_tight(ours, ref, "KDA fp32 [{}]".format(cfg_name))


def test_kda_module_bf16(cfg_name):
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "kda", cfg_dict_of(cfg_name), torch.bfloat16, layer_idx=0)
    hidden = H.seeded_input("kda:" + cfg_name, 2, 33, cfg.hidden_size,
                            dtype=torch.bfloat16)
    with torch.no_grad():
        ours = our_mod(hidden)
        ref = oracle_mod(hidden_states=hidden, attention_mask=None)
    H.assert_bf16_gate(ours, ref, "KDA bf16 [{}]".format(cfg_name))


def test_kda_module_hard_fails():
    ours = H.load_our_modules()
    cfg = H.build_our_config(H.syn25_config_dict())
    mod = ours.model.KimiK3KDAAttention(cfg, layer_idx=0, kda_backend="reference")
    hidden = H.seeded_input("kda:hf", 1, 4, cfg.hidden_size)
    with pytest.raises(NotImplementedError, match="M3"):
        mod(hidden, cache_params=object())
    with pytest.raises(NotImplementedError, match="M4"):
        mod(hidden, attention_mask=torch.ones(1, 4))
    # cu_seqlens is now SUPPORTED (packed prefill), so the assertion moves to
    # the two things that stay contract errors: a packed descriptor on a
    # non-packed batch, and the reference backend, which has no varlen path
    # and would silently run one recurrence across every boundary.
    batched = H.seeded_input("kda:hf:b2", 2, 4, cfg.hidden_size)
    with pytest.raises(ValueError, match="PACKED"):
        mod(batched, cu_seqlens=torch.tensor([0, 4]))
    with pytest.raises(NotImplementedError, match="no packed/varlen path"):
        mod(hidden, cu_seqlens=torch.tensor([0, 4]))
    with pytest.raises(ValueError, match="kda_backend"):
        ours.model.KimiK3KDAAttention(cfg, layer_idx=0, kda_backend="banana")


# --------------------------------------------------------------------------- #
#  T5 — SiTU bit-exactness                                                     #
# --------------------------------------------------------------------------- #
def test_situ_bitexact():
    """Dense sweep over [-30, 30]^2 (covers |gate|>4 and |up|>25, the two tanh
    clamp regions — preflight iv is the sweep itself).  Bit-equality after the
    bf16 downcast is legitimate: same formula, same op order (ML:76-82).  It
    catches interior-dtype mutations the relative gate cannot see."""
    ours = H.load_our_modules()
    situ = ours.model.SituAndMul(beta=4.0, linear_beta=25.0)
    grid = torch.linspace(-30.0, 30.0, 121)
    gate, up = torch.meshgrid(grid, grid, indexing="ij")
    x32 = torch.cat([gate.reshape(-1, 1), up.reshape(-1, 1)], dim=-1)

    def reference(x):
        d = x.shape[-1] // 2
        g = x[..., :d].to(torch.float32)
        u = x[..., d:].to(torch.float32)
        a = 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)
        u = 25.0 * torch.tanh(u / 25.0)
        return (a * u).to(x.dtype)

    for dtype in (torch.float32, torch.bfloat16):
        x = x32.to(dtype)
        assert torch.equal(situ(x), reference(x)), "SiTU mismatch in {}".format(dtype)
    with pytest.raises(ValueError, match="beta"):
        ours.model.SituAndMul(beta=None, linear_beta=25.0)


# --------------------------------------------------------------------------- #
#  T6 — router unit (index-exactness lives HERE, module level)                 #
# --------------------------------------------------------------------------- #
def test_router_unit():
    ours = H.load_our_modules()
    om = H.load_oracle_modules()
    cfg_dict = H.syn25_config_dict()
    cfg = H.build_our_config(cfg_dict)
    ocfg = H.build_oracle_config(cfg_dict)
    our_gate = ours.model.KimiMoEGate(cfg)
    oracle_gate = om.modeling.KimiMoEGate(ocfg)

    n_exp, hidden = cfg.num_experts, cfg.hidden_size
    # adversarial near-tie weights: the top half of the experts duplicates the
    # bottom half + 1e-4 noise, so rank-16/17 near-ties are dense in any input
    w = H.seeded_master("router:base_w", (n_exp, hidden))
    w[n_exp // 2:] = w[: n_exp // 2] + 1e-4 * H.seeded_master(
        "router:tie_noise", (n_exp // 2, hidden))
    bias = H.seeded_master("router:bias", (n_exp,))
    with torch.no_grad():
        for gate in (our_gate, oracle_gate):
            gate.weight.copy_(w)
            gate.e_score_correction_bias.copy_(bias)
    our_gate.eval(), oracle_gate.eval()

    hidden_states = H.seeded_input("router:h", 1, 256, hidden)
    # ---- preflights (discriminative power, computed independently) ----
    logits = hidden_states.view(-1, hidden).float() @ w.t()
    scores = logits.sigmoid()
    top_bias = torch.topk(scores + bias, 16, dim=-1).indices
    top_nobias = torch.topk(scores, 16, dim=-1).indices
    flips_bias = sum(frozenset(a.tolist()) != frozenset(b.tolist())
                     for a, b in zip(top_bias, top_nobias))
    assert flips_bias >= 1, "e_score bias never changes a selection: mutation blind"
    logits_bf16 = logits.bfloat16().float()
    top_bf16 = torch.topk(logits_bf16.sigmoid() + bias, 16, dim=-1).indices
    flips_bf16 = sum(frozenset(a.tolist()) != frozenset(b.tolist())
                     for a, b in zip(top_bias, top_bf16))
    assert flips_bf16 >= 1, "bf16 router rounding never flips a selection: mutation blind"

    # ---- the actual parity gate ----
    with torch.no_grad():
        our_idx, our_w = our_gate(hidden_states)
        oracle_idx, oracle_w = oracle_gate(hidden_states)
    assert our_w.dtype == torch.float32 and oracle_w.dtype == torch.float32
    # topk(sorted=False) order is implementation-defined (flag D): canonicalize
    # by sorting the index axis before comparing — SETS must be exactly equal
    assert H.topk_index_sets(our_idx) == H.topk_index_sets(oracle_idx)
    our_order = our_idx.argsort(dim=-1)
    oracle_order = oracle_idx.argsort(dim=-1)
    H.assert_fp32_tight(
        our_w.gather(1, our_order), oracle_w.gather(1, oracle_order), "router weights")
    ws = our_w.sum(dim=-1)
    H.assert_fp32_tight(ws, torch.ones_like(ws), "router renorm", tol=1e-5)


# --------------------------------------------------------------------------- #
#  T7 — LatentMoE block parity                                                 #
# --------------------------------------------------------------------------- #
def test_latent_moe_fp32(cfg_name):
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "moe", cfg_dict_of(cfg_name), torch.float32)
    hidden = H.seeded_input("moe:" + cfg_name, 2, 24, cfg.hidden_size)
    with torch.no_grad():
        ours = our_mod(hidden)
        ref = oracle_mod(hidden)
        our_idx, _ = our_mod.gate(hidden)
        oracle_idx, _ = oracle_mod.gate(hidden)
    assert H.topk_index_sets(our_idx) == H.topk_index_sets(oracle_idx)
    H.assert_fp32_tight(ours, ref, "MoE fp32 [{}]".format(cfg_name))
    # N=1 token: 16-of-64 selection leaves >=1 expert empty by construction —
    # exercises the empty-expert skip in moe_infer
    one = H.seeded_input("moe1:" + cfg_name, 1, 1, cfg.hidden_size)
    with torch.no_grad():
        H.assert_fp32_tight(our_mod(one), oracle_mod(one),
                            "MoE fp32 single-token [{}]".format(cfg_name))


def test_latent_moe_bf16(cfg_name):
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "moe", cfg_dict_of(cfg_name), torch.bfloat16)
    hidden = H.seeded_input("moe:" + cfg_name, 2, 24, cfg.hidden_size,
                            dtype=torch.bfloat16)
    with torch.no_grad():
        ours = our_mod(hidden)
        ref = oracle_mod(hidden)
    H.assert_bf16_gate(ours, ref, "MoE bf16 [{}]".format(cfg_name))


def test_moe_bf16_bitwise():
    """Identical op order (fp32 router + fp32 combine islands included) implies
    bit-equality in bf16 — the detector for the `combine_bf16` mutation, which
    the fp32 run cannot see (everything is fp32 there) and the relative gate
    forgives.  Fallback decision if a legit reorder ever lands: <=2-ulp gate."""
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "moe", H.syn25_config_dict(), torch.bfloat16)
    hidden = H.seeded_input("moe:syn25", 2, 24, cfg.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(our_mod(hidden), oracle_mod(hidden))


# --------------------------------------------------------------------------- #
#  T8 — reference mixer vs the oracle's own                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nb", [1, 3, 9])
def test_attn_res_reference_fp32(nb):
    ours = H.load_our_modules()
    om = H.load_oracle_modules()
    hidden = 448
    norm = ours.model.KimiRMSNorm(hidden, eps=1e-5)
    proj = torch.nn.Linear(hidden, 1, bias=False)
    with torch.no_grad():
        norm.weight.copy_(H.seeded_master("t8:norm.weight", (hidden,)))
        proj.weight.copy_(H.seeded_master("t8:res_proj.weight", (1, hidden)))
    prefix = H.seeded_input("t8:prefix:{}".format(nb), 64, hidden)
    block = H.seeded_input("t8:block:{}".format(nb), 64, nb, hidden)
    got = H.apply_attn_res_reference(prefix, block, proj, norm)
    want = om.modeling._apply_attn_res(prefix, block, proj, norm)
    H.assert_fp32_tight(got, want, "reference mixer vs oracle nb={}".format(nb))


# --------------------------------------------------------------------------- #
#  T9 — lean mixer == reference mixer, max_abs < 1e-6 (POIS decision 2)        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(64, 1, 448), (4096, 8, 1024),
                                   (4097, 8, 1024), (50, 9, 448)])
def test_attn_res_lean_equiv(shape):
    T, nb, hidden = shape
    ours = H.load_our_modules()
    norm = ours.model.KimiRMSNorm(hidden, eps=1e-5)
    proj = torch.nn.Linear(hidden, 1, bias=False)
    with torch.no_grad():
        norm.weight.copy_(H.seeded_master("t9:{}:norm.weight".format(hidden), (hidden,)))
        proj.weight.copy_(H.seeded_master("t9:{}:res_proj.weight".format(hidden),
                                          (1, hidden)))
    prefix = H.seeded_input("t9:prefix:{}".format(shape), T, hidden)
    block = H.seeded_input("t9:block:{}".format(shape), T, nb, hidden)
    lean = ours.model._apply_attn_res_lean(prefix, block, proj, norm, chunk_size=1024)
    ref = H.apply_attn_res_reference(prefix, block, proj, norm)
    H.assert_fp32_tight(lean, ref, "lean vs reference mixer {}".format(shape))


# --------------------------------------------------------------------------- #
#  T10 — the lean mixer must never materialize the (T, nb+1, H) fp32 tensor    #
# --------------------------------------------------------------------------- #
class _FP32AllocRecorder(TorchDispatchMode):
    """Records EVERY tensor allocation and its BYTE size.

    An fp32-element-only bound is evadable: a variant that materializes the
    full (T, nb+1, H) tensor in BF16 and floats it per chunk passes an
    fp32-shape check while allocating 150 MB.  The load-bearing property is
    memory, so the detector counts bytes in every dtype.
    """

    def __init__(self):
        self.shapes = []          # fp32 shapes (the exact-shape assertion)
        self.allocs = []          # (shape, dtype, nbytes) for every dtype

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                self.allocs.append(
                    (tuple(t.shape), t.dtype, t.numel() * t.element_size()))
                if t.dtype == torch.float32:
                    self.shapes.append(tuple(t.shape))
        return out


def test_attn_res_lean_no_materialization():
    """T=8192, nb=8, H=1024, chunk=1024: the reference form would allocate a
    (8192, 9, 1024) fp32 tensor (302 MB at full K3 hidden scale); the lean
    form's largest fp32 tensor must stay <= chunk*(nb+1)*H (37.7 MB-equivalent
    bound).  The `lean_revert_reference` mutation proves this detector fires."""
    T, nb, hidden, chunk = 8192, 8, 1024, 1024
    ours = H.load_our_modules()
    norm = ours.model.KimiRMSNorm(hidden, eps=1e-5)
    proj = torch.nn.Linear(hidden, 1, bias=False)
    prefix = H.seeded_input("t10:prefix", T, hidden, dtype=torch.bfloat16)
    block = H.seeded_input("t10:block", T, nb, hidden, dtype=torch.bfloat16)
    rec = _FP32AllocRecorder()
    with torch.no_grad(), rec:
        out = ours.model._apply_attn_res_lean(prefix, block, proj, norm,
                                              chunk_size=chunk)
    assert out.shape == (T, hidden)
    full = (T, nb + 1, hidden)
    assert full not in rec.shapes, (
        "the lean mixer materialized the full {} fp32 tensor".format(full))
    bound = int(chunk * (nb + 1) * hidden * 1.05)
    biggest = max((torch.Size(s).numel() for s in rec.shapes), default=0)
    assert biggest <= bound, (
        "largest fp32 allocation {} exceeds the chunked bound {}".format(biggest, bound))
    # BYTE bound across every dtype — closes the bf16-materialization evader.
    byte_bound = int(chunk * (nb + 1) * hidden * 4 * 1.05)
    worst = max(rec.allocs, key=lambda a: a[2], default=((), None, 0))
    assert worst[2] <= byte_bound, (
        "largest allocation {} {} = {} B exceeds the chunked byte bound {} B: "
        "the mixer materialized the full tensor in some dtype".format(
            worst[0], worst[1], worst[2], byte_bound))


# --------------------------------------------------------------------------- #
#  T11 — decoder-layer skeleton (boundary reset / prefix-hidden divergence)    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer_idx,nb", [(0, 0), (3, 1), (4, 2)])
def test_layer_skeleton(layer_idx, nb):
    """Layer 0: KDA + dense MLP, boundary, nb=0 in.  Layer 3: MLA + MoE,
    boundary (3 % 3 == 0) with nb=1 in — snapshot and reset with a non-empty
    residual.  Layer 4: KDA + MoE, non-boundary, nb=2 in — pre-attn mix runs,
    no reset."""
    cfg_dict = H.syn25_config_dict()
    our_mod, oracle_mod, cfg = H.build_pair_module(
        "layer", cfg_dict, torch.float32, layer_idx=layer_idx)
    B, T, hidden = 2, 16, cfg.hidden_size
    hidden_states = H.seeded_input("t11:h:{}".format(layer_idx), B, T, hidden)
    block = H.seeded_input("t11:br:{}".format(layer_idx), B * T, nb, hidden)
    mask = None if cfg.is_kda_layer(layer_idx) else H.build_causal_mask(T, torch.float32)
    with torch.no_grad():
        our_prefix, our_block = our_mod(hidden_states, mask, block)
        oracle_prefix, oracle_block = oracle_mod(
            hidden_states, attention_mask=mask, block_residual=block)
    boundary = (layer_idx % cfg.attn_res_block_size == 0)
    assert our_block.shape[1] == nb + (1 if boundary else 0)
    assert our_block.shape == oracle_block.shape
    H.assert_fp32_tight(our_block, oracle_block,
                        "layer {} block_residual".format(layer_idx), tol=1e-6)
    H.assert_fp32_tight(our_prefix, oracle_prefix,
                        "layer {} prefix_sum".format(layer_idx), tol=1e-6)


# --------------------------------------------------------------------------- #
#  T12/T13 — full-model parity                                                 #
# --------------------------------------------------------------------------- #
def _full_model_case(cfg_dict, dtype, batch, seq_len, tag):
    our_model, oracle_model, cfg = H.build_pair_causallm(cfg_dict, dtype)
    ids = H.seeded_token_ids(tag, batch, seq_len, cfg.vocab_size,
                             cfg.media_placeholder_token_id)
    with torch.no_grad():
        ours = our_model(input_ids=ids)
        ref = H.oracle_forward_logits(oracle_model, ids)
    return ours, ref, cfg


def test_full_model_logits():
    for batch, seq_len in ((4, 128), (1, 257)):     # dense batch + odd length
        ours, ref, _ = _full_model_case(
            H.syn25_config_dict(), torch.bfloat16, batch, seq_len,
            "t12:{}x{}".format(batch, seq_len))
        H.assert_bf16_gate(ours, ref, "full-model logits bf16 {}x{}".format(batch, seq_len))
        # module-level index-exactness lives in T6/T7; at full-model depth the
        # honored form is top-1 agreement at EVERY position
        assert torch.equal(ours.argmax(-1), ref.argmax(-1)), (
            "top-1 disagreement at {}x{}".format(batch, seq_len))


def test_full_model_logits_fp32_loose():
    ours, ref, _ = _full_model_case(
        H.syn25_config_dict(), torch.float32, 2, 64, "t12fp32")
    diff = (ours - ref).abs().max().item()
    assert diff < 1e-3, "fp32 full-model drift {}".format(diff)


def test_full_model_skew():
    for batch, seq_len in ((2, 96), (1, 65)):
        ours, ref, _ = _full_model_case(
            H.skew10_config_dict(), torch.bfloat16, batch, seq_len,
            "t13:{}x{}".format(batch, seq_len))
        H.assert_bf16_gate(ours, ref, "skew full-model bf16 {}x{}".format(batch, seq_len))
        assert torch.equal(ours.argmax(-1), ref.argmax(-1))


# --------------------------------------------------------------------------- #
#  T14 — block_residual is intra-forward scratch                               #
# --------------------------------------------------------------------------- #
def test_forward_twice_identical():
    ours = H.load_our_modules()
    cfg = H.build_our_config(H.skew10_config_dict())
    model = ours.model.KimiK3ForCausalLM(cfg, kda_backend="reference")
    sd, _ = H.make_state_dicts(H.named_shapes_of(model), cfg.kda_num_heads)
    model.load_state_dict(sd, strict=True)
    ours.model.cast_model_to_inference_dtype(model, torch.bfloat16)
    model.eval()
    ids = H.seeded_token_ids("t14", 2, 32, cfg.vocab_size,
                             cfg.media_placeholder_token_id)
    with torch.no_grad():
        first = model(input_ids=ids)
        second = model(input_ids=ids)
    assert torch.equal(first, second), (
        "a second forward differs: block_residual state leaked across forwards")


# --------------------------------------------------------------------------- #
#  T15 — A_log pad entries are inert                                           #
# --------------------------------------------------------------------------- #
def test_a_log_pad_poison():
    ours = H.load_our_modules()
    cfg = H.build_our_config(H.syn25_config_dict())
    mod = ours.model.KimiK3KDAAttention(cfg, layer_idx=0, kda_backend="reference")
    sd, _ = H.make_state_dicts(H.named_shapes_of(mod), cfg.kda_num_heads,
                               prefix="t15:")
    mod.load_state_dict(sd, strict=True)
    ours.model.cast_model_to_inference_dtype(mod, torch.bfloat16)
    mod.eval()
    hidden = H.seeded_input("t15:h", 1, 19, cfg.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        baseline = mod(hidden)
        mod.A_log[cfg.kda_num_heads:] = 1e6
        poisoned = mod(hidden)
    assert torch.equal(baseline, poisoned), (
        "poisoning A_log[{}:] changed the output — the module consumes the "
        "zero-pad region".format(cfg.kda_num_heads))


# --------------------------------------------------------------------------- #
#  T16 — hard-fail perimeter (messages asserted)                               #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_model():
    ours = H.load_our_modules()
    cfg = H.build_our_config(H.skew10_config_dict())
    model = ours.model.KimiK3ForCausalLM(cfg, kda_backend="reference")
    sd, _ = H.make_state_dicts(H.named_shapes_of(model), cfg.kda_num_heads)
    model.load_state_dict(sd, strict=True)
    ours.model.cast_model_to_inference_dtype(model, torch.bfloat16)
    model.eval()
    ids = H.seeded_token_ids("t16", 1, 8, cfg.vocab_size,
                             cfg.media_placeholder_token_id)
    return model, cfg, ids


def test_hard_fail_decode(small_model):
    model, cfg, ids = small_model
    with pytest.raises(NotImplementedError, match="PREFILL-ONLY.*M3"):
        model(input_ids=ids, past_key_values=object())


def test_hard_fail_configure_decoding(small_model):
    model, cfg, ids = small_model
    with pytest.raises(NotImplementedError, match="M3"):
        model.configure_decoding()


def test_hard_fail_padding_mask(small_model):
    model, cfg, ids = small_model
    with pytest.raises(NotImplementedError, match="M4"):
        model(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_hard_fail_position_ids(small_model):
    model, cfg, ids = small_model
    with pytest.raises(ValueError, match="NoPE"):
        model(input_ids=ids, position_ids=torch.arange(ids.shape[1]).unsqueeze(0))


def test_hard_fail_vision_token(small_model):
    model, cfg, ids = small_model
    poisoned = ids.clone()
    poisoned[0, 3] = cfg.media_placeholder_token_id
    with pytest.raises(NotImplementedError, match="media placeholder"):
        model(input_ids=poisoned)


def test_hard_fail_training_mode(small_model):
    model, cfg, ids = small_model
    model.train()
    try:
        with pytest.raises(RuntimeError, match="inference-only"):
            model(input_ids=ids)
    finally:
        model.eval()


def test_hard_fail_input_arg_confusion(small_model):
    model, cfg, ids = small_model
    with pytest.raises(ValueError, match="exactly one"):
        model(input_ids=ids, inputs_embeds=torch.zeros(1, 8, cfg.hidden_size))
    with pytest.raises(ValueError, match="exactly one"):
        model()


def test_hard_fail_missing_weight():
    ours = H.load_our_modules()
    cfg = H.build_our_config(H.skew10_config_dict())
    model = ours.model.KimiK3ForCausalLM(cfg, kda_backend="reference")
    sd, _ = H.make_state_dicts(H.named_shapes_of(model), cfg.kda_num_heads)
    sd.pop("model.layers.0.self_attn.A_log")
    with pytest.raises(RuntimeError, match="Missing"):
        model.load_state_dict(sd, strict=True)


def test_hard_fail_unknown_config_key():
    cfg_dict = H.syn25_config_dict()
    cfg_dict["text_config"]["banana_field"] = 1
    with pytest.raises(ValueError, match="banana_field"):
        H.build_our_config(cfg_dict)
    cfg_dict = H.syn25_config_dict()
    cfg_dict["surprise_top_level"] = {}
    with pytest.raises(ValueError, match="surprise_top_level"):
        H.build_our_config(cfg_dict)
    cfg_dict = H.syn25_config_dict()
    cfg_dict["text_config"]["linear_attn_config"]["mystery_knob"] = 3
    with pytest.raises(ValueError, match="mystery_knob"):
        H.build_our_config(cfg_dict)


def test_hard_fail_config_invariants():
    cfg_dict = H.syn25_config_dict()
    del cfg_dict["text_config"]["routed_expert_hidden_size"]
    with pytest.raises(ValueError, match="routed_expert_hidden_size"):
        H.build_our_config(cfg_dict)

    cfg_dict = H.syn25_config_dict()
    cfg_dict["text_config"]["num_expert_group"] = 2
    with pytest.raises(ValueError, match="num_expert_group"):
        H.build_our_config(cfg_dict)

    cfg_dict = H.syn25_config_dict()
    cfg_dict["text_config"]["num_nextn_predict_layers"] = 1
    with pytest.raises(NotImplementedError, match="MTP"):
        H.build_our_config(cfg_dict)

    with pytest.raises(ValueError, match="text_config"):
        H.build_our_config(H.syn25_config_dict()["text_config"])

    cfg_dict = H.syn25_config_dict()
    cfg_dict["text_config"]["linear_attn_config"]["kda_layers"] = list(range(0, 22))
    with pytest.raises(ValueError, match="partition"):
        H.build_our_config(cfg_dict)
