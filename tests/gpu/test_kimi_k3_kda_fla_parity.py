# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""STAGED GPU validation for Kimi-K3 M2 — run on a CUDA GPU.

This is the closure for everything the CPU suite cannot see (the fla CPU shim
backs BOTH stacks there, so the kernel interior cancels):

  PRE  fla CAPABILITY probe: chunk_kda must NAME use_beta_sigmoid_in_kernel
       (a version-string pin is what misled this file before; fla < 0.5
       swallows the kwarg and consumes beta RAW).
  A    real triton ``chunk_kda`` (exact oracle flag set: l2norm/gate-in-kernel/
       safe_gate lower_bound=-5.0/transpose_state_layout) vs the vendored
       pure-torch composition (kda_reference.py), at synthetic dims, REAL dims
       (96 heads x 128), odd T, and one varlen case; plus naive_chunk vs
       naive_recurrent self-consistency; plus two flag-semantics assertions
       (sigmoid applied EXACTLY once, and the flag is honored -- not dead).
       Acceptance for kernel-vs-reference is fla's OWN err_ratio bar
       (RMS-relative < 0.005), not the per-element bf16 gate: see
       kimi_k3_harness.assert_kernel_err_ratio for the measured rationale.
  B    our KimiK3KDAAttention (kda_backend='fla_chunk') vs the vendored oracle
       KimiDeltaAttention running the REAL fla — synthetic and REAL dims; plus
       micro-parity of our pure-torch conv / gated-norm vs fla's
       ShortConvolution / FusedRMSNormGated.
  C    A_log zero-pad poison invariance against the real triton kernel.
  D    lean AttnRes mixer on GPU: < 1e-6 vs the verbatim reference + a peak-
       memory bound at T=16384, nb=8, H=7168.
  E    full K3-SYN-25 model, GPU bf16: ours (fla backend) vs the oracle
       (real fla, MLA forced to eager).

Launch (from the repo root on the GPU machine; see run_kimi_k3_kda.sh):
    K3_GPU_STAGE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest \
        tests/gpu/test_kimi_k3_kda_fla_parity.py -x -q -rA

On a machine without CUDA the file SKIPS (it is a staged artifact, the CPU
suite is the M2 gate) — unless K3_GPU_STAGE=1 is set, in which case a missing
GPU is a hard RuntimeError (launch-verification rule: no silent green).

IMPORTANT: never import tests/kimi_k3_harness.py's ``load_oracle_modules``
here — it installs the CPU fla shim into sys.modules, clobbering the real fla
this test exists to exercise.  The oracle is loaded by ``_load_oracle_gpu``
below, shim-free.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kimi_k3_harness as H  # noqa: E402


if os.environ.get("K3_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError(
        "K3_GPU_STAGE=1 but CUDA is unavailable — this staged run must not "
        "silently skip. Check CUDA_VISIBLE_DEVICES / the driver.")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="staged GPU validation (see tests/gpu/run_kimi_k3_kda.sh)")

DEV = "cuda"


# --------------------------------------------------------------------------- #
#  Shim-free oracle loading (REAL fla)                                          #
# --------------------------------------------------------------------------- #
_ORACLE_GPU = None


def _load_oracle_gpu():
    global _ORACLE_GPU
    if _ORACLE_GPU is not None:
        return _ORACLE_GPU
    H.check_oracle_pins()
    fla_mod = sys.modules.get("fla")
    if fla_mod is not None and "shim" in getattr(fla_mod, "__version__", ""):
        raise RuntimeError(
            "The CPU fla shim is installed in this process — the GPU test "
            "must run in a process that never imported the CPU harness "
            "oracle loader (load_oracle_modules)")
    import fla  # noqa: F401  (real fla must be importable here)
    H._stub_package("_k3_oracle_gpu", H.ORACLE_ASSETS_DIR)
    configuration = H._load_by_path(
        "_k3_oracle_gpu.configuration_kimi_k3",
        H.ORACLE_ASSETS_DIR / "configuration_kimi_k3.py")
    modeling = H._load_by_path(
        "_k3_oracle_gpu.modeling_kimi_linear",
        H.ORACLE_ASSETS_DIR / "modeling_kimi_linear.py")
    _ORACLE_GPU = (configuration, modeling)
    return _ORACLE_GPU


def _kda_ref():
    return H.load_our_modules().kda_reference


# --------------------------------------------------------------------------- #
#  PRE — environment pins                                                       #
# --------------------------------------------------------------------------- #
def test_pre_environment_pins():
    """Pin the CAPABILITY, not the version string.

    An earlier revision asserted ``fla.__version__ == "0.4.2"``, which was
    doubly wrong: 0.4.2 is the version whose chunk kernel silently consumes
    beta RAW, and the version string is what misled us in the first place
    (the vendored torch reference was labelled 0.4.2 while being 0.5.2).
    What the call contract actually needs is a kernel that NAMES
    ``use_beta_sigmoid_in_kernel`` — older ones swallow it via ``**kwargs``.
    """
    import fla
    import inspect
    from fla.ops.kda import chunk_kda
    assert "use_beta_sigmoid_in_kernel" in inspect.signature(chunk_kda).parameters, (
        "fla {} does not name use_beta_sigmoid_in_kernel: it swallows the "
        "kwarg and feeds RAW beta logits to the delta rule. Install "
        "fla-core >= 0.5.0.".format(fla.__version__))
    import transformers
    from packaging import version as _v
    assert _v.parse(transformers.__version__) >= _v.parse("4.56.0")
    print("\n[pins] fla={} torch={} cuda={} device={}".format(
        fla.__version__, torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0)))


# --------------------------------------------------------------------------- #
#  A — real chunk_kda vs the vendored torch composition                         #
# --------------------------------------------------------------------------- #
def _kda_case(tag, B, T, Hh, K, V, dtype=torch.bfloat16):
    q = H.seeded_input("gpuA:q:" + tag, B, T, Hh, K, dtype=dtype).to(DEV)
    k = H.seeded_input("gpuA:k:" + tag, B, T, Hh, K, dtype=dtype).to(DEV)
    v = H.seeded_input("gpuA:v:" + tag, B, T, Hh, V, dtype=dtype).to(DEV)
    g = H.seeded_input("gpuA:g:" + tag, B, T, Hh, K, dtype=dtype).to(DEV)
    beta = H.seeded_input("gpuA:b:" + tag, B, T, Hh).to(DEV)          # fp32 raw
    A_log = torch.log(
        H.seeded_input("gpuA:A:" + tag, Hh).abs() * 4 + 1.0).to(DEV)  # fp32 [H]
    dt_bias = H.seeded_input("gpuA:dt:" + tag, Hh * K).mul(0.5).to(DEV)
    return q, k, v, g, beta, A_log, dt_bias


def _run_real_chunk_kda(q, k, v, g, beta, A_log, dt_bias, **overrides):
    from fla.ops.kda import chunk_kda
    kwargs = dict(
        q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True, lower_bound=-5.0,
        transpose_state_layout=True, cu_seqlens=None,
    )
    kwargs.update(overrides)
    return chunk_kda(**kwargs)


@pytest.mark.parametrize("dims", [
    ("syn", 4, 256, 8, 64, 64),        # synthetic dims (exercises BS 32/64 autotune)
    ("real", 2, 1024, 96, 128, 128),   # the true K3 KDA geometry
    ("oddT", 2, 1023, 96, 128, 128),   # T not a multiple of the 64 chunk
])
def test_A_kernel_vs_reference(dims):
    tag, B, T, Hh, K, V = dims
    ref = _kda_ref()
    q, k, v, g, beta, A_log, dt_bias = _kda_case(tag, B, T, Hh, K, V)
    o_kernel, _ = _run_real_chunk_kda(q, k, v, g, beta, A_log, dt_bias)
    o_ref = ref.kda_reference_prefill(
        q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias,
        lower_bound=-5.0, use_qk_l2norm=True)
    H.assert_kernel_err_ratio(
        o_kernel, o_ref, "chunk_kda vs torch reference [{}]".format(tag))


def test_A_kernel_vs_reference_varlen():
    """cu_seqlens=[0,300,1024] on a flattened B=1: kernel varlen vs the naive
    reference run per segment and concatenated."""
    ref = _kda_ref()
    q, k, v, g, beta, A_log, dt_bias = _kda_case("varlen", 1, 1024, 8, 64, 64)
    cu = torch.tensor([0, 300, 1024], dtype=torch.int32, device=DEV)
    o_kernel, _ = _run_real_chunk_kda(q, k, v, g, beta, A_log, dt_bias, cu_seqlens=cu)
    parts = []
    for s, e in ((0, 300), (300, 1024)):
        parts.append(ref.kda_reference_prefill(
            q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
            A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0, use_qk_l2norm=True))
    o_ref = torch.cat(parts, dim=1)
    H.assert_kernel_err_ratio(
        o_kernel, o_ref, "chunk_kda varlen vs per-segment reference")


def test_A_naive_chunk_vs_recurrent_selfconsistency():
    """The two vendored torch forms agree (< 1e-4 fp32, T % 64 == 0) — the
    mutation-discipline check on the reference itself."""
    ref = _kda_ref()
    q, k, v, g, beta, A_log, dt_bias = _kda_case(
        "selfc", 2, 256, 8, 64, 64, dtype=torch.float32)
    ql, kl = ref.l2norm_ref(q), ref.l2norm_ref(k)
    beta_post = beta.sigmoid()
    g_log = ref.naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=-5.0)
    o_rec, _ = ref.naive_recurrent_kda(ql, kl, v, g_log, beta_post)
    o_chk, _ = ref.naive_chunk_kda(ql, kl, v, g_log, beta_post)
    diff = (o_rec - o_chk).abs().max().item()
    assert diff < 1e-4, "naive chunk vs recurrent drift {}".format(diff)


def test_A_beta_sigmoid_applied_exactly_once():
    """THE discriminating test for the beta convention.

    ``kernel(raw, flag=True)`` and ``kernel(sigmoid(raw), flag=False)`` agree
    if and only if sigmoid is applied to the raw logits exactly once — by the
    kernel on the left, by us on the right.  An fla that ignores the flag
    fails here, because the left side then consumes RAW beta while the right
    consumes the squashed one.

    Its predecessor asserted merely that raw and pre-sigmoided inputs give
    DIFFERENT outputs, which is true whether the kernel sigmoids or not; it
    could not detect the failure it existed to exclude.
    """
    q, k, v, g, beta, A_log, dt_bias = _kda_case("betasig", 2, 256, 8, 64, 64)
    o_kernel_side, _ = _run_real_chunk_kda(
        q, k, v, g, beta, A_log, dt_bias, use_beta_sigmoid_in_kernel=True)
    o_host_side, _ = _run_real_chunk_kda(
        q, k, v, g, beta.sigmoid(), A_log, dt_bias,
        use_beta_sigmoid_in_kernel=False)
    H.assert_bf16_gate(o_kernel_side, o_host_side,
                       "sigmoid(beta) in-kernel vs host-side (must be the "
                       "same single application)")


def test_A_use_beta_sigmoid_kwarg_is_honored():
    """The flag must MATTER.  If flipping it changes nothing, the kernel is
    ignoring it (fla < 0.5 swallows it via **kwargs) and our call site is
    silently feeding raw logits into the delta rule."""
    q, k, v, g, beta, A_log, dt_bias = _kda_case("livekw", 2, 256, 8, 64, 64)
    o_true, _ = _run_real_chunk_kda(q, k, v, g, beta, A_log, dt_bias,
                                    use_beta_sigmoid_in_kernel=True)
    o_false, _ = _run_real_chunk_kda(q, k, v, g, beta, A_log, dt_bias,
                                     use_beta_sigmoid_in_kernel=False)
    diff = (o_true.float() - o_false.float()).abs().max().item()
    assert diff > 1e-3, (
        "use_beta_sigmoid_in_kernel is inert (max diff {}): this fla ignores "
        "it and consumes beta RAW — the exact silent-wrong-numerics case the "
        "import guard exists to prevent".format(diff))


# --------------------------------------------------------------------------- #
#  B — module-level: our KDA (fla backend) vs the oracle on real fla            #
# --------------------------------------------------------------------------- #
def _build_kda_pair_gpu(cfg_dict, tag):
    ours = H.load_our_modules()
    configuration, modeling = _load_oracle_gpu()
    cfg = H.build_our_config(cfg_dict)
    ocfg = configuration.KimiLinearConfig(**cfg_dict["text_config"])
    ocfg._attn_implementation = "eager"
    our_mod = ours.model.KimiK3KDAAttention(cfg, layer_idx=0, kda_backend="fla_chunk")
    oracle_mod = modeling.KimiDeltaAttention(ocfg, layer_idx=0)
    ours_sd, oracle_sd = H.make_state_dicts(
        H.named_shapes_of(our_mod), cfg.kda_num_heads, prefix="gpuB:{}:".format(tag))
    our_mod.load_state_dict(ours_sd, strict=True)
    oracle_mod.load_state_dict(oracle_sd, strict=True)
    ours.model.cast_model_to_inference_dtype(our_mod, torch.bfloat16)
    H.cast_selective(oracle_mod, torch.bfloat16)
    return our_mod.to(DEV).eval(), oracle_mod.to(DEV).eval(), cfg


def test_B_kda_module_synthetic_dims():
    our_mod, oracle_mod, cfg = _build_kda_pair_gpu(H.syn25_config_dict(), "syn")
    hidden = H.seeded_input("gpuB:h:syn", 2, 333, cfg.hidden_size,
                            dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        ours = our_mod(hidden)
        ref = oracle_mod(hidden_states=hidden, attention_mask=None)
    H.assert_bf16_gate(ours, ref, "KDA module GPU synthetic dims")


def test_B_kda_module_real_dims():
    """One layer at the TRUE K3 geometry: hidden 7168, 96 heads x 128, ~890 MB
    of bf16 weights per stack."""
    our_mod, oracle_mod, cfg = _build_kda_pair_gpu(H.real_config_dict(), "real")
    hidden = H.seeded_input("gpuB:h:real", 1, 512, cfg.hidden_size,
                            dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        ours = our_mod(hidden)
        ref = oracle_mod(hidden_states=hidden, attention_mask=None)
    H.assert_bf16_gate(ours, ref, "KDA module GPU real dims (96x128)")


def test_B_conv_micro_parity():
    """Our pure-torch CausalConv1dSilu vs the real fla ShortConvolution —
    closes the CPU-shim seam for the conv."""
    from fla.modules import ShortConvolution
    ours = H.load_our_modules()
    D, W, B, T = 512, 4, 2, 333
    our_conv = ours.model.CausalConv1dSilu(D, W)
    fla_conv = ShortConvolution(hidden_size=D, kernel_size=W, activation="silu")
    w = H.seeded_master("gpuB:conv.q_conv1d.weight", (D, 1, W))
    with torch.no_grad():
        our_conv.weight.copy_(w)
        fla_conv.weight.copy_(w)
    our_conv = our_conv.to(DEV)
    fla_conv = fla_conv.to(DEV)   # fla keeps its own dtype handling
    x = H.seeded_input("gpuB:conv:x", B, T, D, dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        y_ours = our_conv(x)
        y_fla, _ = fla_conv(x)
    H.assert_bf16_gate(y_ours, y_fla, "CausalConv1dSilu vs fla ShortConvolution")


def test_B_gated_norm_micro_parity():
    """Our KimiGatedRMSNormSigmoid vs the real fla FusedRMSNormGated — closes
    the CPU-shim seam for o_norm."""
    from fla.modules import FusedRMSNormGated
    ours = H.load_our_modules()
    D, B, T, Hh = 128, 2, 64, 8
    our_norm = ours.model.KimiGatedRMSNormSigmoid(D, eps=1e-5)
    fla_norm = FusedRMSNormGated(D, eps=1e-5, activation="sigmoid")
    w = H.seeded_master("gpuB:o_norm.weight", (D,))
    with torch.no_grad():
        our_norm.weight.copy_(w)
        fla_norm.weight.copy_(w)
    our_norm = our_norm.to(DEV)
    fla_norm = fla_norm.to(DEV)
    x = H.seeded_input("gpuB:onorm:x", B, T, Hh, D, dtype=torch.bfloat16).to(DEV)
    g = H.seeded_input("gpuB:onorm:g", B, T, Hh, D, dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        y_ours = our_norm(x, g)
        y_fla = fla_norm(x, g)
    H.assert_bf16_gate(y_ours, y_fla, "KimiGatedRMSNormSigmoid vs fla FusedRMSNormGated")


# --------------------------------------------------------------------------- #
#  C — A_log pad poison against the real kernel                                 #
# --------------------------------------------------------------------------- #
def test_C_a_log_pad_poison_real_kernel():
    our_mod, _, cfg = _build_kda_pair_gpu(H.syn25_config_dict(), "poison")
    hidden = H.seeded_input("gpuC:h", 1, 128, cfg.hidden_size,
                            dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        baseline = our_mod(hidden)
        our_mod.A_log[cfg.kda_num_heads:] = 1e6
        poisoned = our_mod(hidden)
    assert torch.equal(baseline, poisoned), (
        "the real triton kernel consumed the A_log zero-pad region")


# --------------------------------------------------------------------------- #
#  D — lean mixer on GPU: equivalence + peak-memory bound                       #
# --------------------------------------------------------------------------- #
def test_D_lean_mixer_gpu():
    ours = H.load_our_modules()
    T, nb, hidden = 16384, 8, 7168
    norm = ours.model.KimiRMSNorm(hidden, eps=1e-5).to(DEV)
    proj = torch.nn.Linear(hidden, 1, bias=False).to(DEV)
    with torch.no_grad():
        norm.weight.copy_(H.seeded_master("gpuD:norm.weight", (hidden,)))
        proj.weight.copy_(H.seeded_master("gpuD:res_proj.weight", (1, hidden)))
    prefix32 = H.seeded_input("gpuD:prefix", T, hidden).to(DEV)
    block32 = H.seeded_input("gpuD:block", T, nb, hidden).to(DEV)
    with torch.no_grad():
        lean = ours.model._apply_attn_res_lean(prefix32, block32, proj, norm,
                                               chunk_size=1024)
        ref = H.apply_attn_res_reference(prefix32, block32, proj, norm)
    H.assert_fp32_tight(lean, ref, "lean mixer GPU fp32", tol=1e-6)

    prefix = prefix32.bfloat16()
    block = block32.bfloat16()
    del prefix32, block32, ref, lean
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = ours.model._apply_attn_res_lean(prefix, block, proj, norm,
                                              chunk_size=1024)
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - base
    assert out.shape == (T, hidden)
    limit = 1 << 30
    assert transient <= limit, (
        "lean mixer transient GPU memory {} B exceeds the 1 GiB bound "
        "(the reference form would need ~{} B for the fp32 cat alone)".format(
            transient, T * (nb + 1) * hidden * 4))


# --------------------------------------------------------------------------- #
#  E — full K3-SYN-25 model on GPU                                              #
# --------------------------------------------------------------------------- #
def test_E_full_model_gpu():
    ours = H.load_our_modules()
    configuration, modeling = _load_oracle_gpu()
    cfg_dict = H.syn25_config_dict()
    cfg = H.build_our_config(cfg_dict)
    our_model = ours.model.KimiK3ForCausalLM(cfg, kda_backend="fla_chunk")
    ocfg = configuration.KimiLinearConfig(**cfg_dict["text_config"])
    ocfg._attn_implementation = "eager"
    ocfg.use_cache = False
    oracle_model = modeling.KimiLinearForCausalLM(ocfg)
    oracle_model.config._attn_implementation = "eager"   # undo the fa2 force-set
    oracle_model.model._use_flash_attention_2 = False
    print("\n[E] oracle MLA path forced to eager (fp32-softmax reference); "
          "the FA2 path is exercised in serving, not in this parity gate")
    ours_sd, oracle_sd = H.make_state_dicts(
        H.named_shapes_of(our_model), cfg.kda_num_heads)
    our_model.load_state_dict(ours_sd, strict=True)
    oracle_model.load_state_dict(oracle_sd, strict=True)
    ours.model.cast_model_to_inference_dtype(our_model, torch.bfloat16)
    H.cast_selective(oracle_model, torch.bfloat16)
    our_model = our_model.to(DEV).eval()
    oracle_model = oracle_model.to(DEV).eval()

    ids = H.seeded_token_ids("gpuE", 2, 512, cfg.vocab_size,
                             cfg.media_placeholder_token_id).to(DEV)
    with torch.no_grad():
        ours_logits = our_model(input_ids=ids)
        ref_logits = oracle_model(input_ids=ids, use_cache=False).logits
    H.assert_bf16_gate(ours_logits, ref_logits, "full model GPU bf16")
    assert torch.equal(ours_logits.argmax(-1), ref_logits.argmax(-1)), (
        "top-1 disagreement between our model and the oracle on GPU")
