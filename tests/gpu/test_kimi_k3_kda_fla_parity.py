# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""STAGED GPU validation for Kimi-K3 M2 — run on h20-instance-1, GPU 0.

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

Launch (from the repo root ON instance-1; see run_kimi_k3_kda_instance1.sh):
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
        "silently skip. Check CUDA_VISIBLE_DEVICES / the driver on instance-1.")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="staged for h20-instance-1 GPU 0 (see tests/gpu/run_kimi_k3_kda_instance1.sh)")

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
    """Our KDA module's COMPOSITION vs the oracle's, on shared kernels.

    Like test_E, this comparison would otherwise straddle two equally-correct
    implementations of conv and gated-norm; chunk_kda's ~1e-5 perturbation
    floor turns that 1e-7 seam into a fail_frac of 3.3e-3 (measured before
    sharing). The kernels themselves are gated by test_B_conv_micro_parity /
    test_B_gated_norm_micro_parity; what is under test HERE is our wiring —
    projection order, the A_log slice, dt_bias layout, beta at the call site,
    the o_norm gate source. Sharing makes it bit-exact, which is a strictly
    stronger assertion than the gate it replaces.
    """
    ours_mods = H.load_our_modules()
    our_mod, oracle_mod, cfg = _build_kda_pair_gpu(H.syn25_config_dict(), "syn")
    _share_fla_kernels(our_mod, ours_mods)
    hidden = H.seeded_input("gpuB:h:syn", 2, 333, cfg.hidden_size,
                            dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        ours = our_mod(hidden)
        ref = oracle_mod(hidden_states=hidden, attention_mask=None)
    assert torch.equal(ours, ref), (
        "KDA module GPU synthetic dims: not bit-identical on shared kernels; "
        "max_abs={:.3e}".format((ours.float() - ref.float()).abs().max().item()))


def test_B_kda_module_real_dims():
    """One layer at the TRUE K3 geometry: hidden 7168, 96 heads x 128, ~890 MB
    of bf16 weights per stack. Shared kernels, same reasoning as above."""
    ours_mods = H.load_our_modules()
    our_mod, oracle_mod, cfg = _build_kda_pair_gpu(H.real_config_dict(), "real")
    _share_fla_kernels(our_mod, ours_mods)
    hidden = H.seeded_input("gpuB:h:real", 1, 512, cfg.hidden_size,
                            dtype=torch.bfloat16).to(DEV)
    with torch.no_grad():
        ours = our_mod(hidden)
        ref = oracle_mod(hidden_states=hidden, attention_mask=None)
    assert torch.equal(ours, ref), (
        "KDA module GPU real dims (96x128): not bit-identical on shared "
        "kernels; max_abs={:.3e}".format(
            (ours.float() - ref.float()).abs().max().item()))


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

    # The bound is DERIVED, not a round number. The old `1 << 30` was
    # arbitrary and the true peak (1138 MiB) sat 8% over it, which says
    # nothing about correctness. Peak = output + N live fp32 chunk working
    # sets; measured N = 3.63 at chunk 1024 (2026-08-05, H20), and the
    # measured peak scales linearly with chunk_size (1138 / 665 / 446 MiB at
    # 1024 / 512 / 256), confirming the model. Allow N = 4.
    out_bytes = out.numel() * out.element_size()
    chunk_set = 1024 * (nb + 1) * hidden * 4
    limit = out_bytes + 4 * chunk_set
    assert transient <= limit, (
        "lean mixer transient {} B exceeds the derived bound {} B = output "
        "({}) + 4 fp32 chunk sets ({} each); the reference form would need "
        "~{} B for the fp32 cat alone".format(
            transient, limit, out_bytes, chunk_set, T * (nb + 1) * hidden * 4))

    # The property that actually matters is that the (T, nb+1, H) tensor is
    # never built, i.e. the NON-OUTPUT transient does not grow with T. A
    # magnitude bound cannot distinguish "chunked" from "materialized but
    # small T"; this can. Reverting to the reference form makes the
    # non-output transient scale with T and trips this immediately.
    prefix2 = torch.cat([prefix, prefix], dim=0)
    block2 = torch.cat([block, block], dim=0)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base2 = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out2 = ours.model._apply_attn_res_lean(prefix2, block2, proj, norm,
                                               chunk_size=1024)
    torch.cuda.synchronize()
    transient2 = torch.cuda.max_memory_allocated() - base2
    scratch = transient - out_bytes
    scratch2 = transient2 - out2.numel() * out2.element_size()
    print("\n[D] non-output transient: T={} -> {:.1f} MiB, T={} -> {:.1f} MiB"
          .format(T, scratch / 1048576, 2 * T, scratch2 / 1048576))
    assert scratch2 <= 1.15 * scratch, (
        "non-output transient grew with T ({} B at T={} vs {} B at T={}): the "
        "mixer is materializing something proportional to sequence length"
        .format(scratch2, 2 * T, scratch, T))


# --------------------------------------------------------------------------- #
#  E — full K3-SYN-25 model on GPU                                              #
# --------------------------------------------------------------------------- #
def _share_fla_kernels(model, ours):
    """Make OUR stack use fla's conv / gated-norm kernels, sharing weights.

    test_E exists to check COMPOSITION — layer map, Block Attention Residuals,
    MLA, LatentMoE, router, SiTU, dtype policy.  It is not a test of which
    conv kernel we use.  Without this, the comparison necessarily straddles
    two independent-but-equally-correct kernel implementations, and K3 turns
    that 1e-7 seam into an O(1) logit difference (see
    test_E_kernel_seam_amplification for the measured chain).  The pure-torch
    modules keep their own coverage in test_B_conv_micro_parity and
    test_B_gated_norm_micro_parity, where they are compared against these very
    kernels at the module level — which is where that comparison belongs.
    """
    from fla.modules import FusedRMSNormGated, ShortConvolution
    m = ours.model
    swapped = [0, 0]
    for mod in model.modules():
        if isinstance(mod, m.CausalConv1dSilu):
            sc = ShortConvolution(mod.hidden_size, mod.kernel_size,
                                  activation="silu").to(mod.weight.device)
            sc.weight = mod.weight                     # SHARE, never copy
            mod._fla = sc
            mod.forward = (lambda x, _m=mod: _m._fla(
                x=x, cache=None, output_final_state=False, cu_seqlens=None)[0])
            swapped[0] += 1
        elif isinstance(mod, m.KimiGatedRMSNormSigmoid):
            fn = FusedRMSNormGated(mod.weight.shape[0], eps=mod.variance_epsilon,
                                   activation="sigmoid").to(mod.weight.device)
            fn.weight = mod.weight                     # SHARE, never copy
            mod._fla = fn
            mod.forward = lambda x, g, _m=mod: _m._fla(x, g)
            swapped[1] += 1
    assert swapped[0] and swapped[1], (
        "kernel sharing matched nothing ({} conv, {} gated-norm) — the module "
        "classes were renamed and this test silently reverted to comparing "
        "across kernel families".format(*swapped))
    print("\n[E] sharing fla kernels: {} conv, {} gated-norm".format(*swapped))


def test_E_kernel_seam_amplification():
    """Pin the MEASURED reason whole-model logit parity across kernel families
    is unattainable — so nobody 'fixes' a future failure by loosening a gate.

    Two independent amplifiers, both dtype-independent (this is NOT a bf16
    artifact — the seed exists at 5e-8 in pure fp32):

      1. `chunk_kda` has a perturbation-response FLOOR. Its output difference
         saturates near 1e-5 however small the input difference: measured
         gains 113x at eps 1e-7, 31x at 1e-6, 11x at 1e-5. The shrinking gain
         is the signature of a noise floor, not of ill-conditioning — the
         underlying math is well conditioned (an fp64 torch reference shows
         gain ~1.0).
      2. The top-16-of-64 sigmoid router is DISCONTINUOUS. 11-20 tokens per
         1024 sit within 1e-4 of the rank-16/rank-17 boundary, so a ~1e-5
         perturbation flips expert assignments; one flipped token accounted
         for essentially all of layer 1's MoE divergence (7.466e-3 of
         7.472e-3), and the chain runs away to 955/1024 tokens by layer 24.
    """
    ref = _kda_ref()
    q, k, v, g, beta, A_log, dt_bias = _kda_case(
        "seam", 2, 256, 8, 64, 64, dtype=torch.float32)

    # 1. the kernel's perturbation floor
    o0, _ = _run_real_chunk_kda(q, k, v, g, beta, A_log, dt_bias)
    eps = 1e-7
    qp = q * (1.0 + eps)
    o1, _ = _run_real_chunk_kda(qp, k, v, g, beta, A_log, dt_bias)
    rel_in = ((qp - q).pow(2).mean().sqrt() / q.pow(2).mean().sqrt()).item()
    rel_out = ((o1.float() - o0.float()).pow(2).mean().sqrt()
               / o0.float().pow(2).mean().sqrt()).item()
    gain = rel_out / rel_in
    print("\n[seam] chunk_kda: rel_in={:.2e} -> rel_out={:.2e}  gain={:.0f}x"
          .format(rel_in, rel_out, gain))
    assert gain > 10.0, (
        "chunk_kda no longer amplifies a {:.1e} input perturbation (gain {:.1f}x). "
        "If the kernel became perturbation-stable, test_E could compare across "
        "kernel families directly and _share_fla_kernels may be removable — "
        "re-measure before changing anything.".format(rel_in, gain))

    # 2. the router's discontinuity: how close tokens sit to the top-k edge
    cfg = H.build_our_config(H.syn25_config_dict())
    top = int(cfg.num_experts_per_token)
    torch.manual_seed(0)
    scores = torch.rand(1024, int(cfg.num_experts), device=DEV)
    srt = scores.sort(dim=-1, descending=True).values
    boundary_gap = (srt[:, top - 1] - srt[:, top]).abs()
    near = (boundary_gap < 1e-4).sum().item()
    print("[seam] router top-{} of {}: {} / 1024 tokens within 1e-4 of the "
          "rank-{}/{} boundary (min gap {:.2e})".format(
              top, int(cfg.num_experts), near, top, top + 1,
              boundary_gap.min().item()))
    # A ~1e-5 perturbation (amplifier 1's output floor) flips any token whose
    # boundary gap is below it, and one flip was enough to carry layer 1's
    # entire MoE divergence. Requiring only that the min gap is under 1e-4
    # keeps this robust: the claim is that the boundary is APPROACHED, not
    # that a fixed count sits there.
    assert boundary_gap.min().item() < 1e-4, (
        "closest token sits {:.2e} from the top-k boundary — far above the "
        "~1e-5 perturbation floor, so the router-flip half of the "
        "amplification story no longer reproduces; re-derive it before "
        "trusting any whole-model parity claim".format(
            boundary_gap.min().item()))



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
    # VERIFIED (2026-08-06), do not "re-fix" this: transformers prints
    # "Ignoring the provided attention implementation eager / Using
    # flash_attention_2 backend instead" during KimiLinearModel.__init__
    # (modeling_kimi_linear.py:1110-1117), which force-sets the field. That
    # warning is STALE OUTPUT -- the reset two lines above still takes effect,
    # because oracle_model.config, model.config and every layer's
    # self_attn.config are the SAME object. At forward time
    # _attn_implementation == "eager" on all handles, create_causal_mask
    # returns a real (B,1,T,T) mask, eager_attention_forward runs, and every
    # MLA module is bit-identical (0.000e+00) to ours.
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

    _share_fla_kernels(our_model, ours)

    ids = H.seeded_token_ids("gpuE", 2, 512, cfg.vocab_size,
                             cfg.media_placeholder_token_id).to(DEV)
    with torch.no_grad():
        ours_logits = our_model(input_ids=ids)
        ref_logits = oracle_model(input_ids=ids, use_cache=False).logits
    # TIGHTENED, not loosened: with the kernel seam removed the two stacks are
    # BIT-identical, so this is a stronger assertion than the bf16 gate it
    # replaces. Measured 2026-08-06 on H20: err_ratio 0.000000, top-1 100.00%,
    # max_abs 0.0, in BOTH fp32 and bf16.
    assert torch.equal(ours_logits, ref_logits), (
        "full model GPU: our stack and the oracle must be bit-identical once "
        "they share fla's conv/gated-norm kernels. err_ratio={:.6e}, top-1={:.4f}"
        .format(
            (ours_logits.float() - ref_logits.float()).pow(2).mean().sqrt().item()
            / (ref_logits.float().pow(2).mean().sqrt().item() + 1e-8),
            (ours_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()))
