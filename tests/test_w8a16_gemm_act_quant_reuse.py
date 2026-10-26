# ---------------------------------------------------------------------------- #
#  BatchGen                                                                    #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""CPU contract for ``w8a16_gemm``'s pre-quantized-activation seam.

``w8a16_gemm`` used to run ``act_quant(x)`` unconditionally, so a caller that
fed ONE activation to several GEMMs quantized it once per GEMM.  The dense
GLM-5 MLP is the worst case: ``Glm5MLP._forward_slice``'s gate and up read the
same ``x``, so every dense layer paid a second, byte-for-byte identical
quantize every decode step.  ``w8a16_gemm`` now takes that pair through
``x_fp8=``, and ``w8a16_gemm_act_quant`` is the one definition of what it
quantizes.

What is pinned here:

  A  The ``x_fp8`` path and the internal path hand DeepGEMM the SAME bytes,
     and a pair that is already 2D reaches DeepGEMM as the very object the
     caller built -- not a copy, not a re-quantize.  Together with
     ``act_quant`` being deterministic, that is the byte-identity argument:
     the two paths are the same GEMM, so the change needs no opt-in switch.
  B  Each known duplicate call site quantizes its shared activation exactly
     once and hands the pair to both GEMMs (AST pins, so a later edit that
     re-introduces a second quantize fails here rather than silently costing
     a launch).  Sites whose activations are DIFFERENT tensors keep their own
     quantize and are pinned as such.
  C  A malformed or mismatched pair fails loud instead of quietly computing a
     different GEMM.

``fa3_backend`` is imported with stubbed GPU-only dependencies so this runs on
the dev box; no kernel body executes.  Numerical equivalence against the
pre-change path on real hardware is a GPU gate, not this file.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
import types

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _stub_missing(name: str, build: callable) -> list[str]:
    """Install ``build()`` at ``sys.modules[name]`` only if ``name`` is absent.

    On a GPU box the real module imports and is left completely alone; on the
    dev box the stub stands in for it.  Returns the names this call inserted
    so the caller can take them back out of ``sys.modules``.
    """
    if name in sys.modules:
        return []
    try:
        importlib.import_module(name)
    except Exception:
        pass
    else:
        return []
    sys.modules[name] = build()
    return [name]


def _import_fa3_backend() -> types.ModuleType:
    """Import the real ``fa3_backend`` module, stubbing GPU-only deps.

    The stub entries are removed from ``sys.modules`` afterwards so they
    cannot leak into another test module; ``fa3_backend`` already holds its
    own references to what it imported, and nothing in this file executes a
    Triton or DeepGEMM body.
    """
    inserted: list[str] = []

    def _triton() -> types.ModuleType:
        module = types.ModuleType("triton")
        module.jit = lambda fn: fn
        module.Config = object
        language = types.ModuleType("triton.language")
        language.constexpr = object
        module.language = language
        sys.modules.setdefault("triton.language", language)
        inserted.append("triton.language")
        return module

    def _flash_attn() -> types.ModuleType:
        module = types.ModuleType("flash_attn_interface")
        module.flash_attn_varlen_func = lambda *a, **k: None
        return module

    def _deep_gemm() -> types.ModuleType:
        module = types.ModuleType("deep_gemm")
        module.fp8_gemm_nt = lambda *a, **k: None
        return module

    def _fused_dequant_gemm() -> types.ModuleType:
        module = types.ModuleType("batchgen.moe.fused_dequant_gemm")
        module.fused_fp8_bf16_gemm = lambda *a, **k: None
        return module

    inserted += _stub_missing("triton", _triton)
    inserted += _stub_missing("flash_attn_interface", _flash_attn)
    inserted += _stub_missing("deep_gemm", _deep_gemm)
    inserted += _stub_missing("batchgen.moe.fused_dequant_gemm", _fused_dequant_gemm)
    try:
        return importlib.import_module("batchgen.attention.mla.fa3_backend")
    finally:
        for name in inserted:
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def backend() -> types.ModuleType:
    return _import_fa3_backend()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

BLOCK = 128
M, K, N = 6, 256, 32
NUM_BLOCKS = K // BLOCK


def _activation(shape=(M, K)) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(*shape, dtype=torch.bfloat16)


def _weight() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(12)
    weight = torch.randn(N, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale = torch.randn(N, NUM_BLOCKS, dtype=torch.float32)
    return weight, scale


def _install_recorders(backend, monkeypatch):
    """Stub ``act_quant`` deterministically and record DeepGEMM's inputs.

    The fake quantize derives its output from x, so a pair reused across two
    GEMMs and a pair re-quantized per GEMM are comparable byte for byte.
    """
    quantized_inputs: list[torch.Tensor] = []
    gemm_calls: list[tuple] = []

    def fake_act_quant(x, block_size=BLOCK, **_kwargs):
        """Deterministic stand-in that keeps the real act_quant's shapes.

        The real kernel returns the fp8 tensor in the INPUT's shape and the
        scale in ``(*shape[:-1], num_blocks)``, so a 3D activation yields a 3D
        pair.  Mirroring that is what makes the 3D case below exercise
        ``w8a16_gemm``'s view-to-2D of a caller-supplied pair.
        """
        quantized_inputs.append(x)
        flat = x.reshape(-1, x.shape[-1])
        rows, cols = flat.shape
        blocks = (cols + block_size - 1) // block_size
        fp8 = (flat * 3.0).to(torch.float8_e4m3fn)
        scale = flat.float().reshape(rows, blocks, -1).abs().amax(dim=-1) + 1.0
        return fp8.view(*x.shape), scale.view(*x.shape[:-1], blocks)

    monkeypatch.setattr(backend, "act_quant", fake_act_quant)
    monkeypatch.setattr(
        backend.deep_gemm,
        "fp8_gemm_nt",
        lambda x_fp8, y_fp8, out, *a, **k: gemm_calls.append((x_fp8, y_fp8, out)),
    )
    return quantized_inputs, gemm_calls


# --------------------------------------------------------------------------- #
# A. Byte identity of the pair path                                           #
# --------------------------------------------------------------------------- #


def test_pair_path_hands_deepgemm_the_same_bytes_as_the_internal_path(
    backend, monkeypatch
):
    """A: x_fp8= and x_fp8=None reach DeepGEMM with identical inputs."""
    x = _activation()
    weight, weight_scale = _weight()
    quantized_inputs, gemm_calls = _install_recorders(backend, monkeypatch)

    backend.w8a16_gemm(weight, weight_scale, x)
    pair = backend.w8a16_gemm_act_quant(x)
    backend.w8a16_gemm(weight, weight_scale, x, x_fp8=pair)

    assert len(gemm_calls) == 2
    internal_fp8, internal_scale = gemm_calls[0][0]
    reused_fp8, reused_scale = gemm_calls[1][0]
    assert internal_fp8.shape == reused_fp8.shape == (M, K)
    assert internal_scale.shape == reused_scale.shape == (M, NUM_BLOCKS)
    assert torch.equal(internal_fp8, reused_fp8)
    assert torch.equal(internal_scale, reused_scale)
    # The hoisted helper quantized exactly the tensor the GEMM would have.
    assert torch.equal(quantized_inputs[0], quantized_inputs[1])


def test_already_2d_pair_reaches_deepgemm_as_the_caller_built_it(backend, monkeypatch):
    """A: validation checks the pair, it does not copy or re-lay it out."""
    x = _activation()
    weight, weight_scale = _weight()
    _, gemm_calls = _install_recorders(backend, monkeypatch)

    pair = backend.w8a16_gemm_act_quant(x)
    backend.w8a16_gemm(weight, weight_scale, x, x_fp8=pair)

    handed_fp8, handed_scale = gemm_calls[0][0]
    assert handed_fp8 is pair[0]
    assert handed_scale is pair[1]


def test_gate_up_pattern_quantizes_the_shared_activation_once(backend, monkeypatch):
    """A: the hoist is what removes the launch, measured as act_quant calls."""
    x = _activation()
    weight, weight_scale = _weight()
    quantized_inputs, gemm_calls = _install_recorders(backend, monkeypatch)

    # Today's dense-MLP shape: gate and up over one x, then down over another.
    pair = backend.w8a16_gemm_act_quant(x)
    backend.w8a16_gemm(weight, weight_scale, x, x_fp8=pair)
    backend.w8a16_gemm(weight, weight_scale, x, x_fp8=pair)
    intermediate = torch.randn(M, K, dtype=torch.bfloat16)
    backend.w8a16_gemm(weight, weight_scale, intermediate)

    assert len(gemm_calls) == 3
    # One quantize for the shared x, one for down's distinct activation.
    assert len(quantized_inputs) == 2
    # Both of gate/up saw the same pair object.
    assert gemm_calls[0][0][0] is gemm_calls[1][0][0]
    assert gemm_calls[2][0][0] is not gemm_calls[0][0][0]


def test_unhoisted_same_x_still_quantizes_twice(backend, monkeypatch):
    """A: the baseline this change removes, so the pin above means something."""
    x = _activation()
    weight, weight_scale = _weight()
    quantized_inputs, _ = _install_recorders(backend, monkeypatch)

    backend.w8a16_gemm(weight, weight_scale, x)
    backend.w8a16_gemm(weight, weight_scale, x)

    assert len(quantized_inputs) == 2


def test_pair_from_unflattened_3d_activation_matches_internal_path(
    backend, monkeypatch
):
    """A: a 3D activation's pair is viewed to 2D, never reshaped or copied."""
    x = _activation((2, 3, K))
    weight, weight_scale = _weight()
    _, gemm_calls = _install_recorders(backend, monkeypatch)

    backend.w8a16_gemm(weight, weight_scale, x)
    fp8, scale = backend.act_quant(x)  # the un-flattened shape a caller may hold
    backend.w8a16_gemm(weight, weight_scale, x, x_fp8=(fp8, scale))

    internal_fp8, internal_scale = gemm_calls[0][0]
    reused_fp8, reused_scale = gemm_calls[1][0]
    assert reused_fp8.shape == (6, K)
    assert reused_scale.shape == (6, NUM_BLOCKS)
    assert torch.equal(internal_fp8, reused_fp8)
    assert torch.equal(internal_scale, reused_scale)


def test_act_quant_helper_matches_the_gemms_own_flattening(backend, monkeypatch):
    """A: the helper IS the GEMM's quantize, not a parallel reimplementation."""
    x = _activation((2, 3, K))
    quantized_inputs, _ = _install_recorders(backend, monkeypatch)

    backend.w8a16_gemm_act_quant(x)

    assert len(quantized_inputs) == 1
    assert quantized_inputs[0].shape == (6, K)
    assert torch.equal(quantized_inputs[0], x.view(-1, K))


def test_docstring_records_the_w8a8_naming_decision(backend):
    """The name says w8a16 while the helper is w8a8; that must be written down."""
    doc = backend.w8a16_gemm.__doc__ or ""
    assert "w8a8" in doc
    assert "w8a16_gemm_dequant" in doc


# --------------------------------------------------------------------------- #
# C. Fail-loud validation                                                     #
# --------------------------------------------------------------------------- #


def _gemm_with_pair(backend, pair, activation=None):
    weight, weight_scale = _weight()
    x = _activation() if activation is None else activation
    return backend.w8a16_gemm(weight, weight_scale, x, x_fp8=pair)


def _good_pair() -> tuple[torch.Tensor, torch.Tensor]:
    fp8 = torch.zeros(M, K, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(M, NUM_BLOCKS, dtype=torch.float32)
    return fp8, scale


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(torch.zeros(M, K, dtype=torch.float8_e4m3fn), id="bare-tensor"),
        pytest.param("x_fp8", id="string"),
        pytest.param((torch.zeros(M, K, dtype=torch.float8_e4m3fn),), id="length-1"),
        pytest.param(
            (
                torch.zeros(M, K, dtype=torch.float8_e4m3fn),
                torch.zeros(M, NUM_BLOCKS),
                torch.zeros(1),
            ),
            id="length-3",
        ),
    ],
)
def test_pair_must_be_a_two_element_pair(backend, monkeypatch, bad):
    _install_recorders(backend, monkeypatch)
    with pytest.raises(TypeError, match="2-tuple"):
        _gemm_with_pair(backend, bad)


def test_pair_members_must_be_tensors(backend, monkeypatch):
    _install_recorders(backend, monkeypatch)
    with pytest.raises(TypeError, match="pair of tensors"):
        _gemm_with_pair(backend, (torch.zeros(M, K, dtype=torch.float8_e4m3fn), 1.0))


def test_quantized_member_must_be_fp8(backend, monkeypatch):
    _install_recorders(backend, monkeypatch)
    _, scale = _good_pair()
    with pytest.raises(TypeError, match=r"x_fp8\[0\] must be float8_e4m3fn"):
        _gemm_with_pair(backend, (torch.zeros(M, K, dtype=torch.bfloat16), scale))


def test_scale_member_must_be_fp32(backend, monkeypatch):
    _install_recorders(backend, monkeypatch)
    fp8, _ = _good_pair()
    with pytest.raises(TypeError, match=r"x_fp8\[1\] must be float32"):
        _gemm_with_pair(
            backend, (fp8, torch.zeros(M, NUM_BLOCKS, dtype=torch.bfloat16))
        )


def test_pair_must_be_on_the_activations_device(backend, monkeypatch):
    _install_recorders(backend, monkeypatch)
    fp8 = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device="meta")
    scale = torch.zeros(M, NUM_BLOCKS, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="activation's device"):
        _gemm_with_pair(backend, (fp8, scale))


def test_pair_quantized_from_a_different_activation_is_rejected(backend, monkeypatch):
    """The decisive case: a stale pair would silently compute another GEMM."""
    _install_recorders(backend, monkeypatch)
    _, scale = _good_pair()
    wrong = torch.zeros(M + 1, K, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=r"x_fp8\[0\] has \d+ elements"):
        _gemm_with_pair(backend, (wrong, scale))


def test_scale_with_the_wrong_block_count_is_rejected(backend, monkeypatch):
    _install_recorders(backend, monkeypatch)
    fp8, _ = _good_pair()
    wrong = torch.zeros(M, NUM_BLOCKS + 3, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"x_fp8\[1\] has \d+ elements"):
        _gemm_with_pair(backend, (fp8, wrong))


def test_unviewable_layout_is_rejected_rather_than_copied(backend, monkeypatch):
    """Right element count, wrong shape, strides a view cannot satisfy."""
    _install_recorders(backend, monkeypatch)
    _, scale = _good_pair()
    transposed = torch.zeros(M, K, dtype=torch.float8_e4m3fn).t()
    assert tuple(transposed.shape) == (K, M)
    with pytest.raises(ValueError, match="cannot be viewed as"):
        _gemm_with_pair(backend, (transposed, scale))


def test_act_quant_helper_rejects_unsupported_rank(backend, monkeypatch):
    _install_recorders(backend, monkeypatch)
    with pytest.raises(ValueError, match="2D .* or 3D"):
        backend.w8a16_gemm_act_quant(torch.zeros(2, 2, 2, K, dtype=torch.bfloat16))


# --------------------------------------------------------------------------- #
# B. Call-site AST pins                                                       #
# --------------------------------------------------------------------------- #

QUANT_HELPER = "w8a16_gemm_act_quant"
GEMM = "w8a16_gemm"

# (file, qualified target, expected total w8a16_gemm calls, expected hoisted calls)
FOLDED_SITES = [
    ("batchgen/models/glm/glm5/model.py", "Glm5MLP._forward_slice", 3, 2),
    ("batchgen/models/glm/glm5/model.py", "Glm5Expert.deepgemm_forward", 3, 2),
    (
        "batchgen/models/glm/glm5/wrappers.py",
        "GLM5ExpertWrapper._forward_unfused",
        3,
        2,
    ),
    (
        "batchgen/models/minimax/minimax_m25/model.py",
        "MiniMaxM25Expert.deepgemm_forward",
        3,
        2,
    ),
    (
        "batchgen/models/minimax/minimax_m25/wrappers.py",
        "MiniMaxM25ExpertWrapper.forward",
        3,
        2,
    ),
    (
        "batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py",
        "DeepseekV3MLP.deepgemm_forward",
        3,
        2,
    ),
    (
        "batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py",
        "DeepseekV3MoE_Decoding_FP8.single_expert_forward",
        3,
        2,
    ),
    (
        "batchgen/models/moonshotai/kimi_k25/assets/modeling_deepseek.py",
        "DeepseekV3MLP.deepgemm_forward",
        3,
        2,
    ),
    (
        "batchgen/models/moonshotai/kimi_k25/assets/modeling_deepseek.py",
        "DeepseekV3MoE_Decoding_FP8.single_expert_forward",
        3,
        2,
    ),
]

# Sites that call the GEMM once, so there is no second quantize to remove:
# they must keep their own per-call quantize and must NOT grow a pair.
UNFOLDED_SITES = [
    ("batchgen/models/glm/glm5/model.py", "Glm5Indexer.compute_indexer_kv"),
    ("batchgen/models/glm/glm5/model.py", "Glm5Indexer.score_and_select"),
    (
        "batchgen/models/glm/glm5/model.py",
        "Glm5Indexer.score_and_select_relu_gated",
    ),
]


def _module_ast(relative_path: str) -> ast.Module:
    return ast.parse((_REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _find_function(tree: ast.Module, qualified: str) -> ast.FunctionDef:
    """Locate ``Class.method``/``function`` by walking the real nesting."""
    wanted = qualified.split(".")

    def walk(nodes, trail):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                here = trail + [node.name]
                if here[-len(wanted) :] == wanted and isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    return node
                found = walk(node.body, here)
                if found is not None:
                    return found
        return None

    found = walk(tree.body, [])
    assert found is not None, f"{qualified} not found"
    return found


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(fn: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and _called_name(node) == name
    ]


@pytest.mark.parametrize(
    "relative_path,qualified,total_gemms,hoisted",
    FOLDED_SITES,
    ids=[f"{p.split('/')[-1]}:{q}" for p, q, _t, _h in FOLDED_SITES],
)
def test_duplicate_call_site_quantizes_its_shared_activation_once(
    relative_path, qualified, total_gemms, hoisted
):
    """B: one hoisted quantize, handed to every GEMM that reads that tensor."""
    fn = _find_function(_module_ast(relative_path), qualified)

    quant_calls = _calls(fn, QUANT_HELPER)
    assert len(quant_calls) == 1, (
        f"{relative_path}:{qualified} must quantize its shared activation "
        f"exactly once, found {len(quant_calls)} {QUANT_HELPER} calls"
    )

    # The hoisted pair is bound to a single name ...
    assignment = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(c, ast.Call) and _called_name(c) == QUANT_HELPER
            for c in ast.walk(node.value)
        )
    )
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    pair_name = assignment.targets[0].id

    # ... and the activation it quantizes is what the hoisted GEMMs pass.
    quantized_arg = quant_calls[0].args[0]
    assert isinstance(quantized_arg, ast.Name)
    shared_activation = quantized_arg.id

    gemm_calls = _calls(fn, GEMM)
    assert len(gemm_calls) == total_gemms

    with_pair = []
    for call in gemm_calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        if "x_fp8" not in keywords:
            continue
        value = keywords["x_fp8"]
        assert isinstance(value, ast.Name) and value.id == pair_name, (
            f"{relative_path}:{qualified} passes an x_fp8 that is not the "
            f"hoisted {pair_name}"
        )
        activation = call.args[2] if len(call.args) > 2 else None
        assert isinstance(activation, ast.Name), (
            f"{relative_path}:{qualified} hands {pair_name} to a GEMM whose "
            "activation is not a plain name -- cannot prove it is the same x"
        )
        assert activation.id == shared_activation, (
            f"{relative_path}:{qualified} hands {pair_name} to a GEMM over "
            f"{activation.id}, but the pair was quantized from "
            f"{shared_activation}"
        )
        with_pair.append(call)

    assert len(with_pair) == hoisted, (
        f"{relative_path}:{qualified} should hand the hoisted pair to "
        f"{hoisted} GEMMs, found {len(with_pair)}"
    )


@pytest.mark.parametrize(
    "relative_path,qualified",
    UNFOLDED_SITES,
    ids=[f"{p.split('/')[-1]}:{q}" for p, q in UNFOLDED_SITES],
)
def test_single_gemm_call_sites_keep_their_own_quantize(relative_path, qualified):
    """B: a lone GEMM has nothing to share, so it must not grow a pair."""
    fn = _find_function(_module_ast(relative_path), qualified)
    assert len(_calls(fn, GEMM)) == 1
    assert _calls(fn, QUANT_HELPER) == []
    assert all("x_fp8" not in {kw.arg for kw in c.keywords} for c in _calls(fn, GEMM))
