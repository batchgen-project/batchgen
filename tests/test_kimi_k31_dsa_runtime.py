# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""The native code the Kimi-K3.1 DSA path loads is declared, built and preloaded.

``kimi_linear/dsa_runtime.py`` is the model-side list the startup seams read:
the ``kimi_k31`` runtime-preflight contract and the parallel strategy
manager's ``preload_runtime_extensions`` (run by the pre-readiness startup).
This module holds that list to the code, from both ends:

* every extension a route-B DSA seam resolves is in the list — traced through
  the source, statically (the ``batchgen_kernels.attention.dsa`` package
  imports triton, so it cannot be imported on a CPU box);
* every extension in the list is in the unconditional AOT build
  (``batchgen_kernels/setup.py``) and the dev-JIT registry;
* the preload is AOT-only (a plain import; a missing build raises before
  readiness) and checks every third-party callable the DSA paths call;
* the manager runs it for a DSA model and nothing for a dense one.

GPU-deferred: that the built extensions import and run on the target arch.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batchgen.attention.mla import sparse_dispatch  # noqa: E402
from batchgen.models.moonshotai.kimi_linear import dsa_runtime  # noqa: E402

KERNELS = ROOT / "batchgen_kernels"
DSA = KERNELS / "attention" / "dsa"
KIMI = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
SEAM = ROOT / "batchgen" / "attention" / "mla" / "sparse_dispatch.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _imports(function: ast.FunctionDef):
    return [(n.module, [a.name for a in n.names])
            for n in ast.walk(function) if isinstance(n, ast.ImportFrom)]


def _string_constants(tree: ast.Module):
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


# --------------------------------------------------------------------------- #
#  Code -> declaration                                                        #
# --------------------------------------------------------------------------- #

def test_the_decode_writer_seam_is_backed_by_the_declared_writer_extension():
    seam = _function(_tree(KIMI / "dsa_decode.py"), "_fused_index_k_writer")
    assert _imports(seam) == [(
        "batchgen_kernels.attention.dsa.indexer.indexer_k_norm_rope_write_fp8",
        ["fused_indexer_k_norm_rope_write_fp8"],
    )]
    writer = _tree(DSA / "indexer" / "indexer_k_norm_rope_write_fp8.py")
    assert dsa_runtime.FUSED_INDEX_K_WRITER_EXTENSION in _string_constants(writer)


def test_both_scorers_end_in_the_declared_fast_topk_extension():
    decode = _function(_tree(KIMI / "dsa_decode.py"), "_paged_scorer")
    graph = _function(_tree(KIMI / "dsa_decode.py"), "_paged_scorer_out")
    assert [m for m, _ in _imports(decode)] == [
        "batchgen_kernels.attention.dsa.paged_mqa_scorer"]
    assert [m for m, _ in _imports(graph)] == [
        "batchgen_kernels.attention.dsa.paged_mqa_scorer"]
    scorer = _tree(DSA / "paged_mqa_scorer.py")
    assert ("batchgen_kernels.attention.dsa.fast_topk_cuda", ["fast_topk_2048_out"]) in [
        (n.module, [a.name for a in n.names])
        for n in scorer.body if isinstance(n, ast.ImportFrom)
    ]
    prefill = _function(_tree(KIMI / "dsa_sparse_prefill.py"), "_score_and_select_prefill")
    assert ("batchgen_kernels.attention.dsa.fast_topk_cuda", ["fast_topk_2048_out"]) in (
        _imports(prefill))
    fast_topk = _tree(DSA / "fast_topk_cuda.py")
    assert dsa_runtime.FAST_TOPK_EXTENSION in _string_constants(fast_topk)


def test_the_third_party_callables_the_dsa_paths_call_are_declared():
    scorer = (DSA / "paged_mqa_scorer.py").read_text()
    for symbol in ("fp8_paged_mqa_logits", "get_paged_mqa_logits_metadata", "get_num_sms"):
        assert f"deep_gemm.{symbol}" in scorer
        assert symbol in dsa_runtime.K31_DSA_SITE_SYMBOLS["deep_gemm"]
    prefill = (KIMI / "dsa_sparse_prefill.py").read_text()
    assert "deep_gemm.fp8_mqa_logits" in prefill
    assert "fp8_mqa_logits" in dsa_runtime.K31_DSA_SITE_SYMBOLS["deep_gemm"]
    # Sparse attention is the seam's: its per-backend declaration names
    # exactly what each backend imports.
    seam_tree = _tree(SEAM)
    fa3 = _function(seam_tree, "_fa3_with_kvcache")
    assert _imports(fa3) == [("flash_attn_interface", ["flash_attn_with_kvcache"])]
    assert sparse_dispatch.SITE_SYMBOLS["fa3"] == {
        "flash_attn_interface": ("flash_attn_with_kvcache",)}
    # The Blackwell decode launch is the one the dense seam shares.
    decode = _function(seam_tree, "_trtllm_sparse_decode")
    assert "trtllm_gen_mla.trtllm_gen_mla_decode" in ast.unparse(decode)
    shared = _function(_tree(SEAM.parent / "trtllm_gen_mla.py"), "trtllm_gen_mla_decode")
    prefill_seam = _function(seam_tree, "_flashmla_sparse_prefill")
    assert _imports(shared) == [
        ("flashinfer.mla", ["trtllm_batch_decode_with_kv_cache_mla"])]
    assert _imports(prefill_seam) == [("flash_mla", ["flash_mla_sparse_fwd"])]
    assert sparse_dispatch.SITE_SYMBOLS["blackwell"] == {
        "flashinfer.mla": ("trtllm_batch_decode_with_kv_cache_mla",),
        "flash_mla": ("flash_mla_sparse_fwd",),
    }
    assert "flash_attn_interface" not in dsa_runtime.K31_DSA_SITE_SYMBOLS


def test_the_dsa_modules_reach_sparse_attention_only_through_the_seam():
    for name in ("dsa_decode.py", "dsa_sparse_prefill.py"):
        source = (KIMI / name).read_text()
        assert "flash_attn_interface" not in source, name
        assert "fa3_backend" not in source, name
        assert "from batchgen.attention.mla.sparse_dispatch import" in source, name


def test_the_retired_rotation_kernel_is_no_longer_declared():
    """Route B rotates nothing (identity RoPE, no Hadamard): the fused
    RoPE + Hadamard extension is not on the path and must not be required."""
    assert "batchgen_kernels.attention._C_fused_rope_hadamard" not in (
        dsa_runtime.K31_DSA_AOT_EXTENSIONS)


# --------------------------------------------------------------------------- #
#  Declaration -> build                                                       #
# --------------------------------------------------------------------------- #

def _setup_extension_names() -> dict:
    """``{extension name: list it is appended to}`` from batchgen_kernels/setup.py."""
    tree = _tree(KERNELS / "setup.py")
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.List):
            continue
        for element in node.value.elts:
            if (isinstance(element, ast.Call)
                    and getattr(element.func, "id", None) == "CUDAExtension"):
                for kw in element.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        found[kw.value.value] = target.id
    return found


def _unconditionally_built_lists() -> set:
    """Lists ``setup.py`` extends ``_ext_modules`` with OUTSIDE any ``if``."""
    tree = _tree(KERNELS / "setup.py")
    lists = set()
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "extend"
                and getattr(node.value.func.value, "id", None) == "_ext_modules"
                and isinstance(node.value.args[0], ast.Name)):
            lists.add(node.value.args[0].id)
    return lists


def _default_build_arch() -> str:
    tree = _tree(KERNELS / "setup.py")
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "_build_arch"):
            call = node.value
            assert ast.unparse(call.args[0]) == "'BUILD_ARCH'"
            return call.args[1].value
    raise AssertionError("_build_arch not found in setup.py")


@pytest.mark.parametrize("extension", dsa_runtime.K31_DSA_AOT_EXTENSIONS)
def test_every_declared_extension_is_in_the_default_aot_build(extension):
    """Built by a default (Hopper, ``BUILD_ARCH=sm90a``) kernel build. The
    index-key writer is Hopper-only — like GLM's DSA path, Kimi-K3.1's DSA
    serving needs an sm90a build; a ``BUILD_ARCH=sm100`` build is refused by
    the preflight contract rather than failing at the first DSA layer."""
    declared = _setup_extension_names()
    assert extension in declared, f"{extension} is not a CUDAExtension in setup.py"
    built_by_default = _unconditionally_built_lists() | (
        {"_sm90a_extensions"} if _default_build_arch() in ("sm90a", "all") else set())
    assert declared[extension] in built_by_default, (
        f"{extension} is not in the default build ({declared[extension]})")


@pytest.mark.parametrize("extension", dsa_runtime.K31_DSA_AOT_EXTENSIONS)
def test_every_declared_extension_is_in_the_jit_registry(extension):
    """The registry mirrors setup.py; a missing entry would make the dev-mode
    source pyroot unable to name the extension at all."""
    assert f'"{extension}"' in (KERNELS / "_jit_registry.py").read_text()


def test_the_k31_contract_covers_the_model_declared_dsa_extensions():
    """The model package declares what its DSA path loads
    (``kimi_linear/dsa_runtime.py``); the startup contract must check all of
    it, or a partial build passes preflight and fails at the first DSA layer."""
    from batchgen import runtime_preflight

    extensions = runtime_preflight._CONTRACTS["kimi_k31"].extensions
    for name in dsa_runtime.K31_DSA_AOT_EXTENSIONS:
        assert name in extensions
    assert "deep_gemm" in dsa_runtime.K31_DSA_SITE_SYMBOLS
    assert "flash_attn_interface" in runtime_preflight._CONTRACTS["kimi_k31"].modules


def test_the_blackwell_k31_contract_carries_the_dsa_runtime():
    """On 10.x the K3.1 contract swaps FA3 for the Blackwell sparse kernels
    but keeps every DSA extension and DeepGEMM."""
    from batchgen import runtime_preflight

    contract = runtime_preflight._contract_for_capability("kimi_k31", (10, 3))
    for name in dsa_runtime.K31_DSA_AOT_EXTENSIONS:
        assert name in contract.extensions
    assert contract.require_deepgemm
    assert set(contract.modules) == {"flash_mla", "flashinfer"}
    assert len(set(contract.extensions)) == len(contract.extensions)


def test_declaration_has_no_duplicates_and_uses_full_module_names():
    names = dsa_runtime.K31_DSA_AOT_EXTENSIONS
    assert len(set(names)) == len(names)
    assert all(n.startswith("batchgen_kernels.") and "._C_" in n for n in names)


# --------------------------------------------------------------------------- #
#  The preload                                                                #
# --------------------------------------------------------------------------- #

#: The Hopper seam's modules; the preload is handed them explicitly because
#: the default resolves the device's backend.
ATTN = sparse_dispatch.SITE_SYMBOLS["fa3"]


def _site_module(missing=()):
    return SimpleNamespace(**{
        s: (lambda *a, **k: None)
        for table in (dsa_runtime.K31_DSA_SITE_SYMBOLS, ATTN)
        for symbols in table.values()
        for s in symbols if s not in missing
    })


def test_the_preload_imports_everything_in_order():
    seen = []

    def importer(name):
        seen.append(name)
        return _site_module()

    resolved = dsa_runtime.preload_dsa_runtime(importer, ATTN)
    expected = (list(dsa_runtime.K31_DSA_AOT_EXTENSIONS)
                + list(dsa_runtime.K31_DSA_SITE_SYMBOLS) + list(ATTN))
    assert seen == expected and list(resolved) == expected


def test_a_missing_build_fails_the_preload_without_any_jit():
    def importer(name):
        if name == dsa_runtime.FAST_TOPK_EXTENSION:
            raise ImportError("no module named _C_dsa_fast_topk")
        return _site_module()

    with pytest.raises(dsa_runtime.DsaRuntimeError, match="_C_dsa_fast_topk is not built"):
        dsa_runtime.preload_dsa_runtime(importer, ATTN)


def test_the_preload_uses_a_plain_import_not_the_dev_jit_loader():
    source = (KIMI / "dsa_runtime.py").read_text()
    assert "load_extension" not in source
    tree = _tree(KIMI / "dsa_runtime.py")
    preload = _function(tree, "preload_dsa_runtime")
    default = preload.args.defaults[0]
    assert ast.unparse(default) == "importlib.import_module"


def test_a_missing_third_party_callable_fails_the_preload():
    def importer(name):
        return _site_module(missing=("get_paged_mqa_logits_metadata",))

    with pytest.raises(dsa_runtime.DsaRuntimeError,
                       match="deep_gemm.get_paged_mqa_logits_metadata"):
        dsa_runtime.preload_dsa_runtime(importer, ATTN)


def test_an_unimportable_third_party_module_fails_the_preload():
    def importer(name):
        if name == "deep_gemm":
            raise ImportError("no deep_gemm")
        return _site_module()

    with pytest.raises(dsa_runtime.DsaRuntimeError, match="needs deep_gemm"):
        dsa_runtime.preload_dsa_runtime(importer, ATTN)


# --------------------------------------------------------------------------- #
#  The manager hook                                                           #
# --------------------------------------------------------------------------- #

def _hook():
    """``KimiLinearParallelStrategyManager.preload_runtime_extensions`` compiled
    out of its module (the manager imports the native loader)."""
    import copy
    import logging

    tree = _tree(KIMI / "Parallel_Strategy_Manager.py")
    method = copy.deepcopy(_function(tree, "preload_runtime_extensions"))
    namespace = {"logging": logging, "__name__": "batchgen.models.moonshotai.kimi_linear.psm_hook",
                 "__package__": "batchgen.models.moonshotai.kimi_linear"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 "psm_hook", "exec"), namespace)
    return namespace["preload_runtime_extensions"]


def test_the_manager_preloads_the_dsa_runtime_for_a_dsa_model(monkeypatch):
    calls = []
    monkeypatch.setattr(dsa_runtime, "preload_dsa_runtime",
                        lambda: calls.append("dsa") or ("x",))
    manager = SimpleNamespace(rank=0, loaded_model_config=SimpleNamespace(dsa=lambda: object()))
    assert _hook()(manager) == ("x",)
    assert calls == ["dsa"]


@pytest.mark.parametrize("config", [
    SimpleNamespace(dsa=lambda: None),      # dense Kimi-K3
    SimpleNamespace(),                      # a config without the accessor
])
def test_the_manager_adds_nothing_for_a_dense_model(monkeypatch, config):
    monkeypatch.setattr(dsa_runtime, "preload_dsa_runtime",
                        lambda: pytest.fail("a dense model must not load DSA kernels"))
    assert _hook()(SimpleNamespace(rank=0, loaded_model_config=config)) == ()
