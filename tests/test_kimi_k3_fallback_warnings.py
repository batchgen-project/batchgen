"""Kimi-K3 fallback downgrades are never silent (silent-fallback audit 2026-11-26).

Pure-CPU: availability flags are monkeypatched; no triton/CUDA import is
required. The CUDA-only call sites (model.py, whole_model_cuda_graph_segments)
are held to the warning wiring by AST source contracts, since neither module
imports on a CPU-only machine.
"""
import ast
import logging
from pathlib import Path

import pytest

from batchgen.models.moonshotai.kimi_linear import (
    row_gather_triton,
    situ_triton,
)
from batchgen.models.moonshotai.kimi_linear import rmsnorm_triton

ROOT = Path(__file__).resolve().parents[1]
KIMI = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"


# --------------------------------------------------------------------- helpers

def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _calls_named(node, name):
    return [
        call for call in ast.walk(node)
        if isinstance(call, ast.Call) and (
            (isinstance(call.func, ast.Name) and call.func.id == name)
            or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
        )
    ]


# ------------------------------------------------- import-error introspection

def test_triton_import_error_is_consistent_with_availability():
    for module, available in (
        (rmsnorm_triton, rmsnorm_triton.triton_rmsnorm_available),
        (situ_triton, situ_triton.situ_triton_available),
        (row_gather_triton, row_gather_triton.row_gather_triton_available),
    ):
        if available():
            assert module._TRITON_IMPORT_ERROR is None
        else:
            assert isinstance(module._TRITON_IMPORT_ERROR, BaseException)


# --------------------------------------------------- warn-once helpers (C2/C3)

@pytest.mark.parametrize("module,helper,text", [
    (situ_triton, "warn_situ_eager_fallback_once", "situ_and_mul"),
    (row_gather_triton, "warn_row_gather_fallback_once", "row gathers"),
])
def test_fallback_warn_helpers_log_exactly_once(
    monkeypatch, caplog, module, helper, text
):
    monkeypatch.setattr(module, "_FALLBACK_WARNED", False)
    monkeypatch.setattr(
        module, "_TRITON_IMPORT_ERROR", ImportError("no triton wheel")
    )
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        getattr(module, helper)()
        getattr(module, helper)()
    warnings = [
        record for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert text in warnings[0].getMessage()
    assert "no triton wheel" in warnings[0].getMessage()


# ------------------------------------------------ CUDA call-site wiring (AST)

def test_fused_rmsnorm_ladder_warns_on_every_rung():
    """model.py:_fused_rmsnorm_op: the Triton rung (miss or raise) and the
    CUDA-extension rung each log a warning with the caught reason (C1)."""
    source = (KIMI / "model.py").read_text()
    node = _function(ast.parse(source), "_fused_rmsnorm_op")
    warnings = _calls_named(node, "warning")
    assert len(warnings) >= 3, "each resolution rung must warn on downgrade"
    segment = ast.get_source_segment(source, node)
    assert segment.count("Triton fused RMSNorm unavailable") == 2
    assert "fused RMSNorm extension unavailable" in segment


def test_situ_forward_warns_before_eager_on_cuda():
    source = (KIMI / "model.py").read_text()
    tree = ast.parse(source)
    klass = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "SituAndMul"
    )
    forward = _function(klass, "forward")
    assert _calls_named(forward, "warn_situ_eager_fallback_once")


def test_whole_model_graph_warns_on_row_gather_fallback():
    source = (KIMI / "whole_model_cuda_graph_segments.py").read_text()
    forwards = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    assert any(
        _calls_named(node, "_warn_row_gather_fallback_once")
        for node in forwards
    )


# ------------------------------------- k3_skip_kv_callback debug key (C4)

def test_skip_kv_callback_reads_debug_state_and_warns_once(
    monkeypatch, caplog
):
    from batchgen.models.moonshotai.kimi_linear import wrappers
    from batchgen.models.wrappers import AttnWrapperBase

    monkeypatch.setattr(wrappers, "_SKIP_KV_CALLBACK_WARNED", False)

    monkeypatch.setattr(AttnWrapperBase, "batchgen_debug", None, raising=False)
    assert wrappers._skip_kv_callback_active() is False

    monkeypatch.setattr(
        AttnWrapperBase, "batchgen_debug", {"k3_skip_kv_callback": False},
        raising=False,
    )
    assert wrappers._skip_kv_callback_active() is False

    monkeypatch.setattr(
        AttnWrapperBase, "batchgen_debug", {"k3_skip_kv_callback": True},
        raising=False,
    )
    with caplog.at_level(logging.WARNING, logger=wrappers.__name__):
        assert wrappers._skip_kv_callback_active() is True
        assert wrappers._skip_kv_callback_active() is True
    warnings = [
        record for record in caplog.records
        if record.levelno == logging.WARNING
        and "k3_skip_kv_callback" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_the_env_guard_is_gone_from_the_source_tree():
    hits = [
        path for top in ("batchgen", "batchgen_kernels")
        for path in sorted((ROOT / top).rglob("*.py"))
        if "BATCHGEN_SKIP_KV_CALLBACK" in path.read_text()
    ]
    assert not hits, f"BATCHGEN_SKIP_KV_CALLBACK still read in {hits}"


def test_the_skip_key_is_declared_in_the_kimi_registry():
    from batchgen.models.moonshotai.kimi_linear.server_debug_keys import (
        DEBUG_KEYS,
    )

    item = next(k for k in DEBUG_KEYS if k.name == "k3_skip_kv_callback")
    assert item.owner == "kimi_linear"
    assert item.phase == "decode"
    assert item.parse("1") is True and item.parse("off") is False
