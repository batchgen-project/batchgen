# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                             #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The GLM-5 late-compile guards report every compile after readiness.

Under test: the readiness flag of ``batchgen_kernels/triton_prewarm.py``
(``LateCompileGuard.mark_ready`` / ``arm_at_readiness``), the GLM-5 phase
runner that honours it, the DeepGEMM kernel-name baseline of
``deepgemm_prewarm.py``, and the unset-bucket report of
``triton_prewarm.py``.

  * before readiness a prewarm phase disarms the Triton guard; after it the
    guard stays armed, so the phase's compiles are logged as late and the
    DeepGEMM kernels it compiled are reported by name;
  * the DeepGEMM guard diffs kernel NAMES, so a kernel that replaces another
    (same count) is still caught, and each late name is reported once;
  * an unarmed DeepGEMM guard says so at WARNING instead of returning 0;
  * ``decode_graph_buckets=None`` with the whole-model graph on is logged,
    once per process.

The prewarm modules are loaded by file path; no CUDA, Triton or DeepGEMM.

Run: python -m pytest tests/unit/test_glm5_late_compile_guard.py -q
"""

import ast
import importlib.util
import logging
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_GLM5_DIR = _ROOT / "batchgen" / "models" / "glm" / "glm5"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DGP = _load(_GLM5_DIR / "deepgemm_prewarm.py", "_glm5_deepgemm_prewarm_for_guard")
TP = _load(_GLM5_DIR / "triton_prewarm.py", "_glm5_triton_prewarm_for_guard")
# The core module TP itself runs on: its guard, recorder and error classes.
CORE = TP._core


# --------------------------------------------------------------------------- #
#  Triton: readiness
# --------------------------------------------------------------------------- #
def test_ready_guard_cannot_be_disarmed():
    guard = CORE.LateCompileGuard(debug_source=dict)
    assert not guard.ready
    assert guard.disarm() is True and guard.armed_mode == CORE.DISARMED
    guard.mark_ready()
    assert guard.ready and guard.armed_mode == CORE.LOG
    assert guard.disarm() is False
    assert guard.state() == CORE.LOG
    # A strict guard stays strict when readiness is marked again.
    guard.arm(strict=True)
    guard.mark_ready()
    assert guard.armed_mode == CORE.STRICT
    other = CORE.LateCompileGuard(debug_source=dict)
    other.mark_ready(strict=True)
    assert other.armed_mode == CORE.STRICT


def test_core_arm_at_readiness_installs_and_arms(monkeypatch, caplog):
    recorder = CORE.TritonCompileRecorder(
        guard=CORE.LateCompileGuard(debug_source=dict), log_tag="[T]")
    monkeypatch.setattr(CORE, "_RECORDER", recorder)
    monkeypatch.setattr(recorder, "install", lambda runtime=None: True)
    with caplog.at_level(logging.INFO, logger=CORE.logger.name):
        assert CORE.arm_at_readiness(rank=5) is True
    assert recorder.guard.ready and recorder.guard.armed_mode == CORE.LOG
    assert any(r.getMessage().startswith("[T] rank=5 readiness")
               for r in caplog.records)


def _payload(name):
    fn = type("Fn", (), {"name": name, "module": "m", "jit_function": None})()
    return {"fn": fn, "key": name, "compile": {}}


def _phase_with_compile(monkeypatch, recorder, *, deepgemm_new=()):
    """Run ``TP._run_phase`` over one entry that fires Triton's two hooks
    for kernel ``m.k`` and adds ``deepgemm_new`` to the DeepGEMM cache."""
    names = set(DGP.cache_kernel_names())

    def entry():
        recorder.cache_hook(**_payload("k"))
        recorder.post_compile_hook(**_payload("k"))
        names.update(deepgemm_new)

    monkeypatch.setattr(DGP, "cache_kernel_names",
                        lambda path=None: frozenset(names))
    TP._run_phase("decode", [CORE.PrewarmEntry("e", entry)], part_index=0,
                  part_count=1, barrier=None, rank=2, arm_after=False)


@pytest.fixture
def glm5_recorder(monkeypatch, tmp_path):
    recorder = CORE.TritonCompileRecorder(
        guard=CORE.LateCompileGuard(debug_source=dict), log_tag=TP.LOG_TAG)
    monkeypatch.setattr(recorder, "install", lambda runtime=None: True)
    monkeypatch.setattr(TP, "get_recorder", lambda: recorder)
    monkeypatch.setattr(TP, "_deepgemm_prewarm", lambda: DGP)
    monkeypatch.setattr(CORE, "_cuda_synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(DGP, "_BASELINE", None)
    monkeypatch.setattr(DGP, "_UNARMED_REPORTED", set())
    return recorder


def test_phase_before_readiness_is_not_late(glm5_recorder, monkeypatch,
                                            caplog):
    glm5_recorder.guard.arm()
    with caplog.at_level(logging.WARNING):
        _phase_with_compile(monkeypatch, glm5_recorder)
    assert [e.late for e in glm5_recorder.events] == [False]
    # The guard is back in the mode it had before the phase.
    assert glm5_recorder.guard.armed_mode == CORE.LOG
    assert not [r for r in caplog.records if "late compile" in r.getMessage()]


def test_phase_after_readiness_reports_triton_and_deepgemm_late(
        glm5_recorder, monkeypatch, caplog):
    DGP._BASELINE = frozenset()
    glm5_recorder.guard.mark_ready()
    with caplog.at_level(logging.WARNING):
        _phase_with_compile(
            monkeypatch, glm5_recorder,
            deepgemm_new=("kernel.sm90_fp8_gemm_1d2d.abc",))
    assert [e.late for e in glm5_recorder.events] == [True]
    assert glm5_recorder.guard.armed_mode == CORE.LOG
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "decode phase runs after readiness" in text
    assert "[GLM5_TRITON_WARMUP] late compile kernel=m.k" in text
    assert "seen at decode phase after readiness" in text
    assert "sm90_fp8_gemm_1d2d x1" in text
    assert "kernel.sm90_fp8_gemm_1d2d.abc" in text


def test_strict_late_compile_in_a_phase_propagates(glm5_recorder,
                                                   monkeypatch):
    glm5_recorder.guard.mark_ready(strict=True)
    with pytest.raises(CORE.LateTritonCompileError):
        _phase_with_compile(monkeypatch, glm5_recorder)


def test_glm5_arm_at_readiness_arms_both_guards(glm5_recorder, monkeypatch,
                                                tmp_path):
    monkeypatch.setattr(CORE, "get_recorder", lambda *a, **k: glm5_recorder)
    (tmp_path / "cache" / "kernel.sm90_fp8_gemm_1d2d.a").mkdir(parents=True)
    TP.arm_at_readiness(rank=0)
    assert glm5_recorder.guard.ready
    assert glm5_recorder.guard.armed_mode == CORE.LOG
    assert DGP._BASELINE == frozenset({"kernel.sm90_fp8_gemm_1d2d.a"})


# --------------------------------------------------------------------------- #
#  DeepGEMM: kernel-name baseline
# --------------------------------------------------------------------------- #
@pytest.fixture
def dg_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(DGP, "_BASELINE", None)
    monkeypatch.setattr(DGP, "_UNARMED_REPORTED", set())
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


def test_unarmed_guard_is_loud_and_returns_none(dg_cache, caplog):
    (dg_cache / "kernel.sm90_fp8_gemm_1d2d.a").mkdir()
    with caplog.at_level(logging.WARNING, logger=DGP.logger.name):
        assert DGP.check_late_compiles(1, "configure_prefill") is None
        assert DGP.check_late_compiles(1, "configure_prefill") is None
        assert DGP.check_late_compiles(1, "admit") is None
    messages = [r.getMessage() for r in caplog.records]
    # Once per call site.
    assert len(messages) == 2
    assert all("late-compile guard not armed" in m for m in messages)
    assert "configure_prefill" in messages[0] and "admit" in messages[1]


def test_replaced_kernel_is_late_although_the_count_is_unchanged(dg_cache,
                                                                 caplog):
    (dg_cache / "kernel.sm90_fp8_gemm_1d2d.a").mkdir()
    (dg_cache / "kernel.transpose_fp32.b").mkdir()
    (dg_cache / "tmp.partial").mkdir()
    (dg_cache / "kernel.stray_file").write_text("")
    baseline = DGP.record_baseline(0)
    assert baseline == frozenset({"kernel.sm90_fp8_gemm_1d2d.a",
                                  "kernel.transpose_fp32.b"})
    assert DGP.check_late_compiles(0, "configure_prefill") == ()
    # One kernel leaves the cache while a late compile adds another: the
    # count is unchanged, so a count diff saw nothing.
    (dg_cache / "kernel.transpose_fp32.b").rmdir()
    (dg_cache / "kernel.sm90_paged_mqa_logits_metadata.c").mkdir()
    assert len(DGP.cache_kernel_names()) == len(baseline)
    with caplog.at_level(logging.WARNING, logger=DGP.logger.name):
        late = DGP.check_late_compiles(3, "configure_decoding")
    assert late == ("kernel.sm90_paged_mqa_logits_metadata.c",)
    message, = [r.getMessage() for r in caplog.records]
    assert message.startswith("[GLM5_DEEPGEMM_WARMUP] rank=3 DeepGEMM "
                              "compiled 1 kernels after the prewarm")
    assert "configure_decoding" in message
    assert "sm90_paged_mqa_logits_metadata x1" in message
    # Reported once.
    assert DGP.check_late_compiles(3, "configure_prefill") == ()


def test_late_names_carry_their_family_and_strict_raises(dg_cache):
    DGP.record_baseline(0)
    for name in ("kernel.sm90_fp8_gemm_1d2d.x", "kernel.sm90_fp8_gemm_1d2d.y",
                 "kernel.transpose_fp32.z"):
        (dg_cache / name).mkdir()
    with pytest.raises(DGP.LateDeepGemmCompileError,
                       match="sm90_fp8_gemm_1d2d x2, transpose_fp32 x1"):
        DGP.check_late_compiles(0, "admit", strict=True)
    assert DGP.kernel_family("kernel.sm90_fp8_gemm_1d2d.x") == (
        "sm90_fp8_gemm_1d2d")
    assert DGP.kernel_family("kernel.odd") == "odd"


# --------------------------------------------------------------------------- #
#  decode_graph_buckets wiring
# --------------------------------------------------------------------------- #
def test_unset_buckets_with_the_graph_on_is_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(TP, "_BUCKETS_UNSET_REPORTED", False)
    with caplog.at_level(logging.WARNING, logger=TP.logger.name):
        assert TP.report_unset_decode_buckets(None, whole_model_graph=True,
                                              rank=4) is True
        assert TP.report_unset_decode_buckets(None, whole_model_graph=True,
                                              rank=4) is False
    message, = [r.getMessage() for r in caplog.records]
    assert message.startswith("[GLM5_TRITON_WARMUP] rank=4 "
                              "decode_graph_buckets is None")
    assert "whole-model decode CUDA graph" in message


@pytest.mark.parametrize("buckets,graph", [((1, 2, 4), True), (None, False)])
def test_buckets_report_is_silent_otherwise(monkeypatch, caplog, buckets,
                                            graph):
    monkeypatch.setattr(TP, "_BUCKETS_UNSET_REPORTED", False)
    with caplog.at_level(logging.WARNING, logger=TP.logger.name):
        assert TP.report_unset_decode_buckets(
            buckets, whole_model_graph=graph, rank=0) is False
    assert caplog.records == []


def test_psm_reports_unset_buckets_on_the_decode_and_geometry_phases():
    tree = ast.parse((_GLM5_DIR / "Parallel_Strategy_Manager.py").read_text())
    method, = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)
               and node.name == "_prewarm_triton_kernels"]
    calls = [node for node in ast.walk(method)
             if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "report_unset_decode_buckets"]
    assert len(calls) == 1
    text = ast.unparse(calls[0])
    assert "self.decode_graph_buckets" in text
    assert "enable_cuda_graphs" in text
    guard, = [node for node in ast.walk(method)
              if isinstance(node, ast.If) and calls[0] in ast.walk(node)]
    assert ast.unparse(guard.test) == "phase in (PHASE_DECODE, PHASE_GEOMETRY)"
