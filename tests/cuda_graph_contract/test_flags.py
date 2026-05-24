"""Tests for `batchgen.cuda_graph.flags.DecodeGraphFlags`."""

from __future__ import annotations

import logging
import os

import pytest

from batchgen.cuda_graph import flags as flags_mod


_ALL_ENV = (
    "BATCHGEN_DECODE_GRAPH_COMPARE",
    "BATCHGEN_DECODE_GRAPH_COMPARE_FAIL",
    "BATCHGEN_DECODE_GRAPH_COMPARE_ATOL",
    "BATCHGEN_DECODE_GRAPH_COMPARE_RTOL",
    "BATCHGEN_DECODE_GRAPH_TIMING",
    "BATCHGEN_DECODE_GRAPH_PROBE_LAYERS",
    "BATCHGEN_DECODE_GRAPH_PATH_LOG",
    "BATCHGEN_DECODE_GRAPH_MAX_SEQLEN",
    "BATCHGEN_DECODE_GRAPH_MEMORY_DIAG",
    "BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH",
    "BATCHGEN_GLM5_LAYER_CUDA_GRAPH",
    "BATCHGEN_GLM5_DSA_CUDA_GRAPH",
    "BATCHGEN_GLM5_DSA_FULL_CUDA_GRAPH",
    "BATCHGEN_GLM5_MOE_CUDA_GRAPH",
    "BATCHGEN_SEGMENTED_GRAPH",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ALL_ENV:
        monkeypatch.delenv(v, raising=False)
    # Reset the one-shot guard so each test starts fresh.
    flags_mod._warned_glm5_env = False


def test_defaults_all_off():
    f = flags_mod.DecodeGraphFlags.from_env()
    assert f.compare is False
    assert f.compare_fail is False
    assert f.timing is False
    assert f.path_log is False
    assert f.memory_diag is False
    assert f.probe_layers == ()
    assert f.max_seqlen is None
    assert f.compare_atol == pytest.approx(1e-2)
    assert f.compare_rtol == pytest.approx(1e-2)


def test_to_debug_opts_mirrors_flags(monkeypatch):
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_COMPARE", "1")
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_COMPARE_FAIL", "1")
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_TIMING", "1")
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_PATH_LOG", "1")
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_PROBE_LAYERS", "0,12,40")
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_COMPARE_ATOL", "5e-3")
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_COMPARE_RTOL", "5e-3")
    opts = flags_mod.DecodeGraphFlags.from_env().to_debug_opts()
    assert opts.compare_against_eager is True
    assert opts.fail_on_mismatch is True
    assert opts.timing is True
    assert opts.log_path_breadcrumbs is True
    assert opts.probe_layers == (0, 12, 40)
    assert opts.compare_atol == pytest.approx(5e-3)
    assert opts.compare_rtol == pytest.approx(5e-3)


def test_probe_layers_all_marker(monkeypatch):
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_PROBE_LAYERS", "all")
    assert flags_mod.DecodeGraphFlags.from_env().probe_layers == (-1,)


def test_invalid_float_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("BATCHGEN_DECODE_GRAPH_COMPARE_ATOL", "not-a-float")
    f = flags_mod.DecodeGraphFlags.from_env()
    assert f.compare_atol == pytest.approx(1e-2)


def test_warn_on_removed_glm5_env_vars_one_shot(monkeypatch, caplog):
    monkeypatch.setenv("BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    with caplog.at_level(logging.WARNING, logger="batchgen.cuda_graph.flags"):
        flags_mod.warn_on_removed_glm5_env_vars()
        flags_mod.warn_on_removed_glm5_env_vars()
    relevant = [r for r in caplog.records if "no longer recognized" in r.getMessage()]
    assert len(relevant) == 1
    msg = relevant[0].getMessage()
    assert "BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH" in msg
    assert "BATCHGEN_GLM5_DSA_CUDA_GRAPH" in msg


def test_no_warning_when_no_removed_vars_set(caplog):
    with caplog.at_level(logging.WARNING, logger="batchgen.cuda_graph.flags"):
        flags_mod.warn_on_removed_glm5_env_vars()
    assert not any("no longer recognized" in r.getMessage() for r in caplog.records)
