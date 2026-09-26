import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import batchgen.runtime_preflight as preflight

SITE = Path("/opt/conda/lib/python3.11/site-packages")


def test_unknown_model_type_fails_closed(monkeypatch):
    monkeypatch.setattr(preflight, "_check_torch", lambda: None)
    monkeypatch.setattr(preflight, "_detect_model_type_from_identifier", lambda _: "new_model")

    with pytest.raises(preflight.RuntimePreflightError, match="no declared runtime contract"):
        preflight.run_runtime_preflight(SimpleNamespace(model="new-model"))


def test_preflight_checks_contract_before_core_engine(monkeypatch):
    calls = []

    monkeypatch.setattr(preflight, "_check_torch", lambda: calls.append("torch"))
    monkeypatch.setattr(preflight, "_resolve_model_type", lambda _: "gpt_oss")
    monkeypatch.setattr(preflight, "_check_tokenizer", lambda _: calls.append("tokenizer"))
    monkeypatch.setattr(preflight, "_batchgen_root", lambda: SITE)
    monkeypatch.setattr(preflight, "_site_roots", lambda: (SITE,))

    def fake_import(name, *, origin=None):
        calls.append((name, origin))
        if name == "batchgen.core_engine":
            return SimpleNamespace(__file__=str(SITE / "batchgen/core_engine.so"))
        if name == "libucx":
            return SimpleNamespace(load_library=lambda: calls.append("ucx-load"))
        if name == "deep_gemm":
            return SimpleNamespace(fp8_mqa_logits=lambda max_seqlen_k: None)
        return SimpleNamespace(__file__=str(SITE / "module.py"))

    monkeypatch.setattr(preflight, "_import_required", fake_import)
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda _: "0.1.5.post3")

    assert preflight.run_runtime_preflight(SimpleNamespace(model="openai/gpt-oss-120b")) == "gpt_oss"
    assert calls[0] == "torch"
    assert calls.index("ucx-load") < calls.index(("batchgen.core_engine", "batchgen"))
    assert ("batchgen_kernels.attention._C_fused_ops", "batchgen") in calls
    assert ("flash_attn_interface", "site") in calls


def test_non_aot_core_engine_fails(monkeypatch):
    monkeypatch.setattr(preflight, "_import_required", lambda name, *, origin=None: SimpleNamespace(
        __file__="/tmp/core_engine.py"
    ))

    with pytest.raises(preflight.RuntimePreflightError, match="not an AOT native module"):
        preflight._check_core_engine()


def test_ucx_load_failure_fails():
    def broken():
        raise OSError("libucp.so.0: cannot open shared object file")

    with pytest.raises(preflight.RuntimePreflightError, match="libucp"):
        preflight._check_ucx(SimpleNamespace(load_library=broken))


# --- provenance: same-root rule ------------------------------------------------


@pytest.fixture
def worktrees(tmp_path, monkeypatch):
    """Two source worktrees plus a pyroot that symlinks into worktree A."""

    a, b = tmp_path / "wt_a", tmp_path / "wt_b"
    for wt in (a, b):
        (wt / "batchgen").mkdir(parents=True)
        (wt / "batchgen_kernels/attention").mkdir(parents=True)
        (wt / "core").mkdir()
    pyroot = tmp_path / "pyroot_a"
    pyroot.mkdir()
    os.symlink(a / "batchgen", pyroot / "batchgen")
    os.symlink(a / "batchgen_kernels", pyroot / "batchgen_kernels")
    monkeypatch.setattr(preflight, "_site_roots", lambda: (SITE,))
    monkeypatch.setattr(preflight, "_batchgen_root", lambda: a.resolve())
    return SimpleNamespace(a=a, b=b, pyroot=pyroot)


def _mod(path):
    return SimpleNamespace(__file__=str(path))


def test_pyroot_kernels_from_same_worktree_pass(worktrees):
    ext = worktrees.pyroot / "batchgen_kernels/attention/_C_fused_ops.so"
    (worktrees.a / "batchgen_kernels/attention/_C_fused_ops.so").touch()
    resolved = preflight._require_origin("k", _mod(ext), preflight._ORIGIN_BATCHGEN)
    assert resolved == (worktrees.a / "batchgen_kernels/attention/_C_fused_ops.so").resolve()


def test_kernels_from_other_worktree_fail(worktrees):
    ext = worktrees.b / "batchgen_kernels/attention/_C_fused_ops.so"
    with pytest.raises(preflight.RuntimePreflightError, match="same install or worktree"):
        preflight._require_origin("k", _mod(ext), preflight._ORIGIN_BATCHGEN)


def test_kernels_from_site_packages_fail_for_source_batchgen(worktrees):
    ext = SITE / "batchgen_kernels/attention/_C_fused_ops.so"
    with pytest.raises(preflight.RuntimePreflightError, match="same install or worktree"):
        preflight._require_origin("k", _mod(ext), preflight._ORIGIN_BATCHGEN)


def test_third_party_native_shadowed_by_worktree_fails(worktrees):
    with pytest.raises(preflight.RuntimePreflightError, match="outside site-packages"):
        preflight._require_origin(
            "flash_mla", _mod(worktrees.a / "flash_mla/__init__.py"), preflight._ORIGIN_SITE
        )


# --- core_engine freshness in a source worktree --------------------------------


def _core_engine_at(monkeypatch, so_path):
    monkeypatch.setattr(
        preflight,
        "_import_required",
        lambda name, *, origin=None: _mod(so_path),
    )


def test_stale_core_engine_fails(worktrees, monkeypatch):
    so = worktrees.a / "batchgen/core_engine.cpython-311-x86_64-linux-gnu.so"
    so.touch()
    src = worktrees.a / "core/batchgen.cpp"
    src.touch()
    os.utime(so, (1000, 1000))
    os.utime(src, (2000, 2000))
    _core_engine_at(monkeypatch, so)

    with pytest.raises(preflight.RuntimePreflightError, match="older than"):
        preflight._check_core_engine()


def test_fresh_core_engine_passes(worktrees, monkeypatch):
    so = worktrees.a / "batchgen/core_engine.cpython-311-x86_64-linux-gnu.so"
    so.touch()
    src = worktrees.a / "core/batchgen.cpp"
    src.touch()
    os.utime(src, (1000, 1000))
    os.utime(so, (2000, 2000))
    _core_engine_at(monkeypatch, so)

    preflight._check_core_engine()


def test_installed_core_engine_skips_source_freshness(monkeypatch):
    monkeypatch.setattr(preflight, "_site_roots", lambda: (SITE,))
    monkeypatch.setattr(preflight, "_batchgen_root", lambda: SITE)
    _core_engine_at(monkeypatch, SITE / "batchgen/core_engine.so")

    preflight._check_core_engine()
