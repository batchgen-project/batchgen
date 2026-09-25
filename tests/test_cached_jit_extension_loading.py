import importlib
import os

import pytest

import batchgen_kernels
import batchgen_kernels._jit_registry as jit_registry


MODULE_NAME = "batchgen_kernels.moe._C_test_cached"


def test_aot_only_load_accepts_prewarmed_jit(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(batchgen_kernels, "_DEV_MODE", True)
    monkeypatch.setattr(
        batchgen_kernels.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("AOT missing")),
    )
    monkeypatch.setattr(batchgen_kernels, "_load_cached_jit", lambda _name: sentinel)
    monkeypatch.setattr(
        batchgen_kernels,
        "_jit_compile",
        lambda _name: pytest.fail("cached AOT-only load must not compile"),
    )

    assert batchgen_kernels.load_extension(MODULE_NAME, allow_dev_jit=False) is sentinel


def test_aot_only_load_rejects_cache_miss_without_compiling(monkeypatch):
    monkeypatch.setattr(batchgen_kernels, "_DEV_MODE", True)
    monkeypatch.setattr(
        batchgen_kernels.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("AOT missing")),
    )
    monkeypatch.setattr(batchgen_kernels, "_load_cached_jit", lambda _name: None)
    monkeypatch.setattr(
        batchgen_kernels,
        "_jit_compile",
        lambda _name: pytest.fail("AOT-only cache miss must not compile"),
    )

    with pytest.raises(ImportError, match="AOT missing"):
        batchgen_kernels.load_extension(MODULE_NAME, allow_dev_jit=False)


def test_cached_jit_loader_requires_fresh_artifact(monkeypatch, tmp_path):
    package_dir = tmp_path / "batchgen_kernels"
    source = package_dir / "src" / "test.cu"
    source.parent.mkdir(parents=True)
    source.write_text("// source")

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    artifact = build_dir / "_C_test_cached.so"
    artifact.write_bytes(b"binary")

    sentinel = object()
    imported = []
    monkeypatch.setattr(batchgen_kernels, "__file__", str(package_dir / "__init__.py"))
    monkeypatch.setattr(
        jit_registry,
        "get_registry",
        lambda: {MODULE_NAME: {"sources": ["src/test.cu"]}},
    )

    cpp_extension = importlib.import_module("torch.utils.cpp_extension")
    monkeypatch.setattr(
        cpp_extension, "_get_build_directory", lambda _name, verbose=False: str(build_dir)
    )
    monkeypatch.setattr(
        cpp_extension,
        "_import_module_from_library",
        lambda *args, **kwargs: imported.append((args, kwargs)) or sentinel,
    )

    os.utime(source, (100, 100))
    os.utime(artifact, (101, 101))
    assert batchgen_kernels._load_cached_jit(MODULE_NAME) is sentinel
    assert len(imported) == 1

    imported.clear()
    os.utime(source, (102, 102))
    assert batchgen_kernels._load_cached_jit(MODULE_NAME) is None
    assert imported == []
