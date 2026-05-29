"""Tests for :func:`batchgen.server.gpu_arch.detect_gpu_arch`.

These exercise the new Blackwell (sm_100) branch added in Phase 1 of the
Blackwell-support effort, plus the existing Hopper and Ampere branches and
the unsupported-arch error path.

The module is loaded via ``importlib.util.spec_from_file_location`` so the
test does not trigger ``batchgen.server.__init__`` (which imports
``http_server`` → ``uvicorn`` and the rest of the FastAPI stack). This mirrors
the standalone-module-load pattern used by ``tests/test_engine_config.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_gpu_arch_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "batchgen"
        / "server"
        / "gpu_arch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_batchgen_server_gpu_arch", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_arch_module = _load_gpu_arch_module()


@pytest.fixture
def fake_cuda(monkeypatch):
    """Patch the ``torch.cuda`` calls used by ``detect_gpu_arch``.

    Returns a setter ``(major, minor, name)`` that the test can call to
    pick which compute capability ``detect_gpu_arch`` will observe.
    """
    state = {"major": 9, "minor": 0, "name": "NVIDIA Fake"}

    monkeypatch.setattr(
        gpu_arch_module.torch.cuda, "is_available", lambda: True
    )
    monkeypatch.setattr(
        gpu_arch_module.torch.cuda,
        "get_device_capability",
        lambda *_args, **_kw: (state["major"], state["minor"]),
    )
    monkeypatch.setattr(
        gpu_arch_module.torch.cuda,
        "get_device_name",
        lambda *_args, **_kw: state["name"],
    )

    def _set(major: int, minor: int = 0, name: str = "NVIDIA Fake") -> None:
        state["major"] = major
        state["minor"] = minor
        state["name"] = name

    return _set


def test_detect_gpu_arch_blackwell(fake_cuda):
    fake_cuda(10, 0, "NVIDIA B200")
    assert gpu_arch_module.detect_gpu_arch() == "blackwell"


def test_detect_gpu_arch_blackwell_minor_version(fake_cuda):
    """Any 10.x compute capability should still classify as blackwell."""
    fake_cuda(10, 2, "NVIDIA B200")
    assert gpu_arch_module.detect_gpu_arch() == "blackwell"


def test_detect_gpu_arch_hopper(fake_cuda):
    fake_cuda(9, 0, "NVIDIA H20")
    assert gpu_arch_module.detect_gpu_arch() == "hopper"


def test_detect_gpu_arch_ampere(fake_cuda):
    fake_cuda(8, 0, "NVIDIA A100")
    assert gpu_arch_module.detect_gpu_arch() == "ampere"


def test_detect_gpu_arch_unsupported_raises(fake_cuda):
    fake_cuda(7, 5, "NVIDIA T4")
    with pytest.raises(RuntimeError, match="Unsupported GPU architecture"):
        gpu_arch_module.detect_gpu_arch()


def test_detect_gpu_arch_no_cuda(monkeypatch):
    monkeypatch.setattr(
        gpu_arch_module.torch.cuda, "is_available", lambda: False
    )
    with pytest.raises(RuntimeError, match="No CUDA devices available"):
        gpu_arch_module.detect_gpu_arch()


def test_worker_manager_reexports_detect_gpu_arch():
    """``worker_manager.detect_gpu_arch`` must remain importable for callers
    that already use ``from batchgen.server.worker_manager import detect_gpu_arch``.

    Skipped if optional server deps (e.g. ``uvicorn``) are not installed in
    the test environment — those deps are required to import
    ``batchgen.server`` at all, so the re-export only matters where the
    server can also be imported.
    """
    pytest.importorskip("uvicorn")
    from batchgen.server.gpu_arch import detect_gpu_arch as canonical
    from batchgen.server.worker_manager import (
        detect_gpu_arch as reexported,
    )

    assert reexported is canonical
