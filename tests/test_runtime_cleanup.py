"""Runtime cleanup must leave neighboring run namespaces untouched."""

from __future__ import annotations

import runpy
import uuid
from pathlib import Path

import pytest


PROCESS_UTILS = (
    Path(__file__).resolve().parents[1]
    / "batchgen"
    / "server"
    / "process_utils.py"
)
cleanup_shm_files = runpy.run_path(str(PROCESS_UTILS))["cleanup_shm_files"]
cleanup_model_shm_files = runpy.run_path(str(PROCESS_UTILS))["cleanup_model_shm_files"]


def test_cleanup_removes_only_the_selected_run_prefix():
    shm_dir = Path("/dev/shm")
    if not shm_dir.is_dir():
        return

    tag = uuid.uuid4().hex
    first_prefix = f"batchgen_lane-a_{tag}"
    second_prefix = f"batchgen_lane-b_{tag}"
    first = shm_dir / f"{first_prefix}_sentinel"
    second = shm_dir / f"{second_prefix}_sentinel"
    first.touch(exist_ok=False)
    second.touch(exist_ok=False)
    try:
        assert cleanup_shm_files(second_prefix) == 1
        assert first.exists()
        assert not second.exists()
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def test_cleanup_model_shm_files_removes_both_exact_names(tmp_path):
    weight = tmp_path / f"shm_{uuid.uuid4()}"
    metadata = tmp_path / f"shm_{uuid.uuid4()}"
    neighbor = tmp_path / f"{weight.name}_neighbor"
    for path in (weight, metadata, neighbor):
        path.touch()
    model_info = {
        "shm_name": f"/{weight.name}",
        "tensor_meta_shm_name": f"/{metadata.name}",
    }

    assert cleanup_model_shm_files(model_info, shm_dir=tmp_path) == 2
    assert not weight.exists()
    assert not metadata.exists()
    assert neighbor.exists()
    assert "shm_name" not in model_info
    assert "tensor_meta_shm_name" not in model_info


def test_cleanup_model_shm_files_rejects_path_escape(tmp_path):
    weight = tmp_path / f"shm_{uuid.uuid4()}"
    weight.touch()
    model_info = {
        "shm_name": f"/{weight.name}",
        "tensor_meta_shm_name": "/../other-shm",
    }

    with pytest.raises(ValueError, match="Invalid model SHM name"):
        cleanup_model_shm_files(model_info, shm_dir=tmp_path)
    assert weight.exists()
    assert model_info["shm_name"] == f"/{weight.name}"


def test_cleanup_model_shm_files_refuses_symlink(tmp_path):
    target = tmp_path / "neighbor"
    target.touch()
    link = tmp_path / f"shm_{uuid.uuid4()}"
    link.symlink_to(target)
    model_info = {"shm_name": f"/{link.name}"}

    with pytest.raises(RuntimeError, match="Refusing non-file model SHM"):
        cleanup_model_shm_files(model_info, shm_dir=tmp_path)
    assert link.is_symlink()
    assert target.exists()
    assert model_info["shm_name"] == f"/{link.name}"
