"""Runtime cleanup must leave neighboring run namespaces untouched."""

from __future__ import annotations

import runpy
import uuid
from pathlib import Path


PROCESS_UTILS = (
    Path(__file__).resolve().parents[1]
    / "batchgen"
    / "server"
    / "process_utils.py"
)
cleanup_shm_files = runpy.run_path(str(PROCESS_UTILS))["cleanup_shm_files"]


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
