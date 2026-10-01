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
_NAMESPACE = runpy.run_path(str(PROCESS_UTILS))
cleanup_shm_files = _NAMESPACE["cleanup_shm_files"]


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


def test_no_model_shm_provenance_api_remains():
    """The model regions are unnamed memfds in every mode, hugetlbfs included.

    Nothing is left on disk for a supervisor to attribute or for the release
    path to prove absent, so recording a name would be a lie.
    """
    for dead in (
        "record_model_shm_provenance",
        "verify_model_shm_absent",
        "MODEL_SHM_KEYS",
        "MODEL_NAMED_SHM_KEYS",
        "MODEL_SHM_PROVENANCE_FILE",
        "_validated_shm_entry_name",
    ):
        assert dead not in _NAMESPACE, dead

    source = PROCESS_UTILS.read_text()
    assert "model_shm.json" not in source
    assert "/dev/hugepages" not in source
