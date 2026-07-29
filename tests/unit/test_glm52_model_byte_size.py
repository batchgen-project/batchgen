"""Focused regression for GLM-5.2 hugepage sizing."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_process_utils():
    path = REPO_ROOT / "batchgen" / "server" / "process_utils.py"
    spec = importlib.util.spec_from_file_location("glm52_process_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_glm52_model_byte_sizes_use_exact_entries():
    process_utils = _load_process_utils()

    assert process_utils.get_model_byte_size("zai-org/GLM-5.2-FP8") == 760 * 1024**3
    assert process_utils.get_model_byte_size("zai-org/GLM-5.2") == 1400 * 1024**3
