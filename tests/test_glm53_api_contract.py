"""CPU-only request-schema checks for GLM-5.3 chat controls."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_io_struct():
    path = Path(__file__).resolve().parents[1] / "batchgen/server/io_struct.py"
    spec = importlib.util.spec_from_file_location("batchgen.server.io_struct", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_glm53_reasoning_effort_and_clear_thinking_are_schema_fields():
    io_struct = _load_io_struct()
    request = io_struct.ChatCompletionRequest(
        model="zai-org/GLM-5.3-FP8",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="max",
        clear_thinking=True,
    )
    assert request.reasoning_effort == "max"
    assert request.clear_thinking is True


def test_reasoning_effort_rejects_unknown_values():
    io_struct = _load_io_struct()
    try:
        io_struct.ChatCompletionRequest(
            model="zai-org/GLM-5.3-FP8",
            messages=[{"role": "user", "content": "hello"}],
            reasoning_effort="extreme",
        )
    except Exception as exc:
        assert "reasoning_effort" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("unknown reasoning_effort must be rejected")
