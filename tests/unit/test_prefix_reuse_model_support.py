"""Pure-Python tests for prefix-reuse model-name gating."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_model_name_utils():
    module_name = "batchgen.config.model_name_utils"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "batchgen" / "config" / "model_name_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prefix_reuse_model_gate_accepts_supported_backends():
    model_name_utils = _load_model_name_utils()

    supported_model_names = (
        "openai/gpt-oss-120b",
        "zai-org/GLM-5.1-FP8",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
        "moonshotai/Kimi-K2.5",
        "moonshotai/Kimi-K2.6",
        "MiniMaxAI/MiniMax-M2.5",
    )

    for model_name in supported_model_names:
        assert model_name_utils.is_prefix_reuse_supported_model(model_name)


def test_prefix_reuse_model_gate_rejects_unsupported_backends():
    model_name_utils = _load_model_name_utils()

    unsupported_model_names = (
        None,
        "",
        "deepseek-ai/DeepSeek-V2",
        "deepseek-ai/DeepSeek-V2-Lite",
        "Qwen/Qwen2-7B",
        "mistralai/Mixtral-8x7B",
    )

    for model_name in unsupported_model_names:
        assert not model_name_utils.is_prefix_reuse_supported_model(model_name)

