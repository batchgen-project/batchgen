"""Per-model routing of --enable-cuda-graph (sub-project batchgen-gptoss-cuda-graph-flag-ignored).

GPT-OSS-120B used to ignore the flag silently and decode eager. The flag must
turn on the GPT-OSS and K2.5 whole-model graphs, leave GLM-5 FP8 to its own
router, and be rejected loudly for every model it cannot affect.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

from batchgen.config.cuda_graph_flag_policy import (
    GLM5_FP8_GRAPH_FAMILY,
    GPT_OSS_GRAPH_FAMILY,
    KIMI_K25_GRAPH_FAMILY,
    UnsupportedCudaGraphFlagError,
    cli_enables_whole_model_graph,
    cuda_graph_flag_family,
    require_cuda_graph_flag_supported,
)

ROOT = Path(__file__).resolve().parents[1]

SUPPORTED = [
    ("openai/gpt-oss-120b", GPT_OSS_GRAPH_FAMILY, True),
    ("moonshotai/Kimi-K2.5", KIMI_K25_GRAPH_FAMILY, True),
    ("moonshotai/Kimi-K2.6", KIMI_K25_GRAPH_FAMILY, True),
    # GLM-5 FP8 is accepted but routed by glm5_any_cuda_graph_requested_for_model.
    ("zai-org/GLM-5-FP8", GLM5_FP8_GRAPH_FAMILY, False),
    ("zai-org/GLM-5.2-FP8", GLM5_FP8_GRAPH_FAMILY, False),
]

UNSUPPORTED = [
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V4-Flash",
    "MiniMaxAI/MiniMax-M2.5",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "zai-org/GLM-5",  # BF16 GLM-5 has no whole-model graph
]

KIMI_LINEAR = ["moonshotai/Kimi-K3", "moonshotai/Kimi-Linear-48B-A3B-Instruct"]


@pytest.mark.parametrize(("model", "family", "enables"), SUPPORTED)
def test_supported_models_route_the_flag(model, family, enables):
    assert cuda_graph_flag_family(model) == family
    require_cuda_graph_flag_supported(model, enable_cuda_graph=True)
    assert cli_enables_whole_model_graph(model, enable_cuda_graph=True) is enables


@pytest.mark.parametrize("model", [m for m, _, _ in SUPPORTED] + UNSUPPORTED + KIMI_LINEAR)
def test_without_the_flag_nothing_is_enabled_or_rejected(model):
    require_cuda_graph_flag_supported(model, enable_cuda_graph=False)
    assert cli_enables_whole_model_graph(model, enable_cuda_graph=False) is False


@pytest.mark.parametrize("model", UNSUPPORTED)
def test_unsupported_models_reject_the_flag(model):
    assert cuda_graph_flag_family(model) is None
    with pytest.raises(UnsupportedCudaGraphFlagError, match="not supported"):
        require_cuda_graph_flag_supported(model, enable_cuda_graph=True)
    assert cli_enables_whole_model_graph(model, enable_cuda_graph=True) is False


@pytest.mark.parametrize("model", KIMI_LINEAR)
def test_kimi_linear_rejection_names_the_planner_knob(model):
    with pytest.raises(UnsupportedCudaGraphFlagError, match="kimi_decode_graph_mode"):
        require_cuda_graph_flag_supported(model, enable_cuda_graph=True)


def _load_server_args_module():
    """Import the leaf module without running the HTTP server package init."""
    package_name = "batchgen.server"
    previous = sys.modules.get(package_name)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return importlib.import_module("batchgen.server.server_args")
    finally:
        if previous is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous


def _server_args(tmp_path, monkeypatch, **overrides):
    module = _load_server_args_module()
    monkeypatch.setattr(module, "_ensure_local_port_free", lambda port, label: None)
    values = {
        "model": "openai/gpt-oss-120b",
        "host_kv_cache_size": 1,
        "storage_path": tmp_path / "storage",
    }
    values.update(overrides)
    return module, module.ServerArgs(**values)


def test_server_validation_accepts_flag_for_gpt_oss(tmp_path, monkeypatch):
    module, args = _server_args(tmp_path, monkeypatch, enable_cuda_graph=True)
    module.validate_server_args(args)


def test_server_validation_rejects_flag_before_workers_start(tmp_path, monkeypatch):
    module, args = _server_args(
        tmp_path, monkeypatch, model="deepseek-ai/DeepSeek-R1", enable_cuda_graph=True
    )
    with pytest.raises(UnsupportedCudaGraphFlagError, match="DeepSeek-R1"):
        module.validate_server_args(args)
