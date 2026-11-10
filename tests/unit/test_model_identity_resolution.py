"""A --model value resolves to the registered model it names, or fails loud.

Host-KV profiles were looked up by exact lowercase name, so a local
checkpoint directory (``/data/zai-org/GLM-5.3-FP8``) or a base/dated variant
(``moonshotai/Kimi-Linear-48B-A3B``) was refused as "Unsupported model", and
the server caught that refusal, logged a WARNING and started without a host
KV cache (and, for a DSA model, without its indexer region).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.config import model_identity

ROOT = Path(__file__).resolve().parents[2]
HOST_KV_CONFIG = ROOT / "batchgen" / "kv_cache" / "host_kv_mananger_config.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"
GIB = 1024 ** 3


@pytest.fixture
def hk(monkeypatch):
    gpu_config = types.ModuleType("batchgen.kv_cache.gpu_paged_kv_manager")
    gpu_config.GPUPagedKVConfig = type("GPUPagedKVConfig", (SimpleNamespace,), {})
    native_loader = types.ModuleType("batchgen.native_loader")
    native_loader.core_engine = SimpleNamespace()
    monkeypatch.setitem(sys.modules, gpu_config.__name__, gpu_config)
    monkeypatch.setitem(sys.modules, native_loader.__name__, native_loader)
    name = "_identity_host_kv_config_for_test"
    spec = importlib.util.spec_from_file_location(name, HOST_KV_CONFIG)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "model, primary, indexer",
    [
        # Local checkpoint directories named after the release.
        ("/data/zai-org/GLM-5.3-FP8", "_GLM5_MLA_PROFILE", "_GLM5_2_INDEXER_PROFILE"),
        ("/staged/models/GLM-5.3-FP8/", "_GLM5_MLA_PROFILE", "_GLM5_2_INDEXER_PROFILE"),
        ("/shared/models/Kimi-K3.1-Pro", "_KIMI_K3_MLA_PROFILE", "_KIMI_K31_INDEXER_PROFILE"),
        ("/shared/models/Kimi-K3", "_KIMI_K3_MLA_PROFILE", None),
        # Base, instruct and dated variants of a registered id.
        ("moonshotai/Kimi-Linear-48B-A3B", "_KIMI_LINEAR_MLA_PROFILE", None),
        ("moonshotai/Kimi-Linear-48B-A3B-Base", "_KIMI_LINEAR_MLA_PROFILE", None),
        ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "_KIMI_LINEAR_MLA_PROFILE", None),
        ("deepseek-ai/DeepSeek-V3-0324", "_DEEPSEEK_MLA_PROFILE", None),
        # Registered ids keep resolving exactly as before.
        ("zai-org/GLM-5.2-FP8", "_GLM5_MLA_PROFILE", "_GLM5_2_INDEXER_PROFILE"),
        ("zai-org/GLM-5.1", "_GLM5_MLA_PROFILE", "_GLM5_INDEXER_PROFILE"),
        ("openai/gpt-oss-120b", "_GPT_OSS_GQA_PROFILE", None),
    ],
)
def test_a_model_value_resolves_to_the_registered_profiles(hk, model, primary, indexer):
    assert hk._resolve_profile(model) is getattr(hk, primary)
    expected = None if indexer is None else getattr(hk, indexer)
    assert hk._resolve_indexer_profile(model) is expected
    assert hk.is_dsa_model(model) is (indexer is not None)


@pytest.mark.parametrize(
    "model",
    [
        "moonshotai/Kimi-K3.2-Pro",       # unlisted release: never its neighbour
        "/data/models/Kimi-K3.2-Pro",
        "/dockerdata/converted",          # a directory that names no release
        "",
    ],
)
def test_a_value_that_names_no_registered_model_fails_loud(hk, model):
    with pytest.raises(ValueError, match="Unsupported model"):
        hk._resolve_profile(model)


def test_variant_stripping_never_merges_two_registered_models():
    with pytest.raises(ValueError, match="variant suffix cannot be ignored"):
        model_identity.build_index({"org/m-instruct": "a", "org/m": "b"})


def test_weight_envelope_resolves_a_local_directory_named_after_the_release():
    # By file path: batchgen.server's package init imports the GPU stack.
    spec = importlib.util.spec_from_file_location(
        "_identity_process_utils_for_test",
        ROOT / "batchgen" / "server" / "process_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_model_byte_size = module.get_model_byte_size

    assert get_model_byte_size("/data/zai-org/GLM-5.3-FP8") == 760 * GIB
    assert get_model_byte_size("/staged/GLM-5.2") == 1400 * GIB


def test_a_failed_host_kv_allocation_fails_the_start():
    """With --host-kv-cache-size the server must not start without it."""
    tree = ast.parse(WORKER_MANAGER.read_text())
    start = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_impl"
    )
    handlers = [
        handler
        for node in ast.walk(start)
        if isinstance(node, ast.Try)
        and "allocate_host_kv_cache" in ast.unparse(node.body[0] if len(node.body) == 1 else node)
        for handler in node.handlers
    ]
    assert handlers, "the host KV allocation is no longer wrapped"
    for handler in handlers:
        assert isinstance(handler.body[-1], ast.Raise), ast.unparse(handler)
        assert "host_kv_manager = None" not in ast.unparse(handler)
