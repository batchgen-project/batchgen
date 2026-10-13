import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_host_kv_config(monkeypatch):
    gpu_config = types.ModuleType(
        "batchgen.kv_cache.gpu_paged_kv_manager"
    )
    gpu_config.GPUPagedKVConfig = object
    engine_loader = types.ModuleType("batchgen.models.engine_loader")
    engine_loader.core_engine = types.SimpleNamespace()
    monkeypatch.setitem(
        sys.modules,
        "batchgen.kv_cache.gpu_paged_kv_manager",
        gpu_config,
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen.models.engine_loader",
        engine_loader,
    )

    path = ROOT / "batchgen" / "kv_cache" / "host_kv_mananger_config.py"
    module_name = "_glm52_host_kv_config_for_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_glm52_uses_glm5_primary_and_indexer_host_kv_profiles(monkeypatch):
    config = _load_host_kv_config(monkeypatch)

    for model_name in (
        "zai-org/GLM-5.2-FP8",
        "zai-org/GLM-5.2",
        "GLM-5.2-FP8",
        "GLM-5.2",
    ):
        assert config._resolve_profile(model_name) is config._GLM5_MLA_PROFILE
        assert (
            config._resolve_indexer_profile(model_name)
            is config._GLM5_INDEXER_PROFILE
        )


def test_glm53_uses_glm5_primary_and_indexer_host_kv_profiles(monkeypatch):
    config = _load_host_kv_config(monkeypatch)

    for model_name in (
        "zai-org/GLM-5.3-FP8",
        "zai-org/GLM-5.3",
        "GLM-5.3-FP8",
        "GLM-5.3",
    ):
        assert config._resolve_profile(model_name) is config._GLM5_MLA_PROFILE
        assert (
            config._resolve_indexer_profile(model_name)
            is config._GLM5_INDEXER_PROFILE
        )
