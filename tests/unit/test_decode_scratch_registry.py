import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_package_stubs(monkeypatch):
    batchgen_stub = types.ModuleType("batchgen")
    batchgen_stub.__path__ = [str(REPO_ROOT / "batchgen")]
    monkeypatch.setitem(sys.modules, "batchgen", batchgen_stub)
    models_stub = types.ModuleType("batchgen.models")
    models_stub.__path__ = [str(REPO_ROOT / "batchgen" / "models")]
    monkeypatch.setitem(sys.modules, "batchgen.models", models_stub)
    wrappers_stub = types.ModuleType("batchgen.models.wrappers")
    wrappers_stub.__path__ = [str(REPO_ROOT / "batchgen" / "models" / "wrappers")]
    monkeypatch.setitem(sys.modules, "batchgen.models.wrappers", wrappers_stub)
    openai_stub = types.ModuleType("batchgen.models.openai")
    openai_stub.__path__ = [str(REPO_ROOT / "batchgen" / "models" / "openai")]
    monkeypatch.setitem(sys.modules, "batchgen.models.openai", openai_stub)
    gpt_pkg_stub = types.ModuleType("batchgen.models.openai.gpt_oss_120b")
    gpt_pkg_stub.__path__ = [
        str(REPO_ROOT / "batchgen" / "models" / "openai" / "gpt_oss_120b")
    ]
    monkeypatch.setitem(
        sys.modules,
        "batchgen.models.openai.gpt_oss_120b",
        gpt_pkg_stub,
    )
    gpt_scratch_stub = types.ModuleType(
        "batchgen.models.openai.gpt_oss_120b.decode_scratch"
    )
    gpt_scratch_stub.estimate_gpt_oss_decode_scratch_reserve_gb = (
        lambda **kwargs: 2.5
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen.models.openai.gpt_oss_120b.decode_scratch",
        gpt_scratch_stub,
    )


def _registry_module(monkeypatch):
    _install_package_stubs(monkeypatch)
    return importlib.import_module("batchgen.models.wrappers.decode_scratch")


def test_decode_scratch_registry_requires_registered_model(monkeypatch):
    registry = _registry_module(monkeypatch)

    with pytest.raises(RuntimeError, match="not registered"):
        registry.estimate_decode_scratch_reserve_gb(
            model_config=SimpleNamespace(model_type="unknown_model"),
            world_size=1,
            max_num_seq_per_rank=1,
        )


def test_decode_scratch_registry_supports_explicit_no_reserve(monkeypatch):
    registry = _registry_module(monkeypatch)

    reserve = registry.estimate_decode_scratch_reserve_gb(
        model_config=SimpleNamespace(model_type="glm_moe_dsa"),
        world_size=8,
        max_num_seq_per_rank=32,
    )

    assert reserve == 0.0


def test_decode_scratch_registry_dispatches_gpt_oss(monkeypatch):
    registry = _registry_module(monkeypatch)

    reserve = registry.estimate_decode_scratch_reserve_gb(
        model_config=SimpleNamespace(model_type="gpt_oss"),
        world_size=2,
        max_num_seq_per_rank=4,
    )

    assert reserve == 2.5
