# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""Headless regression tests for the decoupled BatchGenModelConfig resolver.

These tests must run WITHOUT torch / the C++ engine. The production
``batchgen.config`` package ``__init__`` eagerly imports the tokenizer stack
(torch) and the model registry (whose auto-import triggers an engine JIT
build), so — mirroring tests/test_glm5_planner.py — we pre-register stub
package modules and load the specific source files directly via importlib.

Coverage:
  * Regression lock: resolve() for GLM-5-FP8 reproduces the exact values the
    old glm5_initializer._parse_model_config hardcoded, AND the engine
    projection maps head_dim <- qk_head_dim (256, the head_dim TRAP).
  * resolve() for GLM-5.2-FP8 reads the real checkpoint config.json and yields
    the correct dims + GLM-5.2-only indexer fields + nested rope_theta, with a
    distinct model_type.
  * Pattern ordering: GLM-5.2 resolves to GLM52Config, not GLM5Config.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "batchgen" / "config"
_GLM5_CONFIG = _REPO_ROOT / "batchgen" / "models" / "glm" / "glm5" / "config.py"

# Point GLM52_CKPT_DIR at a local GLM-5.2-FP8 checkout to exercise the
# real-config resolution tests; unset, they skip (as they do in CI).
_GLM52_CKPT = os.environ.get("GLM52_CKPT_DIR", "")

# Old _parse_model_config hardcoded values — the regression baseline.
_LEGACY_GLM5 = {
    "model_type": "glm_moe_dsa",
    "num_hidden_layers": 78,
    "num_local_experts": 256,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 256,  # == qk_head_dim (the projection uses qk_head_dim)
    "compressed_kv_dim": 576,
    "first_k_dense_replace": 3,
}


def _load_module(fqname: str, path: Path):
    """Exec a source file as ``fqname`` without running package __init__."""
    if fqname in sys.modules:
        return sys.modules[fqname]
    spec = importlib.util.spec_from_file_location(fqname, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[fqname] = module
    spec.loader.exec_module(module)
    return module


def _stub_pkg(fqname: str, dir_path: Path):
    if fqname not in sys.modules:
        pkg = types.ModuleType(fqname)
        pkg.__path__ = [str(dir_path)]
        sys.modules[fqname] = pkg


def _bootstrap():
    """Wire up a torch-free import graph and return (BatchGenModelConfig, mod)."""
    # Namespace stubs so no real package __init__ (torch/engine) runs.
    _stub_pkg("batchgen", _REPO_ROOT / "batchgen")
    _stub_pkg("batchgen.config", _CONFIG_DIR)
    _stub_pkg("batchgen.models", _REPO_ROOT / "batchgen" / "models")
    _stub_pkg("batchgen.models.glm", _REPO_ROOT / "batchgen" / "models" / "glm")
    _stub_pkg("batchgen.models.glm.glm5", _REPO_ROOT / "batchgen" / "models" / "glm" / "glm5")

    # Minimal, torch-free stand-in for model_registry (the real one auto-imports
    # every family config, dragging the engine). Provides just what glm5/config
    # needs: a working register_config decorator + a registry dict.
    if "batchgen.config.model_registry" not in sys.modules:
        reg = types.ModuleType("batchgen.config.model_registry")
        reg.CONFIG_REGISTRY = {}

        def register_config(model_type):
            def deco(cls):
                reg.CONFIG_REGISTRY[model_type] = cls
                return cls
            return deco

        reg.register_config = register_config
        sys.modules["batchgen.config.model_registry"] = reg

    _load_module("batchgen.config.model_config", _CONFIG_DIR / "model_config.py")
    # Pre-load glm5 config so the resolver's import_module hits the cache.
    _load_module("batchgen.models.glm.glm5.config", _GLM5_CONFIG)
    bmc = _load_module(
        "batchgen.config.batchgen_model_config",
        _CONFIG_DIR / "batchgen_model_config.py",
    )
    return bmc.BatchGenModelConfig, bmc


BatchGenModelConfig, _bmc_mod = _bootstrap()


def _project(rich):
    """Replicate glm5_initializer's engine projection (the load-bearing part).

    Mirrors the minimal ModelConfig the C++ engine reads. Crucially maps
    head_dim <- qk_head_dim, not rich.head_dim.
    """
    return {
        "model_type": rich.model_type,
        "num_hidden_layers": rich.num_hidden_layers,
        "num_local_experts": rich.num_local_experts,
        "num_attention_heads": rich.num_attention_heads,
        "num_key_value_heads": rich.num_key_value_heads,
        "head_dim": rich.qk_head_dim,  # TRAP: qk_head_dim, not head_dim
        "compressed_kv_dim": rich.compressed_kv_dim,
        "first_k_dense_replace": rich.first_k_dense_replace,
    }


def test_resolver_imports_without_torch():
    assert "torch" not in sys.modules, (
        "batchgen_model_config must import cleanly headless; something dragged torch."
    )


def test_glm5_fp8_regression_matches_legacy_hardcoded_values():
    # No GLM-5-FP8 checkpoint on disk -> resolver uses GLM5Config defaults,
    # which must reproduce the old _parse_model_config hardcoded values.
    rich = BatchGenModelConfig.resolve("zai-org/GLM-5-FP8", checkpoint_path=None)
    assert type(rich).__name__ == "GLM5Config"
    projected = _project(rich)
    assert projected == _LEGACY_GLM5


def test_glm5_head_dim_trap_projection_uses_qk_head_dim():
    rich = BatchGenModelConfig.resolve("GLM-5-FP8", checkpoint_path=None)
    # rich.head_dim is 64; the engine must receive qk_head_dim (256).
    assert rich.head_dim != 256
    assert rich.qk_head_dim == 256
    assert _project(rich)["head_dim"] == 256


def test_glm52_pattern_resolves_to_glm52_config_not_glm5():
    target = BatchGenModelConfig._match_variant("zai-org/GLM-5.2-FP8")
    assert target == ("batchgen.models.glm.glm5.config", "GLM52Config")
    # And GLM-5.1 / GLM-5 do not get swallowed by GLM-5.2.
    assert BatchGenModelConfig._match_variant("GLM-5.1-FP8")[1] == "GLM5Config"
    assert BatchGenModelConfig._match_variant("GLM-5-FP8")[1] == "GLM5Config"


@pytest.mark.skipif(
    not (Path(_GLM52_CKPT) / "config.json").exists(),
    reason="GLM-5.2-FP8 checkpoint config.json not available",
)
def test_glm52_fp8_reads_checkpoint_dims():
    rich = BatchGenModelConfig.resolve("zai-org/GLM-5.2-FP8", checkpoint_path=_GLM52_CKPT)
    assert type(rich).__name__ == "GLM52Config"

    # Distinct config identity.
    assert rich.model_type == "glm_moe_dsa_5_2"

    # Core dims read from the real config.json.
    assert rich.num_hidden_layers == 78
    assert rich.num_local_experts == 256  # from n_routed_experts
    assert rich.n_routed_experts == 256
    assert rich.num_attention_heads == 64
    assert rich.num_key_value_heads == 64
    assert rich.qk_head_dim == 256
    assert rich.head_dim == 192  # HF head_dim (distinct from qk_head_dim)
    assert rich.kv_lora_rank == 512
    assert rich.qk_rope_head_dim == 64
    assert rich.compressed_kv_dim == 576  # kv_lora_rank + qk_rope_head_dim
    assert rich.first_k_dense_replace == 3

    # GLM-5.2 specifics.
    assert rich.max_position_embeddings == 1048576
    assert rich.rope_theta == 8000000  # nested under rope_parameters
    assert rich.index_topk_freq == 4
    assert rich.index_skip_topk_offset == 3
    assert rich.indexer_types is not None and len(rich.indexer_types) == 78

    # Engine projection still uses qk_head_dim.
    assert _project(rich)["head_dim"] == 256


def test_unlisted_variant_warns_and_falls_back(caplog):
    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        rich = BatchGenModelConfig.resolve("Some-Unknown-Model", checkpoint_path=None)
    assert type(rich).__name__ == "GLM5Config"
    assert any("matched no supported variant" in r.message for r in caplog.records)


def test_unlisted_glm5_minor_warns_before_falling_back_to_base(caplog):
    """An unlisted GLM-5.x minor must warn loudly, not silently bind to base."""
    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        rich = BatchGenModelConfig.resolve("zai/GLM-5.3", checkpoint_path=None)
    assert type(rich).__name__ == "GLM5Config"  # base fallback
    assert any("unlisted GLM-5.3 variant" in r.message for r in caplog.records)


def test_glm5_superstring_warns(caplog):
    """GLM-50 is a superstring of GLM-5 and must warn, not bind silently."""
    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        BatchGenModelConfig.resolve("GLM-50-foo", checkpoint_path=None)
    assert any("superstring" in r.message for r in caplog.records)


def _write_config(tmp_path, data):
    import json
    (tmp_path / "config.json").write_text(json.dumps(data))
    return str(tmp_path)


def test_empty_config_fails_loud(tmp_path):
    """A truncated config.json ({}) must FAIL, not silently backfill defaults."""
    ckpt = _write_config(tmp_path, {})
    with pytest.raises(ValueError, match="missing required fields"):
        BatchGenModelConfig.resolve("GLM-5.2-FP8", checkpoint_path=ckpt)


def test_null_required_field_fails_loud(tmp_path):
    """A required field explicitly null in config.json must FAIL loud."""
    import json
    if not (Path(_GLM52_CKPT) / "config.json").exists():
        pytest.skip("GLM-5.2-FP8 checkpoint config.json not available")
    data = json.loads((Path(_GLM52_CKPT) / "config.json").read_text())
    data["num_hidden_layers"] = None
    ckpt = _write_config(tmp_path, data)
    with pytest.raises(ValueError, match="missing required fields"):
        BatchGenModelConfig.resolve("GLM-5.2-FP8", checkpoint_path=ckpt)


def test_noncontiguous_mlp_layer_types_fails_loud(tmp_path):
    """A non-contiguous dense/sparse layout the scalar first_k_dense_replace
    cannot represent must FAIL loud rather than be silently mis-modelled."""
    import json
    if not (Path(_GLM52_CKPT) / "config.json").exists():
        pytest.skip("GLM-5.2-FP8 checkpoint config.json not available")
    data = json.loads((Path(_GLM52_CKPT) / "config.json").read_text())
    layer_types = list(data["mlp_layer_types"])
    layer_types[0] = "sparse"   # break the contiguous dense prefix
    layer_types[5] = "dense"
    data["mlp_layer_types"] = layer_types
    ckpt = _write_config(tmp_path, data)
    with pytest.raises(ValueError, match="mlp_layer_types"):
        BatchGenModelConfig.resolve("GLM-5.2-FP8", checkpoint_path=ckpt)
