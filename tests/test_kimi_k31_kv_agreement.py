# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Model-side gate: the core KV profiles agree with the model's declaration.

``tensor_map`` declares the admission geometry as a property of the MODEL
(derived from ``dsa_config`` and the checkpoint); the Host KV profile tables
declare it as a property of the CACHE (``tests/unit/test_kimi_k31_kv_profile.py``
covers those on their own).  A K3.1 serving build is only correct if the two
agree, and they land in different change sets, so the cross-assertion lives
with the later one.  CPU only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# The core suite owns the profile loader (``host_kv_mananger_config`` by file
# path, its package imports triton) and the per-token byte arithmetic; the
# model suite owns the K3.1 config stand-in.  Reusing both keeps each side of
# the comparison defined exactly once.
_core = _load("_k31_kv_profile_core", ROOT / "tests" / "unit" / "test_kimi_k31_kv_profile.py")
_model = _load("_k31_model_side_tensor_map", ROOT / "tests" / "test_kimi_k31_tensor_map.py")


def test_core_profiles_agree_with_the_model_side_declaration(monkeypatch):
    config = _core._load_host_kv_config(monkeypatch)
    per_token = _core._profile_bytes_per_token
    tm = _model.tm
    cfg = _model._K31Config()
    k3_cfg = _model.k3t._K3Config()

    primary = config._resolve_profile("moonshotai/Kimi-K3.1-Pro")
    aux = config._resolve_indexer_profile("moonshotai/Kimi-K3.1-Pro")

    assert tm.k3_mla_kv_bytes_per_token(cfg) == per_token(primary)
    assert tm.k31_indexer_kv_bytes_per_token(cfg) == per_token(aux)
    assert tm.k3_paged_kv_bytes_per_token(cfg) == per_token(primary) + per_token(aux)

    # And the dense K3 side: no aux region, no indexer bytes.
    assert config._resolve_indexer_profile("moonshotai/Kimi-K3") is None
    assert tm.k31_indexer_kv_bytes_per_token(k3_cfg) == 0
    assert (tm.k3_paged_kv_bytes_per_token(k3_cfg)
            == per_token(config._resolve_profile("moonshotai/Kimi-K3")))

    # The indexer's per-layer width is the MQA key the model caches, not the
    # 32-head query width: a per-head aux region would be 32x too large.
    indexer_dims = tm.k3_indexer_dims(cfg)
    assert aux.k_data_dim() == indexer_dims["head_dim"]
    assert aux.k_data_dim() * indexer_dims["heads"] == indexer_dims["index_dim"]
    assert aux.num_layers == len(cfg.dsa_config["dsa_layers"])


def test_the_name_pattern_shortcut_cannot_serve_k31_unvalidated():
    """Detection is fixed, but the shortcut still returns BUILT-IN DEFAULTS.

    ``load_config`` on a non-local identifier returns ``KimiLinearConfig()``,
    whose ``model_type`` field is the literal ``"kimi_linear"`` and which
    carries no ``dsa_config``.  The loader gate must reject it, so the only way
    to serve K3.1 remains a ``--cache-dir`` with the real ``config.json``.
    """
    from batchgen.config.model_registry import CONFIG_REGISTRY, load_config
    from batchgen.models.moonshotai.kimi_linear.k3 import tensor_map as tm

    assert "kimi_k31" in CONFIG_REGISTRY
    cfg = load_config("moonshotai/Kimi-K3.1-Pro")
    assert cfg.dsa_config is None
    with pytest.raises(ValueError, match="model_type"):
        tm.validate_k3_config(cfg)
