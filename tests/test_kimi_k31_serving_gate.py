# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 serving gate in ``KimiLinearInitializer.__init__``.

With the DSA serving path in place (index keys written by prefill and decode,
sparse prefill past ``index_topk``, sparse decode, the captured graph step),
the initializer builds K3.1 like K3 — after one check: the indexer geometry
must be the one the DSA kernels are instantiated for
(``dsa_runtime.require_servable_indexer``), refused before any weight is
mapped rather than at the first long prompt.

It also pins the ``is_k3`` fix: K3.1 shares every ``is_k3``-keyed serving
behaviour with K3 (resident prefill, streamed MXFP4 rings, TP8 latent/vocab
sharding, KDA state plan, graph buckets), so a ``== "kimi_k3"`` test would
route a 2.8T K3.1 down the 48B testbed plan.

``kimi_initializer`` imports the native loader (UCX + the compiled engine), so
it cannot be imported on a CPU box; ``__init__`` is compiled out of the
module's AST onto a namespace of fakes instead (the
``tests/unit/test_k3_prefix_pool_group.py`` pattern).
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
INITIALIZER = (ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
               / "kimi_initializer.py")
K3_CONFIG_JSON = (ROOT / "batchgen" / "models" / "moonshotai" / "kimi_k3"
                  / "assets" / "config.json")

sys.path.insert(0, str(ROOT))
from tests.test_kimi_k31_config import _apply_k31_delta  # noqa: E402

INITIALIZER_TREE = ast.parse(INITIALIZER.read_text(), filename=str(INITIALIZER))


def _init_function():
    for node in ast.walk(INITIALIZER_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    raise AssertionError("KimiLinearInitializer.__init__ not found")


def _build_initializer(config, model_id):
    """Run ``KimiLinearInitializer.__init__`` against a fake environment.

    Returns ``(instance, raised)``.  The constructor continues past the K3.1
    gate into plain attribute assignment and other guards that need a real
    deployment, so a later failure is expected and reported rather than
    treated as the result: every assertion below is about state the gate has
    already decided.
    """
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig

    module = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(_init_function())], type_ignores=[]))
    from batchgen.models.moonshotai.kimi_linear.config import is_k3_model_type
    from batchgen.models.moonshotai.kimi_linear.dsa_runtime import (
        require_servable_indexer,
    )

    namespace = {
        "load_config": lambda _name: config,
        "KimiLinearConfig": KimiLinearConfig,
        "is_k3_model_type": is_k3_model_type,
        "require_servable_indexer": require_servable_indexer,
        "os": os,
        "logging": logging,
        "torch": types.SimpleNamespace(),
    }
    exec(compile(module, str(INITIALIZER), "exec"), namespace)

    instance = SimpleNamespace()
    args = SimpleNamespace(
        huggingface_ckpt_name=model_id,
        cache_dir=None,
        host_kv_cache_size=64,
        global_host_kv_cache_size_gb=64,
        local_rank=0,
        global_rank=0,
        world_size=8,
    )
    raised = None
    try:
        namespace["__init__"](instance, args)
    except Exception as exc:  # noqa: BLE001 — classified by the caller
        raised = exc
    return instance, raised


def _config_from(raw: dict):
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig

    return KimiLinearConfig.from_hf_dict(raw)


@pytest.fixture()
def k3_raw() -> dict:
    with open(K3_CONFIG_JSON, "r") as handle:
        return json.load(handle)


@pytest.fixture()
def k31_raw(k3_raw) -> dict:
    return _apply_k31_delta(copy.deepcopy(k3_raw))


# --------------------------------------------------------------------------- #
#  The gate                                                                   #
# --------------------------------------------------------------------------- #

def test_a_released_k31_config_builds_past_the_gate(k31_raw):
    config = _config_from(k31_raw)
    assert config.model_type == "kimi_k31"
    instance, raised = _build_initializer(config, "moonshotai/Kimi-K3.1-Pro")
    assert not isinstance(raised, NotImplementedError), raised
    assert instance.is_k31 is True and instance.is_k3 is True
    # past the gate: the constructor reached the plain assignments
    assert instance.loaded_model_config is config
    assert instance.host_kv_cache_size == 64


@pytest.mark.parametrize("key, value, match", [
    ("index_topk", 1024, "index_topk=1024"),
    ("index_head_dim", 64, "index_head_dim=64"),
    ("index_num_heads", 48, "index_num_heads=48"),
])
def test_an_unservable_indexer_geometry_is_refused_before_any_weight_map(
        k31_raw, key, value, match):
    raw = copy.deepcopy(k31_raw)
    raw["text_config"]["dsa_config"][key] = value
    config = _config_from(raw)
    instance, raised = _build_initializer(config, "moonshotai/Kimi-K3.1-Pro")
    assert isinstance(raised, NotImplementedError), raised
    assert match in str(raised)
    assert instance.is_k31 is True
    assert not hasattr(instance, "loaded_model_config")


def test_the_gate_reads_the_parsed_dsa_config():
    from batchgen.models.moonshotai.kimi_linear.dsa_runtime import (
        require_servable_indexer,
    )

    with pytest.raises(ValueError, match="without dsa_config"):
        require_servable_indexer(None)


# --------------------------------------------------------------------------- #
#  is_k3 covers both identities                                               #
# --------------------------------------------------------------------------- #

def test_k3_still_builds_past_the_gate_with_is_k3_set(k3_raw):
    config = _config_from(k3_raw)
    assert config.model_type == "kimi_k3"
    instance, raised = _build_initializer(config, "moonshotai/Kimi-K3")
    assert not isinstance(raised, NotImplementedError), raised
    assert instance.is_k3 is True
    assert instance.is_k31 is False
    # Past the gate: the constructor got as far as the plain assignments.
    assert instance.loaded_model_config is config
    assert instance.host_kv_cache_size == 64


def test_k31_sets_is_k3_so_it_cannot_take_the_48b_plan(k31_raw):
    """``is_k3`` feeds the planner (``planner.py`` ``is_k3=``), which selects
    the 93-layer / 896-expert streamed plan.  A K3.1 that reported False would
    be planned as the 48B testbed."""
    config = _config_from(k31_raw)
    instance, _ = _build_initializer(config, "moonshotai/Kimi-K3.1-Pro")
    assert instance.is_k3 is True


def test_the_48b_is_neither_k3_nor_k31():
    config = _config_from({
        "model_type": "kimi_linear",
        "hidden_size": 2304,
        "num_hidden_layers": 27,
    })
    instance, raised = _build_initializer(
        config, "moonshotai/Kimi-Linear-48B-A3B-Instruct")
    assert not isinstance(raised, NotImplementedError), raised
    assert instance.is_k3 is False
    assert instance.is_k31 is False


# --------------------------------------------------------------------------- #
#  The pre-existing mis-resolution guard still works, for both identities      #
# --------------------------------------------------------------------------- #

def test_a_k31_identifier_resolving_to_48b_defaults_is_refused():
    """The name-pattern shortcut hands back ``KimiLinearConfig()`` with 48B
    defaults, whose ``model_type`` is the literal ``"kimi_linear"``."""
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig

    instance, raised = _build_initializer(
        KimiLinearConfig(), "moonshotai/Kimi-K3.1-Pro")
    assert isinstance(raised, RuntimeError)
    assert not isinstance(raised, NotImplementedError)
    assert "kimi_k31" in str(raised)
    assert "--cache-dir" in str(raised)


def test_a_k3_identifier_resolving_to_48b_defaults_is_still_refused():
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig

    instance, raised = _build_initializer(
        KimiLinearConfig(), "moonshotai/Kimi-K3")
    assert isinstance(raised, RuntimeError)
    assert "--cache-dir" in str(raised)
