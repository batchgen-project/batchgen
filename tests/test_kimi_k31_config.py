# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Strict-parser gate for Kimi-K3.1's ``dsa_config``.  CPU only, no weights.

Loads ``kimi_k3/config.py`` by file path (the ``tests/test_kimi_k3_tensor_map.py``
pattern), so no ``import batchgen`` and no core-engine build.

The K3.1 fixture is built as the VERIFIED DELTA on top of the vendored K3
``assets/config.json`` — see :data:`K31_DELTA` for the complete list, measured
by diffing the two released files leaf key by leaf key (6 differences plus the
``dsa_config`` block).  Expressing it as a delta rather than a second vendored
file is what makes ``test_fixture_is_k3_plus_exactly_the_dsa_delta``
meaningful: the K3 side of every assertion below is the real K3 config, byte
for byte.  ``_apply_k31_delta`` applied to the vendored K3 config was checked
deep-equal to the released ``moonshotai/Kimi-K3.1-Pro/config.json``
(2026-11-05), so the fixture is that file and not an approximation of it.

One of those differences is itself a finding: K3's ``auto_map`` points
``AutoModelForCausalLM`` at ``modeling_kimi_k3.py`` and its
``text_config.auto_map`` at ``modeling_kimi_linear.py``, while **K3.1 ships
``AutoConfig`` alone and no ``text_config.auto_map`` at all** — it publishes no
modeling code, so there is no upstream statement of the index math (both raw
``modeling_*.py`` paths return HTTP 404).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
K3_PKG_DIR = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_k3"
K3_CONFIG_JSON = K3_PKG_DIR / "assets" / "config.json"
GLM5_DSA_SCHEDULE = ROOT / "batchgen" / "models" / "glm" / "glm5" / "dsa_schedule.py"


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_pkg = types.ModuleType("_k31_cfg_pkg")
_pkg.__path__ = [str(K3_PKG_DIR)]
sys.modules.setdefault("_k31_cfg_pkg", _pkg)
cfgmod = _load("_k31_cfg_pkg.config", K3_PKG_DIR / "config.py")
glm5_schedule = _load("_k31_glm5_dsa_schedule", GLM5_DSA_SCHEDULE)


# --------------------------------------------------------------------------- #
#  Released facts                                                             #
# --------------------------------------------------------------------------- #

#: ``text_config.dsa_config`` of ``moonshotai/Kimi-K3.1-Pro``, verbatim.
K31_DSA_CONFIG = {
    "dsa_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                   64, 68, 72, 76, 80, 84, 88, 92, 93],
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}

#: The two patterns K3.1's ``quantization_config['ignore']`` adds to K3's,
#: at the position the released file puts them (after the dense-MLP pattern).
#: The gate compares the list as a set, so the position is cosmetic — it is
#: reproduced only so the fixture is the released file byte for byte.
K31_ADDED_QUANT_IGNORE = [
    r"re:.*block_sparse_moe\.gate$",
    r"re:.*block_sparse_moe\.routed_expert_(up|down)_proj.*",
]
K31_QUANT_IGNORE_INSERT_AT = 3

#: K3.1 publishes a config and a processor, and no modeling code.
K31_AUTO_MAP = {"AutoConfig": "configuration_kimi_k3.KimiK3Config"}

#: Everything the released K3.1 ``config.json`` changes relative to K3's,
#: as a leaf-key diff of the two files.
K31_DELTA = (
    "model_type: kimi_k3 -> kimi_k31",
    "architectures: KimiK3ForConditionalGeneration -> KimiK31ForConditionalGeneration",
    "auto_map: drops AutoModel / AutoModelForCausalLM (no modeling code shipped)",
    "text_config.auto_map: removed entirely",
    "text_config.transformers_version: 4.56.2 -> 4.57.6",
    "text_config.dsa_config: added (the 8 keys below)",
    "text_config.quantization_config.ignore: + 2 patterns (see above)",
)


def _apply_k31_delta(raw: dict) -> dict:
    raw["model_type"] = "kimi_k31"
    raw["architectures"] = ["KimiK31ForConditionalGeneration"]
    raw["auto_map"] = dict(K31_AUTO_MAP)
    raw["text_config"].pop("auto_map", None)
    raw["text_config"]["transformers_version"] = "4.57.6"
    raw["text_config"]["dsa_config"] = copy.deepcopy(K31_DSA_CONFIG)
    ignore = list(raw["text_config"]["quantization_config"]["ignore"])
    ignore[K31_QUANT_IGNORE_INSERT_AT:K31_QUANT_IGNORE_INSERT_AT] = list(
        K31_ADDED_QUANT_IGNORE)
    raw["text_config"]["quantization_config"]["ignore"] = ignore
    return raw

#: 0-BASED checkpoint layer indexes carrying ``self_attn.indexer.*`` — read out
#: of the released ``model.safetensors.index.json`` diff (5 patterns x these 24
#: layers = 120 new tensors, 0 removed).  Identical to the layers carrying
#: ``self_attn.q_a_proj``, i.e. the full-attention layers.
K31_INDEXER_CKPT_LAYERS = tuple(range(3, 92, 4)) + (92,)


@pytest.fixture()
def k3_raw() -> dict:
    with open(K3_CONFIG_JSON, "r") as handle:
        return json.load(handle)


@pytest.fixture()
def k31_raw(k3_raw) -> dict:
    return _apply_k31_delta(copy.deepcopy(k3_raw))


def test_fixture_is_k3_plus_exactly_the_dsa_delta(k3_raw, k31_raw):
    """The fixture must not drift into a second, independently edited config.

    Reverting every item of :data:`K31_DELTA` must give back the vendored K3
    config exactly, so no assertion below can rest on an accidental edit.
    """
    stripped = copy.deepcopy(k31_raw)
    stripped["model_type"] = k3_raw["model_type"]
    stripped["architectures"] = k3_raw["architectures"]
    stripped["auto_map"] = k3_raw["auto_map"]
    stripped["text_config"]["auto_map"] = k3_raw["text_config"]["auto_map"]
    stripped["text_config"]["transformers_version"] = (
        k3_raw["text_config"]["transformers_version"])
    del stripped["text_config"]["dsa_config"]
    ignore = stripped["text_config"]["quantization_config"]["ignore"]
    for pattern in K31_ADDED_QUANT_IGNORE:
        assert pattern in ignore
        ignore.remove(pattern)
    assert stripped == k3_raw


def test_k31_publishes_no_modeling_code(k3_raw, k31_raw):
    """K3's auto_map names two modeling modules; K3.1's names none.

    This is why the indexer has no upstream Python reference to mirror: the
    only published statement about it is the eight ``dsa_config`` keys and the
    120 tensors in the checkpoint.
    """
    assert set(k3_raw["auto_map"]) == {
        "AutoConfig", "AutoModel", "AutoModelForCausalLM"}
    assert "modeling_kimi_k3" in k3_raw["auto_map"]["AutoModelForCausalLM"]
    assert "modeling_kimi_linear" in (
        k3_raw["text_config"]["auto_map"]["AutoModelForCausalLM"])

    assert k31_raw["auto_map"] == K31_AUTO_MAP
    assert "auto_map" not in k31_raw["text_config"]
    assert not any("modeling" in value for value in k31_raw["auto_map"].values())
    # Parsing must not depend on an auto_map being present at all.
    cfgmod.parse_k3_config(k31_raw)


# --------------------------------------------------------------------------- #
#  K3.1 parses; K3 is untouched                                               #
# --------------------------------------------------------------------------- #

def test_k31_parses_and_round_trips_every_dsa_key(k31_raw):
    cfg = cfgmod.parse_k3_config(k31_raw)
    assert cfg.model_type == "kimi_k31"
    dsa = cfg.dsa_config
    assert dsa is not None
    assert dsa.dsa_layers == tuple(K31_DSA_CONFIG["dsa_layers"])
    assert dsa.index_topk == 2048
    assert dsa.index_head_dim == 128
    assert dsa.index_num_heads == 32
    assert dsa.index_rope_dim == 0
    assert dsa.index_k_norm_type == "layernorm"
    assert dsa.index_topk_freq == 1
    assert dsa.index_skip_topk_offset == 1
    assert dsa.index_dim == 4096


def test_dsa_layers_are_exactly_the_checkpoint_indexer_layers(k31_raw):
    """1-based config list vs 0-based checkpoint names — the off-by-one that
    would load every indexer into the wrong layer."""
    cfg = cfgmod.parse_k3_config(k31_raw)
    dsa_layers = [i for i in range(cfg.num_hidden_layers) if cfg.is_dsa_layer(i)]
    mla_layers = [i for i in range(cfg.num_hidden_layers) if cfg.is_mla_layer(i)]
    assert tuple(dsa_layers) == K31_INDEXER_CKPT_LAYERS
    assert dsa_layers == mla_layers
    assert len(dsa_layers) == 24


def test_k3_config_is_dense_and_unchanged(k3_raw):
    cfg = cfgmod.parse_k3_config(k3_raw)
    assert cfg.model_type == "kimi_k3"
    assert cfg.dsa_config is None
    assert not any(cfg.is_dsa_layer(i) for i in range(cfg.num_hidden_layers))
    # The K3 layer split itself must be untouched by the DSA work.
    assert sum(cfg.is_kda_layer(i) for i in range(93)) == 69
    assert cfg.is_kda_layer(0)


def test_k3_must_not_carry_a_dsa_config(k3_raw):
    k3_raw["text_config"]["dsa_config"] = copy.deepcopy(K31_DSA_CONFIG)
    with pytest.raises(ValueError, match="must not carry a dsa_config"):
        cfgmod.parse_k3_config(k3_raw)


def test_k31_without_dsa_config_is_refused(k31_raw):
    del k31_raw["text_config"]["dsa_config"]
    with pytest.raises(ValueError, match="requires text_config.dsa_config"):
        cfgmod.parse_k3_config(k31_raw)


def test_unknown_wrapper_model_type_is_refused(k31_raw):
    k31_raw["model_type"] = "kimi_k32"
    with pytest.raises(ValueError, match="top-level model_type"):
        cfgmod.parse_k3_config(k31_raw)


# --------------------------------------------------------------------------- #
#  Unknown / missing / unimplemented                                          #
# --------------------------------------------------------------------------- #

def test_unknown_dsa_key_is_refused(k31_raw):
    k31_raw["text_config"]["dsa_config"]["index_rope_theta"] = 10000.0
    with pytest.raises(ValueError, match="unknown key"):
        cfgmod.parse_k3_config(k31_raw)


@pytest.mark.parametrize("key", sorted(K31_DSA_CONFIG))
def test_every_dsa_key_is_required(k31_raw, key):
    """No defaulting: each of the eight keys sizes a tensor, a cache row or the
    selection itself."""
    del k31_raw["text_config"]["dsa_config"][key]
    with pytest.raises(ValueError, match=r"required key '{}'".format(key)):
        cfgmod.parse_k3_config(k31_raw)


@pytest.mark.parametrize("key", sorted(K31_DSA_CONFIG))
def test_a_null_dsa_key_is_refused(k31_raw, key):
    k31_raw["text_config"]["dsa_config"][key] = None
    with pytest.raises(ValueError):
        cfgmod.parse_k3_config(k31_raw)


def test_index_rope_dim_zero_is_the_only_implemented_index_path(k31_raw):
    k31_raw["text_config"]["dsa_config"]["index_rope_dim"] = 64
    with pytest.raises(NotImplementedError, match="index_rope_dim"):
        cfgmod.parse_k3_config(k31_raw)


def test_only_layernorm_k_norm_is_implemented(k31_raw):
    k31_raw["text_config"]["dsa_config"]["index_k_norm_type"] = "rmsnorm"
    with pytest.raises(NotImplementedError, match="index_k_norm_type"):
        cfgmod.parse_k3_config(k31_raw)


def test_a_topk_reuse_schedule_is_refused(k31_raw):
    """``index_topk_freq > 1`` means some DSA layers reuse a previous layer's
    selection and ship no indexer weights.  K3.1 ships 24/24, so a config
    declaring reuse contradicts its own checkpoint."""
    k31_raw["text_config"]["dsa_config"]["index_topk_freq"] = 4
    k31_raw["text_config"]["dsa_config"]["index_skip_topk_offset"] = 3
    with pytest.raises(NotImplementedError, match="REUSE"):
        cfgmod.parse_k3_config(k31_raw)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_topk_freq_is_refused(k31_raw, bad):
    k31_raw["text_config"]["dsa_config"]["index_topk_freq"] = bad
    with pytest.raises(ValueError, match="index_topk_freq must be positive"):
        cfgmod.parse_k3_config(k31_raw)


def test_zero_indexed_dsa_layers_are_refused(k31_raw):
    """A 0-indexed read would put the indexer one layer off on every layer and
    on layer 3 (a KDA layer) at the end of the stack."""
    k31_raw["text_config"]["dsa_config"]["dsa_layers"] = [
        i - 1 for i in K31_DSA_CONFIG["dsa_layers"]]
    with pytest.raises(ValueError, match="full_attn_layers"):
        cfgmod.parse_k3_config(k31_raw)


def test_dsa_layers_must_match_full_attn_layers(k31_raw):
    layers = list(K31_DSA_CONFIG["dsa_layers"])
    layers[0] = 5          # 5 is a KDA layer (1-based) — no MLA to gate
    k31_raw["text_config"]["dsa_config"]["dsa_layers"] = layers
    with pytest.raises(ValueError, match="full_attn_layers"):
        cfgmod.parse_k3_config(k31_raw)


def test_out_of_range_dsa_layer_is_refused(k31_raw):
    k31_raw["text_config"]["dsa_config"]["dsa_layers"] = (
        list(K31_DSA_CONFIG["dsa_layers"]) + [94])
    with pytest.raises(ValueError, match="outside 1"):
        cfgmod.parse_k3_config(k31_raw)


def test_duplicate_dsa_layer_is_refused(k31_raw):
    k31_raw["text_config"]["dsa_config"]["dsa_layers"] = (
        list(K31_DSA_CONFIG["dsa_layers"]) + [4])
    with pytest.raises(ValueError, match="duplicates"):
        cfgmod.parse_k3_config(k31_raw)


@pytest.mark.parametrize("key", ["index_topk", "index_head_dim", "index_num_heads"])
def test_non_positive_geometry_is_refused(k31_raw, key):
    k31_raw["text_config"]["dsa_config"][key] = 0
    with pytest.raises(ValueError, match="must be positive"):
        cfgmod.parse_k3_config(k31_raw)


# --------------------------------------------------------------------------- #
#  The reuse schedule agrees with the one the repo already carries            #
# --------------------------------------------------------------------------- #

class _FreqOffset:
    def __init__(self, freq, offset):
        self.index_topk_freq = freq
        self.index_skip_topk_offset = offset


@pytest.mark.parametrize("freq", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("offset", [1, 2, 3, 4, 5])
def test_schedule_formula_matches_the_glm5_port(freq, offset):
    """``kimi_k3.config.dsa_layer_skips_topk`` restates, as a pure function of
    the two integers, the formula ``glm5/dsa_schedule.py`` already carries.
    Any divergence would schedule the indexer differently from the only other
    DSA model in the tree."""
    holder = _FreqOffset(freq, offset)
    for layer in range(40):
        assert (cfgmod.dsa_layer_skips_topk(freq, offset, layer)
                == glm5_schedule.dsa_layer_skips_topk(holder, layer)), (
            "freq={} offset={} layer={}".format(freq, offset, layer))


def test_freq_one_never_skips():
    assert not any(cfgmod.dsa_layer_skips_topk(1, 1, layer) for layer in range(200))
