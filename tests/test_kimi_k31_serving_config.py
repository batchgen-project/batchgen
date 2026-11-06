# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Registration seam for Kimi-K3.1: the serving config class, the config
registry and the tokenizer family.  CPU only, no weights, no GPU.

Unlike ``tests/test_kimi_k31_config.py`` (which loads the strict parser by file
path) this module exercises ``KimiLinearConfig`` — the class the serving stack,
the parameter server and the planner actually hold — so it imports ``batchgen``.

It also PINS the two places a K3.1 identity is resolved from a HuggingFace
model id rather than from an on-disk ``config.json``, because one of them is
correct today and the other is not:

  * the tokenizer family resolves ``moonshotai/Kimi-K3.1-Pro`` to ``kimi_k3``,
    which is RIGHT: the released K3.1 ``tokenizer_config.json`` is
    field-for-field the vendored K3 one (same ``added_tokens_decoder`` names at
    163584/163586-163591, same ``tokenizer_class``, same
    ``additional_special_tokens``, and no ``chat_template`` — the XTML renderer
    with ``thinking_effort`` in {low, high, max}, defaulting to max, is shared);
  * the CONFIG name-pattern shortcut now resolves it to ``kimi_k31`` as well,
    through the explicit ``"Kimi-K3.1"`` patterns the companion `core` change
    put ahead of the broad ``"Kimi-K3"`` entry in
    ``batchgen/config/model_registry.py``.  That shortcut still returns BUILT-IN
    DEFAULTS rather than the checkpoint's own fields, so the supported path
    remains a ``--cache-dir`` with the real ``config.json``; the loader gate
    rejects the defaults, which ``tests/unit/test_kimi_k31_kv_profile.py``
    pins.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from batchgen.config.model_registry import (
    CONFIG_REGISTRY,
    _detect_model_type_from_identifier,
    load_config,
)
from batchgen.config.tokenizer_registry import resolve_tokenizer_type
from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig
from batchgen.models.moonshotai.kimi_linear.k3 import tensor_map as tm

ROOT = Path(__file__).resolve().parents[1]
K3_ASSETS = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_k3" / "assets"
K3_CONFIG_JSON = K3_ASSETS / "config.json"
K3_TOKENIZER_CONFIG_JSON = K3_ASSETS / "tokenizer_config.json"

K31_MODEL_ID = "moonshotai/Kimi-K3.1-Pro"

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
K31_ADDED_QUANT_IGNORE = [
    r"re:.*block_sparse_moe\.gate$",
    r"re:.*block_sparse_moe\.routed_expert_(up|down)_proj.*",
]
K31_INDEXER_CKPT_LAYERS = tuple(range(3, 92, 4)) + (92,)


@pytest.fixture()
def k31_config_json(tmp_path) -> Path:
    """The released K3.1 ``config.json``, as K3's plus the verified delta.

    The delta is enumerated and reverted in
    ``tests/test_kimi_k31_config.py::test_fixture_is_k3_plus_exactly_the_dsa_delta``;
    it is applied here too so this module exercises the same bytes.
    """
    with open(K3_CONFIG_JSON, "r") as handle:
        raw = json.load(handle)
    raw = copy.deepcopy(raw)
    raw["model_type"] = "kimi_k31"
    raw["architectures"] = ["KimiK31ForConditionalGeneration"]
    raw["auto_map"] = {"AutoConfig": "configuration_kimi_k3.KimiK3Config"}
    raw["text_config"].pop("auto_map", None)
    raw["text_config"]["transformers_version"] = "4.57.6"
    raw["text_config"]["dsa_config"] = copy.deepcopy(K31_DSA_CONFIG)
    ignore = list(raw["text_config"]["quantization_config"]["ignore"])
    ignore[3:3] = list(K31_ADDED_QUANT_IGNORE)
    raw["text_config"]["quantization_config"]["ignore"] = ignore
    path = tmp_path / "config.json"
    with open(path, "w") as handle:
        json.dump(raw, handle)
    return path


# --------------------------------------------------------------------------- #
#  Registry                                                                   #
# --------------------------------------------------------------------------- #

def test_kimi_k31_is_a_registered_config_type():
    assert CONFIG_REGISTRY["kimi_k31"] is KimiLinearConfig
    # and the pre-existing identities are untouched
    assert CONFIG_REGISTRY["kimi_k3"] is KimiLinearConfig
    assert CONFIG_REGISTRY["kimi_linear"] is KimiLinearConfig


def test_load_config_on_a_k31_directory_stamps_the_k31_identity(k31_config_json):
    cfg = load_config(str(k31_config_json.parent))
    assert cfg.model_type == "kimi_k31"
    assert cfg.dsa_config == K31_DSA_CONFIG
    assert cfg.num_dsa_layers == 24


def test_load_config_on_the_k3_assets_stays_dense():
    cfg = load_config(str(K3_CONFIG_JSON.parent))
    assert cfg.model_type == "kimi_k3"
    assert cfg.dsa_config is None
    assert cfg.num_dsa_layers == 0


# --------------------------------------------------------------------------- #
#  KimiLinearConfig                                                           #
# --------------------------------------------------------------------------- #

def test_k31_config_round_trips_the_dsa_block(k31_config_json):
    cfg = KimiLinearConfig.from_json(str(k31_config_json))
    assert cfg.model_type == "kimi_k31"
    assert cfg.quantization == "mxfp4"
    assert cfg.dsa_config == K31_DSA_CONFIG

    dsa = cfg.dsa()
    assert dsa.index_topk == 2048
    assert dsa.index_num_heads == 32
    assert dsa.index_head_dim == 128
    assert dsa.index_dim == 4096
    assert dsa.index_rope_dim == 0


def test_dsa_layers_are_the_full_attention_layers(k31_config_json):
    cfg = KimiLinearConfig.from_json(str(k31_config_json))
    dsa_layers = [i for i in range(cfg.num_hidden_layers) if cfg.is_dsa_layer(i)]
    mla_layers = [i for i in range(cfg.num_hidden_layers) if cfg.is_mla_layer(i)]
    assert tuple(dsa_layers) == K31_INDEXER_CKPT_LAYERS
    assert dsa_layers == mla_layers


def test_k3_config_is_unchanged():
    cfg = KimiLinearConfig.from_json(str(K3_CONFIG_JSON))
    assert cfg.model_type == "kimi_k3"
    assert cfg.quantization == "mxfp4"
    assert cfg.dsa_config is None
    assert cfg.dsa() is None
    assert not any(cfg.is_dsa_layer(i) for i in range(cfg.num_hidden_layers))
    assert cfg.num_kda_layers == 69
    assert cfg.num_mla_layers == 24


def test_a_nested_config_with_an_unknown_wrapper_type_is_refused(k31_config_json):
    with open(k31_config_json, "r") as handle:
        raw = json.load(handle)
    raw["model_type"] = "kimi_k9"
    with pytest.raises(ValueError, match="wrapper model_type"):
        KimiLinearConfig.from_hf_dict(raw)


def test_the_flat_48b_config_still_loads_as_kimi_linear():
    """Kimi-Linear-48B has no ``text_config`` and must not be touched by the
    wrapper-identity requirement."""
    cfg = KimiLinearConfig.from_hf_dict({
        "model_type": "kimi_linear",
        "hidden_size": 2304,
        "num_hidden_layers": 27,
    })
    assert cfg.model_type == "kimi_linear"
    assert cfg.dsa_config is None
    assert cfg.num_dsa_layers == 0


# --------------------------------------------------------------------------- #
#  Loader gate                                                                #
# --------------------------------------------------------------------------- #

def test_the_loader_gate_accepts_k31_and_declares_the_indexer(k31_config_json):
    cfg = KimiLinearConfig.from_json(str(k31_config_json))
    tm.validate_k3_config(cfg)
    name_map, task = tm.build_k3_state_dict_name_map(cfg)
    indexer = [n for n in name_map if ".self_attn.indexer." in n]
    assert len(indexer) == 120
    assert len(task["attn"]) == 24
    assert tm.k3_paged_kv_bytes_per_token(cfg) == 30_816


def test_the_loader_gate_still_accepts_k3_with_its_own_numbers():
    cfg = KimiLinearConfig.from_json(str(K3_CONFIG_JSON))
    tm.validate_k3_config(cfg)
    name_map, _ = tm.build_k3_state_dict_name_map(cfg)
    assert len(name_map) == 496_026
    assert not any(".indexer." in n for n in name_map)
    assert tm.k3_paged_kv_bytes_per_token(cfg) == 27_648


def test_load_k3_config_reads_a_k31_directory(k31_config_json):
    cfg = tm.load_k3_config(str(k31_config_json.parent))
    assert cfg.model_type == "kimi_k31"
    assert cfg.num_dsa_layers == 24


# --------------------------------------------------------------------------- #
#  Tokenizer family                                                           #
# --------------------------------------------------------------------------- #

def test_k31_resolves_to_the_k3_tokenizer_family():
    assert resolve_tokenizer_type(K31_MODEL_ID) == "kimi_k3"
    assert resolve_tokenizer_type("moonshotai/Kimi-K3") == "kimi_k3"
    assert resolve_tokenizer_type("moonshotai/Kimi-Linear-48B-A3B") == "kimi_linear"


def test_the_vendored_k3_tokenizer_config_matches_the_released_k31_one():
    """Pins the fields that make the family shared, as read from
    ``moonshotai/Kimi-K3.1-Pro/tokenizer_config.json``.  If K3.1 had renamed a
    control token or shipped a Jinja ``chat_template``, the shared family above
    would be a silent cross-load (bug_log.md 2026-07-31: a 12-token K3 fragment
    renders as 32 marker-free tokens under the 48B's table)."""
    with open(K3_TOKENIZER_CONFIG_JSON, "r") as handle:
        vendored = json.load(handle)
    assert "chat_template" not in vendored
    assert vendored["tokenizer_class"] == "TikTokenTokenizer"
    assert vendored["bos_token"] == "[BOS]"
    assert vendored["eos_token"] == "[EOS]"
    assert vendored["pad_token"] == "[PAD]"
    assert vendored["additional_special_tokens"] == [
        "<|end_of_msg|>", "[start_header_id]", "[end_header_id]", "[EOT]",
        "<|media_begin|>", "<|media_content|>", "<|media_end|>",
        "<|media_pad|>", "<osagent_mode>",
    ]
    decoder = vendored["added_tokens_decoder"]
    assert {
        token_id: decoder[token_id]["content"]
        for token_id in ("163584", "163586", "163587", "163588", "163589",
                         "163590", "163591")
    } == {
        "163584": "[BOS]",
        "163586": "<|end_of_msg|>",
        "163587": "<|open|>",
        "163588": "<|close|>",
        "163589": "<|sep|>",
        "163590": "[start_header_id]",
        "163591": "[end_header_id]",
    }


def test_the_thinking_effort_contract_is_shared_with_k3():
    """K3.1 ships no Jinja template either, so the XTML renderer's effort
    levels and its default are what a K3.1 request gets."""
    from batchgen.models.moonshotai.kimi_k3.assets import encoding_k3

    assert encoding_k3._VALID_THINKING_EFFORTS == {"low", "high", "max"}
    source = (K3_ASSETS / "tokenization_kimi.py").read_text()
    assert 'kwargs.setdefault("thinking_effort", "max")' in source


# --------------------------------------------------------------------------- #
#  The HF-id name-pattern shortcut                                            #
# --------------------------------------------------------------------------- #

def test_the_name_pattern_shortcut_resolves_the_k31_id_to_k31():
    """``"Kimi-K3"`` is a substring of ``"Kimi-K3.1-Pro"``, so until an
    explicit earlier pattern existed the non-local identifier path bound K3.1
    to the DENSE identity.

    The fix is a `core` change (``batchgen/config/model_registry.py`` is
    outside the model-change allowlist, PR_MERGE_POLICY §2.5) and is covered in
    detail by ``tests/unit/test_kimi_k31_kv_profile.py``; this is the
    model-side end of the same contract.
    """
    assert _detect_model_type_from_identifier(K31_MODEL_ID) == "kimi_k31"
    assert _detect_model_type_from_identifier("moonshotai/Kimi-K3") == "kimi_k3"
