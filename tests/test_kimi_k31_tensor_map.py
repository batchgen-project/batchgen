# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Offline gate for the Kimi-K3.1 name map / module shapes.  CPU only.

Same contract as ``tests/test_kimi_k3_tensor_map.py`` and built ON it: that
module's template tables ARE the released Kimi-K3 checkpoint, and this one adds
the five indexer patterns on the 24 full-attention layers — the complete,
measured delta between the two released ``model.safetensors.index.json`` files
(120 tensors added, 0 removed, 0 changed).

Nothing in the fixture is computed from ``tensor_map``: the indexer shapes
below come from the index diff's own byte arithmetic
(357,052,416 B / 24 layers = 14,877,184 B/layer), and
``test_fixture_reproduces_the_released_k31_index`` asserts that summing the
whole fixture gives K3.1's ``metadata.total_size``.  So the reconciliation has
no free parameter, and a mutation of the declarations moves one side only.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# The K3 suite owns the loader shims, the reconciler handle and the checkpoint
# template tables; reusing them is what keeps the K3 side of every comparison
# below byte-identical to the K3 gate's own.
k3t = _load("_k31_k3_tensor_map", ROOT / "tests" / "test_kimi_k3_tensor_map.py")
tm = k3t.tm
wr = k3t.wr

BF16 = torch.bfloat16


# --------------------------------------------------------------------------- #
#  Pinned facts of the released Kimi-K3.1-Pro index                           #
# --------------------------------------------------------------------------- #

K31_CKPT_TENSORS = 497_340                     # K3: 497,220 (+120)
K31_CKPT_TOTAL_BYTES = 1_561_217_377_280       # K3: 1,560,860,324,864
K31_INDEXER_TENSORS = 120                      # 5 patterns x 24 layers
K31_INDEXER_BYTES = 357_052_416                # = total(K3.1) - total(K3)
K31_INDEXER_BYTES_PER_LAYER = 14_877_184       # = K31_INDEXER_BYTES / 24
K31_N_MODULE_TENSORS = k3t.N_MODULE_TENSORS + K31_INDEXER_TENSORS
# Vision is byte-identical between the two checkpoints (the diff adds and
# removes nothing under vision_tower./mm_projector.).
K31_N_IGNORED_TENSORS = k3t.N_IGNORED_TENSORS
K31_IGNORED_BYTES = k3t.IGNORED_BYTES

#: Dense MLA ring slot (K3) plus the indexer.
K3_ATTN_SLOT_BYTES = 464_392_192
K31_ATTN_SLOT_BYTES = K3_ATTN_SLOT_BYTES + K31_INDEXER_BYTES_PER_LAYER

#: Per-layer indexer templates, with the shape and dtype implied by the
#: released index's byte delta and the declared ``dsa_config`` geometry
#: (index_num_heads 32, index_head_dim 128, q_lora_rank 1536, hidden 7168):
#: 12,582,912 + 1,835,008 + 458,752 + 256 + 256 = 14,877,184 B.
_INDEXER_TEMPLATES: Tuple[Tuple[str, List[int], torch.dtype], ...] = (
    ("self_attn.indexer.wq_b.weight", [4096, 1536], BF16),
    ("self_attn.indexer.wk.weight", [128, 7168], BF16),
    ("self_attn.indexer.weights_proj.weight", [32, 7168], BF16),
    ("self_attn.indexer.k_norm.weight", [128], BF16),
    ("self_attn.indexer.k_norm.bias", [128], BF16),
)

_PREFIX = k3t._PREFIX

#: The released ``dsa_config``, 1-INDEXED like ``linear_attn_config``'s lists.
_DSA_1IDX = list(k3t._FULL_ATTN_1IDX)
_K31_DSA_CONFIG: Dict[str, Any] = {
    "dsa_layers": _DSA_1IDX,
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}

#: K3.1's ``quantization_config['ignore']`` adds two patterns naming modules
#: the K3 checkpoint already ships BF16, so no module_shapes entry changes.
_K31_ADDED_QUANT_IGNORE = (
    r"re:.*block_sparse_moe\.gate$",
    r"re:.*block_sparse_moe\.routed_expert_(up|down)_proj.*",
)


@dataclass
class _K31Config(k3t._K3Config):
    """The K3 stand-in with the K3.1 identity and indexer block."""

    model_type: str = "kimi_k31"
    dsa_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(_K31_DSA_CONFIG))

    def __post_init__(self):
        ignore = list(self.quantization_config["ignore"])
        for pattern in _K31_ADDED_QUANT_IGNORE:
            if pattern not in ignore:
                ignore.append(pattern)
        self.quantization_config["ignore"] = ignore


@pytest.fixture()
def cfg():
    return _K31Config()


@pytest.fixture()
def k3_cfg():
    return k3t._K3Config()


# --------------------------------------------------------------------------- #
#  The fixture must reproduce the artifact                                     #
# --------------------------------------------------------------------------- #

def _k31_checkpoint(cfg, with_tensors: bool = True):
    """The released K3.1 checkpoint = the K3 fixture + the 120 indexer tensors."""
    base = k3t._checkpoint(k3t._K3Config(), with_tensors=True)
    tensors = dict(base.tensors)
    dsa_1idx = set(int(x) for x in cfg.dsa_config["dsa_layers"])
    for layer in range(cfg.num_hidden_layers):
        if (layer + 1) not in dsa_1idx:
            continue
        for suffix, shape, dtype in _INDEXER_TEMPLATES:
            name = "{}model.layers.{}.{}".format(_PREFIX, layer, suffix)
            tensors[name] = wr.CkptTensor(tuple(shape), dtype,
                                          wr.nbytes(shape, dtype))
    return wr.CheckpointIndex(
        tensor_names=set(tensors),
        total_bytes=sum(t.nbytes for t in tensors.values()),
        num_shards=k3t.CKPT_SHARDS,
        tensors=tensors if with_tensors else None,
    )


def test_fixture_reproduces_the_released_k31_index(cfg):
    checkpoint = _k31_checkpoint(cfg)
    assert len(checkpoint.tensor_names) == K31_CKPT_TENSORS
    assert checkpoint.total_bytes == K31_CKPT_TOTAL_BYTES


def test_the_delta_against_k3_is_exactly_the_indexer(cfg, k3_cfg):
    k3 = k3t._checkpoint(k3_cfg, with_tensors=True)
    k31 = _k31_checkpoint(cfg)
    added = k31.tensor_names - k3.tensor_names
    removed = k3.tensor_names - k31.tensor_names
    assert not removed
    assert len(added) == K31_INDEXER_TENSORS
    assert all(".self_attn.indexer." in name for name in added)
    assert (k31.total_bytes - k3.total_bytes) == K31_INDEXER_BYTES
    assert K31_INDEXER_BYTES == 24 * K31_INDEXER_BYTES_PER_LAYER


def test_indexer_sits_on_the_zero_indexed_full_attention_layers(cfg):
    """The config lists the layers 1-based; the checkpoint names them 0-based.
    Reading one as the other puts every indexer on a KDA layer."""
    import re

    k31 = _k31_checkpoint(cfg)
    layers = sorted({int(re.search(r"layers\.(\d+)\.", n).group(1))
                     for n in k31.tensor_names if ".self_attn.indexer." in n})
    assert layers == [i - 1 for i in _DSA_1IDX]
    assert layers == list(range(3, 92, 4)) + [92]
    q_a_layers = sorted({int(re.search(r"layers\.(\d+)\.", n).group(1))
                         for n in k31.tensor_names
                         if n.endswith(".self_attn.q_a_proj.weight")})
    assert layers == q_a_layers


# --------------------------------------------------------------------------- #
#  Config validation                                                           #
# --------------------------------------------------------------------------- #

def test_valid_k31_config_passes(cfg):
    tm.validate_k3_config(cfg)


def test_k31_without_dsa_config_is_rejected(cfg):
    cfg.dsa_config = None
    with pytest.raises(ValueError, match="dsa_config is absent"):
        tm.validate_k3_config(cfg)


def test_k3_with_a_dsa_config_is_rejected(k3_cfg):
    k3_cfg.dsa_config = dict(_K31_DSA_CONFIG)
    with pytest.raises(ValueError, match="dsa_config is set but model_type"):
        tm.validate_k3_config(k3_cfg)


def test_unknown_dsa_key_is_rejected(cfg):
    cfg.dsa_config = dict(cfg.dsa_config, index_rope_theta=10000.0)
    with pytest.raises(ValueError, match="unknown key"):
        tm.validate_k3_config(cfg)


@pytest.mark.parametrize("key", sorted(_K31_DSA_CONFIG))
def test_every_dsa_key_is_required(cfg, key):
    block = dict(cfg.dsa_config)
    block.pop(key)
    cfg.dsa_config = block
    with pytest.raises(ValueError, match="missing required key"):
        tm.validate_k3_config(cfg)


def test_zero_indexed_dsa_layers_are_rejected(cfg):
    cfg.dsa_config = dict(cfg.dsa_config,
                          dsa_layers=[i - 1 for i in _DSA_1IDX])
    with pytest.raises(ValueError, match="full_attn_layers"):
        tm.validate_k3_config(cfg)


def test_rotary_index_path_is_rejected(cfg):
    cfg.dsa_config = dict(cfg.dsa_config, index_rope_dim=64)
    with pytest.raises(ValueError, match="index_rope_dim"):
        tm.validate_k3_config(cfg)


def test_topk_reuse_schedule_is_rejected(cfg):
    cfg.dsa_config = dict(cfg.dsa_config, index_topk_freq=4,
                          index_skip_topk_offset=3)
    with pytest.raises(ValueError, match="index_topk_freq"):
        tm.validate_k3_config(cfg)


def test_the_k31_quant_ignore_list_is_pinned_separately(cfg, k3_cfg):
    """K3's pin must still reject K3.1's list and vice versa — the two
    checkpoints state the BF16 partition differently and the gate is exact."""
    layout = k3t.mxfp4_layout
    assert layout.K31_EXPECTED_QUANT_IGNORE - layout.K3_EXPECTED_QUANT_IGNORE == set(
        _K31_ADDED_QUANT_IGNORE)
    layout.validate_quantization_config(cfg.quantization_config, "kimi_k31")
    layout.validate_quantization_config(k3_cfg.quantization_config, "kimi_k3")
    with pytest.raises(layout.K3QuantContractError, match="ignore"):
        layout.validate_quantization_config(cfg.quantization_config, "kimi_k3")
    with pytest.raises(layout.K3QuantContractError, match="ignore"):
        layout.validate_quantization_config(k3_cfg.quantization_config, "kimi_k31")


# --------------------------------------------------------------------------- #
#  Name map                                                                    #
# --------------------------------------------------------------------------- #

def test_name_map_counts(cfg):
    name_map, task = tm.build_k3_state_dict_name_map(cfg)
    assert len(name_map) == K31_N_MODULE_TENSORS
    # The indexer joins the EXISTING attn ring; it is not a fifth module type.
    assert len(task["attn"]) == 24
    assert len(task["kda_attn"]) == 69
    assert sorted(task) == sorted(tm.K3_MODULE_TYPES)


def test_indexer_names_map_into_their_own_layers_attn_module(cfg):
    name_map, _ = tm.build_k3_state_dict_name_map(cfg)
    indexer = {n: v for n, v in name_map.items() if ".self_attn.indexer." in n}
    assert len(indexer) == K31_INDEXER_TENSORS
    for name, entry in indexer.items():
        layer = int(name.split("model.layers.")[1].split(".")[0])
        assert entry["module_key"] == "attn_{}".format(layer)
        # tensor_key == checkpoint suffix == model.py parameter name
        assert name.endswith(".self_attn." + entry["tensor_key"])
        assert entry["tensor_key"] in tm.K3_MLA_INDEXER_TENSOR_NAMES


def test_no_indexer_name_lands_on_a_kda_layer(cfg):
    name_map, _ = tm.build_k3_state_dict_name_map(cfg)
    for name, entry in name_map.items():
        if ".self_attn.indexer." in name:
            assert entry["module_key"].startswith("attn_")


def test_k3_name_map_is_unchanged(k3_cfg):
    """The dense map must be bit-for-bit what it was before K3.1 existed."""
    name_map, task = tm.build_k3_state_dict_name_map(k3_cfg)
    assert len(name_map) == k3t.N_MODULE_TENSORS
    assert not any(".indexer." in n for n in name_map)
    shapes, _, _ = tm.k3_module_shapes(k3_cfg)
    assert sorted(shapes["attn"]) == sorted(tm.K3_MLA_TENSOR_NAMES)


# --------------------------------------------------------------------------- #
#  Module shapes                                                               #
# --------------------------------------------------------------------------- #

def _attn_bytes(cfg) -> int:
    shapes, weight_dtypes, tensor_dtypes = tm.k3_module_shapes(cfg)
    return k3t._module_bytes(shapes, weight_dtypes, tensor_dtypes, "attn")


def test_attn_slot_grows_by_exactly_the_indexer(cfg, k3_cfg):
    assert _attn_bytes(k3_cfg) == K3_ATTN_SLOT_BYTES
    assert _attn_bytes(cfg) == K31_ATTN_SLOT_BYTES
    assert _attn_bytes(cfg) - _attn_bytes(k3_cfg) == K31_INDEXER_BYTES_PER_LAYER


def test_indexer_shapes(cfg):
    shapes, weight_dtypes, tensor_dtypes = tm.k3_module_shapes(cfg)
    attn = shapes["attn"]
    assert attn["indexer.wq_b.weight"] == [4096, 1536]
    assert attn["indexer.wk.weight"] == [128, 7168]
    assert attn["indexer.weights_proj.weight"] == [32, 7168]
    assert attn["indexer.k_norm.weight"] == [128]
    assert attn["indexer.k_norm.bias"] == [128]
    # BF16, never quantized: the MXFP4 ignore list covers re:.*self_attn.*,
    # so no per-tensor dtype override may appear for the indexer.
    assert weight_dtypes["attn"] is torch.bfloat16
    assert "attn" not in tensor_dtypes


def test_index_k_is_one_mqa_key_per_token_not_one_per_head(cfg):
    """``wk`` projects to ``index_head_dim``, not ``n_heads*index_head_dim``.
    The per-head form would be a 32x larger index cache per token."""
    shapes, _, _ = tm.k3_module_shapes(cfg)
    heads = cfg.dsa_config["index_num_heads"]
    head_dim = cfg.dsa_config["index_head_dim"]
    assert shapes["attn"]["indexer.wk.weight"][0] == head_dim
    assert shapes["attn"]["indexer.wq_b.weight"][0] == heads * head_dim


# --------------------------------------------------------------------------- #
#  Paged-KV geometry (the hand-off to the core profile tables)                 #
# --------------------------------------------------------------------------- #

def test_kv_bytes_per_token(cfg, k3_cfg):
    assert tm.k3_mla_kv_bytes_per_token(k3_cfg) == 24 * 576 * 2 == 27_648
    assert tm.k31_indexer_kv_bytes_per_token(k3_cfg) == 0
    assert tm.k3_paged_kv_bytes_per_token(k3_cfg) == 27_648

    # Page-split FP8 index keys: 128 e4m3 bytes + one fp32 scale per layer.
    assert tm.k31_index_k_bytes_per_token_per_layer(cfg) == 128 + 4 == 132
    assert tm.k31_index_k_bytes_per_token_per_layer(k3_cfg) == 0
    assert tm.k31_indexer_kv_bytes_per_token(cfg) == 24 * 132 == 3_168
    assert tm.k3_paged_kv_bytes_per_token(cfg) == 27_648 + 3_168 == 30_816


def test_indexer_dims(cfg, k3_cfg):
    assert tm.k3_indexer_dims(k3_cfg) is None
    dims = tm.k3_indexer_dims(cfg)
    assert dims == {"heads": 32, "head_dim": 128, "index_dim": 4096, "topk": 2048}


# --------------------------------------------------------------------------- #
#  Reconciliation against the released checkpoint                              #
# --------------------------------------------------------------------------- #

def test_declarations_partition_the_k31_checkpoint(cfg):
    report = wr.reconcile(_k31_checkpoint(cfg), tm.k3_reconcile_spec(cfg))
    assert report.ok, report.render()
    assert report.n_checkpoint == K31_CKPT_TENSORS
    assert report.n_mapped == K31_N_MODULE_TENSORS
    assert report.n_skeleton == k3t.N_SKELETON_TENSORS
    assert report.n_ignored == K31_N_IGNORED_TENSORS
    assert report.declared_bytes == K31_CKPT_TOTAL_BYTES
    assert report.declared_ignored_bytes == K31_IGNORED_BYTES


def test_k3_declarations_still_partition_the_k3_checkpoint(k3_cfg):
    """Regression guard: the K3 gate's own numbers, recomputed here."""
    report = wr.reconcile(k3t._checkpoint(k3_cfg), tm.k3_reconcile_spec(k3_cfg))
    assert report.ok, report.render()
    assert report.n_checkpoint == k3t.CKPT_TENSORS
    assert report.n_mapped == k3t.N_MODULE_TENSORS
    assert report.declared_bytes == k3t.CKPT_TOTAL_BYTES


def test_a_k3_map_against_a_k31_checkpoint_leaves_the_indexer_unmapped(cfg, k3_cfg):
    """What a missing ``dsa_config`` costs: 120 tensors with no destination.

    This is the failure the config gate prevents, shown rather than asserted
    about — a silently dense K3.1 would load 120 indexer tensors into the
    skeleton or nowhere at all.
    """
    report = wr.reconcile(_k31_checkpoint(cfg), tm.k3_reconcile_spec(k3_cfg))
    assert not report.ok
    assert report.declared_bytes == k3t.CKPT_TOTAL_BYTES
