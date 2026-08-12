"""Offline gate for the Kimi-K3 name map / module shapes. CPU only, no weights.

Runs anywhere: the modules under test are loaded by file path (the
``tests/test_engine_config.py`` pattern) so importing the ``batchgen`` package —
and with it a JIT build of the core engine — is never required.

The checkpoint fixture is built from the template tables below: the 60 name
templates of the released checkpoint with the shape and dtype read out of all 96
shard headers.  It is INDEPENDENT of the module under test
(nothing in it is computed from ``k3_module_shapes``), and
``test_fixture_reproduces_the_released_index`` asserts that summing it gives the
index's own ``metadata.total_size`` — so a mutation of the declarations moves
one side only.
"""

import importlib.util
import inspect
import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
K3_DIR = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "k3"


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stub_package(name: str, search_path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(search_path)]
    sys.modules[name] = package


# `batchgen.models.weight_reconciler` is pre-registered so the `from ... import`
# inside tensor_map resolves without importing the `batchgen` package.
wr = _load("batchgen.models.weight_reconciler",
           ROOT / "batchgen" / "models" / "weight_reconciler.py")
_stub_package("_k3_under_test", K3_DIR)
mxfp4_layout = _load("_k3_under_test.mxfp4_layout", K3_DIR / "mxfp4_layout.py")
tm = _load("_k3_under_test.tensor_map", K3_DIR / "tensor_map.py")

BF16 = torch.bfloat16
F32 = torch.float32
U8 = torch.uint8


# --------------------------------------------------------------------------- #
#  A K3 config, standing in for KimiLinearConfig                               #
#                                                                              #
#  Field names and semantics mirror KimiLinearConfig exactly (including the    #
#  1-INDEXED kda_layers rule).  The real class cannot be imported here without  #
#  pulling in the compiled core engine; it IS exercised by                     #
#  test_reconciles_against_the_real_checkpoint, which runs where it exists.    #
# --------------------------------------------------------------------------- #

_K3_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "quantization_status": "compressed",
    "global_compression_ratio": None,
    "kv_cache_scheme": None,
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
    "config_groups": {
        "group_0": {
            "format": "mxfp4-pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 32,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "scale_dtype": "torch.uint8",
                "strategy": "group",
                "symmetric": True,
                "type": "float",
                "zp_dtype": None,
            },
        }
    },
}

# 1-indexed, as the checkpoint declares them: MLA every 4th layer, plus a
# double-MLA tail at 92/93 (0-indexed 91/92).
_FULL_ATTN_1IDX = list(range(4, 93, 4)) + [93]
_KDA_1IDX = [i for i in range(1, 94) if i not in set(_FULL_ATTN_1IDX)]


@dataclass
class _K3Config:
    model_type: str = "kimi_k3"
    vocab_size: int = 163840
    hidden_size: int = 7168
    intermediate_size: int = 33792
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_output_gate: bool = True
    n_routed_experts: int = 896
    n_shared_experts: int = 2
    first_k_dense_replace: int = 1
    moe_intermediate_size: int = 3072
    routed_expert_hidden_size: Optional[int] = 3584
    latent_moe_use_norm: bool = True
    attn_res_block_size: Optional[int] = 12
    hidden_act: str = "situ"
    activation_situ_beta: Optional[float] = 4.0
    activation_situ_linear_beta: Optional[float] = 25.0
    num_nextn_predict_layers: int = 0
    quantization_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: json.loads(json.dumps(_K3_QUANT_CONFIG))
    )
    linear_attn_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "kda_layers": list(_KDA_1IDX),
            "full_attn_layers": list(_FULL_ATTN_1IDX),
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        }
    )

    def is_kda_layer(self, layer_idx: int) -> bool:
        lac = self.linear_attn_config
        return bool(lac) and (layer_idx + 1) in lac.get("kda_layers", [])


@pytest.fixture()
def cfg():
    return _K3Config()


# --------------------------------------------------------------------------- #
#  Pinned facts of the released checkpoint                                     #
# --------------------------------------------------------------------------- #

CKPT_TENSORS = 497_220
CKPT_TOTAL_BYTES = 1_560_860_324_864
CKPT_SHARDS = 96
N_MODULE_TENSORS = 496_026
N_SKELETON_TENSORS = 1_026
N_IGNORED_TENSORS = 168
IGNORED_BYTES = 894_717_952

_PREFIX = "language_model."

# Per-layer templates: (suffix, shape, dtype, kind)
#   kind: "kda" (69 layers) | "mla" (24) | "any" (93) | "moe" (92) | "dense" (1)
_LAYER_TEMPLATES: Tuple[Tuple[str, List[int], torch.dtype, str], ...] = (
    # --- MLA only ---
    ("self_attn.q_a_proj.weight", [1536, 7168], BF16, "mla"),
    ("self_attn.q_a_layernorm.weight", [1536], BF16, "mla"),
    ("self_attn.q_b_proj.weight", [18432, 1536], BF16, "mla"),
    ("self_attn.kv_a_proj_with_mqa.weight", [576, 7168], BF16, "mla"),
    ("self_attn.kv_a_layernorm.weight", [512], BF16, "mla"),
    ("self_attn.kv_b_proj.weight", [24576, 512], BF16, "mla"),
    # --- KDA only ---
    ("self_attn.q_proj.weight", [12288, 7168], BF16, "kda"),
    ("self_attn.k_proj.weight", [12288, 7168], BF16, "kda"),
    ("self_attn.v_proj.weight", [12288, 7168], BF16, "kda"),
    ("self_attn.q_conv1d.weight", [12288, 1, 4], F32, "kda"),
    ("self_attn.k_conv1d.weight", [12288, 1, 4], F32, "kda"),
    ("self_attn.v_conv1d.weight", [12288, 1, 4], F32, "kda"),
    ("self_attn.A_log", [128], F32, "kda"),
    ("self_attn.f_a_proj.weight", [128, 7168], BF16, "kda"),
    ("self_attn.f_b_proj.weight", [12288, 128], BF16, "kda"),
    ("self_attn.dt_bias", [12288], F32, "kda"),
    ("self_attn.b_proj.weight", [96, 7168], BF16, "kda"),
    ("self_attn.o_norm.weight", [128], F32, "kda"),
    # --- both layer kinds ---
    ("self_attn.g_proj.weight", [12288, 7168], BF16, "any"),
    ("self_attn.o_proj.weight", [7168, 12288], BF16, "any"),
    ("input_layernorm.weight", [7168], BF16, "any"),
    ("post_attention_layernorm.weight", [7168], BF16, "any"),
    ("self_attention_res_norm.weight", [7168], BF16, "any"),
    ("self_attention_res_proj.weight", [1, 7168], BF16, "any"),
    ("mlp_res_norm.weight", [7168], BF16, "any"),
    ("mlp_res_proj.weight", [1, 7168], BF16, "any"),
    # --- MoE layers ---
    ("block_sparse_moe.gate.weight", [896, 7168], BF16, "moe"),
    ("block_sparse_moe.gate.e_score_correction_bias", [896], F32, "moe"),
    ("block_sparse_moe.routed_expert_down_proj.weight", [3584, 7168], BF16, "moe"),
    ("block_sparse_moe.routed_expert_up_proj.weight", [7168, 3584], BF16, "moe"),
    ("block_sparse_moe.routed_expert_norm.weight", [3584], BF16, "moe"),
    ("block_sparse_moe.shared_experts.gate_proj.weight", [6144, 7168], BF16, "moe"),
    ("block_sparse_moe.shared_experts.up_proj.weight", [6144, 7168], BF16, "moe"),
    ("block_sparse_moe.shared_experts.down_proj.weight", [7168, 6144], BF16, "moe"),
    # --- dense layer 0 ---
    ("mlp.gate_proj.weight", [33792, 7168], BF16, "dense"),
    ("mlp.up_proj.weight", [33792, 7168], BF16, "dense"),
    ("mlp.down_proj.weight", [7168, 33792], BF16, "dense"),
)

_EXPERT_TEMPLATES: Tuple[Tuple[str, List[int]], ...] = (
    ("w1.weight_packed", [3072, 1792]),
    ("w1.weight_scale", [3072, 112]),
    ("w2.weight_packed", [3584, 1536]),
    ("w2.weight_scale", [3584, 96]),
    ("w3.weight_packed", [3072, 1792]),
    ("w3.weight_scale", [3072, 112]),
)

_GLOBAL_TEMPLATES: Tuple[Tuple[str, List[int], torch.dtype], ...] = (
    ("model.embed_tokens.weight", [163840, 7168], BF16),
    ("lm_head.weight", [163840, 7168], BF16),
    ("model.norm.weight", [7168], BF16),
    ("model.output_attn_res_norm.weight", [7168], BF16),
    ("model.output_attn_res_proj.weight", [1, 7168], BF16),
)

# Vision: out of scope, but its bytes are part of the checkpoint total and of
# the ignored-set pin, so it is in the fixture with its real shapes.
_VISION_BLOCK_TEMPLATES: Tuple[Tuple[str, List[int]], ...] = (
    ("mlp.fc0.weight", [4096, 1024]),
    ("mlp.fc1.weight", [1024, 4096]),
    ("norm0.weight", [1024]),
    ("norm1.weight", [1024]),
    ("wo.weight", [1024, 1536]),
    ("wqkv.weight", [4608, 1024]),
)
_VISION_SINGLETONS: Tuple[Tuple[str, List[int]], ...] = (
    ("vision_tower.encoder.final_layernorm.weight", [1024]),
    ("vision_tower.patch_embed.pos_emb.weight", [64, 64, 1024]),
    ("vision_tower.patch_embed.proj.weight", [1024, 3, 14, 14]),
    ("mm_projector.post_norm.weight", [7168]),
    ("mm_projector.proj.0.weight", [4096, 4096]),
    ("mm_projector.proj.2.weight", [7168, 4096]),
)

_CKPT_CACHE: Dict[Any, Any] = {}


def _checkpoint(cfg, with_tensors: bool = True):
    """The released checkpoint, rebuilt from the template tables.

    ``with_tensors=True`` carries per-tensor shape+dtype, i.e. what
    ``read_safetensors_headers`` returns on the mounted artifact.  ``False``
    gives the weaker index view (names + aggregate total) for the tests that
    exist to show the difference.
    """
    key = (cfg.num_hidden_layers, cfg.n_routed_experts,
           cfg.first_k_dense_replace, with_tensors)
    if key in _CKPT_CACHE:
        return _CKPT_CACHE[key]

    tensors: Dict[str, Any] = {}

    def add(name: str, shape: List[int], dtype: torch.dtype) -> None:
        tensors[name] = wr.CkptTensor(tuple(shape), dtype,
                                      wr.nbytes(shape, dtype))

    for suffix, shape, dtype in _GLOBAL_TEMPLATES:
        add(_PREFIX + suffix, shape, dtype)

    kda_set = set(_KDA_1IDX)
    for layer in range(cfg.num_hidden_layers):
        is_kda = (layer + 1) in kda_set
        is_moe = layer >= cfg.first_k_dense_replace
        base = "{}model.layers.{}.".format(_PREFIX, layer)
        for suffix, shape, dtype, kind in _LAYER_TEMPLATES:
            if kind == "kda" and not is_kda:
                continue
            if kind == "mla" and is_kda:
                continue
            if kind == "moe" and not is_moe:
                continue
            if kind == "dense" and is_moe:
                continue
            add(base + suffix, shape, dtype)
        if is_moe:
            for expert in range(cfg.n_routed_experts):
                expert_base = "{}block_sparse_moe.experts.{}.".format(base, expert)
                for suffix, shape in _EXPERT_TEMPLATES:
                    add(expert_base + suffix, shape, U8)

    for block in range(27):
        for suffix, shape in _VISION_BLOCK_TEMPLATES:
            add("vision_tower.encoder.blocks.{}.{}".format(block, suffix),
                shape, BF16)
    for name, shape in _VISION_SINGLETONS:
        add(name, shape, BF16)

    checkpoint = wr.CheckpointIndex(
        tensor_names=set(tensors),
        total_bytes=sum(t.nbytes for t in tensors.values()),
        num_shards=CKPT_SHARDS,
        tensors=tensors if with_tensors else None,
    )
    _CKPT_CACHE[key] = checkpoint
    return checkpoint


# --------------------------------------------------------------------------- #
#  The fixture itself must reproduce the artifact                              #
# --------------------------------------------------------------------------- #

def test_fixture_reproduces_the_released_index(cfg):
    """A fixture fitted to the code under test proves nothing.

    Both numbers come from the released checkpoint (497,220 tensors,
    metadata.total_size 1,560,860,324,864); nothing here is computed from
    tensor_map, so the byte reconciliation below has no free parameter.
    """
    checkpoint = _checkpoint(cfg)
    assert len(checkpoint.tensor_names) == CKPT_TENSORS
    assert checkpoint.total_bytes == CKPT_TOTAL_BYTES
    ignored = sum(t.nbytes for name, t in checkpoint.tensors.items()
                  if name.startswith(("vision_tower.", "mm_projector.")))
    assert ignored == IGNORED_BYTES


# --------------------------------------------------------------------------- #
#  Config validation                                                           #
# --------------------------------------------------------------------------- #

def test_valid_config_passes(cfg):
    tm.validate_k3_config(cfg)
    assert sum(cfg.is_kda_layer(i) for i in range(93)) == 69
    assert cfg.is_kda_layer(0)
    assert not cfg.is_kda_layer(91) and not cfg.is_kda_layer(92)


@pytest.mark.parametrize("field_name,bad_value", [
    ("model_type", "kimi_linear"),          # 48B defaults reached
    ("q_lora_rank", None),                  # MLA would want a direct q_proj
    ("mla_use_output_gate", False),         # g_proj dropped
    ("routed_expert_hidden_size", None),    # LatentMoE off -> wrong expert K
    ("latent_moe_use_norm", False),
    ("attn_res_block_size", None),
    ("hidden_act", "silu"),
    ("activation_situ_beta", None),
    ("num_nextn_predict_layers", 1),
    ("quantization_config", None),          # top-level read instead of text_config
])
def test_dropped_config_switch_is_rejected(cfg, field_name, bad_value):
    setattr(cfg, field_name, bad_value)
    with pytest.raises(ValueError):
        tm.validate_k3_config(cfg)


def test_zero_indexed_layer_lists_are_rejected(cfg):
    """The D7 trap: reading the layer lists 0-indexed offsets every layer."""
    cfg.linear_attn_config = dict(cfg.linear_attn_config)
    cfg.linear_attn_config["kda_layers"] = [i - 1 for i in _KDA_1IDX]
    cfg.linear_attn_config["full_attn_layers"] = [i - 1 for i in _FULL_ATTN_1IDX]
    with pytest.raises(ValueError):
        tm.validate_k3_config(cfg)


def test_missing_kda_layers_key_is_rejected(cfg):
    cfg.linear_attn_config = dict(cfg.linear_attn_config)
    cfg.linear_attn_config.pop("kda_layers")
    with pytest.raises(ValueError):
        tm.validate_k3_config(cfg)


def test_low_rank_gate_declaration_is_rejected(cfg):
    cfg.linear_attn_config = dict(cfg.linear_attn_config)
    cfg.linear_attn_config["use_full_rank_gate"] = False
    with pytest.raises(ValueError):
        tm.validate_k3_config(cfg)


def test_a_shallower_k3_is_accepted(cfg):
    """K3-24L staging must not be rejected by a hardcoded depth."""
    cfg.num_hidden_layers = 24
    cfg.linear_attn_config = dict(cfg.linear_attn_config)
    cfg.linear_attn_config["full_attn_layers"] = [4, 8, 12, 16, 20, 24]
    cfg.linear_attn_config["kda_layers"] = [
        i for i in range(1, 25) if i % 4 != 0]
    tm.validate_k3_config(cfg)
    _, task = tm.build_k3_state_dict_name_map(cfg)
    assert len(task["attn"]) == 6 and len(task["kda_attn"]) == 18


# --------------------------------------------------------------------------- #
#  Name map                                                                    #
# --------------------------------------------------------------------------- #

def test_name_map_counts(cfg):
    name_map, task = tm.build_k3_state_dict_name_map(cfg)
    assert len(name_map) == N_MODULE_TENSORS
    assert len(task["attn"]) == 24
    assert len(task["kda_attn"]) == 69
    assert len(task["shared_expert"]) == 92
    assert len(task["routed_expert"]) == 92 * 896 == 82_432


def test_every_name_carries_the_language_model_prefix(cfg):
    name_map, _ = tm.build_k3_state_dict_name_map(cfg)
    assert all(n.startswith("language_model.model.layers.") for n in name_map)


def test_one_g_proj_handler_serves_both_layer_kinds(cfg):
    name_map, _ = tm.build_k3_state_dict_name_map(cfg)
    assert name_map["language_model.model.layers.0.self_attn.g_proj.weight"][
        "module_key"] == "kda_attn_0"
    assert name_map["language_model.model.layers.3.self_attn.g_proj.weight"][
        "module_key"] == "attn_3"
    g_proj = [n for n in name_map if n.endswith(".self_attn.g_proj.weight")]
    assert len(g_proj) == 93


def test_routed_experts_are_packed_pairs_never_dot_weight(cfg):
    name_map, _ = tm.build_k3_state_dict_name_map(cfg)
    base = "language_model.model.layers.1.block_sparse_moe.experts.0."
    for projection in ("w1", "w2", "w3"):
        assert base + projection + ".weight_packed" in name_map
        assert base + projection + ".weight_scale" in name_map
        assert base + projection + ".weight" not in name_map


def test_layer_zero_is_dense_and_has_no_moe_subtree(cfg):
    name_map, _ = tm.build_k3_state_dict_name_map(cfg)
    assert not any(".layers.0.block_sparse_moe." in n for n in name_map)
    skeleton = tm.k3_skeleton_declaration(cfg)
    assert "language_model.model.layers.0.mlp.gate_proj.weight" in skeleton
    assert "language_model.model.layers.1.mlp.gate_proj.weight" not in skeleton


def test_skeleton_partition(cfg):
    skeleton = tm.k3_skeleton_declaration(cfg)
    assert len(skeleton) == N_SKELETON_TENSORS
    for name in ("model.embed_tokens.weight", "lm_head.weight",
                 "model.output_attn_res_proj.weight",
                 "model.layers.0.mlp.gate_proj.weight",
                 "model.layers.1.block_sparse_moe.routed_expert_down_proj.weight",
                 "model.layers.92.mlp_res_norm.weight"):
        assert tm.k3_skeleton_key(name) in skeleton


# --------------------------------------------------------------------------- #
#  Module shapes and bytes                                                     #
# --------------------------------------------------------------------------- #

def _module_bytes(module_shapes, weight_dtypes, tensor_dtypes, module_type):
    total = 0
    for name, shape in module_shapes[module_type].items():
        dtype = tensor_dtypes.get(module_type, {}).get(
            name, weight_dtypes[module_type])
        total += wr.nbytes(shape, dtype)
    return total


def test_per_module_bytes_match_the_checkpoint(cfg):
    shapes, weight_dtypes, tensor_dtypes = tm.k3_module_shapes(cfg)
    assert _module_bytes(shapes, weight_dtypes, tensor_dtypes, "attn") == 464_392_192
    assert _module_bytes(shapes, weight_dtypes, tensor_dtypes, "kda_attn") == 887_800_832
    assert _module_bytes(shapes, weight_dtypes, tensor_dtypes, "shared_expert") == 264_241_152
    assert _module_bytes(shapes, weight_dtypes, tensor_dtypes, "routed_expert") == 17_547_264


def test_shapes_that_a_48b_shaped_guess_gets_wrong(cfg):
    shapes, _, tensor_dtypes = tm.k3_module_shapes(cfg)
    # kv_b_proj is heads*(nope+v) x kv_lora, NOT heads*v x kv_lora.
    assert shapes["attn"]["kv_b_proj.weight"] == [24576, 512]
    assert shapes["attn"]["q_b_proj.weight"] == [18432, 1536]
    # A_log is a per-head [96] vector zero-padded to 128 (ACTIVATION_FLOW D2).
    assert shapes["kda_attn"]["A_log"] == [128]
    # K3 conv1d / o_norm are F32; the 48B's are BF16.
    for name in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
                 "o_norm.weight", "A_log", "dt_bias"):
        assert tensor_dtypes["kda_attn"][name] is torch.float32
    # the KDA gate is a single full-rank g_proj
    assert "g_a_proj.weight" not in shapes["kda_attn"]
    assert shapes["kda_attn"]["g_proj.weight"] == [12288, 7168]


def test_routed_expert_slot_is_six_uint8_tensors(cfg):
    shapes, weight_dtypes, tensor_dtypes = tm.k3_module_shapes(cfg)
    assert weight_dtypes["routed_expert"] is torch.uint8
    assert sorted(shapes["routed_expert"]) == sorted([
        "w1.weight_packed", "w1.weight_scale",
        "w2.weight_packed", "w2.weight_scale",
        "w3.weight_packed", "w3.weight_scale",
    ])
    assert shapes["routed_expert"]["w1.weight_packed"] == [3072, 1792]
    assert shapes["routed_expert"]["w1.weight_scale"] == [3072, 112]
    assert shapes["routed_expert"]["w2.weight_packed"] == [3584, 1536]
    assert shapes["routed_expert"]["w2.weight_scale"] == [3584, 96]
    assert all(d is torch.uint8 for d in tensor_dtypes["routed_expert"].values())


# --------------------------------------------------------------------------- #
#  Reconciliation against the released checkpoint                              #
# --------------------------------------------------------------------------- #

def test_declarations_partition_the_checkpoint(cfg):
    report = wr.reconcile(_checkpoint(cfg), tm.k3_reconcile_spec(cfg))
    assert report.ok, report.render()
    assert report.n_checkpoint == CKPT_TENSORS
    assert report.n_mapped == N_MODULE_TENSORS
    assert report.n_skeleton == N_SKELETON_TENSORS
    assert report.n_ignored == N_IGNORED_TENSORS
    assert report.declared_bytes == CKPT_TOTAL_BYTES
    assert report.declared_ignored_bytes == IGNORED_BYTES


def test_reconcile_defaults_to_shard_headers():
    """Index mode compares names and the AGGREGATE total only.

    Making it the default is what would let a byte-neutral shape error through
    the production gate (see the next test), so the default is pinned here.
    """
    default = inspect.signature(
        tm.reconcile_k3_checkpoint).parameters["use_shard_headers"].default
    assert default is True


def test_a_byte_neutral_transpose_is_caught_only_by_header_mode(cfg):
    """Why the gate must read shard headers.

    o_proj declared [12288, 7168] instead of [7168, 12288]: same bytes, so the
    aggregate total still reconciles, the torch::zeros slot is the right size,
    blocking_copy_ succeeds -- and the GEMM gets a transposed weight.
    """
    spec = tm.k3_reconcile_spec(cfg)
    spec.module_shapes["attn"]["o_proj.weight"] = [12288, 7168]

    index_only = wr.reconcile(_checkpoint(cfg, with_tensors=False), spec)
    assert index_only.ok, "index mode is blind to a transpose -- that is the point"

    headers = wr.reconcile(_checkpoint(cfg), spec)
    assert not headers.ok
    assert "byte_total" not in headers.counts
    assert headers.counts["tensor_mismatch"] == 24


def test_the_current_kimi_linear_naming_fails_against_k3(cfg):
    """The blocker, as a test: today's 48B name lists find nothing in K3."""
    spec = tm.k3_reconcile_spec(cfg)
    # replay the 48B assumptions: no prefix, `.weight` routed experts
    broken = {}
    for ckpt_name, entry in spec.name_map.items():
        name = ckpt_name[len("language_model."):]
        key = entry["tensor_key"]
        if key.endswith(".weight_packed"):
            name = name.replace(".weight_packed", ".weight")
            key = key.replace(".weight_packed", ".weight")
        elif key.endswith(".weight_scale"):
            continue
        broken[name] = {"module_key": entry["module_key"], "tensor_key": key}
    report = wr.reconcile(
        _checkpoint(cfg),
        wr.ReconcileSpec(name_map=broken, module_shapes=spec.module_shapes,
                         weight_dtypes=spec.weight_dtypes,
                         tensor_dtypes=spec.tensor_dtypes,
                         skeleton=spec.skeleton,
                         ignore_rules=spec.ignore_rules),
    )
    assert not report.ok
    assert report.counts["dangling"] == len(broken)
    assert report.counts["unaccounted"] == N_MODULE_TENSORS


@pytest.mark.parametrize("module_type,tensor_key,bad_shape,n_bad", [
    ("kda_attn", "A_log", [96], 69),                    # derived from num_heads
    ("attn", "kv_b_proj.weight", [16384, 512], 24),     # 48B-shaped guess
    ("routed_expert", "w2.weight_scale", [3584, 192], 82_432),  # K/16 not K/32
])
def test_a_wrong_declared_shape_is_reported(cfg, module_type, tensor_key,
                                            bad_shape, n_bad):
    spec = tm.k3_reconcile_spec(cfg)
    spec.module_shapes[module_type][tensor_key] = bad_shape
    report = wr.reconcile(_checkpoint(cfg), spec)
    assert "byte_total" in report.counts, report.render()
    assert report.counts["tensor_mismatch"] == n_bad


def test_a_wrong_declared_dtype_is_reported(cfg):
    spec = tm.k3_reconcile_spec(cfg)
    spec.tensor_dtypes["kda_attn"]["q_conv1d.weight"] = torch.bfloat16
    report = wr.reconcile(_checkpoint(cfg), spec)
    assert "byte_total" in report.counts
    assert report.counts["tensor_mismatch"] == 69


def test_dropping_g_proj_leaves_orphan_checkpoint_tensors(cfg):
    """The 48B MLA list has no g_proj; K3 ships it on all 93 layers."""
    spec = tm.k3_reconcile_spec(cfg)
    spec.name_map = {n: e for n, e in spec.name_map.items()
                     if not n.endswith(".self_attn.g_proj.weight")}
    report = wr.reconcile(_checkpoint(cfg), spec)
    assert report.counts["unaccounted"] == 93
    assert report.counts["slot_never_written"] == 2   # attn + kda_attn
    assert "byte_total" in report.counts


def test_vision_tensors_are_ignored_with_a_reason(cfg):
    spec = tm.k3_reconcile_spec(cfg)
    spec.ignore_rules = ()
    report = wr.reconcile(_checkpoint(cfg), spec)
    assert report.counts["unaccounted"] == N_IGNORED_TENSORS
    assert all(rule.reason for rule in tm.K3_IGNORE_RULES)


def test_a_changed_ignored_set_is_reported(cfg):
    """A revised vision tower must not quietly change the shm reservation."""
    spec = tm.k3_reconcile_spec(cfg, ignored_count=170)
    report = wr.reconcile(_checkpoint(cfg), spec)
    assert report.counts["ignored_count"] == 1


def test_a_stage_without_vision_can_be_declared_explicitly(cfg):
    """K3-24L staging: dropping the ignored set must be visible at the call site."""
    cfg.num_hidden_layers = 24
    cfg.linear_attn_config = dict(cfg.linear_attn_config)
    cfg.linear_attn_config["full_attn_layers"] = [4, 8, 12, 16, 20, 24]
    cfg.linear_attn_config["kda_layers"] = [
        i for i in range(1, 25) if i % 4 != 0]
    checkpoint = _checkpoint(cfg)
    names = {n for n in checkpoint.tensor_names
             if not n.startswith(("vision_tower.", "mm_projector."))}
    tensors = {n: checkpoint.tensors[n] for n in names}
    staged = wr.CheckpointIndex(
        tensor_names=names,
        total_bytes=sum(t.nbytes for t in tensors.values()),
        num_shards=8, tensors=tensors,
    )
    assert not wr.reconcile(staged, tm.k3_reconcile_spec(cfg)).ok
    report = wr.reconcile(
        staged, tm.k3_reconcile_spec(cfg, ignored_count=0, ignored_bytes=0))
    assert report.ok, report.render()


def test_shm_reservation_is_computed_from_the_index(cfg, tmp_path):
    index = {
        "metadata": {"total_size": CKPT_TOTAL_BYTES},
        "weight_map": {
            "t{}".format(i): "model-{:05d}-of-000096.safetensors".format(i + 1)
            for i in range(CKPT_SHARDS)
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert tm.k3_shm_byte_size(str(tmp_path)) == CKPT_TOTAL_BYTES + CKPT_SHARDS * 4096
    # 13.2x the 110 GiB constant the 48B path hardcodes
    assert tm.k3_shm_byte_size(str(tmp_path)) > 13 * 110 * 1024 ** 3


def test_shm_reservation_refuses_to_guess(tmp_path):
    with pytest.raises(FileNotFoundError):
        tm.k3_shm_byte_size(str(tmp_path))


# --------------------------------------------------------------------------- #
#  MXFP4 contract                                                              #
# --------------------------------------------------------------------------- #

def test_quantization_config_accepts_the_released_declaration():
    mxfp4_layout.validate_quantization_config(
        json.loads(json.dumps(_K3_QUANT_CONFIG)))


@pytest.mark.parametrize("mutate", [
    lambda q: q.__setitem__("quant_method", "awq"),
    lambda q: q["config_groups"]["group_0"]["weights"].__setitem__("group_size", 16),
    lambda q: q["config_groups"]["group_0"]["weights"].__setitem__("num_bits", 8),
    lambda q: q["config_groups"]["group_0"]["weights"].__setitem__(
        "scale_dtype", "torch.bfloat16"),
    lambda q: q["config_groups"]["group_0"]["weights"].__setitem__("actorder", "group"),
    lambda q: q["config_groups"]["group_0"].__setitem__(
        "input_activations", {"num_bits": 4, "dynamic": True}),
    lambda q: q["config_groups"]["group_0"].__setitem__("format", "int4-pack-quantized"),
    lambda q: q.__setitem__("quantization_status", "frozen"),
    # a module leaving the ignore list has become MXFP4 while module_shapes
    # still declares it BF16
    lambda q: q["ignore"].remove("re:.*self_attn.*"),
    # ...and one joining it has become BF16, possibly where we declare uint8
    lambda q: q["ignore"].append("re:.*block_sparse_moe.experts.*"),
])
def test_unsupported_quantization_declaration_is_rejected(mutate):
    quant = json.loads(json.dumps(_K3_QUANT_CONFIG))
    mutate(quant)
    with pytest.raises(mxfp4_layout.K3QuantContractError):
        mxfp4_layout.validate_quantization_config(quant)


def test_nested_quantization_config_read_from_the_top_level_is_rejected():
    with pytest.raises(mxfp4_layout.K3QuantContractError) as excinfo:
        mxfp4_layout.validate_quantization_config(None)
    assert "text_config" in str(excinfo.value)


def _good_slot():
    shapes = mxfp4_layout.routed_expert_module_shapes(3072, 3584)
    return {name: torch.zeros(shape, dtype=torch.uint8)
            for name, shape in shapes.items()}, shapes


def test_valid_slot_passes():
    weights, shapes = _good_slot()
    mxfp4_layout.validate_routed_expert_slot("routed_expert_1_0", weights, shapes)


def test_slot_missing_a_scale_is_rejected():
    weights, shapes = _good_slot()
    weights.pop("w1.weight_scale")
    with pytest.raises(mxfp4_layout.K3QuantContractError):
        mxfp4_layout.validate_routed_expert_slot("routed_expert_1_0", weights, shapes)


def test_slot_with_a_dequantized_weight_key_is_rejected():
    """The K2.5 anti-pattern: re-keying `.weight_packed` to `.weight`."""
    weights, shapes = _good_slot()
    weights["w1.weight"] = weights.pop("w1.weight_packed")
    with pytest.raises(mxfp4_layout.K3QuantContractError):
        mxfp4_layout.validate_routed_expert_slot("routed_expert_1_0", weights, shapes)


def test_slot_with_a_bf16_scale_is_rejected():
    weights, shapes = _good_slot()
    weights["w1.weight_scale"] = weights["w1.weight_scale"].to(torch.bfloat16)
    with pytest.raises(mxfp4_layout.K3QuantContractError):
        mxfp4_layout.validate_routed_expert_slot("routed_expert_1_0", weights, shapes)


def test_slot_with_mismatched_packed_and_scale_k_is_rejected():
    weights, shapes = _good_slot()
    shapes = dict(shapes)
    shapes["w1.weight_scale"] = [3072, 224]
    weights["w1.weight_scale"] = torch.zeros(3072, 224, dtype=torch.uint8)
    with pytest.raises(mxfp4_layout.K3QuantContractError):
        mxfp4_layout.validate_routed_expert_slot("routed_expert_1_0", weights, shapes)


def test_unpadded_k_is_required():
    with pytest.raises(mxfp4_layout.K3QuantContractError):
        mxfp4_layout.routed_expert_module_shapes(3072, 3580)


# --------------------------------------------------------------------------- #
#  Import seam: everything above loads by file path, so exercise the real one  #
# --------------------------------------------------------------------------- #

def test_the_real_import_path_resolves():
    """tensor_map does `from batchgen.models.weight_reconciler import ...`."""
    try:
        import batchgen.models.weight_reconciler as real
    except Exception as exc:                       # noqa: BLE001
        pytest.skip("batchgen package not importable here: {}".format(exc))
    assert hasattr(real, "reconcile") and hasattr(real, "read_safetensors_headers")


# --------------------------------------------------------------------------- #
#  The real checkpoint (K3_CKPT_DIR); skipped when it is not mounted           #
# --------------------------------------------------------------------------- #

_CKPT = os.environ.get("K3_CKPT_DIR", "/path/to/Kimi-K3")


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_CKPT, "model.safetensors.index.json")),
    reason="K3 checkpoint not mounted",
)
def test_reconciles_against_the_real_checkpoint():
    """Same gate, real artifact, real KimiLinearConfig, real shard headers."""
    cfg = tm.load_k3_config(_CKPT)          # absolute import, no relative escape

    # from_json must stamp model_type even though text_config says otherwise:
    # `merged.update(text_config)` then `merged["model_type"] = "kimi_k3"`
    # (config.py:137-139). Reordering those two lines silently reverts K3 to the
    # 48B identity, which every other guard here is downstream of.
    with open(os.path.join(_CKPT, "config.json")) as handle:
        raw = json.load(handle)
    assert raw["text_config"]["model_type"] == "kimi_linear"
    assert cfg.model_type == "kimi_k3"

    report = tm.reconcile_k3_checkpoint(_CKPT, cfg)   # header mode by default
    assert report.ok, report.render()
    assert report.n_checkpoint == CKPT_TENSORS
    assert report.n_mapped == N_MODULE_TENSORS
    assert report.n_skeleton == N_SKELETON_TENSORS
    assert report.n_ignored == N_IGNORED_TENSORS
    assert report.declared_bytes == CKPT_TOTAL_BYTES
    assert report.declared_ignored_bytes == IGNORED_BYTES
    assert "tensor_mismatch" not in report.counts
    assert tm.k3_shm_byte_size(_CKPT) == CKPT_TOTAL_BYTES + CKPT_SHARDS * 4096
