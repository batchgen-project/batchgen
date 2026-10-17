"""Kimi-K3 compact distributed host-weight store builder.

Every layout assertion here restates the consumer's own rule independently of
the builder:

* the per-module offset is recomputed with the daemon's ``RemoteOffset``
  formula (``distributed_weight_daemon.cpp:1156-1164``);
* published stores are read back ONLY through the consumer boundary:
  ``load_distributed_weight_config`` (the loader every launch path calls,
  including its completion-manifest gate) and ``read_distributed_metadata``
  (the ``Weights_Storage.cpp`` row contract), never through the builder's
  plan;
* a node's ``store.bin`` is byte-compared against the synthetic source bytes;
* the mini checkpoint's expected names come from the real K3 tensor map built
  from a real ``config.json`` through ``load_k3_config``.
"""

import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batchgen.models.moonshotai.kimi_linear import store_builder as sb  # noqa: E402
from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (  # noqa: E402
    DEFAULT_RAIL_DEVICES,
    load_distributed_weight_config,
    read_distributed_metadata,
)
from batchgen.models.moonshotai.kimi_linear.k3 import tensor_map as tm  # noqa: E402
from batchgen.models.moonshotai.kimi_linear.k3.tensor_map import (  # noqa: E402
    K3_CKPT_PREFIX,
    K3_KDA_TENSOR_NAMES,
    K3_MLA_TENSOR_NAMES,
    K3_SHARED_EXPERT_TENSOR_NAMES,
)

# --------------------------------------------------------------------------- #
#  Synthetic mini checkpoint: 2 nodes, 4 layers (KDA 0-2, MLA 3), 8 experts   #
# --------------------------------------------------------------------------- #

NUM_NODES = 2
N_LAYERS = 4
MOE_LAYERS = (1, 2, 3)  # layer 0 is the dense KDA layer, as in the real K3
EXPERTS = 8
TOPOLOGY = (len(MOE_LAYERS), EXPERTS)
EXPERTS_PER_OWNER = EXPERTS // NUM_NODES
SHARDS = 3
NODE_IPS = "10.0.0.1,10.0.0.2"

#: Same six names as the real routed expert, scaled down. Sizes sum to
#: MODULE_BYTES with no padding.
EXPERT_TENSORS = {
    "w1.weight_packed": ("int32", [2, 4]),
    "w1.weight_scale": ("uint8", [2, 2]),
    "w3.weight_packed": ("int32", [2, 4]),
    "w3.weight_scale": ("uint8", [2, 2]),
    "w2.weight_packed": ("int32", [4, 2]),
    "w2.weight_scale": ("uint8", [4, 1]),
}
MODULE_BYTES = 3 * (32 + 4)

IGNORED = (
    ("vision_tower.blocks.0.proj.weight", "bfloat16", [2, 2]),
    ("mm_projector.weight", "bfloat16", [2, 2]),
)

_QUANT_CONFIG = {
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


def mini_config_json():
    """A real K3 ``config.json`` (nested ``text_config``), scaled down."""
    return {
        "model_type": "kimi_k3",
        "text_config": {
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": N_LAYERS,
            "num_attention_heads": 2,
            "kv_lora_rank": 4,
            "q_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "mla_use_output_gate": True,
            "num_experts": EXPERTS,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "moe_intermediate_size": 4,
            "routed_expert_hidden_size": 4,
            "latent_moe_use_norm": True,
            "attn_res_block_size": 2,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "num_nextn_predict_layers": 0,
            "quantization_config": _QUANT_CONFIG,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "full_attn_layers": [4],
                "num_heads": 2,
                "head_dim": 2,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
            },
        },
    }


def write_model_dir(root, config=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config or mini_config_json()))
    return root


def _byte_size(dtype, shape):
    size = sb.DTYPE_ITEMSIZE[dtype]
    for dim in shape:
        size *= dim
    return size


def _payload(name, size):
    """Deterministic, name-dependent bytes, so a byte-compare is meaningful."""
    blob = b""
    counter = 0
    while len(blob) < size:
        blob += hashlib.sha256(f"{name}:{counter}".encode()).digest()
        counter += 1
    return blob[:size]


def _declarations(cfg):
    """(ckpt_name, dtype, shape) for the whole mini checkpoint.

    Exactly the real tensor map + skeleton declaration of ``cfg``, plus the
    two explicitly ignored vision roots.
    """
    name_map, _ = tm.build_k3_state_dict_name_map(cfg)
    out = []
    for name, entry in name_map.items():
        if entry["module_key"].startswith("routed_expert_"):
            dtype, shape = EXPERT_TENSORS[entry["tensor_key"]]
        else:
            dtype, shape = "bfloat16", [2, 4]
        out.append((name, dtype, shape))
    for name in tm.k3_skeleton_declaration(cfg):
        out.append((name, "float32", [4]))
    out.extend(IGNORED)
    return out


def write_mini_checkpoint(root, declarations):
    """Write the shard ``*.json`` + ``*.bin`` pairs; return name -> bytes."""
    root.mkdir(parents=True, exist_ok=True)
    # Interleaved, so a single expert module's six tensors come from different
    # shards. The builder must still place them contiguously.
    chunks = [declarations[index::SHARDS] for index in range(SHARDS)]
    payloads = {}
    for index, chunk in enumerate(chunks):
        file_name = f"shard-{index:05d}.bin"
        blob = bytearray()
        state_dict = {}
        for name, dtype, shape in chunk:
            size = _byte_size(dtype, shape)
            data = _payload(name, size)
            state_dict[name] = {
                "dtype": dtype,
                "shape": shape,
                "offset": len(blob),
                "byte_size": size,
            }
            payloads[name] = data
            blob += data
        (root / file_name).write_bytes(bytes(blob))
        (root / f"shard-{index:05d}.json").write_text(
            json.dumps(
                {
                    "file_name": file_name,
                    "total_byte_size": len(blob),
                    "state_dict": state_dict,
                }
            )
        )
    return payloads


def _sidecar(root, index=0):
    path = root / f"shard-{index:05d}.json"
    return path, json.loads(path.read_text())


class Env:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.model_dir = write_model_dir(tmp_path / "hf")
        self.cfg = tm.load_k3_config(str(self.model_dir))
        self.ckpt = tmp_path / "converted_ckpt"
        self.payloads = write_mini_checkpoint(self.ckpt, _declarations(self.cfg))

    def argv(self, output, node_rank, extra=(), ckpt=None):
        return [
            "--converted-ckpt",
            str(ckpt or self.ckpt),
            "--cache-dir",
            str(self.model_dir),
            "--output-dir",
            str(output),
            "--num-nodes",
            str(NUM_NODES),
            "--node-rank",
            str(node_rank),
            "--node-ips",
            NODE_IPS,
            *extra,
        ]

    def build(self, output, node_rank, extra=(), ckpt=None):
        return sb.build(
            self.argv(output, node_rank, extra, ckpt), expected_topology=TOPOLOGY
        )

    def plan(self, node_rank=0, ckpt=None):
        return sb.build_store_plan(
            ckpt or self.ckpt,
            NUM_NODES,
            node_rank,
            self.cfg,
            expected_topology=TOPOLOGY,
        )

    def rewrite(self, declarations, name="ckpt2"):
        root = self.tmp / name
        return root, write_mini_checkpoint(root, declarations)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# --------------------------------------------------------------------------- #
#  Layout                                                                      #
# --------------------------------------------------------------------------- #


def test_module_offsets_match_the_daemon_formula(env):
    plan = env.plan(0)

    assert plan.module_bytes == MODULE_BYTES
    assert plan.experts_per_owner == EXPERTS_PER_OWNER

    modules = {}
    for record in plan.expert_records:
        modules.setdefault(record.module_key, []).append(record)
    assert len(modules) == len(MOE_LAYERS) * EXPERTS

    for layer in MOE_LAYERS:
        for expert in range(EXPERTS):
            records = modules[f"routed_expert_{layer}_{expert}"]
            # distributed_weight_daemon.cpp:1156-1164, restated here.
            expected_base = plan.replicated_bytes + (
                (layer - 1) * EXPERTS_PER_OWNER + expert % EXPERTS_PER_OWNER
            ) * plan.module_bytes
            offsets = [record.compact_offset for record in records]
            ends = [record.compact_offset + record.byte_size for record in records]
            assert min(offsets) == expected_base
            # The engine's remote-module contract: relative offsets start at 0
            # and the maximum relative end is exactly module_bytes.
            assert max(ends) - expected_base == plan.module_bytes
            assert {record.owner for record in records} == {
                expert // EXPERTS_PER_OWNER
            }
            ordered = sorted(records, key=lambda r: r.tensor_key)
            cursor = expected_base
            for record in ordered:
                assert record.compact_offset == cursor
                cursor += record.byte_size
            assert cursor == expected_base + plan.module_bytes


def test_replicated_prefix_is_packed_in_source_read_order(env):
    plan = env.plan(0)

    keys = [(r.shard_file, r.source_offset) for r in plan.replicated_records]
    assert keys == sorted(keys)
    cursor = 0
    for record in plan.replicated_records:
        assert record.owner == sb.REPLICATED_OWNER
        assert record.compact_offset == cursor
        cursor += record.byte_size
    assert cursor == plan.replicated_bytes
    assert plan.replicated_bytes > 0


def test_store_bytes_is_an_exact_identity(env):
    plan = env.plan(1)

    assert plan.store_bytes == (
        plan.replicated_bytes + len(MOE_LAYERS) * EXPERTS_PER_OWNER * plan.module_bytes
    )
    checkpoint_bytes = sum(
        _byte_size(dtype, shape) for _, dtype, shape in _declarations(env.cfg)
    )
    assert (
        plan.replicated_bytes + len(MOE_LAYERS) * EXPERTS * plan.module_bytes
        == checkpoint_bytes
    )


# --------------------------------------------------------------------------- #
#  Naming: the config's tensor map decides every key                           #
# --------------------------------------------------------------------------- #


def test_module_and_tensor_keys_are_the_mapped_names(env):
    plan = env.plan(0)
    by_key = {}
    for record in plan.records:
        by_key.setdefault(record.module_key, set()).add(record.tensor_key)

    # The ring comes from the config's is_kda_layer: 0-2 KDA, 3 MLA.
    for layer in (0, 1, 2):
        assert by_key[f"kda_attn_{layer}"] == set(K3_KDA_TENSOR_NAMES)
    assert by_key["attn_3"] == set(K3_MLA_TENSOR_NAMES)
    for layer in MOE_LAYERS:
        assert by_key[f"shared_expert_{layer}"] == set(K3_SHARED_EXPERT_TENSOR_NAMES)
        assert by_key[f"routed_expert_{layer}_0"] == set(EXPERT_TENSORS)
    assert "shared_expert_0" not in by_key

    # Unmapped tensors keep their RAW checkpoint name, under the one module
    # key batchgen_worker.py:990-998 asks for.
    skeleton = by_key[sb.SKELETON_MODULE_KEY]
    assert K3_CKPT_PREFIX + "model.embed_tokens.weight" in skeleton
    assert K3_CKPT_PREFIX + "model.layers.0.mlp.gate_proj.weight" in skeleton
    assert "vision_tower.blocks.0.proj.weight" in skeleton
    assert "mm_projector.weight" in skeleton
    assert plan.ignored_tensors == len(IGNORED)
    assert {
        record.owner
        for record in plan.records
        if record.module_key == sb.SKELETON_MODULE_KEY
    } == {sb.REPLICATED_OWNER}


# --------------------------------------------------------------------------- #
#  Completeness: the expected map, held in both directions                     #
# --------------------------------------------------------------------------- #


def _without(env, predicate):
    return [decl for decl in _declarations(env.cfg) if not predicate(decl[0])]


def test_a_missing_shared_expert_module_is_refused(env):
    root, _ = env.rewrite(
        _without(env, lambda name: ".layers.2.block_sparse_moe.shared_experts." in name)
    )
    with pytest.raises(sb.K3StoreBuilderError) as excinfo:
        env.plan(0, ckpt=root)
    message = str(excinfo.value)
    assert "shared_expert_2 (3 missing)" in message
    assert "3 expected tensors are absent" in message


def test_a_missing_attention_layer_is_refused(env):
    root, _ = env.rewrite(_without(env, lambda name: ".layers.1.self_attn." in name))
    with pytest.raises(
        sb.K3StoreBuilderError,
        match=rf"kda_attn_1 \({len(K3_KDA_TENSOR_NAMES)} missing\)",
    ):
        env.plan(0, ckpt=root)


def test_one_missing_skeleton_tensor_is_refused(env):
    dropped = K3_CKPT_PREFIX + "model.layers.3.mlp_res_norm.weight"
    root, _ = env.rewrite(_without(env, lambda name: name == dropped))
    with pytest.raises(sb.K3StoreBuilderError) as excinfo:
        env.plan(0, ckpt=root)
    assert "__skeleton__ (1 missing)" in str(excinfo.value)
    assert dropped in str(excinfo.value)


def test_a_missing_routed_expert_is_refused(env):
    root, _ = env.rewrite(
        _without(env, lambda name: ".block_sparse_moe.experts.5." in name)
    )
    with pytest.raises(sb.K3StoreBuilderError, match="routed_expert_1_5 \\(6 missing\\)"):
        env.plan(0, ckpt=root)


def test_an_unexpected_tensor_is_refused(env):
    stray = K3_CKPT_PREFIX + "model.layers.1.self_attn.q_a_proj.weight"
    root, _ = env.rewrite(_declarations(env.cfg) + [(stray, "bfloat16", [2, 4])])
    with pytest.raises(sb.K3StoreBuilderError) as excinfo:
        env.plan(0, ckpt=root)
    assert "1 tensors are in neither the Kimi-K3 tensor map" in str(excinfo.value)
    assert stray in str(excinfo.value)


def test_a_layer_shipping_the_other_ring_is_refused_both_ways(env):
    """Layer 3 is MLA in the config; shipping KDA names there is not re-ringed."""
    stem = f"{K3_CKPT_PREFIX}model.layers.3.self_attn."
    declarations = _without(env, lambda name: name.startswith(stem))
    declarations += [(stem + name, "bfloat16", [2, 4]) for name in K3_KDA_TENSOR_NAMES]
    root, _ = env.rewrite(declarations)
    with pytest.raises(sb.K3StoreBuilderError) as excinfo:
        env.plan(0, ckpt=root)
    message = str(excinfo.value)
    assert "attn_3 (" in message
    assert "in neither the Kimi-K3 tensor map" in message


def test_ignored_roots_are_optional(env):
    root, _ = env.rewrite(
        _without(env, lambda name: name.startswith(("vision_tower.", "mm_projector.")))
    )
    assert env.plan(0, ckpt=root).ignored_tensors == 0


def test_a_byte_nonuniform_expert_module_is_refused(env):
    target = f"{K3_CKPT_PREFIX}model.layers.2.block_sparse_moe.experts.5.w2.weight_scale"
    declarations = [
        (name, dtype, [8, 1] if name == target else shape)
        for name, dtype, shape in _declarations(env.cfg)
    ]
    root, _ = env.rewrite(declarations)
    with pytest.raises(sb.K3StoreBuilderError, match="not byte-uniform"):
        env.plan(0, ckpt=root)


# --------------------------------------------------------------------------- #
#  Source integrity                                                           #
# --------------------------------------------------------------------------- #


def _first_tensor(sidecar):
    name = sorted(sidecar["state_dict"], key=lambda n: sidecar["state_dict"][n]["offset"])[0]
    return name, sidecar["state_dict"][name]


def test_a_negative_offset_is_refused(env):
    path, sidecar = _sidecar(env.ckpt)
    name, info = _first_tensor(sidecar)
    info["offset"] = -4
    path.write_text(json.dumps(sidecar))
    with pytest.raises(sb.K3StoreBuilderError, match=rf"tensor {name} has negative offset -4"):
        sb.read_shard_metadata(env.ckpt)


def test_a_span_past_the_end_of_the_bin_is_refused(env):
    path, sidecar = _sidecar(env.ckpt)
    last = max(sidecar["state_dict"], key=lambda n: sidecar["state_dict"][n]["offset"])
    sidecar["state_dict"][last]["offset"] += 1
    path.write_text(json.dumps(sidecar))
    with pytest.raises(sb.K3StoreBuilderError, match=rf"tensor {last} spans .* but .*shard-00000.bin is only"):
        sb.read_shard_metadata(env.ckpt)


def test_overlapping_spans_are_refused(env):
    path, sidecar = _sidecar(env.ckpt)
    ordered = sorted(sidecar["state_dict"], key=lambda n: sidecar["state_dict"][n]["offset"])
    # Point the second tensor at the first one's bytes: both stay in bounds.
    sidecar["state_dict"][ordered[1]]["offset"] = sidecar["state_dict"][ordered[0]]["offset"]
    path.write_text(json.dumps(sidecar))
    with pytest.raises(sb.K3StoreBuilderError, match="overlaps tensor"):
        sb.read_shard_metadata(env.ckpt)


def test_total_byte_size_must_equal_the_tensor_sum(env):
    path, sidecar = _sidecar(env.ckpt)
    sidecar["total_byte_size"] += 8
    path.write_text(json.dumps(sidecar))
    with pytest.raises(sb.K3StoreBuilderError, match="sum to .* but total_byte_size is"):
        sb.read_shard_metadata(env.ckpt)


def test_the_bin_length_must_equal_total_byte_size(env):
    with open(env.ckpt / "shard-00001.bin", "ab") as handle:
        handle.write(b"\x00" * 8)
    with pytest.raises(sb.K3StoreBuilderError, match="shard-00001.bin is .* but shard-00001.json declares"):
        sb.read_shard_metadata(env.ckpt)


def test_a_boolean_or_float_offset_is_refused(env):
    path, sidecar = _sidecar(env.ckpt)
    name, info = _first_tensor(sidecar)
    info["offset"] = True
    path.write_text(json.dumps(sidecar))
    with pytest.raises(sb.K3StoreBuilderError, match="offset must be an integer"):
        sb.read_shard_metadata(env.ckpt)


def test_malformed_json_is_a_named_error_with_the_path(env):
    (env.ckpt / "shard-00002.json").write_text("{not json")
    with pytest.raises(sb.K3StoreBuilderError, match=r"shard-00002.json is not valid JSON"):
        sb.read_shard_metadata(env.ckpt)


def test_unknown_dtype_is_refused_not_defaulted_to_fp8(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir(parents=True)
    (root / "shard-00000.bin").write_bytes(b"\x00" * 16)
    (root / "shard-00000.json").write_text(
        json.dumps(
            {
                "file_name": "shard-00000.bin",
                "total_byte_size": 16,
                "state_dict": {
                    "language_model.model.norm.weight": {
                        "dtype": "float4_e2m1",
                        "shape": [4, 4],
                        "offset": 0,
                        "byte_size": 16,
                    }
                },
            }
        )
    )
    with pytest.raises(sb.K3StoreBuilderError, match="falling back to fp8"):
        sb.read_shard_metadata(root)


def test_shape_and_byte_size_must_agree(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir(parents=True)
    (root / "shard-00000.bin").write_bytes(b"\x00" * 32)
    (root / "shard-00000.json").write_text(
        json.dumps(
            {
                "file_name": "shard-00000.bin",
                "total_byte_size": 32,
                "state_dict": {
                    "language_model.model.norm.weight": {
                        "dtype": "float32",
                        "shape": [4],
                        "offset": 0,
                        "byte_size": 32,
                    }
                },
            }
        )
    )
    with pytest.raises(sb.K3StoreBuilderError, match="declares byte_size"):
        sb.read_shard_metadata(root)


def test_an_empty_or_missing_checkpoint_directory_is_refused(tmp_path):
    with pytest.raises(sb.K3StoreBuilderError, match="does not exist"):
        sb.read_shard_metadata(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(sb.K3StoreBuilderError, match="no converted-checkpoint"):
        sb.read_shard_metadata(empty)


def test_a_sidecar_without_its_bin_is_refused(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "shard-00000.json").write_text(
        json.dumps(
            {
                "file_name": "shard-00000.bin",
                "total_byte_size": 4,
                "state_dict": {
                    "language_model.model.norm.weight": {
                        "dtype": "float32",
                        "shape": [1],
                        "offset": 0,
                        "byte_size": 4,
                    }
                },
            }
        )
    )
    with pytest.raises(sb.K3StoreBuilderError, match="which is not a file"):
        sb.read_shard_metadata(root)


# --------------------------------------------------------------------------- #
#  Published store, read back only through the consumer boundary               #
# --------------------------------------------------------------------------- #


def _local_rows(config, n_experts=EXPERTS):
    rows = read_distributed_metadata(
        config["metadata_path"],
        store_bytes=config["store_bytes"],
        num_nodes=config["num_nodes"],
        n_experts=n_experts,
        module_bytes=config["module_bytes"],
    )
    return rows, [r for r in rows if r.owner in (-1, config["node_rank"])]


def test_published_store_is_consumed_through_the_real_loader(env):
    """Engine boundary: config -> metadata rows -> bytes, no builder plan."""
    output = env.tmp / "node1"
    env.build(output, 1)

    config = load_distributed_weight_config(output / "node1.json")
    rows, local = _local_rows(config)
    # The checkpoint name each (module_key, tensor_key) must hold, from the
    # tensor map itself.
    name_map, _ = tm.build_k3_state_dict_name_map(env.cfg)
    source_of = {
        (entry["module_key"], entry["tensor_key"]): name
        for name, entry in name_map.items()
    }
    blob = Path(config["store_path"]).read_bytes()
    assert len(blob) == config["store_bytes"]
    assert len(rows) == len(_declarations(env.cfg))
    covered = 0
    for row in local:
        name = (
            row.tensor_key
            if row.module_key == sb.SKELETON_MODULE_KEY
            else source_of[(row.module_key, row.tensor_key)]
        )
        span = blob[row.compact_offset : row.compact_offset + row.byte_size]
        assert span == env.payloads[name], name
        covered += row.byte_size
    assert covered == config["store_bytes"]
    owned = {row.module_key for row in local if row.owner == 1}
    assert owned == {
        f"routed_expert_{layer}_{expert}"
        for layer in MOE_LAYERS
        for expert in range(EXPERTS_PER_OWNER, EXPERTS)
    }


def test_metadata_is_node_independent_and_covers_every_node(env):
    first, second = env.tmp / "node0", env.tmp / "node1"
    env.build(first, 0)
    env.build(second, 1)

    assert (first / "metadata.tsv").read_bytes() == (second / "metadata.tsv").read_bytes()
    config = load_distributed_weight_config(first / "node0.json")
    rows, _ = _local_rows(config)
    assert {row.owner for row in rows} == {-1, 0, 1}
    assert [row.index for row in rows] == list(range(len(rows)))


def test_node_config_validates_through_the_real_loader(env):
    output = env.tmp / "node0"
    plan = env.build(output, 0)

    config = load_distributed_weight_config(output / "node0.json")
    assert config["num_nodes"] == NUM_NODES
    assert config["node_rank"] == 0
    assert config["workers"] == 8
    assert config["worker_sharded"] is True
    assert config["transport"] == "host_rdma"
    assert config["store_bytes"] == plan.store_bytes
    assert config["replicated_bytes"] == plan.replicated_bytes
    assert config["module_bytes"] == plan.module_bytes
    assert config["base_port"] == sb.DEFAULT_BASE_PORT
    assert config["store_path"] == str(output / "store.bin")
    assert config["metadata_path"] == str(output / "metadata.tsv")
    assert config["manifest_path"] == str(output / sb.MANIFEST_FILE_NAME)
    assert config["daemon_socket"] == str(output / "daemon.sock")
    assert config["summary_path"] == str(output / "summary.json")
    # H200 HCA numbering, NOT the loader's H20 default.
    assert config["rail_devices"] == list(sb.H200_RAIL_DEVICES)
    assert config["rail_devices"] != list(DEFAULT_RAIL_DEVICES)
    # Nothing but the published set is left: no staging directory, and no
    # summary.json (the daemon's runtime output).
    assert sorted(path.name for path in output.iterdir()) == sorted(
        [sb.LOCK_FILE_NAME, sb.MANIFEST_FILE_NAME, "metadata.tsv", "node0.json", "store.bin"]
    )


def test_node_config_honours_the_optional_overrides(env):
    output = env.tmp / "node1"
    socket = env.tmp / "custom.sock"
    summary = env.tmp / "custom-summary.json"
    rails = ",".join(f"mlx5_{index}:1" for index in range(8))
    env.build(
        output,
        1,
        extra=[
            "--daemon-socket", str(socket),
            "--summary-path", str(summary),
            "--base-port", "27100",
            "--transport", "hierarchical_gdr",
            "--rail-devices", rails,
        ],
    )

    config = load_distributed_weight_config(output / "node1.json")
    assert config["node_rank"] == 1
    assert config["daemon_socket"] == str(socket)
    assert config["summary_path"] == str(summary)
    assert config["base_port"] == 27100
    assert config["transport"] == "hierarchical_gdr"
    assert config["rail_devices"] == rails.split(",")
    # The recorded command reproduces every node argument.
    manifest = json.loads((output / sb.MANIFEST_FILE_NAME).read_text())
    for token in ("--transport hierarchical_gdr", "--base-port 27100", f"--rail-devices {rails}"):
        assert token in manifest["build_command"]


def test_the_two_node_stores_differ_only_in_the_expert_region(env):
    first = env.build(env.tmp / "n0", 0)
    second = env.build(env.tmp / "n1", 1)
    blob0 = (env.tmp / "n0" / "store.bin").read_bytes()
    blob1 = (env.tmp / "n1" / "store.bin").read_bytes()

    assert first.store_bytes == second.store_bytes
    prefix = first.replicated_bytes
    assert blob0[:prefix] == blob1[:prefix]
    assert blob0[prefix:] != blob1[prefix:]


def test_a_small_read_buffer_produces_the_same_store(env):
    plain, chunked = env.tmp / "plain", env.tmp / "chunked"
    env.build(plain, 0)
    env.build(chunked, 0, extra=["--read-buffer-bytes", "7", "--hash-workers", "1"])

    assert (plain / "store.bin").read_bytes() == (chunked / "store.bin").read_bytes()
    manifests = [
        json.loads((path / sb.MANIFEST_FILE_NAME).read_text())
        for path in (plain, chunked)
    ]
    assert manifests[0]["store"] == manifests[1]["store"]
    assert manifests[0]["metadata"] == manifests[1]["metadata"]


# --------------------------------------------------------------------------- #
#  Completion manifest                                                         #
# --------------------------------------------------------------------------- #


def test_manifest_records_identity_sizes_and_segment_hashes(env, monkeypatch):
    # Small segments, so the store is hashed in several pieces.
    monkeypatch.setattr(sb, "SEGMENT_TARGET_BYTES", 64)
    output = env.tmp / "node0"
    plan = env.build(output, 0)
    manifest = json.loads((output / sb.MANIFEST_FILE_NAME).read_text())

    assert manifest["format"] == sb.MANIFEST_FORMAT
    assert manifest["format_version"] == sb.MANIFEST_FORMAT_VERSION
    assert manifest["builder"] == {"module": sb.BUILDER_MODULE, "version": sb.STORE_BUILDER_VERSION}
    assert manifest["complete"] is True
    assert manifest["hash_algorithm"] == "blake2b-128"
    assert (manifest["node_rank"], manifest["num_nodes"]) == (0, NUM_NODES)
    assert manifest["store"]["bytes"] == plan.store_bytes
    assert manifest["metadata"]["digest"] == hashlib.blake2b(
        (output / "metadata.tsv").read_bytes(), digest_size=16
    ).hexdigest()
    identity = sb.checkpoint_identity(env.ckpt)
    assert manifest["source"]["fingerprint"] == identity["fingerprint"]
    assert manifest["source"]["converted_ckpt"] == str(env.ckpt)
    assert manifest["source"]["model_config"]["digest"] == hashlib.blake2b(
        (env.model_dir / "config.json").read_bytes(), digest_size=16
    ).hexdigest()
    assert sum(s["bytes_copied"] for s in manifest["source"]["shards"]) == plan.store_bytes
    assert manifest["build_command"].startswith(f"python -m {sb.BUILDER_MODULE} ")

    segments = manifest["store"]["segments"]
    assert len(segments) > 3
    cursor = 0
    for segment in segments:
        assert segment["offset"] == cursor
        cursor += segment["bytes"]
    assert cursor == plan.store_bytes

    # Independently recompute one segment's digest from the store bytes and
    # the metadata rows: rollup of per-tensor BLAKE2b-128 in store order.
    config = load_distributed_weight_config(output / "node0.json")
    _, local = _local_rows(config)
    local.sort(key=lambda row: row.compact_offset)
    blob = (output / "store.bin").read_bytes()
    segment = segments[1]
    rollup = hashlib.blake2b(digest_size=16)
    for row in local:
        if segment["offset"] <= row.compact_offset < segment["offset"] + segment["bytes"]:
            rollup.update(
                hashlib.blake2b(
                    blob[row.compact_offset : row.compact_offset + row.byte_size],
                    digest_size=16,
                ).digest()
            )
    assert rollup.hexdigest() == segment["digest"]


def test_preflight_verification_accepts_a_published_store(env, monkeypatch):
    monkeypatch.setattr(sb, "SEGMENT_TARGET_BYTES", 64)
    output = env.tmp / "node0"
    env.build(output, 0)
    result = sb.verify_published_store(
        output / "node0.json", converted_ckpt=env.ckpt, cache_dir=env.model_dir
    )
    count = len(result["manifest"]["store"]["segments"])
    assert result["spot_checked_segments"] == sorted({0, count // 2, count - 1})


def test_preflight_refuses_a_corrupted_published_store(env, monkeypatch):
    monkeypatch.setattr(sb, "SEGMENT_TARGET_BYTES", 64)
    output = env.tmp / "node0"
    env.build(output, 0)
    with open(output / "store.bin", "r+b") as handle:
        handle.seek(3)
        byte = handle.read(1)
        handle.seek(3)
        handle.write(bytes([byte[0] ^ 0xFF]))
    # Size and metadata are intact, so the load-time gate cannot see a flipped
    # byte; the launch preflight re-hashes segment 0 and must.
    load_distributed_weight_config(output / "node0.json")
    with pytest.raises(sb.K3StoreManifestError) as excinfo:
        sb.verify_published_store(output / "node0.json", converted_ckpt=env.ckpt)
    message = str(excinfo.value)
    assert "segment 0" in message and "does not hash to the manifest" in message
    assert f"Rebuild it with: python -m {sb.BUILDER_MODULE} --converted-ckpt {env.ckpt}" in message


def test_preflight_refuses_a_store_built_from_another_checkpoint(env):
    output = env.tmp / "node0"
    env.build(output, 0)
    other, _ = env.rewrite(_declarations(env.cfg)[::-1], name="other_ckpt")
    with pytest.raises(sb.K3StoreManifestError) as excinfo:
        sb.verify_published_store(output / "node0.json", converted_ckpt=other)
    message = str(excinfo.value)
    assert "the store is stale" in message
    assert f"--converted-ckpt {other}" in message


def test_preflight_refuses_a_different_model_config(env):
    output = env.tmp / "node0"
    env.build(output, 0)
    other = mini_config_json()
    other["text_config"]["rms_norm_eps"] = 1e-6
    other_dir = write_model_dir(env.tmp / "other_hf", other)
    with pytest.raises(sb.K3StoreManifestError, match="is not the config.json the store was built from"):
        sb.verify_published_store(
            output / "node0.json", converted_ckpt=env.ckpt, cache_dir=other_dir
        )


# --------------------------------------------------------------------------- #
#  Atomic publication, crash residue, dry run                                  #
# --------------------------------------------------------------------------- #


class SimulatedCrash(BaseException):
    """Stands in for the process dying: not an Exception, so nothing handles it."""


def test_interrupt_before_publish_leaves_nothing_a_loader_accepts(env, monkeypatch):
    output = env.tmp / "node0"

    def crash(*_args, **_kwargs):
        raise SimulatedCrash()

    monkeypatch.setattr(sb, "_publish", crash)
    with pytest.raises(SimulatedCrash):
        env.build(output, 0)
    assert not (output / "node0.json").exists()
    assert not (output / "store.bin").exists()
    assert not (output / sb.MANIFEST_FILE_NAME).exists()
    with pytest.raises(FileNotFoundError):
        load_distributed_weight_config(output / "node0.json")


def test_interrupt_before_publish_keeps_the_previous_store_whole(env, monkeypatch):
    output = env.tmp / "node0"
    env.build(output, 0)
    before = (output / "store.bin").read_bytes()
    other, _ = env.rewrite(_declarations(env.cfg)[::-1], name="other_ckpt")

    monkeypatch.setattr(sb, "_publish", lambda *a, **k: (_ for _ in ()).throw(SimulatedCrash()))
    with pytest.raises(SimulatedCrash):
        env.build(output, 0, ckpt=other)
    # The old build is untouched and still self-consistent.
    assert (output / "store.bin").read_bytes() == before
    load_distributed_weight_config(output / "node0.json")


def test_a_crash_mid_publish_is_refused_by_the_loader(env, monkeypatch):
    output = env.tmp / "node0"
    env.build(output, 0)
    other, _ = env.rewrite(_declarations(env.cfg)[::-1], name="other_ckpt")

    real_replace = os.replace

    def replace(source, destination):
        if Path(destination).name == sb.MANIFEST_FILE_NAME:
            raise SimulatedCrash()
        return real_replace(source, destination)

    monkeypatch.setattr(sb.os, "replace", replace)
    with pytest.raises(SimulatedCrash):
        env.build(output, 0, ckpt=other)
    monkeypatch.setattr(sb.os, "replace", real_replace)

    # New store + metadata + config are in place, the commit point is not.
    assert (output / "store.bin").exists()
    assert not (output / sb.MANIFEST_FILE_NAME).exists()
    with pytest.raises(sb.K3StoreManifestError) as excinfo:
        load_distributed_weight_config(output / "node0.json")
    message = str(excinfo.value)
    assert "no completion manifest" in message
    assert f"Rebuild it with: python -m {sb.BUILDER_MODULE}" in message
    assert "--node-rank 0" in message and f"--node-ips {NODE_IPS}" in message


def test_staging_residue_of_a_killed_build_is_swept_and_never_adopted(env):
    output = env.tmp / "node0"
    output.mkdir()
    residue = output / f"{sb.STAGING_PREFIX}killed"
    residue.mkdir()
    (residue / "store.bin").write_bytes(b"partial")
    env.build(output, 0)
    assert not residue.exists()
    assert not list(output.glob(sb.STAGING_PREFIX + "*"))


def test_a_failed_build_publishes_nothing_and_cleans_its_staging(env, monkeypatch):
    output = env.tmp / "node0"
    real_write_store = sb.write_store

    def corrupting_write_store(plan, store_path, read_buffer_bytes):
        result = real_write_store(plan, store_path, read_buffer_bytes)
        with open(store_path, "r+b") as handle:
            handle.seek(plan.store_bytes - 1)
            last = handle.read(1)
            handle.seek(plan.store_bytes - 1)
            handle.write(bytes([last[0] ^ 0x01]))
        return result

    monkeypatch.setattr(sb, "write_store", corrupting_write_store)
    with pytest.raises(sb.K3StoreBuilderError, match="do not hash to the bytes copied"):
        env.build(output, 0)
    assert sorted(path.name for path in output.iterdir()) == [sb.LOCK_FILE_NAME]


def test_a_concurrent_build_into_the_same_directory_is_refused(env):
    output = env.tmp / "node0"
    output.mkdir()
    descriptor = os.open(output / sb.LOCK_FILE_NAME, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(sb.K3StoreBuilderError, match="another store build holds"):
            env.build(output, 0)
    finally:
        os.close(descriptor)


def test_a_directory_holding_another_rank_is_refused_before_writing(env):
    output = env.tmp / "shared"
    env.build(output, 0)
    with pytest.raises(sb.K3StoreBuilderError, match="holds node0.json, the config of a different node rank"):
        env.build(output, 1)
    load_distributed_weight_config(output / "node0.json")


def test_dry_run_writes_nothing_and_plans_the_real_rows(env):
    dry = env.tmp / "dry"
    plan = env.build(dry, 0, extra=["--dry-run"])
    assert not dry.exists()

    real = env.tmp / "real"
    env.build(real, 0)
    assert sb.metadata_bytes(plan) == (real / "metadata.tsv").read_bytes()
    assert plan.store_bytes == (real / "store.bin").stat().st_size


def test_dry_run_over_an_unpublished_store_refuses(env):
    output = env.tmp / "node0"
    output.mkdir()
    (output / "store.bin").write_bytes(b"\x00" * 64)
    (output / "node0.json").write_text("{}")
    with pytest.raises(sb.K3StoreBuilderError, match="--dry-run refuses: .* but no store_manifest.json"):
        env.build(output, 0, extra=["--dry-run"])
    # Nothing was adopted or rewritten.
    assert (output / "node0.json").read_text() == "{}"
    assert not (output / sb.MANIFEST_FILE_NAME).exists()


def test_dry_run_over_a_matching_published_store_passes(env):
    output = env.tmp / "node0"
    env.build(output, 0)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    env.build(output, 0, extra=["--dry-run"])
    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert before == after


def test_dry_run_over_a_store_of_another_checkpoint_refuses(env):
    output = env.tmp / "node0"
    env.build(output, 0)
    other, _ = env.rewrite(_declarations(env.cfg)[::-1], name="other_ckpt")
    with pytest.raises(sb.K3StoreBuilderError, match="does not match this plan .*checkpoint fingerprint"):
        env.build(output, 0, extra=["--dry-run"], ckpt=other)


# --------------------------------------------------------------------------- #
#  Load-time gate on stores that changed after publication                     #
# --------------------------------------------------------------------------- #


def test_loader_refuses_a_truncated_or_missing_store(env):
    output = env.tmp / "node0"
    env.build(output, 0)
    with open(output / "store.bin", "r+b") as handle:
        handle.truncate(10)
    with pytest.raises(ValueError, match="byte size mismatch|is 10 B but the manifest"):
        load_distributed_weight_config(output / "node0.json")
    (output / "store.bin").unlink()
    with pytest.raises(sb.K3StoreManifestError, match="is not readable"):
        load_distributed_weight_config(output / "node0.json")


def test_loader_refuses_metadata_of_another_build(env):
    first, second = env.tmp / "node0", env.tmp / "node0b"
    env.build(first, 0)
    other, _ = env.rewrite(_declarations(env.cfg)[::-1], name="other_ckpt")
    env.build(second, 0, ckpt=other)
    (first / "metadata.tsv").write_bytes((second / "metadata.tsv").read_bytes())
    with pytest.raises(sb.K3StoreManifestError, match="does not hash to the manifest's metadata digest"):
        load_distributed_weight_config(first / "node0.json")


def test_loader_refuses_a_config_without_a_manifest(env):
    output = env.tmp / "node0"
    env.build(output, 0)
    config = json.loads((output / "node0.json").read_text())
    config.pop("manifest_path")
    hand_made = env.tmp / "hand.json"
    hand_made.write_text(json.dumps(config))
    with pytest.raises(sb.K3StoreManifestError, match="names no manifest_path"):
        load_distributed_weight_config(hand_made)


# --------------------------------------------------------------------------- #
#  Refusals: every unsupported input hard-fails with a named error             #
# --------------------------------------------------------------------------- #


def test_real_topology_is_enforced_on_the_cli_path(env):
    with pytest.raises(sb.K3StoreBuilderError, match="must be laid out for"):
        # The default expected_topology is the engine's 92 x 896.
        sb.build(env.argv(env.tmp / "out", 0, extra=["--dry-run"]))


def test_node_ip_count_must_match_num_nodes(env):
    argv = env.argv(env.tmp / "out", 0)
    argv[argv.index("--node-ips") + 1] = "10.0.0.1"
    with pytest.raises(sb.K3StoreBuilderError, match="--node-ips lists"):
        sb.build(argv, expected_topology=TOPOLOGY)


def test_bad_node_rank_and_node_count_are_refused(env):
    with pytest.raises(sb.K3StoreBuilderError, match="node_rank must be"):
        sb.build_store_plan(env.ckpt, NUM_NODES, 2, env.cfg, expected_topology=TOPOLOGY)
    with pytest.raises(sb.K3StoreBuilderError, match="num_nodes must be"):
        sb.build_store_plan(env.ckpt, 3, 0, env.cfg, expected_topology=TOPOLOGY)


def test_a_missing_or_invalid_model_config_is_refused(env):
    argv = env.argv(env.tmp / "out", 0)
    argv[argv.index("--cache-dir") + 1] = str(env.tmp / "absent")
    with pytest.raises(sb.K3StoreBuilderError, match="cannot read the Kimi-K3 model config"):
        sb.build(argv, expected_topology=TOPOLOGY)
    broken = mini_config_json()
    broken["text_config"]["linear_attn_config"].pop("kda_layers")
    broken_dir = write_model_dir(env.tmp / "broken", broken)
    argv[argv.index("--cache-dir") + 1] = str(broken_dir)
    with pytest.raises(sb.K3StoreBuilderError, match="is not a loadable Kimi-K3 config"):
        sb.build(argv, expected_topology=TOPOLOGY)


def test_main_reports_a_named_failure_without_raising(env):
    argv = env.argv(env.tmp / "out", 0, extra=["--dry-run"], ckpt=env.tmp / "absent")
    assert sb.main(argv) == 1
