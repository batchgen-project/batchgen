"""Kimi-K3 compact distributed host-weight store builder.

Every layout assertion here restates the consumer's own rule independently of
the builder:

* the per-module offset is recomputed with the daemon's ``RemoteOffset``
  formula (``distributed_weight_daemon.cpp:1156-1164``);
* a metadata row is re-split exactly as ``Weights_Storage.cpp:377-400`` splits
  it (nine fields, ``T`` tag, ``H`` header skipped);
* the emitted config is validated through the real
  ``load_distributed_weight_config``;
* node 1's ``store.bin`` is byte-compared against the synthetic source bytes.
"""

import hashlib
import json
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
)
from batchgen.models.moonshotai.kimi_linear.k3.tensor_map import (  # noqa: E402
    K3_CKPT_PREFIX,
    K3_KDA_TENSOR_NAMES,
    K3_MLA_TENSOR_NAMES,
    K3_SHARED_EXPERT_TENSOR_NAMES,
)

# --------------------------------------------------------------------------- #
#  Synthetic mini checkpoint: 2 nodes, MoE layers 1..3, 8 experts             #
# --------------------------------------------------------------------------- #

NUM_NODES = 2
MOE_LAYERS = (1, 2, 3)
ALL_LAYERS = (0,) + MOE_LAYERS  # layer 0 is dense MLA, as in the real K3
EXPERTS = 8
TOPOLOGY = (len(MOE_LAYERS), EXPERTS)
EXPERTS_PER_OWNER = EXPERTS // NUM_NODES
SHARDS = 3

#: Same six names and the same packed/scale dtype split as the real routed
#: expert, scaled down. Sizes sum to MODULE_BYTES with no padding.
EXPERT_TENSORS = (
    ("w1.weight_packed", "int32", [2, 4]),
    ("w1.weight_scale", "uint8", [2, 2]),
    ("w3.weight_packed", "int32", [2, 4]),
    ("w3.weight_scale", "uint8", [2, 2]),
    ("w2.weight_packed", "int32", [4, 2]),
    ("w2.weight_scale", "uint8", [4, 1]),
)
MODULE_BYTES = 3 * (32 + 4)


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


def _declarations():
    """(ckpt_name, dtype, shape) for the whole mini checkpoint."""
    out = [
        (K3_CKPT_PREFIX + "model.embed_tokens.weight", "bfloat16", [8, 4]),
        (K3_CKPT_PREFIX + "lm_head.weight", "bfloat16", [8, 4]),
        (K3_CKPT_PREFIX + "model.norm.weight", "float32", [4]),
        # Explicitly ignored roots: unprefixed, and promoted to the skeleton by
        # the replicated path exactly like any other unmapped tensor.
        ("vision_tower.blocks.0.proj.weight", "bfloat16", [2, 2]),
        ("mm_projector.weight", "bfloat16", [2, 2]),
    ]
    for layer in ALL_LAYERS:
        stem = f"{K3_CKPT_PREFIX}model.layers.{layer}"
        out.append((f"{stem}.input_layernorm.weight", "float32", [4]))
        attention = (
            K3_MLA_TENSOR_NAMES if layer == 0 else K3_KDA_TENSOR_NAMES
        )
        for name in attention:
            out.append((f"{stem}.self_attn.{name}", "bfloat16", [2, 4]))
        if layer == 0:
            continue
        for name in K3_SHARED_EXPERT_TENSOR_NAMES:
            out.append(
                (
                    f"{stem}.block_sparse_moe.shared_experts.{name}",
                    "bfloat16",
                    [4, 4],
                )
            )
        for expert in range(EXPERTS):
            for name, dtype, shape in EXPERT_TENSORS:
                out.append(
                    (
                        f"{stem}.block_sparse_moe.experts.{expert}.{name}",
                        dtype,
                        shape,
                    )
                )
    return out


def write_mini_checkpoint(root, declarations=None):
    """Write the shard ``*.json`` + ``*.bin`` pairs; return name -> bytes."""
    root.mkdir(parents=True, exist_ok=True)
    declarations = declarations or _declarations()
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


def run_build(checkpoint, output, node_rank, extra=()):
    argv = [
        "--converted-ckpt",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--num-nodes",
        str(NUM_NODES),
        "--node-rank",
        str(node_rank),
        "--node-ips",
        "10.0.0.1,10.0.0.2",
        *extra,
    ]
    return sb.build(argv, expected_topology=TOPOLOGY)


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "converted_ckpt_marlin"
    return root, write_mini_checkpoint(root)


# --------------------------------------------------------------------------- #
#  Layout                                                                      #
# --------------------------------------------------------------------------- #


def test_module_offsets_match_the_daemon_formula(checkpoint):
    root, _ = checkpoint
    plan = sb.build_store_plan(
        root, NUM_NODES, node_rank=0, expected_topology=TOPOLOGY
    )

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
            ends = [
                record.compact_offset + record.byte_size for record in records
            ]
            assert min(offsets) == expected_base
            # The engine's remote-module contract: relative offsets start at 0
            # and the maximum relative end is exactly module_bytes.
            assert max(ends) - expected_base == plan.module_bytes
            assert {record.owner for record in records} == {
                expert // EXPERTS_PER_OWNER
            }
            # Contiguous in sorted tensor_key order, no padding.
            ordered = sorted(records, key=lambda r: r.tensor_key)
            cursor = expected_base
            for record in ordered:
                assert record.compact_offset == cursor
                cursor += record.byte_size
            assert cursor == expected_base + plan.module_bytes


def test_replicated_prefix_is_packed_in_source_read_order(checkpoint):
    root, _ = checkpoint
    plan = sb.build_store_plan(
        root, NUM_NODES, node_rank=0, expected_topology=TOPOLOGY
    )

    keys = [
        (record.shard_file, record.source_offset)
        for record in plan.replicated_records
    ]
    assert keys == sorted(keys)
    cursor = 0
    for record in plan.replicated_records:
        assert record.owner == sb.REPLICATED_OWNER
        assert record.compact_offset == cursor
        cursor += record.byte_size
    assert cursor == plan.replicated_bytes
    assert plan.replicated_bytes > 0


def test_store_bytes_is_an_exact_identity(checkpoint):
    root, _ = checkpoint
    plan = sb.build_store_plan(
        root, NUM_NODES, node_rank=1, expected_topology=TOPOLOGY
    )

    assert plan.store_bytes == (
        plan.replicated_bytes
        + len(MOE_LAYERS) * EXPERTS_PER_OWNER * plan.module_bytes
    )
    checkpoint_bytes = sum(
        _byte_size(dtype, shape) for _, dtype, shape in _declarations()
    )
    assert (
        plan.replicated_bytes
        + len(MOE_LAYERS) * EXPERTS * plan.module_bytes
        == checkpoint_bytes
    )


# --------------------------------------------------------------------------- #
#  Naming: mapped module/tensor keys, raw names only under __skeleton__        #
# --------------------------------------------------------------------------- #


def test_module_and_tensor_keys_are_the_mapped_names(checkpoint):
    root, _ = checkpoint
    plan = sb.build_store_plan(
        root, NUM_NODES, node_rank=0, expected_topology=TOPOLOGY
    )
    by_key = {}
    for record in plan.records:
        by_key.setdefault(record.module_key, set()).add(record.tensor_key)

    # Layer 0 ships the MLA suffix set, layers 1..3 the KDA one; the ring is
    # chosen by exact set match, never guessed.
    assert by_key["attn_0"] == set(K3_MLA_TENSOR_NAMES)
    for layer in MOE_LAYERS:
        assert by_key[f"kda_attn_{layer}"] == set(K3_KDA_TENSOR_NAMES)
        assert by_key[f"shared_expert_{layer}"] == set(
            K3_SHARED_EXPERT_TENSOR_NAMES
        )
        assert by_key[f"routed_expert_{layer}_0"] == {
            name for name, _, _ in EXPERT_TENSORS
        }

    # Unmapped tensors keep their RAW checkpoint name, under the one module
    # key batchgen_worker.py:990-998 asks for.
    skeleton = by_key[sb.SKELETON_MODULE_KEY]
    assert K3_CKPT_PREFIX + "model.embed_tokens.weight" in skeleton
    assert K3_CKPT_PREFIX + "lm_head.weight" in skeleton
    assert "vision_tower.blocks.0.proj.weight" in skeleton
    assert "mm_projector.weight" in skeleton
    assert all("." in name for name in skeleton)
    # The skeleton is replicated, so get_tensor() can hand it to Torch on
    # every node (Weights_Storage.cpp:797-806 rejects a remote module).
    assert {
        record.owner
        for record in plan.records
        if record.module_key == sb.SKELETON_MODULE_KEY
    } == {sb.REPLICATED_OWNER}


def test_classify_tensor_routes_each_name_shape():
    assert sb.classify_tensor("some.unmapped.tensor") == (
        sb.SKELETON_MODULE_KEY,
        "some.unmapped.tensor",
        None,
        None,
    )
    assert sb.classify_tensor(
        K3_CKPT_PREFIX + "model.layers.7.block_sparse_moe.experts.3.w2.weight_scale"
    ) == ("routed_expert_7_3", "w2.weight_scale", 7, 3)
    assert sb.classify_tensor(
        K3_CKPT_PREFIX + "model.layers.7.block_sparse_moe.shared_experts.up_proj.weight"
    ) == ("shared_expert_7", "up_proj.weight", 7, None)
    assert sb.classify_tensor(
        K3_CKPT_PREFIX + "model.layers.7.self_attn.o_proj.weight"
    ) == ("attn?", "o_proj.weight", 7, None)


# --------------------------------------------------------------------------- #
#  metadata.tsv                                                                #
# --------------------------------------------------------------------------- #


def test_metadata_rows_parse_as_the_engine_parses_them(checkpoint, tmp_path):
    root, _ = checkpoint
    output = tmp_path / "node0"
    plan = run_build(root, output, 0)

    lines = (output / "metadata.tsv").read_text().splitlines()
    assert lines[0] == "H"
    assert len(lines) == len(plan.records) + 1

    for index, line in enumerate(lines[1:]):
        fields = line.split("\t")
        assert len(fields) == 9
        assert fields[0] == "T"
        assert int(fields[1]) == index
        owner = int(fields[4])
        offset = int(fields[5])
        byte_size = int(fields[6])
        assert -1 <= owner < NUM_NODES
        assert fields[7] in sb.DTYPE_ITEMSIZE
        shape = json.loads(fields[8])
        assert isinstance(shape, list)
        assert all(isinstance(dim, int) for dim in shape)
        assert _byte_size(fields[7], shape) == byte_size
        assert offset + byte_size <= plan.store_bytes
        if owner >= 0:
            # Only routed experts may be owned (Weights_Storage.cpp:412-423).
            layer, expert = fields[2].removeprefix("routed_expert_").split("_")
            assert owner == int(expert) // EXPERTS_PER_OWNER
            assert 1 <= int(layer) <= len(MOE_LAYERS)


def test_metadata_covers_every_node_and_is_node_independent(
    checkpoint, tmp_path
):
    root, _ = checkpoint
    first = tmp_path / "node0"
    second = tmp_path / "node1"
    plan0 = run_build(root, first, 0)
    run_build(root, second, 1)

    assert (first / "metadata.tsv").read_bytes() == (
        second / "metadata.tsv"
    ).read_bytes()
    owners = {record.owner for record in plan0.expert_records}
    assert owners == set(range(NUM_NODES))


def test_dry_run_metadata_equals_the_real_build(checkpoint, tmp_path):
    root, _ = checkpoint
    real = tmp_path / "real"
    dry = tmp_path / "dry"
    run_build(root, real, 0)
    plan = run_build(root, dry, 0, extra=["--dry-run"])

    assert (dry / "metadata.tsv").read_bytes() == (
        real / "metadata.tsv"
    ).read_bytes()
    assert json.loads((dry / "node0.json").read_text()) == json.loads(
        (real / "node0.json").read_text()
    ) | {"store_path": str(dry / "store.bin"),
         "metadata_path": str(dry / "metadata.tsv"),
         "daemon_socket": str(dry / "daemon.sock"),
         "summary_path": str(dry / "summary.json")}
    # --dry-run writes no store, and never a summary.json: that file is the
    # daemon's runtime output (distributed_weight_daemon.cpp:1337-1378).
    assert not (dry / "store.bin").exists()
    assert not (dry / "summary.json").exists()
    assert not (real / "summary.json").exists()
    assert plan.store_bytes == (real / "store.bin").stat().st_size


# --------------------------------------------------------------------------- #
#  node<R>.json                                                                #
# --------------------------------------------------------------------------- #


def test_node_config_validates_through_the_real_loader(checkpoint, tmp_path):
    root, _ = checkpoint
    output = tmp_path / "node0"
    plan = run_build(root, output, 0)

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
    assert config["daemon_socket"] == str(output / "daemon.sock")
    assert config["summary_path"] == str(output / "summary.json")
    # H200 HCA numbering, NOT the loader's H20 default.
    assert config["rail_devices"] == list(sb.H200_RAIL_DEVICES)
    assert config["rail_devices"] != list(DEFAULT_RAIL_DEVICES)


def test_node_config_honours_the_optional_overrides(checkpoint, tmp_path):
    root, _ = checkpoint
    output = tmp_path / "node1"
    socket = tmp_path / "custom.sock"
    summary = tmp_path / "custom-summary.json"
    rails = ",".join(f"mlx5_{index}:1" for index in range(8))
    run_build(
        root,
        output,
        1,
        extra=[
            "--daemon-socket",
            str(socket),
            "--summary-path",
            str(summary),
            "--base-port",
            "27100",
            "--transport",
            "hierarchical_gdr",
            "--rail-devices",
            rails,
        ],
    )

    config = load_distributed_weight_config(output / "node1.json")
    assert config["node_rank"] == 1
    assert config["daemon_socket"] == str(socket)
    assert config["summary_path"] == str(summary)
    assert config["base_port"] == 27100
    assert config["transport"] == "hierarchical_gdr"
    assert config["rail_devices"] == rails.split(",")


# --------------------------------------------------------------------------- #
#  store.bin bytes                                                             #
# --------------------------------------------------------------------------- #


def test_node_store_holds_exactly_its_own_experts(checkpoint, tmp_path):
    root, payloads = checkpoint
    output = tmp_path / "node1"
    plan = run_build(root, output, 1)
    blob = (output / "store.bin").read_bytes()

    assert len(blob) == plan.store_bytes

    seen = bytearray(plan.store_bytes)
    local_experts = set()
    for record in plan.records:
        span = blob[
            record.compact_offset : record.compact_offset + record.byte_size
        ]
        if record.owner in (sb.REPLICATED_OWNER, 1):
            assert span == payloads[record.ckpt_name], record.ckpt_name
            for index in range(
                record.compact_offset,
                record.compact_offset + record.byte_size,
            ):
                assert seen[index] == 0, "two tensors overlap"
                seen[index] = 1
            if record.owner == 1:
                local_experts.add(record.module_key)

    # Node 1 holds the replicated prefix plus exactly its half of the experts,
    # and those spans tile the file with no gap and no padding.
    assert all(seen)
    assert len(local_experts) == len(MOE_LAYERS) * EXPERTS_PER_OWNER
    assert {
        int(key.rsplit("_", 1)[1]) for key in local_experts
    } == set(range(EXPERTS_PER_OWNER, EXPERTS))


def test_the_two_node_stores_differ_only_in_the_expert_region(
    checkpoint, tmp_path
):
    root, _ = checkpoint
    first = run_build(root, tmp_path / "n0", 0)
    second = run_build(root, tmp_path / "n1", 1)
    blob0 = (tmp_path / "n0" / "store.bin").read_bytes()
    blob1 = (tmp_path / "n1" / "store.bin").read_bytes()

    assert first.store_bytes == second.store_bytes
    prefix = first.replicated_bytes
    assert blob0[:prefix] == blob1[:prefix]
    assert blob0[prefix:] != blob1[prefix:]


def test_a_small_read_buffer_produces_the_same_store(checkpoint, tmp_path):
    root, _ = checkpoint
    plain = tmp_path / "plain"
    chunked = tmp_path / "chunked"
    run_build(root, plain, 0)
    run_build(root, chunked, 0, extra=["--read-buffer-bytes", "7"])

    assert (plain / "store.bin").read_bytes() == (
        chunked / "store.bin"
    ).read_bytes()


# --------------------------------------------------------------------------- #
#  Refusals: every unsupported input hard-fails with a named error             #
# --------------------------------------------------------------------------- #


def test_real_topology_is_enforced_on_the_cli_path(checkpoint, tmp_path):
    root, _ = checkpoint
    with pytest.raises(sb.K3StoreBuilderError, match="must be laid out for"):
        # The default expected_topology is the engine's 92 x 896.
        sb.build(
            [
                "--converted-ckpt",
                str(root),
                "--output-dir",
                str(tmp_path / "out"),
                "--num-nodes",
                "2",
                "--node-rank",
                "0",
                "--node-ips",
                "10.0.0.1,10.0.0.2",
                "--dry-run",
            ]
        )


def test_node_ip_count_must_match_num_nodes(checkpoint, tmp_path):
    root, _ = checkpoint
    with pytest.raises(sb.K3StoreBuilderError, match="--node-ips lists"):
        sb.build(
            [
                "--converted-ckpt",
                str(root),
                "--output-dir",
                str(tmp_path / "out"),
                "--num-nodes",
                "2",
                "--node-rank",
                "0",
                "--node-ips",
                "10.0.0.1",
            ],
            expected_topology=TOPOLOGY,
        )


def test_bad_node_rank_and_node_count_are_refused(checkpoint):
    root, _ = checkpoint
    with pytest.raises(sb.K3StoreBuilderError, match="node_rank must be"):
        sb.build_store_plan(
            root, NUM_NODES, node_rank=2, expected_topology=TOPOLOGY
        )
    with pytest.raises(sb.K3StoreBuilderError, match="num_nodes must be"):
        sb.build_store_plan(
            root, 3, node_rank=0, expected_topology=TOPOLOGY
        )


def test_unknown_dtype_is_refused_not_defaulted_to_fp8(tmp_path):
    root = tmp_path / "ckpt"
    declarations = [
        (name, "float4_e2m1" if name.endswith("w1.weight_packed") else dtype, shape)
        for name, dtype, shape in _declarations()
    ]
    root.mkdir(parents=True)
    # Written by hand: write_mini_checkpoint would need the itemsize first.
    (root / "shard-00000.bin").write_bytes(b"\x00" * 16)
    (root / "shard-00000.json").write_text(
        json.dumps(
            {
                "file_name": "shard-00000.bin",
                "total_byte_size": 16,
                "state_dict": {
                    declarations[0][0]: {
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


def test_an_unrecognised_attention_suffix_set_is_refused(tmp_path):
    root = tmp_path / "ckpt"
    declarations = [
        decl
        for decl in _declarations()
        # Drop one KDA tensor from layer 1, so its set matches neither ring.
        if decl[0]
        != f"{K3_CKPT_PREFIX}model.layers.1.self_attn.{K3_KDA_TENSOR_NAMES[0]}"
    ]
    write_mini_checkpoint(root, declarations)
    with pytest.raises(
        sb.K3StoreBuilderError, match="match neither K3_MLA_TENSOR_NAMES"
    ):
        sb.build_store_plan(
            root, NUM_NODES, node_rank=0, expected_topology=TOPOLOGY
        )


def test_an_incomplete_expert_module_is_refused(tmp_path):
    root = tmp_path / "ckpt"
    dropped = (
        f"{K3_CKPT_PREFIX}model.layers.2.block_sparse_moe.experts.5."
        "w2.weight_scale"
    )
    declarations = [decl for decl in _declarations() if decl[0] != dropped]
    write_mini_checkpoint(root, declarations)
    with pytest.raises(sb.K3StoreBuilderError, match="expected the six names"):
        sb.build_store_plan(
            root, NUM_NODES, node_rank=0, expected_topology=TOPOLOGY
        )


def test_a_byte_nonuniform_expert_module_is_refused(tmp_path):
    root = tmp_path / "ckpt"
    target = (
        f"{K3_CKPT_PREFIX}model.layers.2.block_sparse_moe.experts.5."
        "w2.weight_scale"
    )
    declarations = [
        (name, dtype, [8, 1] if name == target else shape)
        for name, dtype, shape in _declarations()
    ]
    write_mini_checkpoint(root, declarations)
    with pytest.raises(
        sb.K3StoreBuilderError, match="not byte-uniform"
    ):
        sb.build_store_plan(
            root, NUM_NODES, node_rank=0, expected_topology=TOPOLOGY
        )


def test_a_missing_expert_breaks_the_contiguous_run(tmp_path):
    root = tmp_path / "ckpt"
    declarations = [
        decl
        for decl in _declarations()
        if ".block_sparse_moe.experts.5." not in decl[0]
    ]
    write_mini_checkpoint(root, declarations)
    with pytest.raises(
        sb.K3StoreBuilderError, match="contiguous run 0..N-1"
    ):
        sb.build_store_plan(
            root, NUM_NODES, node_rank=0, expected_topology=TOPOLOGY
        )


def test_a_moe_layer_run_not_starting_at_one_is_refused(tmp_path):
    root = tmp_path / "ckpt"
    declarations = [
        decl
        for decl in _declarations()
        if ".layers.1.block_sparse_moe." not in decl[0]
    ]
    write_mini_checkpoint(root, declarations)
    with pytest.raises(
        sb.K3StoreBuilderError, match="contiguous run 1..N"
    ):
        sb.build_store_plan(
            root, NUM_NODES, node_rank=0, expected_topology=(2, EXPERTS)
        )


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


def test_main_reports_a_named_failure_without_raising(tmp_path):
    assert (
        sb.main(
            [
                "--converted-ckpt",
                str(tmp_path / "absent"),
                "--output-dir",
                str(tmp_path / "out"),
                "--num-nodes",
                "2",
                "--node-rank",
                "0",
                "--node-ips",
                "10.0.0.1,10.0.0.2",
                "--dry-run",
            ]
        )
        == 1
    )
