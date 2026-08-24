import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
INITIALIZER = (
    ROOT
    / "batchgen"
    / "models"
    / "moonshotai"
    / "kimi_linear"
    / "kimi_initializer.py"
)


def _class(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(klass, name):
    return next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_distributed_weight_config_reaches_kimi_initializer():
    """The worker and initializer must observe the same distributed store path."""
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    worker = _class(tree, "BatchGenWorker")
    args = _class(tree, "InputArguments")

    fields = {
        node.target.id
        for node in args.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    assert "distributed_weight_config" in fields

    initialize = _method(worker, "_initialize_core_components")
    assignment = next(
        node
        for node in ast.walk(initialize)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "input_arguments"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Dict)

    config_entry = next(
        value
        for key, value in zip(assignment.value.keys, assignment.value.values)
        if isinstance(key, ast.Constant)
        and key.value == "distributed_weight_config"
    )
    assert isinstance(config_entry, ast.Attribute)
    assert isinstance(config_entry.value, ast.Attribute)
    assert isinstance(config_entry.value.value, ast.Name)
    assert config_entry.value.value.id == "self"
    assert config_entry.value.attr == "args"
    assert config_entry.attr == "distributed_weight_config"


def _write_store_config(tmp_path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "store.bin"
    store.write_bytes(b"store")
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text("H\n")
    config = {
        "node_rank": 0,
        "node_ips": ["n0", "n1", "n2", "n3"],
        "workers": 8,
        "store_path": str(store),
        "metadata_path": str(metadata),
        "daemon_socket": str(tmp_path / "daemon.sock"),
        "summary_path": str(tmp_path / "summary.json"),
        "store_bytes": store.stat().st_size,
        "replicated_bytes": 0,
        "module_bytes": 1,
        "worker_sharded": True,
    }
    config.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_weight_transport_defaults_to_host_rdma_and_accepts_hierarchical(
    tmp_path,
):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        load_distributed_weight_config,
    )

    # Backward compatibility: a config written before the transport key exists
    # keeps the validated host-RDMA behaviour, and the key is normalized in so
    # callers never have to re-apply the default.
    legacy = load_distributed_weight_config(_write_store_config(tmp_path))
    assert legacy["transport"] == "host_rdma"

    selected = load_distributed_weight_config(
        _write_store_config(tmp_path / "gdr", transport="hierarchical_gdr")
    )
    assert selected["transport"] == "hierarchical_gdr"


def test_store_config_normalizes_two_and_four_node_topologies(tmp_path):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        DISTRIBUTED_NODE_COUNTS,
        load_distributed_weight_config,
    )

    assert DISTRIBUTED_NODE_COUNTS == (2, 4)

    two = load_distributed_weight_config(
        _write_store_config(
            tmp_path / "two", node_ips=["n0", "n1"], node_rank=1
        )
    )
    assert two["num_nodes"] == 2

    four = load_distributed_weight_config(
        _write_store_config(tmp_path / "four", node_rank=3)
    )
    assert four["num_nodes"] == 4


def test_store_config_rejects_node_counts_other_than_two_or_four(tmp_path):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        load_distributed_weight_config,
    )

    for count in (1, 3, 5, 8):
        path = _write_store_config(
            tmp_path / f"n{count}",
            node_ips=[f"n{index}" for index in range(count)],
        )
        with pytest.raises(ValueError, match="exactly 2 or 4 node"):
            load_distributed_weight_config(path)


def test_store_config_rejects_node_rank_outside_its_topology(tmp_path):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        load_distributed_weight_config,
    )

    # Valid on four nodes, out of range on two.
    two = _write_store_config(
        tmp_path / "two", node_ips=["n0", "n1"], node_rank=2
    )
    with pytest.raises(ValueError, match=r"node_rank must be in \[0, 2\)"):
        load_distributed_weight_config(two)

    four = _write_store_config(tmp_path / "four", node_rank=4)
    with pytest.raises(ValueError, match=r"node_rank must be in \[0, 4\)"):
        load_distributed_weight_config(four)


def test_store_config_rejects_num_nodes_disagreeing_with_the_ip_list(tmp_path):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        load_distributed_weight_config,
    )

    stale = _write_store_config(
        tmp_path / "stale", node_ips=["n0", "n1"], num_nodes=4
    )
    with pytest.raises(ValueError, match="num_nodes 4 disagrees"):
        load_distributed_weight_config(stale)


def test_weight_transport_rejects_values_outside_the_two_allowed(tmp_path):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        WEIGHT_TRANSPORTS,
        load_distributed_weight_config,
    )

    assert WEIGHT_TRANSPORTS == ("host_rdma", "hierarchical_gdr")
    for value in ("gdr", "nvlink", "HOST_RDMA", "", None, 1):
        path = _write_store_config(tmp_path / str(value), transport=value)
        with pytest.raises(ValueError, match="transport must be one of"):
            load_distributed_weight_config(path)


def test_rail_devices_default_and_override_are_normalized(tmp_path):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        DEFAULT_RAIL_DEVICES,
        load_distributed_weight_config,
    )

    legacy = load_distributed_weight_config(_write_store_config(tmp_path))
    assert legacy["rail_devices"] == list(DEFAULT_RAIL_DEVICES)

    h200_devices = [f"mlx5_bond_{index}:1" for index in range(8)]
    selected = load_distributed_weight_config(
        _write_store_config(tmp_path / "h200", rail_devices=h200_devices)
    )
    assert selected["rail_devices"] == h200_devices


@pytest.mark.parametrize(
    "rail_devices",
    (
        "mlx5_bond_0:1",
        [],
        [f"mlx5_bond_{index}:1" for index in range(7)],
        [f"mlx5_bond_{index}:1" for index in range(8)] + ["extra:1"],
        [f"mlx5_bond_{index}:1" for index in range(7)] + [""],
        [f"mlx5_bond_{index}:1" for index in range(7)] + [1],
    ),
)
def test_rail_devices_require_eight_non_empty_strings(
    tmp_path, rail_devices
):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        load_distributed_weight_config,
    )

    path = _write_store_config(tmp_path, rail_devices=rail_devices)
    with pytest.raises(ValueError, match="exactly eight non-empty strings"):
        load_distributed_weight_config(path)


def test_initializer_publishes_the_selected_transport_without_env_vars():
    tree = ast.parse(INITIALIZER.read_text(), filename=str(INITIALIZER))
    source = INITIALIZER.read_text()

    # The value reaches Basic_Config from the store config, never getenv.
    stamped = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "distributed_weight_transport"
            and ast.unparse(target).endswith(
                "Basic_Config.distributed_weight_transport"
            )
            for target in node.targets
        )
    ]
    assert len(stamped) == 1
    assert (
        ast.unparse(stamped[0].value) == "self.distributed_weight_transport"
    )
    assert "load_distributed_weight_config" in source
    assert "TRANSPORT" not in source
    assert 'self.distributed_weight_transport = "host_rdma"' in source
