"""Server-argument validation for the K3 distributed host-weight topologies."""

import importlib
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_server_args_module():
    """Import the leaf module without running the HTTP server package init."""
    package_name = "batchgen.server"
    previous = sys.modules.get(package_name)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return importlib.import_module("batchgen.server.server_args")
    finally:
        if previous is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous


server_args_module = _load_server_args_module()
ServerArgs = server_args_module.ServerArgs
validate_server_args = server_args_module.validate_server_args
validate_shared_runtime_capability = (
    server_args_module.validate_shared_runtime_capability
)


@pytest.fixture(autouse=True)
def _skip_port_probes(monkeypatch):
    """Validation binds real sockets; the topology rules under test do not."""
    monkeypatch.setattr(
        server_args_module,
        "_ensure_local_port_free",
        lambda port, label: None,
    )


def _args(tmp_path, **overrides):
    config = tmp_path / "distributed_weights.json"
    config.write_text("{}")
    values = {
        "model": "moonshotai/Kimi-K3",
        "distributed_weight_config": config,
        "nnodes": 4,
        "node_rank": 0,
        "world_size": 32,
        "host_kv_cache_size": 1,
        "storage_path": tmp_path / "storage",
    }
    values.update(overrides)
    return ServerArgs(**values)


def test_distributed_weights_accept_two_and_four_node_topologies(tmp_path):
    validate_server_args(_args(tmp_path, nnodes=2, world_size=16))
    validate_server_args(_args(tmp_path, nnodes=4, world_size=32))


def test_distributed_weights_reject_other_node_counts(tmp_path):
    for nnodes in (1, 3, 8):
        with pytest.raises(ValueError, match="--nnodes 2 --world-size 16"):
            validate_server_args(
                _args(tmp_path, nnodes=nnodes, world_size=nnodes * 8)
            )


def test_distributed_weights_require_eight_ranks_per_node(tmp_path):
    for nnodes, world_size in ((2, 32), (4, 16), (2, 8), (4, 8)):
        with pytest.raises(ValueError, match="--nnodes 2 --world-size 16"):
            validate_server_args(
                _args(tmp_path, nnodes=nnodes, world_size=world_size)
            )


def test_topology_rules_apply_only_to_distributed_host_weights(tmp_path):
    # Without --distributed-weight-config any positive world_size stands.
    validate_server_args(
        _args(
            tmp_path,
            distributed_weight_config=None,
            nnodes=3,
            world_size=24,
        )
    )


def test_world_size_must_divide_evenly_across_nodes(tmp_path):
    with pytest.raises(ValueError, match="world_size must be divisible"):
        validate_server_args(
            _args(
                tmp_path,
                distributed_weight_config=None,
                nnodes=3,
                world_size=8,
            )
        )


def test_pynccl_range_must_be_positive_and_fit_port_space(tmp_path):
    with pytest.raises(ValueError, match="pynccl_port_span must be positive"):
        validate_server_args(
            _args(
                tmp_path,
                distributed_weight_config=None,
                nnodes=1,
                world_size=1,
                pynccl_port_span=0,
            )
        )

    with pytest.raises(ValueError, match="must end at or before 65535"):
        validate_server_args(
            _args(
                tmp_path,
                distributed_weight_config=None,
                nnodes=1,
                world_size=1,
                pynccl_port_base=65535,
                pynccl_port_span=2,
            )
        )


def _shared_args(tmp_path, **overrides):
    values = {
        "model": "openai/gpt-oss-120b",
        "instance_id": "lane-0",
        "runtime_mode": "shared",
        "lane_lease_manifest_fd": 99,
        "cache_dir": tmp_path / "checkpoint",
        "converted_ckpt_dir": tmp_path / "converted",
        "nnodes": 1,
        "node_rank": 0,
        "world_size": 1,
        "host_kv_cache_size": 1,
        "listen_port": 11000,
        "dist_init_addr": "localhost:12000",
        "pynccl_port_base": 21000,
        "pynccl_port_span": 4,
        "storage_path": tmp_path / "storage",
    }
    values.update(overrides)
    return ServerArgs(**values)


def test_shared_capability_accepts_only_qualified_shape(tmp_path):
    validate_shared_runtime_capability(_shared_args(tmp_path))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model": "other/model"}, "qualified only"),
        ({"world_size": 3}, "world_size"),
        ({"nnodes": 2}, "one node"),
        ({"fast_init": True}, "rejects fast-init"),
        ({"enable_hugetlbfs": True}, "rejects fast-init"),
        ({"enable_deepep": True}, "rejects fast-init"),
        ({"enable_ep_with_offloading": True}, "EP offloading"),
        ({"enable_cuda_graph": True}, "CUDA graph"),
        ({"watchdog_timeout": 30}, "watchdog"),
        ({"decode_step_timeout": 30}, "watchdog"),
        ({"lane_lease_manifest_fd": None}, "lease-manifest"),
        ({"cache_dir": None}, "explicit cache_dir"),
    ],
)
def test_shared_capability_rejects_unqualified_shapes(
    tmp_path, override, message
):
    with pytest.raises(ValueError, match=message):
        validate_shared_runtime_capability(
            _shared_args(tmp_path, **override)
        )


def test_shared_capability_rejects_overlapping_ports(tmp_path):
    with pytest.raises(ValueError, match="ports must differ"):
        validate_shared_runtime_capability(
            _shared_args(tmp_path, dist_init_addr="localhost:11000")
        )
    with pytest.raises(ValueError, match="allocations overlap"):
        validate_shared_runtime_capability(
            _shared_args(tmp_path, listen_port=21001, world_size=2)
        )
