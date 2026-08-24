"""Server-argument validation for the K3 distributed host-weight topologies."""

import pytest

from batchgen.server import server_args as server_args_module
from batchgen.server.server_args import ServerArgs, validate_server_args


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
