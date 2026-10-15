"""Kimi-K3 multi-node defaults to the distributed host-weight store.

The replicated parameter server stages the whole checkpoint per node, so it is
an explicit `--replicated-weights` opt-in rather than the silent default.
"""

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


@pytest.fixture(autouse=True)
def _skip_port_probes(monkeypatch):
    """Validation binds real sockets; the weight-source rules do not."""
    monkeypatch.setattr(
        server_args_module,
        "_ensure_local_port_free",
        lambda port, label: None,
    )


def _args(tmp_path, *, with_config=False, **overrides):
    values = {
        "model": "moonshotai/Kimi-K3",
        "distributed_weight_config": None,
        "nnodes": 2,
        "node_rank": 0,
        "world_size": 16,
        "host_kv_cache_size": 1,
        "storage_path": tmp_path / "storage",
    }
    if with_config:
        config = tmp_path / "node0.json"
        config.write_text("{}")
        values["distributed_weight_config"] = config
    values.update(overrides)
    return ServerArgs(**values)


def test_multinode_without_a_weight_source_names_the_store_flag(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        validate_server_args(_args(tmp_path))
    message = str(excinfo.value)
    assert "--distributed-weight-config" in message
    assert "--replicated-weights" in message
    assert "deploy-kimi-k3-h200.md" in message


def test_multinode_accepts_the_replicated_opt_in(tmp_path):
    validate_server_args(_args(tmp_path, replicated_weights=True))


def test_multinode_accepts_the_distributed_store(tmp_path):
    validate_server_args(_args(tmp_path, with_config=True))


def test_distributed_store_constraints_still_apply(tmp_path):
    """The default-path rule must not shadow the existing topology rules."""
    with pytest.raises(ValueError, match="--nnodes 2 --world-size 16"):
        validate_server_args(
            _args(tmp_path, with_config=True, nnodes=3, world_size=24)
        )
    with pytest.raises(
        ValueError, match="currently supports Kimi-K3 only"
    ):
        validate_server_args(
            _args(tmp_path, with_config=True, model="openai/gpt-oss-120b")
        )


def test_both_weight_sources_are_contradictory(tmp_path):
    with pytest.raises(ValueError, match="contradictory"):
        validate_server_args(
            _args(tmp_path, with_config=True, replicated_weights=True)
        )


def test_contradiction_is_checked_for_every_model_and_topology(tmp_path):
    for overrides in (
        {"model": "openai/gpt-oss-120b"},
        {"nnodes": 1, "world_size": 8},
    ):
        with pytest.raises(ValueError, match="contradictory"):
            validate_server_args(
                _args(
                    tmp_path,
                    with_config=True,
                    replicated_weights=True,
                    **overrides,
                )
            )


def test_other_models_keep_the_replicated_default_on_multiple_nodes(tmp_path):
    validate_server_args(_args(tmp_path, model="openai/gpt-oss-120b"))
    validate_server_args(
        _args(tmp_path, model="zai-org/GLM-5", nnodes=4, world_size=32)
    )


def test_single_node_kimi_k3_is_unchanged(tmp_path):
    validate_server_args(_args(tmp_path, nnodes=1, world_size=8))
    # The opt-in flag is inert where there is no multi-node default to switch.
    validate_server_args(
        _args(
            tmp_path, nnodes=1, world_size=8, replicated_weights=True
        )
    )


def test_replicated_weights_defaults_to_false():
    assert ServerArgs(model="moonshotai/Kimi-K3").replicated_weights is False


def test_cli_exposes_the_opt_in_as_a_store_true_flag():
    parser = server_args_module._build_parser()
    base = ["--model", "moonshotai/Kimi-K3"]
    assert parser.parse_args(base).replicated_weights is False
    assert parser.parse_args(
        base + ["--replicated-weights"]
    ).replicated_weights is True
