"""Server-argument validation for the K3 distributed host-weight topologies."""

import importlib
import os
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
    # Acceptance needs a real published store: the launch preflight verifies
    # its completion manifest (tests/test_kimi_k3_distributed_store_preflight.py).
    from tests.test_kimi_k3_distributed_store_preflight import build_store
    from tests.test_kimi_k3_store_builder import Env

    env = Env(tmp_path / "k3")
    for nnodes in (2, 4):
        config = build_store(env, tmp_path / f"store{nnodes}", num_nodes=nnodes)
        validate_server_args(
            _args(
                tmp_path,
                nnodes=nnodes,
                world_size=nnodes * 8,
                distributed_weight_config=config,
                converted_ckpt_dir=env.ckpt,
                cache_dir=env.model_dir,
            )
        )


def test_distributed_weights_reject_other_node_counts(tmp_path):
    for nnodes in (1, 3, 8):
        with pytest.raises(ValueError, match="require --nnodes 2 or 4"):
            validate_server_args(
                _args(tmp_path, nnodes=nnodes, world_size=nnodes * 8)
            )


def _published(tmp_path, *extra, nnodes=2):
    from tests.test_kimi_k3_distributed_store_preflight import TOPOLOGY
    from tests.test_kimi_k3_store_builder import Env
    from batchgen.models.moonshotai.kimi_linear import store_builder as sb

    env = Env(tmp_path / "k3")
    output = tmp_path / f"store-{'-'.join(extra) or 'default'}"
    argv = env.argv(output, 0)
    argv[argv.index("--num-nodes") + 1] = str(nnodes)
    argv[argv.index("--node-ips") + 1] = ",".join(
        f"10.0.0.{index + 1}" for index in range(nnodes)
    )
    sb.build(argv + list(extra), expected_topology=TOPOLOGY)
    return env, output / "node0.json"


def test_distributed_weights_require_the_store_workers_per_node(tmp_path):
    env, config = _published(tmp_path)
    for world_size in (8, 32):
        with pytest.raises(ValueError, match="8 workers per host require --world-size 16"):
            validate_server_args(
                _args(
                    tmp_path, nnodes=2, world_size=world_size,
                    distributed_weight_config=config,
                    converted_ckpt_dir=env.ckpt, cache_dir=env.model_dir,
                )
            )


def test_a_four_worker_local_store_runs_two_trays_as_world8(tmp_path):
    env, config = _published(
        tmp_path, "--transport", "local", "--workers-per-node", "4",
    )
    common = dict(
        nnodes=2, distributed_weight_config=config,
        converted_ckpt_dir=env.ckpt, cache_dir=env.model_dir,
    )
    validate_server_args(_args(tmp_path, world_size=8, **common))
    with pytest.raises(ValueError, match="4 workers per host require --world-size 8"):
        validate_server_args(_args(tmp_path, world_size=16, **common))


def test_resident_ep_prefill_with_the_prefix_cache_is_refused_at_startup(
    tmp_path, monkeypatch,
):
    import batchgen.prefix_reuse.config as prefix_config

    monkeypatch.setattr(
        prefix_config, "require_prefix_cache_model_support", lambda model: None
    )
    with pytest.raises(ValueError, match="prefill prefix pool runs per DP rank"):
        validate_server_args(
            _args(
                tmp_path,
                distributed_weight_config=None,
                nnodes=1,
                world_size=8,
                enable_prefix_cache=True,
                prefill_moe_mode="resident_ep",
            )
        )


def test_prefill_moe_mode_flag_parses_to_server_args():
    parser = server_args_module._build_parser()
    parsed = parser.parse_args(
        ["--model", "m", "--host-kv-cache-size", "1",
         "--prefill-moe-mode", "resident_ep"]
    )
    assert parsed.prefill_moe_mode == "resident_ep"
    assert parser.parse_args(
        ["--model", "m", "--host-kv-cache-size", "1"]
    ).prefill_moe_mode is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model", "m", "--host-kv-cache-size", "1",
             "--prefill-moe-mode", "streamed"]
        )


def test_topology_rules_apply_only_to_distributed_host_weights(tmp_path):
    # Off the distributed store any positive world_size stands. Kimi-K3
    # multi-node now defaults to that store, so the replicated path this case
    # describes has to be requested explicitly.
    validate_server_args(
        _args(
            tmp_path,
            distributed_weight_config=None,
            replicated_weights=True,
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


@pytest.mark.parametrize("max_pool_size", [0, -1])
def test_non_pool_mode_is_rejected(tmp_path, max_pool_size):
    """--max-pool-size 0 used to select the removed non-pool path."""
    with pytest.raises(ValueError, match="legacy non-pool mode was removed"):
        validate_server_args(
            _args(
                tmp_path,
                distributed_weight_config=None,
                nnodes=1,
                world_size=1,
                max_pool_size=max_pool_size,
            )
        )


def test_default_pool_size_is_accepted(tmp_path):
    args = _args(
        tmp_path, distributed_weight_config=None, nnodes=1, world_size=1
    )
    assert args.max_pool_size > 0
    validate_server_args(args)


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


def test_log_options_resolved_absolute_with_info_default(tmp_path, monkeypatch):
    """The log root is fixed against the launch directory at parse time, so
    every process of the server -- workers included -- names the same one."""
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    argv = [
        "--model", "openai/gpt-oss-120b",
        "--host-kv-cache-size", "1",
        "--storage-path", str(tmp_path / "storage"),
    ]

    defaults = server_args_module.prepare_server_args(argv)
    assert defaults.log_dir == os.path.join(cwd, "batchgen_logs")
    assert defaults.log_level == "INFO"

    explicit = server_args_module.prepare_server_args(
        argv + ["--log-dir", "logs/run", "--log-level", "warning"]
    )
    assert explicit.log_dir == os.path.join(cwd, "logs", "run")
    assert explicit.log_level == "WARNING"
