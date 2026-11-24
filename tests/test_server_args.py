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
    base = ["--model", "m", "--host-kv-cache-size", "1"]
    for mode in ("streamed", "streamed_sp8", "resident_ep"):
        parsed = parser.parse_args(base + ["--prefill-moe-mode", mode])
        assert parsed.prefill_moe_mode == mode
    assert parser.parse_args(base).prefill_moe_mode is None
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--prefill-moe-mode", "resident"])


def test_prefix_rank_launch_parameters_parse_and_default_to_the_packer():
    parser = server_args_module._build_parser()
    base = ["--model", "m", "--host-kv-cache-size", "1"]
    parsed = parser.parse_args(base)
    assert parsed.prefix_rank_assignment_mode == "packing_v2"
    assert parsed.prefix_rank_cost_profile == "on"
    assert ServerArgs(model="m").prefix_rank_assignment_mode == "packing_v2"
    assert ServerArgs(model="m").prefix_rank_cost_profile == "on"
    parsed = parser.parse_args(base + [
        "--prefix-rank-assignment-mode", "global_idx",
        "--prefix-rank-cost-profile", "off",
    ])
    assert parsed.prefix_rank_assignment_mode == "global_idx"
    assert parsed.prefix_rank_cost_profile == "off"
    for flag in ("--prefix-rank-assignment-mode", "--prefix-rank-cost-profile"):
        with pytest.raises(SystemExit):
            parser.parse_args(base + [flag, "maybe"])


@pytest.mark.parametrize(
    "field", ["prefix_rank_assignment_mode", "prefix_rank_cost_profile"]
)
def test_prefix_rank_launch_parameters_reach_the_worker(field):
    # Parsed -> ServerArgs -> the worker's args: a flag that is accepted and
    # then dropped on the way would silently run the default arm.
    root = Path(server_args_module.__file__).resolve().parents[1]
    server_args_source = Path(server_args_module.__file__).read_text()
    assert f"{field}=parsed.{field}," in server_args_source
    manager_source = (root / "server" / "worker_manager.py").read_text()
    assert f"{field}=self.args.{field}," in manager_source
    worker_source = (root / "batchgen_worker.py").read_text()
    assert f"=self.args.{field}," in worker_source


LAYOUT_GROUP_FLAGS = {
    "prefill_attention_tp": "--prefill-attention-tp",
    "prefill_moe_ep": "--prefill-moe-ep",
    "decode_attention_tp": "--decode-attention-tp",
    "decode_moe_ep": "--decode-moe-ep",
}


@pytest.mark.parametrize("field, flag", sorted(LAYOUT_GROUP_FLAGS.items()))
def test_layout_group_flags_parse_and_default_to_the_model(field, flag):
    parser = server_args_module._build_parser()
    base = ["--model", "m", "--host-kv-cache-size", "1"]
    assert getattr(parser.parse_args(base), field) is None
    assert getattr(parser.parse_args(base + [flag, "8"]), field) == 8
    assert getattr(ServerArgs(model="m"), field) is None
    with pytest.raises(SystemExit):
        parser.parse_args(base + [flag, "eight"])


@pytest.mark.parametrize("field", sorted(LAYOUT_GROUP_FLAGS))
def test_layout_group_flags_reach_the_worker_and_the_model_initializer(field):
    # Parsed -> ServerArgs -> the worker's args -> the initializer's input
    # arguments: a flag accepted and then dropped on the way would silently
    # run the default layout.
    root = Path(server_args_module.__file__).resolve().parents[1]
    assert f"{field}=parsed.{field}," in Path(server_args_module.__file__).read_text()
    assert f"{field}=self.args.{field}," in (root / "server" / "worker_manager.py").read_text()
    worker_source = (root / "batchgen_worker.py").read_text()
    assert f'"{field}": self.args.{field},' in worker_source
    assert worker_source.count(f"\t{field}: Optional[int] = None") == 2


def test_debug_num_layers_flag_parses_and_defaults_to_all_layers():
    parser = server_args_module._build_parser()
    base = ["--model", "m", "--host-kv-cache-size", "1"]
    assert parser.parse_args(base + ["--debug-num-layers", "4"]).debug_num_layers == 4
    assert parser.parse_args(base).debug_num_layers is None
    assert ServerArgs(model="m").debug_num_layers is None


def test_every_parsed_flag_reaches_server_args():
    """A flag the parser accepts but the conversion drops is silently ignored."""
    import ast
    import dataclasses

    source = (ROOT / "batchgen" / "server" / "server_args.py").read_text()
    prepare = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_server_args"
    )
    forwarded = {
        keyword.arg
        for call in ast.walk(prepare)
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "ServerArgs"
        for keyword in call.keywords
    }
    fields = {field.name for field in dataclasses.fields(ServerArgs)}
    parsed = {action.dest for action in server_args_module._build_parser()._actions}
    assert sorted((fields & parsed) - forwarded) == []


@pytest.mark.parametrize("bad", [0, -3])
def test_debug_num_layers_must_be_positive(tmp_path, bad):
    with pytest.raises(ValueError, match="--debug-num-layers"):
        validate_server_args(
            _args(
                tmp_path,
                distributed_weight_config=None,
                nnodes=1,
                world_size=8,
                debug_num_layers=bad,
            )
        )


def test_debug_num_layers_refuses_the_distributed_weight_store(tmp_path):
    # The K3 distributed store is built and served at full depth
    # (store_builder K3_LAYERS / distributed_weight_daemon.cpp kLayers).
    with pytest.raises(ValueError, match="--distributed-weight-config"):
        validate_server_args(_args(tmp_path, debug_num_layers=4))


def test_debug_num_layers_refuses_the_prefix_cache(tmp_path, monkeypatch):
    import batchgen.prefix_reuse.config as prefix_config

    monkeypatch.setattr(
        prefix_config, "require_prefix_cache_model_support", lambda model: None
    )
    with pytest.raises(ValueError, match="--enable-prefix-cache"):
        validate_server_args(
            _args(
                tmp_path,
                distributed_weight_config=None,
                nnodes=1,
                world_size=8,
                enable_prefix_cache=True,
                debug_num_layers=4,
            )
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


@pytest.mark.parametrize("value", ["-1", "4096"])
def test_max_model_len_cli(value):
    parsed = server_args_module._build_parser().parse_args([
        "--model", "test-model", "--max-model-len", value,
    ])
    assert parsed.max_model_len == int(value)


def test_max_model_len_cli_default_is_model_context():
    assert server_args_module._build_parser().parse_args([
        "--model", "test-model",
    ]).max_model_len is None


@pytest.mark.parametrize("value", [0, 1, -2])
def test_max_model_len_rejects_invalid_values(tmp_path, value):
    args = ServerArgs(
        model="gpt-oss-120b", storage_path=tmp_path,
        host_kv_cache_size=1, max_model_len=value,
    )
    with pytest.raises(ValueError, match="max_model_len"):
        validate_server_args(args)


@pytest.mark.parametrize("flags, expected", [
    ([], (None, None)),
    (["--parse-thinking", "--parse-tool-call"], (True, True)),
    (["--no-parse-thinking", "--no-parse-tool-call"], (False, False)),
    (["--no-parse-thinking"], (False, None)),
    (["--parse-tool-call"], (None, True)),
])
def test_output_parsing_flags_are_tri_state(tmp_path, monkeypatch, flags, expected):
    """Unset = the model's declared default (None); --[no-]parse-* force it."""
    monkeypatch.chdir(tmp_path)
    args = server_args_module.prepare_server_args([
        "--model", "openai/gpt-oss-120b",
        "--host-kv-cache-size", "1",
        "--storage-path", str(tmp_path / "storage"),
        *flags,
    ])
    assert (args.parse_thinking, args.parse_tool_call) == expected


def test_output_parsing_dataclass_defaults_are_model_default():
    args = ServerArgs(model="openai/gpt-oss-120b")
    assert (args.parse_thinking, args.parse_tool_call) == (None, None)
