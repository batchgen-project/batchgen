"""Kimi-K3 weight-source preflight, driven end to end through
``prepare_server_args(argv)``.

Every store here is a real one: built by the store builder from a mini
converted checkpoint and a real ``config.json`` (the fixtures of
``tests/test_kimi_k3_store_builder.py``), then judged by the same preflight
``launch_http_server`` runs before any worker, SHM segment or port exists.
"""

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batchgen.models.moonshotai.kimi_linear import store_builder as sb  # noqa: E402
from tests.test_kimi_k3_store_builder import (  # noqa: E402
    TOPOLOGY,
    Env,
    _declarations,
    write_mini_checkpoint,
)


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
prepare_server_args = server_args_module.prepare_server_args
WeightSourceError = server_args_module.WeightSourceError


@pytest.fixture(autouse=True)
def _skip_port_probes(monkeypatch):
    """Validation binds real sockets; the weight-source rules do not."""
    monkeypatch.setattr(
        server_args_module, "_ensure_local_port_free", lambda port, label: None
    )


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def build_store(env, output, node_rank=0, num_nodes=2, ckpt=None):
    """Publish one node's store; return its node config path."""
    argv = env.argv(output, node_rank, ckpt=ckpt)
    argv[argv.index("--num-nodes") + 1] = str(num_nodes)
    argv[argv.index("--node-ips") + 1] = ",".join(
        f"10.0.0.{index + 1}" for index in range(num_nodes)
    )
    sb.build(argv, expected_topology=TOPOLOGY)
    return output / f"node{node_rank}.json"


def launch(tmp_path, *extra, nnodes=2, node_rank=0, model="moonshotai/Kimi-K3"):
    return prepare_server_args(
        [
            "--model", model,
            "--nnodes", str(nnodes),
            "--world-size", str(nnodes * 8),
            "--node-rank", str(node_rank),
            "--host-kv-cache-size", "1",
            "--storage-path", str(tmp_path / "storage"),
            *[str(item) for item in extra],
        ]
    )


def store_flags(env, config, converted=None):
    return [
        "--distributed-weight-config", config,
        "--converted-ckpt-dir", converted or env.ckpt,
        "--cache-dir", env.model_dir,
    ]


def _assert_rebuild_command(message, env, output, converted=None):
    assert "Rebuild it with: python -m " + sb.BUILDER_MODULE in message
    assert f"--converted-ckpt {converted or env.ckpt}" in message
    assert f"--cache-dir {env.model_dir}" in message
    assert f"--output-dir {output}" in message
    assert "--num-nodes 2 --node-rank 0 --node-ips 10.0.0.1,10.0.0.2" in message


# --------------------------------------------------------------------------- #
#  Default weight source                                                      #
# --------------------------------------------------------------------------- #


def test_multinode_k3_without_a_weight_source_is_refused(tmp_path):
    with pytest.raises(WeightSourceError) as excinfo:
        launch(tmp_path)
    message = str(excinfo.value)
    assert "--distributed-weight-config" in message
    assert "--replicated-weights" in message
    assert sb.BUILDER_MODULE in message


def test_multinode_k3_accepts_the_replicated_opt_in(tmp_path):
    args = launch(tmp_path, "--replicated-weights")
    assert args.replicated_weights is True
    assert args.distributed_weight_config is None


# --------------------------------------------------------------------------- #
#  A valid store                                                              #
# --------------------------------------------------------------------------- #


def test_a_valid_store_passes_the_launch_preflight(env, tmp_path):
    config = build_store(env, tmp_path / "store0")
    args = launch(tmp_path, *store_flags(env, config))
    assert args.distributed_weight_config == config


def test_a_four_node_store_passes_for_its_own_rank(env, tmp_path):
    config = build_store(env, tmp_path / "store3", node_rank=3, num_nodes=4)
    args = launch(tmp_path, *store_flags(env, config), nnodes=4, node_rank=3)
    assert args.distributed_weight_config == config


def test_the_converted_checkpoint_resolves_from_cache_dir(env, tmp_path):
    """No --converted-ckpt-dir: <cache-dir>/converted_ckpt, as the worker does."""
    nested = env.model_dir / "converted_ckpt"
    write_mini_checkpoint(nested, _declarations(env.cfg))
    config = build_store(env, tmp_path / "store0", ckpt=nested)
    launch(
        tmp_path,
        "--distributed-weight-config", config,
        "--cache-dir", env.model_dir,
    )


# --------------------------------------------------------------------------- #
#  Stale, incomplete, foreign stores                                          #
# --------------------------------------------------------------------------- #


def test_no_config_file_is_refused(env, tmp_path):
    with pytest.raises(WeightSourceError, match="distributed_weight_config does not exist"):
        launch(tmp_path, *store_flags(env, tmp_path / "absent.json"))


def test_an_unpublished_store_is_refused_with_the_rebuild_command(env, tmp_path):
    output = tmp_path / "store0"
    config = build_store(env, output)
    (output / sb.MANIFEST_FILE_NAME).unlink()  # the publish never committed
    with pytest.raises(sb.K3StoreManifestError) as excinfo:
        launch(tmp_path, *store_flags(env, config))
    message = str(excinfo.value)
    assert "no completion manifest" in message
    _assert_rebuild_command(message, env, output)


def test_an_incomplete_manifest_is_refused(env, tmp_path):
    output = tmp_path / "store0"
    config = build_store(env, output)
    manifest_path = output / sb.MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(sb.K3StoreManifestError, match="does not mark the build complete"):
        launch(tmp_path, *store_flags(env, config))


def test_a_manifest_of_another_format_version_is_refused(env, tmp_path):
    output = tmp_path / "store0"
    config = build_store(env, output)
    manifest_path = output / sb.MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = sb.MANIFEST_FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(sb.K3StoreManifestError, match="expected .* version"):
        launch(tmp_path, *store_flags(env, config))


def test_a_store_cleared_from_under_its_config_is_refused(env, tmp_path):
    """The tmpfs was cleared but node<R>.json survived on disk."""
    output = tmp_path / "store0"
    config = build_store(env, output)
    (output / "store.bin").unlink()
    with pytest.raises(sb.K3StoreManifestError) as excinfo:
        launch(tmp_path, *store_flags(env, config))
    message = str(excinfo.value)
    assert "is not readable" in message
    _assert_rebuild_command(message, env, output)


def test_a_partially_copied_store_is_refused(env, tmp_path):
    output = tmp_path / "store0"
    config = build_store(env, output)
    with open(output / "store.bin", "r+b") as handle:
        handle.truncate(17)
    with pytest.raises(sb.K3StoreManifestError, match="is 17 B but the manifest records"):
        launch(tmp_path, *store_flags(env, config))


def test_a_store_of_another_checkpoint_is_refused_as_stale(env, tmp_path):
    output = tmp_path / "store0"
    config = build_store(env, output)
    other, _ = env.rewrite(_declarations(env.cfg)[::-1], name="other_ckpt")
    with pytest.raises(sb.K3StoreManifestError) as excinfo:
        launch(tmp_path, *store_flags(env, config, converted=other))
    message = str(excinfo.value)
    assert "the store is stale" in message
    _assert_rebuild_command(message, env, output, converted=other)


def test_a_launch_naming_no_checkpoint_is_refused(env, tmp_path):
    config = build_store(env, tmp_path / "store0")
    with pytest.raises(sb.K3StoreManifestError, match="names no converted checkpoint"):
        launch(tmp_path, "--distributed-weight-config", config)


def test_corrupted_store_bytes_are_refused(env, tmp_path):
    output = tmp_path / "store0"
    config = build_store(env, output)
    with open(output / "store.bin", "r+b") as handle:
        handle.seek(0)
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0x80]))
    with pytest.raises(sb.K3StoreManifestError, match="does not hash to the manifest"):
        launch(tmp_path, *store_flags(env, config))


def test_another_nodes_config_is_refused(env, tmp_path):
    config = build_store(env, tmp_path / "store1", node_rank=1)
    with pytest.raises(sb.K3StoreManifestError) as excinfo:
        launch(tmp_path, *store_flags(env, config), node_rank=0)
    message = str(excinfo.value)
    assert "is node 1 of 2, but this launch is --node-rank 0" in message
    assert "--node-rank 0" in message.split("build it with:")[1]


# --------------------------------------------------------------------------- #
#  Flag rules                                                                 #
# --------------------------------------------------------------------------- #


def test_the_contradiction_is_the_first_failure(env, tmp_path):
    # --max-pool-size 0 is otherwise the very first check in validation.
    with pytest.raises(WeightSourceError, match="contradictory"):
        launch(
            tmp_path,
            "--distributed-weight-config", tmp_path / "absent.json",
            "--replicated-weights",
            "--max-pool-size", "0",
        )


@pytest.mark.parametrize(
    "model, nnodes",
    (("moonshotai/Kimi-K3", 1), ("openai/gpt-oss-120b", 2), ("zai-org/GLM-5", 2)),
)
def test_replicated_weights_outside_multinode_k3_is_refused(tmp_path, model, nnodes):
    with pytest.raises(WeightSourceError, match="only selects the weight source"):
        launch(tmp_path, "--replicated-weights", model=model, nnodes=nnodes)


def test_k3_is_the_registered_identity_not_a_substring(tmp_path):
    is_k3 = server_args_module._is_kimi_k3_model
    assert is_k3("moonshotai/Kimi-K3")
    assert not is_k3("moonshotai/Kimi-Linear-48B-A3B-Instruct")
    assert not is_k3("openai/gpt-oss-120b")
    # Not bound to kimi_k3 by the registry, so the default flip does not
    # claim it -- and the distributed store refuses it.
    assert not is_k3("someorg/kimi-k3-unregistered")
    launch(tmp_path, model="someorg/kimi-k3-unregistered")
    with pytest.raises(WeightSourceError, match="supports Kimi-K3 only"):
        launch(
            tmp_path,
            "--distributed-weight-config", Path(__file__),
            model="someorg/kimi-k3-unregistered",
        )
