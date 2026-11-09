"""The launch readiness probe: every check, and the CLI that reports them.

CPU only, and deliberately without a real node: each check reaches the
environment through one injected callable, so a wrong mount flag, an
unreadable kernel knob, a taken port, a busy GPU and -- the case this module
exists for -- a read that never returns are all driven here.

The mount and store rules themselves are NOT restated: the staging module the
server uses is loaded and called, so a mount it would refuse is refused here
identically.
"""

import importlib
import json
import socket
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batchgen.launch_readiness import (  # noqa: E402
    MAX_WATERMARK_SCALE_FACTOR,
    STAGING_HEADROOM_BYTES,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_WARN,
    LaunchReadinessError,
    LaunchReadinessProbe,
    LaunchTarget,
    StalledReadError,
    is_port_free,
    main,
    read_with_deadline,
    run_launch_readiness,
)


def _load_leaf_module(module_name):
    """Import one ``batchgen.server`` leaf without running the package init.

    ``batchgen/server/__init__.py`` imports the HTTP server and its ASGI
    stack, which the probe deliberately never pulls in.
    """
    package_name = "batchgen.server"
    previous = sys.modules.get(package_name)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return importlib.import_module(module_name)
    finally:
        if previous is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous


staging_module = _load_leaf_module("batchgen.server.weight_store_staging")
server_args_module = _load_leaf_module("batchgen.server.server_args")


# --------------------------------------------------------------------------- #
#  Fixtures: a node config, a mount table, a blocking file                    #
# --------------------------------------------------------------------------- #

STORE_BYTES = 4 * (1 << 30)


def write_node_config(tmp_path, *, staging_dir, residency="run", store=None):
    """A schema-valid node config naming ``staging_dir``.

    Only the schema half of the loader reads it, which is exactly what the
    staging check needs: which directory is staged and how many bytes land in
    it. The manifest half is covered by its own check below.
    """
    store_path = Path(store) if store is not None else tmp_path / "source" / "store.bin"
    config = {
        "node_rank": 0,
        "num_nodes": 2,
        "node_ips": ["10.0.0.1", "10.0.0.2"],
        "workers": 8,
        "worker_sharded": True,
        "store_path": str(store_path),
        "metadata_path": str(store_path.with_name("metadata.tsv")),
        "manifest_path": str(store_path.with_name("store_manifest.json")),
        "daemon_socket": str(tmp_path / "daemon.sock"),
        "summary_path": str(tmp_path / "summary.json"),
        "store_bytes": STORE_BYTES,
        "replicated_bytes": 1 << 20,
        "module_bytes": 1 << 20,
    }
    if residency is not None:
        config["store_residency"] = residency
        config["store_staging_dir"] = str(staging_dir)
    path = tmp_path / "node0.json"
    path.write_text(json.dumps(config))
    return path


def mounts(mountpoint, fstype="tmpfs", options="rw,huge=within_size,size=800g"):
    return f"tmpfs {mountpoint} {fstype} {options} 0 0\n"


class BlockingFile:
    """A file whose read never returns, like a stalled network-FS read."""

    def __init__(self, released):
        self._released = released

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _size):
        # Waited on rather than slept so the abandoned thread is released when
        # the test ends instead of outliving it.
        self._released.wait()
        return b""


def probe(target, **kwargs):
    kwargs.setdefault("staging", staging_module)
    return LaunchReadinessProbe(target, **kwargs)


def sysfs(values):
    def read_text(path):
        if path not in values:
            raise FileNotFoundError(2, "No such file or directory", path)
        value = values[path]
        if isinstance(value, Exception):
            raise value
        return value

    return read_text


# --------------------------------------------------------------------------- #
#  1. the staging tmpfs                                                       #
# --------------------------------------------------------------------------- #


def test_staging_tmpfs_passes_on_a_huge_page_tmpfs_with_room(tmp_path):
    staging_dir = tmp_path / "shm" / "stores"
    staging_dir.mkdir(parents=True)
    config = write_node_config(tmp_path, staging_dir=staging_dir)
    result = probe(
        LaunchTarget(model="m", distributed_weight_config=config, fast_init=True),
        read_mounts=lambda: mounts(str(tmp_path / "shm")),
        free_bytes_of=lambda path: STORE_BYTES + STAGING_HEADROOM_BYTES,
    ).check_staging_tmpfs()
    assert result.status == STATUS_OK
    assert "huge=within_size" in result.evidence


def test_staging_tmpfs_refuses_a_non_tmpfs_mount(tmp_path):
    staging_dir = tmp_path / "disk" / "stores"
    staging_dir.mkdir(parents=True)
    config = write_node_config(tmp_path, staging_dir=staging_dir)
    result = probe(
        LaunchTarget(model="m", distributed_weight_config=config),
        read_mounts=lambda: f"/dev/sda1 {tmp_path / 'disk'} ext4 rw 0 0\n",
    ).check_staging_tmpfs()
    assert result.status == STATUS_FAIL
    assert "not a tmpfs" in result.evidence


def test_staging_tmpfs_refuses_a_tmpfs_without_huge_pages_under_fast_init(tmp_path):
    """The exact failure that was only reported minutes into staging."""
    staging_dir = tmp_path / "shm" / "stores"
    staging_dir.mkdir(parents=True)
    config = write_node_config(tmp_path, staging_dir=staging_dir)
    result = probe(
        LaunchTarget(model="m", distributed_weight_config=config, fast_init=True),
        read_mounts=lambda: mounts(str(tmp_path / "shm"), options="rw,size=800g"),
    ).check_staging_tmpfs()
    assert result.status == STATUS_FAIL
    assert "huge=within_size" in result.evidence
    assert "mount -o remount" in result.evidence


def test_staging_tmpfs_refuses_a_mount_too_small_for_the_store(tmp_path):
    staging_dir = tmp_path / "shm" / "stores"
    staging_dir.mkdir(parents=True)
    config = write_node_config(tmp_path, staging_dir=staging_dir)
    result = probe(
        LaunchTarget(model="m", distributed_weight_config=config),
        read_mounts=lambda: mounts(str(tmp_path / "shm")),
        free_bytes_of=lambda path: STORE_BYTES,
    ).check_staging_tmpfs()
    assert result.status == STATUS_FAIL
    assert str(STORE_BYTES + STAGING_HEADROOM_BYTES) in result.evidence
    # The advice names the host-sized remount (MemTotal - 300 GiB, whole GiB).
    assert "mount -o remount," in result.evidence
    assert "huge=within_size {}".format(tmp_path / "shm") in result.evidence


def test_staging_tmpfs_sizes_the_mount_that_will_hold_a_missing_directory(tmp_path):
    """The staging directory is created by the staging step, not the probe."""
    staging_dir = tmp_path / "shm" / "not-created-yet"
    (tmp_path / "shm").mkdir()
    config = write_node_config(tmp_path, staging_dir=staging_dir)
    measured = []
    result = probe(
        LaunchTarget(model="m", distributed_weight_config=config),
        read_mounts=lambda: mounts(str(tmp_path / "shm")),
        free_bytes_of=lambda path: measured.append(path)
        or STORE_BYTES + STAGING_HEADROOM_BYTES,
    ).check_staging_tmpfs()
    assert result.status == STATUS_OK
    assert measured == [tmp_path / "shm"]
    assert not staging_dir.exists()


def test_staging_tmpfs_is_not_applicable_without_a_staged_store(tmp_path):
    config = write_node_config(tmp_path, staging_dir=None, residency=None)
    result = probe(
        LaunchTarget(model="m", distributed_weight_config=config)
    ).check_staging_tmpfs()
    assert result.status == STATUS_OK
    assert "store_path" in result.evidence

    bare = probe(LaunchTarget(model="m")).check_staging_tmpfs()
    assert bare.status == STATUS_OK


def test_an_unusable_node_config_fails_the_staging_check(tmp_path):
    broken = tmp_path / "node0.json"
    broken.write_text("{}")
    result = probe(
        LaunchTarget(model="m", distributed_weight_config=broken)
    ).check_staging_tmpfs()
    assert result.status == STATUS_FAIL
    assert "misses" in result.evidence


# --------------------------------------------------------------------------- #
#  2. vm.watermark_scale_factor                                               #
# --------------------------------------------------------------------------- #


def test_watermark_scale_factor_within_the_bound_passes():
    result = probe(
        LaunchTarget(model="m"),
        read_text=sysfs({"/proc/sys/vm/watermark_scale_factor": "10\n"}),
    ).check_watermark_scale_factor()
    assert result.status == STATUS_OK
    assert "10" in result.evidence


def test_watermark_scale_factor_above_the_bound_fails():
    result = probe(
        LaunchTarget(model="m"),
        read_text=sysfs(
            {
                "/proc/sys/vm/watermark_scale_factor": str(
                    MAX_WATERMARK_SCALE_FACTOR + 1
                )
            }
        ),
    ).check_watermark_scale_factor()
    assert result.status == STATUS_FAIL
    assert "sysctl -w vm.watermark_scale_factor" in result.evidence


def test_an_unreadable_watermark_scale_factor_only_warns():
    result = probe(
        LaunchTarget(model="m"), read_text=sysfs({})
    ).check_watermark_scale_factor()
    assert result.status == STATUS_WARN


def test_a_non_numeric_watermark_scale_factor_only_warns():
    result = probe(
        LaunchTarget(model="m"),
        read_text=sysfs({"/proc/sys/vm/watermark_scale_factor": "nonsense"}),
    ).check_watermark_scale_factor()
    assert result.status == STATUS_WARN


# --------------------------------------------------------------------------- #
#  3. THP for shared memory                                                   #
# --------------------------------------------------------------------------- #

_THP_PATH = "/sys/kernel/mm/transparent_hugepage/shmem_enabled"


@pytest.mark.parametrize("line", ["always [within_size] advise never", "[always] never"])
def test_thp_shmem_backed_by_huge_pages_passes(line):
    result = probe(
        LaunchTarget(model="m"), read_text=sysfs({_THP_PATH: line})
    ).check_thp_shmem()
    assert result.status == STATUS_OK


def test_thp_shmem_off_only_warns_without_fast_init():
    result = probe(
        LaunchTarget(model="m"),
        read_text=sysfs({_THP_PATH: "always within_size advise [never]"}),
    ).check_thp_shmem()
    assert result.status == STATUS_WARN


def test_thp_shmem_off_fails_under_fast_init():
    """--fast-init is refused by the server itself, so the probe refuses it."""
    result = probe(
        LaunchTarget(model="m", fast_init=True),
        read_text=sysfs({_THP_PATH: "always within_size advise [never]"}),
    ).check_thp_shmem()
    assert result.status == STATUS_FAIL
    assert "echo always" in result.evidence


def test_an_unreadable_thp_knob_fails_only_under_fast_init():
    assert (
        probe(LaunchTarget(model="m"), read_text=sysfs({})).check_thp_shmem().status
        == STATUS_WARN
    )
    assert (
        probe(
            LaunchTarget(model="m", fast_init=True), read_text=sysfs({})
        ).check_thp_shmem().status
        == STATUS_FAIL
    )


# --------------------------------------------------------------------------- #
#  4. the ports a launch binds                                                #
# --------------------------------------------------------------------------- #


def test_the_probe_port_check_matches_the_servers_own():
    """The probe restates server_args.is_port_available; it must not drift."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert is_port_free(taken) is server_args_module.is_port_available(taken)
        assert is_port_free(taken) is False
    free = 0
    assert is_port_free(free) is server_args_module.is_port_available(free)


def test_free_launch_ports_pass():
    result = probe(
        LaunchTarget(model="m", listen_port=10900),
        port_probe=lambda port: True,
        read_text=sysfs({}),
    ).check_ports()
    assert result.status == STATUS_OK
    assert "listen=10900" in result.evidence


def test_a_port_already_in_use_fails():
    result = probe(
        LaunchTarget(model="m", listen_port=10900),
        port_probe=lambda port: False,
        read_text=sysfs({}),
    ).check_ports()
    assert result.status == STATUS_FAIL
    assert "listen port 10900 is already in use" in result.evidence


def test_a_port_inside_the_ephemeral_range_warns():
    result = probe(
        LaunchTarget(model="m", listen_port=40000),
        port_probe=lambda port: True,
        read_text=sysfs({"/proc/sys/net/ipv4/ip_local_port_range": "32768\t60999\n"}),
    ).check_ports()
    assert result.status == STATUS_WARN
    assert "ip_local_reserved_ports" in result.evidence


def test_a_reserved_port_inside_the_ephemeral_range_passes():
    result = probe(
        LaunchTarget(model="m", listen_port=40000),
        port_probe=lambda port: True,
        read_text=sysfs(
            {
                "/proc/sys/net/ipv4/ip_local_port_range": "32768 60999",
                "/proc/sys/net/ipv4/ip_local_reserved_ports": "39000-40100,50000",
            }
        ),
    ).check_ports()
    assert result.status == STATUS_OK


def test_the_rendezvous_ports_are_derived_on_the_hosting_node_only():
    checked = []

    def record(port):
        checked.append(port)
        return True

    target = LaunchTarget(
        model="m",
        listen_port=10900,
        dist_init_addr="10.0.0.1:20000",
        pynccl_port_base=20003,
        nnodes=2,
        node_rank=0,
        world_size=16,
    )
    probe(target, port_probe=record, read_text=sysfs({})).check_ports()
    # listen, dist, the derived control store (dist + 1), and pynccl.
    assert checked == [10900, 20000, 20001, 20003]

    checked.clear()
    peer = LaunchTarget(**{**target.__dict__, "node_rank": 1})
    probe(peer, port_probe=record, read_text=sysfs({})).check_ports()
    assert checked == [10900]


# --------------------------------------------------------------------------- #
#  5. bounded reads of every path a launch names                              #
# --------------------------------------------------------------------------- #


def test_read_with_deadline_returns_the_bytes_it_read(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"0123456789")
    assert read_with_deadline(path, size=4) == 4


def test_read_with_deadline_reports_a_read_that_never_returns(tmp_path):
    released = threading.Event()
    try:
        with pytest.raises(StalledReadError) as refusal:
            read_with_deadline(
                tmp_path / "store.bin",
                deadline_s=0.05,
                opener=lambda path, mode: BlockingFile(released),
            )
        assert "read stalled" in str(refusal.value)
    finally:
        released.set()


def test_every_launch_named_path_is_read(tmp_path):
    cache = tmp_path / "cache"
    converted = cache / "converted_ckpt"
    converted.mkdir(parents=True)
    (cache / "config.json").write_bytes(b"{}")
    (converted / "a-00001.json").write_bytes(b"{}")
    (converted / "b-00002.json").write_bytes(b"{}")
    store = tmp_path / "source"
    store.mkdir()
    (store / "store.bin").write_bytes(b"\0" * 2048)
    config = write_node_config(
        tmp_path, staging_dir=tmp_path / "shm", store=store / "store.bin"
    )
    result = probe(
        LaunchTarget(model="m", cache_dir=cache, distributed_weight_config=config)
    ).check_path_reads()
    assert result.status == STATUS_OK
    # The model config, the FIRST sidecar in sorted order, and the store.
    assert "a-00001.json" in result.evidence
    assert "b-00002.json" not in result.evidence
    assert "store.bin (2048 B read)" in result.evidence


def test_a_missing_launch_named_path_fails(tmp_path):
    result = probe(
        LaunchTarget(model="m", cache_dir=tmp_path / "absent")
    ).check_path_reads()
    assert result.status == STATUS_FAIL
    assert "cannot be read" in result.evidence


def test_a_converted_checkpoint_without_a_sidecar_fails(tmp_path):
    converted = tmp_path / "converted"
    converted.mkdir()
    result = probe(
        LaunchTarget(model="m", converted_ckpt_dir=converted)
    ).check_path_reads()
    assert result.status == STATUS_FAIL


def test_a_stalled_store_read_fails_the_path_check(tmp_path):
    """The 2026-11-06 failure mode: one node's filesystem stops answering and
    every rank spins with nothing in the log."""
    released = threading.Event()
    config = write_node_config(tmp_path, staging_dir=tmp_path / "shm")
    try:
        result = probe(
            LaunchTarget(model="m", distributed_weight_config=config),
            read_deadline_s=0.05,
            opener=lambda path, mode: BlockingFile(released),
        ).check_path_reads()
        assert result.status == STATUS_FAIL
        assert "read stalled" in result.evidence
        assert "weight store" in result.evidence
    finally:
        released.set()


# --------------------------------------------------------------------------- #
#  6. the GPUs a launch takes                                                 #
# --------------------------------------------------------------------------- #


def test_idle_gpus_pass():
    result = probe(
        LaunchTarget(model="m"), run_nvidia_smi=lambda argv, timeout: "\n"
    ).check_gpu_idle()
    assert result.status == STATUS_OK


def test_resident_compute_processes_fail():
    result = probe(
        LaunchTarget(model="m"),
        run_nvidia_smi=lambda argv, timeout: "41, 512 MiB\n77, 64 MiB\n",
        pid_visible=lambda pid: True,
    ).check_gpu_idle()
    assert result.status == STATUS_FAIL
    assert "2 compute process(es)" in result.evidence


def test_contexts_outside_this_pid_namespace_only_warn():
    """Host PIDs this container cannot see (another container, or a context
    the driver never reclaimed) cannot be stopped from here and run no work:
    they cost memory headroom, not the launch."""
    result = probe(
        LaunchTarget(model="m"),
        run_nvidia_smi=lambda argv, timeout: "191475, 474 MiB\n191476, 474 MiB\n",
        pid_visible=lambda pid: False,
    ).check_gpu_idle()
    assert result.status == STATUS_WARN
    assert "2 GPU context(s)" in result.evidence
    assert "474 MiB" in result.evidence


def test_a_visible_process_fails_even_beside_foreign_contexts():
    result = probe(
        LaunchTarget(model="m"),
        run_nvidia_smi=lambda argv, timeout: "41, 512 MiB\n191475, 474 MiB\n",
        pid_visible=lambda pid: pid == 41,
    ).check_gpu_idle()
    assert result.status == STATUS_FAIL
    assert "1 compute process(es)" in result.evidence
    assert "191475" not in result.evidence


def test_a_missing_nvidia_smi_only_warns():
    def missing(argv, timeout):
        raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")

    result = probe(LaunchTarget(model="m"), run_nvidia_smi=missing).check_gpu_idle()
    assert result.status == STATUS_WARN


def test_an_unresponsive_driver_fails():
    import subprocess

    def wedged(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    result = probe(LaunchTarget(model="m"), run_nvidia_smi=wedged).check_gpu_idle()
    assert result.status == STATUS_FAIL


# --------------------------------------------------------------------------- #
#  7. the store config itself                                                 #
# --------------------------------------------------------------------------- #


def test_a_store_without_a_completion_manifest_fails(tmp_path):
    config = write_node_config(tmp_path, staging_dir=tmp_path / "shm")
    result = probe(
        LaunchTarget(model="m", distributed_weight_config=config)
    ).check_weight_store_config()
    assert result.status == STATUS_FAIL


def test_a_store_that_passes_the_manifest_gate_is_reported(tmp_path):
    config = write_node_config(tmp_path, staging_dir=tmp_path / "shm")
    accepting = types.SimpleNamespace(
        read_distributed_weight_config=staging_module.read_distributed_weight_config,
        load_distributed_weight_config=lambda path: None,
    )
    result = LaunchReadinessProbe(
        LaunchTarget(model="m", distributed_weight_config=config),
        staging=accepting,
    ).check_weight_store_config()
    assert result.status == STATUS_OK
    assert "node 0 of 2" in result.evidence
    assert str(STORE_BYTES) in result.evidence


def test_no_store_config_is_reported_as_such():
    result = probe(LaunchTarget(model="m")).check_weight_store_config()
    assert result.status == STATUS_OK


# --------------------------------------------------------------------------- #
#  the report, the gate, and the CLI                                          #
# --------------------------------------------------------------------------- #


def test_the_report_names_every_check_and_every_failure():
    report = probe(
        LaunchTarget(model="m", listen_port=10900),
        read_text=sysfs({}),
        port_probe=lambda port: False,
        run_nvidia_smi=lambda argv, timeout: "",
    ).run()
    assert not report.ok
    assert [check.name for check in report.checks] == [
        "weight-store-config",
        "staging-tmpfs",
        "watermark-scale-factor",
        "thp-shmem",
        "ports",
        "path-reads",
        "gpu-idle",
    ]
    assert [check.name for check in report.failures] == ["ports"]
    assert "ports: listen port 10900 is already in use" in report.message()
    assert report.to_json()["ok"] is False
    assert len(report.to_json()["checks"]) == 7


def test_the_startup_gate_refuses_a_failing_node(caplog):
    server_args = types.SimpleNamespace(
        model="m",
        listen_port=10900,
        distributed_weight_config=None,
        cache_dir=None,
        converted_ckpt_dir=None,
        fast_init=False,
        dist_init_addr=None,
        restart_control_addr=None,
        pynccl_port_base=None,
        nnodes=1,
        node_rank=0,
        world_size=1,
    )
    with pytest.raises(LaunchReadinessError) as refusal:
        run_launch_readiness(
            server_args,
            staging=staging_module,
            read_text=sysfs({}),
            port_probe=lambda port: False,
            run_nvidia_smi=lambda argv, timeout: "",
        )
    assert "already in use" in str(refusal.value)


def test_the_startup_gate_passes_a_ready_node():
    server_args = types.SimpleNamespace(model="m", listen_port=None, nnodes=1)
    report = run_launch_readiness(
        server_args,
        staging=staging_module,
        read_text=sysfs(
            {
                "/proc/sys/vm/watermark_scale_factor": "10",
                _THP_PATH: "[always] never",
            }
        ),
        run_nvidia_smi=lambda argv, timeout: "",
    )
    assert report.ok
    assert not report.warnings


def test_the_cli_exits_zero_and_prints_json_when_every_check_passes(capsys):
    assert main(["--model", "m", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert {check["name"] for check in payload["checks"]} >= {"ports", "path-reads"}


def test_the_cli_exits_one_on_a_failure(tmp_path, capsys):
    broken = tmp_path / "node0.json"
    broken.write_text("{}")
    assert (
        main(["--model", "m", "--distributed-weight-config", str(broken)]) == 1
    )
    captured = capsys.readouterr()
    assert "launch readiness failed" in captured.err
    assert "[fail] weight-store-config" in captured.out
