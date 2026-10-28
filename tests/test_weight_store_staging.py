"""Server-owned staging of the Kimi-K3 distributed weight store.

Hermetic: real store-builder output from a mini checkpoint, temporary
directories standing in for the tmpfs staging directory, an injected mount
table, and real ``flock`` holders in child processes. No GPU, no tmpfs.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batchgen.models.moonshotai.kimi_linear import store_builder as sb  # noqa: E402
from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (  # noqa: E402
    DEFAULT_STORE_STAGING_DIR,
    load_distributed_weight_config,
    read_distributed_weight_config,
)
from tests.test_kimi_k3_store_builder import Env  # noqa: E402


def _load_staging_module():
    """Import ``batchgen.server.weight_store_staging`` without the server
    package's HTTP-stack ``__init__`` (as tests/test_runtime_locks.py does)."""
    package_name = "batchgen.server"
    module_name = "batchgen.server.weight_store_staging"
    previous_package = sys.modules.get(package_name)
    previous_module = sys.modules.pop(module_name, None)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
        if previous_package is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous_package


st = _load_staging_module()


def _mounts(path: Path, fstype: str = "tmpfs", options: str = "rw,huge=within_size") -> str:
    return (
        "sysfs /sys sysfs rw 0 0\n"
        "overlay / overlay rw 0 0\n"
        f"{fstype} {os.path.realpath(path)} {fstype} {options} 0 0\n"
    )


class Deployment:
    """One node's published store plus a fake tmpfs staging directory."""

    def __init__(
        self, tmp_path: Path, monkeypatch, *, residency="run", node_rank=0, env=None
    ):
        # Small segments, so the copy is verified in several pieces.
        monkeypatch.setattr(sb, "SEGMENT_TARGET_BYTES", 64)
        self.env = env or Env(tmp_path)
        self.shm = tmp_path / "shm"
        self.shm.mkdir(exist_ok=True)
        self.staging_dir = self.shm / "batchgen-weight-stores"
        self.output = tmp_path / f"store-node{node_rank}-{residency}"
        self.env.build(
            self.output,
            node_rank,
            extra=(
                "--store-residency", residency,
                "--store-staging-dir", str(self.staging_dir),
            ),
        )
        self.config_path = self.output / f"node{node_rank}.json"
        self.runtime_dir = tmp_path / f"run-node{node_rank}-{residency}"
        self.runtime_dir.mkdir()
        self.interleaved = 0

    def interleave(self):
        deployment = self

        class _Recorder:
            def __enter__(self):
                deployment.interleaved += 1

            def __exit__(self, *exc):
                return False

        return _Recorder()

    def stage(self, **overrides):
        kwargs = dict(
            fast_init=False,
            read_mounts=lambda: _mounts(self.shm),
            interleave=self.interleave,
        )
        kwargs.update(overrides)
        return st.stage_distributed_weight_store(
            self.config_path, self.runtime_dir, **kwargs
        )

    def staged_id(self):
        manifest = (self.output / sb.MANIFEST_FILE_NAME).read_bytes()
        node_rank = json.loads(self.config_path.read_text())["node_rank"]
        return st.staged_store_id(node_rank, manifest)

    def hide_source_store(self):
        """Make the source store.bin unreadable by any path: move it away."""
        source = self.output / sb.STORE_FILE_NAME
        source.rename(self.output / "store.bin.moved-away")


def _hold_lock(path: Path, *, mkdir=True):
    """A child process holding ``path`` LOCK_SH until its stdin closes."""
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)\n"
            "fcntl.flock(fd, fcntl.LOCK_SH)\n"
            "print('held', flush=True)\n"
            "sys.stdin.read()\n",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout.readline().strip() == "held"
    return child


def _drop(child):
    child.stdin.close()
    child.wait(timeout=30)


def _fake_staged_dir(staging_dir: Path, name: str, residency: str = "resident"):
    directory = staging_dir / name
    directory.mkdir(parents=True)
    (directory / st.USE_LOCK_NAME).touch()
    (directory / "store.bin").write_bytes(b"x" * 32)
    (directory / st.RESIDENCY_FILE_NAME).write_text(residency + "\n")
    (directory / st.COMPLETE_FILE_NAME).write_text(name + "\n")
    return directory


# --------------------------------------------------------------------------- #
#  Config schema and builder                                                   #
# --------------------------------------------------------------------------- #


def _schema_config(tmp_path, **extra):
    config = {
        "node_rank": 0,
        "node_ips": ["10.0.0.1", "10.0.0.2"],
        "workers": 8,
        "worker_sharded": True,
        "store_path": "/stores/store.bin",
        "metadata_path": "/stores/metadata.tsv",
        "daemon_socket": "/stores/daemon.sock",
        "summary_path": "/stores/summary.json",
        "store_bytes": 1,
        "replicated_bytes": 1,
        "module_bytes": 1,
    }
    config.update(extra)
    path = tmp_path / "node0.json"
    path.write_text(json.dumps(config))
    return path


def test_config_schema_residency_and_staging_dir(tmp_path):
    legacy = read_distributed_weight_config(_schema_config(tmp_path))
    assert legacy["store_residency"] is None
    assert legacy["store_staging_dir"] is None
    assert legacy["store_path"] == "/stores/store.bin"

    run = read_distributed_weight_config(_schema_config(tmp_path, store_residency="run"))
    assert run["store_residency"] == "run"
    assert run["store_staging_dir"] == DEFAULT_STORE_STAGING_DIR == (
        "/dev/shm/batchgen-weight-stores"
    )
    resident = read_distributed_weight_config(
        _schema_config(tmp_path, store_residency="resident", store_staging_dir="/mnt/t")
    )
    assert (resident["store_residency"], resident["store_staging_dir"]) == (
        "resident", "/mnt/t",
    )

    with pytest.raises(ValueError, match="store_residency must be one of"):
        read_distributed_weight_config(_schema_config(tmp_path, store_residency="forever"))
    with pytest.raises(ValueError, match="store_staging_dir is set but store_residency is not"):
        read_distributed_weight_config(_schema_config(tmp_path, store_staging_dir="/mnt/t"))
    with pytest.raises(ValueError, match="must be an absolute path"):
        read_distributed_weight_config(
            _schema_config(tmp_path, store_residency="run", store_staging_dir="rel/dir")
        )


def test_builder_writes_residency_and_rebuild_command_keeps_it(tmp_path):
    env = Env(tmp_path)
    default = tmp_path / "default"
    env.build(default, 0)
    config = json.loads((default / "node0.json").read_text())
    assert config["store_residency"] == "run"
    assert config["store_staging_dir"] == DEFAULT_STORE_STAGING_DIR

    chosen = tmp_path / "chosen"
    env.build(
        chosen, 0, extra=("--store-residency", "resident", "--store-staging-dir", "/mnt/t")
    )
    config = json.loads((chosen / "node0.json").read_text())
    assert (config["store_residency"], config["store_staging_dir"]) == ("resident", "/mnt/t")
    # The builder output location is unchanged: it is the source.
    assert config["store_path"] == str(chosen / sb.STORE_FILE_NAME)
    command = sb.rebuild_command(chosen / "node0.json", config)
    assert "--store-residency resident" in command
    assert "--store-staging-dir /mnt/t" in command

    with pytest.raises(sb.K3StoreBuilderError, match="must be an absolute path"):
        env.build(tmp_path / "bad", 0, extra=("--store-staging-dir", "relative"))


# --------------------------------------------------------------------------- #
#  Mount gate                                                                  #
# --------------------------------------------------------------------------- #


def test_non_tmpfs_staging_dir_refused(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    with pytest.raises(st.WeightStoreStagingError, match="not a tmpfs") as excinfo:
        deployment.stage(read_mounts=lambda: _mounts(deployment.shm, fstype="ext4", options="rw"))
    assert f"ext4 mount {os.path.realpath(deployment.shm)}" in str(excinfo.value)
    assert not deployment.staging_dir.exists()
    assert deployment.interleaved == 0


def test_fast_init_requires_huge_tmpfs_and_names_the_remount_command(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    plain = lambda: _mounts(deployment.shm, options="rw,nosuid,size=100g")  # noqa: E731
    mountpoint = os.path.realpath(deployment.shm)
    with pytest.raises(st.WeightStoreStagingError) as excinfo:
        deployment.stage(fast_init=True, read_mounts=plain)
    message = str(excinfo.value)
    assert "it is huge=never" in message
    assert message.endswith(f"mount -o remount,huge=within_size {mountpoint}")
    assert not deployment.staging_dir.exists()

    advise = lambda: _mounts(deployment.shm, options="rw,huge=advise")  # noqa: E731
    with pytest.raises(st.WeightStoreStagingError, match="it is huge=advise"):
        deployment.stage(fast_init=True, read_mounts=advise)

    # Without --fast-init a plain tmpfs is enough; with it, always is too.
    deployment.stage(fast_init=False, read_mounts=plain).release()
    always = lambda: _mounts(deployment.shm, options="rw,huge=always")  # noqa: E731
    deployment.stage(fast_init=True, read_mounts=always).release()


def test_find_mount_picks_the_innermost_and_topmost_mount(tmp_path):
    table = (
        "overlay / overlay rw 0 0\n"
        "tmpfs /dev/shm tmpfs rw,huge=never 0 0\n"
        "tmpfs /dev/shm tmpfs rw,huge=within_size 0 0\n"
        "tmpfs /dev/shm/x\\040y tmpfs rw 0 0\n"
    )
    entry = st.find_mount("/dev/shm/batchgen-weight-stores", table)
    assert (entry.mountpoint, entry.option("huge")) == ("/dev/shm", "within_size")
    assert st.find_mount("/dev/shm/x y/z", table).mountpoint == "/dev/shm/x y"
    assert st.find_mount("/dev/shmother", table).mountpoint == "/"


# --------------------------------------------------------------------------- #
#  Stage, reuse, residency                                                     #
# --------------------------------------------------------------------------- #


def test_stage_then_reuse_without_reading_the_source(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch, residency="resident")
    source_bytes = (deployment.output / sb.STORE_FILE_NAME).read_bytes()

    first = deployment.stage()
    directory = deployment.staging_dir / deployment.staged_id()
    assert first.directory == directory and not first.reused
    assert deployment.interleaved == 1
    assert (directory / sb.STORE_FILE_NAME).read_bytes() == source_bytes
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [
            sb.STORE_FILE_NAME, sb.METADATA_FILE_NAME, sb.MANIFEST_FILE_NAME,
            st.RESIDENCY_FILE_NAME, st.COMPLETE_FILE_NAME, st.USE_LOCK_NAME,
        ]
    )
    assert (directory / st.RESIDENCY_FILE_NAME).read_text() == "resident\n"
    first.release()
    assert (directory / st.COMPLETE_FILE_NAME).is_file()

    deployment.hide_source_store()
    second = deployment.stage()
    assert second.reused and second.directory == directory
    assert deployment.interleaved == 1  # no second copy
    assert (directory / sb.STORE_FILE_NAME).read_bytes() == source_bytes
    config = load_distributed_weight_config(second.effective_config_path)
    assert config["store_path"] == str(directory / sb.STORE_FILE_NAME)
    second.release()


def test_preflight_samples_the_staged_copy_on_a_resident_restart(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch, residency="resident")
    assert st.complete_staged_store(deployment.config_path) is None
    deployment.stage().release()
    staged = st.complete_staged_store(deployment.config_path)
    assert staged == deployment.staging_dir / deployment.staged_id() / sb.STORE_FILE_NAME

    def preflight(**extra):
        return sb.verify_published_store(
            deployment.config_path,
            converted_ckpt=deployment.env.ckpt,
            cache_dir=deployment.env.model_dir,
            **extra,
        )

    # The source keeps its size (the cheap gate stats it) but its bytes are
    # unreadable: only a preflight that samples the staged copy passes.
    source = deployment.output / sb.STORE_FILE_NAME
    source.chmod(0)
    try:
        with pytest.raises(sb.K3StoreManifestError):
            preflight()
        assert preflight(spot_check_store_path=staged)["spot_checked_segments"]
    finally:
        source.chmod(0o644)

    # A corrupt staged copy is refused and named; rebuilding the source would
    # not fix it.
    with open(staged, "r+b") as handle:
        handle.write(b"\xff" * 8)
    with pytest.raises(sb.K3StoreManifestError, match="staged copy"):
        preflight(spot_check_store_path=staged)


def test_effective_config_passes_the_manifest_gate_and_names_the_copy(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    staged = deployment.stage()
    directory = staged.directory
    assert staged.effective_config_path.parent == deployment.runtime_dir
    effective = json.loads(staged.effective_config_path.read_text())
    source = json.loads(deployment.config_path.read_text())
    assert effective["store_path"] == str(directory / sb.STORE_FILE_NAME)
    assert effective["metadata_path"] == str(directory / sb.METADATA_FILE_NAME)
    assert effective["manifest_path"] == str(directory / sb.MANIFEST_FILE_NAME)
    assert effective["staged_from"] == str(deployment.config_path)
    unchanged = set(source) - {"store_path", "metadata_path", "manifest_path"}
    assert {key: effective[key] for key in unchanged} == {key: source[key] for key in unchanged}
    load_distributed_weight_config(staged.effective_config_path)
    # The id is the manifest's own content hash.
    assert directory.name == deployment.staged_id()
    assert (directory / sb.MANIFEST_FILE_NAME).read_bytes() == (
        deployment.output / sb.MANIFEST_FILE_NAME
    ).read_bytes()

    # A refusal of the staged copy names the builder's output, not the run dir.
    (directory / sb.STORE_FILE_NAME).write_bytes(b"short")
    with pytest.raises(sb.K3StoreManifestError) as excinfo:
        load_distributed_weight_config(staged.effective_config_path)
    assert f"--output-dir {deployment.output} " in str(excinfo.value)
    staged.release()


def test_run_copy_is_removed_on_stop_and_resident_copy_is_kept(tmp_path, monkeypatch):
    run = Deployment(tmp_path, monkeypatch, residency="run")
    staged = run.stage()
    assert staged.directory.is_dir()
    staged.release()
    assert not staged.directory.exists()
    assert list(run.staging_dir.iterdir()) == []
    # The source is never touched.
    load_distributed_weight_config(run.config_path)

    resident = Deployment(
        tmp_path, monkeypatch, residency="resident", node_rank=1, env=run.env
    )
    staged = resident.stage()
    staged.release()
    assert (staged.directory / st.COMPLETE_FILE_NAME).is_file()
    assert (staged.directory / st.RESIDENCY_FILE_NAME).read_text() == "resident\n"


def test_run_copy_mapped_by_another_server_is_left_to_it(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch, residency="run")
    first = deployment.stage()
    holder = _hold_lock(first.directory / st.USE_LOCK_NAME, mkdir=False)
    try:
        first.release()
        assert (first.directory / st.COMPLETE_FILE_NAME).is_file()
    finally:
        _drop(holder)
    second = deployment.stage()
    assert second.reused
    second.release()
    assert not second.directory.exists()


# --------------------------------------------------------------------------- #
#  Sweep and crash leftovers                                                   #
# --------------------------------------------------------------------------- #


def test_sweep_removes_idle_other_ids_but_never_a_locked_one(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    idle_resident = _fake_staged_dir(deployment.staging_dir, "node0-" + "a" * 16)
    idle_run = _fake_staged_dir(deployment.staging_dir, "node0-" + "b" * 16, "run")
    no_lock = deployment.staging_dir / ("node1-" + "c" * 16)
    no_lock.mkdir()
    (no_lock / "store.bin.partial").write_bytes(b"y")
    removing = deployment.staging_dir / (st.REMOVING_PREFIX + "node0-" + "d" * 16)
    removing.mkdir()
    in_use = _fake_staged_dir(deployment.staging_dir, "node1-" + "e" * 16)
    holder = _hold_lock(in_use / st.USE_LOCK_NAME, mkdir=False)
    try:
        staged = deployment.stage()
        names = sorted(path.name for path in deployment.staging_dir.iterdir())
        assert names == sorted([in_use.name, staged.directory.name])
        assert not idle_resident.exists() and not idle_run.exists()
        assert (in_use / st.COMPLETE_FILE_NAME).is_file()
        staged.release()
    finally:
        _drop(holder)
    # Once its holder is gone it is idle, and the next start sweeps it.
    deployment.stage().release()
    assert not in_use.exists()


_STAGE_AND_HANG = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, sys.argv[1])
    from tests.test_weight_store_staging import st
    from batchgen.models.moonshotai.kimi_linear import store_builder as sb
    sb.SEGMENT_TARGET_BYTES = 64
    staged = st.stage_distributed_weight_store(
        sys.argv[2], sys.argv[3], fast_init=False, read_mounts=lambda: sys.argv[4]
    )
    print("staged", staged.directory, flush=True)
    time.sleep(600)
    """
)


def _stage_then_kill9(deployment) -> Path:
    child = subprocess.Popen(
        [
            sys.executable, "-c", _STAGE_AND_HANG, str(ROOT),
            str(deployment.config_path), str(deployment.runtime_dir),
            _mounts(deployment.shm),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = child.stdout.readline().split()
        assert line[:1] == ["staged"], line
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
    assert child.returncode == -signal.SIGKILL
    return Path(line[1])


def test_kill9_leftover_is_reused_for_its_id_and_swept_for_another(tmp_path, monkeypatch):
    node0 = Deployment(tmp_path, monkeypatch, residency="run")
    leftover = _stage_then_kill9(node0)
    assert (leftover / st.COMPLETE_FILE_NAME).is_file()

    # Same build: the run copy a killed server left is valid by construction.
    node0.hide_source_store()
    staged = node0.stage()
    assert staged.reused and staged.directory == leftover
    assert node0.interleaved == 0
    staged._lock_fd, fd = None, staged._lock_fd
    os.close(fd)  # this server dies too, without its stop path

    # Another build: the idle leftover is swept.
    node1 = Deployment(
        tmp_path, monkeypatch, residency="run", node_rank=1, env=node0.env
    )
    other = node1.stage()
    assert not leftover.exists()
    assert [path.name for path in node1.staging_dir.iterdir()] == [other.directory.name]
    other.release()


_DIE_MID_COPY = textwrap.dedent(
    """
    import os, signal, sys
    sys.path.insert(0, sys.argv[1])
    from tests.test_weight_store_staging import st
    from batchgen.models.moonshotai.kimi_linear import store_builder as sb
    sb.SEGMENT_TARGET_BYTES = 64
    copied = []
    real = st.hash_store_segment

    def hash_then_die(*args, **kwargs):
        if len(copied) == 2:
            os.kill(os.getpid(), signal.SIGKILL)
        copied.append(1)
        return real(*args, **kwargs)

    st.hash_store_segment = hash_then_die
    st.stage_distributed_weight_store(
        sys.argv[2], sys.argv[3], fast_init=False, read_mounts=lambda: sys.argv[4]
    )
    """
)


def test_kill9_mid_copy_leaves_no_complete_and_is_restaged(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    child = subprocess.run(
        [
            sys.executable, "-c", _DIE_MID_COPY, str(ROOT),
            str(deployment.config_path), str(deployment.runtime_dir),
            _mounts(deployment.shm),
        ],
        timeout=120,
    )
    assert child.returncode == -signal.SIGKILL
    directory = deployment.staging_dir / deployment.staged_id()
    assert (directory / st.PARTIAL_STORE_NAME).is_file()
    assert not (directory / st.COMPLETE_FILE_NAME).exists()

    staged = deployment.stage()
    assert not staged.reused and deployment.interleaved == 1
    assert (directory / sb.STORE_FILE_NAME).read_bytes() == (
        deployment.output / sb.STORE_FILE_NAME
    ).read_bytes()
    staged.release()


def test_partial_copy_without_complete_is_restaged(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    directory = deployment.staging_dir / deployment.staged_id()
    directory.mkdir(parents=True)
    source = (deployment.output / sb.STORE_FILE_NAME).read_bytes()
    (directory / st.PARTIAL_STORE_NAME).write_bytes(source[:10])
    (directory / sb.STORE_FILE_NAME).write_bytes(b"\0" * len(source))
    (directory / sb.MANIFEST_FILE_NAME).write_bytes(
        (deployment.output / sb.MANIFEST_FILE_NAME).read_bytes()
    )
    (directory / sb.METADATA_FILE_NAME).write_bytes(
        (deployment.output / sb.METADATA_FILE_NAME).read_bytes()
    )

    staged = deployment.stage()
    assert not staged.reused and deployment.interleaved == 1
    assert (directory / sb.STORE_FILE_NAME).read_bytes() == source
    assert not (directory / st.PARTIAL_STORE_NAME).exists()
    assert (directory / st.COMPLETE_FILE_NAME).is_file()
    staged.release()


def test_complete_copy_that_fails_the_gate_is_restaged(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch, residency="resident")
    deployment.stage().release()
    directory = deployment.staging_dir / deployment.staged_id()
    (directory / sb.STORE_FILE_NAME).write_bytes(b"truncated")
    staged = deployment.stage()
    assert not staged.reused and deployment.interleaved == 2
    assert (directory / sb.STORE_FILE_NAME).read_bytes() == (
        deployment.output / sb.STORE_FILE_NAME
    ).read_bytes()
    staged.release()


# --------------------------------------------------------------------------- #
#  Refusals                                                                    #
# --------------------------------------------------------------------------- #


def test_corrupted_source_segment_is_refused_with_no_complete(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    manifest = json.loads((deployment.output / sb.MANIFEST_FILE_NAME).read_text())
    segment = manifest["store"]["segments"][2]
    with open(deployment.output / sb.STORE_FILE_NAME, "r+b") as handle:
        handle.seek(segment["offset"] + 1)
        byte = handle.read(1)
        handle.seek(segment["offset"] + 1)
        handle.write(bytes([byte[0] ^ 0xFF]))
    # Size and metadata are intact: only the copy's hashing can see it.
    load_distributed_weight_config(deployment.config_path)

    with pytest.raises(st.WeightStoreStagingError) as excinfo:
        deployment.stage()
    message = str(excinfo.value)
    assert "segment 2 of the source weight store" in message
    assert f"Rebuild it with: python -m {sb.BUILDER_MODULE}" in message
    assert f"--output-dir {deployment.output} " in message
    assert list(deployment.staging_dir.iterdir()) == []
    assert not list(deployment.staging_dir.rglob(st.COMPLETE_FILE_NAME))


def _corrupt_segment(deployment, index):
    manifest = json.loads((deployment.output / sb.MANIFEST_FILE_NAME).read_text())
    segment = manifest["store"]["segments"][index]
    with open(deployment.output / sb.STORE_FILE_NAME, "r+b") as handle:
        handle.seek(segment["offset"] + 1)
        byte = handle.read(1)
        handle.seek(segment["offset"] + 1)
        handle.write(bytes([byte[0] ^ 0xFF]))


def test_refusal_names_the_first_bad_segment_whatever_finishes_first(
    tmp_path, monkeypatch
):
    deployment = Deployment(tmp_path, monkeypatch)
    _corrupt_segment(deployment, 1)
    _corrupt_segment(deployment, 3)
    import threading
    import time as _time

    real = st.hash_store_segment
    lock = threading.Lock()
    started = []
    segment_3_done = threading.Event()

    def slow_low_segments(descriptor, spans, *args):
        # Workers take segments in submission order: the second call is
        # segment 1, and it finishes only after segment 3 has hashed (badly).
        with lock:
            started.append(None)
            call = len(started)
        if call == 2:
            assert segment_3_done.wait(10)
            _time.sleep(0.05)
        digest = real(descriptor, spans, *args)
        if call == 4:
            segment_3_done.set()
        return digest

    monkeypatch.setattr(st, "hash_store_segment", slow_low_segments)
    with pytest.raises(st.WeightStoreStagingError) as excinfo:
        deployment.stage(copy_workers=4)
    assert "segment 1 of the source weight store" in str(excinfo.value)
    assert not list(deployment.staging_dir.rglob(st.COMPLETE_FILE_NAME))


def test_copy_threads_run_inside_the_interleave_policy(tmp_path, monkeypatch):
    import threading

    deployment = Deployment(tmp_path, monkeypatch)
    inside = threading.Event()

    @contextlib.contextmanager
    def interleave():
        inside.set()
        try:
            yield
        finally:
            inside.clear()

    real = st.hash_store_segment
    calls = []

    def recording(descriptor, spans, *args):
        calls.append((threading.current_thread().name, inside.is_set()))
        return real(descriptor, spans, *args)

    monkeypatch.setattr(st, "hash_store_segment", recording)
    staged = deployment.stage(interleave=interleave, copy_workers=4)
    try:
        segments = json.loads(
            (deployment.output / sb.MANIFEST_FILE_NAME).read_text()
        )["store"]["segments"]
        assert len(calls) == len(segments) > 4
        assert all(within for _, within in calls)
        names = {name for name, _ in calls}
        assert all(name.startswith("weight-store-copy") for name in names)
        assert 1 < len(names) <= 4
        staged_bytes = (staged.directory / sb.STORE_FILE_NAME).read_bytes()
        assert staged_bytes == (deployment.output / sb.STORE_FILE_NAME).read_bytes()
    finally:
        staged.release()


def test_free_space_shortfall_is_refused_before_copying(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    store_bytes = (deployment.output / sb.STORE_FILE_NAME).stat().st_size
    seen = []

    def free(path):
        seen.append(path)
        return store_bytes - 1

    with pytest.raises(st.WeightStoreStagingError, match=r"has \d+ B free but the staged store needs"):
        deployment.stage(free_bytes_of=free)
    assert seen == [deployment.staging_dir / deployment.staged_id()]
    assert deployment.interleaved == 0
    assert list(deployment.staging_dir.iterdir()) == []


def test_legacy_config_maps_store_path_unchanged(tmp_path, monkeypatch):
    deployment = Deployment(tmp_path, monkeypatch)
    raw = json.loads(deployment.config_path.read_text())
    del raw["store_residency"], raw["store_staging_dir"]
    deployment.config_path.write_text(json.dumps(raw))

    assert deployment.stage() is None
    assert not deployment.staging_dir.exists()
    assert list(deployment.runtime_dir.iterdir()) == []
    config = load_distributed_weight_config(deployment.config_path)
    assert config["store_path"] == str(deployment.output / sb.STORE_FILE_NAME)
    assert config["store_residency"] is None


# --------------------------------------------------------------------------- #
#  WorkerManager wiring                                                        #
# --------------------------------------------------------------------------- #


def _method(tree, name):
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(function):
    return [
        ast.unparse(node.func) for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]


def test_start_stages_after_compaction_and_before_the_daemon():
    from tests.test_worker_pool_restart import _worker_manager_tree

    tree = _worker_manager_tree()
    calls = _calls(_method(tree, "_start_impl"))
    assert calls.index("self._compact_memory") < calls.index(
        "self._stage_distributed_weight_store"
    ) < calls.index("self._start_distributed_weight_daemon")

    daemon = _method(tree, "_start_distributed_weight_daemon")
    source = ast.unparse(daemon)
    assert "config_path = self._distributed_weight_config_path" in source
    assert "self.args.distributed_weight_config" not in source.split("config_path =")[1]

    spawn = _method(tree, "_spawn_workers")
    args_call = next(
        node for node in ast.walk(spawn)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "BatchGenWorkerArgs"
    )
    keywords = {kw.arg: ast.unparse(kw.value) for kw in args_call.keywords}
    assert "str(self._distributed_weight_config_path)" in keywords["distributed_weight_config"]


def test_daemon_start_starts_the_event_watcher_after_the_daemon():
    from tests.test_worker_pool_restart import _worker_manager_tree

    calls = _calls(_method(_worker_manager_tree(), "_start_distributed_weight_daemon"))
    assert calls.index("self.distributed_weight_daemon.start") < calls.index(
        "weight_daemon_events.WeightDaemonWatcher"
    ) < calls.index("watcher.start")


class _FakeStaged:
    def __init__(self, order):
        self.order = order
        self.directory = Path("/staged")

    def release(self):
        self.order.append("store_release")


class _FakeWatcher:
    """Records its stop, and whether the daemon was still held at that point:
    the real watcher polls inside the daemon, which must outlive it."""

    def __init__(self, order, manager):
        self.order = order
        self.manager = manager
        self.daemon_held_at_stop = None

    def stop(self):
        self.order.append("watcher_stop")
        self.daemon_held_at_stop = self.manager.distributed_weight_daemon is not None


def _stopping_manager(monkeypatch, tmp_path, order):
    from tests.test_worker_pool_restart import _swap_manager

    manager = _swap_manager(monkeypatch, order)
    manager.args = SimpleNamespace(
        runtime_identity=SimpleNamespace(shm_prefix="run.", runtime_dir=tmp_path / "unused")
    )
    manager._runtime_dir_created = False
    manager._runtime_namespace_owned = False
    manager._runtime_locks = None
    manager._lane_lease = None
    manager._run_lock = None
    manager.parameter_server_instance = None
    manager.model_info = {}
    manager.skeleton_state_dict_file = None
    manager.prefix_cache_runtime_config = None
    manager._close_skeleton_memfd = lambda: None
    manager._staged_weight_store = _FakeStaged(order)
    manager._weight_daemon_watcher = _FakeWatcher(order, manager)
    return manager


def test_stop_releases_the_store_after_the_daemon_and_workers(monkeypatch, tmp_path):
    order = []
    manager = _stopping_manager(monkeypatch, tmp_path, order)
    watcher = manager._weight_daemon_watcher
    manager.stop()
    assert order == [
        "prepare_stop", "teardown", "daemon_stop", "watcher_stop", "store_release",
    ]
    assert watcher.daemon_held_at_stop is True
    assert manager._weight_daemon_watcher is None
    assert manager.distributed_weight_daemon is None
    assert manager._staged_weight_store is None


def test_partial_start_failure_still_releases_the_store(monkeypatch, tmp_path):
    # Staging done, daemon never constructed: the start's rollback stop()
    # must still release it.
    order = []
    manager = _stopping_manager(monkeypatch, tmp_path, order)
    manager.distributed_weight_daemon = None
    manager.worker_process = None
    manager.started = False
    manager.stop()
    assert order == ["store_release"]


def test_stop_keeps_the_store_when_worker_teardown_is_unconfirmed(monkeypatch, tmp_path):
    order = []
    manager = _stopping_manager(monkeypatch, tmp_path, order)

    def teardown_fails():
        order.append("teardown")
        raise RuntimeError("worker still alive")

    manager._stop_workers = teardown_fails
    with pytest.raises(RuntimeError, match="worker still alive"):
        manager.stop()
    assert "store_release" not in order
    assert manager._staged_weight_store is not None


def test_stop_keeps_the_store_while_the_daemon_failed_to_stop(monkeypatch, tmp_path):
    # The daemon lives in this process and still maps the copy.
    order = []
    manager = _stopping_manager(monkeypatch, tmp_path, order)

    def daemon_stop_fails():
        order.append("daemon_stop")
        raise RuntimeError("daemon stop failed")

    manager.distributed_weight_daemon.stop = daemon_stop_fails
    with pytest.raises(RuntimeError, match="daemon stop failed"):
        manager.stop()
    # The watcher keeps reporting on the daemon that is still held.
    assert order == ["prepare_stop", "teardown", "daemon_stop"]
    assert manager._staged_weight_store is not None
    assert manager._weight_daemon_watcher is not None


def test_worker_restart_never_touches_the_store(monkeypatch, tmp_path):
    order = []
    manager = _stopping_manager(monkeypatch, tmp_path, order)
    manager.restart_workers(timeout_s=1)
    assert "store_release" not in order
    assert isinstance(manager._staged_weight_store, _FakeStaged)
