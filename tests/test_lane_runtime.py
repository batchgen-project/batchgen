"""Lane supervisor admission and ownership contracts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "lane_runtime.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lane_runtime_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


lane_runtime = _load_module()


def _load_core_lane_lease():
    package_name = "batchgen.server"
    module_name = "batchgen.server.runtime_lease"
    previous_package = sys.modules.get(package_name)
    previous_module = sys.modules.pop(module_name, None)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return __import__(module_name, fromlist=["LaneLease"])
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
        if previous_package is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous_package


def _candidate(instance_id="lane-0", **overrides):
    value = {
        "instance_id": instance_id,
        "gpu_uuids": ["GPU-a"],
        "listen_port": 11000,
        "dist_init_port": 12000,
        "pynccl_port_base": 21000,
        "pynccl_port_span": 4,
        "paths": {"storage": f"/lanes/{instance_id}/storage"},
    }
    value.update(overrides)
    return value


def test_overlap_checks_gpu_ports_ranges_and_writable_paths():
    active = [_candidate()]

    with pytest.raises(lane_runtime.LaneError, match="GPU UUID"):
        lane_runtime._check_no_overlap(
            _candidate("lane-1", listen_port=11001, dist_init_port=12001),
            active,
        )
    with pytest.raises(lane_runtime.LaneError, match="TCP port"):
        lane_runtime._check_no_overlap(
            _candidate("lane-1", gpu_uuids=["GPU-b"], dist_init_port=12001),
            active,
        )
    with pytest.raises(lane_runtime.LaneError, match="PyNccl"):
        lane_runtime._check_no_overlap(
            _candidate(
                "lane-1",
                gpu_uuids=["GPU-b"],
                listen_port=11001,
                dist_init_port=12001,
                pynccl_port_base=21002,
            ),
            active,
        )
    with pytest.raises(lane_runtime.LaneError, match="writable path"):
        lane_runtime._check_no_overlap(
            _candidate(
                "lane-1",
                gpu_uuids=["GPU-b"],
                listen_port=11001,
                dist_init_port=12001,
                pynccl_port_base=22000,
                paths={"storage": "/lanes/lane-0/storage"},
            ),
            active,
        )


def test_overlap_checks_nested_and_intra_lane_writable_paths():
    active = [
        _candidate(
            paths={"storage": "/lanes/lane-0/storage"},
        )
    ]
    with pytest.raises(lane_runtime.LaneError, match="active lane"):
        lane_runtime._check_no_overlap(
            _candidate(
                "lane-1",
                gpu_uuids=["GPU-b"],
                listen_port=11001,
                dist_init_port=12001,
                pynccl_port_base=22000,
                paths={"storage": "/lanes/lane-0/storage/nested"},
            ),
            active,
        )
    with pytest.raises(lane_runtime.LaneError, match="each other"):
        lane_runtime._check_no_overlap(
            _candidate(
                paths={
                    "storage": "/lanes/lane-0/storage",
                    "temp": "/lanes/lane-0/storage/tmp",
                }
            ),
            [],
        )


def test_overlap_checks_ports_against_other_lane_pynccl_range():
    active = [_candidate(gpu_uuids=["GPU-a", "GPU-b"])]
    with pytest.raises(lane_runtime.LaneError, match="PyNccl"):
        lane_runtime._check_no_overlap(
            _candidate(
                "lane-1",
                gpu_uuids=["GPU-c", "GPU-d"],
                listen_port=21001,
                dist_init_port=12001,
                pynccl_port_base=22000,
            ),
            active,
        )
    with pytest.raises(lane_runtime.LaneError, match="PyNccl"):
        lane_runtime._check_no_overlap(
            _candidate(
                "lane-1",
                gpu_uuids=["GPU-c", "GPU-d"],
                listen_port=11001,
                dist_init_port=13001,
                pynccl_port_base=12000,
            ),
            active,
        )
    lane_runtime._check_no_overlap(
        _candidate(
            "lane-1",
            gpu_uuids=["GPU-c", "GPU-d"],
            listen_port=21004,
            dist_init_port=13001,
            pynccl_port_base=22000,
        ),
        active,
    )


def test_gpu_admission_requires_present_and_idle_physical_uuids(monkeypatch):
    monkeypatch.setattr(lane_runtime, "_gpu_inventory", lambda: {"GPU-a"})
    monkeypatch.setattr(lane_runtime, "_gpu_processes", lambda: [])
    lane_runtime._check_gpus_free(["GPU-a"])

    with pytest.raises(lane_runtime.LaneError, match="not present"):
        lane_runtime._check_gpus_free(["GPU-b"])

    monkeypatch.setattr(
        lane_runtime,
        "_gpu_processes",
        lambda: [("GPU-a", 123)],
    )
    with pytest.raises(lane_runtime.LaneError, match="live compute"):
        lane_runtime._check_gpus_free(["GPU-a"])


def test_resource_names_match_core_lane_lease_contract():
    candidate = _candidate(
        gpu_uuids=["GPU-a", "GPU-b"],
        paths={
            "storage": "/lanes/lane-0/storage",
            "temp": "/lanes/lane-0/tmp",
        },
    )
    resources = lane_runtime._resource_names(candidate)

    assert "gpu:GPU-a" in resources
    assert "gpu:GPU-b" in resources
    assert "port:11000" in resources
    assert "port:12000" in resources
    assert "pynccl:21000:4" in resources
    for path in candidate["paths"].values():
        digest = lane_runtime.hashlib.sha256(path.encode()).hexdigest()
        resource = f"path:{digest}"
        assert resource in resources
        assert lane_runtime._resource_filename(resource) == (
            lane_runtime.hashlib.sha256(resource.encode()).hexdigest()
            + ".lock"
        )


def test_lock_conflict_is_non_destructive(tmp_path):
    path = tmp_path / "resource.lock"
    first = lane_runtime._open_lock(path, fcntl.LOCK_EX)
    try:
        with pytest.raises(BlockingIOError):
            lane_runtime._open_lock(path, fcntl.LOCK_EX)
    finally:
        lane_runtime._close_fd(first)

    replacement = lane_runtime._open_lock(path, fcntl.LOCK_EX)
    lane_runtime._close_fd(replacement)


def test_parent_close_does_not_unlock_child_open_file_description(tmp_path):
    path = tmp_path / "resource.lock"
    parent_fd = lane_runtime._open_lock(path, fcntl.LOCK_EX)
    child_fd = os.dup(parent_fd)
    lane_runtime._close_inherited_parent_fd(parent_fd)
    try:
        with pytest.raises(BlockingIOError):
            lane_runtime._open_lock(path, fcntl.LOCK_EX)
    finally:
        lane_runtime._close_fd(child_fd)


def test_pid_identity_uses_boot_id_start_time_and_process_group(
    tmp_path, monkeypatch
):
    proc_root = tmp_path / "proc"
    (proc_root / "sys/kernel/random").mkdir(parents=True)
    (proc_root / "sys/kernel/random/boot_id").write_text("boot-a\n")
    pid_dir = proc_root / "123"
    pid_dir.mkdir()
    prefix = "123 (batchgen worker) "
    fields = ["S"] + ["0"] * 18 + ["987"] + ["0"] * 4
    (pid_dir / "stat").write_text(prefix + " ".join(fields))
    monkeypatch.setattr(lane_runtime.os, "getpgid", lambda pid: 456)
    manifest = {
        "pid": 123,
        "boot_id": "boot-a",
        "pid_start_time": 987,
        "process_group": 456,
    }

    assert lane_runtime._pid_identity_matches(manifest, proc_root)
    manifest["pid_start_time"] = 988
    assert not lane_runtime._pid_identity_matches(manifest, proc_root)


def test_process_group_liveness_does_not_depend_on_leader_pid(monkeypatch):
    def fake_killpg(process_group, sig):
        assert process_group == 456
        assert sig == 0

    monkeypatch.setattr(lane_runtime.os, "killpg", fake_killpg)
    assert lane_runtime._process_group_exists(456)

    def missing_group(process_group, sig):
        raise ProcessLookupError

    monkeypatch.setattr(lane_runtime.os, "killpg", missing_group)
    assert not lane_runtime._process_group_exists(456)


def test_pidfd_owner_change_after_open_refuses_signal(monkeypatch):
    source = os.open(os.devnull, os.O_RDONLY)
    opened = []

    def open_pidfd(pid):
        assert pid == 123
        fd = os.dup(source)
        opened.append(fd)
        return fd

    monkeypatch.setattr(lane_runtime.os, "pidfd_open", open_pidfd, raising=False)
    monkeypatch.setattr(
        lane_runtime.signal,
        "pidfd_send_signal",
        lambda *args: pytest.fail("unverified owner must not be signaled"),
        raising=False,
    )
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: False)

    try:
        with pytest.raises(lane_runtime.LaneError, match="identity changed"):
            lane_runtime._open_verified_owner_pidfd({"pid": 123})
        with pytest.raises(OSError):
            os.fstat(opened[0])
    finally:
        os.close(source)


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open") or not hasattr(lane_runtime.signal, "pidfd_send_signal"),
    reason="Linux pidfd APIs are required",
)
def test_pidfd_signals_only_the_verified_process():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    pidfd = None
    try:
        manifest = {
            "pid": process.pid,
            "boot_id": lane_runtime._boot_id(),
            "pid_start_time": lane_runtime._proc_start_time(process.pid),
            "process_group": os.getpgid(process.pid),
        }
        pidfd = lane_runtime._open_verified_owner_pidfd(manifest)
        lane_runtime._signal_pidfd(pidfd, lane_runtime.signal.SIGTERM)
        assert process.wait(timeout=5) == -lane_runtime.signal.SIGTERM
    finally:
        if process.poll() is None:
            if pidfd is not None:
                lane_runtime._signal_pidfd(pidfd, lane_runtime.signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
        if pidfd is not None:
            os.close(pidfd)


def test_live_lane_stop_requires_pidfd_support(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    manifest = _candidate()
    manifest.update({"state": "admitted", "pid": 123, "process_group": 456})
    state_path = state_root / "lane-0.json"
    lane_runtime._atomic_json(state_path, manifest)
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: True)
    monkeypatch.delattr(lane_runtime.os, "pidfd_open", raising=False)
    monkeypatch.delattr(lane_runtime.signal, "pidfd_send_signal", raising=False)
    monkeypatch.setattr(
        lane_runtime.os,
        "kill",
        lambda *args: pytest.fail("numeric PID must not be signaled"),
    )
    monkeypatch.setattr(
        lane_runtime.os,
        "killpg",
        lambda *args: pytest.fail("process group must not be signaled"),
    )

    args = SimpleNamespace(state_root=state_root, instance_id="lane-0", timeout=0)
    with pytest.raises(lane_runtime.LaneError, match="requires Linux pidfd"):
        lane_runtime.stop_lane(args)
    assert lane_runtime._read_json(state_path)["state"] == "admitted"


@pytest.mark.parametrize("command", ["status", "stop", "verify"])
def test_lane_commands_reject_instance_id_path_traversal(tmp_path, command):
    state_root = tmp_path / "state"
    state_root.mkdir()
    (tmp_path / "foreign.json").write_text('{"unrelated": true}')
    args = SimpleNamespace(
        state_root=state_root,
        instance_id="../foreign",
        timeout=0,
    )
    operation = {
        "status": lane_runtime.lane_status,
        "stop": lane_runtime.stop_lane,
        "verify": lane_runtime.verify_lane,
    }[command]

    with pytest.raises(lane_runtime.LaneError, match="instance_id"):
        operation(args)


@pytest.mark.parametrize("owner_still_valid", [False, True])
def test_stop_pidfd_escalation_requires_current_owner_identity(
    tmp_path, monkeypatch, owner_still_valid
):
    state_root = tmp_path / "state"
    manifest = _candidate()
    manifest.update(
        {
            "state": "admitted",
            "pid": 123,
            "process_group": 456,
            "paths": {"temp": str(tmp_path / "lane-tmp")},
        }
    )
    state_path = state_root / "lane-0.json"
    lane_runtime._atomic_json(state_path, manifest)
    identity_checks = iter([True, True, owner_still_valid])
    monkeypatch.setattr(
        lane_runtime,
        "_pid_identity_matches",
        lambda value: next(identity_checks),
    )
    group_checks = iter([True, False, False])
    monkeypatch.setattr(
        lane_runtime,
        "_process_group_exists",
        lambda process_group: next(group_checks),
    )
    signals = []
    pidfd_source = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(
        lane_runtime.os,
        "pidfd_open",
        lambda pid: os.dup(pidfd_source),
        raising=False,
    )
    monkeypatch.setattr(
        lane_runtime.signal,
        "pidfd_send_signal",
        lambda fd, sig: signals.append(("pidfd", sig)),
        raising=False,
    )
    monkeypatch.setattr(
        lane_runtime.os,
        "kill",
        lambda pid, sig: pytest.fail("numeric PID must not be signaled"),
    )
    monkeypatch.setattr(
        lane_runtime.os,
        "killpg",
        lambda process_group, sig: pytest.fail("process group must not be signaled"),
    )
    monkeypatch.setattr(lane_runtime, "_gpu_processes", lambda: [])
    args = SimpleNamespace(
        state_root=state_root,
        instance_id="lane-0",
        timeout=0,
    )

    try:
        if owner_still_valid:
            assert lane_runtime.stop_lane(args)["state"] == "stopped"
        else:
            with pytest.raises(lane_runtime.LaneError, match="owner identity changed"):
                lane_runtime.stop_lane(args)
        assert lane_runtime._read_json(state_path)["state"] == (
            "stopped" if owner_still_valid else "failed"
        )
        assert ("pidfd", lane_runtime.signal.SIGTERM) in signals
        assert (("pidfd", lane_runtime.signal.SIGKILL) in signals) == owner_still_valid
    finally:
        os.close(pidfd_source)


def test_stop_reaps_dead_owner_without_signaling(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    manifest = _candidate()
    manifest.update(
        {
            "state": "admitted",
            "pid": 123,
            "process_group": 456,
            "paths": {"temp": str(tmp_path / "lane-tmp")},
        }
    )
    state_path = state_root / "lane-0.json"
    lane_runtime._atomic_json(state_path, manifest)
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_process_group_exists", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_gpu_processes", lambda: [])
    monkeypatch.setattr(
        lane_runtime.os,
        "kill",
        lambda *args: pytest.fail("dead owner must not be signaled"),
    )
    monkeypatch.setattr(
        lane_runtime.os,
        "killpg",
        lambda *args: pytest.fail("dead process group must not be signaled"),
    )
    args = SimpleNamespace(state_root=state_root, instance_id="lane-0", timeout=0)

    result = lane_runtime.stop_lane(args)

    assert result["state"] == "stopped"
    assert lane_runtime._read_json(state_path)["state"] == "stopped"


def test_stop_preserves_dead_owner_with_live_group(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    manifest = _candidate()
    manifest.update({"state": "admitted", "pid": 123, "process_group": 456})
    state_path = state_root / "lane-0.json"
    lane_runtime._atomic_json(state_path, manifest)
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_process_group_exists", lambda value: True)
    monkeypatch.setattr(
        lane_runtime.os,
        "kill",
        lambda *args: pytest.fail("unverified owner must not be signaled"),
    )
    monkeypatch.setattr(
        lane_runtime.os,
        "killpg",
        lambda *args: pytest.fail("unverified group must not be signaled"),
    )
    args = SimpleNamespace(state_root=state_root, instance_id="lane-0", timeout=0)

    with pytest.raises(lane_runtime.LaneError, match="unverified lane owner"):
        lane_runtime.stop_lane(args)

    assert lane_runtime._read_json(state_path)["state"] == "admitted"


def test_stop_cannot_overwrite_manifest_during_admission(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    state_path = state_root / "lane-0.json"
    manifest = _candidate()
    manifest.update(
        {
            "state": "admitted",
            "process_group": 456,
            "paths": {"temp": str(tmp_path / "lane-tmp")},
        }
    )
    lane_runtime._atomic_json(state_path, manifest)
    monkeypatch.setattr(lane_runtime, "HOST_LOCK_ROOT", tmp_path / "locks")
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_process_group_exists", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_gpu_processes", lambda: [])
    lock_root = lane_runtime._ensure_private_dir(lane_runtime.HOST_LOCK_ROOT)
    admission_fd = lane_runtime._open_lock(lock_root / "admission.lock", fcntl.LOCK_EX)
    try:
        args = SimpleNamespace(state_root=state_root, instance_id="lane-0", timeout=0)
        with pytest.raises(BlockingIOError):
            lane_runtime.stop_lane(args)
        assert lane_runtime._read_json(state_path) == manifest
    finally:
        lane_runtime._close_fd(admission_fd)
    assert lane_runtime.stop_lane(args)["state"] == "stopped"


def test_stop_preserves_dead_owner_with_residual_gpu(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    manifest = _candidate()
    manifest.update(
        {
            "state": "admitted",
            "pid": 123,
            "process_group": 456,
            "paths": {"temp": str(tmp_path / "lane-tmp")},
        }
    )
    state_path = state_root / "lane-0.json"
    lane_runtime._atomic_json(state_path, manifest)
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_process_group_exists", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_gpu_processes", lambda: [("GPU-a", 789)])
    monkeypatch.setattr(
        lane_runtime.os,
        "kill",
        lambda *args: pytest.fail("dead owner must not be signaled"),
    )
    args = SimpleNamespace(state_root=state_root, instance_id="lane-0", timeout=0)

    with pytest.raises(lane_runtime.LaneError, match="residual lane resources"):
        lane_runtime.stop_lane(args)

    failed = lane_runtime._read_json(state_path)
    assert failed["state"] == "failed"
    assert failed["residual_gpu_processes"] == [["GPU-a", 789]]


def test_stop_preserves_state_when_assigned_gpu_still_has_a_process(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "state"
    manifest = _candidate()
    manifest.update(
        {
            "state": "admitted",
            "pid": 123,
            "process_group": 456,
            "paths": {"temp": str(tmp_path / "lane-tmp")},
        }
    )
    state_path = state_root / "lane-0.json"
    lane_runtime._atomic_json(state_path, manifest)
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: True)
    monkeypatch.setattr(lane_runtime, "_process_group_exists", lambda value: False)
    pidfd_source = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(
        lane_runtime.os,
        "pidfd_open",
        lambda pid: os.dup(pidfd_source),
        raising=False,
    )
    monkeypatch.setattr(
        lane_runtime.signal,
        "pidfd_send_signal",
        lambda fd, sig: None,
        raising=False,
    )
    monkeypatch.setattr(
        lane_runtime,
        "_gpu_processes",
        lambda: [("GPU-a", 789)],
    )
    args = SimpleNamespace(
        state_root=state_root,
        instance_id="lane-0",
        timeout=0,
    )

    try:
        with pytest.raises(lane_runtime.LaneError, match="residual lane resources"):
            lane_runtime.stop_lane(args)
    finally:
        os.close(pidfd_source)

    failed = lane_runtime._read_json(state_path)
    assert failed["state"] == "failed"
    assert failed["residual_gpu_processes"] == [["GPU-a", 789]]


def test_verify_rejects_foreign_process_only_on_assigned_gpu(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    manifest = _candidate()
    manifest.update(
        {
            "state": "admitted",
            "pid": 123,
            "process_group": 456,
            "paths": {"temp": str(tmp_path / "lane-tmp")},
        }
    )
    state_path = state_root / "lane-0.json"
    lane_runtime._atomic_json(state_path, manifest)
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: True)
    monkeypatch.setattr(lane_runtime, "_resource_names", lambda value: ())
    monkeypatch.setattr(lane_runtime.os, "getpgid", lambda pid: 789)
    args = SimpleNamespace(state_root=state_root, instance_id="lane-0")

    monkeypatch.setattr(lane_runtime, "_gpu_processes", lambda: [("GPU-b", 999)])
    assert lane_runtime.verify_lane(args)["verified"] is True

    monkeypatch.setattr(lane_runtime, "_gpu_processes", lambda: [("GPU-a", 999)])
    with pytest.raises(lane_runtime.LaneError, match="foreign GPU processes"):
        lane_runtime.verify_lane(args)
    assert lane_runtime._read_json(state_path) == manifest


@pytest.mark.parametrize("same_lane", [False, True])
def test_dead_lane_closeout_matches_only_exact_instance_shm(
    tmp_path, monkeypatch, same_lane
):
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    shm_name = (
        f"batchgen_lane_{'a' * 32}.host_kv"
        if same_lane
        else f"batchgen_lane_{'a' * 32}_b_{'b' * 32}.host_kv"
    )
    shm_object = shm_dir / shm_name
    shm_object.touch()
    state_root = tmp_path / "state"
    manifest = _candidate("lane")
    manifest.update(
        {
            "state": "admitted",
            "process_group": 456,
            "paths": {"temp": str(tmp_path / "lane-tmp")},
        }
    )
    lane_runtime._atomic_json(state_root / "lane.json", manifest)
    real_path = Path
    monkeypatch.setattr(
        lane_runtime,
        "Path",
        lambda value: shm_dir if value == "/dev/shm" else real_path(value),
    )
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_process_group_exists", lambda value: False)
    monkeypatch.setattr(lane_runtime, "_gpu_processes", lambda: [])
    args = SimpleNamespace(state_root=state_root, instance_id="lane")

    if same_lane:
        with pytest.raises(lane_runtime.LaneError, match="residual lane resources"):
            lane_runtime.stop_lane(args)
        assert lane_runtime._read_json(state_root / "lane.json")["state"] == "failed"
    else:
        assert lane_runtime.stop_lane(args)["state"] == "stopped"
    assert shm_object.exists()


def test_prepare_paths_pins_both_packages_to_one_worktree(tmp_path):
    worktree = tmp_path / "worktree"
    (worktree / "batchgen").mkdir(parents=True)
    (worktree / "batchgen_kernels").mkdir()
    lane_root = tmp_path / "lane"
    converted = lane_root / "converted"

    pyroot, paths, log_path = lane_runtime._prepare_paths(
        lane_root,
        worktree,
        converted,
    )

    assert (pyroot / "batchgen").resolve() == (worktree / "batchgen").resolve()
    assert (pyroot / "batchgen_kernels").resolve() == (
        worktree / "batchgen_kernels"
    ).resolve()
    assert Path(paths["converted_checkpoint"]).is_relative_to(lane_root)
    assert log_path == lane_root / "logs" / "server.log"


def test_lane_root_inside_worktree_is_rejected_before_creation(tmp_path):
    worktree = tmp_path / "worktree"
    (worktree / "batchgen").mkdir(parents=True)
    (worktree / "batchgen_kernels").mkdir()
    lane_root = worktree / "lane"

    with pytest.raises(lane_runtime.LaneError, match="worktree"):
        lane_runtime._prepare_paths(lane_root, worktree, lane_root / "converted")

    assert not lane_root.exists()


def test_nested_lane_root_is_rejected_before_touching_active_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(lane_runtime, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(lane_runtime, "HOST_LOCK_ROOT", tmp_path / "host-locks")
    monkeypatch.setattr(lane_runtime, "LANE_LOCK_ROOT", tmp_path / "lane-locks")
    active_root = tmp_path / "lane-0"
    monkeypatch.setattr(
        lane_runtime,
        "_active_manifests",
        lambda state_root: [{"lane_root": str(active_root)}],
    )
    monkeypatch.setattr(
        lane_runtime,
        "_check_memory",
        lambda *args: pytest.fail("admission continued into memory checks"),
    )
    worktree = tmp_path / "worktree"
    (worktree / "batchgen").mkdir(parents=True)
    (worktree / "batchgen_kernels").mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    nested_root = active_root / "logs" / "lane-1"
    args = argparse.Namespace(
        instance_id="lane-1",
        model="openai/gpt-oss-120b",
        gpu_uuid=["GPU-b"],
        listen_port=11001,
        dist_port=12001,
        pynccl_port_base=22000,
        pynccl_port_span=4,
        host_kv_cache_gb=1,
        host_memory_gb=192,
        shm_gb=128,
        safety_gb=1,
        checkpoint=checkpoint,
        converted_ckpt_dir=nested_root / "converted",
        worktree=worktree,
        lane_root=nested_root,
        python="/usr/bin/python3",
    )

    with pytest.raises(lane_runtime.LaneError, match="lane root overlaps"):
        lane_runtime.start_lane(args)

    assert not nested_root.exists()


def test_lane_root_admission_fails_closed_on_unknown_or_parent_root(tmp_path):
    active_root = tmp_path / "lane-0"
    with pytest.raises(lane_runtime.LaneError, match="valid lane root"):
        lane_runtime._check_lane_root_available(tmp_path / "lane-1", [{}])
    with pytest.raises(lane_runtime.LaneError, match="lane root overlaps"):
        lane_runtime._check_lane_root_available(
            tmp_path, [{"lane_root": str(active_root)}]
        )
    lane_runtime._check_lane_root_available(
        tmp_path / "lane-1", [{"lane_root": str(active_root)}]
    )


def test_o200k_base_cache_is_verified_and_lane_local(tmp_path, monkeypatch):
    data = b"qualified-encoding-asset"
    monkeypatch.setattr(
        lane_runtime, "_O200K_BASE_SHA256", hashlib.sha256(data).hexdigest()
    )
    source = tmp_path / "o200k_base.tiktoken"
    source.write_bytes(data)
    first = lane_runtime._seed_o200k_base(tmp_path / "lane-0", source)
    second = lane_runtime._seed_o200k_base(tmp_path / "lane-1", source)

    assert first != second
    assert (first / lane_runtime._O200K_BASE_CACHE_KEY).read_bytes() == data
    assert (second / lane_runtime._O200K_BASE_CACHE_KEY).read_bytes() == data
    assert lane_runtime._seed_o200k_base(tmp_path / "lane-0", source) == first

    (first / lane_runtime._O200K_BASE_CACHE_KEY).write_bytes(b"corrupt")
    with pytest.raises(lane_runtime.LaneError, match="cache checksum mismatch"):
        lane_runtime._seed_o200k_base(tmp_path / "lane-0", source)

    source.write_bytes(b"wrong-source")
    with pytest.raises(lane_runtime.LaneError, match="encoding checksum mismatch"):
        lane_runtime._seed_o200k_base(tmp_path / "lane-1", source)


def test_server_command_carries_exact_shared_contract(tmp_path):
    args = argparse.Namespace(
        python="/env/bin/python",
        model="openai/gpt-oss-120b",
        instance_id="lane-0",
        checkpoint=tmp_path / "checkpoint",
        listen_port=11000,
        dist_port=12000,
        pynccl_port_base=21000,
        pynccl_port_span=4,
        host_kv_cache_gb=16,
        gpu_uuid=["GPU-a", "GPU-b"],
    )
    candidate = _candidate(
        gpu_uuids=args.gpu_uuid,
        paths={
            "storage": str(tmp_path / "storage"),
            "converted_checkpoint": str(tmp_path / "converted"),
        },
    )
    command = lane_runtime._server_command(args, candidate, manifest_fd=17)

    assert command[:3] == ["/env/bin/python", "-m", "batchgen.launch_http_server"]
    assert command[command.index("--runtime-mode") + 1] == "shared"
    assert command[command.index("--lane-lease-manifest-fd") + 1] == "17"
    assert command[command.index("--world-size") + 1] == "2"
    assert "--fast-init" not in command
    assert "--enable-hugetlbfs" not in command


def test_memory_formula_charges_shmem_as_nonreclaimable():
    gib = 1024**3
    meminfo = {
        "MemTotal": 100 * gib,
        "MemFree": 10 * gib,
        "Buffers": 2 * gib,
        "Cached": 30 * gib,
        "Shmem": 8 * gib,
        "SReclaimable": 5 * gib,
    }
    assert lane_runtime._nonreclaimable_bytes(meminfo) == 61 * gib


def test_gptoss_admission_rejects_underreported_lane_memory(tmp_path):
    args = SimpleNamespace(
        instance_id="lane-0",
        model="openai/gpt-oss-120b",
        gpu_uuid=["GPU-a"],
        listen_port=11000,
        dist_port=12000,
        pynccl_port_base=21000,
        pynccl_port_span=4,
        host_kv_cache_gb=64,
        host_memory_gb=256,
        shm_gb=128,
        safety_gb=64,
        checkpoint=tmp_path,
    )
    with pytest.raises(lane_runtime.LaneError, match="SHM reservation"):
        lane_runtime._validate_start_args(args)

    args.shm_gb = 150
    args.host_memory_gb = 200
    with pytest.raises(lane_runtime.LaneError, match="host-memory reservation"):
        lane_runtime._validate_start_args(args)

    args.host_memory_gb = 214
    lane_runtime._validate_start_args(args)


def test_supervisor_bundle_is_accepted_by_core_lane_lease(
    tmp_path, monkeypatch
):
    core_lease = _load_core_lane_lease()
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    paths = {
        "storage": str(tmp_path / "storage"),
        "converted_checkpoint": str(tmp_path / "converted"),
        "temp": str(tmp_path / "tmp"),
        "torch_extensions": str(tmp_path / "torch-extensions"),
        "triton": str(tmp_path / "triton"),
        "torchinductor": str(tmp_path / "torchinductor"),
        "cuda": str(tmp_path / "cuda"),
    }
    env_names = {
        "TMPDIR": "temp",
        "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "CUDA_CACHE_PATH": "cuda",
    }
    for env_name, path_name in env_names.items():
        monkeypatch.setenv(env_name, paths[path_name])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    candidate = _candidate(
        gpu_uuids=["GPU-a", "GPU-b"],
        paths=paths,
    )
    resources = {}
    for resource in lane_runtime._resource_names(candidate):
        path = lock_root / lane_runtime._resource_filename(resource)
        resources[resource] = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    manifest = {
        "version": 1,
        "instance_id": "lane-0",
        "gpu_uuids": ["GPU-a", "GPU-b"],
        "listen_port": 11000,
        "dist_init_port": 12000,
        "pynccl_port_base": 21000,
        "pynccl_port_span": 4,
        "paths": paths,
        "resources": resources,
    }
    manifest_path = tmp_path / "lease.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_fd = os.open(manifest_path, os.O_RDONLY)
    args = SimpleNamespace(
        lane_lease_manifest_fd=manifest_fd,
        instance_id="lane-0",
        world_size=2,
        listen_port=11000,
        dist_init_addr="localhost:12000",
        pynccl_port_base=21000,
        pynccl_port_span=4,
        storage_path=Path(paths["storage"]),
        converted_ckpt_dir=Path(paths["converted_checkpoint"]),
    )

    lease = core_lease.LaneLease.acquire(args, lock_root=lock_root)
    lease.close()


def test_ambiguous_stale_manifest_blocks_without_ttl_takeover(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "state"
    manifest = _candidate()
    manifest.update({"state": "admitted", "pid": 123})
    lane_runtime._atomic_json(state_root / "lane-0.json", manifest)
    monkeypatch.setattr(
        lane_runtime,
        "_pid_identity_matches",
        lambda value: False,
    )

    with pytest.raises(lane_runtime.LaneError, match="ambiguous stale"):
        lane_runtime._active_manifests(state_root)


@pytest.mark.parametrize(
    "corruption",
    [
        "valid",
        "missing_gpu_uuids",
        "unhashable_gpu",
        "missing_host_reservation",
        "missing_shm_reservation",
        "missing_temp_path",
        "filename_mismatch",
        "stopped_live_owner",
    ],
)
def test_manifest_admission_validates_live_ownership(
    tmp_path, monkeypatch, corruption
):
    state_root = tmp_path / "state"
    lane_root = tmp_path / "lane-0"
    paths = lane_runtime._canonical_paths(lane_root, lane_root / "converted")
    manifest = _candidate(paths=paths)
    manifest.update(
        version=1,
        state="admitted",
        lane_root=str(lane_root),
        host_memory_reservation_bytes=192 * 1024**3,
        shm_reservation_bytes=128 * 1024**3,
        pid=123,
        process_group=123,
    )
    if corruption == "missing_gpu_uuids":
        del manifest["gpu_uuids"]
    elif corruption == "unhashable_gpu":
        manifest["gpu_uuids"] = [{"uuid": "GPU-a"}]
    elif corruption == "missing_host_reservation":
        del manifest["host_memory_reservation_bytes"]
    elif corruption == "missing_shm_reservation":
        del manifest["shm_reservation_bytes"]
    elif corruption == "missing_temp_path":
        del manifest["paths"]["temp"]
    elif corruption == "stopped_live_owner":
        manifest["state"] = "stopped"
    name = "other.json" if corruption == "filename_mismatch" else "lane-0.json"
    lane_runtime._atomic_json(state_root / name, manifest)
    monkeypatch.setattr(lane_runtime, "_pid_identity_matches", lambda value: True)

    if corruption == "valid":
        assert lane_runtime._active_manifests(state_root) == [manifest]
    else:
        with pytest.raises(lane_runtime.LaneError, match="manifest"):
            lane_runtime._active_manifests(state_root)
