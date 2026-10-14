"""GPU workers must die with the server process that spawned them."""

import ast
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROCESS_UTILS = ROOT / "batchgen" / "server" / "process_utils.py"
MAIN_LOOP = ROOT / "batchgen" / "server_worker_main_loop.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-only"
)


def _load_process_utils():
    """Load process_utils by path: importing the package pulls in the server."""
    spec = importlib.util.spec_from_file_location(
        "batchgen_process_utils_under_test", PROCESS_UTILS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The intermediate process stands in for the HTTP server: it spawns one child
# worker, then gets SIGKILLed the way an OOM kill would end the real server.
_PARENT_SCRIPT = '''
import importlib.util
import multiprocessing
import os
import time

_spec = importlib.util.spec_from_file_location("process_utils", {process_utils!r})
_process_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_process_utils)


def child(pid_file):
    _process_utils.die_with_parent()
    with open(pid_file, "w") as handle:
        handle.write(str(os.getpid()))
    time.sleep(300)


if __name__ == "__main__":
    proc = multiprocessing.get_context("spawn").Process(
        target=child, args=({pid_file!r},)
    )
    proc.start()
    proc.join()
'''


def _process_gone(pid):
    """True if the pid is unused or only a zombie awaiting a reaper."""
    try:
        with open(f"/proc/{pid}/stat") as handle:
            stat = handle.read()
    except FileNotFoundError:
        return True
    # "<pid> (<comm>) <state> ..."; comm may itself contain spaces or ')'.
    return stat[stat.rindex(")") + 2] == "Z"


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@linux_only
def test_child_dies_when_spawning_process_is_killed(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        _PARENT_SCRIPT.format(
            process_utils=str(PROCESS_UTILS), pid_file=str(pid_file)
        )
    )

    parent = subprocess.Popen([sys.executable, str(script)])
    child_pid = None
    try:
        assert _wait_for(lambda: pid_file.is_file() and pid_file.read_text()), (
            "child never reported its pid"
        )
        child_pid = int(pid_file.read_text())
        assert not _process_gone(child_pid)

        parent.kill()
        parent.wait(timeout=10)

        assert _wait_for(lambda: _process_gone(child_pid)), (
            f"child {child_pid} survived its killed parent"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        if child_pid is not None and not _process_gone(child_pid):
            os.kill(child_pid, signal.SIGKILL)


class _Exited(Exception):
    """Raised in place of os._exit so the test can observe the exit."""


def _patch_linux(monkeypatch, module):
    """Run the Linux path with the real prctl call stubbed out."""
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "_set_pdeathsig", lambda: None)


def test_exits_when_parent_died_before_the_signal_was_armed(monkeypatch):
    module = _load_process_utils()
    _patch_linux(monkeypatch, module)
    monkeypatch.setattr(module.os, "getppid", lambda: 1)
    monkeypatch.setattr(
        module.multiprocessing, "parent_process", lambda: SimpleNamespace(pid=4242)
    )

    def _exit(code):
        raise _Exited(code)

    monkeypatch.setattr(module.os, "_exit", _exit)

    with pytest.raises(_Exited) as excinfo:
        module.die_with_parent()
    assert excinfo.value.args[0] == 1


def test_returns_when_the_parent_is_still_alive(monkeypatch):
    module = _load_process_utils()
    _patch_linux(monkeypatch, module)
    monkeypatch.setattr(module.os, "getppid", lambda: 4242)
    monkeypatch.setattr(
        module.multiprocessing, "parent_process", lambda: SimpleNamespace(pid=4242)
    )
    monkeypatch.setattr(
        module.os, "_exit", lambda code: pytest.fail(f"unexpected exit {code}")
    )

    module.die_with_parent()


def test_worker_entry_arms_parent_death_first():
    tree = ast.parse(MAIN_LOOP.read_text(), filename=str(MAIN_LOOP))
    func = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_server_worker_main_impl"
    )
    body = func.body
    if ast.get_docstring(func) is not None:
        body = body[1:]
    first = body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), (
        "first statement of the worker entry is not a call"
    )
    assert getattr(first.value.func, "id", None) == "die_with_parent"


def test_worker_spawn_requires_the_main_thread():
    text = WORKER_MANAGER.read_text()
    guard = text[: text.index("mp.spawn(")]
    assert "threading.current_thread() is not threading.main_thread()" in guard[-800:]
    assert "RuntimeError" in guard[-800:]
