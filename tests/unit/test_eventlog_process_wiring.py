"""``batchgen.eventlog.process``: how every BatchGen process wires the
event log -- CLI options, the run identity, per-process configuration, and the
flushing ``os._exit`` path -- plus source checks that no runtime module still
configures logging on its own.

CPU only; no CUDA, no torch.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()


def _load_logging_setup():
    """Import ``eventlog.process`` and ``runtime_identity`` under stub packages.

    ``batchgen/__init__`` and ``batchgen/server/__init__`` pull in torch and
    the whole server, so both packages are stubbed for the import exactly as
    ``eventlog_testutil`` stubs ``batchgen``, and the stubs are removed again.
    """
    names = (
        "batchgen.eventlog.process",
        "batchgen.server.runtime_identity",
    )
    if all(name in sys.modules for name in names):
        return tuple(sys.modules[name] for name in names)
    stubs = []
    for name, path in (
        ("batchgen", tu.REPO_ROOT / "batchgen"),
        ("batchgen.server", tu.REPO_ROOT / "batchgen" / "server"),
    ):
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]
        stub.__package__ = name
        sys.modules[name] = stub
        stubs.append((name, stub))
    try:
        return tuple(importlib.import_module(name) for name in names)
    finally:
        for name, stub in stubs:
            if sys.modules.get(name) is stub:
                del sys.modules[name]


ls, runtime_identity = _load_logging_setup()

RUN = ls.RunIdentity(tu.RUN_ID, tu.RUN_STARTED_AT.timestamp())


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(ls, "_run", None)
    monkeypatch.setattr(ls, "_settings", None)
    monkeypatch.setattr(ls, "_fallback_run_id", None)
    with tu.isolated_logging(ev):
        yield


def _log(message: str) -> None:
    logging.getLogger("batchgen.test").info(message)


# ----------------------------------------------------------- CLI options


def test_add_logging_args_defaults_and_choices():
    parser = argparse.ArgumentParser()
    ls.add_logging_args(parser)

    defaults = parser.parse_args([])
    assert defaults.log_dir is None
    assert defaults.log_level == "INFO"
    assert parser.parse_args(["--log-level", "warning"]).log_level == "WARNING"
    with pytest.raises(SystemExit):
        parser.parse_args(["--log-level", "LOUD"])


def test_resolve_log_dir_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()

    assert ls.resolve_log_dir(None) == os.path.join(cwd, "batchgen_logs")
    assert ls.resolve_log_dir("logs/x") == os.path.join(cwd, "logs", "x")
    absolute = os.path.join(cwd, "abs")
    assert ls.resolve_log_dir(absolute) == absolute


# ------------------------------------------------------ the run identity


def test_run_identity_json_roundtrip():
    payload = RUN.to_json()

    assert json.loads(payload) == {
        "run_id": tu.RUN_ID,
        "run_started_at": RUN.run_started_at,
    }
    assert ls.RunIdentity.from_json(payload) == RUN
    with pytest.raises(ValueError):
        ls.RunIdentity("not-a-run-id", RUN.run_started_at)


# ---------------------------------------------------- per-process config


def test_configure_server_with_run_writes_node0_server_jsonl(tmp_path):
    # Configuring twice replaces, never stacks, the root handler.
    for _ in range(2):
        ls.configure_process(
            "server", node=0, run=RUN, log_dir=str(tmp_path), level="INFO"
        )
    handlers = [
        handler for handler in logging.getLogger().handlers
        if isinstance(handler, ev.handlers.EventQueueHandler)
    ]
    assert len(handlers) == 1

    _log("hello")
    assert ev.flush(10.0)

    (rec,) = tu.read_jsonl(tmp_path / tu.RUN_DIR / "node0" / "server.jsonl")
    assert rec["run_id"] == tu.RUN_ID
    assert rec["node"] == 0
    assert rec["proc"] == "server"
    assert rec["msg"] == "hello"
    assert ls.current_run() == RUN
    assert ls.current_settings() == (str(tmp_path), "INFO")


def test_configure_server_without_run_buffers_until_bind_run(tmp_path):
    ls.configure_process(
        "server", node=1, run=None, log_dir=str(tmp_path), level="INFO"
    )
    _log("before bind")
    assert ev.flush(10.0)
    assert not (tmp_path / tu.RUN_DIR).exists()
    assert ls.current_run() is None

    ls.bind_run(RUN)
    _log("after bind")
    assert ev.flush(10.0)

    recs = tu.read_jsonl(tmp_path / tu.RUN_DIR / "node1" / "server.jsonl")
    assert [rec["msg"] for rec in recs] == ["before bind", "after bind"]
    assert {rec["node"] for rec in recs} == {1}
    assert {rec["run_id"] for rec in recs} == {tu.RUN_ID}
    assert ls.current_run() == RUN


def test_configure_rank_stamps_rank_local_rank_cohort(tmp_path):
    ls.configure_process(
        "rank",
        node=1,
        rank=9,
        local_rank=1,
        cohort=2,
        run=RUN,
        log_dir=str(tmp_path),
        level="INFO",
    )
    _log("rank record")
    assert ev.flush(10.0)

    (rec,) = tu.read_jsonl(tmp_path / tu.RUN_DIR / "node1" / "rank9.jsonl")
    assert (rec["proc"], rec["node"], rec["rank"]) == ("rank", 1, 9)
    assert (rec["local_rank"], rec["cohort"]) == (1, 2)
    assert rec["run_id"] == tu.RUN_ID


def test_configure_paramserver_fresh_identity(tmp_path):
    run = ls.new_run_identity(
        runtime_identity.RuntimeIdentity.create("paramserver").run_id
    )
    assert re.fullmatch(r"[0-9a-f]{32}", run.run_id)
    assert abs(run.run_started_at - time.time()) < 60

    ls.configure_process(
        "paramserver", node=0, run=run, log_dir=str(tmp_path), level="INFO"
    )
    _log("param server up")
    assert ev.flush(10.0)

    path = Path(ev.jsonl_path())
    assert path.name == "paramserver.jsonl"
    assert path.parent.name == "node0"
    assert path.parent.parent.name.endswith("-" + run.run_id[:8])
    (rec,) = tu.read_jsonl(path)
    assert rec["run_id"] == run.run_id
    assert rec["proc"] == "paramserver"


def test_bind_run_same_run_noop_other_run_raises(tmp_path):
    ls.configure_process(
        "server", node=1, run=None, log_dir=str(tmp_path), level="INFO"
    )
    ls.bind_run(RUN)
    ls.bind_run(RUN)

    other = ls.RunIdentity("0" * 32, RUN.run_started_at)
    with pytest.raises(RuntimeError, match="already bound"):
        ls.bind_run(other)
    assert ls.current_run() == RUN


# ------------------------------------------------------- fallback run


FALLBACK_ID = "f" * 32


def _run_dirs(root: Path) -> list:
    return sorted(path.name for path in root.iterdir())


def test_ensure_run_bound_writes_the_buffer_under_the_fallback(tmp_path):
    ls.configure_process(
        "server", node=1, run=None, log_dir=str(tmp_path), level="INFO",
        fallback_run_id=FALLBACK_ID,
    )
    _log("startup failed before node 0 answered")
    ls.ensure_run_bound()
    assert ev.flush(10.0)

    (run_dir,) = _run_dirs(tmp_path)
    assert run_dir.endswith("-" + FALLBACK_ID[:8])
    (rec,) = tu.read_jsonl(tmp_path / run_dir / "node1" / "server.jsonl")
    assert rec["run_id"] == FALLBACK_ID
    assert rec["msg"] == "startup failed before node 0 answered"


def test_ensure_run_bound_keeps_an_existing_run(tmp_path):
    ls.configure_process(
        "server", node=0, run=RUN, log_dir=str(tmp_path), level="INFO",
        fallback_run_id=FALLBACK_ID,
    )
    ls.ensure_run_bound()
    assert ls.current_run() == RUN
    assert _run_dirs(tmp_path) == [tu.RUN_DIR]


def test_ensure_run_bound_is_a_noop_without_configuration():
    ls.ensure_run_bound(FALLBACK_ID)
    assert ls.current_run() is None


# ------------------------------------------------------------ exit path


_EXIT_SCRIPT = """
import logging
import sys
sys.path.insert(0, {tests!r})
import test_eventlog_process_wiring as t
ls = t.ls
{body}
"""


def _run_exit_script(body: str) -> subprocess.CompletedProcess:
    script = _EXIT_SCRIPT.format(tests=str(tu.TESTS_DIR), body=body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_exit_process_critical_record_survives_os_exit(tmp_path):
    done = _run_exit_script(
        f"ls.configure_process('server', node=0, run=t.RUN, "
        f"log_dir={str(tmp_path)!r}, level='INFO')\n"
        "logging.getLogger('batchgen.test').info('still running')\n"
        "ls.exit_process(7, 'test exit')\n"
    )

    assert done.returncode == 7, done.stderr
    recs = tu.read_jsonl(tmp_path / tu.RUN_DIR / "node0" / "server.jsonl")
    assert recs[0]["msg"] == "still running"
    last = recs[-1]
    assert last["event"] == "process.exit"
    assert last["level"] == "CRITICAL"
    assert last["data"] == {"code": 7, "reason": "test exit"}


def test_exit_process_binds_the_fallback_before_flushing(tmp_path):
    done = _run_exit_script(
        f"ls.configure_process('server', node=1, run=None, "
        f"log_dir={str(tmp_path)!r}, level='INFO', "
        f"fallback_run_id={FALLBACK_ID!r})\n"
        "logging.getLogger('batchgen.test').info('buffered')\n"
        "ls.exit_process(3, 'startup timeout')\n"
    )

    assert done.returncode == 3, done.stderr
    (run_dir,) = _run_dirs(tmp_path)
    recs = tu.read_jsonl(tmp_path / run_dir / "node1" / "server.jsonl")
    assert recs[0]["msg"] == "buffered"
    assert recs[-1]["event"] == "process.exit"
    assert recs[-1]["run_id"] == FALLBACK_ID


def test_exit_process_records_the_active_exception(tmp_path):
    done = _run_exit_script(
        f"ls.configure_process('server', node=0, run=t.RUN, "
        f"log_dir={str(tmp_path)!r}, level='INFO')\n"
        "try:\n"
        "    raise ValueError('boom')\n"
        "except ValueError:\n"
        "    ls.exit_process(1, 'unhandled', exc_info=True)\n"
    )

    assert done.returncode == 1, done.stderr
    recs = tu.read_jsonl(tmp_path / tu.RUN_DIR / "node0" / "server.jsonl")
    assert "ValueError: boom" in recs[-1]["exc"]
    assert "boom" not in recs[-1]["msg"]


def test_exit_process_unconfigured_exits_with_code():
    done = _run_exit_script("ls.exit_process(5, 'early exit')\n")

    assert done.returncode == 5, done.stderr
    assert "process.exit code=5 reason=early exit" in done.stderr


# ----------------------------------------------------- source invariants


_RUNTIME_MODULES = (
    "batchgen/launch_http_server.py",
    "batchgen/server_worker_main_loop.py",
    "batchgen/parameter_server.py",
    "batchgen/entrypoint.py",
    "batchgen/batchgen_.py",
    "batchgen/server/http_server.py",
)


def _calls(tree: ast.AST, name: str) -> list:
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name
    ]


def test_runtime_modules_have_no_basicconfig_and_uvicorn_log_config_none():
    trees = {
        path: ast.parse((tu.REPO_ROOT / path).read_text(), filename=path)
        for path in _RUNTIME_MODULES
    }
    for path, tree in trees.items():
        assert not _calls(tree, "logging.basicConfig"), path

    (run_call,) = _calls(trees["batchgen/server/http_server.py"], "uvicorn.run")
    keywords = {kw.arg: kw.value for kw in run_call.keywords}
    for name in ("log_config", "log_level"):
        value = keywords[name]
        assert isinstance(value, ast.Constant) and value.value is None, name


# ------------------------------------------------ process_utils exit routing

PROCESS_UTILS = tu.REPO_ROOT / "batchgen" / "server" / "process_utils.py"


class _Exited(BaseException):
    pass


def _load_process_utils():
    spec = importlib.util.spec_from_file_location(
        "process_utils_exit_routing_under_test", PROCESS_UTILS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_utils_exit_uses_the_event_log_when_loaded(monkeypatch):
    module = _load_process_utils()
    calls = []

    def fake_record_exit(code, reason, logger=None):
        calls.append((code, reason))

    def fake_exit(code):
        raise _Exited(code)

    monkeypatch.setitem(
        sys.modules,
        "batchgen.eventlog.process",
        types.SimpleNamespace(record_exit=fake_record_exit),
    )
    monkeypatch.setattr(module.os, "_exit", fake_exit)
    with pytest.raises(_Exited) as exc:
        module._exit_process(143, "worker received SIGTERM")
    assert exc.value.args == (143,)
    assert calls == [(143, "worker received SIGTERM")]


def test_process_utils_exit_falls_back_when_not_loaded(monkeypatch, caplog):
    module = _load_process_utils()
    monkeypatch.delitem(sys.modules, "batchgen.eventlog.process", raising=False)

    def fake_exit(code):
        raise _Exited(code)

    monkeypatch.setattr(module.os, "_exit", fake_exit)
    with caplog.at_level(logging.CRITICAL), pytest.raises(_Exited) as exc:
        module._exit_process(1, "parent died")
    assert exc.value.args == (1,)
    assert "process.exit code=1 reason=parent died" in caplog.text
