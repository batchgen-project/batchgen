"""CPU checks of the code identity a worker cohort publishes.

A worker restart re-imports BatchGen from the source tree on disk, so the
revision of that tree when a cohort starts is what the cohort runs. Covers
the identity of a clean, dirty and git-less tree (real git where present,
fakes for every way git can fail), the pinned log line, and the publication
path: worker main loop -> scheduler -> WorkerManager.
"""

import ast
import asyncio
import queue
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.worker import code_identity as ci
from tests.test_batch_in_progress_failure import bs

ROOT = Path(__file__).resolve().parents[2]
REV = "0123456789abcdef0123456789abcdef01234567"
HAVE_GIT = shutil.which("git") is not None


def _git(tree, *argv):
    subprocess.run(
        ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgsign=false", *argv],
        check=True, capture_output=True, text=True,
    )
    return subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()


def _repo(tmp_path):
    tree = tmp_path / "tree"
    (tree / "batchgen").mkdir(parents=True)
    (tree / "batchgen" / "__init__.py").write_text("")
    (tree / ".gitignore").write_text("*.so\n")
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    _git(tree, "add", "-A")
    return tree, _git(tree, "commit", "-q", "-m", "one")


class _Run:
    """Stands in for subprocess.run and records the argv it was given."""

    def __init__(self, stdout="", returncode=0, raises=None):
        self.stdout, self.returncode, self.raises = stdout, returncode, raises
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(stdout=self.stdout, returncode=self.returncode)


def _git_tree(tmp_path):
    (tmp_path / ".git").mkdir()
    return tmp_path


# -- identity of a real tree -------------------------------------------------

@pytest.mark.skipif(not HAVE_GIT, reason="git binary not installed")
def test_clean_tree_reports_its_commit(tmp_path):
    tree, head = _repo(tmp_path)
    identity = ci.compute_code_identity(tree)
    assert identity == ci.CodeIdentity(source="git", revision=head, dirty=False)


@pytest.mark.skipif(not HAVE_GIT, reason="git binary not installed")
def test_tracked_change_or_untracked_file_is_dirty_and_ignored_file_is_not(tmp_path):
    tree, head = _repo(tmp_path)
    (tree / "batchgen" / "core_engine.so").write_bytes(b"\0")
    assert ci.compute_code_identity(tree).dirty is False
    (tree / "batchgen" / "new_module.py").write_text("x = 1\n")
    assert ci.compute_code_identity(tree) == ci.CodeIdentity("git", head, True)
    (tree / "batchgen" / "new_module.py").unlink()
    (tree / "batchgen" / "__init__.py").write_text("y = 2\n")
    assert ci.compute_code_identity(tree) == ci.CodeIdentity("git", head, True)


@pytest.mark.skipif(not HAVE_GIT, reason="git binary not installed")
def test_moved_head_is_seen_by_the_next_computation(tmp_path):
    """What a restarted cohort sees: the same tree, a new HEAD."""
    tree, first = _repo(tmp_path)
    (tree / "batchgen" / "__init__.py").write_text("z = 3\n")
    second = _git(tree, "commit", "-q", "-am", "two")
    assert second != first
    assert ci.compute_code_identity(tree) == ci.CodeIdentity("git", second, False)


@pytest.mark.skipif(not HAVE_GIT, reason="git binary not installed")
def test_linked_worktree_reports_its_own_head(tmp_path):
    tree, first = _repo(tmp_path)
    linked = tmp_path / "linked"
    _git(tree, "worktree", "add", "-q", "--detach", str(linked))
    (tree / "batchgen" / "__init__.py").write_text("w = 4\n")
    second = _git(tree, "commit", "-q", "-am", "two")
    # .git is a file in a linked worktree; its HEAD is still the first commit.
    assert (linked / ".git").is_file()
    assert ci.compute_code_identity(linked) == ci.CodeIdentity("git", first, False)
    assert ci.compute_code_identity(tree).revision == second


@pytest.mark.skipif(not HAVE_GIT, reason="git binary not installed")
def test_repository_without_a_commit_is_unknown(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    identity = ci.compute_code_identity(tmp_path)
    assert identity == ci.CodeIdentity("git", "unknown", None, "git_no_commit")


def test_no_git_metadata_is_unknown_without_calling_git(tmp_path):
    run = _Run()
    identity = ci.compute_code_identity(tmp_path, run=run)
    assert identity == ci.CodeIdentity("none", "unknown", None, "no_git_metadata")
    assert run.calls == []


def test_repository_further_up_does_not_describe_the_package(tmp_path):
    # An installed package under some checkout (a venv inside a repository).
    (tmp_path / ".git").mkdir()
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    run = _Run(stdout=f"# branch.oid {REV}\n")
    assert ci.compute_code_identity(site, run=run).revision == "unknown"
    assert run.calls == []


def test_package_tree_is_this_checkout():
    assert ci.package_tree() == ROOT


# -- every way git can fail, and the one call it gets -------------------------

def test_one_status_call_with_timeout_and_no_stdin(tmp_path):
    run = _Run(stdout=f"# branch.oid {REV}\n# branch.head main\n")
    tree = _git_tree(tmp_path)
    assert ci.compute_code_identity(tree, run=run, timeout_s=7.0) == ci.CodeIdentity("git", REV, False)
    ((argv, kwargs),) = run.calls
    assert argv[0] == "git" and "--no-optional-locks" in argv
    assert argv[argv.index("-C") + 1] == str(tree)
    assert argv[argv.index("-C") + 2:] == [
        "status", "--porcelain=v2", "--branch", "--no-ahead-behind",
        "--untracked-files=normal",
    ]
    assert kwargs["timeout"] == 7.0 and kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["check"] is False


@pytest.mark.parametrize("stdout, expected", [
    (f"# branch.oid {REV}\n1 .M N... 100644 100644 100644 a b batchgen/x.py\n",
     ci.CodeIdentity("git", REV, True)),
    (f"# branch.oid {REV}\n? batchgen/new.py\n", ci.CodeIdentity("git", REV, True)),
    ("# branch.oid (initial)\n", ci.CodeIdentity("git", "unknown", None, "git_no_commit")),
    ("", ci.CodeIdentity("git", "unknown", None, "git_unparseable")),
    ("# branch.oid not-a-sha\n", ci.CodeIdentity("git", "unknown", None, "git_unparseable")),
])
def test_status_output(tmp_path, stdout, expected):
    assert ci.compute_code_identity(_git_tree(tmp_path), run=_Run(stdout=stdout)) == expected


@pytest.mark.parametrize("run, reason", [
    (_Run(raises=FileNotFoundError("git")), "git_unavailable"),
    (_Run(raises=PermissionError("git")), "git_unavailable"),
    (_Run(raises=subprocess.TimeoutExpired("git", 30)), "git_timeout"),
    (_Run(returncode=128, stdout="ignored"), "git_exit_128"),
])
def test_git_failure_degrades_to_unknown(tmp_path, run, reason):
    identity = ci.compute_code_identity(_git_tree(tmp_path), run=run)
    assert identity == ci.CodeIdentity("git", "unknown", None, reason)


# -- the published value and the log line ------------------------------------

def test_log_line_format_is_pinned():
    clean = ci.CodeIdentity("git", REV, False)
    assert clean.log_line(2, 0) == (
        f"[CODE_IDENTITY] generation=2 node=0 source=git revision={REV} dirty=false reason=-"
    )
    assert ci.CodeIdentity("git", REV, True).log_line(1, 3) == (
        f"[CODE_IDENTITY] generation=1 node=3 source=git revision={REV} dirty=true reason=-"
    )
    assert ci.CodeIdentity("none", "unknown", None, "no_git_metadata").log_line(1, 0) == (
        "[CODE_IDENTITY] generation=1 node=0 source=none revision=unknown dirty=unknown "
        "reason=no_git_metadata"
    )
    assert ci.CodeIdentity("git", "unknown", None, "git_exit_128").log_line(5, 1) == (
        "[CODE_IDENTITY] generation=5 node=1 source=git revision=unknown dirty=unknown "
        "reason=git_exit_128"
    )


def test_identity_never_names_the_tree(tmp_path):
    tree = _git_tree(tmp_path)
    for run in (_Run(stdout=f"# branch.oid {REV}\n"), _Run(returncode=128),
                _Run(raises=subprocess.TimeoutExpired("git", 1))):
        identity = ci.compute_code_identity(tree, run=run)
        assert str(tmp_path) not in identity.log_line(1, 0)
        assert str(tmp_path) not in repr(identity.as_dict())


def test_round_trip_and_refusals():
    for identity in (ci.CodeIdentity("git", REV, True),
                     ci.CodeIdentity("none", "unknown", None, "no_git_metadata"),
                     ci.CodeIdentity("git", "unknown", None, "git_exit_1")):
        assert ci.CodeIdentity.from_dict(identity.as_dict()) == identity
    bad = [
        None,
        {"source": "git", "revision": REV, "dirty": False},
        {"source": "git", "revision": REV, "dirty": False, "reason": None, "path": "/x"},
        {"source": "git", "revision": REV[:12], "dirty": False, "reason": None},
        {"source": "git", "revision": REV, "dirty": None, "reason": None},
        {"source": "git", "revision": REV, "dirty": False, "reason": "git_timeout"},
        {"source": "none", "revision": REV, "dirty": False, "reason": None},
        {"source": "git", "revision": "unknown", "dirty": False, "reason": "git_timeout"},
        {"source": "git", "revision": "unknown", "dirty": None, "reason": "fatal: dubious ownership"},
        {"source": "none", "revision": "unknown", "dirty": None, "reason": "git_timeout"},
        {"source": "svn", "revision": "unknown", "dirty": None, "reason": "no_git_metadata"},
    ]
    for payload in bad:
        with pytest.raises(ValueError):
            ci.CodeIdentity.from_dict(payload)


# -- publication: worker main loop -> scheduler -> WorkerManager -------------

def test_main_loop_computes_before_the_model_build_and_publishes_with_the_capacity():
    tree = ast.parse((ROOT / "batchgen/server_worker_main_loop.py").read_text())
    impl = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "_server_worker_main_impl")
    (compute,) = [node for node in ast.walk(impl) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "compute_code_identity"]
    (build,) = [node for node in ast.walk(impl) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "BatchGenWorker"]
    (init_pg,) = [node for node in ast.walk(impl) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "init_process_group"]
    assert compute.lineno < init_pg.lineno < build.lineno
    (message,) = [node for node in ast.walk(impl) if isinstance(node, ast.Dict)
                  and any(isinstance(v, ast.Constant) and v.value == "deployment_capacity"
                          for v in node.values)]
    entries = {k.value: ast.unparse(v) for k, v in zip(message.keys, message.values)}
    assert entries["code_identity"] == "code_identity.as_dict()"
    assert entries["cohort"] == "args.log_cohort"


def _transient():
    return [{"flavor": "eager", "seam": "startup-deployment-prepare",
             "padding_bsz": 1, "unavailable": "CUDA is unavailable"}]


# A minimal valid [SERVER_CONFIG] publication (undeclared model, no graphs).
SERVER_CONFIG = {
    "prefill_attn": "undeclared", "prefill_moe": "undeclared",
    "decode_attn": "undeclared", "decode_moe": "undeclared",
    "cuda_graphs": False, "graph_buckets": None,
    "host_kv_gib": 1.0, "host_kv_pages": 8,
    "device_kv_gib": 1.0, "device_kv_pages": 8,
    "gpu_memory_frac": 0.9, "max_pool": 8,
}


def test_scheduler_records_the_identity_before_the_deployment_is_ready():
    async def run():
        recorded = []
        worker = SimpleNamespace(response_queue=queue.Queue())
        scheduler = bs.BatchScheduler(SimpleNamespace(), worker, SimpleNamespace())

        def record(cohort, identity):
            # Nothing of this generation can be dispatched yet.
            assert not scheduler._deployment_ready.is_set()
            recorded.append((cohort, identity))

        worker.record_code_identity = record
        worker.record_server_config = lambda cohort, config: None
        await scheduler.start()
        try:
            identity = ci.CodeIdentity("git", REV, False).as_dict()
            # A publication without the identity never makes the deployment ready.
            worker.response_queue.put({"type": "deployment_capacity", "max_model_len": 129,
                                       "startup_transient": _transient(),
                                       "server_config": SERVER_CONFIG, "cohort": 0})
            await asyncio.sleep(0.2)
            assert not scheduler._deployment_ready.is_set() and recorded == []
            worker.response_queue.put({"type": "deployment_capacity", "max_model_len": 129,
                                       "startup_transient": _transient(), "cohort": 3,
                                       "server_config": SERVER_CONFIG,
                                       "code_identity": identity})
            await asyncio.wait_for(scheduler.wait_for_deployment(), timeout=5)
            assert recorded == [(3, identity)]
            assert scheduler.max_model_len == 129
        finally:
            await scheduler.stop()
            worker.response_queue.put(None)

    asyncio.run(run())
