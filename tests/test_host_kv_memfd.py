"""The host paged KV cache must never be a named /dev/shm object.

A named region outlives the processes that mapped it, so a crashed server can
leave the whole host KV budget pinned until something unlinks the name. The
region is therefore always an anonymous memfd, in every mode, and `--fast-init`
only adds transparent huge pages and prefaulting on top of it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "core" / "KV_Storage" / "host_paged_kv_backend.cpp"
BACKEND_HEADER = ROOT / "core" / "KV_Storage" / "host_paged_kv_backend.h"
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"
COORDINATOR = ROOT / "batchgen" / "kv_cache" / "dual_host_kv_coordinator.py"

# Host-KV-only identity fields. Weight-storage memfd plumbing keeps its own
# names and is deliberately not covered here.
HOST_KV_MEMFD_NAMES = (
    "kv_memfd_pid",
    "kv_memfd_fd",
    "kv_aux_memfd_fd",
    "memfd_creator_pid",
)


def _mentions(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
        if isinstance(child, ast.Attribute) and child.attr == name:
            return True
    return False


def _branches_guarded_by(path: Path, guard_name: str) -> list[str]:
    """Dump every branch whose condition tests `guard_name`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    dumps: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            branches: list[ast.AST] = list(node.body) + list(node.orelse)
        elif isinstance(node, ast.IfExp):
            branches = [node.body, node.orelse]
        else:
            continue
        if _mentions(node.test, guard_name):
            dumps.extend(ast.dump(branch) for branch in branches)
    return dumps


def _method_dump(path: Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.dump(method)


def test_backend_never_opens_a_named_shm_object():
    source = BACKEND.read_text()

    assert "shm_open" not in source
    assert "shm_unlink" not in source
    # One definition plus exactly one call site: creation is unconditional.
    assert source.count("memfd_create_wrapper(") == 2
    assert 'memfd_create_wrapper("batchgen_kv", MFD_CLOEXEC)' in source
    assert "O_RDWR | O_CLOEXEC" in source
    assert "enable_memfd" not in source
    assert "enable_memfd" not in BACKEND_HEADER.read_text()


def test_backend_errors_never_blame_fast_init():
    """memfd is unconditional, so no failure in it can be a fast-init failure."""
    body = BACKEND.read_text().split(
        "SharedState::Initialize(bool create_region) {", 1
    )[1]
    body = body.split("\nstd::vector<std::int32_t> ", 1)[0]

    assert "fast-init" not in body
    assert "fast_init" not in body


def test_backend_pins_base_page_size_unless_thp_is_requested():
    source = BACKEND.read_text()

    assert "enable_thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE" in source
    # Creator and attacher both advise, so one side cannot promote the mapping
    # while the other keeps it at the base page size.
    assert (
        source.count(
            "AdviseTransparentHugePages(mapped, total_bytes, config.enable_thp)"
        )
        == 2
    )

    # The 2 MiB prefault stays behind the THP opt-in.
    assert source.count("TouchPagesMultiThreaded(") == 2
    call = source.index("TouchPagesMultiThreaded(mapped")
    guard = source.rindex("if (config.enable_thp)", 0, call)
    assert 0 < call - guard < 120


def test_config_hash_covers_layout_only():
    header = BACKEND_HEADER.read_text()
    body = header.split("inline std::uint64_t HashHostKVConfig", 1)[1]
    body = body.split("class HostPagedKVBackend", 1)[0]
    combines = [line for line in body.splitlines() if "HashCombine(" in line]

    assert combines
    # enable_thp is a paging hint and memfd identity is per-process; neither may
    # make a creator and an attacher disagree.
    assert not any("enable_thp" in line for line in combines)
    assert not any("memfd" in line for line in combines)


def test_worker_host_kv_attach_is_not_gated_on_fast_init():
    guarded = _branches_guarded_by(WORKER, "fast_init")

    for name in HOST_KV_MEMFD_NAMES:
        assert not any(name in dump for dump in guarded), name

    source = WORKER.read_text()
    assert "worker_kv_config.memfd_creator_pid = args.kv_memfd_pid" in source
    assert "enable_memfd" not in source


@pytest.mark.parametrize(
    "method_name",
    ["_get_kv_memfd_pid", "_get_kv_memfd_fd", "_get_kv_aux_memfd_fd"],
)
def test_worker_manager_publishes_host_kv_memfd_in_every_mode(method_name):
    dump = _method_dump(WORKER_MANAGER, "WorkerManager", method_name)

    assert "fast_init" not in dump


def test_coordinator_always_forwards_the_creator_identity():
    source = COORDINATOR.read_text()

    assert "enable_memfd" not in source
    assert "primary_config.memfd_creator_pid = memfd_creator_pid" in source
    assert "aux_config.memfd_fd = aux_memfd_fd" in source
    assert not any(
        name in dump
        for dump in _branches_guarded_by(COORDINATOR, "enable_thp")
        for name in HOST_KV_MEMFD_NAMES
    )
