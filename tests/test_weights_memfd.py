"""Model weights and tensor metadata must never be named objects.

A named region outlives the processes that mapped it, so a crashed server can
leave the whole model weight budget pinned until something unlinks the name.
Both regions are therefore anonymous memfds in every mode: `--enable-hugetlbfs`
only takes the weights memfd from the host's huge page pool instead of base
pages, and `--fast-init` only adds transparent huge pages and prefaulting.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
POSIX_SHM = ROOT / "core" / "Parameter_Server" / "posix_shm.cpp"
POSIX_SHM_HEADER = ROOT / "core" / "Parameter_Server" / "posix_shm.h"
PARAMETER_SERVER = ROOT / "core" / "Parameter_Server" / "Parameter_Server.cpp"
PARAMETER_SERVER_HEADER = ROOT / "core" / "Parameter_Server" / "Parameter_Server.h"
WEIGHTS_STORAGE = ROOT / "core" / "Weights_Storage" / "Weights_Storage.cpp"
WEIGHTS_STORAGE_HEADER = ROOT / "core" / "Weights_Storage" / "Weights_Storage.h"
BINDING = ROOT / "core" / "batchgen_Binding.cpp"
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"
PROCESS_UTILS = ROOT / "batchgen" / "server" / "process_utils.py"

PARAMETER_SERVERS = sorted(
    (ROOT / "batchgen" / "models").glob("**/*parameter_server.py")
)

# Weight-storage-only identity fields. Host KV memfd plumbing keeps its own
# names and is covered by tests/test_host_kv_memfd.py.
WEIGHTS_MEMFD_NAMES = (
    "weights_memfd_pid",
    "weights_memfd_fd",
    "tensor_meta_memfd_fd",
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


def test_parameter_server_sources_never_open_a_named_shm_object():
    source = POSIX_SHM.read_text()

    assert "shm_open" not in source
    assert "shm_unlink" not in source
    assert "shm_open" not in PARAMETER_SERVER.read_text()
    assert "shm_unlink" not in PARAMETER_SERVER.read_text()
    # The hugetlbfs mode used to open /dev/hugepages/<name>, which survived a
    # crash until the destructor ran.
    assert "/dev/hugepages" not in source
    assert "unlink" not in source
    # One definition plus exactly two call sites: the weights region and the
    # tensor-metadata region, both created unconditionally.
    assert source.count("create_anonymous_memfd(") == 3
    assert 'create_anonymous_memfd("batchgen_weights")' in source
    assert 'create_anonymous_memfd("batchgen_tensor_meta")' in source
    assert "syscall(SYS_memfd_create, name, MFD_CLOEXEC)" in source
    # Attachers open the creator's fd through /proc, so their handles must not
    # survive an exec either.
    assert "O_RDWR | O_CLOEXEC" in source
    assert "O_RDONLY | O_CLOEXEC" in source


def test_weight_allocation_failures_never_blame_fast_init():
    """memfd is unconditional, so no failure in it can be a fast-init failure."""
    lines = POSIX_SHM.read_text().splitlines()

    # The only remaining fast-init mentions are the prefault it still owns.
    for line in lines:
        if "fast-init" in line and not line.lstrip().startswith("//"):
            assert "touch" in line, line


def test_weights_follow_system_page_size_unless_thp_is_requested():
    source = POSIX_SHM.read_text()

    # Without --fast-init nothing is advised: the host's shmem_enabled decides.
    assert "MADV_NOHUGEPAGE" not in source
    assert "if (!enable_thp) {" in source
    # Creator and attacher advise identically, except that THP cannot apply to
    # a mapping whose pages already come from the huge page pool.
    assert (
        source.count("advise_transparent_huge_pages(ptr, allocated_size, enable_thp)")
        == 1
    )
    assert "enable_thp && !fd_is_hugetlb);" in source

    # The 2 MiB prefault stays behind the THP opt-in; without it the mapping
    # keeps the base-page-stride touch.
    assert source.count("if (enable_thp) {") == 1
    thp_block = source.split("if (enable_thp) {", 1)[1]
    thp_block = thp_block.split("} else if (!touch_pages", 1)[0]
    assert "touch_pages(ptr, allocated_size, huge_page_size" in thp_block
    assert "} else if (!touch_pages(ptr, size, page_size, true)) {" in source


def test_allocator_signature_drops_named_shm_ownership():
    header = POSIX_SHM_HEADER.read_text()

    assert "bool enable_thp = false," in header
    assert "out_posix_shm_owned" not in header
    assert "out_posix_shm_owned" not in POSIX_SHM.read_text()
    assert "int serialize_to_memfd(" in header
    assert "deserialize_from_memfd(int memfd_creator_pid, int memfd_fd)" in header
    assert "serialize_to_shared_memory" not in header
    assert "deserialize_from_shared_memory" not in header


def test_parameter_server_owns_both_memfds_and_unlinks_nothing():
    header = PARAMETER_SERVER_HEADER.read_text()
    source = PARAMETER_SERVER.read_text()

    assert "Parameter_Server(bool enable_hugetlbfs, bool enable_thp = false);" in header
    assert "int tensor_meta_memfd_fd() const" in header
    assert "weight_posix_shm_owned_" not in header
    assert "tensor_meta_shm_owned_" not in header
    # Nothing names the hugetlbfs region any more, so no path is tracked for it.
    assert "weight_hugetlbfs_owned_" not in header
    assert "weight_hugetlbfs_path_" not in header

    assert "serialize_to_memfd(this->module_weights_storage_)" in source
    destructor = source.split("Parameter_Server::~Parameter_Server()", 1)[1]
    destructor = destructor.split("Parameter_Server::get_skeleton_state_dict", 1)[0]
    assert "close(this->weights_memfd_fd_)" in destructor
    assert "close(this->tensor_meta_memfd_fd_)" in destructor
    assert "unlink(" not in destructor


def test_hugetlbfs_weights_are_an_unnamed_huge_page_memfd():
    source = POSIX_SHM.read_text()
    header = POSIX_SHM_HEADER.read_text()

    assert "syscall(SYS_memfd_create, name, MFD_CLOEXEC | MFD_HUGETLB)" in source
    assert 'create_huge_anonymous_memfd("batchgen_weights")' in source
    # The huge page size is the host's default, so the region is rounded and
    # mapped to exactly that, not to a hardcoded 2 MB.
    assert "default_huge_page_size()" in source
    assert "ftruncate64(fd, hugetlb_size)" in source
    # Only the creator allocates; an attacher cannot name a memfd, so it takes
    # the same /proc path in both modes.
    assert "if (enable_hugetlbfs && create) {" in source
    # The ownership bookkeeping a named file needed is gone from the signature.
    for dead in ("out_hugetlbfs_owned", "out_hugetlbfs_path", "hugepage_path"):
        assert dead not in source, dead
        assert dead not in header, dead


def test_hugepage_setup_reserves_the_pool_without_mounting_hugetlbfs():
    source = WORKER_MANAGER.read_text()
    config = source.split("def _config_hugepages(", 1)[1]
    config = config.split("\n    def ", 1)[0]

    # memfd needs the reservation, which stays host/operator configuration.
    assert '["sysctl", "-w", f"vm.nr_hugepages={num_hugepages}"]' in config
    # It needs no mount point, so the server no longer creates or mounts one.
    assert '"mount"' not in config
    assert '"mkdir"' not in config
    assert "/dev/hugepages" not in source


def test_worker_side_attaches_both_regions_through_the_creator_fds():
    header = WEIGHTS_STORAGE_HEADER.read_text()
    source = WEIGHTS_STORAGE.read_text()

    assert "int tensor_meta_memfd_fd = -1);" in header
    assert "deserialize_from_memfd(memfd_creator_pid, tensor_meta_memfd_fd)" in source
    assert "enable_thp, memfd_creator_pid, memfd_fd_arg," in source
    assert "deserialize_from_shared_memory" not in source


def test_bindings_expose_thp_and_the_metadata_fd():
    source = BINDING.read_text()

    assert source.count('py::arg("enable_thp") = false') == 3
    assert source.count('py::arg("tensor_meta_memfd_fd") = -1') == 2
    assert '.def("tensor_meta_memfd_fd", &Parameter_Server::tensor_meta_memfd_fd)' in source


def test_worker_weight_attach_is_not_gated_on_fast_init():
    guarded = _branches_guarded_by(WORKER, "fast_init")

    for name in WEIGHTS_MEMFD_NAMES:
        assert not any(name in dump for dump in guarded), name

    source = WORKER.read_text()
    assert "memfd_creator_pid=args.weights_memfd_pid" in source
    assert "memfd_fd=args.weights_memfd_fd" in source
    assert "tensor_meta_memfd_fd=args.tensor_meta_memfd_fd" in source
    assert "enable_thp=args.fast_init" in source


@pytest.mark.parametrize(
    "method_name",
    [
        "_get_weights_memfd_pid",
        "_get_weights_memfd_fd",
        "_get_tensor_meta_memfd_fd",
    ],
)
def test_worker_manager_publishes_weight_memfds_in_every_mode(method_name):
    dump = _method_dump(WORKER_MANAGER, "WorkerManager", method_name)

    assert "fast_init" not in dump


def test_worker_manager_forwards_the_metadata_fd_to_workers():
    source = WORKER_MANAGER.read_text()

    assert "tensor_meta_memfd_fd=self._get_tensor_meta_memfd_fd()," in source
    # --fast-init now means THP only, for every parameter server.
    assert "enable_memfd" not in source
    assert source.count("enable_thp=self.args.fast_init,") == 9


def test_model_parameter_servers_exist():
    assert len(PARAMETER_SERVERS) == 8


@pytest.mark.parametrize(
    "path", PARAMETER_SERVERS, ids=lambda path: path.parent.name
)
def test_model_parameter_servers_forward_thp_and_drop_the_shm_check(path):
    source = path.read_text()

    assert "enable_thp" in source
    assert "enable_memfd" not in source
    assert "self.enable_thp" in source
    # memfd does not live on /dev/shm, so its free space cannot gate the weights.
    assert 'disk_usage("/dev/shm")' not in source


def test_release_path_has_no_named_model_region_left_to_verify():
    process_utils = PROCESS_UTILS.read_text()
    manager = WORKER_MANAGER.read_text()

    # No mode leaves a model region on disk, so there is nothing to record and
    # nothing whose absence could be proven.
    for dead in (
        "MODEL_SHM_KEYS",
        "MODEL_NAMED_SHM_KEYS",
        "MODEL_SHM_PROVENANCE_FILE",
        "model_shm.json",
        "record_model_shm_provenance",
        "verify_model_shm_absent",
        "hugepages_dir",
    ):
        assert dead not in process_utils, dead
        assert dead not in manager, dead

    # The owner still has to be dropped, which is what closes the memfds.
    assert "self.parameter_server_instance = None" in manager
    assert 'self.model_info.pop("shm_name", None)' in manager
    assert 'self.model_info.pop("tensor_meta_shm_name", None)' in manager


_HUGE_MEMFD_PROGRAM = """
import ctypes
import mmap
import os
import sys

MFD_CLOEXEC = 0x0001
MFD_HUGETLB = 0x0004

libc = ctypes.CDLL(None, use_errno=True)
libc.memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
fd = libc.memfd_create(b"batchgen_weights", MFD_CLOEXEC | MFD_HUGETLB)
if fd < 0:
    sys.exit(f"memfd_create(MFD_HUGETLB) failed: {os.strerror(ctypes.get_errno())}")
size = int(sys.argv[1])
os.ftruncate(fd, size)
region = mmap.mmap(fd, size, flags=mmap.MAP_SHARED)
region[0] = 7
# The region is unnamed: only the creator's fd reaches it.
assert "memfd:batchgen_weights" in os.readlink(f"/proc/self/fd/{fd}")
assert not os.path.exists("/dev/hugepages/batchgen_weights")
region.close()
os.close(fd)
"""


def _free_huge_pages() -> tuple[int, int]:
    """Return (free huge pages, huge page size in bytes) from /proc/meminfo."""
    free, size = 0, 0
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("HugePages_Free:"):
                free = int(line.split()[1])
            elif line.startswith("Hugepagesize:"):
                size = int(line.split()[1]) * 1024
    return free, size


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="MFD_HUGETLB requires Linux"
)
def test_native_huge_page_memfd_carries_no_name():
    """The real syscall, when the host actually has huge pages reserved."""
    free, huge_page_size = _free_huge_pages()
    if huge_page_size == 0 or free < 4:
        pytest.skip(f"host has {free} free huge pages; need 4")

    # A child process keeps the huge page mapping out of this process.
    result = subprocess.run(
        [sys.executable, "-c", _HUGE_MEMFD_PROGRAM, str(2 * huge_page_size)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
