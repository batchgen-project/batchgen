"""Host page geometry: every page-size dependent stride, alignment and rounding
derives from the running host (base page, THP size, hugetlb size, shmem THP
policy), never from a 2 MiB constant.

The table runs against the Python helper on any machine. With a compiled
core_engine on Linux it also runs against the C++ helper, and the C++ fd
probe is checked on a real memfd and a real tmpfs file of this host.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

from batchgen import host_page_geometry as hpg
from batchgen.host_page_geometry import HostPageGeometry, HostPageGeometryError

# Loaded by path: the batchgen.server package imports the HTTP stack, which
# the reservation logic does not need.
_SPEC = importlib.util.spec_from_file_location(
	"batchgen_process_utils_under_test",
	Path(__file__).resolve().parents[1] / "batchgen" / "server" / "process_utils.py",
)
process_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(process_utils)

KIB, MIB = 1024, 1024 * 1024
POLICIES = ("always", "within_size", "advise", "never", "deny", "force")


def _shmem_enabled(active):
	return " ".join(f"[{p}]" if p == active else p for p in POLICIES)


def _meminfo(hugepagesize_kib=None, total=None):
	lines = ["MemTotal:       1048576 kB"]
	if total is not None:
		lines.append(f"HugePages_Total: {total}")
	if hugepagesize_kib is not None:
		lines.append(f"Hugepagesize:    {hugepagesize_kib} kB")
	return "\n".join(lines) + "\n"


def x86_host(policy="always"):
	"""H20/H200: 4 KiB base pages, 2 MiB THP and hugetlb pages."""
	return (4096, "2097152\n", _meminfo(2048), _shmem_enabled(policy))


def gb300_host(policy="never"):
	"""GB300 (Grace, 64 KiB kernel): THP and hugetlb pages are 512 MiB."""
	return (65536, "536870912\n", _meminfo(524288), _shmem_enabled(policy))


def no_thp_no_hugetlb_host():
	return (4096, "", _meminfo(None), "")


def _implementations():
	impls = [pytest.param(HostPageGeometry.from_text, id="python")]
	try:
		import torch  # noqa: F401  (core_engine links libc10; torch loads it)
		from batchgen.native_loader import core_engine

		impls.append(pytest.param(core_engine.HostPageGeometry.from_text, id="cpp"))
	except Exception:
		impls.append(
			pytest.param(
				None, id="cpp", marks=pytest.mark.skip(reason="compiled core_engine unavailable")
			)
		)
	return impls


@pytest.fixture(params=_implementations())
def from_text(request):
	return request.param


# --- x86 parity: on a 4 KiB-page, 2 MiB-THP host every derived size equals
# --- the constant it replaces.

OLD_2MIB_CONSTANT = 2 * MIB


@pytest.mark.parametrize("policy", ["always", "within_size"])
def test_x86_thp_host_derives_every_former_2mib_constant(from_text, policy):
	g = from_text(*x86_host(policy))
	assert g.base_page_bytes == 4096
	# posix_shm.cpp weights memfd: size/address alignment and --fast-init
	# touch stride (were huge_page_size = 2 MiB).
	assert g.shmem_backing_bytes(madvised=True) == OLD_2MIB_CONSTANT
	# host_paged_kv_backend.cpp data_offset, mapping alignment and touch
	# stride (were kHugePageAlignment / kHugePageSize = 2 MiB), both with and
	# without --fast-init under this policy.
	assert g.shmem_backing_bytes(madvised=False) == OLD_2MIB_CONSTANT
	# Weights_Storage.cpp pin alignment of the staged store on a
	# huge=within_size tmpfs (was kPinAlignment = 2 MiB).
	assert g.shmem_backing_bytes(madvised=True, mount_huge="within_size") == OLD_2MIB_CONSTANT
	# posix_shm.cpp / process_utils.py hugetlb size (fallback was 2 MiB).
	assert g.hugetlb_page_bytes == OLD_2MIB_CONSTANT


def test_x86_fast_init_under_advise_policy_keeps_2mib(from_text):
	g = from_text(*x86_host("advise"))
	assert g.shmem_backing_bytes(madvised=True) == OLD_2MIB_CONSTANT
	# Unadvised under "advise", THP never backs the memfd: base pages.
	assert g.shmem_backing_bytes(madvised=False) == 4096


def test_python_weights_region_on_x86_equals_former_constant():
	g = HostPageGeometry.from_text(*x86_host("always"))
	for hugetlbfs in (False, True):
		for thp in (False, True):
			assert (
				g.weights_region_backing_bytes(enable_hugetlbfs=hugetlbfs, enable_thp=thp)
				== OLD_2MIB_CONSTANT
			)


# --- GB300: 64 KiB base pages, 512 MiB THP/hugetlb pages.


def test_gb300_with_thp_off_uses_64k_base_pages(from_text):
	g = from_text(*gb300_host("never"))
	assert (g.base_page_bytes, g.thp_pmd_bytes, g.hugetlb_page_bytes) == (
		64 * KIB,
		512 * MIB,
		512 * MIB,
	)
	assert g.shmem_thp_policy == "never"
	for madvised in (False, True):
		assert g.shmem_thp_active(madvised=madvised) is False
		assert g.shmem_backing_bytes(madvised=madvised) == 64 * KIB


@pytest.mark.parametrize("policy", ["always", "within_size"])
def test_gb300_with_shmem_thp_on_uses_512m_pages(from_text, policy):
	g = from_text(*gb300_host(policy))
	assert g.shmem_backing_bytes(madvised=True) == 512 * MIB
	assert g.shmem_backing_bytes(madvised=False) == 512 * MIB


# --- policy rules shared by both implementations.


@pytest.mark.parametrize(
	"policy, madvised, mount_huge, active",
	[
		("always", False, None, True),
		("within_size", False, None, True),
		("advise", True, None, True),
		("advise", False, None, False),
		("never", True, None, False),
		("deny", True, None, False),
		("deny", True, "always", False),  # deny overrides every mount
		("force", False, None, True),
		("force", False, "never", True),  # force overrides every mount
		("never", True, "within_size", True),  # a tmpfs mount has its own huge=
		("always", True, "never", False),
		("never", True, "advise", True),
		("never", False, "advise", False),
	],
)
def test_shmem_thp_policy_table(from_text, policy, madvised, mount_huge, active):
	g = from_text(*x86_host(policy))
	assert g.shmem_thp_active(madvised=madvised, mount_huge=mount_huge) is active
	expected = OLD_2MIB_CONSTANT if active else 4096
	assert g.shmem_backing_bytes(madvised=madvised, mount_huge=mount_huge) == expected


def test_kernel_without_thp_or_hugetlb(from_text):
	g = from_text(*no_thp_no_hugetlb_host())
	assert (g.thp_pmd_bytes, g.hugetlb_page_bytes, g.shmem_thp_policy) == (0, 0, "")
	assert g.shmem_backing_bytes(madvised=True, mount_huge="always") == 4096


def test_missing_hugetlb_is_a_named_error_python():
	g = HostPageGeometry.from_text(*no_thp_no_hugetlb_host())
	with pytest.raises(HostPageGeometryError, match="HostPageGeometryError: .*no Hugepagesize"):
		g.require_hugetlb_page_bytes()
	with pytest.raises(HostPageGeometryError):
		g.weights_region_backing_bytes(enable_hugetlbfs=True, enable_thp=False)


@pytest.mark.parametrize(
	"args",
	[
		(3000, "2097152", _meminfo(2048), ""),
		(0, "2097152", _meminfo(2048), ""),
		(4096, "2M", _meminfo(2048), ""),
		(4096, "2097152", "Hugepagesize:    2048 MB\n", ""),
		(4096, "2097152", "Hugepagesize:\n", ""),
	],
)
def test_malformed_host_text_is_a_named_error(from_text, args):
	with pytest.raises(Exception, match="HostPageGeometryError"):
		from_text(*args)


def test_read_host_reads_sysfs_and_meminfo(tmp_path):
	thp = tmp_path / "thp"
	thp.mkdir()
	(thp / "hpage_pmd_size").write_text("536870912\n")
	(thp / "shmem_enabled").write_text(_shmem_enabled("within_size") + "\n")
	meminfo = tmp_path / "meminfo"
	meminfo.write_text(_meminfo(524288, total=3))
	g = HostPageGeometry.read_host(str(thp), str(meminfo))
	assert g.base_page_bytes == os.sysconf("SC_PAGESIZE")
	assert (g.thp_pmd_bytes, g.hugetlb_page_bytes, g.shmem_thp_policy) == (
		512 * MIB,
		512 * MIB,
		"within_size",
	)
	assert hpg.read_hugepages_total(str(meminfo)) == 3


def test_read_host_without_thp_sysfs(tmp_path):
	meminfo = tmp_path / "meminfo"
	meminfo.write_text(_meminfo(None))
	g = HostPageGeometry.read_host(str(tmp_path / "absent"), str(meminfo))
	assert (g.thp_pmd_bytes, g.hugetlb_page_bytes, g.shmem_thp_policy) == (0, 0, "")
	with pytest.raises(HostPageGeometryError, match="HugePages_Total"):
		hpg.read_hugepages_total(str(meminfo))


# --- huge page pool reservation (worker_manager._config_hugepages).


def _reserve(monkeypatch, page_bytes, total_after, sysctl_error=None):
	calls = []

	def run(cmd, **kwargs):
		calls.append(cmd)
		if sysctl_error is not None:
			raise sysctl_error

	monkeypatch.setattr(process_utils, "get_hugepage_size", lambda: page_bytes)
	monkeypatch.setattr(process_utils, "read_hugepages_total", lambda: total_after)
	return calls, run


def test_reservation_counts_pages_of_the_host_size(monkeypatch):
	calls, run = _reserve(monkeypatch, 2 * MIB, total_after=358400)
	assert process_utils.reserve_hugepages(700 * 1024 * MIB, run=run) == 358400
	assert calls == [["sysctl", "-w", "vm.nr_hugepages=358400"]]
	calls, run = _reserve(monkeypatch, 512 * MIB, total_after=1400)
	assert process_utils.reserve_hugepages(700 * 1024 * MIB, run=run) == 1400
	assert calls == [["sysctl", "-w", "vm.nr_hugepages=1400"]]


def test_reservation_shortfall_fails_loudly(monkeypatch):
	_, run = _reserve(monkeypatch, 512 * MIB, total_after=1399)
	with pytest.raises(process_utils.HugePagesReservationError, match="1400 huge pages of 512 MiB"):
		process_utils.reserve_hugepages(700 * 1024 * MIB, run=run)


def test_reservation_trusts_a_pool_the_operator_already_reserved(monkeypatch):
	_, run = _reserve(monkeypatch, 2 * MIB, total_after=400000, sysctl_error=PermissionError("ro"))
	assert process_utils.reserve_hugepages(700 * 1024 * MIB, run=run) == 358400


def test_reservation_refuses_failed_sysctl_without_a_pool(monkeypatch):
	_, run = _reserve(monkeypatch, 2 * MIB, total_after=0, sysctl_error=PermissionError("ro"))
	with pytest.raises(process_utils.HugePagesReservationError):
		process_utils.reserve_hugepages(700 * 1024 * MIB, run=run)


@pytest.mark.parametrize("byte_size", [None, 0, -1])
def test_reservation_needs_a_size(monkeypatch, byte_size):
	_, run = _reserve(monkeypatch, 2 * MIB, total_after=10)
	with pytest.raises(ValueError):
		process_utils.reserve_hugepages(byte_size, run=run)


def test_hugepage_size_has_no_assumed_fallback(monkeypatch):
	monkeypatch.setattr(
		process_utils,
		"host_page_geometry",
		lambda: HostPageGeometry.from_text(*no_thp_no_hugetlb_host()),
	)
	with pytest.raises(HostPageGeometryError):
		process_utils.get_hugepage_size()
	assert not hasattr(process_utils, "DEFAULT_HUGEPAGE_SIZE")


# --- the C++ fd probe on this host (Linux with a compiled core_engine).


def _core_engine():
	try:
		import torch  # noqa: F401  (core_engine links libc10; torch loads it)
		from batchgen.native_loader import core_engine
	except Exception as exc:
		pytest.skip(f"compiled core_engine unavailable: {exc}")
	return core_engine


def test_cpp_and_python_read_the_same_host():
	ce = _core_engine()
	cpp, py = ce.HostPageGeometry.host(), hpg.HostPageGeometry.read_host()
	assert (cpp.base_page_bytes, cpp.thp_pmd_bytes, cpp.hugetlb_page_bytes, cpp.shmem_thp_policy) == (
		py.base_page_bytes,
		py.thp_pmd_bytes,
		py.hugetlb_page_bytes,
		py.shmem_thp_policy,
	)


@pytest.mark.parametrize("madvised", [False, True])
def test_cpp_memfd_backing_follows_shmem_enabled(madvised):
	ce = _core_engine()
	fd = os.memfd_create("batchgen-geometry-test")
	try:
		expected = hpg.HostPageGeometry.read_host().shmem_backing_bytes(madvised=madvised)
		assert ce.mapping_backing_bytes(fd, madvised) == expected
	finally:
		os.close(fd)


def _mount_huge_option(path):
	st = os.stat(path)
	dev = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
	with open("/proc/self/mountinfo") as handle:
		for line in handle:
			fields = line.split()
			if fields[2] != dev:
				continue
			tail = line.split(" - ", 1)[1].split()
			if tail[0] != "tmpfs":
				continue
			for option in tail[2].split(","):
				if option.startswith("huge="):
					return option[5:]
			return "never"
	return None


def test_cpp_tmpfs_file_backing_follows_its_mount():
	ce = _core_engine()
	if not os.path.isdir("/dev/shm"):
		pytest.skip("no /dev/shm")
	with tempfile.NamedTemporaryFile(dir="/dev/shm") as handle:
		mount_huge = _mount_huge_option(handle.name)
		assert mount_huge is not None, "/dev/shm is expected to be a visible tmpfs mount"
		expected = hpg.HostPageGeometry.read_host().shmem_backing_bytes(
			madvised=True, mount_huge=mount_huge
		)
		assert ce.mapping_backing_bytes(handle.fileno(), True) == expected
