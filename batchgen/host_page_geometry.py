# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Page geometry of the running host, the single Python source for every
page-size dependent size (C++ mirror: ``core/host_page_geometry.h``, which
applies the same rules).

* base page: ``os.sysconf("SC_PAGESIZE")`` (4 KiB on x86_64; 4, 16 or 64 KiB
  on arm64)
* THP size: ``/sys/kernel/mm/transparent_hugepage/hpage_pmd_size`` (2 MiB on
  x86_64, 512 MiB on a 64 KiB-page arm64 kernel)
* hugetlb size: ``Hugepagesize`` in ``/proc/meminfo`` (the default huge page
  size that ``vm.nr_hugepages`` and ``MFD_HUGETLB`` use)

A shmem mapping (memfd or tmpfs file) is backed by THP only when the policy
that governs it allows it; its backing page is then the THP size, otherwise
the base page.

Stdlib only: imported by launch-layer modules that must not load torch.
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass
from typing import Optional

THP_SYSFS_DIR = "/sys/kernel/mm/transparent_hugepage"
MEMINFO_PATH = "/proc/meminfo"


class HostPageGeometryError(RuntimeError):
	"""The host does not provide the page geometry a caller requires."""


def _read(path: str) -> str:
	try:
		with open(path, "r", encoding="ascii") as handle:
			return handle.read()
	except FileNotFoundError:
		return ""


@dataclass(frozen=True)
class HostPageGeometry:
	base_page_bytes: int
	thp_pmd_bytes: int  # 0: the kernel has no THP
	hugetlb_page_bytes: int  # 0: the kernel has no hugetlb
	shmem_thp_policy: str  # active shmem_enabled mode, "" without THP

	@classmethod
	def from_text(
		cls,
		base_page_bytes: int,
		hpage_pmd_size_text: str,
		meminfo_text: str,
		shmem_enabled_text: str,
	) -> "HostPageGeometry":
		"""Parse the three host sources; empty THP text means no THP and a
		meminfo without ``Hugepagesize`` means no hugetlb."""
		if base_page_bytes <= 0 or base_page_bytes & (base_page_bytes - 1):
			raise HostPageGeometryError(
				f"HostPageGeometryError: base page size {base_page_bytes} is not a power of two"
			)
		pmd_text = hpage_pmd_size_text.strip()
		if pmd_text and not pmd_text.isdigit():
			raise HostPageGeometryError(
				f"HostPageGeometryError: unparsable hpage_pmd_size {hpage_pmd_size_text!r}"
			)
		hugetlb = 0
		for line in meminfo_text.splitlines():
			if not line.startswith("Hugepagesize:"):
				continue
			fields = line.split()
			if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
				raise HostPageGeometryError(
					f"HostPageGeometryError: unparsable meminfo line {line!r}"
				)
			hugetlb = int(fields[1]) * 1024
		match = re.search(r"\[([a-z_]+)\]", shmem_enabled_text)
		return cls(
			base_page_bytes=base_page_bytes,
			thp_pmd_bytes=int(pmd_text) if pmd_text else 0,
			hugetlb_page_bytes=hugetlb,
			shmem_thp_policy=match.group(1) if match else "",
		)

	@classmethod
	def read_host(
		cls, thp_sysfs_dir: str = THP_SYSFS_DIR, meminfo_path: str = MEMINFO_PATH
	) -> "HostPageGeometry":
		return cls.from_text(
			os.sysconf("SC_PAGESIZE"),
			_read(os.path.join(thp_sysfs_dir, "hpage_pmd_size")),
			_read(meminfo_path),
			_read(os.path.join(thp_sysfs_dir, "shmem_enabled")),
		)

	def shmem_thp_active(self, *, madvised: bool, mount_huge: Optional[str] = None) -> bool:
		"""Whether THP backs a shmem mapping.

		``mount_huge`` is the ``huge=`` option of the tmpfs mount holding the
		object, or None for the kernel's internal mount (memfd), which
		``shmem_enabled`` governs. ``deny``/``force`` override every mount.
		"""
		if self.thp_pmd_bytes == 0 or self.shmem_thp_policy == "deny":
			return False
		if self.shmem_thp_policy == "force":
			return True
		mode = self.shmem_thp_policy if mount_huge is None else mount_huge
		return mode in ("always", "within_size") or (mode == "advise" and madvised)

	def shmem_backing_bytes(self, *, madvised: bool, mount_huge: Optional[str] = None) -> int:
		if self.shmem_thp_active(madvised=madvised, mount_huge=mount_huge):
			return max(self.thp_pmd_bytes, self.base_page_bytes)
		return self.base_page_bytes

	def require_hugetlb_page_bytes(self) -> int:
		if self.hugetlb_page_bytes == 0:
			raise HostPageGeometryError(
				"HostPageGeometryError: /proc/meminfo has no Hugepagesize; "
				"this kernel provides no hugetlb pages"
			)
		return self.hugetlb_page_bytes

	def weights_region_backing_bytes(self, *, enable_hugetlbfs: bool, enable_thp: bool) -> int:
		"""Page backing the parameter server's weights memfd (posix_shm.cpp):
		the hugetlb size under --enable-hugetlbfs, else the shmem backing of a
		memfd, madvised under --fast-init."""
		if enable_hugetlbfs:
			return self.require_hugetlb_page_bytes()
		return self.shmem_backing_bytes(madvised=enable_thp)


@functools.lru_cache(maxsize=1)
def host_page_geometry() -> HostPageGeometry:
	"""The running host's geometry, read once per process."""
	return HostPageGeometry.read_host()


def read_hugepages_total(meminfo_path: str = MEMINFO_PATH) -> int:
	"""``HugePages_Total`` (pages of the default huge page size)."""
	for line in _read(meminfo_path).splitlines():
		if line.startswith("HugePages_Total:"):
			fields = line.split()
			if len(fields) == 2 and fields[1].isdigit():
				return int(fields[1])
			raise HostPageGeometryError(
				f"HostPageGeometryError: unparsable meminfo line {line!r}"
			)
	raise HostPageGeometryError(
		"HostPageGeometryError: /proc/meminfo has no HugePages_Total; "
		"this kernel provides no hugetlb pages"
	)
