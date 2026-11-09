"""Host CPU detection in op_builder: -march/-mcpu and SIMD width on x86_64 and
aarch64, with py-cpuinfo, lscpu, or only the kernel machine name available."""

import sys
import types

import pytest

from op_builder import builder as builder_mod
from op_builder.core_engine import CoreEngineBuilder

LSCPU_GRACE = """Architecture:                       aarch64
CPU op-mode(s):                     64-bit
Byte Order:                         Little Endian
CPU(s):                             144
Vendor ID:                          ARM
Model name:                         Neoverse-V2
Flags:                              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti
"""

LSCPU_X86_AVX512 = """Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Vendor ID:                          GenuineIntel
Flags:                              fpu vme sse sse2 avx avx2 avx512f avx512dq
"""

LSCPU_X86_AVX2_HYGON = """Architecture:                       x86_64
Vendor ID:                          HygonGenuine
Flags:                              fpu sse sse2 avx avx2
"""

LSCPU_RISCV = """Architecture:                       riscv64
Byte Order:                         Little Endian
"""


@pytest.fixture
def op():
	return CoreEngineBuilder()


def _no_cpuinfo(monkeypatch):
	monkeypatch.setitem(sys.modules, "cpuinfo", None)  # import raises ImportError


def _cpuinfo(monkeypatch, info=None, exc=None):
	mod = types.ModuleType("cpuinfo")

	def get_cpu_info():
		if exc is not None:
			raise exc
		return info

	mod.get_cpu_info = get_cpu_info
	monkeypatch.setitem(sys.modules, "cpuinfo", mod)


def _lscpu(monkeypatch, text):
	monkeypatch.setattr(builder_mod.OpBuilder, "command_exists", lambda self, cmd: text is not None)
	monkeypatch.setattr(builder_mod.subprocess, "check_output", lambda *a, **k: text.encode())


def _machine(monkeypatch, name):
	monkeypatch.setattr(builder_mod.platform, "machine", lambda: name)


def test_grace_without_py_cpuinfo_uses_lscpu(monkeypatch, op):
	# The GB300 crash: lscpu fallback returned arch None -> None.startswith.
	_no_cpuinfo(monkeypatch)
	_lscpu(monkeypatch, LSCPU_GRACE)
	_machine(monkeypatch, "x86_64")  # lscpu's Architecture line must win
	assert op.cpu_arch() == "-mcpu=native"
	assert op.simd_width() == "-D__SCALAR__"


def test_aarch64_from_py_cpuinfo(monkeypatch, op):
	_cpuinfo(monkeypatch, {"arch": "ARM_8", "flags": ["asimd", "sve2"]})
	assert op.cpu_arch() == "-mcpu=native"
	assert op.simd_width() == "-D__SCALAR__"


def test_aarch64_without_py_cpuinfo_or_lscpu(monkeypatch, op):
	_no_cpuinfo(monkeypatch)
	_lscpu(monkeypatch, None)
	monkeypatch.setattr(builder_mod.OpBuilder, "warning", lambda self, msg: None)
	_machine(monkeypatch, "aarch64")
	assert op.cpu_arch() == "-mcpu=native"
	assert op.simd_width() == "-D__SCALAR__"


def test_x86_avx512_from_py_cpuinfo(monkeypatch, op):
	_cpuinfo(monkeypatch, {"arch": "X86_64", "flags": ["avx2", "avx512f"]})
	assert op.cpu_arch() == "-march=native"
	assert op.simd_width() == "-D__AVX512__"


def test_x86_avx512_from_lscpu(monkeypatch, op):
	_no_cpuinfo(monkeypatch)
	_lscpu(monkeypatch, LSCPU_X86_AVX512)
	assert op.cpu_arch() == "-march=native"
	assert op.simd_width() == "-D__AVX512__"


def test_x86_vendor_outside_intel_amd_is_still_x86(monkeypatch, op):
	_no_cpuinfo(monkeypatch)
	_lscpu(monkeypatch, LSCPU_X86_AVX2_HYGON)
	assert op.cpu_arch() == "-march=native"
	assert op.simd_width() == "-D__AVX256__"


def test_py_cpuinfo_failure_falls_back_to_lscpu(monkeypatch, op):
	_cpuinfo(monkeypatch, exc=RuntimeError("no /proc/cpuinfo"))
	_lscpu(monkeypatch, LSCPU_GRACE)
	assert op.cpu_arch() == "-mcpu=native"


def test_py_cpuinfo_without_arch_falls_back_to_lscpu(monkeypatch, op):
	_cpuinfo(monkeypatch, {"flags": []})
	_lscpu(monkeypatch, LSCPU_X86_AVX512)
	assert op.cpu_arch() == "-march=native"


def test_ppc64le_keeps_mcpu(monkeypatch, op):
	_cpuinfo(monkeypatch, {"arch": "PPC_64", "flags": []})
	assert op.cpu_arch() == "-mcpu=native"


def test_unsupported_arch_is_a_named_error(monkeypatch, op):
	_no_cpuinfo(monkeypatch)
	_lscpu(monkeypatch, LSCPU_RISCV)
	with pytest.raises(builder_mod.UnsupportedCPUArchError, match="RISCV64"):
		op.cpu_arch()
