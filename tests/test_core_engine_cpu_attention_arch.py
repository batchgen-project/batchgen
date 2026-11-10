"""The AVX2 CPU attention kernel (attention modes 0 and 2) is built only on
x86_64: op_builder leaves its source out elsewhere, and Hetero_Attn guards the
kernel calls with __x86_64__ so a non-x86 build refuses those modes with
CPUAttentionUnavailableError instead of falling back to another mode."""

import re
from pathlib import Path

import pytest

from op_builder import core_engine as core_engine_mod
from op_builder.core_engine import X86_CPU_ATTENTION_SOURCE, CoreEngineBuilder

ROOT = Path(__file__).resolve().parents[1]
HETERO_ATTN = ROOT / "core" / "Hetero_Attn" / "Hetero_Attn.cpp"
CPU_KERNELS_H = ROOT / "core" / "Hetero_Attn" / "CPU_Kernels" / "cpu_kernels.h"
KERNEL_CALL = "grouped_query_attention_cpu_avx2("


def _sources_for(monkeypatch, arch):
	monkeypatch.setattr(
		CoreEngineBuilder, "_cpu_info", lambda self: {"arch": arch, "flags": ""}
	)
	return CoreEngineBuilder().sources()


def test_x86_keeps_the_avx2_source_in_its_original_place(monkeypatch):
	sources = _sources_for(monkeypatch, "X86_64")
	assert sources.count(X86_CPU_ATTENTION_SOURCE) == 1
	position = sources.index(X86_CPU_ATTENTION_SOURCE)
	assert sources[position - 1].endswith("KV_Storage/uva_copy_kernel.cu")
	assert sources[position + 1].endswith("allocator.cpp")
	assert len(sources) == 22


@pytest.mark.parametrize("arch", ["ARM_8", "PPC_64"])
def test_non_x86_builds_leave_the_avx2_source_out(monkeypatch, arch):
	x86 = _sources_for(monkeypatch, "X86_64")
	other = _sources_for(monkeypatch, arch)
	assert X86_CPU_ATTENTION_SOURCE not in other
	assert other == [s for s in x86 if s != X86_CPU_ATTENTION_SOURCE]


def test_no_source_of_an_aarch64_build_includes_x86_intrinsics(monkeypatch):
	for source in _sources_for(monkeypatch, "ARM_8"):
		text = (ROOT / source).read_text()
		assert "immintrin.h" not in text, source


def test_avx2_source_is_the_one_that_needs_x86_intrinsics():
	assert "#include <immintrin.h>" in (ROOT / X86_CPU_ATTENTION_SOURCE).read_text()


def test_source_selection_uses_the_builders_host_detection():
	# One detection path for -march/-mcpu and for the source list.
	body = core_engine_mod.CoreEngineBuilder.sources.__code__.co_names
	assert "_cpu_info" in body


_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def _preprocess(text, x86):
	"""Keep the lines a compiler keeps, evaluating only __x86_64__ tests.

	Every conditional in the guarded files is on ``defined(__x86_64__)``; any
	other condition makes this helper fail rather than guess.
	"""
	kept, stack = [], []
	for line in text.splitlines():
		match = _DIRECTIVE.match(line)
		if not match:
			if all(stack):
				kept.append(line)
			continue
		kind, rest = match.group(1), match.group(2).split("//")[0].strip()
		if kind == "if":
			if rest == "defined(__x86_64__)":
				stack.append(x86)
			elif rest == "!defined(__x86_64__)":
				stack.append(not x86)
			else:
				raise AssertionError(f"unexpected conditional: {line!r}")
		elif kind == "else":
			stack[-1] = not stack[-1]
		elif kind == "endif":
			stack.pop()
		else:
			raise AssertionError(f"unexpected conditional: {line!r}")
	assert not stack, "unbalanced conditionals"
	return "\n".join(kept)


@pytest.mark.parametrize("path", [HETERO_ATTN, CPU_KERNELS_H])
def test_non_x86_translation_units_never_name_the_avx2_kernel(path):
	assert KERNEL_CALL not in _preprocess(path.read_text(), x86=False)


def test_x86_translation_unit_keeps_both_kernel_calls_and_no_refusal():
	x86 = _preprocess(HETERO_ATTN.read_text(), x86=True)
	assert x86.count(KERNEL_CALL) == 2
	assert "throw_cpu_attention_unavailable" not in x86
	assert KERNEL_CALL in _preprocess(CPU_KERNELS_H.read_text(), x86=True)


def test_non_x86_build_refuses_modes_0_and_2_at_construction_and_call():
	arm = _preprocess(HETERO_ATTN.read_text(), x86=False)
	assert '"CPUAttentionUnavailableError: attn_mode "' in arm
	ctor = arm.split("Hetero_Attn::Hetero_Attn(", 1)[1].split("\n};", 1)[0]
	assert "this->attn_mode_ == 0 || this->attn_mode_ == 2" in ctor
	assert "throw_cpu_attention_unavailable(this->attn_mode_);" in ctor
	mode0 = arm.split("torch::Tensor Hetero_Attn::_attn_mode_0(", 1)[1].split("\n};", 1)[0]
	assert "throw_cpu_attention_unavailable(0);" in mode0
	cpu_mechanism = arm.split("Hetero_Attn::CPU_attn_mechanism(", 1)[1].split("\n};", 1)[0]
	assert "throw_cpu_attention_unavailable(2);" in cpu_mechanism
