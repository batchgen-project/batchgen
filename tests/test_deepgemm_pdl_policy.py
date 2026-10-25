import ast
import sys
import types
from pathlib import Path

import pytest

from batchgen.gemm import deepgemm_pdl_policy
from batchgen.gemm.deepgemm_pdl_policy import (
	DeepGemmPdlStateError,
	DeepGemmPdlUnsupportedError,
	apply_deepgemm_pdl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_fake_deep_gemm(monkeypatch, *, with_toggle=True):
	calls = []
	mod = types.ModuleType("deep_gemm")
	if with_toggle:
		state = {"pdl": False}

		def set_pdl(value):
			calls.append(value)
			state["pdl"] = value

		mod.set_pdl = set_pdl
		mod.get_pdl = lambda: state["pdl"]
	monkeypatch.setitem(sys.modules, "deep_gemm", mod)
	return calls


def test_switch_defaults_on():
	assert deepgemm_pdl_policy.DEEPGEMM_PDL is True


def test_apply_enables_pdl(monkeypatch):
	calls = _install_fake_deep_gemm(monkeypatch)
	monkeypatch.setattr(deepgemm_pdl_policy, "DEEPGEMM_PDL", True)
	assert apply_deepgemm_pdl() is True
	assert calls == [True]


def test_switch_off_makes_no_deep_gemm_call(monkeypatch):
	calls = _install_fake_deep_gemm(monkeypatch)
	monkeypatch.setattr(deepgemm_pdl_policy, "DEEPGEMM_PDL", False)
	assert apply_deepgemm_pdl() is False
	assert calls == []


def test_switch_off_never_imports_deep_gemm(monkeypatch):
	monkeypatch.setattr(deepgemm_pdl_policy, "DEEPGEMM_PDL", False)
	monkeypatch.delitem(sys.modules, "deep_gemm", raising=False)
	assert apply_deepgemm_pdl() is False
	assert "deep_gemm" not in sys.modules


def test_missing_toggle_fails_loud(monkeypatch):
	_install_fake_deep_gemm(monkeypatch, with_toggle=False)
	monkeypatch.setattr(deepgemm_pdl_policy, "DEEPGEMM_PDL", True)
	with pytest.raises(DeepGemmPdlUnsupportedError):
		apply_deepgemm_pdl()


def test_ineffective_set_pdl_fails_loud(monkeypatch):
	_install_fake_deep_gemm(monkeypatch)
	mod = sys.modules["deep_gemm"]
	mod.set_pdl = lambda value: None  # accepted but ignored
	monkeypatch.setattr(deepgemm_pdl_policy, "DEEPGEMM_PDL", True)
	with pytest.raises(DeepGemmPdlStateError):
		apply_deepgemm_pdl()


def test_worker_wires_pdl_before_any_model_work():
	"""apply_deepgemm_pdl() must run in worker __init__ right after set_device."""
	src = (REPO_ROOT / "batchgen" / "batchgen_worker.py").read_text()
	tree = ast.parse(src)
	call_lines = [
		node.lineno
		for node in ast.walk(tree)
		if isinstance(node, ast.Call)
		and isinstance(node.func, ast.Name)
		and node.func.id == "apply_deepgemm_pdl"
	]
	assert len(call_lines) == 1, "expected exactly one apply_deepgemm_pdl() call"
	set_device_lines = [
		node.lineno
		for node in ast.walk(tree)
		if isinstance(node, ast.Call)
		and isinstance(node.func, ast.Attribute)
		and node.func.attr == "set_device"
		and ast.unparse(node.args[0]) == "self.local_rank"
	]
	assert set_device_lines, "worker __init__ set_device(self.local_rank) not found"
	assert call_lines[0] > min(set_device_lines)
	assert call_lines[0] - min(set_device_lines) < 10, (
		"apply_deepgemm_pdl() drifted away from the set_device init seam"
	)
