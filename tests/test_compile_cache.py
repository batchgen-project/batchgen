"""Compile-cache wiring for the native builds (batchgen_kernels/_compile_cache.py,
scripts/cache_cxx.sh): CXX must be one executable on every supported torch."""

import importlib.util
import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CXX_WRAPPER = os.path.realpath(SCRIPTS / "cache_cxx.sh")
NVCC_WRAPPER = os.path.realpath(SCRIPTS / "cache_nvcc.sh")


def _load():
	spec = importlib.util.spec_from_file_location(
		"compile_cache_under_test", ROOT / "batchgen_kernels" / "_compile_cache.py"
	)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


cc = _load()


def _which(*present):
	return lambda tool: f"/usr/bin/{tool}" if tool in present else None


def test_ccache_installs_single_executable_cxx():
	env = {}
	tool = cc.configure_compile_cache(env, str(SCRIPTS), "/cuda", which=_which("ccache"))
	assert tool == "ccache"
	assert env["CXX"] == CXX_WRAPPER
	assert len(shlex.split(env["CXX"])) == 1
	assert env["CACHE_CXX_LAUNCHER"] == "ccache"
	assert env["TORCH_NO_COMPILER_WRAPPER"] == "1"
	assert env["PYTORCH_NVCC"] == f"{NVCC_WRAPPER} ccache /cuda/bin/nvcc"


def test_sccache_is_preferred():
	env = {}
	assert cc.configure_compile_cache(env, str(SCRIPTS), "/cuda", which=_which("ccache", "sccache")) == "sccache"
	assert env["CACHE_CXX_LAUNCHER"] == "sccache"


@pytest.mark.parametrize("cxx", ["ccache g++", "sccache g++", "g++ -O2"])
def test_multi_word_cxx_is_refused(cxx):
	env = {"CXX": cxx}
	with pytest.raises(cc.MultiWordCompilerError, match="more than one program"):
		cc.configure_compile_cache(env, str(SCRIPTS), "/cuda", which=_which("ccache"))
	assert env == {"CXX": cxx}


def test_multi_word_cxx_is_refused_without_cache_tool():
	with pytest.raises(cc.MultiWordCompilerError):
		cc.configure_compile_cache({"CXX": "ccache g++"}, str(SCRIPTS), "/cuda", which=_which())


def test_caller_compiler_is_kept_and_torch_launcher_left_on():
	env = {"CXX": "clang++"}
	assert cc.configure_compile_cache(env, str(SCRIPTS), "/cuda", which=_which("ccache")) == "ccache"
	assert env["CXX"] == "clang++"
	assert "TORCH_NO_COMPILER_WRAPPER" not in env
	assert "CACHE_CXX_LAUNCHER" not in env
	assert env["PYTORCH_NVCC"].startswith(NVCC_WRAPPER)


def test_caller_nvcc_is_kept():
	env = {"PYTORCH_NVCC": "/my/nvcc"}
	cc.configure_compile_cache(env, str(SCRIPTS), "/cuda", which=_which("ccache"))
	assert env["PYTORCH_NVCC"] == "/my/nvcc"


def test_no_cache_tool_configures_nothing():
	env = {}
	assert cc.configure_compile_cache(env, str(SCRIPTS), "/cuda", which=_which()) is None
	assert env == {}


def test_outside_checkout_configures_nothing(tmp_path):
	env = {}
	assert cc.configure_compile_cache(env, str(tmp_path), "/cuda", which=_which("ccache")) is None
	assert env == {}


def _fake_launcher(tmp_path):
	launcher = tmp_path / "fakecache"
	launcher.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
	launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
	return launcher


def test_cache_cxx_runs_launcher_with_compiler(tmp_path):
	launcher = _fake_launcher(tmp_path)
	env = dict(os.environ, CACHE_CXX_LAUNCHER=str(launcher))
	env.pop("CACHE_CXX_COMPILER", None)
	out = subprocess.run(
		[CXX_WRAPPER, "-MMD", "-c", "a b.cpp", "-o", "x.o"],
		env=env, check=True, capture_output=True, text=True,
	).stdout.splitlines()
	assert out == ["g++", "-MMD", "-c", "a b.cpp", "-o", "x.o"]
	env["CACHE_CXX_COMPILER"] = "clang++"
	out = subprocess.run([CXX_WRAPPER, "-v"], env=env, check=True, capture_output=True, text=True).stdout
	assert out.splitlines() == ["clang++", "-v"]


def test_cache_cxx_without_launcher_fails_loudly():
	env = dict(os.environ)
	env.pop("CACHE_CXX_LAUNCHER", None)
	proc = subprocess.run([CXX_WRAPPER, "-c", "x.cpp"], env=env, capture_output=True, text=True)
	assert proc.returncode != 0
	assert "CACHE_CXX_LAUNCHER" in proc.stderr


def test_torch_ninja_file_has_exactly_one_launcher(tmp_path, monkeypatch):
	"""Real torch boundary: the generated build.ninja runs the wrapper directly
	(torch >= 2.15 would otherwise prepend ccache found on PATH)."""
	cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
	bindir = tmp_path / "bin"
	bindir.mkdir()
	for tool in ("ccache", "sccache"):
		fake = bindir / tool
		fake.write_text("#!/bin/sh\n")
		fake.chmod(0o755)
	monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
	env = {}
	assert cc.configure_compile_cache(env, str(SCRIPTS), "/cuda") == "sccache"
	for var, value in env.items():
		monkeypatch.setenv(var, value)

	ninja = tmp_path / "build.ninja"
	cpp_extension._write_ninja_file(
		path=str(ninja), cflags=["-O2"], post_cflags=None, cuda_cflags=None,
		cuda_post_cflags=None, cuda_dlink_post_cflags=None, sycl_cflags=None,
		sycl_post_cflags=None, sycl_dlink_post_cflags=None,
		sources=[str(tmp_path / "a.cpp")], objects=[str(tmp_path / "a.o")],
		ldflags=None, library_target=None, with_cuda=True, with_sycl=False,
	)
	text = ninja.read_text()
	cxx_line = re.search(r"^cxx = (.*)$", text, re.MULTILINE).group(1)
	nvcc_line = re.search(r"^nvcc = (.*)$", text, re.MULTILINE).group(1)
	assert shlex.split(cxx_line) == [CXX_WRAPPER]
	assert shlex.split(nvcc_line) == [NVCC_WRAPPER, "sccache", "/cuda/bin/nvcc"]
