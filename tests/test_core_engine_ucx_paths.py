import sys
import types

import pytest

from op_builder.core_engine import _libucx_paths


def test_missing_libucx_wheel_is_refused(monkeypatch):
	monkeypatch.setitem(sys.modules, "libucx", None)
	with pytest.raises(RuntimeError, match="libucx-cu12"):
		_libucx_paths()


def test_libucx_wheel_without_headers_is_refused(monkeypatch, tmp_path):
	(tmp_path / "lib").mkdir()
	fake = types.ModuleType("libucx")
	fake.__file__ = str(tmp_path / "__init__.py")
	monkeypatch.setitem(sys.modules, "libucx", fake)
	with pytest.raises(RuntimeError, match="include/ and lib/"):
		_libucx_paths()


def test_libucx_wheel_paths(monkeypatch, tmp_path):
	(tmp_path / "include").mkdir()
	(tmp_path / "lib").mkdir()
	fake = types.ModuleType("libucx")
	fake.__file__ = str(tmp_path / "__init__.py")
	monkeypatch.setitem(sys.modules, "libucx", fake)
	assert _libucx_paths() == (str(tmp_path / "include"), str(tmp_path / "lib"))
