"""The one CPU stand-in for Triton that unit tests install.

Several ``tests/unit`` files import production modules that define Triton
kernels at import time (``batchgen.kv_cache`` via ``batchgen_kernels.triton``,
``batchgen.models.glm.glm5.model``). On a CPU-only box without Triton those
imports need a decorator-only stand-in; no kernel body ever runs.

``sys.modules`` is process-global and every test module is imported during
collection, so a stub installed at module level outlives its file. When each
file built its own partial stub, the first one collected won and the others
imported against it: a ``triton.language`` that was not a package broke the
``from triton.language.extra import libdevice`` in
``batchgen_kernels/triton/fp8_quantize.py``, and a stub without ``__spec__``
made a later ``importlib.util.find_spec("triton")`` raise. The result depended
on file order.

This module is therefore the single installer: one complete stub, installed
at most once per process, recognised by a marker, and never installed when a
real Triton is importable (CI keeps the real one). A test that needs a
working Triton gates on ``importorskip_real_triton()``, which skips on the
stub as it does on a missing Triton, whatever file was collected first.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

_MARKER = "__batchgen_cpu_stub__"
_STUB_NAMES = (
    "triton",
    "triton.language",
    "triton.language.extra",
    "triton.language.extra.libdevice",
)


class ForeignTritonStubError(RuntimeError):
    """``sys.modules['triton']`` is neither real Triton nor this stub."""


class _AnyName:
    """Stand-in for a ``tl.*`` / ``libdevice.*`` name used at import time."""

    def __getitem__(self, item):
        return item

    def __call__(self, *args, **kwargs):
        return None


def _module(name: str, *, package: bool) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=package
    )
    if package:
        module.__path__ = []
    module.__version__ = "0.0.0-cpu-stub"
    setattr(module, _MARKER, True)
    return module


def _any_name_getattr(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return _AnyName()


def _decorator(fn=None, **_kwargs):
    if fn is None:
        return lambda f: f
    return fn


def _next_power_of_2(n):
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def is_triton_stubbed() -> bool:
    """True when the importable ``triton`` is this CPU stub."""
    return bool(getattr(sys.modules.get("triton"), _MARKER, False))


def importorskip_real_triton():
    """``pytest.importorskip("triton")`` that does not accept this stub.

    For a test that needs a working Triton: once another file has installed
    the stub, a plain ``importorskip`` would import it and the test would run
    (or fail to collect) only when it happens to be collected after that file.
    """
    import pytest

    if is_triton_stubbed():
        pytest.skip(
            "triton is absent; only the CPU stub from tests/triton_cpu_stub.py "
            "is importable",
            allow_module_level=True,
        )
    return pytest.importorskip("triton")


def install_triton_cpu_stub() -> bool:
    """Install the stub if Triton is absent; return True if the stub is live.

    Idempotent. Returns False, installing nothing, when a real Triton is
    importable or already imported.
    """
    existing = sys.modules.get("triton")
    if existing is not None:
        if getattr(existing, _MARKER, False):
            return True
        if getattr(existing, "__file__", None) is None:
            raise ForeignTritonStubError(
                "sys.modules['triton'] is a stub not built by "
                "tests/triton_cpu_stub.py; install Triton stubs only through "
                "install_triton_cpu_stub() so every test sees the same one"
            )
        return False
    if importlib.util.find_spec("triton") is not None:
        return False

    triton = _module("triton", package=True)
    triton.jit = _decorator
    triton.autotune = lambda *args, **kwargs: (lambda fn: fn)
    triton.heuristics = lambda *args, **kwargs: (lambda fn: fn)
    triton.Config = lambda *args, **kwargs: None
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = _next_power_of_2

    language = _module("triton.language", package=True)
    language.__getattr__ = _any_name_getattr
    extra = _module("triton.language.extra", package=True)
    libdevice = _module("triton.language.extra.libdevice", package=False)
    libdevice.__getattr__ = _any_name_getattr

    extra.libdevice = libdevice
    language.extra = extra
    triton.language = language
    for module in (triton, language, extra, libdevice):
        sys.modules[module.__name__] = module
    assert all(sys.modules[name].__name__ == name for name in _STUB_NAMES)
    return True
