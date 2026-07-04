import builtins
import importlib
import sys

import pytest


def _reimport_fa2_backend_without_flash_attn(monkeypatch):
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(
        sys.modules, "batchgen.attention.mla.fa2_backend", raising=False
    )

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "flash_attn" or name.startswith("flash_attn."):
            raise ImportError("No module named 'flash_attn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    return importlib.import_module("batchgen.attention.mla.fa2_backend")


def test_fa2_backend_imports_without_flash_attn(monkeypatch):
    fa2_backend = _reimport_fa2_backend_without_flash_attn(monkeypatch)

    assert hasattr(fa2_backend, "mla_prefill_flashattention2")
    assert hasattr(fa2_backend, "mla_chunked_prefill_flashattention2")


def test_fa2_backend_call_raises_clear_error_without_flash_attn(monkeypatch):
    fa2_backend = _reimport_fa2_backend_without_flash_attn(monkeypatch)

    with pytest.raises(ImportError) as excinfo:
        fa2_backend.flash_attn_varlen_func()

    assert "flash_attn" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)
