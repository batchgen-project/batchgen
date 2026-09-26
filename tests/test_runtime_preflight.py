from types import SimpleNamespace

import pytest

import batchgen.runtime_preflight as preflight


def test_unknown_model_type_fails_closed(monkeypatch):
    monkeypatch.setattr(preflight, "_check_torch", lambda: None)
    monkeypatch.setattr(preflight, "_detect_model_type_from_identifier", lambda _: "new_model")

    with pytest.raises(preflight.RuntimePreflightError, match="no declared runtime contract"):
        preflight.run_runtime_preflight(SimpleNamespace(model="new-model"))


def test_preflight_checks_contract_before_core_engine(monkeypatch):
    calls = []

    monkeypatch.setattr(preflight, "_check_torch", lambda: calls.append("torch"))
    monkeypatch.setattr(preflight, "_resolve_model_type", lambda _: "gpt_oss")
    monkeypatch.setattr(preflight, "_check_tokenizer", lambda _: calls.append("tokenizer"))

    def fake_import(name, *, site_package=False):
        calls.append((name, site_package))
        if name == "batchgen.core_engine":
            return SimpleNamespace(__file__="/opt/conda/lib/python3.11/site-packages/core_engine.so")
        if name == "libucx":
            return SimpleNamespace(load_library=lambda: calls.append("ucx-load"))
        if name == "deep_gemm":
            return SimpleNamespace(fp8_mqa_logits=lambda max_seqlen_k: None)
        return SimpleNamespace(__file__="/opt/conda/lib/python3.11/site-packages/module.py")

    monkeypatch.setattr(preflight, "_import_required", fake_import)
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda _: "0.1.5.post3")

    assert preflight.run_runtime_preflight(SimpleNamespace(model="openai/gpt-oss-120b")) == "gpt_oss"
    assert calls[0] == "torch"
    assert calls.index("ucx-load") < calls.index(("batchgen.core_engine", False))


def test_non_aot_core_engine_fails(monkeypatch):
    monkeypatch.setattr(preflight, "_check_torch", lambda: None)
    monkeypatch.setattr(preflight, "_resolve_model_type", lambda _: "mixtral")
    monkeypatch.setattr(preflight, "_check_tokenizer", lambda _: None)

    def fake_import(name, *, site_package=False):
        if name == "batchgen.core_engine":
            return SimpleNamespace(__file__="/tmp/core_engine.py")
        if name == "libucx":
            return SimpleNamespace(load_library=lambda: None)
        return SimpleNamespace(__file__="/opt/conda/lib/python3.11/site-packages/module.py")

    monkeypatch.setattr(preflight, "_import_required", fake_import)

    with pytest.raises(preflight.RuntimePreflightError, match="not an AOT native module"):
        preflight.run_runtime_preflight(SimpleNamespace(model="mixtral"))
