from types import SimpleNamespace


def test_load_extension_caches_successful_jit_result(monkeypatch):
    import batchgen_kernels

    module_name = "batchgen_kernels.test_extension"
    jit_module = SimpleNamespace(name=module_name)
    import_calls = []
    jit_calls = []

    def fake_import(name):
        import_calls.append(name)
        raise ImportError(name)

    def fake_jit(name):
        jit_calls.append(name)
        return jit_module

    monkeypatch.setattr(batchgen_kernels, "_EXTENSION_CACHE", {})
    monkeypatch.setattr(batchgen_kernels, "_DEV_MODE", True)
    monkeypatch.setattr(batchgen_kernels.importlib, "import_module", fake_import)
    monkeypatch.setattr(batchgen_kernels, "_jit_compile", fake_jit)

    assert batchgen_kernels.load_extension(module_name) is jit_module
    assert batchgen_kernels.load_extension(module_name) is jit_module
    assert import_calls == [module_name]
    assert jit_calls == [module_name]
