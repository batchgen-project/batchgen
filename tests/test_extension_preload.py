from types import SimpleNamespace


def test_dispatch_preload_reuses_extension_cache(monkeypatch):
    import batchgen_kernels
    from batchgen.moe import dispatch_scatter_3d

    fake = SimpleNamespace()
    calls = []

    def load_extension(name):
        calls.append(name)
        return fake

    monkeypatch.setattr(batchgen_kernels, "load_extension", load_extension)
    monkeypatch.setattr(dispatch_scatter_3d, "_dispatch_reduce_module", None)

    assert dispatch_scatter_3d.require_dispatch_scatter_3d_kernels() is fake
    assert dispatch_scatter_3d.require_dispatch_scatter_3d_kernels() is fake
    assert calls == ["batchgen_kernels.moe._C_dispatch_scatter_3d"]


def test_fused_attention_preload_reuses_extension_cache(monkeypatch):
    import batchgen_kernels
    from batchgen.attention.fused_kernels import ops

    fake = object()
    calls = []

    def load_extension(name):
        calls.append(name)
        return fake

    monkeypatch.setattr(batchgen_kernels, "load_extension", load_extension)
    monkeypatch.setattr(ops, "_ext", None)

    assert ops.preload_fused_attention_kernels() is fake
    assert ops.preload_fused_attention_kernels() is fake
    assert calls == ["batchgen_kernels.attention._C_fused_ops"]
