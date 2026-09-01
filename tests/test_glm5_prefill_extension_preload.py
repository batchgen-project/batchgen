from types import SimpleNamespace


def test_ragged_dispatch_and_ordered_reduce_share_one_extension_cache(monkeypatch):
    import batchgen_kernels
    from batchgen.moe import dispatch_scatter_3d
    from batchgen.models.glm.glm5 import moe_ragged

    fake = SimpleNamespace(
        dispatch_scatter_ragged=object(),
        reduce_weighted_scatter_bf16_ordered=object(),
    )
    calls = []

    def load_extension(name):
        calls.append(name)
        return fake

    monkeypatch.setattr(batchgen_kernels, "load_extension", load_extension)
    monkeypatch.setattr(dispatch_scatter_3d, "_dispatch_reduce_module", None)

    assert moe_ragged._require_dispatch_module() is fake
    assert dispatch_scatter_3d.require_dispatch_scatter_3d_kernels() is fake
    assert calls == ["batchgen_kernels.moe._C_dispatch_scatter_3d"]


def test_fused_attention_preload_is_idempotent(monkeypatch):
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
