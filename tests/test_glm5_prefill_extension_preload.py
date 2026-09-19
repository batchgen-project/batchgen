from types import SimpleNamespace


def test_ragged_dispatch_uses_shared_extension_cache(monkeypatch):
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


def test_prefill_preload_sync_waits_for_every_distributed_rank(monkeypatch):
    from batchgen.models.glm.glm5 import Parallel_Strategy_Manager as psm

    calls = []
    monkeypatch.setattr(psm.dist, "is_available", lambda: True)
    monkeypatch.setattr(psm.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(psm.dist, "barrier", lambda: calls.append("barrier"))

    psm._synchronize_prefill_preloads()
    assert calls == ["barrier"]

    calls.clear()
    monkeypatch.setattr(psm.dist, "is_initialized", lambda: False)
    psm._synchronize_prefill_preloads()
    assert calls == []
