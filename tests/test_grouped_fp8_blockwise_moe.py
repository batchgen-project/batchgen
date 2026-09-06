from types import SimpleNamespace


def test_grouped_fp8_blockwise_moe_uses_shared_extension_loader(monkeypatch):
    import batchgen_kernels
    from batchgen.moe import grouped_fp8_blockwise_moe as mod

    calls = []

    def fake_load_extension(name):
        calls.append(name)
        return SimpleNamespace(
            fp8_blockwise_grouped_gemm=lambda *args, **kwargs: "grouped",
            fp8_blockwise_fused_s1=lambda *args, **kwargs: "fused",
        )

    monkeypatch.setattr(batchgen_kernels, "load_extension", fake_load_extension)

    mod._grouped_module = None
    mod._warned_import = False
    mod._warned_fused_s1 = False

    assert mod._get_kernel()("x") == "grouped"
    assert mod._get_fused_s1_kernel()("x") == "fused"
    assert calls == [
        "batchgen_kernels.moe._C_fp8_blockwise_gemm",
        "batchgen_kernels.moe._C_fp8_blockwise_gemm",
    ]
