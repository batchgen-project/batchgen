"""The ragged FP8 ops extension is loaded through the shared loader once."""

import batchgen.models.glm.glm5.moe_ragged as moe_ragged


def test_act_quant_ragged_loads_fp8_ops_through_shared_loader_once(monkeypatch):
    import batchgen_kernels

    calls = []

    class _FakeFp8Ops:
        @staticmethod
        def act_quant_ragged(x, seqlens, cu_seqlens, output, scale, row_align):
            calls.append((x, seqlens, cu_seqlens, output, scale, row_align))

    loads = []

    def _load_extension(name):
        loads.append(name)
        return _FakeFp8Ops()

    monkeypatch.setattr(batchgen_kernels, "load_extension", _load_extension)
    monkeypatch.setattr(moe_ragged, "_ops_module", None)
    x, seqlens, cu_seqlens, output, scale = (object() for _ in range(5))

    moe_ragged.act_quant_ragged(x, seqlens, cu_seqlens, output, scale)
    moe_ragged.act_quant_ragged(x, seqlens, cu_seqlens, output, scale)

    assert loads == ["batchgen_kernels.moe._C_fp8_blockwise_ops"]
    assert calls == [
        (x, seqlens, cu_seqlens, output, scale, moe_ragged.ROW_ALIGN),
        (x, seqlens, cu_seqlens, output, scale, moe_ragged.ROW_ALIGN),
    ]


def test_loader_refuses_an_ops_module_without_the_ragged_kernel(monkeypatch):
    import pytest

    import batchgen_kernels

    monkeypatch.setattr(
        batchgen_kernels, "load_extension", lambda name: object()
    )
    monkeypatch.setattr(moe_ragged, "_ops_module", None)

    with pytest.raises(RuntimeError, match="act_quant_ragged"):
        moe_ragged._require_ops_module()
