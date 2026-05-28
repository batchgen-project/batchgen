# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _ref_dequant(weight, scale, dtype):
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        _dequant_fp4_e2m1_weight,
    )

    return _dequant_fp4_e2m1_weight(weight, scale, dtype)


def _kernel_dequant(weight, scale, dtype):
    from batchgen_kernels.common.v4_fp4_dequant import dequant_fp4_e2m1

    return dequant_fp4_e2m1(weight, scale, dtype)


def test_all_16_fp4_values():
    from batchgen_kernels.common.v4_fp4_dequant import FP4_E2M1_TABLE

    packed = torch.arange(16, dtype=torch.uint8, device="cuda").unsqueeze(0)
    scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")

    result = _kernel_dequant(packed, scale, torch.float32)

    expected = torch.tensor(FP4_E2M1_TABLE, dtype=torch.float32, device="cuda")
    low_vals = result[0, 0::2]
    high_vals = result[0, 1::2]

    all_vals = torch.zeros(16, dtype=torch.float32, device="cuda")
    for i in range(16):
        nibble = i
        all_vals[nibble] = expected[nibble]

    for i in range(16):
        lo = packed[0, i].item() & 0x0F
        hi = (packed[0, i].item() >> 4) & 0x0F
        assert low_vals[i].item() == expected[lo].item()
        assert high_vals[i].item() == expected[hi].item()


def test_nibble_unpack():
    packed = torch.tensor([[0xA5, 0x3F]], dtype=torch.uint8, device="cuda")
    scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")

    result = _kernel_dequant(packed, scale, torch.float32)
    from batchgen_kernels.common.v4_fp4_dequant import FP4_E2M1_TABLE

    table = FP4_E2M1_TABLE

    assert result[0, 0].item() == table[0x5]
    assert result[0, 1].item() == table[0xA]
    assert result[0, 2].item() == table[0xF]
    assert result[0, 3].item() == table[0x3]


def test_scale_application():
    torch.manual_seed(42)
    packed = torch.randint(0, 256, (32, 512), dtype=torch.uint8, device="cuda")
    scale = (
        torch.rand(32, 512 * 2 // 32, dtype=torch.float32, device="cuda") + 0.1
    )

    result = _kernel_dequant(packed, scale, torch.float32)
    ref = _ref_dequant(packed, scale, torch.float32)

    torch.testing.assert_close(result, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape,scale_cols",
    [
        ((2048, 1024), 1024 * 2 // 32),
        ((3072, 3584), 3584 * 2 // 32),
    ],
    ids=["flash_expert", "pro_expert"],
)
def test_e2e_matches_ref(shape, scale_cols):
    torch.manual_seed(7)
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    scale = (
        torch.rand(shape[0], scale_cols, dtype=torch.float32, device="cuda")
        + 0.01
    )

    result = _kernel_dequant(packed, scale, torch.bfloat16)
    ref = _ref_dequant(packed, scale, torch.bfloat16)

    assert torch.equal(result, ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_output_dtype(dtype):
    packed = torch.randint(0, 256, (64, 64), dtype=torch.uint8, device="cuda")
    scale = torch.ones(64, 4, dtype=torch.float32, device="cuda")

    result = _kernel_dequant(packed, scale, dtype)
    assert result.dtype == dtype


def test_all_zero_packed():
    packed = torch.zeros(2048, 1024, dtype=torch.uint8, device="cuda")
    scale = torch.ones(2048, 64, dtype=torch.float32, device="cuda")

    result = _kernel_dequant(packed, scale, torch.bfloat16)
    assert torch.count_nonzero(result).item() == 0


def test_zero_scale():
    torch.manual_seed(1)
    packed = torch.randint(0, 256, (64, 64), dtype=torch.uint8, device="cuda")
    scale = torch.zeros(64, 4, dtype=torch.float32, device="cuda")

    result = _kernel_dequant(packed, scale, torch.float32)
    assert torch.count_nonzero(result).item() == 0


def test_non_aligned_shape():
    packed = torch.randint(
        0, 256, (2048, 500), dtype=torch.uint8, device="cuda"
    )
    n_unpacked = 500 * 2
    n_scale_cols = (n_unpacked + 31) // 32
    scale = torch.ones(2048, n_scale_cols, dtype=torch.float32, device="cuda")

    result = _kernel_dequant(packed, scale, torch.bfloat16)
    assert result.shape == (2048, 1000)


def test_flash_expert_shape():
    packed = torch.randint(
        0, 256, (2048, 1024), dtype=torch.uint8, device="cuda"
    )
    scale = torch.ones(2048, 64, dtype=torch.float32, device="cuda")

    result = _kernel_dequant(packed, scale, torch.bfloat16)
    assert result.shape == (2048, 2048)


def test_pro_expert_shape():
    packed = torch.randint(
        0, 256, (3072, 3584), dtype=torch.uint8, device="cuda"
    )
    scale = torch.ones(3072, 224, dtype=torch.float32, device="cuda")

    result = _kernel_dequant(packed, scale, torch.bfloat16)
    assert result.shape == (3072, 7168)


@pytest.mark.parametrize(
    "shape",
    [(2048, 1024), (3072, 3584)],
    ids=["flash_2048x2048", "pro_3072x7168"],
)
def test_benchmark(shape):
    from tests.kernels.conftest import _bench

    torch.manual_seed(0)
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    scale = torch.ones(
        shape[0], shape[1] * 2 // 32, dtype=torch.float32, device="cuda"
    )

    from batchgen_kernels.common.v4_fp4_dequant import dequant_fp4_e2m1

    ms = _bench(dequant_fp4_e2m1, packed, scale, torch.bfloat16)
    print(f"\nK13 dequant {shape}: {ms:.3f} ms")
