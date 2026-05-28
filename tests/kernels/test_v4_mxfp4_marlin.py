"""Tests for MXFP4 Marlin MoE weight preparation and reference forward."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

try:
    from batchgen.moe.marlin_weight_prep import (
        _marlin_pack_weights,
        get_weight_perm,
    )
except ImportError:
    pytest.skip("Marlin extension unavailable", allow_module_level=True)

from batchgen.moe.v4_mxfp4_marlin_moe import (
    Mxfp4MarlinMoEMethod,
    mxfp4_dequant_weight,
    mxfp4_expert_mlp_ref,
    prepare_moe_mxfp4_layer_for_marlin,
)
from batchgen.quantization.mxfp4 import FP4_LOOKUP_TABLE, MXFP4_BLOCK_SIZE

NUM_EXPERTS = 8
INTERMEDIATE_SIZE = 2880
HIDDEN_SIZE = 2880
ATOL = 0.05
DEVICE = "cuda"


def _make_mxfp4_weights(
    rows: int, cols: int, device: str = DEVICE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random MXFP4 packed weights and E8M0 scales.

    Returns:
        packed: [rows, cols // 2] uint8
        scales: [rows, cols // 32] uint8 (raw E8M0 exponent bytes)
    """
    n_packed = cols // 2
    n_scales = cols // MXFP4_BLOCK_SIZE
    packed = torch.randint(
        0, 256, (rows, n_packed), dtype=torch.uint8, device=device
    )
    scales = torch.randint(
        120, 135, (rows, n_scales), dtype=torch.uint8, device=device
    )
    return packed, scales


def _make_layer(device: str = DEVICE) -> nn.Module:
    layer = nn.Module()
    E, N, K = NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE

    w13_packed, w13_scales = _make_mxfp4_weights(2 * N, K, device)
    w2_packed, w2_scales = _make_mxfp4_weights(K, N, device)

    layer.w13_weight = nn.Parameter(
        w13_packed.unsqueeze(0).expand(E, -1, -1).contiguous(),
        requires_grad=False,
    )
    layer.w13_weight_scale_inv = nn.Parameter(
        w13_scales.unsqueeze(0).expand(E, -1, -1).contiguous(),
        requires_grad=False,
    )
    layer.w2_weight = nn.Parameter(
        w2_packed.unsqueeze(0).expand(E, -1, -1).contiguous(),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = nn.Parameter(
        w2_scales.unsqueeze(0).expand(E, -1, -1).contiguous(),
        requires_grad=False,
    )
    return layer


class TestMxfp4DequantWeight:
    def test_shape_and_dtype(self):
        packed, scales = _make_mxfp4_weights(64, 128)
        out = mxfp4_dequant_weight(packed, scales, torch.bfloat16)
        assert out.shape == (64, 128)
        assert out.dtype == torch.bfloat16

    def test_known_values(self):
        fp4_table = FP4_LOOKUP_TABLE
        packed = torch.tensor([[0x10]], dtype=torch.uint8, device=DEVICE)
        scales = torch.tensor([[127]], dtype=torch.uint8, device=DEVICE)
        packed_full = torch.zeros(1, 16, dtype=torch.uint8, device=DEVICE)
        scales_full = torch.full((1, 1), 127, dtype=torch.uint8, device=DEVICE)
        packed_full[0, 0] = 0x10
        out = mxfp4_dequant_weight(packed_full, scales_full, torch.float32)
        assert out[0, 0].item() == fp4_table[0].item()
        assert out[0, 1].item() == fp4_table[1].item()


class TestPrepareWeightsForMarlin:
    def test_shapes_after_preparation(self):
        layer = _make_layer()
        E, N, K = NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE

        prepare_moe_mxfp4_layer_for_marlin(layer)

        assert layer.w13_weight.dtype == torch.int32
        assert layer.w2_weight.dtype == torch.int32
        assert layer.w13_weight.shape[0] == E
        assert layer.w2_weight.shape[0] == E
        assert hasattr(layer, "workspace")

    def test_scales_are_e8m0(self):
        layer = _make_layer()
        prepare_moe_mxfp4_layer_for_marlin(layer)
        assert layer.w13_weight_scale_inv.dtype == torch.float8_e8m0fnu
        assert layer.w2_weight_scale_inv.dtype == torch.float8_e8m0fnu


class TestMxfp4MarlinMoEMethod:
    def test_create_weights(self):
        method = Mxfp4MarlinMoEMethod(
            NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE
        )
        layer = nn.Module()
        method.create_weights(layer, torch.device(DEVICE))

        E, N, K = NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
        assert layer.w13_weight.shape == (E, 2 * N, K // 2)
        assert layer.w13_weight.dtype == torch.uint8
        assert layer.w2_weight.shape == (E, K, N // 2)
        assert layer.w13_weight_scale_inv.shape == (E, 2 * N, K // 32)
        assert layer.w2_weight_scale_inv.shape == (E, K, N // 32)

    def test_process_weights_after_loading(self):
        method = Mxfp4MarlinMoEMethod(
            NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE
        )
        layer = _make_layer()
        method.process_weights_after_loading(layer)
        assert layer.w13_weight.dtype == torch.int32
        assert layer.w2_weight.dtype == torch.int32


class TestForwardSingleExpert:
    def test_reference_forward_matches_manual(self):
        E, N, K = NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
        M = 4
        torch.manual_seed(42)

        w13_packed, w13_scales = _make_mxfp4_weights(2 * N, K)
        w2_packed, w2_scales = _make_mxfp4_weights(K, N)

        w13_packed_e = w13_packed.unsqueeze(0).expand(E, -1, -1).contiguous()
        w13_scales_e = w13_scales.unsqueeze(0).expand(E, -1, -1).contiguous()
        w2_packed_e = w2_packed.unsqueeze(0).expand(E, -1, -1).contiguous()
        w2_scales_e = w2_scales.unsqueeze(0).expand(E, -1, -1).contiguous()

        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
        expert_idx = 0

        method = Mxfp4MarlinMoEMethod(E, K, N)
        out_method = method.forward_single_expert(
            x,
            expert_idx,
            w13_packed_e,
            w13_scales_e,
            w2_packed_e,
            w2_scales_e,
        )

        gate_w = mxfp4_dequant_weight(
            w13_packed_e[expert_idx, :N, :],
            w13_scales_e[expert_idx, :N, :],
            torch.bfloat16,
        )
        up_w = mxfp4_dequant_weight(
            w13_packed_e[expert_idx, N:, :],
            w13_scales_e[expert_idx, N:, :],
            torch.bfloat16,
        )
        down_w = mxfp4_dequant_weight(
            w2_packed_e[expert_idx],
            w2_scales_e[expert_idx],
            torch.bfloat16,
        )
        gate_out = x @ gate_w.T
        up_out = x @ up_w.T
        intermediate = F.silu(gate_out) * up_out
        out_ref = intermediate @ down_w.T

        diff = (out_method.float() - out_ref.float()).abs()
        assert (
            diff.max().item() < ATOL
        ), f"max abs diff = {diff.max().item():.6f}, expected < {ATOL}"

    def test_output_shape(self):
        E, N, K = NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
        M = 8
        torch.manual_seed(0)

        w13_packed, w13_scales = _make_mxfp4_weights(2 * N, K)
        w2_packed, w2_scales = _make_mxfp4_weights(K, N)

        w13_packed_e = w13_packed.unsqueeze(0).expand(E, -1, -1).contiguous()
        w13_scales_e = w13_scales.unsqueeze(0).expand(E, -1, -1).contiguous()
        w2_packed_e = w2_packed.unsqueeze(0).expand(E, -1, -1).contiguous()
        w2_scales_e = w2_scales.unsqueeze(0).expand(E, -1, -1).contiguous()

        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
        method = Mxfp4MarlinMoEMethod(E, K, N)
        out = method.forward_single_expert(
            x,
            3,
            w13_packed_e,
            w13_scales_e,
            w2_packed_e,
            w2_scales_e,
        )
        assert out.shape == (M, K)
        assert out.dtype == torch.bfloat16
