# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

"""Unit tests for MXFP4 MoE module.

Run with: pytest batchgen/moe/test_moe_mxfp4.py -v
"""

import pytest
import torch
import torch.nn.functional as F

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestTokenDispatch:
    """Tests for token dispatch/combine operations."""

    def test_local_dispatcher_basic(self):
        """Test LocalTokenDispatcher with simple input."""
        from batchgen.moe.token_dispatch import LocalTokenDispatcher

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        num_tokens = 8
        hidden_dim = 64
        num_experts = 4
        k = 2

        dispatcher = LocalTokenDispatcher()

        x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
        topk_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)
        topk_weights = F.softmax(torch.randn(num_tokens, k, device=device), dim=-1)

        # Dispatch
        result = dispatcher.dispatch(x, topk_indices, topk_weights)

        # Verify shapes
        assert result.dispatched_x.shape == (num_tokens * k, hidden_dim)
        assert result.expert_ids.shape == (num_tokens * k,)
        assert result.original_indices.shape == (num_tokens * k,)
        assert result.expert_counts.shape == (num_experts,)
        assert result.expert_offsets.shape == (num_experts,)

        # Verify expert counts sum to total
        assert result.expert_counts.sum().item() == num_tokens * k

        # Verify offsets are correct
        expected_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
        expected_offsets[1:] = torch.cumsum(result.expert_counts[:-1], dim=0)
        assert torch.equal(result.expert_offsets, expected_offsets)

    def test_local_dispatcher_roundtrip(self):
        """Test that dispatch -> identity GEMM -> combine recovers weighted input."""
        from batchgen.moe.token_dispatch import LocalTokenDispatcher

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        num_tokens = 16
        hidden_dim = 64
        num_experts = 8
        k = 2

        dispatcher = LocalTokenDispatcher()

        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device=device)
        topk_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)
        topk_weights = F.softmax(torch.randn(num_tokens, k, device=device), dim=-1)

        # Dispatch
        result = dispatcher.dispatch(x, topk_indices, topk_weights)

        # "Identity" expert: just return dispatched_x
        expert_output = result.dispatched_x.clone()

        # Combine
        output = dispatcher.combine(expert_output, result, num_tokens)

        # Expected: weighted sum of x for each selected expert
        expected = torch.zeros_like(x)
        for i in range(num_tokens):
            for j in range(k):
                expected[i] += topk_weights[i, j] * x[i]

        # Should match
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_local_dispatcher_sorted_by_expert(self):
        """Verify dispatched tokens are sorted by expert ID."""
        from batchgen.moe.token_dispatch import LocalTokenDispatcher

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        dispatcher = LocalTokenDispatcher()

        num_tokens = 32
        hidden_dim = 64
        num_experts = 8
        k = 4

        x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
        topk_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)
        topk_weights = F.softmax(torch.randn(num_tokens, k, device=device), dim=-1)

        result = dispatcher.dispatch(x, topk_indices, topk_weights)

        # Verify expert_ids are sorted
        sorted_ids = torch.sort(result.expert_ids).values
        assert torch.equal(result.expert_ids, sorted_ids)


class TestRouting:
    """Tests for MoE routing."""

    def test_routing_basic(self):
        """Test basic top-k routing."""
        from batchgen.moe.routing import moe_routing

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        num_tokens = 16
        hidden_size = 64
        num_experts = 8
        k = 4

        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        gate_weight = torch.randn(hidden_size, num_experts, dtype=torch.bfloat16, device=device)
        gate_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

        topk_indices, topk_weights = moe_routing(x, gate_weight, gate_bias, k)

        # Verify shapes
        assert topk_indices.shape == (num_tokens, k)
        assert topk_weights.shape == (num_tokens, k)

        # Verify weights sum to 1 (softmax normalized)
        weight_sums = topk_weights.sum(dim=-1)
        torch.testing.assert_close(
            weight_sums,
            torch.ones(num_tokens, device=device, dtype=topk_weights.dtype),
            rtol=1e-3, atol=1e-3
        )

        # Verify indices are in valid range
        assert (topk_indices >= 0).all()
        assert (topk_indices < num_experts).all()

    def test_routing_with_aux_loss(self):
        """Test routing with auxiliary loss."""
        from batchgen.moe.routing import moe_routing_with_auxiliary_loss

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        num_tokens = 32
        hidden_size = 64
        num_experts = 8
        k = 2

        x = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device)
        gate_weight = torch.randn(hidden_size, num_experts, dtype=torch.float32, device=device)

        topk_indices, topk_weights, aux_loss = moe_routing_with_auxiliary_loss(
            x, gate_weight, None, k, num_experts, aux_loss_coef=0.01
        )

        # Verify aux_loss is a scalar
        assert aux_loss.dim() == 0
        assert aux_loss.item() >= 0

    def test_expert_load_stats(self):
        """Test expert load statistics computation."""
        from batchgen.moe.routing import compute_expert_load_stats

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        num_tokens = 100
        num_experts = 8
        k = 4

        # Create routing with known distribution
        topk_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)

        stats = compute_expert_load_stats(topk_indices, num_experts)

        # Verify counts sum correctly
        assert stats['counts'].sum().item() == num_tokens * k

        # Verify statistics are reasonable
        assert stats['mean'] == num_tokens * k / num_experts
        assert 0 <= stats['utilization'] <= 1


class TestSwiGLU:
    """Tests for SwiGLU activation."""

    def test_swiglu_basic(self):
        """Test SwiGLU activation."""
        from batchgen.moe.moe_mxfp4 import swiglu

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        batch = 8
        intermediate = 64

        x = torch.randn(batch, intermediate * 2, dtype=torch.bfloat16, device=device)

        output = swiglu(x, limit=7.0)

        # Verify shape (splits in half)
        assert output.shape == (batch, intermediate)

        # Verify against reference
        gate, up = x.chunk(2, dim=-1)
        gate_clamped = gate.clamp(-7.0, 7.0)
        expected = F.silu(gate_clamped) * up

        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    def test_swiglu_no_limit(self):
        """Test SwiGLU without clamping."""
        from batchgen.moe.moe_mxfp4 import swiglu

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        x = torch.randn(8, 128, dtype=torch.bfloat16, device=device)

        # With limit=0, no clamping
        output = swiglu(x, limit=0)

        gate, up = x.chunk(2, dim=-1)
        expected = F.silu(gate) * up

        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


class TestFusedMXFP4GEMM:
    """Tests for fused MXFP4 GEMM operations."""

    def test_single_expert_gemm(self):
        """Test single-expert fused MXFP4 GEMM."""
        from batchgen.moe.fused_mxfp4_gemm import fused_mxfp4_gemm
        from batchgen.quantization.mxfp4 import mxfp4_dequantize_reference

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        M, N, K = 64, 128, 256

        lhs = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        rhs_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        rhs_scales = torch.randint(100, 150, (N, K // 32), dtype=torch.uint8, device=device)

        # Fused kernel
        output = fused_mxfp4_gemm(lhs, rhs_packed, rhs_scales)

        # Reference: dequant + GEMM
        rhs_dequant = mxfp4_dequantize_reference(rhs_packed, rhs_scales, dtype=torch.bfloat16)
        expected = lhs @ rhs_dequant.T

        # Compare
        max_diff = (output - expected).abs().max().item()
        assert max_diff < 0.1, f"Max diff too large: {max_diff}"

    def test_moe_gemm_vs_sequential(self):
        """Test fused MoE GEMM against sequential reference."""
        from batchgen.moe.fused_mxfp4_gemm import (
            fused_mxfp4_moe_gemm,
            fused_mxfp4_moe_gemm_sequential,
        )
        from batchgen.moe.token_dispatch import LocalTokenDispatcher

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        num_tokens = 64
        hidden_size = 128
        output_size = 64
        num_experts = 4
        k = 2

        # Create dispatcher and test data
        dispatcher = LocalTokenDispatcher()

        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        topk_indices = torch.randint(0, num_experts, (num_tokens, k), device=device)
        topk_weights = F.softmax(torch.randn(num_tokens, k, device=device), dim=-1)

        # Dispatch
        result = dispatcher.dispatch(x, topk_indices, topk_weights)

        # Create weight tensors
        w_packed = torch.randint(
            0, 256, (num_experts, output_size, hidden_size // 2),
            dtype=torch.uint8, device=device
        )
        w_scales = torch.randint(
            100, 150, (num_experts, output_size, hidden_size // 32),
            dtype=torch.uint8, device=device
        )

        # Fused kernel
        output_fused = fused_mxfp4_moe_gemm(
            result.dispatched_x,
            w_packed,
            w_scales,
            result.expert_counts,
            result.expert_offsets,
        )

        # Sequential reference
        output_seq = fused_mxfp4_moe_gemm_sequential(
            result.dispatched_x,
            w_packed,
            w_scales,
            result.expert_counts,
            result.expert_offsets,
        )

        # Compare
        max_diff = (output_fused - output_seq).abs().max().item()
        assert max_diff < 0.01, f"Max diff too large: {max_diff}"


class TestMoEForward:
    """Tests for complete MoE forward pass."""

    def test_moe_forward_basic(self):
        """Test basic MoE forward pass."""
        from batchgen.moe.moe_mxfp4 import moe_mxfp4_forward

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        num_tokens = 32
        hidden_size = 128
        intermediate_size = 64
        num_experts = 8
        k = 2

        # Create inputs
        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        gate_weight = torch.randn(hidden_size, num_experts, dtype=torch.bfloat16, device=device)
        gate_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

        # Create weights
        w1_packed = torch.randint(
            0, 256, (num_experts, intermediate_size * 2, hidden_size // 2),
            dtype=torch.uint8, device=device
        )
        w1_scales = torch.randint(
            100, 150, (num_experts, intermediate_size * 2, hidden_size // 32),
            dtype=torch.uint8, device=device
        )
        w1_bias = torch.randn(num_experts, intermediate_size * 2, dtype=torch.bfloat16, device=device)

        w2_packed = torch.randint(
            0, 256, (num_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8, device=device
        )
        w2_scales = torch.randint(
            100, 150, (num_experts, hidden_size, intermediate_size // 32),
            dtype=torch.uint8, device=device
        )
        w2_bias = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

        # Forward
        output = moe_mxfp4_forward(
            x,
            gate_weight, gate_bias,
            w1_packed, w1_scales, w1_bias,
            w2_packed, w2_scales, w2_bias,
            experts_per_token=k,
            swiglu_limit=7.0,
        )

        # Verify output shape
        assert output.shape == x.shape
        assert output.dtype == torch.bfloat16

        # Verify output is not all zeros or NaN
        assert not torch.isnan(output).any()
        assert output.abs().sum() > 0

    def test_moe_forward_vs_reference(self):
        """Test MoE forward against reference implementation."""
        from batchgen.moe.moe_mxfp4 import moe_mxfp4_forward, moe_mxfp4_forward_reference

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        num_tokens = 16
        hidden_size = 64
        intermediate_size = 32
        num_experts = 4
        k = 2

        # Create inputs
        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        gate_weight = torch.randn(hidden_size, num_experts, dtype=torch.bfloat16, device=device)
        gate_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

        # Create weights
        w1_packed = torch.randint(
            0, 256, (num_experts, intermediate_size * 2, hidden_size // 2),
            dtype=torch.uint8, device=device
        )
        w1_scales = torch.randint(
            100, 150, (num_experts, intermediate_size * 2, hidden_size // 32),
            dtype=torch.uint8, device=device
        )

        w2_packed = torch.randint(
            0, 256, (num_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8, device=device
        )
        w2_scales = torch.randint(
            100, 150, (num_experts, hidden_size, intermediate_size // 32),
            dtype=torch.uint8, device=device
        )

        # Fused forward
        output = moe_mxfp4_forward(
            x, gate_weight, gate_bias,
            w1_packed, w1_scales, None,
            w2_packed, w2_scales, None,
            experts_per_token=k,
        )

        # Reference forward
        output_ref = moe_mxfp4_forward_reference(
            x, gate_weight, gate_bias,
            w1_packed, w1_scales, None,
            w2_packed, w2_scales, None,
            experts_per_token=k,
        )

        # Compare
        max_diff = (output - output_ref).abs().max().item()
        assert max_diff < 0.01, f"Max diff vs reference: {max_diff}"


class TestMoEMXFP4Layer:
    """Tests for MoEMXFP4Layer module."""

    def test_layer_initialization(self):
        """Test layer initialization."""
        from batchgen.moe.moe_mxfp4 import MoEMXFP4Layer

        device = torch.device("cuda:0")

        layer = MoEMXFP4Layer(
            hidden_size=128,
            intermediate_size=64,
            num_experts=8,
            experts_per_token=2,
            device=device,
        )

        # Verify parameter shapes
        assert layer.gate_weight.shape == (128, 8)
        assert layer.gate_bias.shape == (8,)
        assert layer.w1_packed.shape == (8, 128, 64)
        assert layer.w1_scales.shape == (8, 128, 4)
        assert layer.w2_packed.shape == (8, 128, 32)
        assert layer.w2_scales.shape == (8, 128, 2)

    def test_layer_forward(self):
        """Test layer forward pass."""
        from batchgen.moe.moe_mxfp4 import MoEMXFP4Layer

        torch.manual_seed(42)
        device = torch.device("cuda:0")

        layer = MoEMXFP4Layer(
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            experts_per_token=2,
            device=device,
        )

        # Initialize with random weights
        torch.nn.init.normal_(layer.gate_weight)
        torch.nn.init.normal_(layer.gate_bias)
        layer.w1_packed.random_(0, 256)
        layer.w1_scales.fill_(127)  # Scale of 1.0
        layer.w2_packed.random_(0, 256)
        layer.w2_scales.fill_(127)
        torch.nn.init.normal_(layer.w1_bias)
        torch.nn.init.normal_(layer.w2_bias)

        # Test with 2D input
        x_2d = torch.randn(16, 64, dtype=torch.bfloat16, device=device)
        output_2d = layer(x_2d)
        assert output_2d.shape == x_2d.shape

        # Test with 3D input
        x_3d = torch.randn(2, 8, 64, dtype=torch.bfloat16, device=device)
        output_3d = layer(x_3d)
        assert output_3d.shape == x_3d.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
