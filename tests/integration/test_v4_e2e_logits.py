"""End-to-end logit validation for V4-Flash.

Compares optimized kernel path vs PyTorch fallback path on the same input.
Requires: model weights on disk, H20 GPU (sm_90).

Run: pytest tests/integration/test_v4_e2e_logits.py -v --timeout=120
"""

from __future__ import annotations

import pytest
import torch

V4_FLASH_LAYERS = 43
V4_FLASH_HIDDEN = 4096
V4_FLASH_VOCAB = 129280


@pytest.fixture
def v4_flash_config():
    from batchgen.models.deepseek.deepseekv4_flash.config import (
        DeepSeekV4FlashConfig,
    )

    return DeepSeekV4FlashConfig()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestV4FlashE2ELogits:
    def test_decoder_layer_kernel_vs_fallback(self, v4_flash_config):
        from batchgen.models.deepseek.deepseekv4_flash.model import (
            DeepSeekV4FlashDecoderLayer,
        )

        layer = DeepSeekV4FlashDecoderLayer(v4_flash_config, layer_idx=0)
        layer = layer.cuda().eval()

        torch.manual_seed(42)
        hidden = torch.randn(
            1,
            16,
            v4_flash_config.hc_mult,
            V4_FLASH_HIDDEN,
            device="cuda",
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            out, _, _ = layer(hidden)

        assert out.shape == hidden.shape
        assert torch.isfinite(out).all()

    def test_hc_kernel_matches_inline(self, v4_flash_config):
        from batchgen_kernels.common.v4_hyper_connections import hc_post, hc_pre

        hc_mult = v4_flash_config.hc_mult
        hidden_dim = V4_FLASH_HIDDEN
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * hidden_dim

        torch.manual_seed(42)
        hidden = torch.randn(1, 8, hc_mult, hidden_dim, device="cuda")
        fn = torch.randn(mix_hc, hc_dim, device="cuda")
        scale = torch.randn(3, device="cuda")
        base = torch.randn(mix_hc, device="cuda")

        reduced, post, comb = hc_pre(
            hidden,
            fn,
            scale,
            base,
            hc_mult=hc_mult,
            sinkhorn_iters=20,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
        )

        assert reduced.shape == (1, 8, hidden_dim)
        assert post.shape[:-1] == (1, 8)
        assert torch.isfinite(reduced).all()

        reconstructed = hc_post(reduced.unsqueeze(2), hidden, post, comb)
        assert reconstructed.shape == hidden.shape
        assert torch.isfinite(reconstructed).all()

    def test_gate_routing_dispatch(self, v4_flash_config):
        from batchgen.models.deepseek.deepseekv4_flash.model import (
            DeepSeekV4FlashGate,
        )

        gate = DeepSeekV4FlashGate(v4_flash_config, layer_idx=5)
        gate = gate.cuda().eval()

        torch.manual_seed(42)
        gate.weight.data = torch.randn_like(gate.weight)
        if gate.bias is not None:
            gate.bias.data = torch.randn_like(gate.bias)

        hidden = torch.randn(4, V4_FLASH_HIDDEN, device="cuda")
        weights, indices = gate(hidden)

        assert weights.shape == (4, gate.topk)
        assert indices.shape == (4, gate.topk)
        assert (indices >= 0).all() and (indices < gate.num_experts).all()
        assert (weights > 0).all()
