"""V4 wiring correctness and performance tests.

Validates that model.py components produce correct output after
kernel wiring (HC, gate routing, expert activation, KV cache layout).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

V4_FLASH_HIDDEN = 4096
V4_FLASH_EXPERTS = 256
V4_FLASH_INTER = 2048
V4_FLASH_TOPK = 6

V4_PRO_HIDDEN = 7168
V4_PRO_EXPERTS = 384
V4_PRO_INTER = 3072


def _make_v4_flash_config():
    from batchgen.models.deepseek.deepseekv4_flash.config import (
        DeepSeekV4FlashConfig,
    )

    return DeepSeekV4FlashConfig()


def _ref_sqrtsoftplus_topk(hidden, weight, bias, topk=6, route_scale=1.5):
    scores = F.linear(hidden.float(), weight.float())
    scores = F.softplus(scores).sqrt()
    select_scores = scores + bias.float().unsqueeze(0)
    topk_indices = torch.topk(select_scores, k=topk, dim=-1).indices
    topk_weights = scores.gather(-1, topk_indices)
    topk_weights = topk_weights / (
        topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    )
    return topk_weights * route_scale, topk_indices


def _ref_hash_routing(
    input_ids, tid2eid, hidden, weight, topk=6, route_scale=1.5
):
    scores = F.linear(hidden.float(), weight.float())
    scores = F.softplus(scores).sqrt()
    topk_indices = tid2eid[input_ids].long()
    topk_weights = scores.gather(-1, topk_indices)
    topk_weights = topk_weights / (
        topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    )
    return topk_weights * route_scale, topk_indices


def _make_expert_weights(
    hidden_size: int, inter_size: int
) -> Dict[str, torch.Tensor]:
    return {
        "w1.weight": torch.randn(
            inter_size, hidden_size, device="cuda", dtype=torch.bfloat16
        ),
        "w3.weight": torch.randn(
            inter_size, hidden_size, device="cuda", dtype=torch.bfloat16
        ),
        "w2.weight": torch.randn(
            hidden_size, inter_size, device="cuda", dtype=torch.bfloat16
        ),
    }


# ─── T1: Gate wiring ──────────────────────────────────────────────────────── #


@pytest.mark.parametrize("tokens", [1, 32, 1024])
@pytest.mark.parametrize(
    "hidden,experts",
    [(V4_FLASH_HIDDEN, V4_FLASH_EXPERTS), (V4_PRO_HIDDEN, V4_PRO_EXPERTS)],
)
def test_gate_sqrtsoftplus_wiring(tokens, hidden, experts):
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashGate,
    )

    cfg = SimpleNamespace(
        hidden_size=hidden,
        n_routed_experts=experts,
        num_experts_per_tok=V4_FLASH_TOPK,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        num_hash_layers=0,
        vocab_size=129280,
    )
    gate = DeepSeekV4FlashGate(cfg, layer_idx=5).cuda()
    torch.manual_seed(tokens + hidden)
    gate.weight.data = torch.randn_like(gate.weight)
    gate.bias.data = torch.randn_like(gate.bias)

    hidden_states = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16
    )
    weights, indices = gate(hidden_states)

    ref_weights, ref_indices = _ref_sqrtsoftplus_topk(
        hidden_states,
        gate.weight,
        gate.bias,
        topk=V4_FLASH_TOPK,
        route_scale=1.5,
    )

    assert torch.equal(indices, ref_indices)
    assert torch.allclose(weights, ref_weights, atol=1e-4)


@pytest.mark.parametrize("tokens", [1, 32])
def test_gate_hash_routing_wiring(tokens):
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashGate,
    )

    cfg = SimpleNamespace(
        hidden_size=V4_FLASH_HIDDEN,
        n_routed_experts=V4_FLASH_EXPERTS,
        num_experts_per_tok=V4_FLASH_TOPK,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        num_hash_layers=3,
        vocab_size=129280,
    )
    gate = DeepSeekV4FlashGate(cfg, layer_idx=0).cuda()
    torch.manual_seed(42)
    gate.weight.data = torch.randn_like(gate.weight)
    gate.tid2eid.data = torch.randint(
        0, V4_FLASH_EXPERTS, (129280, V4_FLASH_TOPK), device="cuda"
    )

    hidden_states = torch.randn(
        tokens, V4_FLASH_HIDDEN, device="cuda", dtype=torch.bfloat16
    )
    input_ids = torch.randint(0, 129280, (tokens,), device="cuda")
    weights, indices = gate(hidden_states, input_ids)

    ref_weights, ref_indices = _ref_hash_routing(
        input_ids,
        gate.tid2eid,
        hidden_states,
        gate.weight,
        topk=V4_FLASH_TOPK,
        route_scale=1.5,
    )

    assert torch.equal(indices, ref_indices)
    assert torch.allclose(weights, ref_weights, atol=1e-4)


# ─── T2: Expert activation wiring ─────────────────────────────────────────── #


@pytest.mark.parametrize("T", [1, 32, 128])
@pytest.mark.parametrize("inter", [V4_FLASH_INTER, V4_PRO_INTER])
def test_expert_silu_quant_wiring(T, inter):
    from batchgen_kernels.moe.silu_mul_quant import fused_silu_mul_quant_cuda

    torch.manual_seed(T * 100 + inter)
    gate = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)

    gate_clamped = gate.float().clamp(max=10.0).to(torch.bfloat16)
    up_clamped = up.float().clamp(min=-10.0, max=10.0).to(torch.bfloat16)
    out_fp8, scales = fused_silu_mul_quant_cuda(gate_clamped, up_clamped)
    kernel_activated = out_fp8.float() * scales.unsqueeze(-1)

    ref_activated = F.silu(gate_clamped.float()) * up_clamped.float()

    from tests.kernels.conftest import _assert_fp8_close

    _assert_fp8_close(
        kernel_activated, ref_activated, msg="silu_mul_quant CUDA vs PyTorch"
    )


# ─── T3: Attention wrapper decode contract ─────────────────────────────────── #


@pytest.mark.parametrize("batch", [1, 4])
def test_attn_decode_projection_shapes(batch):
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashAttention,
    )

    cfg = _make_v4_flash_config()
    attn = DeepSeekV4FlashAttention(cfg, layer_idx=5).cuda()

    torch.manual_seed(42)
    attn.set_runtime_tensors(
        {
            "wq_a.weight": torch.randn(
                cfg.q_lora_rank,
                cfg.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "wq_b.weight": torch.randn(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.q_lora_rank,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "wkv.weight": torch.randn(
                cfg.head_dim,
                cfg.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "wo_a.weight": torch.randn(
                cfg.o_groups * cfg.o_lora_rank,
                cfg.num_attention_heads * cfg.head_dim // cfg.o_groups,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "wo_b.weight": torch.randn(
                cfg.hidden_size,
                cfg.o_groups * cfg.o_lora_rank,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "q_norm.weight": torch.ones(
                cfg.q_lora_rank, device="cuda", dtype=torch.float32
            ),
            "kv_norm.weight": torch.ones(
                cfg.head_dim, device="cuda", dtype=torch.float32
            ),
        }
    )

    hidden = torch.randn(
        batch, 1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    result = attn(hidden)

    attn_output, _, kv = result
    assert attn_output.shape == (batch, 1, cfg.hidden_size)
    assert kv.shape == (batch, 1, cfg.head_dim)
    assert torch.isfinite(attn_output).all()
    assert torch.isfinite(kv).all()

    attn.clear_runtime_tensors()


# ─── T4: KV cache bytes_per_page() ────────────────────────────────────────── #


def test_kv_cache_v4_flash_byte_size():
    from batchgen.kv_cache.host_kv_mananger_config import (
        _DEEPSEEK_V4_FLASH_PROFILE,
    )
    from batchgen_kernels.triton.v4_cache_utils import TOKEN_BYTES

    assert _DEEPSEEK_V4_FLASH_PROFILE.raw_bytes_per_token == TOKEN_BYTES
    assert _DEEPSEEK_V4_FLASH_PROFILE.raw_bytes_per_token == 584
    assert _DEEPSEEK_V4_FLASH_PROFILE.bytes_per_page() == 64 * 584


def test_kv_cache_v4_pro_byte_size():
    from batchgen.kv_cache.host_kv_mananger_config import (
        _DEEPSEEK_V4_PRO_PROFILE,
    )

    assert _DEEPSEEK_V4_PRO_PROFILE.raw_bytes_per_token == 584
    assert _DEEPSEEK_V4_PRO_PROFILE.bytes_per_page() == 64 * 584
    assert _DEEPSEEK_V4_PRO_PROFILE.num_layers == 61


def test_kv_cache_non_v4_unaffected():
    from batchgen.kv_cache.host_kv_mananger_config import (
        _DEEPSEEK_MLA_PROFILE,
    )

    assert _DEEPSEEK_MLA_PROFILE.raw_bytes_per_token is None
    assert _DEEPSEEK_MLA_PROFILE.bytes_per_page() == 64 * 1 * 576 * 2


# ─── T5: Decoder layer forward with HC kernel path ────────────────────────── #


def test_decoder_layer_hc_kernel_forward():
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashDecoderLayer,
    )

    cfg = _make_v4_flash_config()
    layer = DeepSeekV4FlashDecoderLayer(cfg, layer_idx=5).cuda().eval()

    torch.manual_seed(42)

    layer.self_attn.set_runtime_tensors(
        {
            "wq_a.weight": torch.randn(
                cfg.q_lora_rank,
                cfg.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "wq_b.weight": torch.randn(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.q_lora_rank,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "wkv.weight": torch.randn(
                cfg.head_dim,
                cfg.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "wo_a.weight": torch.randn(
                cfg.o_groups * cfg.o_lora_rank,
                cfg.num_attention_heads * cfg.head_dim // cfg.o_groups,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "wo_b.weight": torch.randn(
                cfg.hidden_size,
                cfg.o_groups * cfg.o_lora_rank,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "q_norm.weight": torch.ones(
                cfg.q_lora_rank, device="cuda", dtype=torch.float32
            ),
            "kv_norm.weight": torch.ones(
                cfg.head_dim, device="cuda", dtype=torch.float32
            ),
        }
    )

    expert_w = _make_expert_weights(cfg.hidden_size, cfg.moe_intermediate_size)
    layer.mlp.shared_experts.set_runtime_tensors(expert_w)
    layer.mlp.experts[0].set_runtime_tensors(
        _make_expert_weights(cfg.hidden_size, cfg.moe_intermediate_size)
    )

    def _force_single_expert(hidden_states, input_ids=None):
        T = hidden_states.shape[0]
        return (
            torch.ones(T, 1, device=hidden_states.device),
            torch.zeros(T, 1, dtype=torch.long, device=hidden_states.device),
        )

    layer.mlp.gate.forward = _force_single_expert
    layer.mlp.num_experts_per_tok = 1

    hidden = torch.randn(
        1,
        16,
        cfg.hc_mult,
        cfg.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with torch.no_grad():
        out, _, _ = layer(hidden)

    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()
    assert not torch.equal(out, hidden)

    out2, _, _ = layer(hidden)
    assert torch.equal(out, out2)

    layer.self_attn.clear_runtime_tensors()
    layer.mlp.shared_experts.clear_runtime_tensors()
    layer.mlp.experts[0].clear_runtime_tensors()


# ─── T6: Performance regression gate ──────────────────────────────────────── #


@pytest.mark.parametrize("T", [128, 1024])
def test_gate_perf_not_slower_than_pytorch(T):
    from tests.kernels.conftest import _bench

    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashGate,
    )

    cfg = SimpleNamespace(
        hidden_size=V4_FLASH_HIDDEN,
        n_routed_experts=V4_FLASH_EXPERTS,
        num_experts_per_tok=V4_FLASH_TOPK,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        num_hash_layers=0,
        vocab_size=129280,
    )
    gate = DeepSeekV4FlashGate(cfg, layer_idx=5).cuda()
    torch.manual_seed(42)
    gate.weight.data = torch.randn_like(gate.weight)
    gate.bias.data = torch.randn_like(gate.bias)

    hidden = torch.randn(
        T, V4_FLASH_HIDDEN, device="cuda", dtype=torch.bfloat16
    )

    kernel_ms = _bench(gate, hidden)

    def _ref_fn():
        return _ref_sqrtsoftplus_topk(
            hidden,
            gate.weight,
            gate.bias,
            topk=V4_FLASH_TOPK,
            route_scale=1.5,
        )

    ref_ms = _bench(_ref_fn)

    ratio = kernel_ms / ref_ms if ref_ms > 0 else 0
    print(
        f"\nGate T={T}: kernel={kernel_ms:.3f}ms ref={ref_ms:.3f}ms ratio={ratio:.2f}x"
    )
    assert (
        ratio <= 1.5
    ), f"kernel path {ratio:.2f}x slower than PyTorch (regression)"
