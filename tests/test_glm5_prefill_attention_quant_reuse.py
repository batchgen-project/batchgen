import types

import torch

from batchgen.attention.mla import fa3_backend


class _IdentityNorm:
    def __call__(self, value):
        return value


class _Rotary:
    def __call__(self, value, seq_len):
        width = value.shape[-1]
        return (
            torch.ones((seq_len, width), dtype=torch.float32),
            torch.zeros((seq_len, width), dtype=torch.float32),
        )


def _attention():
    hidden_size = 4
    q_lora_rank = 2
    kv_lora_rank = 2
    qk_nope = 2
    qk_rope = 2
    v_head = 2

    def linear(out_features, in_features):
        return types.SimpleNamespace(
            weight=types.SimpleNamespace(
                data=torch.empty((out_features, in_features))
            )
        )

    return types.SimpleNamespace(
        layer_idx=0,
        num_heads=1,
        q_head_dim=qk_nope + qk_rope,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head,
        softmax_scale=1.0,
        q_a_proj=linear(q_lora_rank, hidden_size),
        q_a_layernorm=_IdentityNorm(),
        q_b_proj=linear(qk_nope + qk_rope, q_lora_rank),
        kv_a_proj_with_mqa=linear(kv_lora_rank + qk_rope, hidden_size),
        kv_a_layernorm=_IdentityNorm(),
        kv_b_proj=linear(qk_nope + v_head, kv_lora_rank),
        o_proj=linear(hidden_size, v_head),
        rotary_emb=_Rotary(),
    )


def _scales():
    return {
        "q_a_proj.weight_scale_inv": torch.empty(0),
        "q_b_proj.weight_scale_inv": torch.empty(0),
        "kv_a_proj_with_mqa.weight_scale_inv": torch.empty(0),
        "kv_b_proj.weight_scale_inv": torch.empty(0),
        "o_proj.weight_scale_inv": torch.empty(0),
    }


def _run(monkeypatch):
    monkeypatch.setattr(
        fa3_backend,
        "flash_attn_varlen_func",
        lambda query, key, value, **kwargs: value,
    )
    hidden = torch.zeros((2, 4), dtype=torch.bfloat16)
    return fa3_backend.mla_prefill_flashattention3_w8a16_deepgemm_prepacked(
        _attention(),
        hidden,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.int32),
        2,
        1,
        _scales(),
    )


def test_q_a_and_kv_a_reuse_one_activation_quantization(monkeypatch):
    sentinel = (torch.empty(0), torch.empty(0))
    quantized = []
    gemm_calls = []

    def fake_quant(value):
        quantized.append(value)
        return sentinel

    def fake_gemm(weight, scale, activation, activation_fp8=None):
        gemm_calls.append((weight.shape[0], activation_fp8))
        return torch.zeros(
            (activation.shape[0], weight.shape[0]), dtype=torch.bfloat16
        )

    monkeypatch.delenv("BATCHGEN_W8A16_DEQUANT", raising=False)
    monkeypatch.setattr(fa3_backend, "act_quant", fake_quant)
    monkeypatch.setattr(fa3_backend, "w8a16_gemm", fake_gemm)

    output, offload_kv = _run(monkeypatch)

    assert len(quantized) == 1
    assert gemm_calls[0][1] is sentinel
    assert gemm_calls[1][1] is sentinel
    assert [call[1] for call in gemm_calls[2:]] == [None] * 3
    assert output.shape == (2, 4)
    assert offload_kv.shape == (2, 4)


def test_dequant_fallback_does_not_quantize_or_pass_fp8(monkeypatch):
    gemm_calls = []

    def fail_quant(value):
        raise AssertionError("dequant fallback must not quantize activations")

    def fake_dequant_gemm(weight, scale, activation):
        gemm_calls.append(weight.shape[0])
        return torch.zeros(
            (activation.shape[0], weight.shape[0]), dtype=torch.bfloat16
        )

    monkeypatch.setenv("BATCHGEN_W8A16_DEQUANT", "1")
    monkeypatch.setattr(fa3_backend, "act_quant", fail_quant)
    monkeypatch.setattr(fa3_backend, "w8a16_gemm_dequant", fake_dequant_gemm)

    output, offload_kv = _run(monkeypatch)

    assert len(gemm_calls) == 5
    assert output.shape == (2, 4)
    assert offload_kv.shape == (2, 4)
