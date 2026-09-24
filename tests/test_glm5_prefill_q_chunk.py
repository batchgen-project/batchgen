from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA3 requires CUDA")
def test_single_sequence_q_chunks_match_full_fa3(monkeypatch):
    from batchgen.attention.mla import fa3_backend

    torch.manual_seed(23)
    device = "cuda"
    dtype = torch.bfloat16
    tokens, hidden, heads, q_lora, kv_lora = 19, 128, 2, 64, 64
    nope, rope, value = 192, 64, 256

    def linear(out_dim, in_dim):
        return SimpleNamespace(weight=torch.randn(out_dim, in_dim, device=device, dtype=dtype) * 0.02)

    def rotary(_x, seq_len):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freq = torch.arange(rope // 2, device=device, dtype=torch.float32) / (rope // 2)
        phase = positions[:, None] * torch.exp(-freq)[None, :]
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos(), phase.sin()

    attn = SimpleNamespace(
        num_heads=heads,
        q_head_dim=nope + rope,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=value,
        kv_lora_rank=kv_lora,
        softmax_scale=(nope + rope) ** -0.5,
        q_a_proj=linear(q_lora, hidden),
        q_a_layernorm=lambda x: x,
        q_b_proj=linear(heads * (nope + rope), q_lora),
        kv_a_proj_with_mqa=linear(kv_lora + rope, hidden),
        kv_a_layernorm=lambda x: x,
        kv_b_proj=linear(heads * (nope + value), kv_lora),
        o_proj=linear(hidden, heads * value),
        rotary_emb=rotary,
    )
    scales = {
        f"{name}.weight_scale_inv": torch.ones(1, device=device)
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
    }
    hidden_states = torch.randn(tokens, hidden, device=device, dtype=dtype)
    positions = torch.arange(tokens, device=device)
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    def gemm(weight, _scale, activation):
        return F.linear(activation, weight)

    monkeypatch.setattr(fa3_backend, "w8a16_gemm", gemm)
    full_out, full_kv = fa3_backend.mla_prefill_flashattention3_w8a16_deepgemm_prepacked(
        attn, hidden_states, positions, cu_seqlens, tokens, 1, scales,
    )
    chunk_out, chunk_kv = fa3_backend._mla_prefill_single_sequence_chunked(
        attn, hidden_states, positions, tokens, scales, gemm, 5,
    )

    torch.testing.assert_close(chunk_out, full_out, atol=0.02, rtol=0.02)
    torch.testing.assert_close(chunk_kv, full_kv, atol=0.02, rtol=0.02)
