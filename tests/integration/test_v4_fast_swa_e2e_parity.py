from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_rope_cache(max_pos, rope_dim=64, base=10000.0):
    device = torch.device("cuda")
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
            / rope_dim
        )
    )
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _run_c128_decode(fast_enabled, seq_len=200):
    import batchgen.attention.dsa.v4_flashmla_adapter as adapter
    from batchgen.attention.dsa.v4_flashmla_adapter import (
        DeepSeekV4FlashMLADecodeAdapter,
        build_v4_decode_attn_metadata,
    )

    adapter._V4_FAST_PREFIX_INDICES = fast_enabled
    from batchgen.attention.v4_backend import (
        DeepseekV4AttnBackend,
        build_layer_configs_from_compress_ratios,
    )
    from batchgen.kv_cache.deepseek_v4_kv_coordinator import (
        DeepSeekV4KVCoordinator,
    )
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    device = torch.device("cuda")
    num_heads = 64
    head_dim = 512
    layer_idx = 2
    compress_ratios = [0, 4, 128]
    sequence_ids = [31337]
    softmax_scale = head_dim**-0.5

    torch.manual_seed(0)
    hidden_states = (
        torch.randn(seq_len, head_dim, dtype=torch.float32, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    kv_tokens = (
        torch.randn(seq_len, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    q_tokens = torch.randn(
        seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    q_tokens = (
        q_tokens
        * torch.rsqrt(q_tokens.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).clamp_(-1, 1)
    rope_cache = _make_rope_cache(seq_len + 4)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)

    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=compress_ratios,
        num_pages=256,
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [seq_len])
        page_tables = coordinator.rebuild_page_table(sequence_ids)
        layer_config = build_layer_configs_from_compress_ratios(
            compress_ratios=compress_ratios,
            n_heads=num_heads,
            head_dim=head_dim,
            rope_head_dim=64,
        )[layer_idx]
        backend = DeepseekV4AttnBackend(
            layer_configs=[layer_config],
            page_size=coordinator.swa.page_size_tokens,
            flashmla_backend=DeepSeekV4FlashMLADecodeAdapter(coordinator),
        )
        compressor = DeepSeekV4Compressor(
            head_dim, head_dim, 64, 128, 1e-6, overlap=False
        ).to(device)

        outs = []
        for step in range(seq_len):
            metadata = build_v4_decode_attn_metadata(
                coordinator=coordinator,
                sequence_ids=sequence_ids,
                cache_seqlens=torch.tensor(
                    [step + 1], dtype=torch.int32, device=device
                ),
                positions=torch.tensor(
                    [step], dtype=torch.int32, device=device
                ),
                page_tables=page_tables,
                rope_cache=rope_cache,
            )
            backend.init_metadata(metadata)
            out = backend.forward(
                layer_config=layer_config,
                q=q_tokens[step : step + 1],
                kv=kv_tokens[step : step + 1],
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
                compressor=compressor,
                compress_hidden_states=hidden_states[step : step + 1],
            )
            outs.append(out.clone())
        return torch.stack(outs)
    finally:
        coordinator.destroy()


def test_fast_swa_c128_e2e_matches_slow():
    import batchgen.attention.dsa.v4_flashmla_adapter as adapter

    saved = adapter._V4_FAST_PREFIX_INDICES
    try:
        out_fast = _run_c128_decode(fast_enabled=True)
        out_slow = _run_c128_decode(fast_enabled=False)
    finally:
        adapter._V4_FAST_PREFIX_INDICES = saved
    assert out_fast.shape == out_slow.shape
    torch.testing.assert_close(out_fast, out_slow, atol=0.0, rtol=0.0)
