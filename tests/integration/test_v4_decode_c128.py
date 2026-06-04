from __future__ import annotations

import pytest
import torch

from batchgen.attention.dsa.v4_flashmla_adapter import (
    DeepSeekV4FlashMLADecodeAdapter,
    build_v4_decode_attn_metadata,
)
from batchgen.attention.v4_backend import (
    DeepseekV4AttnBackend,
    build_layer_configs_from_compress_ratios,
)
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator
from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for V4 c128 decode"
)

_SWA_WINDOW = 128


def _make_rope_cache(
    max_pos: int, rope_dim: int = 64, base: float = 10000.0
) -> torch.Tensor:
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


def _apply_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_cache: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    out = x.clone()
    half = 32
    rope = out[..., -64:].float().view(*out.shape[:-1], half, 2)
    cache = rope_cache.index_select(0, positions.long())
    view_shape = (positions.shape[0],) + (1,) * (rope.ndim - 3) + (half,)
    cos = cache[:, :half].view(view_shape)
    sin = cache[:, half:].view(view_shape)
    even = rope[..., 0]
    odd = rope[..., 1]
    if inverse:
        rot_even = even * cos + odd * sin
        rot_odd = odd * cos - even * sin
    else:
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
    out[..., -64:] = (
        torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(x.dtype)
    )
    return out


def _rmsnorm_ref(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_fp32 = x.float()
    return (
        x_fp32
        * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    ).to(x.dtype)


def _canonical_c128_chunks(
    compressor: DeepSeekV4Compressor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    rope_cache: torch.Tensor,
) -> torch.Tensor:
    ratio = compressor.compress_ratio
    num_chunks = hidden_states.shape[0] // ratio
    if num_chunks == 0:
        return hidden_states.new_empty(0, compressor.head_dim)
    hidden_states = hidden_states[: num_chunks * ratio].float()
    positions = positions[: num_chunks * ratio]
    kv = compressor.wkv(hidden_states).view(
        num_chunks, ratio, compressor.head_dim
    )
    gate = compressor.wgate(hidden_states).view(
        num_chunks, ratio, compressor.head_dim
    )
    scores = gate + compressor.ape.view(ratio, compressor.head_dim).unsqueeze(0)
    weights = torch.softmax(scores, dim=1)
    pooled = (kv * weights).sum(dim=1)
    pooled = _rmsnorm_ref(pooled, compressor.norm.weight, compressor.norm.eps)
    chunk_starts = positions.view(num_chunks, ratio)[:, 0]
    return _apply_rope_ref(pooled.to(torch.bfloat16), chunk_starts, rope_cache)


def _dense_reference(
    q: torch.Tensor,
    selected_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    scores = (
        torch.einsum("bhd,btd->bht", q.float(), selected_kv.float())
        * softmax_scale
    )
    scores_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - scores_max)
    sink = torch.exp(attn_sink.float().view(1, -1, 1) - scores_max)
    weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + sink)
    return torch.einsum(
        "bht,btd->bhd", weights.to(selected_kv.dtype), selected_kv
    )


def test_v4_c128_decode_matches_independent_reference():
    device = torch.device("cuda")
    batch_size = 1
    seq_len = 256
    num_heads = 64
    head_dim = 512
    layer_idx = 2
    compress_ratios = [0, 4, 128]
    sequence_ids = [31337]
    softmax_scale = 512**-0.5

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

    compressor = DeepSeekV4Compressor(
        head_dim, head_dim, 64, 128, 1e-6, overlap=False
    ).to(device)
    canonical_compressed = _canonical_c128_chunks(
        compressor,
        hidden_states,
        torch.arange(seq_len, device=device, dtype=torch.int64),
        rope_cache,
    )

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

        actual = None
        for step in range(seq_len):
            cur_seq_len = step + 1
            metadata = build_v4_decode_attn_metadata(
                coordinator=coordinator,
                sequence_ids=sequence_ids,
                cache_seqlens=torch.tensor(
                    [cur_seq_len], dtype=torch.int32, device=device
                ),
                positions=torch.tensor(
                    [step], dtype=torch.int32, device=device
                ),
                page_tables=page_tables,
                rope_cache=rope_cache,
            )
            backend.init_metadata(metadata)
            actual = backend.forward(
                layer_config=layer_config,
                q=q_tokens[step : step + 1],
                kv=kv_tokens[step : step + 1],
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
                compressor=compressor,
                compress_hidden_states=hidden_states[step : step + 1],
            )

        route = coordinator.get_layer_routing(layer_idx)
        assert route.c128_layer_idx is not None
        c128_slots = coordinator.c128.sequence_token_slots(
            sequence_ids[0], [0, 1]
        )
        stored_compressed = coordinator.c128.debug_read_kv(
            layer_idx=route.c128_layer_idx,
            token_slots=c128_slots,
        )
        assert stored_compressed.shape == canonical_compressed[:2].shape

        full_swa = _apply_rope_ref(
            kv_tokens,
            torch.arange(seq_len, device=device, dtype=torch.long),
            rope_cache,
        )
        selected_window = full_swa[-_SWA_WINDOW:].unsqueeze(0)
        selected_kv = torch.cat(
            (
                selected_window,
                canonical_compressed[: seq_len // 128].unsqueeze(0),
            ),
            dim=1,
        )
        q_roped = _apply_rope_ref(
            q_tokens[-1:].clone(),
            torch.tensor([seq_len - 1], device=device, dtype=torch.long),
            rope_cache,
        )
        expected = _dense_reference(
            q=q_roped,
            selected_kv=selected_kv,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
        )
        expected = _apply_rope_ref(
            expected,
            torch.tensor([seq_len - 1], device=device, dtype=torch.long),
            rope_cache,
            inverse=True,
        )

        assert actual is not None
        assert actual.shape == (batch_size, num_heads, head_dim)
        assert torch.allclose(actual, expected, atol=0.05, rtol=0)
    finally:
        coordinator.destroy()
