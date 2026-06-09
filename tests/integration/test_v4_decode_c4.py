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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for V4 c4 decode"
)

_SWA_WINDOW = 128
_TOPK = 512


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


def _naive_c4_topk(
    q_index: torch.Tensor,
    cached_k: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    scores = torch.einsum("bhd,btd->bht", q_index.float(), cached_k.float())
    scores = scores * head_gates.float().unsqueeze(-1)
    aggregated = scores.sum(dim=1)
    key_pos = torch.arange(cached_k.size(1), device=cached_k.device).unsqueeze(
        0
    )
    aggregated = aggregated.masked_fill(
        key_pos >= cache_seqlens.long().unsqueeze(1), float("-inf")
    )
    effective_topk = min(
        topk, cached_k.size(1), int(cache_seqlens.min().item())
    )
    return torch.topk(aggregated, effective_topk, dim=-1).indices


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


def test_v4_c4_decode_matches_independent_sparse_reference():
    device = torch.device("cuda")
    batch_size = 1
    seq_len = 2304
    compress_len = seq_len // 4
    assert compress_len > _TOPK
    num_heads = 64
    head_dim = 512
    index_dim = 128
    layer_idx = 1
    compress_ratios = [0, 4, 128]
    cache_seqlens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    compress_cache_seqlens = torch.tensor(
        [compress_len], dtype=torch.int32, device=device
    )
    sequence_ids = [2026]
    positions = cache_seqlens.long() - 1
    rope_cache = _make_rope_cache(seq_len + 4)
    softmax_scale = 512**-0.5

    torch.manual_seed(0)
    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=compress_ratios,
        num_pages=2048,
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()

    try:
        coordinator.allocate_pages_for_sequences(
            sequence_ids, cache_seqlens.tolist()
        )
        page_tables = coordinator.rebuild_page_table(sequence_ids)
        route = coordinator.get_layer_routing(layer_idx)
        assert route.c4_layer_idx is not None
        assert route.indexer_layer_idx is not None

        q_index = torch.randn(
            batch_size,
            num_heads,
            index_dim,
            dtype=torch.bfloat16,
            device=device,
        ).clamp_(-1, 1)
        q_attn = torch.randn(
            batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        q_attn = (
            q_attn
            * torch.rsqrt(q_attn.square().mean(dim=-1, keepdim=True) + 1e-6)
        ).clamp_(-1, 1)
        head_gates = torch.rand(
            batch_size, num_heads, dtype=torch.float32, device=device
        )
        attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)

        swa_history = (
            torch.randn(
                batch_size,
                seq_len - 1,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            .div_(10)
            .clamp_(-1, 1)
        )
        current_kv = (
            torch.randn(
                batch_size, head_dim, dtype=torch.bfloat16, device=device
            )
            .div_(10)
            .clamp_(-1, 1)
        )
        c4_kv = (
            torch.randn(
                batch_size,
                compress_len,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            .div_(10)
            .clamp_(-1, 1)
        )
        indexer_cached_k = (
            torch.randn(
                batch_size,
                compress_len,
                index_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            .div_(10)
            .clamp_(-1, 1)
        )

        hist_pos = torch.arange(seq_len - 1, device=device, dtype=torch.long)
        swa_history_roped = _apply_rope_ref(
            swa_history[0], hist_pos, rope_cache
        )
        swa_slots = coordinator.swa.sequence_token_slots(
            sequence_ids[0], hist_pos
        )
        coordinator.swa.store_kv(
            layer_idx=route.swa_layer_idx,
            token_slots=swa_slots,
            kv_processed=swa_history_roped,
        )

        c4_positions = torch.arange(
            compress_len, device=device, dtype=torch.long
        )
        c4_slots = coordinator.c4.sequence_token_slots(
            sequence_ids[0], c4_positions
        )
        coordinator.c4.store_kv(
            layer_idx=route.c4_layer_idx,
            token_slots=c4_slots,
            kv_processed=c4_kv[0],
        )

        index_slots = coordinator.indexer.sequence_token_slots(
            sequence_ids[0], c4_positions
        )
        coordinator.indexer.store_indexer(
            layer_idx=route.indexer_layer_idx,
            token_slots=index_slots,
            index_k=indexer_cached_k[0],
        )

        gathered_index_k = coordinator.indexer.debug_read_indexer(
            layer_idx=route.indexer_layer_idx,
            token_slots=index_slots,
        ).view(batch_size, compress_len, index_dim)
        assert torch.allclose(
            gathered_index_k, indexer_cached_k, atol=0.05, rtol=0
        )

        metadata = build_v4_decode_attn_metadata(
            coordinator=coordinator,
            sequence_ids=sequence_ids,
            cache_seqlens=cache_seqlens,
            positions=positions,
            page_tables=page_tables,
            rope_cache=rope_cache,
        )
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
        backend.init_metadata(metadata)

        actual = backend.forward(
            layer_config=layer_config,
            q=q_index,
            kv=gathered_index_k,
            attn_sink=attn_sink,
            head_gates=head_gates,
            q_attn=q_attn,
            current_kv=current_kv,
            softmax_scale=softmax_scale,
        )

        topk_indices = _naive_c4_topk(
            q_index=q_index,
            cached_k=gathered_index_k,
            head_gates=head_gates,
            cache_seqlens=compress_cache_seqlens,
            topk=_TOPK,
        )
        window_start = seq_len - _SWA_WINDOW
        full_swa = torch.cat(
            (
                swa_history_roped,
                _apply_rope_ref(current_kv, positions, rope_cache),
            ),
            dim=0,
        )
        selected_window = full_swa[window_start:seq_len].unsqueeze(0)
        selected_c4 = c4_kv.index_select(1, topk_indices[0].long())
        selected_kv = torch.cat((selected_window, selected_c4), dim=1)

        q_attn_roped = _apply_rope_ref(q_attn, positions, rope_cache)
        expected = _dense_reference(
            q=q_attn_roped,
            selected_kv=selected_kv,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
        )
        expected = _apply_rope_ref(
            expected, positions, rope_cache, inverse=True
        )

        assert topk_indices.shape == (batch_size, _TOPK)
        assert actual.shape == (batch_size, num_heads, head_dim)
        assert torch.allclose(actual, expected, atol=0.05, rtol=0)
    finally:
        coordinator.destroy()
