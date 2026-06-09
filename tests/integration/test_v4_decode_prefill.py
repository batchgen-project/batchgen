from __future__ import annotations

import pytest
import torch

from batchgen.attention.dsa.v4_flashmla_adapter import (
    DeepSeekV4FlashMLADecodeAdapter,
    build_v4_decode_attn_metadata,
)
from batchgen.attention.dsa.v4_prefill_populate import (
    populate_v4_prefill_coordinator,
)
from batchgen.attention.v4_backend import (
    DeepseekV4AttnBackend,
    build_layer_configs_from_compress_ratios,
)
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator
from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for V4 prefill/decode"
)

_SWA_WINDOW = 128
_C4_TOPK = 512


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


def _dense_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    scores = (
        torch.einsum("bhd,btd->bht", q.float(), kv_cache.float())
        * softmax_scale
    )
    scores_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - scores_max)
    sink = torch.exp(attn_sink.float().view(1, -1, 1) - scores_max)
    weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + sink)
    return torch.einsum("bht,btd->bhd", weights.to(kv_cache.dtype), kv_cache)


def _naive_c4_topk(
    q_index: torch.Tensor,
    cached_k: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
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
    return torch.topk(
        aggregated,
        min(_C4_TOPK, cached_k.size(1), int(cache_seqlens.min().item())),
        dim=-1,
    ).indices


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
    hidden_states = hidden_states[: num_chunks * ratio].float()
    positions = positions[: num_chunks * ratio]
    if hidden_states.numel() == 0:
        return hidden_states.new_empty(0, compressor.head_dim)
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


def _make_backend(
    *,
    coordinator: DeepSeekV4KVCoordinator,
    compress_ratios: list[int],
    layer_idx: int,
    num_heads: int,
    head_dim: int,
) -> tuple[DeepseekV4AttnBackend, object]:
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
    return backend, layer_config


def test_prefill_populate_dense_handoff():
    device = torch.device("cuda")
    compress_ratios = [0, 4, 128]
    layer_idx = 0
    prompt_len = 9
    total_len = prompt_len + 1
    num_heads = 64
    head_dim = 512
    sequence_id = 7001
    rope_cache = _make_rope_cache(total_len + 4)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)
    softmax_scale = 512**-0.5

    torch.manual_seed(0)
    prompt_kv = (
        torch.randn(prompt_len, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    current_kv = (
        torch.randn(1, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q = (q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + 1e-6)).clamp_(
        -1, 1
    )

    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=compress_ratios,
        num_pages=16,
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()
    try:
        coordinator.allocate_pages_for_sequences([sequence_id], [total_len])
        page_tables = coordinator.rebuild_page_table([sequence_id])
        prompt_positions = torch.arange(
            prompt_len, device=device, dtype=torch.long
        )
        populate_v4_prefill_coordinator(
            coordinator=coordinator,
            layer_idx=layer_idx,
            sequence_id=sequence_id,
            prompt_positions=prompt_positions,
            swa_kv=prompt_kv,
            rope_cache=rope_cache,
        )
        metadata = build_v4_decode_attn_metadata(
            coordinator=coordinator,
            sequence_ids=[sequence_id],
            cache_seqlens=torch.tensor(
                [total_len], dtype=torch.int32, device=device
            ),
            positions=torch.tensor(
                [prompt_len], dtype=torch.int32, device=device
            ),
            page_tables=page_tables,
            rope_cache=rope_cache,
        )
        backend, layer_config = _make_backend(
            coordinator=coordinator,
            compress_ratios=compress_ratios,
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        backend.init_metadata(metadata)
        actual = backend.forward(
            layer_config=layer_config,
            q=q,
            kv=current_kv,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
        )

        prompt_roped = _apply_rope_ref(prompt_kv, prompt_positions, rope_cache)
        current_roped = _apply_rope_ref(
            current_kv,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
        )
        q_roped = _apply_rope_ref(
            q,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
        )
        expected = _dense_reference(
            q_roped,
            torch.cat((prompt_roped, current_roped), dim=0).unsqueeze(0),
            attn_sink,
            softmax_scale,
        )
        expected = _apply_rope_ref(
            expected,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
            inverse=True,
        )
        assert torch.allclose(actual, expected, atol=0.05, rtol=0)
    finally:
        coordinator.destroy()


def test_prefill_populate_c4_handoff():
    device = torch.device("cuda")
    compress_ratios = [0, 4, 128]
    layer_idx = 1
    prompt_len = 2304
    total_len = prompt_len + 1
    compress_len = prompt_len // 4
    assert compress_len > _C4_TOPK
    num_heads = 64
    head_dim = 512
    index_dim = 128
    sequence_id = 7002
    rope_cache = _make_rope_cache(total_len + 4)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)
    softmax_scale = 512**-0.5

    torch.manual_seed(1)
    prompt_kv = (
        torch.randn(prompt_len, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    c4_kv = (
        torch.randn(compress_len, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    indexer_k = (
        torch.randn(
            compress_len, index_dim, dtype=torch.bfloat16, device=device
        )
        .div_(10)
        .clamp_(-1, 1)
    )
    current_kv = (
        torch.randn(1, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    q_index = torch.randn(
        1, num_heads, index_dim, dtype=torch.bfloat16, device=device
    ).clamp_(-1, 1)
    q_attn = torch.randn(
        1, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    q_attn = (
        q_attn * torch.rsqrt(q_attn.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).clamp_(-1, 1)
    head_gates = torch.rand(1, num_heads, dtype=torch.float32, device=device)

    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=compress_ratios,
        num_pages=2048,
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()
    try:
        coordinator.allocate_pages_for_sequences([sequence_id], [total_len])
        page_tables = coordinator.rebuild_page_table([sequence_id])
        prompt_positions = torch.arange(
            prompt_len, device=device, dtype=torch.long
        )
        populate_v4_prefill_coordinator(
            coordinator=coordinator,
            layer_idx=layer_idx,
            sequence_id=sequence_id,
            prompt_positions=prompt_positions,
            swa_kv=prompt_kv,
            rope_cache=rope_cache,
            c4_kv=c4_kv,
            indexer_k=indexer_k,
        )
        route = coordinator.get_layer_routing(layer_idx)
        assert route.indexer_layer_idx is not None
        index_slots = coordinator.indexer.sequence_token_slots(
            sequence_id,
            torch.arange(compress_len, device=device, dtype=torch.long),
        )
        cached_k = coordinator.indexer.debug_read_indexer(
            layer_idx=route.indexer_layer_idx,
            token_slots=index_slots,
        ).view(1, compress_len, index_dim)

        metadata = build_v4_decode_attn_metadata(
            coordinator=coordinator,
            sequence_ids=[sequence_id],
            cache_seqlens=torch.tensor(
                [total_len], dtype=torch.int32, device=device
            ),
            positions=torch.tensor(
                [prompt_len], dtype=torch.int32, device=device
            ),
            page_tables=page_tables,
            rope_cache=rope_cache,
        )
        backend, layer_config = _make_backend(
            coordinator=coordinator,
            compress_ratios=compress_ratios,
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        backend.init_metadata(metadata)
        actual = backend.forward(
            layer_config=layer_config,
            q=q_index,
            kv=cached_k,
            attn_sink=attn_sink,
            head_gates=head_gates,
            q_attn=q_attn,
            current_kv=current_kv,
            softmax_scale=softmax_scale,
        )

        topk = _naive_c4_topk(
            q_index=q_index,
            cached_k=indexer_k.unsqueeze(0),
            head_gates=head_gates,
            cache_seqlens=torch.tensor(
                [compress_len], dtype=torch.int32, device=device
            ),
        )
        prompt_roped = _apply_rope_ref(prompt_kv, prompt_positions, rope_cache)
        current_roped = _apply_rope_ref(
            current_kv,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
        )
        selected_window = torch.cat(
            (prompt_roped[-(_SWA_WINDOW - 1) :], current_roped), dim=0
        ).unsqueeze(0)
        selected_c4 = c4_kv.index_select(0, topk[0].long()).unsqueeze(0)
        q_attn_roped = _apply_rope_ref(
            q_attn,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
        )
        expected = _dense_reference(
            q_attn_roped,
            torch.cat((selected_window, selected_c4), dim=1),
            attn_sink,
            softmax_scale,
        )
        expected = _apply_rope_ref(
            expected,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
            inverse=True,
        )
        assert torch.allclose(actual, expected, atol=0.05, rtol=0)
    finally:
        coordinator.destroy()


def test_prefill_populate_c128_handoff():
    device = torch.device("cuda")
    compress_ratios = [0, 4, 128]
    layer_idx = 2
    prompt_len = 256
    total_len = prompt_len + 1
    num_heads = 64
    head_dim = 512
    sequence_id = 7003
    rope_cache = _make_rope_cache(total_len + 4)
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)
    softmax_scale = 512**-0.5

    torch.manual_seed(2)
    prompt_hidden = (
        torch.randn(prompt_len, head_dim, dtype=torch.float32, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    prompt_kv = (
        torch.randn(prompt_len, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    current_kv = (
        torch.randn(1, head_dim, dtype=torch.bfloat16, device=device)
        .div_(10)
        .clamp_(-1, 1)
    )
    q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    q = (q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + 1e-6)).clamp_(
        -1, 1
    )
    compressor = DeepSeekV4Compressor(
        head_dim, head_dim, 64, 128, 1e-6, overlap=False
    ).to(device)
    canonical_compressed = _canonical_c128_chunks(
        compressor,
        prompt_hidden,
        torch.arange(prompt_len, device=device, dtype=torch.int64),
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
        coordinator.allocate_pages_for_sequences([sequence_id], [total_len])
        page_tables = coordinator.rebuild_page_table([sequence_id])
        prompt_positions = torch.arange(
            prompt_len, device=device, dtype=torch.long
        )
        populated = populate_v4_prefill_coordinator(
            coordinator=coordinator,
            layer_idx=layer_idx,
            sequence_id=sequence_id,
            prompt_positions=prompt_positions,
            swa_kv=prompt_kv,
            rope_cache=rope_cache,
            c128_hidden_states=prompt_hidden,
            compressor=compressor,
        )
        assert torch.allclose(
            populated["c128_kv"].to(torch.bfloat16),
            canonical_compressed,
            atol=0.05,
            rtol=0,
        )

        metadata = build_v4_decode_attn_metadata(
            coordinator=coordinator,
            sequence_ids=[sequence_id],
            cache_seqlens=torch.tensor(
                [total_len], dtype=torch.int32, device=device
            ),
            positions=torch.tensor(
                [prompt_len], dtype=torch.int32, device=device
            ),
            page_tables=page_tables,
            rope_cache=rope_cache,
        )
        backend, layer_config = _make_backend(
            coordinator=coordinator,
            compress_ratios=compress_ratios,
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        backend.init_metadata(metadata)
        actual = backend.forward(
            layer_config=layer_config,
            q=q,
            kv=current_kv,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
        )

        prompt_roped = _apply_rope_ref(prompt_kv, prompt_positions, rope_cache)
        current_roped = _apply_rope_ref(
            current_kv,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
        )
        selected_window = torch.cat(
            (prompt_roped[-(_SWA_WINDOW - 1) :], current_roped), dim=0
        ).unsqueeze(0)
        q_roped = _apply_rope_ref(
            q,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
        )
        expected = _dense_reference(
            q_roped,
            torch.cat(
                (selected_window, canonical_compressed.unsqueeze(0)), dim=1
            ),
            attn_sink,
            softmax_scale,
        )
        expected = _apply_rope_ref(
            expected,
            torch.tensor([prompt_len], device=device, dtype=torch.long),
            rope_cache,
            inverse=True,
        )
        assert torch.allclose(actual, expected, atol=0.05, rtol=0)
    finally:
        coordinator.destroy()
