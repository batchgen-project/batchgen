from __future__ import annotations

from typing import Any

import torch

from batchgen.kv_cache.deepseek_v4_single_kv_pool import (
    HEAD_DIM,
    NOPE_DIM,
    TOKEN_BYTES,
    TOKEN_DATA_SIZE,
    dequantize_nope_from_fp8,
)

_TOKEN_SCALE_BYTES = TOKEN_BYTES - TOKEN_DATA_SIZE


def _validate_k_cache_tensor(k_cache: torch.Tensor) -> tuple[int, int]:
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            f"k_cache must have shape [num_pages, page_size, 1, {TOKEN_BYTES}], got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != TOKEN_BYTES:
        raise ValueError(
            f"expected packed token size {TOKEN_BYTES}, got {k_cache.shape[-1]}"
        )
    if k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"k_cache must use torch.float8_e4m3fn view, got {k_cache.dtype}"
        )
    return int(k_cache.shape[0]), int(k_cache.shape[1])


def _select_token_slots(
    indices_row: torch.Tensor,
    valid_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if indices_row.ndim != 1:
        raise ValueError(
            f"indices row must be 1D, got {tuple(indices_row.shape)}"
        )
    if valid_length < 0 or valid_length > indices_row.shape[0]:
        raise ValueError(
            f"valid length {valid_length} is out of range for {indices_row.shape[0]} indices"
        )
    token_slots = indices_row[:valid_length].to(
        device=device, dtype=torch.int64
    )
    if token_slots.numel() == 0:
        return token_slots
    return token_slots[token_slots >= 0]


def _gather_packed_rows_from_raw_cache(
    k_cache: torch.Tensor,
    token_slots: torch.Tensor,
) -> torch.Tensor:
    num_pages, page_size_tokens = _validate_k_cache_tensor(k_cache)
    rows = token_slots.to(device=k_cache.device, dtype=torch.int64).view(-1)
    if rows.numel() == 0:
        return torch.empty(
            (0, TOKEN_BYTES), dtype=torch.uint8, device=k_cache.device
        )

    max_slot = num_pages * page_size_tokens
    if (rows < 0).any() or (rows >= max_slot).any():
        raise ValueError(
            f"token slots out of range for capacity {max_slot}: {rows.tolist()}"
        )

    # Matches DeepSeekV4SingleKVPool._gather_packed_rows / debug_read_kv.
    raw_pages = k_cache.view(torch.uint8).reshape(
        num_pages, page_size_tokens * TOKEN_BYTES
    )
    body_bytes = page_size_tokens * TOKEN_DATA_SIZE
    scale_bytes = page_size_tokens * _TOKEN_SCALE_BYTES
    body_view = raw_pages[:, :body_bytes].reshape(
        num_pages, page_size_tokens, TOKEN_DATA_SIZE
    )
    scale_view = raw_pages[:, body_bytes : body_bytes + scale_bytes].reshape(
        num_pages, page_size_tokens, _TOKEN_SCALE_BYTES
    )

    page_indices = torch.div(rows, page_size_tokens, rounding_mode="floor")
    token_offsets = torch.remainder(rows, page_size_tokens)
    bodies = body_view[page_indices, token_offsets]
    scales = scale_view[page_indices, token_offsets]
    return torch.cat((bodies, scales), dim=-1)


def _gather_kv_from_raw_cache(
    k_cache: torch.Tensor,
    token_slots: torch.Tensor,
) -> torch.Tensor:
    packed = _gather_packed_rows_from_raw_cache(k_cache, token_slots)
    if packed.numel() == 0:
        return torch.empty(
            (0, HEAD_DIM), dtype=torch.bfloat16, device=k_cache.device
        )

    nope_fp8 = packed[:, :NOPE_DIM].contiguous().view(torch.float8_e4m3fn)
    rope = packed[:, NOPE_DIM:TOKEN_DATA_SIZE].contiguous().view(torch.bfloat16)
    scales = packed[:, TOKEN_DATA_SIZE:TOKEN_BYTES][:, : NOPE_DIM // 64]
    nope = dequantize_nope_from_fp8(nope_fp8, scales)
    return torch.cat((nope.to(torch.bfloat16), rope), dim=-1)


def flashmla_decode_torch_reference(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    head_dim_v: int,
    tile_scheduler_metadata: Any,
    num_splits: Any,
    softmax_scale: float,
    causal: bool,
    is_fp8_kvcache: bool,
    indices: torch.Tensor,
    attn_sink: torch.Tensor | None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    del block_table, cache_seqlens, tile_scheduler_metadata, num_splits

    if causal:
        raise NotImplementedError(
            "flashmla_decode_torch_reference only supports causal=False"
        )
    if not is_fp8_kvcache:
        raise NotImplementedError(
            "flashmla_decode_torch_reference only supports fp8 KV cache"
        )
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"q must have shape [B, 1, H, {HEAD_DIM}], got {tuple(q.shape)}"
        )
    if head_dim_v != HEAD_DIM:
        raise ValueError(
            f"head_dim_v must be {HEAD_DIM} for V4 MLA decode, got {head_dim_v}"
        )
    if (
        indices.ndim != 3
        or indices.shape[0] != q.shape[0]
        or indices.shape[1] != 1
    ):
        raise ValueError(
            f"indices must have shape [B, 1, K], got {tuple(indices.shape)}"
        )
    if topk_length is None:
        raise ValueError("topk_length is required")
    if topk_length.ndim != 1 or topk_length.shape[0] != q.shape[0]:
        raise ValueError(
            f"topk_length must have shape [{q.shape[0]}], got {tuple(topk_length.shape)}"
        )
    if attn_sink is not None and (
        attn_sink.ndim != 1 or attn_sink.shape[0] != q.shape[2]
    ):
        raise ValueError(
            f"attn_sink must have shape [{q.shape[2]}], got {tuple(attn_sink.shape)}"
        )
    if extra_k_cache is not None:
        if extra_indices_in_kvcache is None or extra_topk_length is None:
            raise ValueError(
                "extra_k_cache requires extra_indices_in_kvcache and extra_topk_length"
            )
        if (
            extra_indices_in_kvcache.ndim != 3
            or extra_indices_in_kvcache.shape[0] != q.shape[0]
            or extra_indices_in_kvcache.shape[1] != 1
        ):
            raise ValueError(
                "extra_indices_in_kvcache must have shape [B, 1, K_extra]"
            )
        if (
            extra_topk_length.ndim != 1
            or extra_topk_length.shape[0] != q.shape[0]
        ):
            raise ValueError(
                f"extra_topk_length must have shape [{q.shape[0]}], got {tuple(extra_topk_length.shape)}"
            )

    batch_size = q.shape[0]
    num_heads = q.shape[2]
    attn_out = torch.zeros(
        (batch_size, 1, num_heads, head_dim_v),
        dtype=q.dtype,
        device=q.device,
    )
    main_lengths = topk_length.to(device=q.device, dtype=torch.int64)
    extra_lengths = None
    if extra_topk_length is not None:
        extra_lengths = extra_topk_length.to(device=q.device, dtype=torch.int64)
    sink = None
    if attn_sink is not None:
        sink = attn_sink.to(device=q.device, dtype=torch.float32).view(
            num_heads, 1
        )

    for batch_idx in range(batch_size):
        kv_chunks: list[torch.Tensor] = []

        main_slots = _select_token_slots(
            indices[batch_idx, 0],
            int(main_lengths[batch_idx].item()),
            device=k_cache.device,
        )
        if main_slots.numel() > 0:
            kv_chunks.append(_gather_kv_from_raw_cache(k_cache, main_slots))

        if (
            extra_k_cache is not None
            and extra_indices_in_kvcache is not None
            and extra_lengths is not None
        ):
            extra_slots = _select_token_slots(
                extra_indices_in_kvcache[batch_idx, 0],
                int(extra_lengths[batch_idx].item()),
                device=extra_k_cache.device,
            )
            if extra_slots.numel() > 0:
                kv_chunks.append(
                    _gather_kv_from_raw_cache(extra_k_cache, extra_slots)
                )

        if kv_chunks:
            kv_rows = torch.cat(kv_chunks, dim=0).to(device=q.device)
        else:
            kv_rows = torch.empty(
                (0, HEAD_DIM), dtype=torch.bfloat16, device=q.device
            )

        if kv_rows.numel() == 0 and sink is None:
            continue

        q_row = q[batch_idx, 0].to(dtype=torch.float32)
        kv_rows_f32 = kv_rows.to(dtype=torch.float32)
        if kv_rows_f32.shape[0] > 0:
            scores = torch.matmul(q_row, kv_rows_f32.transpose(0, 1))
            scores = scores * softmax_scale
        else:
            scores = torch.empty(
                (num_heads, 0), dtype=torch.float32, device=q.device
            )

        if sink is not None:
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[
                :, : scores.shape[-1]
            ]
        else:
            probs = torch.softmax(scores, dim=-1)

        attn_out[batch_idx, 0] = torch.matmul(probs, kv_rows_f32).to(q.dtype)

    return attn_out


__all__ = ["flashmla_decode_torch_reference"]
