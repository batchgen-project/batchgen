from __future__ import annotations

from typing import Any, Optional

import torch

from batchgen.attention.dsa.v4_flashmla_adapter import _apply_rope
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator


def populate_v4_prefill_coordinator(
    *,
    coordinator: DeepSeekV4KVCoordinator,
    layer_idx: int,
    sequence_id: int,
    prompt_positions: torch.Tensor,
    swa_kv: torch.Tensor,
    rope_cache: torch.Tensor,
    c4_kv: Optional[torch.Tensor] = None,
    indexer_k: Optional[torch.Tensor] = None,
    c128_hidden_states: Optional[torch.Tensor] = None,
    compressor: Optional[Any] = None,
    compress_rope_cache: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Populate DeepSeek-V4 coordinator pools for a prompt.

    This helper is intentionally side-effect only: it does not change prefill
    attention math, it only writes the prompt-resident KV/views that later decode
    steps consume.

    Expected inputs are the prefill-produced tensors for a single sequence:
    * ``swa_kv``: prompt KV after ``wkv`` + ``kv_norm`` and before RoPE.
    * ``c4_kv`` / ``indexer_k``: prompt compressed/indexer projections for ratio-4.
    * ``c128_hidden_states`` + ``compressor``: prompt hidden states for ratio-128.
    """

    if prompt_positions.ndim != 1:
        raise ValueError(
            f"prompt_positions must be 1D, got {tuple(prompt_positions.shape)}"
        )
    if swa_kv.ndim != 2:
        raise ValueError(f"swa_kv must be [T,D], got {tuple(swa_kv.shape)}")
    if swa_kv.shape[0] != prompt_positions.shape[0]:
        raise ValueError(
            "swa_kv and prompt_positions must align in sequence length"
        )

    route = coordinator.get_layer_routing(layer_idx)
    swa_kv_roped = _apply_rope(swa_kv, prompt_positions, rope_cache)
    swa_slots = coordinator.swa.sequence_token_slots(
        sequence_id, prompt_positions
    )
    coordinator.swa.store_kv(
        layer_idx=route.swa_layer_idx,
        token_slots=swa_slots,
        kv_processed=swa_kv_roped.contiguous(),
    )

    outputs = {"swa_kv_roped": swa_kv_roped}

    if route.c4_layer_idx is not None:
        if c4_kv is None or indexer_k is None:
            raise ValueError(
                "ratio-4 prefill population requires c4_kv and indexer_k"
            )
        if c4_kv.ndim != 2 or indexer_k.ndim != 2:
            raise ValueError("c4_kv and indexer_k must be rank-2 tensors")
        if c4_kv.shape[0] != indexer_k.shape[0]:
            raise ValueError("c4_kv and indexer_k must align in token count")
        c4_positions = torch.arange(
            c4_kv.shape[0], device=prompt_positions.device, dtype=torch.long
        )
        c4_slots = coordinator.c4.sequence_token_slots(
            sequence_id, c4_positions
        )
        coordinator.c4.store_kv(
            layer_idx=route.c4_layer_idx,
            token_slots=c4_slots,
            kv_processed=c4_kv.to(torch.bfloat16).contiguous(),
        )
        indexer_slots = coordinator.indexer.sequence_token_slots(
            sequence_id, c4_positions
        )
        coordinator.indexer.store_indexer(
            layer_idx=route.indexer_layer_idx,
            token_slots=indexer_slots,
            index_k=indexer_k.to(torch.bfloat16).contiguous(),
        )
        outputs["c4_kv"] = c4_kv
        outputs["indexer_k"] = indexer_k

    if route.c128_layer_idx is not None:
        if c128_hidden_states is None or compressor is None:
            raise ValueError(
                "ratio-128 prefill population requires c128_hidden_states and compressor"
            )
        compressed = compressor.forward_prefill(
            c128_hidden_states,
            prompt_positions.to(
                dtype=torch.int64, device=c128_hidden_states.device
            ),
            compress_rope_cache
            if compress_rope_cache is not None
            else rope_cache,
        )
        if compressed.numel():
            c128_positions = torch.arange(
                compressed.shape[0],
                device=prompt_positions.device,
                dtype=torch.long,
            )
            c128_slots = coordinator.c128.sequence_token_slots(
                sequence_id, c128_positions
            )
            coordinator.c128.store_kv(
                layer_idx=route.c128_layer_idx,
                token_slots=c128_slots,
                kv_processed=compressed.to(torch.bfloat16).contiguous(),
            )
        outputs["c128_kv"] = compressed

    return outputs


__all__ = ["populate_v4_prefill_coordinator"]
