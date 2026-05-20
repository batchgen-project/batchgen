from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
from trace_common import (
    assert_close,
    destroy_runtime,
    import_reference_model_module,
    init_runtime,
    ranked_output_path,
    replay_sparse_attention,
)

from batchgen.kv_cache.compressed_ratio_gpu_paged_kv_manager import (
    CompressedRatioGPUPagedKVCacheManager,
)
from batchgen.kv_cache.compressed_state_gpu_manager import (
    CompressedStateGPUConfig,
    CompressedStateGPUManager,
)
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig
from batchgen.kv_cache.swa_gpu_paged_kv_manager import (
    SWAGPUPagedKVCacheManager,
)


def _ceil_div(value: int, divisor: int) -> int:
    return math.ceil(int(value) / int(divisor))


def _manager_config(
    *,
    num_layers: int,
    num_pages: int,
    page_size_tokens: int,
    head_dim: int,
    dtype: torch.dtype,
    cuda_graph_max_slots: int = 1,
) -> GPUPagedKVConfig:
    return GPUPagedKVConfig(
        num_layers=num_layers,
        num_pages=num_pages,
        page_size_tokens=page_size_tokens,
        num_k_heads=1,
        k_head_dim=head_dim,
        num_v_heads=0,
        v_head_dim=0,
        kv_dtype=dtype,
        cuda_graph_max_pages_per_sequence=num_pages,
        cuda_graph_max_slots=cuda_graph_max_slots,
    )


def _copy_dense_tokens_to_manager(
    manager,
    *,
    layer_id: int,
    sequence_id: int,
    dense_tokens: torch.Tensor,
) -> None:
    if dense_tokens.ndim != 3 or dense_tokens.shape[0] != 1:
        raise ValueError(
            "dense_tokens must have shape [1, tokens, head_dim], got "
            f"{tuple(dense_tokens.shape)}"
        )
    token_count = int(dense_tokens.shape[1])
    if token_count == 0:
        return

    k_cache, _ = manager.get_kv_tensors()
    pages = manager._sequences[int(sequence_id)].pages
    page_size = int(manager.config.page_size_tokens)
    source = dense_tokens.to(
        device=manager.device, dtype=manager.config.kv_dtype
    )
    layer_id = manager.resolve_physical_layer(int(layer_id))

    copied = 0
    for page_offset, page_id_tensor in enumerate(pages):
        if copied >= token_count:
            break
        page_id = int(page_id_tensor.item())
        chunk = min(page_size, token_count - copied)
        k_cache[
            layer_id,
            page_id,
            :chunk,
            0,
            :,
        ].copy_(source[0, copied : copied + chunk])
        copied += chunk

    if copied != token_count:
        raise RuntimeError(
            f"copied {copied} tokens, expected {token_count}; manager did not "
            "have enough allocated pages"
        )


def _gather_dense_tokens_from_manager(
    manager,
    *,
    layer_id: int,
    sequence_id: int,
    token_count: int,
) -> torch.Tensor:
    k_cache, _ = manager.get_kv_tensors()
    pages = manager._sequences[int(sequence_id)].pages
    page_size = int(manager.config.page_size_tokens)
    layer_id = manager.resolve_physical_layer(int(layer_id))

    chunks: list[torch.Tensor] = []
    gathered = 0
    for page_id_tensor in pages:
        if gathered >= token_count:
            break
        page_id = int(page_id_tensor.item())
        chunk = min(page_size, token_count - gathered)
        chunks.append(k_cache[layer_id, page_id, :chunk, 0, :])
        gathered += chunk

    if gathered != token_count:
        raise RuntimeError(
            f"gathered {gathered} tokens, expected {token_count}; manager did "
            "not have enough allocated pages"
        )
    if not chunks:
        head_dim = int(manager.config.k_head_dim)
        return torch.empty(
            (1, 0, head_dim),
            dtype=manager.config.kv_dtype,
            device=manager.device,
        )
    return torch.cat(chunks, dim=0).unsqueeze(0)


def _decode_update(
    manager,
    *,
    layer_id: int,
    sequence_id: int,
    raw_position: int,
    token: Optional[torch.Tensor],
) -> None:
    manager.prepare_decode_step(
        [int(sequence_id)],
        torch.tensor(
            [int(raw_position)], dtype=torch.int32, device=manager.device
        ),
    )
    if token is None:
        return
    token = token.to(device=manager.device, dtype=manager.config.kv_dtype)
    token = token.view(1, 1, 1, int(manager.config.k_head_dim))
    manager.update_layer_decode_new_token(
        token,
        None,
        None,
        int(layer_id),
        assume_prepared=True,
    )


def _build_swa_manager(
    *,
    layer_id: int,
    prefill_tokens: int,
    window_size: int,
    page_size_tokens: int,
    dtype: torch.dtype,
    head_dim: int,
    sequence_id: int,
    prefill_swa_kv: torch.Tensor,
    new_swa_kv: torch.Tensor,
    device: torch.device,
) -> SWAGPUPagedKVCacheManager:
    active_after = min(prefill_tokens + 1, window_size)
    num_pages = max(4, _ceil_div(active_after, page_size_tokens) + 2)
    manager = SWAGPUPagedKVCacheManager(
        config=_manager_config(
            num_layers=layer_id + 1,
            num_pages=num_pages,
            page_size_tokens=page_size_tokens,
            head_dim=head_dim,
            dtype=dtype,
        ),
        device=device,
        window_size_tokens=window_size,
    )
    manager.initialize()
    manager.allocate_pages_for_sequences([sequence_id], [prefill_tokens])
    manager.rebuild_page_table([sequence_id])
    _copy_dense_tokens_to_manager(
        manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        dense_tokens=prefill_swa_kv,
    )
    _decode_update(
        manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        raw_position=prefill_tokens,
        token=new_swa_kv,
    )
    return manager


def _build_compressed_manager(
    *,
    layer_id: int,
    prefill_tokens: int,
    compressed_tokens_after: int,
    compression_ratio: int,
    page_size_tokens: int,
    dtype: torch.dtype,
    head_dim: int,
    sequence_id: int,
    prefill_compressed_kv: torch.Tensor,
    new_compressed_kv: Optional[torch.Tensor],
    device: torch.device,
) -> CompressedRatioGPUPagedKVCacheManager:
    num_pages = max(
        4, _ceil_div(max(1, compressed_tokens_after), page_size_tokens) + 2
    )
    manager = CompressedRatioGPUPagedKVCacheManager(
        config=_manager_config(
            num_layers=layer_id + 1,
            num_pages=num_pages,
            page_size_tokens=page_size_tokens,
            head_dim=head_dim,
            dtype=dtype,
        ),
        device=device,
        compression_ratio=compression_ratio,
    )
    manager.initialize()
    manager.allocate_pages_for_sequences([sequence_id], [prefill_tokens])
    manager.rebuild_page_table([sequence_id])
    _copy_dense_tokens_to_manager(
        manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        dense_tokens=prefill_compressed_kv,
    )
    _decode_update(
        manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        raw_position=prefill_tokens,
        token=new_compressed_kv,
    )
    return manager


def _build_indexer_manager(
    *,
    layer_id: int,
    prefill_tokens: int,
    compressed_tokens_after: int,
    compression_ratio: int,
    page_size_tokens: int,
    dtype: torch.dtype,
    head_dim: int,
    sequence_id: int,
    prefill_indexer_kv: torch.Tensor,
    new_indexer_kv: Optional[torch.Tensor],
    device: torch.device,
) -> CompressedRatioGPUPagedKVCacheManager:
    return _build_compressed_manager(
        layer_id=layer_id,
        prefill_tokens=prefill_tokens,
        compressed_tokens_after=compressed_tokens_after,
        compression_ratio=compression_ratio,
        page_size_tokens=page_size_tokens,
        dtype=dtype,
        head_dim=head_dim,
        sequence_id=sequence_id,
        prefill_compressed_kv=prefill_indexer_kv,
        new_compressed_kv=new_indexer_kv,
        device=device,
    )


def _state_rows(state: torch.Tensor) -> torch.Tensor:
    if state.ndim < 2 or int(state.shape[0]) != 1:
        raise ValueError(
            "compressed state must have shape [1, ring_size, ...], got "
            f"{tuple(state.shape)}"
        )
    return state.contiguous().view(1, int(state.shape[1]), -1)


def _build_compressed_state_manager(
    *,
    layer_id: int,
    sequence_id: int,
    prefill_state: torch.Tensor,
    compression_ratio: int,
    overlap: bool,
    device: torch.device,
) -> CompressedStateGPUManager:
    rows = _state_rows(prefill_state)
    manager = CompressedStateGPUManager(
        config=CompressedStateGPUConfig(
            num_layers=layer_id + 1,
            num_state_items=4,
            ring_size=int(rows.shape[1]),
            state_dim=int(rows.shape[2]),
            state_dtype=prefill_state.dtype,
            cuda_graph_max_slots=max(8, int(rows.shape[1])),
        ),
        device=device,
        ratio=compression_ratio,
        overlap=overlap,
    )
    manager.initialize()
    allocations = manager.allocate_state_items_for_sequences([sequence_id])
    state_item_id = allocations[int(sequence_id)]
    manager.state_cache[
        manager.resolve_physical_layer(layer_id),
        state_item_id,
    ].copy_(rows[0].to(device=device, dtype=prefill_state.dtype))
    return manager


def _update_manager_state_to_match_reference(
    manager: CompressedStateGPUManager,
    *,
    layer_id: int,
    sequence_id: int,
    before_state: torch.Tensor,
    after_state: torch.Tensor,
) -> None:
    before_rows = _state_rows(before_state)
    after_rows = _state_rows(after_state)
    if before_rows.shape != after_rows.shape:
        raise ValueError(
            "before/after compressed state shapes differ: "
            f"{tuple(before_rows.shape)} vs {tuple(after_rows.shape)}"
        )

    changed_rows = (
        before_rows[0].to(dtype=torch.float32)
        != after_rows[0].to(dtype=torch.float32)
    ).any(dim=1)
    row_ids = torch.nonzero(changed_rows, as_tuple=False).flatten()
    if int(row_ids.numel()) == 0:
        return

    slots = torch.tensor(
        [
            manager.resolve_state_slot(sequence_id, int(row_id.item()))
            for row_id in row_ids
        ],
        dtype=torch.int32,
        device=manager.device,
    )
    manager.update_layer_state_slots(
        after_rows[0, row_ids].to(
            device=manager.device,
            dtype=manager.config.state_dtype,
        ),
        slots,
        layer_id,
    )


def _assert_manager_state(
    manager: CompressedStateGPUManager,
    *,
    layer_id: int,
    sequence_id: int,
    expected_state: torch.Tensor,
) -> None:
    expected_rows = _state_rows(expected_state)
    state_item_id = manager._sequence_state_items[int(sequence_id)]
    actual = manager.get_layer_state_buffer(layer_id)[state_item_id]
    torch.testing.assert_close(
        actual.float().cpu(),
        expected_rows[0].float().cpu(),
        atol=0,
        rtol=0,
        equal_nan=True,
    )


def _verify_state_component_replay(
    *,
    layer_id: int,
    sequence_id: int,
    compression_ratio: int,
    overlap: bool,
    prefill_state: torch.Tensor,
    after_states: list[torch.Tensor],
    device: torch.device,
) -> None:
    manager = _build_compressed_state_manager(
        layer_id=layer_id,
        sequence_id=sequence_id,
        prefill_state=prefill_state,
        compression_ratio=compression_ratio,
        overlap=overlap,
        device=device,
    )
    try:
        current = prefill_state
        _assert_manager_state(
            manager,
            layer_id=layer_id,
            sequence_id=sequence_id,
            expected_state=current,
        )
        for after_state in after_states:
            _update_manager_state_to_match_reference(
                manager,
                layer_id=layer_id,
                sequence_id=sequence_id,
                before_state=current,
                after_state=after_state,
            )
            _assert_manager_state(
                manager,
                layer_id=layer_id,
                sequence_id=sequence_id,
                expected_state=after_state,
            )
            current = after_state
    finally:
        manager.destroy()


def _verify_compressed_state_replay(
    layer_export: dict,
    *,
    sequence_id: int,
    device: torch.device,
) -> None:
    prefill = layer_export["prefill"]
    decode_steps = _layer_decode_steps(layer_export)
    if not int(layer_export["compress_ratio"]):
        return

    layer_id = int(layer_export["layer_id"])
    compression_ratio = int(layer_export["compress_ratio"])
    overlap = compression_ratio == 4
    components = [
        (
            "compressor_kv_state",
            "compressor_kv_state_after",
        ),
        (
            "compressor_score_state",
            "compressor_score_state_after",
        ),
    ]
    if "indexer_compressor_kv_state" in prefill:
        components.extend(
            [
                (
                    "indexer_compressor_kv_state",
                    "indexer_compressor_kv_state_after",
                ),
                (
                    "indexer_compressor_score_state",
                    "indexer_compressor_score_state_after",
                ),
            ]
        )

    for prefill_key, after_key in components:
        after_states = [decode[after_key] for decode in decode_steps]
        _verify_state_component_replay(
            layer_id=layer_id,
            sequence_id=sequence_id,
            compression_ratio=compression_ratio,
            overlap=overlap,
            prefill_state=prefill[prefill_key],
            after_states=after_states,
            device=device,
        )
    print(
        f"layer {layer_id}: compressed state replay matched "
        f"(ratio={compression_ratio})"
    )


def _verify_indexer_replay(
    layer_export: dict,
    *,
    prefill_tokens: int,
    sequence_id: int,
    page_size_tokens: int,
    device: torch.device,
    managers: list[object],
) -> Optional[torch.Tensor]:
    indexer_decode = layer_export.get("indexer_decode")
    if indexer_decode is None:
        return None

    import torch.distributed as dist

    layer_id = int(layer_export["layer_id"])
    prefill = layer_export["prefill"]
    decode = layer_export["decode"]
    compression_ratio = int(indexer_decode["compress_ratio"])
    compressed_tokens_after = int(indexer_decode["compressed_tokens_after"])
    prefill_indexer_kv = prefill["indexer_kv"]
    new_indexer_kv = decode.get("new_indexer_kv")
    indexer_manager = _build_indexer_manager(
        layer_id=layer_id,
        prefill_tokens=prefill_tokens,
        compressed_tokens_after=compressed_tokens_after,
        compression_ratio=compression_ratio,
        page_size_tokens=page_size_tokens,
        dtype=prefill_indexer_kv.dtype,
        head_dim=int(prefill_indexer_kv.shape[-1]),
        sequence_id=sequence_id,
        prefill_indexer_kv=prefill_indexer_kv,
        new_indexer_kv=new_indexer_kv,
        device=device,
    )
    managers.append(indexer_manager)
    indexer_kv = _gather_dense_tokens_from_manager(
        indexer_manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        token_count=compressed_tokens_after,
    )

    q = indexer_decode["q"].to(device)
    weights = indexer_decode["weights"].to(device)
    index_score = torch.einsum("bshd,btd->bsht", q, indexer_kv)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(index_score)
    topk_count = min(int(indexer_decode["index_topk"]), compressed_tokens_after)
    topk_idxs = index_score.topk(topk_count, dim=-1)[1] + int(
        indexer_decode["offset"]
    )
    expected = indexer_decode["topk_idxs"].to(device)
    if not torch.equal(topk_idxs.cpu(), expected.cpu()):
        raise AssertionError(
            f"layer {layer_id}: manager indexer topk mismatch; "
            f"actual shape={tuple(topk_idxs.shape)}, expected shape={tuple(expected.shape)}"
        )
    window_topk = decode["topk_idxs"][..., : int(decode["window_size"])].to(
        device
    )
    return torch.cat([window_topk, topk_idxs], dim=-1)


def _layer_decode_steps(layer_export: dict) -> list[dict]:
    return list(layer_export.get("decode_steps") or [layer_export["decode"]])


def _layer_indexer_decode_steps(layer_export: dict) -> list[dict]:
    if "indexer_decode_steps" in layer_export:
        return list(layer_export["indexer_decode_steps"])
    if "indexer_decode" in layer_export:
        return [layer_export["indexer_decode"]]
    return []


def _swa_storage_tokens_after(decode: dict) -> int:
    return int(
        decode.get(
            "swa_storage_tokens_after",
            decode["swa_active_tokens_after"],
        )
    )


def _map_reference_topk_to_manager_layout(
    decode: dict,
    *,
    page_size_tokens: int,
) -> torch.Tensor:
    topk = decode["topk_idxs"].clone()
    window_size = int(decode["window_size"])
    raw_end = int(decode["start_pos"]) + int(decode["seqlen"])
    strict_start = max(0, raw_end - window_size)
    storage_start = int(
        decode.get(
            "swa_storage_start_token",
            (strict_start // int(page_size_tokens)) * int(page_size_tokens),
        )
    )
    swa_storage_tokens = _swa_storage_tokens_after(decode)

    mapped = topk.clone()
    valid_window = (topk >= 0) & (topk < window_size)
    if valid_window.any():
        slots = topk[valid_window].to(dtype=torch.long)
        raw_tokens = strict_start + (
            (slots - (strict_start % window_size)) % window_size
        )
        raw_tokens = torch.where(
            raw_tokens >= raw_end,
            raw_tokens - window_size,
            raw_tokens,
        )
        mapped[valid_window] = (raw_tokens - storage_start).to(
            dtype=mapped.dtype
        )

    compressed = topk >= window_size
    if compressed.any():
        mapped[compressed] = topk[compressed] - window_size + swa_storage_tokens
    return mapped


def _indexer_topk_from_manager(
    *,
    layer_export: dict,
    decode: dict,
    indexer_decode: dict,
    indexer_manager,
    sequence_id: int,
    page_size_tokens: int,
) -> torch.Tensor:
    import torch.distributed as dist

    layer_id = int(layer_export["layer_id"])
    compressed_tokens_after = int(indexer_decode["compressed_tokens_after"])
    indexer_kv = _gather_dense_tokens_from_manager(
        indexer_manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        token_count=compressed_tokens_after,
    )

    q = indexer_decode["q"].to(indexer_manager.device)
    weights = indexer_decode["weights"].to(indexer_manager.device)
    index_score = torch.einsum("bshd,btd->bsht", q, indexer_kv)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(index_score)
    topk_count = min(int(indexer_decode["index_topk"]), compressed_tokens_after)
    topk_idxs_zero_based = index_score.topk(topk_count, dim=-1)[1]
    topk_idxs = topk_idxs_zero_based + int(indexer_decode["offset"])
    expected = indexer_decode["topk_idxs"].to(indexer_manager.device)
    if not torch.equal(topk_idxs.cpu(), expected.cpu()):
        raise AssertionError(
            f"layer {layer_id}: manager indexer topk mismatch at "
            f"start_pos={indexer_decode['start_pos']}; "
            f"actual shape={tuple(topk_idxs.shape)}, expected shape={tuple(expected.shape)}"
        )
    mapped_reference_topk = _map_reference_topk_to_manager_layout(
        decode,
        page_size_tokens=page_size_tokens,
    ).to(indexer_manager.device)
    window_topk = mapped_reference_topk[..., : int(decode["window_size"])]
    manager_compressed_topk = topk_idxs_zero_based + int(
        _swa_storage_tokens_after(decode)
    )
    return torch.cat([window_topk, manager_compressed_topk], dim=-1)


def _manager_kv_after_decode_step(
    layer_export: dict,
    *,
    step_index: int,
    prefill_tokens: int,
    sequence_id: int,
    page_size_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[object], Optional[torch.Tensor]]:
    decode_steps = _layer_decode_steps(layer_export)
    if step_index >= len(decode_steps):
        raise ValueError(
            f"requested decode step {step_index}, but trace only has "
            f"{len(decode_steps)} step(s)"
        )

    layer_id = int(layer_export["layer_id"])
    first_decode = decode_steps[0]
    target_decode = decode_steps[step_index]
    prefill = layer_export["prefill"]
    window_size = int(first_decode["window_size"])
    compress_ratio = int(first_decode["compress_ratio"])
    dtype = first_decode["kv_reference"].dtype
    head_dim = int(prefill["swa_kv"].shape[-1])
    managers: list[object] = []

    swa_manager = _build_swa_manager(
        layer_id=layer_id,
        prefill_tokens=prefill_tokens,
        window_size=window_size,
        page_size_tokens=page_size_tokens,
        dtype=dtype,
        head_dim=head_dim,
        sequence_id=sequence_id,
        prefill_swa_kv=prefill["swa_kv"],
        new_swa_kv=first_decode["new_swa_kv"],
        device=device,
    )
    managers.append(swa_manager)
    for decode in decode_steps[1 : step_index + 1]:
        _decode_update(
            swa_manager,
            layer_id=layer_id,
            sequence_id=sequence_id,
            raw_position=int(decode["start_pos"]),
            token=decode["new_swa_kv"],
        )
    swa_kv = _gather_dense_tokens_from_manager(
        swa_manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        token_count=_swa_storage_tokens_after(target_decode),
    )
    manager_topk = _map_reference_topk_to_manager_layout(
        target_decode,
        page_size_tokens=page_size_tokens,
    )

    if not compress_ratio:
        return swa_kv, managers, manager_topk

    compressed_manager = _build_compressed_manager(
        layer_id=layer_id,
        prefill_tokens=prefill_tokens,
        compressed_tokens_after=int(first_decode["compressed_tokens_after"]),
        compression_ratio=compress_ratio,
        page_size_tokens=page_size_tokens,
        dtype=dtype,
        head_dim=head_dim,
        sequence_id=sequence_id,
        prefill_compressed_kv=prefill["compressed_kv"],
        new_compressed_kv=first_decode["new_compressed_kv"],
        device=device,
    )
    managers.append(compressed_manager)
    for decode in decode_steps[1 : step_index + 1]:
        _decode_update(
            compressed_manager,
            layer_id=layer_id,
            sequence_id=sequence_id,
            raw_position=int(decode["start_pos"]),
            token=decode.get("new_compressed_kv"),
        )
    compressed_kv = _gather_dense_tokens_from_manager(
        compressed_manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        token_count=int(target_decode["compressed_tokens_after"]),
    )

    indexer_decode_steps = _layer_indexer_decode_steps(layer_export)
    if indexer_decode_steps:
        if step_index >= len(indexer_decode_steps):
            raise ValueError(
                f"requested indexer decode step {step_index}, but trace only "
                f"has {len(indexer_decode_steps)} step(s)"
            )
        first_indexer_decode = indexer_decode_steps[0]
        target_indexer_decode = indexer_decode_steps[step_index]
        indexer_manager = _build_indexer_manager(
            layer_id=layer_id,
            prefill_tokens=prefill_tokens,
            compressed_tokens_after=int(
                first_indexer_decode["compressed_tokens_after"]
            ),
            compression_ratio=int(first_indexer_decode["compress_ratio"]),
            page_size_tokens=page_size_tokens,
            dtype=prefill["indexer_kv"].dtype,
            head_dim=int(prefill["indexer_kv"].shape[-1]),
            sequence_id=sequence_id,
            prefill_indexer_kv=prefill["indexer_kv"],
            new_indexer_kv=first_decode.get("new_indexer_kv"),
            device=device,
        )
        managers.append(indexer_manager)
        for decode in decode_steps[1 : step_index + 1]:
            _decode_update(
                indexer_manager,
                layer_id=layer_id,
                sequence_id=sequence_id,
                raw_position=int(decode["start_pos"]),
                token=decode.get("new_indexer_kv"),
            )
        manager_topk = _indexer_topk_from_manager(
            layer_export=layer_export,
            decode=target_decode,
            indexer_decode=target_indexer_decode,
            indexer_manager=indexer_manager,
            sequence_id=sequence_id,
            page_size_tokens=page_size_tokens,
        )

    return torch.cat([swa_kv, compressed_kv], dim=1), managers, manager_topk


def _manager_kv_for_layer(
    layer_export: dict,
    *,
    prefill_tokens: int,
    sequence_id: int,
    page_size_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[object]]:
    layer_id = int(layer_export["layer_id"])
    decode = layer_export["decode"]
    prefill = layer_export["prefill"]
    window_size = int(decode["window_size"])
    compress_ratio = int(decode["compress_ratio"])
    dtype = decode["kv_reference"].dtype
    head_dim = int(prefill["swa_kv"].shape[-1])

    if (prefill_tokens + 1) % window_size != 0:
        raise ValueError(
            "manager replay currently expects the first decode step to end on "
            "a window boundary so the reference ring window and manager-local "
            f"window have the same order; got decode_end={prefill_tokens + 1}, "
            f"window_size={window_size}"
        )

    managers: list[object] = []
    swa_manager = _build_swa_manager(
        layer_id=layer_id,
        prefill_tokens=prefill_tokens,
        window_size=window_size,
        page_size_tokens=page_size_tokens,
        dtype=dtype,
        head_dim=head_dim,
        sequence_id=sequence_id,
        prefill_swa_kv=prefill["swa_kv"],
        new_swa_kv=decode["new_swa_kv"],
        device=device,
    )
    managers.append(swa_manager)
    swa_kv = _gather_dense_tokens_from_manager(
        swa_manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        token_count=_swa_storage_tokens_after(decode),
    )

    if not compress_ratio:
        return swa_kv, managers

    compressed_manager = _build_compressed_manager(
        layer_id=layer_id,
        prefill_tokens=prefill_tokens,
        compressed_tokens_after=int(decode["compressed_tokens_after"]),
        compression_ratio=compress_ratio,
        page_size_tokens=page_size_tokens,
        dtype=dtype,
        head_dim=head_dim,
        sequence_id=sequence_id,
        prefill_compressed_kv=prefill["compressed_kv"],
        new_compressed_kv=decode["new_compressed_kv"],
        device=device,
    )
    managers.append(compressed_manager)
    compressed_kv = _gather_dense_tokens_from_manager(
        compressed_manager,
        layer_id=layer_id,
        sequence_id=sequence_id,
        token_count=int(decode["compressed_tokens_after"]),
    )
    return torch.cat([swa_kv, compressed_kv], dim=1), managers


def replay_manager_trace(
    *,
    trace_path: Path,
    page_size_tokens: Optional[int],
    atol: float,
    rtol: float,
) -> None:
    runtime = init_runtime()
    if runtime.rank == 0:
        print("manager replay: runtime initialized", flush=True)
    ref_model = import_reference_model_module()
    if runtime.rank == 0:
        print("manager replay: reference model module imported", flush=True)
    ref_model.world_size = runtime.world_size
    ref_model.rank = runtime.rank
    try:
        resolved_trace_path = trace_path
        if not resolved_trace_path.exists():
            resolved_trace_path = ranked_output_path(trace_path, runtime)
        trace = torch.load(resolved_trace_path, map_location="cpu")
        if runtime.rank == 0:
            print(
                f"manager replay: loaded trace {resolved_trace_path}",
                flush=True,
            )
        metadata = trace["metadata"]
        if int(metadata["world_size"]) != runtime.world_size:
            raise ValueError(
                "manager replay must run with the same world size as export; "
                f"trace world_size={metadata['world_size']}, runtime world_size={runtime.world_size}"
            )
        prefill_tokens = int(metadata["prefill_tokens"])
        sequence_id = int(metadata["sequence_id"])
        resolved_page_size_tokens = int(
            page_size_tokens or metadata.get("page_size_tokens", 64)
        )

        for layer_id, layer_export in sorted(trace["layers"].items()):
            managers: list[object] = []
            try:
                manager_kv, managers = _manager_kv_for_layer(
                    layer_export,
                    prefill_tokens=prefill_tokens,
                    sequence_id=sequence_id,
                    page_size_tokens=resolved_page_size_tokens,
                    device=runtime.device,
                )
                manager_topk = _verify_indexer_replay(
                    layer_export,
                    prefill_tokens=prefill_tokens,
                    sequence_id=sequence_id,
                    page_size_tokens=resolved_page_size_tokens,
                    device=runtime.device,
                    managers=managers,
                )
                actual = replay_sparse_attention(
                    ref_model,
                    layer_export,
                    manager_kv,
                    topk_idxs=manager_topk,
                )
                expected = layer_export["decode"]["output"]
                assert_close(actual, expected, atol=atol, rtol=rtol)
                ratio = int(layer_export["decode"]["compress_ratio"])
                print(
                    f"layer {layer_id}: manager attention replay matched (ratio={ratio})"
                )
                _verify_compressed_state_replay(
                    layer_export,
                    sequence_id=sequence_id,
                    device=runtime.device,
                )
                decode_steps = _layer_decode_steps(layer_export)
                if len(decode_steps) >= 2:
                    next_managers: list[object] = []
                    try:
                        next_kv, next_managers, next_topk = (
                            _manager_kv_after_decode_step(
                                layer_export,
                                step_index=1,
                                prefill_tokens=prefill_tokens,
                                sequence_id=sequence_id,
                                page_size_tokens=resolved_page_size_tokens,
                                device=runtime.device,
                            )
                        )
                        next_actual = replay_sparse_attention(
                            ref_model,
                            layer_export,
                            next_kv,
                            topk_idxs=next_topk,
                            decode=decode_steps[1],
                        )
                        assert_close(
                            next_actual,
                            decode_steps[1]["output"],
                            atol=atol,
                            rtol=rtol,
                        )
                        print(
                            f"layer {layer_id}: next-decode replay matched "
                            f"after manager KV writes (ratio={ratio})"
                        )
                    finally:
                        for manager in next_managers:
                            manager.destroy()
            finally:
                for manager in managers:
                    manager.destroy()
    finally:
        destroy_runtime()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay exported DeepSeek-V4 Flash decode attention using BatchGen managers."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--page-size-tokens", type=int, default=None)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("manager replay requires CUDA")
    replay_manager_trace(
        trace_path=args.trace,
        page_size_tokens=args.page_size_tokens,
        atol=args.atol,
        rtol=args.rtol,
    )


if __name__ == "__main__":
    main()
