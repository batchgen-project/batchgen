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
        token_count=int(decode["swa_active_tokens_after"]),
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
    ref_model = import_reference_model_module()
    ref_model.world_size = runtime.world_size
    ref_model.rank = runtime.rank
    try:
        resolved_trace_path = trace_path
        if not resolved_trace_path.exists():
            resolved_trace_path = ranked_output_path(trace_path, runtime)
        trace = torch.load(resolved_trace_path, map_location="cpu")
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
