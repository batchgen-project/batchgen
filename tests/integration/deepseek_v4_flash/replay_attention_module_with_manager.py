from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import torch
from replay_manager_decode_attention import _manager_kv_for_layer
from trace_common import (
    assert_close,
    build_model_args,
    configure_reference_globals,
    destroy_runtime,
    import_reference_model_module,
    init_runtime,
    load_model_config,
    parse_layers,
    ranked_output_path,
)


def _load_rank_trace(trace_path: Path, runtime) -> dict:
    resolved_trace_path = trace_path
    if not resolved_trace_path.exists():
        resolved_trace_path = ranked_output_path(trace_path, runtime)
    trace = torch.load(resolved_trace_path, map_location="cpu")
    metadata = trace["metadata"]
    if int(metadata["world_size"]) != runtime.world_size:
        raise ValueError(
            "module replay must run with the same world size as export; "
            f"trace world_size={metadata['world_size']}, "
            f"runtime world_size={runtime.world_size}"
        )
    return trace


def _load_attention_state(
    attention: torch.nn.Module,
    *,
    model_dir: Path,
    runtime,
    layer_id: int,
) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required for module checkpoint loading"
        ) from exc

    shard_path = (
        model_dir / f"model{runtime.rank}-mp{runtime.world_size}.safetensors"
    )
    if not shard_path.exists():
        raise FileNotFoundError(
            f"missing DeepSeek-V4 Flash MP shard {shard_path}; expected "
            "model{rank}-mp{world_size}.safetensors under --model-dir"
        )

    prefix = f"layers.{layer_id}.attn."
    state = {}
    with safe_open(
        str(shard_path), framework="pt", device=str(runtime.device)
    ) as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = f.get_tensor(key)
    if not state:
        raise RuntimeError(
            f"no attention parameters found for prefix {prefix!r}"
        )

    missing, unexpected = attention.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"failed to load layer {layer_id} Attention state: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _copy_prefix(target: torch.Tensor, source: Optional[torch.Tensor]) -> None:
    if source is None:
        return
    source = source.to(device=target.device, dtype=target.dtype)
    slices = tuple(slice(0, int(dim)) for dim in source.shape)
    target[slices].copy_(source)


def _restore_ring_window(
    kv_cache: torch.Tensor,
    chronological_kv: torch.Tensor,
    *,
    raw_end: int,
    window_size: int,
) -> None:
    token_count = int(chronological_kv.shape[1])
    if token_count == 0:
        return
    first_token = int(raw_end) - token_count
    slots = torch.arange(
        first_token,
        int(raw_end),
        device=kv_cache.device,
        dtype=torch.long,
    ) % int(window_size)
    source = chronological_kv.to(device=kv_cache.device, dtype=kv_cache.dtype)
    kv_cache[: source.shape[0], slots].copy_(source)


def _restore_prefill_state(
    attention: torch.nn.Module,
    layer_export: dict,
    *,
    prefill_tokens: int,
) -> None:
    prefill = layer_export["prefill"]
    window_size = int(layer_export["decode"]["window_size"])
    compress_ratio = int(layer_export["decode"]["compress_ratio"])

    attention.kv_cache.zero_()
    _restore_ring_window(
        attention.kv_cache,
        prefill["swa_strict_kv"],
        raw_end=prefill_tokens,
        window_size=window_size,
    )

    if not compress_ratio:
        return

    compressed_kv = prefill["compressed_kv"]
    if compressed_kv is not None and int(compressed_kv.shape[1]) > 0:
        target = attention.kv_cache[
            : compressed_kv.shape[0],
            window_size : window_size + compressed_kv.shape[1],
        ]
        target.copy_(compressed_kv.to(device=target.device, dtype=target.dtype))

    _copy_prefix(attention.compressor.kv_state, prefill["compressor_kv_state"])
    _copy_prefix(
        attention.compressor.score_state,
        prefill["compressor_score_state"],
    )
    attention.compressor.kv_cache = None
    attention.compressor.freqs_cis = None

    if getattr(attention, "indexer", None) is None:
        return

    attention.indexer.kv_cache.zero_()
    _copy_prefix(attention.indexer.kv_cache, prefill["indexer_kv"])
    _copy_prefix(
        attention.indexer.compressor.kv_state,
        prefill["indexer_compressor_kv_state"],
    )
    _copy_prefix(
        attention.indexer.compressor.score_state,
        prefill["indexer_compressor_score_state"],
    )
    attention.indexer.compressor.kv_cache = None
    attention.indexer.compressor.freqs_cis = None
    attention.indexer.freqs_cis = None


@contextmanager
def _use_manager_sparse_attn(
    ref_model, manager_kv: torch.Tensor
) -> Iterator[None]:
    original_sparse_attn = ref_model.sparse_attn

    def manager_sparse_attn(
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        del kv
        return original_sparse_attn(
            q,
            manager_kv,
            attn_sink,
            topk_idxs,
            softmax_scale,
        )

    ref_model.sparse_attn = manager_sparse_attn
    try:
        yield
    finally:
        ref_model.sparse_attn = original_sparse_attn


def _trace_layers(
    trace: dict, requested_layers: Optional[str]
) -> dict[int, dict]:
    layers_by_id = {
        int(layer_id): layer_export
        for layer_id, layer_export in trace["layers"].items()
    }
    if requested_layers is None:
        return dict(sorted(layers_by_id.items()))
    selected = parse_layers(requested_layers)
    missing = sorted(set(selected) - set(layers_by_id))
    if missing:
        raise ValueError(
            f"requested layers {missing} are not present in the trace"
        )
    return {layer_id: layers_by_id[layer_id] for layer_id in selected}


@torch.inference_mode()
def replay_attention_modules(
    *,
    model_dir: Path,
    config_path: Path,
    trace_path: Path,
    layers: Optional[str],
    page_size_tokens: Optional[int],
    atol: float,
    rtol: float,
) -> None:
    runtime = init_runtime()
    try:
        trace = _load_rank_trace(trace_path, runtime)
        metadata = trace["metadata"]
        prefill_tokens = int(metadata["prefill_tokens"])
        sequence_id = int(metadata["sequence_id"])
        resolved_page_size_tokens = int(
            page_size_tokens or metadata.get("page_size_tokens", 64)
        )

        ref_model = import_reference_model_module()
        config = load_model_config(config_path)
        model_args = build_model_args(ref_model, config, max_batch_size=1)
        configure_reference_globals(ref_model, model_args, runtime)
        torch.set_default_dtype(torch.bfloat16)

        for layer_id, layer_export in _trace_layers(trace, layers).items():
            if "module_decode" not in layer_export:
                raise RuntimeError(
                    "trace is missing module_decode; regenerate it with "
                    "export_reference_decode_attention.py"
                )

            managers: list[object] = []
            try:
                manager_kv, managers = _manager_kv_for_layer(
                    layer_export,
                    prefill_tokens=prefill_tokens,
                    sequence_id=sequence_id,
                    page_size_tokens=resolved_page_size_tokens,
                    device=runtime.device,
                )

                with torch.device(runtime.device):
                    attention = ref_model.Attention(layer_id, model_args)
                attention.eval()
                _load_attention_state(
                    attention,
                    model_dir=model_dir,
                    runtime=runtime,
                    layer_id=layer_id,
                )
                _restore_prefill_state(
                    attention,
                    layer_export,
                    prefill_tokens=prefill_tokens,
                )

                module_decode = layer_export["module_decode"]
                decode_input = module_decode["input"].to(runtime.device)
                with _use_manager_sparse_attn(ref_model, manager_kv):
                    actual = attention(decode_input, prefill_tokens)
                expected = module_decode["output"]
                assert_close(actual, expected, atol=atol, rtol=rtol)

                ratio = int(layer_export["decode"]["compress_ratio"])
                print(
                    f"layer {layer_id}: single Attention module replay matched "
                    f"(ratio={ratio})"
                )
            finally:
                for manager in managers:
                    manager.destroy()
    finally:
        destroy_runtime()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay exported DeepSeek-V4 Flash decode Attention.forward with "
            "only the selected Attention modules loaded."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--page-size-tokens", type=int, default=None)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    args = parser.parse_args()

    config_path = args.config or args.model_dir / "config.json"
    replay_attention_modules(
        model_dir=args.model_dir,
        config_path=config_path,
        trace_path=args.trace,
        layers=args.layers,
        page_size_tokens=args.page_size_tokens,
        atol=args.atol,
        rtol=args.rtol,
    )


if __name__ == "__main__":
    main()
