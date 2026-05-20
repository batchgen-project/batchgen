from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from trace_common import (
    DEFAULT_TRACE_LAYERS,
    MultiLayerSparseAttentionHook,
    clone_to_cpu,
    destroy_runtime,
    init_runtime,
    load_prompt_tokens,
    load_reference_model,
    page_aligned_prefill_swa_kv,
    parse_layers,
    ranked_output_path,
    trace_to_dict,
    validate_representative_layers,
    window_cache_in_chronological_order,
)


def _compressed_cache_slice(
    kv_cache: torch.Tensor,
    *,
    window_size: int,
    compressed_tokens: int,
) -> torch.Tensor:
    if compressed_tokens <= 0:
        return kv_cache[:, window_size:window_size].contiguous()
    return kv_cache[
        :,
        window_size : window_size + compressed_tokens,
    ].contiguous()


def _capture_prefill_layer_state(
    model: torch.nn.Module,
    *,
    layer_id: int,
    prefill_tokens: int,
    page_size_tokens: int,
    prefill_trace: dict,
) -> dict:
    attention = model.layers[layer_id].attn
    window_size = int(attention.window_size)
    compress_ratio = int(attention.compress_ratio)
    swa_kv, swa_start_token = page_aligned_prefill_swa_kv(
        prefill_trace["kv_reference"],
        raw_end=prefill_tokens,
        window_size=window_size,
        page_size_tokens=page_size_tokens,
    )
    prefill: dict[str, object] = {
        "swa_kv": clone_to_cpu(swa_kv),
        "swa_storage_start_token": int(swa_start_token),
        "swa_active_tokens": int(swa_kv.shape[1]),
        "swa_strict_kv": clone_to_cpu(
            window_cache_in_chronological_order(
                attention.kv_cache,
                raw_end=prefill_tokens,
                window_size=window_size,
            )
        ),
    }

    if compress_ratio:
        compressed_tokens = prefill_tokens // compress_ratio
        prefill["compressed_kv"] = clone_to_cpu(
            _compressed_cache_slice(
                attention.kv_cache,
                window_size=window_size,
                compressed_tokens=compressed_tokens,
            )
        )
        prefill["compressed_tokens"] = compressed_tokens
        prefill["compressor_kv_state"] = clone_to_cpu(
            attention.compressor.kv_state
        )
        prefill["compressor_score_state"] = clone_to_cpu(
            attention.compressor.score_state
        )

        if getattr(attention, "indexer", None) is not None:
            prefill["indexer_kv"] = clone_to_cpu(
                attention.indexer.kv_cache[:, :compressed_tokens]
            )
            prefill["indexer_compressor_kv_state"] = clone_to_cpu(
                attention.indexer.compressor.kv_state
            )
            prefill["indexer_compressor_score_state"] = clone_to_cpu(
                attention.indexer.compressor.score_state
            )
    else:
        prefill["compressed_kv"] = None
        prefill["compressed_tokens"] = 0

    return prefill


def _capture_decode_layer_updates(
    model: torch.nn.Module,
    *,
    layer_id: int,
    decode_start_pos: int,
) -> dict:
    attention = model.layers[layer_id].attn
    window_size = int(attention.window_size)
    compress_ratio = int(attention.compress_ratio)
    decode_end = decode_start_pos + 1
    decode_window_slot = decode_start_pos % window_size

    updates: dict[str, object] = {
        "new_swa_kv": clone_to_cpu(
            attention.kv_cache[
                :,
                decode_window_slot : decode_window_slot + 1,
            ]
        ),
        "decode_window_slot": decode_window_slot,
        "swa_active_tokens_after": min(decode_end, window_size),
    }

    if compress_ratio:
        before = decode_start_pos // compress_ratio
        after = decode_end // compress_ratio
        updates["compressed_tokens_after"] = after
        if after > before:
            updates["new_compressed_kv"] = clone_to_cpu(
                attention.kv_cache[
                    :,
                    window_size + after - 1 : window_size + after,
                ]
            )
            if getattr(attention, "indexer", None) is not None:
                updates["new_indexer_kv"] = clone_to_cpu(
                    attention.indexer.kv_cache[:, after - 1 : after]
                )
        else:
            updates["new_compressed_kv"] = None
    else:
        updates["compressed_tokens_after"] = 0
        updates["new_compressed_kv"] = None

    return updates


@torch.inference_mode()
def export_reference_trace(
    *,
    model_dir: Path,
    config_path: Path,
    tokenizer_path: Optional[Path],
    output_path: Path,
    prompt: str,
    prefill_tokens: int,
    page_size_tokens: int,
    layer_ids: list[int],
    decode_token_id: Optional[int],
) -> Path:
    runtime = init_runtime()
    try:
        ref_model, model, config = load_reference_model(
            model_dir=model_dir,
            config_path=config_path,
            runtime=runtime,
        )
        validate_representative_layers(config, layer_ids)
        prompt_tokens = load_prompt_tokens(
            model_dir=model_dir,
            tokenizer_path=tokenizer_path,
            prompt=prompt,
            min_tokens=prefill_tokens,
        )

        device = runtime.device
        prefill_input = torch.tensor(
            [prompt_tokens],
            dtype=torch.long,
            device=device,
        )
        with MultiLayerSparseAttentionHook(
            ref_model=ref_model,
            model=model,
            layer_ids=layer_ids,
            phases=("prefill", "decode"),
        ) as hook:
            prefill_logits = model(prefill_input, 0)
            prefill_traces_by_layer = {
                trace.layer_id: trace_to_dict(trace)
                for trace in hook.traces
                if trace.phase == "prefill"
            }
            missing_prefill = sorted(
                set(layer_ids) - set(prefill_traces_by_layer)
            )
            if missing_prefill:
                raise RuntimeError(
                    f"missing prefill traces for layers {missing_prefill}"
                )

            layer_exports = {
                layer_id: {
                    "layer_id": layer_id,
                    "compress_ratio": int(config["compress_ratios"][layer_id]),
                    "prefill_trace": prefill_traces_by_layer[layer_id],
                    "prefill": _capture_prefill_layer_state(
                        model,
                        layer_id=layer_id,
                        prefill_tokens=prefill_tokens,
                        page_size_tokens=page_size_tokens,
                        prefill_trace=prefill_traces_by_layer[layer_id],
                    ),
                }
                for layer_id in layer_ids
            }

            if decode_token_id is None:
                decode_token = prefill_logits.argmax(dim=-1).view(1, 1)
            else:
                decode_token = torch.tensor(
                    [[int(decode_token_id)]],
                    dtype=torch.long,
                    device=device,
                )

            decode_logits = model(decode_token, prefill_tokens)

        traces_by_layer = {
            trace.layer_id: trace
            for trace in hook.traces
            if trace.phase == "decode"
        }
        missing = sorted(set(layer_ids) - set(traces_by_layer))
        if missing:
            raise RuntimeError(f"missing decode traces for layers {missing}")

        for layer_id in layer_ids:
            module_key = (layer_id, "decode")
            if module_key not in hook.attention_forward_traces:
                raise RuntimeError(
                    f"missing Attention.forward decode trace for layer {layer_id}"
                )
            layer_exports[layer_id]["module_decode"] = (
                hook.attention_forward_traces[module_key]
            )
            layer_exports[layer_id]["decode"] = trace_to_dict(
                traces_by_layer[layer_id]
            )
            layer_exports[layer_id]["decode"].update(
                _capture_decode_layer_updates(
                    model,
                    layer_id=layer_id,
                    decode_start_pos=prefill_tokens,
                )
            )
            if int(config["compress_ratios"][layer_id]) == 4:
                if layer_id not in hook.indexer_traces:
                    raise RuntimeError(
                        f"missing C4 indexer trace for layer {layer_id}"
                    )
                layer_exports[layer_id]["indexer_decode"] = hook.indexer_traces[
                    layer_id
                ]

        output = {
            "metadata": {
                "rank": runtime.rank,
                "world_size": runtime.world_size,
                "local_rank": runtime.local_rank,
                "model_dir": str(model_dir),
                "config_path": str(config_path),
                "prefill_tokens": prefill_tokens,
                "page_size_tokens": page_size_tokens,
                "decode_start_pos": prefill_tokens,
                "decode_end_pos": prefill_tokens + 1,
                "layers": layer_ids,
                "sequence_id": 0,
            },
            "prompt_tokens": prompt_tokens,
            "decode_token": clone_to_cpu(decode_token),
            "prefill_logits": clone_to_cpu(prefill_logits),
            "decode_logits": clone_to_cpu(decode_logits),
            "layers": layer_exports,
        }

        ranked_path = ranked_output_path(output_path, runtime)
        ranked_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, ranked_path)
        return ranked_path
    finally:
        destroy_runtime()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-V4 Flash reference prefill/decode and export attention traces."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain why cache layout matters for sparse decode attention.",
    )
    parser.add_argument("--prefill-tokens", type=int, default=255)
    parser.add_argument("--page-size-tokens", type=int, default=64)
    parser.add_argument(
        "--layers",
        type=str,
        default=",".join(str(layer) for layer in DEFAULT_TRACE_LAYERS),
    )
    parser.add_argument("--decode-token-id", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config or args.model_dir / "config.json"
    path = export_reference_trace(
        model_dir=args.model_dir,
        config_path=config_path,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output,
        prompt=args.prompt,
        prefill_tokens=args.prefill_tokens,
        page_size_tokens=args.page_size_tokens,
        layer_ids=parse_layers(args.layers),
        decode_token_id=args.decode_token_id,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
