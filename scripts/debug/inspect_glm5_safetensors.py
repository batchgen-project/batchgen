"""Header-only walk of a GLM-5-FP8 checkpoint — dumps every tensor's name,
shape, dtype, and shard to a Markdown reference doc.

Why: the 3D-MoE port assumed minimax-style per-expert shapes
(`gate.weight = [N, K]`, `down.weight = [K, N]`) but L2 accuracy
regressed 71.9% → 3.9% after enabling it. Before patching the port,
we need ground truth for the on-disk layout. This script produces a
git-tracked markdown doc at `docs/glm5_fp8_weight_layout.md` so
future sessions don't need to re-walk 780 GB of shards.

Usage (on H20 where weights live):

    python scripts/debug/inspect_glm5_safetensors.py \\
        --model-path /data2/models/zai-org/GLM-5-FP8 \\
        --out docs/glm5_fp8_weight_layout.md

No tensor data is ever loaded — only headers via `safe_open(...).get_slice(k)`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open


def _shape_str(shape):
    return "[" + ", ".join(str(d) for d in shape) + "]"


def _dtype_str(dtype):
    # safetensors returns a torch-style string like "F32", "BF16", "F8_E4M3".
    return str(dtype)


def _group_of(name: str):
    """Classify a tensor name into one of the role groups the doc organizes by."""
    if name in ("model.embed_tokens.weight",):
        return "embed"
    if name in ("lm_head.weight",):
        return "lm_head"
    if name == "model.norm.weight":
        return "model_norm"

    m = re.match(r"model\.layers\.(\d+)\.(.+)", name)
    if not m:
        return "other"
    layer_idx = int(m.group(1))
    rest = m.group(2)

    if rest.endswith("_layernorm.weight") or rest.endswith("_layernorm.bias"):
        return "layer_norm"
    if rest == "input_layernorm.weight" or rest == "post_attention_layernorm.weight":
        return "layer_norm"
    if rest.startswith("self_attn.indexer"):
        return "indexer"
    if rest.startswith("self_attn"):
        return "attn"
    if rest.startswith("mlp.gate.") or rest.startswith("mlp.gate_bias") or rest.startswith("mlp.e_score"):
        return "moe_gate"
    if rest.startswith("mlp.experts."):
        return "moe_experts"
    if rest.startswith("mlp.shared_experts"):
        return "moe_shared"
    if rest.startswith("mlp."):
        # Dense MLP (layers 0..first_k_dense_replace-1)
        return "dense_mlp"
    return "other"


def _canonical_layer_expert(name: str):
    """Return (layer_idx, expert_idx) for routed-expert tensors, else (None, None)."""
    m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\..+", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def walk(model_path: Path):
    idx_path = model_path / "model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]  # {tensor_name: shard_file}

    shard_to_keys = defaultdict(list)
    for key, shard in weight_map.items():
        shard_to_keys[shard].append(key)

    records = {}  # name -> {shape, dtype, shard}
    for shard, keys in sorted(shard_to_keys.items()):
        shard_path = str(model_path / shard)
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                # get_slice is header-only — no tensor data read.
                sl = f.get_slice(key)
                shape = tuple(sl.get_shape())
                dtype = _dtype_str(sl.get_dtype())
                records[key] = {"shape": shape, "dtype": dtype, "shard": shard}
    return records


def _fmt_table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    out = [head, sep]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)) + " |")
    return "\n".join(out)


def render(records, model_path: Path, config: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("# GLM-5-FP8 weight layout (on-disk)")
    lines.append("")
    lines.append(f"- **Checkpoint:** `{model_path}`")
    lines.append(f"- **Total tensors:** {len(records)}")
    lines.append(f"- **Generated:** {now}")
    lines.append("- **Regenerate:** `python scripts/debug/inspect_glm5_safetensors.py "
                 "--model-path <path> --out docs/glm5_fp8_weight_layout.md`")
    lines.append("")

    # --- Config summary ---
    lines.append("## Config summary")
    lines.append("")
    keys = [
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "moe_intermediate_size", "n_routed_experts",
        "num_experts_per_tok", "n_shared_experts", "first_k_dense_replace",
        "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
        "q_lora_rank", "rope_theta", "index_topk", "index_head_dim", "index_n_heads",
        "tie_word_embeddings", "vocab_size", "torch_dtype", "quantization_config",
    ]
    rows = []
    for k in keys:
        if k in config:
            v = config[k]
            if isinstance(v, dict):
                v = json.dumps(v, separators=(",", ":"))
            rows.append((k, str(v)))
    if rows:
        lines.append(_fmt_table(rows, ["key", "value"]))
    lines.append("")

    # --- Group tensors ---
    by_group = defaultdict(list)
    for name, meta in records.items():
        by_group[_group_of(name)].append((name, meta))

    def sorted_names(items):
        # Natural sort by layer idx then expert idx then name
        def key(it):
            n = it[0]
            m = re.match(r"model\.layers\.(\d+)\.(?:mlp\.experts\.(\d+)\.)?(.+)", n)
            if m:
                layer = int(m.group(1))
                expert = int(m.group(2)) if m.group(2) else -1
                return (layer, expert, m.group(3))
            return (-1, -1, n)
        return sorted(items, key=key)

    def dump_group(title, group_key, limit=None):
        items = by_group.get(group_key, [])
        if not items:
            return
        lines.append(f"## {title} ({len(items)} tensors)")
        lines.append("")
        items = sorted_names(items)
        if limit:
            items = items[:limit]
        rows = [(n, _shape_str(m["shape"]), m["dtype"], m["shard"]) for n, m in items]
        lines.append(_fmt_table(rows, ["tensor", "shape", "dtype", "shard"]))
        lines.append("")

    dump_group("Embeddings + lm_head", "embed")
    dump_group("lm_head", "lm_head")
    dump_group("Model norm", "model_norm")
    dump_group("Per-layer norms", "layer_norm", limit=12)
    dump_group("Attention weights (first 2 layers)", "attn", limit=18)
    dump_group("DSA indexer (first 2 layers with indexer)", "indexer", limit=14)
    dump_group("Dense MLP (layers 0..first_k_dense_replace-1)", "dense_mlp")
    dump_group("MoE routing (gate + e_score bias)", "moe_gate", limit=6)
    dump_group("MoE shared experts (layer 3 sample)", "moe_shared", limit=6)

    # --- MoE routed experts: layer 3 expert 0, layer 3 expert 1, layer 77 expert 0 ---
    lines.append("## MoE routed experts (canonical samples)")
    lines.append("")
    samples = []
    for name, meta in by_group.get("moe_experts", []):
        layer, expert = _canonical_layer_expert(name)
        if (layer, expert) in ((3, 0), (3, 1), (77, 0)):
            samples.append((name, meta))
    samples = sorted_names(samples)
    rows = [(n, _shape_str(m["shape"]), m["dtype"], m["shard"]) for n, m in samples]
    lines.append(_fmt_table(rows, ["tensor", "shape", "dtype", "shard"]))
    lines.append("")

    # Count total expert tensors for sanity.
    total_experts_tensors = len(by_group.get("moe_experts", []))
    lines.append(f"_MoE routed experts total tensors: {total_experts_tensors} "
                 f"(expected ≈ (num_hidden_layers − first_k_dense_replace) × n_routed_experts × 6 tensors per expert — "
                 f"3 projections × (weight + weight_scale_inv))_")
    lines.append("")

    # --- Port headline diff ---
    lines.append("## 3D-MoE port: on-disk vs minimax stacking assumption")
    lines.append("")
    lines.append("The 3D-MoE port (`batchgen/models/glm/glm5/model.py:_init_fp8_blockwise_weights`)")
    lines.append("stacks per-expert weights into 3D tensors assuming the minimax layout:")
    lines.append("")
    lines.append("- `gate.weight` = `[N, K]` → stacked `[E, N, K]`   (N=moe_intermediate_size, K=hidden_size)")
    lines.append("- `up.weight`   = `[N, K]` → stacked `[E, N, K]`")
    lines.append("- `down.weight` = `[K, N]` → stacked `[E, K, N]`")
    lines.append("")
    lines.append("If GLM-5's on-disk shapes below don't match this convention, the 3D stacking")
    lines.append("is reinterpreting transposed weights and the FP8 blockwise GEMMs will silently")
    lines.append("compute garbage — which is exactly what the post-port L2 run (3.9% accuracy)")
    lines.append("exhibited.")
    lines.append("")
    headline_keys = [
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.layers.3.mlp.experts.0.up_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight_scale_inv",
        "model.layers.3.mlp.experts.0.down_proj.weight",
        "model.layers.3.mlp.experts.0.down_proj.weight_scale_inv",
    ]
    rows = []
    for k in headline_keys:
        if k in records:
            m = records[k]
            rows.append((k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1],
                         _shape_str(m["shape"]), m["dtype"]))
    lines.append(_fmt_table(rows, ["tensor", "shape", "dtype"]))
    lines.append("")

    # --- Full expert inventory (one row per unique tensor name within layer 3 expert 0
    #                            + any other unique name across the rest) ---
    lines.append("## Gotchas observed")
    lines.append("")
    lines.append("_(empty — append findings here as we debug the 3D-MoE port.)_")
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    model_path = args.model_path
    with open(model_path / "config.json") as f:
        config = json.load(f)

    records = walk(model_path)
    doc = render(records, model_path, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({len(records)} tensors)")


if __name__ == "__main__":
    main()
