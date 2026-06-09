#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

TARGETS = [
    "reduced",
    "mlp_input",
    "routed_before_allreduce",
    "routed_after_allreduce",
    "shared",
    "mlp_out",
]
LAYERS = [4, 5, 6]


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pick_moe_records(path: Path) -> dict[int, dict[str, Any]]:
    records = load_records(path)
    out: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("kind") != "moe_internals":
            continue
        layer = int(record["layer_idx"])
        if layer in LAYERS:
            out[layer] = record
    missing = [layer for layer in LAYERS if layer not in out]
    if missing:
        raise RuntimeError(
            f"{path}: missing moe_internals for layers {missing}"
        )
    return out


def as_tensor(record: dict[str, Any], name: str) -> torch.Tensor:
    value = record[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"record[{name!r}] is not a tensor: {type(value)}")
    return value.detach().to(torch.float32).reshape(-1)


def stats(record: dict[str, Any], name: str) -> dict[str, float]:
    cached = record.get("stats", {}).get(name)
    if cached is not None:
        return {
            "rms": float(cached["rms"]),
            "l2": float(cached["l2"]),
            "max_abs": float(cached["max_abs"]),
        }
    tensor = as_tensor(record, name)
    return {
        "rms": float(tensor.square().mean().sqrt().item()),
        "l2": float(torch.linalg.vector_norm(tensor).item()),
        "max_abs": float(tensor.abs().max().item()),
    }


def cosine(
    record_a: dict[str, Any], record_b: dict[str, Any], name: str
) -> float:
    ta = as_tensor(record_a, name)
    tb = as_tensor(record_b, name)
    return float(
        F.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0), dim=1).item()
    )


def median2(a: float, b: float) -> float:
    return float((a + b) / 2.0)


def ratio(
    records: dict[int, dict[str, Any]], name: str, prompt: str, key: str
) -> float:
    l4 = stats(records[4], name)[key]
    l5 = stats(records[5], name)[key]
    l6 = stats(records[6], name)[key]
    denom = median2(l4, l6)
    if abs(denom) < 1e-12:
        return math.inf if abs(l5) > 0 else 1.0
    return l5 / denom


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.6e}"


def extras_summary(record: dict[str, Any]) -> str:
    extras = record.get("extras", {})
    before = extras.get("routed_before_allreduce_global", {})
    after = extras.get("routed_after_allreduce_global", {})
    seg_before = extras.get("routed_before_allreduce_segments", [])
    seg_after = extras.get("routed_after_allreduce_segments", [])
    return (
        f"global_before_l2={fmt(float(before.get('l2', float('nan'))))} "
        f"global_after_l2={fmt(float(after.get('l2', float('nan'))))} "
        f"segments_before={[round(float(seg.get('l2', float('nan'))), 6) for seg in seg_before]} "
        f"segments_after={[round(float(seg.get('l2', float('nan'))), 6) for seg in seg_after]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank0", type=Path)
    parser.add_argument("rank1", type=Path)
    args = parser.parse_args()

    prompt_b = pick_moe_records(args.rank0)
    prompt_a = pick_moe_records(args.rank1)

    header = [
        "layer",
        "name",
        "A_rms",
        "A_l2",
        "A_max_abs",
        "B_rms",
        "B_l2",
        "B_max_abs",
        "cos(A,B)",
        "A_L5/med(L4,L6)_rms",
        "A_L5/med(L4,L6)_l2",
        "A_L5/med(L4,L6)_max",
        "B_L5/med(L4,L6)_rms",
        "B_L5/med(L4,L6)_l2",
        "B_L5/med(L4,L6)_max",
    ]
    print("\t".join(header))
    for layer in LAYERS:
        for name in TARGETS:
            a_stats = stats(prompt_a[layer], name)
            b_stats = stats(prompt_b[layer], name)
            row = [
                str(layer),
                name,
                fmt(a_stats["rms"]),
                fmt(a_stats["l2"]),
                fmt(a_stats["max_abs"]),
                fmt(b_stats["rms"]),
                fmt(b_stats["l2"]),
                fmt(b_stats["max_abs"]),
                fmt(cosine(prompt_a[layer], prompt_b[layer], name)),
            ]
            if layer == 5:
                row.extend(
                    [
                        fmt(ratio(prompt_a, name, "A", "rms")),
                        fmt(ratio(prompt_a, name, "A", "l2")),
                        fmt(ratio(prompt_a, name, "A", "max_abs")),
                        fmt(ratio(prompt_b, name, "B", "rms")),
                        fmt(ratio(prompt_b, name, "B", "l2")),
                        fmt(ratio(prompt_b, name, "B", "max_abs")),
                    ]
                )
            else:
                row.extend([""] * 6)
            print("\t".join(row))

    print("\n# routed global / segment diagnostics")
    for prompt_name, records in [
        ("A(rank1)", prompt_a),
        ("B(rank0)", prompt_b),
    ]:
        for layer in LAYERS:
            print(
                f"{prompt_name} layer={layer} {extras_summary(records[layer])}"
            )


if __name__ == "__main__":
    main()
