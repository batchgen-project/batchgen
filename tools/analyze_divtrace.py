#!/usr/bin/env python3

import glob
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

BOUNDARIES = ["h_in", "attn_out", "h_after_attn", "h_after_ffn"]
PROMPT_A_SEQLEN = 6
PROMPT_B_SEQLEN = 16


def _artifact_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_records(base_dir: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(base_dir, "divtrace_rank*.pt")))
    if not paths:
        raise FileNotFoundError(f"no divtrace_rank*.pt under {base_dir}")
    records = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, list):
            raise TypeError(
                f"expected list payload in {path}, got {type(payload)!r}"
            )
        records.extend(payload)
    return records


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.to(torch.float32).reshape(-1)
    return {
        "norm": torch.linalg.vector_norm(flat).item(),
        "abs_mean": flat.abs().mean().item(),
        "rms": flat.square().mean().sqrt().item(),
        "max_abs": flat.abs().max().item(),
    }


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a_flat = a.to(torch.float32).reshape(-1)
    b_flat = b.to(torch.float32).reshape(-1)
    diff = a_flat - b_flat
    return {
        "rel_l2": (
            torch.linalg.vector_norm(diff)
            / (torch.linalg.vector_norm(a_flat) + 1e-6)
        ).item(),
        "cosine": F.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1
        ).item(),
    }


def _index_boundary_records(
    records: list[dict],
) -> dict[tuple[int, int, str], dict]:
    grouped: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("kind") != "boundary":
            continue
        name = record.get("name")
        cache_seqlen = record.get("cache_seqlen")
        layer_idx = record.get("layer_idx")
        if cache_seqlen is None or layer_idx is None or name is None:
            continue
        grouped[(int(cache_seqlen), int(layer_idx), str(name))].append(record)
    indexed: dict[tuple[int, int, str], dict] = {}
    for key, items in grouped.items():
        items = sorted(
            items,
            key=lambda item: (
                int(item.get("rank", -1)),
                str(item.get("seq_id")),
            ),
        )
        indexed[key] = items[0]
    return indexed


def _index_final_topk(records: list[dict]) -> dict[int, dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("kind") != "final_topk":
            continue
        cache_seqlen = record.get("cache_seqlen")
        if cache_seqlen is None:
            continue
        grouped[int(cache_seqlen)].append(record)
    indexed: dict[int, dict] = {}
    for key, items in grouped.items():
        items = sorted(items, key=lambda item: int(item.get("rank", -1)))
        indexed[key] = items[0]
    return indexed


def _print_table(rows: list[dict]) -> None:
    header = (
        "layer boundary        A_rank B_rank   rel_l2       cosine      "
        "A_norm       B_norm       A_rms        B_rms"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['layer_idx']:>5} {row['name']:<14} "
            f"{row['rank_a']:>6} {row['rank_b']:>6} "
            f"{row['rel_l2']:<12.6e} {row['cosine']:<11.6f} "
            f"{row['norm_a']:<12.6e} {row['norm_b']:<12.6e} "
            f"{row['rms_a']:<12.6e} {row['rms_b']:<12.6e}"
        )


def _collapse_candidate(rows: list[dict]) -> dict | None:
    for row in rows:
        if row["name"] == "h_in":
            continue
        if row["rel_l2"] <= 1e-3 and row["cosine"] >= 0.9999:
            return row
    best = None
    for row in rows:
        if row["name"] == "h_in":
            continue
        score = (1.0 - row["cosine"]) + row["rel_l2"]
        if best is None or score < best[0]:
            best = (score, row)
    return None if best is None else best[1]


def main() -> None:
    base_dir = _artifact_dir()
    records = _load_records(base_dir)
    boundaries = _index_boundary_records(records)
    topk = _index_final_topk(records)

    rows = []
    for layer_idx in sorted(
        {layer for (_, layer, name) in boundaries.keys() if name in BOUNDARIES}
    ):
        for name in BOUNDARIES:
            rec_a = boundaries.get((PROMPT_A_SEQLEN, layer_idx, name))
            rec_b = boundaries.get((PROMPT_B_SEQLEN, layer_idx, name))
            if rec_a is None or rec_b is None:
                continue
            tensor_a = rec_a["tensor"]
            tensor_b = rec_b["tensor"]
            cmp_stats = _compare(tensor_a, tensor_b)
            stats_a = _tensor_stats(tensor_a)
            stats_b = _tensor_stats(tensor_b)
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "name": name,
                    "rank_a": int(rec_a["rank"]),
                    "rank_b": int(rec_b["rank"]),
                    "rel_l2": cmp_stats["rel_l2"],
                    "cosine": cmp_stats["cosine"],
                    "norm_a": stats_a["norm"],
                    "norm_b": stats_b["norm"],
                    "rms_a": stats_a["rms"],
                    "rms_b": stats_b["rms"],
                }
            )

    if not rows:
        raise RuntimeError(
            "no comparable boundary pairs found for cache_seqlens 6 and 16"
        )

    print(f"loaded {len(records)} records from {base_dir}")
    print(
        f"prompt A cache_seqlen={PROMPT_A_SEQLEN}, prompt B cache_seqlen={PROMPT_B_SEQLEN}"
    )
    _print_table(rows)

    final_a = boundaries.get((PROMPT_A_SEQLEN, -1, "final_norm"))
    final_b = boundaries.get((PROMPT_B_SEQLEN, -1, "final_norm"))
    if final_a is not None and final_b is not None:
        cmp_stats = _compare(final_a["tensor"], final_b["tensor"])
        stats_a = _tensor_stats(final_a["tensor"])
        stats_b = _tensor_stats(final_b["tensor"])
        print("\nfinal_norm")
        print(
            "  "
            f"rel_l2={cmp_stats['rel_l2']:.6e} cosine={cmp_stats['cosine']:.6f} "
            f"A_norm={stats_a['norm']:.6e} B_norm={stats_b['norm']:.6e}"
        )

    topk_a = topk.get(PROMPT_A_SEQLEN)
    topk_b = topk.get(PROMPT_B_SEQLEN)
    if topk_a is not None and topk_b is not None:
        print("\nfinal logits top-20")
        print(
            f"  A(rank={topk_a['rank']}): ids={topk_a['ids']} values={[round(float(v), 6) for v in topk_a['values']]}"
        )
        print(
            f"  B(rank={topk_b['rank']}): ids={topk_b['ids']} values={[round(float(v), 6) for v in topk_b['values']]}"
        )
        overlap = sorted(
            set(int(v) for v in topk_a["ids"])
            & set(int(v) for v in topk_b["ids"])
        )
        print(f"  overlap_ids={overlap}")

    candidate = _collapse_candidate(rows)
    if candidate is not None:
        print("\nfirst collapse candidate")
        print(
            "  "
            f"layer={candidate['layer_idx']} boundary={candidate['name']} "
            f"rel_l2={candidate['rel_l2']:.6e} cosine={candidate['cosine']:.6f}"
        )


if __name__ == "__main__":
    torch.set_printoptions(linewidth=200)
    main()
