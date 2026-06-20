"""Diff two step-tap dumps (e.g. native vs sglang) to localize where decode
diverges. Loads taps_<tagA>_rank*.pt and taps_<tagB>_rank*.pt from a directory,
aligns by (ctx, name), and reports cosine + relative error per point.

Usage:
    python -m batchgen.debug.compare_step_taps <dir> [tagA] [tagB]
    # default tagA=native, tagB=sglang

Each tap is the decode token's value at LAYER 0 (see step_tap.py). ctx
(cache_seqlens, unique per prompt) is the alignment key, so the two runs match by
prompt regardless of rank placement. `indexer_sel` is compared as index sets, not
cosine. Note: native and sglang use different FP8 kernels, so even correct points
differ slightly — read "divergence" (cosine << 1 / rel-err >> 0), not bit-equality.
"""
from __future__ import annotations

import glob
import os
import sys

import torch

# Canonical substep order within a layer (names are "L{nn}.{substep}").
ORDER = ["hidden_in", "attn_out", "mlp_out", "hidden_out", "logits", "indexer_sel"]


def _sortkey(name):
    """Sort by (layer, substep-order). Names look like 'L03.attn_out'."""
    layer, _, sub = name.partition(".")
    try:
        ln = int(layer[1:])
    except ValueError:
        ln = 99
    return (ln, ORDER.index(sub) if sub in ORDER else 99)


def _load(dir_path, tag):
    merged = {}
    for f in sorted(glob.glob(os.path.join(dir_path, f"taps_{tag}_rank*.pt"))):
        d = torch.load(f, map_location="cpu")
        for k, v in d.items():
            merged[k] = v  # (ctx, name) -> tensor
    return merged


def _cos_relerr(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    rel = float((a - b).norm() / (a.norm() + 1e-12))
    return cos, rel, (a.numel(), b.numel())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    dir_path = sys.argv[1]
    tagA = sys.argv[2] if len(sys.argv) > 2 else "native"
    tagB = sys.argv[3] if len(sys.argv) > 3 else "sglang"

    A, B = _load(dir_path, tagA), _load(dir_path, tagB)
    ctxs = sorted({k[0] for k in A} | {k[0] for k in B})
    names = sorted({k[1] for k in A} | {k[1] for k in B}, key=_sortkey)
    print(f"[compare] {tagA} taps={len(A)}  {tagB} taps={len(B)}  "
          f"ctxs={ctxs}\n")

    hdr = f"{'name':<12} {'ctx':>6} {'cosine':>9} {'rel_err':>9}  shapes/note"
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        for ctx in ctxs:
            ka, kb = (ctx, name), (ctx, name)
            ta, tb = A.get(ka), B.get(kb)
            if ta is None or tb is None:
                where = tagA if ta is not None else tagB if tb is not None else "neither"
                print(f"{name:<12} {ctx:>6} {'—':>9} {'—':>9}  only in {where}")
                continue
            if name.endswith("indexer_sel"):
                sa = set(int(x) for x in ta.flatten().tolist() if x >= 0)
                sb = set(int(x) for x in tb.flatten().tolist() if x >= 0)
                jac = len(sa & sb) / max(1, len(sa | sb))
                print(f"{name:<12} {ctx:>6} {'jac=':>4}{jac:>5.3f} {'':>9}  "
                      f"|A|={len(sa)} |B|={len(sb)} a-only={len(sa-sb)} b-only={len(sb-sa)}")
                continue
            if name.endswith("mlp_in"):
                # native = [1,H] (this rank's token); sglang = [N_gathered,H]
                # (dp-gathered buffer). Best-match native's row against sglang rows
                # => is the token present in the gather, and at which offset?
                H = ta.shape[-1]
                a = ta.reshape(-1, H)[0].float()
                bb = tb.reshape(-1, H).float()
                csim = torch.nn.functional.cosine_similarity(
                    a.unsqueeze(0), bb, dim=1)
                best = int(csim.argmax())
                print(f"{name:<12} {ctx:>6} {float(csim[best]):>9.5f} "
                      f"{'':>9}  best_row={best}/{bb.shape[0]} "
                      f"(gather buffer rows={bb.shape[0]})")
                continue
            cos, rel, shapes = _cos_relerr(ta, tb)
            flag = "  <<< DIVERGES" if (cos < 0.99 or rel > 0.05) else ""
            print(f"{name:<12} {ctx:>6} {cos:>9.5f} {rel:>9.5f}  "
                  f"{shapes[0]}v{shapes[1]}{flag}")
        print()

    # Compact divergence curve: mean over ctx per (layer.substep).
    print("\n=== SUMMARY (mean over ctx) — divergence curve by depth ===")
    print(f"{'name':<16} {'mean_cos':>9} {'mean_rel':>9}  n")
    print("-" * 40)
    for name in names:
        if name.endswith("indexer_sel") or name.endswith("mlp_in"):
            continue
        cs, rs = [], []
        for ctx in ctxs:
            ta, tb = A.get((ctx, name)), B.get((ctx, name))
            if ta is None or tb is None:
                continue
            cos, rel, _ = _cos_relerr(ta, tb)
            cs.append(cos)
            rs.append(rel)
        if cs:
            mc = sum(cs) / len(cs)
            mr = sum(rs) / len(rs)
            flag = "  <<<" if (mc < 0.99 or mr > 0.05) else ""
            print(f"{name:<16} {mc:>9.5f} {mr:>9.5f}  {len(cs)}{flag}")
    print("\nRead the curve: smooth growth => FP8 accumulation; a SPIKE at one "
          "layer's substep => that step is the bug (attn_out vs mlp_out localizes "
          "attention vs MoE within the layer).")


if __name__ == "__main__":
    main()
