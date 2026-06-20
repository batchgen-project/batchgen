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

# Canonical layer-0 substep order (native taps; indexer_sel only on sglang).
ORDER = ["hidden_in", "attn_out", "mlp_out", "hidden_out", "logits", "indexer_sel"]


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
    names = sorted({k[1] for k in A} | {k[1] for k in B},
                   key=lambda n: ORDER.index(n) if n in ORDER else 99)
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
            if name == "indexer_sel":
                sa = set(int(x) for x in ta.flatten().tolist() if x >= 0)
                sb = set(int(x) for x in tb.flatten().tolist() if x >= 0)
                jac = len(sa & sb) / max(1, len(sa | sb))
                print(f"{name:<12} {ctx:>6} {'jac=':>4}{jac:>5.3f} {'':>9}  "
                      f"|A|={len(sa)} |B|={len(sb)} a-only={len(sa-sb)} b-only={len(sb-sa)}")
                continue
            cos, rel, shapes = _cos_relerr(ta, tb)
            flag = "  <<< DIVERGES" if (cos < 0.99 or rel > 0.05) else ""
            print(f"{name:<12} {ctx:>6} {cos:>9.5f} {rel:>9.5f}  "
                  f"{shapes[0]}v{shapes[1]}{flag}")
        print()

    print("Read: hidden_in should match (same input). The FIRST substep that "
          "DIVERGES localizes the bug (attn_out => attention; hidden_out only => "
          "MLP/residual; logits only => lm_head/norm).")


if __name__ == "__main__":
    main()
