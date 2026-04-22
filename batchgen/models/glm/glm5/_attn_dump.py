"""Per-rank attention-output dumper for the GLM-5 prefill hot path.

Gate: BATCHGEN_GLM5_ATTN_DUMP=<dir>. Off by default. When set to a
directory path, every rank *independently* writes its own
`rows.rank{NN}.jsonl` + `manifest.rank{NN}.json` into that directory.
Nothing is aggregated or dropped at runtime — every GPU keeps a complete
record of what it computed.

Used for the "2×16 vs 1×32" batching-delta experiment: the same 32
prompts are run two ways, per-seq attention outputs are dumped at fixed
(layer, site, token_idx) keys, and a post-hoc comparator pairs them by
`global_seq_id` to quantify the divergence introduced purely by
multi-seq-prepacked prefill.

Design choices:
  - per-rank files (no cross-rank coordination; hold the whole device's
    record no matter how ranks shard seqs)
  - bounded cost: 6 token positions per seq per site per layer (~45k
    rows × ~40 KB each = 1–3 GB per run — fits on /data2)
  - row = full float list + first-8-floats summary + scalar norm, so the
    comparator can do both fast (first8) and exact (full) diffs
"""

import json
import logging
import os
import threading

import torch

_ENV = "BATCHGEN_GLM5_ATTN_DUMP"
_DIR = os.environ.get(_ENV, "")
_LOCK = threading.Lock()
_ROWS_FH: dict = {}
_MANIFEST: dict = {}
_log = logging.getLogger("batchgen.attn_dump")


def enabled() -> bool:
    return bool(_DIR)


def _rows_fh(rank: int):
    if rank not in _ROWS_FH:
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"rows.rank{rank:02d}.jsonl")
        _ROWS_FH[rank] = open(path, "a", buffering=1)
        _log.warning(f"[ATTN-DUMP] rank={rank} opened {path}")
    return _ROWS_FH[rank]


def note_microbatch(rank: int, mb_idx: int, global_ids, cu_seqlens,
                    max_seqlen: int, prepack_mode: bool) -> None:
    """Record per-microbatch metadata in the rank's manifest AND flush it
    to disk right away.

    The flush-per-microbatch policy is deliberate: benchmark scripts
    routinely kill the server with SIGKILL (pkill -9 python) between
    runs, so a defer-to-graceful-shutdown manifest would be lost. The
    file is tiny (a few KB) so flushing every call is free.
    """
    if not enabled():
        return
    m = _MANIFEST.setdefault(
        rank,
        {
            "rank": rank,
            "dir": _DIR,
            "allow_multi_seq_prepack": os.environ.get(
                "BATCHGEN_GLM5_ALLOW_MULTI_SEQ_PREPACK", "0"
            ) == "1",
            "microbatches": [],
        },
    )
    m["microbatches"].append({
        "mb_idx": int(mb_idx),
        "global_ids": list(global_ids) if global_ids is not None else [],
        "cu_seqlens": (cu_seqlens.tolist() if torch.is_tensor(cu_seqlens)
                       else list(cu_seqlens)),
        "max_seqlen": int(max_seqlen),
        "prepack_mode": bool(prepack_mode),
    })
    # Atomic-ish write: dump to tmp file then rename. SIGKILL-safe.
    path = os.path.join(_DIR, f"manifest.rank{rank:02d}.json")
    tmp = path + ".tmp"
    with _LOCK:
        os.makedirs(_DIR, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(m, f, indent=2)
        os.replace(tmp, path)


def dump_rows(rank: int, site: str, layer_idx: int, global_ids, cu_seqlens,
              tensor: torch.Tensor,
              token_idxs=(0, 1, 2, -3, -2, -1)) -> None:
    """Dump rows of `tensor` sliced per-seq at fixed token positions.

    Args:
        rank: DP rank.
        site: short site tag, e.g. "A_post_fa3" / "B_post_oproj" / "C_layer_out".
        layer_idx: layer index.
        global_ids: per-seq global ids in admit order (same ordering as
                    cu_seqlens). Used as the cross-run pairing key.
        cu_seqlens: [num_seqs + 1] cumulative seq lengths over `tensor`.
        tensor: [total_tokens, ...] — will be flattened past dim 0 for dump.
        token_idxs: which tokens to sample per seq. Negatives count from
                    the end of the seq. Out-of-range indices are skipped.

    Writes one JSON row per (rank, gid, site, layer, token_idx):
      {rank, gid, site, layer, token_idx, norm, first8, full}
    """
    if not enabled():
        return
    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    t = tensor.detach().float().cpu()
    rows = []
    for i, gid in enumerate(global_ids):
        start, end = int(cu[i]), int(cu[i + 1])
        L = end - start
        if L <= 0:
            continue
        for ti in token_idxs:
            ti_abs = ti + L if ti < 0 else ti
            if ti_abs < 0 or ti_abs >= L:
                continue
            sl = t[start + ti_abs].reshape(-1)
            rows.append({
                "rank": int(rank),
                "gid": gid,
                "site": site,
                "layer": int(layer_idx),
                "token_idx": int(ti),  # preserve the original request so the
                                       # comparator can pair "last-3" on both runs
                "token_idx_abs": int(ti_abs),
                "norm": float(sl.norm().item()),
                "first8": sl[:8].tolist(),
                "full": sl.tolist(),
            })
    if not rows:
        return
    with _LOCK:
        f = _rows_fh(rank)
        for r in rows:
            f.write(json.dumps(r) + "\n")


def close_rank(rank: int) -> None:
    """Flush the rank's manifest to disk and close the row file.

    Call at server shutdown. Safe to call multiple times.
    """
    if not enabled():
        return
    with _LOCK:
        if rank in _MANIFEST:
            path = os.path.join(_DIR, f"manifest.rank{rank:02d}.json")
            with open(path, "w") as f:
                json.dump(_MANIFEST[rank], f, indent=2)
            _log.warning(f"[ATTN-DUMP] rank={rank} flushed manifest to {path}")
        if rank in _ROWS_FH:
            _ROWS_FH[rank].close()
            del _ROWS_FH[rank]
