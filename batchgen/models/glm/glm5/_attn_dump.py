"""Per-rank attention-output dumper for the GLM-5 prefill hot path.

Gate: BATCHGEN_GLM5_ATTN_DUMP=<base_dir>. Off by default. When set,
the module reads an **active-run pointer file** on every dump call so
that a single long-running server can route dumps to different
subdirectories back-to-back without a restart:

    ${BATCHGEN_GLM5_ATTN_DUMP}/           # base dir (from env, read once)
      .active_run                         # sidecar; contains current subdir name
                                          # (empty/missing → dumping disabled)
      ref16A/
        rows.rank00.jsonl
        manifest.rank00.json
        ...
      ref16B/
        rows.rank00.jsonl
        ...
      test32/
        rows.rank00.jsonl
        ...

The experiment driver writes `ref16A` / `ref16B` / `test32` into
`.active_run` before each client submission, so each client's prefill
lands in its own subdir. File handles are re-opened when the active
run name changes.

Designed for the "2x16 vs 1x32" batching-delta experiment. Per-rank
files (no cross-rank coordination), bounded cost (6 token positions
per seq per site per layer), full float lists + first-8 + norm per row
for both fast (first8) and exact (full) diffs.
"""

import json
import logging
import os
import threading

import torch

_ENV = "BATCHGEN_GLM5_ATTN_DUMP"
_BASE_DIR = os.environ.get(_ENV, "")
_ACTIVE_FILE = os.path.join(_BASE_DIR, ".active_run") if _BASE_DIR else ""
_LOCK = threading.Lock()

# Per-rank state keyed by `(active_run_name, rank)` so that when the
# active run changes between client submissions, the old file handle is
# closed and a new one is opened in the new subdir.
_ROWS_FH: dict = {}
_MANIFEST: dict = {}
_log = logging.getLogger("batchgen.attn_dump")


def _active_run() -> str:
    """Read the current active-run name from the sidecar file.

    Returns empty string if BATCHGEN_GLM5_ATTN_DUMP isn't set, the
    sidecar is missing, or the sidecar is empty (dumping disabled).
    """
    if not _ACTIVE_FILE:
        return ""
    try:
        with open(_ACTIVE_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def enabled() -> bool:
    """True iff BATCHGEN_GLM5_ATTN_DUMP is set AND a non-empty active
    run name is currently written to the sidecar file."""
    return bool(_BASE_DIR) and bool(_active_run())


def _active_dir() -> str:
    """Full path of the currently active run's subdir, or '' if off."""
    run = _active_run()
    if not run:
        return ""
    return os.path.join(_BASE_DIR, run)


def _rows_fh(rank: int, active: str):
    """Return the open file handle for this (active_run, rank).

    When active_run changes, closes the old handle and opens a new one
    in the new subdir. Thread-safe via module-level _LOCK (caller holds it).
    """
    key = (active, rank)
    if key in _ROWS_FH:
        return _ROWS_FH[key]
    # Close any stale handles for this rank under a different run.
    for (prev_run, r), fh in list(_ROWS_FH.items()):
        if r == rank and prev_run != active:
            try:
                fh.close()
            except Exception:
                pass
            del _ROWS_FH[(prev_run, r)]
    sub = os.path.join(_BASE_DIR, active)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"rows.rank{rank:02d}.jsonl")
    _ROWS_FH[key] = open(path, "a", buffering=1)
    _log.warning(f"[ATTN-DUMP] rank={rank} run={active} opened {path}")
    return _ROWS_FH[key]


def note_microbatch(rank: int, mb_idx: int, global_ids, cu_seqlens,
                    max_seqlen: int, prepack_mode: bool) -> None:
    """Record per-microbatch metadata AND flush the manifest atomically.

    Flush-per-microbatch is deliberate: the experiment driver kills the
    server with pkill -9 at the end, so deferring the flush to a
    graceful shutdown hook would lose the manifest. The file is tiny
    (a few KB) so flushing every call is free.
    """
    if not enabled():
        return
    active = _active_run()
    key = (active, rank)
    m = _MANIFEST.setdefault(
        key,
        {
            "rank": rank,
            "run": active,
            "base_dir": _BASE_DIR,
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
    # Atomic write: dump to tmp + rename. SIGKILL-safe.
    sub = os.path.join(_BASE_DIR, active)
    with _LOCK:
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"manifest.rank{rank:02d}.json")
        tmp = path + ".tmp"
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
      {rank, run, gid, site, layer, token_idx, norm, first8, full}
    """
    if not enabled():
        return
    active = _active_run()
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
                "run": active,
                "gid": gid,
                "site": site,
                "layer": int(layer_idx),
                "token_idx": int(ti),
                "token_idx_abs": int(ti_abs),
                "norm": float(sl.norm().item()),
                "first8": sl[:8].tolist(),
                "full": sl.tolist(),
            })
    if not rows:
        return
    with _LOCK:
        f = _rows_fh(rank, active)
        for r in rows:
            f.write(json.dumps(r) + "\n")


def dump_full_tensor(rank: int, name: str, layer_idx: int,
                     tensor: torch.Tensor) -> None:
    """Serialize a FULL tensor (no token sampling) to a .pt file.

    Intended for the one-shot layer-0 Q/K/V dump used to verify whether
    the middle-token Q/K/V diverges between single-seq and multi-seq
    prefill paths (which the site-D row sampling can't detect).

    Gated by active run; writes
      ${BATCHGEN_GLM5_ATTN_DUMP}/<active_run>/full_<name>_rank{NN}_layer{LL}.pt
    """
    if not enabled():
        return
    active = _active_run()
    sub = os.path.join(_BASE_DIR, active)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"full_{name}_rank{rank:02d}_layer{layer_idx:02d}.pt")
    # Detach + move to CPU so the file is dtype-exact and allocator-free.
    t = tensor.detach().cpu()
    torch.save(t, path)
    _log.warning(f"[ATTN-DUMP-FULL] rank={rank} layer={layer_idx} wrote {path} "
                 f"shape={list(t.shape)} dtype={t.dtype}")


def close_rank(rank: int) -> None:
    """Close all open file handles for this rank. Safe to call multiple times."""
    if not _BASE_DIR:
        return
    with _LOCK:
        for key in list(_ROWS_FH.keys()):
            _, r = key
            if r != rank:
                continue
            try:
                _ROWS_FH[key].close()
            except Exception:
                pass
            del _ROWS_FH[key]
