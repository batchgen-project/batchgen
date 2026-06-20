"""Cross-path decode step-tap: snapshot named intermediate tensors for the
decode token at ONE layer (layer 0) so the BatchGen-native and SGLang decode
paths can be diffed step-by-step to localize where they diverge.

Ground-truth comparison infra (reusable for future model bring-up): run native
and SGLang decode on the SAME prefilled KV + same decode token, tap the same
named points in each, then `compare_step_taps.py` aligns by (ctx, name) and
prints the first point exceeding tolerance.

Enable per-job via the batch-level flag (no env guard, no cold restart — rides
the same AttnWrapperBase.batchgen_debug path as glm5_moe_mode):

    batchgen_debug = {"step_tap": "<run_tag>"}     # e.g. "native" or "sglang"

Each tap stores `tensor.detach().to(float32).cpu()` — a read-only copy; the GPU
compute path is untouched (bf16/fp8 unchanged). fp32+CPU so the saved value and
the offline diff carry no extra bf16 rounding. Captures the FIRST decode forward
only. Dumps one .pt per rank to BATCHGEN_STEP_TAP_DIR (default /tmp/step_tap),
keyed (ctx, name); ctx (= cache_seqlens) is a natural per-prompt key (each of the
8 MMLU prompts has a distinct context length), so native<->sglang align by ctx
regardless of which rank handled which prompt.
"""
from __future__ import annotations

import logging
import os

import torch

TAP_LAYER = 0  # back-compat alias (layer 0)
# Depth sweep: layers 0-2 are dense MLP, 3+ are MoE (first_k_dense_replace=3).
# Layer 0 matched native within FP8 noise; divergence accumulates -> logits, so
# sweep depth (incl. the first MoE layer 3) to find where it first spikes.
TAP_LAYERS = (0, 3, 8, 20, 40, 60, 77)

_active_tag = None      # run tag for the in-flight first forward, else None
_rank = None
_done = False           # capture the FIRST decode forward only
_store = {}             # (ctx:int, name:str) -> fp32 cpu tensor


def _tag():
    """Run tag: batch-level flag preferred (no restart); env fallback is natural
    here since native vs SGLang are separate server launches anyway."""
    from batchgen.models.wrappers import AttnWrapperBase

    debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
    return debug.get("step_tap") or os.getenv("BATCHGEN_STEP_TAP")


def _ctx_for_row(row=0):
    """Per-rank decode ctx (cache_seqlens). dp16 => 1 seq/rank, so row 0."""
    from batchgen.models.wrappers import AttnWrapperBase

    cs = getattr(AttnWrapperBase, "cache_seqlens", None)
    if cs is None or cs.numel() <= row:
        return -1
    return int(cs.view(-1)[row].item())


def begin(rank=None):
    """Call at the start of a decode model forward. Activates tapping for the
    FIRST forward when batchgen_debug.step_tap is set; returns the run tag (or
    None). Subsequent forwards are inert (one-shot)."""
    global _active_tag, _rank
    if _done:
        _active_tag = None
        return None
    tag = _tag()
    _active_tag = tag if tag else None
    if rank is None:
        try:
            rank = torch.distributed.get_rank()
        except Exception:  # noqa: BLE001
            rank = 0
    _rank = rank
    return _active_tag


def active():
    return _active_tag is not None


def tap(name, tensor, layer_id=0, row=0):
    """Snapshot `tensor` (the decode token's value) under (ctx, "L{layer}.{name}").
    No-op unless tapping is active and layer_id is in the sweep set."""
    if _active_tag is None or layer_id not in TAP_LAYERS:
        return
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return
    ctx = _ctx_for_row(row)
    if ctx <= 0:  # idle dp rank (0 local seqs) — nothing to compare
        return
    _store[(ctx, "L%02d.%s" % (int(layer_id), name))] = (
        tensor.detach().to(torch.float32).cpu())


def flush():
    """Dump this rank's taps and disarm (one-shot). Call at end of the forward."""
    global _active_tag, _done
    if _active_tag is None:
        return
    if _store:
        d = os.getenv("BATCHGEN_STEP_TAP_DIR", "/tmp/step_tap")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"taps_{_active_tag}_rank{_rank}.pt")
        torch.save(_store, path)
        logging.getLogger().warning(
            "[STEP-TAP] tag=%s rank=%s wrote %d taps -> %s (keys=%s)",
            _active_tag, _rank, len(_store), path,
            sorted({k[1] for k in _store}),
        )
    _store.clear()
    _active_tag = None
    _done = True
