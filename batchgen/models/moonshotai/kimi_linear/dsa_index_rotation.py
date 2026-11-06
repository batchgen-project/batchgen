# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 DSA: the index-path rotation, and the ONLY place it is decided.

Kimi-K3.1 declares ``index_rope_dim = 0``: nothing on the index path is
position-rotated.  Its index keys live in DeepGEMM's page-split FP8 layout
(the ``kimi_k31_indexer`` Host-KV profile) and are scored by the ReLU-gated
``fp8_paged_mqa_logits`` — the route the production page-split DSA models
already run.  Three producers touch an index vector, and they must agree:

* the PREFILL K writer, which quantizes a prompt's keys for the Host region;
* the DECODE K writer, ``fused_indexer_k_norm_rope_write_fp8``, which owns
  LayerNorm + RoPE + optional Hadamard + quantize + page write in one launch
  and therefore needs a cos/sin table and a Hadamard flag;
* the Q producer, which feeds the scorer.

Two facts make all three trivially consistent on this route:

1. **Identity RoPE.**  Interleaved RoPE with ``cos == 1`` and ``sin == 0`` is
   ``x * 1 - x_pair * 0 = x``, exactly, and position-INDEPENDENT, so every row
   reads table row 0: a ``[1, 64]`` table with all-zero positions.  The kernel
   reads ``cos_ptr[pos * cos_stride + cos_idx]`` with ``cos_idx < 32``
   (``batchgen_kernels/src/attention/hadamard/fused_rope_hadamard.cu:141-146,
   227-229``; the fused writer reads its table the same way) and never
   inspects the table length, so this is bit-identical to a position-indexed
   identity table.  It was also checked on-device: the identity table leaves
   q and k bit-identical.
2. **No Hadamard.**  The repository's single rotation predicate,
   ``indexer_applies_hadamard(page_split=True)``, says the page-split route
   drops the rotation on BOTH q and every index-K producer (upstream V3.2
   semantics).

Together: on this route the index path applies NO transform at all.  The Q
producer and the prefill K writer therefore skip the rotation kernel, and the
decode writer is handed the identity tables plus the predicate's answer —
which this module asserts is "no rotation" before any of the three proceeds.
If the predicate ever changes, all three stop here, together, rather than one
of them quietly rotating into a different space.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

#: The fused index kernels (decode K writer, page-split scorer) are
#: instantiated for a 128-wide row only.
KERNEL_INDEX_HEAD_DIM = 128

#: Width of the cos/sin tables the fused writer reads (the leading 64 of the
#: 128 dims are the "rope" part in its layout; 32 pairs are read).
INDEX_ROPE_TABLE_DIM = 64

#: The K3.1 auxiliary index-K region is page-split FP8 (the
#: ``kimi_k31_indexer`` Host-KV profile). The one fact the rotation predicate
#: is resolved from.
K31_INDEX_CACHE_PAGE_SPLIT = True

_TABLE_CACHE: Dict[Tuple[str, Optional[int]], Tuple[torch.Tensor, torch.Tensor]] = {}


def check_index_head_dim(index_head_dim: int) -> None:
    """Refuse a geometry the fused index kernels are not instantiated for."""
    if int(index_head_dim) != KERNEL_INDEX_HEAD_DIM:
        raise NotImplementedError(
            f"index_head_dim={index_head_dim}: the fused DSA index kernels "
            f"(K writer and scorer) handle a {KERNEL_INDEX_HEAD_DIM}-wide row "
            "only"
        )


def _hadamard_predicate():
    """The repository's single rotation predicate, imported on first use (its
    package ``__init__`` imports triton, which a CPU box does not have)."""
    from batchgen_kernels.attention.dsa.indexer_k_quant import (
        indexer_applies_hadamard,
    )

    return indexer_applies_hadamard


def index_applies_hadamard() -> bool:
    """Whether index vectors are Hadamard-rotated, for the K3.1 aux layout."""
    return bool(_hadamard_predicate()(page_split=K31_INDEX_CACHE_PAGE_SPLIT))


def require_identity_index_rotation() -> None:
    """Stop unless the index path applies no transform at all.

    The Q producer and the prefill K writer skip the rotation kernel on the
    strength of this; the decode writer receives the same answer as its flag.
    A predicate that started rotating page-split records would invalidate all
    three at once, so all three refuse at once.
    """
    if index_applies_hadamard():
        raise RuntimeError(
            "indexer_applies_hadamard(page_split="
            f"{K31_INDEX_CACHE_PAGE_SPLIT}) now returns True. Kimi-K3.1's Q "
            "producer and prefill K writer apply no rotation on the "
            "page-split route; they would score Q and K in different spaces."
        )


def identity_rope_tables(device) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[1, 64]`` float32 identity cos/sin tables, cached per device.

    Never written after creation, so a captured CUDA graph may bake their
    addresses. Read through all-zero positions (:func:`identity_positions`).
    """
    device = torch.device(device)
    key = (device.type, device.index)
    tables = _TABLE_CACHE.get(key)
    if tables is None:
        cos = torch.ones(1, INDEX_ROPE_TABLE_DIM, dtype=torch.float32, device=device)
        sin = torch.zeros(1, INDEX_ROPE_TABLE_DIM, dtype=torch.float32, device=device)
        tables = (cos, sin)
        _TABLE_CACHE[key] = tables
    return tables


def identity_positions(num_rows: int, device) -> torch.Tensor:
    """All-zero int64 positions: every row reads identity table row 0."""
    return torch.zeros(int(num_rows), dtype=torch.int64, device=device)


def decode_writer_rotation_args(num_rows: int, device) -> Dict[str, object]:
    """The rotation arguments the fused decode K writer takes, from one place.

    ``cos_cache`` / ``sin_cache`` / ``positions`` / ``apply_hadamard`` as
    ``write_aux_indexer_k_norm_rope`` names them.
    """
    require_identity_index_rotation()
    cos, sin = identity_rope_tables(device)
    return {
        "cos_cache": cos,
        "sin_cache": sin,
        "positions": identity_positions(num_rows, device),
        "apply_hadamard": index_applies_hadamard(),
    }


def index_q_rows(q: torch.Tensor) -> torch.Tensor:
    """Q side: the index query exactly as the scorer must see it.

    On this route that is the ``wq_b`` output unchanged (identity RoPE, no
    Hadamard); returned as-is after the identity check.
    """
    check_index_head_dim(q.shape[-1])
    require_identity_index_rotation()
    return q


def prefill_index_k_rows(keys: torch.Tensor) -> torch.Tensor:
    """Prefill K side: ``k_norm(wk(hidden))`` exactly as decode writes it.

    The decode writer applies identity RoPE and (on this route) no Hadamard
    to the bf16-rounded LayerNorm output, then quantizes. Prefill produces the
    same pre-quantize values by skipping the no-op rotation.
    """
    check_index_head_dim(keys.shape[-1])
    require_identity_index_rotation()
    return keys


# --------------------------------------------------------------------------- #
#  Reference (CPU-provable statement of the same transform)                     #
# --------------------------------------------------------------------------- #

def _sylvester_hadamard(dim: int, dtype, device) -> torch.Tensor:
    if dim & (dim - 1):
        raise ValueError(f"Hadamard needs a power-of-two dim, got {dim}")
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def _interleaved_rope(rows: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE, transcribed from the repo's own reference
    ``apply_rope_interleaved`` (``batchgen_kernels/attention/dsa/
    fused_indexer_score.py:853``). Only the IDENTITY case is claimed here:
    with ``cos == 1`` / ``sin == 0`` every pairing convention returns its
    input."""
    rope_dim = cos.shape[-1]
    x_rope = rows[..., :rope_dim]
    x_nope = rows[..., rope_dim:]
    x1 = x_rope[..., 0::2]
    x2 = x_rope[..., 1::2]
    cos_h = cos[..., :rope_dim // 2]
    sin_h = sin[..., :rope_dim // 2]
    r1 = x1 * cos_h - x2 * sin_h
    r2 = x2 * cos_h + x1 * sin_h
    x_rot = torch.stack([r1, r2], dim=-1).flatten(-2)
    return torch.cat([x_rot, x_nope], dim=-1)


def rotate_index_rows_reference(rows: torch.Tensor) -> torch.Tensor:
    """What the decode writer does to a row before quantizing, in float32:
    identity RoPE through position 0, then ``x @ H / sqrt(d)`` iff the
    predicate says so. On this route that is the identity."""
    check_index_head_dim(rows.shape[-1])
    x = rows.float()
    cos, sin = identity_rope_tables(x.device)
    x = _interleaved_rope(x, cos[0], sin[0])
    if index_applies_hadamard():
        h = _sylvester_hadamard(KERNEL_INDEX_HEAD_DIM, torch.float32, x.device)
        x = (x @ h) * (KERNEL_INDEX_HEAD_DIM ** -0.5)
    return x
