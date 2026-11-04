# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Decode tokens-per-expert capture for grouped-MoE expert paths.

WHAT THIS OBSERVES
------------------
Every grouped-MoE expert path already computes, per layer and per decode step,
the number of dispatched rows per LOCAL expert: the ``expert_counts`` tensor the
grouped GEMM needs in order to size its row tiles.  The kernel consumes it and
it is then discarded, so the per-rank expert load distribution of a decode wave
has never been observable.  This module accumulates exactly that tensor into a
persistent ``[num_layers, num_local_experts]`` int64 matrix and reads it back
once, at the end of the batch.

Each rank sees only its own local expert shard.  Stitching the per-rank matrices
of an expert-parallel group gives the global per-layer expert popularity; the
JSON payload therefore records this rank's global expert id range so the
stitching is unambiguous.

LIFECYCLE
---------
1. ``register_geometry`` is called once per rank, when the resident expert
   shards are built.  It records the layer ids, the local expert count, the
   global expert offset and the device.  It allocates nothing.
2. The engine ``arm``s the capture at a decode batch boundary, from a
   batch-level debug flag.  Arming allocates ONE persistent accumulator and
   zeroes it.  Both happen eagerly, on the calling stream, outside any
   CUDA-graph capture.
3. Each expert path calls ``accumulate(slot, counts)``.  That is a single
   device-side ``add_`` into a fixed row of the persistent accumulator: no host
   read, no synchronization, no allocation, no host-visible branch.
4. The engine ``flush``es at the end of the decode batch: ONE device-to-host
   copy of the whole matrix, a one-line digest, and the full matrix as JSON.

CUDA-GRAPH SAFETY
-----------------
``accumulate`` is reachable from Python only while the capture is armed, so when
the flag is off nothing is recorded into a graph and the cost of the off path is
one module-level attribute read per expert path, and nothing at all on the
device.  The direct consequence is that the armed state is BAKED INTO a captured
graph: a graph captured while unarmed will never accumulate, and a graph
captured while armed will keep accumulating.

``graph_epoch`` exposes that state as a single integer for a graph owner to
compare against the epoch its graphs were captured under, and to drop and
re-capture them when the two differ.  The accumulator is allocated once per
geometry and is deliberately NEVER freed (it is tens of kilobytes), so a re-arm
re-zeroes the SAME device address and ``graph_epoch`` returns to the value it
had during the previous armed window -- graphs captured in an earlier armed
window stay valid and do not need a second re-capture.

Accumulating into a fixed address is what makes the add graph-safe, and it is
also why the readout is a batch-level quantity: the matrix is the sum over every
decode step of the armed window, not a per-step series.
"""

import json
import logging
from typing import List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# Log tag of every line this module emits; the engine reuses it so one grep
# finds the digest, the JSON payload and the artifact path together.
LOG_TAG = "[EXPERT_COUNTS]"


class ExpertCountsCaptureError(RuntimeError):
    """Raised when a capture is requested but cannot be served exactly.

    This diagnostic never degrades silently: a run that asked for the capture
    and did not get it would publish a load distribution that is not the one the
    decode actually produced.
    """


class ExpertCountsGeometry(object):
    """Immutable per-rank shape of the capture."""

    __slots__ = ("layer_ids", "num_local_experts", "expert_start", "device")

    def __init__(self, layer_ids, num_local_experts, expert_start, device):
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self.num_local_experts = int(num_local_experts)
        self.expert_start = int(expert_start)
        self.device = device
        if not self.layer_ids:
            raise ExpertCountsCaptureError(
                "expert-counts capture geometry needs at least one MoE layer"
            )
        if self.num_local_experts <= 0:
            raise ExpertCountsCaptureError(
                "expert-counts capture geometry needs a positive local expert "
                "count, got %d" % self.num_local_experts
            )

    @property
    def num_layers(self):
        return len(self.layer_ids)

    def key(self):
        return (self.layer_ids, self.num_local_experts, self.expert_start,
                str(self.device))


class ExpertCountsCapture(object):
    """Persistent ``[num_layers, num_local_experts]`` int64 accumulator.

    One instance per geometry, created on the first arm and reused by every
    later armed window so the device address stays stable across CUDA-graph
    captures (see the module docstring).
    """

    def __init__(self, geometry: ExpertCountsGeometry, generation: int):
        self.geometry = geometry
        self.generation = int(generation)
        self.steps = 0
        self._accum = torch.zeros(
            (geometry.num_layers, geometry.num_local_experts),
            dtype=torch.int64,
            device=geometry.device,
        )
        # Pre-materialized row views: ``accumulate`` must not build a new view
        # per call, because it runs once per MoE layer per decode step and is
        # recorded into a CUDA graph.  Each view aliases the persistent
        # accumulator, so the recorded add targets a fixed address.
        self._rows = [self._accum[i] for i in range(geometry.num_layers)]

    # -- hot path ---------------------------------------------------------- #

    def accumulate(self, slot: int, counts: torch.Tensor) -> None:
        """Add one layer's per-local-expert row counts into the accumulator.

        ``slot`` is the layer's position in the registered layer id list, NOT
        the model layer index.  Graph-safe: one in-place device add, no host
        read of ``counts``, no allocation.
        """
        if slot < 0 or slot >= len(self._rows):
            raise ExpertCountsCaptureError(
                "expert-counts capture slot %r is outside the registered "
                "range [0, %d)" % (slot, len(self._rows))
            )
        if counts.numel() != self.geometry.num_local_experts:
            raise ExpertCountsCaptureError(
                "expert-counts capture expected %d local expert counts, got a "
                "tensor of %d elements"
                % (self.geometry.num_local_experts, counts.numel())
            )
        self._rows[slot].add_(counts.view(-1))

    def mark_step(self) -> None:
        """Count one decode step of the armed window (host-side only)."""
        self.steps += 1

    def reset(self) -> None:
        """Zero the accumulator in place, keeping its device address."""
        self._accum.zero_()
        self.steps = 0

    # -- readout ----------------------------------------------------------- #

    def readout(self) -> List[List[int]]:
        """One device-to-host copy of the whole matrix."""
        return self._accum.detach().to("cpu").tolist()


def layer_stats(row: Sequence[int]) -> Tuple[int, float, int, int]:
    """``(max, mean, min, total)`` over one layer's local experts."""
    if not row:
        raise ExpertCountsCaptureError(
            "expert-counts capture cannot summarize an empty layer row"
        )
    total = int(sum(int(v) for v in row))
    return (
        max(int(v) for v in row),
        total / float(len(row)),
        min(int(v) for v in row),
        total,
    )


def summary_line(rank: int, steps: int, matrix: Sequence[Sequence[int]],
                 geometry: Optional[ExpertCountsGeometry] = None) -> str:
    """One-line per-rank digest of a captured matrix.

    Reports, over the layers, the distribution of each layer's max/mean/min
    over local experts, plus the per-layer imbalance ratio ``max/mean`` -- the
    quantity that bounds what a rebalancing of this rank's expert set could
    recover -- and the grand total of dispatched rows.
    """
    if not matrix:
        raise ExpertCountsCaptureError(
            "expert-counts capture produced no layers to summarize"
        )
    num_layers = len(matrix)
    num_experts = len(matrix[0])
    per_layer = [layer_stats(row) for row in matrix]
    total = sum(stat[3] for stat in per_layer)

    maxima = [stat[0] for stat in per_layer]
    means = [stat[1] for stat in per_layer]
    minima = [stat[2] for stat in per_layer]
    ratios = [
        (stat[0] / stat[1]) if stat[1] > 0 else 0.0 for stat in per_layer
    ]
    worst = max(range(num_layers), key=lambda i: ratios[i])
    best = min(range(num_layers), key=lambda i: ratios[i])
    layer_ids = (
        geometry.layer_ids if geometry is not None else tuple(range(num_layers))
    )
    expert_range = (
        "%d..%d"
        % (geometry.expert_start,
           geometry.expert_start + geometry.num_local_experts - 1)
        if geometry is not None
        else "0..%d" % (num_experts - 1)
    )

    return (
        "%s rank=%d steps=%d layers=%d local_experts=%d global_expert_ids=%s "
        "total_rows=%d | over layers, per-layer max: max=%d mean=%.1f min=%d "
        "| per-layer mean: max=%.1f mean=%.1f min=%.1f "
        "| per-layer min: max=%d mean=%.1f min=%d "
        "| imbalance max/mean: worst=%.3f@layer%d mean=%.3f best=%.3f@layer%d"
        % (
            LOG_TAG, int(rank), int(steps), num_layers, num_experts,
            expert_range, total,
            max(maxima), sum(maxima) / float(num_layers), min(maxima),
            max(means), sum(means) / float(num_layers), min(means),
            max(minima), sum(minima) / float(num_layers), min(minima),
            ratios[worst], layer_ids[worst],
            sum(ratios) / float(num_layers),
            ratios[best], layer_ids[best],
        )
    )


def payload(rank: int, steps: int, matrix: Sequence[Sequence[int]],
            geometry: Optional[ExpertCountsGeometry] = None) -> dict:
    """JSON-serializable full capture: the matrix plus its per-layer stats."""
    if not matrix:
        raise ExpertCountsCaptureError(
            "expert-counts capture produced no layers to serialize"
        )
    layer_ids = (
        list(geometry.layer_ids) if geometry is not None
        else list(range(len(matrix)))
    )
    per_layer = []
    for row_idx, row in enumerate(matrix):
        row_max, row_mean, row_min, row_total = layer_stats(row)
        per_layer.append({
            "layer": layer_ids[row_idx],
            "max": row_max,
            "mean": row_mean,
            "min": row_min,
            "total": row_total,
            "imbalance_max_over_mean": (
                row_max / row_mean if row_mean > 0 else 0.0
            ),
        })
    out = {
        "rank": int(rank),
        "steps": int(steps),
        "num_layers": len(matrix),
        "num_local_experts": len(matrix[0]),
        "layer_ids": layer_ids,
        "per_layer": per_layer,
        "counts": [[int(v) for v in row] for row in matrix],
    }
    if geometry is not None:
        out["global_expert_start"] = geometry.expert_start
    return out


# -------------------------------------------------------------------------- #
#  module-level registry                                                      #
# -------------------------------------------------------------------------- #

_GEOMETRY: Optional[ExpertCountsGeometry] = None
_CAPTURE: Optional[ExpertCountsCapture] = None   # allocated once per geometry
_ACTIVE: Optional[ExpertCountsCapture] = None    # non-None only while armed
_GENERATION = 0


def register_geometry(layer_ids, num_local_experts, expert_start, device) -> None:
    """Record this rank's capture shape.  Allocates nothing.

    Called from the expert-shard build, which is the only place that knows the
    rank's MoE layer list and local expert range.  A second registration with a
    different shape invalidates any accumulator built from the old one.
    """
    global _GEOMETRY, _CAPTURE, _ACTIVE, _GENERATION
    geometry = ExpertCountsGeometry(
        layer_ids, num_local_experts, expert_start, device
    )
    if _GEOMETRY is not None and _GEOMETRY.key() == geometry.key():
        return
    _GEOMETRY = geometry
    # A shape change retires the old accumulator; the next arm allocates a new
    # one and bumps the epoch so a graph owner re-captures against it.
    _CAPTURE = None
    _ACTIVE = None
    _GENERATION += 1


def geometry() -> Optional[ExpertCountsGeometry]:
    return _GEOMETRY


def active() -> Optional[ExpertCountsCapture]:
    """The armed capture, or ``None``.  One attribute read; the off path."""
    return _ACTIVE


def is_armed() -> bool:
    return _ACTIVE is not None


def graph_epoch() -> int:
    """Integer identity of what a CUDA-graph capture would record right now.

    ``0`` while unarmed (no add recorded).  While armed it is the accumulator's
    generation, which changes only when a NEW accumulator is allocated, so
    re-arming the same accumulator reproduces the same epoch.
    """
    return 0 if _ACTIVE is None else _ACTIVE.generation


def arm() -> ExpertCountsCapture:
    """Allocate (once) and zero the accumulator, and arm the capture.

    Must be called eagerly, outside CUDA-graph capture, before the decode steps
    whose expert paths should accumulate.  Idempotent within one armed window:
    a second call on an already-armed capture does nothing, so the per-step
    engine hook can call it unconditionally without discarding counts.
    """
    global _CAPTURE, _ACTIVE, _GENERATION
    if _ACTIVE is not None:
        return _ACTIVE
    if _GEOMETRY is None:
        raise ExpertCountsCaptureError(
            "expert-counts capture was requested but no MoE geometry is "
            "registered on this rank: the running model does not build "
            "resident grouped-MoE expert shards, so there are no per-expert "
            "dispatch counts to capture"
        )
    if _CAPTURE is None:
        _GENERATION += 1
        _CAPTURE = ExpertCountsCapture(_GEOMETRY, _GENERATION)
    else:
        _CAPTURE.reset()
    _ACTIVE = _CAPTURE
    return _ACTIVE


def disarm() -> None:
    """Stop accumulating.  Keeps the accumulator and its device address."""
    global _ACTIVE
    _ACTIVE = None


def mark_step() -> None:
    """Count one decode step, if armed.  Host-side only."""
    capture = _ACTIVE
    if capture is not None:
        capture.mark_step()


def flush(rank: int, reason: str, artifact_dir: Optional[str] = None) -> Optional[dict]:
    """Read the matrix back once, log the digest and the JSON, then disarm.

    Returns the JSON payload, or ``None`` when nothing was armed.  ``reason``
    names the boundary that triggered the flush so interleaved decode windows
    stay attributable in the log.
    """
    capture = _ACTIVE
    if capture is None:
        return None
    matrix = capture.readout()
    out = payload(rank, capture.steps, matrix, capture.geometry)
    out["reason"] = str(reason)
    logger.info("%s", summary_line(rank, capture.steps, matrix, capture.geometry))
    written = None
    if artifact_dir:
        written = _write_artifact(out, rank, artifact_dir)
    if written is None:
        logger.info(
            "%s rank=%d reason=%s full matrix: %s",
            LOG_TAG, int(rank), reason, json.dumps(out, sort_keys=True),
        )
    else:
        logger.info(
            "%s rank=%d reason=%s full matrix written to %s",
            LOG_TAG, int(rank), reason, written,
        )
    disarm()
    return out


def _write_artifact(out: dict, rank: int, artifact_dir: str) -> Optional[str]:
    """Write the payload to ``<artifact_dir>/expert_counts_rank<rank>.json``.

    A failed artifact write must not lose the measurement, so it falls back to
    the log line the caller emits instead; it is reported, never swallowed.
    """
    import os

    path = os.path.join(artifact_dir, "expert_counts_rank%d.json" % int(rank))
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(out, handle, sort_keys=True)
    except OSError as exc:
        logger.warning(
            "%s rank=%d artifact write to %s failed (%s); logging the matrix "
            "inline instead", LOG_TAG, int(rank), path, exc,
        )
        return None
    return path


def _reset_for_tests() -> None:
    """Drop all module state.  Test-only."""
    global _GEOMETRY, _CAPTURE, _ACTIVE, _GENERATION
    _GEOMETRY = None
    _CAPTURE = None
    _ACTIVE = None
    _GENERATION = 0
