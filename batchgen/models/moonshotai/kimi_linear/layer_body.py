# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
# ---------------------------------------------------------------------------- #

"""The single Kimi decoder-layer body, shared by eager and CUDA-graph paths.

Kimi has TWO legitimate layer bodies, one per architecture — which one runs is
a property of the checkpoint (``use_attn_residuals`` =
``attn_res_block_size is not None``, config.py), never of the serving phase:

* :func:`run_kimi_layer_body` — the plain pre-norm body
  (norm -> attention -> residual -> norm -> FFN -> residual), e.g.
  Kimi-Linear-48B.
* :func:`run_kimi_block_residual_layer_body` — the K3 / K3.1 Block Attention
  Residual body (depth mix -> boundary snapshot -> norm -> attention ->
  prefix merge -> depth mix -> norm -> FFN -> prefix merge).

Each body is written ONCE, here, in graph-safe form: static shapes for a
bucket, no host sync, no Python branching on tensor values. The former three
spellings of the plain body (``KimiDecoderLayer.forward``,
``KimiLinearSpanSegment.forward``, the compare-mode ``_eager_span``) and the
former two spellings of the block-residual body
(``KimiDecoderLayer._forward_attn_residual``,
``KimiLinearSpanSegment._forward_block_residual``) all route through these two
functions; the CUDA-graph segment CAPTURES them, the eager forward calls them
outside capture, and the whole-model K3 graph reuses the segment's call.

The body is a function of (layer weight handles, per-phase operator set,
activation, state). It reads no global config and no debug state: everything
phase- or capture-specific arrives through :class:`KimiLayerBodyOps`, whose
operator callables close over their own phase state (the eager attention
kwargs and streamed-SP8 hooks on one side; the graph segment's static buffers
on the other).

Kernel identity (why eager and graph calls stay bit-identical to what each
path launched before the extraction):

* The depth mixer (``apply_attn_res``) was ALREADY one shared function —
  ``model.py`` aliased it as ``_apply_attn_res`` and the segment imported it
  directly — so the bodies launch the same mixer both sides launched before.
* The eager operator set binds ``_run_attn`` / ``_run_ffn``, the carried
  profiler hooks, the growing ``BlockResidualBuffer`` boundary append, and the
  in-place prefix merges, so the body reduces to exactly the old eager
  sequence, including its memory behaviour.
* The graph operator set binds the graph-safe attention, ``layer.mlp`` on a
  fold-FFN span, the static-buffer boundary write (captured) or the private
  ``torch.cat`` (compare), and out-of-place prefix merges, so the captured and
  compare sequences reproduce the old segment launch for launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from .block_residual import apply_attn_res


def view_rows_as_bsh(t: torch.Tensor, batch_size: int, seq_len: int,
                     hidden_size: int) -> torch.Tensor:
    """``(rows, H)`` -> ``(B, rows // B, H)``, valid when the batch is EMPTY.

    ``view(batch_size, -1, hidden_size)`` cannot infer ``-1`` from a 0-element
    tensor and raises "cannot reshape tensor of 0 elements ... is ambiguous".
    A decode batch drains to zero rows whenever the last sequences of a run
    complete, which killed a 512-request decode run at 506/512.

    ONLY the empty case is special-cased; a non-empty tensor keeps exactly the
    shape ``-1`` inferred. That matters: the row count here is NOT the caller's
    entry ``seq_len`` -- ``scatter_rows`` shards rows across the TP group -- so
    hoisting a fixed seq_len for the NON-empty case breaks prefill with
    "shape '[1, N, H]' is invalid for input of size ...".

    The empty case must reuse the caller's entry ``seq_len``, not 0. ``numel()
    == 0`` means ``B == 0`` or ``seq_len == 0``, so every ``(B, seq_len, H)``
    holds 0 elements and no sharding can invalidate the choice -- but the
    residual that stays behind keeps the ENTRY shape, so emitting ``(B, 0, H)``
    here makes ``prefix_sum.add_(hidden_states)`` fail with "output with shape
    [0, 1, H] doesn't match the broadcast shape [0, 0, H]".
    """
    if t.numel() == 0:
        return t.view(batch_size, seq_len, hidden_size)
    return t.view(batch_size, -1, hidden_size)


@dataclass
class KimiLayerBodyOps:
    """Per-phase operator set for one Kimi decoder layer.

    attn: ``(normed) -> (attn_output, extras)``. ``extras`` is passed through
        unchanged into the body's output dict (the graph path's host-KV append
        tensors ``k_tensor`` / ``index_k`` / ``index_scales``; the eager and
        compare paths pass ``{}``).
    ffn: ``(normed_out) -> Tensor``, or ``None`` when this phase does NOT run
        the FFN half inside the body: the graph span of a collective-carrying
        MoE layer ends after the post-attention norm and returns the
        ``normed`` / ``residual`` halves for the driver to finish, exactly as
        the segment always has. The eager set always binds ``_run_ffn``; a
        fold-FFN span binds ``layer.mlp``.

    The remaining fields exist only for the block-residual body:

    append_boundary: ``(flat_prefix, block_residual) -> block_residual`` —
        append this boundary layer's pre-mix prefix as the next residual
        column. Eager binds the growing ``BlockResidualBuffer`` append (with
        the streamed-SP8 row gather); the captured graph binds an in-place
        column write into the bucket's static buffer; compare mode binds a
        private ``torch.cat`` so the reference cannot advance the static
        state before the replay. Required by the block-residual body, refused
        by the plain body.
    residual_add_in_place: the prefix merges run as ``prefix_sum.add_(x)``
        (the eager set: at K3 scale each avoided full hidden tensor is
        896 MiB per rank) instead of ``prefix_sum + x`` (the graph set:
        compare mode must not mutate the replay's input, and the captured
        allocations are graph-private either way).
    reshard_prefix: ``(prefix_sum) -> prefix_sum`` for the streamed-SP8
        prefill case where attention returned a row shard and the carried
        prefix must be scattered to the same rows before the merge; ``None``
        (the graph sets) hard-fails if the shapes ever diverge rather than
        broadcasting garbage.
    record_finite: ``(stage, tensor) -> None`` streamed-SP8 finite-check
        probe (``_record_prefill_finite``), or ``None`` (the graph sets),
        which records nothing.
    pre_ffn: ``(block_residual) -> None`` hook between the post-attention
        norm and the FFN — the eager streamed-SP8 carry publishes the global
        row count to the MoE block here. ``None`` runs nothing.
    """

    attn: Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
    ffn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    append_boundary: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None
    residual_add_in_place: bool = False
    reshard_prefix: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    record_finite: Optional[Callable[[str, torch.Tensor], None]] = None
    pre_ffn: Optional[Callable[[torch.Tensor], None]] = None


def run_kimi_layer_body(
    *,
    layer,
    ops: KimiLayerBodyOps,
    hidden_states: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run one PLAIN (no attention-residual) decoder layer.

    ``layer`` is the weight-handle object: anything exposing
    ``input_layernorm`` / ``post_attention_layernorm``. The eager path passes
    the ``KimiDecoderLayer`` itself; the graph segment passes the layer it
    wraps.

    Returns ``{"hidden": ...}`` when ``ops.ffn`` ran the FFN half, else the
    ``{"normed": ..., "residual": ...}`` halves for the caller's FFN stitch,
    plus whatever extras the attention operator produced.
    """
    bound = [
        name
        for name, value in (
            ("append_boundary", ops.append_boundary),
            ("reshard_prefix", ops.reshard_prefix),
            ("record_finite", ops.record_finite),
            ("pre_ffn", ops.pre_ffn),
            ("residual_add_in_place", ops.residual_add_in_place or None),
        )
        if value is not None
    ]
    if bound:
        raise ValueError(
            "run_kimi_layer_body: this layer has no block residuals, but the "
            f"operator set carries {bound}; the block-residual architecture "
            "runs through run_kimi_block_residual_layer_body"
        )

    normed = layer.input_layernorm(hidden_states)
    attn_out, attn_extras = ops.attn(normed)
    residual = hidden_states + attn_out
    normed_out = layer.post_attention_layernorm(residual)
    if ops.ffn is None:
        out = {"normed": normed_out, "residual": residual}
    else:
        out = {"hidden": residual + ops.ffn(normed_out)}
    out.update(attn_extras)
    return out


def _merge_attn_prefix(
    prefix_sum: Optional[torch.Tensor],
    attn_out: torch.Tensor,
    ops: KimiLayerBodyOps,
) -> torch.Tensor:
    """The post-attention prefix merge: ``prefix_sum (+)= attn_out`` per the
    phase's operator set. A fresh prefix (``None``, the post-boundary case)
    adopts the attention output. This merge alone owns the streamed-SP8
    row-shard case — the attention operator may return this rank's rows while
    the carried prefix is still full — which the FFN merge never has (the
    pre-FFN depth mix already settled the row layout)."""
    if prefix_sum is None:
        return attn_out
    if prefix_sum.numel() != attn_out.numel():
        if ops.reshard_prefix is None:
            raise ValueError(
                "kimi layer body: the attention output rows do not match the "
                "carried prefix and this operator set has no reshard_prefix "
                "(streamed-SP8 is an eager prefill mode)"
            )
        prefix_sum = ops.reshard_prefix(prefix_sum)
    if ops.residual_add_in_place:
        # The surviving prefix buffer is updated in place: the merged-in
        # tensor is dead after this point and exact-64K K3 is 896 MiB per
        # full hidden tensor.
        prefix_sum.add_(attn_out)
        return prefix_sum
    return prefix_sum + attn_out


def run_kimi_block_residual_layer_body(
    *,
    layer,
    ops: KimiLayerBodyOps,
    hidden_states: torch.Tensor,
    block_residual: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Run one K3 Block-Attention-Residual decoder layer.

    ``layer`` additionally exposes ``layer_idx``, ``attn_res_block_size``, the
    ``self_attention_res_{proj,norm}`` / ``mlp_res_{proj,norm}`` depth-mix
    weights.

    ``block_residual`` is the ACTIVE residual bank this layer consumes:
    ``shape[1]`` counts the boundaries seen so far (the eager caller threads
    the growing view; the graph segment passes its static buffer narrowed to
    ``num_blocks_before``). A boundary layer appends its pre-mix prefix
    through ``ops.append_boundary`` and the widened bank is returned under
    ``"block_residual"`` — the eager caller threads it on; the graph callers
    pop it (their bank is bucket-owned state).

    Returns ``{"hidden": ..., "block_residual": ...}`` when ``ops.ffn`` ran
    the FFN half, else ``{"normed": ..., "residual": ..., "block_residual":
    ...}``, plus the attention operator's extras.
    """
    if ops.append_boundary is None:
        raise ValueError(
            "run_kimi_block_residual_layer_body: the operator set must bind "
            "append_boundary (this architecture snapshots a residual column "
            "at every block boundary)"
        )

    batch_size, seq_len, hidden_size = hidden_states.shape
    record = ops.record_finite
    if record is not None:
        record("entry", hidden_states)

    flat_prefix = hidden_states.reshape(-1, hidden_size)
    prefix_sum = hidden_states

    attn_input = hidden_states
    if block_residual is not None and block_residual.shape[1] > 0:
        attn_input = view_rows_as_bsh(
            apply_attn_res(
                flat_prefix,
                block_residual,
                layer.self_attention_res_proj,
                layer.self_attention_res_norm,
            ),
            batch_size,
            seq_len,
            hidden_size,
        )
        if record is not None:
            record("post_input_depth_mix", attn_input)

    if layer.layer_idx % layer.attn_res_block_size == 0:
        # Boundary: snapshot the PRE-mix prefix_sum as the next residual
        # column, then RESET the running prefix (assignment, not add). HOW the
        # column lands is the phase's: eager writes the next column of the
        # pass's preallocated buffer (gathering sharded rows first); the
        # captured graph writes its bucket's static column; compare cats
        # privately.
        block_residual = ops.append_boundary(flat_prefix, block_residual)
        prefix_sum = None

    normed = layer.input_layernorm(attn_input)
    attn_out, attn_extras = ops.attn(normed)
    if record is not None:
        record("post_attention", attn_out)

    prefix_sum = _merge_attn_prefix(prefix_sum, attn_out, ops)
    if record is not None:
        record("post_attention_residual", prefix_sum)

    ffn_input = view_rows_as_bsh(
        apply_attn_res(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            layer.mlp_res_proj,
            layer.mlp_res_norm,
        ),
        batch_size,
        seq_len,
        hidden_size,
    )
    if record is not None:
        record("post_mlp_depth_mix", ffn_input)

    normed_out = layer.post_attention_layernorm(ffn_input)
    if record is not None:
        record("post_mlp_norm", normed_out)

    if ops.ffn is None:
        out = {
            "normed": normed_out,
            "residual": prefix_sum,
            "block_residual": block_residual,
        }
    else:
        if ops.pre_ffn is not None:
            ops.pre_ffn(block_residual)
        ffn_out = ops.ffn(normed_out)
        if record is not None:
            record("post_ffn", ffn_out)
        if ops.residual_add_in_place:
            # The FFN output is dead after this residual merge; reuse the
            # surviving prefix buffer (the old eager body's add_).
            prefix_sum.add_(ffn_out)
            hidden = prefix_sum
        else:
            hidden = prefix_sum + ffn_out
        if record is not None:
            record("output", hidden)
        out = {"hidden": hidden, "block_residual": block_residual}
    out.update(attn_extras)
    return out


__all__ = [
    "KimiLayerBodyOps",
    "run_kimi_block_residual_layer_body",
    "run_kimi_layer_body",
    "view_rows_as_bsh",
]
