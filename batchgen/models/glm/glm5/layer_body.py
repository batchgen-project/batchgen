# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""The single GLM-5 decoder-layer body, shared by eager and CUDA-graph paths.

The layer body -- norm -> attention -> fused add+norm -> MoE/MLP -> residual --
is written ONCE, here, in graph-safe form: static shapes for a bucket, no host
sync, no Python branching on tensor values. ``Glm5DecoderLayer.forward``
(eager, both phases) calls this function outside capture;
``Glm5DecoderLayerGraphSegment.forward`` is captured into the decode CUDA
graph and runs the SAME function instead of spelling the body out again.

The body is a function of (layer weight handles, per-phase operator set,
activation, state). It reads no global config and no debug state: everything
phase- or capture-specific arrives through ``Glm5LayerBodyOps``, whose
operator callables close over their own phase state (eager attention kwargs
on one side; the DSA/MoE graph-segment statics on the other).

Kernel identity (why eager and graph calls stay bit-identical to what each
path launched before the extraction):

* The eager operator set leaves every fusion switch False and the row mask
  absent, so the body reduces to exactly the old eager sequence --
  ``input_layernorm`` module call, attention, ``cuda_add_rmsnorm``, MLP,
  materialized ``residual + mlp_output``.
* The graph operator set carries the inner segments' fusion decisions, so the
  body reproduces the old segment sequence launch for launch, including the
  ``cuda_add_rmsnorm_fp8_quant`` fusions and the deferred-residual chaining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


@dataclass
class Glm5LayerBodyOps:
    """Per-phase operator set for one decoder layer.

    attn: keyword callable ``(hidden_states=, hidden_fp8=, hidden_scale=)``
        returning a dict with at least ``"attn_output"``. Every OTHER key is
        passed through unchanged to the body's output dict (graph: the
        host-KV append tensors ``primary_k_tensor`` / ``indexer_k_tensor`` /
        ``indexer_k_scale``; eager: ``attn_weights`` and ``present``).
    ffn: callable ``(hidden_states, hidden_fp8, hidden_scale) -> Tensor``.
        The FP8 pair is non-None only when ``consumes_fused_padded_fp8``
        asked the post-attention norm launch to emit it.
    consumes_fused_hidden_fp8: the attention operator wants its qkv_a input
        emitted by the preceding norm launch as a blocked-FP8 pair (graph
        decode only; the eager operator set leaves it False). The inner
        segment owns that decision -- see norm_quant_fuse_policy.py; this
        body only runs the matching norm version.
    consumes_fused_padded_fp8: the same question for the FFN operator's
        shared-expert input.
    attn_output_row_mask: ``(batch_size) -> Optional[Tensor]`` row gate the
        body folds into the post-attention add+norm launch (the attention
        operator is told to skip its own mask launch by its caller). ``None``
        -- the eager set -- folds nothing and the launch receives no mask.
    criticality_probe: decode criticality probe (criticality_probe.py), or
        ``None`` (the default and the eager value), which launches nothing.
    """

    attn: Callable[..., Dict[str, torch.Tensor]]
    ffn: Callable[
        [torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        torch.Tensor,
    ]
    consumes_fused_hidden_fp8: bool = False
    consumes_fused_padded_fp8: bool = False
    attn_output_row_mask: Optional[
        Callable[[int], Optional[torch.Tensor]]
    ] = None
    criticality_probe: Optional[object] = None


def run_glm5_layer_body(
    *,
    layer,
    ops: Glm5LayerBodyOps,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    defer_residual_add: bool = False,
    num_valid_tokens: Optional[torch.Tensor] = None,
    layer_idx: int = -1,
) -> Dict[str, torch.Tensor]:
    """Run one decoder layer.

    ``layer`` is the weight-handle object: anything exposing
    ``input_layernorm`` / ``post_attention_layernorm`` (norm modules with
    ``weight`` / ``eps``) and, for the dense path's eager twin, ``mlp``
    behind ``ops.ffn``. The eager path passes the ``Glm5DecoderLayer``
    itself; the graph segment passes the layer it wraps. The post-attention
    norm is read only AFTER the attention operator ran, never eagerly.

    ``residual`` / ``defer_residual_add`` let a driver chain the layers in
    the fused-add form. With ``defer_residual_add`` the layer does NOT
    materialize ``residual + mlp_output``; it returns the two halves and the
    NEXT layer's ``input_layernorm`` becomes the fused add+RMSNorm that was
    going to run anyway, so the standalone add disappears from the graph.
    ``residual`` is the previous layer's carried residual and is ``None``
    only for the first layer, whose input comes straight from the
    embedding. The driver owes the final add after the last layer.

    Mixing the two conventions mid-chain would drop a residual, so
    ``residual`` without ``defer_residual_add`` is a hard error rather than
    a silent fallback.
    """
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm

    if residual is not None and not defer_residual_add:
        raise ValueError(
            "run_glm5_layer_body: residual was carried in "
            "but defer_residual_add is False, which would materialize this "
            "layer's add while the previous layer's add is still pending; "
            "a driver must use one convention for the whole layer chain"
        )

    batch_size = int(hidden_states.shape[0])

    # Does the attention operator below want its qkv_a activation emitted BY
    # this input norm instead of quantized by its own launch? The operator set
    # owns that decision (it was fixed before capture); this body only has to
    # run the matching norm version. See norm_quant_fuse_policy.py.
    fuse_attn_input = bool(ops.consumes_fused_hidden_fp8)
    attn_input_fp8 = attn_input_scale = None

    if residual is None:
        # First layer of the chain: its input IS the residual stream.
        residual = hidden_states
        attn_input = layer.input_layernorm(hidden_states)
        if fuse_attn_input:
            # There is no residual add in the chain's FIRST layer, so its
            # input norm is the standalone `cuda_rmsnorm`
            # (rmsnorm_bf16_6144_kernel), which has no fused FP8 version --
            # fusing a kernel that is not the one the other 77 layers run
            # would be a second variant for ONE launch per step. That one
            # quantize stays standalone, which also keeps the attention
            # operator on a single contract: it always receives the FP8 pair.
            from batchgen.attention.mla.fa3_backend import act_quant
            attn_input_fp8, attn_input_scale = act_quant(
                attn_input.view(batch_size, attn_input.shape[-1]),
                num_valid_tokens=num_valid_tokens,
                scale_tma_aligned=num_valid_tokens is not None,
            )
    elif fuse_attn_input:
        # The previous layer's deferred `residual + mlp_output`, the input
        # norm, AND the qkv_a blocked-FP8 cast, in ONE launch: the FP8
        # bytes come from the registers that held the normalized row, so
        # the attention operator's standalone quantize is not launched at
        # all. `normed`/`residual` are bit-for-bit `cuda_add_rmsnorm`'s and
        # the FP8 pair is bit-for-bit `act_quant`'s (byte-identity note in
        # batchgen_kernels/src/attention/csrc/add_rmsnorm_fp8_quant.cu).
        from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
            cuda_add_rmsnorm_fp8_quant,
        )
        (
            attn_input,
            residual,
            attn_input_fp8,
            attn_input_scale,
        ) = cuda_add_rmsnorm_fp8_quant(
            residual,
            hidden_states,
            layer.input_layernorm.weight,
            layer.input_layernorm.eps,
            # The norm itself is NOT row-gated here, exactly as the
            # unfused call below is not: every row is normalized, padding
            # included. Only the FP8 half is gated, which is what the
            # standalone `act_quant(..., num_valid_tokens=...)` did.
            quant_num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=num_valid_tokens is not None,
        )
    else:
        # The previous layer's deferred `residual + mlp_output` happens
        # HERE, inside the input-norm launch. `hidden_states` is that
        # layer's mlp_output.
        attn_input, residual = cuda_add_rmsnorm(
            residual,
            hidden_states,
            layer.input_layernorm.weight,
            layer.input_layernorm.eps,
        )
    if ops.criticality_probe is not None:
        ops.criticality_probe.mark("after_input_norm_quant", layer_idx)

    attn_out = ops.attn(
        hidden_states=attn_input,
        hidden_fp8=attn_input_fp8,
        hidden_scale=attn_input_scale,
    )
    if ops.criticality_probe is not None:
        ops.criticality_probe.mark("after_attention", layer_idx)

    # Same question for the FFN block's shared expert: does it want its x
    # activation emitted BY this post-attention norm? A dense layer's
    # operator set has no MoE segment and therefore no shared expert, so it
    # never fuses; the eager set never fuses either.
    fuse_moe_input = bool(ops.consumes_fused_padded_fp8)
    moe_input_fp8 = moe_input_scale = None
    # The attention operator's row gate (padding rows of the bucket) is
    # absorbed into the add+norm launch below instead of costing the
    # attention path its own masking launch. The eager set has no mask and
    # the launch is exactly the historical unmasked `cuda_add_rmsnorm`.
    attn_output_row_mask = (
        None
        if ops.attn_output_row_mask is None
        else ops.attn_output_row_mask(batch_size)
    )
    if fuse_moe_input:
        # The residual add, the post-attention norm AND the shared
        # expert's x blocked-FP8 cast in ONE launch. The shared expert's
        # own quantize passed `scale_tma_aligned=True` and NO
        # num_valid_tokens (it quantizes every row of the bucket), so this
        # call gates neither half -- the padding rows carry exactly the
        # values they carried before.
        from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
            cuda_add_rmsnorm_fp8_quant,
        )
        (
            hidden_states,
            residual,
            moe_input_fp8,
            moe_input_scale,
        ) = cuda_add_rmsnorm_fp8_quant(
            residual,
            attn_out["attn_output"],
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
            row_valid_mask=attn_output_row_mask,
            scale_tma_aligned=True,
        )
    else:
        # Fused residual add + RMSNorm (saves one HBM pass of [B, 6144]).
        hidden_states, residual = cuda_add_rmsnorm(
            residual,
            attn_out["attn_output"],
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
            row_valid_mask=attn_output_row_mask,
        )
    if ops.criticality_probe is not None:
        ops.criticality_probe.mark("after_post_attn_norm_quant", layer_idx)

    mlp_output = ops.ffn(hidden_states, moe_input_fp8, moe_input_scale)

    if defer_residual_add:
        # `mlp_output` may be the MoE pool's shared static buffer, which
        # the NEXT layer's MoE block overwrites. That is safe only because
        # the next layer's add+norm READS it to produce the input its own
        # MoE block consumes, so the overwrite is downstream of the read by
        # a true data dependency, not by launch-order luck.
        out: Dict[str, torch.Tensor] = {
            "hidden_states": mlp_output,
            "residual": residual,
        }
    else:
        out = {
            "hidden_states": residual + mlp_output,
        }
    # Everything else the attention operator produced rides along: the graph
    # path's host-KV append tensors, the eager path's attention weights and
    # present-KV tuple.
    for key, value in attn_out.items():
        if key != "attn_output":
            out[key] = value
    return out
