"""Composition helpers for CUDA-graph capturable segments.

These utilities formalize the natural growth path described in
`batchgen_design/cuda_graph/cuda_graph_contract.md` §C:
  segmented (attn / moe) -> layer -> whole_model

`CaptureContext` genericizes the `AttnWrapperBase` class-attribute bind/restore
pattern that the GLM-5 whole-model segment currently inlines (see
`batchgen/models/glm/glm5/whole_model_cuda_graph_segments.py` lines 394-430).
Composition helpers (`compose_sequential`, `compose_layer_from_segments`,
`compose_whole_model`) provide the lifecycle plumbing so adopters don't
reimplement `setup_static_buffers` / `release_static_buffers` cascades and
input/output spec unions.

Constraints enforced (contract §C):
  - Shared bucketing across all child segments.
  - Static-buffer ownership cascades: outer.setup() calls inner.setup().
  - Input-spec dict unions must not have key conflicts.
  - No dynamic shapes — shapes derive from `bucket_size` + constants.

Adapters may use these helpers OR construct their own `CapturableSegment`
implementations directly. GLM-5 today uses direct construction; new adopters
benefit from these helpers to avoid copy-paste.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from batchgen.cuda_graph.graph_manager import CapturableSegment, TensorSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CaptureContext: bind/restore class attributes for the duration of a forward
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def CaptureContext(target: type, attrs: Mapping[str, Any]):
    """Context manager that binds class attributes on `target` and restores them.

    Use this instead of the inline save-restore pattern at
    `Glm5WholeModelSegment.forward` (whole_model_cuda_graph_segments.py:394-430).
    The contract requires that no segment monkey-patches
    `AttnWrapperBase` ClassVars outside a `CaptureContext`.

    Args:
        target: the class whose attributes are bound (typically
            `batchgen.models.wrappers.attention.AttnWrapperBase`).
        attrs: name -> value mapping. Each name MUST already exist as a
            class attribute on `target` (a missing attribute is a programming
            error, not a silent add).

    Example:
        with CaptureContext(AttnWrapperBase, {
            "cache_seqlens": cache_seqlens,
            "position_ids": position_ids,
            "max_seqlen": max_seqlen,
            "kv_append_callback": copy_primary_kv,
            "kv_append_callback_aux": copy_aux_kv,
        }):
            outputs = run_decode_forward(...)
    """
    saved: Dict[str, Any] = {}
    missing = [name for name in attrs if not hasattr(target, name)]
    if missing:
        raise AttributeError(
            f"CaptureContext: target {target.__name__} has no attribute(s): "
            f"{', '.join(missing)}. Adding new ClassVars implicitly would "
            f"leak across modes — declare them on the target class first."
        )
    try:
        for name, value in attrs.items():
            saved[name] = getattr(target, name)
            setattr(target, name, value)
        yield
    finally:
        for name, prev in saved.items():
            setattr(target, name, prev)


# ---------------------------------------------------------------------------
# compose_sequential: generic CapturableSegment over a sequence of inner ones
# ---------------------------------------------------------------------------

class _ComposedSegment:
    """Internal CapturableSegment built from inner segments + a glue forward.

    Implements the protocol mechanically:
      - `get_static_input_specs`: union of inner specs plus extras (asserts no
        key conflicts on overlap).
      - `get_static_output_specs`: caller-provided.
      - `setup_static_buffers`: cascades to inner segments in order.
      - `release_static_buffers`: cascades in reverse.
      - `forward`: dispatches to caller-provided `forward_fn(inner_segments,
        **inputs) -> Dict[str, Tensor]`.
    """

    def __init__(
        self,
        *,
        inner: List[CapturableSegment],
        extra_input_specs: Mapping[str, TensorSpec],
        output_specs: Mapping[str, TensorSpec],
        forward_fn: Callable[..., Dict[str, torch.Tensor]],
    ):
        self._inner = list(inner)
        self._extra_inputs = dict(extra_input_specs)
        self._outputs = dict(output_specs)
        self._forward_fn = forward_fn

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        merged: Dict[str, TensorSpec] = {}
        for seg in self._inner:
            for k, v in seg.get_static_input_specs(bucket_size).items():
                if k in merged and merged[k] != v:
                    raise ValueError(
                        f"compose_sequential: conflicting spec for input '{k}' "
                        f"between inner segments ({merged[k]} vs {v})"
                    )
                merged[k] = v
        for k, v in self._extra_inputs.items():
            if k in merged and merged[k] != v:
                raise ValueError(
                    f"compose_sequential: extra input '{k}' conflicts with "
                    f"inner-segment spec ({merged[k]} vs {v})"
                )
            merged[k] = v
        return merged

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return dict(self._outputs)

    def setup_static_buffers(self, bucket_size: int) -> None:
        for seg in self._inner:
            setup = getattr(seg, "setup_static_buffers", None)
            if callable(setup):
                setup(bucket_size)

    def release_static_buffers(self, bucket_size: int) -> None:
        for seg in reversed(self._inner):
            release = getattr(seg, "release_static_buffers", None)
            if callable(release):
                release(bucket_size)

    def forward(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._forward_fn(self._inner, **inputs)


def compose_sequential(
    *,
    inner: List[CapturableSegment],
    extra_input_specs: Mapping[str, TensorSpec],
    output_specs: Mapping[str, TensorSpec],
    forward_fn: Callable[..., Dict[str, torch.Tensor]],
) -> CapturableSegment:
    """Build a `CapturableSegment` from a sequence of inner segments + glue.

    `forward_fn(inner_segments, **inputs) -> Dict[str, Tensor]` is the
    user-provided glue that interleaves inner-segment `.forward(...)` calls
    with residuals, norms, or any other model-specific logic.

    The composed segment cascades setup/release to inner segments and unions
    input specs automatically.
    """
    if not inner:
        raise ValueError("compose_sequential: at least one inner segment required")
    return _ComposedSegment(
        inner=inner,
        extra_input_specs=extra_input_specs,
        output_specs=output_specs,
        forward_fn=forward_fn,
    )


# ---------------------------------------------------------------------------
# compose_layer_from_segments: attn + moe -> layer
# ---------------------------------------------------------------------------

def compose_layer_from_segments(
    *,
    attn_segment: CapturableSegment,
    moe_segment: Optional[CapturableSegment],
    glue: Callable[..., Dict[str, torch.Tensor]],
    extra_input_specs: Optional[Mapping[str, TensorSpec]] = None,
    output_specs: Mapping[str, TensorSpec],
) -> CapturableSegment:
    """Compose an attention segment with an optional MoE segment.

    Args:
        attn_segment: per-layer attention `CapturableSegment`. Output dict
            must include the keys `glue` reads (typically `attn_output`,
            `primary_k_tensor`, optionally `indexer_k_tensor`).
        moe_segment: per-layer MoE `CapturableSegment`, or None for dense MLP
            layers (caller's glue then provides the MLP forward inline).
        glue: callable invoked as
            `glue(attn=attn_segment, moe=moe_segment, **inputs)
            -> Dict[str, Tensor]`. Implements residual + post-attn norm +
            (optional MoE) + final residual.
        extra_input_specs: extra inputs the glue needs beyond what
            `attn_segment` / `moe_segment` declare (e.g. `hidden_states`).
        output_specs: outputs the layer segment produces (typically
            `hidden_states`, `primary_k_tensor`, optionally
            `indexer_k_tensor`).

    Returns a `CapturableSegment` ready to register with `CUDAGraphManager`
    or to feed into `compose_whole_model`.
    """
    inner: List[CapturableSegment] = [attn_segment]
    if moe_segment is not None:
        inner.append(moe_segment)

    def _forward(inner_segs: List[CapturableSegment], **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        kwargs = dict(inputs)
        kwargs["attn"] = inner_segs[0]
        kwargs["moe"] = inner_segs[1] if len(inner_segs) > 1 else None
        return glue(**kwargs)

    return compose_sequential(
        inner=inner,
        extra_input_specs=extra_input_specs or {},
        output_specs=output_specs,
        forward_fn=_forward,
    )


# ---------------------------------------------------------------------------
# allocate_kv_staging: contiguous primary/aux KV staging tensors
# ---------------------------------------------------------------------------

def allocate_kv_staging(
    *,
    num_layers: int,
    max_bucket: int,
    kv_staging_dim: Mapping[str, int],
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Allocate one contiguous KV staging tensor per stream.

    The contract (§C constraint #2 + §A finding #6) requires a single
    contiguous `[num_layers, max_bucket, 1, 1, kv_dim]` tensor per kv stream
    so the worker can clone all layers once per step instead of per-layer.

    Args:
        kv_staging_dim: name -> kv-dim, e.g. {"primary": 576, "aux": 128}.

    Returns mapping `stream_name -> tensor` with shape
    `[num_layers, max_bucket, 1, 1, kv_dim]`.
    """
    if max_bucket <= 0 or num_layers <= 0:
        raise ValueError(
            f"allocate_kv_staging: num_layers={num_layers}, max_bucket={max_bucket}"
        )
    out: Dict[str, torch.Tensor] = {}
    for name, kv_dim in kv_staging_dim.items():
        if kv_dim <= 0:
            raise ValueError(f"allocate_kv_staging: {name} kv_dim={kv_dim} must be > 0")
        out[name] = torch.zeros(
            (num_layers, max_bucket, 1, 1, kv_dim), dtype=dtype, device=device,
        )
    return out


# ---------------------------------------------------------------------------
# compose_whole_model: layer segments + embed + norm + lm_head -> whole model
# ---------------------------------------------------------------------------

def compose_whole_model(
    *,
    layer_segments: List[CapturableSegment],
    embed: torch.nn.Module,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    kv_staging_dim: Mapping[str, int],
    kv_staging_dtype: torch.dtype,
    max_bucket: int,
    device: torch.device,
    extra_input_specs: Mapping[str, TensorSpec],
    output_specs: Mapping[str, TensorSpec],
    forward_fn: Callable[..., Dict[str, torch.Tensor]],
) -> CapturableSegment:
    """Compose layer segments + embed + final_norm + lm_head into one segment.

    The composed segment allocates contiguous KV staging buffers via
    `allocate_kv_staging` (one per `kv_staging_dim` entry). The worker
    post-replay clones the staging tensors ONCE per kv stream and dispatches
    `AttnWrapperBase.kv_append_callback` per layer index — see contract §C
    constraint #2.

    Args:
        layer_segments: list of per-layer `CapturableSegment`s, in order.
        embed, final_norm, lm_head: real modules called at the segment
            boundaries; not captured as separate segments because they have
            no per-layer KV interaction.
        kv_staging_dim: per-stream KV dim (e.g. {"primary": 576} for MLA, or
            {"primary": kv_dim, "aux": aux_dim} for hybrid attention).
        forward_fn: `(layer_segments, embed, final_norm, lm_head, kv_staging,
            **inputs) -> Dict[str, Tensor]`. Implements embed -> layers ->
            final_norm -> lm_head.

    Returns a `CapturableSegment` ready to register as the whole-model graph.
    """
    if not layer_segments:
        raise ValueError("compose_whole_model: at least one layer segment required")

    kv_staging = allocate_kv_staging(
        num_layers=len(layer_segments),
        max_bucket=max_bucket,
        kv_staging_dim=kv_staging_dim,
        dtype=kv_staging_dtype,
        device=device,
    )

    def _forward(inner_segs: List[CapturableSegment], **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return forward_fn(
            layer_segments=inner_segs,
            embed=embed,
            final_norm=final_norm,
            lm_head=lm_head,
            kv_staging=kv_staging,
            **inputs,
        )

    composed = compose_sequential(
        inner=layer_segments,
        extra_input_specs=extra_input_specs,
        output_specs=output_specs,
        forward_fn=_forward,
    )
    # Attach KV staging so adapters can read it from stage_post_graph_kv.
    composed.kv_staging = kv_staging  # type: ignore[attr-defined]
    return composed


__all__ = [
    "CaptureContext",
    "allocate_kv_staging",
    "compose_sequential",
    "compose_layer_from_segments",
    "compose_whole_model",
]
