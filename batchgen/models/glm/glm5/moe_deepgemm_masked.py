"""DeepGEMM masked m-grouped routed GEMMs for the DeepEP decode MoE path.

Replaces, on the DeepEP low-latency decode path only, our persistent CuTe
S1/S3 pair with DeepGEMM's masked m-grouped FP8 kernels — the same kernels the
reference FP8 MoE runner uses behind a low-latency dispatch. The prefill /
ragged path keeps the CuTe kernels; this module is reached only from
``Glm5MoEGraphSegment._forward_deepep``.

Why
---
Our S1/S3 schedule a whole 16-row tile per 16 rows an expert received, so a
layer's cost grows with the number of occupied tiles and the slowest rank's
routing skew sets the combine's waiting time. DeepGEMM's masked kernel works on
a fixed 64-row block per expert, so the same layer costs the same up to 64
rows/expert and the skew stops being amplified.

Chain, per routed layer
-----------------------
    R1_gate : masked( (x, x_scales), (W_gate, S_gate) ) -> gate   [E, S, I]
    R1_up   : masked( (x, x_scales), (W_up,   S_up)   ) -> up     [E, S, I]
    act     : quant(silu(gate) * up)                   -> (a, a_scales)
    R2      : masked( (a, a_scales), (W_down, S_down) ) -> out    [E, S, H]

The reference runner issues ONE R1 over a per-expert-fused ``[E, 2I, H]``
gate/up weight. We keep the two projections separate because our routed-expert
stacks are per-projection ``[E, N, K]`` blocks whose expert stride the CuTe
kernels hardcode as ``N * K``: handing those kernels halves of a fused
``[E, 2I, H]`` buffer would silently stride them wrong, and holding a fused
copy ON TOP of the per-projection stacks would double the routed-expert
footprint. Two masked launches read the same weight bytes and the same number
of output tiles as one fused launch; the cost is one extra launch per routed
layer, against the two launches (descriptor prep + bounded scale repack) this
path drops. A fused ``[E, 2I, H]`` load-time layout is a follow-up that has to
move the preallocated expert stacks with it.

Layout contract
---------------
``fp8_m_grouped_gemm_nt_masked`` takes, per group:

    A  [E, S, K]  float8_e4m3fn, row-major
    SA [E, S, K // 128] f32, MN-MAJOR: ``stride(-2) == 1`` and the MN extent
       padded to 4 floats (16 B) for the TMA load
    B  [E, N, K]  float8_e4m3fn, row-major  (the "nt" of the name)
    SB [E, N // 128, K // 128] f32, row-major
    D  [E, S, N]  bfloat16, row-major

Both A-scale operands already satisfy the MN-major contract without a repack,
which is why this path has no scale-preparation launch at all:

* the dispatch's ``recv_scales`` is a ``(1, 2)`` transposed view of a
  contiguous ``[E, K // 128, S]`` block, i.e. shape ``[E, S, K // 128]`` with
  strides ``(K // 128 * S, 1, S)``;
* the activation scales are written straight into the same form by
  :func:`~batchgen_kernels.triton.moe_masked_act_quant.silu_mul_masked_group_quant`.

:func:`require_mn_major_tma_aligned` asserts exactly that form rather than
calling DeepGEMM's ``get_mn_major_tma_aligned_tensor``, which would be a no-op
on a conforming tensor and a silent per-call transpose (and allocation, inside
a capture) on any other.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FP8_DTYPE = torch.float8_e4m3fn
SCALE_BLOCK = 128
# The masked kernel's m-block. An expert's row extent is padded to the receive
# stride, so the stride must be a multiple of this for every block a group
# schedules to lie inside that group's slab.
MASKED_BLOCK_M = 64
# DeepGEMM pads the MN extent of an A-scale tensor to this many floats.
SCALE_TMA_ALIGN_FLOATS = 4


def masked_gemm_fn():
    """DeepGEMM's masked m-grouped FP8 GEMM, or a named error.

    Resolved per call rather than cached: the resolution only ever runs on the
    host, during warmup and capture, never on a graph replay.
    """
    try:
        import deep_gemm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the DeepEP decode MoE path runs DeepGEMM's masked m-grouped FP8 "
            "GEMM, but deep_gemm is not importable in this runtime"
        ) from exc
    fn = getattr(deep_gemm, "fp8_m_grouped_gemm_nt_masked", None)
    if fn is None:
        raise RuntimeError(
            "the installed deep_gemm has no fp8_m_grouped_gemm_nt_masked; the "
            "DeepEP decode MoE path needs the masked m-grouped FP8 GEMM "
            f"(module exports: {sorted(n for n in dir(deep_gemm) if 'masked' in n)})"
        )
    return fn


def masked_act_quant_fn():
    """The masked ``quant(silu(gate) * up)`` kernel.

    Resolved at call time for the same reason as the GEMM, and because
    importing ``batchgen_kernels.triton`` pulls in Triton, which the NCCL
    decode path must not pay for at module import.
    """
    from batchgen_kernels.triton.moe_masked_act_quant import (
        silu_mul_masked_group_quant,
    )

    return silu_mul_masked_group_quant


def expected_m_per_expert(
    *,
    local_rows: int,
    world_size: int,
    topk: int,
    num_experts_global: int,
) -> int:
    """The masked kernel's row-count hint: mean rows per GLOBAL expert.

    ``local_rows * world_size * topk`` routed (token, expert) pairs spread over
    ``num_experts_global`` experts, rounded up. Identical to the reference
    dispatcher's formula, and a function of the captured bucket only, so it is
    a host constant inside a capture.
    """
    if min(local_rows, world_size, topk, num_experts_global) <= 0:
        raise ValueError(
            "expected_m needs positive local_rows / world_size / topk / "
            f"num_experts_global; got {local_rows} / {world_size} / {topk} / "
            f"{num_experts_global}"
        )
    pairs = int(local_rows) * int(world_size) * int(topk)
    return (pairs + int(num_experts_global)) // int(num_experts_global)


def is_mn_major_tma_aligned(scales: torch.Tensor) -> bool:
    """True when ``scales`` is already in the masked GEMM's A-scale form."""
    if scales.dim() != 3 or scales.dtype != torch.float32:
        return False
    groups, rows, blocks = scales.shape
    return (
        rows % SCALE_TMA_ALIGN_FLOATS == 0
        and scales.stride() == (blocks * rows, 1, rows)
    )


def require_mn_major_tma_aligned(scales: torch.Tensor, name: str) -> None:
    """Fail loudly unless ``scales`` needs no preparation launch."""
    if scales.dim() != 3:
        raise RuntimeError(
            f"{name}: the masked GEMM takes logical [E, S, K // {SCALE_BLOCK}] "
            f"A-scales; got {tuple(scales.shape)}"
        )
    if scales.dtype != torch.float32:
        raise RuntimeError(
            f"{name}: the masked GEMM takes f32 A-scales; got {scales.dtype}"
        )
    groups, rows, blocks = scales.shape
    if rows % SCALE_TMA_ALIGN_FLOATS != 0:
        raise RuntimeError(
            f"{name}: the masked GEMM's A-scale TMA load needs the row extent "
            f"{rows} padded to {SCALE_TMA_ALIGN_FLOATS} floats"
        )
    want = (blocks * rows, 1, rows)
    if scales.stride() != want:
        raise RuntimeError(
            f"{name}: the masked GEMM needs MN-major A-scales, i.e. a "
            f"[{groups}, {blocks}, {rows}] contiguous block viewed as "
            f"[{groups}, {rows}, {blocks}] (strides {want}); got strides "
            f"{tuple(scales.stride())}. Repacking it here would allocate "
            "inside the capture."
        )


@dataclass(frozen=True)
class DeepGemmMaskedExpertWeights:
    """One routed layer's weights in the masked GEMM's ``nt`` layout.

    ``gate_w`` / ``up_w`` are ``[E, I, H]`` and ``down_w`` is ``[E, H, I]``:
    both are already ``[E, N, K]`` for their own GEMM, so the routed-expert
    stacks are handed over as they are loaded — no conversion, no second
    resident copy.
    """

    gate_w: torch.Tensor
    gate_scale: torch.Tensor
    up_w: torch.Tensor
    up_scale: torch.Tensor
    down_w: torch.Tensor
    down_scale: torch.Tensor

    def validate(
        self, *, num_experts: int, hidden_size: int, intermediate_size: int
    ) -> None:
        e, h, i = int(num_experts), int(hidden_size), int(intermediate_size)
        if h % SCALE_BLOCK or i % SCALE_BLOCK:
            raise RuntimeError(
                "the masked GEMM is block-scaled: hidden "
                f"{h} and intermediate {i} must both be multiples of "
                f"{SCALE_BLOCK}"
            )
        for name, weight, shape in (
            ("gate_w", self.gate_w, (e, i, h)),
            ("up_w", self.up_w, (e, i, h)),
            ("down_w", self.down_w, (e, h, i)),
        ):
            if tuple(weight.shape) != shape:
                raise RuntimeError(
                    f"{name}: the masked GEMM takes [E, N, K] routed weights "
                    f"{shape}; got {tuple(weight.shape)}"
                )
            if weight.dtype != FP8_DTYPE:
                raise RuntimeError(
                    f"{name}: the masked GEMM takes {FP8_DTYPE} weights; got "
                    f"{weight.dtype}"
                )
            if not weight.is_contiguous():
                raise RuntimeError(
                    f"{name}: the masked GEMM takes a contiguous [E, N, K] "
                    f"block; got strides {tuple(weight.stride())}"
                )
        for name, scale, shape in (
            ("gate_scale", self.gate_scale, (e, i // SCALE_BLOCK, h // SCALE_BLOCK)),
            ("up_scale", self.up_scale, (e, i // SCALE_BLOCK, h // SCALE_BLOCK)),
            ("down_scale", self.down_scale, (e, h // SCALE_BLOCK, i // SCALE_BLOCK)),
        ):
            if tuple(scale.shape) != shape:
                raise RuntimeError(
                    f"{name}: the masked GEMM takes [E, N // {SCALE_BLOCK}, "
                    f"K // {SCALE_BLOCK}] weight scales {shape}; got "
                    f"{tuple(scale.shape)}. Our stacks pad the trailing block "
                    "axis to 4 for the CuTe kernels, which the masked GEMM "
                    "cannot read — the padded axis must already be exact."
                )
            if scale.dtype != torch.float32:
                raise RuntimeError(
                    f"{name}: the masked GEMM takes f32 weight scales; got "
                    f"{scale.dtype}"
                )
            if not scale.is_contiguous():
                raise RuntimeError(
                    f"{name}: the masked GEMM takes contiguous weight scales; "
                    f"got strides {tuple(scale.stride())}"
                )


def run_deepgemm_masked_moe(
    *,
    x_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    weights: DeepGemmMaskedExpertWeights,
    masked_m: torch.Tensor,
    expected_m: int,
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
    act_fp8: torch.Tensor,
    act_scales: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run one routed layer's masked chain into caller-owned buffers.

    Allocates nothing, synchronizes nothing, and reads its per-step row counts
    from ``masked_m`` on the device, so the whole chain captures. ``masked_m``
    is the dispatch's ``recv_count``; ``expected_m`` is a host constant from
    :func:`expected_m_per_expert`.
    """
    if not isinstance(expected_m, int):
        raise RuntimeError(
            "expected_m must be a host int (it is a captured constant); got "
            f"{type(expected_m).__name__}"
        )
    require_mn_major_tma_aligned(x_scales, "dispatch activation scales")
    require_mn_major_tma_aligned(act_scales, "down-projection input scales")
    masked_gemm = masked_gemm_fn()
    act_quant = masked_act_quant_fn()

    masked_gemm(
        (x_fp8, x_scales),
        (weights.gate_w, weights.gate_scale),
        gate_out,
        masked_m,
        expected_m,
    )
    masked_gemm(
        (x_fp8, x_scales),
        (weights.up_w, weights.up_scale),
        up_out,
        masked_m,
        expected_m,
    )

    act_quant(
        gate_out, up_out, act_fp8, act_scales, masked_m, group_size=SCALE_BLOCK
    )

    masked_gemm(
        (act_fp8, act_scales),
        (weights.down_w, weights.down_scale),
        out,
        masked_m,
        expected_m,
    )
    return out
