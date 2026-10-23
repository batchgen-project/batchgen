"""BF16 strided-batched MLA absorb for GLM-5 decode, and container dispatch.

Slice 2c-2. Deliberately a LEAF module with no Triton import and no
intra-package import at module scope, so the layout and contract can be tested
on a CPU-only dev machine; the FP8/Triton path is imported lazily, only when an
``FP8AbsorbWeights`` container is actually passed. ``fp8_absorb.py`` is not
modified by this slice at all.

Why bf16 ``torch.bmm`` replaces the Triton FP8 WGMMA absorb
-----------------------------------------------------------
Measured 2026-10-23 under CUDA-graph replay, B=64, H=64, both production
shapes, four arms in one probe:

    arm                                  q 192->512   out 512->256   calc_diff
    Triton fp8 (what this replaces)        14.048 us     17.952 us    ~6.6e-04
    bmm with .contiguous() permutes        17.952 us     21.072 us     0.00e+00
    bmm on transposed VIEWS  <-- this      10.912 us     14.528 us     0.00e+00
    bmm, pre-staged [H,B,*] (a floor)      10.752 us     14.304 us     0.00e+00

Expected production effect **-0.372 ms/step**, obtained by scaling these
ratios onto the 2026-10-17 A/B harness baseline (23.21 us/layer for q+out =
1.810 ms/step). Two earlier figures for this slice, 0.678 and 0.512 ms/step,
are superseded and must not be requoted: the first came from reading a
permute-free *floor* as achievable, the second from this probe's absolute
numbers, which run hot because it builds fresh weights per shape with no
cold-weight pooling.

It is also numerically EXACT (``calc_diff`` 0 against a bf16 reference, versus
~6.6e-04 for the fp8 kernel), so it removes an fp8 quantisation error source
from the attention critical path. That is an independent benefit, not a
rounding detail.

Cost: bf16 absorb weights are twice the fp8 ones. Once the fp8 copies stop
being built, the net change is **+14.68 MB/layer = +1.15 GB/GPU**, which
competes with the KV cache. That trade is the slice's one open question and is
settled by the paired cold M1 plus the HBM/KV-pool readings, not here.

The layout IS the win
---------------------
Note the second row above: a bf16 bmm that pays ``.contiguous()`` permutes is
**1.17-1.28x slower than the Triton kernel it would replace**. The entire gain
comes from not permuting, which needs two things:

* the weights pre-stored ONCE at load as ``[H, K, N]`` (free, done here);
* the activation and the output passed as transposed VIEWS at runtime.

A transposed view is already a valid strided-batched GEMM operand:
``x[B,H,K].transpose(0,1)`` is ``[H,B,K]`` with strides ``(K, H*K, 1)`` -- a
constant batch stride and ``lda = H*K`` -- which is exactly what cuBLAS
strided-batched GEMM takes, and the result can be written straight through
``out[B,H,N].transpose(0,1)``. Verified in the same probe: cuBLASLt keeps the
same ``nvjet_tst_*`` kernel for the views and emits **no** extra copy, while
the ``.contiguous()`` arm shows an additional ``elementwise_kernel``.
"""

from __future__ import annotations

import torch

__all__ = [
    "BF16AbsorbWeights",
    "absorb_out",
    "absorb_out_out",
    "absorb_q",
    "absorb_q_out",
]


class BF16AbsorbWeights:
    """BF16 absorb weights in the strided-batched-GEMM layout, ``[H, K, N]``.

    Both GEMMs want ``[H, K, N]``, prepared once at load:

    * ``q_absorb``   arrives ``[H, qk_nope, kv_lora]`` -- already ``[H, K, N]``,
      so only ``.contiguous()``;
    * ``out_absorb`` arrives ``[H, v_head, kv_lora]`` -- i.e. ``[H, N, K]``, so
      it needs ``transpose(1, 2)``.

    On ``num_valid_tokens``: this path accepts and IGNORES it
    --------------------------------------------------------
    The Triton kernel masks rows ``>= num_valid_tokens`` to zero. Its own
    comment gives the motive, and it is **not** that any consumer needs zeros:

        "Keep zero-valid tiles on the same straight-line path: Triton dynamic
        early-return poisoned CUDA graph capture on zero-local-rank buckets."

    The mask exists so a Triton program never takes a dynamic early return
    inside a captured graph. **A bmm has no early return and no dynamic branch**,
    so the motive does not transfer. **Anyone reintroducing a Triton path here,
    or adding any dynamic branch to this one, must restore that masking.** That
    sentence is the reason this paragraph exists.

    Ignoring it is safe on the consuming paths, established at source level
    2026-10-23:

    1. ``models/glm/glm5/reuse_topk_segment.py`` (~:532) already documents the
       invariant that masked-GEMM outputs carry stale rows and are left
       unmasked, because "every consumer of a masked-GEMM output is row-local
       (fused_rmsnorm, act_quant, RoPE, the absorb GEMVs, FA3)". The absorb is
       named there.
    2. ``act_quant`` (``attention/mla/fa3_backend.act_quant_kernel_2d``) breaks
       the chain unconditionally: scales are per ``(row, 128-block)`` with no
       cross-row reduction, and for ``pid_m >= num_valid_tokens`` it stores
       zeros plus an ``eps`` scale and returns **without ever loading the row**.
       Whatever the absorb leaves in an invalid row is never read.
    3. The only cross-row consumer is the MoE router (gate top-k and ragged
       dispatch do mix rows) and it is protected by the one load-bearing mask,
       ``attn_output.mul_(valid_rows_bf16)``, whose behaviour does not depend on
       the absorb.
    4. ``models/glm/glm5/cuda_graph_segments.py`` never passes
       ``num_valid_tokens`` to the absorbs at all.

    PRE-EXISTING, NOT INTRODUCED HERE: the graph buffers are ``torch.empty``,
    so a first capture can hold NaN in invalid rows, and the load-bearing mask
    is a multiply (``NaN * 0 == NaN``). That route runs through
    ``w8a8_deepgemm``, which leaves invalid rows unwritten regardless of the
    absorb, so this change neither creates nor widens it. Recorded as a
    standing suspect if a precision gate ever reports an unexplained NaN.
    """

    def __init__(
        self,
        q_absorb_bf16: torch.Tensor,     # [H, qk_nope, kv_lora]
        out_absorb_bf16: torch.Tensor,   # [H, v_head,  kv_lora]
    ):
        if q_absorb_bf16.dim() != 3 or out_absorb_bf16.dim() != 3:
            raise ValueError(
                "absorb weights must be 3D [H, *, kv_lora]; got "
                f"{tuple(q_absorb_bf16.shape)} and {tuple(out_absorb_bf16.shape)}"
            )
        if q_absorb_bf16.shape[0] != out_absorb_bf16.shape[0]:
            raise ValueError(
                f"head counts differ: q {q_absorb_bf16.shape[0]} vs "
                f"out {out_absorb_bf16.shape[0]}"
            )
        if q_absorb_bf16.shape[2] != out_absorb_bf16.shape[2]:
            raise ValueError(
                "both absorb weights must share kv_lora_rank; got "
                f"{q_absorb_bf16.shape[2]} and {out_absorb_bf16.shape[2]}"
            )
        # q is already [H, K, N]; out is [H, N, K] and is flipped ONCE here.
        self.q_absorb_knn = q_absorb_bf16.to(torch.bfloat16).contiguous()
        self.out_absorb_knn = (
            out_absorb_bf16.to(torch.bfloat16).transpose(1, 2).contiguous()
        )
        # One-time layout assertions. The entire gain depends on these holding,
        # and a silent layout drift would surface only as lost performance --
        # the hardest kind of regression to notice.
        H, _, Nq = self.q_absorb_knn.shape
        Ho, Ko, _ = self.out_absorb_knn.shape
        if (Ho, Ko) != (H, Nq):
            raise RuntimeError(
                "out_absorb must be [H, kv_lora, v_head] after the flip; got "
                f"{tuple(self.out_absorb_knn.shape)} against q "
                f"{tuple(self.q_absorb_knn.shape)}"
            )
        for name, t in (("q_absorb_knn", self.q_absorb_knn),
                        ("out_absorb_knn", self.out_absorb_knn)):
            if not t.is_contiguous():
                raise RuntimeError(f"{name} must be a contiguous [H, K, N]")
            if t.dtype != torch.bfloat16:
                raise RuntimeError(f"{name} must be bfloat16, got {t.dtype}")

    def nbytes(self) -> int:
        """Resident bytes. The slice's HBM cost is reported from this."""
        return self.q_absorb_knn.nbytes + self.out_absorb_knn.nbytes


def _bmm_absorb_out(x_bhk: torch.Tensor, w_knn: torch.Tensor,
                    out_bhn: torch.Tensor) -> torch.Tensor:
    """``out[B,H,N] = x[B,H,K] @ w[H,K,N]`` with NO permute on either side.

    Both operands go to ``torch.bmm`` as transposed views and the result is
    written through a transposed view of ``out``. See the module docstring for
    why that is the point of the slice rather than an implementation detail.
    """
    if x_bhk.dim() != 3 or out_bhn.dim() != 3:
        raise ValueError(
            f"absorb bmm needs 3D [B,H,*]; got x {tuple(x_bhk.shape)} "
            f"out {tuple(out_bhn.shape)}"
        )
    B, H, K = x_bhk.shape
    if w_knn.shape[0] != H or w_knn.shape[1] != K:
        raise ValueError(
            f"weight {tuple(w_knn.shape)} does not match x {tuple(x_bhk.shape)}; "
            "expected [H, K, N]"
        )
    if out_bhn.shape[:2] != (B, H) or out_bhn.shape[2] != w_knn.shape[2]:
        raise ValueError(
            f"out {tuple(out_bhn.shape)} does not match "
            f"[{B}, {H}, {w_knn.shape[2]}]"
        )
    torch.bmm(x_bhk.transpose(0, 1), w_knn, out=out_bhn.transpose(0, 1))
    return out_bhn


def _is_fp8_container(weights) -> bool:
    """True for ``FP8AbsorbWeights``, imported lazily to stay Triton-free."""
    try:
        from batchgen_kernels.attention.dsa.fp8_absorb import FP8AbsorbWeights
    except Exception:
        return False
    return isinstance(weights, FP8AbsorbWeights)


def _reject(fn_name: str, weights) -> None:
    raise TypeError(
        f"{fn_name} needs BF16AbsorbWeights or FP8AbsorbWeights, got "
        f"{type(weights).__name__}"
    )


def _squeeze4d(x, out=None):
    """Callers pass either ``[B,H,C]`` or ``[B,1,H,C]``; normalise to 3D."""
    if x.dim() != 4:
        return x, out, False
    x = x.squeeze(1)
    if out is not None:
        if out.dim() != 4:
            raise ValueError("out must be 4D when the input is 4D")
        out = out.squeeze(1)
    return x, out, True


def absorb_q(x, weights):
    """q-absorb, dispatched on the container type."""
    if isinstance(weights, BF16AbsorbWeights):
        out = torch.empty(
            x.shape[0], x.shape[1], weights.q_absorb_knn.shape[2],
            dtype=torch.bfloat16, device=x.device,
        )
        return _bmm_absorb_out(x, weights.q_absorb_knn, out)
    if _is_fp8_container(weights):
        from batchgen_kernels.attention.dsa.fp8_absorb import fp8_q_absorb
        return fp8_q_absorb(x, weights)
    _reject("absorb_q", weights)


def absorb_q_out(x, weights, out, num_valid_tokens=None):
    """Out-buffer q-absorb for CUDA-graph capture."""
    if isinstance(weights, BF16AbsorbWeights):
        # num_valid_tokens deliberately unused -- see BF16AbsorbWeights.
        return _bmm_absorb_out(x, weights.q_absorb_knn, out)
    if _is_fp8_container(weights):
        from batchgen_kernels.attention.dsa.fp8_absorb import fp8_q_absorb_out
        return fp8_q_absorb_out(x, weights, out, num_valid_tokens=num_valid_tokens)
    _reject("absorb_q_out", weights)


def absorb_out(x, weights):
    """out-absorb, dispatched on the container type."""
    if isinstance(weights, BF16AbsorbWeights):
        x3, _, squeezed = _squeeze4d(x)
        out = torch.empty(
            x3.shape[0], x3.shape[1], weights.out_absorb_knn.shape[2],
            dtype=torch.bfloat16, device=x3.device,
        )
        _bmm_absorb_out(x3, weights.out_absorb_knn, out)
        return out.unsqueeze(1) if squeezed else out
    if _is_fp8_container(weights):
        from batchgen_kernels.attention.dsa.fp8_absorb import fp8_out_absorb
        return fp8_out_absorb(x, weights)
    _reject("absorb_out", weights)


def absorb_out_out(x, weights, out, num_valid_tokens=None):
    """Out-buffer out-absorb for CUDA-graph capture."""
    if isinstance(weights, BF16AbsorbWeights):
        x3, out3, squeezed = _squeeze4d(x, out)
        # num_valid_tokens deliberately unused -- see BF16AbsorbWeights.
        _bmm_absorb_out(x3, weights.out_absorb_knn, out3)
        return out if squeezed else out3
    if _is_fp8_container(weights):
        from batchgen_kernels.attention.dsa.fp8_absorb import fp8_out_absorb_out
        return fp8_out_absorb_out(x, weights, out,
                                  num_valid_tokens=num_valid_tokens)
    _reject("absorb_out_out", weights)
