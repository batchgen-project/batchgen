"""Fused INT4 grouped + SiLU (SwiGLU) Triton kernels for SM100 (Blackwell).

Pure-Triton port of the Hopper WGMMA INT4 MoE expert path (K2.5 decode). The
SM90a stage-1 kernel computes `silu(gate) * up` and stage-2 the down
projection. On sm_100a those `.cu` kernels are not built, so we compose the
sub-task-7 INT4 GEMM building block with a small `silu_mul` epilogue and a
second INT4 GEMM for the down projection.

SwiGLU convention (matches `single_expert_int4_wgmma.cu` epilogue):
    out = (gate * sigmoid(gate)) * up        # silu(gate) * up, computed in FP32
"""

import torch
import triton
import triton.language as tl

from batchgen_kernels.triton.int4_grouped_gemm import int4_grouped_gemm


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up   # silu(gate) * up
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Elementwise SwiGLU activation: silu(gate) * up (FP32 math, BF16 out)."""
    assert gate.shape == up.shape, f"shape mismatch {gate.shape} vs {up.shape}"
    out = torch.empty_like(gate, dtype=torch.bfloat16)
    n = gate.numel()
    if n == 0:
        return out
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _silu_mul_kernel[grid](gate.contiguous(), up.contiguous(), out, n, BLOCK=BLOCK)
    return out


def fused_int4_grouped_silu(
    x: torch.Tensor,
    wg_packed: torch.Tensor, wg_scales: torch.Tensor,
    wu_packed: torch.Tensor, wu_scales: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """silu(x @ dequant(Wg).T) * (x @ dequant(Wu).T) with INT4 weights.

    Returns the stage-1 SwiGLU activation [M, N_intermediate] BF16.
    """
    gate = int4_grouped_gemm(x, wg_packed, wg_scales, group_size)
    up = int4_grouped_gemm(x, wu_packed, wu_scales, group_size)
    return silu_mul(gate, up)


def int4_expert_mlp(
    x: torch.Tensor,                       # [M, K] BF16
    gate_packed: torch.Tensor, gate_scale: torch.Tensor,
    up_packed: torch.Tensor, up_scale: torch.Tensor,
    down_packed: torch.Tensor, down_scale: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """Full INT4 expert MLP: stage1 (gate+up+SiLU) + stage2 (down).

    Pure-Triton equivalent of `single_expert_int4_forward`. Returns [M, K] BF16.
    """
    intermediate = fused_int4_grouped_silu(
        x, gate_packed, gate_scale, up_packed, up_scale, group_size,
    )
    return int4_grouped_gemm(intermediate, down_packed, down_scale, group_size)


def int4_grouped_moe_forward(
    hidden_states: torch.Tensor,       # [num_tokens, K] BF16
    topk_indices: torch.Tensor,        # [num_tokens, topk] int
    topk_weights: torch.Tensor,        # [num_tokens, topk] float
    expert_indices,                    # iterable of global expert idx to process
    gate_packed, gate_scale,           # List[Tensor] indexed by global expert idx
    up_packed, up_scale,
    down_packed, down_scale,
    group_size: int = 32,
) -> torch.Tensor:
    """SM100 INT4 grouped MoE forward (correctness-first per-expert masked loop).

    Mirrors the MXFP4 sm100 path: each routed expert runs the full INT4 MLP over
    its masked tokens, accumulated in FP32 with slot-specific routing weights.

    Returns:
        Output [num_tokens, K] BF16 (routing-weighted sum of expert outputs).
    """
    num_tokens, K = hidden_states.shape
    output = torch.zeros(num_tokens, K, dtype=torch.float32, device=hidden_states.device)

    active_experts = set(topk_indices.flatten().tolist())
    for e in expert_indices:
        if e not in active_experts:
            continue
        mask = (topk_indices == e).any(dim=-1)
        x_e = hidden_states[mask].contiguous()

        out_e = int4_expert_mlp(
            x_e,
            gate_packed[e], gate_scale[e],
            up_packed[e], up_scale[e],
            down_packed[e], down_scale[e],
            group_size,
        )

        sel_idx = topk_indices[mask]
        sel_w = topk_weights[mask]
        w_e = torch.where(
            sel_idx == e, sel_w, torch.zeros_like(sel_w)
        ).sum(dim=-1).float()
        output[mask] += out_e.float() * w_e.unsqueeze(-1)

    return output.to(hidden_states.dtype)
