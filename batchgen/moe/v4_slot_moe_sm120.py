"""Slot-based grouped MXFP4 MoE for DeepSeek-V4-Flash decode on Blackwell sm120.

Replaces the per-expert Python loop (`DeepSeekV4FlashMoE._run_owned_experts`) with two
fused FP4-dequant+GEMV Triton kernels over a fixed (token, expert) slot grid. No per-expert
`.item()`/`torch.where` syncs and no per-token full-weight re-dequant.

Adapted from SGLang's sm120 MXFP4 MoE kernel (commit 578f232e,
python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py). V4-specific
deltas vs that reference:
  - Expert-parallel owned range: topk indices are GLOBAL [0, total_experts); the stacked
    weight buffers hold only this rank's owned experts. Slots outside the owned range are
    masked to zero (mirrors `_run_owned_experts` which only runs owned experts and relies
    on a later all_reduce to combine ranks).
  - V4 activation is silu(gate)*up with optional clamp to swiglu_limit (model.py expert
    forward), NOT OpenAI-style GLU.
  - Routing weight is applied to the down-projection output then summed over topk. This is
    algebraically identical to V4 applying it to the activated intermediate (w2 is linear).

The FP4 E2M1 decode is bitwise-identical to model.py `_dequant_fp4_e2m1_weight`
(verified by .sisyphus/blackwell/test_v4_stack_dequant.py).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_fp4_e2m1(nibble):
    sign_bit = (nibble >> 3) & 1
    exp_bits = (nibble >> 1) & 3
    man_bit = nibble & 1
    is_subnormal = exp_bits == 0
    mantissa = 1.0 + man_bit.to(tl.float32) * 0.5
    exponent = tl.math.exp2((exp_bits - 1).to(tl.float32))
    val = tl.where(
        is_subnormal, man_bit.to(tl.float32) * 0.5, mantissa * exponent
    )
    val = tl.where(sign_bit != 0, -val, val)
    return val


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def _slot_gemv_kernel(
    A_ptr,
    B_packed_ptr,
    B_scale_ptr,
    C_ptr,
    token_ids_ptr,
    expert_ids_ptr,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,
    stride_bn: tl.int32,
    stride_bk2: tl.int32,
    stride_bsn: tl.int32,
    stride_bsk32: tl.int32,
    expert_b_stride: tl.int64,
    expert_s_stride: tl.int64,
    stride_cm: tl.int32,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    slot_id = tl.program_id(0)
    n_block = tl.program_id(1)

    token_id = tl.load(token_ids_ptr + slot_id).to(tl.int64)
    expert_id = tl.load(expert_ids_ptr + slot_id).to(tl.int64)

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    b_base = expert_id * expert_b_stride
    s_base = expert_id * expert_s_stride
    a_base = token_id * stride_am

    for k_start in range(0, K, BLOCK_K):
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        b_mask = n_mask[:, None] & (offs_k2[None, :] < K // 2)
        b_packed = tl.load(
            B_packed_ptr
            + b_base
            + offs_n[:, None] * stride_bn
            + offs_k2[None, :] * stride_bk2,
            mask=b_mask,
            other=0,
        )
        b_u8 = b_packed.to(tl.int32)
        val_lo = _dequant_fp4_e2m1(b_u8 & 0x0F)
        val_hi = _dequant_fp4_e2m1((b_u8 >> 4) & 0x0F)

        group_ids = tl.arange(0, BLOCK_K // 2) // 16
        s_mask = n_mask[:, None] & (
            (k_start // 32 + group_ids[None, :]) < K // 32
        )
        scales = tl.load(
            B_scale_ptr
            + s_base
            + offs_n[:, None] * stride_bsn
            + (k_start // 32 + group_ids[None, :]) * stride_bsk32,
            mask=s_mask,
            other=1.0,
        )
        val_lo = val_lo * scales
        val_hi = val_hi * scales

        offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2
        offs_k_odd = offs_k_even + 1
        a_even = tl.load(
            A_ptr + a_base + offs_k_even, mask=offs_k_even < K, other=0.0
        ).to(tl.float32)
        a_odd = tl.load(
            A_ptr + a_base + offs_k_odd, mask=offs_k_odd < K, other=0.0
        ).to(tl.float32)

        acc += tl.sum(a_even[None, :] * val_lo, axis=1)
        acc += tl.sum(a_odd[None, :] * val_hi, axis=1)

    tl.store(
        C_ptr + slot_id * stride_cm + offs_n, acc.to(tl.bfloat16), mask=n_mask
    )


def _ensure_f32_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.float32:
        return scale.to(torch.float32)
    return scale


def v4_slot_moe_forward(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Grouped MXFP4 MoE over this rank's owned experts; returns routed [G, hidden] fp32.

    Mirrors `DeepSeekV4FlashMoE._run_owned_experts`: only experts in
    [owned_start, owned_start+owned_count) contribute; all other (token, expert) slots
    contribute exactly zero so a downstream all_reduce can combine ranks.

    Args:
      token_states:  [G, hidden] bf16 input rows.
      topk_weights:  [G, topk] router weights (already include any route_scale).
      topk_indices:  [G, topk] GLOBAL expert ids.
      w13_packed:    [owned_count, 2*I, hidden//2] uint8 (gate rows then up rows).
      w13_scale:     [owned_count, 2*I, hidden//32] E8M0/float32.
      w2_packed:     [owned_count, hidden, I//2] uint8.
      w2_scale:      [owned_count, hidden, I//32] E8M0/float32.
      swiglu_limit:  clamp limit (>0 enables clamp), matching the eager expert forward.
    """
    import torch.nn.functional as F

    G, hidden = token_states.shape
    topk = topk_indices.shape[1]
    two_I = w13_packed.shape[1]
    I = two_I // 2
    num_slots = G * topk
    device = token_states.device
    dtype = token_states.dtype

    token_states = token_states.contiguous()
    w13_u8 = w13_packed.view(torch.uint8).contiguous()
    w2_u8 = w2_packed.view(torch.uint8).contiguous()
    w13_scale = _ensure_f32_scale(w13_scale).contiguous()
    w2_scale = _ensure_f32_scale(w2_scale).contiguous()

    global_eids = topk_indices.reshape(-1)
    local_eids = global_eids - owned_start
    valid = (global_eids >= owned_start) & (
        global_eids < owned_start + owned_count
    )
    local_eids = torch.where(
        valid, local_eids, torch.zeros_like(local_eids)
    ).to(torch.int32)

    token_ids = (
        torch.arange(G, device=device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(G, topk)
        .reshape(-1)
        .contiguous()
    )

    intermediate = torch.empty(num_slots, two_I, dtype=dtype, device=device)
    grid1 = lambda meta: (num_slots, triton.cdiv(two_I, meta["BLOCK_N"]))
    _slot_gemv_kernel[grid1](
        token_states,
        w13_u8,
        w13_scale,
        intermediate,
        token_ids,
        local_eids,
        two_I,
        hidden,
        token_states.stride(0),
        w13_u8.stride(1),
        w13_u8.stride(2),
        w13_scale.stride(1),
        w13_scale.stride(2),
        w13_u8.stride(0),
        w13_scale.stride(0),
        intermediate.stride(0),
    )

    gate = intermediate[:, :I].float()
    up = intermediate[:, I:].float()
    if swiglu_limit and swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    activated = (F.silu(gate) * up).to(dtype).contiguous()

    down = torch.empty(num_slots, hidden, dtype=dtype, device=device)
    slot_ids = torch.arange(num_slots, device=device, dtype=torch.int32)
    grid2 = lambda meta: (num_slots, triton.cdiv(hidden, meta["BLOCK_N"]))
    _slot_gemv_kernel[grid2](
        activated,
        w2_u8,
        w2_scale,
        down,
        slot_ids,
        local_eids,
        hidden,
        I,
        activated.stride(0),
        w2_u8.stride(1),
        w2_u8.stride(2),
        w2_scale.stride(1),
        w2_scale.stride(2),
        w2_u8.stride(0),
        w2_scale.stride(0),
        down.stride(0),
    )

    valid_mask = valid.unsqueeze(1).to(torch.float32)
    weights = topk_weights.reshape(-1).unsqueeze(1).to(torch.float32)
    weighted = down.float() * weights * valid_mask
    return weighted.view(G, topk, hidden).sum(dim=1)
