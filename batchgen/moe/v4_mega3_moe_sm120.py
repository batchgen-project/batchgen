from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from batchgen.moe.v4_ragged_moe_sm120 import (
    RaggedRoutingMetadata,
    _use_all_owned_routing_fast_path,
    build_ragged_routing_metadata,
)
from batchgen_kernels.moe.mega_moe_sm120 import (
    is_mega_moe_sm120_available,
    mega_moe_sm120_forward,
)

_MEGA3_STAGE1_CFG = {
    "block_m": 16,
    "block_i": 64,
    "block_k": 256,
    "num_warps": 4,
    "num_stages": 1,
}

_MEGA3_STAGE2_CFG = {
    "block_m": 16,
    "block_n": 128,
    "block_k": 256,
    "num_warps": 4,
    "num_stages": 1,
}


@dataclass
class Mega3Scratch:
    batch_max: int
    hidden: int
    topk: int
    intermediate: int
    slots_max: int
    activated: torch.Tensor


def prepare_mega3_scratch(
    batch_max: int,
    hidden: int,
    intermediate: int,
    device: torch.device,
    *,
    topk: int,
) -> Mega3Scratch:
    if batch_max <= 0:
        raise ValueError("batch_max must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")
    slots_max = batch_max * topk
    return Mega3Scratch(
        batch_max=batch_max,
        hidden=hidden,
        topk=topk,
        intermediate=intermediate,
        slots_max=slots_max,
        activated=torch.empty(
            (slots_max, intermediate), device=device, dtype=torch.bfloat16
        ),
    )


def _ensure_mega3_scratch(
    weight_ptrs: dict[str, object],
    *,
    num_tokens: int,
    hidden: int,
    topk: int,
    intermediate: int,
    device: torch.device,
) -> Mega3Scratch:
    scratch = weight_ptrs.get("mega3_scratch")
    if isinstance(scratch, Mega3Scratch):
        if (
            scratch.batch_max >= num_tokens
            and scratch.hidden == hidden
            and scratch.topk == topk
            and scratch.intermediate == intermediate
            and scratch.activated.device == device
        ):
            return scratch
    scratch = prepare_mega3_scratch(
        max(1, num_tokens),
        hidden,
        intermediate,
        device,
        topk=topk,
    )
    weight_ptrs["mega3_scratch"] = scratch
    return scratch


def route_pack(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    owned_start: int,
    owned_count: int,
    *,
    global_expert_count: int | None = None,
) -> RaggedRoutingMetadata | None:
    """Build compact routing metadata fully on GPU.

    Reuses the compact on-device counting/sort path from the ragged kernel rather
    than the old Triton route-pack kernel that was exploding IR.
    """

    return build_ragged_routing_metadata(
        topk_indices,
        topk_weights,
        owned_start,
        owned_count,
        block_m=_MEGA3_STAGE1_CFG["block_m"],
        assume_all_owned=_use_all_owned_routing_fast_path(
            owned_start,
            owned_count,
            global_expert_count,
        ),
    )


@triton.jit
def stage1_swiglu_kernel(
    hidden_states_ptr,
    sorted_token_ids_ptr,
    sorted_weights_ptr,
    block_experts_ptr,
    block_slot_starts_ptr,
    block_row_starts_ptr,
    expt_hist_ptr,
    stage1_weight_ptr,
    stage1_scale_ptr,
    activated_ptr,
    hidden,
    intermediate,
    stride_hidden_m,
    stride_hidden_k,
    stride_stage1_e,
    stride_stage1_k,
    stride_stage1_n,
    stride_stage1_se,
    stride_stage1_sn,
    stride_stage1_sk,
    stride_activated_m,
    stride_activated_n,
    swiglu_limit,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 32 == 0)

    pid = tl.program_id(0)
    grid_i = tl.cdiv(intermediate, BLOCK_I)
    block_idx = pid // grid_i
    pid_i = pid % grid_i

    expert = tl.load(block_experts_ptr + block_idx)
    slot_start = tl.load(block_slot_starts_ptr + block_idx)
    row_start = tl.load(block_row_starts_ptr + block_idx)
    e_rows = tl.load(expt_hist_ptr + expert)

    expert_i64 = tl.cast(expert, tl.int64)
    stride_hidden_m_i64 = tl.cast(stride_hidden_m, tl.int64)
    stride_hidden_k_i64 = tl.cast(stride_hidden_k, tl.int64)
    stride_stage1_e_i64 = tl.cast(stride_stage1_e, tl.int64)
    stride_stage1_k_i64 = tl.cast(stride_stage1_k, tl.int64)
    stride_stage1_n_i64 = tl.cast(stride_stage1_n, tl.int64)
    stride_stage1_se_i64 = tl.cast(stride_stage1_se, tl.int64)
    stride_stage1_sn_i64 = tl.cast(stride_stage1_sn, tl.int64)
    stride_stage1_sk_i64 = tl.cast(stride_stage1_sk, tl.int64)
    stride_activated_m_i64 = tl.cast(stride_activated_m, tl.int64)
    stride_activated_n_i64 = tl.cast(stride_activated_n, tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    up_cols = intermediate + offs_i
    slot_rows = slot_start + offs_m
    mask_m = (row_start + offs_m) < e_rows
    mask_i = offs_i < intermediate

    slot_rows_i64 = tl.cast(slot_rows, tl.int64)
    offs_i_i64 = tl.cast(offs_i, tl.int64)
    up_cols_i64 = tl.cast(up_cols, tl.int64)
    token_ids = tl.load(sorted_token_ids_ptr + slot_rows_i64, mask=mask_m, other=0)
    token_ids_i64 = tl.cast(token_ids, tl.int64)
    slot_weights = tl.load(sorted_weights_ptr + slot_rows_i64, mask=mask_m, other=0.0)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    for k0 in tl.range(0, hidden, BLOCK_K, num_stages=1, loop_unroll_factor=1):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        offs_k_scale = (k0 // 32) + tl.arange(0, BLOCK_K // 32)

        offs_k_i64 = tl.cast(offs_k, tl.int64)
        offs_k_packed_i64 = tl.cast(offs_k_packed, tl.int64)
        offs_k_scale_i64 = tl.cast(offs_k_scale, tl.int64)

        x = tl.load(
            hidden_states_ptr
            + token_ids_i64[:, None] * stride_hidden_m_i64
            + offs_k_i64[None, :] * stride_hidden_k_i64,
            mask=mask_m[:, None] & (offs_k[None, :] < hidden),
            other=0,
        )
        gate_w = tl.load(
            stage1_weight_ptr
            + expert_i64 * stride_stage1_e_i64
            + offs_k_packed_i64[:, None] * stride_stage1_k_i64
            + offs_i_i64[None, :] * stride_stage1_n_i64,
            mask=(offs_k_packed[:, None] < (hidden // 2))
            & mask_i[None, :],
            other=0,
        )
        gate_scale = tl.load(
            stage1_scale_ptr
            + expert_i64 * stride_stage1_se_i64
            + offs_i_i64[:, None] * stride_stage1_sn_i64
            + offs_k_scale_i64[None, :] * stride_stage1_sk_i64,
            mask=mask_i[:, None] & (offs_k_scale[None, :] < (hidden // 32)),
            other=127,
        )
        up_w = tl.load(
            stage1_weight_ptr
            + expert_i64 * stride_stage1_e_i64
            + offs_k_packed_i64[:, None] * stride_stage1_k_i64
            + up_cols_i64[None, :] * stride_stage1_n_i64,
            mask=(offs_k_packed[:, None] < (hidden // 2))
            & mask_i[None, :],
            other=0,
        )
        up_scale = tl.load(
            stage1_scale_ptr
            + expert_i64 * stride_stage1_se_i64
            + up_cols_i64[:, None] * stride_stage1_sn_i64
            + offs_k_scale_i64[None, :] * stride_stage1_sk_i64,
            mask=mask_i[:, None] & (offs_k_scale[None, :] < (hidden // 32)),
            other=127,
        )
        acc_gate = tl.dot_scaled(
            x,
            None,
            "bf16",
            gate_w,
            gate_scale,
            "e2m1",
            acc=acc_gate,
            fast_math=True,
            rhs_k_pack=True,
        )
        acc_up = tl.dot_scaled(
            x,
            None,
            "bf16",
            up_w,
            up_scale,
            "e2m1",
            acc=acc_up,
            fast_math=True,
            rhs_k_pack=True,
        )

    gate = acc_gate
    up = acc_up
    if swiglu_limit > 0:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.maximum(tl.minimum(up, swiglu_limit), -swiglu_limit)
    activated = (gate * tl.sigmoid(gate)) * up
    activated = activated * slot_weights[:, None].to(tl.float32)

    tl.store(
        activated_ptr
        + slot_rows_i64[:, None] * stride_activated_m_i64
        + offs_i_i64[None, :] * stride_activated_n_i64,
        activated.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_i[None, :],
    )


@triton.jit
def stage2_scatter_kernel(
    activated_ptr,
    sorted_token_ids_ptr,
    block_experts_ptr,
    block_slot_starts_ptr,
    block_row_starts_ptr,
    expt_hist_ptr,
    stage2_weight_ptr,
    stage2_scale_ptr,
    output_ptr,
    hidden,
    intermediate,
    stride_activated_m,
    stride_activated_k,
    stride_stage2_e,
    stride_stage2_k,
    stride_stage2_n,
    stride_stage2_se,
    stride_stage2_sn,
    stride_stage2_sk,
    stride_output_m,
    stride_output_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 32 == 0)

    pid = tl.program_id(0)
    grid_n = tl.cdiv(hidden, BLOCK_N)
    block_idx = pid // grid_n
    pid_n = pid % grid_n

    expert = tl.load(block_experts_ptr + block_idx)
    slot_start = tl.load(block_slot_starts_ptr + block_idx)
    row_start = tl.load(block_row_starts_ptr + block_idx)
    e_rows = tl.load(expt_hist_ptr + expert)

    expert_i64 = tl.cast(expert, tl.int64)
    stride_activated_m_i64 = tl.cast(stride_activated_m, tl.int64)
    stride_activated_k_i64 = tl.cast(stride_activated_k, tl.int64)
    stride_stage2_e_i64 = tl.cast(stride_stage2_e, tl.int64)
    stride_stage2_k_i64 = tl.cast(stride_stage2_k, tl.int64)
    stride_stage2_n_i64 = tl.cast(stride_stage2_n, tl.int64)
    stride_stage2_se_i64 = tl.cast(stride_stage2_se, tl.int64)
    stride_stage2_sn_i64 = tl.cast(stride_stage2_sn, tl.int64)
    stride_stage2_sk_i64 = tl.cast(stride_stage2_sk, tl.int64)
    stride_output_m_i64 = tl.cast(stride_output_m, tl.int64)
    stride_output_n_i64 = tl.cast(stride_output_n, tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    slot_rows = slot_start + offs_m
    mask_m = (row_start + offs_m) < e_rows
    mask_n = offs_n < hidden

    slot_rows_i64 = tl.cast(slot_rows, tl.int64)
    offs_n_i64 = tl.cast(offs_n, tl.int64)
    token_ids = tl.load(sorted_token_ids_ptr + slot_rows_i64, mask=mask_m, other=0)
    token_ids_i64 = tl.cast(token_ids, tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, intermediate, BLOCK_K, num_stages=1, loop_unroll_factor=1):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        offs_k_scale = (k0 // 32) + tl.arange(0, BLOCK_K // 32)

        offs_k_i64 = tl.cast(offs_k, tl.int64)
        offs_k_packed_i64 = tl.cast(offs_k_packed, tl.int64)
        offs_k_scale_i64 = tl.cast(offs_k_scale, tl.int64)

        x = tl.load(
            activated_ptr
            + slot_rows_i64[:, None] * stride_activated_m_i64
            + offs_k_i64[None, :] * stride_activated_k_i64,
            mask=mask_m[:, None] & (offs_k[None, :] < intermediate),
            other=0,
        )
        w = tl.load(
            stage2_weight_ptr
            + expert_i64 * stride_stage2_e_i64
            + offs_k_packed_i64[:, None] * stride_stage2_k_i64
            + offs_n_i64[None, :] * stride_stage2_n_i64,
            mask=(offs_k_packed[:, None] < (intermediate // 2)) & mask_n[None, :],
            other=0,
        )
        scale = tl.load(
            stage2_scale_ptr
            + expert_i64 * stride_stage2_se_i64
            + offs_n_i64[:, None] * stride_stage2_sn_i64
            + offs_k_scale_i64[None, :] * stride_stage2_sk_i64,
            mask=mask_n[:, None] & (offs_k_scale[None, :] < (intermediate // 32)),
            other=127,
        )
        acc = tl.dot_scaled(
            x,
            None,
            "bf16",
            w,
            scale,
            "e2m1",
            acc=acc,
            fast_math=True,
            rhs_k_pack=True,
        )

    out_ptrs = (
        output_ptr
        + token_ids_i64[:, None] * stride_output_m_i64
        + offs_n_i64[None, :] * stride_output_n_i64
    )
    tl.atomic_add(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def _launch_stage1_swiglu(
    token_states: torch.Tensor,
    routing: RaggedRoutingMetadata,
    stage1_weight: torch.Tensor,
    stage1_scale: torch.Tensor,
    activated: torch.Tensor,
    swiglu_limit: float,
) -> None:
    intermediate = stage1_weight.shape[2] // 2
    grid = (routing.num_blocks * triton.cdiv(intermediate, _MEGA3_STAGE1_CFG["block_i"]),)
    stage1_swiglu_kernel[grid](
        token_states,
        routing.sorted_token_ids,
        routing.sorted_weights,
        routing.block_experts,
        routing.block_slot_starts,
        routing.block_row_starts,
        routing.expt_hist,
        stage1_weight,
        stage1_scale,
        activated,
        token_states.shape[1],
        intermediate,
        token_states.stride(0),
        token_states.stride(1),
        stage1_weight.stride(0),
        stage1_weight.stride(1),
        stage1_weight.stride(2),
        stage1_scale.stride(0),
        stage1_scale.stride(1),
        stage1_scale.stride(2),
        activated.stride(0),
        activated.stride(1),
        float(swiglu_limit),
        BLOCK_M=_MEGA3_STAGE1_CFG["block_m"],
        BLOCK_I=_MEGA3_STAGE1_CFG["block_i"],
        BLOCK_K=_MEGA3_STAGE1_CFG["block_k"],
        num_warps=_MEGA3_STAGE1_CFG["num_warps"],
        num_stages=_MEGA3_STAGE1_CFG["num_stages"],
    )


def _launch_stage2_scatter(
    activated: torch.Tensor,
    routing: RaggedRoutingMetadata,
    stage2_weight: torch.Tensor,
    stage2_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    grid = (routing.num_blocks * triton.cdiv(output.shape[1], _MEGA3_STAGE2_CFG["block_n"]),)
    stage2_scatter_kernel[grid](
        activated,
        routing.sorted_token_ids,
        routing.block_experts,
        routing.block_slot_starts,
        routing.block_row_starts,
        routing.expt_hist,
        stage2_weight,
        stage2_scale,
        output,
        output.shape[1],
        activated.shape[1],
        activated.stride(0),
        activated.stride(1),
        stage2_weight.stride(0),
        stage2_weight.stride(1),
        stage2_weight.stride(2),
        stage2_scale.stride(0),
        stage2_scale.stride(1),
        stage2_scale.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_M=_MEGA3_STAGE2_CFG["block_m"],
        BLOCK_N=_MEGA3_STAGE2_CFG["block_n"],
        BLOCK_K=_MEGA3_STAGE2_CFG["block_k"],
        num_warps=_MEGA3_STAGE2_CFG["num_warps"],
        num_stages=_MEGA3_STAGE2_CFG["num_stages"],
    )


def _use_native_sm120_kernel() -> bool:
    if os.environ.get("BATCHGEN_V4_MEGA_FORCE_TRITON", "0") == "1":
        return False
    if os.environ.get("BATCHGEN_V4_MEGA_USE_NATIVE", "0") != "1":
        return False
    return is_mega_moe_sm120_available()


@torch.inference_mode()
def v4_mega3_moe_forward(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    token_states = token_states.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_indices = topk_indices.contiguous()
    num_tokens, hidden = token_states.shape
    if num_tokens == 0:
        return torch.empty((0, hidden), dtype=torch.float32, device=token_states.device)
    if topk_indices.ndim != 2 or topk_weights.shape != topk_indices.shape:
        raise ValueError("topk_indices/topk_weights must be rank-2 tensors with matching shape")

    routing = route_pack(
        topk_indices,
        topk_weights,
        owned_start,
        owned_count,
        global_expert_count=int(weight_ptrs.get("global_expert_count", owned_count)),
    )
    if routing is None:
        return torch.zeros((num_tokens, hidden), dtype=torch.float32, device=token_states.device)

    ragged = weight_ptrs["ragged_bundle"]
    stage1_weight = ragged["stage1_weight"]
    stage1_scale = ragged["stage1_scale"]
    stage2_weight = ragged["stage2_weight"]
    stage2_scale = ragged["stage2_scale"]
    intermediate = int(stage2_weight.shape[1] * 2)
    topk = int(topk_indices.shape[1])

    scratch = _ensure_mega3_scratch(
        weight_ptrs,
        num_tokens=num_tokens,
        hidden=hidden,
        topk=topk,
        intermediate=intermediate,
        device=token_states.device,
    )
    activated = scratch.activated[: routing.num_slots]
    output = torch.zeros((num_tokens, hidden), device=token_states.device, dtype=torch.float32)

    if _use_native_sm120_kernel():
        return mega_moe_sm120_forward(
            token_states,
            routing.sorted_token_ids,
            routing.sorted_weights,
            routing.block_experts,
            routing.block_slot_starts,
            routing.block_row_starts,
            torch.tensor([routing.num_blocks], device=token_states.device, dtype=torch.int32),
            routing.expt_hist,
            stage1_weight,
            stage1_scale,
            stage2_weight,
            stage2_scale,
            output,
            swiglu_limit=float(swiglu_limit),
        )

    _launch_stage1_swiglu(
        token_states,
        routing,
        stage1_weight,
        stage1_scale,
        activated,
        swiglu_limit,
    )
    _launch_stage2_scatter(
        activated,
        routing,
        stage2_weight,
        stage2_scale,
        output,
    )
    return output
