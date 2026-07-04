from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from batchgen_kernels.moe.mega_moe_sm120 import (
    is_mega_moe_sm120_available,
    mega_moe_sm120_forward,
)

_MEGA_CFG = {
    "block_m": 16,
    "block_k": 256,
    "block_i": 128,
    "block_n": 128,
    "num_warps": 4,
    "num_stages": 1,
    "zero_block": 256,
}


@dataclass
class MegaScratch:
    batch_max: int
    hidden: int
    topk: int
    experts_max: int
    slots_max: int
    blocks_max: int
    slot_token_ids: torch.Tensor
    slot_weights: torch.Tensor
    block_experts: torch.Tensor
    block_slot_starts: torch.Tensor
    block_rows: torch.Tensor
    num_blocks: torch.Tensor
    expt_hist: torch.Tensor
    expt_offsets: torch.Tensor
    expt_write_offsets: torch.Tensor


def prepare_mega_scratch(
    batch_max: int,
    hidden: int,
    device: torch.device,
    *,
    topk: int = 6,
    experts_max: int = 256,
    block_m: int = _MEGA_CFG["block_m"],
) -> MegaScratch:
    if batch_max <= 0:
        raise ValueError("batch_max must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if experts_max <= 0:
        raise ValueError("experts_max must be positive")
    slots_max = batch_max * topk
    del block_m
    blocks_max = slots_max
    return MegaScratch(
        batch_max=batch_max,
        hidden=hidden,
        topk=topk,
        experts_max=experts_max,
        slots_max=slots_max,
        blocks_max=blocks_max,
        slot_token_ids=torch.empty((slots_max,), device=device, dtype=torch.int64),
        slot_weights=torch.empty((slots_max,), device=device, dtype=torch.float32),
        block_experts=torch.empty((blocks_max,), device=device, dtype=torch.int32),
        block_slot_starts=torch.empty((blocks_max,), device=device, dtype=torch.int32),
        block_rows=torch.empty((blocks_max,), device=device, dtype=torch.int32),
        num_blocks=torch.empty((1,), device=device, dtype=torch.int32),
        expt_hist=torch.empty((experts_max,), device=device, dtype=torch.int32),
        expt_offsets=torch.empty((experts_max + 1,), device=device, dtype=torch.int32),
        expt_write_offsets=torch.empty((experts_max,), device=device, dtype=torch.int32),
    )


def _ensure_mega_scratch(
    weight_ptrs: dict[str, object],
    *,
    num_tokens: int,
    hidden: int,
    topk: int,
    experts_max: int,
    device: torch.device,
) -> MegaScratch:
    scratch = weight_ptrs.get("mega_scratch")
    if isinstance(scratch, MegaScratch):
        if (
            scratch.batch_max >= num_tokens
            and scratch.hidden == hidden
            and scratch.topk == topk
            and scratch.experts_max >= experts_max
            and scratch.slot_token_ids.device == device
        ):
            return scratch
    scratch = prepare_mega_scratch(
        max(1, num_tokens),
        hidden,
        device,
        topk=topk,
        experts_max=experts_max,
    )
    weight_ptrs["mega_scratch"] = scratch
    return scratch


@triton.jit
def route_pack_kernel(
    topk_indices_ptr,
    topk_weights_ptr,
    slot_token_ids_ptr,
    slot_weights_ptr,
    block_experts_ptr,
    block_slot_starts_ptr,
    block_rows_ptr,
    num_blocks_ptr,
    expt_hist_ptr,
    expt_offsets_ptr,
    expt_write_offsets_ptr,
    output_ptr,
    num_tokens,
    hidden,
    owned_start,
    owned_count,
    stride_index_m,
    stride_index_k,
    stride_weight_m,
    stride_weight_k,
    stride_output_m,
    stride_output_n,
    BATCH_MAX: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS_MAX: tl.constexpr,
    BLOCKS_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = num_tokens * hidden
    if pid > 0:
        offs = (pid - 1) * ZERO_BLOCK + tl.arange(0, ZERO_BLOCK)
        mask = offs < total_elements
        rows = offs // hidden
        cols = offs % hidden
        tl.store(
            output_ptr + rows * stride_output_m + cols * stride_output_n,
            0.0,
            mask=mask,
        )
        return

    expert_offs = tl.arange(0, EXPERTS_MAX)
    tl.store(expt_hist_ptr + expert_offs, 0, mask=expert_offs < EXPERTS_MAX)
    tl.store(
        expt_write_offsets_ptr + expert_offs,
        0,
        mask=expert_offs < EXPERTS_MAX,
    )
    tl.store(expt_offsets_ptr + expert_offs, 0, mask=expert_offs < EXPERTS_MAX)
    tl.store(expt_offsets_ptr + EXPERTS_MAX, 0)
    tl.store(num_blocks_ptr, 0)

    total_slots = num_tokens * TOPK
    owned_end = owned_start + owned_count
    for flat in range(0, BATCH_MAX * TOPK):
        if flat < total_slots:
            tok = flat // TOPK
            lane = flat % TOPK
            expert = tl.load(topk_indices_ptr + tok * stride_index_m + lane * stride_index_k)
            valid = (expert >= owned_start) & (expert < owned_end)
            if valid:
                local_e = (expert - owned_start).to(tl.int32)
                count = tl.load(expt_hist_ptr + local_e)
                tl.store(expt_hist_ptr + local_e, count + 1)

    running = 0
    for expert_idx in range(0, EXPERTS_MAX):
        tl.store(expt_offsets_ptr + expert_idx, running)
        if expert_idx < owned_count:
            count = tl.load(expt_hist_ptr + expert_idx)
            tl.store(expt_write_offsets_ptr + expert_idx, running)
            running += count
    tl.store(expt_offsets_ptr + EXPERTS_MAX, running)

    for flat_idx in range(0, BATCH_MAX * TOPK):
        if flat_idx < total_slots:
            tok = flat_idx // TOPK
            lane = flat_idx % TOPK
            expert = tl.load(topk_indices_ptr + tok * stride_index_m + lane * stride_index_k)
            valid = (expert >= owned_start) & (expert < owned_end)
            if valid:
                local_e = (expert - owned_start).to(tl.int32)
                dst = tl.load(expt_write_offsets_ptr + local_e)
                weight = tl.load(topk_weights_ptr + tok * stride_weight_m + lane * stride_weight_k)
                tl.store(slot_token_ids_ptr + dst, tok.to(tl.int64))
                tl.store(slot_weights_ptr + dst, weight)
                tl.store(expt_write_offsets_ptr + local_e, dst + 1)

    block_counter = 0
    for expert_idx in range(0, EXPERTS_MAX):
        if expert_idx < owned_count:
            count = tl.load(expt_hist_ptr + expert_idx)
            if count > 0:
                start = tl.load(expt_offsets_ptr + expert_idx)
                blocks_for_expert = (count + BLOCK_M - 1) // BLOCK_M
                for block in range(0, BLOCKS_MAX):
                    if block < blocks_for_expert:
                        dst = block_counter + block
                        tl.store(block_experts_ptr + dst, expert_idx)
                        tl.store(block_slot_starts_ptr + dst, start + block * BLOCK_M)
                        tl.store(block_rows_ptr + dst, block * BLOCK_M)
                block_counter += blocks_for_expert
    tl.store(num_blocks_ptr, block_counter)


@triton.jit
def moe_mega_kernel(
    hidden_states_ptr,
    slot_token_ids_ptr,
    slot_weights_ptr,
    block_experts_ptr,
    block_slot_starts_ptr,
    block_rows_ptr,
    num_blocks_ptr,
    expt_hist_ptr,
    stage1_weight_ptr,
    stage1_scale_ptr,
    stage2_weight_ptr,
    stage2_scale_ptr,
    output_ptr,
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
    stride_stage2_e,
    stride_stage2_k,
    stride_stage2_n,
    stride_stage2_se,
    stride_stage2_sn,
    stride_stage2_sk,
    stride_output_m,
    stride_output_n,
    swiglu_limit,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 32 == 0)
    tl.static_assert(BLOCK_I % 32 == 0)

    pid = tl.program_id(0)
    grid_n = tl.cdiv(hidden, BLOCK_N)
    block_idx = pid // grid_n
    pid_n = pid % grid_n
    active_blocks = tl.load(num_blocks_ptr)
    if block_idx >= active_blocks:
        return

    expert = tl.load(block_experts_ptr + block_idx)
    slot_start = tl.load(block_slot_starts_ptr + block_idx)
    row_start = tl.load(block_rows_ptr + block_idx)
    e_rows = tl.load(expt_hist_ptr + expert)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    slot_rows = slot_start + offs_m
    mask_m = (row_start + offs_m) < e_rows
    mask_n = offs_n < hidden
    token_ids = tl.load(slot_token_ids_ptr + slot_rows, mask=mask_m, other=0)
    acc_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i0 in range(0, intermediate, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        offs_i_packed = (i0 // 2) + tl.arange(0, BLOCK_I // 2)
        offs_i_scale = (i0 // 32) + tl.arange(0, BLOCK_I // 32)
        offs_up = intermediate + offs_i

        acc_gate = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        for k0 in range(0, hidden, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
            offs_k_scale = (k0 // 32) + tl.arange(0, BLOCK_K // 32)

            x = tl.load(
                hidden_states_ptr
                + token_ids[:, None] * stride_hidden_m
                + offs_k[None, :] * stride_hidden_k,
                mask=mask_m[:, None] & (offs_k[None, :] < hidden),
                other=0,
            )
            gate_w = tl.load(
                stage1_weight_ptr
                + expert * stride_stage1_e
                + offs_k_packed[:, None] * stride_stage1_k
                + offs_i[None, :] * stride_stage1_n,
                mask=(offs_k_packed[:, None] < (hidden // 2))
                & (offs_i[None, :] < intermediate),
                other=0,
            )
            gate_scale = tl.load(
                stage1_scale_ptr
                + expert * stride_stage1_se
                + offs_i[:, None] * stride_stage1_sn
                + offs_k_scale[None, :] * stride_stage1_sk,
                mask=(offs_i[:, None] < intermediate)
                & (offs_k_scale[None, :] < (hidden // 32)),
                other=127,
            )
            up_w = tl.load(
                stage1_weight_ptr
                + expert * stride_stage1_e
                + offs_k_packed[:, None] * stride_stage1_k
                + offs_up[None, :] * stride_stage1_n,
                mask=(offs_k_packed[:, None] < (hidden // 2))
                & (offs_up[None, :] < (2 * intermediate)),
                other=0,
            )
            up_scale = tl.load(
                stage1_scale_ptr
                + expert * stride_stage1_se
                + offs_up[:, None] * stride_stage1_sn
                + offs_k_scale[None, :] * stride_stage1_sk,
                mask=(offs_up[:, None] < (2 * intermediate))
                & (offs_k_scale[None, :] < (hidden // 32)),
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

        if swiglu_limit > 0:
            acc_gate = tl.minimum(acc_gate, swiglu_limit)
            acc_up = tl.maximum(tl.minimum(acc_up, swiglu_limit), -swiglu_limit)
        activated = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
        activated_bf16 = activated.to(tl.bfloat16)

        stage2_w = tl.load(
            stage2_weight_ptr
            + expert * stride_stage2_e
            + offs_i_packed[:, None] * stride_stage2_k
            + offs_n[None, :] * stride_stage2_n,
            mask=(offs_i_packed[:, None] < (intermediate // 2)) & mask_n[None, :],
            other=0,
        )
        stage2_scale = tl.load(
            stage2_scale_ptr
            + expert * stride_stage2_se
            + offs_n[:, None] * stride_stage2_sn
            + offs_i_scale[None, :] * stride_stage2_sk,
            mask=mask_n[:, None] & (offs_i_scale[None, :] < (intermediate // 32)),
            other=127,
        )
        acc_out = tl.dot_scaled(
            activated_bf16,
            None,
            "bf16",
            stage2_w,
            stage2_scale,
            "e2m1",
            acc=acc_out,
            fast_math=True,
            rhs_k_pack=True,
        )

    slot_weight = tl.load(slot_weights_ptr + slot_rows, mask=mask_m, other=0).to(tl.float32)
    acc_out = acc_out * slot_weight[:, None]
    out_ptrs = output_ptr + token_ids[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    tl.atomic_add(out_ptrs, acc_out, mask=mask_m[:, None] & mask_n[None, :])


def snapshot_route_pack(
    scratch: MegaScratch,
    *,
    owned_count: int,
) -> dict[str, torch.Tensor]:
    num_blocks = int(scratch.num_blocks.item())
    expt_hist = scratch.expt_hist[:owned_count].clone()
    num_slots = int(expt_hist.sum().item())
    return {
        "sorted_token_ids": scratch.slot_token_ids[:num_slots].clone(),
        "sorted_weights": scratch.slot_weights[:num_slots].clone(),
        "expt_hist": expt_hist,
        "expt_offsets": scratch.expt_offsets[: owned_count + 1].clone(),
        "block_experts": scratch.block_experts[:num_blocks].clone(),
        "block_slot_starts": scratch.block_slot_starts[:num_blocks].clone(),
        "block_row_starts": scratch.block_rows[:num_blocks].clone(),
        "num_blocks": scratch.num_blocks.clone(),
    }


def _use_native_sm120_kernel() -> bool:
    if os.environ.get("BATCHGEN_V4_MEGA_FORCE_TRITON", "0") == "1":
        return False
    return is_mega_moe_sm120_available()


@torch.inference_mode()
def v4_mega_moe_forward(
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

    ragged = weight_ptrs["ragged_bundle"]
    stage1_weight = ragged["stage1_weight"]
    stage1_scale = ragged["stage1_scale"]
    stage2_weight = ragged["stage2_weight"]
    stage2_scale = ragged["stage2_scale"]
    topk = int(topk_indices.shape[1])
    intermediate = int(stage2_weight.shape[1] * 2)

    scratch = _ensure_mega_scratch(
        weight_ptrs,
        num_tokens=num_tokens,
        hidden=hidden,
        topk=topk,
        experts_max=int(stage1_weight.shape[0]),
        device=token_states.device,
    )
    output = torch.empty((num_tokens, hidden), device=token_states.device, dtype=torch.float32)

    route_grid = lambda meta: (1 + triton.cdiv(num_tokens * hidden, meta["ZERO_BLOCK"]),)
    route_pack_kernel[route_grid](
        topk_indices,
        topk_weights,
        scratch.slot_token_ids,
        scratch.slot_weights,
        scratch.block_experts,
        scratch.block_slot_starts,
        scratch.block_rows,
        scratch.num_blocks,
        scratch.expt_hist,
        scratch.expt_offsets,
        scratch.expt_write_offsets,
        output,
        num_tokens,
        hidden,
        owned_start,
        owned_count,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        output.stride(0),
        output.stride(1),
        BATCH_MAX=scratch.batch_max,
        TOPK=scratch.topk,
        EXPERTS_MAX=scratch.experts_max,
        BLOCKS_MAX=scratch.blocks_max,
        BLOCK_M=_MEGA_CFG["block_m"],
        ZERO_BLOCK=_MEGA_CFG["zero_block"],
        num_warps=1,
        num_stages=1,
    )

    if _use_native_sm120_kernel():
        return mega_moe_sm120_forward(
            token_states,
            scratch.slot_token_ids,
            scratch.slot_weights,
            scratch.block_experts,
            scratch.block_slot_starts,
            scratch.block_rows,
            scratch.num_blocks,
            scratch.expt_hist,
            stage1_weight,
            stage1_scale,
            stage2_weight,
            stage2_scale,
            output,
            swiglu_limit=float(swiglu_limit),
        )

    mega_grid = (scratch.blocks_max * triton.cdiv(hidden, _MEGA_CFG["block_n"]),)
    moe_mega_kernel[mega_grid](
        token_states,
        scratch.slot_token_ids,
        scratch.slot_weights,
        scratch.block_experts,
        scratch.block_slot_starts,
        scratch.block_rows,
        scratch.num_blocks,
        scratch.expt_hist,
        stage1_weight,
        stage1_scale,
        stage2_weight,
        stage2_scale,
        output,
        hidden,
        intermediate,
        token_states.stride(0),
        token_states.stride(1),
        stage1_weight.stride(0),
        stage1_weight.stride(1),
        stage1_weight.stride(2),
        stage1_scale.stride(0),
        stage1_scale.stride(1),
        stage1_scale.stride(2),
        stage2_weight.stride(0),
        stage2_weight.stride(1),
        stage2_weight.stride(2),
        stage2_scale.stride(0),
        stage2_scale.stride(1),
        stage2_scale.stride(2),
        output.stride(0),
        output.stride(1),
        float(swiglu_limit),
        BLOCK_M=_MEGA_CFG["block_m"],
        BLOCK_N=_MEGA_CFG["block_n"],
        BLOCK_K=_MEGA_CFG["block_k"],
        BLOCK_I=_MEGA_CFG["block_i"],
        num_warps=_MEGA_CFG["num_warps"],
        num_stages=_MEGA_CFG["num_stages"],
    )
    return output
