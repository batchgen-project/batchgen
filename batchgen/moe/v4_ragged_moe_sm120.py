from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

try:
    from batchgen.moe.fp4_utils import dequant_fp4_e2m1_weight
except ModuleNotFoundError:
    _fp4_utils_path = Path(__file__).resolve().with_name("fp4_utils.py")
    _fp4_utils_spec = importlib.util.spec_from_file_location(
        "batchgen_moe_fp4_utils_fallback", _fp4_utils_path
    )
    if _fp4_utils_spec is None or _fp4_utils_spec.loader is None:
        raise RuntimeError(f"Failed to load FP4 utils from {_fp4_utils_path}")
    _fp4_utils = importlib.util.module_from_spec(_fp4_utils_spec)
    sys.modules[_fp4_utils_spec.name] = _fp4_utils
    _fp4_utils_spec.loader.exec_module(_fp4_utils)
    dequant_fp4_e2m1_weight = _fp4_utils.dequant_fp4_e2m1_weight

_RAGGED_STAGE_CFG = {
    "block_m": 16,
    "block_n": 256,
    "block_k": 256,
    "num_warps": 4,
    "num_stages": 1,
}


@dataclass(frozen=True)
class RaggedRoutingMetadata:
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    expt_hist: torch.Tensor
    expt_offsets: torch.Tensor
    block_experts: torch.Tensor
    block_slot_starts: torch.Tensor
    block_row_starts: torch.Tensor

    @property
    def num_slots(self) -> int:
        return int(self.sorted_token_ids.numel())

    @property
    def num_blocks(self) -> int:
        return int(self.block_experts.numel())


@triton.jit
def _ragged_mxfp4_matmul_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    block_experts_ptr,
    block_slot_starts_ptr,
    block_row_starts_ptr,
    expt_hist_ptr,
    y_ptr,
    num_slots,
    out_features,
    k_features,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_se,
    stride_sn,
    stride_sk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 32 == 0)
    pid = tl.program_id(0)
    grid_n = tl.cdiv(out_features, BLOCK_N)
    block_idx = pid // grid_n
    pid_n = pid % grid_n

    expert = tl.load(block_experts_ptr + block_idx)
    slot_start = tl.load(block_slot_starts_ptr + block_idx)
    row_start = tl.load(block_row_starts_ptr + block_idx)
    e_rows = tl.load(expt_hist_ptr + expert)

    expert_i64 = tl.cast(expert, tl.int64)
    stride_xm_i64 = tl.cast(stride_xm, tl.int64)
    stride_xk_i64 = tl.cast(stride_xk, tl.int64)
    stride_we_i64 = tl.cast(stride_we, tl.int64)
    stride_wk_i64 = tl.cast(stride_wk, tl.int64)
    stride_wn_i64 = tl.cast(stride_wn, tl.int64)
    stride_se_i64 = tl.cast(stride_se, tl.int64)
    stride_sn_i64 = tl.cast(stride_sn, tl.int64)
    stride_sk_i64 = tl.cast(stride_sk, tl.int64)
    stride_ym_i64 = tl.cast(stride_ym, tl.int64)
    stride_yn_i64 = tl.cast(stride_yn, tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    slot_rows = slot_start + offs_m
    mask_m = (row_start + offs_m) < e_rows
    mask_n = offs_n < out_features

    slot_rows_i64 = tl.cast(slot_rows, tl.int64)
    offs_n_i64 = tl.cast(offs_n, tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k_features, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        offs_k_scale = (k0 // 32) + tl.arange(0, BLOCK_K // 32)
        offs_k_i64 = tl.cast(offs_k, tl.int64)
        offs_k_packed_i64 = tl.cast(offs_k_packed, tl.int64)
        offs_k_scale_i64 = tl.cast(offs_k_scale, tl.int64)

        x = tl.load(
            x_ptr + slot_rows_i64[:, None] * stride_xm_i64 + offs_k_i64[None, :] * stride_xk_i64,
            mask=mask_m[:, None] & (offs_k[None, :] < k_features),
            other=0,
        )
        w = tl.load(
            w_ptr
            + expert_i64 * stride_we_i64
            + offs_k_packed_i64[:, None] * stride_wk_i64
            + offs_n_i64[None, :] * stride_wn_i64,
            mask=(offs_k_packed[:, None] < (k_features // 2))
            & mask_n[None, :],
            other=0,
        )
        scale = tl.load(
            scale_ptr
            + expert_i64 * stride_se_i64
            + offs_n_i64[:, None] * stride_sn_i64
            + offs_k_scale_i64[None, :] * stride_sk_i64,
            mask=mask_n[:, None] & (offs_k_scale[None, :] < (k_features // 32)),
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

    tl.store(
        y_ptr + slot_rows_i64[:, None] * stride_ym_i64 + offs_n_i64[None, :] * stride_yn_i64,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _ragged_mxfp4_stage1_swiglu_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    block_experts_ptr,
    block_slot_starts_ptr,
    block_row_starts_ptr,
    expt_hist_ptr,
    slot_weights_ptr,
    y_ptr,
    intermediate,
    k_features,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_se,
    stride_sn,
    stride_sk,
    stride_sw,
    stride_ym,
    stride_yn,
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
    stride_xm_i64 = tl.cast(stride_xm, tl.int64)
    stride_xk_i64 = tl.cast(stride_xk, tl.int64)
    stride_we_i64 = tl.cast(stride_we, tl.int64)
    stride_wk_i64 = tl.cast(stride_wk, tl.int64)
    stride_wn_i64 = tl.cast(stride_wn, tl.int64)
    stride_se_i64 = tl.cast(stride_se, tl.int64)
    stride_sn_i64 = tl.cast(stride_sn, tl.int64)
    stride_sk_i64 = tl.cast(stride_sk, tl.int64)
    stride_sw_i64 = tl.cast(stride_sw, tl.int64)
    stride_ym_i64 = tl.cast(stride_ym, tl.int64)
    stride_yn_i64 = tl.cast(stride_yn, tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    up_cols = intermediate + offs_i
    slot_rows = slot_start + offs_m
    mask_m = (row_start + offs_m) < e_rows
    mask_i = offs_i < intermediate

    slot_rows_i64 = tl.cast(slot_rows, tl.int64)
    offs_i_i64 = tl.cast(offs_i, tl.int64)
    up_cols_i64 = tl.cast(up_cols, tl.int64)

    slot_weights = tl.load(slot_weights_ptr + slot_rows_i64 * stride_sw_i64, mask=mask_m, other=0.0)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    for k0 in range(0, k_features, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k_packed = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        offs_k_scale = (k0 // 32) + tl.arange(0, BLOCK_K // 32)
        offs_k_i64 = tl.cast(offs_k, tl.int64)
        offs_k_packed_i64 = tl.cast(offs_k_packed, tl.int64)
        offs_k_scale_i64 = tl.cast(offs_k_scale, tl.int64)

        x = tl.load(
            x_ptr + slot_rows_i64[:, None] * stride_xm_i64 + offs_k_i64[None, :] * stride_xk_i64,
            mask=mask_m[:, None] & (offs_k[None, :] < k_features),
            other=0,
        )
        gate_w = tl.load(
            w_ptr
            + expert_i64 * stride_we_i64
            + offs_k_packed_i64[:, None] * stride_wk_i64
            + offs_i_i64[None, :] * stride_wn_i64,
            mask=(offs_k_packed[:, None] < (k_features // 2))
            & mask_i[None, :],
            other=0,
        )
        gate_scale = tl.load(
            scale_ptr
            + expert_i64 * stride_se_i64
            + offs_i_i64[:, None] * stride_sn_i64
            + offs_k_scale_i64[None, :] * stride_sk_i64,
            mask=mask_i[:, None] & (offs_k_scale[None, :] < (k_features // 32)),
            other=127,
        )
        up_w = tl.load(
            w_ptr
            + expert_i64 * stride_we_i64
            + offs_k_packed_i64[:, None] * stride_wk_i64
            + up_cols_i64[None, :] * stride_wn_i64,
            mask=(offs_k_packed[:, None] < (k_features // 2))
            & mask_i[None, :],
            other=0,
        )
        up_scale = tl.load(
            scale_ptr
            + expert_i64 * stride_se_i64
            + up_cols_i64[:, None] * stride_sn_i64
            + offs_k_scale_i64[None, :] * stride_sk_i64,
            mask=mask_i[:, None] & (offs_k_scale[None, :] < (k_features // 32)),
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
        y_ptr + slot_rows_i64[:, None] * stride_ym_i64 + offs_i_i64[None, :] * stride_yn_i64,
        activated.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_i[None, :],
    )


def _round_pow2_scale(scale: torch.Tensor) -> torch.Tensor:
    return torch.pow(
        torch.full_like(scale, 2.0, dtype=torch.float32),
        torch.round(torch.log2(torch.clamp(scale, min=2.0**-20))),
    )


def _canonicalize_dense_to_mxfp4(dense_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, k_features = dense_weight.shape
    if k_features % 32 != 0:
        raise ValueError("MXFP4 quantization requires K divisible by 32")
    blocks = dense_weight.float().reshape(out_features, k_features // 32, 32)
    block_scale = _round_pow2_scale(blocks.abs().amax(dim=-1) / 6.0)
    normalized = (blocks / block_scale.unsqueeze(-1)).reshape(out_features, k_features)
    packed = MXFP4Tensor(normalized).to_packed_tensor(dim=1).contiguous()
    scale = MXScaleTensor(block_scale).data.contiguous()
    return packed.transpose(0, 1).contiguous(), scale


def _canonicalize_expert_weight(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dense = dequant_fp4_e2m1_weight(weight, scale, torch.bfloat16)
    return _canonicalize_dense_to_mxfp4(dense)


def prepare_ragged_weight_bundle(expert_weights: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not expert_weights:
        raise ValueError("expert_weights must be non-empty")

    num_experts = len(expert_weights)
    stage1_w_bundle = None
    stage1_s_bundle = None
    stage2_w_bundle = None
    stage2_s_bundle = None

    for expert_idx, expert in enumerate(expert_weights):
        gate = dequant_fp4_e2m1_weight(expert["w1.weight"], expert["w1.scale"], torch.bfloat16)
        up = dequant_fp4_e2m1_weight(expert["w3.weight"], expert["w3.scale"], torch.bfloat16)
        fused_dense = torch.cat([gate, up], dim=0)
        del gate, up

        fused_w, fused_s = _canonicalize_dense_to_mxfp4(fused_dense)
        del fused_dense

        down_w, down_s = _canonicalize_expert_weight(expert["w2.weight"], expert["w2.scale"])
        fused_s_u8 = fused_s.view(torch.uint8)
        down_s_u8 = down_s.view(torch.uint8)

        if stage1_w_bundle is None:
            stage1_w_bundle = torch.empty(
                (num_experts, *fused_w.shape),
                device=fused_w.device,
                dtype=fused_w.dtype,
            )
            stage1_s_bundle = torch.empty(
                (num_experts, *fused_s_u8.shape),
                device=fused_s_u8.device,
                dtype=fused_s_u8.dtype,
            )
            stage2_w_bundle = torch.empty(
                (num_experts, *down_w.shape),
                device=down_w.device,
                dtype=down_w.dtype,
            )
            stage2_s_bundle = torch.empty(
                (num_experts, *down_s_u8.shape),
                device=down_s_u8.device,
                dtype=down_s_u8.dtype,
            )

        stage1_w_bundle[expert_idx].copy_(fused_w)
        stage1_s_bundle[expert_idx].copy_(fused_s_u8)
        stage2_w_bundle[expert_idx].copy_(down_w)
        stage2_s_bundle[expert_idx].copy_(down_s_u8)

        del fused_w, fused_s, fused_s_u8, down_w, down_s, down_s_u8

    return {
        "stage1_weight": stage1_w_bundle,
        "stage1_scale": stage1_s_bundle,
        "stage2_weight": stage2_w_bundle,
        "stage2_scale": stage2_s_bundle,
    }


def build_ragged_routing_metadata(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    owned_start: int,
    owned_count: int,
    *,
    block_m: int = _RAGGED_STAGE_CFG["block_m"],
    assume_all_owned: bool = False,
) -> RaggedRoutingMetadata | None:
    device = topk_indices.device
    _, topk = topk_indices.shape
    flat_weights = topk_weights.reshape(-1)
    if assume_all_owned:
        sorted_eids, order = torch.sort(topk_indices.reshape(-1).to(torch.int32))
        sorted_token_ids = torch.div(order, topk, rounding_mode="floor").to(torch.int64)
        sorted_weights = flat_weights.index_select(0, order)
        expt_hist = torch.bincount(sorted_eids.to(torch.int64), minlength=owned_count).to(torch.int32)
    else:
        local_eids = topk_indices.to(torch.int32) - owned_start
        valid_mask = (local_eids >= 0) & (local_eids < owned_count)
        sentinel = torch.full_like(local_eids, owned_count)
        sort_keys = torch.where(valid_mask, local_eids, sentinel).reshape(-1)
        sorted_eids_all, order = torch.sort(sort_keys)
        hist_full = torch.bincount(sorted_eids_all.to(torch.int64), minlength=owned_count + 1)
        expt_hist = hist_full[:-1].to(torch.int32)
        valid_order = order.masked_select(sorted_eids_all < owned_count)
        sorted_token_ids = torch.div(valid_order, topk, rounding_mode="floor").to(torch.int64)
        sorted_weights = flat_weights.index_select(0, valid_order)

    expt_offsets = torch.empty(owned_count + 1, device=device, dtype=torch.int32)
    expt_offsets[0] = 0
    expt_offsets[1:] = torch.cumsum(expt_hist, dim=0)

    block_counts = torch.div(expt_hist + (block_m - 1), block_m, rounding_mode="floor")
    block_experts = torch.repeat_interleave(
        torch.arange(owned_count, device=device, dtype=torch.int32),
        block_counts,
    )
    if block_experts.numel() == 0:
        return None
    block_offsets = torch.empty(owned_count + 1, device=device, dtype=torch.int32)
    block_offsets[0] = 0
    block_offsets[1:] = torch.cumsum(block_counts, dim=0)
    block_ids = torch.arange(block_experts.numel(), device=device, dtype=torch.int32)
    block_row_starts = (block_ids - block_offsets.index_select(0, block_experts.to(torch.int64))) * block_m
    block_slot_starts = expt_offsets.index_select(0, block_experts.to(torch.int64)) + block_row_starts

    return RaggedRoutingMetadata(
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        expt_hist=expt_hist,
        expt_offsets=expt_offsets,
        block_experts=block_experts,
        block_slot_starts=block_slot_starts,
        block_row_starts=block_row_starts,
    )


def _use_all_owned_routing_fast_path(
    owned_start: int,
    owned_count: int,
    global_expert_count: int | None,
) -> bool:
    return (
        global_expert_count is not None
        and owned_start == 0
        and owned_count == global_expert_count
    )


def _launch_ragged_stage(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    routing: RaggedRoutingMetadata,
    out_features: int,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    out = torch.zeros((routing.num_slots, out_features), device=x.device, dtype=torch.bfloat16)
    grid = (routing.num_blocks * triton.cdiv(out_features, block_n),)
    _ragged_mxfp4_matmul_kernel[grid](
        x,
        weight,
        scale,
        routing.block_experts,
        routing.block_slot_starts,
        routing.block_row_starts,
        routing.expt_hist,
        out,
        routing.num_slots,
        out_features,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _launch_ragged_stage1_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    routing: RaggedRoutingMetadata,
    out_features: int,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    swiglu_limit: float,
) -> torch.Tensor:
    out = torch.zeros((routing.num_slots, out_features), device=x.device, dtype=torch.bfloat16)
    grid = (routing.num_blocks * triton.cdiv(out_features, block_n // 2),)
    _ragged_mxfp4_stage1_swiglu_kernel[grid](
        x,
        weight,
        scale,
        routing.block_experts,
        routing.block_slot_starts,
        routing.block_row_starts,
        routing.expt_hist,
        routing.sorted_weights,
        out,
        out_features,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        routing.sorted_weights.stride(0),
        out.stride(0),
        out.stride(1),
        float(swiglu_limit),
        BLOCK_M=block_m,
        BLOCK_I=block_n // 2,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@torch.inference_mode()
def v4_grouped_mxfp4_moe_forward_ragged_ptrs(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    token_states = token_states.contiguous()
    num_tokens, hidden = token_states.shape
    ragged = weight_ptrs["ragged_bundle"]
    global_expert_count = weight_ptrs.get("global_expert_count")
    routing = build_ragged_routing_metadata(
        topk_indices,
        topk_weights,
        owned_start,
        owned_count,
        assume_all_owned=_use_all_owned_routing_fast_path(
            owned_start,
            owned_count,
            int(global_expert_count) if global_expert_count is not None else None,
        ),
    )
    if routing is None:
        return torch.zeros((num_tokens, hidden), dtype=torch.float32, device=token_states.device)

    stage1_weight = ragged["stage1_weight"]
    stage1_scale = ragged["stage1_scale"]
    stage2_weight = ragged["stage2_weight"]
    stage2_scale = ragged["stage2_scale"]

    sorted_hidden = token_states.index_select(0, routing.sorted_token_ids)
    intermediate = stage1_weight.shape[2] // 2

    stage2_in = _launch_ragged_stage1_swiglu(
        sorted_hidden,
        stage1_weight,
        stage1_scale,
        routing,
        intermediate,
        **_RAGGED_STAGE_CFG,
        swiglu_limit=swiglu_limit,
    )
    stage2 = _launch_ragged_stage(stage2_in, stage2_weight, stage2_scale, routing, hidden, **_RAGGED_STAGE_CFG)

    output = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=token_states.device)
    output.index_add_(0, routing.sorted_token_ids, stage2.float())
    return output
