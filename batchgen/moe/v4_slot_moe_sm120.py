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


@triton.jit
def _e8m0_scale_to_f32(scale_u8):
    return tl.math.exp2(scale_u8.to(tl.float32) - 127.0)


@triton.jit
def _slot_gemv_ptr_kernel(
    A_ptr,
    B_ptrs_ptr,
    S_ptrs_ptr,
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
    stride_cm: tl.int32,
    SCALE_IS_E8M0: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    slot_id = tl.program_id(0)
    n_block = tl.program_id(1)

    token_id = tl.load(token_ids_ptr + slot_id).to(tl.int64)
    expert_id = tl.load(expert_ids_ptr + slot_id).to(tl.int64)

    b_base_ptr = tl.load(B_ptrs_ptr + expert_id).to(tl.pointer_type(tl.uint8))
    if SCALE_IS_E8M0:
        s_base_ptr = tl.load(S_ptrs_ptr + expert_id).to(
            tl.pointer_type(tl.uint8)
        )
    else:
        s_base_ptr = tl.load(S_ptrs_ptr + expert_id).to(
            tl.pointer_type(tl.float32)
        )

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    a_base = token_id * stride_am
    for k_start in range(0, K, BLOCK_K):
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        b_mask = n_mask[:, None] & (offs_k2[None, :] < K // 2)
        b_packed = tl.load(
            b_base_ptr
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
        raw_scales = tl.load(
            s_base_ptr
            + offs_n[:, None] * stride_bsn
            + (k_start // 32 + group_ids[None, :]) * stride_bsk32,
            mask=s_mask,
            other=127 if SCALE_IS_E8M0 else 1.0,
        )
        if SCALE_IS_E8M0:
            scales = _e8m0_scale_to_f32(raw_scales)
        else:
            scales = raw_scales.to(tl.float32)
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


def setup_v4_expert_weight_pointers(
    expert_weights: list[dict[str, torch.Tensor]],
) -> dict[str, object]:
    """Create device pointer arrays for resident V4 expert weights.

    The tensors remain owned by the model/parameter-server wrappers; this helper
    only materializes small int64 pointer arrays, avoiding per-layer stacked
    copies of the FP4 weights.
    """
    if not expert_weights:
        raise ValueError("expert_weights must be non-empty")
    required = (
        "w1.weight",
        "w1.scale",
        "w3.weight",
        "w3.scale",
        "w2.weight",
        "w2.scale",
    )
    first = expert_weights[0]
    device = first["w1.weight"].device
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    for rw in expert_weights:
        for name in required:
            if name not in rw:
                raise KeyError(name)
            ref = first[name]
            if rw[name].device != device:
                raise ValueError(f"{name} must be on device {device}")
            if rw[name].shape != ref.shape:
                raise ValueError(f"{name} shape must match first expert")
            if rw[name].stride() != ref.stride():
                raise ValueError(f"{name} stride must match first expert")
            if not rw[name].is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous for pointer staging"
                )
            if name.endswith(".weight") and rw[name].element_size() != 1:
                raise ValueError(f"{name} must be byte-packed FP4")
            if name.endswith(".scale") and rw[name].element_size() not in (
                1,
                4,
            ):
                raise ValueError(f"{name} scale must be E8M0/uint8 or float32")
            if name.endswith(".scale") and rw[name].element_size() == 1:
                if (
                    rw[name].dtype != torch.uint8
                    and rw[name].dtype != e8m0_dtype
                ):
                    raise ValueError(
                        f"{name} 1-byte scale must be uint8 or E8M0"
                    )
            if name.endswith(".scale") and rw[name].element_size() == 4:
                if rw[name].dtype != torch.float32:
                    raise ValueError(f"{name} 4-byte scale must be float32")

    def ptrs(name: str) -> torch.Tensor:
        return torch.tensor(
            [rw[name].data_ptr() for rw in expert_weights],
            dtype=torch.int64,
            device=device,
        )

    return {
        "gate_ptrs": ptrs("w1.weight"),
        "gate_scale_ptrs": ptrs("w1.scale"),
        "up_ptrs": ptrs("w3.weight"),
        "up_scale_ptrs": ptrs("w3.scale"),
        "down_ptrs": ptrs("w2.weight"),
        "down_scale_ptrs": ptrs("w2.scale"),
        "gate_weight_ref": first["w1.weight"],
        "gate_scale_ref": first["w1.scale"],
        "up_weight_ref": first["w3.weight"],
        "up_scale_ref": first["w3.scale"],
        "down_weight_ref": first["w2.weight"],
        "down_scale_ref": first["w2.scale"],
        "expert_refs": expert_weights,
    }


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


def v4_slot_moe_forward_ptrs(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Pointer-array variant of v4_slot_moe_forward for resident expert weights."""
    import torch.nn.functional as F

    gate_ref = weight_ptrs["gate_weight_ref"]
    gate_scale_ref = weight_ptrs["gate_scale_ref"]
    up_ref = weight_ptrs["up_weight_ref"]
    up_scale_ref = weight_ptrs["up_scale_ref"]
    down_ref = weight_ptrs["down_weight_ref"]
    down_scale_ref = weight_ptrs["down_scale_ref"]

    G, hidden = token_states.shape
    topk = topk_indices.shape[1]
    I = gate_ref.shape[0]
    num_slots = G * topk
    device = token_states.device
    dtype = token_states.dtype

    token_states = token_states.contiguous()
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

    gate = torch.empty(num_slots, I, dtype=dtype, device=device)
    up = torch.empty(num_slots, I, dtype=dtype, device=device)
    grid1 = lambda meta: (num_slots, triton.cdiv(I, meta["BLOCK_N"]))
    _slot_gemv_ptr_kernel[grid1](
        token_states,
        weight_ptrs["gate_ptrs"],
        weight_ptrs["gate_scale_ptrs"],
        gate,
        token_ids,
        local_eids,
        I,
        hidden,
        token_states.stride(0),
        gate_ref.stride(0),
        gate_ref.stride(1),
        gate_scale_ref.stride(0),
        gate_scale_ref.stride(1),
        gate.stride(0),
        gate_scale_ref.element_size() == 1,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
    )
    _slot_gemv_ptr_kernel[grid1](
        token_states,
        weight_ptrs["up_ptrs"],
        weight_ptrs["up_scale_ptrs"],
        up,
        token_ids,
        local_eids,
        I,
        hidden,
        token_states.stride(0),
        up_ref.stride(0),
        up_ref.stride(1),
        up_scale_ref.stride(0),
        up_scale_ref.stride(1),
        up.stride(0),
        up_scale_ref.element_size() == 1,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
    )

    gate_f = gate.float()
    up_f = up.float()
    if swiglu_limit and swiglu_limit > 0:
        gate_f = torch.clamp(gate_f, max=swiglu_limit)
        up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
    activated = (F.silu(gate_f) * up_f).to(dtype).contiguous()

    down = torch.empty(num_slots, hidden, dtype=dtype, device=device)
    slot_ids = torch.arange(num_slots, device=device, dtype=torch.int32)
    grid2 = lambda meta: (num_slots, triton.cdiv(hidden, meta["BLOCK_N"]))
    _slot_gemv_ptr_kernel[grid2](
        activated,
        weight_ptrs["down_ptrs"],
        weight_ptrs["down_scale_ptrs"],
        down,
        slot_ids,
        local_eids,
        hidden,
        I,
        activated.stride(0),
        down_ref.stride(0),
        down_ref.stride(1),
        down_scale_ref.stride(0),
        down_scale_ref.stride(1),
        down.stride(0),
        down_scale_ref.element_size() == 1,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
    )

    valid_mask = valid.unsqueeze(1).to(torch.float32)
    weights = topk_weights.reshape(-1).unsqueeze(1).to(torch.float32)
    weighted = down.float() * weights * valid_mask
    return weighted.view(G, topk, hidden).sum(dim=1)


def v4_grouped_mxfp4_moe_forward_3d_ptrs(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    import torch.nn.functional as F

    from batchgen.moe.mxfp4_grouped_gemm import (
        gather_from_3d_expert_layout,
        grouped_mxfp4_gemm_3d,
        reshape_to_3d_expert_layout,
    )

    token_states = token_states.contiguous()
    G, hidden = token_states.shape
    topk = topk_indices.shape[1]
    refs = (
        weight_ptrs["gate_weight_ref"],
        weight_ptrs["gate_scale_ref"],
        weight_ptrs["up_weight_ref"],
        weight_ptrs["up_scale_ref"],
        weight_ptrs["down_weight_ref"],
        weight_ptrs["down_scale_ref"],
    )
    scale_refs = refs[1::2]
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    for scale in scale_refs:
        if scale.element_size() != 1:
            raise ValueError(
                "3D grouped V4 MoE currently requires 1-byte E8M0/uint8 scales"
            )
        if scale.dtype != torch.uint8 and scale.dtype != e8m0_dtype:
            raise ValueError("3D grouped V4 MoE scale dtype must be uint8/E8M0")
    if hidden % 32 != 0 or weight_ptrs["gate_weight_ref"].shape[0] % 32 != 0:
        raise ValueError(
            "3D grouped V4 MoE requires hidden/intermediate divisible by 32"
        )

    flat_global = topk_indices.reshape(-1)
    valid = (flat_global >= owned_start) & (
        flat_global < owned_start + owned_count
    )
    if not bool(valid.any()):
        return torch.zeros(
            G, hidden, dtype=torch.float32, device=token_states.device
        )

    local_eids = (flat_global[valid] - owned_start).to(torch.int64)
    token_ids = (
        torch.arange(G, device=token_states.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(G, topk)
        .reshape(-1)[valid]
    )
    routing_weights = topk_weights.reshape(-1)[valid]
    sorted_eids, order = torch.sort(local_eids)
    sorted_token_ids = token_ids[order]
    sorted_weights = routing_weights[order]
    sorted_hidden = token_states[sorted_token_ids]
    expert_counts = torch.bincount(sorted_eids, minlength=owned_count).to(
        torch.int32
    )
    max_expert_tokens = int(expert_counts.max().item())
    intermediate = int(weight_ptrs["gate_weight_ref"].shape[0])
    max_3d_elements = int(
        torch.tensor(
            [
                owned_count * max_expert_tokens * hidden,
                owned_count * max_expert_tokens * intermediate,
            ],
            device=token_states.device,
        )
        .max()
        .item()
    )
    max_3d_bytes = max_3d_elements * token_states.element_size()
    max_allowed_bytes = int(
        torch.cuda.get_device_properties(token_states.device).total_memory
        * 0.10
    )
    if max_3d_bytes > max_allowed_bytes:
        raise RuntimeError(
            "3D grouped V4 MoE padding would allocate too much memory: "
            f"{max_3d_bytes / (1024**3):.2f} GiB"
        )

    hidden_3d, _ = reshape_to_3d_expert_layout(
        sorted_hidden, expert_counts, owned_count
    )
    I = intermediate
    gate_3d = grouped_mxfp4_gemm_3d(
        hidden_3d,
        weight_ptrs["gate_ptrs"],
        weight_ptrs["gate_scale_ptrs"],
        expert_counts,
        I,
        weight_ptrs["gate_weight_ref"],
        weight_ptrs["gate_scale_ref"],
    )
    up_3d = grouped_mxfp4_gemm_3d(
        hidden_3d,
        weight_ptrs["up_ptrs"],
        weight_ptrs["up_scale_ptrs"],
        expert_counts,
        I,
        weight_ptrs["up_weight_ref"],
        weight_ptrs["up_scale_ref"],
    )
    gate_f = gate_3d.float()
    up_f = up_3d.float()
    if swiglu_limit and swiglu_limit > 0:
        gate_f = torch.clamp(gate_f, max=swiglu_limit)
        up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
    intermediate_3d = (F.silu(gate_f) * up_f).to(token_states.dtype)
    output_3d = grouped_mxfp4_gemm_3d(
        intermediate_3d,
        weight_ptrs["down_ptrs"],
        weight_ptrs["down_scale_ptrs"],
        expert_counts,
        hidden,
        weight_ptrs["down_weight_ref"],
        weight_ptrs["down_scale_ref"],
    )
    sorted_output = gather_from_3d_expert_layout(
        output_3d, expert_counts, int(sorted_hidden.shape[0])
    )
    output = torch.zeros(
        G, hidden, dtype=torch.float32, device=token_states.device
    )
    output.scatter_add_(
        0,
        sorted_token_ids.unsqueeze(-1).expand(-1, hidden),
        sorted_output.float() * sorted_weights.float().unsqueeze(-1),
    )
    return output
