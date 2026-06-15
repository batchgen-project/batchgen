"""Grouped MXFP4 MoE for DeepSeek-V4-Flash decode on Blackwell sm120.

Replaces the per-expert Python loop (`DeepSeekV4FlashMoE._run_owned_experts`)
with a 3D grouped MXFP4 GEMM over this rank's resident owned experts. This is
the fastest grouped path measured on Blackwell sm120 (moe_expert_loop ~19 ms at
b256 / ~11 ms at b128, vs ~92 ms for the slot-GEMV path and ~75 ms for the
FlashInfer native-FP4 path; see .sisyphus/blackwell timing CSVs). Those two
alternative paths were removed; this 3D path is the sole grouped implementation.

V4-specific behavior:
  - Expert-parallel owned range: topk indices are GLOBAL [0, total_experts); the
    resident weight pointers hold only this rank's owned experts. Slots outside
    the owned range contribute zero (mirrors `_run_owned_experts`, which only
    runs owned experts and relies on a later all_reduce to combine ranks).
  - V4 activation is silu(gate)*up with optional clamp to swiglu_limit
    (model.py expert forward), NOT OpenAI-style GLU.
  - Routing weight is applied to the down-projection output then summed over
    topk. This is algebraically identical to V4 applying it to the activated
    intermediate (w2 is linear).
"""

from __future__ import annotations

import torch


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


def v4_grouped_mxfp4_moe_forward_3d_ptrs(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    # NOT QAT-FAITHFUL. This path runs grouped_mxfp4_gemm_3d, which dequantizes
    # the FP4 expert weights to bf16 and matmuls against bf16 (non-quantized)
    # activations. The official model act-quantizes the activation to fp8
    # (block-128, ue8m0) before each fp4 GEMM, so this introduces ~5e-2 rel
    # error per GEMM vs the QAT path (test_grouped_moe_kernel_vs_per_expert_parity
    # measures cos~0.9988). Kept as the FAST throughput path; for character-exact
    # output use v4_grouped_mxfp4_moe_forward_qat (BATCHGEN_V4_QAT_MOE=1).
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


def _qat_fp4_linear(x, weight, scale, kern):
    # Bit-exact V4 FP4 linear: act-quant x to fp8 (block-128, ue8m0), then
    # fp4_gemm against the e8m0-scaled FP4 weight. Mirrors model._qat_linear
    # exactly so the grouped path matches the per-expert/official numerics.
    fp4_dtype = torch.float4_e2m1fn_x2
    if weight.dtype in (torch.uint8, torch.int8):
        weight = weight.view(fp4_dtype)
    wscale = (
        scale
        if scale.dtype == torch.float8_e8m0fnu
        else scale.view(torch.float8_e8m0fnu)
        if scale.dtype == torch.uint8
        else scale.to(torch.float32).to(torch.float8_e8m0fnu)
    )
    x2d = x.reshape(-1, x.shape[-1])
    if x2d.dtype != torch.bfloat16:
        x2d = x2d.to(torch.bfloat16)
    xq, xs = kern.act_quant(x2d, 128, "ue8m0", torch.float8_e8m0fnu)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        out = kern.fp4_gemm(xq, xs, weight, wscale, torch.float8_e8m0fnu)
    finally:
        torch.set_default_dtype(prev)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def v4_grouped_mxfp4_moe_forward_qat(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """QAT-faithful grouped MoE: per-owned-expert official act_quant + fp4_gemm.

    Numerically matches the per-expert reference (DeepSeekV4FlashExpertPlaceholder
    under BATCHGEN_V4_QAT_LINEAR) and the official Expert, unlike
    v4_grouped_mxfp4_moe_forward_3d_ptrs which dequantizes weights to bf16.
    Routing/combine semantics are identical to that function. Per-128 K
    requirement: hidden and intermediate must be divisible by 128.
    """
    import torch.nn.functional as F

    from batchgen.models.deepseek.deepseekv4_flash.model import (
        _v4_official_kernels,
    )

    kern = _v4_official_kernels()
    refs = weight_ptrs["expert_refs"]
    token_states = token_states.contiguous()
    G, hidden = token_states.shape
    topk = topk_indices.shape[1]

    output = torch.zeros(
        G, hidden, dtype=torch.float32, device=token_states.device
    )

    flat_global = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    token_for_slot = (
        torch.arange(G, device=token_states.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(G, topk)
        .reshape(-1)
    )

    for local_e in range(owned_count):
        global_e = owned_start + local_e
        slot_mask = flat_global == global_e
        if not bool(slot_mask.any()):
            continue
        tok_idx = token_for_slot[slot_mask]
        w = flat_weights[slot_mask]
        rw = refs[local_e]
        x = token_states[tok_idx]

        gate = _qat_fp4_linear(x, rw["w1.weight"], rw["w1.scale"], kern).float()
        up = _qat_fp4_linear(x, rw["w3.weight"], rw["w3.scale"], kern).float()
        if swiglu_limit and swiglu_limit > 0:
            gate = torch.clamp(gate, max=swiglu_limit)
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        activated = F.silu(gate) * up
        activated = activated * w.float().unsqueeze(-1)
        down = _qat_fp4_linear(
            activated.to(token_states.dtype),
            rw["w2.weight"],
            rw["w2.scale"],
            kern,
        )
        output.index_add_(0, tok_idx, down.float())

    return output
