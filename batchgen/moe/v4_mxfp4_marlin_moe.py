"""MXFP4 (E8M0 scales) MoE quantization using the Marlin backend.

Ported from sglang ``Mxfp4MarlinMoEMethod``.

Difference vs ``marlin_grouped_moe.py``
---------------------------------------
``marlin_grouped_moe.py``
    INT4 W4A16 for Kimi-K2.5 checkpoints.  Weights are INT4 nibbles with
    BF16 per-group scales (gs=32).  Uses ``batchgen_kernels._C_marlin_grouped_gemm``
    which has INT4 offset-dequant (nibble - 8) baked into the kernel.

This module (``v4_mxfp4_marlin_moe.py``)
    MXFP4 for GPT-OSS-style checkpoints.  Weights are FP4 (E2M1) nibbles with
    E8M0 exponent scales (gs=32).  Requires a Marlin kernel that understands
    the ``float4_e2m1f`` scalar type (e.g. ``sgl_kernel.moe_wna16_marlin_gemm``).

Both share the same Marlin tile layout for packed weights, but differ in:

1. **Value encoding** -- INT4 offset (nibble - 8) vs FP4 lookup table.
2. **Scale format** -- BF16 group scales vs E8M0 exponent scales.
3. **Kernel dequant** -- INT4 linear vs FP4 non-linear.

The weight-preparation pipeline (repack + scale permutation) is fully ported
and self-contained.  The forward path currently uses a PyTorch reference
(dequant + matmul) because batchgen's Marlin kernel does not yet support
MXFP4 scalar types.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.moe.marlin_weight_prep import (
    _marlin_pack_weights,
    _marlin_permute_scales,
    get_weight_perm,
    INT4_GROUP_SIZE,
)
from batchgen.quantization.mxfp4 import FP4_LOOKUP_TABLE, MXFP4_BLOCK_SIZE

logger = logging.getLogger(__name__)

MXFP4_GROUP_SIZE = 32


def _normalize_scale_tensor(
    scales: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Normalise E8M0 scale tensor to *target_dtype* numerical values.

    Checkpoint loaders may store E8M0 exponents in various container dtypes.
    This function converts them all to the numerical 2**e representation in
    *target_dtype*.
    """
    if scales.dtype == torch.uint8:
        return scales.view(torch.float8_e8m0fnu).to(target_dtype)
    if scales.dtype == torch.int8:
        return (
            scales.view(torch.uint8).view(torch.float8_e8m0fnu).to(target_dtype)
        )
    if scales.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return scales.to(target_dtype)
    if (
        hasattr(torch, "float8_e8m0fnu")
        and scales.dtype == torch.float8_e8m0fnu
    ):
        return scales.to(target_dtype)
    raise TypeError(f"Unsupported MXFP4 scale dtype for Marlin: {scales.dtype}")


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    input_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Post-process Marlin-permuted scales for MXFP4 kernel consumption.

    1. Reorder columns for the Marlin MXFP4 kernel's expected access pattern
       (swap pairs within groups of 4 when using 16-bit activations).
    2. Convert to ``float8_e8m0fnu`` (the native E8M0 exponent type).
    3. Optionally bias exponents for FP8 activation path.
    """
    if input_dtype is None or input_dtype.itemsize == 2:
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
            marlin_scales.size(0), -1
        )
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        assert marlin_scales.max() <= 249
        marlin_scales = marlin_scales + 6  # exponent_bias(fp4->fp8) = 2^3 - 2^1
        marlin_scales = marlin_scales.view(torch.float8_e8m0fnu)
    return marlin_scales


def _unpack_mxfp4_to_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``[..., K//2]`` uint8 MXFP4 tensor to ``[..., K]`` int32 nibbles."""
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    out = torch.empty(
        *packed.shape[:-1],
        packed.shape[-1] * 2,
        dtype=torch.int32,
        device=packed.device,
    )
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _repack_mxfp4_weight_for_marlin(
    weight: torch.Tensor,
    num_experts: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    """Repack MXFP4 weight ``[E, N, K//2]`` uint8 -> Marlin packed ``[E, ...]`` int32.

    Uses batchgen's ``_marlin_pack_weights`` (CPU/numpy) which is functionally
    equivalent to sglang's ``gptq_marlin_repack`` C++ kernel.
    """
    assert (
        weight.shape == (num_experts, size_n, size_k // 2)
    ), f"Expected [{num_experts}, {size_n}, {size_k // 2}], got {list(weight.shape)}"
    perm = get_weight_perm(4)
    result_list = []
    for i in range(num_experts):
        nibbles_nk = _unpack_mxfp4_to_nibbles(weight[i])
        nibbles_kn = nibbles_nk.t().contiguous()
        marlin_qw = _marlin_pack_weights(nibbles_kn, size_k, size_n, perm)
        result_list.append(marlin_qw)
    return torch.stack(result_list)


def _permute_mxfp4_scales_for_marlin(
    scales: torch.Tensor,
    num_experts: int,
    size_n: int,
    size_k: int,
    param_dtype: torch.dtype,
) -> torch.Tensor:
    """Permute MXFP4 E8M0 scales ``[E, N, K//32]`` -> Marlin layout ``[E, ...]``.

    Normalises to *param_dtype*, applies Marlin scale permutation, then
    converts to E8M0 via ``mxfp4_marlin_process_scales``.
    """
    scales = _normalize_scale_tensor(scales, param_dtype)
    result_list = []
    for i in range(num_experts):
        s = scales[i].T.contiguous()
        s_perm = _marlin_permute_scales(s, size_k, size_n, MXFP4_GROUP_SIZE)
        s_e8m0 = mxfp4_marlin_process_scales(s_perm, input_dtype=param_dtype)
        result_list.append(s_e8m0)
    return torch.stack(result_list)


def prepare_moe_mxfp4_layer_for_marlin(layer: nn.Module) -> None:
    """Transform MXFP4 MoE layer weights into Marlin-compatible format.

    Modifies *layer* in-place, replacing ``w13_weight``, ``w2_weight``,
    ``w13_weight_scale_inv``, and ``w2_weight_scale_inv`` with their
    Marlin-repacked equivalents.

    Expected input shapes (GPT-OSS convention):
        w13_weight:          [E, 2*intermediate, hidden//2]  uint8
        w2_weight:           [E, hidden, intermediate//2]    uint8
        w13_weight_scale_inv: [E, 2*intermediate, hidden//32] uint8/E8M0
        w2_weight_scale_inv:  [E, hidden, intermediate//32]   uint8/E8M0
    """
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = layer.w13_weight_scale_inv.data
    w2_scale = layer.w2_weight_scale_inv.data

    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2
    hidden_size = w13.shape[2] * 2

    param_dtype = getattr(layer, "orig_dtype", torch.bfloat16)

    w13_marlin = _repack_mxfp4_weight_for_marlin(
        w13,
        num_experts,
        intermediate_size * 2,
        hidden_size,
    )
    w2_marlin = _repack_mxfp4_weight_for_marlin(
        w2,
        num_experts,
        hidden_size,
        intermediate_size,
    )

    w13_scale_marlin = _permute_mxfp4_scales_for_marlin(
        w13_scale,
        num_experts,
        intermediate_size * 2,
        hidden_size,
        param_dtype,
    )
    w2_scale_marlin = _permute_mxfp4_scales_for_marlin(
        w2_scale,
        num_experts,
        hidden_size,
        intermediate_size,
        param_dtype,
    )

    layer.w13_weight = nn.Parameter(w13_marlin, requires_grad=False)
    layer.w2_weight = nn.Parameter(w2_marlin, requires_grad=False)
    layer.w13_weight_scale_inv = nn.Parameter(
        w13_scale_marlin, requires_grad=False
    )
    layer.w2_weight_scale_inv = nn.Parameter(
        w2_scale_marlin, requires_grad=False
    )

    device = w13_marlin.device
    layer.workspace = torch.zeros(64, dtype=torch.int32, device=device)


def mxfp4_dequant_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantise a single MXFP4 weight ``[N, K//2]`` uint8 -> ``[N, K]`` dtype.

    *scales* is ``[N, K//32]`` in uint8 (raw E8M0 bytes) or float.
    """
    device = packed.device
    fp4_table = FP4_LOOKUP_TABLE.to(device)

    nibbles = _unpack_mxfp4_to_nibbles(packed)
    values = fp4_table[nibbles.long()]

    if scales.dtype == torch.uint8:
        exponents = scales.to(torch.int32) - 127
    elif (
        hasattr(torch, "float8_e8m0fnu")
        and scales.dtype == torch.float8_e8m0fnu
    ):
        exponents = scales.view(torch.uint8).to(torch.int32) - 127
    else:
        scale_expanded = (
            scales.unsqueeze(-1)
            .expand(
                *scales.shape,
                MXFP4_BLOCK_SIZE,
            )
            .reshape(*scales.shape[:-1], scales.shape[-1] * MXFP4_BLOCK_SIZE)
        )
        if scale_expanded.shape[-1] > values.shape[-1]:
            scale_expanded = scale_expanded[..., : values.shape[-1]]
        return (values * scale_expanded.float()).to(dtype)

    exponents = exponents.clamp(min=-126, max=127)
    exp_expanded = (
        exponents.unsqueeze(-1)
        .expand(
            *exponents.shape,
            MXFP4_BLOCK_SIZE,
        )
        .reshape(*exponents.shape[:-1], exponents.shape[-1] * MXFP4_BLOCK_SIZE)
    )
    if exp_expanded.shape[-1] > values.shape[-1]:
        exp_expanded = exp_expanded[..., : values.shape[-1]]

    result = torch.ldexp(values, exp_expanded)
    return result.to(dtype)


def mxfp4_expert_mlp_ref(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    s_gate: torch.Tensor,
    w_up: torch.Tensor,
    s_up: torch.Tensor,
    w_down: torch.Tensor,
    s_down: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference single-expert MLP forward with MXFP4 weights.

    Args:
        x:      [M, K]       input activations (BF16)
        w_gate: [N, K//2]    packed FP4 gate weight
        s_gate: [N, K//32]   E8M0 gate scales
        w_up:   [N, K//2]    packed FP4 up weight
        s_up:   [N, K//32]   E8M0 up scales
        w_down: [K, N//2]    packed FP4 down weight
        s_down: [K, N//32]   E8M0 down scales
        dtype:  compute dtype

    Returns:
        [M, K] output activations
    """
    gate_w = mxfp4_dequant_weight(w_gate, s_gate, dtype)
    up_w = mxfp4_dequant_weight(w_up, s_up, dtype)
    down_w = mxfp4_dequant_weight(w_down, s_down, dtype)

    gate_out = x.to(dtype) @ gate_w.T
    up_out = x.to(dtype) @ up_w.T
    intermediate = F.silu(gate_out) * up_out
    output = intermediate @ down_w.T
    return output


class Mxfp4MarlinMoEMethod:
    """MXFP4 (E8M0 scales) MoE quantization method using the Marlin backend.

    Lifecycle:
        1. ``create_weights`` — allocate raw MXFP4 weight buffers on the layer.
        2. (loader fills ``layer.w13_weight``, etc.)
        3. ``process_weights_after_loading`` — repack to Marlin tile layout.
        4. ``forward_single_expert`` — reference forward for one expert.
    """

    def __init__(
        self, num_experts: int, hidden_size: int, intermediate_size: int
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def create_weights(self, layer: nn.Module, device: torch.device) -> None:
        """Allocate raw MXFP4 weight placeholders on *layer*."""
        E = self.num_experts
        N = self.intermediate_size
        K = self.hidden_size

        layer.w13_weight = nn.Parameter(
            torch.empty(E, 2 * N, K // 2, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        layer.w2_weight = nn.Parameter(
            torch.empty(E, K, N // 2, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        layer.w13_weight_scale_inv = nn.Parameter(
            torch.empty(
                E,
                2 * N,
                K // MXFP4_GROUP_SIZE,
                dtype=torch.uint8,
                device=device,
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale_inv = nn.Parameter(
            torch.empty(
                E, K, N // MXFP4_GROUP_SIZE, dtype=torch.uint8, device=device
            ),
            requires_grad=False,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Repack raw MXFP4 weights into Marlin tile layout.

        After this call the weight tensors on *layer* are in Marlin format
        and the original MXFP4 layout is discarded.
        """
        K = self.hidden_size
        N = self.intermediate_size

        if K % 64 != 0:
            raise RuntimeError(
                f"hidden_size={K} must be divisible by 64 for Marlin."
            )
        if N % 64 != 0:
            raise RuntimeError(
                f"intermediate_size={N} must be divisible by 64 for Marlin."
            )

        logger.info(
            "Preparing MXFP4 experts for Marlin backend "
            "(E=%d, N=%d, K=%d)...",
            self.num_experts,
            N,
            K,
        )
        prepare_moe_mxfp4_layer_for_marlin(layer)

    def forward_single_expert(
        self,
        x: torch.Tensor,
        expert_idx: int,
        w13_packed: torch.Tensor,
        w13_scales: torch.Tensor,
        w2_packed: torch.Tensor,
        w2_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Reference single-expert MLP forward using raw MXFP4 weights.

        This uses PyTorch dequant + matmul (no Marlin kernel).  Intended for
        correctness testing and as a fallback.

        Args:
            x:           [M, K]  BF16 input
            expert_idx:  which expert to run
            w13_packed:  [E, 2*N, K//2]  raw MXFP4 packed weights (gate+up)
            w13_scales:  [E, 2*N, K//32] raw E8M0 scales
            w2_packed:   [E, K, N//2]    raw MXFP4 packed weights (down)
            w2_scales:   [E, K, N//32]   raw E8M0 scales

        Returns:
            [M, K] output
        """
        N = self.intermediate_size
        e = expert_idx

        w_gate = w13_packed[e, :N, :]
        s_gate = w13_scales[e, :N, :]
        w_up = w13_packed[e, N:, :]
        s_up = w13_scales[e, N:, :]
        w_down = w2_packed[e]
        s_down = w2_scales[e]

        return mxfp4_expert_mlp_ref(
            x,
            w_gate,
            s_gate,
            w_up,
            s_up,
            w_down,
            s_down,
            dtype=x.dtype,
        )
