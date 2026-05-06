"""Marlin weight preprocessing: convert K2.5 INT4 → Marlin packed format."""

import logging
import numpy as np
import torch

GPTQ_MARLIN_TILE = 16
INT4_GROUP_SIZE = 32  # K2.5 checkpoint group size


# ============================================================================
# Marlin format utilities (from sglang)
# ============================================================================

def get_weight_perm(num_bits: int = 4):
    """Compute weight permutation for Marlin mma.sync m16n8k16 fragment layout."""
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7]) if num_bits == 4 else np.array([0, 2, 1, 3])
    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return torch.from_numpy(perm)


def _marlin_permute_weights(q_w, size_k, size_n, perm):
    """Permute quantized weight [K, N] for Marlin tile layout."""
    q_w = q_w.reshape((size_k // GPTQ_MARLIN_TILE, GPTQ_MARLIN_TILE,
                        size_n // GPTQ_MARLIN_TILE, GPTQ_MARLIN_TILE))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // GPTQ_MARLIN_TILE, size_n * GPTQ_MARLIN_TILE))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    return q_w


def _marlin_pack_weights(q_w, size_k, size_n, perm):
    """Permute and pack quantized weights [K, N] into Marlin int32 format."""
    q_w = _marlin_permute_weights(q_w, size_k, size_n, perm)
    pack_factor = 8  # 32 / 4
    q_np = q_w.cpu().numpy().astype(np.uint32)
    q_packed = np.zeros((q_np.shape[0], q_np.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_np[:, i::pack_factor] << (4 * i)
    return torch.from_numpy(q_packed.astype(np.int32)).to(q_w.device)


def _get_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def _marlin_permute_scales(s, size_k, size_n, group_size):
    """Permute scales for Marlin kernel access pattern."""
    scale_perm, scale_perm_single = _get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def _quantize_weights_uint4b8(w_fp16, group_size):
    """Quantize [K, N] FP16 weight to uint4b8 with given group_size.

    Returns: (w_ref [K,N] FP16, q_w [K,N] int, scales [K//gs, N] FP16)
    """
    size_k, size_n = w_fp16.shape
    device = w_fp16.device

    if group_size < size_k:
        w_grouped = w_fp16.reshape((-1, group_size, size_n)).permute(1, 0, 2)
        w_flat = w_grouped.reshape((group_size, -1))
    else:
        w_flat = w_fp16

    max_val = torch.max(w_flat, 0, keepdim=True).values
    min_val = torch.min(w_flat, 0, keepdim=True).values
    w_s = torch.max(abs(max_val / 7.0), abs(min_val / -8.0)).clamp(min=1e-10)

    w_q = torch.round(w_flat / w_s).int() + 8
    w_q = torch.clamp(w_q, 0, 15)
    w_ref = (w_q - 8).to(w_fp16.dtype) * w_s

    if group_size < size_k:
        def reshape_w(t):
            return t.reshape((group_size, -1, size_n)).permute(1, 0, 2).reshape((size_k, size_n)).contiguous()
        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()
    else:
        w_s = w_s.reshape((1, size_n)).contiguous()

    return w_ref.to(device), w_q.to(device), w_s.to(torch.float16).to(device)


# ============================================================================
# K2.5 INT4 → Marlin converter
# ============================================================================

def _dequantize_k25_int4(packed_int32, scales_bf16, K, N):
    """Dequantize K2.5 INT4 packed [N, K//8] int32 → [K, N] FP16.

    K2.5 format: 8 nibbles per int32, offset encoding (nibble - 8),
    group_size=32 BF16 scales.
    """
    assert packed_int32.shape == (N, K // 8), f"Expected [{N}, {K // 8}], got {packed_int32.shape}"
    device = packed_int32.device

    # Unpack 8 nibbles from each int32
    unpacked = torch.empty(N, K // 8, 8, dtype=torch.int32, device=device)
    for i in range(8):
        unpacked[:, :, i] = ((packed_int32 >> (i * 4)) & 0xF) - 8  # offset decode

    unpacked_flat = unpacked.view(N, K).float()  # [N, K]

    # Apply group scales
    n_groups = K // INT4_GROUP_SIZE
    unpacked_grouped = unpacked_flat.view(N, n_groups, INT4_GROUP_SIZE)
    scales_expanded = scales_bf16.float().unsqueeze(-1)  # [N, n_groups, 1]
    result = (unpacked_grouped * scales_expanded).view(N, K)

    # Transpose to [K, N] for Marlin (which expects [K, N] input)
    return result.t().to(torch.float16).contiguous()


def repack_int4_to_marlin_gs32(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    K: int, N: int,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Repack K2.5 INT4 gs=32 → Marlin tile layout WITHOUT requantization.

    Direct nibble rearrangement: K2.5 [N, K//8] → Marlin [K, N//8].
    Zero quantization error — only layout transformation.

    Args:
        weight_packed: [N, K//8] int32 (K2.5 checkpoint, 8 nibbles per int32)
        weight_scale: [N, K//32] BF16 (K2.5 group_size=32 scales)
        K: hidden dimension (in_features)
        N: intermediate dimension (out_features)
        compute_dtype: output scale dtype

    Returns:
        marlin_qw: [K, N//8] int32 Marlin-packed weight (same nibble values, different layout)
        marlin_s: [K//32, N] permuted scales in compute_dtype
    """
    assert weight_packed.shape == (N, K // 8), f"Expected [{N}, {K // 8}], got {weight_packed.shape}"
    device = weight_packed.device

    # Step 1: Unpack all nibbles to [N, K] uint4 values (0-15, NOT offset decoded)
    unpacked = torch.empty(N, K // 8, 8, dtype=torch.int32, device=device)
    for i in range(8):
        unpacked[:, :, i] = (weight_packed >> (i * 4)) & 0xF
    q_w_nk = unpacked.view(N, K)  # [N, K] nibble values 0-15

    # Step 2: Transpose to [K, N] for Marlin layout
    q_w = q_w_nk.t().contiguous()  # [K, N]

    # Step 3: Marlin tile permutation + packing (no quantization change)
    perm = get_weight_perm(4)
    marlin_qw = _marlin_pack_weights(q_w, K, N, perm)

    # Step 4: Transpose scales [N, K//32] → [K//32, N] and permute for Marlin
    marlin_s = weight_scale.t().contiguous().to(torch.float16)  # [K//32, N]
    marlin_s = _marlin_permute_scales(marlin_s, K, N, INT4_GROUP_SIZE)
    marlin_s = marlin_s.to(compute_dtype)

    return marlin_qw, marlin_s


def convert_int4_to_marlin(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    K: int, N: int,
    marlin_group_size: int = 128,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Convert K2.5 INT4 checkpoint weights → Marlin packed format.

    Args:
        weight_packed: [N, K//8] int32 (K2.5 checkpoint format)
        weight_scale: [N, K//32] BF16 (K2.5 group_size=32 scales)
        K: hidden dimension (in_features)
        N: intermediate dimension (out_features)
        marlin_group_size: target group size for Marlin (128 for optimal perf)
        compute_dtype: output scale dtype (bfloat16 for BF16 compute kernel)

    Returns:
        marlin_qw: packed int32 Marlin weight
        marlin_s: permuted scales in compute_dtype
    """
    # Step 1: Dequantize K2.5 INT4 (gs=32) → FP16 [K, N]
    w_fp16 = _dequantize_k25_int4(weight_packed, weight_scale, K, N)

    # Step 2: Re-quantize with Marlin group_size (128)
    w_ref, q_w, s_fp16 = _quantize_weights_uint4b8(w_fp16, marlin_group_size)

    # Step 3: Permute + pack into Marlin format
    perm = get_weight_perm(4)
    marlin_qw = _marlin_pack_weights(q_w, K, N, perm)

    # Step 4: Permute scales
    marlin_s = _marlin_permute_scales(s_fp16, K, N, marlin_group_size)

    # Convert scales to compute dtype
    marlin_s = marlin_s.to(compute_dtype)

    return marlin_qw, marlin_s
