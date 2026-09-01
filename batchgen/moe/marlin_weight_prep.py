"""Marlin weight preprocessing: convert K2.5 INT4 / K3 MXFP4 → Marlin packed format."""

import logging
import numpy as np
import torch

from batchgen.moe.mxfp4_oracle_vector import (
    MXFP4_E2M1_LUT,
    MXFP4_E8M0_BIAS,
    MXFP4_FORBIDDEN_SCALE_BYTES,
    MXFP4_GROUP_SIZE,
    MXFP4_LOW_NIBBLE_IS_EVEN_K,
    check_dequant_fn,
)

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


# ============================================================================
# K3 MXFP4 (E2M1 nibbles + E8M0 uint8 scales) → Marlin converter — task #34
#
# The Marlin tile permutation moves opaque 4-bit codes; it performs NO
# arithmetic on them, so E2M1 codes ride through the exact machinery the K2.5
# INT4 production path uses. The ONLY format-specific pieces are:
#   (a) the source nibble-extraction convention (low nibble = even K index —
#       frozen against the compressed-tensors oracle, see mxfp4_oracle_vector),
#   (b) E8M0 uint8 scale handling (index-permute only; optional EXACT bf16
#       materialization via bit shift — never a value cast),
#   (c) the in-kernel decode (dequant_e2m1 + SiTU, marlin_grouped_gemm.cu).
#
# FORBIDDEN per the 2026-08-04 POIS decision ledger (HARD-FAIL policy):
#   - any decode→requantize path (E2M1 magnitudes are non-uniform; INT4+scale
#     re-quantization is a second lossy quantization). Only nibble
#     rearrangement is allowed here. Contrast convert_int4_to_marlin above,
#     which REQUANTIZES and must never be extended to MXFP4.
#   - .to(torch.float16)/.to(float) on scale BYTES or scale VALUES
#     (fp16 overflows at byte >= 143; byte-value casts are garbage).
#   - silent clamping of E8M0 edge bytes 0x00/0xFF: the repack RAISES instead.
# ============================================================================

_mxfp4_convention_verified = False


def _unpack_mxfp4_nibbles(weight_packed: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Unpack compressed-tensors mxfp4-pack-quantized [N, K//2] uint8 → [N, K] codes.

    Convention (frozen, oracle-verified): low nibble of byte j = K index 2j,
    high nibble = K index 2j+1. Because int32 little-endian byte order composes
    with intra-byte low-first, this is bit-identical to the K2.5 raw INT4
    int32 unpacking (nibble i of int32 word w = K index 8w+i) — so we view the
    uint8 buffer as int32 and reuse the exact production unpack loop.

    Returns [N, K] int32 with values 0..15 (raw E2M1 codes, NOT decoded).
    """
    if not MXFP4_LOW_NIBBLE_IS_EVEN_K:
        # The int32-view shortcut below encodes low-first. If the frozen verdict
        # ever changes, this function must be rewritten — refuse loudly.
        raise ValueError(
            "MXFP4_LOW_NIBBLE_IS_EVEN_K is no longer True; _unpack_mxfp4_nibbles "
            "hard-codes the low-nibble-first convention and must be updated.")
    packed_i32 = weight_packed.contiguous().view(N, K // 2).view(torch.int32)  # [N, K//8]
    unpacked = torch.empty(N, K // 8, 8, dtype=torch.int32, device=weight_packed.device)
    for i in range(8):
        unpacked[:, :, i] = (packed_i32 >> (i * 4)) & 0xF
    return unpacked.view(N, K)


def _mxfp4_dequant_via_unpack(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequant path built on THIS module's unpack helper (used by the R8 self-check)."""
    N, half_k = packed.shape
    K = half_k * 2
    codes = _unpack_mxfp4_nibbles(packed, K, N)
    lut = torch.tensor(list(MXFP4_E2M1_LUT) + [-v for v in MXFP4_E2M1_LUT],
                       dtype=torch.float32)
    vals = lut[codes.long()]
    exps = (scales.to(torch.int32) - MXFP4_E8M0_BIAS).repeat_interleave(
        MXFP4_GROUP_SIZE, dim=-1)
    return torch.ldexp(vals, exps).to(torch.bfloat16)


def _verify_mxfp4_convention() -> None:
    """R8: one-time self-check of the nibble convention against the frozen
    real-checkpoint oracle vector. Raises if the module's unpack logic ever
    diverges from the recorded compressed-tensors verdict."""
    global _mxfp4_convention_verified
    if _mxfp4_convention_verified:
        return
    check_dequant_fn(_mxfp4_dequant_via_unpack)
    _mxfp4_convention_verified = True


def mxfp4_scale_e8m0_to_bf16(scale_u8: torch.Tensor) -> torch.Tensor:
    """EXACT E8M0 uint8 → bf16 conversion: 2^(e8 - 127) for e8 in [1, 254].

    bf16 layout is 1s/8e/7m, so the bit pattern uint16(e8) << 7 IS the value
    2^(e8-127) exactly — no rounding anywhere. This is a bit shift, not a
    value cast; power-of-two scales times E2M1 magnitudes (<=2 significant
    bits) stay exact in bf16 through the kernel's scale_op.

    Edge bytes RAISE (hard-fail policy): 0x00 would need the bf16 subnormal
    0x0040 (not e8<<7) and 0xFF is NaN per OCP MX — neither occurs in K3 data.
    """
    if scale_u8.dtype != torch.uint8:
        raise ValueError(
            f"mxfp4_scale_e8m0_to_bf16 expects uint8 E8M0 bytes, got {scale_u8.dtype}")
    n_bad = int(((scale_u8 == MXFP4_FORBIDDEN_SCALE_BYTES[0]) |
                 (scale_u8 == MXFP4_FORBIDDEN_SCALE_BYTES[1])).sum())
    if n_bad:
        raise ValueError(
            f"E8M0 edge byte 0x00/0xFF present (count={n_bad}, observed byte range "
            f"[{int(scale_u8.min())}, {int(scale_u8.max())}]): outside the validated "
            f"exact window [1, 254]. Edge semantics per the compressed-tensors "
            f"verdict (0x00 -> 2^-127, 0xFF -> inf/NaN) — refusing, no silent clamp.")
    return (scale_u8.to(torch.int16) << 7).view(torch.bfloat16)


def _check_mxfp4_repack_contract(
    weight_packed: torch.Tensor, weight_scale: torch.Tensor, K: int, N: int,
) -> None:
    """Hard-fail contract checks R1–R7 (ValueError, never assert)."""
    # R1
    if weight_packed.dtype != torch.uint8:
        raise ValueError(
            f"MXFP4 weight_packed must be uint8 [N, K//2], got {weight_packed.dtype} "
            f"{tuple(weight_packed.shape)}")
    # R2
    if weight_scale.dtype != torch.uint8:
        raise ValueError(
            f"MXFP4 weight_scale must be uint8 E8M0 [N, K//32], got {weight_scale.dtype} "
            f"— bf16 scales mean an INT4 checkpoint; use repack_int4_to_marlin_gs32")
    # R3
    if tuple(weight_packed.shape) != (N, K // 2):
        raise ValueError(
            f"MXFP4 packed dim != K//2: got {tuple(weight_packed.shape)}, "
            f"expected ({N}, {K // 2}) for K={K}")
    # R4
    if K % MXFP4_GROUP_SIZE != 0 or weight_scale.shape[-1] != K // MXFP4_GROUP_SIZE:
        raise ValueError(
            f"MXFP4 scale groups != K/32: got {tuple(weight_scale.shape)}, expected "
            f"({N}, {K // MXFP4_GROUP_SIZE}) — group size 32 is the only supported "
            f"MXFP4 group (kernel GROUP_BLOCKS=2)")
    # R5
    if weight_scale.shape[0] != weight_packed.shape[0]:
        raise ValueError(
            f"N mismatch packed vs scale: {weight_packed.shape[0]} vs "
            f"{weight_scale.shape[0]}")
    # R6 (K%16 for marlin tiles — already subsumed by R4's K%32, kept as
    #     defense in depth for the perm math; N%64 because the weight/scale
    #     permutations act on 64-column blocks)
    if K % 16 != 0 or N % 64 != 0:
        raise ValueError(
            f"Marlin tiling requires K%16==0 and N%64==0 (perm block = 64 N-cols), "
            f"got K={K}, N={N}")
    # R7
    n_bad = int(((weight_scale == MXFP4_FORBIDDEN_SCALE_BYTES[0]) |
                 (weight_scale == MXFP4_FORBIDDEN_SCALE_BYTES[1])).sum())
    if n_bad:
        raise ValueError(
            f"E8M0 edge byte 0x00/0xFF present in weight_scale (count={n_bad}, "
            f"observed byte range [{int(weight_scale.min())}, {int(weight_scale.max())}]): "
            f"outside the validated exact window; refusing per the hard-fail policy "
            f"(no silent clamp).")


def repack_mxfp4_to_marlin_gs32(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    K: int, N: int,
    emit_scale: str = "e8m0",
) -> tuple:
    """Repack K3 MXFP4 gs=32 → Marlin tile layout. PURE NIBBLE REARRANGEMENT.

    No value is decoded, quantized, or clamped anywhere in this function.
    E2M1 codes move as opaque 4-bit fields through the identical permutation
    machinery the K2.5 INT4 production path uses (get_weight_perm(4) +
    _marlin_pack_weights); E8M0 scale BYTES are index-permuted only.

    Args:
        weight_packed: [N, K//2] uint8 (compressed-tensors mxfp4-pack-quantized;
            low nibble = even K index — frozen oracle verdict)
        weight_scale: [N, K//32] uint8 E8M0
        K: in_features (K3: 3584 for w1/w3, 3072 for w2)
        N: out_features (K3: 3072 for w1/w3, 3584 for w2)
        emit_scale:
            "e8m0" — carry E8M0 uint8 bytes through (index-permuted only).
                This is the K3 Marlin kernel boundary format.
            "bf16" — materialize the EXACT bf16 values 2^(e8-127) here
                (bit shift, provably lossless; see mxfp4_scale_e8m0_to_bf16).
                Retained for conversion parity and legacy test references.

    Returns:
        marlin_qw: [K//16, N*2] int32 — same nibbles, Marlin tile order
                   (byte count unchanged vs source)
        marlin_s:  [K//32, N] uint8 (emit_scale="e8m0") or bf16 ("bf16"),
                   Marlin scale-permuted
    """
    _check_mxfp4_repack_contract(weight_packed, weight_scale, K, N)
    _verify_mxfp4_convention()  # R8

    # Step 1: unpack nibbles [N, K] (raw codes 0-15, NOT decoded)
    q_w_nk = _unpack_mxfp4_nibbles(weight_packed, K, N)

    # Step 2: transpose to [K, N] for Marlin layout
    q_w = q_w_nk.t().contiguous()

    # Step 3: Marlin tile permutation + packing (value-agnostic)
    perm = get_weight_perm(4)
    marlin_qw = _marlin_pack_weights(q_w, K, N, perm)

    # Step 4: scales — transpose [N, K//32] → [K//32, N], index-permute.
    # NOTE: no dtype conversion here (contrast the INT4 path's .to(float16),
    # which would silently corrupt E8M0 bytes — see module banner).
    s = weight_scale.t().contiguous()
    marlin_s = _marlin_permute_scales(s, K, N, MXFP4_GROUP_SIZE)
    if emit_scale == "bf16":
        marlin_s = mxfp4_scale_e8m0_to_bf16(marlin_s)
    elif emit_scale != "e8m0":
        raise ValueError(f"emit_scale must be 'e8m0' or 'bf16', got {emit_scale!r}")

    return marlin_qw, marlin_s


def repack_mxfp4_w13_to_marlin_gs32(
    w1_packed: torch.Tensor, w1_scale: torch.Tensor,
    w3_packed: torch.Tensor, w3_scale: torch.Tensor,
    K: int, N: int,
    emit_scale: str = "e8m0",
) -> tuple:
    """Fused w1‖w3 repack: two complete Marlin tensors, storage-adjacent.

    The fused _s1 kernel takes SEPARATE gate/up pointer arrays and derives its
    B row stride from the per-branch prob_n — a column slice of one wide
    [K, 2N] marlin tensor has the WRONG stride and can never be passed. The
    checkpoint-coordinate concat [2N, K//2] is legal (packing is along K) but
    is only the debug-reference layout. Production "fusion" = adjacency:

        qw[0] = gate (w1) marlin tensor, qw[1] = up (w3) marlin tensor
        up_qw_ptr = gate_qw_ptr + K//16 * N*2 * 4 bytes  (qw is contiguous)

    Gate-first order matches the HF reference (gate_up = cat([w1, w3])) and is
    SILENT if swapped — the GPU SiTU parity mutation test is the guard.

    Args (per branch): same contracts as repack_mxfp4_to_marlin_gs32.
        K = 3584, N = 3072 for K3 routed experts.

    Returns:
        qw: [2, K//16, N*2] int32 contiguous (index 0 = gate/w1, 1 = up/w3)
        s:  [2, K//32, N] uint8 or bf16 contiguous (same order)
    """
    if tuple(w1_packed.shape) != tuple(w3_packed.shape):
        raise ValueError(
            f"w1/w3 packed shape mismatch: {tuple(w1_packed.shape)} vs "
            f"{tuple(w3_packed.shape)}")
    if tuple(w1_scale.shape) != tuple(w3_scale.shape):
        raise ValueError(
            f"w1/w3 scale shape mismatch: {tuple(w1_scale.shape)} vs "
            f"{tuple(w3_scale.shape)}")
    gate_qw, gate_s = repack_mxfp4_to_marlin_gs32(w1_packed, w1_scale, K, N, emit_scale)
    up_qw, up_s = repack_mxfp4_to_marlin_gs32(w3_packed, w3_scale, K, N, emit_scale)
    qw = torch.stack([gate_qw, up_qw], dim=0).contiguous()
    s = torch.stack([gate_s, up_s], dim=0).contiguous()
    return qw, s


# --- exact inverse (round-trip proof; local copies of the inverse perms so
#     this module stays importable without compiled batchgen_kernels, unlike
#     marlin_transform.py which imports the CUDA extension at module level) ---

def _inverse_weight_perm(num_bits: int = 4) -> torch.Tensor:
    perm = get_weight_perm(num_bits)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm))
    return inv_perm


def _inverse_scale_perm() -> list:
    scale_perm, _ = _get_scale_perms()
    inv = [0] * len(scale_perm)
    for i, p in enumerate(scale_perm):
        inv[p] = i
    return inv


def marlin_mxfp4_to_raw_cpu(
    marlin_qw: torch.Tensor,
    marlin_s: torch.Tensor,
    K: int, N: int,
) -> tuple:
    """Exact inverse of repack_mxfp4_to_marlin_gs32 (pure rearrangement proof).

    Accepts marlin_s as uint8 (e8m0 passthrough) or bf16 (exact expansion;
    inverted losslessly via bits >> 7).

    Returns:
        raw_packed: [N, K//2] uint8 — must be BYTE-IDENTICAL to the source
        raw_scale:  [N, K//32] uint8
    """
    if tuple(marlin_qw.shape) != (K // GPTQ_MARLIN_TILE, N * 2):
        raise ValueError(
            f"Expected marlin_qw [{K // GPTQ_MARLIN_TILE}, {N * 2}], "
            f"got {tuple(marlin_qw.shape)}")

    # Weights: unpack int32 nibbles → inverse tile perm → undo tile transpose
    flat = marlin_qw.reshape(-1)
    unpacked = torch.empty(flat.numel(), 8, dtype=torch.int32, device=flat.device)
    for i in range(8):
        unpacked[:, i] = (flat >> (i * 4)) & 0xF
    q_marlin = unpacked.view(K // GPTQ_MARLIN_TILE, N * GPTQ_MARLIN_TILE)
    inv_perm = _inverse_weight_perm(4).to(flat.device)
    q_tiled = q_marlin.reshape(-1, inv_perm.numel())[:, inv_perm].reshape(q_marlin.shape)
    q_tiled = q_tiled.reshape(K // GPTQ_MARLIN_TILE, N // GPTQ_MARLIN_TILE,
                              GPTQ_MARLIN_TILE, GPTQ_MARLIN_TILE)
    q_raw_kn = q_tiled.permute(0, 2, 1, 3).reshape(K, N)
    q_raw_nk = q_raw_kn.t().contiguous()  # [N, K]

    # Pack back to uint8, low nibble = even K index
    raw_packed = (q_raw_nk[:, 0::2] | (q_raw_nk[:, 1::2] << 4)).to(torch.uint8)

    # Scales
    if marlin_s.dtype == torch.bfloat16:
        s_kn = (marlin_s.view(torch.int16) >> 7).to(torch.uint8)
    elif marlin_s.dtype == torch.uint8:
        s_kn = marlin_s
    else:
        raise ValueError(
            f"marlin_s must be uint8 (e8m0) or bf16 (exact expansion), "
            f"got {marlin_s.dtype}")
    inv_scale_perm = _inverse_scale_perm()
    s_inv = s_kn.reshape(-1, len(inv_scale_perm))[:, inv_scale_perm]
    raw_scale = s_inv.reshape(-1, N).t().contiguous()  # [N, K//32]

    return raw_packed, raw_scale
