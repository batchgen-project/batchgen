"""FROZEN MXFP4 oracle verdict for Kimi-K3 (task #34 gate; settled 2026-08-04).

Single source of truth for the MXFP4 packing convention. Production code
(marlin_weight_prep.repack_mxfp4_to_marlin_gs32) and the test suite both import
from here. Do NOT re-derive these facts; they were settled against an
independent oracle on a real K3 checkpoint tensor.

ORACLE: compressed-tensors 0.17.1 (pure-torch CPU reference for the
"mxfp4-pack-quantized" format K3 is stored in). Source citations (paths within
the installed package):
  - Nibble order:  compressors/nvfp4/helpers.py:72 (pack)
        packed = indices[:, 0] | (indices[:, 1] << 4)
    and compressors/nvfp4/helpers.py:96-100 (unpack): low = byte & 0x0F is the
    FIRST (even-K) element, high = byte >> 4 is the SECOND (odd-K) element.
  - FP4 code: sign-magnitude. bit3 (0x08) = sign, bits0-2 index the E2M1 LUT
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  (helpers.py:29-31, 103-108).
  - E8M0 scale: compressors/mx_utils.py:43-44
        scale_float = 2.0 ** (uint8 - 127)      # NO clamp, NO special cases
    => byte 0x00 -> 2^-127 (subnormal, valid); byte 0xFF -> 2^128 -> +inf in
    bf16 (OCP MX spec says 0xFF is NaN; compressed_tensors yields inf).
  - Group mapping: quantization/lifecycle/forward_helpers.py:149-151 —
    scale[r, j] covers contiguous columns [32*j, 32*j+32) of the unpacked K dim.
  - Format registration: compressors/mxfp4/base.py:25-26 (MXFP4PackedCompressor).

REAL-TENSOR VERDICT (language_model.model.layers.4.block_sparse_moe.experts.0.w1,
model-00005-of-000096.safetensors, packed U8[3072,1792] at abs offset 1268562960,
scale U8[3072,112] at abs offset 1274067984):
  - BatchGen batchgen/quantization/mxfp4.py::mxfp4_dequantize_reference is
    BIT-EXACT vs MXFP4PackedCompressor.decompress on all 11,010,048 elements
    (bf16 and fp32). Nibble convention low-first is CONFIRMED independently.
  - Swapped-nibble mutation mismatches 91.72% of elements -> test has teeth.
  - Scale-byte range observed: w1/w3 in [112, 122] (2^-15..2^-5), w2 in
    [119, 122]. ZERO occurrences of 0x00 or 0xFF in any of the three tensors.
  - The exponent clamp to [-126, 127] in quantization/mxfp4.py diverges from
    the oracle ONLY for bytes 0x00 (2x too large: 2^-126 vs 2^-127) and 0xFF
    (2^127 vs +inf). Neither byte occurs in K3 data; per the HARD-FAIL policy
    the marlin-MXFP4 load contract asserts scale bytes not in {0x00, 0xFF}
    instead of silently clamping (see repack_mxfp4_to_marlin_gs32).

Raw-source checksums (bytes sliced straight from the shard, md5 verified
remote==local):
  sha256(w1 packed bytes) = cf822517403f5ccb418150e10b303568f617f3099bea6dcc9af3a7b6a48e3501
  sha256(w1 scale  bytes) = b1cd3499f23097edbbc3f4dc3304c573dbdc49602404b90bdfb1b3dbb0b4ea92
"""
import torch

# --- packing convention constants (from compressed_tensors, independently verified) ---
MXFP4_LOW_NIBBLE_IS_EVEN_K = True          # byte & 0x0F -> element 2i; byte >> 4 -> 2i+1
MXFP4_SIGN_BIT_MASK = 0x08                  # bit 3 of each nibble
MXFP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MXFP4_E8M0_BIAS = 127                       # scale_float = 2 ** (uint8 - 127)
MXFP4_GROUP_SIZE = 32                       # scale[r, j] covers K columns [32j, 32j+32)
MXFP4_FORBIDDEN_SCALE_BYTES = (0x00, 0xFF)  # never occur in K3; loader must HARD-FAIL
# Observed on layer4/expert0 (w1, w2, w3): all scale bytes within [112, 122].

# Identifier stamped into converted-checkpoint metadata by the conversion seam
# (R11 in the task-#34 design). The kernel-side contract checks match on this.
MXFP4_MARLIN_FORMAT_ID = "mxfp4_marlin_gs32_v1"

# --- frozen real-data test vector: layer 4, expert 0, w1, row 0, group 0 ---
# 16 packed bytes + 1 E8M0 scale byte -> 32 bf16 outputs (exact bit patterns).
VEC_PACKED = bytes([0xAB, 0x4C, 0x18, 0x65, 0x2C, 0x91, 0x04, 0x39,
                    0x48, 0x94, 0x58, 0x33, 0x94, 0xB5, 0x5B, 0x30])
VEC_SCALE_BYTE = 0x79                       # 121 -> 2^-6 = 0.015625
VEC_EXPECTED_BF16_BITS = (
    0xBCC0, 0xBC80, 0xBD00, 0x3D00, 0x8000, 0x3C00, 0x3D40, 0x3D80,
    0xBD00, 0x3C80, 0x3C00, 0xBC00, 0x3D00, 0x0000, 0xBC00, 0x3CC0,
    0x8000, 0x3D00, 0x3D00, 0xBC00, 0x8000, 0x3D40, 0x3CC0, 0x3CC0,
    0x3D00, 0xBC00, 0x3D40, 0xBCC0, 0xBCC0, 0x3D40, 0x0000, 0x3CC0,
)
# Full-tensor pin: sha256 of the little-endian bf16 buffer of the dequantized
# [3072, 3584] w1 weight (identical from ct's bf16 pipeline and an fp32-exact path).
VEC_FULL_W1_DEQUANT_BF16_SHA256 = \
    "eaeed3d0fd8378496f60174c74737f6c29dd97ec2b854da64b0966ead7f2090f"
VEC_SOURCE = ("Kimi-K3 model-00005-of-000096.safetensors "
              "language_model.model.layers.4.block_sparse_moe.experts.0.w1 "
              "row 0, K-group 0; oracle compressed-tensors 0.17.1")


def mxfp4_dequantize_oracle(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Oracle-faithful MXFP4 dequant (pure torch, CPU-safe, NO exponent clamp).

    Bit-exact reimplementation of compressed_tensors 0.17.1 semantics:
    low nibble first, sign-magnitude E2M1 LUT, scale = 2^(uint8 - 127) via
    torch.ldexp with no clamp (0x00 -> 2^-127, 0xFF -> inf in bf16).

    This is the parity reference for all marlin-MXFP4 tests. It intentionally
    does NOT import batchgen.quantization.mxfp4 (which carries a clamp and a
    triton import).

    Args:
        packed: [..., K//2] uint8
        scales: [..., K//32] uint8 (E8M0)
    Returns: [..., K] in `dtype`.
    """
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(
            f"mxfp4_dequantize_oracle expects uint8 packed/scales, got "
            f"{packed.dtype}/{scales.dtype}")
    lut = torch.tensor(
        list(MXFP4_E2M1_LUT) + [-v for v in MXFP4_E2M1_LUT],
        dtype=torch.float32, device=packed.device)
    idx_lo = (packed & 0x0F).to(torch.long)
    idx_hi = (packed >> 4).to(torch.long)
    out_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    vals = torch.empty(out_shape, dtype=torch.float32, device=packed.device)
    vals[..., 0::2] = lut[idx_lo]   # low nibble -> even K index
    vals[..., 1::2] = lut[idx_hi]   # high nibble -> odd K index
    exponents = scales.to(torch.int32) - MXFP4_E8M0_BIAS  # NO clamp (oracle semantics)
    exponents = exponents.repeat_interleave(MXFP4_GROUP_SIZE, dim=-1)
    return torch.ldexp(vals, exponents).to(dtype)


def vec_expected_bf16() -> torch.Tensor:
    """The 32 expected dequantized values as bf16 (exact bits)."""
    signed = [b - 0x10000 if b >= 0x8000 else b for b in VEC_EXPECTED_BF16_BITS]
    return torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16)


def vec_packed_tensor() -> torch.Tensor:
    return torch.tensor(list(VEC_PACKED), dtype=torch.uint8).unsqueeze(0)  # [1, 16]


def vec_scale_tensor() -> torch.Tensor:
    return torch.tensor([[VEC_SCALE_BYTE]], dtype=torch.uint8)             # [1, 1]


def check_dequant_fn(dequant_fn) -> None:
    """Assert dequant_fn(packed[1,16], scale[1,1]) -> bf16 [1,32] bit-exact.

    Any wrong nibble order, wrong LUT, wrong sign bit, wrong bias, or wrong
    group mapping fails this check (verified by mutation on the real tensor).
    """
    out = dequant_fn(vec_packed_tensor(), vec_scale_tensor()).reshape(-1)
    if out.dtype != torch.bfloat16:
        raise AssertionError(f"expected bf16, got {out.dtype}")
    exp = vec_expected_bf16()
    same = (out.view(torch.int16) == exp.view(torch.int16))
    if not bool(same.all()):
        raise AssertionError(
            f"MXFP4 dequant mismatch vs frozen compressed-tensors oracle vector "
            f"({VEC_SOURCE}): got {out.float().tolist()} "
            f"expected {exp.float().tolist()}")
