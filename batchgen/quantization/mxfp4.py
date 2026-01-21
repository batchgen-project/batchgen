"""MXFP4 (Microscaling FP4) dequantization for GPT-OSS-120B.

MXFP4 Format (from OpenAI gpt_oss/torch/weights.py):
- Block size: 32 FP4 values packed in 16 bytes (2 values per uint8)
- Packing: Low nibble = first value, high nibble = second value
- FP4 Lookup Table: 16 values [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
- Scales: uint8 stored, exponent = scale.to(int32) - 127
- Dequantization: torch.ldexp(fp4_value, exponent) = fp4_value * 2^exponent
"""

import torch
from typing import Tuple
import triton
import triton.language as tl

# FP4 lookup table (16 values)
# Indices 0-7: positive values [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Indices 8-15: negative values [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
FP4_LOOKUP_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32
)

# MXFP4 block size: 32 FP4 values per scale
MXFP4_BLOCK_SIZE = 32


def mxfp4_dequantize_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Reference implementation of MXFP4 dequantization (pure PyTorch).

    This follows the OpenAI gpt_oss/torch/weights.py implementation exactly.

    Args:
        packed: Packed FP4 values as uint8 [M, G, B] or [M, N//2] (2 values per byte)
        scales: Scale factors as uint8 [M, G] or [M, N//32] (one per 32 elements)
        dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized tensor in the specified dtype [M, N]
    """
    device = packed.device
    fp4_table = FP4_LOOKUP_TABLE.to(device)

    # Handle 3D packed tensors from GPT-OSS: [rows, blocks, bytes_per_block]
    # Flatten to 2D: [rows, blocks * bytes_per_block]
    if packed.dim() == 3:
        M, G, B = packed.shape
        packed = packed.view(M, G * B)  # [M, G*B]
        # scales is already [M, G], no change needed

    # Get packed shape after potential flattening
    packed_shape = packed.shape

    # Unpack two FP4 values from each uint8 byte
    # Low nibble (bits 0-3) = first value (even positions)
    # High nibble (bits 4-7) = second value (odd positions)
    idx_lo = (packed & 0x0F).to(torch.long)
    idx_hi = (packed >> 4).to(torch.long)

    # Lookup FP4 values from table
    val_lo = fp4_table[idx_lo]  # Even positions
    val_hi = fp4_table[idx_hi]  # Odd positions

    # Interleave: [lo0, hi0, lo1, hi1, ...]
    # Shape: [..., N//2] -> [..., N]
    output_shape = packed_shape[:-1] + (packed_shape[-1] * 2,)
    unpacked = torch.empty(output_shape, dtype=torch.float32, device=device)
    unpacked[..., 0::2] = val_lo
    unpacked[..., 1::2] = val_hi

    # Apply block scaling
    # Each scale covers 32 consecutive FP4 values
    # scales shape: [..., N//32], unpacked shape: [..., N]
    # Expand scales to match unpacked: each scale repeats 32 times
    exponents = scales.to(torch.int32) - 127

    # Broadcast scales: [..., N//32] -> [..., N]
    # Each scale value is used for 32 consecutive elements
    n_elements = unpacked.shape[-1]
    n_blocks = scales.shape[-1]

    # Repeat each scale 32 times along the last dimension
    expanded_exponents = exponents.unsqueeze(-1).expand(
        *exponents.shape, MXFP4_BLOCK_SIZE
    ).reshape(*exponents.shape[:-1], n_blocks * MXFP4_BLOCK_SIZE)

    # Handle case where n_elements is not exactly divisible by 32
    if expanded_exponents.shape[-1] > n_elements:
        expanded_exponents = expanded_exponents[..., :n_elements]

    # Apply ldexp: result = fp4_value * 2^exponent
    result = torch.ldexp(unpacked, expanded_exponents)

    return result.to(dtype)


@triton.jit
def mxfp4_dequant_kernel_2d(
    packed_ptr,      # Input: uint8 packed FP4 values [M, n_packed]
    scales_ptr,      # Input: uint8 scales [M, n_scales]
    output_ptr,      # Output: dequantized BF16 values [M, n_output]
    n_packed,        # Number of packed bytes per row (K // 2)
    n_scales,        # Number of scales per row (K // 32)
    n_output,        # Number of output elements per row (K)
    stride_packed_row,   # Stride between rows in packed (n_packed)
    stride_scales_row,   # Stride between rows in scales (n_scales)
    stride_output_row,   # Stride between rows in output (n_output)
    BLOCK_SIZE: tl.constexpr,  # Elements per thread block
):
    """Triton kernel for MXFP4 dequantization with 2D grid.

    Grid: (num_col_blocks, num_rows)
    Each thread block processes BLOCK_SIZE packed bytes (2*BLOCK_SIZE FP4 values).
    """
    # 2D grid: pid_x = column block, pid_y = row
    pid_x = tl.program_id(0)  # Column block index
    pid_y = tl.program_id(1)  # Row index

    # Offsets within this row
    block_start = pid_x * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_packed

    # Calculate row offsets
    packed_row_offset = pid_y * stride_packed_row
    scales_row_offset = pid_y * stride_scales_row
    output_row_offset = pid_y * stride_output_row

    # Load packed bytes
    packed = tl.load(packed_ptr + packed_row_offset + offsets, mask=mask, other=0)
    packed = packed.to(tl.uint8)

    # Extract nibbles
    idx_lo = (packed & 0x0F).to(tl.int32)
    idx_hi = ((packed >> 4) & 0x0F).to(tl.int32)

    # Low nibble values
    val_lo = _fp4_lookup(idx_lo)
    val_hi = _fp4_lookup(idx_hi)

    # Calculate scale indices for each output position
    # Each scale covers 32 FP4 values = 16 packed bytes
    scale_idx_lo = offsets // 16  # Scale for even output positions
    scale_idx_hi = (offsets * 2 + 1) // 32  # Scale for odd output positions

    # Load scales
    scale_mask_lo = scale_idx_lo < n_scales
    scale_mask_hi = scale_idx_hi < n_scales

    scales_lo = tl.load(scales_ptr + scales_row_offset + scale_idx_lo,
                        mask=mask & scale_mask_lo, other=0)
    scales_hi = tl.load(scales_ptr + scales_row_offset + scale_idx_hi,
                        mask=mask & scale_mask_hi, other=0)

    # Convert scales: exponent = scale_uint8 - 127
    exp_lo = scales_lo.to(tl.int32) - 127
    exp_hi = scales_hi.to(tl.int32) - 127

    # Apply ldexp: result = value * 2^exponent
    result_lo = _triton_ldexp(val_lo, exp_lo)
    result_hi = _triton_ldexp(val_hi, exp_hi)

    # Store interleaved results
    # Output positions: 2*offset (even) and 2*offset+1 (odd)
    out_offsets_lo = offsets * 2
    out_offsets_hi = offsets * 2 + 1

    out_mask_lo = out_offsets_lo < n_output
    out_mask_hi = out_offsets_hi < n_output

    tl.store(output_ptr + output_row_offset + out_offsets_lo,
             result_lo.to(tl.bfloat16), mask=mask & out_mask_lo)
    tl.store(output_ptr + output_row_offset + out_offsets_hi,
             result_hi.to(tl.bfloat16), mask=mask & out_mask_hi)


@triton.jit
def _fp4_lookup(idx):
    """Lookup FP4 value from index using conditional logic.

    FP4 table:
    0: 0.0,  1: 0.5,  2: 1.0,  3: 1.5,  4: 2.0,  5: 3.0,  6: 4.0,  7: 6.0
    8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0
    """
    # Positive values (idx 0-7)
    val = tl.where(idx == 0, 0.0, 0.0)
    val = tl.where(idx == 1, 0.5, val)
    val = tl.where(idx == 2, 1.0, val)
    val = tl.where(idx == 3, 1.5, val)
    val = tl.where(idx == 4, 2.0, val)
    val = tl.where(idx == 5, 3.0, val)
    val = tl.where(idx == 6, 4.0, val)
    val = tl.where(idx == 7, 6.0, val)

    # Negative values (idx 8-15)
    val = tl.where(idx == 8, -0.0, val)
    val = tl.where(idx == 9, -0.5, val)
    val = tl.where(idx == 10, -1.0, val)
    val = tl.where(idx == 11, -1.5, val)
    val = tl.where(idx == 12, -2.0, val)
    val = tl.where(idx == 13, -3.0, val)
    val = tl.where(idx == 14, -4.0, val)
    val = tl.where(idx == 15, -6.0, val)

    return val.to(tl.float32)


@triton.jit
def _triton_ldexp(mantissa, exponent):
    """Compute mantissa * 2^exponent using bit manipulation.

    For float32: exponent is stored in bits 23-30 (biased by 127)
    ldexp(m, e) = m * 2^e
    """
    # Clamp exponent to valid range for float32
    exp_clamped = tl.minimum(tl.maximum(exponent, -126), 127)

    # Create 2^exponent as float
    # float32 bit layout: [sign(1)][exponent(8)][mantissa(23)]
    # To create 2^e: set exponent bits to (e + 127), mantissa = 0
    exp_bits = (exp_clamped + 127).to(tl.int32)
    exp_bits = exp_bits << 23
    power_of_2 = exp_bits.to(tl.float32, bitcast=True)

    return mantissa * power_of_2


def mxfp4_dequantize_triton(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Triton-accelerated MXFP4 dequantization with 2D grid.

    Uses a single kernel launch with 2D grid instead of per-row launches,
    providing ~1000x speedup for large tensors.

    Args:
        packed: Packed FP4 values as uint8 [M, K//2] (2 values per byte)
        scales: Scale factors as uint8 [M, K//32] (one per 32 elements)
        dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized tensor [..., K] in the specified dtype
    """
    assert packed.dtype == torch.uint8, f"packed must be uint8, got {packed.dtype}"
    assert scales.dtype == torch.uint8, f"scales must be uint8, got {scales.dtype}"
    assert packed.is_contiguous(), "packed must be contiguous"
    assert scales.is_contiguous(), "scales must be contiguous"

    # Handle different input shapes
    original_shape = packed.shape
    if packed.dim() == 1:
        packed = packed.unsqueeze(0)
        scales = scales.unsqueeze(0)

    # Flatten to 2D for processing
    if packed.dim() > 2:
        batch_dims = packed.shape[:-1]
        packed = packed.view(-1, packed.shape[-1])
        scales = scales.view(-1, scales.shape[-1])
    else:
        batch_dims = None

    M, n_packed = packed.shape
    n_output = n_packed * 2  # 2 FP4 values per byte
    n_scales = scales.shape[-1]

    # Get device index for CUDA context
    device_idx = packed.device.index if packed.device.index is not None else 0

    # Use device guard to ensure Triton launches on correct GPU in multi-GPU setup
    with torch.cuda.device(device_idx):
        # Allocate output on the same device
        output = torch.empty((M, n_output), dtype=dtype, device=packed.device)

        # Launch kernel with 2D grid: (num_col_blocks, num_rows)
        BLOCK_SIZE = 256  # Tune based on hardware
        grid = (triton.cdiv(n_packed, BLOCK_SIZE), M)

        mxfp4_dequant_kernel_2d[grid](
            packed,
            scales,
            output,
            n_packed,
            n_scales,
            n_output,
            n_packed,   # stride_packed_row (contiguous)
            n_scales,   # stride_scales_row (contiguous)
            n_output,   # stride_output_row (contiguous)
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Restore original shape
    if batch_dims is not None:
        output = output.view(*batch_dims, n_output)
    if len(original_shape) == 1:
        output = output.squeeze(0)

    return output


def mxfp4_dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    use_triton: bool = False  # Disabled: Triton 2D kernel has CUDA issues; use vectorized PyTorch
) -> torch.Tensor:
    """Dequantize MXFP4 packed tensor to the specified dtype.

    Args:
        packed: Packed FP4 values as uint8 (2 values per byte)
        scales: Scale factors as uint8 (one per 32 elements)
        dtype: Output dtype (default: bfloat16)
        use_triton: Whether to use Triton kernel (default: True)

    Returns:
        Dequantized tensor in the specified dtype
    """
    if use_triton and packed.is_cuda:
        return mxfp4_dequantize_triton(packed, scales, dtype)
    else:
        return mxfp4_dequantize_reference(packed, scales, dtype)


def load_mxfp4_weight(
    blocks_path: str,
    scales_path: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
) -> torch.Tensor:
    """Load and dequantize MXFP4 weights from disk.

    Args:
        blocks_path: Path to packed FP4 blocks (.blocks file)
        scales_path: Path to scale factors (.scales file)
        shape: Expected output shape
        dtype: Output dtype (default: bfloat16)
        device: Target device

    Returns:
        Dequantized weight tensor
    """
    # Load packed data
    packed = torch.load(blocks_path, map_location=device)
    scales = torch.load(scales_path, map_location=device)

    # Ensure uint8 dtype
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    if scales.dtype != torch.uint8:
        scales = scales.to(torch.uint8)

    # Dequantize
    weight = mxfp4_dequantize(packed, scales, dtype)

    # Reshape to expected shape
    if weight.shape != shape:
        weight = weight.view(shape)

    return weight
