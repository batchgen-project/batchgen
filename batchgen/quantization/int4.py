"""INT4 W4A16 (Weight-Only) dequantization for Kimi K2.5.

INT4 Format (compressed-tensors pack-quantized, symmetric):
- Block size: 32 INT4 values per scale group (group_size=32)
- Packing: 2 INT4 values per uint8 byte (low nibble = first, high nibble = second)
- INT4 range: signed [-8, +7] (4-bit offset encoding, nibble - 8)
- Scales: bf16, one per group of 32 elements
- Symmetric: no zero-point needed
- Tensor names: .weight_packed (uint8), .weight_scale (bf16)

W4A16 means:
- Weights: INT4 quantized (stored packed in uint8)
- Activations: BF16 (untouched, no activation quantization)
- GEMM: dequant INT4→BF16 in-register, then BF16 tensor core mma

Dequantization: int4_value * bf16_scale → bf16 weight
"""

import torch
import triton
import triton.language as tl

# INT4 group size: 32 elements per scale
INT4_GROUP_SIZE = 32


def int4_dequantize_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Reference implementation of INT4 W4A16 dequantization (pure PyTorch).

    Dequantizes INT4 packed weights to bf16 using group scales.
    No activation quantization — this is weight-only dequantization.

    Args:
        packed: Packed INT4 values as uint8 [M, K//2] (2 values per byte)
        scales: Scale factors as bf16 [M, K//32] (one per group of 32 elements)
        dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized tensor [M, K] in the specified dtype
    """
    device = packed.device

    # Extract nibbles from packed uint8 bytes
    # Low nibble (bits 0-3) = first value (even positions)
    # High nibble (bits 4-7) = second value (odd positions)
    # Offset encoding (compressed-tensors standard): signed = nibble - 8
    lo = (packed & 0x0F).to(torch.int32) - 8
    hi = (packed >> 4).to(torch.int32) - 8

    # Interleave lo/hi to reconstruct original element order [M, K]
    # lo goes to even positions (0, 2, 4, ...), hi to odd positions (1, 3, 5, ...)
    M = packed.shape[0]
    K_half = packed.shape[-1]
    K = K_half * 2
    unpacked = torch.empty(M, K, dtype=torch.int32, device=device)
    unpacked[:, 0::2] = lo
    unpacked[:, 1::2] = hi

    # Apply group scales: each scale covers 32 consecutive elements
    # scales: [M, K//32], unpacked: [M, K]
    n_groups = scales.shape[-1]
    unpacked_float = unpacked.to(dtype)  # [M, K] in target dtype

    # Reshape for grouped broadcast multiply
    unpacked_grouped = unpacked_float.view(M, n_groups, INT4_GROUP_SIZE)  # [M, G, 32]
    scales_expanded = scales.to(dtype).unsqueeze(-1)  # [M, G, 1]

    result = unpacked_grouped * scales_expanded  # [M, G, 32]
    return result.view(M, K).to(dtype)


@triton.jit
def int4_dequant_kernel_2d(
    packed_ptr,      # Input: uint8 packed INT4 values [M, n_packed]
    scales_ptr,      # Input: bf16 scales [M, n_scales]
    output_ptr,      # Output: dequantized bf16 values [M, n_output]
    n_packed,        # Number of packed bytes per row (K // 2)
    n_scales,        # Number of scales per row (K // 32)
    n_output,        # Number of output elements per row (K)
    stride_packed_row,   # Stride between rows in packed (n_packed)
    stride_scales_row,   # Stride between rows in scales (n_scales)
    stride_output_row,   # Stride between rows in output (n_output)
    BLOCK_SIZE: tl.constexpr,  # Packed bytes per thread block
):
    """Triton kernel for INT4 W4A16 dequantization with 2D grid.

    Grid: (num_col_blocks, num_rows)
    Each thread block processes BLOCK_SIZE packed bytes (2*BLOCK_SIZE INT4 values).

    Simpler than MXFP4 kernel:
    - No FP4 lookup table — simple sign-extend
    - No ldexp — direct bf16 multiply with bf16 scale
    """
    # 2D grid: pid_x = column block, pid_y = row
    pid_x = tl.program_id(0)  # Column block index
    pid_y = tl.program_id(1)  # Row index

    # Offsets within this row (in packed byte space)
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

    # Extract nibbles and apply offset encoding (compressed-tensors standard)
    lo = (packed & 0x0F).to(tl.int32) - 8
    hi = ((packed >> 4) & 0x0F).to(tl.int32) - 8

    # Convert to bf16 for scaling
    lo_bf16 = lo.to(tl.bfloat16)
    hi_bf16 = hi.to(tl.bfloat16)

    # Load scales (bf16 — simpler than MXFP4's uint8 exponent)
    # Both lo and hi from the same byte share the same scale,
    # since group_size=32 aligns with 16 packed bytes:
    #   lo at output pos 2*i → group (2*i)//32 = i//16
    #   hi at output pos 2*i+1 → group (2*i+1)//32 = i//16
    scale_idx = offsets // 16  # 16 packed bytes per group (32 elements / 2 per byte)
    scale_mask = scale_idx < n_scales

    scale_val = tl.load(scales_ptr + scales_row_offset + scale_idx,
                        mask=mask & scale_mask, other=0.0)
    scale_val = scale_val.to(tl.bfloat16)

    # Apply scale: dequantized = int4_value * scale
    result_lo = lo_bf16 * scale_val
    result_hi = hi_bf16 * scale_val

    # Store interleaved results
    # Output positions: 2*offset (even) and 2*offset+1 (odd)
    out_offsets_lo = offsets * 2
    out_offsets_hi = offsets * 2 + 1

    out_mask_lo = out_offsets_lo < n_output
    out_mask_hi = out_offsets_hi < n_output

    tl.store(output_ptr + output_row_offset + out_offsets_lo,
             result_lo, mask=mask & out_mask_lo)
    tl.store(output_ptr + output_row_offset + out_offsets_hi,
             result_hi, mask=mask & out_mask_hi)


def int4_dequantize_triton(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Triton-accelerated INT4 W4A16 dequantization with 2D grid.

    Uses a single kernel launch with 2D grid (column_blocks, rows).

    Args:
        packed: Packed INT4 values as uint8 [M, K//2] (2 values per byte)
        scales: Scale factors as bf16 [M, K//32] (one per group of 32)
        dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized tensor [M, K] in the specified dtype
    """
    assert packed.dtype == torch.uint8, f"packed must be uint8, got {packed.dtype}"
    assert scales.dtype == torch.bfloat16, f"scales must be bfloat16, got {scales.dtype}"
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
    n_output = n_packed * 2  # 2 INT4 values per byte
    n_scales = scales.shape[-1]

    # Get device index for CUDA context
    device_idx = packed.device.index if packed.device.index is not None else 0

    # Use device guard to ensure Triton launches on correct GPU in multi-GPU setup
    with torch.cuda.device(device_idx):
        output = torch.empty((M, n_output), dtype=dtype, device=packed.device)

        # Launch kernel with 2D grid: (num_col_blocks, num_rows)
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_packed, BLOCK_SIZE), M)

        int4_dequant_kernel_2d[grid](
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


def int4_dequantize_int32_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Reference implementation of INT4 W4A16 dequantization for int32 packed format.

    Kimi K2.5 uses int32 packing: 8 INT4 values per int32 word (32 bits / 4 bits = 8).

    Bit layout in each int32 word:
    - bits 0-3:   value 0
    - bits 4-7:   value 1
    - bits 8-11:  value 2
    - bits 12-15: value 3
    - bits 16-19: value 4
    - bits 20-23: value 5
    - bits 24-27: value 6
    - bits 28-31: value 7

    Args:
        packed: Packed INT4 values as int32 [M, K//8] (8 values per word)
        scales: Scale factors as bf16 [M, K//32] (one per group of 32 elements)
        dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized tensor [M, K] in the specified dtype
    """
    device = packed.device
    M = packed.shape[0]
    K_div8 = packed.shape[-1]
    K = K_div8 * 8

    # Extract 8 nibbles from each int32 word
    # Use bit shifts and mask to extract 4-bit values
    unpacked = torch.empty(M, K_div8, 8, dtype=torch.int32, device=device)

    for i in range(8):
        # Extract nibble at position i (bits i*4 to i*4+3)
        nibble = (packed >> (i * 4)) & 0xF
        # Apply offset encoding: signed = nibble - 8
        unpacked[:, :, i] = nibble.to(torch.int32) - 8

    # Reshape to [M, K]
    unpacked_flat = unpacked.view(M, K)

    # Apply group scales: each scale covers 32 consecutive elements
    n_groups = scales.shape[-1]
    unpacked_float = unpacked_flat.to(dtype)  # [M, K] in target dtype

    # Reshape for grouped broadcast multiply
    unpacked_grouped = unpacked_float.view(M, n_groups, INT4_GROUP_SIZE)  # [M, G, 32]
    scales_expanded = scales.to(dtype).unsqueeze(-1)  # [M, G, 1]

    result = unpacked_grouped * scales_expanded  # [M, G, 32]
    return result.view(M, K).to(dtype)


def int4_dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    use_triton: bool = False  # Default to PyTorch reference; enable after Triton validation
) -> torch.Tensor:
    """Dequantize INT4 W4A16 packed tensor to the specified dtype.

    Automatically detects packing format:
    - uint8: 2 INT4 values per byte (standard format)
    - int32: 8 INT4 values per word (Kimi K2.5 format)

    Use cases:
    1. Pre-dequant persistent experts to BF16 at init (EP mode, world_size >= 4)
    2. On-the-fly dequant for offloaded experts (host → device → dequant → BF16 GEMM)
    3. Validation: compare fused kernel output against this reference

    Args:
        packed: Packed INT4 values as uint8 or int32
        scales: Scale factors as bf16 (one per group of 32 elements)
        dtype: Output dtype (default: bfloat16)
        use_triton: Whether to use Triton kernel (default: False)

    Returns:
        Dequantized tensor in the specified dtype
    """
    # Detect packing format by dtype
    if packed.dtype == torch.int32:
        # Kimi K2.5 format: 8 INT4 values per int32 word
        return int4_dequantize_int32_reference(packed, scales, dtype)
    elif packed.dtype == torch.uint8:
        # Standard format: 2 INT4 values per uint8 byte
        if use_triton and packed.is_cuda:
            return int4_dequantize_triton(packed, scales, dtype)
        else:
            return int4_dequantize_reference(packed, scales, dtype)
    else:
        raise ValueError(
            f"Unsupported packed dtype: {packed.dtype}. "
            f"Expected torch.uint8 or torch.int32"
        )
