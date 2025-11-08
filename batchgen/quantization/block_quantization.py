import torch
from typing import Tuple
import triton
import triton.language as tl
# @triton.jit
# def act_quant_kernel_2d_transposed_scale(
#     x_ptr,
#     y_ptr,
#     scale_ptr,
#     M, N,
#     tma_aligned_M: tl.constexpr,  # Aligned M dimension for output
#     eps: tl.constexpr,
#     fp8_max: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr
# ):
#     """
#     2D quantization kernel that outputs scales in transposed layout.
#     Scale layout: [num_blocks, tma_aligned_M] instead of [M, num_blocks]
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
    
#     # Each program handles one block in a row
#     row_start = x_ptr + pid_m * N
#     block_start = pid_n * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < N
    
#     # Load block
#     x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    
#     # Compute scale (absmax)
#     absmax = tl.max(tl.abs(x), axis=0)
#     scale = tl.maximum(absmax, eps) / fp8_max
    
#     # Quantize
#     x_scaled = x / scale
#     x_scaled = tl.minimum(x_scaled, fp8_max)
#     x_scaled = tl.maximum(x_scaled, -fp8_max)
    
#     # Store quantized values
#     y = x_scaled.to(y_ptr.dtype.element_ty)
#     y_row_start = y_ptr + pid_m * N
#     tl.store(y_row_start + offsets, y, mask=mask)
    
#     # Store scale in TRANSPOSED layout: [num_blocks, tma_aligned_M]
#     # This matches the output of transpose_fp32: [SF_K, tma_aligned_MN]
#     scale_offset = pid_n * tma_aligned_M + pid_m
#     tl.store(scale_ptr + scale_offset, scale)


# def act_quant_transposed_scale(
#     x: torch.Tensor, 
#     block_size: int = 128,
#     eps: float = 1e-12,
#     tma_align: int = 4  # 16 bytes / 4 bytes per float
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     BF16 to FP8 E4M3 block quantization with transposed scale output.
    
#     Args:
#         x: Input tensor in BF16/FP16/FP32
#         block_size: Block size for quantization (typically 128 or 256)
#         eps: Epsilon for numerical stability (1e-12 is standard)
#         tma_align: Alignment requirement (4 floats = 16 bytes for TMA)
    
#     Returns:
#         y: Quantized tensor in FP8 E4M3
#         scale: Per-block scaling factors in TRANSPOSED layout [num_blocks, aligned_M]
#                Ready for direct use without additional transpose!
#     """
#     assert x.is_contiguous(), 'Input must be contiguous'
    
#     # FP8 E4M3 characteristics
#     fp8_max = 448.0
    
#     # Flatten all dimensions except last for block processing
#     original_shape = x.shape
#     x_flat = x.view(-1, x.shape[-1])
#     M, N = x_flat.shape
    
#     # Calculate aligned M dimension
#     tma_aligned_M = ((M + tma_align - 1) // tma_align) * tma_align
    
#     # Allocate outputs
#     y = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
#     num_blocks = (N + block_size - 1) // block_size
    
#     # Scale in TRANSPOSED layout: [num_blocks, tma_aligned_M]
#     scale = torch.empty(
#         (num_blocks, tma_aligned_M), 
#         dtype=torch.float32, 
#         device=x.device
#     )
    
#     # Launch kernel
#     grid = (M, num_blocks)
#     act_quant_kernel_2d_transposed_scale[grid](
#         x_flat, y, scale,
#         M, N,
#         tma_aligned_M=tma_aligned_M,
#         eps=eps,
#         fp8_max=fp8_max,
#         BLOCK_SIZE=block_size
#     )
    
#     # Restore original shape for quantized output
#     y = y.view(original_shape)
    
#     return y, scale

@triton.jit
def act_quant_kernel_2d_transposed_scale(
    x_ptr,
    y_ptr,
    scale_ptr,
    M, K,  # Changed from N to K for clarity
    tma_aligned_M: tl.constexpr,
    eps: tl.constexpr,
    fp8_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Quantizes along K dimension in blocks.
    Each row is divided into blocks of size BLOCK_SIZE along K.
    """
    pid_m = tl.program_id(axis=0)  # Row index
    pid_k = tl.program_id(axis=1)  # Block index along K
    
    # Each program handles one block in a row
    row_start = x_ptr + pid_m * K
    block_start = pid_k * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < K
    
    # Load block
    x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute scale (absmax per block)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(absmax, eps) / fp8_max
    
    # Quantize
    x_scaled = x / scale
    x_scaled = tl.minimum(x_scaled, fp8_max)
    x_scaled = tl.maximum(x_scaled, -fp8_max)
    
    # Store quantized values
    y = x_scaled.to(y_ptr.dtype.element_ty)
    y_row_start = y_ptr + pid_m * K
    tl.store(y_row_start + offsets, y, mask=mask)
    
    # Store scale in TRANSPOSED layout: [num_blocks_k, tma_aligned_M]
    scale_offset = pid_k * tma_aligned_M + pid_m
    tl.store(scale_ptr + scale_offset, scale)

def act_quant_transposed_scale(
    x: torch.Tensor, 
    block_size: int = 128,  # This should match the K-dimension blocking (typically 128)
    eps: float = 1e-12,
    tma_align: int = 4  # 16 bytes / 4 bytes per float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BF16 to FP8 E4M3 block quantization with transposed scale output.
    Quantizes along K dimension (last dimension) in blocks.
    
    Args:
        x: Input tensor [M, K] in BF16/FP16/FP32
        block_size: Block size for K-dimension quantization (must match deep_gemm recipe, typically 128)
        eps: Epsilon for numerical stability
        tma_align: Alignment requirement (4 floats = 16 bytes for TMA)
    
    Returns:
        y: Quantized tensor in FP8 E4M3 [M, K]
        scale: Per-block scaling factors [num_blocks_k, tma_aligned_m]
               where num_blocks_k = ceil_div(K, block_size)
    """
    assert x.is_contiguous(), 'Input must be contiguous'
    assert x.dim() == 2, 'Input must be 2D [M, K]'
    
    fp8_max = 448.0
    M, K = x.shape
    
    # Calculate aligned M dimension
    tma_aligned_M = ((M + tma_align - 1) // tma_align) * tma_align
    
    # Allocate outputs
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    num_blocks_k = (K + block_size - 1) // block_size
    
    # Scale in TRANSPOSED layout: [num_blocks_k, tma_aligned_M]
    # This matches what deep_gemm expects: sf.size(-2) == ceil_div(K, block_size)
    scale = torch.empty(
        (num_blocks_k, tma_aligned_M), 
        dtype=torch.float32, 
        device=x.device
    )
    
    # Launch kernel
    grid = (M, num_blocks_k)
    act_quant_kernel_2d_transposed_scale[grid](
        x, y, scale,
        M, K,
        tma_aligned_M=tma_aligned_M,
        eps=eps,
        fp8_max=fp8_max,
        BLOCK_SIZE=block_size
    )
    
    return y, scale