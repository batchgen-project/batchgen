"""BF16 -> FP8 block-wise activation quantization (block 128 along the last dim).

Backend-neutral: it lived in ``batchgen.attention.mla.fa3_backend``, whose
module import requires FlashAttention-3, so any non-Hopper caller (the
Kimi-K3.1 DSA prefill scorer on Blackwell) could not reach it. That module
re-exports these names unchanged.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

@triton.jit
def act_quant_kernel_2d(
    x_ptr,
    y_ptr,
    scale_ptr,
    num_valid_tokens_ptr,
    M, N,
    eps: tl.constexpr,
    fp8_max: tl.constexpr,
    HAS_VALID_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    2D Block Quantization Kernel.
    Treats input as (Rows, Cols) -> Quantizes row segments of size BLOCK_SIZE.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate offsets
    row_start = x_ptr + pid_m * N
    block_start = pid_n * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    if HAS_VALID_TOKENS:
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        if pid_m >= num_valid_tokens:
            y_row_start = y_ptr + pid_m * N
            tl.store(y_row_start + offsets, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
            num_blocks_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
            scale_offset = pid_m * num_blocks_n + pid_n
            tl.store(scale_ptr + scale_offset, eps)
            return
    
    # Load block
    x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute scale (absmax)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(absmax, eps) / fp8_max
    
    # Quantize
    x_scaled = x / scale
    x_scaled = tl.minimum(x_scaled, fp8_max)
    x_scaled = tl.maximum(x_scaled, -fp8_max)
    
    # Store Quantized Data
    y = x_scaled.to(y_ptr.dtype.element_ty)
    y_row_start = y_ptr + pid_m * N
    tl.store(y_row_start + offsets, y, mask=mask)
    
    # Store Scale
    # Layout: (Rows, NumBlocks)
    num_blocks_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_offset = pid_m * num_blocks_n + pid_n
    tl.store(scale_ptr + scale_offset, scale)

# Maximum rows per kernel launch to avoid CUDA grid dimension limits
# and memory access issues with very long sequences
ACT_QUANT_MAX_ROWS = 32768  # 32K rows per chunk


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-12,
    num_valid_tokens: torch.Tensor | None = None,
    scale_tma_aligned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 → FP8 blockwise quantization for the attention hot path.

    Routes bf16 block-128 inputs to `per_token_blocked_quantize_bf16_to_fp8_1d`
    (batchgen_kernels.triton.fp8_quantize). That kernel launches one CTA per
    token with a 2D [NUM_BLOCKS_P2, BLOCK_K] tile and reduces amax along
    axis=1 for all K-blocks simultaneously. Benchmarked vs the CUDA
    `act_quant_3d` kernel at M=1..100000, K=6144:
      - Bit-exact or within single-ULP rounding noise vs CUDA.
      - ~1.85x faster at M >= 4k (CUDA's warp-stride-over-K-blocks pattern
        serializes the reductions; the 2D-tile variant parallelizes them).
      - No grid-X cap concern (Triton 1D grid -> 2^31-1 on CC >= 3.0).

    Legacy Triton fallback (`act_quant_kernel_2d` below) is kept for edge
    cases where x is non-bf16 or block_size != 128.
    """
    assert x.is_contiguous(), 'Input must be contiguous'
    if num_valid_tokens is not None:
        if num_valid_tokens.device != x.device:
            raise ValueError("num_valid_tokens must be on the same device as x")
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}")
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                f"num_valid_tokens must contain one element, got {tuple(num_valid_tokens.shape)}"
            )

    original_shape = x.shape
    x_flat = x.view(-1, original_shape[-1])
    M, N = x_flat.shape

    if x.dtype == torch.bfloat16 and M > 0 and block_size == 128:
        if scale_tma_aligned and x.dim() != 2:
            raise ValueError("scale_tma_aligned act_quant currently requires a 2D input")
        from batchgen_kernels.triton.fp8_quantize import per_token_blocked_quantize_bf16_to_fp8_1d
        y_flat, scale_flat = per_token_blocked_quantize_bf16_to_fp8_1d(
            x_flat,
            block_size=block_size,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=scale_tma_aligned,
        )
        num_blocks = scale_flat.size(-1)
        y = y_flat.view(*original_shape)
        scale = scale_flat if scale_tma_aligned else scale_flat.view(*original_shape[:-1], num_blocks)
        return y, scale

    fp8_max = 448.0
    if scale_tma_aligned and x.dim() != 2:
        raise ValueError("scale_tma_aligned act_quant currently requires a 2D input")
    
    # 2. Allocate Outputs
    y = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
    num_blocks = (N + block_size - 1) // block_size
    
    # Scale shape: (Rows, NumBlocks)
    if scale_tma_aligned:
        aligned_m = ((M + 3) // 4) * 4
        scale = torch.empty((num_blocks, aligned_m), dtype=torch.float32, device=x.device).transpose(0, 1)[:M, :]
    else:
        scale = torch.empty((M, num_blocks), dtype=torch.float32, device=x.device)
    
    # 3. Launch Kernel - chunk if needed for long sequences
    if M <= ACT_QUANT_MAX_ROWS:
        # Single kernel launch for short sequences
        grid = (M, num_blocks)
        act_quant_kernel_2d[grid](
            x_flat, y, scale,
            num_valid_tokens if num_valid_tokens is not None else scale,
            M, N,
            eps=eps,
            fp8_max=fp8_max,
            HAS_VALID_TOKENS=num_valid_tokens is not None,
            BLOCK_SIZE=block_size
        )
    else:
        if num_valid_tokens is not None:
            raise ValueError("num_valid_tokens is not supported by chunked fallback act_quant")
        # Chunked processing for long sequences
        for chunk_start in range(0, M, ACT_QUANT_MAX_ROWS):
            chunk_end = min(chunk_start + ACT_QUANT_MAX_ROWS, M)
            chunk_size = chunk_end - chunk_start
            
            # Get views into the chunk
            x_chunk = x_flat[chunk_start:chunk_end]
            y_chunk = y[chunk_start:chunk_end]
            scale_chunk = scale[chunk_start:chunk_end]
            
            grid = (chunk_size, num_blocks)
            act_quant_kernel_2d[grid](
                x_chunk, y_chunk, scale_chunk,
                scale_chunk,
                chunk_size, N,
                eps=eps,
                fp8_max=fp8_max,
                HAS_VALID_TOKENS=False,
                BLOCK_SIZE=block_size
            )
    
    # 4. Restore Shapes
    # Reshape Y back to (E, T, H)
    y = y.view(original_shape)
    
    # CRITICAL: Reshape Scale back to (E, T, NumBlocks)
    # This allows downstream kernels to calculate strides correctly.
    if not scale_tma_aligned:
        scale_shape = list(original_shape[:-1]) + [num_blocks]
        scale = scale.view(*scale_shape)

    return y, scale
