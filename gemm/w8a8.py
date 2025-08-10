
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
import math

@triton.autotune(
    configs=[
        # Small tiles for small matrices
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        
        # Medium tiles
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        
        # Large tiles for large matrices
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def w8a8_gemm_kernel(
    # Pointers to matrices
    a_ptr, w_ptr, c_ptr,
    # Pointers to scales
    a_scale_ptr, w_scale_ptr,
    # Matrix dimensions
    M, N, K,
    # Quantization block sizes
    a_block_size, w_block_size_k, w_block_size_n,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,  # A is (M, K)
    stride_wk, stride_wn,  # W is (K, N) 
    stride_cm, stride_cn,  # C is (M, N)
    stride_a_scale_m, stride_a_scale_k,  # a_scale is (M, ceil(K/a_block_size))
    stride_w_scale_k, stride_w_scale_n,  # w_scale is (ceil(K/w_block_size_k), ceil(N/w_block_size_n))
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A @ W with quantized inputs.
    
    A has shape (M, K) with per-row quantization
    W has shape (K, N) with 2D block quantization  
    C has shape (M, N) and will be in bf16
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A, W, and their scales.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    
    # Initialize the accumulator.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and W.
        a_fp8 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w_fp8 = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Convert fp8e4m3 to float32 for computation
        a_f32 = a_fp8.to(tl.float32)
        w_f32 = w_fp8.to(tl.float32)
        
        # Load quantization scales
        # For A: per-row quantization, scale varies across K dimension in blocks
        k_block_idx = (k * BLOCK_SIZE_K) // a_block_size
        a_scale_ptrs = a_scale_ptr + (offs_am * stride_a_scale_m + k_block_idx * stride_a_scale_k)
        a_scales = tl.load(a_scale_ptrs, mask=offs_am < M, other=1.0)
        
        # For W: 2D block quantization
        k_start = k * BLOCK_SIZE_K
        w_k_block_idx = k_start // w_block_size_k
        w_n_block_start = pid_n * BLOCK_SIZE_N
        
        # Load W scales for this K block and all N blocks we're computing
        w_scale_ptrs_base = w_scale_ptr + w_k_block_idx * stride_w_scale_k
        
        # We need to handle the case where our BLOCK_SIZE_N spans multiple w_block_size_n blocks
        w_scales = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        
        for n_offset in range(0, BLOCK_SIZE_N, w_block_size_n):
            n_end = min(n_offset + w_block_size_n, BLOCK_SIZE_N)
            actual_n_pos = w_n_block_start + n_offset
            w_n_block_idx = actual_n_pos // w_block_size_n
            
            if actual_n_pos < N:
                w_scale_ptr_curr = w_scale_ptrs_base + w_n_block_idx * stride_w_scale_n
                scale_val = tl.load(w_scale_ptr_curr)
                
                # Fill the corresponding positions in w_scales
                for i in range(n_offset, n_end):
                    if w_n_block_start + i < N:
                        w_scales = tl.where(
                            tl.arange(0, BLOCK_SIZE_N) == i,
                            scale_val,
                            w_scales
                        )
        
        # Apply scales to dequantize
        # A scales: shape (BLOCK_SIZE_M,) -> (BLOCK_SIZE_M, 1)
        a_dequant = a_f32 * a_scales[:, None]
        
        # W scales: shape (BLOCK_SIZE_N,) -> (1, BLOCK_SIZE_N)  
        w_dequant = w_f32 * w_scales[None, :]
        
        # We accumulate along the K dimension.
        accumulator = tl.dot(a_dequant, w_dequant, accumulator)
        
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # Convert accumulator from fp32 to bf16
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_gemm(
    a: torch.Tensor,           # [bsz, N] - activations in fp8e4m3
    a_scale: torch.Tensor,     # [bsz, ceil(N / block_size)] - per-row quantization scales
    w: torch.Tensor,           # [K, N] - weights in fp8e4m3  
    w_scale: torch.Tensor,     # [ceil(K/block_size_K), ceil(N/block_size_N)] - 2D block scales
    a_block_size: int = 128,   # Block size for A quantization
    w_block_size_k: int = 128, # Block size for W quantization along K
    w_block_size_n: int = 128, # Block size for W quantization along N
) -> torch.Tensor:
    """
    Performs quantized GEMM: C = A @ W^T
    
    Args:
        a: Activations tensor [M, K] in fp8e4m3
        a_scale: Per-row quantization scales [M, ceil(K/a_block_size)]
        w: Weight tensor [N, K] in fp8e4m3 (will be transposed internally)
        w_scale: 2D block quantization scales [ceil(N/w_block_size_k), ceil(K/w_block_size_n)]
        a_block_size: Block size for activation quantization
        w_block_size_k: Block size for weight quantization along K dimension
        w_block_size_n: Block size for weight quantization along N dimension
    
    Returns:
        torch.Tensor: Output tensor [M, N] in bfloat16
    """
    # Check input shapes and dtypes
    assert a.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3 for activations, got {a.dtype}"
    assert w.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3 for weights, got {w.dtype}"
    assert a_scale.dtype in [torch.float32, torch.bfloat16], f"Scale dtype should be fp32 or bf16, got {a_scale.dtype}"
    assert w_scale.dtype in [torch.float32, torch.bfloat16], f"Scale dtype should be fp32 or bf16, got {w_scale.dtype}"
    
    # Get dimensions
    M, K = a.shape
    N, K_w = w.shape
    assert K == K_w, f"Dimension mismatch: A has {K} columns, W has {K_w} columns"
    
    # Transpose weights to get [K, N] layout for better memory access
    w = w.transpose(0, 1).contiguous()  # Now [K, N]
    
    # Validate scale tensor shapes
    expected_a_scale_shape = (M, math.ceil(K / a_block_size))
    expected_w_scale_shape = (math.ceil(K / w_block_size_k), math.ceil(N / w_block_size_n))
    
    assert a_scale.shape == expected_a_scale_shape, \
        f"a_scale shape mismatch: expected {expected_a_scale_shape}, got {a_scale.shape}"
    assert w_scale.shape == expected_w_scale_shape, \
        f"w_scale shape mismatch: expected {expected_w_scale_shape}, got {w_scale.shape}"
    
    # Allocate output tensor in bfloat16
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
    # Ensure scales are contiguous and in fp32 for computation
    a_scale = a_scale.contiguous().to(torch.float32)
    w_scale = w_scale.contiguous().to(torch.float32)
    
    # Launch kernel with auto-tuning
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    w8a8_gemm_kernel[grid](
        a, w, c,
        a_scale, w_scale,
        M, N, K,
        a_block_size, w_block_size_k, w_block_size_n,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        w_scale.stride(0), w_scale.stride(1),
    )
    
    return c