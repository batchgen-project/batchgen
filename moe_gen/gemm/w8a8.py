
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
import math

# @triton.autotune(
#     configs=[
#         # Small tiles for small matrices
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        
#         # Medium tiles
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        
#         # Large tiles for large matrices
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#     ],
#     key=['M', 'N', 'K'],
# )
@triton.jit
def w8a8_gemm_kernel(
    # Pointers to matrices
    a_ptr, w_ptr, c_ptr,
    # Pointers to scales
    a_scale_ptr, w_scale_ptr,
    # Matrix dimensions
    M, N: tl.constexpr, K: tl.constexpr,
    # Quantization block sizes
    a_block_size, w_block_size_k, w_block_size_n,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,  # A is (M, K)
    stride_wn, stride_wk,  # W is (N, K) 
    stride_cm, stride_cn,  # C is (M, N)
    stride_a_scale_m, stride_a_scale_k,  # a_scale is (M, ceil(K/a_block_size))
    stride_w_scale_n, stride_w_scale_k,  # w_scale is (ceil(N/w_block_size_n), ceil(K/w_block_size_k))
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A @ W^T with quantized inputs.
    
    A has shape (M, K) with per-row quantization
    W has shape (N, K) with 2D block quantization
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
    w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)
    
    # Initialize the accumulator.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and W.
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a_fp8 = tl.load(a_ptrs, mask=k_mask, other=0.0)
        
        w_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Load quantization scales
        # For A: per-row quantization, scale varies across K dimension in blocks
        k_start = k * BLOCK_SIZE_K
        a_k_block_idx = k_start // a_block_size
        a_scale_ptrs = a_scale_ptr + (offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k)
        a_scales = tl.load(a_scale_ptrs, mask=offs_am < M, other=1.0)
        
        # For W: 2D block quantization
        # Calculate which blocks we're in for both N and K dimensions
        w_k_block_idx = k_start // w_block_size_k
        
        # For each element in offs_bn, calculate which N block it belongs to
        n_block_indices = offs_bn // w_block_size_n
        
        # Load the appropriate scales for each N block
        w_scale_ptrs = w_scale_ptr + (n_block_indices * stride_w_scale_n + w_k_block_idx * stride_w_scale_k)
        w_scales = tl.load(w_scale_ptrs, mask=offs_bn < N, other=1.0)

        # Apply scales and accumulate
        # a_scales is [BLOCK_SIZE_M], w_scales is [BLOCK_SIZE_N]
        # We need to broadcast them correctly for the dot product
        scaled_a = a_fp8 * a_scales[:, None]
        scaled_w = w_fp8 * w_scales[:, None]  # w_fp8 is [BLOCK_SIZE_N, BLOCK_SIZE_K]
        
        accumulator += tl.dot(scaled_a, tl.trans(scaled_w))
        
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
    a: torch.Tensor,           # [M, K] - activations in fp8e4m3
    a_scale: torch.Tensor,     # [M, ceil(K / a_block_size)] - per-row quantization scales
    w: torch.Tensor,           # [N, K] - weights in fp8e4m3  
    w_scale: torch.Tensor,     # [ceil(N/w_block_size_n), ceil(K/w_block_size_k)] - 2D block scales
    a_block_size: int = 128,   # Block size for A quantization
    w_block_size_k: int = 128, # Block size for W quantization along K
    w_block_size_n: int = 128, # Block size for W quantization along N
) -> torch.Tensor:
    """
    Performs quantized GEMM: C = A @ W^T
    This kernel assumes:
    - A is quantized with per-token (row) scales with block size `a_block_size`
    - W is quantized with 2D block scales with block sizes `w_block_size_k` and `w_block_size_n`
    - W is [N, K] 
    - Scales are in fp32 format.
    - Both A and W are in fp8e4m3 format
    - Output C is in bfloat16 format
    
    Args:
        a: Activations tensor [M, K] in fp8e4m3
        a_scale: Per-row quantization scales [M, ceil(K/a_block_size)]
        w: Weight tensor [N, K] in fp8e4m3
        w_scale: 2D block quantization scales [ceil(N/w_block_size_n), ceil(K/w_block_size_k)]
        a_block_size: Block size for activation quantization
        w_block_size_k: Block size for weight quantization along K dimension
        w_block_size_n: Block size for weight quantization along N dimension
    
    Returns:
        torch.Tensor: Output tensor [M, N] in bfloat16
    """
    # Check input shapes and dtypes
    assert a.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3 for activations, got {a.dtype}"
    assert w.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3 for weights, got {w.dtype}"
    assert a_scale.dtype == torch.float32, f"Expected float32 for a_scale, got {a_scale.dtype}"
    assert w_scale.dtype == torch.float32, f"Expected float32 for w_scale, got {w_scale.dtype}"
    
    # Get dimensions
    M, K = a.shape
    N, K_w = w.shape
    assert K == K_w, f"Dimension mismatch: A has {K} columns, W has {K_w} columns"
        
    # Validate scale tensor shapes
    expected_a_scale_shape = (M, math.ceil(K / a_block_size))
    expected_w_scale_shape = (math.ceil(N / w_block_size_n), math.ceil(K / w_block_size_k))
    
    assert a_scale.shape == expected_a_scale_shape, \
        f"a_scale shape mismatch: expected {expected_a_scale_shape}, got {a_scale.shape}"
    assert w_scale.shape == expected_w_scale_shape, \
        f"w_scale shape mismatch: expected {expected_w_scale_shape}, got {w_scale.shape}"
    
    # Allocate output tensor in bfloat16
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        
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
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=4
    )
    
    return c


def test_w8a8_gemm():
    # Example test case
    M, K, N = 128, 256, 64
    a = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn)
    a_scale = torch.randn(M, math.ceil(K / 128), dtype=torch.float32, device='cuda')
    w_scale = torch.randn(math.ceil(N / 128), math.ceil(K / 128), dtype=torch.float32, device='cuda')
    
    c = w8a8_gemm(a, a_scale, w, w_scale)
    assert c.dtype == torch.bfloat16
    assert c.shape == (M, N)
    print("Test passed! Output shape:", c.shape)



if __name__ == "__main__":
    test_w8a8_gemm()
    print("w8a8_gemm kernel is ready for use!")