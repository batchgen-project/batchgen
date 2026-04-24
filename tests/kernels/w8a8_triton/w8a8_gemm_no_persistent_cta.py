import torch
import triton
import triton.language as tl
from typing import Optional
import math


# ==================== KERNEL 1: SMALL-M OPTIMIZED (M <= 32) ====================
@triton.jit
def w8a8_gemm_small_m_kernel(
    a_ptr, w_ptr, c_ptr,
    a_scale_ptr, w_scale_ptr,
    M, N, K,
    a_block_size: tl.constexpr,
    w_block_size_k: tl.constexpr,
    w_block_size_n: tl.constexpr,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k,
    stride_w_scale_n, stride_w_scale_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    EXTREME optimization for small M (1-32).
    Strategy: Each program handles one M-row, process multiple N-blocks.
    This eliminates GROUP_SIZE_M overhead and maximizes N-direction vectorization.
    """
    pid_m = tl.program_id(axis=0)  # One program per M row
    pid_n = tl.program_id(axis=1)  # Multiple programs for N
    
    # This M-row
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    
    # This N-block
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    
    # FP32 accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K with software pipelining
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    
    for k_idx in range(num_k_blocks):
        k_start = k_idx * BLOCK_SIZE_K
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load scales (these are tiny, should stay in cache)
        a_k_block = k_start // a_block_size
        a_scale_ptrs = a_scale_ptr + offs_m * stride_a_scale_m + a_k_block * stride_a_scale_k
        a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
        
        w_k_block = k_start // w_block_size_k
        w_n_block = offs_n // w_block_size_n
        w_scale_ptrs = w_scale_ptr + w_n_block * stride_w_scale_n + w_k_block * stride_w_scale_k
        w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
        
        # Load activations and weights
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        
        a_mask = mask_m[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # FP8 matmul with scale
        partial = tl.dot(a_tile, tl.trans(w_tile), out_dtype=tl.float32)
        acc += partial * (a_scales[:, None] * w_scales[None, :])
        # acc += tl.dot(a_tile, tl.trans(w_tile), out_dtype=tl.float32) * (a_scales[:, None] * w_scales[None, :])
    
    # Store result
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ==================== KERNEL 2: MEDIUM-M WITH ASYNC PIPELINE (32 < M < 128) ====================
@triton.jit
def w8a8_gemm_medium_m_kernel(
    a_ptr, w_ptr, c_ptr,
    a_scale_ptr, w_scale_ptr,
    M, N, K,
    a_block_size: tl.constexpr,
    w_block_size_k: tl.constexpr,
    w_block_size_n: tl.constexpr,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k,
    stride_w_scale_n, stride_w_scale_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized for medium M with better swizzling and 2-stage pipeline.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Super-grouping for better L2 locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_am < M
    mask_n = offs_bn < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main K-loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load scales
        a_k_block_idx = k_start // a_block_size
        a_scale_ptrs = a_scale_ptr + offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k
        a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
        
        w_k_block_idx = k_start // w_block_size_k
        w_n_block_idx = offs_bn // w_block_size_n
        w_scale_ptrs = w_scale_ptr + w_n_block_idx * stride_w_scale_n + w_k_block_idx * stride_w_scale_k
        w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
        
        # Load data
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        
        a_mask = mask_m[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        
        a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Compute
        partial = tl.dot(a_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
        acc += partial * (a_scales[:, None] * w_scales[None, :])
    
    # Store
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ==================== KERNEL 3: LARGE-M HIGH-THROUGHPUT (M >= 128) ====================
@triton.jit
def w8a8_gemm_large_m_kernel(
    a_ptr, w_ptr, c_ptr,
    a_scale_ptr, w_scale_ptr,
    M, N, K,
    a_block_size: tl.constexpr,
    w_block_size_k: tl.constexpr,
    w_block_size_n: tl.constexpr,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k,
    stride_w_scale_n, stride_w_scale_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Large M: maximize throughput with larger tiles and more stages.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_am < M
    mask_n = offs_bn < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        a_k_block_idx = k_start // a_block_size
        a_scale_ptrs = a_scale_ptr + offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k
        a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
        
        w_k_block_idx = k_start // w_block_size_k
        w_n_block_idx = offs_bn // w_block_size_n
        w_scale_ptrs = w_scale_ptr + w_n_block_idx * stride_w_scale_n + w_k_block_idx * stride_w_scale_k
        w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
        
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        
        a_mask = mask_m[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        
        a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        partial = tl.dot(a_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
        acc += partial * (a_scales[:, None] * w_scales[None, :])
    
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ==================== ULTRA-FAST DISPATCH SYSTEM ====================
class W8A8GemmConfig:
    """Pre-optimized configurations to minimize JIT overhead."""
    
    # Small M configs (M <= 32): One warp per M-row
    SMALL_M_CONFIGS = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        (16, 128, 128, 4, 3),   # Ultra-small M (1-16)
        (32, 128, 128, 4, 3),   # Small M (17-32)
        # (64, 16, 128, 4, 2),   # Ultra-small M (1-16)
        # (64, 16, 128, 4, 2),   # Small M (17-32)
    ]
    
    # Medium M configs (32 < M < 128)
    MEDIUM_M_CONFIGS = [
        (64, 16, 128, 4, 2),   # M in [32, 64]
        (64, 16, 128, 4, 2),  # M in [64, 128]
    ]
    
    # Large M configs (M >= 128)
    LARGE_M_CONFIGS = [
        (64, 16, 128, 4, 3),  # M in [128, 512]
        (64, 16, 128, 4, 3),  # M >= 512, wide N
    ]
    
    @staticmethod
    def select_config(M: int, N: int, K: int):
        """Lightning-fast config selection with zero overhead."""
        if M <= 16:
            return (*W8A8GemmConfig.SMALL_M_CONFIGS[0], 4, 'small')
        elif M <= 32:
            return (*W8A8GemmConfig.SMALL_M_CONFIGS[1], 4, 'small')
        elif M <= 64:
            return (*W8A8GemmConfig.MEDIUM_M_CONFIGS[0], 8, 'medium')
        elif M < 128:
            return (*W8A8GemmConfig.MEDIUM_M_CONFIGS[1], 8, 'medium')
        elif M < 512 or N < 2048:
            return (*W8A8GemmConfig.LARGE_M_CONFIGS[0], 8, 'large')
        else:
            return (*W8A8GemmConfig.LARGE_M_CONFIGS[1], 8, 'large')


def w8a8_gemm_dispatch(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    a_block_size: int = 128,
    w_block_size_k: int = 128,
    w_block_size_n: int = 128,
) -> torch.Tensor:
    """
    🔥 EXTREME PERFORMANCE DISPATCH 🔥
    
    Zero-overhead kernel selection based on problem size.
    Uses specialized kernels for different M ranges.
    """
    M, K = a.shape
    N = w.shape[0]
    
    # Validate inputs
    assert a.dtype == torch.float8_e4m3fn or a.dtype == torch.float8_e5m2, "A must be FP8"
    assert w.dtype == torch.float8_e4m3fn or w.dtype == torch.float8_e5m2, "W must be FP8"
    assert a.is_contiguous() and w.is_contiguous(), "Tensors must be contiguous"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
    # Fast config selection
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kernel_type = \
        W8A8GemmConfig.select_config(M, N, K)
    
    # Select kernel and grid
    if kernel_type == 'small':
        # 2D grid: (M_blocks, N_blocks)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        kernel = w8a8_gemm_small_m_kernel[grid]
        
        kernel(
            a, w, c,
            a_scale, w_scale,
            M, N, K,
            a_block_size, w_block_size_k, w_block_size_n,
            a.stride(0), a.stride(1),
            w.stride(0), w.stride(1),
            c.stride(0), c.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    
    elif kernel_type == 'medium':
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        kernel = w8a8_gemm_medium_m_kernel[grid]
        
        kernel(
            a, w, c,
            a_scale, w_scale,
            M, N, K,
            a_block_size, w_block_size_k, w_block_size_n,
            a.stride(0), a.stride(1),
            w.stride(0), w.stride(1),
            c.stride(0), c.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    
    else:  # large
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        kernel = w8a8_gemm_large_m_kernel[grid]
        
        kernel(
            a, w, c,
            a_scale, w_scale,
            M, N, K,
            a_block_size, w_block_size_k, w_block_size_n,
            a.stride(0), a.stride(1),
            w.stride(0), w.stride(1),
            c.stride(0), c.stride(1),
            a_scale.stride(0), a_scale.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    
    return c


# ==================== KERNEL WARMUP ====================
def warmup_kernels(device='cuda'):
    """
    Pre-compile all kernel variants to eliminate first-call JIT overhead.
    Call this once during model initialization.
    """
    print("🔥 Warming up W8A8 GEMM kernels...")
    
    test_sizes = [
        (1, 4096, 4096),    # Tiny M
        (8, 4096, 11008),   # Small M
        (13, 4096, 11008),  # Small M
        (16, 4096, 11008),  # Small M
        (32, 4096, 11008),  # Medium M boundary
        (64, 4096, 11008),  # Medium M
        (82, 4096, 11008),  # Medium M
        (128, 4096, 11008), # Large M
        (200, 4096, 11008), # Large M
    ]
    
    for M, N, K in test_sizes:
        a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        a_scale = torch.ones(M, (K + 127) // 128, device=device, dtype=torch.float32)
        w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32)
        
        # Warmup call
        _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
    
    torch.cuda.synchronize()
    print("✅ Kernels warmed up and ready!")


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Warmup (do this once at model load time)
    warmup_kernels()
    
    # Example inference
    M, N, K = 16, 7168, 32768  # Typical LLM MLP shape
    
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    
    a_scale = torch.ones(M, (K + 127) // 128, device='cuda', dtype=torch.float32)
    w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device='cuda', dtype=torch.float32)
    
    # Fast dispatch
    result = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
    
    print(f"Output shape: {result.shape}, dtype: {result.dtype}")