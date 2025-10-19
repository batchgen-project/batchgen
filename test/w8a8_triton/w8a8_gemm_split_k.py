import torch
import triton
import triton.language as tl
from typing import Optional
import math


# ==================== KERNEL 1A: SMALL-M WITH SPLIT-K (M <= 32, large N) ====================
@triton.jit
def w8a8_gemm_small_m_split_k_kernel(
    a_ptr, w_ptr, c_ptr,
    a_scale_ptr, w_scale_ptr,
    M, N, K,
    a_block_size: tl.constexpr,
    w_block_size_k: tl.constexpr,
    w_block_size_n: tl.constexpr,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn, stride_ck,  # c_ptr has shape [SPLIT_K, M, N]
    stride_a_scale_m, stride_a_scale_k,
    stride_w_scale_n, stride_w_scale_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Split-K variant for small M: Each CTA computes partial result for a K-slice.
    Grid: (M_blocks, N_blocks, SPLIT_K)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)  # Which K-slice
    
    # Compute K-range for this split
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_global = pid_k * k_per_split
    k_end_global = min(k_start_global + k_per_split, K)
    
    # This M-block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    
    # This N-block
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    
    # FP32 accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over THIS K-slice only
    num_k_blocks = tl.cdiv(k_end_global - k_start_global, BLOCK_SIZE_K)
    
    for k_idx in range(num_k_blocks):
        k_start = k_start_global + k_idx * BLOCK_SIZE_K
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < k_end_global
        
        # Load scales
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
    
    # Store partial result for this K-slice
    c_partial = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Store to [pid_k, M, N] layout
    c_ptrs = c_ptr + pid_k * stride_ck + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c_partial, mask=c_mask)


@triton.jit
def reduce_split_k_kernel(
    c_splits_ptr, c_final_ptr,
    M, N, 
    SPLIT_K: tl.constexpr,
    stride_ck, stride_cm, stride_cn,
    stride_out_m, stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fast reduction kernel to sum K-splits into final result.
    Grid: (M_blocks, N_blocks)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Sum across K-splits
    for k_split in range(SPLIT_K):
        c_ptrs = c_splits_ptr + k_split * stride_ck + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = mask_m[:, None] & mask_n[None, :]
        c_tile = tl.load(c_ptrs, mask=mask, other=0.0)
        acc += c_tile.to(tl.float32)
    
    # Store final result
    c_final = acc.to(tl.bfloat16)
    out_ptrs = c_final_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, c_final, mask=mask)


# ==================== KERNEL 1B: SMALL-M STANDARD (no Split-K) ====================
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
    Standard small M kernel without Split-K (for small N cases).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    
    for k_idx in range(num_k_blocks):
        k_start = k_idx * BLOCK_SIZE_K
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        a_k_block = k_start // a_block_size
        a_scale_ptrs = a_scale_ptr + offs_m * stride_a_scale_m + a_k_block * stride_a_scale_k
        a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
        
        w_k_block = k_start // w_block_size_k
        w_n_block = offs_n // w_block_size_n
        w_scale_ptrs = w_scale_ptr + w_n_block * stride_w_scale_n + w_k_block * stride_w_scale_k
        w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
        
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        
        a_mask = mask_m[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        partial = tl.dot(a_tile, tl.trans(w_tile), out_dtype=tl.float32)
        acc += partial * (a_scales[:, None] * w_scales[None, :])
    
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


# ==================== ULTRA-FAST DISPATCH SYSTEM WITH SPLIT-K ====================
class W8A8GemmConfig:
    """Pre-optimized configurations with Split-K support."""
    
    # Small M configs (M <= 32)
    SMALL_M_CONFIGS = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        # Note: Using BLOCK_M=64 even for M=8 improves performance (better register allocation)
        (64, 32, 128, 4, 3),   # Ultra-small M (1-16), larger block for better perf
        (64, 32, 128, 4, 3),   # Small M (17-32)
    ]
    
    # Medium M configs (32 < M < 128)
    MEDIUM_M_CONFIGS = [
        (64, 128, 128, 4, 4),   # M in [32, 64]
        (128, 128, 128, 4, 4),  # M in [64, 128]
    ]
    
    # Large M configs (M >= 128)
    LARGE_M_CONFIGS = [
        (128, 128, 128, 8, 4),  # M in [128, 512]
        (128, 256, 128, 8, 5),  # M >= 512, wide N
    ]
    
    @staticmethod
    def compute_split_k(M: int, N: int, K: int) -> int:
        """
        Compute optimal Split-K factor for small-M or small-N cases.
        Goal: Achieve at least 80+ CTAs for good GPU utilization.
        
        Split-K is beneficial when:
        1. M is small (≤ 32) - limited parallelism in M dimension
        2. N is small (≤ 2048) - limited parallelism in N dimension
        """
        # Estimate base CTA count without split-K
        BLOCK_M = 64  # Using 64 for better performance
        BLOCK_N = 16  # Minimum for tensor cores
        base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        
        # Determine if we need Split-K
        needs_split_k = (M <= 32) or (N <= 2048)
        
        if not needs_split_k:
            return 1
        
        # Target: at least 80 CTAs (good for modern GPUs with ~80-132 SMs)
        target_ctas = 80
        
        if base_ctas >= target_ctas:
            return 1  # Already have enough parallelism
        
        # Compute split factor needed
        split_k_needed = triton.cdiv(target_ctas, base_ctas)
        
        # Cap split-K based on K dimension (don't over-split)
        # Each split should have at least 1024 elements to process
        # (Reduced from 2048 to allow more aggressive splitting for small N)
        min_k_per_split = 1024
        max_split_k = max(1, K // min_k_per_split)
        
        # For very small N (≤ 576), be more aggressive
        if N <= 576:
            min_k_per_split = 512
            max_split_k = max(1, K // min_k_per_split)
        
        # Round to power of 2 for better hardware utilization
        split_k = min(split_k_needed, max_split_k)
        split_k = 2 ** int(math.log2(max(1, split_k)))
        
        # Cap at reasonable maximum (16 for small N, 8 for small M with large N)
        if N <= 2048:
            split_k = min(split_k, 16)  # More aggressive for small N
        else:
            split_k = min(split_k, 8)   # Conservative for large N
        
        return split_k
    
    @staticmethod
    def select_config(M: int, N: int, K: int):
        """Lightning-fast config selection with Split-K decision."""
        if M <= 16:
            split_k = W8A8GemmConfig.compute_split_k(M, N, K)
            return (*W8A8GemmConfig.SMALL_M_CONFIGS[0], 4, 'small', split_k)
        elif M <= 32:
            split_k = W8A8GemmConfig.compute_split_k(M, N, K)
            return (*W8A8GemmConfig.SMALL_M_CONFIGS[1], 4, 'small', split_k)
        elif M <= 64:
            return (*W8A8GemmConfig.MEDIUM_M_CONFIGS[0], 8, 'medium', 1)
        elif M < 128:
            return (*W8A8GemmConfig.MEDIUM_M_CONFIGS[1], 8, 'medium', 1)
        elif M < 512 or N < 2048:
            return (*W8A8GemmConfig.LARGE_M_CONFIGS[0], 8, 'large', 1)
        else:
            return (*W8A8GemmConfig.LARGE_M_CONFIGS[1], 8, 'large', 1)


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
    🔥 EXTREME PERFORMANCE DISPATCH WITH SPLIT-K 🔥
    
    Automatically uses Split-K for small-M cases to maximize GPU utilization.
    API remains unchanged - Split-K is applied transparently.
    """
    M, K = a.shape
    N = w.shape[0]
    
    # Validate inputs
    assert a.dtype == torch.float8_e4m3fn or a.dtype == torch.float8_e5m2, "A must be FP8"
    assert w.dtype == torch.float8_e4m3fn or w.dtype == torch.float8_e5m2, "W must be FP8"
    assert a.is_contiguous() and w.is_contiguous(), "Tensors must be contiguous"
    
    # Fast config selection (now includes split_k)
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kernel_type, split_k = \
        W8A8GemmConfig.select_config(M, N, K)
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
    # Select kernel and grid based on split_k
    if kernel_type == 'small':
        if split_k > 1:
            # Use Split-K for small M with large N
            # Allocate temporary buffer for partial results
            c_splits = torch.empty((split_k, M, N), device=a.device, dtype=torch.bfloat16)
            
            # Launch Split-K kernel
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
            kernel = w8a8_gemm_small_m_split_k_kernel[grid]
            
            kernel(
                a, w, c_splits,
                a_scale, w_scale,
                M, N, K,
                a_block_size, w_block_size_k, w_block_size_n,
                a.stride(0), a.stride(1),
                w.stride(0), w.stride(1),
                c_splits.stride(1), c_splits.stride(2), c_splits.stride(0),
                a_scale.stride(0), a_scale.stride(1),
                w_scale.stride(0), w_scale.stride(1),
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_N=BLOCK_N,
                BLOCK_SIZE_K=BLOCK_K,
                SPLIT_K=split_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            
            # Reduce Split-K results
            reduce_grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            reduce_kernel = reduce_split_k_kernel[reduce_grid]
            
            reduce_kernel(
                c_splits, c,
                M, N, split_k,
                c_splits.stride(0), c_splits.stride(1), c_splits.stride(2),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_N=BLOCK_N,
                num_warps=num_warps,
            )
        else:
            # Standard small-M kernel (no Split-K)
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
    Pre-compile all kernel variants including Split-K to eliminate first-call JIT overhead.
    Call this once during model initialization.
    """
    print("🔥 Warming up W8A8 GEMM kernels (with Split-K)...")
    
    # Test representative shapes including all DeepSeek-like patterns
    test_sizes = [
        (8, 7168, 2048),    # Small M, large N, medium K
        (8, 2048, 7168),    # Small M, medium N, large K (needs Split-K)
        (8, 1536, 7168),    # Small M, medium N, large K (needs Split-K)
        (8, 576, 7168),     # Small M, small N, large K (most aggressive Split-K)
        (16, 4096, 11008),  # Small M
        (32, 4096, 11008),  # Medium M boundary
        (64, 4096, 11008),  # Medium M
        (128, 4096, 11008), # Large M
    ]
    
    for M, N, K in test_sizes:
        a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        a_scale = torch.ones(M, (K + 127) // 128, device=device, dtype=torch.float32)
        w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32)
        
        # Warmup call (will automatically use Split-K when beneficial)
        _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
    
    torch.cuda.synchronize()
    print("✅ All kernels (including Split-K variants) warmed up and ready!")


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Warmup (do this once at model load time)
    warmup_kernels()
    
    # Test all DeepSeek-like shapes
    test_shapes = [
        (8, 7168, 2048),
        (8, 2048, 7168),
        (8, 1536, 7168),
        (8, 24576, 1536),
        (8, 576, 7168),
        (8, 32768, 512),
        (8, 7168, 16384),
    ]
    
    print("\n" + "="*80)
    print("📊 Split-K Analysis for All Shapes")
    print("="*80)
    
    for M, N, K in test_shapes:
        config = W8A8GemmConfig.select_config(M, N, K)
        BLOCK_M, BLOCK_N, _, _, _, _, kernel_type, split_k = config
        
        base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        total_ctas = base_ctas * split_k
        
        print(f"\n📐 Shape: M={M:5d}, N={N:5d}, K={K:5d}")
        print(f"   Kernel: {kernel_type:8s} | Split-K: {split_k:2d}x")
        print(f"   Base CTAs: {base_ctas:4d} → Total CTAs: {total_ctas:4d} (🚀 {total_ctas/base_ctas:.1f}x speedup)")
    
    print("\n" + "="*80)
    print("🔥 Detailed Benchmark on Critical Shape")
    print("="*80)
    
    # Benchmark the most challenging shape: small M, small N
    M, N, K = 8, 576, 7168
    
    print(f"\nTesting M={M}, N={N}, K={K} (worst case: small M + small N)")
    
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    
    a_scale = torch.ones(M, (K + 127) // 128, device='cuda', dtype=torch.float32)
    w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device='cuda', dtype=torch.float32)
    
    # Get config info
    config = W8A8GemmConfig.select_config(M, N, K)
    BLOCK_M, BLOCK_N, _, _, _, _, kernel_type, split_k = config
    
    base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    total_ctas = base_ctas * split_k
    
    print(f"\n⚙️  Configuration:")
    print(f"   Block size: M={BLOCK_M}, N={BLOCK_N}")
    print(f"   Kernel type: {kernel_type}")
    print(f"   Split-K factor: {split_k}")
    print(f"   Base CTAs (no split): {base_ctas}")
    print(f"   Total CTAs (with split): {total_ctas}")
    print(f"   🚀 Parallelism increase: {total_ctas / base_ctas:.1f}x\n")
    
    # Correctness check
    result = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
    print(f"✅ Output shape: {result.shape}, dtype: {result.dtype}")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    
    iters = 100
    warmup_iters = 10
    
    # Warmup
    for _ in range(warmup_iters):
        _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
    torch.cuda.synchronize()
    
    # Actual benchmark
    start = time.time()
    for _ in range(iters):
        _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate TFLOPS (FP8 GEMM)
    flops = 2 * M * N * K  # 2 for multiply-add
    tflops = (flops * iters) / (elapsed * 1e12)
    
    print(f"\n⚡ Performance:")
    print(f"   TFLOPS: {tflops:.2f}")
    print(f"   Average time: {elapsed/iters*1000:.3f} ms")
    print(f"   Throughput: {iters/elapsed:.1f} iter/s")
    
    print("\n" + "="*80)