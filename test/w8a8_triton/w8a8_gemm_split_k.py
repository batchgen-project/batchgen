# import torch
# import triton
# import triton.language as tl
# from typing import Optional
# import math


# # ==================== KERNEL 1A: SMALL-M WITH SPLIT-K (M <= 32, large N) ====================
# @triton.jit
# def w8a8_gemm_small_m_split_k_kernel(
#     a_ptr, w_ptr, c_ptr,
#     a_scale_ptr, w_scale_ptr,
#     M, N, K,
#     a_block_size: tl.constexpr,
#     w_block_size_k: tl.constexpr,
#     w_block_size_n: tl.constexpr,
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn, stride_ck,  # c_ptr has shape [SPLIT_K, M, N]
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     SPLIT_K: tl.constexpr,
# ):
#     """
#     Split-K variant for small M: Each CTA computes partial result for a K-slice.
#     Grid: (M_blocks, N_blocks, SPLIT_K)
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
#     pid_k = tl.program_id(axis=2)  # Which K-slice
	
#     # Compute K-range for this split
#     k_per_split = tl.cdiv(K, SPLIT_K)
#     k_start_global = pid_k * k_per_split
#     k_end_global = min(k_start_global + k_per_split, K)
	
#     # This M-block
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     mask_m = offs_m < M
	
#     # This N-block
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_n = offs_n < N
	
#     # FP32 accumulator
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     # Iterate over THIS K-slice only
#     num_k_blocks = tl.cdiv(k_end_global - k_start_global, BLOCK_SIZE_K)
	
#     for k_idx in range(num_k_blocks):
#         k_start = k_start_global + k_idx * BLOCK_SIZE_K
#         offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
#         mask_k = offs_k < k_end_global
		
#         # Load scales
#         a_k_block = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + offs_m * stride_a_scale_m + a_k_block * stride_a_scale_k
#         a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
		
#         w_k_block = k_start // w_block_size_k
#         w_n_block = offs_n // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + w_n_block * stride_w_scale_n + w_k_block * stride_w_scale_k
#         w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
		
#         # Load activations and weights
#         a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
		
#         a_mask = mask_m[:, None] & mask_k[None, :]
#         w_mask = mask_n[:, None] & mask_k[None, :]
		
#         a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
		
#         # FP8 matmul with scale
#         partial = tl.dot(a_tile, tl.trans(w_tile), out_dtype=tl.float32)
#         acc += partial * (a_scales[:, None] * w_scales[None, :])
	
#     # Store partial result for this K-slice
#     c_partial = acc.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	
#     # Store to [pid_k, M, N] layout
#     c_ptrs = c_ptr + pid_k * stride_ck + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c_partial, mask=c_mask)


# @triton.jit
# def reduce_split_k_kernel(
#     c_splits_ptr, c_final_ptr,
#     M, N, 
#     SPLIT_K: tl.constexpr,
#     stride_ck, stride_cm, stride_cn,
#     stride_out_m, stride_out_n,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
# ):
#     """
#     Fast reduction kernel to sum K-splits into final result.
#     Grid: (M_blocks, N_blocks)
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
	
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_m = offs_m < M
#     mask_n = offs_n < N
	
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     # Sum across K-splits
#     for k_split in range(SPLIT_K):
#         c_ptrs = c_splits_ptr + k_split * stride_ck + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
#         mask = mask_m[:, None] & mask_n[None, :]
#         c_tile = tl.load(c_ptrs, mask=mask, other=0.0)
#         acc += c_tile.to(tl.float32)
	
#     # Store final result
#     c_final = acc.to(tl.bfloat16)
#     out_ptrs = c_final_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
#     mask = mask_m[:, None] & mask_n[None, :]
#     tl.store(out_ptrs, c_final, mask=mask)


# # ==================== KERNEL 1B: SMALL-M STANDARD (no Split-K) ====================
# @triton.jit
# def w8a8_gemm_small_m_kernel(
#     a_ptr, w_ptr, c_ptr,
#     a_scale_ptr, w_scale_ptr,
#     M, N, K,
#     a_block_size: tl.constexpr,
#     w_block_size_k: tl.constexpr,
#     w_block_size_n: tl.constexpr,
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     Standard small M kernel without Split-K (for small N cases).
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
	
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     mask_m = offs_m < M
	
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_n = offs_n < N
	
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
	
#     for k_idx in range(num_k_blocks):
#         k_start = k_idx * BLOCK_SIZE_K
#         offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
#         mask_k = offs_k < K
		
#         a_k_block = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + offs_m * stride_a_scale_m + a_k_block * stride_a_scale_k
#         a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
		
#         w_k_block = k_start // w_block_size_k
#         w_n_block = offs_n // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + w_n_block * stride_w_scale_n + w_k_block * stride_w_scale_k
#         w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
		
#         a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
		
#         a_mask = mask_m[:, None] & mask_k[None, :]
#         w_mask = mask_n[:, None] & mask_k[None, :]
		
#         a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
		
#         partial = tl.dot(a_tile, tl.trans(w_tile), out_dtype=tl.float32)
#         acc += partial * (a_scales[:, None] * w_scales[None, :])
	
#     c = acc.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# # ==================== KERNEL 2: MEDIUM-M WITH ASYNC PIPELINE (32 < M < 128) ====================
# @triton.jit
# def w8a8_gemm_medium_m_kernel(
#     a_ptr, w_ptr, c_ptr,
#     a_scale_ptr, w_scale_ptr,
#     M, N, K,
#     a_block_size: tl.constexpr,
#     w_block_size_k: tl.constexpr,
#     w_block_size_n: tl.constexpr,
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     """
#     Optimized for medium M with better swizzling and 2-stage pipeline.
#     """
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m
	
#     offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_m = offs_am < M
#     mask_n = offs_bn < N
	
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         k_start = k * BLOCK_SIZE_K
#         offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
#         mask_k = offs_k < K
		
#         a_k_block_idx = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k
#         a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
		
#         w_k_block_idx = k_start // w_block_size_k
#         w_n_block_idx = offs_bn // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + w_n_block_idx * stride_w_scale_n + w_k_block_idx * stride_w_scale_k
#         w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
		
#         a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)
		
#         a_mask = mask_m[:, None] & mask_k[None, :]
#         w_mask = mask_n[:, None] & mask_k[None, :]
		
#         a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
		
#         partial = tl.dot(a_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
#         acc += partial * (a_scales[:, None] * w_scales[None, :])
	
#     c = acc.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# # ==================== KERNEL 3: LARGE-M HIGH-THROUGHPUT (M >= 128) ====================
# @triton.jit
# def w8a8_gemm_large_m_kernel(
#     a_ptr, w_ptr, c_ptr,
#     a_scale_ptr, w_scale_ptr,
#     M, N, K,
#     a_block_size: tl.constexpr,
#     w_block_size_k: tl.constexpr,
#     w_block_size_n: tl.constexpr,
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     """
#     Large M: maximize throughput with larger tiles and more stages.
#     """
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m
	
#     offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_m = offs_am < M
#     mask_n = offs_bn < N
	
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         k_start = k * BLOCK_SIZE_K
#         offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
#         mask_k = offs_k < K
		
#         a_k_block_idx = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k
#         a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
		
#         w_k_block_idx = k_start // w_block_size_k
#         w_n_block_idx = offs_bn // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + w_n_block_idx * stride_w_scale_n + w_k_block_idx * stride_w_scale_k
#         w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
		
#         a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)
		
#         a_mask = mask_m[:, None] & mask_k[None, :]
#         w_mask = mask_n[:, None] & mask_k[None, :]
		
#         a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
		
#         partial = tl.dot(a_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
#         acc += partial * (a_scales[:, None] * w_scales[None, :])
	
#     c = acc.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# # ==================== ULTRA-FAST DISPATCH SYSTEM WITH SPLIT-K ====================
# class W8A8GemmConfig:
#     """Pre-optimized configurations with Split-K support."""
	
#     # Small M configs (M <= 32)
#     SMALL_M_CONFIGS = [
#         # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
#         # Note: Using BLOCK_M=64 even for M=8 improves performance (better register allocation)
#         (64, 128, 128, 4, 3),   # Ultra-small M (1-16), larger block for better perf
#         (64, 128, 128, 4, 3),   # Small M (17-32)
#     ]
	
#     # Medium M configs (32 < M < 128)
#     MEDIUM_M_CONFIGS = [
#         (64, 128, 128, 4, 4),   # M in [32, 64]
#         (128, 128, 128, 4, 4),  # M in [64, 128]
#     ]
	
#     # Large M configs (M >= 128)
#     LARGE_M_CONFIGS = [
#         (128, 128, 128, 8, 4),  # M in [128, 512]
#         (128, 256, 128, 8, 5),  # M >= 512, wide N
#     ]
	
#     @staticmethod
#     def compute_split_k(M: int, N: int, K: int) -> int:
#         """
#         Compute optimal Split-K factor for small-M or small-N cases.
#         Goal: Achieve at least 80+ CTAs for good GPU utilization.
		
#         Split-K is beneficial when:
#         1. M is small (≤ 32) - limited parallelism in M dimension
#         2. N is small (≤ 2048) - limited parallelism in N dimension
#         """
#         # Estimate base CTA count without split-K
#         BLOCK_M = 64  # Using 64 for better performance
#         BLOCK_N = 16  # Minimum for tensor cores
#         base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
		
#         # Determine if we need Split-K
#         needs_split_k = (M <= 32) or (N <= 2048)
		
#         if not needs_split_k:
#             return 1
		
#         # Target: at least 80 CTAs (good for modern GPUs with ~80-132 SMs)
#         target_ctas = 114
		
#         if base_ctas >= target_ctas:
#             return 1  # Already have enough parallelism
		
#         # Compute split factor needed
#         split_k_needed = triton.cdiv(target_ctas, base_ctas)
		
#         # Cap split-K based on K dimension (don't over-split)
#         # Each split should have at least 1024 elements to process
#         # (Reduced from 2048 to allow more aggressive splitting for small N)
#         min_k_per_split = 1024
#         max_split_k = max(1, K // min_k_per_split)
		
#         # For very small N (≤ 576), be more aggressive
#         if N <= 576:
#             min_k_per_split = 512
#             max_split_k = max(1, K // min_k_per_split)
		
#         # Round to power of 2 for better hardware utilization
#         split_k = min(split_k_needed, max_split_k)
#         split_k = 2 ** int(math.log2(max(1, split_k)))
		
#         # Cap at reasonable maximum (16 for small N, 8 for small M with large N)
#         if N <= 2048:
#             split_k = min(split_k, 8)  # More aggressive for small N
#         else:
#             split_k = min(split_k, 4)   # Conservative for large N
#         split_k = 1
#         return split_k
	
#     @staticmethod
#     def select_config(M: int, N: int, K: int):
#         """Lightning-fast config selection with Split-K decision."""
#         if M <= 16:
#             split_k = W8A8GemmConfig.compute_split_k(M, N, K)
#             return (*W8A8GemmConfig.SMALL_M_CONFIGS[0], 4, 'small', split_k)
#         elif M <= 32:
#             split_k = W8A8GemmConfig.compute_split_k(M, N, K)
#             return (*W8A8GemmConfig.SMALL_M_CONFIGS[1], 4, 'small', split_k)
#         elif M <= 64:
#             return (*W8A8GemmConfig.MEDIUM_M_CONFIGS[0], 8, 'medium', 1)
#         elif M < 128:
#             return (*W8A8GemmConfig.MEDIUM_M_CONFIGS[1], 8, 'medium', 1)
#         elif M < 512 or N < 2048:
#             return (*W8A8GemmConfig.LARGE_M_CONFIGS[0], 8, 'large', 1)
#         else:
#             return (*W8A8GemmConfig.LARGE_M_CONFIGS[1], 8, 'large', 1)


# def w8a8_gemm_dispatch(
#     a: torch.Tensor,
#     a_scale: torch.Tensor,
#     w: torch.Tensor,
#     w_scale: torch.Tensor,
#     a_block_size: int = 128,
#     w_block_size_k: int = 128,
#     w_block_size_n: int = 128,
# ) -> torch.Tensor:
#     """
#     🔥 EXTREME PERFORMANCE DISPATCH WITH SPLIT-K 🔥
	
#     Automatically uses Split-K for small-M cases to maximize GPU utilization.
#     API remains unchanged - Split-K is applied transparently.
#     """
#     M, K = a.shape
#     N = w.shape[0]
	
#     # Validate inputs
#     assert a.dtype == torch.float8_e4m3fn or a.dtype == torch.float8_e5m2, "A must be FP8"
#     assert w.dtype == torch.float8_e4m3fn or w.dtype == torch.float8_e5m2, "W must be FP8"
#     assert a.is_contiguous() and w.is_contiguous(), "Tensors must be contiguous"
	
#     # Fast config selection (now includes split_k)
#     BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kernel_type, split_k = \
#         W8A8GemmConfig.select_config(M, N, K)
#     # Allocate output
#     c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
	
#     # Select kernel and grid based on split_k
#     if kernel_type == 'small':
#         if split_k > 1:
#             # Use Split-K for small M with large N
#             # Allocate temporary buffer for partial results
#             c_splits = torch.empty((split_k, M, N), device=a.device, dtype=torch.bfloat16)
			
#             # Launch Split-K kernel
#             grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
#             kernel = w8a8_gemm_small_m_split_k_kernel[grid]
			
#             kernel(
#                 a, w, c_splits,
#                 a_scale, w_scale,
#                 M, N, K,
#                 a_block_size, w_block_size_k, w_block_size_n,
#                 a.stride(0), a.stride(1),
#                 w.stride(0), w.stride(1),
#                 c_splits.stride(1), c_splits.stride(2), c_splits.stride(0),
#                 a_scale.stride(0), a_scale.stride(1),
#                 w_scale.stride(0), w_scale.stride(1),
#                 BLOCK_SIZE_M=BLOCK_M,
#                 BLOCK_SIZE_N=BLOCK_N,
#                 BLOCK_SIZE_K=BLOCK_K,
#                 SPLIT_K=split_k,
#                 num_warps=num_warps,
#                 num_stages=num_stages,
#             )
			
#             # Reduce Split-K results
#             reduce_grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
#             reduce_kernel = reduce_split_k_kernel[reduce_grid]
			
#             reduce_kernel(
#                 c_splits, c,
#                 M, N, split_k,
#                 c_splits.stride(0), c_splits.stride(1), c_splits.stride(2),
#                 c.stride(0), c.stride(1),
#                 BLOCK_SIZE_M=BLOCK_M,
#                 BLOCK_SIZE_N=BLOCK_N,
#                 num_warps=num_warps,
#             )
#         else:
#             # Standard small-M kernel (no Split-K)
#             grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
#             kernel = w8a8_gemm_small_m_kernel[grid]
			
#             kernel(
#                 a, w, c,
#                 a_scale, w_scale,
#                 M, N, K,
#                 a_block_size, w_block_size_k, w_block_size_n,
#                 a.stride(0), a.stride(1),
#                 w.stride(0), w.stride(1),
#                 c.stride(0), c.stride(1),
#                 a_scale.stride(0), a_scale.stride(1),
#                 w_scale.stride(0), w_scale.stride(1),
#                 BLOCK_SIZE_M=BLOCK_M,
#                 BLOCK_SIZE_N=BLOCK_N,
#                 BLOCK_SIZE_K=BLOCK_K,
#                 num_warps=num_warps,
#                 num_stages=num_stages,
#             )
	
#     elif kernel_type == 'medium':
#         grid = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#         )
#         kernel = w8a8_gemm_medium_m_kernel[grid]
		
#         kernel(
#             a, w, c,
#             a_scale, w_scale,
#             M, N, K,
#             a_block_size, w_block_size_k, w_block_size_n,
#             a.stride(0), a.stride(1),
#             w.stride(0), w.stride(1),
#             c.stride(0), c.stride(1),
#             a_scale.stride(0), a_scale.stride(1),
#             w_scale.stride(0), w_scale.stride(1),
#             BLOCK_SIZE_M=BLOCK_M,
#             BLOCK_SIZE_N=BLOCK_N,
#             BLOCK_SIZE_K=BLOCK_K,
#             GROUP_SIZE_M=GROUP_M,
#             num_warps=num_warps,
#             num_stages=num_stages,
#         )
	
#     else:  # large
#         grid = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#         )
#         kernel = w8a8_gemm_large_m_kernel[grid]
		
#         kernel(
#             a, w, c,
#             a_scale, w_scale,
#             M, N, K,
#             a_block_size, w_block_size_k, w_block_size_n,
#             a.stride(0), a.stride(1),
#             w.stride(0), w.stride(1),
#             c.stride(0), c.stride(1),
#             a_scale.stride(0), a_scale.stride(1),
#             w_scale.stride(0), w_scale.stride(1),
#             BLOCK_SIZE_M=BLOCK_M,
#             BLOCK_SIZE_N=BLOCK_N,
#             BLOCK_SIZE_K=BLOCK_K,
#             GROUP_SIZE_M=GROUP_M,
#             num_warps=num_warps,
#             num_stages=num_stages,
#         )
	
#     return c


# # ==================== KERNEL WARMUP ====================
# def warmup_kernels(device='cuda'):
#     """
#     Pre-compile all kernel variants including Split-K to eliminate first-call JIT overhead.
#     Call this once during model initialization.
#     """
#     print("🔥 Warming up W8A8 GEMM kernels (with Split-K)...")
	
#     # Test representative shapes including all DeepSeek-like patterns
#     test_sizes = [
#         (8, 7168, 2048),    # Small M, large N, medium K
#         (8, 2048, 7168),    # Small M, medium N, large K (needs Split-K)
#         (8, 1536, 7168),    # Small M, medium N, large K (needs Split-K)
#         (8, 576, 7168),     # Small M, small N, large K (most aggressive Split-K)
#         (16, 4096, 11008),  # Small M
#         (32, 4096, 11008),  # Medium M boundary
#         (64, 4096, 11008),  # Medium M
#         (128, 4096, 11008), # Large M
#     ]
	
#     for M, N, K in test_sizes:
#         a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#         w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#         a_scale = torch.ones(M, (K + 127) // 128, device=device, dtype=torch.float32)
#         w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32)
		
#         # Warmup call (will automatically use Split-K when beneficial)
#         _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
	
#     torch.cuda.synchronize()
#     print("✅ All kernels (including Split-K variants) warmed up and ready!")


# # ==================== USAGE EXAMPLE ====================
# if __name__ == "__main__":
#     # Warmup (do this once at model load time)
#     warmup_kernels()
	
#     # Test all DeepSeek-like shapes
#     test_shapes = [
#         (8, 7168, 2048),
#         (8, 2048, 7168),
#         (8, 1536, 7168),
#         (8, 24576, 1536),
#         (8, 576, 7168),
#         (8, 32768, 512),
#         (8, 7168, 16384),
#     ]
	
#     print("\n" + "="*80)
#     print("📊 Split-K Analysis for All Shapes")
#     print("="*80)
	
#     for M, N, K in test_shapes:
#         config = W8A8GemmConfig.select_config(M, N, K)
#         BLOCK_M, BLOCK_N, _, _, _, _, kernel_type, split_k = config
		
#         base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
#         total_ctas = base_ctas * split_k
		
#         print(f"\n📐 Shape: M={M:5d}, N={N:5d}, K={K:5d}")
#         print(f"   Kernel: {kernel_type:8s} | Split-K: {split_k:2d}x")
#         print(f"   Base CTAs: {base_ctas:4d} → Total CTAs: {total_ctas:4d} (🚀 {total_ctas/base_ctas:.1f}x speedup)")
	
#     print("\n" + "="*80)
#     print("🔥 Detailed Benchmark on Critical Shape")
#     print("="*80)
	
#     # Benchmark the most challenging shape: small M, small N
#     M, N, K = 8, 576, 7168
	
#     print(f"\nTesting M={M}, N={N}, K={K} (worst case: small M + small N)")
	
#     a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#     w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
	
#     a_scale = torch.ones(M, (K + 127) // 128, device='cuda', dtype=torch.float32)
#     w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device='cuda', dtype=torch.float32)
	
#     # Get config info
#     config = W8A8GemmConfig.select_config(M, N, K)
#     BLOCK_M, BLOCK_N, _, _, _, _, kernel_type, split_k = config
	
#     base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
#     total_ctas = base_ctas * split_k
	
#     print(f"\n⚙️  Configuration:")
#     print(f"   Block size: M={BLOCK_M}, N={BLOCK_N}")
#     print(f"   Kernel type: {kernel_type}")
#     print(f"   Split-K factor: {split_k}")
#     print(f"   Base CTAs (no split): {base_ctas}")
#     print(f"   Total CTAs (with split): {total_ctas}")
#     print(f"   🚀 Parallelism increase: {total_ctas / base_ctas:.1f}x\n")
	
#     # Correctness check
#     result = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
#     print(f"✅ Output shape: {result.shape}, dtype: {result.dtype}")
	
#     # Benchmark
#     import time
#     torch.cuda.synchronize()
	
#     iters = 100
#     warmup_iters = 10
	
#     # Warmup
#     for _ in range(warmup_iters):
#         _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
#     torch.cuda.synchronize()
	
#     # Actual benchmark
#     start = time.time()
#     for _ in range(iters):
#         _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
	
#     # Calculate TFLOPS (FP8 GEMM)
#     flops = 2 * M * N * K  # 2 for multiply-add
#     tflops = (flops * iters) / (elapsed * 1e12)
	
#     print(f"\n⚡ Performance:")
#     print(f"   TFLOPS: {tflops:.2f}")
#     print(f"   Average time: {elapsed/iters*1000:.3f} ms")
#     print(f"   Throughput: {iters/elapsed:.1f} iter/s")
	
#     print("\n" + "="*80)

# import torch
# import triton
# import triton.language as tl
# from typing import Optional
# import math


# # ==================== KERNEL 1A: SMALL-M WITH SPLIT-K (M <= 32, large N) ====================
# @triton.jit
# def w8a8_gemm_small_m_split_k_kernel(
#     a_ptr, w_ptr, c_ptr,
#     a_scale_ptr, w_scale_ptr,
#     M, N, K,
#     a_block_size: tl.constexpr,
#     w_block_size_k: tl.constexpr,
#     w_block_size_n: tl.constexpr,
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn, stride_ck,  # c_ptr has shape [SPLIT_K, M, N]
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     SPLIT_K: tl.constexpr,
# ):
#     """
#     Split-K variant for small M: Each CTA computes partial result for a K-slice.
#     Grid: (M_blocks, N_blocks, SPLIT_K)
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
#     pid_k = tl.program_id(axis=2)  # Which K-slice
	
#     # Compute K-range for this split
#     k_per_split = tl.cdiv(K, SPLIT_K)
#     k_start_global = pid_k * k_per_split
#     k_end_global = min(k_start_global + k_per_split, K)
	
#     # This M-block
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     mask_m = offs_m < M
	
#     # This N-block
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_n = offs_n < N
	
#     # FP32 accumulator
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     # Iterate over THIS K-slice only
#     num_k_blocks = tl.cdiv(k_end_global - k_start_global, BLOCK_SIZE_K)
	
#     for k_idx in range(num_k_blocks):
#         k_start = k_start_global + k_idx * BLOCK_SIZE_K
#         offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
#         mask_k = offs_k < k_end_global
		
#         # Load scales
#         a_k_block = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + offs_m * stride_a_scale_m + a_k_block * stride_a_scale_k
#         a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
		
#         w_k_block = k_start // w_block_size_k
#         w_n_block = offs_n // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + w_n_block * stride_w_scale_n + w_k_block * stride_w_scale_k
#         w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
		
#         # Load activations and weights
#         a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
		
#         a_mask = mask_m[:, None] & mask_k[None, :]
#         w_mask = mask_n[:, None] & mask_k[None, :]
		
#         a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
		
#         # FP8 matmul with scale
#         partial = tl.dot(a_tile, tl.trans(w_tile), out_dtype=tl.float32)
#         acc += partial * (a_scales[:, None] * w_scales[None, :])
	
#     # Store partial result for this K-slice
#     c_partial = acc.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	
#     # Store to [pid_k, M, N] layout
#     c_ptrs = c_ptr + pid_k * stride_ck + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c_partial, mask=c_mask)


# @triton.jit
# def reduce_split_k_kernel(
#     c_splits_ptr, c_final_ptr,
#     M, N, 
#     SPLIT_K: tl.constexpr,
#     stride_ck, stride_cm, stride_cn,
#     stride_out_m, stride_out_n,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
# ):
#     """
#     Fast reduction kernel to sum K-splits into final result.
#     Grid: (M_blocks, N_blocks)
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
	
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_m = offs_m < M
#     mask_n = offs_n < N
	
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     # Sum across K-splits
#     for k_split in range(SPLIT_K):
#         c_ptrs = c_splits_ptr + k_split * stride_ck + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
#         mask = mask_m[:, None] & mask_n[None, :]
#         c_tile = tl.load(c_ptrs, mask=mask, other=0.0)
#         acc += c_tile.to(tl.float32)
	
#     # Store final result
#     c_final = acc.to(tl.bfloat16)
#     out_ptrs = c_final_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
#     mask = mask_m[:, None] & mask_n[None, :]
#     tl.store(out_ptrs, c_final, mask=mask)


# # ==================== KERNEL 1B: SMALL-M STANDARD (no Split-K) ====================
# @triton.jit
# def w8a8_gemm_small_m_kernel(
#     a_ptr, w_ptr, c_ptr,
#     a_scale_ptr, w_scale_ptr,
#     M, N, K,
#     a_block_size: tl.constexpr,
#     w_block_size_k: tl.constexpr,
#     w_block_size_n: tl.constexpr,
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     Standard small M kernel without Split-K (for small N cases).
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
	
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     mask_m = offs_m < M
	
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_n = offs_n < N
	
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
	
#     for k_idx in range(num_k_blocks):
#         k_start = k_idx * BLOCK_SIZE_K
#         offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
#         mask_k = offs_k < K
		
#         a_k_block = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + offs_m * stride_a_scale_m + a_k_block * stride_a_scale_k
#         a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
		
#         w_k_block = k_start // w_block_size_k
#         w_n_block = offs_n // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + w_n_block * stride_w_scale_n + w_k_block * stride_w_scale_k
#         w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
		
#         a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
		
#         a_mask = mask_m[:, None] & mask_k[None, :]
#         w_mask = mask_n[:, None] & mask_k[None, :]
		
#         a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
		
#         partial = tl.dot(a_tile, tl.trans(w_tile), out_dtype=tl.float32)
#         acc += partial * (a_scales[:, None] * w_scales[None, :])
	
#     c = acc.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# # ==================== KERNEL 2: MEDIUM-M WITH ASYNC PIPELINE (32 < M < 128) ====================
# @triton.jit
# def w8a8_gemm_medium_m_kernel(
#     a_ptr, w_ptr, c_ptr,
#     a_scale_ptr, w_scale_ptr,
#     M, N, K,
#     a_block_size: tl.constexpr,
#     w_block_size_k: tl.constexpr,
#     w_block_size_n: tl.constexpr,
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     """
#     Optimized for medium M with better swizzling and 2-stage pipeline.
#     """
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m
	
#     offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_m = offs_am < M
#     mask_n = offs_bn < N
	
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         k_start = k * BLOCK_SIZE_K
#         offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
#         mask_k = offs_k < K
		
#         a_k_block_idx = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k
#         a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
		
#         w_k_block_idx = k_start // w_block_size_k
#         w_n_block_idx = offs_bn // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + w_n_block_idx * stride_w_scale_n + w_k_block_idx * stride_w_scale_k
#         w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
		
#         a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)
		
#         a_mask = mask_m[:, None] & mask_k[None, :]
#         w_mask = mask_n[:, None] & mask_k[None, :]
		
#         a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
		
#         partial = tl.dot(a_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
#         acc += partial * (a_scales[:, None] * w_scales[None, :])
	
#     c = acc.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# # ==================== KERNEL 3: LARGE-M HIGH-THROUGHPUT (M >= 128) ====================
# @triton.jit
# def w8a8_gemm_large_m_kernel(
#     a_ptr, w_ptr, c_ptr,
#     a_scale_ptr, w_scale_ptr,
#     M, N, K,
#     a_block_size: tl.constexpr,
#     w_block_size_k: tl.constexpr,
#     w_block_size_n: tl.constexpr,
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     """
#     Large M: maximize throughput with larger tiles and more stages.
#     """
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m
	
#     offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     mask_m = offs_am < M
#     mask_n = offs_bn < N
	
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         k_start = k * BLOCK_SIZE_K
#         offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
#         mask_k = offs_k < K
		
#         a_k_block_idx = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k
#         a_scales = tl.load(a_scale_ptrs, mask=mask_m, other=1.0)
		
#         w_k_block_idx = k_start // w_block_size_k
#         w_n_block_idx = offs_bn // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + w_n_block_idx * stride_w_scale_n + w_k_block_idx * stride_w_scale_k
#         w_scales = tl.load(w_scale_ptrs, mask=mask_n, other=1.0)
		
#         a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)
		
#         a_mask = mask_m[:, None] & mask_k[None, :]
#         w_mask = mask_n[:, None] & mask_k[None, :]
		
#         a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
		
#         partial = tl.dot(a_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
#         acc += partial * (a_scales[:, None] * w_scales[None, :])
	
#     c = acc.to(tl.bfloat16)
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# # ==================== ULTRA-FAST DISPATCH SYSTEM WITH SPLIT-K ====================
# class W8A8GemmConfig:
#     """
#     Pre-optimized configurations with device-aware Split-K and shape-specific strategies.
	
#     Two-tier strategy:
	
#     **BAD SHAPES** (N ≤ 1536 AND K > 4096): "Large Tile Strategy"
#     - Problem: Can't generate enough CTAs due to small M×N
#     - Solution: Use large BLOCK_N (128-256) to maximize work per CTA
#     - Benefits: Better memory coalescing, better cache utilization, no Split-K overhead
#     - Example: (8, 576, 7168) → BLOCK_M=16, BLOCK_N=128, Split-K=1
	
#     **GOOD SHAPES** (all others): "Default Tiling with Split-K"
#     - Use standard BLOCK_N=32 (good balance)
#     - Apply Split-K when K ≤ 4096 to increase parallelism
#     - Benefits: Good parallelism, reasonable efficiency
#     - Example: (8, 2048, 7168) → BLOCK_M=16, BLOCK_N=32, Split-K=4
	
#     Key principles:
#     1. Adaptive BLOCK_M: 16 for M≤16, 32 for M≤32 (minimizes padding)
#     2. Bad shapes: Maximize efficiency per CTA (few CTAs doing good work)
#     3. Good shapes: Maximize parallelism with Split-K (many CTAs)
#     4. K-dimension awareness: Limit Split-K for K > 4096
#     """
	
#     # Small M configs (M <= 32)
#     # Note: BLOCK_M and BLOCK_N are now determined dynamically by compute_optimal_config()
#     # to minimize padding waste and overhead
#     SMALL_M_BASE_CONFIG = (128, 4, 3)  # (BLOCK_K, num_warps, num_stages)
	
#     # Medium M configs (32 < M < 128)
#     MEDIUM_M_CONFIGS = [
#         (64, 128, 128, 4, 4),   # M in [32, 64]
#         (128, 128, 128, 4, 4),  # M in [64, 128]
#     ]
	
#     # Large M configs (M >= 128)
#     LARGE_M_CONFIGS = [
#         (128, 128, 128, 8, 4),  # M in [128, 512]
#         (128, 256, 128, 8, 5),  # M >= 512, wide N
#     ]
	
#     # Cache SM count to avoid repeated queries
#     _sm_count_cache = {}
	
#     @staticmethod
#     def get_sm_count(device) -> int:
#         """Get the number of SMs for the given device."""
#         device_idx = device.index if device.index is not None else 0
		
#         if device_idx not in W8A8GemmConfig._sm_count_cache:
#             props = torch.cuda.get_device_properties(device_idx)
#             W8A8GemmConfig._sm_count_cache[device_idx] = props.multi_processor_count
		
#         return W8A8GemmConfig._sm_count_cache[device_idx]
	
#     @staticmethod
#     def compute_optimal_config(M: int, N: int, K: int, device, oversubscribe_factor: float = 1.5):
#         """
#         Compute optimal BLOCK_N and Split-K factor together.
		
#         Two-tier strategy:
#         1. BAD SHAPES (N ≤ 1536 AND K > 4096): Always use BLOCK_N=128, Split-K=1
#            - Can't get parallelism anyway, so maximize efficiency per CTA
#         2. GOOD SHAPES (all others): Use BLOCK_N=32-64 with Split-K for parallelism
#            - Standard approach with good balance
		
#         Args:
#             M, N, K: Problem dimensions
#             device: Torch device
#             oversubscribe_factor: Target CTAs = num_SMs * this factor
		
#         Returns:
#             (BLOCK_M, BLOCK_N, split_k): Optimal block sizes and split factor
#         """
#         # Get device SM count
#         num_sms = W8A8GemmConfig.get_sm_count(device)
#         target_ctas = int(num_sms * oversubscribe_factor)
		
#         # Choose BLOCK_M based on M size to minimize padding waste
#         if M <= 16:
#             BLOCK_M = 16
#         elif M <= 32:
#             BLOCK_M = 32
#         else:
#             BLOCK_M = 64
		
#         # CRITICAL: Identify "bad shapes" - small N + large K
#         # These are fundamentally CTA-limited: (8, 576, 7168) and (8, 1536, 7168)
#         is_bad_shape = (N <= 1536) and (K > 4096)
		
#         if is_bad_shape:
#             # BAD SHAPE STRATEGY: Large tiles, no Split-K
#             # Rationale: Can't get parallelism (M×N too small), so maximize efficiency
#             # ALWAYS use BLOCK_N=128 for these shapes
#             return BLOCK_M, 128, 1
		
#         # GOOD SHAPES: Use default tiling with Split-K
#         # Large problem case
#         if M > 32 and N > 2048:
#             return BLOCK_M, 128, 1
		
#         # Default: BLOCK_N=64 for good balance between coalescing and parallelism
#         BLOCK_N = 64
		
#         # Compute base CTAs
#         base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
		
#         # If already have good parallelism, done
#         if base_ctas >= target_ctas:
#             return BLOCK_M, BLOCK_N, 1
		
#         # Need Split-K for more parallelism
#         split_k_needed = triton.cdiv(target_ctas, base_ctas)
		
#         # Determine Split-K based on K size
#         if K <= 2048:
#             # Small K: can split aggressively
#             min_k_per_split = 512
#             max_split_k = max(1, K // min_k_per_split)
#             max_split_k = min(max_split_k, 8)
#         elif K <= 4096:
#             # Medium K: moderate splitting
#             min_k_per_split = 1024
#             max_split_k = max(1, K // min_k_per_split)
#             max_split_k = min(max_split_k, 4)
#         else:
#             # Large K (> 4096): Conservative splitting
#             min_k_per_split = 2048
#             max_split_k = max(1, K // min_k_per_split)
#             max_split_k = min(max_split_k, 4)
			
#             # For large K shapes with insufficient parallelism, try smaller BLOCK_N
#             if base_ctas < num_sms // 2:
#                 BLOCK_N = 32
#                 base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
#                 split_k_needed = triton.cdiv(target_ctas, base_ctas)
		
#         # Compute Split-K (power of 2)
#         split_k = min(split_k_needed, max_split_k)
#         split_k = 2 ** int(math.log2(max(1, split_k)))
		
#         return BLOCK_M, BLOCK_N, split_k
	
#     @staticmethod
#     def select_config(M: int, N: int, K: int, device):
#         """
#         Lightning-fast config selection with device-aware Split-K and adaptive BLOCK_M/BLOCK_N.
#         Returns: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kernel_type, split_k)
#         """
#         if M <= 32:
#             # Small M: Use adaptive config to minimize padding and overhead
#             BLOCK_M, BLOCK_N, split_k = W8A8GemmConfig.compute_optimal_config(M, N, K, device)
#             return (BLOCK_M, BLOCK_N, 128, 4, 3, 4, 'small', split_k)
#         elif M <= 64:
#             return (64, 128, 128, 4, 4, 8, 'medium', 1)
#         elif M < 128:
#             return (128, 128, 128, 4, 4, 8, 'medium', 1)
#         elif M < 512 or N < 2048:
#             return (128, 128, 128, 8, 4, 8, 'large', 1)
#         else:
#             return (128, 256, 128, 8, 5, 8, 'large', 1)


# def w8a8_gemm_dispatch(
#     a: torch.Tensor,
#     a_scale: torch.Tensor,
#     w: torch.Tensor,
#     w_scale: torch.Tensor,
#     a_block_size: int = 128,
#     w_block_size_k: int = 128,
#     w_block_size_n: int = 128,
# ) -> torch.Tensor:
#     """
#     🔥 EXTREME PERFORMANCE DISPATCH WITH DEVICE-AWARE ADAPTIVE CONFIG 🔥
	
#     Optimizations:
#     1. Device-aware Split-K: Queries actual SM count, targets num_SMs * 1.5 CTAs
#     2. Adaptive BLOCK_N: Prefers larger blocks (64/32) if Split-K can provide enough CTAs
#     3. BLOCK_M = 64: Empirically faster than 16/32 for small M
	
#     Trade-off: Fewer large tiles (better efficiency) vs many small tiles (more parallelism)
#     Solution: Use Split-K to maintain parallelism while using larger tiles
	
#     API remains unchanged - all optimizations applied transparently.
#     """
#     M, K = a.shape
#     N = w.shape[0]
	
#     # Validate inputs
#     assert a.dtype == torch.float8_e4m3fn or a.dtype == torch.float8_e5m2, "A must be FP8"
#     assert w.dtype == torch.float8_e4m3fn or w.dtype == torch.float8_e5m2, "W must be FP8"
#     assert a.is_contiguous() and w.is_contiguous(), "Tensors must be contiguous"
	
#     # Fast config selection with adaptive BLOCK_N and device-aware Split-K
#     BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kernel_type, split_k = \
#         W8A8GemmConfig.select_config(M, N, K, a.device)
	
#     # Allocate output
#     c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
	
#     # Select kernel and grid based on split_k
#     if kernel_type == 'small':
#         if split_k > 1:
#             # Use Split-K for small M with large N
#             # Allocate temporary buffer for partial results
#             c_splits = torch.empty((split_k, M, N), device=a.device, dtype=torch.bfloat16)
			
#             # Launch Split-K kernel
#             grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
#             kernel = w8a8_gemm_small_m_split_k_kernel[grid]
			
#             kernel(
#                 a, w, c_splits,
#                 a_scale, w_scale,
#                 M, N, K,
#                 a_block_size, w_block_size_k, w_block_size_n,
#                 a.stride(0), a.stride(1),
#                 w.stride(0), w.stride(1),
#                 c_splits.stride(1), c_splits.stride(2), c_splits.stride(0),
#                 a_scale.stride(0), a_scale.stride(1),
#                 w_scale.stride(0), w_scale.stride(1),
#                 BLOCK_SIZE_M=BLOCK_M,
#                 BLOCK_SIZE_N=BLOCK_N,
#                 BLOCK_SIZE_K=BLOCK_K,
#                 SPLIT_K=split_k,
#                 num_warps=num_warps,
#                 num_stages=num_stages,
#             )
			
#             # Reduce Split-K results
#             reduce_grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
#             reduce_kernel = reduce_split_k_kernel[reduce_grid]
			
#             reduce_kernel(
#                 c_splits, c,
#                 M, N, split_k,
#                 c_splits.stride(0), c_splits.stride(1), c_splits.stride(2),
#                 c.stride(0), c.stride(1),
#                 BLOCK_SIZE_M=BLOCK_M,
#                 BLOCK_SIZE_N=BLOCK_N,
#                 num_warps=num_warps,
#             )
#         else:
#             # Standard small-M kernel (no Split-K)
#             grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
#             kernel = w8a8_gemm_small_m_kernel[grid]
			
#             kernel(
#                 a, w, c,
#                 a_scale, w_scale,
#                 M, N, K,
#                 a_block_size, w_block_size_k, w_block_size_n,
#                 a.stride(0), a.stride(1),
#                 w.stride(0), w.stride(1),
#                 c.stride(0), c.stride(1),
#                 a_scale.stride(0), a_scale.stride(1),
#                 w_scale.stride(0), w_scale.stride(1),
#                 BLOCK_SIZE_M=BLOCK_M,
#                 BLOCK_SIZE_N=BLOCK_N,
#                 BLOCK_SIZE_K=BLOCK_K,
#                 num_warps=num_warps,
#                 num_stages=num_stages,
#             )
	
#     elif kernel_type == 'medium':
#         grid = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#         )
#         kernel = w8a8_gemm_medium_m_kernel[grid]
		
#         kernel(
#             a, w, c,
#             a_scale, w_scale,
#             M, N, K,
#             a_block_size, w_block_size_k, w_block_size_n,
#             a.stride(0), a.stride(1),
#             w.stride(0), w.stride(1),
#             c.stride(0), c.stride(1),
#             a_scale.stride(0), a_scale.stride(1),
#             w_scale.stride(0), w_scale.stride(1),
#             BLOCK_SIZE_M=BLOCK_M,
#             BLOCK_SIZE_N=BLOCK_N,
#             BLOCK_SIZE_K=BLOCK_K,
#             GROUP_SIZE_M=GROUP_M,
#             num_warps=num_warps,
#             num_stages=num_stages,
#         )
	
#     else:  # large
#         grid = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#         )
#         kernel = w8a8_gemm_large_m_kernel[grid]
		
#         kernel(
#             a, w, c,
#             a_scale, w_scale,
#             M, N, K,
#             a_block_size, w_block_size_k, w_block_size_n,
#             a.stride(0), a.stride(1),
#             w.stride(0), w.stride(1),
#             c.stride(0), c.stride(1),
#             a_scale.stride(0), a_scale.stride(1),
#             w_scale.stride(0), w_scale.stride(1),
#             BLOCK_SIZE_M=BLOCK_M,
#             BLOCK_SIZE_N=BLOCK_N,
#             BLOCK_SIZE_K=BLOCK_K,
#             GROUP_SIZE_M=GROUP_M,
#             num_warps=num_warps,
#             num_stages=num_stages,
#         )
	
#     return c


# # ==================== KERNEL WARMUP ====================
# def warmup_kernels(device='cuda'):
#     """
#     Pre-compile all kernel variants including Split-K to eliminate first-call JIT overhead.
#     Call this once during model initialization.
#     """
#     print("🔥 Warming up W8A8 GEMM kernels (with Split-K)...")
	
#     # Test representative shapes including all DeepSeek-like patterns
#     test_sizes = [
#         (8, 7168, 2048),    # Small M, large N, medium K
#         (8, 2048, 7168),    # Small M, medium N, large K (needs Split-K)
#         (8, 1536, 7168),    # Small M, medium N, large K (needs Split-K)
#         (8, 576, 7168),     # Small M, small N, large K (most aggressive Split-K)
#         (16, 4096, 11008),  # Small M
#         (32, 4096, 11008),  # Medium M boundary
#         (64, 4096, 11008),  # Medium M
#         (128, 4096, 11008), # Large M
#     ]
	
#     for M, N, K in test_sizes:
#         a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#         w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#         a_scale = torch.ones(M, (K + 127) // 128, device=device, dtype=torch.float32)
#         w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32)
		
#         # Warmup call (will automatically use Split-K when beneficial)
#         _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
	
#     torch.cuda.synchronize()
#     print("✅ All kernels (including Split-K variants) warmed up and ready!")


# # ==================== USAGE EXAMPLE ====================
# if __name__ == "__main__":
#     # Warmup (do this once at model load time)
#     warmup_kernels()
	
#     # Get device info
#     device = torch.device('cuda')
#     props = torch.cuda.get_device_properties(0)
#     num_sms = props.multi_processor_count
	
#     print("\n" + "="*80)
#     print(f"🖥️  Device: {props.name}")
#     print(f"   SM Count: {num_sms}")
#     print(f"   Target CTAs: {int(num_sms * 1.5)} (1.5x oversubscription)")
#     print("="*80)
	
#     # Test all DeepSeek-like shapes
#     test_shapes = [
#         (8, 7168, 2048),
#         (8, 2048, 7168),
#         (8, 1536, 7168),
#         (8, 24576, 1536),
#         (8, 576, 7168),
#         (8, 32768, 512),
#         (8, 7168, 16384),
#     ]
	
#     print("\n" + "="*80)
#     print("📊 Adaptive Config Selection for All Shapes")
#     print("="*80)
	
#     for M, N, K in test_shapes:
#         config = W8A8GemmConfig.select_config(M, N, K, device)
#         BLOCK_M, BLOCK_N, _, _, _, _, kernel_type, split_k = config
		
#         base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
#         total_ctas = base_ctas * split_k
#         sm_utilization = (total_ctas / num_sms) * 100
#         k_iterations = triton.cdiv(K, 128)
		
#         # Identify strategy
#         is_bad_shape = (N <= 1536) and (K > 4096)
		
#         if is_bad_shape:
#             strategy = f"🔧 BAD SHAPE: Large tile (BN={BLOCK_N})"
#         elif split_k > 1:
#             strategy = f"✅ DEFAULT: Split-K={split_k}×"
#         else:
#             strategy = f"✅ DEFAULT: No split needed"
		
#         # Show if we're limited
#         limitation = ""
#         if sm_utilization < 50:
#             limitation = f" ⚠️ {sm_utilization:.0f}% SM util"
		
#         print(f"\n📐 Shape: M={M:5d}, N={N:5d}, K={K:5d}")
#         print(f"   Strategy: {strategy}")
#         print(f"   Config: BLOCK_M={BLOCK_M:2d}, BLOCK_N={BLOCK_N:3d} | CTAs: {total_ctas:4d}{limitation}")
	
#     print("\n" + "="*80)
#     print("🔥 Detailed Benchmark on Challenging Shape")
#     print("="*80)
	
#     # Benchmark a challenging shape: small M, small N, LARGE K
#     M, N, K = 8, 576, 7168
	
#     print(f"\nTesting M={M}, N={N}, K={K}")
#     print(f"Challenge: Small N + Large K → Limited parallelism + Many K-iterations")
#     print(f"Note: Split-K is DISABLED for large K to avoid overhead")
#     print()
	
#     a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#     w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
	
#     a_scale = torch.ones(M, (K + 127) // 128, device='cuda', dtype=torch.float32)
#     w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device='cuda', dtype=torch.float32)
	
#     # Get config info
#     config = W8A8GemmConfig.select_config(M, N, K, device)
#     BLOCK_M, BLOCK_N, _, _, _, _, kernel_type, split_k = config
	
#     base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
#     total_ctas = base_ctas * split_k
	
#     # Calculate problem characteristics
#     total_flops = 2 * M * N * K
#     problem_size = M * N * K
#     padding_waste_m = ((BLOCK_M - (M % BLOCK_M)) % BLOCK_M) / BLOCK_M * 100
#     k_iterations = triton.cdiv(K, 128)
#     sm_coverage = (total_ctas / num_sms) * 100
	
#     is_bad_shape = (N <= 1536) and (K > 4096)
	
#     print(f"\n⚙️  Configuration:")
#     print(f"   Device: {props.name} ({num_sms} SMs)")
#     print(f"   Problem: {total_flops/1e6:.1f} MFLOPs, {problem_size/1e6:.1f}M elements")
	
#     if is_bad_shape:
#         print(f"   🔧 BAD SHAPE detected (N={N} ≤ 1536 AND K={K} > 4096)")
#         print(f"   Strategy: Large tile approach")
#         print(f"   - Using BLOCK_N={BLOCK_N} (large tiles for efficiency)")
#         print(f"   - Split-K=1 (disabled to avoid overhead)")
#         print(f"   - Rationale: Can't get parallelism, so maximize work/CTA")
#     else:
#         print(f"   ✅ GOOD SHAPE - Using default strategy")
#         print(f"   Strategy: Standard tiling with Split-K")
#         print(f"   - Using BLOCK_N={BLOCK_N} (balanced)")
#         if split_k > 1:
#             print(f"   - Split-K={split_k}× (for more parallelism)")
#         else:
#             print(f"   - Split-K=1 (sufficient parallelism)")
	
#     print(f"   Block size: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K=128")
#     print(f"   Padding waste (M): {padding_waste_m:.1f}% (M={M} in BLOCK_M={BLOCK_M})")
#     print(f"   K-dimension: {K} elements → {k_iterations} K-loop iterations")
#     print(f"   Base CTAs: {base_ctas}")
#     print(f"   Total CTAs: {total_ctas}")
#     print(f"   📊 SM Coverage: {sm_coverage:.1f}%")
	
#     # Warning for severe underutilization
#     if sm_coverage < 50:
#         print(f"   ⚠️  LOW SM utilization! Only {total_ctas}/{num_sms} SMs active")
#     print()
	
#     # Correctness check
#     result = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
#     print(f"✅ Output shape: {result.shape}, dtype: {result.dtype}")
	
#     # Benchmark
#     import time
#     torch.cuda.synchronize()
	
#     iters = 100
#     warmup_iters = 10
	
#     # Warmup
#     for _ in range(warmup_iters):
#         _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
#     torch.cuda.synchronize()
	
#     # Actual benchmark
#     start = time.time()
#     for _ in range(iters):
#         _ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
	
#     # Calculate TFLOPS
#     flops = 2 * M * N * K
#     tflops = (flops * iters) / (elapsed * 1e12)
	
#     # Estimate theoretical peak (assuming H100/H20 class GPU)
#     theoretical_peak_tflops = 1000  # Approximate for FP8 on modern GPUs
#     utilization = (tflops / theoretical_peak_tflops) * 100
	
#     print(f"\n⚡ Performance:")
#     print(f"   TFLOPS: {tflops:.2f}")
#     print(f"   GPU Utilization: {utilization:.2f}%")
#     print(f"   Average time: {elapsed/iters*1000:.3f} ms")
#     print(f"   Throughput: {iters/elapsed:.1f} iter/s")
	
#     # Compare with a better shape
#     compare_M, compare_N, compare_K = 8, 24576, 1536
#     compare_flops = 2 * compare_M * compare_N * compare_K
#     compare_config = W8A8GemmConfig.select_config(compare_M, compare_N, compare_K, device)
#     compare_BLOCK_M, compare_BLOCK_N, _, _, _, _, _, compare_split_k = compare_config
#     compare_ctas = triton.cdiv(compare_M, compare_BLOCK_M) * triton.cdiv(compare_N, compare_BLOCK_N)
	
#     print(f"\n📊 Why This Shape Is Challenging:")
#     print(f"   Current shape:  M={M:5d}, N={N:5d}, K={K:5d}")
#     print(f"   - Base parallelism: {base_ctas} CTAs (only {sm_coverage:.0f}% of {num_sms} SMs!)")
#     print(f"   - K-iterations: {k_iterations} (long serial work per CTA)")
#     print(f"   - BLOCK_N={BLOCK_N} (trying to maximize work per CTA)")
#     print(f"   - Result: {total_flops/1e6:.1f} MFLOPs spread across few CTAs")
#     print(f"\n   Better shape:   M={compare_M:5d}, N={compare_N:5d}, K={compare_K:5d}")
#     print(f"   - Base parallelism: {compare_ctas} CTAs ({(compare_ctas/num_sms)*100:.0f}% SM utilization)")
#     print(f"   - K-iterations: {triton.cdiv(compare_K, 128)} (short work per CTA)")
#     print(f"   - BLOCK_N={compare_BLOCK_N}")
#     print(f"   - Result: {compare_flops/1e6:.1f} MFLOPs (9× more work BUT 10× more parallelism)")
	
#     print(f"\n   🔑 Key Insight:")
#     print(f"   Shape quality = f(M×N parallelism, K serial work)")
#     print(f"   Bad:  Small M×N ({M}×{N}) + Large K ({K}) = Few CTAs doing long work")
#     print(f"   Good: Large M×N ({compare_M}×{compare_N}) + Small K ({compare_K}) = Many CTAs doing short work")
	
#     if sm_coverage < 50:
#         print(f"\n   💡 Recommendations:")
#         print(f"   1. **Batch multiple M=8 problems** → Increase M to 64, 128, etc.")
#         print(f"   2. **Transpose if semantically valid**: (M={M}, N={N}, K={K}) → (M={M}, N={K}, K={N})")
#         print(f"      This would give {triton.cdiv(K, 64)} N-blocks instead of {base_ctas}!")
#         print(f"   3. **Accept suboptimal perf for this shape** - it's fundamentally GPU-unfriendly")
#         print(f"   4. **Use cuBLAS which may have better small-shape heuristics**")
	
#     print("\n" + "="*80)



"""
🔥 W8A8 GEMM: High-Performance FP8 Matrix Multiplication with Integrated Benchmarking

Simplified Tiling Strategy:
- BAD SHAPES (small N + large K): (16, 32, 128) tiles, NO split-K
- GOOD SHAPES (everything else): (64, 32, 128) tiles, WITH split-K (4-8×)
"""

import torch
import triton
import triton.language as tl
from typing import List, Tuple, Dict, Optional
import math
import time
import numpy as np
from dataclasses import dataclass


# ==================== TRITON KERNELS ====================

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
	stride_cm, stride_cn, stride_ck,
	stride_a_scale_m, stride_a_scale_k,
	stride_w_scale_n, stride_w_scale_k,
	BLOCK_SIZE_M: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr,
	BLOCK_SIZE_K: tl.constexpr,
	SPLIT_K: tl.constexpr,
):
	"""Split-K variant: Each CTA computes partial result for a K-slice."""
	pid_m = tl.program_id(axis=0)
	pid_n = tl.program_id(axis=1)
	pid_k = tl.program_id(axis=2)
	
	k_per_split = tl.cdiv(K, SPLIT_K)
	k_start_global = pid_k * k_per_split
	k_end_global = min(k_start_global + k_per_split, K)
	
	offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	mask_m = offs_m < M
	
	offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	mask_n = offs_n < N
	
	acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
	num_k_blocks = tl.cdiv(k_end_global - k_start_global, BLOCK_SIZE_K)
	
	for k_idx in range(num_k_blocks):
		k_start = k_start_global + k_idx * BLOCK_SIZE_K
		offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
		mask_k = offs_k < k_end_global
		
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
	
	c_partial = acc.to(tl.bfloat16)
	offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	
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
	"""Fast reduction kernel to sum K-splits into final result."""
	pid_m = tl.program_id(axis=0)
	pid_n = tl.program_id(axis=1)
	
	offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	mask_m = offs_m < M
	mask_n = offs_n < N
	
	acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
	for k_split in range(SPLIT_K):
		c_ptrs = c_splits_ptr + k_split * stride_ck + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
		mask = mask_m[:, None] & mask_n[None, :]
		c_tile = tl.load(c_ptrs, mask=mask, other=0.0)
		acc += c_tile.to(tl.float32)
	
	c_final = acc.to(tl.bfloat16)
	out_ptrs = c_final_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
	mask = mask_m[:, None] & mask_n[None, :]
	tl.store(out_ptrs, c_final, mask=mask)


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
	"""Standard small M kernel without Split-K."""
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
	"""Optimized for medium M with better swizzling."""
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
	"""Large M: maximize throughput with larger tiles."""
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


# ==================== SIMPLIFIED CONFIG SELECTOR ====================

class W8A8GemmConfig:
	"""
	Simplified tiling strategy:
	
	For M ≤ 32 (small-M cases):
	1. BAD SHAPES (small N + large K, like N≤2048 and K>4096):
	   - Use (16, 32, 128) tiles
	   - NO split-K (overhead too high)
	   - Minimize padding waste with small M tile
	   
	2. GOOD SHAPES (everything else):
	   - Use (64, 32, 128) tiles
	   - WITH split-K (4-8×) for parallelism
	   - Larger M tile for efficiency
	"""
	
	_sm_count_cache = {}
	
	@staticmethod
	def get_sm_count(device) -> int:
		"""Get the number of SMs for the given device."""
		device_idx = device.index if device.index is not None else 0
		
		if device_idx not in W8A8GemmConfig._sm_count_cache:
			props = torch.cuda.get_device_properties(device_idx)
			W8A8GemmConfig._sm_count_cache[device_idx] = props.multi_processor_count
		
		return W8A8GemmConfig._sm_count_cache[device_idx]
	
	@staticmethod
	def select_config(M: int, N: int, K: int, device):
		"""
		Simplified config selection.
		Returns: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kernel_type, split_k)
		"""
		if M <= 32:
			# Small M: Choose between two strategies
			is_bad_shape = (N <= 2048) and (K > 4096)
			
			if is_bad_shape:
				# Strategy 1: Small tiles, no split-K
				# (16, 32, 128) - minimize padding waste for small M
				BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 128
				split_k = 1
				num_warps = 4
				num_stages = 3
			else:
				# Strategy 2: Larger tiles, with split-K
				# (64, 32, 128) - better efficiency, use split-K for parallelism
				BLOCK_M, BLOCK_N, BLOCK_K = 64, 32, 128
				
				# Compute split-K factor
				num_sms = W8A8GemmConfig.get_sm_count(device)
				target_ctas = int(num_sms * 1.5)
				base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
				
				if base_ctas >= target_ctas:
					split_k = 1
				else:
					# Need split-K for more parallelism
					split_k_needed = triton.cdiv(target_ctas, base_ctas)
					
					# Limit split based on K size
					if K <= 2048:
						max_split_k = 8
					elif K <= 4096:
						max_split_k = 4
					else:
						max_split_k = 2
					
					split_k = min(split_k_needed, max_split_k)
					split_k = 2 ** int(math.log2(max(1, split_k)))
				
				num_warps = 4
				num_stages = 3
			
			return (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, 4, 'small', split_k)
		
		elif M <= 64:
			return (64, 128, 128, 4, 4, 8, 'medium', 1)
		elif M < 128:
			return (128, 128, 128, 4, 4, 8, 'medium', 1)
		elif M < 512 or N < 2048:
			return (128, 128, 128, 8, 4, 8, 'large', 1)
		else:
			return (128, 256, 128, 8, 5, 8, 'large', 1)


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
	🔥 W8A8 GEMM with Simplified Tiling Strategy 🔥
	
	Two-tier approach:
	- Bad shapes (small N + large K): (16, 32, 128) tiles, no split-K
	- Good shapes: (64, 32, 128) tiles, with split-K (4-8×)
	"""
	M, K = a.shape
	N = w.shape[0]
	
	assert a.dtype == torch.float8_e4m3fn or a.dtype == torch.float8_e5m2, "A must be FP8"
	assert w.dtype == torch.float8_e4m3fn or w.dtype == torch.float8_e5m2, "W must be FP8"
	assert a.is_contiguous() and w.is_contiguous(), "Tensors must be contiguous"
	
	BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kernel_type, split_k = \
		W8A8GemmConfig.select_config(M, N, K, a.device)
	
	c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
	
	if kernel_type == 'small':
		if split_k > 1:
			c_splits = torch.empty((split_k, M, N), device=a.device, dtype=torch.bfloat16)
			
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


# ==================== BENCHMARKING INFRASTRUCTURE ====================

@dataclass
class BenchmarkResult:
	"""Container for benchmark results."""
	shape: Tuple[int, int, int]
	time_ms: float
	tflops: float
	bandwidth_gb_s: float
	kernel_type: str
	config: Dict


class W8A8Benchmarker:
	"""Comprehensive benchmarking suite."""
	
	def __init__(self, device='cuda', warmup_iters=10, bench_iters=100):
		self.device = device
		self.warmup_iters = warmup_iters
		self.bench_iters = bench_iters
		self.results = []
	
	def benchmark_kernel(
		self,
		M: int,
		N: int,
		K: int,
		a_block_size: int = 128,
		w_block_size: int = 128,
	) -> BenchmarkResult:
		"""Benchmark a single configuration."""
		a = torch.randn(M, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
		w = torch.randn(N, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
		
		a_scale = torch.ones(M, (K + 127) // 128, device=self.device, dtype=torch.float32)
		w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device, dtype=torch.float32)
		
		config_tuple = W8A8GemmConfig.select_config(M, N, K, torch.device(self.device))
		BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, kernel_type, split_k = config_tuple
		
		# Warmup
		for _ in range(self.warmup_iters):
			_ = w8a8_gemm_dispatch(a, a_scale, w, w_scale, a_block_size, w_block_size, w_block_size)
		
		torch.cuda.synchronize()
		
		# Benchmark
		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		
		start_event.record()
		for _ in range(self.bench_iters):
			_ = w8a8_gemm_dispatch(a, a_scale, w, w_scale, a_block_size, w_block_size, w_block_size)
		end_event.record()
		
		torch.cuda.synchronize()
		
		total_time_ms = start_event.elapsed_time(end_event)
		avg_time_ms = total_time_ms / self.bench_iters
		
		flops = 2 * M * N * K
		tflops = (flops / avg_time_ms / 1e9)
		
		bytes_read = M * K + N * K + M * ((K + 127) // 128) * 4 + ((N + 127) // 128) * ((K + 127) // 128) * 4
		bytes_write = M * N * 2
		total_bytes = bytes_read + bytes_write
		bandwidth_gb_s = (total_bytes / avg_time_ms / 1e6)
		
		base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
		total_ctas = base_ctas * split_k
		k_iterations = triton.cdiv(K, BLOCK_K)
		k_iters_per_split = k_iterations // split_k if split_k > 1 else k_iterations
		
		# Determine strategy
		is_bad_shape = (N <= 2048) and (K > 4096) and (M <= 32)
		strategy = "BAD (16×32)" if is_bad_shape else "GOOD (64×32)"
		
		result = BenchmarkResult(
			shape=(M, N, K),
			time_ms=avg_time_ms,
			tflops=tflops,
			bandwidth_gb_s=bandwidth_gb_s,
			kernel_type=kernel_type,
			config={
				'BLOCK_M': BLOCK_M,
				'BLOCK_N': BLOCK_N,
				'BLOCK_K': BLOCK_K,
				'split_k': split_k,
				'num_warps': num_warps,
				'num_stages': num_stages,
				'base_ctas': base_ctas,
				'total_ctas': total_ctas,
				'k_iterations': k_iterations,
				'k_iters_per_split': k_iters_per_split,
				'strategy': strategy,
			},
		)
		
		self.results.append(result)
		return result
	
	def sweep_shapes(self, test_shapes: List[Tuple[int, int, int]]) -> List[BenchmarkResult]:
		"""Sweep through multiple problem sizes."""
		results = []
		
		print("\n" + "=" * 150)
		print("🔥 PERFORMANCE SWEEP - SIMPLIFIED TILING STRATEGY")
		print("=" * 150)
		print(f"{'M':>5} {'N':>6} {'K':>6} │ {'Strategy':>13} │ {'BM':>3} {'BN':>3} {'BK':>3} │ {'SK':>2} │ {'K-iters':>7} │ {'CTAs':>5} │ {'Time(ms)':>9} {'TFLOPS':>7} {'BW(GB/s)':>9}")
		print("-" * 150)
		
		for M, N, K in test_shapes:
			result = self.benchmark_kernel(M, N, K)
			results.append(result)
			
			cfg = result.config
			
			if cfg['split_k'] > 1:
				k_iter_str = f"{cfg['k_iters_per_split']}×{cfg['split_k']}"
			else:
				k_iter_str = f"{cfg['k_iterations']}"
			
			print(f"{M:>5} {N:>6} {K:>6} │ {cfg['strategy']:>13} │ "
				  f"{cfg['BLOCK_M']:>3} {cfg['BLOCK_N']:>3} {cfg['BLOCK_K']:>3} │ "
				  f"{cfg['split_k']:>2} │ "
				  f"{k_iter_str:>7} │ "
				  f"{cfg['total_ctas']:>5} │ "
				  f"{result.time_ms:>9.4f} {result.tflops:>7.2f} {result.bandwidth_gb_s:>9.2f}")
		
		print("=" * 150)
		print("Strategy: BAD (16×32) = Small tiles, no split-K | GOOD (64×32) = Large tiles, with split-K")
		print("=" * 150)
		
		return results
	
	def compare_with_torch(self, M: int, N: int, K: int):
		"""Compare with PyTorch baseline."""
		print(f"\n🔥 COMPARISON: Custom vs PyTorch (M={M}, N={N}, K={K})")
		print("=" * 80)
		
		a_fp16 = torch.randn(M, K, device=self.device, dtype=torch.float16)
		w_fp16 = torch.randn(N, K, device=self.device, dtype=torch.float16)
		
		a_fp8 = a_fp16.to(torch.float8_e4m3fn)
		w_fp8 = w_fp16.to(torch.float8_e4m3fn)
		
		a_scale = torch.ones(M, (K + 127) // 128, device=self.device, dtype=torch.float32)
		w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device, dtype=torch.float32)
		
		torch.cuda.synchronize()
		torch_times = []
		for _ in range(self.bench_iters):
			start = time.perf_counter()
			_ = torch.matmul(a_fp16, w_fp16.t())
			torch.cuda.synchronize()
			torch_times.append((time.perf_counter() - start) * 1000)
		torch_time_ms = np.median(torch_times)
		
		result = self.benchmark_kernel(M, N, K)
		speedup = torch_time_ms / result.time_ms
		
		print(f"PyTorch FP16:     {torch_time_ms:.4f} ms")
		print(f"Custom FP8:       {result.time_ms:.4f} ms")
		print(f"Speedup:          {speedup:.2f}x")
		print(f"TFLOPS:           {result.tflops:.2f}")
		print(f"Configuration:    {result.config['strategy']}, "
			  f"Tiles={result.config['BLOCK_M']}×{result.config['BLOCK_N']}×{result.config['BLOCK_K']}, "
			  f"Split-K={result.config['split_k']}")
		print("=" * 80)
		
		return speedup


class W8A8Validator:
	"""Numerical validation suite."""
	
	@staticmethod
	def validate_correctness(
		M: int = 16,
		N: int = 128,
		K: int = 256,
		rtol: float = 1e-1,
		atol: float = 1e-2,
	) -> bool:
		"""Validate kernel correctness."""
		print(f"\n✅ CORRECTNESS VALIDATION (M={M}, N={N}, K={K})")
		print("=" * 80)
		
		device = 'cuda'
		
		torch.manual_seed(42)
		a_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
		w_fp16 = torch.randn(N, K, device=device, dtype=torch.float16)
		
		a_fp8 = a_fp16.to(torch.float8_e4m3fn)
		w_fp8 = w_fp16.to(torch.float8_e4m3fn)
		
		a_scale = torch.ones(M, (K + 127) // 128, device=device, dtype=torch.float32) * 0.5
		w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32) * 0.5
		
		a_dequant = a_fp8.to(torch.float16)
		w_dequant = w_fp8.to(torch.float16)
		
		reference = torch.matmul(a_dequant, w_dequant.t()) * 0.25
		custom = w8a8_gemm_dispatch(a_fp8, a_scale, w_fp8, w_scale, 128, 128, 128)
		
		max_diff = torch.max(torch.abs(reference.to(torch.bfloat16) - custom)).item()
		mean_diff = torch.mean(torch.abs(reference.to(torch.bfloat16) - custom)).item()
		rel_error = mean_diff / (torch.mean(torch.abs(reference)).item() + 1e-8)
		
		print(f"Max absolute error:   {max_diff:.6f}")
		print(f"Mean absolute error:  {mean_diff:.6f}")
		print(f"Relative error:       {rel_error:.6f}")
		
		passed = rel_error < rtol and max_diff < atol
		
		if passed:
			print("✅ PASSED: Kernel is numerically correct!")
		else:
			print("❌ FAILED: Numerical errors exceed tolerance!")
		
		print("=" * 80)
		return passed


def warmup_kernels(device='cuda'):
	"""Pre-compile all kernel variants."""
	print("🔥 Warming up W8A8 GEMM kernels...")
	
	test_sizes = [
		(8, 7168, 2048),
		(8, 2048, 7168),
		(8, 1536, 7168),
		(8, 576, 7168),
		(16, 4096, 11008),
		(32, 4096, 11008),
		(64, 4096, 11008),
		(128, 4096, 11008),
	]
	
	for M, N, K in test_sizes:
		a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
		w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
		a_scale = torch.ones(M, (K + 127) // 128, device=device, dtype=torch.float32)
		w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32)
		
		_ = w8a8_gemm_dispatch(a, a_scale, w, w_scale)
	
	torch.cuda.synchronize()
	print("✅ All kernels warmed up and ready!")


def run_full_benchmark_suite():
	"""Run comprehensive benchmark suite."""
	print("\n" + "=" * 80)
	print("🚀 W8A8 GEMM COMPREHENSIVE BENCHMARK SUITE")
	print("=" * 80)
	
	device = torch.device('cuda')
	props = torch.cuda.get_device_properties(0)
	num_sms = props.multi_processor_count
	
	print(f"\n🖥️  Device: {props.name}")
	print(f"   SM Count: {num_sms}")
	
	print(f"\n📋 TILING STRATEGY:")
	print(f"   BAD SHAPES (N≤2048 AND K>4096):  16×32×128 tiles, NO split-K")
	print(f"   GOOD SHAPES (everything else):    64×32×128 tiles, WITH split-K (4-8×)")
	
	warmup_kernels()
	
	validator = W8A8Validator()
	validator.validate_correctness()
	
	benchmarker = W8A8Benchmarker(warmup_iters=10, bench_iters=100)
	
	# Test shapes covering both strategies
	llm_shapes = [
		# Token generation (M=1)
		(1, 7168, 2048),
		(1, 2048, 7168),
		(1, 1536, 7168),
		(1, 24576, 1536),
		(1, 576, 7168),
		(1, 32768, 512),
		(1, 7168, 16384),

		# (1, 11008, 4096),
		
		# Small batch inference (M=8-16)
		(8, 7168, 2048),
		(8, 2048, 7168),
		(8, 1536, 7168),
		(8, 24576, 1536),
		(8, 576, 7168),
		(8, 32768, 512),
		(8, 7168, 16384),
		
		(16, 7168, 2048),
		(16, 2048, 7168),
		(16, 1536, 7168),
		(16, 24576, 1536),
		(16, 576, 7168),
		(16, 32768, 512),
		(16, 7168, 16384),
		
		
		# Medium batch (M=32)
		(32, 7168, 2048),
		(32, 2048, 7168),
		(32, 1536, 7168),
		(32, 24576, 1536),
		(32, 576, 7168),
		(32, 32768, 512),
		(32, 7168, 16384),
		
		# Larger batches
		(64, 7168, 2048),
		(64, 2048, 7168),
		(64, 1536, 7168),
		(64, 24576, 1536),
		(64, 576, 7168),
		(64, 32768, 512),
		(64, 7168, 16384),

		(128, 7168, 2048),
		(128, 2048, 7168),
		(128, 1536, 7168),
		(128, 24576, 1536),
		(128, 576, 7168),
		(128, 32768, 512),
		(128, 7168, 16384),
	]
	
	
	results = benchmarker.sweep_shapes(llm_shapes)
	
	print("\n" + "=" * 80)
	print("🔥 PYTORCH COMPARISON")
	print("=" * 80)
	for M in [8, 16, 32]:
		benchmarker.compare_with_torch(M, 4096, 11008)
	
	print("\n" + "=" * 80)
	print("📊 PERFORMANCE SUMMARY")
	print("=" * 80)
	
	best_tflops = max(results, key=lambda r: r.tflops)
	print(f"Peak TFLOPS:        {best_tflops.tflops:.2f} at shape {best_tflops.shape}")
	print(f"                    Strategy: {best_tflops.config['strategy']}, "
		  f"Split-K={best_tflops.config['split_k']}")
	
	best_bw = max(results, key=lambda r: r.bandwidth_gb_s)
	print(f"Peak Bandwidth:     {best_bw.bandwidth_gb_s:.2f} GB/s at shape {best_bw.shape}")
	
	small_m_results = [r for r in results if r.shape[0] <= 16]
	if small_m_results:
		best_small_m = max(small_m_results, key=lambda r: r.tflops)
		print(f"Best Small-M:       {best_small_m.tflops:.2f} TFLOPS at shape {best_small_m.shape}")
		print(f"                    Strategy: {best_small_m.config['strategy']}, "
			  f"Split-K={best_small_m.config['split_k']}")
	
	# Analyze by strategy
	bad_shape_results = [r for r in results if r.config['strategy'] == "BAD (16×32)"]
	good_shape_results = [r for r in results if r.config['strategy'] == "GOOD (64×32)"]
	
	if bad_shape_results:
		avg_bad = np.mean([r.tflops for r in bad_shape_results])
		print(f"\nBAD shapes avg:     {avg_bad:.2f} TFLOPS (16×32 tiles, no split-K)")
	
	if good_shape_results:
		avg_good = np.mean([r.tflops for r in good_shape_results])
		print(f"GOOD shapes avg:    {avg_good:.2f} TFLOPS (64×32 tiles, with split-K)")
	
	print("=" * 80)


if __name__ == "__main__":
	run_full_benchmark_suite()