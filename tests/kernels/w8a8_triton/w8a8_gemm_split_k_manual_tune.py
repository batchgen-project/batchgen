"""
🔥 W8A8 GEMM: High-Performance FP8 Matrix Multiplication with Integrated Benchmarking

Features:
- Adaptive block sizing and device-aware split-K
- Comprehensive benchmarking and validation suite
- Detailed performance analysis with configuration info
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
	"""Split-K variant for small M: Each CTA computes partial result for a K-slice."""
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


# ==================== CONFIG SELECTOR ====================

class W8A8GemmConfig:
	"""Pre-optimized configurations with device-aware Split-K."""
	
	SMALL_M_BASE_CONFIG = (128, 4, 3)  # (BLOCK_K, num_warps, num_stages)
	
	MEDIUM_M_CONFIGS = [
		(64, 128, 128, 4, 4),
		(128, 128, 128, 4, 4),
	]
	
	LARGE_M_CONFIGS = [
		(128, 128, 128, 8, 4),
		(128, 256, 128, 8, 5),
	]
	
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
	def compute_optimal_config(M: int, N: int, K: int, device, oversubscribe_factor: float = 1.5):
		"""Compute optimal BLOCK_N and Split-K factor together."""
		num_sms = W8A8GemmConfig.get_sm_count(device)
		target_ctas = int(num_sms * oversubscribe_factor)
		
		if M <= 16:
			BLOCK_M = 16
		elif M <= 32:
			BLOCK_M = 32
		else:
			BLOCK_M = 64
		
		needs_split_k = (M <= 32) or (N <= 2048)
		
		if not needs_split_k:
			return BLOCK_M, 128, 1
		
		if K > 4096:
			base_ctas_16 = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, 16)
			base_ctas_64 = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, 64)
			base_ctas_128 = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, 128)
			base_ctas_256 = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, 256)
			
			if base_ctas_16 < num_sms // 2:
				if base_ctas_256 >= 4:
					return BLOCK_M, 256, 1
				elif base_ctas_128 >= 8:
					return BLOCK_M, 128, 1
				elif base_ctas_64 >= 16:
					return BLOCK_M, 64, 1
				else:
					return BLOCK_M, 32, 1
			else:
				if base_ctas_128 >= num_sms // 4:
					return BLOCK_M, 128, 1
				elif base_ctas_64 >= num_sms // 3:
					return BLOCK_M, 64, 1
				else:
					return BLOCK_M, 32, 1
		
		candidate_block_ns = [64, 32, 16]
		
		for BLOCK_N in candidate_block_ns:
			base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
			
			if base_ctas >= target_ctas:
				return BLOCK_M, BLOCK_N, 1
			
			split_k_needed = triton.cdiv(target_ctas, base_ctas)
			
			min_k_per_split = 1024
			if N <= 576:
				min_k_per_split = 512
			
			max_split_k = max(1, K // min_k_per_split)
			split_k = min(split_k_needed, max_split_k)
			split_k = 2 ** int(math.log2(max(1, split_k)))
			
			if K <= 2048:
				split_k = min(split_k, 4)
			elif K <= 4096:
				split_k = min(split_k, 8)
			else:
				split_k = 1
			
			total_ctas = base_ctas * split_k
			if total_ctas >= target_ctas or split_k >= max_split_k:
				return BLOCK_M, BLOCK_N, split_k
		
		BLOCK_N = 16
		base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
		split_k_needed = triton.cdiv(target_ctas, base_ctas)
		
		min_k_per_split = 512 if N <= 576 else 1024
		max_split_k = max(1, K // min_k_per_split)
		
		split_k = min(split_k_needed, max_split_k)
		split_k = 2 ** int(math.log2(max(1, split_k)))
		
		if K <= 2048:
			split_k = min(split_k, 4)
		elif K <= 4096:
			split_k = min(split_k, 8)
		else:
			split_k = 1
		
		return BLOCK_M, BLOCK_N, split_k
	
	@staticmethod
	def select_config(M: int, N: int, K: int, device):
		"""Select optimal configuration."""
		if M <= 32:
			BLOCK_M, BLOCK_N, split_k = W8A8GemmConfig.compute_optimal_config(M, N, K, device)
			split_k = 2
			return (BLOCK_M, BLOCK_N, 128, 4, 3, 4, 'small', split_k)
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
	"""Device-aware W8A8 GEMM with automatic Split-K."""
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
		# Create test data
		a = torch.randn(M, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
		w = torch.randn(N, K, device=self.device, dtype=torch.float16).to(torch.float8_e4m3fn)
		
		a_scale = torch.ones(M, (K + 127) // 128, device=self.device, dtype=torch.float32)
		w_scale = torch.ones((N + 127) // 128, (K + 127) // 128, device=self.device, dtype=torch.float32)
		
		# Get config info
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
		
		# Calculate metrics
		total_time_ms = start_event.elapsed_time(end_event)
		avg_time_ms = total_time_ms / self.bench_iters
		
		flops = 2 * M * N * K
		tflops = (flops / avg_time_ms / 1e9)
		
		bytes_read = M * K + N * K + M * ((K + 127) // 128) * 4 + ((N + 127) // 128) * ((K + 127) // 128) * 4
		bytes_write = M * N * 2
		total_bytes = bytes_read + bytes_write
		bandwidth_gb_s = (total_bytes / avg_time_ms / 1e6)
		
		# Calculate CTAs
		base_ctas = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
		total_ctas = base_ctas * split_k
		
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
			},
		)
		
		self.results.append(result)
		return result
	
	def sweep_shapes(self, test_shapes: List[Tuple[int, int, int]]) -> List[BenchmarkResult]:
		"""Sweep through multiple problem sizes with detailed config output."""
		results = []
		
		print("\n" + "=" * 120)
		print("🔥 PERFORMANCE SWEEP WITH DETAILED CONFIGURATION")
		print("=" * 120)
		print(f"{'M':>5} {'N':>6} {'K':>6} │ {'BM':>3} {'BN':>3} {'BK':>3} │ {'SK':>2} │ {'CTAs':>5} │ {'Type':>6} │ {'Time(ms)':>9} {'TFLOPS':>7} {'BW(GB/s)':>9}")
		print("-" * 120)
		
		for M, N, K in test_shapes:
			result = self.benchmark_kernel(M, N, K)
			results.append(result)
			
			cfg = result.config
			
			# Format output with all details
			print(f"{M:>5} {N:>6} {K:>6} │ "
				  f"{cfg['BLOCK_M']:>3} {cfg['BLOCK_N']:>3} {cfg['BLOCK_K']:>3} │ "
				  f"{cfg['split_k']:>2} │ "
				  f"{cfg['total_ctas']:>5} │ "
				  f"{result.kernel_type:>6} │ "
				  f"{result.time_ms:>9.4f} {result.tflops:>7.2f} {result.bandwidth_gb_s:>9.2f}")
		
		print("=" * 120)
		print("Legend: BM/BN/BK = Block sizes (M/N/K), SK = Split-K factor, CTAs = Total thread blocks")
		print("=" * 120)
		
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
		
		# PyTorch benchmark
		torch.cuda.synchronize()
		torch_times = []
		for _ in range(self.bench_iters):
			start = time.perf_counter()
			_ = torch.matmul(a_fp16, w_fp16.t())
			torch.cuda.synchronize()
			torch_times.append((time.perf_counter() - start) * 1000)
		torch_time_ms = np.median(torch_times)
		
		# Custom kernel benchmark
		result = self.benchmark_kernel(M, N, K)
		
		speedup = torch_time_ms / result.time_ms
		
		print(f"PyTorch FP16:     {torch_time_ms:.4f} ms")
		print(f"Custom FP8:       {result.time_ms:.4f} ms")
		print(f"Speedup:          {speedup:.2f}x")
		print(f"TFLOPS:           {result.tflops:.2f}")
		print(f"Configuration:    BM={result.config['BLOCK_M']}, BN={result.config['BLOCK_N']}, "
			  f"BK={result.config['BLOCK_K']}, Split-K={result.config['split_k']}")
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
	
	# Device info
	device = torch.device('cuda')
	props = torch.cuda.get_device_properties(0)
	num_sms = props.multi_processor_count
	
	print(f"\n🖥️  Device: {props.name}")
	print(f"   SM Count: {num_sms}")
	print(f"   Target CTAs: {int(num_sms * 1.5)} (1.5x oversubscription)")
	
	# Warmup
	warmup_kernels()
	
	# Validation
	validator = W8A8Validator()
	validator.validate_correctness()
	
	# Benchmarking
	benchmarker = W8A8Benchmarker(warmup_iters=10, bench_iters=100)
	
	# Test shapes (DeepSeek-like patterns)
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
	
	# PyTorch comparison
	print("\n" + "=" * 80)
	print("🔥 PYTORCH COMPARISON")
	print("=" * 80)
	for M in [1, 8, 16, 32]:
		benchmarker.compare_with_torch(M, 4096, 11008)
	
	# Summary
	print("\n" + "=" * 80)
	print("📊 PERFORMANCE SUMMARY")
	print("=" * 80)
	
	best_tflops = max(results, key=lambda r: r.tflops)
	print(f"Peak TFLOPS:        {best_tflops.tflops:.2f} at shape {best_tflops.shape}")
	print(f"                    Config: BM={best_tflops.config['BLOCK_M']}, "
		  f"BN={best_tflops.config['BLOCK_N']}, BK={best_tflops.config['BLOCK_K']}, "
		  f"Split-K={best_tflops.config['split_k']}")
	
	best_bw = max(results, key=lambda r: r.bandwidth_gb_s)
	print(f"Peak Bandwidth:     {best_bw.bandwidth_gb_s:.2f} GB/s at shape {best_bw.shape}")
	
	small_m_results = [r for r in results if r.shape[0] <= 16]
	if small_m_results:
		best_small_m = max(small_m_results, key=lambda r: r.tflops)
		print(f"Best Small-M:       {best_small_m.tflops:.2f} TFLOPS at shape {best_small_m.shape}")
		print(f"                    Config: BM={best_small_m.config['BLOCK_M']}, "
			  f"BN={best_small_m.config['BLOCK_N']}, BK={best_small_m.config['BLOCK_K']}, "
			  f"Split-K={best_small_m.config['split_k']}")
	
	print("=" * 80)


if __name__ == "__main__":
	run_full_benchmark_suite()