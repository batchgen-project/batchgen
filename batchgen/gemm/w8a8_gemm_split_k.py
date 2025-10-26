import torch
import triton
import triton.language as tl
import math

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
	
	assert a.dtype == torch.float8_e4m3fn, "A must be FP8"
	assert w.dtype == torch.float8_e4m3fn, "W must be FP8"
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