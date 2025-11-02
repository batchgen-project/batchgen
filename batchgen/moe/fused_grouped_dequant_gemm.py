import torch
import triton
import triton.language as tl
import os
# os.environ["TRITON_CACHE_SIZE"] = "2048"
import torch.distributed as dist
import logging

@triton.jit
def fused_dequant_grouped_gemm_bf16_fp8_kernel(
	lhs_ptr, rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	output_ptr,
	N, K, num_groups,
	stride_lhs_m, stride_lhs_k,
	stride_rhs_n, stride_rhs_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices, stride_rhs_ptrs, stride_rhs_scale_ptrs,
	
	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	"""
		Fused dequantization and grouped GEMM kernel for bf16 lhs and fp8 rhs matrices.
		Args:
			- lhs_ptr: Pointer to the lhs matrix (M, K) in bf16 dtype.
			- rhs_ptrs_ptr: Pointer to a tensor of pointers to rhs matrices (N, K) in fp8 dtype.
			- rhs_scale_ptrs_ptr: Pointer to a tensor of pointers to scale factors for rhs. 
				Each scale tensor is of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
			- group_sizes_ptr: Pointer to a tensor of group sizes. Shape is (num_groups,).
			- group_start_indices_ptr: Pointer to a tensor. Each element is the starting index of the group in lhs.
			- output_ptr: Pointer to the output tensor (M, N) in bf16

		Notes:
			- One-D CTA launch. Each program process associated tiles for each group.
			- For each group, M is different while K and N are the same across all groups.
			- Require lhs to be sorted by group ID.
	"""
	pid = tl.program_id(axis=0)
	num_programs = tl.num_programs(axis=0)
	lhs_dtype = tl.bfloat16
	rhs_dtype = tl.float8e4nv

	for g in range(num_groups):
		# Get group size: gm
		gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
		group_idx = tl.load(group_idx_ptr + g * stride_group_idx)

		# Get row indices for the current group.
		start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
		base_ptr_g = lhs_ptr + start_idx * stride_lhs_m

		rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(rhs_dtype))
		rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))
		
		# Process tiles for current pid.
		# Global problem size: [gm, k] @ [n, k].T -> [gm, n]
		num_tiles_m = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
		num_tiles_n = tl.cdiv(N, GEMM_BLOCK_SIZE_N)
		num_tiles = num_tiles_m * num_tiles_n
		tile_id = pid
		while tile_id < num_tiles:
			tile_m = tile_id // num_tiles_n
			tile_n = tile_id % num_tiles_n
			
			# Calculate offsets for lhs and rhs
			offs_lhs_m = tile_m * GEMM_BLOCK_SIZE_M + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_rhs_n = tile_n * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			offs_k = tl.arange(0, GEMM_BLOCK_SIZE_K)
			
			# Create pointers for lhs and rhs
			lhs_ptrs = base_ptr_g + (offs_lhs_m[:, None] * stride_lhs_m + offs_k[None, :] * stride_lhs_k)
			rhs_ptrs = rhs_base_ptr + (offs_rhs_n[:, None] * stride_rhs_n + offs_k[None, :] * stride_rhs_k)
			# Create pointers for scale
			n_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)

			# Initialize accumulator
			acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
			# Iterate over K blocks
			for k in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
				k_idx = k * GEMM_BLOCK_SIZE_K

				# Create masks for bounds checking
				lhs_mask = (offs_lhs_m[:, None] < gm) & (k_idx + offs_k[None, :] < K)
				rhs_mask = (offs_rhs_n[:, None] < N) & (k_idx + offs_k[None, :] < K)
				
				# Load lhs and rhs tiles
				lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0,cache_modifier='.cg')
				rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0,cache_modifier='.cg')
				
				# Dequantize rhs from fp8 to bf16
				rhs_fp32 = tl.cast(rhs_fp8, tl.float32)

				# Load scale for this tile
				scale_row = tile_n * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
				scale_col = k_idx // SCALE_BLOCK_SIZE_K
				scale_idx = scale_row * n_scale_k + scale_col
				scale = tl.load(rhs_scale_base_ptr + scale_idx)
				rhs_scaled = rhs_fp32 * scale
				rhs_bf16 = tl.cast(rhs_scaled, lhs_dtype)
				
				# Matrix multiplication with transposed rhs
				acc += tl.dot(lhs, rhs_bf16.T)

				lhs_ptrs += GEMM_BLOCK_SIZE_K * stride_lhs_k
				rhs_ptrs += GEMM_BLOCK_SIZE_K * stride_rhs_k


			# Store the result
			offs_output_m = start_idx + tile_m * GEMM_BLOCK_SIZE_M + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_output_n = tile_n * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
			output_mask = (offs_output_m[:, None] < start_idx + gm) & (offs_output_n[None, :] < N)
			# Convert to bf16 before storing
			output = tl.cast(acc, lhs_dtype)
			tl.store(output_ptrs, output, mask=output_mask)

			tile_id += num_programs  # Move to the next tile for this group

@torch.inference_mode()
def fused_dequant_grouped_gemm_bf16_fp8_triton(
		lhs: torch.Tensor,
		rhs_list: list[torch.Tensor],
		rhs_scale_list: list[torch.Tensor],
		group_sizes: tuple[int, int],
		gemm_block_size=(64, 64, 128), 
		scale_block_size=(128, 128)
):
	"""
		Performs a fused dequantization and grouped_gemm. We dequantize the fp8 rhs to bf16 on the fly.
		Args:
			lhs: torch.Tensor of shape (M, K) in bf16 dtype.
			rhs_list: List of torch.Tensor, each of shape (N, K) in fp8 dtype.
			rhs_scale_list: List of torch.Tensor, each of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
			group_sizes: Tuple of (group ID, group size) for each group.
			gemm_block_size: Tuple of (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
			scale_block_size: Tuple of (SCALE_BLOCK_M, SCALE_BLOCK_K)
		
		Returns:
			C: torch.Tensor of shape (M, N) in bf16 dtype.

		Notes:
			- 
	"""
	assert lhs.dtype == torch.bfloat16, "lhs must be of dtype bfloat16"
	assert all(r.dtype == torch.float8_e4m3fn for r in rhs_list), "All rhs matrices must be of dtype float8_e4m3fn"
	assert all(s.dtype == torch.float32 for s in rhs_scale_list), "All scale tensors must be of dtype float32"
	assert len(rhs_list) == len(rhs_scale_list), "rhs_list and rhs_scale_list must have the same length"

	device = lhs.device
	N = rhs_list[0].shape[0]
	K = lhs.shape[1]
	rhs_ptrs_ptr = torch.tensor([r.data_ptr() for r in rhs_list], dtype=torch.int64, device=device)
	rhs_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in rhs_scale_list], dtype=torch.int64, device=device)
	group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
	activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
	num_groups = len(group_sizes)
	group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
	group_start_indices[0] = 0  # The first group starts at index 0

	output = torch.empty((lhs.shape[0], N), dtype=torch.bfloat16, device=device)
	num_sms = torch.cuda.get_device_properties(device).multi_processor_count
	# grid = (num_sms,)
	# grid = lambda META: (
	# 	triton.cdiv(16, META['GEMM_BLOCK_SIZE_M']) * triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']),
	# )
	grid = lambda META: (
		triton.cdiv(16, META['GEMM_BLOCK_SIZE_M']) * triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']),
	)

	# logging.info(f"Rank {dist.get_rank()} launching fused_dequant_grouped_gemm_bf16_fp8_kernel with input shapes: lhs: {lhs.shape}, rhs_list: {[r.shape for r in rhs_list]}, rhs_scale_list: {[s.shape for s in rhs_scale_list]}, group_sizes: {group_size.tolist()}, group_start_indices: {group_start_indices.tolist()}, selected groups: {activated_group_idx.tolist()}")
	# # Launch the kernel
	try:
		fused_dequant_grouped_gemm_bf16_fp8_kernel[grid](
			lhs, rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
			activated_group_idx, group_size, group_start_indices,
			output,
			N, K, num_groups,
			lhs.stride(0), lhs.stride(1),
			rhs_list[0].stride(0), rhs_list[0].stride(1),
			output.stride(0), output.stride(1),
			activated_group_idx.stride(0), 
			group_size.stride(0), group_start_indices.stride(0),
			rhs_ptrs_ptr.stride(0), rhs_scale_ptrs_ptr.stride(0),
			GEMM_BLOCK_SIZE_M=gemm_block_size[0], GEMM_BLOCK_SIZE_N=gemm_block_size[1], GEMM_BLOCK_SIZE_K=gemm_block_size[2],
			SCALE_BLOCK_SIZE_N=scale_block_size[0], SCALE_BLOCK_SIZE_K=scale_block_size[1],
			num_warps=8,
			# num_stages=5
		)
	except Exception as e:
		print(f"Error launching fused_dequant_grouped_gemm_bf16_fp8_kernel: {e}")
		raise
	return output


@triton.jit
def fused_dequant_grouped_gemm_bf16_fp8_kernel_v2(
	lhs_ptr, rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	output_ptr,
	M, N:tl.constexpr, K:tl.constexpr, num_groups,
	stride_lhs_m, stride_lhs_k,
	stride_rhs_n, stride_rhs_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices, stride_rhs_ptrs, stride_rhs_scale_ptrs,
	
	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	"""
		Fused dequantization and grouped GEMM kernel for bf16 lhs and fp8 rhs matrices.
		Args:
			- lhs_ptr: Pointer to the lhs matrix (M, K) in bf16 dtype.
			- rhs_ptrs_ptr: Pointer to a tensor of pointers to rhs matrices (N, K) in fp8 dtype.
			- rhs_scale_ptrs_ptr: Pointer to a tensor of pointers to scale factors for rhs. 
				Each scale tensor is of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
			- group_sizes_ptr: Pointer to a tensor of group sizes. Shape is (num_groups,).
			- group_start_indices_ptr: Pointer to a tensor. Each element is the starting index of the group in lhs.
			- output_ptr: Pointer to the output tensor (M, N) in bf16

		Notes:
			- One-D CTA launch. Each program process associated tiles for each group.
			- For each group, M is different while K and N are the same across all groups.
			- Require lhs to be sorted by group ID.
	"""
	pid = tl.program_id(axis=0)
	num_programs = tl.num_programs(axis=0)
	lhs_dtype = tl.bfloat16
	rhs_dtype = tl.float8e4nv
	# NOTE: We ensure we launch N // GEMM_BLOCK_SIZE_N programs.
	offsets_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N) 
	scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
	num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
	for g in range(num_groups):
		# Get group size: gm
		gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
		group_idx = tl.load(group_idx_ptr + g * stride_group_idx) # Which group we are working on.
		# Get row indices for the current group.
		start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
		num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)

		# We have determinated the rhs. So we do the base pointer calculation here.
		rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(rhs_dtype))
		rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))

		for sub_group_idx in range(num_sub_groups):
			# Calculate the base pointer for the current sub-group
			sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
			# Remaining rows in the group:
			remaining_rows_in_group = start_idx + gm - sub_group_start_idx
			valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)

			# base_ptr_sub_g = lhs_ptr + sub_group_start_idx * stride_lhs_m
			# # Process the associated tile
			offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)

			# Loop along K dimension
			acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
			for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
				offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
				# Create pointers for lhs and rhs
				abs_row_indices = sub_group_start_idx + offsets_m
				# lhs_ptrs = base_ptr_sub_g + (offsets_m[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
				
				# Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
				# Find out which scale block this tile is on:
				scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
				scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				# Load the scale for this tile
				scale = tl.load(scale_ptr)

				# Create masks for lhs and rhs
				# lhs_mask = (offsets_m[:, None] < valid_rows_this_block) & (offsets_k[None, :] < K)
				lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
				rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

				# Load rhs tile:
				rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				# Dequantize rhs from fp8 to bf16
				rhs_fp32 = tl.cast(rhs_fp8, tl.float32)
				rhs_scaled = rhs_fp32 * scale
				rhs_bf16 = tl.cast(rhs_scaled, lhs_dtype)

				# Load lhs tile:
				lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
				# , cache_modifier='.cg'
				# Product
				acc += tl.dot(lhs, tl.trans(rhs_bf16))
			# Store the result
			offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
			# output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
			output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)


			# Convert to bf16 before storing
			output = tl.cast(acc, lhs_dtype)
			tl.store(output_ptrs, output, mask=output_mask)

@triton.jit
def fused_dequant_grouped_gemm_bf16_fp8_kernel_fp32(
	lhs_ptr, rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	output_ptr,
	M, N: tl.constexpr, K: tl.constexpr, num_groups,
	stride_lhs_m, stride_lhs_k,
	stride_rhs_n, stride_rhs_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices, stride_rhs_ptrs, stride_rhs_scale_ptrs,
	
	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	"""
		Fused dequantization and grouped GEMM kernel for bf16 lhs and fp8 rhs matrices.
		Args:
			- lhs_ptr: Pointer to the lhs matrix (M, K) in bf16 dtype.
			- rhs_ptrs_ptr: Pointer to a tensor of pointers to rhs matrices (N, K) in fp8 dtype.
			- rhs_scale_ptrs_ptr: Pointer to a tensor of pointers to scale factors for rhs. 
				Each scale tensor is of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
			- group_sizes_ptr: Pointer to a tensor of group sizes. Shape is (num_groups,).
			- group_start_indices_ptr: Pointer to a tensor. Each element is the starting index of the group in lhs.
			- output_ptr: Pointer to the output tensor (M, N) in bf16

		Notes:
			- One-D CTA launch. Each program process associated tiles for each group.
			- For each group, M is different while K and N are the same across all groups.
			- Require lhs to be sorted by group ID.
	"""
	pid = tl.program_id(axis=0)
	num_programs = tl.num_programs(axis=0)
	lhs_dtype = tl.bfloat16
	rhs_dtype = tl.float8e4nv
	# NOTE: We ensure we launch N // GEMM_BLOCK_SIZE_N programs.
	offsets_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N) 
	scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
	num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
	for g in range(num_groups):
		# Get group size: gm
		gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
		group_idx = tl.load(group_idx_ptr + g * stride_group_idx) # Which group we are working on.
		# Get row indices for the current group.
		start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
		num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)

		# We have determinated the rhs. So we do the base pointer calculation here.
		rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(rhs_dtype))
		rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))

		for sub_group_idx in range(num_sub_groups):
			# Calculate the base pointer for the current sub-group
			sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
			# Remaining rows in the group:
			remaining_rows_in_group = start_idx + gm - sub_group_start_idx
			valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)

			# base_ptr_sub_g = lhs_ptr + sub_group_start_idx * stride_lhs_m
			# # Process the associated tile
			offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)

			# Loop along K dimension
			acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
			for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
				offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
				# Create pointers for lhs and rhs
				abs_row_indices = sub_group_start_idx + offsets_m
				# lhs_ptrs = base_ptr_sub_g + (offsets_m[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
				
				# Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
				# Find out which scale block this tile is on:
				scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
				scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				# Load the scale for this tile
				scale = tl.load(scale_ptr)

				# Create masks for lhs and rhs
				# lhs_mask = (offsets_m[:, None] < valid_rows_this_block) & (offsets_k[None, :] < K)
				lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
				rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

				# Load rhs tile:
				rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				# Dequantize rhs from fp8 to bf16
				rhs_fp32 = tl.cast(rhs_fp8, tl.float32)
				rhs_scaled = rhs_fp32 * scale
				# rhs_bf16 = tl.cast(rhs_scaled, lhs_dtype)

				# Load lhs tile:
				lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
				# , cache_modifier='.cg'
				# Product
				lhs_fp32 = tl.cast(lhs, tl.float32)
				acc += tl.dot(lhs_fp32, tl.trans(rhs_scaled))
			# Store the result
			offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
			# output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
			output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)


			# Convert to bf16 before storing
			output = tl.cast(acc, lhs_dtype)
			tl.store(output_ptrs, output, mask=output_mask)


										
@torch.inference_mode()
def fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
		lhs: torch.Tensor,
		rhs_list: list[torch.Tensor],
		rhs_scale_list: list[torch.Tensor],
		group_sizes: tuple[int, int],
		group_start_indices: torch.Tensor,
		gemm_block_size=(64, 64, 128), 
		scale_block_size=(128, 128),
		num_stages=2,
		num_warps=4
):
	"""
		Performs a fused dequantization and grouped_gemm. We dequantize the fp8 rhs to bf16 on the fly.
		Args:
			lhs: torch.Tensor of shape (M, K) in bf16 dtype.
			rhs_list: List of torch.Tensor, each of shape (N, K) in fp8 dtype.
			rhs_scale_list: List of torch.Tensor, each of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
			group_sizes: Tuple of (group ID, group size) for each group.
			gemm_block_size: Tuple of (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
			scale_block_size: Tuple of (SCALE_BLOCK_M, SCALE_BLOCK_K)
		
		Returns:
			C: torch.Tensor of shape (M, N) in bf16 dtype.

	"""
	assert lhs.dtype == torch.bfloat16, "lhs must be of dtype bfloat16"
	assert all(r.dtype == torch.float8_e4m3fn for r in rhs_list), "All rhs matrices must be of dtype float8_e4m3fn"
	assert all(s.dtype == torch.float32 for s in rhs_scale_list), "All scale tensors must be of dtype float32"
	assert len(rhs_list) == len(rhs_scale_list), "rhs_list and rhs_scale_list must have the same length"

	device = lhs.device
	N = rhs_list[0].shape[0]
	K = lhs.shape[1]
	rhs_ptrs_ptr = torch.tensor([r.data_ptr() for r in rhs_list], dtype=torch.int64, device=device)
	rhs_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in rhs_scale_list], dtype=torch.int64, device=device)
	group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
	activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
	num_groups = len(group_sizes)
	# group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
	# group_start_indices[0] = 0  # The first group starts at index 0

	output = torch.zeros((lhs.shape[0], N), dtype=torch.bfloat16, device=device)
	num_sms = torch.cuda.get_device_properties(device).multi_processor_count
	# grid = (num_sms,)
	# grid = lambda META: (
	# 	triton.cdiv(16, META['GEMM_BLOCK_SIZE_M']) * triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']),
	# )
	# grid = lambda META: (
	# 	triton.cdiv(16, META['GEMM_BLOCK_SIZE_M']) * triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']),
	# )
	grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
	# logging.info(f"Rank {dist.get_rank()} launching fused_dequant_grouped_gemm_bf16_fp8_kernel with input shapes: lhs: {lhs.shape}, rhs_list: {[r.shape for r in rhs_list]}, rhs_scale_list: {[s.shape for s in rhs_scale_list]}, group_sizes: {group_size.tolist()}, group_start_indices: {group_start_indices.tolist()}, selected groups: {activated_group_idx.tolist()}")
	# # Launch the kernel
	try:
		fused_dequant_grouped_gemm_bf16_fp8_kernel_v2[grid](
			lhs, rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
			activated_group_idx, group_size, group_start_indices,
			output,
			lhs.shape[0], N, K, num_groups,
			lhs.stride(0), lhs.stride(1),
			rhs_list[0].stride(0), rhs_list[0].stride(1),
			output.stride(0), output.stride(1),
			activated_group_idx.stride(0), 
			group_size.stride(0), group_start_indices.stride(0),
			rhs_ptrs_ptr.stride(0), rhs_scale_ptrs_ptr.stride(0),
			GEMM_BLOCK_SIZE_M=gemm_block_size[0], GEMM_BLOCK_SIZE_N=gemm_block_size[1], GEMM_BLOCK_SIZE_K=gemm_block_size[2],
			SCALE_BLOCK_SIZE_N=scale_block_size[0], SCALE_BLOCK_SIZE_K=scale_block_size[1],
			num_warps=num_warps,
			num_stages=num_stages
		)
	except Exception as e:
		print(f"Error launching fused_dequant_grouped_gemm_bf16_fp8_kernel: {e}")
		raise
	return output




# @triton.jit
# def fused_dequant_grouped_gemm_fp8_fp8_kernel(
# 	lhs_ptr, lhs_scale_ptr,
# 	rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
# 	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
# 	output_ptr,
# 	M, N:tl.constexpr, K:tl.constexpr, num_groups,
# 	stride_lhs_m, stride_lhs_k,
# 	stride_lhs_scale_m, stride_lhs_scale_k,
# 	stride_rhs_n, stride_rhs_k,
# 	stride_output_m, stride_output_n,
# 	stride_group_idx, stride_group_sizes, stride_group_start_indices, stride_rhs_ptrs, stride_rhs_scale_ptrs,
	
# 	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
# 	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
# ):
# 	"""
# 		Fused dequantization and grouped GEMM kernel for bf16 lhs and fp8 rhs matrices.
# 		Args:
# 			- lhs_ptr: Pointer to the lhs matrix (M, K) in bf16 dtype.
# 			- rhs_ptrs_ptr: Pointer to a tensor of pointers to rhs matrices (N, K) in fp8 dtype.
# 			- rhs_scale_ptrs_ptr: Pointer to a tensor of pointers to scale factors for rhs. 
# 				Each scale tensor is of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
# 			- group_sizes_ptr: Pointer to a tensor of group sizes. Shape is (num_groups,).
# 			- group_start_indices_ptr: Pointer to a tensor. Each element is the starting index of the group in lhs.
# 			- output_ptr: Pointer to the output tensor (M, N) in bf16

# 		Notes:
# 			- One-D CTA launch. Each program process associated tiles for each group.
# 			- For each group, M is different while K and N are the same across all groups.
# 			- Require lhs to be sorted by group ID.
# 	"""
# 	pid = tl.program_id(axis=0)
# 	num_programs = tl.num_programs(axis=0)
# 	lhs_dtype = tl.bfloat16
# 	rhs_dtype = tl.float8e4nv
# 	# NOTE: We ensure we launch N // GEMM_BLOCK_SIZE_N programs.
# 	offsets_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N) 
# 	scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
# 	num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
# 	for g in range(num_groups):
# 		# Get group size: gm
# 		gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
# 		group_idx = tl.load(group_idx_ptr + g * stride_group_idx) # Which group we are working on.
# 		# Get row indices for the current group.
# 		start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
# 		num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)

# 		# We have determinated the rhs. So we do the base pointer calculation here.
# 		rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(rhs_dtype))
# 		rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))

# 		for sub_group_idx in range(num_sub_groups):
# 			# Calculate the base pointer for the current sub-group
# 			sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
# 			# Remaining rows in the group:
# 			remaining_rows_in_group = start_idx + gm - sub_group_start_idx
# 			valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)

# 			# base_ptr_sub_g = lhs_ptr + sub_group_start_idx * stride_lhs_m
# 			# # Process the associated tile
# 			offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)

# 			# Loop along K dimension
# 			acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
# 			for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
# 				offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
# 				# Create pointers for lhs and rhs
# 				abs_row_indices = sub_group_start_idx + offsets_m
# 				# lhs_ptrs = base_ptr_sub_g + (offsets_m[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
# 				lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
# 				rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
				
# 				# Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
# 				# Find out which scale block this tile is on:
# 				scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
# 				scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
# 				# Load the scale for this tile
# 				rhs_scale = tl.load(scale_ptr)

# 				# Create masks for lhs and rhs
# 				# lhs_mask = (offsets_m[:, None] < valid_rows_this_block) & (offsets_k[None, :] < K)
# 				lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
# 				rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

# 				# Load rhs tile:
# 				rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
# 				# Dequantize rhs from fp8 to bf16
# 				# rhs_fp32 = tl.cast(rhs_fp8, tl.float32)
# 				# rhs_scaled = rhs_fp32 * scale
# 				# rhs_bf16 = tl.cast(rhs_scaled, lhs_dtype)

# 				# Load lhs tile:
# 				lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
# 				lhs_scale_k = k_idx * GEMM_BLOCK_SIZE_K // 128
# 				l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
# 				lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0, cache_modifier='.cg')
				

# 				# Product
# 				acc += tl.dot(lhs, tl.trans(rhs_fp8)) * lhs_scale * rhs_scale
# 			# Store the result
# 			offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
# 			offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
# 			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
# 			# output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
# 			output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)


# 			# Convert to bf16 before storing
# 			output = tl.cast(acc, lhs_dtype)
# 			tl.store(output_ptrs, output, mask=output_mask)


# @torch.inference_mode()
# def fused_dequant_grouped_gemm_fp8_fp8_triton(
# 		lhs: torch.Tensor,
# 		lhs_scale: torch.Tensor,
# 		rhs_list: list[torch.Tensor],
# 		rhs_ptrs_ptr: torch.Tensor,
# 		rhs_scale_list: list[torch.Tensor],
# 		rhs_scale_ptrs_ptr: torch.Tensor,
# 		# group_sizes: tuple[int, int],
# 		group_size: torch.Tensor,
# 		activated_group_idx: torch.Tensor,
# 		group_start_indices: torch.Tensor,
# 		gemm_block_size=(64, 32, 128), 
# 		scale_block_size=(128, 128),
# 		num_stages=2,
# 		num_warps=4
# ):
# 	"""
# 		Performs a fused dequantization and grouped_gemm. We dequantize the fp8 rhs to bf16 on the fly.
# 		Args:
# 			lhs: torch.Tensor of shape (M, K) in bf16 dtype.
# 			lhs_scale: torch.Tensor of shape (M, lhs_dim // lhs_scale_block_size) in fp32 dtype.
# 			rhs_list: List of torch.Tensor, each of shape (N, K) in fp8 dtype.
# 			rhs_scale_list: List of torch.Tensor, each of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
# 			group_sizes: Tuple of (group ID, group size) for each group.
# 			gemm_block_size: Tuple of (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
# 			scale_block_size: Tuple of (SCALE_BLOCK_M, SCALE_BLOCK_K)
		
# 		Returns:
# 			C: torch.Tensor of shape (M, N) in bf16 dtype.

# 	"""
# 	assert lhs.dtype == torch.float8_e4m3fn, "lhs must be of dtype float8_e4m3fn"
# 	assert lhs_scale.dtype == torch.float32, "lhs_scale must be of dtype float32"
# 	assert all(r.dtype == torch.float8_e4m3fn for r in rhs_list), "All rhs matrices must be of dtype float8_e4m3fn"
# 	assert all(s.dtype == torch.float32 for s in rhs_scale_list), "All scale tensors must be of dtype float32"
# 	assert len(rhs_list) == len(rhs_scale_list), "rhs_list and rhs_scale_list must have the same length"

# 	device = lhs.device
# 	N = rhs_list[0].shape[0]
# 	K = lhs.shape[1]
# 	# rhs_ptrs_ptr = torch.tensor([r.data_ptr() for r in rhs_list], dtype=torch.int64, device=device)
# 	# rhs_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in rhs_scale_list], dtype=torch.int64, device=device)
# 	# group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
# 	# activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
# 	num_groups = group_size.shape[0]

# 	output = torch.zeros((lhs.shape[0], N), dtype=torch.bfloat16, device=device)
# 	# num_sms = torch.cuda.get_device_properties(device).multi_processor_count
# 	grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
# 	# logging.info(f"Rank {dist.get_rank()} launching fused_dequant_grouped_gemm_bf16_fp8_kernel with input shapes: lhs: {lhs.shape}, rhs_list: {[r.shape for r in rhs_list]}, rhs_scale_list: {[s.shape for s in rhs_scale_list]}, group_sizes: {group_size.tolist()}, group_start_indices: {group_start_indices.tolist()}, selected groups: {activated_group_idx.tolist()}")
# 	# # Launch the kernel
# 	try:
# 		fused_dequant_grouped_gemm_fp8_fp8_kernel[grid](
# 			lhs, lhs_scale,
# 			rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
# 			activated_group_idx, group_size, group_start_indices,
# 			output,
# 			lhs.shape[0], N, K, num_groups,
# 			lhs.stride(0), lhs.stride(1),
# 			lhs_scale.stride(0), lhs_scale.stride(1),
# 			rhs_list[0].stride(0), rhs_list[0].stride(1),
# 			output.stride(0), output.stride(1),
# 			activated_group_idx.stride(0), 
# 			group_size.stride(0), group_start_indices.stride(0),
# 			rhs_ptrs_ptr.stride(0), rhs_scale_ptrs_ptr.stride(0),
# 			GEMM_BLOCK_SIZE_M=gemm_block_size[0], GEMM_BLOCK_SIZE_N=gemm_block_size[1], GEMM_BLOCK_SIZE_K=gemm_block_size[2],
# 			SCALE_BLOCK_SIZE_N=scale_block_size[0], SCALE_BLOCK_SIZE_K=scale_block_size[1],
# 			num_warps=num_warps,
# 			num_stages=num_stages
# 		)
# 	except Exception as e:
# 		print(f"Error launching fused_dequant_grouped_gemm_bf16_fp8_kernel: {e}")
# 		raise
# 	return output


@triton.jit
def fused_dequant_grouped_gemm_fp8_fp8_kernel(
	lhs_ptr, lhs_scale_ptr,
	rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	num_active_experts_ptr,  # <-- ADD THIS: pointer to num_active tensor
	output_ptr,
	M, N:tl.constexpr, K:tl.constexpr,  # <-- REMOVE num_groups from here
	stride_lhs_m, stride_lhs_k,
	stride_lhs_scale_m, stride_lhs_scale_k,
	stride_rhs_n, stride_rhs_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices, stride_rhs_ptrs, stride_rhs_scale_ptrs,
	
	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	"""
		Fused dequantization and grouped GEMM kernel for bf16 lhs and fp8 rhs matrices.
		Args:
			- lhs_ptr: Pointer to the lhs matrix (M, K) in bf16 dtype.
			- rhs_ptrs_ptr: Pointer to a tensor of pointers to rhs matrices (N, K) in fp8 dtype.
			- rhs_scale_ptrs_ptr: Pointer to a tensor of pointers to scale factors for rhs. 
				Each scale tensor is of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
			- group_sizes_ptr: Pointer to a tensor of group sizes. Shape is (num_groups,).
			- group_start_indices_ptr: Pointer to a tensor. Each element is the starting index of the group in lhs.
			- num_active_experts_ptr: Pointer to num_active_experts tensor [1] on device
			- output_ptr: Pointer to the output tensor (M, N) in bf16

		Notes:
			- One-D CTA launch. Each program process associated tiles for each group.
			- For each group, M is different while K and N are the same across all groups.
			- Require lhs to be sorted by group ID.
	"""
	pid = tl.program_id(axis=0)
	num_programs = tl.num_programs(axis=0)
	lhs_dtype = tl.bfloat16
	rhs_dtype = tl.float8e4nv
	
	# ===== LOAD NUM_GROUPS FROM DEVICE (NO CPU-GPU SYNC!) =====
	num_groups = tl.load(num_active_experts_ptr)
	
	# NOTE: We ensure we launch N // GEMM_BLOCK_SIZE_N programs.
	offsets_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N) 
	scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
	num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
	
	for g in range(num_groups):
		# Get group size: gm
		gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
		group_idx = tl.load(group_idx_ptr + g * stride_group_idx) # Which group we are working on.
		# Get row indices for the current group.
		start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
		num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)

		# We have determinated the rhs. So we do the base pointer calculation here.
		rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(rhs_dtype))
		rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))

		for sub_group_idx in range(num_sub_groups):
			# Calculate the base pointer for the current sub-group
			sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
			# Remaining rows in the group:
			remaining_rows_in_group = start_idx + gm - sub_group_start_idx
			valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)

			# base_ptr_sub_g = lhs_ptr + sub_group_start_idx * stride_lhs_m
			# # Process the associated tile
			offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)

			# Loop along K dimension
			acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
			for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
				offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
				# Create pointers for lhs and rhs
				abs_row_indices = sub_group_start_idx + offsets_m
				# lhs_ptrs = base_ptr_sub_g + (offsets_m[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
				
				# Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
				# Find out which scale block this tile is on:
				scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
				scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				# Load the scale for this tile
				rhs_scale = tl.load(scale_ptr)

				# Create masks for lhs and rhs
				# lhs_mask = (offsets_m[:, None] < valid_rows_this_block) & (offsets_k[None, :] < K)
				lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
				rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

				# Load rhs tile:
				rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				# Dequantize rhs from fp8 to bf16
				# rhs_fp32 = tl.cast(rhs_fp8, tl.float32)
				# rhs_scaled = rhs_fp32 * scale
				# rhs_bf16 = tl.cast(rhs_scaled, lhs_dtype)

				# Load lhs tile:
				lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
				lhs_scale_k = k_idx * GEMM_BLOCK_SIZE_K // 128
				l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
				lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0, cache_modifier='.cg')
				

				# Product
				acc += tl.dot(lhs, tl.trans(rhs_fp8)) * lhs_scale * rhs_scale
			# Store the result
			offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
			# output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
			output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)


			# Convert to bf16 before storing
			output = tl.cast(acc, lhs_dtype)
			tl.store(output_ptrs, output, mask=output_mask)


@torch.inference_mode()
def fused_dequant_grouped_gemm_fp8_fp8_triton(
		lhs: torch.Tensor,
		lhs_scale: torch.Tensor,
		rhs_list: list[torch.Tensor],
		rhs_ptrs_ptr: torch.Tensor,
		rhs_scale_list: list[torch.Tensor],
		rhs_scale_ptrs_ptr: torch.Tensor,
		group_size: torch.Tensor,  # <-- Now full-sized (experts_per_rank)
		activated_group_idx: torch.Tensor,  # <-- Now full-sized (experts_per_rank)
		group_start_indices: torch.Tensor,  # <-- Now full-sized (experts_per_rank)
		num_active_experts: torch.Tensor,  # <-- ADD THIS: [1] tensor containing count
		gemm_block_size=(64, 32, 128), 
		scale_block_size=(128, 128),
		num_stages=2,
		num_warps=4
):
	"""
		Performs a fused dequantization and grouped_gemm. We dequantize the fp8 rhs to bf16 on the fly.
		Args:
			lhs: torch.Tensor of shape (M, K) in bf16 dtype.
			lhs_scale: torch.Tensor of shape (M, lhs_dim // lhs_scale_block_size) in fp32 dtype.
			rhs_list: List of torch.Tensor, each of shape (N, K) in fp8 dtype.
			rhs_scale_list: List of torch.Tensor, each of shape (ceil(N / scale_block_size[0]), ceil(K / scale_block_size[1])) in fp32 dtype.
			group_size: Tensor of group sizes (full-sized, experts_per_rank)
			activated_group_idx: Tensor of active expert indices (full-sized, experts_per_rank)
			group_start_indices: Tensor of group start indices (full-sized, experts_per_rank)
			num_active_experts: Tensor [1] containing actual number of active experts
			gemm_block_size: Tuple of (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
			scale_block_size: Tuple of (SCALE_BLOCK_M, SCALE_BLOCK_K)
		
		Returns:
			C: torch.Tensor of shape (M, N) in bf16 dtype.

	"""
	assert lhs.dtype == torch.float8_e4m3fn, "lhs must be of dtype float8_e4m3fn"
	assert lhs_scale.dtype == torch.float32, "lhs_scale must be of dtype float32"
	assert all(r.dtype == torch.float8_e4m3fn for r in rhs_list), "All rhs matrices must be of dtype float8_e4m3fn"
	assert all(s.dtype == torch.float32 for s in rhs_scale_list), "All scale tensors must be of dtype float32"
	assert len(rhs_list) == len(rhs_scale_list), "rhs_list and rhs_scale_list must have the same length"

	device = lhs.device
	N = rhs_list[0].shape[0]
	K = lhs.shape[1]

	# NO LONGER NEEDED: num_groups = group_size.shape[0]

	output = torch.zeros((lhs.shape[0], N), dtype=torch.bfloat16, device=device)
	grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
	
	try:
		fused_dequant_grouped_gemm_fp8_fp8_kernel[grid](
			lhs, lhs_scale,
			rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
			activated_group_idx, group_size, group_start_indices,
			num_active_experts,  # <-- Pass the tensor (NOT .item()!)
			output,
			lhs.shape[0], N, K,  # <-- Remove num_groups from here
			lhs.stride(0), lhs.stride(1),
			lhs_scale.stride(0), lhs_scale.stride(1),
			rhs_list[0].stride(0), rhs_list[0].stride(1),
			output.stride(0), output.stride(1),
			activated_group_idx.stride(0), 
			group_size.stride(0), group_start_indices.stride(0),
			rhs_ptrs_ptr.stride(0), rhs_scale_ptrs_ptr.stride(0),
			GEMM_BLOCK_SIZE_M=gemm_block_size[0], GEMM_BLOCK_SIZE_N=gemm_block_size[1], GEMM_BLOCK_SIZE_K=gemm_block_size[2],
			SCALE_BLOCK_SIZE_N=scale_block_size[0], SCALE_BLOCK_SIZE_K=scale_block_size[1],
			num_warps=num_warps,
			num_stages=num_stages
		)
	except Exception as e:
		print(f"Error launching fused_dequant_grouped_gemm_bf16_fp8_kernel: {e}")
		raise
	return output

# =============================================================================
# ALLOCATOR SETUP
# =============================================================================
_allocator_set = False

def _setup_allocator_once():
	"""Set up Triton allocator for TMA descriptors (call once per process)."""
	global _allocator_set
	if not _allocator_set:
		def alloc_fn(size: int, alignment: int, stream: int):
			return torch.empty(size, device='cuda', dtype=torch.int8)
		
		triton.set_allocator(alloc_fn)
		_allocator_set = True


@triton.jit
def fp8_grouped_gemm_persistent_tma_kernel(
	lhs_ptr, lhs_scale_ptr,
	rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	num_active_experts_ptr,
	output_ptr,
	M, N: tl.constexpr, K: tl.constexpr,
	stride_lhs_m, stride_lhs_k,
	stride_lhs_scale_m, stride_lhs_scale_k,
	stride_rhs_n, stride_rhs_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices,
	stride_rhs_ptrs, stride_rhs_scale_ptrs,
	GEMM_BLOCK_SIZE_M: tl.constexpr,
	GEMM_BLOCK_SIZE_N: tl.constexpr,
	GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_K: tl.constexpr,
	NUM_SMS: tl.constexpr,
	NUM_N_BLOCKS: tl.constexpr,
):
	"""
	Persistent kernel for FP8 grouped GEMM with TMA descriptors.
	
	Key features from v3:
	- Persistent scheduling (work stealing across experts)
	- TMA descriptors for efficient memory access
	- Larger N-tiles to reduce overhead
	- Combined scale factors for better ILP
	- Hoisted invariants
	
	One CTA processes multiple (expert, N-block) pairs.
	"""
	start_pid = tl.program_id(axis=0)
	
	# Load actual number of active experts
	num_groups = tl.load(num_active_experts_ptr)
	total_work_items = num_groups * NUM_N_BLOCKS
	
	# Early exit if no work
	if start_pid >= total_work_items:
		return
	
	# --- Create Static Descriptors ONCE (for LHS and output) ---
	lhs_desc = tl.make_tensor_descriptor(
		lhs_ptr,
		shape=[M, K],
		strides=[stride_lhs_m, stride_lhs_k],
		block_shape=[GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_K]
	)
	output_desc = tl.make_tensor_descriptor(
		output_ptr,
		shape=[M, N],
		strides=[stride_output_m, stride_output_n],
		block_shape=[GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N]
	)
	
	# Pre-compute constants (hoisted)
	num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
	num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
	offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
	offsets_n = tl.arange(0, GEMM_BLOCK_SIZE_N)
	
	# --- Persistent Loop: Work Stealing ---
	work_item_id = start_pid
	while work_item_id < total_work_items:
		
		# Unpack work item into (group_id, n_block_id)
		group_pid = work_item_id // NUM_N_BLOCKS
		n_pid = work_item_id % NUM_N_BLOCKS
		
		# Load expert metadata
		gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
		
		if gm > 0:
			# Load expert data
			group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
			start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
			
			# Load RHS weight pointers for this expert
			rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(tl.float8e4nv))
			rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))
			
			# Create TMA descriptor for RHS (dynamic per expert)
			rhs_desc = tl.make_tensor_descriptor(
				rhs_base_ptr,
				shape=[N, K],
				strides=[stride_rhs_n, stride_rhs_k],
				block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K]
			)
			
			# N-block offsets (hoisted out of M-loop)
			offs_bn = n_pid * GEMM_BLOCK_SIZE_N
			offs_n = offs_bn + offsets_n
			n_mask = offs_n < N
			scale_n_idx = offs_bn // SCALE_BLOCK_SIZE_K
			
			# M-loop bounds
			num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
			
			# --- M-block loop ---
			for sub_group_idx in range(num_sub_groups):
				sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
				offs_am = sub_group_start_idx
				
				remaining_rows = start_idx + gm - sub_group_start_idx
				valid_rows = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows)
				
				abs_row_indices = sub_group_start_idx + offsets_m
				m_mask = abs_row_indices < M
				valid_mask = (offsets_m < valid_rows)[:, None]
				
				# Initialize accumulator
				acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
				
				# --- K-block loop ---
				for k_block_idx in range(num_k_blocks):
					offs_k = k_block_idx * GEMM_BLOCK_SIZE_K
					
					# Load scales
					scale_offset = scale_n_idx * num_scale_k + k_block_idx
					rhs_scale = tl.load(rhs_scale_base_ptr + scale_offset)
					
					lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
													  k_block_idx * stride_lhs_scale_k)
					lhs_scale_mask = m_mask[:, None] & valid_mask
					lhs_scale = tl.load(lhs_scale_ptrs, mask=lhs_scale_mask, other=1.0)
					
					# Load data via TMA descriptors
					lhs = lhs_desc.load([offs_am, offs_k])
					rhs_fp8 = rhs_desc.load([offs_bn, offs_k])
					
					# Apply masking to LHS
					lhs = tl.where(valid_mask, lhs, 0.0)
					
					# Combined scale factors (better ILP)
					combined_scale = lhs_scale * rhs_scale
					
					# GEMM accumulation
					acc += tl.dot(lhs, tl.trans(rhs_fp8), out_dtype=tl.float32) * combined_scale
				
				# Convert to output dtype
				output = acc.to(tl.bfloat16)
				
				# Store with masking
				output_mask = m_mask[:, None] & n_mask[None, :] & valid_mask
				output_masked = tl.where(output_mask, output, 0.0)
				output_desc.store([offs_am, offs_bn], output_masked)
		
		# Work stealing: grab next work item
		work_item_id += NUM_SMS


@torch.inference_mode()
def fp8_grouped_gemm_persistent_tma(
	lhs: torch.Tensor,
	lhs_scale: torch.Tensor,
	rhs_list: list[torch.Tensor],
	rhs_ptrs_ptr: torch.Tensor,
	rhs_scale_list: list[torch.Tensor],
	rhs_scale_ptrs_ptr: torch.Tensor,
	group_sizes: torch.Tensor,
	activated_group_idx: torch.Tensor,
	group_start_indices: torch.Tensor,
	num_active_experts: torch.Tensor,
	num_groups: int,
	gemm_block_size=[64, 128, 128],  # Larger N-tiles like v3!
	scale_block_size=128,
	num_stages=3,
	num_warps=8
):
	"""
	FP8 Grouped GEMM with persistent kernel + TMA descriptors.
	
	Args:
		lhs: (M, K) FP8 tensor
		lhs_scale: (M, K//scale_block_size) FP32 scale factors
		rhs_list: List of (N, K) FP8 weight tensors
		rhs_ptrs_ptr: Tensor of pointers to RHS matrices
		rhs_scale_list: List of (N//scale_block_size, K//scale_block_size) FP32 scales
		rhs_scale_ptrs_ptr: Tensor of pointers to RHS scale matrices
		group_sizes: (num_experts,) tensor of group sizes
		activated_group_idx: (num_experts,) tensor of active expert indices
		group_start_indices: (num_experts,) tensor of start indices
		num_active_experts: (1,) tensor with count of active experts
		gemm_block_size: [M, N, K] tile sizes (default [64, 128, 128])
		scale_block_size: Scale block size for K dimension
		num_stages: Number of pipeline stages (for Triton's auto-pipelining)
		num_warps: Number of warps per CTA
	
	Returns:
		output: (M, N) BF16 tensor
	"""
	_setup_allocator_once()
	
	assert lhs.dtype == torch.float8_e4m3fn
	assert lhs_scale.dtype == torch.float32
	assert all(r.dtype == torch.float8_e4m3fn for r in rhs_list)
	assert all(s.dtype == torch.float32 for s in rhs_scale_list)
	assert gemm_block_size[2] == scale_block_size
	
	device = lhs.device
	M = lhs.shape[0]
	N = rhs_list[0].shape[0]
	K = lhs.shape[1]
	
	output = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
	
	# Calculate work distribution
	NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
	num_n_blocks = triton.cdiv(N, gemm_block_size[1])
	# actual_num_experts = num_active_experts.item()
	actual_num_experts = num_groups
	total_work_items = actual_num_experts * num_n_blocks
	
	# Launch grid: Use all SMs up to total work
	grid = (min(NUM_SMS, total_work_items),)
	
	fp8_grouped_gemm_persistent_tma_kernel[grid](
		lhs, lhs_scale,
		rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
		activated_group_idx, group_sizes, group_start_indices,
		num_active_experts,
		output,
		M, N, K,
		lhs.stride(0), lhs.stride(1),
		lhs_scale.stride(0), lhs_scale.stride(1),
		rhs_list[0].stride(0), rhs_list[0].stride(1),
		output.stride(0), output.stride(1),
		activated_group_idx.stride(0),
		group_sizes.stride(0),
		group_start_indices.stride(0),
		rhs_ptrs_ptr.stride(0),
		rhs_scale_ptrs_ptr.stride(0),
		GEMM_BLOCK_SIZE_M=gemm_block_size[0],
		GEMM_BLOCK_SIZE_N=gemm_block_size[1],
		GEMM_BLOCK_SIZE_K=gemm_block_size[2],
		SCALE_BLOCK_SIZE_K=scale_block_size,
		NUM_SMS=NUM_SMS,
		NUM_N_BLOCKS=num_n_blocks,
		num_stages=num_stages,
		num_warps=num_warps
	)
	
	return output