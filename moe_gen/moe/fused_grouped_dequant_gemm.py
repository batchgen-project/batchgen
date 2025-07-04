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
			# num_warps=4,
    		# num_stages=4
		)
	except Exception as e:
		print(f"Error launching fused_dequant_grouped_gemm_bf16_fp8_kernel: {e}")
		raise
	return output



@triton.jit
def fused_dequant_grouped_gemm_fp8_fp8_kernel(
	lhs_ptr, lhs_scale_ptr,
	rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
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
		Fused grouped GEMM kernel for fp8 lhs and fp8 rhs matrices.
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