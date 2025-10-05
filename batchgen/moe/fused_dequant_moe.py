import torch
import triton
import triton.language as tl
import os
import torch.distributed as dist
import logging
# os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.triton/cache")
# os.environ["TRITON_CACHE_MANAGER"] = "1"


# @triton.autotune(
# 	configs=[
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 64, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 64}),
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 64, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 128}),
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 64, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 32}),
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 32, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 64}),
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 32, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 32}),
# 	],
# 	key=['N', 'K'],  # Autotune based on these input shapes
# 	# warmup=25,            # Number of warmup iterations
# 	# rep=100,              # Number of measurement iterations
# 	use_cuda_graph=True   # Use CUDA graphs for more accurate timing
# )
@triton.jit
def fused_dequant_weighted_moe_stage_1_kernel(
	lhs_ptr, gate_ptrs_ptr, up_ptrs_ptr,
	gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	output_ptr,
	M, N: tl.constexpr, K: tl.constexpr, num_groups,
	stride_lhs_m, stride_lhs_k,
	stride_gate_n, stride_gate_k,
	stride_up_n, stride_up_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices,
	stride_weight_ptrs, stride_scale_ptrs,

	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	"""
		# (act(hidden_states @ deq(gate)) * (hidden_states @ deq(up))
		Note: act - silu. And we assume the gate and up weights have the same shape which is common in MoE models.
	"""
	pid = tl.program_id(axis=0)
	lhs_dtype = tl.bfloat16
	rhs_dtype = tl.float8e4nv
	scale_dtype = tl.float32
	acc_dtype = tl.float32


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
		gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
		up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
		gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))
		up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))

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
			gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype= acc_dtype)
			up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype= acc_dtype)
			for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
				offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
				# Create pointers for lhs and rhs
				abs_row_indices = sub_group_start_idx + offsets_m
				# lhs_ptrs = base_ptr_sub_g + (offsets_m[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				# rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
				gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
				up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
				
				# Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
				# Find out which scale block this tile is on:
				scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
				# scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				gate_scale_ptr = gate_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				up_scale_ptr = up_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				# Load the scale for this tile
				gate_scale = tl.load(gate_scale_ptr)
				up_scale = tl.load(up_scale_ptr)

				# Create masks for lhs and rhs
				lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
				rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

				# Load rhs tile:
				gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0, cache_modifier='.cg')

				gate_fp32 = tl.cast(gate_fp8, tl.float32)
				gate_scaled = gate_fp32 * gate_scale
				gate_bf16 = tl.cast(gate_scaled, lhs_dtype)
				gate_acc += tl.dot(lhs, tl.trans(gate_bf16))
				# gate_acc = tl.dot(lhs, tl.trans(gate_bf16), acc=gate_acc)

				up_fp32 = tl.cast(up_fp8, tl.float32)
				up_scaled = up_fp32 * up_scale
				up_bf16 = tl.cast(up_scaled, lhs_dtype)
				up_acc += tl.dot(lhs, tl.trans(up_bf16))
				# up_acc = tl.dot(lhs, tl.trans(up_bf16), acc=up_acc)

			# Store the result
			offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
			# output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
			output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)


			# Convert to bf16 before storing
			output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc 
			output = tl.cast(output_acc, lhs_dtype)
			tl.store(output_ptrs, output, mask=output_mask)


@torch.inference_mode()
def fused_dequant_weighted_moe_stage_1(
	hidden_states: torch.Tensor,
	gate_weight_list: list[torch.Tensor],
	up_weight_list: list[torch.Tensor],
	gate_scale_list: list[torch.Tensor],
	up_scale_list: list[torch.Tensor],
	group_sizes: tuple[int, int],
	group_start_indices: torch.Tensor,
	gate_gemm_block_size=[64,16,128],
	up_gemm_block_size=[64,16,128],
	scale_block_size=[128,128]
):
	assert hidden_states.dtype == torch.bfloat16, "hidden_states must be of dtype bfloat16"
	assert all(r.dtype == torch.float8_e4m3fn for r in gate_weight_list), "All gate weights must be of dtype float8_e4m3fn"
	assert all(r.dtype == torch.float8_e4m3fn for r in up_weight_list), "All up weights must be of dtype float8_e4m3fn"
	assert all(s.dtype == torch.float32 for s in gate_scale_list), "All gate scales must be of dtype float32"
	assert all(s.dtype == torch.float32 for s in up_scale_list), "All up scales must be of dtype float32"
	assert len(gate_weight_list) == len(gate_scale_list), "gate_weight_list and gate_scale_list must have the same length"
	assert len(up_weight_list) == len(up_scale_list), "up_weight_list and up_scale_list must have the same length"

	device = hidden_states.device
	M = hidden_states.shape[0]
	N = gate_weight_list[0].shape[0]
	K = hidden_states.shape[1]

	gate_ptrs_ptr = torch.tensor([r.data_ptr() for r in gate_weight_list], dtype=torch.int64, device=device)
	up_ptrs_ptr = torch.tensor([r.data_ptr() for r in up_weight_list], dtype=torch.int64, device=device)
	gate_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in gate_scale_list], dtype=torch.int64, device=device)
	up_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in up_scale_list], dtype=torch.int64, device=device)
	group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
	activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
	num_groups = len(group_sizes)

	output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
	grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
	# Launch the kernel
	try:
		fused_dequant_weighted_moe_stage_1_kernel[grid](
			hidden_states, gate_ptrs_ptr, up_ptrs_ptr,
			gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
			activated_group_idx, group_size, group_start_indices,
			output,
			M, N, K, num_groups,
			hidden_states.stride(0), hidden_states.stride(1),
			gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
			up_weight_list[0].stride(0), up_weight_list[0].stride(1),
			output.stride(0), output.stride(1),
			activated_group_idx.stride(0), group_size.stride(0), group_start_indices.stride(0),
			gate_ptrs_ptr.stride(0), gate_scale_ptrs_ptr.stride(0),

			gate_gemm_block_size[0], gate_gemm_block_size[1], gate_gemm_block_size[2],
			SCALE_BLOCK_SIZE_N = scale_block_size[0],
			SCALE_BLOCK_SIZE_K = scale_block_size[1]
		)
	except Exception as e:
		logging.error(f"Error in fused_dequant_weighted_moe_stage_1: {e}")
		raise
	return output


@triton.jit
def fused_fp8_moe_stage_1_kernel(
	lhs_ptr, lhs_scale_ptr,
	gate_ptrs_ptr, up_ptrs_ptr,
	gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	output_ptr,
	M, N: tl.constexpr, K: tl.constexpr, num_groups,
	stride_lhs_m, stride_lhs_k,
	stride_lhs_scale_m, stride_lhs_scale_k,
	stride_gate_n, stride_gate_k,
	stride_up_n, stride_up_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices,
	stride_weight_ptrs, stride_scale_ptrs,

	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	"""
		# (act(hidden_states @ deq(gate)) * (hidden_states @ deq(up))
		Note: act - silu. And we assume the gate and up weights have the same shape which is common in MoE models.
	"""
	pid = tl.program_id(axis=0)
	lhs_dtype = tl.bfloat16
	rhs_dtype = tl.float8e4nv
	scale_dtype = tl.float32
	acc_dtype = tl.float32


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
		gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
		up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
		gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))
		up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))

		for sub_group_idx in range(num_sub_groups):
			# Calculate the base pointer for the current sub-group
			sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
			# Remaining rows in the group:
			remaining_rows_in_group = start_idx + gm - sub_group_start_idx
			valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)

			# # Process the associated tile
			offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)

			# Loop along K dimension
			gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype= acc_dtype)
			up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype= acc_dtype)
			for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
				offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
				# Create pointers for lhs and rhs
				abs_row_indices = sub_group_start_idx + offsets_m
				lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
				up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
				
				# Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
				# Find out which scale block this tile is on:
				scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
				gate_scale_ptr = gate_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				up_scale_ptr = up_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				# Load the scale for this tile
				gate_scale = tl.load(gate_scale_ptr)
				up_scale = tl.load(up_scale_ptr)

				lhs_scale_k = k_idx * GEMM_BLOCK_SIZE_K // 128
				l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
				lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0, cache_modifier='.cg')

				# Create masks for lhs and rhs
				lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
				rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

				# Load rhs tile:
				gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0, cache_modifier='.cg')

				# gate_fp32 = tl.cast(gate_fp8, tl.float32)
				# gate_scaled = gate_fp32 * gate_scale
				# gate_bf16 = tl.cast(gate_scaled, lhs_dtype)
				gate_acc += tl.dot(lhs, tl.trans(gate_fp8)) * lhs_scale * gate_scale

				# up_fp32 = tl.cast(up_fp8, tl.float32)
				# up_scaled = up_fp32 * up_scale
				# up_bf16 = tl.cast(up_scaled, lhs_dtype)
				# up_acc += tl.dot(lhs, tl.trans(up_bf16))
				up_acc += tl.dot(lhs, tl.trans(up_fp8)) * lhs_scale * up_scale

			# Store the result
			offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
			# output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
			output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)


			# Convert to bf16 before storing
			output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc 
			# output_acc = gate_acc * up_acc
			output = tl.cast(output_acc, lhs_dtype)
			tl.store(output_ptrs, output, mask=output_mask)

@triton.jit
def fused_fp8_moe_stage_1_kernel_v2(
	lhs_ptr, lhs_scale_ptr,
	gate_ptrs_ptr, up_ptrs_ptr,
	gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	num_active_experts_ptr,  # <-- ADD THIS: pointer to num_active tensor
	output_ptr,
	M, N: tl.constexpr, K: tl.constexpr, 
	# REMOVE: num_groups parameter (we'll load it from device)
	stride_lhs_m, stride_lhs_k,
	stride_lhs_scale_m, stride_lhs_scale_k,
	stride_gate_n, stride_gate_k,
	stride_up_n, stride_up_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices,
	stride_weight_ptrs, stride_scale_ptrs,

	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	"""
	FP8 MoE Stage 1: act(hidden_states @ deq(gate)) * (hidden_states @ deq(up))
	
	Correctly handles GEMM tiles that span multiple quantization blocks.
	GEMM_BLOCK_SIZE_K can be > SCALE_BLOCK_SIZE_K (e.g., 256 vs 128).
	"""
	pid = tl.program_id(axis=0)
	lhs_dtype = tl.bfloat16
	rhs_dtype = tl.float8e4nv
	scale_dtype = tl.float32
	acc_dtype = tl.float32

	offsets_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
	scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
	num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
	
	# Calculate how many quantization blocks fit in one GEMM block
	num_quant_blocks_per_gemm = tl.cdiv(GEMM_BLOCK_SIZE_K, SCALE_BLOCK_SIZE_K)
	
	# ===== LOAD NUM_GROUPS FROM DEVICE (NO CPU-GPU SYNC!) =====
	num_groups = tl.load(num_active_experts_ptr)
	
	for g in range(num_groups):
		# Get group size: gm
		gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
		group_idx = tl.load(group_idx_ptr + g * stride_group_idx)
		# Get row indices for the current group
		start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
		num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)

		# Load expert weight pointers
		gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
		up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
		gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))
		up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))

		for sub_group_idx in range(num_sub_groups):
			# Calculate the base pointer for the current sub-group
			sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
			# Remaining rows in the group:
			remaining_rows_in_group = start_idx + gm - sub_group_start_idx
			valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)

			# Process the associated tile
			offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
			abs_row_indices = sub_group_start_idx + offsets_m

			# Initialize accumulators
			gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
			up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
			
			# ===== OUTER LOOP: Iterate over GEMM tiles in K dimension =====
			num_gemm_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
			for gemm_k_idx in range(num_gemm_k_blocks):
				gemm_k_start = gemm_k_idx * GEMM_BLOCK_SIZE_K
				
				# ===== INNER LOOP: Iterate over quantization sub-blocks within this GEMM tile =====
				for quant_sub_idx in range(num_quant_blocks_per_gemm):
					# Calculate K range for this quantization sub-block
					sub_k_start = gemm_k_start + quant_sub_idx * SCALE_BLOCK_SIZE_K
					
					# CRITICAL: Use masking instead of break
					# This sub-block is valid only if sub_k_start < K
					sub_block_valid = sub_k_start < K
					
					# Offsets for this sub-block (aligned to quantization boundaries)
					offsets_k = sub_k_start + tl.arange(0, SCALE_BLOCK_SIZE_K)
					
					# ===== LOAD SCALES (one scale per quantization block) =====
					# Weight scales: which quantization block in K dimension?
					scale_k = sub_k_start // SCALE_BLOCK_SIZE_K
					gate_scale_ptr = gate_scale_base_ptr + (scale_n * num_scale_k + scale_k)
					up_scale_ptr = up_scale_base_ptr + (scale_n * num_scale_k + scale_k)
					gate_scale = tl.load(gate_scale_ptr)
					up_scale = tl.load(up_scale_ptr)
					
					# LHS scales: per-row quantization along K
					lhs_scale_k = sub_k_start // SCALE_BLOCK_SIZE_K
					l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
					lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0, cache_modifier='.cg')
					
					# ===== LOAD DATA =====
					# Create pointers for lhs and rhs
					lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
					gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
					up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
					
					# Create masks - include sub_block_valid check
					k_mask = (offsets_k < K) & sub_block_valid
					lhs_mask = (abs_row_indices[:, None] < M) & k_mask[None, :] & (offsets_m[:, None] < valid_rows_this_block)
					rhs_mask = (offsets_n[:, None] < N) & k_mask[None, :]
					
					# Load FP8 data
					lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0, cache_modifier='.cg')
					gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
					up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
					
					# ===== COMPUTE AND ACCUMULATE =====
					# Each sub-block uses its own scale
					# When sub_block_valid is False, loaded values are 0.0, so contribution is 0
					gate_acc += tl.dot(lhs, tl.trans(gate_fp8)) * lhs_scale * gate_scale
					up_acc += tl.dot(lhs, tl.trans(up_fp8)) * lhs_scale * up_scale

			# ===== APPLY ACTIVATION AND STORE =====
			# SiLU activation: x / (1 + exp(-x))
			output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
			output = tl.cast(output_acc, lhs_dtype)
			
			# Store output
			offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
			output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)
			
			tl.store(output_ptrs, output, mask=output_mask)


# @torch.inference_mode()
# def fused_fp8_moe_stage_1(
# 	hidden_states: torch.Tensor,
# 	hidden_states_scale: torch.Tensor,
# 	gate_weight_list: list[torch.Tensor],
# 	gate_ptrs_ptr: torch.Tensor,
# 	up_weight_list: list[torch.Tensor],
# 	up_ptrs_ptr: torch.Tensor,
# 	gate_scale_list: list[torch.Tensor],
# 	gate_scale_ptrs_ptr: torch.Tensor,
# 	up_scale_list: list[torch.Tensor],
# 	up_scale_ptrs_ptr: torch.Tensor,
# 	group_sizes: torch.Tensor,
# 	activated_group_idx: torch.Tensor,
# 	group_start_indices: torch.Tensor,
# 	gate_gemm_block_size=[64,16,256],
# 	up_gemm_block_size=[64,16,128],
# 	scale_block_size=[128,128],
# 	num_stages = 2,
# 	num_warps = 4
# ):
# 	# assert hidden_states.dtype == torch.float8_e4m3fn, "hidden_states must be of dtype float8_e4m3fn"
# 	# assert hidden_states_scale.dtype == torch.float32, "hiden_states_scale must be of dtype float32"
# 	# assert hidden_states.shape[0] == hidden_states_scale.shape[0], "hidden_states and hiden_states_scale must have the same batch size"
# 	# assert hidden_states_scale.shape[1] == hidden_states.shape[1] // 128, "hiden_states_scale must have the same number of columns as hidden_states divided by 128"
# 	# assert all(r.dtype == torch.float8_e4m3fn for r in gate_weight_list), "All gate weights must be of dtype float8_e4m3fn"
# 	# assert all(r.dtype == torch.float8_e4m3fn for r in up_weight_list), "All up weights must be of dtype float8_e4m3fn"
# 	# assert all(s.dtype == torch.float32 for s in gate_scale_list), "All gate scales must be of dtype float32"
# 	# assert all(s.dtype == torch.float32 for s in up_scale_list), "All up scales must be of dtype float32"
# 	# assert len(gate_weight_list) == len(gate_scale_list), "gate_weight_list and gate_scale_list must have the same length"
# 	# assert len(up_weight_list) == len(up_scale_list), "up_weight_list and up_scale_list must have the same length"

# 	device = hidden_states.device
# 	M = hidden_states.shape[0]
# 	N = gate_weight_list[0].shape[0]
# 	K = hidden_states.shape[1]

# 	# gate_ptrs_ptr = torch.tensor([r.data_ptr() for r in gate_weight_list], dtype=torch.int64, device=device)
# 	# up_ptrs_ptr = torch.tensor([r.data_ptr() for r in up_weight_list], dtype=torch.int64, device=device)
# 	# gate_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in gate_scale_list], dtype=torch.int64, device=device)
# 	# up_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in up_scale_list], dtype=torch.int64, device=device)
# 	# group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
# 	# activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
# 	num_groups = group_sizes.shape[0]

# 	output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
# 	# TMA descriptors require a global memory allocation
# 	grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
# 	# Launch the kernel
# 	try:
# 		fused_fp8_moe_stage_1_kernel_v2[grid](
# 			hidden_states, hidden_states_scale,
# 			gate_ptrs_ptr, up_ptrs_ptr,
# 			gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
# 			activated_group_idx, group_sizes, group_start_indices,
# 			output,
# 			M, N, K, num_groups,
# 			hidden_states.stride(0), hidden_states.stride(1),
# 			hidden_states_scale.stride(0), hidden_states_scale.stride(1),
# 			gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
# 			up_weight_list[0].stride(0), up_weight_list[0].stride(1),
# 			output.stride(0), output.stride(1),
# 			activated_group_idx.stride(0), group_sizes.stride(0), group_start_indices.stride(0),
# 			gate_ptrs_ptr.stride(0), gate_scale_ptrs_ptr.stride(0),

# 			gate_gemm_block_size[0], gate_gemm_block_size[1], gate_gemm_block_size[2],
# 			SCALE_BLOCK_SIZE_N = scale_block_size[0],
# 			SCALE_BLOCK_SIZE_K = scale_block_size[1],
# 			num_stages=num_stages,
# 			num_warps=num_warps
# 		)
# 	except Exception as e:
# 		logging.error(f"Error in fused_dequant_weighted_moe_stage_1: {e}")
# 		raise
# 	return output

# @torch.inference_mode()
# def fused_fp8_moe_stage_1(
# 	hidden_states: torch.Tensor,
# 	hidden_states_scale: torch.Tensor,
# 	gate_weight_list: list[torch.Tensor],
# 	gate_ptrs_ptr: torch.Tensor,
# 	up_weight_list: list[torch.Tensor],
# 	up_ptrs_ptr: torch.Tensor,
# 	gate_scale_list: list[torch.Tensor],
# 	gate_scale_ptrs_ptr: torch.Tensor,
# 	up_scale_list: list[torch.Tensor],
# 	up_scale_ptrs_ptr: torch.Tensor,
# 	group_sizes: torch.Tensor,  # <-- Now full-sized (experts_per_rank)
# 	activated_group_idx: torch.Tensor,  # <-- Now full-sized (experts_per_rank)
# 	group_start_indices: torch.Tensor,  # <-- Now full-sized (experts_per_rank)
# 	num_active_experts: torch.Tensor,  # <-- ADD THIS: [1] tensor containing count
# 	gate_gemm_block_size=[64,16,256],
# 	up_gemm_block_size=[64,16,128],
# 	scale_block_size=[128,128],
# 	num_stages = 2,
# 	num_warps = 4
# ):
# 	device = hidden_states.device
# 	M = hidden_states.shape[0]
# 	N = gate_weight_list[0].shape[0]
# 	K = hidden_states.shape[1]

# 	# NO LONGER NEEDED: num_groups = group_sizes.shape[0]

# 	output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
# 	grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
	
# 	try:
# 		fused_fp8_moe_stage_1_kernel_optimized[grid](
# 			hidden_states, hidden_states_scale,
# 			gate_ptrs_ptr, up_ptrs_ptr,
# 			gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
# 			activated_group_idx, group_sizes, group_start_indices,
# 			num_active_experts,  # <-- Pass the tensor (NOT .item()!)
# 			output,
# 			M, N, K,  # <-- Remove num_groups from here
# 			hidden_states.stride(0), hidden_states.stride(1),
# 			hidden_states_scale.stride(0), hidden_states_scale.stride(1),
# 			gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
# 			up_weight_list[0].stride(0), up_weight_list[0].stride(1),
# 			output.stride(0), output.stride(1),
# 			activated_group_idx.stride(0), group_sizes.stride(0), group_start_indices.stride(0),
# 			gate_ptrs_ptr.stride(0), gate_scale_ptrs_ptr.stride(0),

# 			gate_gemm_block_size[0], gate_gemm_block_size[1], gate_gemm_block_size[2],
# 			SCALE_BLOCK_SIZE_N = scale_block_size[0],
# 			SCALE_BLOCK_SIZE_K = scale_block_size[1],
# 			num_stages=num_stages,
# 			num_warps=num_warps
# 		)
# 	except Exception as e:
# 		logging.error(f"Error in fused_dequant_weighted_moe_stage_1: {e}")
# 		raise
# 	return output


@torch.inference_mode()
def fused_fp8_moe_stage_1_optimized(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gate_weight_list: list[torch.Tensor],
    gate_ptrs_ptr: torch.Tensor,
    up_weight_list: list[torch.Tensor],
    up_ptrs_ptr: torch.Tensor,
    gate_scale_list: list[torch.Tensor],
    gate_scale_ptrs_ptr: torch.Tensor,
    up_scale_list: list[torch.Tensor],
    up_scale_ptrs_ptr: torch.Tensor,
    group_sizes: torch.Tensor,
    activated_group_idx: torch.Tensor,
    group_start_indices: torch.Tensor,
    num_active_experts: torch.Tensor,
    gate_gemm_block_size=[64, 16, 256],  # Using 256 for unrolling
    scale_block_size=[128, 128],
    num_stages=3,  # More stages for pipelining
    num_warps=4
):
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]

    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
    
    try:
        fused_fp8_moe_stage_1_kernel_optimized[grid](
            hidden_states, hidden_states_scale,
            gate_ptrs_ptr, up_ptrs_ptr,
            gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
            activated_group_idx, group_sizes, group_start_indices,
            num_active_experts,
            output,
            M, N, K,  # ← These become tl.constexpr in the kernel
            hidden_states.stride(0), hidden_states.stride(1),
            hidden_states_scale.stride(0), hidden_states_scale.stride(1),
            gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
            up_weight_list[0].stride(0), up_weight_list[0].stride(1),
            output.stride(0), output.stride(1),
            activated_group_idx.stride(0), 
            group_sizes.stride(0), 
            group_start_indices.stride(0),
            gate_ptrs_ptr.stride(0), 
            gate_scale_ptrs_ptr.stride(0),
            # GEMM config
            GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
            GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
            GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],  # 256
            SCALE_BLOCK_SIZE_N=scale_block_size[0],     # 128
            SCALE_BLOCK_SIZE_K=scale_block_size[1],     # 128
            num_stages=num_stages,
            num_warps=num_warps
        )
    except Exception as e:
        logging.error(f"Error in fused_fp8_moe_stage_1_optimized: {e}")
        raise
    
    return output

@torch.inference_mode()
def fused_fp8_moe_stage_1_bak(
	hidden_states: torch.Tensor,
	hidden_states_scale: torch.Tensor,
	gate_weight_list: list[torch.Tensor],
	up_weight_list: list[torch.Tensor],
	gate_scale_list: list[torch.Tensor],
	up_scale_list: list[torch.Tensor],
	group_sizes: tuple[int, int],
	group_start_indices: torch.Tensor,
	gate_gemm_block_size=[64,16,256],
	up_gemm_block_size=[64,16,128],
	scale_block_size=[128,128],
	num_stages = 2,
	num_warps = 4
):
	assert hidden_states.dtype == torch.float8_e4m3fn, "hidden_states must be of dtype float8_e4m3fn"
	assert hidden_states_scale.dtype == torch.float32, "hiden_states_scale must be of dtype float32"
	assert hidden_states.shape[0] == hidden_states_scale.shape[0], "hidden_states and hiden_states_scale must have the same batch size"
	assert hidden_states_scale.shape[1] == hidden_states.shape[1] // 128, "hiden_states_scale must have the same number of columns as hidden_states divided by 128"
	assert all(r.dtype == torch.float8_e4m3fn for r in gate_weight_list), "All gate weights must be of dtype float8_e4m3fn"
	assert all(r.dtype == torch.float8_e4m3fn for r in up_weight_list), "All up weights must be of dtype float8_e4m3fn"
	assert all(s.dtype == torch.float32 for s in gate_scale_list), "All gate scales must be of dtype float32"
	assert all(s.dtype == torch.float32 for s in up_scale_list), "All up scales must be of dtype float32"
	assert len(gate_weight_list) == len(gate_scale_list), "gate_weight_list and gate_scale_list must have the same length"
	assert len(up_weight_list) == len(up_scale_list), "up_weight_list and up_scale_list must have the same length"

	device = hidden_states.device
	M = hidden_states.shape[0]
	N = gate_weight_list[0].shape[0]
	K = hidden_states.shape[1]

	gate_ptrs_ptr = torch.tensor([r.data_ptr() for r in gate_weight_list], dtype=torch.int64, device=device)
	up_ptrs_ptr = torch.tensor([r.data_ptr() for r in up_weight_list], dtype=torch.int64, device=device)
	gate_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in gate_scale_list], dtype=torch.int64, device=device)
	up_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in up_scale_list], dtype=torch.int64, device=device)
	group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
	activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
	num_groups = len(group_sizes)

	output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
	# TMA descriptors require a global memory allocation
	# def alloc_fn(size: int, alignment: int, stream):
	# 	return torch.empty(size, device="cuda:0", dtype=torch.float8_e4m3fn)

	# triton.set_allocator(alloc_fn)
	grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
	# Launch the kernel
	try:
		fused_fp8_moe_stage_1_kernel[grid](
			hidden_states, hidden_states_scale,
			gate_ptrs_ptr, up_ptrs_ptr,
			gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
			activated_group_idx, group_size, group_start_indices,
			output,
			M, N, K, num_groups,
			hidden_states.stride(0), hidden_states.stride(1),
			hidden_states_scale.stride(0), hidden_states_scale.stride(1),
			gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
			up_weight_list[0].stride(0), up_weight_list[0].stride(1),
			output.stride(0), output.stride(1),
			activated_group_idx.stride(0), group_size.stride(0), group_start_indices.stride(0),
			gate_ptrs_ptr.stride(0), gate_scale_ptrs_ptr.stride(0),

			gate_gemm_block_size[0], gate_gemm_block_size[1], gate_gemm_block_size[2],
			SCALE_BLOCK_SIZE_N = scale_block_size[0],
			SCALE_BLOCK_SIZE_K = scale_block_size[1],
			num_stages=num_stages,
			num_warps=num_warps
		)
	except Exception as e:
		logging.error(f"Error in fused_dequant_weighted_moe_stage_1: {e}")
		raise
	return output


# Autotune configuration for the fused FP8 MoE kernel
# @triton.autotune(
# 	configs=[
# 		triton.Config({'GEMM_BLOCK_SIZE_M': m, 'GEMM_BLOCK_SIZE_N': n, 'GEMM_BLOCK_SIZE_K': k}, 
# 					 num_stages=stages, num_warps=warps)
# 		for m in [16, 32, 64, 128]
# 		for n in [16, 32, 64, 128] 
# 		for k in [32, 64, 128]
# 		for stages in [2, 3, 4]
# 		for warps in [2, 4, 8]
# 	],
# 	key=['N', 'K'],  # Autotuning keys based on problem size
# 	warmup=25,       # Number of warmup iterations
# 	rep=10,         # Number of benchmark repetitions
# 	# cache_results=True # Cache results for faster subsequent runs
# )
# @triton.jit
# def fused_fp8_moe_stage_1_kernel_optimized(
# 	lhs_ptr, lhs_scale_ptr,
# 	gate_ptrs_ptr, up_ptrs_ptr,
# 	gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
# 	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
# 	output_ptr,
# 	M, N: tl.constexpr, K: tl.constexpr, num_groups,
# 	stride_lhs_m, stride_lhs_k,
# 	stride_lhs_scale_m, stride_lhs_scale_k,
# 	stride_gate_n, stride_gate_k,
# 	stride_up_n, stride_up_k,
# 	stride_output_m, stride_output_n,
# 	stride_group_idx, stride_group_sizes, stride_group_start_indices,
# 	stride_weight_ptrs, stride_scale_ptrs,
# 	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
# 	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
# ):
# 	"""
# 	Optimized fused FP8 MoE kernel: parallelize over (expert, N_block) pairs
# 	Formula: act(hidden_states @ dequant(gate)) * (hidden_states @ dequant(up))
# 	where act = SiLU
# 	"""
# 	# Get expert and N block indices from program IDs
# 	expert_id = tl.program_id(axis=0)
# 	n_block_id = tl.program_id(axis=1)
	
# 	# Data types
# 	lhs_dtype = tl.bfloat16
# 	rhs_dtype = tl.float8e4nv
# 	scale_dtype = tl.float32
# 	acc_dtype = tl.float32

# 	# Check if expert is valid (use masking instead of early return)
# 	expert_valid = expert_id < num_groups
	
# 	# Load expert metadata with masking
# 	gm = tl.load(group_sizes_ptr + expert_id * stride_group_sizes, mask=expert_valid, other=0)
# 	group_idx = tl.load(group_idx_ptr + expert_id * stride_group_idx, mask=expert_valid, other=0)
# 	start_idx = tl.load(group_start_indices_ptr + expert_id * stride_group_start_indices, mask=expert_valid, other=0)
	
# 	# Check if expert has tokens (combine with expert_valid)
# 	process_expert = expert_valid & (gm > 0)
	
# 	# Calculate N offsets for this CTA
# 	offsets_n = n_block_id * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
	
# 	# Load expert weight pointers - handle invalid experts by using dummy pointers
# 	gate_ptr_val = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs)
# 	up_ptr_val = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs)
# 	gate_scale_ptr_val = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs)
# 	up_scale_ptr_val = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs)
	
# 	gate_base_ptr = gate_ptr_val.to(tl.pointer_type(rhs_dtype))
# 	up_base_ptr = up_ptr_val.to(tl.pointer_type(rhs_dtype))
# 	gate_scale_base_ptr = gate_scale_ptr_val.to(tl.pointer_type(scale_dtype))
# 	up_scale_base_ptr = up_scale_ptr_val.to(tl.pointer_type(scale_dtype))
	
# 	# Scale calculations
# 	scale_n = n_block_id * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
# 	num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
	
# 	# Calculate maximum possible M blocks (use conservative upper bound)
# 	max_possible_m_blocks = tl.cdiv(M, GEMM_BLOCK_SIZE_M)
	
# 	# Use regular Python range since we need runtime values
# 	for m_block_idx in range(max_possible_m_blocks):
# 		# Calculate M range for this block
# 		m_start = start_idx + m_block_idx * GEMM_BLOCK_SIZE_M
# 		remaining_rows = start_idx + gm - m_start
# 		valid_m_rows = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows)
		
# 		# Check if this M block is valid for this expert
# 		actual_num_m_blocks = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
# 		m_block_valid = process_expert & (m_block_idx < actual_num_m_blocks) & (valid_m_rows > 0)
		
# 		offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
# 		abs_row_indices = m_start + offsets_m
		
# 		# Initialize accumulator
# 		gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
# 		up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
		
# 		# K dimension loop - use regular range since tl.cdiv doesn't preserve constexpr
# 		num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
# 		for k_block_idx in range(num_k_blocks):
# 			offsets_k = k_block_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
			
# 			# Load LHS
# 			lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
# 			lhs_mask = ((abs_row_indices[:, None] < M) & 
# 					   (offsets_k[None, :] < K) & 
# 					   (offsets_m[:, None] < valid_m_rows) &
# 					   m_block_valid)
# 			lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
			
# 			# Load LHS scales
# 			lhs_scale_k = k_block_idx * GEMM_BLOCK_SIZE_K // 128
# 			l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
# 										  lhs_scale_k * stride_lhs_scale_k)
# 			lhs_scale_mask = ((abs_row_indices[:, None] < M) & 
# 							 (offsets_m[:, None] < valid_m_rows) &
# 							 m_block_valid)
# 			lhs_scale = tl.load(l_scale_ptr, mask=lhs_scale_mask, other=1.0)
			
# 			# Load weight scales
# 			scale_k = k_block_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
# 			gate_scale_ptr = gate_scale_base_ptr + (scale_n * num_scale_k + scale_k)
# 			up_scale_ptr = up_scale_base_ptr + (scale_n * num_scale_k + scale_k)
# 			gate_scale = tl.load(gate_scale_ptr, mask=m_block_valid, other=1.0)
# 			up_scale = tl.load(up_scale_ptr, mask=m_block_valid, other=1.0)
			
# 			# Load gate and up weights
# 			gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
# 			up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
			
# 			rhs_mask = ((offsets_n[:, None] < N) & (offsets_k[None, :] < K) & m_block_valid)
# 			gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
# 			up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
			
# 			# Perform GEMM with scaling
# 			gate_acc += tl.dot(lhs, tl.trans(gate_fp8)) * lhs_scale * gate_scale
# 			up_acc += tl.dot(lhs, tl.trans(up_fp8)) * lhs_scale * up_scale
		
# 		# Apply SiLU activation and combine: gate * sigmoid(gate) * up
# 		# sigmoid(x) = 1 / (1 + exp(-x)), SiLU(x) = x * sigmoid(x)
# 		combined_result = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
		
# 		# Store results
# 		output_m_indices = abs_row_indices
# 		output_n_indices = offsets_n
		
# 		output_ptrs = output_ptr + (output_m_indices[:, None] * stride_output_m + 
# 								   output_n_indices[None, :] * stride_output_n)
		
# 		output_mask = ((output_m_indices[:, None] < M) & 
# 					  (output_n_indices[None, :] < N) & 
# 					  (offsets_m[:, None] < valid_m_rows) &
# 					  m_block_valid)
		
# 		output_bf16 = tl.cast(combined_result, lhs_dtype)
# 		tl.store(output_ptrs, output_bf16, mask=output_mask)


# @torch.inference_mode()
# def fused_fp8_moe_stage_1_optimized(
# 	hidden_states: torch.Tensor,
# 	hidden_states_scale: torch.Tensor,
# 	gate_weight_list: list[torch.Tensor],
# 	up_weight_list: list[torch.Tensor],
# 	gate_scale_list: list[torch.Tensor],
# 	up_scale_list: list[torch.Tensor],
# 	group_sizes: tuple[int, int],
# 	group_start_indices: torch.Tensor,
# 	gate_gemm_block_size=[64, 16, 128],
# 	up_gemm_block_size=[64, 16, 128],  # Kept for API compatibility
# 	scale_block_size=[128, 128]
# ):
# 	"""
# 	Optimized fused FP8 MoE computation with autotuning
# 	"""
# 	# Assertions
# 	assert hidden_states.dtype == torch.float8_e4m3fn, "hidden_states must be float8_e4m3fn"
# 	assert hidden_states_scale.dtype == torch.float32, "hidden_states_scale must be float32"
# 	assert hidden_states.shape[0] == hidden_states_scale.shape[0], "Batch size mismatch"
# 	assert hidden_states_scale.shape[1] == hidden_states.shape[1] // 128, "Scale shape mismatch"
# 	assert all(w.dtype == torch.float8_e4m3fn for w in gate_weight_list), "Gate weights must be float8_e4m3fn"
# 	assert all(w.dtype == torch.float8_e4m3fn for w in up_weight_list), "Up weights must be float8_e4m3fn"
# 	assert all(s.dtype == torch.float32 for s in gate_scale_list), "Gate scales must be float32"
# 	assert all(s.dtype == torch.float32 for s in up_scale_list), "Up scales must be float32"
# 	assert len(gate_weight_list) == len(gate_scale_list), "Gate weight/scale list length mismatch"
# 	assert len(up_weight_list) == len(up_scale_list), "Up weight/scale list length mismatch"
	
# 	device = hidden_states.device
# 	M = hidden_states.shape[0]
# 	N = gate_weight_list[0].shape[0]
# 	K = hidden_states.shape[1]
	
# 	# Prepare pointer arrays
# 	gate_ptrs_ptr = torch.tensor([w.data_ptr() for w in gate_weight_list], dtype=torch.int64, device=device)
# 	up_ptrs_ptr = torch.tensor([w.data_ptr() for w in up_weight_list], dtype=torch.int64, device=device)
# 	gate_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in gate_scale_list], dtype=torch.int64, device=device)
# 	up_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in up_scale_list], dtype=torch.int64, device=device)
	
# 	# Prepare group metadata
# 	group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
# 	activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
# 	num_groups = len(group_sizes)
	
# 	# Allocate output
# 	output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
	
# 	# The autotuned kernel will automatically select optimal block sizes
# 	# Launch configuration: parallelize over (expert, N_block) pairs
# 	# Note: grid[1] will be calculated based on the selected GEMM_BLOCK_SIZE_N
# 	def grid_fn(meta):
# 		return (num_groups, triton.cdiv(N, meta['GEMM_BLOCK_SIZE_N']))
	
# 	try:
# 		fused_fp8_moe_stage_1_kernel_optimized[grid_fn](
# 			hidden_states, hidden_states_scale,
# 			gate_ptrs_ptr, up_ptrs_ptr,
# 			gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
# 			activated_group_idx, group_size, group_start_indices,
# 			output,
# 			M, N, K, num_groups,
# 			# Strides
# 			hidden_states.stride(0), hidden_states.stride(1),
# 			hidden_states_scale.stride(0), hidden_states_scale.stride(1),
# 			gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
# 			up_weight_list[0].stride(0), up_weight_list[0].stride(1),
# 			output.stride(0), output.stride(1),
# 			activated_group_idx.stride(0), group_size.stride(0), group_start_indices.stride(0),
# 			gate_ptrs_ptr.stride(0), gate_scale_ptrs_ptr.stride(0),
# 			# Scale block sizes (these remain fixed)
# 			SCALE_BLOCK_SIZE_N=scale_block_size[0],
# 			SCALE_BLOCK_SIZE_K=scale_block_size[1],
# 		)
# 	except Exception as e:
# 		import logging
# 		logging.error(f"Error in optimized fused_fp8_moe_stage_1: {e}")
# 		raise
	
# 	return output

# @triton.jit
# def fused_fp8_moe_stage_1_kernel_optimized(
# 	lhs_ptr, lhs_scale_ptr,
# 	gate_ptrs_ptr, up_ptrs_ptr,
# 	gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
# 	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
# 	output_ptr,
# 	M, N: tl.constexpr, K: tl.constexpr, num_groups,
# 	stride_lhs_m, stride_lhs_k,
# 	stride_lhs_scale_m, stride_lhs_scale_k,
# 	stride_gate_n, stride_gate_k,
# 	stride_up_n, stride_up_k,
# 	stride_output_m, stride_output_n,
# 	stride_group_idx, stride_group_sizes, stride_group_start_indices,
# 	stride_weight_ptrs, stride_scale_ptrs,
# 	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
# 	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
# ):
# 	"""
# 	Optimized fused FP8 MoE kernel: parallelize over (expert, N_block) pairs
# 	Formula: act(hidden_states @ dequant(gate)) * (hidden_states @ dequant(up))
# 	where act = SiLU
# 	"""
# 	# Get expert and N block indices from program IDs
# 	expert_id = tl.program_id(axis=0)
# 	n_block_id = tl.program_id(axis=1)
	
# 	# Data types
# 	lhs_dtype = tl.bfloat16
# 	rhs_dtype = tl.float8e4nv
# 	scale_dtype = tl.float32
# 	acc_dtype = tl.float32

# 	# Check if expert is valid (use masking instead of early return)
# 	expert_valid = expert_id < num_groups
	
# 	# Load expert metadata with masking
# 	gm = tl.load(group_sizes_ptr + expert_id * stride_group_sizes, mask=expert_valid, other=0)
# 	group_idx = tl.load(group_idx_ptr + expert_id * stride_group_idx, mask=expert_valid, other=0)
# 	start_idx = tl.load(group_start_indices_ptr + expert_id * stride_group_start_indices, mask=expert_valid, other=0)
	
# 	# Check if expert has tokens (combine with expert_valid)
# 	process_expert = expert_valid & (gm > 0)
	
# 	# Calculate N offsets for this CTA
# 	offsets_n = n_block_id * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
	
# 	# Load expert weight pointers - handle invalid experts by using dummy pointers
# 	gate_ptr_val = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs)
# 	up_ptr_val = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs)
# 	gate_scale_ptr_val = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs)
# 	up_scale_ptr_val = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs)
	
# 	gate_base_ptr = gate_ptr_val.to(tl.pointer_type(rhs_dtype))
# 	up_base_ptr = up_ptr_val.to(tl.pointer_type(rhs_dtype))
# 	gate_scale_base_ptr = gate_scale_ptr_val.to(tl.pointer_type(scale_dtype))
# 	up_scale_base_ptr = up_scale_ptr_val.to(tl.pointer_type(scale_dtype))
	
# 	# Scale calculations
# 	scale_n = n_block_id * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
# 	num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
	
# 	# Calculate maximum possible M blocks (use conservative upper bound)
# 	max_possible_m_blocks = tl.cdiv(M, GEMM_BLOCK_SIZE_M)
	
# 	# Use regular Python range since we need runtime values
# 	for m_block_idx in range(max_possible_m_blocks):
# 		# Calculate M range for this block
# 		m_start = start_idx + m_block_idx * GEMM_BLOCK_SIZE_M
# 		remaining_rows = start_idx + gm - m_start
# 		valid_m_rows = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows)
		
# 		# Check if this M block is valid for this expert
# 		actual_num_m_blocks = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
# 		m_block_valid = process_expert & (m_block_idx < actual_num_m_blocks) & (valid_m_rows > 0)
		
# 		offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
# 		abs_row_indices = m_start + offsets_m
		
# 		# Initialize accumulator
# 		gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
# 		up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
		
# 		# K dimension loop - use regular range since tl.cdiv doesn't preserve constexpr
# 		num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
# 		for k_block_idx in range(num_k_blocks):
# 			offsets_k = k_block_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
			
# 			# Load LHS
# 			lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
# 			lhs_mask = ((abs_row_indices[:, None] < M) & 
# 					   (offsets_k[None, :] < K) & 
# 					   (offsets_m[:, None] < valid_m_rows) &
# 					   m_block_valid)
# 			lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
			
# 			# Load LHS scales
# 			lhs_scale_k = k_block_idx * GEMM_BLOCK_SIZE_K // 128
# 			l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
# 										  lhs_scale_k * stride_lhs_scale_k)
# 			lhs_scale_mask = ((abs_row_indices[:, None] < M) & 
# 							 (offsets_m[:, None] < valid_m_rows) &
# 							 m_block_valid)
# 			lhs_scale = tl.load(l_scale_ptr, mask=lhs_scale_mask, other=1.0)
			
# 			# Load weight scales
# 			scale_k = k_block_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
# 			gate_scale_ptr = gate_scale_base_ptr + (scale_n * num_scale_k + scale_k)
# 			up_scale_ptr = up_scale_base_ptr + (scale_n * num_scale_k + scale_k)
# 			gate_scale = tl.load(gate_scale_ptr, mask=m_block_valid, other=1.0)
# 			up_scale = tl.load(up_scale_ptr, mask=m_block_valid, other=1.0)
			
# 			# Load gate and up weights
# 			gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
# 			up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
			
# 			rhs_mask = ((offsets_n[:, None] < N) & (offsets_k[None, :] < K) & m_block_valid)
# 			gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
# 			up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
			
# 			# Perform GEMM with scaling
# 			gate_acc += tl.dot(lhs, tl.trans(gate_fp8)) * lhs_scale * gate_scale
# 			up_acc += tl.dot(lhs, tl.trans(up_fp8)) * lhs_scale * up_scale
		
# 		# Apply SiLU activation and combine: gate * sigmoid(gate) * up
# 		# sigmoid(x) = 1 / (1 + exp(-x)), SiLU(x) = x * sigmoid(x)
# 		combined_result = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
		
# 		# Store results
# 		output_m_indices = abs_row_indices
# 		output_n_indices = offsets_n
		
# 		output_ptrs = output_ptr + (output_m_indices[:, None] * stride_output_m + 
# 								   output_n_indices[None, :] * stride_output_n)
		
# 		output_mask = ((output_m_indices[:, None] < M) & 
# 					  (output_n_indices[None, :] < N) & 
# 					  (offsets_m[:, None] < valid_m_rows) &
# 					  m_block_valid)
		
# 		output_bf16 = tl.cast(combined_result, lhs_dtype)
# 		tl.store(output_ptrs, output_bf16, mask=output_mask)


# @torch.inference_mode()
# def fused_fp8_moe_stage_1_optimized(
# 	hidden_states: torch.Tensor,
# 	hidden_states_scale: torch.Tensor,
# 	gate_weight_list: list[torch.Tensor],
# 	up_weight_list: list[torch.Tensor],
# 	gate_scale_list: list[torch.Tensor],
# 	up_scale_list: list[torch.Tensor],
# 	group_sizes: tuple[int, int],
# 	group_start_indices: torch.Tensor,
# 	gate_gemm_block_size=[64, 16, 128],
# 	up_gemm_block_size=[64, 16, 128],  # Kept for API compatibility
# 	scale_block_size=[128, 128]
# ):
# 	"""
# 	Optimized fused FP8 MoE computation
# 	"""
# 	# Assertions
# 	assert hidden_states.dtype == torch.float8_e4m3fn, "hidden_states must be float8_e4m3fn"
# 	assert hidden_states_scale.dtype == torch.float32, "hidden_states_scale must be float32"
# 	assert hidden_states.shape[0] == hidden_states_scale.shape[0], "Batch size mismatch"
# 	assert hidden_states_scale.shape[1] == hidden_states.shape[1] // 128, "Scale shape mismatch"
# 	assert all(w.dtype == torch.float8_e4m3fn for w in gate_weight_list), "Gate weights must be float8_e4m3fn"
# 	assert all(w.dtype == torch.float8_e4m3fn for w in up_weight_list), "Up weights must be float8_e4m3fn"
# 	assert all(s.dtype == torch.float32 for s in gate_scale_list), "Gate scales must be float32"
# 	assert all(s.dtype == torch.float32 for s in up_scale_list), "Up scales must be float32"
# 	assert len(gate_weight_list) == len(gate_scale_list), "Gate weight/scale list length mismatch"
# 	assert len(up_weight_list) == len(up_scale_list), "Up weight/scale list length mismatch"
	
# 	device = hidden_states.device
# 	M = hidden_states.shape[0]
# 	N = gate_weight_list[0].shape[0]
# 	K = hidden_states.shape[1]
	
# 	# Prepare pointer arrays
# 	gate_ptrs_ptr = torch.tensor([w.data_ptr() for w in gate_weight_list], dtype=torch.int64, device=device)
# 	up_ptrs_ptr = torch.tensor([w.data_ptr() for w in up_weight_list], dtype=torch.int64, device=device)
# 	gate_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in gate_scale_list], dtype=torch.int64, device=device)
# 	up_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in up_scale_list], dtype=torch.int64, device=device)
	
# 	# Prepare group metadata
# 	group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
# 	activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
# 	num_groups = len(group_sizes)
	
# 	# Allocate output
# 	output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
	
# 	# Launch configuration: parallelize over (expert, N_block) pairs
# 	grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
	
# 	try:
# 		fused_fp8_moe_stage_1_kernel_optimized[grid](
# 			hidden_states, hidden_states_scale,
# 			gate_ptrs_ptr, up_ptrs_ptr,
# 			gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
# 			activated_group_idx, group_size, group_start_indices,
# 			output,
# 			M, N, K, num_groups,
# 			# Strides
# 			hidden_states.stride(0), hidden_states.stride(1),
# 			hidden_states_scale.stride(0), hidden_states_scale.stride(1),
# 			gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
# 			up_weight_list[0].stride(0), up_weight_list[0].stride(1),
# 			output.stride(0), output.stride(1),
# 			activated_group_idx.stride(0), group_size.stride(0), group_start_indices.stride(0),
# 			gate_ptrs_ptr.stride(0), gate_scale_ptrs_ptr.stride(0),
# 			# Block sizes
# 			GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
# 			GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1], 
# 			GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
# 			SCALE_BLOCK_SIZE_N=scale_block_size[0],
# 			SCALE_BLOCK_SIZE_K=scale_block_size[1],
# 		)
# 	except Exception as e:
# 		logging.error(f"Error in optimized fused_fp8_moe_stage_1: {e}")
# 		raise
	
# 	return output


@triton.jit
def fused_fp8_moe_stage_1_kernel(
	lhs_ptr, lhs_scale_ptr,
	gate_ptrs_ptr, up_ptrs_ptr,
	gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	output_ptr,
	M, N: tl.constexpr, K: tl.constexpr, num_groups,
	stride_lhs_m, stride_lhs_k,
	stride_lhs_scale_m, stride_lhs_scale_k,
	stride_gate_n, stride_gate_k,
	stride_up_n, stride_up_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices,
	stride_weight_ptrs, stride_scale_ptrs,

	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	"""
		# (act(hidden_states @ deq(gate)) * (hidden_states @ deq(up))
		Note: act - silu. And we assume the gate and up weights have the same shape which is common in MoE models.
	"""
	pid = tl.program_id(axis=0)
	lhs_dtype = tl.bfloat16
	rhs_dtype = tl.float8e4nv
	scale_dtype = tl.float32
	acc_dtype = tl.float32


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
		gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
		up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
		gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))
		up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))

		for sub_group_idx in range(num_sub_groups):
			# Calculate the base pointer for the current sub-group
			sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
			# Remaining rows in the group:
			remaining_rows_in_group = start_idx + gm - sub_group_start_idx
			valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)

			# # Process the associated tile
			offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)

			# Loop along K dimension
			gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype= acc_dtype)
			up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype= acc_dtype)
			for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
				offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
				# Create pointers for lhs and rhs
				abs_row_indices = sub_group_start_idx + offsets_m
				lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
				gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
				up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
				
				# Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
				# Find out which scale block this tile is on:
				scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
				gate_scale_ptr = gate_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				up_scale_ptr = up_scale_base_ptr + (scale_n * num_scale_k + scale_k)
				# Load the scale for this tile
				gate_scale = tl.load(gate_scale_ptr)
				up_scale = tl.load(up_scale_ptr)

				lhs_scale_k = k_idx * GEMM_BLOCK_SIZE_K // 128
				l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
				lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0, cache_modifier='.cg')

				# Create masks for lhs and rhs
				lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
				rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

				# Load rhs tile:
				gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
				lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0, cache_modifier='.cg')

				# gate_fp32 = tl.cast(gate_fp8, tl.float32)
				# gate_scaled = gate_fp32 * gate_scale
				# gate_bf16 = tl.cast(gate_scaled, lhs_dtype)
				gate_acc += tl.dot(lhs, tl.trans(gate_fp8)) * lhs_scale * gate_scale

				# up_fp32 = tl.cast(up_fp8, tl.float32)
				# up_scaled = up_fp32 * up_scale
				# up_bf16 = tl.cast(up_scaled, lhs_dtype)
				# up_acc += tl.dot(lhs, tl.trans(up_bf16))
				up_acc += tl.dot(lhs, tl.trans(up_fp8)) * lhs_scale * up_scale

			# Store the result
			offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
			offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
			output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
			# output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
			output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)


			# Convert to bf16 before storing
			output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc 
			output = tl.cast(output_acc, lhs_dtype)
			tl.store(output_ptrs, output, mask=output_mask)


@triton.jit
def fused_fp8_moe_stage_1_v2_kernel(
	lhs_ptr, lhs_scale_ptr,
	gate_ptrs_ptr, up_ptrs_ptr,
	gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
	group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
	output_ptr,
	M, N: tl.constexpr, K: tl.constexpr, num_groups,
	stride_lhs_m, stride_lhs_k,
	stride_lhs_scale_m, stride_lhs_scale_k,
	stride_gate_n, stride_gate_k,
	stride_up_n, stride_up_k,
	stride_output_m, stride_output_n,
	stride_group_idx, stride_group_sizes, stride_group_start_indices,
	stride_weight_ptrs, stride_scale_ptrs,

	GEMM_BLOCK_SIZE_M: tl.constexpr, GEMM_BLOCK_SIZE_N: tl.constexpr, GEMM_BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_SIZE_N: tl.constexpr, SCALE_BLOCK_SIZE_K: tl.constexpr
):
	tile_idx = tl.program_id(axis=0)
	num_ctas = tl.num_programs(axis=0)
	dtype = tl.float8e4nv
	scale_dtype = tl.float32
	acc_dtype = tl.float32
	last_problem_end = 0
	num_n_tiles = tl.cdiv(N, GEMM_BLOCK_SIZE_N)
	for g in range(num_groups):
		# Get the gemm size for the current group
		gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
		group_idx = tl.load(group_idx_ptr + g * stride_group_idx)
		start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
		num_m_tiles = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
		num_tiles = num_m_tiles * num_n_tiles

		lhs_group_ptr = lhs_ptr + (start_idx * stride_lhs_m)
		gate_group_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(dtype))
		up_group_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(dtype))
		if(tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
			lhs_desc = tl.make_tensor_descriptor(
				lhs_group_ptr,
				shape=[gm, K],
				strides=[stride_lhs_m, stride_lhs_k],
				block_size=[GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_K],
			)
			gate_desc = tl.make_tensor_descriptor(
				gate_group_ptr,
				shape=[N, K],
				strides=[stride_gate_n, stride_gate_k],
				block_size=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K],
			)
			up_desc = tl.make_tensor_descriptor(
				up_group_ptr,
				shape=[N, K],
				strides=[stride_up_n, stride_up_k],
				block_size=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K],
			)
			# iterate through the tiles in the current gemm problem.
			while(tile_idx < last_problem_end and tile_idx < last_problem_end + num_tiles):
				# Calculate the tile indices
				tile_idx_in_gemm = tile_idx - last_problem_end
				tile_m_idx = tile_idx_in_gemm // num_n_tiles
				tile_n_idx = tile_idx_in_gemm % num_n_tiles
				
				# Calculate the absolute row indices for this tile
				offs_am = tile_m_idx * GEMM_BLOCK_SIZE_M
				offs_bn = tile_n_idx * GEMM_BLOCK_SIZE_N

				gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype= acc_dtype)
				up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype= acc_dtype)
				# Iterate through the K dimension in blocks
				for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
					# lhs = lhs_desc.load([offs_am, k_idx * GEMM_BLOCK_SIZE_K], mask=(offs_am < gm) & (k_idx * GEMM_BLOCK_SIZE_K < K))
					lhs = lhs_desc.load([offs_am, k_idx * GEMM_BLOCK_SIZE_K])
					gate = gate_desc.load([offs_bn, k_idx * GEMM_BLOCK_SIZE_K])
					up = up_desc.load([offs_bn, k_idx * GEMM_BLOCK_SIZE_K])
					
					gate_acc += tl.dot(lhs, tl.trans(gate)) 
					up_acc += tl.dot(lhs, tl.trans(up))
				# Apply SiLU activation and combine: gate * sigmoid(gate) * up
				# sigmoid(x) = 1 / (1 + exp(-x)), SiLU(x) = x * sigmoid(x)
				output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc 
				output = tl.cast(output_acc, dtype)

				# Store the result
				offs_output_m = start_idx + offs_am
				offs_output_n = tile_n_idx * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
				output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
				output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < tl.minimum(GEMM_BLOCK_SIZE_M, gm - offs_output_m[:, None]))
				tl.store(output_ptrs, output, mask=output_mask)

				tile_idx += num_ctas
			last_problem_end += num_tiles



				
@triton.jit
def fused_fp8_moe_stage_1_kernel_optimized(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,  # All must be tl.constexpr!
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr, 
    GEMM_BLOCK_SIZE_N: tl.constexpr, 
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr, 
    SCALE_BLOCK_SIZE_K: tl.constexpr
):
    """
    Optimized FP8 MoE with unrolled quantization block handling.
    Handles GEMM_BLOCK_SIZE_K = 2 * SCALE_BLOCK_SIZE_K (e.g., 256 vs 128).
    """
    pid = tl.program_id(axis=0)
    lhs_dtype = tl.bfloat16
    rhs_dtype = tl.float8e4nv
    scale_dtype = tl.float32
    acc_dtype = tl.float32

    # Pre-compute block indices
    offsets_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    
    # Compile-time constants - these are all tl.constexpr
    num_scale_k: tl.constexpr = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    num_gemm_k_blocks: tl.constexpr = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
    
    # Load number of active groups from device
    num_groups = tl.load(num_active_experts_ptr)
    
    for g in range(num_groups):
        gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
        group_idx = tl.load(group_idx_ptr + g * stride_group_idx)
        start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
        
        # Load expert weight pointers
        gate_ptr_val = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs)
        up_ptr_val = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs)
        gate_scale_ptr_val = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs)
        up_scale_ptr_val = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs)
        
        gate_base_ptr = gate_ptr_val.to(tl.pointer_type(rhs_dtype))
        up_base_ptr = up_ptr_val.to(tl.pointer_type(rhs_dtype))
        gate_scale_base_ptr = gate_scale_ptr_val.to(tl.pointer_type(scale_dtype))
        up_scale_base_ptr = up_scale_ptr_val.to(tl.pointer_type(scale_dtype))

        # Use runtime division for group-specific calculations
        num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)

        for sub_group_idx in range(num_sub_groups):
            sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
            remaining_rows = start_idx + gm - sub_group_start_idx
            valid_rows = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows)

            offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
            abs_row_indices = sub_group_start_idx + offsets_m

            # Initialize accumulators
            gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
            up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
            
            # Outer loop over GEMM K blocks
            for gemm_k_idx in range(num_gemm_k_blocks):
                gemm_k_start = gemm_k_idx * GEMM_BLOCK_SIZE_K
                
                # ========== FIRST QUANTIZATION BLOCK ==========
                k_start_0 = gemm_k_start
                offsets_k_0 = k_start_0 + tl.arange(0, SCALE_BLOCK_SIZE_K)
                
                # Load scales for first block
                scale_k_0 = gemm_k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
                gate_scale_0 = tl.load(gate_scale_base_ptr + scale_n * num_scale_k + scale_k_0)
                up_scale_0 = tl.load(up_scale_base_ptr + scale_n * num_scale_k + scale_k_0)
                
                lhs_scale_k_0 = gemm_k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
                lhs_scale_ptr_0 = lhs_scale_ptr + abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k_0 * stride_lhs_scale_k
                lhs_scale_0 = tl.load(lhs_scale_ptr_0, mask=(abs_row_indices[:, None] < M), other=1.0)
                
                # Load data for first block
                lhs_ptrs_0 = lhs_ptr + abs_row_indices[:, None] * stride_lhs_m + offsets_k_0[None, :] * stride_lhs_k
                gate_ptrs_0 = gate_base_ptr + offsets_n[:, None] * stride_gate_n + offsets_k_0[None, :] * stride_gate_k
                up_ptrs_0 = up_base_ptr + offsets_n[:, None] * stride_up_n + offsets_k_0[None, :] * stride_up_k
                
                lhs_mask_0 = (abs_row_indices[:, None] < M) & (offsets_k_0[None, :] < K) & (offsets_m[:, None] < valid_rows)
                rhs_mask_0 = (offsets_n[:, None] < N) & (offsets_k_0[None, :] < K)
                
                lhs_0 = tl.load(lhs_ptrs_0, mask=lhs_mask_0, other=0.0)
                gate_fp8_0 = tl.load(gate_ptrs_0, mask=rhs_mask_0, other=0.0)
                up_fp8_0 = tl.load(up_ptrs_0, mask=rhs_mask_0, other=0.0)
                
                # Accumulate first block
                gate_acc += tl.dot(lhs_0, tl.trans(gate_fp8_0)) * lhs_scale_0 * gate_scale_0
                up_acc += tl.dot(lhs_0, tl.trans(up_fp8_0)) * lhs_scale_0 * up_scale_0
                
                # ========== SECOND QUANTIZATION BLOCK (conditional) ==========
                # Only process if GEMM_BLOCK_SIZE_K > SCALE_BLOCK_SIZE_K
                if GEMM_BLOCK_SIZE_K > SCALE_BLOCK_SIZE_K:
                    k_start_1 = gemm_k_start + SCALE_BLOCK_SIZE_K
                    
                    # Check if second block is within bounds (compile-time + runtime check)
                    if k_start_1 < K:  # Runtime bound check
                        offsets_k_1 = k_start_1 + tl.arange(0, SCALE_BLOCK_SIZE_K)
                        
                        # Load scales for second block
                        scale_k_1 = scale_k_0 + 1  # Next scale block
                        gate_scale_1 = tl.load(gate_scale_base_ptr + scale_n * num_scale_k + scale_k_1)
                        up_scale_1 = tl.load(up_scale_base_ptr + scale_n * num_scale_k + scale_k_1)
                        
                        lhs_scale_k_1 = lhs_scale_k_0 + 1
                        lhs_scale_ptr_1 = lhs_scale_ptr + abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k_1 * stride_lhs_scale_k
                        lhs_scale_1 = tl.load(lhs_scale_ptr_1, mask=(abs_row_indices[:, None] < M), other=1.0)
                        
                        # Load data for second block
                        lhs_ptrs_1 = lhs_ptr + abs_row_indices[:, None] * stride_lhs_m + offsets_k_1[None, :] * stride_lhs_k
                        gate_ptrs_1 = gate_base_ptr + offsets_n[:, None] * stride_gate_n + offsets_k_1[None, :] * stride_gate_k
                        up_ptrs_1 = up_base_ptr + offsets_n[:, None] * stride_up_n + offsets_k_1[None, :] * stride_up_k
                        
                        lhs_mask_1 = (abs_row_indices[:, None] < M) & (offsets_k_1[None, :] < K) & (offsets_m[:, None] < valid_rows)
                        rhs_mask_1 = (offsets_n[:, None] < N) & (offsets_k_1[None, :] < K)
                        
                        lhs_1 = tl.load(lhs_ptrs_1, mask=lhs_mask_1, other=0.0)
                        gate_fp8_1 = tl.load(gate_ptrs_1, mask=rhs_mask_1, other=0.0)
                        up_fp8_1 = tl.load(up_ptrs_1, mask=rhs_mask_1, other=0.0)
                        
                        # Accumulate second block
                        gate_acc += tl.dot(lhs_1, tl.trans(gate_fp8_1)) * lhs_scale_1 * gate_scale_1
                        up_acc += tl.dot(lhs_1, tl.trans(up_fp8_1)) * lhs_scale_1 * up_scale_1

            # Apply SiLU activation and store
            output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
            output = tl.cast(output_acc, lhs_dtype)
            
            # Store output
            offs_out_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
            offs_out_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
            output_ptrs = output_ptr + offs_out_m[:, None] * stride_output_m + offs_out_n[None, :] * stride_output_n
            output_mask = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N) & (offsets_m[:, None] < valid_rows)
            
            tl.store(output_ptrs, output, mask=output_mask)

@torch.inference_mode()
def fused_fp8_moe_stage_1(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gate_weight_list: list[torch.Tensor],
    gate_ptrs_ptr: torch.Tensor,
    up_weight_list: list[torch.Tensor],
    up_ptrs_ptr: torch.Tensor,
    gate_scale_list: list[torch.Tensor],
    gate_scale_ptrs_ptr: torch.Tensor,
    up_scale_list: list[torch.Tensor],
    up_scale_ptrs_ptr: torch.Tensor,
    group_sizes: torch.Tensor,
    activated_group_idx: torch.Tensor,
    group_start_indices: torch.Tensor,
    num_active_experts: torch.Tensor,
    gate_gemm_block_size=[64, 16, 128],  # Using 256 for unrolling
    scale_block_size=[128, 128],
    num_stages=3,  # More stages for pipelining
    num_warps=4
):
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]

    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    grid = lambda META: (triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']), )
    
    try:
        fused_fp8_moe_stage_1_kernel_v2[grid](
            hidden_states, hidden_states_scale,
            gate_ptrs_ptr, up_ptrs_ptr,
            gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
            activated_group_idx, group_sizes, group_start_indices,
            num_active_experts,
            output,
            M, N, K,  # ← These become tl.constexpr in the kernel
            hidden_states.stride(0), hidden_states.stride(1),
            hidden_states_scale.stride(0), hidden_states_scale.stride(1),
            gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
            up_weight_list[0].stride(0), up_weight_list[0].stride(1),
            output.stride(0), output.stride(1),
            activated_group_idx.stride(0), 
            group_sizes.stride(0), 
            group_start_indices.stride(0),
            gate_ptrs_ptr.stride(0), 
            gate_scale_ptrs_ptr.stride(0),
            # GEMM config
            GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
            GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
            GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],  # 256
            SCALE_BLOCK_SIZE_N=scale_block_size[0],     # 128
            SCALE_BLOCK_SIZE_K=scale_block_size[1],     # 128
            num_stages=num_stages,
            num_warps=num_warps
        )
    except Exception as e:
        logging.error(f"Error in fused_fp8_moe_stage_1_optimized: {e}")
        raise
    
    return output

	

# ============================================================================
# HELPER: PRE-COMPUTE GROUP TILE MAPPING
# ============================================================================

def compute_group_tile_mapping(group_sizes: torch.Tensor, block_size_m: int = 64):
    """
    Pre-compute mapping from global tile ID to (group_id, local_tile_id).
    
    This avoids sequential scan in kernel for finding which group a tile belongs to.
    
    Returns:
        tile_to_group: [total_tiles] - which group each tile belongs to
        tile_to_local: [total_tiles] - local tile index within that group
        tiles_per_group: [num_groups] - number of tiles for each group
    """
    num_groups = len(group_sizes)
    tiles_per_group = torch.zeros(num_groups, dtype=torch.int32, device=group_sizes.device)
    
    # Calculate tiles per group
    for i in range(num_groups):
        tiles_per_group[i] = (group_sizes[i] + block_size_m - 1) // block_size_m
    
    total_tiles = tiles_per_group.sum().item()
    
    # Build mapping arrays
    tile_to_group = torch.zeros(total_tiles, dtype=torch.int32, device=group_sizes.device)
    tile_to_local = torch.zeros(total_tiles, dtype=torch.int32, device=group_sizes.device)
    
    tile_idx = 0
    for group_id in range(num_groups):
        num_tiles = tiles_per_group[group_id].item()
        for local_tile in range(num_tiles):
            tile_to_group[tile_idx] = group_id
            tile_to_local[tile_idx] = local_tile
            tile_idx += 1
    
    return tile_to_group, tile_to_local, tiles_per_group


# ============================================================================
# OPTIMIZED KERNEL WITH PRE-COMPUTED MAPPING
# ============================================================================

@triton.jit
def fused_fp8_moe_stage_1_kernel_v3(
    # Input
    lhs_ptr, lhs_scale_ptr,
    # Weights
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    # Group metadata
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    # Pre-computed tile mapping (NEW!)
    tile_to_group_ptr, tile_to_local_ptr,
    # Output
    output_ptr,
    # Dimensions
    M, N: tl.constexpr, K: tl.constexpr,
    # Strides
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    # Block sizes
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
):
    """
    V3: Uses pre-computed tile mapping for O(1) group lookup.
    No more sequential scan!
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    lhs_dtype = tl.bfloat16
    rhs_dtype = tl.float8e4nv
    scale_dtype = tl.float32
    acc_dtype = tl.float32
    
    # ===== O(1) GROUP LOOKUP (instead of sequential scan) =====
    group_id = tl.load(tile_to_group_ptr + pid_m)
    local_tile_idx = tl.load(tile_to_local_ptr + pid_m)
    
    # Load group metadata
    group_idx = tl.load(group_idx_ptr + group_id * stride_group_idx)
    gm = tl.load(group_sizes_ptr + group_id * stride_group_sizes)
    start_idx = tl.load(group_start_indices_ptr + group_id * stride_group_start_indices)
    
    # Calculate offsets
    offs_n = pid_n * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    mask_n = offs_n < N
    
    tile_start_m = start_idx + local_tile_idx * GEMM_BLOCK_SIZE_M
    offs_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    abs_row_indices = tile_start_m + offs_m
    
    remaining_rows = start_idx + gm - tile_start_m
    valid_rows = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows)
    mask_m = (offs_m < valid_rows) & (abs_row_indices < M)
    
    # Load weight pointers
    gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
    up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(rhs_dtype))
    gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))
    up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(scale_dtype))
    
    # Initialize accumulators
    gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
    up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype)
    
    # Scale indices
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    scale_n_idx = pid_n * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    
    # Single K-loop
    for k_tile in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
        k_start = k_tile * GEMM_BLOCK_SIZE_K
        offs_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load scales
        scale_k_idx = k_start // SCALE_BLOCK_SIZE_K
        
        gate_scale_addr = gate_scale_base_ptr + scale_n_idx * num_scale_k + scale_k_idx
        up_scale_addr = up_scale_base_ptr + scale_n_idx * num_scale_k + scale_k_idx
        gate_scale = tl.load(gate_scale_addr)
        up_scale = tl.load(up_scale_addr)
        
        lhs_scale_ptrs = lhs_scale_ptr + abs_row_indices * stride_lhs_scale_m + scale_k_idx * stride_lhs_scale_k
        lhs_scale = tl.load(lhs_scale_ptrs, mask=mask_m, other=1.0)
        
        # Load data
        lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offs_k[None, :] * stride_lhs_k)
        lhs_mask = mask_m[:, None] & mask_k[None, :]
        lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
        
        gate_ptrs = gate_base_ptr + (offs_n[:, None] * stride_gate_n + offs_k[None, :] * stride_gate_k)
        gate_mask = mask_n[:, None] & mask_k[None, :]
        gate_fp8 = tl.load(gate_ptrs, mask=gate_mask, other=0.0)
        
        up_ptrs = up_base_ptr + (offs_n[:, None] * stride_up_n + offs_k[None, :] * stride_up_k)
        up_mask = mask_n[:, None] & mask_k[None, :]
        up_fp8 = tl.load(up_ptrs, mask=up_mask, other=0.0)
        
        # Compute
        gate_partial = tl.dot(lhs, tl.trans(gate_fp8), out_dtype=acc_dtype)
        gate_acc += gate_partial * (lhs_scale[:, None] * gate_scale)
        
        up_partial = tl.dot(lhs, tl.trans(up_fp8), out_dtype=acc_dtype)
        up_acc += up_partial * (lhs_scale[:, None] * up_scale)
    
    # Fused SiLU activation
    gate_silu = gate_acc / (1.0 + tl.exp(-gate_acc))
    output_acc = gate_silu * up_acc
    output = output_acc.to(lhs_dtype)
    
    # Store
    offs_output_m = tile_start_m + tl.arange(0, GEMM_BLOCK_SIZE_M)
    offs_output_n = pid_n * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
    output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & mask_m[:, None]
    
    tl.store(output_ptrs, output, mask=output_mask)


@torch.inference_mode()
def fused_fp8_moe_stage_1_v3(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gate_weight_list: list[torch.Tensor],
    gate_ptrs_ptr: torch.Tensor,
    up_weight_list: list[torch.Tensor],
    up_ptrs_ptr: torch.Tensor,
    gate_scale_list: list[torch.Tensor],
    gate_scale_ptrs_ptr: torch.Tensor,
    up_scale_list: list[torch.Tensor],
    up_scale_ptrs_ptr: torch.Tensor,
    group_sizes: torch.Tensor,
    activated_group_idx: torch.Tensor,
    group_start_indices: torch.Tensor,
    num_active_experts: torch.Tensor,
):
    """
    V3: Pre-computes tile mapping for O(1) lookup.
    Best performance for scenarios with many small groups.
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    # Select config
    if N >= 8192:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 128
        num_warps, num_stages = 8, 4
    elif N >= 2048:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128
        num_warps, num_stages = 4, 5
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 128
        num_warps, num_stages = 4, 4
    
    # Pre-compute tile mapping
    tile_to_group, tile_to_local, tiles_per_group = compute_group_tile_mapping(
        group_sizes, BLOCK_M
    )
    total_m_tiles = len(tile_to_group)
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    grid = (total_m_tiles, triton.cdiv(N, BLOCK_N))
    
    fused_fp8_moe_stage_1_kernel_v3[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        activated_group_idx, group_sizes, group_start_indices,
        tile_to_group, tile_to_local,
        output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0),
        group_sizes.stride(0),
        group_start_indices.stride(0),
        gate_ptrs_ptr.stride(0),
        gate_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=BLOCK_M,
        GEMM_BLOCK_SIZE_N=BLOCK_N,
        GEMM_BLOCK_SIZE_K=BLOCK_K,
        SCALE_BLOCK_SIZE_K=128,
        SCALE_BLOCK_SIZE_N=128,
        num_stages=num_stages,
        num_warps=num_warps
    )
    
    return output
