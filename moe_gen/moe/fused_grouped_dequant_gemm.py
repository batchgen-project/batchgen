import torch
import triton
import triton.language as tl
import os
# os.environ["TRITON_CACHE_SIZE"] = "2048"
import torch.distributed as dist
import math
import time

@triton.jit
def grouped_fp8_bf16_gemm_kernel(
	# Pointers to matrices
	a_ptr, b_ptrs_ptr, c_ptr, scale_ptrs_ptr, group_indices_ptr,
	# Matrix dimensions
	M, N, K, num_groups,
	# Strides for A and C
	stride_am, stride_ak,
	stride_cm, stride_cn,
	# B matrix strides (assumed same for all B matrices)
	stride_bk, stride_bn,
	# Meta-parameters
	BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_M: tl.constexpr, SCALE_BLOCK_K: tl.constexpr,
	GROUP_SIZE_M: tl.constexpr
):
	"""
	Grouped GEMM kernel that performs C = A @ B.T where different rows of A
	use different B matrices based on group_indices mapping.
	
	A is a matrix of shape (M, K) with bf16 elements
	B matrices are of shape (N, K) with fp8 elements that need dequantization
	C is a matrix of shape (M, N) with bf16 elements
	"""
	# -----------------------------------------------------------
	# Map program IDs to blocks of C
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
	# Create offset pointers and masks for A and C
	offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offs_k = tl.arange(0, BLOCK_SIZE_K)
	
	# Get the group index for the first row in this block
	# All rows in a block should belong to the same group for efficiency
	first_row_idx = pid_m * BLOCK_SIZE_M
	group_idx = tl.load(group_indices_ptr + first_row_idx)
	
	# Load the appropriate B matrix pointer and scale pointer for this group
	b_ptr = tl.load(b_ptrs_ptr + group_idx)
	scale_ptr = tl.load(scale_ptrs_ptr + group_idx)
	
	# Create pointers for A and B
	a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
	b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
	
	# Create masks for bounds checking
	a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
	b_mask = (offs_bn[:, None] < N) & (offs_k[None, :] < K)

	# -----------------------------------------------------------
	# Initialize accumulator
	acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	
	# -----------------------------------------------------------
	# Iterate to compute a block of the C matrix
	for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
		k_idx = k * BLOCK_SIZE_K
		
		# Load A and B tiles
		a = tl.load(a_ptrs, mask=a_mask, other=0.0)
		b_fp8 = tl.load(b_ptrs, mask=b_mask, other=0.0)
		
		# Load scales for this block
		n_scale_k = tl.cdiv(K, SCALE_BLOCK_K)
		scale_row = (pid_n * BLOCK_SIZE_N) // SCALE_BLOCK_M
		scale_col = k_idx // SCALE_BLOCK_K
		scale_idx = scale_row * n_scale_k + scale_col
		
		# Load the scale and broadcast it
		scale = tl.load(scale_ptr + scale_idx)
		
		# Dequantize B from fp8 to bf16
		b_fp32 = tl.cast(b_fp8, tl.float32)
		b_scaled = b_fp32 * scale
		b = tl.cast(b_scaled, tl.bfloat16)
		
		# Matrix multiplication with transposed B
		acc += tl.dot(a, tl.trans(b))
		
		# Update pointers for next K block
		a_ptrs += BLOCK_SIZE_K * stride_ak
		b_ptrs += BLOCK_SIZE_K * stride_bk
		
	# Store the result
	offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
	c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
	
	# Convert to bf16 before storing
	c = tl.cast(acc, tl.bfloat16)
	tl.store(c_ptrs, c, mask=c_mask)


def fused_grouped_fp8_bf16_gemm(A, group_map, B_fp8_list, B_scale_list, 
								block_size=(64, 64, 32), scale_block_size=(128, 128)):
	"""
	Performs a fused dequantization and matrix multiplication: A @ B_dequantized.T
	
	Args:
		A: torch.Tensor of shape (M, K) with bf16 elements
		group_map: Dict[int, torch.Tensor] mapping group IDs to their corresponding indices in A.
		B_fp8_list: List of torch.Tensor, each of shape (N, K) with fp8 elements
		B_scale_list: List of torch.Tensor, each containing scales in FP32 for dequantization
		block_size: Tuple of (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
		scale_block_size: Tuple of (SCALE_BLOCK_M, SCALE_BLOCK_K)
		
	Returns:
		C: torch.Tensor of shape (M, N) with bf16 elements
	"""
	original_shape = A.shape
	if A.dim() == 3:
		bsz, seq_len, dim = A.shape
		A = A.reshape(bsz * seq_len, dim)
		  
	M, K = A.shape
	N = B_fp8_list[0].shape[0]  # Assuming all B matrices have same N
	num_groups = len(B_fp8_list)
	
	# Create group indices array - maps each row of A to its group
	group_indices = torch.zeros(M, dtype=torch.int32, device=A.device)
	for group_id, indices in group_map.items():
		group_indices[indices] = group_id
	
	# Create arrays of pointers to B matrices and scales
	B_ptrs = torch.zeros(num_groups, dtype=torch.int64, device=A.device)
	scale_ptrs = torch.zeros(num_groups, dtype=torch.int64, device=A.device)
	
	for i, (B, scale) in enumerate(zip(B_fp8_list, B_scale_list)):
		B_ptrs[i] = B.data_ptr()
		scale_ptrs[i] = scale.data_ptr()
	
	# Allocate output
	C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
	
	# Launch parameters
	BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = block_size
	SCALE_BLOCK_M, SCALE_BLOCK_K = scale_block_size
	GROUP_SIZE_M = 8  # Tunable parameter
	
	# Calculate grid dimensions
	grid_m = triton.cdiv(M, BLOCK_SIZE_M)
	grid_n = triton.cdiv(N, BLOCK_SIZE_N)
	grid = (grid_m * grid_n,)
	
	# Launch kernel
	grouped_fp8_bf16_gemm_kernel[grid](
		# Pointers
		A.data_ptr(), B_ptrs.data_ptr(), C.data_ptr(), scale_ptrs.data_ptr(),
		group_indices.data_ptr(),
		# Dimensions
		M, N, K, num_groups,
		# Strides
		A.stride(0), A.stride(1),
		C.stride(0), C.stride(1),
		# B strides (assuming all B matrices have same layout)
		B_fp8_list[0].stride(1), B_fp8_list[0].stride(0),  # Transposed
		# Meta-parameters
		BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
		SCALE_BLOCK_M=SCALE_BLOCK_M, SCALE_BLOCK_K=SCALE_BLOCK_K,
		GROUP_SIZE_M=GROUP_SIZE_M
	)
	
	return C.reshape(original_shape[0], original_shape[1], -1) if len(original_shape) == 3 else C