import torch
import triton
import triton.language as tl
import os
# os.environ["TRITON_CACHE_SIZE"] = "2048"
import torch.distributed as dist
import math
import time

"""
	class DeepseekV3RMSNorm(nn.Module):
		def __init__(self, hidden_size, eps=1e-6):
			super().__init__()
			self.weight = nn.Parameter(torch.ones(hidden_size))
			self.variance_epsilon = eps

		def forward(self, hidden_states):
			input_dtype = hidden_states.dtype
			hidden_states = hidden_states.to(torch.float32)
			variance = hidden_states.pow(2).mean(-1, keepdim=True)
			hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
			return self.weight * hidden_states.to(input_dtype)
"""

"""
	We let each thread block to compute full N. 
	Then the variance for a row can be computed within the same thread block.
	The we can compute each tile of (C * torch.rsqrt(C.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)).to(torch.bfloat16) * weight
	without any redundant memory access.
"""
@triton.jit
def fused_get_query_states_kernel():
	"""
		This kernel fuse the following operations:
		q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
		query_states[:, :, self.kv_lora_rank: ] = rotary_pos_emb(q_pe, cos, sin, q_position_id)
		query_states[:, :, : self.kv_lora_rank] = fused_bhd_hdc(q_nope, q_absorb)

		Args:
			q (torch.Tensor): The input query tensor of shape (B, S, HD).
			cos (torch.Tensor): The cosine part of the rotary embedding.
			sin (torch.Tensor): The sine part of the rotary embedding.
			q_position_id (torch.Tensor): The position indices of the tokens corresponding to the query tensor.
			q_absorb (torch.Tensor): The tensor used for the BHD multiplication.

			qk_nope_head_dim (int): The dimension of the q_nope part.
			qk_rope_head_dim (int): The dimension of the q_pe part.
			kv_lora_rank (int): The rank for the kv lora.
			q_head_dim (int): The head dimension of the query tensor.
		
		Output:
			query_states (torch.Tensor): The output tensor of shape (B, S, QD).

	"""






@triton.jit
def fgemm_fp8_e4m3_bf16_kernel(
	# Pointers to matrices
	a_ptr, b_ptr, c_ptr, scale_ptr,
	# Matrix dimensions
	M, N, K,
	# Strides for A, B, and C
	stride_am, stride_ak,
	stride_bk, stride_bn,  # Transposed from original
	stride_cm, stride_cn,
	# Meta-parameters
	BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
	SCALE_BLOCK_M: tl.constexpr, SCALE_BLOCK_K: tl.constexpr,
	GROUP_SIZE_M: tl.constexpr,
	c_ptr_offset=0,
):
	"""
	Compute the matrix multiplication C = A @ B.T.
	
	A is a matrix of shape (M, K) with bf16 elements
	B is a matrix of shape (N, K) with fp8 elements that needs to be dequantized using scale
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

	c_ptr += c_ptr_offset
	# ----------------------------------------------------------
	# Create offset pointers and masks for A, B, and C
	offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offs_k = tl.arange(0, BLOCK_SIZE_K)
	
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
		# Load A and B tiles and compute matrix multiply
		a = tl.load(a_ptrs, mask=a_mask, other=0.0)
		b_fp8 = tl.load(b_ptrs, mask=b_mask, other=0.0)
		
		# Load scales for this block
		# how many blocks of size SCALE_BLOCK_K fit into the K‐dimension?
		n_scale_k = tl.cdiv(K, SCALE_BLOCK_K)

		# which scale‐block row does this N‐tile live in?
		# pid_n is the N‐tile index, each tile is BLOCK_SIZE_N rows,
		# and each scale‐row covers SCALE_BLOCK_M rows.
		scale_row = (pid_n * BLOCK_SIZE_N) // SCALE_BLOCK_M

		# which scale‐block col does this K‐tile live in?
		# k_idx = k * BLOCK_SIZE_K is the start of this K‐tile;
		# each scale‐col covers SCALE_BLOCK_K entries.
		scale_col = k_idx // SCALE_BLOCK_K

		# flatten into a 1D offset in the [n_scale_m, n_scale_k] scale array
		scale_idx = scale_row * n_scale_k + scale_col

		# load the single fp32 scale and let Triton broadcast it
		scale = tl.load(scale_ptr + scale_idx)
		
		# Convert B from fp8 to bf16
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


def fused_fp8_bf16_gemm(A, B_fp8, B_scale, block_size=(64, 64, 32), scale_block_size=(128, 128), group_size_m=8, out=None, offset=None):
	"""
	Performs a fused dequantization and matrix multiplication: A @ B_dequantized.T
	
	Args:
		A: torch.Tensor of shape (M, K) with bf16 elements
		B_fp8: torch.Tensor of shape (N, K) with fp8 elements
		B_scale: torch.Tensor of scales for dequantization
		block_size: tuple of (M, N, K) block sizes for the kernel
		scale_block_size: tuple of (M, K) block sizes for scales
		group_size_m: group size for M dimension (improves cache locality)
		
	Returns:
		C: torch.Tensor of shape (M, N) with bf16 elements
	"""
	# Handle 3D input case
	original_shape = A.shape
	if A.dim() == 3:
		bsz, seq_len, dim = A.shape
		A = A.reshape(bsz * seq_len, dim)
	
	# Get dimensions
	M, K = A.shape
	N, Kb = B_fp8.shape
	assert K == Kb, f"Incompatible dimensions: A has dims {A.shape}, B has dims {B_fp8.shape}"
	
	# Calculate number of scale blocks
	n_blocks = (N + scale_block_size[0] - 1) // scale_block_size[0]
	k_blocks = (K + scale_block_size[1] - 1) // scale_block_size[1]
	expected_scale_shape = (n_blocks, k_blocks)
	
	# Verify B_scale dimensions
	assert B_scale.numel() == expected_scale_shape[0] * expected_scale_shape[1], \
		f"Expected B_scale to have {expected_scale_shape[0] * expected_scale_shape[1]} elements, got {B_scale.numel()}"
	
	# Ensure inputs are in the proper format
	A = A.to(torch.bfloat16).contiguous()
	B_fp8 = B_fp8.contiguous()
	B_scale = B_scale.reshape(expected_scale_shape).contiguous()
	if out is None:
		C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

	
	
	# Calculate grid size with improved work distribution
	grid = lambda META: (
		triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
	)
	
	# Launch kernel with auto-tuning
	fgemm_fp8_e4m3_bf16_kernel[grid](
		A, B_fp8, out if out is not None else C, B_scale,
		M, N, K,
		A.stride(0), A.stride(1),
		B_fp8.stride(1), B_fp8.stride(0),  # Note the swap for transposition
		# C.stride(0), C.stride(1),
		N, 1,
		BLOCK_SIZE_M=block_size[0], 
		BLOCK_SIZE_N=block_size[1], 
		BLOCK_SIZE_K=block_size[2],
		SCALE_BLOCK_M=scale_block_size[0], 
		SCALE_BLOCK_K=scale_block_size[1],
		GROUP_SIZE_M=group_size_m,
		c_ptr_offset=offset if out is not None else 0,
	)
	
	# Reshape output if necessary
	if out is None:
		return C.reshape(original_shape[0], original_shape[1], -1) if len(original_shape) == 3 else C


