import torch
from math import ceil
import triton
import triton.language as tl

_FP8_MAX = 448.0
def compressed_kv_bf16_to_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Quantize a [bsz, seq, 576] BF16 tensor to FP8 per‐token.
	Returns:
		q:   torch.Tensor[bsz, seq, 576]   dtype=float8_e4m3fn
		s:   torch.Tensor[bsz, seq]        dtype=float32  (the scale factors)
	"""	
	assert x.dtype == torch.bfloat16
	assert x.shape[-1] == 576
	assert x.is_contiguous()
	assert x.dim() == 3, f"Expected x to be 3D tensor with last dimension of size 576, got {x.dim()}D with shape {x.shape}"

	bsz, seq_len, dim = x.shape
	M = bsz * seq_len
	x_flat = x.view(M, dim).float()              # to FP32 for reduction
	amax   = x_flat.abs().amax(dim=1).clamp(min=1e-6)  # [M]
	scale  = amax / _FP8_MAX                         # [M]
	# scale & cast
	y = x_flat / scale.unsqueeze(1)                  # [M, 576]
	q = y.to(torch.float8_e4m3fn)                    # [M, 576] in FP8
	return q.view(bsz, seq_len, dim), scale.view(bsz, seq_len)	

@triton.jit
def per_token_blocked_quantize_bf16_to_fp8_kernel(
	q_ptr, scale_ptr, out_ptr,
	dim: tl.constexpr, block_size: tl.constexpr,
	# Strides
	q_stride0, q_stride1, q_stride2,
	scale_stride0, scale_stride1, scale_stride2,
	out_stride0, out_stride1, out_stride2
):
	"""
	This kernel handle the following operations.def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		assert x.dim() == 2 and x.size(1) % 128 == 0
		m, n = x.shape
		x_view = x.view(m, -1, 128)
		x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
		return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
			torch.float8_e4m3fn
		).view(m, n), (x_amax / 448.0).view(m, -1)

	Args:
		q_ptr: Pointer to the quantized KV tensor [bsz, seq_len, dim], BF16
		scale_ptr: Pointer to the scale tensor [bsz, seq_len, num_blocks], float32
		out_ptr: Pointer to the output tensor [bsz, seq_len, dim], float8_e4m3fn
	
	Notes:
		- This kernel quantizes the input BF16 tensor to FP8 per token with a block size.
			The input dim may not be divisible by the block size for instance, 576. 
			This kernel handles the last block with a size less than the block size.
	"""
	pid_seq = tl.program_id(0)  # seq
	pid_token = tl.program_id(1)
	pid_block = tl.program_id(2)  # quantize block sized 128

	# Calculate the block index and offset in the q tensor
	block_offsets = pid_seq * q_stride0 + pid_token * q_stride1 + (pid_block * block_size + tl.arange(0, block_size)) * q_stride2
	# The size of the block can be less than 128 for the last block
	q_bf16 = tl.load(q_ptr + block_offsets, mask=(pid_block * block_size + tl.arange(0, block_size)) < dim, other=0.0)
	# x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
	# Find max for this block
	# q_abs = tl.abs(q_bf16)
	q_float = q_bf16.to(tl.float32)
	q_abs = tl.abs(q_float)
	amax = tl.max(q_abs, axis=0)  # [M] max across the block
	# amax = tl.where(amax < 1e-4, 1e-4, amax) 
	amax = tl.maximum(amax, 1e-6) 
	
	scale = amax / 448.0 
	q_fp8 = (q_float * (448.0 / amax)).to(tl.float8e4nv)  # quantize to FP8
	# Store the quantized tensor
	out_offsets = pid_seq * out_stride0 + pid_token * out_stride1 + (pid_block * block_size + tl.arange(0, block_size)) * out_stride2
	tl.store(out_ptr + out_offsets, q_fp8, mask=(pid_block * block_size + tl.arange(0, block_size)) < dim)
	# Store the scale factor
	tl.store(scale_ptr + pid_seq * scale_stride0 + pid_token * scale_stride1 + pid_block * scale_stride2, scale)



def per_token_blocked_quantize_bf16_to_fp8(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Quantize q [bsz, seq, token_dim] BF16 tensor to FP8 per token with a default block size 128.
	Returns:
		out:   torch.Tensor[bsz, seq, token_dim]   					dtype=float8_e4m3fn
		scale:   torch.Tensor[bsz, seq, ceil(token_dim / block_size)]   dtype=float32  (the scale factors)
	"""	
	assert x.dtype == torch.bfloat16
	assert x.is_contiguous()
	assert x.dim() == 3, f"Expected x to be 3D tensor got {x.dim()}D with shape {x.shape}"

	bsz, seq_len, dim = x.shape
	M = bsz * seq_len
	# Calculate the number of blocks
	num_blocks = (dim + block_size - 1) // block_size
	# Reshape x to [M, num_blocks, block_size]


	# 3D triton kernel grid
	grid = (bsz, seq_len, num_blocks)
	out = torch.empty((bsz, seq_len, dim), device=x.device, dtype=torch.float8_e4m3fn)
	scale = torch.empty((bsz, seq_len, num_blocks), device=x.device, dtype=torch.float32)
	# Call the kernel
	per_token_blocked_quantize_bf16_to_fp8_kernel[grid](
		x, scale, out,
		dim, block_size,
		x.stride(0), x.stride(1), x.stride(2),
		scale.stride(0), scale.stride(1), scale.stride(2),
		out.stride(0), out.stride(1), out.stride(2)
	)
	return out, scale


def compressed_kv_fp8_to_bf16_per_token(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
		"""
		Dequantize the output of bf16_to_fp8_per_token back to BF16.
		Inputs:
			q     [bsz, seq, 576]  dtype=float8_e4m3fn
			scale [bsz, seq]       dtype=float32
		Returns:
			x_bf16 [bsz, seq, 576] dtype=bfloat16
		"""
		bsz, seq_len, dim = q.shape
		M = bsz * seq_len
		# flatten
		q_flat = q.view(M, dim).float()               # upcast FP8→FP32
		x_rec = q_flat * s.view(M, 1)             # rescale
		return x_rec.to(torch.bfloat16).view(bsz, seq_len, dim)	


def deepseek_v3_dequantization(
		weight_data_fp8: torch.Tensor,
		weight_scale_inv_fp32: torch.Tensor,
		block_size=(128, 128),
) -> torch.Tensor:
		"""
		Vectorized dequantization that removes Python-level loops
		and leverages PyTorch's parallelism.
		"""
		rows, cols = weight_data_fp8.shape
		block_rows, block_cols = block_size

		# Number of blocks in each dimension
		n_block_rows = rows // block_rows
		n_block_cols = cols // block_cols

		# 1) Reshape weight data into 4D block form and cast to float32
		#    shape becomes [n_block_rows, block_rows, n_block_cols, block_cols].
		weight_4d = weight_data_fp8.reshape(
				n_block_rows, block_rows, n_block_cols, block_cols
		).to(torch.float32)

		# 2) Broadcast scale into 4D by unsqueezing along the second and fourth dimensions.
		#    shape becomes [n_block_rows, 1, n_block_cols, 1].
		scale_4d = weight_scale_inv_fp32.unsqueeze(1).unsqueeze(-1)

		# 3) Multiply once using broadcasting
		dequantized_4d = weight_4d * scale_4d

		# 4) Reshape back to [rows, cols] and cast to bfloat16
		dequantized_weight = dequantized_4d.reshape(rows, cols).to(torch.bfloat16)

		return dequantized_weight





@triton.jit
def dequant_compressed_kv_per_token_with_length_kernel(
			kv_ptr, scale_ptr, output_ptr,
			dim, block_size: tl.constexpr,

			# Strides
			kv_stride0, kv_stride1, kv_stride2,
			scale_stride0, scale_stride1, scale_stride2,
			output_stride0, output_stride1, output_stride2,
		
):
		"""
			Kernel to dequantize FP8 KV-Cache tensor with scale.
			Args:
					kv_ptr: Pointer to the quantized KV tensor [bsz, max_seq_len, dim]
					scale_ptr: Pointer to the scale tensor [bsz, max_seq_len, num_blocks]
					output_ptr: Pointer to the output tensor [bsz, padded_seq_len, dim]

			Notes:
					- This kernel load first seq_len elements of each sequence in the KV tensor
						and dequantizes them using the corresponding scale factors.
		"""
		pid_seq = tl.program_id(0) # seq
		pid_token = tl.program_id(1)
		pid_block = tl.program_id(2) # quantize block sized 128

		# Calculate the block index and offset in the kv tensor
		block_offsets = pid_seq * kv_stride0 + pid_token * kv_stride1 + (pid_block * block_size + tl.arange(0, block_size)) * kv_stride2
		# The size of the block can be less than 128 for the last block
		# Load the quantized KV tensor
		kv_fp8 = tl.load(kv_ptr + block_offsets, mask=(pid_block * block_size + tl.arange(0, block_size)) < dim)
		kv_fp32 = tl.cast(kv_fp8, tl.float32)
		# Load the scale tensor
		scale_offsets = pid_seq * scale_stride0 + pid_token * scale_stride1 + pid_block * scale_stride2
		scale = tl.load(scale_ptr + scale_offsets)
	
		# Dequantize
		# kv_fp32 = kv_fp32 * scale  # [block_size] in FP32
		kv_fp32 = kv_fp32 * scale
		out = tl.cast(kv_fp32, tl.bfloat16)

		out_offsets = pid_seq * output_stride0 + pid_token * output_stride1 + (pid_block * block_size + tl.arange(0, block_size)) * output_stride2
		# Store the dequantized output
		tl.store(output_ptr + out_offsets, out, mask=(pid_block * block_size + tl.arange(0, block_size)) < dim)



def dequant_compressed_kv_per_token_with_length(
		q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
):
		"""
		Dequantize FP8 KV-Cache tensor with scale.
		Args:
				q: [bsz, max_seq_len, dim]
				scale: [bsz, max_seq_len, num_blocks]
				seq_len: int, the length of the sequence to dequantize
		Returns:
				x: [bsz, padded_seq_len, dim]

		Notes:
				- return x with padded_seq_len.
					Where padded_seq_len is the nearest multiple of 64(page size for FlashMLA) greater than or equal to seq_len.
				- We dequantize the first seq_len elements of the max_seq_len tensor and write to x.
		"""
		assert q.is_cuda and scale.is_cuda
		assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32, f"Expected q to be float8_e4m3fn and scale to be float32, got {q.dtype} and {scale.dtype}"
		assert seq_len >= 0, f"seq_len must be >= 0, got {seq_len}"
		assert seq_len <= q.size(1), f"seq_len must be <= max_seq_len, got {seq_len} > {q.size(1)}"
		assert q.is_contiguous() and scale.is_contiguous(), "q and scale must be contiguous tensors"

		bsz, max_seq_len, dim = q.shape
		padded_seq_len = ceil(seq_len / 64) * 64  # Nearest multiple of 64

		result = torch.empty((bsz, padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)

		# Construct 3D triton grid: bsz, seq_len, num_blocks
		num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
		grid = (bsz, seq_len, num_blocks)
		# Call the kernel
		dequant_compressed_kv_per_token_with_length_kernel[grid](
				q, scale, result,
				dim, BLOCK_SIZE,
				q.stride(0), q.stride(1), q.stride(2),
				scale.stride(0), scale.stride(1), scale.stride(2),
				result.stride(0), result.stride(1), result.stride(2)
		)
		return result


@triton.jit
def dequant_compressed_kv_per_token_with_length_kernel_v2(
			kv_ptr, scale_ptr, output_ptr,
			dim: tl.constexpr, block_size: tl.constexpr,
			bsz: tl.constexpr, seq_len: tl.constexpr,
			num_blocks_per_token: tl.constexpr,

			# Strides
			kv_stride0, kv_stride1, kv_stride2,
			scale_stride0, scale_stride1, scale_stride2,
			output_stride0, output_stride1, output_stride2,
		
):
		"""
			Kernel to dequantize FP8 KV-Cache tensor with scale.
			Args:
					kv_ptr: Pointer to the quantized KV tensor [bsz, max_seq_len, dim]
					scale_ptr: Pointer to the scale tensor [bsz, max_seq_len, num_blocks]
					output_ptr: Pointer to the output tensor [bsz, padded_seq_len, dim]

			Notes:
					- This kernel load first seq_len elements of each sequence in the KV tensor
						and dequantizes them using the corresponding scale factors.
		"""
		pid_seq = tl.program_id(0) # seq
		pid_chunk = tl.program_id(1)  # Each chunk processes 16 tokens
		num_tokens = min(16, seq_len - pid_chunk * 16)  # Number of tokens in this chunk

		for token_id in range(num_tokens):
			pid_token = pid_chunk * 16 + token_id
			for pid_block in range(num_blocks_per_token):
				# Calculate the block index and offset in the kv tensor
				block_offsets = pid_seq * kv_stride0 + pid_token * kv_stride1 + (pid_block * block_size + tl.arange(0, block_size)) * kv_stride2
				# The size of the block can be less than 128 for the last block
				# Load the quantized KV tensor
				kv_fp8 = tl.load(kv_ptr + block_offsets, mask=(pid_block * block_size + tl.arange(0, block_size)) < dim)
				kv_fp32 = tl.cast(kv_fp8, tl.float32)
				# Load the scale tensor
				scale_offsets = pid_seq * scale_stride0 + pid_token * scale_stride1 + pid_block * scale_stride2
				scale = tl.load(scale_ptr + scale_offsets)
			
				# Dequantize
				# kv_fp32 = kv_fp32 * scale  # [block_size] in FP32
				kv_fp32 = kv_fp32 * scale
				out = tl.cast(kv_fp32, tl.bfloat16)

				out_offsets = pid_seq * output_stride0 + pid_token * output_stride1 + (pid_block * block_size + tl.arange(0, block_size)) * output_stride2
				# Store the dequantized output
				tl.store(output_ptr + out_offsets, out, mask=(pid_block * block_size + tl.arange(0, block_size)) < dim)


def dequant_compressed_kv_per_token_with_length_v2(
		q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
):
		"""
		Dequantize FP8 KV-Cache tensor with scale.
		Args:
				q: [bsz, max_seq_len, dim]
				scale: [bsz, max_seq_len, num_blocks]
		Returns:
				x: [bsz, padded_seq_len, dim]

		Notes:
				- return x with padded_seq_len.
					Where padded_seq_len is the nearest multiple of 64(page size for FlashMLA) greater than or equal to seq_len.
				- We dequantize the first seq_len elements of the max_seq_len tensor and write to x.
		"""
		assert q.is_cuda and scale.is_cuda
		assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32, f"Expected q to be float8_e4m3fn and scale to be float32, got {q.dtype} and {scale.dtype}"
		assert seq_len >= 0, f"seq_len must be >= 0, got {seq_len}"
		assert seq_len <= q.size(1), f"seq_len must be <= max_seq_len, got {seq_len} > {q.size(1)}"
		assert q.is_contiguous() and scale.is_contiguous(), "q and scale must be contiguous tensors"

		bsz, max_seq_len, dim = q.shape
		padded_seq_len = ceil(seq_len / 64) * 64  # Nearest multiple of 64

		result = torch.empty((bsz, padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)

		# Construct 3D triton grid: bsz, seq_len, num_blocks
		num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
		# Each CTA(program in triton) process 16 tokens.
		chunk_size = 16
		num_work_chunk = (ceil(seq_len / chunk_size))
		grid = (bsz, num_work_chunk)
		# Call the kernel
		dequant_compressed_kv_per_token_with_length_kernel_v2[grid](
				q, scale, result,
				dim, BLOCK_SIZE,
				bsz, seq_len,
				# chunk_size,  # tokens per chunk
				num_blocks,

				q.stride(0), q.stride(1), q.stride(2),
				scale.stride(0), scale.stride(1), scale.stride(2),
				result.stride(0), result.stride(1), result.stride(2)
		)
		return result



# @triton.jit
# def dequant_compressed_kv_per_token_with_length_persistant_kernel(
# 			kv_ptr, scale_ptr, output_ptr,
# 			dim: tl.constexpr, block_size: tl.constexpr,
# 			bsz: tl.constexpr, seq_len: tl.constexpr, 
# 			num_blocks_per_token: tl.constexpr, total_num_blocks: tl.constexpr,
# 			block_per_cta: tl.constexpr,

# 			# Strides
# 			kv_stride0, kv_stride1, kv_stride2,
# 			scale_stride0, scale_stride1, scale_stride2,
# 			output_stride0, output_stride1, output_stride2,
		
# ):
# 		"""
# 			Kernel to dequantize FP8 KV-Cache tensor with scale.
# 			Args:
# 					kv_ptr: Pointer to the quantized KV tensor [bsz, max_seq_len, dim]
# 					scale_ptr: Pointer to the scale tensor [bsz, max_seq_len, num_blocks]
# 					output_ptr: Pointer to the output tensor [bsz, padded_seq_len, dim]

# 			Notes:
# 					- Persistant kernel approach applied. 
# 		"""
# 		cta_id = tl.program_id(0)

# 		# Compute the start end of the block range for this CTA
# 		start_block = cta_id * block_per_cta
# 		end_block = min(start_block + block_per_cta, total_num_blocks)

# 		# Iterate over the blocks assigned to this CTA
# 		for block_id in range(start_block, end_block):
# 			# Calculate the sequence and token indices from the block ID
# 			cur_seq = block_id // (num_blocks_per_token * seq_len)
# 			cur_token = (block_id // num_blocks_per_token) % seq_len	
# 			cur_block = block_id % num_blocks_per_token

# 			# Calculate the block index and offset in the kv tensor
# 			block_offsets = cur_seq * kv_stride0 + cur_token * kv_stride1 + (cur_block * block_size + tl.arange(0, block_size)) * kv_stride2
# 			# The size of the block can be less than 128 for the last block
# 			# Load the quantized KV tensor		
# 			kv_fp8 = tl.load(kv_ptr + block_offsets, mask=(cur_block * block_size + tl.arange(0, block_size)) < dim)
# 			kv_fp32 = tl.cast(kv_fp8, tl.float32)
# 			# Load the scale tensor
# 			scale_offsets = cur_seq * scale_stride0 + cur_token * scale_stride1 + cur_block * scale_stride2
# 			scale = tl.load(scale_ptr + scale_offsets)
# 			# Dequantize
# 			kv_fp32 = kv_fp32 * scale  # [block_size] in FP32
# 			out = tl.cast(kv_fp32, tl.bfloat16)
# 			# Calculate the output offsets
# 			out_offsets = cur_seq * output_stride0 + cur_token * output_stride1 + (cur_block * block_size + tl.arange(0, block_size)) * output_stride2
# 			# Store the dequantized output
# 			tl.store(output_ptr + out_offsets, out, mask=(cur_block * block_size + tl.arange(0, block_size)) < dim)
		


# def dequant_compressed_kv_per_token_with_length(
# 		q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
# ):
# 		"""
# 		Dequantize FP8 KV-Cache tensor with scale.
# 		Args:
# 				q: [bsz, max_seq_len, dim]
# 				scale: [bsz, max_seq_len, num_blocks]
# 		Returns:
# 				x: [bsz, padded_seq_len, dim]

# 		Notes:
# 				- return x with padded_seq_len.
# 					Where padded_seq_len is the nearest multiple of 64(page size for FlashMLA) greater than or equal to seq_len.
# 				- We dequantize the first seq_len elements of the max_seq_len tensor and write to x.
# 		"""
# 		assert q.is_cuda and scale.is_cuda
# 		assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32, f"Expected q to be float8_e4m3fn and scale to be float32, got {q.dtype} and {scale.dtype}"
# 		assert seq_len >= 0, f"seq_len must be >= 0, got {seq_len}"
# 		assert seq_len <= q.size(1), f"seq_len must be <= max_seq_len, got {seq_len} > {q.size(1)}"
# 		assert q.is_contiguous() and scale.is_contiguous(), "q and scale must be contiguous tensors"

# 		bsz, max_seq_len, dim = q.shape
# 		padded_seq_len = ceil(seq_len / 64) * 64  # Nearest multiple of 64

# 		result = torch.empty((bsz, padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)

# 		num_blocks_per_token = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
# 		total_num_blocks = bsz * seq_len * num_blocks_per_token
# 		num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
# 		block_per_cta = total_num_blocks // num_sms
# 		if block_per_cta == 0:
# 				block_per_cta = 1  # Ensure at least one block per CTA
# 		grid = (num_sms,)
# 		# Call the kernel
# 		dequant_compressed_kv_per_token_with_length_persistant_kernel[grid](
# 				q, scale, result,
# 				dim, BLOCK_SIZE, bsz, seq_len, num_blocks_per_token, total_num_blocks, block_per_cta,
# 				q.stride(0), q.stride(1), q.stride(2),
# 				scale.stride(0), scale.stride(1), scale.stride(2),
# 				result.stride(0), result.stride(1), result.stride(2)
# 		)
# 		return result


@triton.jit
def dequant_compressed_kv_per_token_kernel(
			kv_ptr, scale_ptr, output_ptr,
			dim: tl.constexpr, quant_block_size: tl.constexpr,
			bsz, padded_seq_len, max_seq_len,

			BLOCK_SIZE_M: tl.constexpr,
			BLOCK_SIZE_N: tl.constexpr,
			# Strides
			kv_stride0, kv_stride1,
			scale_stride0, scale_stride1,
			output_stride0, output_stride1,
		
):
		"""
			Kernel to dequantize FP8 KV-Cache tensor with scale.
			Args:
					kv_ptr: Pointer to the quantized KV tensor [bsz, max_seq_len, dim]
					scale_ptr: Pointer to the scale tensor [bsz, max_seq_len, num_blocks]
					output_ptr: Pointer to the output tensor [bsz, padded_seq_len, dim]

			Notes:
					- This kernel load first seq_len elements of each sequence in the KV tensor
						and dequantizes them using the corresponding scale factors.
		"""
		tile_m = tl.program_id(0)  
		tile_n = tl.program_id(1)  
		token_idx = tile_m * BLOCK_SIZE_M
		seq_idx = token_idx // padded_seq_len
		kv_token_idx = seq_idx * max_seq_len + token_idx % padded_seq_len

		block_idx = tile_n * BLOCK_SIZE_N // quant_block_size

		block_ptr = kv_ptr + kv_token_idx * kv_stride0 + tile_n * BLOCK_SIZE_N * kv_stride1
		# Load the [BLOCK_SIZE_M, BLOCK_SIZE_N] FP8 blocks
		mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * max_seq_len) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < dim)
		fp8_block = tl.load(
			block_ptr + (tl.arange(0, BLOCK_SIZE_M)[:, None] * kv_stride0 + tl.arange(0, BLOCK_SIZE_N)[None, :] * kv_stride1),
			mask=mask, other=0.0, cache_modifier='.cg'
		)
		scale_offsets = scale_ptr + (kv_token_idx + tl.arange(0, BLOCK_SIZE_M)[:, None]) * scale_stride0 + block_idx * scale_stride1
		scale = tl.load(scale_offsets, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * max_seq_len), other=0.0, cache_modifier='.cg')

		# Dequantize the FP8 blocks
		fp32_block = tl.cast(fp8_block, tl.float32) * scale
		# Convert to BF16
		bf16_block = tl.cast(fp32_block, tl.bfloat16)

		# Store the dequantized blocks
		output_ptrs = output_ptr + tile_m * BLOCK_SIZE_M * output_stride0 + tile_n * BLOCK_SIZE_N * output_stride1
		output_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * output_stride0 + tl.arange(0, BLOCK_SIZE_N)[None, :] * output_stride1
		output_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * padded_seq_len) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < dim)
		tl.store(output_ptrs + output_offsets, bf16_block, mask=output_mask)


def dequant_compressed_kv_per_token(
		q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
):
		"""
		Dequantize FP8 KV-Cache tensor with scale.
		Args:
				q: [bsz, max_seq_len, dim]
				scale: [bsz, max_seq_len, num_blocks]
		Returns:
				x: [bsz, padded_seq_len, dim]

		Notes:
				- return x with padded_seq_len.
					Where padded_seq_len is the nearest multiple of 64(page size for FlashMLA) greater than or equal to seq_len.
				- We dequantize the first seq_len elements of the max_seq_len tensor and write to x.
		"""
		assert q.is_cuda and scale.is_cuda
		assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32, f"Expected q to be float8_e4m3fn and scale to be float32, got {q.dtype} and {scale.dtype}"
		assert seq_len >= 0, f"seq_len must be >= 0, got {seq_len}"
		assert seq_len <= q.size(1), f"seq_len must be <= max_seq_len, got {seq_len} > {q.size(1)}"
		assert q.is_contiguous() and scale.is_contiguous(), "q and scale must be contiguous tensors"
		assert q.dim() == scale.dim(), f"Expected q and scale to have the same number of dimensions, got {q.dim()} and {scale.dim()}"

		bsz, max_seq_len, dim = q.shape
		padded_seq_len = ceil(seq_len / 64) * 64  # Nearest multiple of 64

		result = torch.ones((bsz * padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)
		# Assign max value to result tensor
		# result.fill_(1e10)

		# Construct 3D triton grid: bsz, seq_len, num_blocks
		BLOCK_SIZE_M = 64
		BLOCK_SIZE_N = 64
		num_blocks_m = ceil(bsz * padded_seq_len / BLOCK_SIZE_M)
		num_blocks_n = ceil(dim / BLOCK_SIZE_N)
		q = q.view(-1, dim)
		scale = scale.view(-1, scale.shape[-1])  # Flatten the first two dimensions
		
		grid = (num_blocks_m, num_blocks_n)
		# Call the kernel
		dequant_compressed_kv_per_token_kernel[grid](
				q, scale, result,
				dim, BLOCK_SIZE,
				bsz, padded_seq_len, max_seq_len,
				BLOCK_SIZE_M, BLOCK_SIZE_N,

				q.stride(0), q.stride(1),
				scale.stride(0), scale.stride(1),
				result.stride(0), result.stride(1)
		)
		# result = torch.ones((bsz * padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)
		return result.view(bsz, padded_seq_len, dim)  # Reshape back to [bsz, padded_seq_len, dim]


# @triton.jit
# def dequant_compressed_kv_per_token_kernel(
#         kv_ptr, scale_ptr, output_ptr,
#         dim: tl.constexpr, quant_block_size: tl.constexpr,
#         bsz, padded_seq_len, max_seq_len, seq_len,

#         BLOCK_SIZE_M: tl.constexpr,
#         BLOCK_SIZE_N: tl.constexpr,
#         # Strides
#         kv_stride0, kv_stride1,
#         scale_stride0, scale_stride1,
#         output_stride0, output_stride1,
    
# ):
#     """
#         Kernel to dequantize FP8 KV-Cache tensor with scale.
#         Args:
#                 kv_ptr: Pointer to the quantized KV tensor [bsz, max_seq_len, dim]
#                 scale_ptr: Pointer to the scale tensor [bsz, max_seq_len, num_blocks]
#                 output_ptr: Pointer to the output tensor [bsz, padded_seq_len, dim]

#         Notes:
#                 - This kernel loads first seq_len elements of each sequence in the KV tensor
#                     and dequantizes them using the corresponding scale factors.
#     """
#     tile_m = tl.program_id(0)  
#     tile_n = tl.program_id(1)  
    
#     # Calculate batch and sequence indices properly
#     batch_idx = tile_m // tl.cdiv(padded_seq_len, BLOCK_SIZE_M)
#     seq_tile = tile_m % tl.cdiv(padded_seq_len, BLOCK_SIZE_M)
#     seq_start = seq_tile * BLOCK_SIZE_M
    
#     # Calculate dimension indices
#     dim_start = tile_n * BLOCK_SIZE_N
    
#     # Create ranges for this tile
#     seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_M)
#     dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_N)
    
#     # Mask for valid sequences (only process up to seq_len)
#     seq_mask = seq_offsets < seq_len
#     dim_mask = dim_offsets < dim
    
#     # Calculate block index for scale tensor
#     block_idx = dim_start // quant_block_size
    
#     # Load KV data
#     kv_ptrs = (kv_ptr + 
#                batch_idx * max_seq_len * kv_stride0 + 
#                seq_offsets[:, None] * kv_stride0 + 
#                dim_offsets[None, :] * kv_stride1)
    
#     # Mask for loading from KV tensor
#     kv_mask = (seq_mask[:, None] & dim_mask[None, :] & 
#                (batch_idx < bsz))
    
#     fp8_block = tl.load(kv_ptrs, mask=kv_mask, other=0.0, cache_modifier='.cg')

#     # Load scale factors
#     scale_ptrs = (scale_ptr + 
#                   batch_idx * max_seq_len * scale_stride0 + 
#                   seq_offsets[:, None] * scale_stride0 + 
#                   block_idx * scale_stride1)
    
#     scale_mask = seq_mask[:, None] & (batch_idx < bsz)
#     scale = tl.load(scale_ptrs, mask=scale_mask, other=1.0, cache_modifier='.cg')

#     # Dequantize the FP8 blocks
#     fp32_block = tl.cast(fp8_block, tl.float32) * scale
#     # Convert to BF16
#     bf16_block = tl.cast(fp32_block, tl.bfloat16)

#     # Store the dequantized blocks to output
#     output_ptrs = (output_ptr + 
#                    batch_idx * padded_seq_len * output_stride0 + 
#                    seq_offsets[:, None] * output_stride0 + 
#                    dim_offsets[None, :] * output_stride1)
    
#     # Mask for output - ensure we're within bounds for padded output
#     output_seq_mask = seq_offsets < padded_seq_len
#     output_mask = (output_seq_mask[:, None] & dim_mask[None, :] & 
#                    (batch_idx < bsz) & seq_mask[:, None])
    
#     tl.store(output_ptrs, bf16_block, mask=output_mask)


# def dequant_compressed_kv_per_token(
#         q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
# ):
#     """
#     Dequantize FP8 KV-Cache tensor with scale.
#     Args:
#             q: [bsz, max_seq_len, dim]
#             scale: [bsz, max_seq_len, num_blocks]
#     Returns:
#             x: [bsz, padded_seq_len, dim]

#     Notes:
#             - return x with padded_seq_len.
#                 Where padded_seq_len is the nearest multiple of 64(page size for FlashMLA) greater than or equal to seq_len.
#             - We dequantize the first seq_len elements of the max_seq_len tensor and write to x.
#     """
#     assert q.is_cuda and scale.is_cuda
#     assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32, f"Expected q to be float8_e4m3fn and scale to be float32, got {q.dtype} and {scale.dtype}"
#     assert seq_len >= 0, f"seq_len must be >= 0, got {seq_len}"
#     assert seq_len <= q.size(1), f"seq_len must be <= max_seq_len, got {seq_len} > {q.size(1)}"
#     assert q.is_contiguous() and scale.is_contiguous(), "q and scale must be contiguous tensors"
#     assert q.dim() == 3 and scale.dim() == 3, f"Expected q and scale to be 3D tensors, got {q.dim()}D and {scale.dim()}D"

#     bsz, max_seq_len, dim = q.shape
#     padded_seq_len = ((seq_len + 63) // 64) * 64  # Nearest multiple of 64, more explicit

#     # Create output tensor with correct shape
#     result = torch.empty((bsz, padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)

#     # Calculate grid dimensions
#     BLOCK_SIZE_M = 64
#     BLOCK_SIZE_N = 64
#     num_blocks_m = (bsz * ((padded_seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M))
#     num_blocks_n = (dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
#     grid = (num_blocks_m, num_blocks_n)
    
#     # Call the kernel
#     dequant_compressed_kv_per_token_kernel[grid](
#             q, scale, result,
#             dim, BLOCK_SIZE,
#             bsz, padded_seq_len, max_seq_len, seq_len,
#             BLOCK_SIZE_M, BLOCK_SIZE_N,

#             q.stride(0), q.stride(1),
#             scale.stride(0), scale.stride(1),
#             result.stride(0), result.stride(1)
#     )
#     return result







		

