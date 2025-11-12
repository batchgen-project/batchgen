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

# @triton.jit
# def per_token_blocked_quantize_bf16_to_fp8_kernel_v1(
# 	q_ptr, scale_ptr, out_ptr,
# 	dim: tl.constexpr, block_size: tl.constexpr,
# 	# Strides
# 	q_stride0, q_stride1, q_stride2,
# 	scale_stride0, scale_stride1, scale_stride2,
# 	out_stride0, out_stride1, out_stride2
# ):
# 	"""
# 	This kernel handle the following operations.def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# 		assert x.dim() == 2 and x.size(1) % 128 == 0
# 		m, n = x.shape
# 		x_view = x.view(m, -1, 128)
# 		x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
# 		return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
# 			torch.float8_e4m3fn
# 		).view(m, n), (x_amax / 448.0).view(m, -1)

# 	Args:
# 		q_ptr: Pointer to the quantized KV tensor [bsz, seq_len, dim], BF16
# 		scale_ptr: Pointer to the scale tensor [bsz, seq_len, num_blocks], float32
# 		out_ptr: Pointer to the output tensor [bsz, seq_len, dim], float8_e4m3fn
	
# 	Notes:
# 		- This kernel quantizes the input BF16 tensor to FP8 per token with a block size.
# 			The input dim may not be divisible by the block size for instance, 576. 
# 			This kernel handles the last block with a size less than the block size.
# 	"""
# 	pid_seq = tl.program_id(0)  # seq
# 	pid_token = tl.program_id(1)
# 	pid_block = tl.program_id(2)  # quantize block sized 128

# 	# Calculate the block index and offset in the q tensor
# 	block_offsets = pid_seq * q_stride0 + pid_token * q_stride1 + (pid_block * block_size + tl.arange(0, block_size)) * q_stride2
# 	# The size of the block can be less than 128 for the last block
# 	q_bf16 = tl.load(q_ptr + block_offsets, mask=(pid_block * block_size + tl.arange(0, block_size)) < dim, other=0.0)
# 	# x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
# 	# Find max for this block
# 	# q_abs = tl.abs(q_bf16)
# 	q_float = q_bf16.to(tl.float32)
# 	q_abs = tl.abs(q_float)
# 	amax = tl.max(q_abs, axis=0)  # [M] max across the block
# 	# amax = tl.where(amax < 1e-4, 1e-4, amax) 
# 	amax = tl.maximum(amax, 1e-6) 
	
# 	scale = amax / 448.0 
# 	q_fp8 = (q_float * (448.0 / amax)).to(tl.float8e4nv)  # quantize to FP8
# 	# Store the quantized tensor
# 	out_offsets = pid_seq * out_stride0 + pid_token * out_stride1 + (pid_block * block_size + tl.arange(0, block_size)) * out_stride2
# 	tl.store(out_ptr + out_offsets, q_fp8, mask=(pid_block * block_size + tl.arange(0, block_size)) < dim)
# 	# Store the scale factor
# 	tl.store(scale_ptr + pid_seq * scale_stride0 + pid_token * scale_stride1 + pid_block * scale_stride2, scale)


@triton.jit
def per_token_blocked_quantize_bf16_to_fp8_kernel(
    q_ptr, scale_ptr, out_ptr,
    dim: tl.constexpr, block_size: tl.constexpr,
    # Strides
    q_stride0, q_stride1, q_stride2,
    scale_stride0, scale_stride1, scale_stride2,
    out_stride0, out_stride1, out_stride2
):
    # Constants matching C++ implementation
    FP8_SAFE_MAX: tl.constexpr = 440.0  # Leave headroom
    FP8_E4M3_MIN_NORMAL: tl.constexpr = 1.52587890625e-05
    EPSILON: tl.constexpr = 1e-12
    
    pid_seq = tl.program_id(0)
    pid_token = tl.program_id(1)
    pid_block = tl.program_id(2)
    
    # Calculate offsets
    block_start = pid_block * block_size
    block_offsets = pid_seq * q_stride0 + pid_token * q_stride1 + \
                    (block_start + tl.arange(0, block_size)) * q_stride2
    
    # Load with masking for partial blocks
    mask = (block_start + tl.arange(0, block_size)) < dim
    q_bf16 = tl.load(q_ptr + block_offsets, mask=mask, other=0.0)
    
    # Convert to float32 for computation
    q_float = q_bf16.to(tl.float32)
    
    # Compute absolute maximum (symmetric quantization)
    q_abs = tl.abs(q_float)
    amax = tl.max(q_abs, axis=0)
    
    # Apply minimum threshold (matching C++)
    amax = tl.maximum(amax, FP8_E4M3_MIN_NORMAL)
    
    # Compute scale factor
    scale = tl.maximum(amax / FP8_SAFE_MAX, EPSILON)
    
    # Quantize with explicit clamping
    q_scaled = q_float / scale
    q_scaled = tl.minimum(q_scaled, FP8_SAFE_MAX)
    q_scaled = tl.maximum(q_scaled, -FP8_SAFE_MAX)
    
    # Handle NaN/Inf (set to 0)
    is_finite = tl.abs(q_scaled) < 1e30  # Triton doesn't have isfinite
    q_scaled = tl.where(is_finite, q_scaled, 0.0)
    
    # Convert to FP8
    q_fp8 = q_scaled.to(tl.float8e4nv)
    
    # Store results
    out_offsets = pid_seq * out_stride0 + pid_token * out_stride1 + \
                  (block_start + tl.arange(0, block_size)) * out_stride2
    tl.store(out_ptr + out_offsets, q_fp8, mask=mask)
    tl.store(scale_ptr + pid_seq * scale_stride0 + pid_token * scale_stride1 + \
             pid_block * scale_stride2, scale)



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


# @triton.jit
# def dequant_compressed_kv_per_token_kernel_dep(
# 			kv_ptr, scale_ptr, output_ptr,
# 			dim: tl.constexpr, quant_block_size: tl.constexpr,
# 			bsz, padded_seq_len, max_seq_len,

# 			BLOCK_SIZE_M: tl.constexpr,
# 			BLOCK_SIZE_N: tl.constexpr,
# 			# Strides
# 			kv_stride0, kv_stride1,
# 			scale_stride0, scale_stride1,
# 			output_stride0, output_stride1,
		
# ):
# 		"""
# 			Kernel to dequantize FP8 KV-Cache tensor with scale.
# 			Args:
# 					kv_ptr: Pointer to the quantized KV tensor [bsz, max_seq_len, dim]
# 					scale_ptr: Pointer to the scale tensor [bsz, max_seq_len, num_blocks]
# 					output_ptr: Pointer to the output tensor [bsz, padded_seq_len, dim]

# 			Notes:
# 					- This kernel load first seq_len elements of each sequence in the KV tensor
# 						and dequantizes them using the corresponding scale factors.
# 		"""
# 		tile_m = tl.program_id(0)  
# 		tile_n = tl.program_id(1)  
# 		token_idx = tile_m * BLOCK_SIZE_M
# 		seq_idx = token_idx // padded_seq_len
# 		kv_token_idx = seq_idx * max_seq_len + token_idx % padded_seq_len

# 		block_idx = tile_n * BLOCK_SIZE_N // quant_block_size

# 		block_ptr = kv_ptr + kv_token_idx * kv_stride0 + tile_n * BLOCK_SIZE_N * kv_stride1
# 		# Load the [BLOCK_SIZE_M, BLOCK_SIZE_N] FP8 blocks
# 		mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * max_seq_len) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < dim)
# 		fp8_block = tl.load(
# 			block_ptr + (tl.arange(0, BLOCK_SIZE_M)[:, None] * kv_stride0 + tl.arange(0, BLOCK_SIZE_N)[None, :] * kv_stride1),
# 			mask=mask, other=0.0, cache_modifier='.cg'
# 		)
# 		scale_offsets = scale_ptr + (kv_token_idx + tl.arange(0, BLOCK_SIZE_M)[:, None]) * scale_stride0 + block_idx * scale_stride1
# 		scale = tl.load(scale_offsets, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * max_seq_len), other=0.0, cache_modifier='.cg')

# 		# Dequantize the FP8 blocks
# 		fp32_block = tl.cast(fp8_block, tl.float32) * scale
# 		# Convert to BF16
# 		bf16_block = tl.cast(fp32_block, tl.bfloat16)

# 		# Store the dequantized blocks
# 		output_ptrs = output_ptr + tile_m * BLOCK_SIZE_M * output_stride0 + tile_n * BLOCK_SIZE_N * output_stride1
# 		output_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * output_stride0 + tl.arange(0, BLOCK_SIZE_N)[None, :] * output_stride1
# 		output_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * padded_seq_len) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < dim)
# 		tl.store(output_ptrs + output_offsets, bf16_block, mask=output_mask)


# def dequant_compressed_kv_per_token(
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
# 		assert q.dim() == scale.dim(), f"Expected q and scale to have the same number of dimensions, got {q.dim()} and {scale.dim()}"

# 		bsz, max_seq_len, dim = q.shape
# 		padded_seq_len = ceil(seq_len / 64) * 64  # Nearest multiple of 64

# 		result = torch.ones((bsz * padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)
# 		# Assign max value to result tensor
# 		# result.fill_(1e10)

# 		# Construct 3D triton grid: bsz, seq_len, num_blocks
# 		BLOCK_SIZE_M = 64
# 		BLOCK_SIZE_N = 64
# 		num_blocks_m = ceil(bsz * padded_seq_len / BLOCK_SIZE_M)
# 		num_blocks_n = ceil(dim / BLOCK_SIZE_N)
# 		q = q.view(-1, dim)
# 		scale = scale.view(-1, scale.shape[-1])  # Flatten the first two dimensions
		
# 		grid = (num_blocks_m, num_blocks_n)
# 		# Call the kernel
# 		dequant_compressed_kv_per_token_kernel[grid](
# 				q, scale, result,
# 				dim, BLOCK_SIZE,
# 				bsz, padded_seq_len, max_seq_len,
# 				BLOCK_SIZE_M, BLOCK_SIZE_N,

# 				q.stride(0), q.stride(1),
# 				scale.stride(0), scale.stride(1),
# 				result.stride(0), result.stride(1)
# 		)
# 		# result = torch.ones((bsz * padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)
# 		return result.view(bsz, padded_seq_len, dim)  # Reshape back to [bsz, padded_seq_len, dim]



@triton.jit
def dequant_compressed_kv_per_token_kernel_3d(
    kv_ptr, scale_ptr, output_ptr,
    dim: tl.constexpr, 
    quant_block_size: tl.constexpr,
    seq_len: tl.constexpr,
    bsz: tl.constexpr, 
    padded_seq_len: tl.constexpr, 
    max_seq_len: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    kv_stride0, kv_stride1, kv_stride2,
    scale_stride0, scale_stride1, scale_stride2,
    output_stride0, output_stride1, output_stride2,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    # Get sequence tile index
    seq_tile_idx = tl.program_id(1)
    # Get dimension tile index
    dim_tile_idx = tl.program_id(2)
    
    # Calculate starting positions
    seq_start = seq_tile_idx * BLOCK_SIZE_M
    dim_start = dim_tile_idx * BLOCK_SIZE_N
    
    # Create offset arrays
    seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_M)
    dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    seq_mask = seq_offsets < tl.minimum(seq_len, padded_seq_len)
    dim_mask = dim_offsets < dim
    
    # Calculate which quantization blocks we're accessing
    quant_block_start = dim_start // quant_block_size
    quant_block_end = tl.minimum(
        (dim_start + BLOCK_SIZE_N + quant_block_size - 1) // quant_block_size,
        (dim + quant_block_size - 1) // quant_block_size
    )
    
    # Load FP8 data
    kv_base = kv_ptr + batch_idx * kv_stride0
    
    # Create 2D mask
    mask_2d = seq_mask[:, None] & dim_mask[None, :]
    
    # Calculate pointers for KV data
    kv_offsets = seq_offsets[:, None] * kv_stride1 + dim_offsets[None, :] * kv_stride2
    
    # Load FP8 values
    fp8_block = tl.load(
        kv_base + kv_offsets,
        mask=mask_2d,
        other=0.0,
        cache_modifier='.cg'
    )
    
    # For each element, determine which quantization block it belongs to
    # and load the corresponding scale
    scale_base = scale_ptr + batch_idx * scale_stride0
    
    # Initialize output block
    output_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16)
    
    # Process each quantization block separately
    for block_idx in range(quant_block_start, quant_block_end):
        # Determine which elements belong to this quantization block
        block_start_dim = block_idx * quant_block_size
        block_end_dim = tl.minimum((block_idx + 1) * quant_block_size, dim)
        
        # Create mask for elements in this quantization block
        in_block = (dim_offsets >= block_start_dim) & (dim_offsets < block_end_dim)
        
        # Load scales for this block
        scale_offsets = seq_offsets * scale_stride1 + block_idx * scale_stride2
        scales = tl.load(
            scale_base + scale_offsets,
            mask=seq_mask,
            other=1.0,  # Use 1.0 as default to avoid NaN
            cache_modifier='.cg'
        )
        
        # Dequantize elements in this block
        # Broadcast scales and apply only to relevant elements
        block_mask = mask_2d & in_block[None, :]
        dequantized = tl.cast(fp8_block, tl.float32) * scales[:, None]
        
        # Accumulate to output (using where to avoid NaN propagation)
        output_block = tl.where(
            block_mask,
            tl.cast(dequantized, tl.bfloat16),
            output_block
        )
    
    # Store the result
    output_base = output_ptr + batch_idx * output_stride0
    output_offsets = seq_offsets[:, None] * output_stride1 + dim_offsets[None, :] * output_stride2
    
    # Final mask for output
    output_mask = (seq_offsets[:, None] < padded_seq_len) & (dim_offsets[None, :] < dim)
    
    tl.store(
        output_base + output_offsets,
        output_block,
        mask=output_mask
    )

# def dequant_compressed_kv_per_token(
#     q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
# ):
#     assert q.is_cuda and scale.is_cuda
#     assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
#     assert seq_len >= 0 and seq_len <= q.size(1)
#     assert q.is_contiguous() and scale.is_contiguous()
    
#     bsz, max_seq_len, dim = q.shape
#     padded_seq_len = ((seq_len + 63) // 64) * 64
    
#     # Calculate number of quantization blocks per token
#     num_quant_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
#     assert scale.shape == (bsz, max_seq_len, num_quant_blocks), \
#         f"Scale shape mismatch: expected {(bsz, max_seq_len, num_quant_blocks)}, got {scale.shape}"
    
#     result = torch.zeros((bsz, padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)
    
#     # Use 3D grid: (batch, sequence tiles, dim tiles)
#     BLOCK_SIZE_M = 64
#     BLOCK_SIZE_N = 64
    
#     grid = (
#         bsz,
#         (padded_seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
#         (dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
#     )
    
#     dequant_compressed_kv_per_token_kernel[grid](
#         q, scale, result,
#         dim, BLOCK_SIZE, seq_len,
#         bsz, padded_seq_len, max_seq_len,
#         BLOCK_SIZE_M, BLOCK_SIZE_N,
#         q.stride(0), q.stride(1), q.stride(2),
#         scale.stride(0), scale.stride(1), scale.stride(2),
#         result.stride(0), result.stride(1), result.stride(2)
#     )
    
#     return result


# """ V2 """
# @triton.jit
# def dequant_compressed_kv_per_token_kernel(
#     kv_ptr, scale_ptr, output_ptr,
#     dim: tl.constexpr, 
#     quant_block_size: tl.constexpr,
#     seq_len: tl.constexpr,
#     bsz: tl.constexpr, 
#     padded_seq_len: tl.constexpr, 
#     max_seq_len: tl.constexpr,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     kv_stride0, kv_stride1, kv_stride2,
#     scale_stride0, scale_stride1, scale_stride2,
#     output_stride0, output_stride1, output_stride2,
# ):
#     # Get batch index
#     batch_idx = tl.program_id(0)
#     # Get sequence tile index
#     seq_tile_idx = tl.program_id(1)
#     # Get dimension tile index
#     dim_tile_idx = tl.program_id(2)
    
#     # Calculate starting positions
#     seq_start = seq_tile_idx * BLOCK_SIZE_M
#     dim_start = dim_tile_idx * BLOCK_SIZE_N
    
#     # Create offset arrays
#     seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_M)
#     dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_N)
    
#     # Create masks - ensure we respect actual sequence length
#     seq_mask = seq_offsets < tl.minimum(seq_len, padded_seq_len)
#     dim_mask = dim_offsets < dim
    
#     # Calculate which quantization blocks we're accessing
#     quant_block_start = dim_start // quant_block_size
#     quant_block_end = tl.minimum(
#         (dim_start + BLOCK_SIZE_N + quant_block_size - 1) // quant_block_size,
#         (dim + quant_block_size - 1) // quant_block_size
#     )
    
#     # Load FP8 data
#     kv_base = kv_ptr + batch_idx * kv_stride0
    
#     # Create 2D mask
#     mask_2d = seq_mask[:, None] & dim_mask[None, :]
    
#     # Calculate pointers for KV data
#     kv_offsets = seq_offsets[:, None] * kv_stride1 + dim_offsets[None, :] * kv_stride2
    
#     # Load FP8 values with explicit float8 type
#     fp8_block = tl.load(
#         kv_base + kv_offsets,
#         mask=mask_2d,
#         other=tl.cast(0.0, tl.float8e4nv),  # Use FP8 zero
#         cache_modifier='.cg'
#     )
    
#     # Convert FP8 to float32 safely
#     # FP8 E4M3 range is approximately ±448, so clamp to avoid overflow
#     fp32_block = tl.cast(fp8_block, tl.float32)
    
#     # Clamp to valid range to prevent NaN/Inf
#     fp32_block = tl.where(
#         mask_2d,
#         tl.minimum(tl.maximum(fp32_block, -448.0), 448.0),
#         0.0
#     )
    
#     # Initialize output block with zeros
#     output_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     # Scale base pointer
#     scale_base = scale_ptr + batch_idx * scale_stride0
    
#     # Process each quantization block separately
#     for block_idx in range(quant_block_start, quant_block_end):
#         # Determine which elements belong to this quantization block
#         block_start_dim = block_idx * quant_block_size
#         block_end_dim = tl.minimum((block_idx + 1) * quant_block_size, dim)
        
#         # Create mask for elements in this quantization block
#         in_block = (dim_offsets >= block_start_dim) & (dim_offsets < block_end_dim)
        
#         # Load scales for this block
#         scale_offsets = seq_offsets * scale_stride1 + block_idx * scale_stride2
        
#         # Load scales with proper masking
#         scales = tl.load(
#             scale_base + scale_offsets,
#             mask=seq_mask,
#             other=0.0,  # Use 0.0 for invalid positions
#             cache_modifier='.cg'
#         )
        
#         # Ensure scales are valid (non-NaN, non-Inf, non-zero)
#         scales = tl.where(
#             seq_mask,
#             tl.where(
#                 tl.abs(scales) > 1e-10,  # Avoid division by zero
#                 scales,
#                 1.0
#             ),
#             0.0
#         )
        
#         # Check for NaN/Inf in scales and replace with 1.0
#         scale_is_finite = (scales == scales) & (tl.abs(scales) < 1e10)
#         scales = tl.where(scale_is_finite, scales, 1.0)
        
#         # Create combined mask for this block
#         block_mask = mask_2d & in_block[None, :]
        
#         # Dequantize elements in this block
#         # Broadcast scales and apply only to relevant elements
#         dequantized = fp32_block * scales[:, None]
        
#         # Check for NaN/Inf in dequantized values
#         dequantized_is_finite = (dequantized == dequantized) & (tl.abs(dequantized) < 65504.0)  # BF16 max
#         dequantized = tl.where(dequantized_is_finite, dequantized, 0.0)
        
#         # Accumulate to output only for valid positions
#         output_block = tl.where(
#             block_mask,
#             dequantized,
#             output_block
#         )
    
#     # Convert to bfloat16 with clamping
#     output_block = tl.where(
#         mask_2d,
#         tl.cast(
#             tl.minimum(tl.maximum(output_block, -65504.0), 65504.0),  # BF16 range
#             tl.bfloat16
#         ),
#         tl.cast(0.0, tl.bfloat16)
#     )
    
#     # Store the result
#     output_base = output_ptr + batch_idx * output_stride0
#     output_offsets = seq_offsets[:, None] * output_stride1 + dim_offsets[None, :] * output_stride2
    
#     # Final mask for output - only store to valid padded positions
#     output_mask = (seq_offsets[:, None] < padded_seq_len) & (dim_offsets[None, :] < dim)
    
#     tl.store(
#         output_base + output_offsets,
#         output_block,
#         mask=output_mask
#     )



# @triton.jit
# def dequant_compressed_kv_per_token_kernel(
#     kv_ptr, scale_ptr, output_ptr,
#     dim: tl.constexpr, 
#     quant_block_size: tl.constexpr,
#     seq_len: tl.constexpr,
#     bsz: tl.constexpr, 
#     padded_seq_len: tl.constexpr, 
#     max_seq_len: tl.constexpr,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     kv_stride0, kv_stride1, kv_stride2,
#     scale_stride0, scale_stride1, scale_stride2,
#     output_stride0, output_stride1, output_stride2,
# ):
#     """
#     Dequantize FP8 KV cache with per-token block quantization.
    
#     Key features:
#     - Handles partial blocks at the end of dimensions
#     - Returns padded sequence length for alignment
#     - Valid tokens are on the left side
#     """
#     # Get batch index
#     batch_idx = tl.program_id(0)
#     # Get sequence tile index
#     seq_tile_idx = tl.program_id(1)
#     # Get dimension tile index  
#     dim_tile_idx = tl.program_id(2)
    
#     # Calculate starting positions for this tile
#     seq_start = seq_tile_idx * BLOCK_SIZE_M
#     dim_start = dim_tile_idx * BLOCK_SIZE_N
    
#     # Create offset arrays
#     seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_M)
#     dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_N)
    
#     # Create masks - only process valid tokens (up to seq_len)
#     seq_mask = seq_offsets < seq_len
#     dim_mask = dim_offsets < dim
    
#     # Combined mask for loading
#     mask_2d = seq_mask[:, None] & dim_mask[None, :]
    
#     # Load FP8 data from the input tensor
#     kv_base = kv_ptr + batch_idx * kv_stride0
#     kv_offsets = seq_offsets[:, None] * kv_stride1 + dim_offsets[None, :] * kv_stride2
    
#     fp8_block = tl.load(
#         kv_base + kv_offsets,
#         mask=mask_2d,
#         other=0.0,
#         eviction_policy='evict_last'
#     )
    
#     # Convert FP8 to float32 for computation
#     fp32_block = fp8_block.to(tl.float32)
    
#     # Initialize output block
#     output_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     # Scale base pointer
#     scale_base = scale_ptr + batch_idx * scale_stride0
    
#     # Approach: Process each column independently based on its quantization block
#     # This avoids loops and dynamic conditionals
    
#     # Compute which quantization block each column belongs to
#     quant_block_indices = dim_offsets // quant_block_size
    
#     # We know that with BLOCK_SIZE_N=64 and quant_block_size=128,
#     # we can have at most 2 different blocks in a tile
#     # Let's handle up to 4 blocks for generality
    
#     # Process block 0 (if present)
#     block_0_mask = quant_block_indices == 0
#     if tl.sum(block_0_mask.to(tl.int32)) > 0:
#         scale_offsets_0 = seq_offsets * scale_stride1 + 0 * scale_stride2
#         scales_0 = tl.load(scale_base + scale_offsets_0, mask=seq_mask, other=1.0)
#         scales_0 = tl.where(tl.abs(scales_0) < 1e-10, 1.0, scales_0)
#         mask_0 = seq_mask[:, None] & block_0_mask[None, :] & dim_mask[None, :]
#         output_block = tl.where(mask_0, fp32_block * scales_0[:, None], output_block)
    
#     # Process block 1 (if present)
#     block_1_mask = quant_block_indices == 1
#     if tl.sum(block_1_mask.to(tl.int32)) > 0:
#         scale_offsets_1 = seq_offsets * scale_stride1 + 1 * scale_stride2
#         scales_1 = tl.load(scale_base + scale_offsets_1, mask=seq_mask, other=1.0)
#         scales_1 = tl.where(tl.abs(scales_1) < 1e-10, 1.0, scales_1)
#         mask_1 = seq_mask[:, None] & block_1_mask[None, :] & dim_mask[None, :]
#         output_block = tl.where(mask_1, fp32_block * scales_1[:, None], output_block)
    
#     # Process block 2 (if present)
#     block_2_mask = quant_block_indices == 2
#     if tl.sum(block_2_mask.to(tl.int32)) > 0:
#         scale_offsets_2 = seq_offsets * scale_stride1 + 2 * scale_stride2
#         scales_2 = tl.load(scale_base + scale_offsets_2, mask=seq_mask, other=1.0)
#         scales_2 = tl.where(tl.abs(scales_2) < 1e-10, 1.0, scales_2)
#         mask_2 = seq_mask[:, None] & block_2_mask[None, :] & dim_mask[None, :]
#         output_block = tl.where(mask_2, fp32_block * scales_2[:, None], output_block)
    
#     # Process block 3 (if present)
#     block_3_mask = quant_block_indices == 3
#     if tl.sum(block_3_mask.to(tl.int32)) > 0:
#         scale_offsets_3 = seq_offsets * scale_stride1 + 3 * scale_stride2
#         scales_3 = tl.load(scale_base + scale_offsets_3, mask=seq_mask, other=1.0)
#         scales_3 = tl.where(tl.abs(scales_3) < 1e-10, 1.0, scales_3)
#         mask_3 = seq_mask[:, None] & block_3_mask[None, :] & dim_mask[None, :]
#         output_block = tl.where(mask_3, fp32_block * scales_3[:, None], output_block)
    
#     # Process block 4 (if present) 
#     block_4_mask = quant_block_indices == 4
#     if tl.sum(block_4_mask.to(tl.int32)) > 0:
#         scale_offsets_4 = seq_offsets * scale_stride1 + 4 * scale_stride2
#         scales_4 = tl.load(scale_base + scale_offsets_4, mask=seq_mask, other=1.0)
#         scales_4 = tl.where(tl.abs(scales_4) < 1e-10, 1.0, scales_4)
#         mask_4 = seq_mask[:, None] & block_4_mask[None, :] & dim_mask[None, :]
#         output_block = tl.where(mask_4, fp32_block * scales_4[:, None], output_block)
    
#     # Clamp to BF16 range before conversion
#     output_block = tl.minimum(tl.maximum(output_block, -65504.0), 65504.0)
    
#     # Convert to bfloat16
#     output_bf16 = output_block.to(tl.bfloat16)
    
#     # Store the result - write to padded output tensor
#     output_base = output_ptr + batch_idx * output_stride0
#     output_offsets = seq_offsets[:, None] * output_stride1 + dim_offsets[None, :] * output_stride2
    
#     # Output mask ensures we write within padded_seq_len bounds
#     output_mask = (seq_offsets[:, None] < padded_seq_len) & (dim_offsets[None, :] < dim)
    
#     tl.store(
#         output_base + output_offsets,
#         output_bf16,
#         mask=output_mask
#     )


# @triton.jit
# def dequant_compressed_kv_per_token_kernel(
#     kv_ptr, scale_ptr, output_ptr,
#     dim: tl.constexpr, 
#     quant_block_size: tl.constexpr,
#     seq_len: tl.constexpr,
#     bsz: tl.constexpr, 
#     padded_seq_len: tl.constexpr, 
#     max_seq_len: tl.constexpr,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     kv_stride0, kv_stride1, kv_stride2,
#     scale_stride0, scale_stride1, scale_stride2,
#     output_stride0, output_stride1, output_stride2,
# ):
#     # Get program IDs for each dimension
#     batch_idx = tl.program_id(0)
#     seq_tile_idx = tl.program_id(1)
#     dim_tile_idx = tl.program_id(2)
    
#     # Calculate starting offsets for the current tile
#     seq_start = seq_tile_idx * BLOCK_SIZE_M
#     dim_start = dim_tile_idx * BLOCK_SIZE_N
    
#     # Create 1D offset arrays for sequence and dimension
#     seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_M)
#     dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_N)
    
#     # Create masks to avoid out-of-bounds memory access
#     # Mask for valid sequence tokens (used for loading)
#     seq_load_mask = seq_offsets < seq_len
#     # Mask for valid dimensions
#     dim_mask = dim_offsets < dim
    
#     # --- Load Quantized FP8 Data ---
#     kv_base = kv_ptr + batch_idx * kv_stride0
#     kv_offsets = seq_offsets[:, None] * kv_stride1 + dim_offsets[None, :] * kv_stride2
    
#     # Create a 2D mask for loading the FP8 block
#     kv_load_mask = seq_load_mask[:, None] & dim_mask[None, :]
    
#     # Load the FP8 values, using a type consistent with PyTorch (e4m3fn)
#     fp8_block = tl.load(
#         kv_base + kv_offsets,
#         mask=kv_load_mask,
#         other=tl.cast(0.0, tl.float8e4nv), # Use the correct FP8 type
#         cache_modifier='.cg'
#     )
    
#     # Cast FP8 to FP32 for computation. Clamp for safety.
#     fp32_block = tl.cast(fp8_block, tl.float32)
#     fp32_block = tl.minimum(tl.maximum(fp32_block, -448.0), 448.0)
    
#     # --- Dequantization Loop ---
    
#     # **CRITICAL FIX**: Initialize the output block to zeros before the loop
#     # This enables accumulation instead of overwriting.
#     output_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     # Determine the range of quantization blocks this dim_tile overlaps with
#     quant_block_start = dim_start // quant_block_size
#     num_quant_blocks_total = (dim + quant_block_size - 1) // quant_block_size
#     quant_block_end = tl.minimum(
#         (dim_start + BLOCK_SIZE_N + quant_block_size - 1) // quant_block_size,
#         num_quant_blocks_total
#     )
    
#     # Base pointer for scales
#     scale_base = scale_ptr + batch_idx * scale_stride0
    
#     for block_idx in range(quant_block_start, quant_block_end):
#         # Identify elements within this specific quantization block
#         block_start_dim = block_idx * quant_block_size
#         block_end_dim = tl.minimum((block_idx + 1) * quant_block_size, dim)
#         in_block_mask = (dim_offsets[None, :] >= block_start_dim) & (dim_offsets[None, :] < block_end_dim)
        
#         # Load the corresponding scales for each token in the sequence tile
#         scale_offsets = seq_offsets * scale_stride1 + block_idx * scale_stride2
#         scales = tl.load(
#             scale_base + scale_offsets,
#             mask=seq_load_mask,
#             other=1.0, # Default scale to 1.0 for out-of-bounds tokens
#             cache_modifier='.cg'
#         )
        
#         # Simplified Safety Check: Replace non-finite or zero scales with 1.0
#         # The C++ code ensures scales are non-zero, so this is a robust safeguard.
#         scale_is_valid = (scales == scales) & (tl.abs(scales) > 1e-12)
#         scales = tl.where(scale_is_valid, scales, 1.0)
        
#         # Dequantize by multiplying with the broadcasted scale
#         dequantized = fp32_block * scales[:, None]
        
#         # Create a combined mask for this specific quantization block
#         # This ensures we only update the elements relevant to the current scale
#         update_mask = kv_load_mask & in_block_mask
        
#         # **CRITICAL FIX**: Accumulate the results. Add the dequantized values
#         # for the current block to the output_block.
#         output_block += tl.where(update_mask, dequantized, 0.0)

#     # --- Cast and Store the Final Result ---
    
#     # Cast the final accumulated float32 values to bfloat16
#     output_block_bf16 = tl.cast(output_block, tl.bfloat16)

#     # Create a mask for storing, respecting the padded output tensor dimensions
#     output_store_mask = (seq_offsets[:, None] < padded_seq_len) & dim_mask[None, :]
    
#     # Calculate output pointers and store the result
#     output_base = output_ptr + batch_idx * output_stride0
#     output_offsets = seq_offsets[:, None] * output_stride1 + dim_offsets[None, :] * output_stride2
    
#     tl.store(
#         output_base + output_offsets,
#         output_block_bf16,
#         mask=output_store_mask
#     )


def dequant_compressed_kv_per_token_3D(
    q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
):
    """
    Dequantize FP8 compressed KV cache with per-token quantization.
    
    Args:
        q: FP8 quantized tensor of shape (bsz, max_seq_len, dim)
        scale: Scaling factors of shape (bsz, max_seq_len, num_quant_blocks)
        seq_len: Actual sequence length to process
        BLOCK_SIZE: Quantization block size
    
    Returns:
        Dequantized BF16 tensor of shape (bsz, padded_seq_len, dim)
    """
    assert q.is_cuda and scale.is_cuda
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert seq_len >= 0 and seq_len <= q.size(1)
    assert q.is_contiguous() and scale.is_contiguous()
    
    bsz, max_seq_len, dim = q.shape
    padded_seq_len = ((seq_len + 63) // 64) * 64
    
    # Calculate number of quantization blocks per token
    num_quant_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert scale.shape == (bsz, max_seq_len, num_quant_blocks), \
        f"Scale shape mismatch: expected {(bsz, max_seq_len, num_quant_blocks)}, got {scale.shape}"
    
    # Check for NaN/Inf in input tensors
    if torch.isnan(scale).any() or torch.isinf(scale).any():
        print("Warning: NaN or Inf values detected in scale tensor")
        # Replace NaN/Inf with 1.0
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
    
    # Initialize result tensor with zeros
    result = torch.zeros((bsz, padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)
    
    # Use 3D grid: (batch, sequence tiles, dim tiles)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid = (
        bsz,
        (padded_seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    dequant_compressed_kv_per_token_kernel_3d[grid](
        q, scale, result,
        dim, BLOCK_SIZE, seq_len,
        bsz, padded_seq_len, max_seq_len,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        q.stride(0), q.stride(1), q.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        result.stride(0), result.stride(1), result.stride(2)
    )
    
    return result


# # Optional: Debug helper function
# def check_for_nans(tensor, name):
#     """Helper function to check for NaN values in tensors."""
#     if torch.isnan(tensor).any():
#         nan_count = torch.isnan(tensor).sum().item()
#         print(f"Warning: {name} contains {nan_count} NaN values")
#         print(f"  Shape: {tensor.shape}")
#         print(f"  Dtype: {tensor.dtype}")
#         nan_indices = torch.where(torch.isnan(tensor))
#         if len(nan_indices[0]) > 0:
#             print(f"  First NaN at: {[idx[0].item() for idx in nan_indices]}")
#         return True
#     return False




		


# """ V3 """
# @triton.jit
# def dequant_compressed_kv_per_token_kernel(
#     kv_ptr, scale_ptr, output_ptr,
#     dim: tl.constexpr, quant_block_size: tl.constexpr,
#     bsz, padded_seq_len, max_seq_len,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     # Strides
#     kv_stride0, kv_stride1,
#     scale_stride0, scale_stride1,
#     output_stride0, output_stride1,
# ):
#     """
#     Kernel to dequantize FP8 KV-Cache tensor with scale.
#     Args:
#         kv_ptr: Pointer to the quantized KV tensor [bsz, max_seq_len, dim]
#         scale_ptr: Pointer to the scale tensor [bsz, max_seq_len, num_blocks]
#         output_ptr: Pointer to the output tensor [bsz, padded_seq_len, dim]
    
#     Notes:
#         - This kernel loads first seq_len elements of each sequence in the KV tensor
#           and dequantizes them using the corresponding scale factors.
#         - Includes numerical precision protection for FP8->FP32->BF16 conversion
#     """
#     tile_m = tl.program_id(0)  
#     tile_n = tl.program_id(1)  
#     token_idx = tile_m * BLOCK_SIZE_M
#     seq_idx = token_idx // padded_seq_len
#     kv_token_idx = seq_idx * max_seq_len + token_idx % padded_seq_len

#     block_idx = tile_n * BLOCK_SIZE_N // quant_block_size

#     block_ptr = kv_ptr + kv_token_idx * kv_stride0 + tile_n * BLOCK_SIZE_N * kv_stride1
    
#     # Load the [BLOCK_SIZE_M, BLOCK_SIZE_N] FP8 blocks
#     mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * max_seq_len) & \
#            (tl.arange(0, BLOCK_SIZE_N)[None, :] < dim)
    
#     # Load FP8 data with explicit FP8 zero for masked elements
#     fp8_block = tl.load(
#         block_ptr + (tl.arange(0, BLOCK_SIZE_M)[:, None] * kv_stride0 + 
#                     tl.arange(0, BLOCK_SIZE_N)[None, :] * kv_stride1),
#         mask=mask, 
#         other=tl.cast(0.0, tl.float8e4nv),  # Use FP8 zero instead of float zero
#         cache_modifier='.cg'
#     )
    
#     # Load scale values
#     scale_offsets = scale_ptr + (kv_token_idx + tl.arange(0, BLOCK_SIZE_M)[:, None]) * scale_stride0 + \
#                    block_idx * scale_stride1
#     scale_mask = tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * max_seq_len
#     scale = tl.load(
#         scale_offsets, 
#         mask=scale_mask, 
#         other=1.0,  # Use 1.0 as default scale to avoid corruption
#         cache_modifier='.cg'
#     )
    
#     # NUMERICAL FIX 1: Cast FP8 to FP32 with clamping
#     # FP8 E4M3 range is approximately ±448
#     fp32_block = tl.cast(fp8_block, tl.float32)
    
#     # NUMERICAL FIX 2: Clamp FP32 values to valid range before scaling
#     fp32_block = tl.where(
#         mask,
#         tl.minimum(tl.maximum(fp32_block, -448.0), 448.0),
#         0.0
#     )
    
#     # NUMERICAL FIX 3: Validate scales before multiplication
#     # Check for NaN/Inf in scales (NaN != NaN, Inf has large absolute value)
#     scale_is_finite = (scale == scale) & (tl.abs(scale) < 1e10)
#     scale_is_valid = scale_is_finite & (tl.abs(scale) > 1e-10)  # Avoid near-zero scales
    
#     # Use safe scale values (1.0 for invalid scales to preserve the value)
#     safe_scale = tl.where(
#         scale_mask,
#         tl.where(scale_is_valid, scale, 1.0),
#         1.0
#     )
    
#     # NUMERICAL FIX 4: Dequantize with safe multiplication
#     fp32_block = fp32_block * safe_scale
    
#     # NUMERICAL FIX 5: Check for NaN/Inf after dequantization
#     # BF16 has same range as FP32 (±3.39e38) but we clamp to reasonable values
#     MAX_BF16_SAFE = 65504.0  # Use FP16 max as safe value for stability
#     dequant_is_finite = (fp32_block == fp32_block) & (tl.abs(fp32_block) < MAX_BF16_SAFE)
#     fp32_block = tl.where(
#         dequant_is_finite,
#         fp32_block,
#         0.0  # Replace NaN/Inf with zero
#     )
    
#     # NUMERICAL FIX 6: Final clamping before BF16 conversion
#     # This prevents overflow when converting to BF16
#     fp32_block = tl.where(
#         mask,
#         tl.minimum(tl.maximum(fp32_block, -MAX_BF16_SAFE), MAX_BF16_SAFE),
#         0.0
#     )
    
#     # Convert to BF16
#     bf16_block = tl.cast(fp32_block, tl.bfloat16)
    
#     # Store the dequantized blocks
#     output_ptrs = output_ptr + tile_m * BLOCK_SIZE_M * output_stride0 + tile_n * BLOCK_SIZE_N * output_stride1
#     output_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * output_stride0 + \
#                     tl.arange(0, BLOCK_SIZE_N)[None, :] * output_stride1
#     output_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < bsz * padded_seq_len) & \
#                   (tl.arange(0, BLOCK_SIZE_N)[None, :] < dim)
    
#     tl.store(output_ptrs + output_offsets, bf16_block, mask=output_mask)


# def dequant_compressed_kv_per_token(
#     q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
# ):
#     """
#     Dequantize FP8 KV-Cache tensor with scale.
#     Args:
#         q: [bsz, max_seq_len, dim] - FP8 quantized tensor
#         scale: [bsz, max_seq_len, num_blocks] - scale factors
#         seq_len: actual sequence length to dequantize
#         BLOCK_SIZE: quantization block size (default 128)
#     Returns:
#         x: [bsz, padded_seq_len, dim] - dequantized BF16 tensor
    
#     Notes:
#         - Returns x with padded_seq_len (nearest multiple of 64 >= seq_len)
#         - Dequantizes first seq_len elements of max_seq_len tensor
#         - Includes numerical safety checks for FP8->FP32->BF16 conversion
#     """
#     assert q.is_cuda and scale.is_cuda, "Tensors must be on CUDA"
#     assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32, \
#         f"Expected q to be float8_e4m3fn and scale to be float32, got {q.dtype} and {scale.dtype}"
#     assert seq_len >= 0, f"seq_len must be >= 0, got {seq_len}"
#     assert seq_len <= q.size(1), f"seq_len must be <= max_seq_len, got {seq_len} > {q.size(1)}"
#     assert q.is_contiguous() and scale.is_contiguous(), "q and scale must be contiguous tensors"
#     assert q.dim() == scale.dim() == 3, f"Expected 3D tensors, got q: {q.dim()}D, scale: {scale.dim()}D"
    
#     bsz, max_seq_len, dim = q.shape
#     _, _, num_blocks = scale.shape
    
#     # Verify scale dimensions match expected number of blocks
#     expected_num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
#     assert num_blocks == expected_num_blocks, \
#         f"Scale has {num_blocks} blocks but expected {expected_num_blocks} for dim={dim}, BLOCK_SIZE={BLOCK_SIZE}"
    
#     padded_seq_len = ceil(seq_len / 64) * 64  # Nearest multiple of 64
    
#     # Initialize output with zeros (safer than arbitrary values)
#     result = torch.zeros((bsz * padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)
    
#     # Set up kernel launch parameters
#     BLOCK_SIZE_M = 64
#     BLOCK_SIZE_N = 64
#     num_blocks_m = ceil(bsz * padded_seq_len / BLOCK_SIZE_M)
#     num_blocks_n = ceil(dim / BLOCK_SIZE_N)
    
#     # Flatten batch dimension for kernel processing
#     q_flat = q.view(-1, dim)
#     scale_flat = scale.view(-1, scale.shape[-1])
    
#     grid = (num_blocks_m, num_blocks_n)
    
#     # Launch kernel with numerical safety
#     dequant_compressed_kv_per_token_kernel[grid](
#         q_flat, scale_flat, result,
#         dim, BLOCK_SIZE,
#         bsz, padded_seq_len, max_seq_len,
#         BLOCK_SIZE_M, BLOCK_SIZE_N,
#         q_flat.stride(0), q_flat.stride(1),
#         scale_flat.stride(0), scale_flat.stride(1),
#         result.stride(0), result.stride(1)
#     )
    
#     return result.view(bsz, padded_seq_len, dim)



""" V4 """
# @triton.jit
# def dequant_compressed_kv_per_token_kernel(
#     kv_ptr, scale_ptr, output_ptr,
#     dim: tl.constexpr, 
#     quant_block_size: tl.constexpr,
#     seq_len: tl.constexpr,
#     bsz: tl.constexpr, 
#     padded_seq_len: tl.constexpr, 
#     max_seq_len: tl.constexpr,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     kv_stride0, kv_stride1, kv_stride2,
#     scale_stride0, scale_stride1, scale_stride2,
#     output_stride0, output_stride1, output_stride2,
# ):
#     # Get batch index
#     batch_idx = tl.program_id(0)
#     # Get sequence tile index
#     seq_tile_idx = tl.program_id(1)
#     # Get dimension tile index
#     dim_tile_idx = tl.program_id(2)
    
#     # Calculate starting positions
#     seq_start = seq_tile_idx * BLOCK_SIZE_M
#     dim_start = dim_tile_idx * BLOCK_SIZE_N
    
#     # Create offset arrays
#     seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_M)
#     dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_N)
    
#     # Create masks - ensure we respect actual sequence length
#     seq_mask = seq_offsets < seq_len  # Only process up to actual seq_len
#     dim_mask = dim_offsets < dim
#     mask_2d = seq_mask[:, None] & dim_mask[None, :]
    
#     # Load FP8 data
#     kv_base = kv_ptr + batch_idx * kv_stride0
#     kv_offsets = seq_offsets[:, None] * kv_stride1 + dim_offsets[None, :] * kv_stride2
    
#     # Load FP8 values
#     fp8_block = tl.load(
#         kv_base + kv_offsets,
#         mask=mask_2d,
#         other=tl.cast(0.0, tl.float8e4nv),
#         cache_modifier='.cg'
#     )
    
#     # Convert FP8 to float32 with proper range handling
#     fp32_block = tl.cast(fp8_block, tl.float32)
    
#     # Initialize output block
#     output_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     # Scale base pointer
#     scale_base = scale_ptr + batch_idx * scale_stride0
    
#     # Calculate quantization block indices that intersect with current tile
#     first_quant_block = dim_start // quant_block_size
#     last_quant_block = (dim_start + BLOCK_SIZE_N - 1) // quant_block_size
    
#     # Process each quantization block
#     for quant_block_idx in range(first_quant_block, last_quant_block + 1):
#         # Calculate the dimension range for this quantization block
#         block_dim_start = quant_block_idx * quant_block_size
#         block_dim_end = tl.minimum(block_dim_start + quant_block_size, dim)
        
#         # Check if this quantization block intersects with current tile
#         tile_dim_end = tl.minimum(dim_start + BLOCK_SIZE_N, dim)
#         if block_dim_start >= tile_dim_end or block_dim_end <= dim_start:
#             continue
            
#         # Create mask for dimensions that belong to this quantization block
#         dim_in_block = (dim_offsets >= block_dim_start) & (dim_offsets < block_dim_end)
#         block_mask = mask_2d & dim_in_block[None, :]
        
#         # Load scales for this quantization block
#         scale_offsets = seq_offsets * scale_stride1 + quant_block_idx * scale_stride2
#         scales = tl.load(
#             scale_base + scale_offsets,
#             mask=seq_mask,
#             other=1.0,  # Default scale of 1.0 for invalid positions
#             cache_modifier='.cg'
#         )
        
#         # Validate and clamp scales to prevent numerical issues
#         # Replace NaN/Inf with 1.0, and very small scales with 1.0 to avoid overflow
#         scales_valid = (scales == scales) & tl.isfinite(scales) & (tl.abs(scales) > 1e-8)
#         scales = tl.where(scales_valid, scales, 1.0)
        
#         # Clamp scales to reasonable range to prevent overflow/underflow
#         scales = tl.minimum(tl.maximum(scales, 1e-6), 1e6)
        
#         # Apply dequantization: quantized_value * scale
#         dequantized = fp32_block * scales[:, None]
        
#         # Check for numerical issues in dequantized values
#         dequantized_valid = (dequantized == dequantized) & tl.isfinite(dequantized)
#         dequantized = tl.where(dequantized_valid, dequantized, 0.0)
        
#         # Clamp to bfloat16 range to prevent overflow during conversion
#         # BF16 range is approximately ±3.39e38, but we use a more conservative range
#         dequantized = tl.minimum(tl.maximum(dequantized, -65504.0), 65504.0)
        
#         # Update output block only for valid positions in this quantization block
#         output_block = tl.where(block_mask, dequantized, output_block)
    
#     # Convert to bfloat16 with final clamping
#     output_block_bf16 = tl.cast(output_block, tl.bfloat16)
    
#     # Store the result only to valid positions within padded sequence length
#     output_base = output_ptr + batch_idx * output_stride0
#     output_offsets = seq_offsets[:, None] * output_stride1 + dim_offsets[None, :] * output_stride2
    
#     # Final storage mask: within padded_seq_len and valid dimensions
#     storage_mask = (seq_offsets[:, None] < padded_seq_len) & (dim_offsets[None, :] < dim)
    
#     tl.store(
#         output_base + output_offsets,
#         output_block_bf16,
#         mask=storage_mask
#     )


# def dequant_compressed_kv_per_token(
#     q: torch.Tensor, scale: torch.Tensor, seq_len: int, BLOCK_SIZE: int = 128
# ):
#     """
#     Dequantize FP8 compressed KV cache with per-token quantization.
    
#     Args:
#         q: FP8 quantized tensor of shape (bsz, max_seq_len, dim)
#         scale: Scaling factors of shape (bsz, max_seq_len, num_quant_blocks)
#         seq_len: Actual sequence length to process
#         BLOCK_SIZE: Quantization block size (should be 128)
    
#     Returns:
#         Dequantized BF16 tensor of shape (bsz, padded_seq_len, dim)
#     """
#     assert q.is_cuda and scale.is_cuda, "Tensors must be on CUDA"
#     assert q.dtype == torch.float8_e4m3fn, f"Expected FP8 input, got {q.dtype}"
#     assert scale.dtype == torch.float32, f"Expected float32 scales, got {scale.dtype}"
#     assert seq_len >= 0 and seq_len <= q.size(1), f"Invalid seq_len: {seq_len}"
#     assert q.is_contiguous() and scale.is_contiguous(), "Tensors must be contiguous"
    
#     bsz, max_seq_len, dim = q.shape
    
#     # Pad sequence length to multiple of 64 for efficient memory access
#     padded_seq_len = ((seq_len + 63) // 64) * 64
    
#     # Calculate expected number of quantization blocks
#     num_quant_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
#     expected_scale_shape = (bsz, max_seq_len, num_quant_blocks)
    
#     assert scale.shape == expected_scale_shape, \
#         f"Scale shape mismatch: expected {expected_scale_shape}, got {scale.shape}"
    
#     # Validate input tensors for numerical issues
#     if torch.isnan(scale).any():
#         print("Warning: NaN values detected in scale tensor, replacing with 1.0")
#         scale = torch.where(torch.isnan(scale), torch.ones_like(scale), scale)
    
#     if torch.isinf(scale).any():
#         print("Warning: Inf values detected in scale tensor, replacing with 1.0")
#         scale = torch.where(torch.isinf(scale), torch.ones_like(scale), scale)
    
#     # Check for extremely small scales that could cause overflow
#     small_scale_mask = torch.abs(scale) < 1e-8
#     if small_scale_mask.any():
#         print("Warning: Very small scale values detected, replacing with 1.0")
#         scale = torch.where(small_scale_mask, torch.ones_like(scale), scale)
    
#     # Initialize result tensor
#     result = torch.zeros(
#         (bsz, padded_seq_len, dim), 
#         device=q.device, 
#         dtype=torch.bfloat16
#     )
    
#     # Configure grid for 3D tiling
#     BLOCK_SIZE_M = 64  # Sequence dimension tile size
#     BLOCK_SIZE_N = 64  # Feature dimension tile size
    
#     grid = (
#         bsz,
#         (padded_seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
#         (dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
#     )
    
#     # Launch kernel
#     dequant_compressed_kv_per_token_kernel[grid](
#         q, scale, result,
#         dim, BLOCK_SIZE, seq_len,
#         bsz, padded_seq_len, max_seq_len,
#         BLOCK_SIZE_M, BLOCK_SIZE_N,
#         q.stride(0), q.stride(1), q.stride(2),
#         scale.stride(0), scale.stride(1), scale.stride(2),
#         result.stride(0), result.stride(1), result.stride(2)
#     )
    
#     return result

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

		result = torch.empty((bsz * padded_seq_len, dim), device=q.device, dtype=torch.bfloat16)
		# Assign max value to result tensor
		# result.fill_(1e10)

		# Construct 3D triton grid: bsz, seq_len, num_blocks
		BLOCK_SIZE_M = 256
		BLOCK_SIZE_N = 128
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