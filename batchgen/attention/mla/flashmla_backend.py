import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_mla import (
	flash_mla_with_kvcache,
	get_mla_metadata,
)
from .rotary_embedding import rotary_pos_emb
import logging
from ..quantization import dequant_per_token_triton, dequant_per_token_return_with_max_seqlen_pad
import triton
import triton.language as tl
from ...moe.fused_dequant_gemm import fused_fp8_bf16_gemm
from .fused_bhd_hdc_kernel import fused_bhd_hdc, fused_bhd_hdc_inplace
from .fused_rotary_embedding import fused_rotary_embedding, fused_rotary_embedding_inplace
from ...quantization.fp8e4m3 import (
	per_token_blocked_quantize_bf16_to_fp8, 
	dequant_compressed_kv_per_token_with_length, 
	dequant_compressed_kv_per_token_with_length_v2,
	dequant_compressed_kv_per_token
)

from typing import Optional, Tuple
import math

try:
	from batchgen_kernels.attention.dsa import FP8AbsorbWeights, fp8_q_absorb, fp8_out_absorb
	_HAS_FP8_ABSORB = True
except (ImportError, Exception):
	_HAS_FP8_ABSORB = False

def quant_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Quantize a [bsz, seq, 576] BF16 tensor to FP8 per 128-element block.
	
	This function exactly reproduces the C++ quantization logic:
	- Processes tensor in 128-element blocks
	- Handles partial last block correctly
	- Uses proper FP8 constants and clamping
	
	Args:
		x: Input tensor of shape [bsz, seq, 576] with dtype bfloat16
		
	Returns:
		q: Quantized tensor [bsz, seq, 576] with dtype float8_e4m3fn
		s: Scale factors [bsz, seq, num_blocks] with dtype float32
	"""
	# Input validation (matching C++ TORCH_CHECK statements)
	assert x.dtype == torch.bfloat16, f"Input tensor must be of dtype BFloat16, got {x.dtype}"
	assert x.size(-1) == 576, f"Last dimension of input tensor must be 576, got {x.size(-1)}"
	assert x.is_contiguous(), "Input tensor must be contiguous"
	assert x.dim() == 3, f"Input tensor must have 3 dimensions, got {x.dim()}"
	
	# Extract dimensions
	bsz, seq_len, dim = x.shape
	M = bsz * seq_len
	
	# Block configuration (matching C++ logic exactly)
	block_size = 128
	num_full_blocks = dim // block_size  # 576 // 128 = 4
	has_last_block = (dim % block_size != 0)
	last_block_size = dim % block_size  # 576 % 128 = 64
	num_blocks = num_full_blocks + (1 if has_last_block else 0)  # 4 + 1 = 5
	
	print(f"Processing tensor: {x.shape}")
	print(f"Block configuration: {num_full_blocks} full blocks + {'1 partial' if has_last_block else '0 partial'} blocks")
	print(f"Last block size: {last_block_size}, Total blocks: {num_blocks}")
	
	# FP8 E4M3FN maximum finite value
	# For float8_e4m3fn: sign(1) + exp(4) + mantissa(3), max finite = 448.0
	FP8_MAX = 448.0
	
	# Flatten and convert to float32 (matching C++ logic)
	x_flat = x.view(M, dim).to(torch.float32)
	
	# Prepare output tensors
	scale_flat = torch.empty(M, num_blocks, dtype=torch.float32, device=x.device)
	q_flat = torch.empty(M, dim, dtype=torch.float8_e4m3fn, device=x.device)
	
	# Process each block independently (exactly matching C++ loop)
	for b in range(num_blocks):
		start = b * block_size
		length = block_size if b < num_full_blocks else last_block_size
		
		print(f"Processing block {b}: elements {start}:{start+length} (length={length})")
		
		# Extract block (matching C++ narrow operation)
		x_block = x_flat[:, start:start+length]  # Shape: [M, length]
		
		# Compute absolute max per sequence (matching C++ amax)
		amax = torch.amax(torch.abs(x_block), dim=1)  # Shape: [M]
		
		# Clamp minimum (matching C++ clamp with 1e-6f)
		amax = torch.clamp(amax, min=1e-6)
		
		# Compute scale for this block (matching C++ scale computation)
		scale = amax / FP8_MAX
		scale_flat[:, b] = scale
		
		# Quantize block (matching C++ quantization)
		# Broadcast scale for division: [M, 1] for proper broadcasting
		y = x_block / scale.unsqueeze(1)
		q_block = y.to(torch.float8_e4m3fn)
		
		# Copy quantized block back to output
		q_flat[:, start:start+length] = q_block
	
	# Reshape back to original dimensions
	q = q_flat.view(bsz, seq_len, dim)
	scale = scale_flat.view(bsz, seq_len, num_blocks)
	
	return q, scale


def dequant_per_token(q, scale):
	"""
	Dequantize a [bsz, seq, 576] FP8 tensor back to BF16 per 128-element block.
	
	Args:
		q: Quantized tensor [bsz, seq, 576] with dtype torch.float8_e4m3fn
		scale: Scale factors [bsz, seq, num_blocks] with dtype torch.float32
		
	Returns:
		x_bf16: Dequantized tensor [bsz, seq, 576] with dtype torch.bfloat16
	"""
	# Input validation
	assert q.dtype == torch.float8_e4m3fn, "Quantized tensor must be of dtype torch.float8_e4m3fn"
	assert scale.dtype == torch.float32, "Scale tensor must be of dtype torch.float32"
	assert q.dim() == 3 and scale.dim() == 3, "Input tensors must be 3D"
	assert q.size(0) == scale.size(0) and q.size(1) == scale.size(1), \
		"Batch and sequence dimensions must match between q and scale"
	
	bsz, seq_len, dim = q.shape
	M = bsz * seq_len
	
	block_size = 128
	num_full_blocks = dim // block_size
	has_last_block = (dim % block_size != 0)
	last_block_size = dim % block_size
	num_blocks = num_full_blocks + (1 if has_last_block else 0)
	
	assert scale.size(2) == num_blocks, \
		"Scale tensor last dimension must match number of blocks"
	
	# Flatten tensors
	q_flat = q.view(M, dim)
	scale_flat = scale.view(M, num_blocks)
	
	# Prepare output buffer in float32
	x_flat = torch.empty((M, dim), dtype=torch.float32, device=q.device)
	
	# Process each block
	for b in range(num_blocks):
		start = b * block_size
		length = block_size if b < num_full_blocks else last_block_size
		
		# Extract block of q, upcast to float
		q_block = q_flat[:, start:start+length].to(torch.float32)
		
		# Get scale for this block [M, 1]
		s_block = scale_flat[:, b:b+1]
		
		# Reconstruct: broadcast multiply
		x_block = q_block * s_block
		
		# Store back
		x_flat[:, start:start+length] = x_block
	
	# Cast back to BF16 and reshape
	x_bf16 = x_flat.to(torch.bfloat16).view(bsz, seq_len, dim)
	return x_bf16


# @triton.jit
# def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
# 	"""
# 	Dequantizes weights using the provided scaling factors and stores the result.

# 	Args:
# 		x_ptr (tl.pointer): Pointer to the quantized weights.
# 		s_ptr (tl.pointer): Pointer to the scaling factors.
# 		y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
# 		M (int): Number of rows in the weight matrix.
# 		N (int): Number of columns in the weight matrix.
# 		BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

# 	Returns:
# 		None
# 	"""
# 	pid_m = tl.program_id(axis=0)
# 	pid_n = tl.program_id(axis=1)
# 	n = tl.cdiv(N, BLOCK_SIZE)
# 	offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# 	offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# 	offs = offs_m[:, None] * N + offs_n[None, :]
# 	mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
# 	x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
# 	s = tl.load(s_ptr + pid_m * n + pid_n)
# 	y = x * s
# 	tl.store(y_ptr + offs, y, mask=mask)
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, 
						  stride_sm, stride_sn,  # <--- Add these arguments
						  BLOCK_SIZE: tl.constexpr):
	pid_m = tl.program_id(axis=0)
	pid_n = tl.program_id(axis=1)
	
	# 1. Calculate offsets for X (Weights)
	offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	offs = offs_m[:, None] * N + offs_n[None, :]
	mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

	# 2. Load X
	x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)

	# 3. Load S (Scales) using explicit strides
	# We assume s is shape (Ceil(M/B), Ceil(N/B))
	# No mask needed here IF s is sized to match the grid (using cdiv).
	# If s is sized using floor div, you are missing data for the edges.
	s_offset = pid_m * stride_sm + pid_n * stride_sn
	s = tl.load(s_ptr + s_offset)

	# 4. Compute and Store
	y = x * s
	tl.store(y_ptr + offs, y, mask=mask)


# def deepseek_v3_dequantization(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
# 	"""
# 	Dequantizes the given weight tensor using the provided scale tensor.

# 	Args:
# 		x (torch.Tensor): The quantized weight tensor of shape (M, N).
# 		s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
# 		block_size (int, optional): The block size to use for dequantization. Defaults to 128.

# 	Returns:
# 		torch.Tensor: The dequantized weight tensor of the same shape as `x`.

# 	Raises:
# 		AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
# 	"""
# 	assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
# 	assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions but got {} and {}'.format(x.shape, s.shape)
# 	M, N = x.size()
# 	y = torch.empty_like(x, dtype=torch.bfloat16)
# 	grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
# 	weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
# 	return y

def deepseek_v3_dequantization(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
	assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
	assert x.dim() == 2 and s.dim() == 2
	
	M, N = x.size()
	
	# VALIDATION: Ensure s is large enough to cover the ceil division
	expected_s_m = (M + block_size - 1) // block_size
	expected_s_n = (N + block_size - 1) // block_size
	
	assert s.shape[0] == expected_s_m and s.shape[1] == expected_s_n, \
		f"Scale tensor shape mismatch. Expected ({expected_s_m}, {expected_s_n}), got {s.shape}"

	y = torch.empty_like(x, dtype=torch.bfloat16)
	
	grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
	
	weight_dequant_kernel[grid](
		x, s, y, 
		M, N, 
		s.stride(0), s.stride(1), # <--- Pass strides explicitly
		BLOCK_SIZE=block_size
	)
	return y

@torch.inference_mode()
def mla_decoding_flashmla_(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
	scale
):
	from batchgen.models.wrappers.attention import AttnWrapperBase
	# Use cache_seqlens directly instead of attention_mask.sum()
	if AttnWrapperBase.cache_seqlens is not None:
		cache_seqlens = AttnWrapperBase.cache_seqlens
	else:
		cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)

	block_size = 64
	max_seqlen = cache_seqlens.max().item()
	max_seqlen_pad = ((max_seqlen + block_size - 1) // block_size) * block_size
	if max_seqlen_pad >= past_key_states.size(1):
		compressed_kv = dequant_per_token_return_with_max_seqlen_pad(
			past_key_states, scale, max_seqlen_pad
		)
	else:
		compressed_kv = past_key_states[:, :max_seqlen_pad, :].contiguous()
		scale = scale[:, :max_seqlen_pad, :].contiguous()
		compressed_kv = dequant_per_token_triton(compressed_kv, scale)

	bsz, q_len, _ = hidden_states.size()
	q_position_id = (cache_seqlens.to(torch.int64) - 1).unsqueeze(-1)
	kv_len = max_seqlen_pad

	
	# Log norm weight dtype
	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	# q = fused_fp8_bf16_gemm(hidden_states, self.q_a_proj.weight, scale["q_a_proj.weight_scale_inv"])
	# q = self.q_a_layernorm(q)
	# q = fused_fp8_bf16_gemm(q, self.q_b_proj.weight, scale["q_b_proj.weight_scale_inv"])
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	# new_compressed_kv = fused_fp8_bf16_gemm(hidden_states, self.kv_a_proj_with_mqa.weight, scale["kv_a_proj_with_mqa.weight_scale_inv"])

	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	k_pe = k_pe.view(bsz, 1, q_len, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=kv_len)
	# k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_id)
	k_pe = fused_rotary_embedding(k_pe, cos, sin, q_position_id)

	kv = self.kv_a_layernorm(kv)
	k_pe = k_pe.view(bsz, q_len, self.qk_rope_head_dim)
	offload_kv = torch.cat([kv, k_pe], dim=-1)

	# batch_indices = torch.arange(bsz, device=hidden_states.device)
	# compressed_kv[batch_indices, q_position_id[:, 0], :] = offload_kv[:, 0, :]
	compressed_kv[torch.arange(bsz, device=compressed_kv.device), q_position_id.squeeze(-1)] = offload_kv.squeeze(1)
	# compressed_kv.scatter_(
	# 	dim=1,
	# 	index=q_position_id[:, 0:1].unsqueeze(-1).expand(-1, -1, compressed_kv.size(-1)),
	# 	src=offload_kv[:, 0:1, :]
	# )
	compressed_kv = compressed_kv.view(bsz, max_seqlen_pad, 1, 576)
	
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	# q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	# out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]
	q_absorb, out_absorb = torch.split(
		kv_b_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=1)

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, qk_head_dim,
		dtype=compressed_kv.dtype,
		device=compressed_kv.device,
	)

	
	# query_states[:, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	q_nope = q_nope.view(bsz, self.num_heads, self.qk_nope_head_dim)
	# query_states[:, :, : self.kv_lora_rank] = fused_bhd_hdc(q_nope, q_absorb)
	fused_bhd_hdc_inplace(q_nope, q_absorb, query_states)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_id)
	query_states[:, :, self.kv_lora_rank :] = q_pe.view(bsz, self.num_heads, self.qk_rope_head_dim)
	# fused_rotary_embedding_inplace(q_pe, cos, sin, q_position_id, query_states, self.kv_lora_rank)
	
	query_states = query_states.view(
		bsz, q_len, self.num_heads, qk_head_dim
	)



	# Pad the compressed_kv_ref tensor to the maximum sequence length
	# if max_seqlen_pad > kv_len:
	# 	compressed_kv = torch.cat(
	# 		[
	# 			compressed_kv,
	# 			torch.full(
	# 				(bsz, max_seqlen_pad - kv_len, 1, compressed_kv.size(-1)),
	# 				# float("nan"),
	# 				0,
	# 				dtype=compressed_kv.dtype,
	# 				device=compressed_kv.device,
	# 			),
	# 		],
	# 		dim=1,
	# 	)
	# else:
	# 	compressed_kv = compressed_kv[:, :max_seqlen_pad, :, :]

	# block_table = torch.arange(
	# 	bsz * max_seqlen_pad // block_size, dtype=torch.int32
	# ).view(bsz, max_seqlen_pad // block_size).to(compressed_kv.device)
	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	"""
	flash_mla_with_kvcache
	Arguments:
		q: (batch_size, seq_len_q, num_heads_q, head_dim).
		k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
		block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
		cache_seqlens: (batch_size), torch.int32.
		head_dim_v: Head dimension of v.
		tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
		num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
		softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
		causal: bool. Whether to apply causal attention mask.

	Returns:
		out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
		softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
	"""
	try:
		attn_out, attention_weights = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	
	# attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	out_absorb = out_absorb.transpose(1,2).contiguous()
	attn_output = fused_bhd_hdc(attn_out.squeeze(1), out_absorb)


	# attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.unsqueeze(1)
	attn_output = attn_output.view(
		bsz, q_len, self.num_heads * self.v_head_dim
	)
	attn_output = self.o_proj(attn_output)

	return (
		attn_output,
		offload_kv,
		torch.tensor([], device=hidden_states.device),
	)


# from batchgen.gemm.w8a8 import w8a8_gemm
# from batchgen.gemm.w8a8_gemm_no_persistent_cta import w8a8_gemm_dispatch as w8a8_gemm
from batchgen.gemm.w8a8_gemm_split_k import w8a8_gemm_dispatch as w8a8_gemm
from batchgen.attention.mla.fa3_backend import act_quant
@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	q_position_ids: torch.Tensor,
	scale: torch.Tensor,
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	weight_scale: dict = None
):
	"""
		MLA decoding function using FlashMLA as the attention mechanism backend.
		Args:
			hidden_states (torch.Tensor): The input hidden states of shape (batch_size, 1, hidden_size).
			past_key_states (torch.Tensor): The past key states of shape (batch_size, max_seqlen, kv_dim)
			past_value_states (torch.Tensor): None. Placeholder.
			attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
			position_ids (torch.Tensor): The position ids of shape (batch_size, 1)
			scale (torch.Tensor): The scale tensor for kv per token quantization. [batch_size, max_seqlen, ceil(kv_dim // 128)]
	
		Note:
			- The past_key_states has the shape of (batch_size, max_seqlen, kv_dim).
			Where max_seqlen is the nearest multiple of 64(kv block size) that is greater than or equal to the full context length.
			- attention_mask has the shape of (batch_size, seq_len).
			Where seq_len is the length of the processed tokens(input prompt(padded) + generated tokens).
			- position_ids record the position of the new kv should be placed in the past compressed kv cache.
			- max_seq_len is reserved for scale. 
			
	"""
	hidden_states = hidden_states.squeeze_(1)
	# Create a block table for the key states
	block_size = 64
	bsz = hidden_states.size(0)
	compressed_kv = dequant_compressed_kv_per_token(past_key_states, scale, max_seqlen)
	max_seqlen_pad = compressed_kv.size(1)

	# q_position_id = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	# kv_len = attention_mask.size(-1)
	# q_position_ids = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	# self.q_a_proj.weight.data = deepseek_v3_dequantization(
	# 	self.q_a_proj.weight.data, weight_scale["q_a_proj.weight_scale_inv"]
	# )
	# self.q_b_proj.weight.data = deepseek_v3_dequantization(
	# 	self.q_b_proj.weight.data, weight_scale["q_b_proj.weight_scale_inv"]
	# )
	x_fp8, x_scale = act_quant(hidden_states)
	x = w8a8_gemm(x_fp8, x_scale, self.q_a_proj.weight.data, weight_scale['q_a_proj.weight_scale_inv'])
	new_compressed_kv = w8a8_gemm(x_fp8, x_scale, self.kv_a_proj_with_mqa.weight.data, weight_scale['kv_a_proj_with_mqa.weight_scale_inv'])
	x = self.q_a_layernorm(x)
	x_fp8, x_scale = act_quant(x)
	q = w8a8_gemm(x_fp8, x_scale, self.q_b_proj.weight.data, weight_scale['q_b_proj.weight_scale_inv'])

	# q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, 1, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	# self.kv_a_proj_with_mqa.weight.data = deepseek_v3_dequantization(
	# 	self.kv_a_proj_with_mqa.weight.data, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]
	# )
	# new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=max_seqlen_pad)
	k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_ids)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)
	kv = self.kv_a_layernorm(kv)
	k_pe = k_pe.view(bsz, 1, self.qk_rope_head_dim)
	kv = kv.view(bsz, 1, self.kv_lora_rank)
	offload_kv = torch.cat([kv, k_pe], dim=-1)
	
	batch_indices = torch.arange(bsz, device=hidden_states.device)
	compressed_kv[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	compressed_kv = compressed_kv.view(bsz, max_seqlen_pad, 1, 576)

	# Quantize and write to past_key_states
	new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
	
	# kv_b_proj = self.kv_b_proj.weight.view(
	# 	self.num_heads, -1, self.kv_lora_rank
	# )
	kv_b_proj = deepseek_v3_dequantization(
		self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	).view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, 1, qk_head_dim,
		dtype=compressed_kv.dtype,
		device=compressed_kv.device,
	)

	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(
		bsz, 1, self.num_heads, qk_head_dim
	)


	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	try:
		attn_out, _ = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)


	attn_output = attn_output.transpose(1, 2).contiguous()
	# attn_output = attn_output.view(
	# 	bsz, 1, self.num_heads * self.v_head_dim
	# )
	# self.o_proj.weight.data = deepseek_v3_dequantization(
	# 	self.o_proj.weight.data, weight_scale["o_proj.weight_scale_inv"]
	# )
	# attn_output = self.o_proj(attn_output)

	attn_output = attn_output.view(
		bsz, self.num_heads * self.v_head_dim
	)
	attn_output_fp8, attn_out_scale = act_quant(attn_output)
	attn_output = w8a8_gemm(
		attn_output_fp8, attn_out_scale, self.o_proj.weight.data, weight_scale["o_proj.weight_scale_inv"]
	)

	return attn_output.unsqueeze_(1), past_key_states, scale


# NOTE: mla_decoding_flashmla_attn_mode_3_bak and update_causal_mask
# were moved to local_archive/deprecated/


@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn_(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	q_position_ids: torch.Tensor,
	scale: torch.Tensor,
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	weight_scale: dict = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	MLA decoding function with FP8 KV cache quantization and BF16 attention computation.

	Args:
		hidden_states: Input hidden states of shape (batch_size, 1, hidden_size).
		past_key_states: Quantized (FP8) past compressed key states cache of shape (batch_size, max_seqlen, kv_dim).
		past_value_states: Not used. Placeholder for compatibility.
		attention_mask: Attention mask of shape (batch_size, seq_len).
		q_position_ids: Position ids of shape (batch_size, 1) indicating where to write the new KV cache.
		scale: Dequantization scale for past_key_states.
		cache_seqlens: Sequence lengths of the cache.
		max_seqlen: Maximum sequence length of the cache.
		weight_scale: Weight quantization scales dictionary.

	Returns:
		Tuple of (attn_output, updated past_key_states, updated scale).
	"""
	bsz, q_len, _ = hidden_states.size()
	assert q_len == 1, "The PyTorch MLA decoding backend currently only supports a query length of 1."

	# Dequantize KV cache
	compressed_kv_ref = dequant_compressed_kv_per_token(past_key_states, scale, max_seqlen)
	max_seqlen_pad = compressed_kv_ref.size(1)

	# Query and key-value projection with W8A8 quantization
	hidden_states = hidden_states.squeeze(1)
	hidden_states, hidden_states_scale = act_quant(hidden_states)
	
	q = w8a8_gemm(hidden_states, hidden_states_scale, self.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"])
	new_compressed_kv = w8a8_gemm(hidden_states, hidden_states_scale, self.kv_a_proj_with_mqa.weight, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]).view(bsz, 1, -1)
	
	q = self.q_a_layernorm(q)
	q, q_scale = act_quant(q)
	q = w8a8_gemm(q, q_scale, self.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"])

	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

	# Split and process KV
	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	
	# Apply RoPE
	k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=max_seqlen_pad)
	k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_ids)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)
	
	kv = self.kv_a_layernorm(kv)
	k_pe = k_pe.view(bsz, 1, self.qk_rope_head_dim)
	offload_kv = torch.cat([kv, k_pe], dim=-1)
	
	# Update cache with new KV
	batch_indices = torch.arange(bsz, device=hidden_states.device)
	compressed_kv_ref[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	
	# Quantize and update past_key_states
	new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
	
	# Prepare absorb weights (cached after first call)
	if _HAS_FP8_ABSORB and not hasattr(self, '_fp8_absorb_weights'):
		kv_b_proj_bf16 = deepseek_v3_dequantization(
			self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
		).view(self.num_heads, -1, self.kv_lora_rank)
		q_absorb_w = kv_b_proj_bf16[:, : self.qk_nope_head_dim, :]
		out_absorb_w = kv_b_proj_bf16[:, self.qk_nope_head_dim :, :]
		self._fp8_absorb_weights = FP8AbsorbWeights(q_absorb_w, out_absorb_w)

	if _HAS_FP8_ABSORB:
		# FP8 WGMMA q_absorb: [B, H, 192] → [B, H, 512]
		q_nope_3d = q_nope.view(bsz, self.num_heads, self.qk_nope_head_dim)
		q_absorbed = fp8_q_absorb(q_nope_3d, self._fp8_absorb_weights)

		qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
		query_states = torch.empty(
			bsz, self.num_heads, 1, qk_head_dim,
			dtype=compressed_kv_ref.dtype,
			device=compressed_kv_ref.device,
		)
		query_states[:, :, 0, : self.kv_lora_rank] = q_absorbed
		query_states[:, :, :, self.kv_lora_rank :] = q_pe
	else:
		kv_b_proj = deepseek_v3_dequantization(
			self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
		).view(self.num_heads, -1, self.kv_lora_rank)
		q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
		out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

		qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
		query_states = torch.empty(
			bsz, self.num_heads, 1, qk_head_dim,
			dtype=compressed_kv_ref.dtype,
			device=compressed_kv_ref.device,
		)
		query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
		query_states[:, :, :, self.kv_lora_rank :] = q_pe

	query_states = query_states.view(
		bsz, 1, self.num_heads, qk_head_dim
	)

	assert qk_head_dim == 576, f"qk_head_dim should be 576, but got {qk_head_dim}"

	block_size = 64
	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv_ref.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv_ref.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv_ref.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)
	try:
		attn_out, _ = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			causal = True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise

	if _HAS_FP8_ABSORB:
		# FP8 WGMMA out_absorb: [B, 1, H, 512] → [B, 1, H, 256] → transpose to [B, H, 1, 256]
		attn_output = fp8_out_absorb(attn_out, self._fp8_absorb_weights)
		attn_output = attn_output.transpose(1, 2)
	else:
		attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is {attn_output.size()}"
		)
	
	# Final projection with W8A8 quantization
	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
	
	attn_output_fp8, attn_output_scale = act_quant(attn_output)
	attn_output = w8a8_gemm(attn_output_fp8, attn_output_scale, self.o_proj.weight, weight_scale["o_proj.weight_scale_inv"])
	attn_output = attn_output.view(bsz, 1, -1)
	
	return attn_output, past_key_states, scale

# from .fused_kv_kernel import fused_kv_update_and_rope
from .fused_rmsnorm_rope import (
	fused_rmsnorm_rope,
	fused_rmsnorm_rope_cache_update,
	fused_rmsnorm_rope_cache_update_with_q,
	fused_rmsnorm_rope_cache_update_with_q_return_new_kv,
	fused_rmsnorm_rope_with_q,
)
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
@triton.jit
def quantize_and_scatter_kernel(
	input_bf16_ptr,           # [bsz, 1, total_dim] - input
	output_fp8_ptr,           # [bsz, max_seq_len, total_dim] - output
	scale_ptr,                # [bsz, max_seq_len, num_blocks] - output
	position_ids_ptr,         # [bsz, 1]
	bsz,
	max_seq_len,
	total_dim,
	quant_block_size: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	"""
	Quantize bf16 input to fp8 and write directly to target position in cache.
	Eliminates fancy indexing overhead.
	"""
	batch_idx = tl.program_id(0)
	quant_block_idx = tl.program_id(1)
	
	# Constants
	FP8_SAFE_MAX: tl.constexpr = 440.0
	FP8_E4M3_MIN_NORMAL: tl.constexpr = 1.52587890625e-05
	EPSILON: tl.constexpr = 1e-12
	
	pos_id = tl.load(position_ids_ptr + batch_idx)
	
	# Input offset: [bsz, 1, total_dim] with seq_dim=1
	input_offset = batch_idx * total_dim
	
	# Output offset: [bsz, max_seq_len, total_dim]
	output_offset = batch_idx * max_seq_len * total_dim + pos_id * total_dim
	
	# This quantization block's range
	block_start = quant_block_idx * quant_block_size
	block_end = tl.minimum(block_start + quant_block_size, total_dim)
	
	# Find amax for this block
	amax = 0.0
	for i in range(block_start, block_end, BLOCK_SIZE):
		offsets = i + tl.arange(0, BLOCK_SIZE)
		mask = (offsets >= block_start) & (offsets < block_end)
		
		data = tl.load(input_bf16_ptr + input_offset + offsets, mask=mask, other=0.0)
		data_fp32 = data.to(tl.float32)
		amax = tl.maximum(amax, tl.max(tl.abs(data_fp32)))
	
	# Compute scale
	amax = tl.maximum(amax, FP8_E4M3_MIN_NORMAL)
	scale = tl.maximum(amax / FP8_SAFE_MAX, EPSILON)
	
	# Store scale
	num_blocks = (total_dim + quant_block_size - 1) // quant_block_size
	scale_offset = batch_idx * max_seq_len * num_blocks + pos_id * num_blocks + quant_block_idx
	tl.store(scale_ptr + scale_offset, scale)
	
	# Quantize and store
	for i in range(block_start, block_end, BLOCK_SIZE):
		offsets = i + tl.arange(0, BLOCK_SIZE)
		mask = (offsets >= block_start) & (offsets < block_end)
		
		data = tl.load(input_bf16_ptr + input_offset + offsets, mask=mask, other=0.0)
		data_fp32 = data.to(tl.float32)
		
		# Quantize
		data_scaled = data_fp32 / scale
		data_scaled = tl.minimum(data_scaled, FP8_SAFE_MAX)
		data_scaled = tl.maximum(data_scaled, -FP8_SAFE_MAX)
		data_fp8 = data_scaled.to(tl.float8e4nv)
		
		# Write directly to target position
		tl.store(output_fp8_ptr + output_offset + offsets, data_fp8, mask=mask)


def quantize_and_scatter_write(
	input_bf16: torch.Tensor,       # [bsz, 1, total_dim]
	output_fp8: torch.Tensor,       # [bsz, max_seq_len, total_dim]
	scale: torch.Tensor,            # [bsz, max_seq_len, num_blocks]
	position_ids: torch.Tensor,     # [bsz, 1]
	quant_block_size: int = 128,
):
	"""Quantize and write directly to cache, eliminating fancy indexing."""
	bsz, _, total_dim = input_bf16.shape
	max_seq_len = output_fp8.shape[1]
	num_blocks = (total_dim + quant_block_size - 1) // quant_block_size
	
	grid = (bsz, num_blocks)
	
	quantize_and_scatter_kernel[grid](
		input_bf16,
		output_fp8,
		scale,
		position_ids,
		bsz,
		max_seq_len,
		total_dim,
		quant_block_size,
		BLOCK_SIZE=64,
	)
@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	q_position_ids: torch.Tensor,
	scale: torch.Tensor,
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	weight_scale: dict = None,
) -> torch.Tensor:  # Only return attn_output now
	"""
	Modified to update past_key_states in-place via the full tensor reference.
	"""
	bsz, q_len, _ = hidden_states.size()
	assert q_len == 1, "The PyTorch MLA decoding backend currently only supports a query length of 1."
	
	compressed_kv_ref = dequant_compressed_kv_per_token(past_key_states, scale, max_seqlen)
	max_seqlen_pad = compressed_kv_ref.size(1)

	# Work on a VIEW of the batch slice
	# past_key_states = past_key_states_full[batch_start_idx:batch_end_idx]
	_, kv_len, _ = past_key_states.size()

	# --- 2. Query and New Key-Value Projection ---
	hidden_states = hidden_states.squeeze(1)
	hidden_states, hidden_states_scale = act_quant(hidden_states)
	q = w8a8_deepgemm(hidden_states, hidden_states_scale, self.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"])
	new_compressed_kv = w8a8_deepgemm(hidden_states, hidden_states_scale, self.kv_a_proj_with_mqa.weight, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]).view(bsz, 1, -1)
	q = self.q_a_layernorm(q)
	q, q_scale = act_quant(q)
	q = w8a8_deepgemm(q, q_scale, self.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"])

	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
	q_pe = q_pe.contiguous()
	cos, sin = self.rotary_emb(q_pe, seq_len=kv_len)
	# q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)
	
	# This modifies past_key_states in-place (which is a view of the full tensor!)
	offload_kv = fused_rmsnorm_rope_cache_update_with_q_return_new_kv(
		new_compressed_kv,
		compressed_kv_ref,  # This is a VIEW, so modifications affect past_key_states_full!
		q_pe,
		cos,
		sin,
		q_position_ids,
		self.kv_a_layernorm.weight,
		self.kv_lora_rank,
		self.qk_rope_head_dim
	)

	batch_indices = torch.arange(bsz, device=hidden_states.device)
	# Quantize and update past_key_states
	new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
	# quantize_and_scatter_write(
	# 	offload_kv,
	# 	past_key_states,  # fp8 cache
	# 	scale,            # fp32 scales
	# 	q_position_ids,
	# 	quant_block_size=128
	# )

	kv_seqlen = past_key_states.size(1)
	kv_b_proj = deepseek_v3_dequantization(
		self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	).view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, 1, qk_head_dim,
		dtype=compressed_kv_ref.dtype,
		device=compressed_kv_ref.device,
	)
	q_nope = q_nope.squeeze(2)
	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('bhd,hdc->bhc', q_nope, q_absorb).view(bsz, self.num_heads, 1, self.kv_lora_rank)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(bsz, 1, self.num_heads, qk_head_dim)

	# Pad the kv cache to be multiple of 64 (create a NEW tensor for computation)
	# if kv_seqlen % 64 != 0:
	# 	pad_len = 64 - (kv_seqlen % 64)
	# 	past_key_states_padded = torch.cat([
	# 		past_key_states, 
	# 		torch.zeros((bsz, pad_len, past_key_states.size(-1)), device=past_key_states.device, dtype=past_key_states.dtype)
	# 	], dim=1)
	# 	kv_seqlen_padded = past_key_states_padded.size(1)
	# else:
	# 	past_key_states_padded = past_key_states
	# 	kv_seqlen_padded = kv_seqlen
	kv_seqlen_padded = compressed_kv_ref.size(1)
	block_size = 64
	block_table = torch.arange(
		bsz * kv_seqlen_padded // block_size, dtype=torch.int32, device=past_key_states.device
	).view(bsz, kv_seqlen_padded // block_size)

	blocked_k = compressed_kv_ref.view(
		bsz * kv_seqlen_padded // block_size, block_size, 1, compressed_kv_ref.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	try:
		attn_out, _ = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	
	# Apply out_absorb projection
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)
	
	# --- 9. Final Projection and Return ---
	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
	attn_output_fp8, attn_output_scale = act_quant(attn_output)
	attn_output = w8a8_deepgemm(attn_output_fp8, attn_output_scale, self.o_proj.weight, weight_scale["o_proj.weight_scale_inv"])
	attn_output = attn_output.view(bsz, 1, -1)
	

	return attn_output, past_key_states, scale


# NOTE: Deprecated functions moved to local_archive/deprecated/:
# - mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn_bak
# - mla_decoding_flashmla_attn_mode_3_bf16_bak


from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
from batchgen.quantization.block_quantization import act_quant_transposed_scale


@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_bf16(
	self,
	hidden_states: torch.Tensor,  # Already sliced: (batch_slice, 1, hidden_size)
	past_key_states_full: torch.Tensor,  # Full tensor: (full_batch, max_seqlen, kv_dim)
	past_value_states: torch.Tensor,  # Not used, kept for compatibility
	attention_mask: torch.Tensor,  # Already sliced
	q_position_ids: torch.Tensor,  # Already sliced
	cache_seqlens: torch.Tensor,  # Already slicedcp
	max_seqlen: int,
	batch_start_idx: int,  # NEW: which batch slice we're working on
	batch_end_idx: int,    # NEW: end of batch slice
	scale: torch.Tensor = None,
	weight_scale: dict = None,
) -> torch.Tensor:  # Only return attn_output now
	"""
	Modified to update past_key_states in-place via the full tensor reference.
	"""
	bsz, q_len, _ = hidden_states.size()
	assert q_len == 1, "The PyTorch MLA decoding backend currently only supports a query length of 1."
	
	# Work on a VIEW of the batch slice
	past_key_states = past_key_states_full[batch_start_idx:batch_end_idx]
	_, kv_len, _ = past_key_states.size()

	# --- 2. Query and New Key-Value Projection ---
	hidden_states = hidden_states.squeeze(1)
	# hidden_states, hidden_states_scale = act_quant(hidden_states)
	hidden_states, hidden_states_scale = act_quant(hidden_states)

	q = w8a8_deepgemm(hidden_states, hidden_states_scale, self.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"])
	new_compressed_kv = w8a8_deepgemm(hidden_states, hidden_states_scale, self.kv_a_proj_with_mqa.weight, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]).view(bsz, 1, -1)
	q = self.q_a_layernorm(q)
	# q, q_scale = act_quant(q)
	q, q_scale = act_quant(q)
	q = w8a8_deepgemm(q, q_scale, self.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"])

	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
	q_pe = q_pe.contiguous()
	cos, sin = self.rotary_emb(q_pe, seq_len=kv_len)
	
	# This modifies past_key_states in-place (which is a view of the full tensor!)
	fused_rmsnorm_rope_cache_update_with_q(
		new_compressed_kv,
		past_key_states,  # This is a VIEW, so modifications affect past_key_states_full!
		q_pe,
		cos,
		sin,
		q_position_ids,
		self.kv_a_layernorm.weight,
		self.kv_lora_rank,
		self.qk_rope_head_dim
	)

	kv_seqlen = past_key_states.size(1)
	kv_b_proj = deepseek_v3_dequantization(
		self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	).view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, 1, qk_head_dim,
		dtype=past_key_states.dtype,
		device=past_key_states.device,
	)
	q_nope = q_nope.squeeze(2)
	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('bhd,hdc->bhc', q_nope, q_absorb).view(bsz, self.num_heads, 1, self.kv_lora_rank)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(bsz, 1, self.num_heads, qk_head_dim)

	# Pad the kv cache to be multiple of 64 (create a NEW tensor for computation)
	if kv_seqlen % 64 != 0:
		pad_len = 64 - (kv_seqlen % 64)
		past_key_states_padded = torch.cat([
			past_key_states, 
			torch.zeros((bsz, pad_len, past_key_states.size(-1)), device=past_key_states.device, dtype=past_key_states.dtype)
		], dim=1)
		kv_seqlen_padded = past_key_states_padded.size(1)
	else:
		past_key_states_padded = past_key_states
		kv_seqlen_padded = kv_seqlen

	block_size = 64
	block_table = torch.arange(
		bsz * kv_seqlen_padded // block_size, dtype=torch.int32, device=past_key_states.device
	).view(bsz, kv_seqlen_padded // block_size)

	blocked_k = past_key_states_padded.view(
		bsz * kv_seqlen_padded // block_size, block_size, 1, past_key_states_padded.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	try:
		attn_out, _ = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	
	# Apply out_absorb projection
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)
	
	# --- 9. Final Projection and Return ---
	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
	# attn_output_fp8, attn_output_scale = act_quant(attn_output)
	attn_output_fp8, attn_output_scale = act_quant(attn_output)
	attn_output = w8a8_deepgemm(attn_output_fp8, attn_output_scale, self.o_proj.weight, weight_scale["o_proj.weight_scale_inv"])
	attn_output = attn_output.view(bsz, 1, -1)
	
	# past_key_states is already updated in-place (it's a view of past_key_states_full)
	# No need to return it!
	return attn_output


@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv(
	self,
	hidden_states: torch.Tensor,
	q_position_ids: torch.Tensor,
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	weight_scale: Optional[dict] = None,
	gpu_paged_kv_manager: Optional[GPUPagedKVCacheManager] = None,
	layer_idx: int = 0,
	batch_slice: Optional[tuple] = None,  # (start_idx, end_idx) for micro-batching
) -> torch.Tensor:
	"""Variant of the BF16 decoder that writes KV tokens via GPUPagedKVCacheManager.
	
	Args:
		batch_slice: Optional tuple (start_idx, end_idx) indicating which slice of the
			full batch this call represents. When provided, the page table will be
			sliced accordingly to match the input hidden_states batch dimension.
	"""
	if gpu_paged_kv_manager is None:
		raise ValueError(
			"gpu_paged_kv_manager must be provided for page-KV decoding backend",
		)
	
	# ============ INPUT VALIDATION ============
	assert hidden_states is not None, "hidden_states is None"
	assert hidden_states.dim() == 3, f"hidden_states must be 3D, got {hidden_states.dim()}D with shape {hidden_states.shape}"
	
	bsz, q_len, hidden_dim = hidden_states.size()
	
	# Early return for empty batch - this is valid when all sequences are filtered out
	if bsz == 0:
		logging.debug(f"[Layer {layer_idx}] Empty batch (bsz=0), returning empty tensor")
		# Return empty tensor with correct shape: (0, 1, hidden_dim)
		return hidden_states, torch.empty(0, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim, 
										   dtype=hidden_states.dtype, device=hidden_states.device)
	
	if q_len != 1:
		raise ValueError("mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv only supports q_len=1")
	
	# Validate q_position_ids
	assert q_position_ids is not None, "q_position_ids is None"
	
	# Shape consistency check (only for non-empty batches)
	if q_position_ids.shape[0] != bsz:
		logging.error(
			f"[Layer {layer_idx}] SHAPE MISMATCH: "
			f"q_position_ids.shape[0]={q_position_ids.shape[0]} != bsz={bsz}, "
			f"hidden_states.shape={hidden_states.shape}"
		)
		raise ValueError(
			f"q_position_ids batch dimension ({q_position_ids.shape[0]}) "
			f"doesn't match hidden_states batch ({bsz})"
		)

	hidden_states = hidden_states.squeeze(1)

	hidden_states, hidden_states_scale = act_quant(hidden_states)

	q = w8a8_deepgemm(
		hidden_states,
		hidden_states_scale,
		self.q_a_proj.weight,
		weight_scale["q_a_proj.weight_scale_inv"],
	)
	
	new_compressed_kv = w8a8_deepgemm(
		hidden_states,
		hidden_states_scale,
		self.kv_a_proj_with_mqa.weight,
		weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
	).view(bsz, 1, -1)
	
	q = self.q_a_layernorm(q)
	q, q_scale = act_quant(q)
	q = w8a8_deepgemm(q, q_scale, self.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"])

	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
	q_pe = q_pe.contiguous()
	cos, sin = self.rotary_emb(q_pe, seq_len=max_seqlen)
	
	# Compute max position ID for RoPE validation
	# Note: bsz > 0 is guaranteed here (empty batch returns early above)
	try:
		max_pos_id = q_position_ids.max().item()
	except Exception as e:
		logging.error(
			f"[Layer {layer_idx}] FAILED to compute q_position_ids.max(): {e}. "
			f"q_position_ids.shape={q_position_ids.shape}, "
			f"q_position_ids.device={q_position_ids.device}, "
			f"q_position_ids.dtype={q_position_ids.dtype}, "
			f"bsz={bsz}"
		)
		raise
	
	cos_seq_len = cos.size(0)

	if max_pos_id >= cos_seq_len:
		logging.error(
			f"RoPE position overflow: max_position_id={max_pos_id}, "
			f"cos_seq_len={cos_seq_len}, max_seqlen={max_seqlen}, "
			f"q_position_ids.shape={q_position_ids.shape}, "
			f"cos.shape={cos.shape}, "  # ← Add this for clarity
			f"q_position_ids={q_position_ids.flatten().tolist()[:10]}..."
		)
		raise ValueError(
			f"q_position_ids (max={max_pos_id}) exceed RoPE cache size ({cos_seq_len})"
		)
	offload_kv = fused_rmsnorm_rope_with_q(
		new_compressed_kv,
		q_pe,
		cos,
		sin,
		q_position_ids,
		self.kv_a_layernorm.weight,
		self.kv_lora_rank,
		self.qk_rope_head_dim,
	)

	manager_device = gpu_paged_kv_manager.device
	k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1)).to(manager_device)
	sequence_lengths = q_position_ids.squeeze(-1).to(dtype=torch.int32, device=manager_device)
	
	gpu_paged_kv_manager.update_layer_decode_new_token(
		k_tensor=k_tensor,
		v_tensor=None,
		sequence_lengths=sequence_lengths,
		layer_idx=layer_idx,
		batch_slice=batch_slice,  # Pass batch slice for micro-batching support
	)

	blocked_k, _, block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(
		layer_idx=layer_idx
	)

	# Apply batch slice to block_table for micro-batching
	if batch_slice is not None:
		start_idx, end_idx = batch_slice
		block_table = block_table[start_idx:end_idx]

	# Validate block_table batch dimension matches input
	assert block_table.shape[0] == bsz, (
		f"[Layer {layer_idx}] block_table batch mismatch: "
		f"block_table.shape[0]={block_table.shape[0]} != bsz={bsz}. "
		f"batch_slice={batch_slice}. "
		f"This indicates GPU page table is out of sync with the current batch."
	)

	kv_b_proj = deepseek_v3_dequantization(
		self.kv_b_proj.weight.data,
		weight_scale["kv_b_proj.weight_scale_inv"],
	).view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz,
		self.num_heads,
		1,
		qk_head_dim,
		dtype=blocked_k.dtype,
		device=blocked_k.device,
	)
	q_nope = q_nope.squeeze(2)
	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum(
		"bhd,hdc->bhc",
		q_nope,
		q_absorb,
	).view(bsz, self.num_heads, 1, self.kv_lora_rank)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(bsz, 1, self.num_heads, qk_head_dim)

	page_size = gpu_paged_kv_manager.config.page_size_tokens
	tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, self.num_heads, 1)

	try:
		attn_out, _ = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			page_size,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True,
		)
	except Exception as exc:  # pragma: no cover - debugging aid
		logging.error("Error in flash_mla_with_kvcache (pagekv): %s", exc)
		raise

	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be {(bsz, self.num_heads, q_len, self.v_head_dim)}, got {attn_output.size()}"
		)
	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
	attn_output_fp8, attn_output_scale = act_quant(attn_output)
	attn_output = w8a8_deepgemm(
		attn_output_fp8,
		attn_output_scale,
		self.o_proj.weight,
		weight_scale["o_proj.weight_scale_inv"],
	)
	attn_output = attn_output.view(bsz, 1, -1)
	return attn_output, k_tensor


def mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv(
	self,
	hidden_states: torch.Tensor,
	q_position_ids: torch.Tensor,
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	weight_scale: Optional[dict] = None,
	gpu_paged_kv_manager: Optional['GPUPagedKVCacheManager'] = None,
	layer_idx: int = 0,
	batch_slice: Optional[tuple] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Pure BF16 MLA decoding for models like Kimi K2.5 (no FP8 quantization).

	This function is identical to mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv
	but uses pure BF16 linear operations instead of FP8 quantized GEMMs.

	Args:
		hidden_states: Input tensor [bsz, 1, hidden_dim]
		q_position_ids: Position IDs [bsz, 1]
		cache_seqlens: Sequence lengths [bsz]
		max_seqlen: Maximum sequence length
		weight_scale: Ignored (for compatibility)
		gpu_paged_kv_manager: Paged KV cache manager
		layer_idx: Current layer index
		batch_slice: Optional (start_idx, end_idx) for micro-batching

	Returns:
		attn_output: Attention output [bsz, 1, hidden_dim]
		k_tensor: KV tensor for callback [bsz, 1, 1, kv_dim]
	"""
	if gpu_paged_kv_manager is None:
		raise ValueError(
			"gpu_paged_kv_manager must be provided for page-KV decoding backend"
		)

	# ============ INPUT VALIDATION ============
	assert hidden_states is not None, "hidden_states is None"
	assert hidden_states.dim() == 3, f"hidden_states must be 3D, got {hidden_states.dim()}D"

	bsz, q_len, hidden_dim = hidden_states.size()

	# Early return for empty batch
	if bsz == 0:
		logging.debug(f"[Layer {layer_idx}] Empty batch (bsz=0), returning empty tensor")
		return hidden_states, torch.empty(
			0, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim,
			dtype=hidden_states.dtype, device=hidden_states.device
		)

	if q_len != 1:
		raise ValueError("mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv only supports q_len=1")

	assert q_position_ids is not None, "q_position_ids is None"

	if q_position_ids.shape[0] != bsz:
		raise ValueError(
			f"q_position_ids batch ({q_position_ids.shape[0]}) "
			f"doesn't match hidden_states batch ({bsz})"
		)

	hidden_states = hidden_states.squeeze(1)  # [bsz, hidden_dim]

	# ============ PER-OP TIMING (BATCHGEN_MLA_TIMING=1) ============
	import os as _os
	_do_timing = _os.environ.get("BATCHGEN_MLA_TIMING", "0") == "1" and layer_idx == 0
	if _do_timing:
		if not hasattr(mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv, '_timing_step'):
			mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv._timing_step = 0
			mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv._timing_accum = {}
		_step = mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv._timing_step
		mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv._timing_step = _step + 1
		_accum = mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv._timing_accum
		_events = []
		def _mark(name):
			e = torch.cuda.Event(enable_timing=True)
			e.record()
			_events.append((name, e))
		_mark("start")

	# ============ PURE BF16 PROJECTIONS (no quantization) ============
	q = F.linear(hidden_states, self.q_a_proj.weight)
	new_compressed_kv = F.linear(hidden_states, self.kv_a_proj_with_mqa.weight).view(bsz, 1, -1)

	q = self.q_a_layernorm(q)
	q = F.linear(q, self.q_b_proj.weight)
	if _do_timing: _mark("projections")

	# ============ ROPE & QUERY PROCESSING ============
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
	q_pe = q_pe.contiguous()
	cos, sin = self.rotary_emb(q_pe, seq_len=max_seqlen)
	if _do_timing: _mark("q_reshape_rope")

	# ============ KV CACHE UPDATE ============
	offload_kv = fused_rmsnorm_rope_with_q(
		new_compressed_kv,
		q_pe,
		cos,
		sin,
		q_position_ids,
		self.kv_a_layernorm.weight,
		self.kv_lora_rank,
		self.qk_rope_head_dim,
	)
	if _do_timing: _mark("fused_kv_norm_rope")

	manager_device = gpu_paged_kv_manager.device
	k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1)).to(manager_device)
	sequence_lengths = q_position_ids.squeeze(-1).to(dtype=torch.int32, device=manager_device)

	gpu_paged_kv_manager.update_layer_decode_new_token(
		k_tensor=k_tensor,
		v_tensor=None,
		sequence_lengths=sequence_lengths,
		layer_idx=layer_idx,
		batch_slice=batch_slice,
	)
	if _do_timing: _mark("kv_cache_update")

	blocked_k, _, block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(
		layer_idx=layer_idx
	)
	if _do_timing: _mark("get_kv_page_table")

	# Apply batch slice for micro-batching
	if batch_slice is not None:
		start_idx, end_idx = batch_slice
		block_table = block_table[start_idx:end_idx]

	assert block_table.shape[0] == bsz, (
		f"[Layer {layer_idx}] block_table batch mismatch: "
		f"block_table.shape[0]={block_table.shape[0]} != bsz={bsz}"
	)

	# ============ KV_B_PROJ (Pure BF16, no dequantization) ============
	kv_b_proj = self.kv_b_proj.weight.data.view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

	# ============ QUERY STATES CONSTRUCTION ============
	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, 1, qk_head_dim,
		dtype=blocked_k.dtype, device=blocked_k.device,
	)
	q_nope = q_nope.squeeze(2)
	query_states[:, :, :, :self.kv_lora_rank] = torch.einsum(
		"bhd,hdc->bhc", q_nope, q_absorb
	).view(bsz, self.num_heads, 1, self.kv_lora_rank)
	query_states[:, :, :, self.kv_lora_rank:] = q_pe
	query_states = query_states.view(bsz, 1, self.num_heads, qk_head_dim)
	if _do_timing: _mark("q_absorb")

	# ============ FLASH MLA ATTENTION ============
	tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 128, 1)

	attn_out, _ = flash_mla_with_kvcache(
		query_states,
		blocked_k,
		block_table,
		cache_seqlens,
		512,
		tile_scheduler_metadata,
		num_splits,
		self.softmax_scale,
		True,
	)
	if _do_timing: _mark("flash_mla")

	# ============ OUTPUT PROJECTION (Pure BF16) ============
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
	attn_output = F.linear(attn_output, self.o_proj.weight)
	attn_output = attn_output.view(bsz, 1, -1)
	if _do_timing: _mark("out_absorb_oproj")

	# ============ ACCUMULATE TIMING (print at step 128, layer 0 only) ============
	if _do_timing:
		if '_pending_events' not in _accum:
			_accum['_pending_events'] = []
		_accum['_pending_events'].append(_events)

		# Print at step 128 (after enough data, skip first 8 warmup steps)
		if _step == 128:
			torch.cuda.synchronize()
			pending = _accum.pop('_pending_events', [])
			# Skip first 8 steps (warmup, Triton JIT, etc.)
			skip = min(8, len(pending) // 2)
			measure = pending[skip:]
			for evts in measure:
				for i in range(1, len(evts)):
					name = evts[i][0]
					dt = evts[i-1][1].elapsed_time(evts[i][1])
					_accum[name] = _accum.get(name, 0.0) + dt
			# Print averages
			parts = []
			total = 0
			n = len(measure)
			for name in [k for k in _accum if not k.startswith('_')]:
				avg_us = _accum[name] / n * 1000
				parts.append(f"{name}={avg_us:.0f}us")
				total += avg_us
			parts.append(f"TOTAL={total:.0f}us")
			import logging as _logging
			_logging.info(f"[MLA TIMING L0] bsz={bsz} (avg over {n} steps, skip {skip}): {', '.join(parts)}")
			_accum.clear()
			_accum['_pending_events'] = []

	return attn_output, k_tensor


@torch.inference_mode()
def mla_decoding_optimized_with_pagekv(
	self,
	hidden_states: torch.Tensor,
	q_position_ids: torch.Tensor,
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	weight_scale: Optional[dict] = None,
	gpu_paged_kv_manager: Optional['GPUPagedKVCacheManager'] = None,
	layer_idx: int = 0,
	batch_slice: Optional[tuple] = None,
):
	"""Fully optimized MLA decode with ALL fused kernels.

	Optimizations vs original mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv:
	1. fused_q_split_cuda: replaces view+transpose+split+contiguous (4 ops → 1)
	2. fused_rmsnorm_rope_cache_update_with_q_return_new_kv: replaces separate
	   RMSNorm+RoPE+KV_cache_update (3 ops → 1, biggest savings)
	3. fused_q_absorb_query_states: replaces einsum+alloc+2×copy (4 ops → 1)
	4. fused_out_absorb_reshape: replaces einsum+transpose+contiguous+reshape (4 ops → 1)

	Bind via Parallel_Strategy_Manager with BATCHGEN_OPTIMIZED_DECODE=1.
	"""
	from batchgen_kernels.triton.fused_q_absorb import fused_q_absorb_query_states
	from batchgen_kernels.triton.fused_out_absorb import fused_out_absorb_reshape

	if gpu_paged_kv_manager is None:
		raise ValueError("gpu_paged_kv_manager is required for optimized decode path")

	bsz = hidden_states.shape[0]
	q_len = 1

	if bsz == 0:
		return hidden_states, None

	hidden_states = hidden_states.squeeze(1)

	# ============ PURE BF16 PROJECTIONS (same as original) ============
	q = F.linear(hidden_states, self.q_a_proj.weight)
	new_compressed_kv = F.linear(hidden_states, self.kv_a_proj_with_mqa.weight).view(bsz, 1, -1)

	q = self.q_a_layernorm(q)
	q = F.linear(q, self.q_b_proj.weight)

	# ============ OPT 1: FUSED Q SPLIT (replaces view+transpose+split+contiguous) ============
	try:
		from batchgen.attention.mla.fused_q_split_cuda import fused_q_split
		q_nope, q_pe = fused_q_split(
			q, num_heads=self.num_heads,
			nope_dim=self.qk_nope_head_dim, rope_dim=self.qk_rope_head_dim,
		)
	except (ImportError, Exception) as _e:
		# Fallback: original ops
		if not getattr(mla_decoding_optimized_with_pagekv, '_warned_q_split', False):
			logging.warning(f"[MLA OPT] fused_q_split_cuda unavailable, using fallback: {_e}")
			mla_decoding_optimized_with_pagekv._warned_q_split = True
		q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
		q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
		q_pe = q_pe.contiguous()
		q_nope = q_nope.squeeze(2)

	cos, sin = self.rotary_emb(q_pe, seq_len=max_seqlen)

	# ============ KV NORM+ROPE (existing production fused kernel — already optimal) ============
	# fused_rmsnorm_rope_with_q does RMSNorm + RoPE on KV + RoPE on all Q heads in 1 kernel
	offload_kv = fused_rmsnorm_rope_with_q(
		new_compressed_kv,
		q_pe if q_pe.dim() == 4 else q_pe.unsqueeze(2),
		cos,
		sin,
		q_position_ids,
		self.kv_a_layernorm.weight,
		self.kv_lora_rank,
		self.qk_rope_head_dim,
	)

	# Paged KV cache update (CUDA kernel — production standard)
	manager_device = gpu_paged_kv_manager.device
	k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1)).to(manager_device)
	sequence_lengths = q_position_ids.squeeze(-1).to(dtype=torch.int32, device=manager_device)

	gpu_paged_kv_manager.update_layer_decode_new_token(
		k_tensor=k_tensor,
		v_tensor=None,
		sequence_lengths=sequence_lengths,
		layer_idx=layer_idx,
		batch_slice=batch_slice,
	)

	blocked_k, _, block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(
		layer_idx=layer_idx
	)

	if batch_slice is not None:
		start_idx, end_idx = batch_slice
		block_table = block_table[start_idx:end_idx]

	# ============ KV_B_PROJ ============
	kv_b_proj = self.kv_b_proj.weight.data.view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

	# ============ OPT 3: FUSED QUERY STATES (replaces einsum+alloc+2×copy) ============
	q_nope_sq = q_nope if q_nope.dim() == 3 else q_nope.squeeze(2)
	query_states = fused_q_absorb_query_states(q_nope_sq, q_absorb,
		q_pe if q_pe.dim() == 4 else q_pe.unsqueeze(2))

	# ============ FLASH MLA ATTENTION ============
	tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 128, 1)

	attn_out, _ = flash_mla_with_kvcache(
		query_states,
		blocked_k,
		block_table,
		cache_seqlens,
		512,
		tile_scheduler_metadata,
		num_splits,
		self.softmax_scale,
		True,
	)

	# ============ OPT 4: FUSED OUTPUT (replaces einsum+transpose+contiguous+reshape) ============
	attn_output = fused_out_absorb_reshape(attn_out, out_absorb)
	attn_output = F.linear(attn_output, self.o_proj.weight)
	attn_output = attn_output.view(bsz, 1, -1)

	return attn_output, k_tensor


# @torch.inference_mode()
# def mla_decoding_flashmla_attn_mode_3_bf16_without_inplace_cache_update(
# 	self,
# 	hidden_states: torch.Tensor,
# 	past_key_states: torch.Tensor,
# 	past_value_states: torch.Tensor,
# 	attention_mask: torch.Tensor,
# 	q_position_ids: torch.Tensor,
# 	cache_seqlens: torch.Tensor,
# 	max_seqlen: int,
# 	scale: torch.Tensor = None,
# 	weight_scale: dict = None,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# 	"""
# 	Revised MLA decoding function with quantization support.
# 	The function signature is aligned with the flash-mla version for interface compatibility.
# 	It now handles quantized key-value caches.

# 	Args:
# 		hidden_states (torch.Tensor): Input hidden states of shape (batch_size, 1, hidden_size).
# 		past_key_states (torch.Tensor): The quantized (e.g., FP8) past compressed key states cache of shape
# 			(batch_size, max_seqlen, kv_dim).
# 		past_value_states (torch.Tensor): Not used. Placeholder for compatibility.
# 		attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
# 		q_position_ids (torch.Tensor): The position ids of shape (batch_size, 1), indicating
# 			where to write the new KV cache.
# 		scale (torch.Tensor): The dequantization scale for `past_key_states`.
# 		cache_seqlens (torch.Tensor): The sequence lengths of the cache. Not directly used here but
# 			kept for compatibility.
# 		max_seqlen (int): The maximum sequence length of the cache.
# 		weight_scale (dict, optional): Not used in this backend. Defaults to None.

# 	Returns:
# 		A tuple containing:
# 		- attn_output (torch.Tensor): The output of the attention module.
# 		- past_key_states (torch.Tensor): The updated quantized key cache.
# 		- scale (torch.Tensor): The updated dequantization scale tensor.
# 	"""
# 	bsz, q_len, _ = hidden_states.size()
# 	assert q_len == 1, "The PyTorch MLA decoding backend currently only supports a query length of 1."
# 	_, kv_len, _ = past_key_states.size()

# 	# --- 2. Query and New Key-Value Projection ---
# 	# q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
# 	hidden_states = hidden_states.squeeze(1)  # Remove seq_len dim for act_quant
# 	hidden_states, hidden_states_scale = act_quant(hidden_states)
# 	q = w8a8_deepgemm(hidden_states, hidden_states_scale, self.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"])
# 	new_compressed_kv = w8a8_deepgemm(hidden_states, hidden_states_scale, self.kv_a_proj_with_mqa.weight, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]).view(bsz, 1, -1)
# 	q = self.q_a_layernorm(q)
# 	q, q_scale = act_quant(q)
# 	q = w8a8_deepgemm(q, q_scale, self.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"])

# 	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
# 	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
# 	q_pe = q_pe.contiguous()
# 	cos, sin = self.rotary_emb(q_pe, seq_len=kv_len)
# 	fused_rmsnorm_rope_cache_update_with_q(
# 		new_compressed_kv,
# 		past_key_states,
# 		q_pe,
# 		cos,
# 		sin,
# 		q_position_ids,
# 		self.kv_a_layernorm.weight,
# 		self.kv_lora_rank,
# 		self.qk_rope_head_dim
# 	)


# 	kv_seqlen = past_key_states.size(1)
# 	kv_b_proj = deepseek_v3_dequantization(
# 		self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
# 	).view(self.num_heads, -1, self.kv_lora_rank)
# 	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
# 	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

# 	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
# 	query_states = torch.empty(
# 		bsz, self.num_heads, 1, qk_head_dim,
# 		dtype=past_key_states.dtype,
# 		device=past_key_states.device,
# 	)
# 	q_nope = q_nope.squeeze(2)
# 	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('bhd,hdc->bhc', q_nope, q_absorb).view(bsz, self.num_heads, 1, self.kv_lora_rank)
# 	# query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
# 	query_states[:, :, :, self.kv_lora_rank :] = q_pe
# 	query_states = query_states.view(
# 		bsz, 1, self.num_heads, qk_head_dim
# 	)

# 	# Pad the kv cache to be multiple of 64
# 	if kv_seqlen % 64 != 0:
# 		pad_len = 64 - (kv_seqlen % 64)
# 		past_key_states = torch.cat([
# 			past_key_states, 
# 			torch.zeros((bsz, pad_len, past_key_states.size(-1)), device=past_key_states.device, dtype=past_key_states.dtype)
# 		], dim=1)
# 		# cache_seqlens = cache_seqlens + pad_len
# 		kv_seqlen = past_key_states.size(1)

# 	block_size = 64
# 	block_table = torch.arange(
# 		bsz * kv_seqlen // block_size, dtype=torch.int32, device=past_key_states.device
# 	).view(bsz, kv_seqlen // block_size)

# 	blocked_k = past_key_states.view(
# 		bsz * kv_seqlen // block_size, block_size, 1, past_key_states.size(-1)
# 	)

# 	tile_scheduler_metadata, num_splits = get_mla_metadata(
# 		cache_seqlens, self.num_heads, 1
# 	)

# 	try:
# 		attn_out, _ = flash_mla_with_kvcache(
# 			query_states,
# 			blocked_k,
# 			block_table,
# 			cache_seqlens,
# 			512,
# 			tile_scheduler_metadata,
# 			num_splits,
# 			self.softmax_scale,
# 			True
# 		)
# 	except Exception as e:
# 		logging.error(f"Error in flash_mla_with_kvcache: {e}")
# 		raise
	
# 	# Apply out_absorb projection
# 	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	
# 	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
# 		raise ValueError(
# 			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
# 			f" {attn_output.size()}"
# 		)
	
# 	# --- 9. Final Projection and Return ---
# 	attn_output = attn_output.transpose(1, 2).contiguous()
# 	# attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
# 	attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
# 	# attn_output = self.o_proj(attn_output)
# 	attn_output_fp8, attn_output_scale = act_quant(attn_output)
# 	attn_output = w8a8_deepgemm(attn_output_fp8, attn_output_scale, self.o_proj.weight, weight_scale["o_proj.weight_scale_inv"])
# 	attn_output = attn_output.view(bsz, 1, -1)
	
# 	return attn_output, past_key_states[:, :kv_len, :]

@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_bf16_triton_gemm(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	q_position_ids: torch.Tensor,
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	scale: torch.Tensor = None,
	weight_scale: dict = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Revised MLA decoding function with quantization support.
	The function signature is aligned with the flash-mla version for interface compatibility.
	It now handles quantized key-value caches.

	Args:
		hidden_states (torch.Tensor): Input hidden states of shape (batch_size, 1, hidden_size).
		past_key_states (torch.Tensor): The quantized (e.g., FP8) past compressed key states cache of shape
			(batch_size, max_seqlen, kv_dim).
		past_value_states (torch.Tensor): Not used. Placeholder for compatibility.
		attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
		q_position_ids (torch.Tensor): The position ids of shape (batch_size, 1), indicating
			where to write the new KV cache.
		scale (torch.Tensor): The dequantization scale for `past_key_states`.
		cache_seqlens (torch.Tensor): The sequence lengths of the cache. Not directly used here but
			kept for compatibility.
		max_seqlen (int): The maximum sequence length of the cache.
		weight_scale (dict, optional): Not used in this backend. Defaults to None.

	Returns:
		A tuple containing:
		- attn_output (torch.Tensor): The output of the attention module.
		- past_key_states (torch.Tensor): The updated quantized key cache.
		- scale (torch.Tensor): The updated dequantization scale tensor.
	"""
	bsz, q_len, _ = hidden_states.size()
	assert q_len == 1, "The PyTorch MLA decoding backend currently only supports a query length of 1."
	_, kv_len, _ = past_key_states.size()

	# --- 2. Query and New Key-Value Projection ---
	# q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	hidden_states = hidden_states.squeeze(1)  # Remove seq_len dim for act_quant
	hidden_states, hidden_states_scale = act_quant(hidden_states)
	q = w8a8_gemm(hidden_states, hidden_states_scale, self.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"])
	new_compressed_kv = w8a8_gemm(hidden_states, hidden_states_scale, self.kv_a_proj_with_mqa.weight, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]).view(bsz, 1, -1)
	q = self.q_a_layernorm(q)
	q, q_scale = act_quant(q)
	q = w8a8_gemm(q, q_scale, self.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"])

	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
	q_pe = q_pe.contiguous()
	cos, sin = self.rotary_emb(q_pe, seq_len=kv_len)
	# q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)
	# Project the new compressed key-value pair in full precision.
	# new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

	# kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	
	# # Apply RoPE to k_pe
	# k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
	# k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_ids)
	# # Apply layer norm to kv
	# kv = self.kv_a_layernorm(kv)
	
	# # Reshape k_pe back for concatenation
	# k_pe = k_pe.view(bsz, 1, self.qk_rope_head_dim)
	
	# # Concatenate kv and k_pe for storage
	# offload_kv = torch.cat([kv, k_pe], dim=-1)

	# offload_kv = fused_rmsnorm_rope(
	# 	new_compressed_kv,
	# 	cos,
	# 	sin,
	# 	q_position_ids,
	# 	self.kv_a_layernorm.weight,
	# 	self.kv_lora_rank,
	# 	self.qk_rope_head_dim
	# )
	
	# --- 3. Update Cache ---
	# Update the dequantized cache at the current position
	# batch_indices = torch.arange(bsz, device=hidden_states.device)
	# past_key_states[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	fused_rmsnorm_rope_cache_update_with_q(
		new_compressed_kv,
		past_key_states,
		q_pe,
		cos,
		sin,
		q_position_ids,
		self.kv_a_layernorm.weight,
		self.kv_lora_rank,
		self.qk_rope_head_dim
	)


	kv_seqlen = past_key_states.size(1)
	kv_b_proj = deepseek_v3_dequantization(
		self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	).view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, 1, qk_head_dim,
		dtype=past_key_states.dtype,
		device=past_key_states.device,
	)
	q_nope = q_nope.squeeze(2)
	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('bhd,hdc->bhc', q_nope, q_absorb).view(bsz, self.num_heads, 1, self.kv_lora_rank)
	# query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(
		bsz, 1, self.num_heads, qk_head_dim
	)

	# Pad the kv cache to be multiple of 64
	if kv_seqlen % 64 != 0:
		pad_len = 64 - (kv_seqlen % 64)
		past_key_states = torch.cat([
			past_key_states, 
			torch.zeros((bsz, pad_len, past_key_states.size(-1)), device=past_key_states.device, dtype=past_key_states.dtype)
		], dim=1)
		# cache_seqlens = cache_seqlens + pad_len
		kv_seqlen = past_key_states.size(1)

	block_size = 64
	block_table = torch.arange(
		bsz * kv_seqlen // block_size, dtype=torch.int32, device=past_key_states.device
	).view(bsz, kv_seqlen // block_size)

	blocked_k = past_key_states.view(
		bsz * kv_seqlen // block_size, block_size, 1, past_key_states.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	try:
		attn_out, _ = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	
	# Apply out_absorb projection
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)
	
	# --- 9. Final Projection and Return ---
	attn_output = attn_output.transpose(1, 2).contiguous()
	# attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
	attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
	# attn_output = self.o_proj(attn_output)
	attn_output_fp8, attn_output_scale = act_quant(attn_output)
	attn_output = w8a8_gemm(attn_output_fp8, attn_output_scale, self.o_proj.weight, weight_scale["o_proj.weight_scale_inv"])
	attn_output = attn_output.view(bsz, 1, -1)
	
	return attn_output, past_key_states[:, :kv_len, :]

@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_bf16_without_fused_rmsnorm_rope(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	q_position_ids: torch.Tensor,
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	scale: torch.Tensor = None,
	weight_scale: dict = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Revised MLA decoding function with quantization support.
	The function signature is aligned with the flash-mla version for interface compatibility.
	It now handles quantized key-value caches.

	Args:
		hidden_states (torch.Tensor): Input hidden states of shape (batch_size, 1, hidden_size).
		past_key_states (torch.Tensor): The quantized (e.g., FP8) past compressed key states cache of shape
			(batch_size, max_seqlen, kv_dim).
		past_value_states (torch.Tensor): Not used. Placeholder for compatibility.
		attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
		q_position_ids (torch.Tensor): The position ids of shape (batch_size, 1), indicating
			where to write the new KV cache.
		scale (torch.Tensor): The dequantization scale for `past_key_states`.
		cache_seqlens (torch.Tensor): The sequence lengths of the cache. Not directly used here but
			kept for compatibility.
		max_seqlen (int): The maximum sequence length of the cache.
		weight_scale (dict, optional): Not used in this backend. Defaults to None.

	Returns:
		A tuple containing:
		- attn_output (torch.Tensor): The output of the attention module.
		- past_key_states (torch.Tensor): The updated quantized key cache.
		- scale (torch.Tensor): The updated dequantization scale tensor.
	"""
	bsz, q_len, _ = hidden_states.size()
	assert q_len == 1, "The PyTorch MLA decoding backend currently only supports a query length of 1."
	_, kv_len, _ = past_key_states.size()

	# --- 2. Query and New Key-Value Projection ---
	# q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	hidden_states = hidden_states.squeeze(1)  # Remove seq_len dim for act_quant
	hidden_states, hidden_states_scale = act_quant(hidden_states)
	q = w8a8_gemm(hidden_states, hidden_states_scale, self.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"])
	new_compressed_kv = w8a8_gemm(hidden_states, hidden_states_scale, self.kv_a_proj_with_mqa.weight, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]).view(bsz, 1, -1)
	q = self.q_a_layernorm(q)
	q, q_scale = act_quant(q)
	q = w8a8_gemm(q, q_scale, self.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"])

	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

	# Project the new compressed key-value pair in full precision.
	# new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	
	# Apply RoPE to k_pe
	k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=kv_len)
	k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_ids)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)
	
	# Apply layer norm to kv
	kv = self.kv_a_layernorm(kv)
	
	# Reshape k_pe back for concatenation
	k_pe = k_pe.view(bsz, 1, self.qk_rope_head_dim)
	
	# Concatenate kv and k_pe for storage
	offload_kv = torch.cat([kv, k_pe], dim=-1)
	
	# --- 3. Update Cache ---
	# Update the dequantized cache at the current position
	batch_indices = torch.arange(bsz, device=hidden_states.device)
	past_key_states[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]



	kv_seqlen = past_key_states.size(1)
	# kv_b_proj = self.kv_b_proj.weight.view(
	# 	self.num_heads, -1, self.kv_lora_rank
	# )
	kv_b_proj = deepseek_v3_dequantization(
		self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	).view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, 1, qk_head_dim,
		dtype=past_key_states.dtype,
		device=past_key_states.device,
	)

	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(
		bsz, 1, self.num_heads, qk_head_dim
	)

	# Pad the kv cache to be multiple of 64
	if kv_seqlen % 64 != 0:
		pad_len = 64 - (kv_seqlen % 64)
		past_key_states = torch.cat([
			past_key_states, 
			torch.zeros((bsz, pad_len, past_key_states.size(-1)), device=past_key_states.device, dtype=past_key_states.dtype)
		], dim=1)
		# cache_seqlens = cache_seqlens + pad_len
		kv_seqlen = past_key_states.size(1)

	block_size = 64
	block_table = torch.arange(
		bsz * kv_seqlen // block_size, dtype=torch.int32, device=past_key_states.device
	).view(bsz, kv_seqlen // block_size)

	blocked_k = past_key_states.view(
		bsz * kv_seqlen // block_size, block_size, 1, past_key_states.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	try:
		attn_out, _ = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	
	# Apply out_absorb projection
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)
	
	# --- 9. Final Projection and Return ---
	attn_output = attn_output.transpose(1, 2).contiguous()
	# attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
	attn_output = attn_output.reshape(bsz, self.num_heads * self.v_head_dim)
	# attn_output = self.o_proj(attn_output)
	attn_output_fp8, attn_output_scale = act_quant(attn_output)
	attn_output = w8a8_gemm(attn_output_fp8, attn_output_scale, self.o_proj.weight, weight_scale["o_proj.weight_scale_inv"])
	attn_output = attn_output.view(bsz, 1, -1)
	
	return attn_output, past_key_states[:, :kv_len, :]


# NOTE: mla_decoding_flashmla_attn_mode_3_bf16_bak was moved to local_archive/deprecated/


@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_dequant_fusion(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
	scale,
	weight_scale: dict[str, torch.Tensor],
):
	"""
		MLA decoding function using FlashMLA as the attention mechanism backend.
		Args:
			hidden_states (torch.Tensor): The input hidden states of shape (batch_size, 1, hidden_size).
			past_key_states (torch.Tensor): The past key states of shape (batch_size, max_seqlen, kv_dim)
			past_value_states (torch.Tensor): None. Placeholder.
			attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
			position_ids (torch.Tensor): The position ids of shape (batch_size, 1)
			scale (torch.Tensor): The scale tensor for kv per token quantization. [batch_size, max_seqlen, ceil(kv_dim // 128)]
	
		Note:
			- The past_key_states has the shape of (batch_size, max_seqlen, kv_dim).
			Where max_seqlen is the nearest multiple of 64(kv block size) that is greater than or equal to the full context length.
			- attention_mask has the shape of (batch_size, seq_len).
			Where seq_len is the length of the processed tokens(input prompt(padded) + generated tokens).
			- position_ids record the position of the new kv should be placed in the past compressed kv cache.
			- max_seq_len is reserved for scale. 
			
	"""
	from batchgen.models.wrappers.attention import AttnWrapperBase
	# Use cache_seqlens directly instead of attention_mask.sum()
	block_size = 64
	if AttnWrapperBase.cache_seqlens is not None:
		cache_seqlens = AttnWrapperBase.cache_seqlens
	else:
		cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)

	bsz = hidden_states.size(0)
	max_seqlen = cache_seqlens.max().item()

	compressed_kv = dequant_compressed_kv_per_token_with_length(past_key_states, scale, max_seqlen)
	max_seqlen_pad = compressed_kv.size(1)

	q_position_ids = (cache_seqlens.to(torch.int64) - 1).unsqueeze(-1)
	
	q = fused_fp8_bf16_gemm(
		hidden_states, self.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"]
	)
	q = self.q_a_layernorm(q)
	q = fused_fp8_bf16_gemm(
		q, self.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"]
	)
	# q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, 1, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	# new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	new_compressed_kv = fused_fp8_bf16_gemm(
		hidden_states, self.kv_a_proj_with_mqa.weight, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]
	)
	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=max_seqlen_pad)
	k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_ids)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)
	kv = self.kv_a_layernorm(kv)
	k_pe = k_pe.view(bsz, 1, self.qk_rope_head_dim)
	offload_kv = torch.cat([kv, k_pe], dim=-1)
	
	batch_indices = torch.arange(bsz, device=hidden_states.device)
	# compressed_kv[batch_indices, position_ids[:, 0], :self.kv_lora_rank] = kv[:, 0, :]
	# compressed_kv[batch_indices, position_ids[:, 0], self.kv_lora_rank:] = k_pe[:, 0, :]
	compressed_kv[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	compressed_kv = compressed_kv.view(bsz, max_seqlen_pad, 1, 576)

	# Quantize and write to past_key_states
	new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
	
	# kv_b_proj = self.kv_b_proj.weight.view(
	# 	self.num_heads, -1, self.kv_lora_rank
	# )
	kv_b_proj = deepseek_v3_dequantization(
		self.kv_b_proj.weight, weight_scale["kv_b_proj.weight_scale_inv"]
	).view(self.num_heads, -1, self.kv_lora_rank)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, 1, qk_head_dim,
		dtype=compressed_kv.dtype,
		device=compressed_kv.device,
	)

	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	# query_states[:, :, :, : self.kv_lora_rank] = fused_bhd_hdc(q_nope, q_absorb)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(
		bsz, 1, self.num_heads, qk_head_dim
	)


	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	try:
		attn_out, _ = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	# out_absorb = out_absorb.transpose(1,2).contiguous()
	# attn_output = fused_bhd_hdc(attn_out.squeeze(1), out_absorb)


	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.view(
		bsz, 1, self.num_heads * self.v_head_dim
	)
	# attn_output = self.o_proj(attn_output)
	attn_output = fused_fp8_bf16_gemm(
		attn_output, self.o_proj.weight, weight_scale["o_proj.weight_scale_inv"]
	)

	return attn_output, past_key_states, scale



@torch.inference_mode()
def mla_decoding_flashmla(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
	scale,
	cache_seqlens_override: torch.Tensor = None,
	max_seqlen_override: int = None,
):
	from batchgen.models.wrappers.attention import AttnWrapperBase
	# Use cache_seqlens directly instead of attention_mask.sum()
	if cache_seqlens_override is not None:
		cache_seqlens = cache_seqlens_override
	elif AttnWrapperBase.cache_seqlens is not None:
		cache_seqlens = AttnWrapperBase.cache_seqlens
	else:
		cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)

	block_size = 64
	max_seqlen = max_seqlen_override if max_seqlen_override is not None else cache_seqlens.max().item()
	max_seqlen_pad = ((max_seqlen + block_size - 1) // block_size) * block_size
	if max_seqlen_pad >= past_key_states.size(1):
		compressed_kv = dequant_per_token_return_with_max_seqlen_pad(
			past_key_states, scale, max_seqlen_pad
		)
	else:
		compressed_kv = past_key_states[:, :max_seqlen_pad, :].contiguous()
		scale = scale[:, :max_seqlen_pad, :].contiguous()
		compressed_kv = dequant_per_token_triton(compressed_kv, scale)

	bsz, q_len, _ = hidden_states.size()
	q_position_id = (cache_seqlens.to(torch.int64) - 1).unsqueeze(-1)
	kv_len = max_seqlen_pad
	
	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	k_pe = k_pe.view(bsz, 1, q_len, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=kv_len)
	k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_id)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_id)
	kv = self.kv_a_layernorm(kv)
	k_pe = k_pe.view(bsz, q_len, self.qk_rope_head_dim)
	offload_kv = torch.cat([kv, k_pe], dim=-1)

	batch_indices = torch.arange(bsz, device=hidden_states.device)
	# compressed_kv[batch_indices, q_position_id[:, 0], :self.kv_lora_rank] = kv[:, 0, :]
	# compressed_kv[batch_indices, q_position_id[:, 0], self.kv_lora_rank:] = k_pe[:, 0, :]
	compressed_kv[batch_indices, q_position_id[:, 0], :] = offload_kv[:, 0, :]
	compressed_kv = compressed_kv.view(bsz, max_seqlen_pad, 1, 576)
	
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, q_len, qk_head_dim,
		dtype=compressed_kv.dtype,
		device=compressed_kv.device,
	)

	# query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(
		bsz, q_len, self.num_heads, qk_head_dim
	)



	# Pad the compressed_kv_ref tensor to the maximum sequence length
	# if max_seqlen_pad > kv_len:
	# 	compressed_kv = torch.cat(
	# 		[
	# 			compressed_kv,
	# 			torch.full(
	# 				(bsz, max_seqlen_pad - kv_len, 1, compressed_kv.size(-1)),
	# 				# float("nan"),
	# 				0,
	# 				dtype=compressed_kv.dtype,
	# 				device=compressed_kv.device,
	# 			),
	# 		],
	# 		dim=1,
	# 	)
	# else:
	# 	compressed_kv = compressed_kv[:, :max_seqlen_pad, :, :]

	# block_table = torch.arange(
	# 	bsz * max_seqlen_pad // block_size, dtype=torch.int32
	# ).view(bsz, max_seqlen_pad // block_size).to(compressed_kv.device)
	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	"""
	flash_mla_with_kvcache
	Arguments:
		q: (batch_size, seq_len_q, num_heads_q, head_dim).
		k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
		block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
		cache_seqlens: (batch_size), torch.int32.
		head_dim_v: Head dimension of v.
		tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
		num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
		softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
		causal: bool. Whether to apply causal attention mask.

	Returns:
		out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
		softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
	"""
	try:
		attn_out, attention_weights = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)


	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.view(
		bsz, q_len, self.num_heads * self.v_head_dim
	)
	attn_output = self.o_proj(attn_output)

	return (
		attn_output,
		offload_kv,
		torch.tensor([], device=hidden_states.device),
	)



@torch.inference_mode()
def get_query_states(self, hidden_states: torch.Tensor):
	bsz, q_len, _ = hidden_states.size()
	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	# out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, q_len, qk_head_dim,
		dtype=hidden_states.dtype,
		device=hidden_states.device,
	)

	# query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(
		bsz, q_len, self.num_heads, qk_head_dim
	)
	return query_states

@torch.inference_mode()
def get_query_states(self, q: torch.Tensor):
	bsz, q_len, _ = q.size()
	# q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	# out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, q_len, qk_head_dim,
		dtype=q.dtype,
		device=q.device,
	)

	# query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(
		bsz, q_len, self.num_heads, qk_head_dim
	)
	return query_states


@triton.jit
def fused_query_kernel(
	# Input pointers
	q_ptr,
	kv_b_proj_ptr,
	# Output pointer (in-place modification of q)
	output_ptr,
	# Dimensions
	bsz,
	q_len,
	num_heads,
	qk_nope_head_dim,
	qk_rope_head_dim,
	kv_lora_rank,
	# Strides
	q_stride_b, q_stride_s, q_stride_h, q_stride_d,
	kv_b_stride_h, kv_b_stride_d, kv_b_stride_r,
	out_stride_b, out_stride_s, out_stride_h, out_stride_d,
	# Block sizes
	BLOCK_SIZE_S: tl.constexpr,
):
	# Get program IDs for batch, sequence, head
	pid_b = tl.program_id(0)
	pid_s = tl.program_id(1) 
	pid_h = tl.program_id(2)
	
	# Early exit if out of bounds
	if pid_b >= bsz or pid_h >= num_heads:
		return
		
	# Calculate sequence indices for this block
	seq_start = pid_s * BLOCK_SIZE_S
	seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_S)
	seq_mask = seq_offsets < q_len
	
	# Base pointers for this batch and head
	q_base = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h
	kv_b_base = kv_b_proj_ptr + pid_h * kv_b_stride_h
	out_base = output_ptr + pid_b * out_stride_b + pid_h * out_stride_h
	
	# Process each sequence position in the block
	for s_idx in range(BLOCK_SIZE_S):
		actual_seq = seq_start + s_idx
		# Use conditional execution instead of break
		if actual_seq < q_len:
			# Calculate pointers for this sequence position
			q_seq_base = q_base + actual_seq * q_stride_s
			out_seq_base = out_base + actual_seq * out_stride_s
			
			# Load q_nope part (first qk_nope_head_dim elements)
			q_nope = tl.load(q_seq_base + tl.arange(0, qk_nope_head_dim) * q_stride_d, 
						   mask=tl.arange(0, qk_nope_head_dim) < qk_nope_head_dim)
			
			# Compute einsum: kv_b_proj @ q_nope for each lora rank
			# Store results in the first kv_lora_rank positions (overwriting q_nope)
			for r in range(kv_lora_rank):
				# Load the r-th column of kv_b_proj for this head  
				kv_col = tl.load(kv_b_base + tl.arange(0, qk_nope_head_dim) * kv_b_stride_d + r * kv_b_stride_r,
							   mask=tl.arange(0, qk_nope_head_dim) < qk_nope_head_dim)
				
				# Compute dot product and store in-place
				dot_result = tl.sum(kv_col * q_nope)
				tl.store(out_seq_base + r * out_stride_d, dot_result)
			
			# q_pe part remains untouched at positions [kv_lora_rank:]


def fused_get_query_states_triton(self, q: torch.Tensor):
	"""
	Fused version that modifies q in-place and returns it with the new layout.
	No intermediate tensor allocations.
	
	In-place operation: only modifies the first kv_lora_rank dimensions,
	q_pe part (at positions [kv_lora_rank:]) remains unchanged.
	"""
	bsz, q_len, total_dim = q.size()
	
	# Reshape q to [bsz, q_len, num_heads, head_dim] for easier processing
	q = q.view(bsz, q_len, self.num_heads, -1)
	
	# Calculate expected output dimension
	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	
	# Validate dimensions - must match exactly for in-place operation
	if q.size(-1) != qk_head_dim:
		raise ValueError(f"Input tensor last dimension {q.size(-1)} must equal "
						f"kv_lora_rank + qk_rope_head_dim = {qk_head_dim} for in-place operation")
	
	# Get kv_b_proj weight reshaped
	kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
	
	# Grid dimensions - one thread block per (batch, head) pair
	# Each thread block processes multiple sequence positions
	grid = lambda meta: (
		bsz,  # One per batch
		triton.cdiv(q_len, meta['BLOCK_SIZE_S']),  # Sequence blocks
		self.num_heads  # One per head
	)
	
	# Launch kernel - modifies q in-place
	fused_query_kernel[grid](
		# Pointers (input and output are the same for in-place)
		q.data_ptr(),
		kv_b_proj.data_ptr(),
		q.data_ptr(),  # Same as input for in-place operation
		# Dimensions
		bsz, q_len, self.num_heads,
		self.qk_nope_head_dim, self.qk_rope_head_dim, self.kv_lora_rank,
		# Input strides
		q.stride(0), q.stride(1), q.stride(2), q.stride(3),
		kv_b_proj.stride(0), kv_b_proj.stride(1), kv_b_proj.stride(2),
		# Output strides (same as input for in-place)
		q.stride(0), q.stride(1), q.stride(2), q.stride(3),
		# Block sizes (tune these for your hardware)
		BLOCK_SIZE_S=32,  # Process 32 sequence positions per block
	)
	
	# Return the modified tensor with the expected layout
	return q.view(bsz, q_len, self.num_heads, qk_head_dim)

@torch.inference_mode()
def mla_decoding_flashmla_v2(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
	scale
):
	"""
		- GPU KV-Cache shape [bsz, max_context_length, 576]
		- Direct write new tokens to the corresponding position.
		- The max_seqlen_pad is the max_context_length (a multiple of block_size).

	"""
	from batchgen.models.wrappers.attention import AttnWrapperBase
	# Use cache_seqlens directly instead of attention_mask.sum()
	block_size = 64
	if AttnWrapperBase.cache_seqlens is not None:
		cache_seqlens = AttnWrapperBase.cache_seqlens
	else:
		cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)

	max_seqlen = cache_seqlens.max().item()
	max_seqlen_pad = ((max_seqlen + block_size - 1) // block_size) * block_size
	if max_seqlen_pad >= past_key_states.size(1):
		compressed_kv = dequant_per_token_return_with_max_seqlen_pad(
			past_key_states, scale, max_seqlen_pad
		)
	else:
		compressed_kv = past_key_states[:, :max_seqlen_pad, :].contiguous()
		scale = scale[:, :max_seqlen_pad, :].contiguous()
		compressed_kv = dequant_per_token_triton(compressed_kv, scale)

	bsz, q_len, _ = hidden_states.size()
	q_position_id = (cache_seqlens.to(torch.int64) - 1).unsqueeze(-1)
	kv_len = max_seqlen_pad
	
	# q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	# q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	# q_nope, q_pe = torch.split(
	# 	q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	# )

	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	# qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	# query_states = torch.empty(
	# 	bsz, self.num_heads, q_len, qk_head_dim,
	# 	dtype=compressed_kv.dtype,
	# 	device=compressed_kv.device,
	# )

	# # query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	# query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	# query_states[:, :, :, self.kv_lora_rank :] = q_pe
	# query_states = query_states.view(
	# 	bsz, q_len, self.num_heads, qk_head_dim
	# )
	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	query_states = self.fused_get_query_states_triton(q)

	new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	k_pe = k_pe.view(bsz, 1, q_len, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=kv_len)
	k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_id)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_id)
	kv = self.kv_a_layernorm(kv)
	k_pe = k_pe.view(bsz, q_len, self.qk_rope_head_dim)
	offload_kv = torch.cat([kv, k_pe], dim=-1)

	batch_indices = torch.arange(bsz, device=hidden_states.device)
	# compressed_kv[batch_indices, q_position_id[:, 0], :self.kv_lora_rank] = kv[:, 0, :]
	# compressed_kv[batch_indices, q_position_id[:, 0], self.kv_lora_rank:] = k_pe[:, 0, :]
	compressed_kv[batch_indices, q_position_id[:, 0], :] = offload_kv[:, 0, :]
	compressed_kv = compressed_kv.view(bsz, max_seqlen_pad, 1, 576)
	

	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, self.num_heads, 1
	)

	"""
	flash_mla_with_kvcache
	Arguments:
		q: (batch_size, seq_len_q, num_heads_q, head_dim).
		k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
		block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
		cache_seqlens: (batch_size), torch.int32.
		head_dim_v: Head dimension of v.
		tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
		num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
		softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
		causal: bool. Whether to apply causal attention mask.

	Returns:
		out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
		softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
	"""
	try:
		attn_out, attention_weights = flash_mla_with_kvcache(
			query_states,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			self.softmax_scale,
			True
		)
	except Exception as e:
		logging.error(f"Error in flash_mla_with_kvcache: {e}")
		raise
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)


	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.view(
		bsz, q_len, self.num_heads * self.v_head_dim
	)
	attn_output = self.o_proj(attn_output)

	return (
		attn_output,
		offload_kv,
		torch.tensor([], device=hidden_states.device),
	)
