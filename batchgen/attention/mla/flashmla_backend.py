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

from typing import Tuple
import math

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


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
	"""
	Dequantizes weights using the provided scaling factors and stores the result.

	Args:
		x_ptr (tl.pointer): Pointer to the quantized weights.
		s_ptr (tl.pointer): Pointer to the scaling factors.
		y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
		M (int): Number of rows in the weight matrix.
		N (int): Number of columns in the weight matrix.
		BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

	Returns:
		None
	"""
	pid_m = tl.program_id(axis=0)
	pid_n = tl.program_id(axis=1)
	n = tl.cdiv(N, BLOCK_SIZE)
	offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
	offs = offs_m[:, None] * N + offs_n[None, :]
	mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
	x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
	s = tl.load(s_ptr + pid_m * n + pid_n)
	y = x * s
	tl.store(y_ptr + offs, y, mask=mask)


def deepseek_v3_dequantization(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
	"""
	Dequantizes the given weight tensor using the provided scale tensor.

	Args:
		x (torch.Tensor): The quantized weight tensor of shape (M, N).
		s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
		block_size (int, optional): The block size to use for dequantization. Defaults to 128.

	Returns:
		torch.Tensor: The dequantized weight tensor of the same shape as `x`.

	Raises:
		AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
	"""
	assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
	assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions but got {} and {}'.format(x.shape, s.shape)
	M, N = x.size()
	y = torch.empty_like(x, dtype=torch.bfloat16)
	grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
	weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
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
	# Create a block table for the key states
	block_size = 64
	cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
	
	max_seqlen = cache_seqlens.max().item()
	max_seqlen_pad = ((max_seqlen + block_size - 1) // block_size) * block_size
	# compressed_kv = dequant_per_token_triton(past_key_states, scale)
	if max_seqlen_pad >= past_key_states.size(1):
		compressed_kv = dequant_per_token_return_with_max_seqlen_pad(
			past_key_states, scale, max_seqlen_pad
		)
	else:
		compressed_kv = past_key_states[:, :max_seqlen_pad, :].contiguous()
		scale = scale[:, :max_seqlen_pad, :].contiguous()
		compressed_kv = dequant_per_token_triton(compressed_kv, scale)

	
	assert attention_mask.dim() == 2
	bsz, q_len, _ = hidden_states.size()
	q_position_id = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	kv_len = attention_mask.size(-1)

	
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
		cache_seqlens, 128, 1
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
	assert attention_mask.dim() == 2
	# Create a block table for the key states
	block_size = 64	
	bsz, seq_len = attention_mask.size()
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
		cache_seqlens, 128, 1
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




@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_bak(
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
	assert attention_mask.dim() == 2
	# Create a block table for the key states
	block_size = 64
	# cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
	
	bsz, seq_len = attention_mask.size()
	# max_seqlen = cache_seqlens.max().item()
	
	# _,max_seq_len, _ = past_key_states.size()
	# compressed_kv = dequant_compressed_kv_per_token_with_length(past_key_states, scale, max_seqlen)
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
	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, 1, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	# self.kv_a_proj_with_mqa.weight.data = deepseek_v3_dequantization(
	# 	self.kv_a_proj_with_mqa.weight.data, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"]
	# )
	new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=max_seqlen_pad)
	k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_ids)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)
	kv = self.kv_a_layernorm(kv)
	k_pe = k_pe.view(bsz, 1, self.qk_rope_head_dim)
	offload_kv = torch.cat([kv, k_pe], dim=-1)
	
	batch_indices = torch.arange(bsz, device=hidden_states.device)
	compressed_kv[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	compressed_kv = compressed_kv.view(bsz, max_seqlen_pad, 1, 576)

	# Quantize and write to past_key_states
	new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
	
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	# kv_b_proj = deepseek_v3_dequantization(
	# 	self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	# ).view(self.num_heads, -1, self.kv_lora_rank)
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

	assert qk_head_dim == 576, f"qk_head_dim should be 576, but got {qk_head_dim}"
	assert self.num_heads == 128, f"num_heads should be 128, but got {self.num_heads}"
	

	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, 128, 1
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
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)


	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.view(
		bsz, 1, self.num_heads * self.v_head_dim
	)
	# self.o_proj.weight.data = deepseek_v3_dequantization(
	# 	self.o_proj.weight.data, weight_scale["o_proj.weight_scale_inv"]
	# )
	attn_output = self.o_proj(attn_output)
	return attn_output, past_key_states, scale


# def update_causal_mask(attention_mask):
#     """
#     Create causal mask for decoding.
    
#     Args:
#         attention_mask: [bsz, seq_len] - 1 for valid tokens, 0 for padding
    
#     Returns:
#         causal_mask: [bsz, 1, 1, seq_len] - additive mask for attention
#     """
#     dtype = torch.bfloat16
#     min_dtype = torch.finfo(dtype).min
#     device = attention_mask.device
#     bsz, seq_len = attention_mask.shape
    
#     # Query length is 1 (single new token being generated)
#     query_length = 1
    
#     # Current position is the last position (seq_len - 1 in 0-indexing)
#     current_position = seq_len - 1
    
#     # Create causal mask: current token can only attend to positions 0 to current_position
#     # Shape: [1, seq_len]
#     causal_mask = torch.zeros((query_length, seq_len), dtype=dtype, device=device)
    
#     # Mask future positions (though there shouldn't be any if we're at the last position)
#     # This is important if seq_len includes future placeholder positions
#     if current_position < seq_len - 1:
#         causal_mask[:, current_position + 1:] = min_dtype
    
#     # Expand to [bsz, 1, 1, seq_len]
#     causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, -1, -1)
    
#     # Apply padding mask: set padding positions to min_dtype
#     # attention_mask: 1 for valid, 0 for padding
#     padding_mask = (attention_mask[:, None, None, :] == 0)
#     causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)
    
#     return causal_mask

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
	
	# Prepare KV projection weights with dequantization
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
	assert self.num_heads == 128, f"num_heads should be 128, but got {self.num_heads}"
	
	block_size = 64	
	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv_ref.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv_ref.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv_ref.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, 128, 1
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
from .fused_rmsnorm_rope import fused_rmsnorm_rope
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
	
	cos, sin = self.rotary_emb(q_pe, seq_len=max_seqlen_pad)
	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)


	# # Split and process KV
	# kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	
	# # Apply RoPE
	# k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
	# k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_ids)
	
	# kv = self.kv_a_layernorm(kv)
	# k_pe = k_pe.view(bsz, 1, self.qk_rope_head_dim)
	# offload_kv = torch.cat([kv, k_pe], dim=-1)
	offload_kv = fused_rmsnorm_rope(
		new_compressed_kv,
		cos,
		sin,
		q_position_ids,
		self.kv_a_layernorm.weight,
		self.kv_lora_rank,
		self.qk_rope_head_dim,
	)
	
	# Update cache with new KV
	batch_indices = torch.arange(bsz, device=hidden_states.device)
	compressed_kv_ref[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	
	# Quantize and update past_key_states
	new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
	
	# Prepare KV projection weights with dequantization
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
	assert self.num_heads == 128, f"num_heads should be 128, but got {self.num_heads}"
	
	block_size = 64	
	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32, device=compressed_kv_ref.device
	).view(bsz, max_seqlen_pad // block_size)

	blocked_k = compressed_kv_ref.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv_ref.size(-1)
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens, 128, 1
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


@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn_bak(
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

	# --- 1. Dequantize KV Cache ---
	# The cache is received in a quantized format. Dequantize it for processing.
	# assert not torch.isnan(past_key_states).any(), "NaN in past_key_states before dequantization"
	# assert not torch.isinf(past_key_states).any(), "Inf in past_key_states before dequantization"
	compressed_kv_ref = dequant_compressed_kv_per_token(past_key_states, scale, max_seqlen)
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref after dequantization"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref after dequantization"
	max_seqlen_pad = compressed_kv_ref.size(1)

	# --- 2. Query and New Key-Value Projection ---
	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

	# Project the new compressed key-value pair in full precision.
	new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
	
	# Apply RoPE to k_pe
	k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=max_seqlen_pad)
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
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref before update"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref before update"
	
	compressed_kv_ref[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	# log if there is NaN or Inf in compressed_kv_ref
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref after update"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref after update"
	
	# Quantize and write to past_key_states
	new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
	
	# --- 4. Split compressed_kv_ref for attention computation ---
	# compressed_kv_ref shape: [bsz, max_seqlen_pad, kv_lora_rank + qk_rope_head_dim]
	kv_states, k_pe_states = torch.split(
		compressed_kv_ref, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	# kv_states shape: [bsz, max_seqlen_pad, kv_lora_rank]
	# k_pe_states shape: [bsz, max_seqlen_pad, qk_rope_head_dim]
	
	# --- 5. Prepare KV projection weights ---
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	# If weight quantization is needed, uncomment:
	# kv_b_proj = deepseek_v3_dequantization(
	#     self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	# ).view(self.num_heads, -1, self.kv_lora_rank)
	
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]
	
	# Apply q_absorb multiplication
	q_nope_absorbed = torch.matmul(q_nope, q_absorb)
	# q_nope_absorbed shape: [bsz, num_heads, 1, kv_lora_rank]
	
	# --- 6. Prepare Attention Mask ---
	kv_len = compressed_kv_ref.size(1)
	
	# Extend or truncate attention mask to match kv_len
	if attention_mask.size(-1) < kv_len:
		attention_mask = torch.cat([
			attention_mask, 
			torch.zeros((bsz, kv_len - attention_mask.size(-1)), device=attention_mask.device)
		], dim=-1)
	else:
		attention_mask = attention_mask[:, :kv_len]
	
	# Convert attention mask to the format expected by attention computation
	# attention_mask_processed = attention_mask.unsqueeze(1).unsqueeze(2)
	# attention_mask_processed = torch.where(
	#     attention_mask_processed == 0,
	#     torch.finfo(hidden_states.dtype).min,
	#     0
	# ).to(hidden_states.dtype)
	# attention_mask_processed = update_causal_mask(attention_mask)
	# Option 2: If you need [bsz, num_heads, 1, seq_len]
	mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)
	mask_4d = mask_4d.expand(bsz, self.num_heads, 1, kv_len)

	# For causal masking in decoder (if needed)
	# Create causal mask and combine with padding mask
	# seq_len = attention_mask.size(1)
	# causal_mask = torch.triu(torch.ones(1, seq_len), diagonal=1)
	# attention_mask_processed = causal_mask.masked_fill(causal_mask == 1, float('-inf')).to(hidden_states.device)
	# attention_mask_processed = torch.where(mask_4d == 1, 0.0, float('-inf')).to(hidden_states.device).to(hidden_states.dtype)
	attention_mask_processed = torch.where(
		mask_4d == 1, 
		0.0, 
		torch.finfo(hidden_states.dtype).min
	).to(hidden_states.device)

	
	# --- 7. Compute Attention Weights ---
	# Compute PE attention weights using einsum for proper broadcasting
	# k_pe_states: [bsz, max_seqlen_pad, qk_rope_head_dim]
	# q_pe: [bsz, num_heads, 1, qk_rope_head_dim]
	# Need to add dimension for broadcasting: k_pe_states -> [bsz, 1, max_seqlen_pad, qk_rope_head_dim]
	k_pe_expanded = k_pe_states.unsqueeze(1)
	# check if there is NaN or Inf in q_pe or k_pe_expanded
	# assert not torch.isnan(q_pe).any(), "NaN in q_pe"
	# assert not torch.isinf(q_pe).any(), "Inf in q_pe"
	# assert not torch.isnan(k_pe_expanded).any(), "NaN in k_pe_expanded"
	# assert not torch.isinf(k_pe_expanded).any(), "Inf in k_pe_expanded"
	attn_weights = torch.einsum("bhqd,blcd->bhqc", q_pe, k_pe_expanded)
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights after PE"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights after PE"
	# attn_weights shape: [bsz, num_heads, 1, max_seqlen_pad]
	
	# Compute nope attention weights using einsum
	# kv_states: [bsz, max_seqlen_pad, kv_lora_rank]
	# q_nope_absorbed: [bsz, num_heads, 1, kv_lora_rank]
	# Add dimension for broadcasting: kv_states -> [bsz, 1, max_seqlen_pad, kv_lora_rank]
	kv_states_expanded = kv_states.unsqueeze(1)
	attn_weights = attn_weights + torch.einsum(
		"bhqd,blcd->bhqc", q_nope_absorbed, kv_states_expanded
	)
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights before scaling"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights before scaling"
	# logging attention weights at 0

	# Apply scaling
	attn_weights = attn_weights * self.softmax_scale
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights before mask"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights before mask"
	# attn_weights = attn_weights / (self.qk_nope_head_dim ** 0.5)
	# attention_weights = 1

	
	if attn_weights.size() != (bsz, self.num_heads, q_len, kv_len):
		raise ValueError(
			f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_len)}, but is"
			f" {attn_weights.size()}"
		)
	
	# Apply attention mask
	attn_weights = attn_weights + attention_mask_processed
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights after mask"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights after mask"
	
	# Apply softmax
	attn_weights = nn.functional.softmax(
		attn_weights, dim=-1, dtype=torch.float32
	).to(q_nope.dtype)
	
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights"
	# --- 8. Compute Attention Output ---
	# Use einsum for the attention output computation
	# attn_weights: [bsz, num_heads, 1, max_seqlen_pad]
	# kv_states: [bsz, max_seqlen_pad, kv_lora_rank]
	attn_output = torch.einsum(
		"bhql,blc->bhqc", attn_weights, kv_states
	)
	# attn_output shape: [bsz, num_heads, 1, kv_lora_rank]
	
	# Apply out_absorb projection
	# attn_output = torch.matmul(attn_output, out_absorb.mT)
	attn_output = torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)
	# attn_output shape: [bsz, num_heads, 1, v_head_dim]
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)
	
	# --- 9. Final Projection and Return ---
	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
	attn_output = self.o_proj(attn_output)
	
	return attn_output, past_key_states, scale



@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_bf16_bak(
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

	# --- 1. Dequantize KV Cache ---
	# The cache is received in a quantized format. Dequantize it for processing.
	# compressed_kv_ref = dequant_compressed_kv_per_token(past_key_states, scale, max_seqlen)
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref after dequantization"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref after dequantization"
	# max_seqlen_pad = compressed_kv_ref.size(1)
	_, kv_len, _ = past_key_states.size()
	# assert not torch.isnan(past_key_states).any(), "NaN in past_key_states"
	# assert not torch.isinf(past_key_states).any(), "Inf in past_key_states"


	# --- 2. Query and New Key-Value Projection ---
	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	# assert not torch.isnan(q).any(), "NaN in q after projection"
	# assert not torch.isinf(q).any(), "Inf in q after projection"
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

	# Project the new compressed key-value pair in full precision.
	new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	# assert not torch.isnan(new_compressed_kv).any(), "NaN in new_compressed_kv after projection"
	# assert not torch.isinf(new_compressed_kv).any(), "Inf in new_compressed_kv after projection"
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
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref before update"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref before update"
	
	past_key_states[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	# log if there is NaN or Inf in compressed_kv_ref
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref after update"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref after update"
	
	# Quantize and write to past_key_states
	# new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	# past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	# scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
	
	# --- 4. Split compressed_kv_ref for attention computation ---
	# compressed_kv_ref shape: [bsz, max_seqlen_pad, kv_lora_rank + qk_rope_head_dim]
	kv_states, k_pe_states = torch.split(
		past_key_states, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	# kv_states shape: [bsz, max_seqlen_pad, kv_lora_rank]
	# k_pe_states shape: [bsz, max_seqlen_pad, qk_rope_head_dim]
	
	# --- 5. Prepare KV projection weights ---
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	# If weight quantization is needed, uncomment:
	# kv_b_proj = deepseek_v3_dequantization(
	#     self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	# ).view(self.num_heads, -1, self.kv_lora_rank)
	
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]
	
	# Apply q_absorb multiplication
	q_nope_absorbed = torch.matmul(q_nope, q_absorb)
	# q_nope_absorbed shape: [bsz, num_heads, 1, kv_lora_rank]
	
	# --- 6. Prepare Attention Mask ---
	# kv_len = past_key_states.size(1)
	
	# Extend or truncate attention mask to match kv_len
	if attention_mask.size(-1) < kv_len:
		attention_mask = torch.cat([
			attention_mask, 
			torch.zeros((bsz, kv_len - attention_mask.size(-1)), device=attention_mask.device)
		], dim=-1)
	else:
		attention_mask = attention_mask[:, :kv_len]
	
	# Convert attention mask to the format expected by attention computation
	# attention_mask_processed = attention_mask.unsqueeze(1).unsqueeze(2)
	# attention_mask_processed = torch.where(
	#     attention_mask_processed == 0,
	#     torch.finfo(hidden_states.dtype).min,
	#     0
	# ).to(hidden_states.dtype)
	# attention_mask_processed = update_causal_mask(attention_mask)
	# Option 2: If you need [bsz, num_heads, 1, seq_len]
	mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)
	mask_4d = mask_4d.expand(bsz, self.num_heads, 1, kv_len)

	# For causal masking in decoder (if needed)
	# Create causal mask and combine with padding mask
	# seq_len = attention_mask.size(1)
	# causal_mask = torch.triu(torch.ones(1, seq_len), diagonal=1)
	# attention_mask_processed = causal_mask.masked_fill(causal_mask == 1, float('-inf')).to(hidden_states.device)
	# attention_mask_processed = torch.where(mask_4d == 1, 0.0, float('-inf')).to(hidden_states.device).to(hidden_states.dtype)
	attention_mask_processed = torch.where(
		mask_4d == 1, 
		0.0, 
		torch.finfo(hidden_states.dtype).min
	).to(hidden_states.device)

	
	# --- 7. Compute Attention Weights ---
	# Compute PE attention weights using einsum for proper broadcasting
	# k_pe_states: [bsz, max_seqlen_pad, qk_rope_head_dim]
	# q_pe: [bsz, num_heads, 1, qk_rope_head_dim]
	# Need to add dimension for broadcasting: k_pe_states -> [bsz, 1, max_seqlen_pad, qk_rope_head_dim]
	k_pe_expanded = k_pe_states.unsqueeze(1)
	# check if there is NaN or Inf in q_pe or k_pe_expanded
	# assert not torch.isnan(q_pe).any(), "NaN in q_pe"
	# assert not torch.isinf(q_pe).any(), "Inf in q_pe"
	# assert not torch.isnan(k_pe_expanded).any(), "NaN in k_pe_expanded"
	# assert not torch.isinf(k_pe_expanded).any(), "Inf in k_pe_expanded"
	attn_weights = torch.einsum("bhqd,blcd->bhqc", q_pe, k_pe_expanded)
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights after PE"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights after PE"
	# attn_weights shape: [bsz, num_heads, 1, max_seqlen_pad]
	
	# Compute nope attention weights using einsum
	# kv_states: [bsz, max_seqlen_pad, kv_lora_rank]
	# q_nope_absorbed: [bsz, num_heads, 1, kv_lora_rank]
	# Add dimension for broadcasting: kv_states -> [bsz, 1, max_seqlen_pad, kv_lora_rank]
	kv_states_expanded = kv_states.unsqueeze(1)
	attn_weights = attn_weights + torch.einsum(
		"bhqd,blcd->bhqc", q_nope_absorbed, kv_states_expanded
	)
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights before scaling"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights before scaling"
	# logging attention weights at 0

	# Apply scaling
	attn_weights = attn_weights * self.softmax_scale
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights before mask"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights before mask"


	
	if attn_weights.size() != (bsz, self.num_heads, q_len, kv_len):
		raise ValueError(
			f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_len)}, but is"
			f" {attn_weights.size()}"
		)
	
	# Apply attention mask
	attn_weights = attn_weights + attention_mask_processed
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights after mask"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights after mask"
	
	# Apply softmax
	attn_weights = nn.functional.softmax(
		attn_weights, dim=-1, dtype=torch.float32
	).to(q_nope.dtype)
	
	# assert not torch.isnan(attn_weights).any(), "NaN in attention weights"
	# assert not torch.isinf(attn_weights).any(), "Inf in attention weights"
	# --- 8. Compute Attention Output ---
	# Use einsum for the attention output computation
	# attn_weights: [bsz, num_heads, 1, max_seqlen_pad]
	# kv_states: [bsz, max_seqlen_pad, kv_lora_rank]
	attn_output = torch.einsum(
		"bhql,blc->bhqc", attn_weights, kv_states
	)
	# attn_output shape: [bsz, num_heads, 1, kv_lora_rank]
	
	# Apply out_absorb projection
	# attn_output = torch.matmul(attn_output, out_absorb.mT)
	attn_output = torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)
	# attn_output shape: [bsz, num_heads, 1, v_head_dim]
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)
	
	# --- 9. Final Projection and Return ---
	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
	# assert not torch.isnan(attn_output).any(), "NaN in attn_output before o_proj"
	# assert not torch.isinf(attn_output).any(), "Inf in attn_output before o_proj"
	attn_output = self.o_proj(attn_output)
	# assert not torch.isnan(attn_output).any(), "NaN in attn_output after o_proj"
	# assert not torch.isinf(attn_output).any(), "Inf in attn_output after o_proj"
	
	return attn_output, past_key_states


@torch.inference_mode()
def mla_decoding_flashmla_attn_mode_3_bf16(
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
		cache_seqlens, 128, 1
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
def mla_decoding_flashmla_attn_mode_3_bf16_bak(
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

	# --- 1. Dequantize KV Cache ---
	# The cache is received in a quantized format. Dequantize it for processing.
	# compressed_kv_ref = dequant_compressed_kv_per_token(past_key_states, scale, max_seqlen)
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref after dequantization"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref after dequantization"
	# max_seqlen_pad = compressed_kv_ref.size(1)
	_, kv_len, _ = past_key_states.size()
	# assert not torch.isnan(past_key_states).any(), "NaN in past_key_states"
	# assert not torch.isinf(past_key_states).any(), "Inf in past_key_states"


	# --- 2. Query and New Key-Value Projection ---
	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	# assert not torch.isnan(q).any(), "NaN in q after projection"
	# assert not torch.isinf(q).any(), "Inf in q after projection"
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

	# Project the new compressed key-value pair in full precision.
	new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	# assert not torch.isnan(new_compressed_kv).any(), "NaN in new_compressed_kv after projection"
	# assert not torch.isinf(new_compressed_kv).any(), "Inf in new_compressed_kv after projection"
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
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref before update"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref before update"
	
	past_key_states[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
	# log if there is NaN or Inf in compressed_kv_ref
	# assert not torch.isnan(compressed_kv_ref).any(), "NaN in compressed_kv_ref after update"
	# assert not torch.isinf(compressed_kv_ref).any(), "Inf in compressed_kv_ref after update"
	
	# Quantize and write to past_key_states
	# new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
	# past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
	# scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]
		
	# Extend or truncate attention mask to match kv_len
	# if attention_mask.size(-1) < kv_len:
	# 	attention_mask = torch.cat([
	# 		attention_mask, 
	# 		torch.zeros((bsz, kv_len - attention_mask.size(-1)), device=attention_mask.device)
	# 	], dim=-1)
	# else:
	# 	attention_mask = attention_mask[:, :kv_len]

	kv_seqlen = past_key_states.size(1)
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	# kv_b_proj = deepseek_v3_dequantization(
	# 	self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"]
	# ).view(self.num_heads, -1, self.kv_lora_rank)
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
		cache_seqlens, 128, 1
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
	# attn_output = torch.matmul(attn_out, out_absorb.mT)
	attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
	# attn_output shape: [bsz, num_heads, 1, v_head_dim]
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)
	
	# --- 9. Final Projection and Return ---
	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
	# assert not torch.isnan(attn_output).any(), "NaN in attn_output before o_proj"
	# assert not torch.isinf(attn_output).any(), "Inf in attn_output before o_proj"
	attn_output = self.o_proj(attn_output)
	# assert not torch.isnan(attn_output).any(), "NaN in attn_output after o_proj"
	# assert not torch.isinf(attn_output).any(), "Inf in attn_output after o_proj"
	
	return attn_output, past_key_states[:, :kv_len, :]



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
	assert attention_mask.dim() == 2
	# Create a block table for the key states
	block_size = 64
	cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
	
	bsz, seq_len = attention_mask.size()
	max_seqlen = cache_seqlens.max().item()
	
	# _,max_seq_len, _ = past_key_states.size()
	compressed_kv = dequant_compressed_kv_per_token_with_length(past_key_states, scale, max_seqlen)
	max_seqlen_pad = compressed_kv.size(1)

	# q_position_id = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	# kv_len = attention_mask.size(-1)
	q_position_ids = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	
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
		cache_seqlens, 128, 1
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
	scale
):
	# Create a block table for the key states
	block_size = 64
	cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
	
	max_seqlen = cache_seqlens.max().item()
	max_seqlen_pad = ((max_seqlen + block_size - 1) // block_size) * block_size
	# compressed_kv = dequant_per_token_triton(past_key_states, scale)
	if max_seqlen_pad >= past_key_states.size(1):
		compressed_kv = dequant_per_token_return_with_max_seqlen_pad(
			past_key_states, scale, max_seqlen_pad
		)
	else:
		compressed_kv = past_key_states[:, :max_seqlen_pad, :].contiguous()
		scale = scale[:, :max_seqlen_pad, :].contiguous()
		compressed_kv = dequant_per_token_triton(compressed_kv, scale)

	
	assert attention_mask.dim() == 2
	bsz, q_len, _ = hidden_states.size()
	q_position_id = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	kv_len = attention_mask.size(-1)
	
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
		cache_seqlens, 128, 1
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
	# Create a block table for the key states
	block_size = 64
	cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
	
	max_seqlen = cache_seqlens.max().item()
	max_seqlen_pad = ((max_seqlen + block_size - 1) // block_size) * block_size
	# compressed_kv = dequant_per_token_triton(past_key_states, scale)
	if max_seqlen_pad >= past_key_states.size(1):
		compressed_kv = dequant_per_token_return_with_max_seqlen_pad(
			past_key_states, scale, max_seqlen_pad
		)
	else:
		compressed_kv = past_key_states[:, :max_seqlen_pad, :].contiguous()
		scale = scale[:, :max_seqlen_pad, :].contiguous()
		compressed_kv = dequant_per_token_triton(compressed_kv, scale)

	
	assert attention_mask.dim() == 2
	bsz, q_len, _ = hidden_states.size()
	q_position_id = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	kv_len = attention_mask.size(-1)
	
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
		cache_seqlens, 128, 1
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
