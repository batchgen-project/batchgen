import torch
import torch.nn as nn
from .modeling_deepseek_v3 import (
	apply_rotary_pos_emb,
	rotate_half,
)
import logging
from flash_mla import (
	flash_mla_with_kvcache,
	get_mla_metadata,
)
from .flash_attn_utils import (
	_upad_input,
	pad_input,
	pad_input_cus
)
from flash_attn_interface import flash_attn_varlen_func 
import triton
import torch.distributed as dist
	

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

def rotary_pos_emb(t, cos, sin, position_ids, unsqueeze_dim=1):
	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	b, h, s, d = t.shape
	t = t.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
	t_embed = (t * cos) + (rotate_half(t) * sin)
	return t_embed



def convert_to_causal_attention_mask(attention_mask):
    """
    Convert a 2D attention mask (batch_size, seq_len) to a 4D causal attention mask (batch_size, 1, 1, seq_len).
    Uses 0 for padding tokens and the minimum value of bfloat16 for non-padding tokens.
    
    Args:
        attention_mask: torch.Tensor of shape (batch_size, seq_len) with 0 for padding and 1 for non-padding
    
    Returns:
        causal_mask: torch.Tensor of shape (batch_size, 1, 1, seq_len) with 0 for padding and min of bfloat16 for non-padding
    """
    # Get batch size and sequence length
    batch_size, seq_len = attention_mask.shape
    
    # Create causal mask (lower triangular matrix)
    # This ensures each token can only attend to itself and previous tokens
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask.device))
    
    # Expand attention_mask to match the causal mask shape for broadcasting
    expanded_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    
    # Combine the attention mask with the causal mask
    # This ensures we mask both padding tokens and future tokens
    combined_mask = causal_mask.unsqueeze(0) * expanded_attention_mask
    
    # Convert 0s to min of bfloat16 (approximately -3.4e+38) and keep 0s as 0s
    min_bfloat16 = torch.finfo(torch.bfloat16).min
    
    # Where the combined mask is 0, keep it as 0 (for padding tokens)
    # Where the combined mask is 1, replace with min of bfloat16 (for non-padding tokens that can be attended to)
    final_mask = torch.zeros_like(combined_mask)
    final_mask = torch.where(combined_mask > 0, min_bfloat16 * torch.ones_like(combined_mask), final_mask)
    
    return final_mask

@torch.inference_mode()
def cus_absorbed_mla_decoding_forward(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
):
	assert attention_mask.dim() == 2
	# position_ids of q_pe is the sum of the attention mask minus one 
	q_position_id = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	# logging.info(f"q_position_id shape: {q_position_id.shape}")
	# logging.info(f"q_position_id: {q_position_id}")
	
	# logging.info(f"Rank: {dist.get_rank()}, attention_mask sum: {attention_mask[0].sum()}")
	# attention_mask = convert_to_causal_attention_mask(attention_mask)
	attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
	attention_mask = torch.where(
		attention_mask == 0,
		torch.finfo(torch.bfloat16).min,
		0).to(hidden_states.dtype)
	# logging.info(f"attention_mask 0 :{attention_mask[0]}")

	bsz, q_len, _ = hidden_states.size()

	if self.q_lora_rank is None:
		q = self.q_proj(hidden_states)
	else:
		q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

	# past_key_states = past_key_states.to(torch.bfloat16)
	# compressed_kv_ref = torch.cat([past_key_states, compressed_kv], dim=1)
	# Copy the new compressed_kv to the corresponding position in the past_key_states
	# The position is given by q_position_id.
	# We first concat the past_key_states with a padding of 0
	compressed_kv_ref = torch.cat(
		[past_key_states, torch.zeros_like(compressed_kv)], dim=1
	)
	for idx in range(bsz):
		compressed_kv_ref[idx, q_position_id[idx], :] = compressed_kv[idx,0,:]


	kv_len = compressed_kv_ref.size(1)
	assert kv_len == attention_mask.size(-1)
	compressed_kv_ref, k_pe = torch.split(
		compressed_kv_ref, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	compressed_kv_ref = self.kv_a_layernorm(compressed_kv_ref)

	k_pe = k_pe.view(bsz, 1, kv_len, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=kv_len)
	k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids)
	# q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	# cos, sin = self.rotary_emb(q_pe, seq_len=q_len)
	# q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids)
	# q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids[:, -1].unsqueeze(-1))
	

	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_id)

	q_nope = torch.matmul(q_nope, q_absorb)
	# attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)) * self.softmax_scale
	attn_weights = torch.einsum("bhqd,bhcd->bhqc", q_pe, k_pe)
	attn_weights = attn_weights + torch.einsum(
		"bhqd,bhcd->bhqc", q_nope, compressed_kv_ref.unsqueeze(-3)
	)
	attn_weights = attn_weights * self.softmax_scale
	if attn_weights.size() != (bsz, self.num_heads, q_len, kv_len):
		raise ValueError(
			f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_len)}, but is"
			f" {attn_weights.size()}"
		)
	attn_weights = attn_weights + attention_mask
	# upcast attention to fp32
	attn_weights = nn.functional.softmax(
		attn_weights, dim=-1, dtype=torch.float32
	).to(q_nope.dtype)
	attn_output = torch.einsum(
		"bhql,blc->bhqc", attn_weights, compressed_kv_ref
	)
	attn_output = torch.matmul(
		attn_output, out_absorb.mT
	)  # torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)

	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)

	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(
		bsz, q_len, self.num_heads * self.v_head_dim
	)
	attn_output = self.o_proj(attn_output)

	return (
		attn_output,
		compressed_kv,
		torch.tensor([], device=hidden_states.device),
	)


@torch.no_grad()
def prefill_attn(
	self,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
):
	bsz, q_len, _ = hidden_states.size()
	if self.q_lora_rank is None:
		q = self.q_proj(hidden_states)
	else:
		q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)
	compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

	kv_len = compressed_kv.size(1)
	assert kv_len == attention_mask.size(-1)
	compressed_kv_ref, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	k_pe = k_pe.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
	kv = (
		self.kv_b_proj(self.kv_a_layernorm(compressed_kv_ref))
		.view(
			bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
		)
		.transpose(1, 2)
	)

	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)
	kv_seq_len = attention_mask.shape[-1]
	cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
	q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

	query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
	query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
	query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

	key_states = k_pe.new_empty(
		bsz, self.num_heads, kv_seq_len, self.q_head_dim
	)
	key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
	key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

	attn_weights = (
		torch.matmul(query_states, key_states.transpose(2, 3))
		* self.softmax_scale
	)

	if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
		raise ValueError(
			f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
			f" {attn_weights.size()}"
		)
	assert attention_mask is not None
	if attention_mask is not None:
		if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
			raise ValueError(
				f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
			)
		attn_weights = attn_weights + attention_mask

	# upcast attention to fp32
	attn_weights = nn.functional.softmax(
		attn_weights, dim=-1, dtype=torch.float32
	).to(query_states.dtype)
	attn_weights = nn.functional.dropout(
		attn_weights, p=self.attention_dropout, training=self.training
	)
	attn_output = torch.matmul(attn_weights, value_states)

	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)

	attn_output = attn_output.transpose(1, 2).contiguous()

	attn_output = attn_output.reshape(
		bsz, q_len, self.num_heads * self.v_head_dim
	)

	attn_output = self.o_proj(attn_output)
	return attn_output, compressed_kv


@torch.no_grad()
def chunked_prefill_attn(
	self,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
	chunk_size: int = 2048,  # Added chunk_size parameter with default None
):
	bsz, q_len, _ = hidden_states.size()
	
	if self.q_lora_rank is None:
		q = self.q_proj(hidden_states)
	else:
		q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)
	
	compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	
	kv_len = compressed_kv.size(1)
	assert kv_len == attention_mask.size(-1)
	
	compressed_kv_ref, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	
	k_pe = k_pe.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
	
	kv = (
		self.kv_b_proj(self.kv_a_layernorm(compressed_kv_ref))
		.view(
			bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
		)
		.transpose(1, 2)
	)
	
	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)
	
	kv_seq_len = attention_mask.shape[-1]
	cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
	q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
	
	query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
	query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
	query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
	
	key_states = k_pe.new_empty(
		bsz, self.num_heads, kv_seq_len, self.q_head_dim
	)
	key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
	key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
	
	# Initialize the final attention output tensor
	attn_output = torch.zeros(
		bsz, self.num_heads, q_len, self.v_head_dim, 
		dtype=query_states.dtype, 
		device=query_states.device
	)
	
	# Implement chunked query attention computation
	if chunk_size is None or q_len <= chunk_size:
		# If chunk_size is None or smaller than q_len, compute attention as before
		attn_weights = (
			torch.matmul(query_states, key_states.transpose(2, 3))
			* self.softmax_scale
		)
		
		if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
			raise ValueError(
				f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
				f" {attn_weights.size()}"
			)
			
		if attention_mask is not None:
			if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
				raise ValueError(
					f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
				)
			attn_weights = attn_weights + attention_mask
		
		# upcast attention to fp32
		attn_weights = nn.functional.softmax(
			attn_weights, dim=-1, dtype=torch.float32
		).to(query_states.dtype)
		
		attn_weights = nn.functional.dropout(
			attn_weights, p=self.attention_dropout, training=self.training
		)
		
		attn_output = torch.matmul(attn_weights, value_states)
	else:
		# Process queries in chunks
		num_chunks = (q_len + chunk_size - 1) // chunk_size  # Ceiling division
		
		for chunk_idx in range(num_chunks):
			chunk_start = chunk_idx * chunk_size
			chunk_end = min(chunk_start + chunk_size, q_len)
			
			# Extract query chunk
			query_chunk = query_states[:, :, chunk_start:chunk_end, :]
			
			# Compute attention weights for this query chunk
			chunk_attn_weights = (
				torch.matmul(query_chunk, key_states.transpose(2, 3))
				* self.softmax_scale
			)
			
			# Apply attention mask for this chunk if provided
			if attention_mask is not None:
				attention_mask_chunk = attention_mask[:, :, chunk_start:chunk_end, :]
				chunk_attn_weights = chunk_attn_weights + attention_mask_chunk
			
			# Apply softmax to get normalized attention weights
			chunk_attn_weights = nn.functional.softmax(
				chunk_attn_weights, dim=-1, dtype=torch.float32
			).to(query_states.dtype)
			
			# Apply dropout
			chunk_attn_weights = nn.functional.dropout(
				chunk_attn_weights, p=self.attention_dropout, training=self.training
			)
			
			# Compute attention output for this chunk
			chunk_output = torch.matmul(chunk_attn_weights, value_states)
			
			# Store the output for this chunk
			attn_output[:, :, chunk_start:chunk_end, :] = chunk_output
	
	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)
	
	attn_output = attn_output.transpose(1, 2).contiguous()
	
	attn_output = attn_output.reshape(
		bsz, q_len, self.num_heads * self.v_head_dim
	)
	
	attn_output = self.o_proj(attn_output)
	return attn_output, compressed_kv


@torch.inference_mode()
def FlashMLA_DeepSeekR1(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
):
	assert attention_mask.dim() == 2
	bsz, q_len, _ = hidden_states.size()

	q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	# compressed_kv_ref = torch.cat([past_key_states, compressed_kv], dim=1)
	compressed_kv_ref = torch.cat(
		[past_key_states, torch.zeros_like(compressed_kv)], dim=1
	)
	q_position_id = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	for idx in range(bsz):
		compressed_kv_ref[idx, q_position_id[idx], :] = compressed_kv[idx,0,:]
	
	
	kv_len = compressed_kv_ref.size(1)
	assert kv_len == attention_mask.size(-1)
	compressed_kv_ref, k_pe = torch.split(
		compressed_kv_ref, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	compressed_kv_ref = self.kv_a_layernorm(compressed_kv_ref)

	k_pe = k_pe.view(bsz, 1, kv_len, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=kv_len)
	k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids)
	
	compressed_kv_ref = compressed_kv_ref.view(
		bsz, 1, kv_len, self.kv_lora_rank
	)
	# Concat k_pe back to compressed_kv_ref
	compressed_kv_ref = torch.cat(
		[compressed_kv_ref, k_pe], dim=-1
	).view(bsz, kv_len, 1, 576)

	
	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_id)
	

	qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
	query_states = torch.empty(
		bsz, self.num_heads, q_len, qk_head_dim,
		dtype=compressed_kv_ref.dtype,
		device=compressed_kv_ref.device,
	)

	# query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
	query_states[:, :, :, self.kv_lora_rank :] = q_pe
	query_states = query_states.view(
		bsz, q_len, self.num_heads, qk_head_dim
	)

	# Create a block table for the key states
	block_size = 64
	cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
	
	# max_seqlen = cache_seqlens.max().item()
	# cache_seqlens = torch.full(
	# 	(bsz,), kv_len, dtype=torch.int32, device=compressed_kv_ref.device
	# )
	max_seqlen = kv_len
	max_seqlen_pad = ((max_seqlen + block_size - 1) // block_size) * block_size
	# max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

	# logging.info(f"max_seqlen_pad: {max_seqlen_pad}")

	# Pad the compressed_kv_ref tensor to the maximum sequence length
	compressed_kv_ref = torch.cat(
		[
			compressed_kv_ref,
			torch.full(
				(bsz, max_seqlen_pad - kv_len, 1, compressed_kv_ref.size(-1)),
				# float("nan"),
				0,
				dtype=compressed_kv_ref.dtype,
				device=compressed_kv_ref.device,
			),
		],
		dim=1,
	)

	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32
	).view(bsz, max_seqlen_pad // block_size).to(compressed_kv_ref.device)

	blocked_k = compressed_kv_ref.view(
		bsz * max_seqlen_pad // block_size, block_size, 1, compressed_kv_ref.size(-1)
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

	if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
		raise ValueError(
			f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
			f" {attn_output.size()}"
		)

	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(
		bsz, q_len, self.num_heads * self.v_head_dim
	)
	attn_output = self.o_proj(attn_output)

	return (
		attn_output,
		compressed_kv,
		torch.tensor([], device=hidden_states.device),
	)

@torch.inference_mode()
def hopper_prefill_mla(
	self,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
):
	"""
		MLA prefifill on hooper device.
		Materialize QKV and call flash_attn_varlen_func(). (flash_attn_3 backend)
	"""
	bsz, seq_len, _ = hidden_states.shape
	query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	query_states = query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	q_nope, q_pe = torch.split(
		query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)	
	compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	compressed_kv_ref, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)	
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
	kv = (
		self.kv_b_proj(self.kv_a_layernorm(compressed_kv_ref))
		.view(
			bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
		)
		.transpose(1, 2)
	)
	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)		
	cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
	q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
	
	
	query_states = k_pe.new_empty(bsz, self.num_heads, seq_len, self.q_head_dim)
	query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
	query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

	key_states = k_pe.new_empty(
		bsz, self.num_heads, seq_len, self.q_head_dim
	)
	key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
	key_states[:, :, :, self.qk_nope_head_dim :] = k_pe	

	query_states = query_states.transpose(1, 2)
	key_states = key_states.transpose(1, 2)
	value_states = value_states.transpose(1, 2)

	# Call flash_attn_varlen_func
	(
		query_states,
		key_states,
		value_states,
		indices_q,
		cu_seq_lens,
		max_seq_lens,
	) = _upad_input(
		query_states, key_states, value_states, attention_mask, seq_len
	)      	
	cu_seqlens_q, cu_seqlens_k = cu_seq_lens
	max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
	attn_output_unpad, _ = flash_attn_varlen_func(
		query_states,
		key_states,
		value_states,
		cu_seqlens_q=cu_seqlens_q,
		cu_seqlens_k=cu_seqlens_k,
		max_seqlen_q=max_seqlen_in_batch_q,
		max_seqlen_k=max_seqlen_in_batch_k,
		softmax_scale=self.softmax_scale,
		causal=True
	)		

	attn_output = pad_input(attn_output_unpad, indices_q, bsz, seq_len).view(
		bsz, seq_len, self.num_heads * self.v_head_dim
	).contiguous()

	attn_output = self.o_proj(attn_output)

	return attn_output, compressed_kv



