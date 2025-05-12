"""
	For Hooper GPU.
	- prefill_fa3()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn_interface import flash_attn_varlen_func 
from .padding import _upad_input, pad_input
from .rotary_embedding import mla_rotary_pos_emb, rotary_pos_emb, apply_rotary_pos_emb
import logging

@torch.inference_mode()
def mla_prefill_flashattention3(
	self,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
		MLA prefifill on hooper device.
		Materialize QKV and call flash_attn_varlen_func(). (flash_attn_3 backend)
	"""
	bsz, seq_len, _ = hidden_states.shape
	query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	# query_states = query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
	query_states = query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim)
	q_nope, q_pe = torch.split(
		query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)	
	compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	compressed_kv, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)	
	# k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
	normed_kv = self.kv_a_layernorm(compressed_kv)
	# kv = self.kv_b_proj(self.kv_a_layernorm(kv))
		# .view(
		# 	bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
		# )
		# .transpose(1, 2)
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(q_pe, seq_len=seq_len)
	q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids, 2)
	k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids, 2)
	k_pe = k_pe.view(bsz, seq_len, self.qk_rope_head_dim)
	# q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
	offload_kv = torch.cat(
		[normed_kv, k_pe], dim=-1
	)

	kv = self.kv_b_proj(normed_kv)
	kv = kv.view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)		

	
	query_states = k_pe.new_empty(bsz, seq_len, self.num_heads, self.q_head_dim)
	query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
	query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

	key_states = k_pe.new_empty(
		bsz, seq_len, self.num_heads, self.q_head_dim
	)
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
	key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
	key_states[:, :, :, self.qk_nope_head_dim :] = k_pe	

	# query_states = query_states.transpose(1, 2)
	# key_states = key_states.transpose(1, 2)
	# value_states = value_states.transpose(1, 2)
	query_states = query_states.contiguous()
	key_states = key_states.contiguous()
	value_states = value_states.contiguous()
	# logging.info(
	# 	f"query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}"
	# )

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

	# logging.info(f"offload_kv: {offload_kv.shape}")
	return attn_output, offload_kv




