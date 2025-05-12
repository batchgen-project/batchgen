import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_mla import (
	flash_mla_with_kvcache,
	get_mla_metadata,
)
from .rotary_embedding import rotary_pos_emb
import logging
import torch.distributed as dist

@torch.inference_mode()
def mla_decoding_flashmla(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
):
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

	compressed_kv = past_key_states
	batch_indices = torch.arange(bsz, device=hidden_states.device)
	compressed_kv[batch_indices, q_position_id[:, 0], :self.kv_lora_rank] = kv[:, 0, :]
	compressed_kv[batch_indices, q_position_id[:, 0], self.kv_lora_rank:] = k_pe[:, 0, :]
	compressed_kv = compressed_kv.view(bsz, kv_len, 1, 576)
	
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

	# Create a block table for the key states
	block_size = 64
	cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
	
	max_seqlen = cache_seqlens.max().item()
	max_seqlen_pad = ((max_seqlen + block_size - 1) // block_size) * block_size

	# Pad the compressed_kv_ref tensor to the maximum sequence length
	if max_seqlen_pad > kv_len:
		compressed_kv = torch.cat(
			[
				compressed_kv,
				torch.full(
					(bsz, max_seqlen_pad - kv_len, 1, compressed_kv.size(-1)),
					# float("nan"),
					0,
					dtype=compressed_kv.dtype,
					device=compressed_kv.device,
				),
			],
			dim=1,
		)
	else:
		compressed_kv = compressed_kv[:, :max_seqlen_pad, :, :]

	block_table = torch.arange(
		bsz * max_seqlen_pad // block_size, dtype=torch.int32
	).view(bsz, max_seqlen_pad // block_size).to(compressed_kv.device)

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
