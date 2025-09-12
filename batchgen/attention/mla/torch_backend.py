import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.attention.attention_mask import convert_to_causal_attention_mask
from batchgen.attention.mla.rotary_embedding import (
	apply_rotary_pos_emb,
	rotary_pos_emb,
)
from batchgen.gemm.w8a8 import w8a8_gemm
from batchgen.quantization.fp8e4m3 import (
	dequant_compressed_kv_per_token,
	per_token_blocked_quantize_bf16_to_fp8,
)


@torch.inference_mode()
def mla_decoding_torch(
	self,
	hidden_states: torch.Tensor,
	past_key_states: torch.Tensor,
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	assert attention_mask.dim() == 2
	q_position_id = (attention_mask.sum(-1) - 1).unsqueeze(-1)
	attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
	attention_mask = torch.where(
		attention_mask == 0,
		torch.finfo(torch.bfloat16).min,
		0).to(hidden_states.dtype)

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
	compressed_kv_ref = past_key_states
	for idx in range(bsz):
		compressed_kv_ref[idx, q_position_id[idx], :] = compressed_kv[idx,0,:]


	kv_len = compressed_kv_ref.size(1)
	assert kv_len == attention_mask.size(-1)
	compressed_kv_ref, k_pe = torch.split(
		compressed_kv_ref, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	compressed_kv_ref = self.kv_a_layernorm(compressed_kv_ref) # 4.8 ms

	k_pe = k_pe.view(bsz, 1, kv_len, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(k_pe, seq_len=kv_len)
	k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids)

	kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
	q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
	out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_id)

	q_nope = torch.matmul(q_nope, q_absorb) # 0.4ms
	attn_weights = torch.einsum("bhqd,bhcd->bhqc", q_pe, k_pe) # 0.3 ms
	attn_weights = attn_weights + torch.einsum(
		"bhqd,bhcd->bhqc", q_nope, compressed_kv_ref.unsqueeze(-3) # einsum 2.3ms
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

@torch.inference_mode()
def mla_decoding_torch_with_fp8_kv(
	self,
	hidden_states: torch.Tensor, # [bs, 1, hidden_dim]
	past_key_states: torch.Tensor, # [bs, max_input + max_output, kv_dim]
	past_value_states: torch.Tensor,
	attention_mask: torch.Tensor, # (bsz, seq_len)
	q_position_ids: torch.Tensor, # (bsz, 1)
	scale: torch.Tensor, # [bs, max_input + max_output, kv_dim]
	cache_seqlens: torch.Tensor,
	max_seqlen: int,
	weight_scale: dict = None
):
    # attention_mask: (bsz, seq_len)
    assert attention_mask.dim() == 2
    bsz, seq_len = attention_mask.size()
    # (bsz, max_seqlen_pad, kv_dim) where max_seqlen_pad is aligned (e.g., to 64)
    compressed_kv = dequant_compressed_kv_per_token(past_key_states, scale, max_seqlen)

    # max_seqlen_pad is the aligned effective max sequence length
    max_seqlen_pad = compressed_kv.size(1)
    
    # Reshape attention_mask to [bs, max_seqlen_pad] while preserving all 1s
    if seq_len > max_seqlen_pad:
        # Remove zeros from the left, keep rightmost max_seqlen_pad tokens (including all 1s)
        attention_mask = attention_mask[:, :max_seqlen_pad]
    elif seq_len == max_seqlen_pad:
        # Length matches exactly, no need to modify
        pass
    else:  # seq_len < max_seqlen_pad
         # pad on the right with zeros using torch.nn.functional.pad
        attention_mask = F.pad(attention_mask, (0, max_seqlen_pad - seq_len), "constant", 0)

    assert attention_mask.size(1) == max_seqlen_pad, f"attention_mask size {attention_mask.size()} does not match max_seqlen_pad {max_seqlen_pad}"
    # Convert to additive mask
    attention_mask = torch.where(
        attention_mask == 0,
        torch.finfo(hidden_states.dtype).min,
        0.0
    ).to(hidden_states.dtype)
    attention_mask = attention_mask[:, None, None, :] # broadcast shape for attn_weights

    
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, 1, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
		q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)
    new_compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

    # import pdb; pdb.set_trace()

    kv, k_pe = torch.split(new_compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) # [bsz, 1, kv_lora_rank], [bsz, 1, qk_rope_head_dim]
    k_pe = k_pe.view(bsz, 1, 1, self.qk_rope_head_dim)
    cos, sin = self.rotary_emb(k_pe, seq_len=max_seqlen_pad)
    k_pe = rotary_pos_emb(k_pe, cos, sin, q_position_ids)
    q_pe = rotary_pos_emb(q_pe, cos, sin, q_position_ids)
    kv = self.kv_a_layernorm(kv)
    k_pe = k_pe.view(bsz, 1, self.qk_rope_head_dim)

    # update compressed_kv with the new key and value
    offload_kv = torch.cat([kv, k_pe], dim=-1)
    batch_indices = torch.arange(bsz, device=hidden_states.device)
    compressed_kv[batch_indices, q_position_ids[:, 0], :] = offload_kv[:, 0, :]
    compressed_kv = compressed_kv.view(bsz, max_seqlen_pad, 1, self.kv_lora_rank + self.qk_rope_head_dim).transpose(1, 2)
    compressed_kv, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )

    # Quantize and write to past_key_states
    new_compressed_kv_fp8, new_scale = per_token_blocked_quantize_bf16_to_fp8(offload_kv)
    past_key_states[batch_indices, q_position_ids[:, 0], :] = new_compressed_kv_fp8[:, 0, :]
    scale[batch_indices, q_position_ids[:, 0], :] = new_scale[:, 0, :]

    kv_b_proj = self.kv_b_proj.weight.view(
		self.num_heads, -1, self.kv_lora_rank
	)
    q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
    out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

    q_nope = torch.matmul(q_nope, q_absorb)

    attn_weights = torch.einsum("bhqd,bhcd->bhqc", q_pe, k_pe) # (bsz, num_heads, 1, max_seqlen_pad)
    attn_weights += torch.einsum(
		"bhqd,bhcd->bhqc", q_nope, compressed_kv # einsum
    )
    
    attn_weights = attn_weights * self.softmax_scale
    if attn_weights.size() != (bsz, self.num_heads, 1, max_seqlen_pad):
        raise ValueError(
			f"Attention weights should be of size {(bsz, self.num_heads, max_seqlen_pad, kv.size(1))}, but is"
			f" {attn_weights.size()}"
		)

    attn_weights = attn_weights + attention_mask
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
		attn_weights, dim=-1, dtype=torch.float32
    ).to(q_nope.dtype)

    attn_output = torch.einsum(
		"bhql,bld->bhqd", attn_weights, compressed_kv.squeeze(1) # einsum
    )
    # import pdb; pdb.set_trace()
    attn_output = torch.matmul(
		attn_output, out_absorb.mT
    )  # torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)

    if attn_output.size() != (bsz, self.num_heads, 1, self.v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, 1, self.v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
		bsz, 1, self.num_heads * self.v_head_dim
    )
    attn_output = self.o_proj(attn_output)

    return attn_output, past_key_states, scale


@torch.inference_mode()
def mla_prefill_torch(
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

@torch.inference_mode()
def mla_chunked_prefill_torch(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    chunk_size: int = 1024,  # Added chunk_size parameter with default None
):
    assert attention_mask.dim() == 2
    attention_mask = convert_to_causal_attention_mask(attention_mask)

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