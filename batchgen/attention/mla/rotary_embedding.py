import torch
import logging
def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
	"""Applies Rotary Position Embedding to the query and key tensors.

	Args:
		q (`torch.Tensor`): The query tensor.
		k (`torch.Tensor`): The key tensor.
		cos (`torch.Tensor`): The cosine part of the rotary embedding.
		sin (`torch.Tensor`): The sine part of the rotary embedding.
		position_ids (`torch.Tensor`):
			The position indices of the tokens corresponding to the query and key tensors. For example, this can be
			used to pass offsetted position ids when working with a KV-cache.
		unsqueeze_dim (`int`, *optional*, defaults to 1):
			The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
			sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
			that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
			k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
			cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
			the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
	Returns:
		`tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
	"""

	# Cos/sin are kept in FP32 for precision (see Glm5RotaryEmbedding); cast
	# back to input dtype at the end so FA3 (bf16/fp16/fp8 only) doesn't
	# reject the rotated tensors.
	q_dtype = q.dtype
	k_dtype = k.dtype

	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	b, h, s, d = q.shape
	q = q.view(b, h ,s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

	b, h, s, d = k.shape
	k = k.view(b, h ,s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)
	return q_embed.to(q_dtype), k_embed.to(k_dtype)

def rotary_pos_emb(t, cos, sin, position_ids, unsqueeze_dim=1):
	# Cos/sin are kept in FP32 (Glm5RotaryEmbedding); rotation auto-promotes
	# to FP32. Cast result back to t.dtype here so downstream kernels (FA3
	# flash_attn_varlen_func, flash-MLA decode) that only accept bf16/fp16/fp8
	# aren't handed an FP32 tensor via pre-allocated assembly buffers.
	orig_dtype = t.dtype
	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	b, h, s, d = t.shape
	t = t.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
	t_embed = (t * cos) + (rotate_half(t) * sin)
	return t_embed.to(orig_dtype)

def mla_rotary_pos_emb(t, cos, sin, position_ids, unsqueeze_dim=1):
	orig_dtype = t.dtype
	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	b, s, d = t.shape
	t = t.view(b, s, d // 2, 2).transpose(3, 2).reshape(b, s, d)
	t_embed = (t * cos) + (rotate_half(t) * sin)
	return t_embed.to(orig_dtype)