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


def rotary_pos_emb_interleaved_native(t, cos_cached, sin_cached, position_ids, unsqueeze_dim=1):
	"""Native interleaved RoPE (DeepSeek-V3.2 / SGLang `is_neox_style=False`).

	Applies pair-wise rotation: for each adjacent pair ``(x[2i], x[2i+1])`` at
	frequency ``inv_freq[i]``, rotates by angle ``position * inv_freq[i]``.
	Returns the rotated tensor in the SAME interleaved layout (no permutation).

	This is the convention the GLM-5-FP8 config declares via
	``rope_interleave=true`` and that SGLang's ``get_rope_wrapper(
	is_neox_style=False, ...)`` produces. The prior ``rotary_pos_emb`` helper
	above uses a ``view/transpose/reshape`` trick that achieves a mathematically
	equivalent ``Q·Kᵀ`` under dot-product invariance but outputs a permuted
	layout. Using this function avoids the permutation and matches HF /
	SGLang / vLLM native interleaved output element-for-element.

	Args:
		t: ``[..., head_dim]``. ``head_dim`` must be even.
		cos_cached: ``[max_pos, head_dim]`` with first half == second half
		    (BatchGen's ``emb = cat((freqs, freqs))`` format). Only the first
		    half is read.
		sin_cached: ``[max_pos, head_dim]``, same format as ``cos_cached``.
		position_ids: integer tensor of position indices.
		unsqueeze_dim: broadcast axis for cos/sin when ``t`` has extra dims
		    (e.g. heads). Matches the convention of ``rotary_pos_emb`` /
		    ``apply_rotary_pos_emb`` so the caller pattern is unchanged.

	Returns:
		Rotated tensor, same shape and dtype as ``t``.
	"""
	orig_dtype = t.dtype
	d = t.shape[-1]
	d_half = d // 2

	# cos_cached is [max_pos, d] with the second half duplicating the first.
	# Native interleaved only needs the first half (one value per pair).
	cos_half = cos_cached[..., :d_half]
	sin_half = sin_cached[..., :d_half]

	# Index by position then broadcast over heads via unsqueeze_dim. Shape:
	#   cos_pos: [B, T, d_half] -> [B, 1, T, d_half] (if unsqueeze_dim=1)
	cos_pos = cos_half[position_ids].unsqueeze(unsqueeze_dim)
	sin_pos = sin_half[position_ids].unsqueeze(unsqueeze_dim)

	# View t as pairs along the last dim: [..., d_half, 2]. FP32 rotation for
	# numerical stability (matches the FP32 cos/sin cache convention).
	t_f32 = t.float().view(*t.shape[:-1], d_half, 2)
	x_even = t_f32[..., 0]   # [..., d_half]
	x_odd = t_f32[..., 1]    # [..., d_half]

	rot_even = x_even * cos_pos - x_odd * sin_pos
	rot_odd = x_even * sin_pos + x_odd * cos_pos

	out = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
	return out.to(orig_dtype)

def mla_rotary_pos_emb(t, cos, sin, position_ids, unsqueeze_dim=1):
	orig_dtype = t.dtype
	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	b, s, d = t.shape
	t = t.view(b, s, d // 2, 2).transpose(3, 2).reshape(b, s, d)
	t_embed = (t * cos) + (rotate_half(t) * sin)
	return t_embed.to(orig_dtype)