import torch
import logging
def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1, interleave=True):
	"""Applies Rotary Position Embedding to the query and key tensors.

	`interleave=True` (default) preserves the DeepSeek-V3 / Kimi-K2.5 behavior: the
	input Q/K last dim is assumed to be in interleaved pair layout
	`[x0, x1, x2, x3, ...]` and is permuted to split-half layout before `rotate_half`.
	`interleave=False` is the HF NeoX/Llama split-half layout used by GLM-5, where
	the caller has already produced split-half-formatted tensors and `rotate_half`
	applies directly without any permutation.
	"""

	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	if interleave:
		b, h, s, d = q.shape
		q = q.view(b, h ,s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

		b, h, s, d = k.shape
		k = k.view(b, h ,s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)
	return q_embed, k_embed

def rotary_pos_emb(t, cos, sin, position_ids, unsqueeze_dim=1, interleave=True):
	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	if interleave:
		b, h, s, d = t.shape
		t = t.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
	t_embed = (t * cos) + (rotate_half(t) * sin)
	return t_embed

def mla_rotary_pos_emb(t, cos, sin, position_ids, unsqueeze_dim=1, interleave=True):
	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	if interleave:
		b, s, d = t.shape
		t = t.view(b, s, d // 2, 2).transpose(3, 2).reshape(b, s, d)
	t_embed = (t * cos) + (rotate_half(t) * sin)
	return t_embed