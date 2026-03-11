"""Token sampling utilities for batch inference.

Supports per-sequence sampling parameters via [B] tensors for temperature,
top_p, and top_k. Scalar params are auto-broadcast for backward compatibility.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
	"""
	Greedily decode the next token from logits.

	Args:
		logits: Tensor of shape [batch_size, vocab_size] containing the logits

	Returns:
		Tensor of shape [batch_size, 1] containing the indices of the selected tokens
	"""
	return torch.argmax(logits, dim=-1, keepdim=True)


@torch.inference_mode()
def sample_tokens(
	logits: torch.Tensor,
	temperature: Union[float, torch.Tensor, None] = None,
	top_p: Union[float, torch.Tensor, None] = None,
	top_k: Union[int, torch.Tensor] = 0,
) -> torch.Tensor:
	"""
	Efficient token sampling with per-sequence temperature, top-p, and top-k.

	Supports both scalar params (applied uniformly) and [B] tensor params
	(per-sequence). Mixed greedy/sampling batches are split and handled
	separately for mathematical exactness.

	Args:
		logits: [batch_size, vocab_size] raw logits from model
		temperature: Scaling factor. None or <=0 = greedy. Scalar or [B] tensor.
		top_p: Nucleus sampling threshold. None or >=1.0 = disabled. Scalar or [B] tensor.
		top_k: Top-k filtering. 0 = disabled. Scalar or [B] tensor.

	Returns:
		[batch_size, 1] sampled token indices
	"""
	B, V = logits.shape

	# --- Determine greedy mask ---
	# Scalar fast path: all greedy or all same params
	if temperature is None or (isinstance(temperature, (int, float)) and temperature <= 0):
		return logits.argmax(dim=-1, keepdim=True)

	# Convert scalars to [B] tensors for uniform code path
	if isinstance(temperature, (int, float)):
		temps = torch.full((B,), temperature, device=logits.device, dtype=logits.dtype)
	else:
		temps = temperature.to(device=logits.device, dtype=logits.dtype)

	if isinstance(top_p, (int, float)):
		top_ps = torch.full((B,), top_p if top_p is not None else 1.0, device=logits.device, dtype=logits.dtype)
	elif top_p is None:
		top_ps = torch.ones(B, device=logits.device, dtype=logits.dtype)
	else:
		top_ps = top_p.to(device=logits.device, dtype=logits.dtype)

	if isinstance(top_k, int):
		top_ks = torch.full((B,), top_k, device=logits.device, dtype=torch.int64)
	else:
		top_ks = top_k.to(device=logits.device, dtype=torch.int64)

	# --- Split greedy vs sampling ---
	greedy_mask = temps <= 0
	sampling_mask = ~greedy_mask

	result = torch.empty(B, 1, dtype=torch.long, device=logits.device)

	# Handle greedy sequences
	if greedy_mask.any():
		result[greedy_mask] = logits[greedy_mask].argmax(dim=-1, keepdim=True)

	# Handle sampling sequences
	if not sampling_mask.any():
		return result

	s_logits = logits[sampling_mask]  # [B_s, V]
	s_temps = temps[sampling_mask]    # [B_s]
	s_top_ps = top_ps[sampling_mask]  # [B_s]
	s_top_ks = top_ks[sampling_mask]  # [B_s]
	B_s = s_logits.shape[0]

	# --- Temperature scaling ---
	s_logits = s_logits / s_temps.unsqueeze(-1)  # [B_s, V] / [B_s, 1]

	# --- Top-k filtering (vectorized per-sequence k) ---
	max_k = s_top_ks.max().item()
	if max_k > 0:
		max_k = min(max_k, V)
		# Sort descending to get ranks
		sorted_logits, sorted_indices = torch.sort(s_logits, dim=-1, descending=True)
		# Create rank tensor [1, V] and compare with per-seq top_k [B_s, 1]
		ranks = torch.arange(V, device=logits.device).unsqueeze(0)  # [1, V]
		# For sequences with top_k=0, set effective_k=V (no filtering)
		effective_k = s_top_ks.clone()
		effective_k[effective_k <= 0] = V
		effective_k = effective_k.clamp(max=V)
		topk_mask = ranks >= effective_k.unsqueeze(-1)  # [B_s, V]
		sorted_logits[topk_mask] = float('-inf')
		# Scatter back to original order
		s_logits = torch.zeros_like(s_logits).scatter(1, sorted_indices, sorted_logits)

	# --- Top-p (nucleus) filtering (vectorized per-sequence threshold) ---
	needs_top_p = (s_top_ps < 1.0).any()
	if needs_top_p:
		sorted_logits, sorted_indices = torch.sort(s_logits, dim=-1, descending=True)
		cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

		# Remove tokens with cumulative prob above threshold, but keep at least one.
		# Same logic as original scalar code: shift right so the token that
		# crosses the threshold is kept, but everything after is removed.
		sorted_mask = cumulative_probs > s_top_ps.unsqueeze(-1)  # [B_s, V]
		# Shift: position i gets mask from position i-1
		sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
		sorted_mask[:, 0] = False  # always keep top-1

		# For sequences with top_p >= 1.0, disable filtering entirely
		no_filter = (s_top_ps >= 1.0).unsqueeze(-1)
		sorted_mask[no_filter.expand_as(sorted_mask)] = False

		# Scatter mask back to original indices and apply
		mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
		s_logits = s_logits.masked_fill(mask, float('-inf'))

	# --- Sample ---
	probs = F.softmax(s_logits, dim=-1)

	# Per-sequence NaN fallback: if filtering removed all tokens, use argmax
	bad_rows = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) < 1e-10)
	if bad_rows.any():
		logger.warning(
			f"Invalid probabilities after filtering for {bad_rows.sum().item()} sequences, "
			"falling back to greedy for those"
		)
		# Fix bad rows with uniform over original logits
		fallback_tokens = logits[sampling_mask][bad_rows].argmax(dim=-1, keepdim=True)
		# Sample good rows
		good_rows = ~bad_rows
		sampled = torch.empty(B_s, 1, dtype=torch.long, device=logits.device)
		if good_rows.any():
			sampled[good_rows] = torch.multinomial(probs[good_rows], num_samples=1)
		sampled[bad_rows] = fallback_tokens
	else:
		sampled = torch.multinomial(probs, num_samples=1)

	result[sampling_mask] = sampled
	return result


# Backward compatibility alias
def sample_with_temperature_top_p(
	logits: torch.Tensor,
	temperature: float = 1.0,
	top_p: float = 0.9,
	top_k: int = None,
	eps: float = 1e-10,
) -> torch.Tensor:
	"""
	Legacy sampling function for backward compatibility.

	Args:
		logits: Tensor of shape [batch_size, vocab_size] containing the logits
		temperature: Float controlling randomness (0.0 = deterministic)
		top_p: Float for nucleus sampling (0.0 to 1.0)
		top_k: Optional int to limit sampling to top k tokens
		eps: Ignored (kept for API compatibility)

	Returns:
		Sampled token indices of shape [batch_size, 1]
	"""
	return sample_tokens(
		logits,
		temperature=temperature if temperature >= 1e-6 else None,
		top_p=top_p,
		top_k=top_k or 0,
	)
