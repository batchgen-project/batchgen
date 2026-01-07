"""Token sampling utilities for batch inference."""

import logging
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
	temperature: float = None,
	top_p: float = None,
	top_k: int = 0,
) -> torch.Tensor:
	"""
	Efficient token sampling with temperature, top-p, and top-k.

	Args:
		logits: [batch_size, vocab_size] raw logits from model
		temperature: Scaling factor. None or <=0 = greedy (deterministic)
		top_p: Nucleus sampling threshold. None or >=1.0 = disabled
		top_k: Top-k filtering. 0 = disabled

	Returns:
		[batch_size, 1] sampled token indices
	"""
	# Fast path: greedy decoding (deterministic)
	if temperature is None or temperature <= 0:
		return logits.argmax(dim=-1, keepdim=True)

	# Apply temperature scaling (in-place for efficiency on contiguous tensors)
	if temperature != 1.0:
		logits = logits / temperature

	# Apply top-k filtering
	if top_k > 0:
		top_k = min(top_k, logits.size(-1))
		# Get threshold value (k-th largest)
		threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
		logits = logits.masked_fill(logits < threshold, float('-inf'))

	# Apply top-p (nucleus) filtering
	if top_p is not None and top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
		cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

		# Remove tokens with cumulative prob above threshold (keep at least one)
		sorted_mask = cumulative_probs > top_p
		sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
		sorted_mask[:, 0] = False

		# Scatter mask back to original indices
		mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
		logits = logits.masked_fill(mask, float('-inf'))

	# Convert to probabilities and sample
	probs = F.softmax(logits, dim=-1)

	# Handle edge cases (all -inf after filtering)
	if torch.isnan(probs).any() or (probs.sum(dim=-1) < 1e-10).any():
		logger.warning("Invalid probabilities after filtering, falling back to greedy")
		return logits.argmax(dim=-1, keepdim=True)

	return torch.multinomial(probs, num_samples=1)


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
