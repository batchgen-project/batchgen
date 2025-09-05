import torch.nn.functional as F
import torch
def greedy_decode(logits):
	"""
	Greedily decode the next token from logits.
	
	Args:
		logits: Tensor of shape [batch_size, vocab_size] containing the logits
	
	Returns:
		Tensor of shape [batch_size, 1] containing the indices of the selected tokens
	"""
	return torch.argmax(logits, dim=-1, keepdim=True)


def sample_with_temperature_top_p(logits, temperature=1.0, top_p=0.9, top_k=None):
    """
    Sample from logits using temperature and top-p (nucleus) sampling.
    Args:
        logits: Tensor of shape [batch_size, vocab_size] containing the logits
        temperature: Float controlling randomness (0.0 = deterministic, higher = more random)
        top_p: Float for nucleus sampling (0.0 to 1.0), keeps smallest set with cumulative prob >= top_p
        top_k: Optional int to limit sampling to top k tokens (can be combined with top_p)
    Returns:
        Sampled token indices of shape [batch_size, 1]
    """
    if temperature < 1e-6:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')

    #
    # === START OF THE FIX ===
    #
    # Add a safeguard to prevent rows with all -inf logits.
    # If a row is all -inf, we'll set the logit for the original most likely token to 0.0.
    # This ensures that softmax doesn't produce NaNs.
    all_filtered = torch.all(logits == -float('Inf'), dim=-1)
    if torch.any(all_filtered):
        # Find the original argmax token indices for the problematic rows
        original_argmax = torch.argmax(logits / temperature, dim=-1)
        # For each problematic row, set its argmax logit back to a valid number (e.g., 0.0)
        # We use a mask to only modify the rows that were all -inf
        logits[all_filtered, original_argmax[all_filtered]] = 0.0
    #
    # === END OF THE FIX ===
    #

    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return sampled