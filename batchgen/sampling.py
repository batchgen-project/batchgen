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


def sample_with_temperature_top_p(logits, temperature=1.0, top_p=0.9, top_k=None, eps=1e-10):
    """
    Sample from logits using temperature and top-p (nucleus) sampling with robust error handling.
    Args:
        logits: Tensor of shape [batch_size, vocab_size] containing the logits
        temperature: Float controlling randomness (0.0 = deterministic, higher = more random)
        top_p: Float for nucleus sampling (0.0 to 1.0), keeps smallest set with cumulative prob >= top_p
        top_k: Optional int to limit sampling to top k tokens (can be combined with top_p)
        eps: Small epsilon value to prevent numerical issues
    Returns:
        Sampled token indices of shape [batch_size, 1]
    """
    # Handle deterministic case
    if temperature < 1e-6:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Clone logits to avoid modifying the original
    logits = logits.clone()
    
    # Check for invalid logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: Found NaN or Inf in logits, using uniform sampling")
        batch_size, vocab_size = logits.shape
        return torch.randint(0, vocab_size, (batch_size, 1), device=logits.device)
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        topk_values, _ = torch.topk(logits, top_k)
        min_values = topk_values[:, -1:].expand_as(logits)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float('inf')), logits)
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Create mask for tokens to remove (keeping at least one token)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False  # Always keep the top token
        
        # Scatter back to original order
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Check for valid probabilities
    if torch.isnan(probs).any():
        print("Warning: NaN in probabilities after softmax, using uniform sampling")
        batch_size, vocab_size = logits.shape
        return torch.randint(0, vocab_size, (batch_size, 1), device=logits.device)
    
    # Check if all probabilities are zero (can happen with extreme filtering)
    prob_sums = probs.sum(dim=-1, keepdim=True)
    if (prob_sums < eps).any():
        print("Warning: All probabilities near zero, using uniform sampling for affected batches")
        # For batches with zero probability sums, use uniform sampling
        uniform_probs = torch.ones_like(probs) / probs.size(-1)
        probs = torch.where(prob_sums < eps, uniform_probs, probs)
    
    # Sample from the distribution
    try:
        sampled = torch.multinomial(probs, num_samples=1)
    except RuntimeError as e:
        print(f"Error in multinomial sampling: {e}")
        print(f"Probs shape: {probs.shape}")
        print(f"Probs sum: {probs.sum(dim=-1)}")
        print(f"Probs min/max: {probs.min()}/{probs.max()}")
        # Fallback to argmax sampling
        sampled = torch.argmax(probs, dim=-1, keepdim=True)
    
    return sampled