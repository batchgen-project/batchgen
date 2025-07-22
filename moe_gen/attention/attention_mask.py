import torch


def convert_to_causal_attention_mask(attention_mask):
    """
    Convert a 2D attention mask (batch_size, seq_len) to a 4D causal attention mask (batch_size, 1, 1, seq_len).
    Uses 0 for padding tokens and the minimum value of bfloat16 for non-padding tokens.

    Args:
        attention_mask: torch.Tensor of shape (batch_size, seq_len) with 0 for padding and 1 for non-padding

    Returns:
        causal_mask: torch.Tensor of shape (batch_size, 1, 1, seq_len) with 0 for padding and min of bfloat16 for non-padding
    """
    # Get batch size and sequence length
    batch_size, seq_len = attention_mask.shape

    # Create causal mask (lower triangular matrix)
    # This ensures each token can only attend to itself and previous tokens
    causal_mask = torch.tril(
        torch.ones((seq_len, seq_len), device=attention_mask.device)
    )

    # Expand attention_mask to match the causal mask shape for broadcasting
    expanded_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Combine the attention mask with the causal mask
    # This ensures we mask both padding tokens and future tokens
    combined_mask = causal_mask.unsqueeze(0) * expanded_attention_mask

    # Convert 0s to min of bfloat16 (approximately -3.4e+38) and keep 0s as 0s
    min_bfloat16 = torch.finfo(torch.bfloat16).min

    # Where the combined mask is 0, keep it as 0 (for padding tokens)
    # Where the combined mask is 1, replace with min of bfloat16 (for non-padding tokens that can be attended to)
    final_mask = torch.zeros_like(combined_mask)
    final_mask = torch.where(
        combined_mask > 0,
        min_bfloat16 * torch.ones_like(combined_mask),
        final_mask,
    )

    return final_mask
