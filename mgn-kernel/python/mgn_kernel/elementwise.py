import torch


def fused_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Fused RMS normalization kernel.

    Args:
        input (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
        weight (torch.Tensor): Weight tensor of shape (hidden_size,).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Normalized output tensor.
    """
    return torch.ops.mgn_kernel.fused_rmsnorm.default(input, weight, eps)
