import torch


def clamp_token_indices_to_seqlens(
    token_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Clamp token indices so each row never exceeds its last valid token."""
    cap = (
        cache_seqlens.to(device=token_indices.device, dtype=torch.long).reshape(token_indices.shape[0]) - 1
    ).clamp(min=0).unsqueeze(-1)
    return torch.minimum(token_indices.to(torch.long), cap)


def build_clamped_dense_token_indices(
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    """Build dense token indices capped at each row's last valid token."""
    batch_size = cache_seqlens.shape[0]
    base = torch.arange(
        max_seqlen, device=device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    return clamp_token_indices_to_seqlens(base, cache_seqlens.to(device=device))
