import torch

from batchgen_kernels import load_extension

_HADAMARD_MODULE = (
    "batchgen_kernels.attention.dsa.indexer."
    "batchgen_dsa_fast_hadamard_transform_cuda"
)
_FUSED_ROPE_HADAMARD_MODULE = (
    "batchgen_kernels.attention.dsa.indexer."
    "batchgen_dsa_fused_rope_hadamard_cuda"
)


def _load_required_extension(module_name: str):
    try:
        return load_extension(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import required DSA Hadamard extension {module_name}. "
            "Install the v0.3.1.post3+ AOT batchgen_kernels wheel; production "
            "GLM-5 DSA no longer runtime-JITs Hadamard extensions. "
            f"Import error: {exc}"
        ) from exc


_hadamard_cuda = _load_required_extension(_HADAMARD_MODULE)
_fused_rope_hadamard_cuda = _load_required_extension(_FUSED_ROPE_HADAMARD_MODULE)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard_cuda.fast_hadamard_transform(x, scale)


def fused_rope_hadamard(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    scale: float = 128 ** -0.5,
) -> torch.Tensor:
    """Fused interleaved RoPE + Hadamard transform for dim=128 bf16.

    Args:
        x: [batch, 128] bf16 tensor (after LayerNorm)
        cos_cache: [max_seq, 64] float32 cos cache from rotary embedding
        sin_cache: [max_seq, 64] float32 sin cache from rotary embedding
        positions: [batch] int64 position indices
        scale: Hadamard scale factor (default 1/sqrt(128))

    Returns:
        [batch, 128] bf16 tensor
    """
    return _fused_rope_hadamard_cuda.fused_rope_hadamard(
        x, cos_cache, sin_cache, positions, scale
    )


__all__ = ["hadamard_transform", "fused_rope_hadamard"]
