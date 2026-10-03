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
        return load_extension(module_name, allow_dev_jit=False)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import required DSA Hadamard extension {module_name}. "
            "Build batchgen_kernels AOT before serving GLM-5; production DSA "
            "must not compile CUDA code at request time. "
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
    return _fused_rope_hadamard_cuda.fused_rope_hadamard(x, cos_cache, sin_cache, positions, scale)


def fused_rope_hadamard_out(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    scale: float = 128 ** -0.5,
) -> torch.Tensor:
    """Out-buffer fused interleaved RoPE + Hadamard transform for graph capture."""
    _fused_rope_hadamard_cuda.fused_rope_hadamard_out(
        x,
        cos_cache,
        sin_cache,
        positions,
        out,
        scale,
    )
    return out
