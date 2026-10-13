import torch
from torch.utils.cpp_extension import load
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_csrc_dir = os.path.join(_current_dir, "csrc")

_cuda_flags = [
    '-O3',
    '-std=c++17',
    '--use_fast_math',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
]

_hadamard_cuda = load(
    name="batchgen_glm5_fast_hadamard_transform_cuda",
    sources=[
        os.path.join(_csrc_dir, "hadamard_binding.cpp"),
        os.path.join(_csrc_dir, "fast_hadamard_transform_cuda.cu"),
    ],
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=_cuda_flags,
    extra_include_paths=[_csrc_dir],
    verbose=False,
)

_fused_rope_hadamard_cuda = load(
    name="batchgen_glm5_fused_rope_hadamard_cuda",
    sources=[
        os.path.join(_csrc_dir, "fused_rope_hadamard_binding.cpp"),
        os.path.join(_csrc_dir, "fused_rope_hadamard.cu"),
    ],
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=_cuda_flags,
    extra_include_paths=[_csrc_dir],
    verbose=False,
)


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
