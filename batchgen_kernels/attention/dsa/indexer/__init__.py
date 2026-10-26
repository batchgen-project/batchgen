import torch
from torch.utils.cpp_extension import load
from typing import Optional
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
    name="fast_hadamard_transform_cuda",
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
    name="fused_rope_hadamard_cuda",
    sources=[
        os.path.join(_csrc_dir, "fused_rope_hadamard_binding.cpp"),
        os.path.join(_csrc_dir, "fused_rope_hadamard.cu"),
    ],
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=_cuda_flags,
    extra_include_paths=[_csrc_dir],
    verbose=False,
)


# Fused indexer-K quantize + page-split write. Unlike the two kernels above it
# is an AOT CUDA extension (setup.py + _jit_registry), and its wrapper loads
# that extension lazily, so importing this package does not build it.
from batchgen_kernels.attention.dsa.indexer.indexer_k_write_fp8 import (  # noqa: E402
    fused_indexer_k_write_fp8,
    page_split_write_plan,
)

# Same AOT/lazy arrangement for the fused LayerNorm + RoPE + Hadamard +
# quantize + write kernel that replaces the three-launch K tail.
from batchgen_kernels.attention.dsa.indexer.indexer_k_norm_rope_write_fp8 import (  # noqa: E402
    fused_indexer_k_norm_rope_write_fp8,
    norm_rope_write_plan,
)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard_cuda.fast_hadamard_transform(x, scale)


def fused_rope_hadamard(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    scale: float = 128 ** -0.5,
    apply_hadamard: bool = True,
) -> torch.Tensor:
    """Fused interleaved RoPE + optional Hadamard transform for dim=128 bf16.

    Args:
        x: [batch, 128] bf16 tensor (after LayerNorm)
        cos_cache: [max_seq, 64] float32 cos cache from rotary embedding
        sin_cache: [max_seq, 64] float32 sin cache from rotary embedding
        positions: [batch] int64 position indices
        scale: Hadamard scale factor (default 1/sqrt(128)); applied only
            when ``apply_hadamard`` is True, because it normalizes H.
        apply_hadamard: run the 128-point Hadamard rotation. True is the bf16
            AoS indexer's semantics; the page-split FP8 route follows upstream
            V3.2 and passes False on BOTH q and k.

    Returns:
        [batch, 128] bf16 tensor
    """
    return _fused_rope_hadamard_cuda.fused_rope_hadamard(
        x, cos_cache, sin_cache, positions, scale, apply_hadamard
    )


def fused_rope_hadamard_out(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    scale: float = 128 ** -0.5,
    out_fp8: Optional[torch.Tensor] = None,
    apply_hadamard: bool = True,
) -> torch.Tensor:
    """Out-buffer fused interleaved RoPE + optional Hadamard for graph capture.

    When ``out_fp8`` is given the kernel epilogue also writes the saturating
    e4m3 copy of the bf16 result, replacing the separate ``clamp`` and bf16 ->
    float8_e4m3fn ATen launches the indexer-Q scorer used to make.

    ``apply_hadamard=False`` selects the kernel's no-rotation instantiation
    (upstream V3.2 semantics, dropped on both q and k); ``scale`` is then
    unused because it normalizes H.
    """
    _fused_rope_hadamard_cuda.fused_rope_hadamard_out(
        x,
        cos_cache,
        sin_cache,
        positions,
        out,
        scale,
        out_fp8,
        apply_hadamard,
    )
    return out
