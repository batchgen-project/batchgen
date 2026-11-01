"""DSA indexer kernels: Hadamard transform, fused RoPE + Hadamard, K writers.

Every native kernel here is an AOT CUDA extension declared in ``setup.py`` and
mirrored in ``_jit_registry.py``; the Hadamard sources live in
``batchgen_kernels/src/attention/hadamard/``. Each extension is imported on
first use, so importing this package neither builds nor loads native code.
"""

from typing import Optional

import torch


class _AotExtension:
    """Imports one AOT extension on first attribute access, then forwards.

    Call sites keep the ``_module.function(...)`` shape while the import (and
    its failure, on a host without the built extension) moves to first use.
    """

    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            # Resolved at call time so tests can substitute the loader.
            from batchgen_kernels import load_extension

            self._module = load_extension(self._module_name, allow_dev_jit=False)
        return self._module

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._load(), name)


_hadamard_cuda = _AotExtension(
    "batchgen_kernels.attention._C_fast_hadamard_transform"
)
_fused_rope_hadamard_cuda = _AotExtension(
    "batchgen_kernels.attention._C_fused_rope_hadamard"
)


def load_hadamard_extensions() -> None:
    """Import both Hadamard extensions now instead of at first call.

    Raises ``ImportError`` when either is not built, for callers whose
    availability check runs at import time.
    """

    _hadamard_cuda._load()
    _fused_rope_hadamard_cuda._load()


# Fused indexer-K quantize + page-split write. Like the two kernels above it
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


def fused_rope_hadamard_strided_out(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    x_row_stride: int,
    scale: float = 128 ** -0.5,
    out_fp8: Optional[torch.Tensor] = None,
    apply_hadamard: bool = True,
) -> torch.Tensor:
    """``fused_rope_hadamard_out`` over a STRIDED ``[rows, heads, 128]`` source.

    ``x`` keeps its heads packed (stride 128) but its rows ``x_row_stride``
    elements apart -- the trailing column slice of the merged ``[q_b | wq_b]``
    GEMM output -- and the kernel reads it in place, so no reshape copy is
    made. ``out`` (and ``out_fp8``) are contiguous ``[rows, heads, 128]``.
    A separate entry point (kernel version management): the contiguous one
    above is unchanged.
    """
    _fused_rope_hadamard_cuda.fused_rope_hadamard_strided_out(
        x,
        cos_cache,
        sin_cache,
        positions,
        out,
        x_row_stride,
        scale,
        out_fp8,
        apply_hadamard,
    )
    return out
