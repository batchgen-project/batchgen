"""Compatibility shim for GLM-5 Hadamard kernels.

The canonical implementation lives in ``batchgen_kernels.attention.dsa.indexer``.
Keeping a second ``torch.utils.cpp_extension.load`` owner here caused duplicate
JIT extension names and multi-rank cold-start races in shared Torch extension
caches.
"""

from batchgen_kernels.attention.dsa.indexer import (
    fused_rope_hadamard,
    hadamard_transform,
)

__all__ = ["hadamard_transform", "fused_rope_hadamard"]
