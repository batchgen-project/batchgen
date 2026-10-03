"""Compatibility exports for the packaged GLM-5 DSA Hadamard kernels.

The model code historically imported this module. Keep that import stable,
but resolve the implementation from ``batchgen_kernels`` so production never
compiles CUDA extensions during startup or the first prefill.
"""

from batchgen_kernels.attention.dsa.indexer import (
    fused_rope_hadamard,
    fused_rope_hadamard_out,
    hadamard_transform,
)


__all__ = ["hadamard_transform", "fused_rope_hadamard", "fused_rope_hadamard_out"]
