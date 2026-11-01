"""Compatibility re-export of the AOT Hadamard kernels.

The kernels, their sources and their extensions live in ``batchgen_kernels``
(``batchgen_kernels.attention.dsa.indexer``); this module keeps the historical
import path.

Importing it imports both AOT extensions (no compilation). Callers such as the
DSA model module treat a failed import here as "kernel unavailable" and report
it from their startup gate, so the failure has to surface at import time, as
it did when this module compiled the kernels itself.
"""

from batchgen_kernels.attention.dsa.indexer import (
    fused_rope_hadamard,
    hadamard_transform,
    load_hadamard_extensions,
)

load_hadamard_extensions()

__all__ = ["fused_rope_hadamard", "hadamard_transform"]
