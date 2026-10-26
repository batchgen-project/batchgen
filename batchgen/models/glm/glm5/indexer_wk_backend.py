# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Which implementation computes the GLM-5 DSA indexer's wk projection.

Kernel version management: BOTH implementations stay in the tree and stay
selectable, and the older one is not edited.

``"fused_cuda"``
    The original pair, unchanged: the custom WGMMA GEMM in
    ``batchgen_kernels/attention/dsa/fused_indexer_kv_proj_cuda.py`` plus that
    kernel's OWN per-row activation quantize of the layer's hidden states.

``"deepgemm"``
    The new version: the private per-row quantize disappears and the
    projection runs as a dense DeepGEMM FP8 GEMM over the 128-block-scaled
    FP8 hidden the SAME layer's qkv_a projection already produces, which the
    segment hoists above the indexer fork so the side stream can read it.

The switch is a module-level constant on purpose:

  * The decode graphs are captured once per server lifetime, so the choice has
    to be fixed before capture -- a batch-level ``batchgen_debug`` flag cannot
    express it.
  * Repo policy forbids a ``BATCHGEN_*`` env guard for model behaviour.

The two settings are not interchangeable at runtime: a segment reads the
backend once in ``__init__`` (before warmup and capture) and keeps it for the
life of the capture.
"""

GLM5_INDEXER_WK_BACKENDS = ("deepgemm", "fused_cuda")

#: Default on this branch. ``"fused_cuda"`` restores the pre-existing path
#: exactly (same kernels, same order, same stream placement).
GLM5_INDEXER_WK_BACKEND = "deepgemm"


def glm5_indexer_wk_backend() -> str:
    """Return the selected backend, failing loud on an unknown value."""
    backend = GLM5_INDEXER_WK_BACKEND
    if backend not in GLM5_INDEXER_WK_BACKENDS:
        raise ValueError(
            f"GLM5_INDEXER_WK_BACKEND must be one of "
            f"{GLM5_INDEXER_WK_BACKENDS}, got {backend!r}"
        )
    return backend


def glm5_indexer_wk_is_deepgemm() -> bool:
    """True when the indexer wk projection runs on the DeepGEMM version."""
    return glm5_indexer_wk_backend() == "deepgemm"
