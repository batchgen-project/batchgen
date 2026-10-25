# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Whether a GLM-5 decode layer emits its FP8 activation FROM the norm kernel.

Three of the decode step's per-token blocked-FP8 casts read a tensor that a
norm kernel wrote one launch earlier and nothing else touches in between:

``qkv_a``
    ``input_layernorm`` (fused add + RMSNorm) -> the DSA segment's
    ``act_quant(hidden_flat)``, which also feeds the indexer's wk GEMM.
``q_a``
    ``q_a_layernorm`` (standalone Triton RMSNorm) -> ``act_quant(q_a_normed)``.
``shared_expert``
    ``post_attention_layernorm`` (fused add + RMSNorm) -> the shared expert's
    ``per_token_blocked_quantize_bf16_to_fp8_1d(x)``.

With the switch on, the norm kernel writes the FP8 bytes and the per-128-block
FP32 scales itself, in the same launch that already holds the normalized row in
registers, and the standalone quantize is not launched at all. The BF16 norm
output is still written, unchanged, because other consumers read it (the
router, DeepEP's own dispatch cast, the residual stream).

Byte identity, not tolerance
----------------------------
The fused epilogues reproduce ``_per_token_blocked_quantize_1d_kernel``
statement for statement (see the module docstrings of
``batchgen_kernels/src/attention/csrc/add_rmsnorm_fp8_quant.cu`` and
``batchgen_kernels/triton/rmsnorm_fp8_quant_fused.py``), and they quantize the
BF16-ROUNDED normalized value -- the same bits the standalone kernel would have
loaded back from global memory -- not the unrounded FP32 accumulator. So the
FP8 bytes and the scales are bit-for-bit what the two-launch form produced, on
valid rows and on padding rows alike. ``tests/test_glm52_norm_quant_fuse.py``
pins the math at the source level; the bitwise gate itself is a GPU test
(``tests/gpu/test_glm52_norm_quant_fuse_bitwise.py``).

Kernel version management: BOTH versions stay in the tree and stay selectable,
and the older one is not edited. ``add_rmsnorm_bf16_6144_kernel`` and
``_rmsnorm_kernel`` are untouched; the fused variants are new files.

Why a module-level constant
---------------------------
  * The decode graphs are captured once per server lifetime, so the choice has
    to be fixed before capture -- a batch-level ``batchgen_debug`` flag cannot
    express it.
  * Repo policy forbids a ``BATCHGEN_*`` env guard for model behaviour.

Dependency-free on purpose (no torch, no kernels): the captured graph segments
import it at top level and it must stay importable on CPU-only machines.
"""

#: Every call site the switch can govern, in decode order.
GLM5_NORM_QUANT_FUSE_SITES = ("qkv_a", "q_a", "shared_expert")

#: Master switch. Default on this branch. ``False`` restores the pre-existing
#: path exactly: the standalone quantizes are launched where they were, the
#: norms run their unfused kernels, and no FP8 output buffer is allocated.
GLM5_NORM_QUANT_FUSE = True

#: Per-site opt-out, for bisecting a regression on the node without rebuilding.
#: A site listed here keeps its standalone quantize even with the master switch
#: on. Must be a subset of ``GLM5_NORM_QUANT_FUSE_SITES``.
GLM5_NORM_QUANT_FUSE_DISABLED_SITES: tuple = ()


def glm5_norm_quant_fuse_enabled() -> bool:
    """Return the master switch, failing loud on anything that is not a bool."""
    value = GLM5_NORM_QUANT_FUSE
    if type(value) is not bool:
        raise TypeError(
            "GLM5_NORM_QUANT_FUSE must be True or False, got "
            f"{value!r} ({type(value).__name__})"
        )
    return value


def glm5_norm_quant_fuse_site_enabled(site: str) -> bool:
    """Whether ``site`` emits its FP8 activation from the norm kernel.

    Fails loud on an unknown site name rather than returning False: a typo in a
    call site's name would silently leave that quantize in the graph while the
    report claimed it was gone.
    """
    if site not in GLM5_NORM_QUANT_FUSE_SITES:
        raise ValueError(
            f"unknown norm/quant fusion site {site!r}; known sites are "
            f"{GLM5_NORM_QUANT_FUSE_SITES}"
        )
    disabled = GLM5_NORM_QUANT_FUSE_DISABLED_SITES
    if not isinstance(disabled, (tuple, list, frozenset, set)):
        raise TypeError(
            "GLM5_NORM_QUANT_FUSE_DISABLED_SITES must be a tuple of site "
            f"names, got {disabled!r} ({type(disabled).__name__})"
        )
    unknown = [s for s in disabled if s not in GLM5_NORM_QUANT_FUSE_SITES]
    if unknown:
        raise ValueError(
            f"GLM5_NORM_QUANT_FUSE_DISABLED_SITES names unknown sites "
            f"{unknown}; known sites are {GLM5_NORM_QUANT_FUSE_SITES}"
        )
    if not glm5_norm_quant_fuse_enabled():
        return False
    return site not in disabled
