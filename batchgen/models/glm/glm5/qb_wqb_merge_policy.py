# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Whether a GLM-5 DSA indexer layer runs ``q_b_proj`` and ``wq_b`` as ONE GEMM.

Both projections consume the same ``q_a_normed`` activation. With the switch
on, a full indexer layer concatenates the two FP8 weights along N at load
time -- ``[q_b (num_heads * q_head_dim) | wq_b (index_n_heads *
index_head_dim)]`` -- and runs a single 128-block-scaled DeepGEMM over the
layer's one blocked activation quantize, instead of the q_b DeepGEMM plus the
indexer's custom WGMMA GEMM and that kernel's private per-row quantize.

Kernel version management: BOTH versions stay in the tree and stay
selectable, and the older one is not edited.

``False``
    The pre-existing path, byte for byte: the q_b DeepGEMM into its own
    buffer, then ``cuda_wq_b_proj_out`` (private per-row quantize + custom
    WGMMA over a re-quantized copy of wq_b) into its own buffer.
``True``
    The merged version. The wq_b rows are the checkpoint's own e4m3 bytes
    with the checkpoint's 128x128 block scales -- there is no dequantize /
    re-quantize round trip -- so the wq_b output is a deliberate numerics
    change versus ``False`` (tolerance-compared), while the q_b output is
    unchanged.

Layers without an indexer (GLM-5.2's ``skip_topk`` layers) have no wq_b and
are never merged; their segment keeps the plain q_b GEMM either way.

The switch is a module-level constant on purpose:

  * The decode graphs are captured once per server lifetime, so the choice has
    to be fixed before capture -- a batch-level ``batchgen_debug`` flag cannot
    express it.
  * Repo policy forbids a ``BATCHGEN_*`` env guard for model behaviour.

Dependency-free on purpose (no torch, no kernels): the captured graph segment
imports it at top level and must stay importable on CPU-only machines.
"""

#: Default on this branch. ``False`` restores the pre-existing path exactly
#: (same kernels, same order, same buffers, same stream placement).
GLM5_QB_WQB_MERGE = True


def glm5_qb_wqb_merge_enabled() -> bool:
    """Return the switch, failing loud on anything that is not a real bool."""
    value = GLM5_QB_WQB_MERGE
    if type(value) is not bool:
        raise TypeError(
            "GLM5_QB_WQB_MERGE must be True or False, got "
            f"{value!r} ({type(value).__name__})"
        )
    return value
