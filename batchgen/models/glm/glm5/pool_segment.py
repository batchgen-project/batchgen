# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Copy a loaded DSA prefix slab into one GLM-5 layer's prefill-pool rows.

The worker loads a committed prefix's two Host regions (MLA latent, indexer
keys) slab by slab (``prefix_reuse.pool_materialize.DualRegionSegmentLoader``);
this is the model side of that seam, kept free of CUDA imports so the copy
contract is testable on CPU.
"""

from __future__ import annotations


def copy_dsa_pool_segment(
    pool, *, layer: int, latent, indexer, slots, has_indexer: bool
) -> int:
    """Write one DSA layer's loaded slab into its pool slots, both regions.

    The pool of a DSA model holds what its forward writes -- the latent row
    (``latent_flat[slots] = primary_kv``) of every layer and the indexer row
    of each layer with an indexer -- and the Host holds the same bytes, so
    this is a copy. ``has_indexer`` is the layer's own fact (a GLM-5.2 shared
    layer reuses a full layer's top-k and keeps no indexer keys). Every check
    runs before the first write: a refused slab leaves the pool untouched.
    """

    tokens = int(latent.shape[0])
    if latent.dim() != 2:
        raise ValueError(
            f"layer {layer}: a loaded latent is [tokens, dim], got "
            f"{tuple(latent.shape)}"
        )
    if int(slots.numel()) != tokens:
        raise ValueError(
            f"layer {layer}: materializing {tokens} token(s) into "
            f"{int(slots.numel())} pool slot(s)"
        )
    if int(pool.k.shape[-2]) != 1 or pool.v is not None:
        raise ValueError(
            f"layer {layer}: a DSA pool holds one latent head and no V, got "
            f"k {tuple(pool.k.shape)}, v {'none' if pool.v is None else 'present'}"
        )
    latent_dim = int(pool.k.shape[-1])
    if int(latent.shape[-1]) != latent_dim or latent.dtype != pool.k.dtype:
        raise ValueError(
            f"layer {layer}: the pool latent is {latent_dim} wide "
            f"{pool.k.dtype}, the loaded latent {int(latent.shape[-1])} wide "
            f"{latent.dtype}"
        )
    row = pool.indexer_row.get(int(layer)) if pool.indexer is not None else None
    if bool(has_indexer) != (row is not None):
        raise RuntimeError(
            f"layer {layer} {'has' if has_indexer else 'has no'} indexer but "
            f"the pool keeps {'no' if row is None else 'an'} indexer row for it"
        )
    if row is None:
        if indexer is not None:
            raise RuntimeError(
                f"layer {layer} keeps no indexer keys, but the load brought some"
            )
    else:
        if indexer is None:
            raise RuntimeError(
                f"layer {layer} scores its top-k from pooled indexer keys; a "
                "load that brings only the latent would leave it scoring "
                "stale slots"
            )
        indexer_dim = int(pool.indexer.shape[-1])
        if (
            tuple(indexer.shape) != (tokens, indexer_dim)
            or indexer.dtype != pool.indexer.dtype
        ):
            raise ValueError(
                f"layer {layer}: the pool indexer row is [{tokens}, "
                f"{indexer_dim}] {pool.indexer.dtype}, the loaded keys "
                f"{tuple(indexer.shape)} {indexer.dtype}"
            )
    pool.k[int(layer)].view(-1, latent_dim).index_copy_(0, slots, latent)
    if row is not None:
        pool.indexer[row].view(-1, indexer_dim).index_copy_(0, slots, indexer)
    return tokens
