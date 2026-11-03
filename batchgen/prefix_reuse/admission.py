"""In-wave shared-prefix structure for admission accounting.

Admission decides how many candidates fit a pool.  When the wave's prompts
share prefixes, the pool holds ONE physical copy of each shared run -- the
reservation plan allocates one Host copy per pooled segment and every member
attaches that copy's pages -- so an admission walk that charges each row its
whole prompt bounds the wave by the logical page sum and admits a fraction of
what fits.

This module turns a wave's prompts into the per-row *segment chains* the
charge ledger consumes.  It reuses the wave prefix tree the pool planner and
the rank packer already build (``wave_plan._build_tree``), so admission's view
of what is shared is the SAME view that later decides what is pooled, rather
than a second estimate that can disagree with it.

Determinism: the tree is a function of the ordered prompt list alone, and the
prompts come from the node-shared trajectory store every rank reads, so every
rank derives identical chains with no communication.  That is required --
admission is a collective walk and a rank that credited a different segment
would admit a different wave.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from batchgen.prefix_reuse.wave_plan import _build_tree

# A run of pages is worth charging once only if at least two sequences read
# it; a single-reader run is that sequence's private prefix and the pool never
# makes a shared copy of it.
MIN_SHARERS = 2

SegmentChain = Tuple[Tuple[int, int], ...]


def plan_admission_shared_segments(
    prompts: Sequence[Sequence[int]],
    page_tokens: int,
    *,
    min_sharers: int = MIN_SHARERS,
) -> Tuple[SegmentChain, ...]:
    """Per prompt, the ``(segment_key, pages)`` chain it shares in this wave.

    ``prompts`` are the candidates' prompt token ids in admission order;
    ``page_tokens`` is the Host page size in tokens, so a segment's pages are
    whole pages and the arithmetic matches the allocator's.

    Keys are wave-local: they identify a shared run WITHIN this admission
    round, which is the scope the in-wave charge needs, and they are equal
    across ranks for the same run.  A prompt's last page never joins a chain
    (``_build_tree`` leaves it unshareable), so every row keeps a private tail
    and no row is ever charged zero.
    """
    if int(page_tokens) <= 0:
        raise ValueError(f"page_tokens must be > 0, got {page_tokens}")
    if int(min_sharers) < 2:
        raise ValueError(f"min_sharers must be >= 2, got {min_sharers}")
    if not prompts:
        return ()

    tree = _build_tree(prompts, int(page_tokens))
    floor = int(min_sharers)
    pages_by_segment = {
        segment: int(tree.end_block[segment]) - int(tree.start_block[segment])
        for segment in range(len(tree.count))
        if int(tree.count[segment]) >= floor
    }
    chains = []
    for path in tree.paths:
        chains.append(
            tuple(
                (int(segment), pages_by_segment[int(segment)])
                for segment in path
                if int(segment) in pages_by_segment
                and pages_by_segment[int(segment)] > 0
            )
        )
    return tuple(chains)


def shared_pages_per_row(chains: Sequence[SegmentChain]) -> Tuple[int, ...]:
    """Pages each row would save if every segment of its chain is already paid.

    Reported for logging and for the admission arithmetic tests; the authority
    on what a row costs remains the ledger, which knows which segments this
    resource has actually charged.
    """
    return tuple(sum(pages for _, pages in chain) for chain in chains)
