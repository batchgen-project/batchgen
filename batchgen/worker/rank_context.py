"""Frozen rank identity of one ``BatchGenWorker`` process.

``RankContext`` is built once, after the parallel strategy manager exists
(that is when the decode attention TP group size G becomes known), and never
changes afterwards. It is the single owner of three terms:

* "node of a rank": ``node_of_rank`` / ``RankContext.node_of``. Every
  host-KV, trajectory-book and memfd computation that needs a rank's physical
  node goes through it.
* the physical node COUNT: ``RankContext.num_nodes``.
* decode-group ownership of a sequence: ``owns_sequence`` (which ranks bind
  the sequence into their local maps) and ``owns_host_kv`` (the single rank
  that drives the per-node shared host-KV region). The decode-group geometry
  itself (``rank_in_decode_group``, ``host_kv_owner_rank``) is owned by
  ``batchgen.decode_dp_group`` and is called, not re-implemented, here.

Imports: stdlib plus ``batchgen.decode_dp_group`` (a torch-free term-owner
module). Nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from batchgen.decode_dp_group import host_kv_owner_rank, rank_in_decode_group


def node_of_rank(rank: int, local_world_size: int) -> int:
    """Physical node id of a global rank: ``rank // local_world_size``.

    Module-level so call sites that run before a ``RankContext`` exists (the
    trajectory-book and memfd setup inside core initialization) use the same
    single definition that ``RankContext.node_of`` uses.
    """
    return rank // local_world_size


@dataclass(frozen=True)
class RankContext:
    """Immutable identity of this rank.

    ``decode_group_size`` is G, the decode attention TP group size
    (``BatchGenWorker._decode_attn_tp_size()`` captured at construction).
    G == 1 is pure DP: a sequence's single ``assigned_rank`` owns it. G > 1 is
    resident TP-G: the G ranks of the sequence's ``decode_dp_group`` hold it.
    """

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    decode_group_size: int

    def node_of(self, rank: int) -> int:
        """Physical node id of ``rank`` (any global rank, not only this one)."""
        return node_of_rank(rank, self.local_world_size)

    @property
    def node_id(self) -> int:
        """Physical node id of this rank."""
        return node_of_rank(self.rank, self.local_world_size)

    @property
    def num_nodes(self) -> int:
        """Number of physical host-KV nodes: ``ceil(world / local_world)``,
        at least 1.

        An uneven world (a partial last node) still counts that node.
        """
        return max(1, math.ceil(self.world_size / self.local_world_size))

    def owns_sequence(self, seq) -> bool:
        """Whether this rank binds ``seq`` into its LOCAL maps and drives it
        through prefill/decode.

        Pure DP (G == 1, the validated path): the single ``assigned_rank`` owns
        it. Option 1 unified resident TP (G > 1): ALL G ranks of the
        sequence's ``decode_dp_group`` hold it. The group runs prefill and
        decode in TP-G lockstep on identical sequences (replicated attention,
        head-sharded KDA state), so the o_proj all_reduce couples matching
        tokens and no prefill->decode reshard is ever needed. Requires
        ``decode_dp_group`` to be stamped (done at admission / prefill config
        before binding).
        """
        G = self.decode_group_size
        if G <= 1:
            return seq.assigned_rank == self.rank
        return rank_in_decode_group(seq.decode_dp_group, self.rank, G)

    def owns_host_kv(self, seq) -> bool:
        """Whether this is the SINGLE rank that drives the per-node SHARED
        host-KV region for ``seq``.

        The host paged KV cache is ONE run-namespaced shm region per node,
        keyed by ``global_idx``, so register / allocate / grow / release must
        fire EXACTLY once per sequence. This is NARROWER than
        ``owns_sequence``: G > 1 replicates a sequence onto all G ranks of its
        group (they each hold GPU KV and head-sharded KDA state), but only the
        group LEADER ``decode_dp_group * G`` may touch the shared host region.
        Otherwise the G ranks double-register (G x host reservation) and
        double-release (the second releaser hits ``IndexError: Sequence ID ...
        not found during release``). G == 1 is the validated single-owner
        path: the ``assigned_rank`` owner, identical to "uuid is bound in the
        local sequence table".
        """
        G = self.decode_group_size
        if G <= 1:
            return seq.assigned_rank == self.rank
        return (
            seq.decode_dp_group is not None
            and host_kv_owner_rank(seq.decode_dp_group, G) == self.rank
        )
