"""UUID ↔ local_idx ↔ global_idx mapping helpers.

Slice 1 of the worker decouple initiative (issue #171). Ports the five
pure-index methods previously inlined on ``BatchGenWorker``
(``_local_to_uuid``, ``_uuid_to_local``, ``_local_indices_to_global_seq_ids``,
``_get_my_sequences_by_status``, ``_get_local_indices_for_uuids``) into a
single sibling module so the worker no longer carries them.

Design follows the Phase A/B/C cuda-graph adapter pattern: a frozen
``IndexLookupRequest`` snapshot carries everything a call consumes, and
``IndexManager`` is a namespace of stateless static methods. The worker
remains the canonical owner of the underlying maps; ``IndexLookupRequest``
is a typed view, not a state copy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Mapping

if TYPE_CHECKING:
    from batchgen.sequence import SequenceBatch, SequenceStatus


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexLookupRequest:
    """Frozen snapshot passed to each ``IndexManager`` call.

    The worker constructs one of these from its canonical fields
    (``self.rank``, ``self._local_to_uuid_map``, ``self._uuid_to_local_map``,
    ``self.global_batch``) per call site. Handler methods MUST NOT mutate
    any contents — Python cannot enforce that for ``Mapping`` values, so
    this is a contract assertion.
    """

    rank: int
    local_to_uuid: Mapping[int, str]
    uuid_to_local: Mapping[str, int]
    global_batch: "SequenceBatch"


class IndexManager:
    """Namespace of pure-function index converters."""

    @staticmethod
    def local_to_uuid(req: IndexLookupRequest, local_idx: int) -> str:
        return req.local_to_uuid.get(local_idx, "")

    @staticmethod
    def uuid_to_local(req: IndexLookupRequest, uuid: str) -> int:
        return req.uuid_to_local.get(uuid, -1)

    @staticmethod
    def local_indices_to_global_seq_ids(
        req: IndexLookupRequest, local_indices: List[int]
    ) -> List[int]:
        """Convert local indices to global sequence IDs (``global_idx`` from ``SequenceEntry``).

        Logs at ``ERROR`` when any local index is unmapped — length
        mismatch between input and output causes KV corruption (wrong
        sequence KV read for wrong batch position).
        """
        global_seq_ids: List[int] = []
        missing_indices: List[int] = []
        for local_idx in local_indices:
            uuid = req.local_to_uuid.get(local_idx)
            if uuid:
                seq = req.global_batch.get_sequence(uuid)
                global_seq_ids.append(seq.global_idx)
            else:
                missing_indices.append(local_idx)

        if missing_indices:
            logger.error(
                f"Rank {req.rank}: MISSING LOCAL INDICES in local_indices_to_global_seq_ids! "
                f"input_len={len(local_indices)}, output_len={len(global_seq_ids)}, "
                f"missing={missing_indices[:10]}..."
            )
        return global_seq_ids

    @staticmethod
    def get_my_sequences_by_status(
        req: IndexLookupRequest, status: "SequenceStatus"
    ) -> List[str]:
        """Get UUIDs of sequences assigned to this rank with given status."""
        return req.global_batch.get_sequences_for_rank_with_status(req.rank, status)

    @staticmethod
    def get_local_indices_for_uuids(
        req: IndexLookupRequest, uuids: List[str]
    ) -> List[int]:
        """Convert global UUIDs to local indices for sequences assigned to this rank.

        Non-owned UUIDs are silently skipped — callers typically pass the
        full cross-rank decode_uuids list and each rank resolves only its
        own slice.
        """
        local_indices: List[int] = []
        for uuid in uuids:
            local_idx = req.uuid_to_local.get(uuid)
            if local_idx is not None:
                local_indices.append(local_idx)
        return local_indices
