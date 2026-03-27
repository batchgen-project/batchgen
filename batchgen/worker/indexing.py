"""
IndexManager: Local↔UUID↔global index mappings.

Pure utility methods — no state machine interaction.
Extracted from batchgen_worker.py lines 2942-2983.
"""
import logging
from typing import Dict, List, Set

from batchgen.worker.state import WorkerState
from batchgen.sequence import SequenceStatus


class IndexManager:
    """Manages bidirectional mappings between local indices, UUIDs, and global sequence IDs.

    All state is stored in WorkerState (shared). IndexManager provides
    the interface methods that other sub-managers use for lookups.
    """

    def __init__(self, state: WorkerState):
        self.state = state

    def local_to_uuid(self, local_idx: int) -> str:
        """Convert local index to UUID."""
        return self.state.local_to_uuid_map.get(local_idx, "")

    def uuid_to_local(self, uuid: str) -> int:
        """Convert UUID to local index. Returns -1 if not found."""
        return self.state.uuid_to_local_map.get(uuid, -1)

    def local_indices_to_global_seq_ids(self, local_indices: List[int]) -> List[int]:
        """Convert local indices to global sequence IDs (global_idx from SequenceEntry).

        CRITICAL: Logs error if any local indices are missing — this causes length
        mismatch which leads to KV corruption (wrong sequence KV for wrong batch position).
        """
        global_seq_ids = []
        missing_indices = []
        for local_idx in local_indices:
            uuid = self.state.local_to_uuid_map.get(local_idx)
            if uuid:
                seq = self.state.global_batch.get_sequence(uuid)
                global_seq_ids.append(seq.global_idx)
            else:
                missing_indices.append(local_idx)

        if missing_indices:
            logging.error(
                f"Rank {self.state.rank}: MISSING LOCAL INDICES in local_indices_to_global_seq_ids! "
                f"input_len={len(local_indices)}, output_len={len(global_seq_ids)}, "
                f"missing={missing_indices[:10]}..."
            )
        return global_seq_ids

    def get_local_indices_for_uuids(self, uuids: List[str]) -> List[int]:
        """Convert global UUIDs to local indices for sequences assigned to this rank."""
        local_indices = []
        for uuid in uuids:
            if uuid in self.state.uuid_to_local_map:
                local_indices.append(self.state.uuid_to_local_map[uuid])
        return local_indices

    def get_my_sequences_by_status(self, status: SequenceStatus) -> List[str]:
        """Get UUIDs of sequences assigned to this rank with given status."""
        return self.state.global_batch.get_sequences_for_rank_with_status(
            self.state.rank, status
        )

    def register_sequence(self, uuid: str) -> int:
        """Assign a local index to a new sequence. Returns the local index."""
        if self.state.free_local_indices:
            local_idx = self.state.free_local_indices.pop()
        else:
            local_idx = self.state.next_local_idx
            self.state.next_local_idx += 1

        self.state.local_to_uuid_map[local_idx] = uuid
        self.state.uuid_to_local_map[uuid] = local_idx
        return local_idx

    def unregister_sequence(self, uuid: str) -> None:
        """Free a local index when sequence completes."""
        local_idx = self.state.uuid_to_local_map.pop(uuid, None)
        if local_idx is not None:
            self.state.local_to_uuid_map.pop(local_idx, None)
            self.state.free_local_indices.add(local_idx)
