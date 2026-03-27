"""
SyncCoordinator: All cross-rank collective operations in one place.

Extracted from batchgen_worker.py lines 3006-3188.
Handles: metadata sync, completion status sync, decode UUID sync.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from batchgen.worker.state import WorkerState
from batchgen.sequence import SequenceStatus


class SyncCoordinator:
    """Centralizes all cross-rank collective operations.

    All collective ops (all_gather, all_reduce, broadcast) go through this class.
    This makes it easy to audit synchronization points and prevent desync bugs.
    """

    def __init__(self, state: WorkerState):
        self.state = state

    def sync_metadata(self, decode_uuids: List[str]) -> None:
        """Synchronize sequence metadata across all ranks.

        Each rank reports its local sequences' state, and all ranks update
        their local SequenceEntry objects with the gathered info.

        Syncs: decoded_length, current_context_length, gpu_pages_allocated,
               eos_reached, host_pages_allocated, host_token_capacity.

        CRITICAL: Validates ctx_len = prompt_len + decoded_len on both
        send and receive sides, correcting if inconsistent.
        """
        if not decode_uuids:
            return

        s = self.state

        # Step 1: Each rank reports state for sequences it owns
        local_state = {}
        for uuid in decode_uuids:
            if uuid in s.uuid_to_local_map:
                seq = s.global_batch.get_sequence(uuid)
                # INVARIANT: current_context_length = original_prompt_length + decoded_length
                expected_ctx = seq.original_prompt_length + seq.decoded_length
                if seq.current_context_length != expected_ctx:
                    logging.warning(
                        f"Rank {s.rank}: Correcting ctx_len for {uuid[:8]} before sync: "
                        f"{seq.current_context_length} -> {expected_ctx}"
                    )
                    seq.current_context_length = expected_ctx

                local_state[uuid] = {
                    'decoded_length': seq.decoded_length,
                    'current_context_length': seq.current_context_length,
                    'gpu_pages_allocated': seq.gpu_pages_allocated,
                    'eos_reached': seq.eos_reached,
                    'prompt_length': seq.prompt_length,
                    'host_pages_allocated': seq.host_pages_allocated,
                    'host_token_capacity': seq.host_token_capacity,
                }

        # Step 2: All-gather state from all ranks
        all_states = [None] * s.world_size
        dist.all_gather_object(all_states, local_state)

        # Step 3: Merge and update local SequenceEntry objects
        for rank_state in all_states:
            if rank_state:
                for uuid, state in rank_state.items():
                    if uuid not in s.uuid_to_local_map:
                        # This sequence belongs to another rank — update our local copy
                        seq = s.global_batch.get_sequence(uuid)
                        if seq is not None:
                            seq.decoded_length = state['decoded_length']
                            seq.current_context_length = state['current_context_length']
                            seq.gpu_pages_allocated = state['gpu_pages_allocated']
                            seq.eos_reached = state['eos_reached']
                            if 'host_pages_allocated' in state:
                                seq.host_pages_allocated = state['host_pages_allocated']
                            if 'host_token_capacity' in state:
                                seq.host_token_capacity = state['host_token_capacity']

                            # VALIDATION: Ensure received ctx_len is consistent
                            expected_ctx = seq.original_prompt_length + seq.decoded_length
                            if seq.current_context_length != expected_ctx:
                                logging.error(
                                    f"Rank {s.rank}: [SYNC-VALIDATE] Received inconsistent ctx_len for {uuid[:8]}: "
                                    f"received={seq.current_context_length}, expected={expected_ctx} "
                                    f"(prompt={seq.prompt_length}, decoded={seq.decoded_length})"
                                )
                                seq.current_context_length = expected_ctx

    def sync_completion_status(
        self,
        decode_uuids: List[str],
    ) -> Tuple[Set[str], List[str]]:
        """Synchronize completion status across all ranks using tensor all_reduce.

        OPTIMIZATION: Uses tensor all_reduce (~0.1ms) instead of all_gather_object (~1-5ms).

        Returns:
            (global_completed_uuids, active_decode_uuids) — both sorted by global_idx
        """
        if not decode_uuids:
            return set(), []

        s = self.state

        # Build global_idx ↔ uuid mapping
        idx_to_uuid = {}
        uuid_to_idx = {}
        for uuid in decode_uuids:
            seq = s.global_batch.get_sequence(uuid)
            if seq is not None:
                idx_to_uuid[seq.global_idx] = uuid
                uuid_to_idx[uuid] = seq.global_idx

        if not idx_to_uuid:
            return set(), []

        max_idx = max(idx_to_uuid.keys())

        # Create completion tensor: 1 = completed, 0 = not
        completion_tensor = torch.zeros(max_idx + 1, dtype=torch.int32, device=s.torch_device)
        for uuid in decode_uuids:
            if uuid in s.uuid_to_local_map:
                seq = s.global_batch.get_sequence(uuid)
                if seq is not None and uuid in uuid_to_idx:
                    is_completed = (seq.status == SequenceStatus.COMPLETED or seq.eos_reached)
                    if is_completed:
                        completion_tensor[uuid_to_idx[uuid]] = 1

        # all_reduce with MAX: if ANY rank marks complete, result is 1
        dist.all_reduce(completion_tensor, op=dist.ReduceOp.MAX)

        # Decode back to UUIDs
        global_completed = set()
        active_uuids = []
        for global_idx in sorted(idx_to_uuid.keys()):
            uuid = idx_to_uuid[global_idx]
            if completion_tensor[global_idx].item() == 1:
                global_completed.add(uuid)
                seq = s.global_batch.get_sequence(uuid)
                if seq is not None:
                    seq.eos_reached = True
                    if seq.status != SequenceStatus.COMPLETED:
                        try:
                            s.global_batch.update_status(uuid, SequenceStatus.COMPLETED)
                        except ValueError as e:
                            logging.debug(
                                f"Rank {s.rank}: Could not update {uuid[:8]} to COMPLETED: {e}"
                            )
            else:
                active_uuids.append(uuid)

        return global_completed, active_uuids

    def sync_decode_uuids(self, decode_uuids: List[str]) -> List[str]:
        """Synchronize decode_uuids across all ranks using tensor all_reduce.

        Uses global_idx as common identifier. Returns sorted list of UUIDs
        that ALL ranks agree on.
        """
        if not decode_uuids:
            return []

        s = self.state

        # Build global_idx ↔ uuid mapping
        idx_to_uuid = {}
        uuid_to_idx = {}
        for seq in s.global_batch:
            idx_to_uuid[seq.global_idx] = seq.uuid
            uuid_to_idx[seq.uuid] = seq.global_idx

        max_idx = max(idx_to_uuid.keys()) if idx_to_uuid else 0

        # Create presence tensor: 1 = in decode_uuids
        presence_tensor = torch.zeros(max_idx + 1, dtype=torch.int32, device=s.torch_device)
        for uuid in decode_uuids:
            if uuid in uuid_to_idx:
                presence_tensor[uuid_to_idx[uuid]] = 1

        # all_reduce with MIN: only sequences present on ALL ranks remain
        dist.all_reduce(presence_tensor, op=dist.ReduceOp.MIN)

        # Extract agreed UUIDs
        synced_uuids = []
        for global_idx in sorted(idx_to_uuid.keys()):
            if presence_tensor[global_idx].item() == 1:
                synced_uuids.append(idx_to_uuid[global_idx])

        return synced_uuids
