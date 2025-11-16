import torch
import torch.distributed as dist
import logging
import time
from typing import List, Dict, Optional, Tuple, Any, Literal

# Assume these classes are correctly imported
from batchgen.sequence_manager.batch_defs import (
    GlobalSequenceRegistry, SequenceStatus, SequenceEntry
)
# from batchgen.host_kv_client import HostKVCacheClient
# from batchgen.gpu_kv_manager import GPUKVCacheManager, GPUPagedKVStats

# Define Decision Types
OrchestratorDecision = Literal["PREFILL", "DECODE", "WAIT"]
WorkAssignment = Dict[int, List[str]] # rank -> list of sequence UUIDs

class PDOrchestrator:
    """
    (L3 Abstraction - Rank 0 Only)
    Orchestrates the overall inference flow, deciding between Prefill and Decode phases
    based on global system state (Host KV Cache usage, sequence statuses).
    It gathers state, runs the scheduling algorithm, and broadcasts decisions/assignments.
    """
    def __init__(self,
                 global_registry: GlobalSequenceRegistry,
                 host_client: Any, # Rank 0's HostKVCacheClient
                 gpu_manager: Any, # Rank 0's GPUKVCacheManager
                 world_size: int,
                 host_kv_watermark_pct: float = 0.30, # Switch to Prefill if host free < 30%
                 max_prefill_batch_size_per_rank: int = 32,
                 max_decode_batch_size_per_rank: int = 64): # Max concurrent seqs

        self.logger = logging.getLogger(f"{self.__class__.__name__}_Rank0")
        if dist.get_rank() != 0:
            raise RuntimeError("PDOrchestrator should only be instantiated on Rank 0.")

        self.global_registry = global_registry
        self.host_client = host_client
        self.gpu_manager = gpu_manager # Needed for local GPU stats (Rank 0)
        self.world_size = world_size
        self.host_kv_watermark_pct = host_kv_watermark_pct
        self.max_prefill_batch_size_per_rank = max_prefill_batch_size_per_rank
        self.max_decode_batch_size_per_rank = max_decode_batch_size_per_rank

        # Internal state tracking the cluster-wide phase
        self.current_phase: OrchestratorDecision = "WAIT" # Start in a waiting state
        # Track sequences currently assigned for processing (across all ranks)
        self.sequences_in_prefill: Set[str] = set()
        self.sequences_in_decode: Set[str] = set()

        self.logger.info("PDOrchestrator initialized on Rank 0.")

    def get_decision_and_new_sequences(self) -> Tuple[OrchestratorDecision, List[str]]:
        """
        The main entry point called by all ranks during their sync point.
        Only Rank 0 performs logic; others wait for broadcast.

        Returns:
            Tuple[OrchestratorDecision, List[str]]:
                - The global decision ("PREFILL", "DECODE", or "WAIT").
                - A list of *new* sequence UUIDs assigned specifically *to the calling rank*.
        """
        decision: OrchestratorDecision = "WAIT"
        work_assignments: WorkAssignment = {rank: [] for rank in range(self.world_size)}

        if dist.get_rank() == 0:
            # --- Rank 0: Gather State & Make Decision ---
            try:
                # 1. Gather Global State
                global_host_stats, global_gpu_stats = self._gather_global_stats()
                sequences_by_status = self._get_sequences_by_status()

                # Calculate global host free percentage
                total_host_pages = sum(s.num_total_pages for s in global_host_stats)
                total_host_free = sum(s.num_free_pages for s in global_host_stats)
                global_host_free_pct = (total_host_free / total_host_pages) if total_host_pages > 0 else 0.0
                self.logger.debug(f"Global Host Free: {global_host_free_pct:.2%} ({total_host_free}/{total_host_pages})")

                # --- Implement Algorithm 1 Logic ---

                # Check if current tasks are finishing
                self._update_internal_state(sequences_by_status) # Update based on completed sequences

                # Decide next phase based on current phase and global state
                if self.current_phase == "WAIT" or \
                   (self.current_phase == "PREFILL" and not self.sequences_in_prefill) or \
                   (self.current_phase == "DECODE" and not self.sequences_in_decode):
                    # Transition out of idle or completed phase
                    if sequences_by_status[SequenceStatus.WAITING_IN_QUEUE] and \
                       global_host_free_pct >= self.host_kv_watermark_pct:
                        decision = "PREFILL"
                        self.logger.info("Decision: Start PREFILL phase.")
                    elif sequences_by_status[SequenceStatus.DECODE_READY]:
                        decision = "DECODE"
                        self.logger.info("Decision: Start DECODE phase.")
                    else:
                        decision = "WAIT" # No pending prefill, nothing ready for decode
                        self.logger.info("Decision: WAIT (no pending or ready sequences).")

                elif self.current_phase == "PREFILL":
                     # Continue prefill if host has space and pending sequences exist
                     if sequences_by_status[SequenceStatus.WAITING_IN_QUEUE] and \
                        global_host_free_pct >= self.host_kv_watermark_pct:
                         decision = "PREFILL"
                         self.logger.debug("Decision: Continue PREFILL phase.")
                     else:
                         # Switch to decode if prefill is blocked/done and decode is possible
                         if sequences_by_status[SequenceStatus.DECODE_READY]:
                              decision = "DECODE"
                              self.logger.info("Decision: Switch from PREFILL to DECODE.")
                         else: # No pending, host full, nothing ready to decode
                              decision = "WAIT"
                              self.logger.info("Decision: WAIT (Prefill blocked/done, nothing ready for decode).")

                elif self.current_phase == "DECODE":
                     # Check watermark condition to switch back to prefill
                     if sequences_by_status[SequenceStatus.WAITING_IN_QUEUE] and \
                        global_host_free_pct < self.host_kv_watermark_pct:
                          decision = "PREFILL"
                          self.logger.info(f"Decision: Switch from DECODE to PREFILL (Host free {global_host_free_pct:.2%} < {self.host_kv_watermark_pct:.2%}).")
                     else:
                          # Continue decoding
                          decision = "DECODE"
                          self.logger.debug("Decision: Continue DECODE phase.")

                # --- Assign Work Based on Decision ---
                if decision == "PREFILL":
                    assigned_uuids = self._select_prefill_batch(sequences_by_status[SequenceStatus.WAITING_IN_QUEUE], global_host_stats)
                    work_assignments = self._distribute_work(assigned_uuids)
                    self.sequences_in_prefill.update(assigned_uuids) # Track newly assigned
                    self.current_phase = "PREFILL"

                elif decision == "DECODE":
                    # Assign initial decode batch OR backfill
                    newly_assigned_decode = []
                    if not self.sequences_in_decode: # Starting decode phase
                        initial_decode_uuids = self._select_initial_decode_batch(sequences_by_status[SequenceStatus.DECODE_READY], global_gpu_stats)
                        work_assignments = self._distribute_work(initial_decode_uuids)
                        newly_assigned_decode = initial_decode_uuids
                    else: # Backfilling during decode
                        backfill_uuids = self._select_decode_backfill(sequences_by_status[SequenceStatus.DECODE_READY], global_gpu_stats)
                        work_assignments = self._distribute_work(backfill_uuids) # Distribute backfill work
                        newly_assigned_decode = backfill_uuids

                    self.sequences_in_decode.update(newly_assigned_decode) # Track newly assigned
                    self.current_phase = "DECODE"

                else: # Decision is WAIT
                     self.current_phase = "WAIT"
                     # work_assignments remains empty

                # Prepare the object to broadcast
                broadcast_data = [(decision, work_assignments)]

            except Exception as e:
                self.logger.error(f"Error during Rank 0 scheduling: {e}", exc_info=True)
                # Broadcast a WAIT signal on error? Or raise? Broadcasting WAIT is safer.
                broadcast_data = [("WAIT", {rank: [] for rank in range(self.world_size)})]

            # 4. Broadcast Decision and Assignments
            self.logger.debug(f"Broadcasting decision '{broadcast_data[0][0]}' and assignments...")
            dist.broadcast_object_list(broadcast_data, src=0)
            self.logger.debug("Broadcast complete.")

        else:
            # --- Ranks 1...N: Wait for Broadcast ---
            self.logger.debug(f"Rank {dist.get_rank()} waiting for broadcast from Rank 0...")
            broadcast_data = [None] # Placeholder to receive the object
            dist.broadcast_object_list(broadcast_data, src=0)
            decision, work_assignments = broadcast_data[0]
            self.logger.debug(f"Rank {dist.get_rank()} received decision '{decision}'.")

        # All ranks return the global decision and *their specific* assigned UUIDs
        my_assigned_uuids = work_assignments.get(dist.get_rank(), [])
        return decision, my_assigned_uuids

    # --- Helper Methods (Executed only on Rank 0) ---

    def _gather_global_stats(self) -> Tuple[List[Any], List[GPUPagedKVStats]]:
        """ Gathers Host and GPU stats from all ranks. """
        self.logger.debug("Gathering global stats...")
        local_host_stats = self.host_client.get_stats().get() # Blocking get
        local_gpu_stats = self.gpu_manager.get_stats()

        all_host_stats = [None] * self.world_size
        all_gpu_stats = [None] * self.world_size

        # Use all_gather_object for simplicity (can be slow for very large world sizes)
        # Consider all_gather with tensors if performance critical
        dist.all_gather_object(all_host_stats, local_host_stats)
        dist.all_gather_object(all_gpu_stats, local_gpu_stats)
        self.logger.debug("Global stats gathered.")
        return all_host_stats, all_gpu_stats

    def _get_sequences_by_status(self) -> Dict[SequenceStatus, List[str]]:
        """ Groups sequences in the global registry by their current status. """
        sequences_by_status = {status: [] for status in SequenceStatus}
        for uuid, seq_entry in self.global_registry.sequences.items():
             # Only consider sequences owned by *some* rank (filter out completed/failed?)
             # if seq_entry.status not in {SequenceStatus.COMPLETED, SequenceStatus.FAILED}:
             sequences_by_status[seq_entry.status].append(uuid)
        return sequences_by_status

    def _update_internal_state(self, sequences_by_status: Dict[SequenceStatus, List[str]]):
         """ Updates internal tracking sets based on global status. """
         # Remove completed/failed sequences from tracking sets
         completed_or_failed = set(sequences_by_status[SequenceStatus.COMPLETED]) | set(sequences_by_status[SequenceStatus.FAILED])
         self.sequences_in_prefill -= completed_or_failed
         self.sequences_in_decode -= completed_or_failed

         # Sanity check: ensure sequences marked DECODE_READY are not still tracked as IN_PREFILL
         ready_for_decode = set(sequences_by_status[SequenceStatus.DECODE_READY])
         self.sequences_in_prefill -= ready_for_decode


    def _select_prefill_batch(self, pending_uuids: List[str], global_host_stats: List[Any]) -> List[str]:
        """ Selects sequences for the next prefill batch globally. """
        # Simple strategy: Fill up to max batch size per rank globally
        num_can_prefill = self.world_size * self.max_prefill_batch_size_per_rank
        num_to_select = min(len(pending_uuids), num_can_prefill - len(self.sequences_in_prefill))
        
        # Prioritize? For now, just take the first N
        selected = pending_uuids[:num_to_select]
        self.logger.info(f"Selected {len(selected)} sequences for prefill.")
        return selected

    def _select_initial_decode_batch(self, ready_uuids: List[str], global_gpu_stats: List[GPUPagedKVStats]) -> List[str]:
        """ Selects the first batch of sequences to start decoding. """
        # Simple strategy: Fill up to max decode batch size per rank globally, respecting GPU limits (approx)
        total_gpu_free_pages = sum(s.num_free_pages for s in global_gpu_stats)
        
        selected = []
        pages_needed_approx = 0
        max_total_seqs = self.world_size * self.max_decode_batch_size_per_rank

        for uuid in ready_uuids:
             if len(selected) >= max_total_seqs: break
             seq = self.global_registry.get_sequence(uuid)
             # Estimate pages needed (prompt pages + buffer) - could be more precise
             est_pages = math.ceil(seq.prompt_length / self.gpu_manager.page_size) + self.PAGE_BUFFER_SIZE
             if pages_needed_approx + est_pages <= total_gpu_free_pages:
                  selected.append(uuid)
                  pages_needed_approx += est_pages
             else:
                  # Stop if we estimate we're out of global GPU space
                  break

        self.logger.info(f"Selected {len(selected)} sequences for initial decode batch.")
        return selected

    def _select_decode_backfill(self, ready_uuids: List[str], global_gpu_stats: List[GPUPagedKVStats]) -> List[str]:
         """ Selects sequences from DECODE_READY to fill empty slots during decode. """
         # Calculate current global decode slots available
         current_decode_count = len(self.sequences_in_decode)
         max_total_seqs = self.world_size * self.max_decode_batch_size_per_rank
         available_slots = max(0, max_total_seqs - current_decode_count)

         num_to_select = min(len(ready_uuids), available_slots)
         if num_to_select == 0:
              return []

         # Similar GPU check as initial batch selection
         total_gpu_free_pages = sum(s.num_free_pages for s in global_gpu_stats)
         selected = []
         pages_needed_approx = 0

         for uuid in ready_uuids:
             if len(selected) >= num_to_select: break
             seq = self.global_registry.get_sequence(uuid)
             est_pages = math.ceil(seq.prompt_length / self.gpu_manager.page_size) + self.PAGE_BUFFER_SIZE
             if pages_needed_approx + est_pages <= total_gpu_free_pages:
                  selected.append(uuid)
                  pages_needed_approx += est_pages
             else:
                  break

         self.logger.info(f"Selected {len(selected)} sequences for decode backfill.")
         return selected

    def _distribute_work(self, global_uuids: List[str]) -> WorkAssignment:
        """ Distributes a list of sequence UUIDs among ranks (simple round-robin). """
        assignments: WorkAssignment = {rank: [] for rank in range(self.world_size)}
        if not global_uuids:
            return assignments

        # Simple round-robin distribution
        # A more complex strategy could consider NUMA, host free space per node, etc.
        for i, uuid in enumerate(global_uuids):
             target_rank = i % self.world_size
             assignments[target_rank].append(uuid)
             # --- CRITICAL: Update Rank Owner ---
             # Rank 0 updates the central registry immediately
             seq = self.global_registry.get_sequence(uuid)
             if seq:
                 seq.rank_owner = target_rank
             else:
                 self.logger.error(f"Sequence {uuid} selected for distribution but not found in registry!")

        self.logger.debug(f"Distributed work: {[len(v) for v in assignments.values()]}")
        return assignments