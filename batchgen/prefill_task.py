import torch
import logging
import time
from typing import List, Dict, Optional, Any
from torch.cuda import Stream, Event # Make sure Event is imported if runtime returns it

# Assume these are correctly imported from your project structure
from batchgen.sequence_manager.batch_defs import (
    GlobalSequenceRegistry, ActiveBatch, SequenceStatus, SequenceEntry
)
from batchgen.inference_runtime import InferenceRuntime
# Assuming HostKVCacheClient and AsyncResult are available
# from batchgen.host_kv_client import HostKVCacheClient, AsyncResult
# Assuming InferenceRuntime is available
# from batchgen.runtime import InferenceRuntime
# Assuming ModelForwardInput/Output are defined appropriately
# from batchgen.model_instance import ModelForwardInput, ModelForwardOutput

class PrefillTask:
    """
    (L2 Abstraction)
    Manages the entire prefill job for a batch of sequences.
    It uses the InferenceRuntime to run the model and handle the offloading behavior.
    Checks runtime state and configures only if necessary.
    """
    def __init__(self,
                 global_registry: GlobalSequenceRegistry,
                 runtime: InferenceRuntime,
                 host_client: 'HostKVCacheClient', # Use HostKVCacheClient type hint
                 device: torch.device): # Added device
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- Owns the L3 registry and L1.5/L0 tools ---
        self.global_registry = global_registry
        self.runtime = runtime
        self.host_client = host_client # Renamed from host_kv_client for consistency
        self.device = device # Needed for ActiveBatch creation

        # Note: kv_copy_engine is assumed to be handled within InferenceRuntime

    def run(self, sequence_uuids: List[str]):
        """
        Executes the entire prefill job for the specified sequence UUIDs.

        Args:
            sequence_uuids: A list of UUIDs belonging to this rank to prefill.
        """
        if not sequence_uuids:
            self.logger.info("PrefillTask received an empty list of sequences. Skipping.")
            return

        self.logger.info(f"Starting PrefillTask for {len(sequence_uuids)} sequences...")

        # --- 1. Get SequenceEntry objects and Update Status ---
        sequences_to_process: List[SequenceEntry] = []
        for uuid in sequence_uuids:
            seq = self.global_registry.get_sequence(uuid)
            if not seq:
                self.logger.error(f"Sequence UUID '{uuid}' not found in GlobalRegistry. Skipping.")
                continue
            # Check if sequence is in a valid starting state
            if seq.status != SequenceStatus.WAITING_IN_QUEUE:
                 self.logger.warning(f"Sequence {uuid} has unexpected status {seq.status.name}. Skipping prefill.")
                 continue
            try:
                seq.update_status(SequenceStatus.PREFILL_PENDING)
                sequences_to_process.append(seq)
            except ValueError as e: # Catch potential invalid transitions if enforced
                self.logger.error(f"Failed to set status for {seq.uuid}: {e}. Skipping.")
                seq.update_status(SequenceStatus.FAILED) # Mark as failed

        if not sequences_to_process:
             self.logger.warning("No valid sequences remaining after status update. PrefillTask exiting.")
             return

        # --- 2. Allocate Host Pages (Async) ---
        self.logger.debug("Allocating host pages...")
        alloc_futures: Dict[str, 'AsyncResult'] = {} # UUID -> AsyncResult
        sequences_pending_alloc = [] # Track sequences for which allocation was requested
        for seq in sequences_to_process:
            try:
                alloc_futures[seq.uuid] = self.host_client.allocate(
                    sequence_id=seq.uuid, # Match server API expectation
                    input_length=seq.prompt_length,
                    max_decode_length=seq.max_output_length
                )
                sequences_pending_alloc.append(seq) # Add to list only if request succeeded
            except Exception as e:
                 self.logger.error(f"Unexpected error during host allocation request for {seq.uuid}: {e}. Marking as FAILED.")
                 seq.update_status(SequenceStatus.FAILED)

        # Update sequences_to_process based on successful alloc requests
        sequences_to_process = sequences_pending_alloc
        if not sequences_to_process:
             self.logger.warning("No sequences remaining after initial host allocation request phase. PrefillTask exiting.")
             return

        # Create the ActiveBatch view for the runtime
        # Assuming runtime provides access to tokenizer's pad_token_id
        pad_token_id = getattr(self.runtime.tokenizer, 'pad_token_id', 0)
        active_batch = ActiveBatch(self.global_registry, [s.uuid for s in sequences_to_process], self.device, pad_token_id)

        try:
            # --- 3. *** NEW: Configure Runtime Only If Necessary *** ---
            current_phase = self.runtime.get_current_phase()
            if current_phase != "prefill":
                self.logger.info(f"Runtime is in phase '{current_phase}'. Configuring for prefill...")
                if current_phase == "decode":
                    # If switching from decode, ensure decode cleanup happens first
                    self.runtime.cleanup_decode()
                self.runtime.config_prefill() # Call your runtime's config method
                self.logger.info("Runtime configured for prefill.")
            else:
                self.logger.info("Runtime already configured for prefill. Skipping reconfiguration.")

            # Update status after configuration attempt (even if skipped)
            for seq in sequences_to_process: seq.update_status(SequenceStatus.IN_PREFILL)

            # --- 4. Wait for Host Allocations ---
            self.logger.debug("Waiting for host page allocations...")
            host_page_map: Dict[str, List[int]] = {}
            failed_alloc_uuids = set()
            start_wait_alloc = time.time()
            # Iterate only over sequences for which allocation was requested
            for seq_id in [s.uuid for s in sequences_to_process]:
                future = alloc_futures.get(seq_id)
                if not future: continue # Should not happen if logic is correct

                try:
                    response = future.get(timeout=60.0)
                    if not response.success:
                        self.logger.error(f"Host allocation failed for {seq_id}: {response.error_message}")
                        failed_alloc_uuids.add(seq_id)
                    else:
                        seq = self.global_registry.get_sequence(seq_id)
                        if seq:
                            seq.host_page_ids = response.physical_page_ids
                            host_page_map[seq_id] = response.physical_page_ids
                        else:
                             self.logger.error(f"Sequence {seq_id} disappeared after successful allocation?")
                             failed_alloc_uuids.add(seq_id)
                except Exception as e:
                     self.logger.error(f"Error or timeout getting host allocation result for {seq_id}: {e}")
                     failed_alloc_uuids.add(seq_id)

            wait_alloc_duration = time.time() - start_wait_alloc
            self.logger.info(f"Host page allocations received/failed in {wait_alloc_duration:.2f}s.")

            # --- Handle Allocation Failures ---
            if failed_alloc_uuids:
                 self.logger.warning(f"Host allocation failed for {len(failed_alloc_uuids)} sequences. Marking as FAILED.")
                 for seq_id in failed_alloc_uuids:
                      seq = self.global_registry.get_sequence(seq_id)
                      if seq: seq.update_status(SequenceStatus.FAILED)
                 # Filter out failed sequences before running the model
                 sequences_to_process = [s for s in sequences_to_process if s.uuid not in failed_alloc_uuids]
                 if not sequences_to_process:
                      self.logger.warning("All sequences failed host allocation. PrefillTask exiting.")
                      # No need to cleanup runtime, leave as prefill
                      return
                 # Recreate active_batch for the filtered list
                 active_batch = ActiveBatch(self.global_registry, [s.uuid for s in sequences_to_process], self.device, pad_token_id)

            # --- 5. Run Prefill Compute & Internal Offload ---
            self.logger.info(f"Running prefill compute & internal offload for {len(active_batch)} sequences...")
            model_input: 'ModelForwardInput' = active_batch.build_prefill_inputs()

            # Pass the host page map relevant to this *active* batch
            current_host_page_map = {uuid: host_page_map[uuid] for uuid in active_batch.active_uuids if uuid in host_page_map}

            # Runtime's prefill method handles compute, layer-wise copies, and returns sync handle
            final_copy_event: Optional[Event] = self.runtime.prefill(model_input, current_host_page_map)
            self.logger.info("Prefill compute & copy launch finished.")

            # --- 6. Synchronize: Wait for all Compute and Copies ---
            if final_copy_event:
                self.logger.debug("Waiting for final copy event from InferenceRuntime...")
                start_sync = time.time()
                # Consider adding a timeout to wait() if runtime might hang
                final_copy_event.wait()
                sync_duration = time.time() - start_sync
                self.logger.info(f"All D2H copies completed (sync via runtime event) in {sync_duration:.2f}s.")
            else:
                 self.logger.warning("InferenceRuntime.prefill did not return a sync event. Using global sync as fallback.")
                 torch.cuda.synchronize()

            # --- 7. Notify Host Server (Async) & Update Final Status ---
            self.logger.debug("Notifying host server of completed prefill...")
            update_futures = []
            final_status = SequenceStatus.DECODE_READY
            for seq in sequences_to_process: # Iterate over successfully processed sequences
                future = self.host_client.update(
                    sequence_id=seq.uuid,
                    new_current_length=seq.prompt_length # Host stores prompt length after prefill
                )
                update_futures.append((seq.uuid, future))
                seq.update_status(final_status) # Update local metadata status

            # --- 8. Wait for Host Updates ---
            self.logger.debug("Waiting for host update acknowledgements...")
            start_wait_update = time.time()
            failed_update_uuids = set()
            for seq_id, future in update_futures:
                try:
                    response = future.get(timeout=10.0)
                    if not response.success:
                         self.logger.error(f"Host metadata update failed for {seq_id}: {response.error_message}")
                         failed_update_uuids.add(seq_id)
                except Exception as e:
                     self.logger.error(f"Error or timeout getting host update result for {seq_id}: {e}")
                     failed_update_uuids.add(seq_id)
            wait_update_duration = time.time() - start_wait_update
            self.logger.debug(f"Host updates acknowledged/failed in {wait_update_duration:.2f}s.")

            if failed_update_uuids:
                 self.logger.error(f"Host update failed for {len(failed_update_uuids)} sequences. Marking as FAILED.")
                 for seq_id in failed_update_uuids:
                      seq = self.global_registry.get_sequence(seq_id)
                      if seq and seq.status == SequenceStatus.DECODE_READY:
                          seq.update_status(SequenceStatus.FAILED)

            self.logger.info(f"PrefillTask completed processing batch.")

        except Exception as e:
            self.logger.error(f"PrefillTask failed critically during execution: {e}", exc_info=True)
            # Mark sequences that were being processed as FAILED
            for seq in sequences_to_process:
                 # Check current status before marking failed
                 current_seq = self.global_registry.get_sequence(seq.uuid)
                 if current_seq and current_seq.status in {SequenceStatus.PREFILL_PENDING, SequenceStatus.IN_PREFILL}:
                      current_seq.update_status(SequenceStatus.FAILED)
            # Re-raise the exception for the main loop / orchestrator
            raise

        # --- NO finally block to clean up runtime ---
        # State is preserved. The next Task (Prefill or Decode) will handle configuration.