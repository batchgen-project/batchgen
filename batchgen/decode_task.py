import torch
import logging
import time
from torch.cuda import Stream, Event
from typing import List, Dict, Optional, Tuple, Any
import math

# Assume these are correctly imported from your project structure
from batchgen.sequence_manager.batch_defs import (
    GlobalSequenceRegistry, ActiveBatch, SequenceStatus, SequenceEntry,
    ModelForwardInput, ModelForwardOutput # Assuming these dataclasses exist
)
# from batchgen.runtime import InferenceRuntime # Your L1.5 Model Runner
# from batchgen.host_kv_client import HostKVCacheClient, AsyncResult # L0
# from batchgen.gpu_kv_manager import GPUKVCacheManager # L1 Vault
# from batchgen.copy_engine import KVCacheCopyEngine # L1.5 Conduit
# from batchgen.orchestrator import PDOrchestrator # Your L3 Orchestrator

logger = logging.getLogger(__name__)

# --- Decode Task --- (L2 Abstraction)
class DecodeTask:
    """
    (L2 Abstraction)
    Manages the continuous batching decode loop for a specific rank.
    Interacts with the PD-Orchestrator, manages GPU KV cache buffers
    via GPUKVCacheManager, handles sequence completions, and uses
    InferenceRuntime for model execution.
    """
    PAGE_BUFFER_SIZE = 2 # The number of extra pages to keep allocated

    def __init__(self,
                 global_registry: GlobalSequenceRegistry,
                 orchestrator: Any, # Use PDOrchestrator type hint
                 runtime: Any, # Use InferenceRuntime type hint
                 gpu_manager: Any, # Use GPUKVCacheManager type hint
                 host_client: Any, # Use HostKVCacheClient type hint
                 copy_engine: Any, # Use KVCacheCopyEngine type hint
                 rank: int,
                 device: torch.device,
                 page_size: int = 64): # Align with FlashMLA block size

        self.logger = logging.getLogger(f"{self.__class__.__name__}_Rank{rank}")

        # --- Owns all its tools ---
        self.global_registry = global_registry
        self.orchestrator = orchestrator # L3 (Interface)
        self.runtime = runtime           # L1.5
        self.gpu_manager = gpu_manager     # L1
        self.host_client = host_client     # L0
        self.copy_engine = copy_engine       # L1.5 (Conduit)

        self.rank = rank
        self.device = device
        self.page_size = page_size
        # Assuming tokenizer info is accessible, e.g., via runtime
        self.pad_token_id = getattr(self.runtime.tokenizer, 'pad_token_id', 0)
        self.eos_token_id = getattr(self.runtime.tokenizer, 'eos_token_id', -1)
        self.eos_token_ids = getattr(self.runtime.tokenizer, 'eos_token_ids', {self.eos_token_id})

        # --- State ---
        self.active_batch: Optional[ActiveBatch] = None # Current batch being decoded
        self.copy_stream = Stream(device=self.device)
        # Tracks pending D2H offloads: seq_id -> (copy_event, metadata_future)
        self.pending_offloads: Dict[str, Tuple[Event, Any]] = {}

    def run(self, initial_batch_uuids: List[str]):
        """
        Executes the continuous batching loop for decoding.

        Args:
            initial_batch_uuids: UUIDs selected by the orchestrator to start decoding.
        """
        self.logger.info(f"Starting DecodeTask with {len(initial_batch_uuids)} initial sequences.")

        # --- Initial Batch Setup ---
        if not initial_batch_uuids:
             self.logger.warning("DecodeTask run() called with empty initial batch.")
             return # Or handle as appropriate

        # Create the initial ActiveBatch view
        self.active_batch = ActiveBatch(self.global_registry, initial_batch_uuids, self.device, self.pad_token_id)
        sequences_in_progress = self.active_batch.active_seqs

        # Load initial KV cache from Host to GPU
        load_event = self._load_sequences_to_gpu(sequences_in_progress)
        if load_event:
            self.logger.debug("Waiting for initial H2D load to complete...")
            load_event.wait() # Ensure initial load is done before first decode
            self.logger.debug("Initial H2D load complete.")

        # Filter out any sequences that failed to load
        sequences_in_progress = [s for s in sequences_in_progress if s.uuid in self.gpu_manager.page_table]
        if not sequences_in_progress:
             self.logger.error("All initial sequences failed to load to GPU. Exiting DecodeTask.")
             # No runtime cleanup needed yet as config hasn't happened
             return
        self.active_batch = ActiveBatch(self.global_registry, [s.uuid for s in sequences_in_progress], self.device, self.pad_token_id)

        try:
            # --- Configure Runtime for Decode (Only if Needed) ---
            current_phase = self.runtime.get_current_phase()
            if current_phase != "decode":
                self.logger.info(f"Runtime is in phase '{current_phase}'. Configuring for decode...")
                if current_phase == "prefill":
                    self.runtime.cleanup_prefill()
                # Pass necessary info like max sequences if needed by config
                self.runtime.config_decode(num_seq=len(self.active_batch), comm=torch.distributed.group.WORLD)
                self.logger.info("Runtime configured for decode.")
            else:
                 self.logger.info("Runtime already configured for decode.")

            # Set initial status
            for seq in sequences_in_progress: seq.update_status(SequenceStatus.IN_DECODE)

            step_count = 0
            # --- The Main Decode Loop ---
            while True:
                if not self.active_batch or len(self.active_batch) == 0:
                     self.logger.info("Active batch is empty. Checking orchestrator before potentially exiting.")
                     # Even with an empty batch, we need to sync and check the orchestrator
                     decision, new_uuids_for_rank = self.orchestrator.get_decision_and_new_sequences()
                     if decision == "SWITCH_TO_PREFILL":
                          self.logger.info("Orchestrator requested SWITCH_TO_PREFILL while batch was empty.")
                          # No flushing needed as batch is empty
                          break # Exit the loop
                     elif new_uuids_for_rank:
                          self.logger.info(f"Received {len(new_uuids_for_rank)} new sequences to load.")
                          # Load these new sequences and restart the loop iteration
                          load_event = self._load_sequences_to_gpu(
                              [self.global_registry.get_sequence(uuid) for uuid in new_uuids_for_rank if self.global_registry.get_sequence(uuid)]
                          )
                          if load_event: load_event.wait()
                          # Recreate active_batch and continue loop
                          self.active_batch = ActiveBatch(self.global_registry, new_uuids_for_rank, self.device, self.pad_token_id)
                          sequences_in_progress = self.active_batch.active_seqs
                          for seq in sequences_in_progress: seq.update_status(SequenceStatus.IN_DECODE)
                          continue
                     else:
                          # No new sequences, batch is empty, continue decision -> wait
                          self.logger.info("No active sequences and no new sequences received. Waiting...")
                          # Add a small sleep or rely on orchestrator sync timeout
                          time.sleep(0.1)
                          step_count = 0 # Reset step count if batch was empty
                          continue # Go back to check orchestrator again

                # --- Synchronization & Decision Point ---
                if step_count % self.page_size == 0:
                    self.logger.debug(f"Reached step {step_count}. Synchronizing and checking orchestrator.")

                    # This call blocks until all ranks are ready and Rank 0 broadcasts
                    decision, new_uuids_for_rank = self.orchestrator.get_decision_and_new_sequences()

                    if decision == "SWITCH_TO_PREFILL":
                        self.logger.info("Orchestrator requested SWITCH_TO_PREFILL.")
                        self._flush_all_sequences_to_host() # Wait for pending copies and flush current state
                        break # Exit the loop

                    elif decision == "CONTINUE":
                        self.logger.debug("Orchestrator: CONTINUE. Running GPU scheduler and checking for new sequences.")
                        # Run GPU scheduler *before* adding new sequences
                        self._run_gpu_scheduler()

                        # Process any new sequences assigned by the orchestrator
                        if new_uuids_for_rank:
                            self.logger.info(f"Received {len(new_uuids_for_rank)} new sequences to load mid-decode.")
                            new_seq_entries = [self.global_registry.get_sequence(uuid) for uuid in new_uuids_for_rank if self.global_registry.get_sequence(uuid)]
                            if new_seq_entries:
                                load_event = self._load_sequences_to_gpu(new_seq_entries)
                                if load_event: load_event.wait() # Ensure loaded before next step
                                # Add successfully loaded sequences to the active batch
                                loaded_uuids = [s.uuid for s in new_seq_entries if s.uuid in self.gpu_manager.page_table]
                                current_uuids = self.active_batch.active_uuids
                                self.active_batch = ActiveBatch(self.global_registry, current_uuids + loaded_uuids, self.device, self.pad_token_id)
                                sequences_in_progress = self.active_batch.active_seqs
                                for seq_id in loaded_uuids: self.global_registry.get_sequence(seq_id).update_status(SequenceStatus.IN_DECODE)
                                self.logger.info(f"Added {len(loaded_uuids)} new sequences to active batch. New size: {len(self.active_batch)}")


                # --- Prepare & Run Decode Step ---
                if not self.active_batch or len(self.active_batch) == 0:
                     self.logger.warning("Active batch became empty unexpectedly before decode step. Skipping step.")
                     continue # Skip to next loop iteration to re-check orchestrator

                self.logger.debug(f"Preparing decode step {step_count} for {len(self.active_batch)} sequences.")
                page_table = self.gpu_manager.build_page_table(self.active_batch.active_uuids)
                model_input: 'ModelForwardInput' = self.active_batch.build_decode_inputs()

                # Run one step of the model
                model_output: 'ModelForwardOutput' = self.runtime.decode(model_input, page_table)

                # --- Post-process Step ---
                new_token_ids = self._process_logits(model_output.logits)
                self.active_batch.update_after_decode_step(new_token_ids)

                completed_uuids = self._find_completed_sequences()
                if completed_uuids:
                    self._handle_completions(completed_uuids)
                    # Recreate active_batch view if sequences were removed
                    current_uuids = [s.uuid for s in self.active_batch.active_seqs if s.uuid not in completed_uuids]
                    self.active_batch = ActiveBatch(self.global_registry, current_uuids, self.device, self.pad_token_id)
                    sequences_in_progress = self.active_batch.active_seqs


                step_count += 1

        except Exception as e:
            self.logger.error(f"DecodeTask failed critically during execution: {e}", exc_info=True)
            # Mark all sequences currently IN_DECODE as FAILED
            if self.active_batch:
                for seq in self.active_batch.active_seqs:
                     # Check current status before marking failed
                     current_seq = self.global_registry.get_sequence(seq.uuid)
                     if current_seq and current_seq.status == SequenceStatus.IN_DECODE:
                          current_seq.update_status(SequenceStatus.FAILED)
            raise # Re-raise for orchestrator/main loop

        finally:
            # --- Cleanup ---
            # Do NOT cleanup runtime if we are preserving state
            # self.runtime.cleanup_decode()
            self.logger.info("DecodeTask run() method finished.")

    # --- GPU Scheduling & Paging ---

    def _run_gpu_scheduler(self):
        """ Checks GPU page usage and offloads sequences if necessary to maintain buffer. """
        stats = self.gpu_manager.get_stats()
        self.logger.debug(f"GPU Stats: Free={stats.num_free_pages}, Used={stats.num_used_pages}, Total={stats.num_total_pages}")

        # Calculate pages needed for current batch + buffer
        pages_needed_for_current_batch = 0
        max_len_in_batch = 0
        if self.active_batch:
            for seq in self.active_batch.active_seqs:
                # Need pages for current length + buffer
                needed = math.ceil(seq.current_length / self.page_size) + self.PAGE_BUFFER_SIZE
                pages_needed_for_current_batch += needed
                max_len_in_batch = max(max_len_in_batch, seq.current_length)

        pages_to_free = max(0, pages_needed_for_current_batch - self.gpu_manager.num_total_pages)

        if pages_to_free > 0:
            self.logger.info(f"GPU scheduler needs to free {pages_to_free} pages to maintain buffer for {len(self.active_batch)} sequences.")
            # Simple greedy: evict longest sequences first
            candidates = self._find_eviction_candidates_greedy(pages_to_free)
            if candidates:
                self._offload_sequences(candidates) # Fire-and-forget offload
                # Update active batch view *after* initiating offload
                current_uuids = [s.uuid for s in self.active_batch.active_seqs if s.uuid not in candidates]
                self.active_batch = ActiveBatch(self.global_registry, current_uuids, self.device, self.pad_token_id)
                self.logger.info(f"Offloaded {len(candidates)} sequences. Active batch size now {len(self.active_batch)}.")
            else:
                 self.logger.warning(f"Needed to free {pages_to_free} pages, but found no eviction candidates!")
                 # This might indicate the batch is too large or memory is too fragmented

        # Now allocate buffer pages for remaining active sequences
        self._allocate_gpu_buffers()


    def _find_eviction_candidates_greedy(self, pages_to_free: int) -> List[str]:
        """ Selects sequences to evict based on longest length first. """
        if not self.active_batch or pages_to_free <= 0:
            return []

        # Sort active sequences by current length, descending
        candidates = sorted(self.active_batch.active_seqs, key=lambda s: s.current_length, reverse=True)

        evicted_uuids = []
        pages_freed_count = 0
        for seq in candidates:
            if pages_freed_count >= pages_to_free:
                break
            # Calculate pages currently held by this sequence
            # Note: This uses the gpu_manager's view, which is the source of truth
            pages_held = len(self.gpu_manager.page_table.get(seq.uuid, []))
            if pages_held > 0:
                evicted_uuids.append(seq.uuid)
                pages_freed_count += pages_held

        if pages_freed_count < pages_to_free:
             self.logger.warning(f"Greedy eviction only freed {pages_freed_count}/{pages_to_free} pages.")

        return evicted_uuids


    def _allocate_gpu_buffers(self):
        """ Ensures all sequences in the active batch have their 2-page buffer allocated. """
        if not self.active_batch:
            return

        self.logger.debug("Allocating GPU buffers...")
        allocated_count = 0
        for seq in self.active_batch.active_seqs:
            if seq.uuid not in self.gpu_manager.page_table:
                 # This sequence might have just been added or failed loading
                 self.logger.warning(f"Sequence {seq.uuid} in active batch but not in GPU page table during buffer allocation.")
                 continue

            current_gpu_pages = len(self.gpu_manager.page_table[seq.uuid])
            # Pages needed for *next* step's potential length + buffer
            pages_needed_total = math.ceil((seq.current_length + 1) / self.page_size) + self.PAGE_BUFFER_SIZE
            pages_to_allocate = max(0, pages_needed_total - current_gpu_pages)

            if pages_to_allocate > 0:
                try:
                    newly_allocated = self.gpu_manager.allocate_pages(seq.uuid, pages_to_allocate)
                    if newly_allocated:
                        allocated_count += len(newly_allocated)
                        # Note: We don't copy H2D here. The kernel only reads allocated pages.
                        # If a sequence was fully evicted and reloaded, _load_sequences_to_gpu handles H2D.
                except RuntimeError as e:
                     # This means scheduler failed to free enough space - critical error
                     self.logger.error(f"Failed to allocate buffer pages for {seq.uuid} even after scheduling: {e}. GPU OOM likely.")
                     # Mark sequence as failed? Or raise exception?
                     seq.update_status(SequenceStatus.FAILED)
                     # Need to remove from active batch safely

        if allocated_count > 0:
             self.logger.debug(f"Allocated {allocated_count} new buffer pages.")


    def _load_sequences_to_gpu(self, sequences_to_load: List['SequenceEntry']) -> Optional[Event]:
        """
        Loads specified sequences from Host KV Cache to GPU KV Cache.
        Allocates necessary GPU pages and triggers H2D copy.
        """
        if not sequences_to_load:
            return None

        self.logger.info(f"Loading {len(sequences_to_load)} sequences from host to GPU...")
        load_futures: Dict[str, 'AsyncResult'] = {}
        host_pages_to_copy: Dict[str, List[int]] = {} # seq_id -> host_page_ids
        gpu_pages_allocated: Dict[str, List[int]] = {} # seq_id -> gpu_page_ids
        sequences_successfully_loaded = []

        # 1. Request host page lists (async)
        for seq in sequences_to_load:
             if seq.status not in {SequenceStatus.DECODE_READY, SequenceStatus.PAUSED_LOW_PRIORITY}:
                  self.logger.warning(f"Cannot load sequence {seq.uuid} with status {seq.status.name}. Skipping.")
                  continue
             load_futures[seq.uuid] = self.host_client.load(sequence_id=seq.uuid)

        # 2. Wait for host responses and allocate GPU pages
        for seq in sequences_to_load:
            if seq.uuid not in load_futures: continue # Skipped or failed request
            future = load_futures[seq.uuid]
            try:
                response = future.get(timeout=30.0) # Timeout for host response
                if not response.success:
                    self.logger.error(f"Failed to load host metadata for {seq.uuid}: {response.error_message}")
                    seq.update_status(SequenceStatus.FAILED)
                    continue

                num_host_pages = math.ceil(response.current_length / self.page_size)
                host_pages = response.physical_page_ids[:num_host_pages] # Only get pages with data
                host_pages_to_copy[seq.uuid] = host_pages

                # Allocate GPU pages (current data pages + buffer)
                num_gpu_pages_needed = num_host_pages + self.PAGE_BUFFER_SIZE
                gpu_pages = self.gpu_manager.allocate_pages(seq.uuid, num_gpu_pages_needed)
                if not gpu_pages: # Allocation failed (OOM)
                    self.logger.error(f"Failed to allocate {num_gpu_pages_needed} GPU pages for {seq.uuid}. Marking FAILED.")
                    seq.update_status(SequenceStatus.FAILED)
                    # Need to potentially release host lock if load implies one? No, load is read-only.
                    continue

                gpu_pages_allocated[seq.uuid] = gpu_pages
                sequences_successfully_loaded.append(seq) # Mark as ready for copy

            except Exception as e:
                self.logger.error(f"Error or timeout during host load/GPU alloc for {seq.uuid}: {e}")
                seq.update_status(SequenceStatus.FAILED)

        # 3. Trigger H2D Copy for successfully loaded sequences
        h2d_page_pairs = []
        for seq in sequences_successfully_loaded:
             host_pages = host_pages_to_copy[seq.uuid]
             # Only copy pages that have data
             gpu_pages_for_data = gpu_pages_allocated[seq.uuid][:len(host_pages)]
             h2d_page_pairs.extend(zip(host_pages, gpu_pages_for_data))

        if h2d_page_pairs:
            self.logger.debug(f"Launching H2D copy for {len(h2d_page_pairs)} pages...")
            self.copy_engine.copy_host_to_gpu(h2d_page_pairs, stream=self.copy_stream)
            # Return an event recorded *after* the copy is launched
            load_event = Event()
            load_event.record(self.copy_stream)
            return load_event
        else:
            self.logger.debug("No H2D copies needed for load.")
            return None


    # --- Sequence Lifecycle & Offload ---

    def _offload_sequences(self, sequence_uuids: List[str]):
        """
        Launches non-blocking D2H copy and host metadata update
        for sequences being evicted from GPU or completed.
        Stores sync handles in self.pending_offloads.
        """
        self.logger.debug(f"Initiating offload for {len(sequence_uuids)} sequences: {sequence_uuids}")
        page_pairs_to_copy = []
        update_futures = []
        sequences_being_offloaded = []

        for seq_id in sequence_uuids:
            seq = self.global_registry.get_sequence(seq_id)
            if not seq or seq_id not in self.gpu_manager.page_table:
                self.logger.warning(f"Cannot offload sequence {seq_id}: Not found or not on GPU.")
                continue

            sequences_being_offloaded.append(seq)
            gpu_pages = self.gpu_manager.page_table[seq_id] # Get pages currently on GPU
            host_pages = seq.host_page_ids             # Get corresponding host pages

            # Determine which pages need copying (e.g., only those containing new tokens)
            # For simplicity, copy all pages up to current length.
            num_pages_to_copy = math.ceil(seq.current_length / self.page_size)
            gpu_pages_to_copy = gpu_pages[:num_pages_to_copy]
            host_pages_to_copy = host_pages[:num_pages_to_copy]

            if len(gpu_pages_to_copy) != len(host_pages_to_copy):
                 self.logger.error(f"Mismatch in GPU ({len(gpu_pages_to_copy)}) vs Host ({len(host_pages_to_copy)}) pages to copy for {seq_id}. Aborting offload for this seq.")
                 # Mark as failed?
                 seq.update_status(SequenceStatus.FAILED)
                 continue # Skip this sequence

            page_pairs_to_copy.extend(zip(gpu_pages_to_copy, host_pages_to_copy))

            # Trigger async host metadata update
            future = self.host_client.update(
                sequence_id=seq.uuid,
                new_current_length=seq.current_length
            )
            update_futures.append((seq.uuid, future))

            # Update status to indicate it's paused/offloaded (unless completed)
            if seq.status == SequenceStatus.IN_DECODE:
                 seq.update_status(SequenceStatus.PAUSED_LOW_PRIORITY)

            # Free GPU pages *after* initiating copy (they are needed for the copy source)
            # The actual freeing happens after copy completes if strict sync is needed,
            # but for fire-and-forget, we assume copy engine handles reads correctly.
            # However, freeing immediately makes pages available sooner. Let's free now.
            self.gpu_manager.free_sequence(seq_id) # This removes from gpu_manager.page_table

        # Launch the non-blocking D2H copy for all sequences together
        if page_pairs_to_copy:
            self.copy_engine.copy_gpu_to_host(page_pairs_to_copy, stream=self.copy_stream)
            # Record the event *after* launching copies
            copy_event = Event()
            copy_event.record(self.copy_stream)

            # Store the sync handles for sequences successfully launched
            successfully_offloaded_ids = {s.uuid for s in sequences_being_offloaded if s.status != SequenceStatus.FAILED}
            for seq_id, future in update_futures:
                 if seq_id in successfully_offloaded_ids:
                     self.pending_offloads[seq_id] = (copy_event, future)
        else:
            self.logger.debug("No pages needed copying for offload.")


    def _flush_all_sequences_to_host(self):
        """
        Performs a *blocking* flush of all active sequences to host
        before switching tasks. Waits for pending offloads and copies
        the current state.
        """
        if not self.active_batch or len(self.active_batch) == 0:
            self.logger.info("Flush requested, but active batch is empty.")
            return

        self.logger.info(f"Flushing all {len(self.active_batch)} active sequences to host...")
        active_uuids_to_flush = self.active_batch.active_uuids

        # 1. Wait for any previously initiated offloads for these sequences
        self.logger.debug("Waiting for any pending offloads to complete...")
        start_wait_pending = time.time()
        for seq_id in list(self.pending_offloads.keys()): # Iterate copy as we modify dict
             if seq_id in active_uuids_to_flush:
                  event, future = self.pending_offloads.pop(seq_id)
                  try:
                       event.wait()
                       response = future.get(timeout=10.0)
                       if not response.success:
                            self.logger.error(f"Pending host update failed for {seq_id} during flush: {response.error_message}")
                            # Mark as failed?
                            seq = self.global_registry.get_sequence(seq_id)
                            if seq: seq.update_status(SequenceStatus.FAILED)
                  except Exception as e:
                       self.logger.error(f"Error waiting for pending offload of {seq_id} during flush: {e}")
                       seq = self.global_registry.get_sequence(seq_id)
                       if seq: seq.update_status(SequenceStatus.FAILED)

        wait_pending_duration = time.time() - start_wait_pending
        self.logger.debug(f"Pending offloads completed/failed in {wait_pending_duration:.2f}s.")

        # 2. Initiate final offload for the current state (blocking within this function)
        # Filter sequences that might have failed during pending wait
        sequences_to_flush_now = [s for s in self.active_batch.active_seqs if s.status != SequenceStatus.FAILED]
        if sequences_to_flush_now:
             page_pairs = []
             update_futures = []
             uuids_in_final_flush = []
             for seq in sequences_to_flush_now:
                  if seq.uuid not in self.gpu_manager.page_table:
                       self.logger.warning(f"Sequence {seq.uuid} intended for flush not found in GPU page table.")
                       continue # Skip if pages already freed somehow

                  uuids_in_final_flush.append(seq.uuid)
                  gpu_pages = self.gpu_manager.page_table[seq.uuid]
                  host_pages = seq.host_page_ids
                  num_pages_to_copy = math.ceil(seq.current_length / self.page_size)
                  gpu_pages_to_copy = gpu_pages[:num_pages_to_copy]
                  host_pages_to_copy = host_pages[:num_pages_to_copy]

                  if len(gpu_pages_to_copy) != len(host_pages_to_copy):
                       self.logger.error(f"Mismatch GPU({len(gpu_pages_to_copy)}) vs Host({len(host_pages_to_copy)}) pages for final flush {seq.uuid}.")
                       seq.update_status(SequenceStatus.FAILED)
                       continue # Skip copy

                  page_pairs.extend(zip(gpu_pages_to_copy, host_pages_to_copy))
                  future = self.host_client.update(seq.uuid, seq.current_length)
                  update_futures.append((seq.uuid, future))
                  # Don't change status here, orchestrator handles state after switch

             if page_pairs:
                  self.logger.debug(f"Launching final D2H copy for {len(uuids_in_final_flush)} sequences...")
                  self.copy_engine.copy_gpu_to_host(page_pairs, stream=self.copy_stream)
                  final_copy_event = Event()
                  final_copy_event.record(self.copy_stream)

                  self.logger.debug("Waiting for final D2H copy to complete...")
                  start_sync = time.time()
                  final_copy_event.wait() # Block for the final copy
                  sync_duration = time.time() - start_sync
                  self.logger.debug(f"Final D2H copy complete in {sync_duration:.2f}s.")

             # Wait for final host updates
             self.logger.debug("Waiting for final host updates...")
             start_wait_update = time.time()
             for seq_id, future in update_futures:
                  try:
                       response = future.get(timeout=10.0)
                       if not response.success:
                            self.logger.error(f"Final host update failed for {seq_id}: {response.error_message}")
                            seq = self.global_registry.get_sequence(seq_id)
                            if seq: seq.update_status(SequenceStatus.FAILED)
                  except Exception as e:
                       self.logger.error(f"Error/timeout on final host update for {seq_id}: {e}")
                       seq = self.global_registry.get_sequence(seq_id)
                       if seq: seq.update_status(SequenceStatus.FAILED)
             wait_update_duration = time.time() - start_wait_update
             self.logger.debug(f"Final host updates acknowledged/failed in {wait_update_duration:.2f}s.")

        # 3. Free all GPU pages for the flushed sequences
        self.logger.debug("Freeing GPU pages for all flushed sequences.")
        for seq_id in active_uuids_to_flush: # Use original list before filtering
            # free_sequence is safe even if seq_id is not in page_table
            self.gpu_manager.free_sequence(seq_id)

        self.active_batch = None # Clear the active batch
        self.logger.info("Flush complete.")


    def _handle_completions(self, completed_uuids: List[str]):
        """
        Handles completed sequences (<EOS> or max_length).
        Initiates final non-blocking offload and frees GPU resources.
        """
        if not completed_uuids:
            return

        self.logger.info(f"Handling {len(completed_uuids)} completed sequences: {completed_uuids}")

        # Update status in global registry
        for seq_id in completed_uuids:
             seq = self.global_registry.get_sequence(seq_id)
             if seq and seq.status == SequenceStatus.IN_DECODE: # Ensure it was decoding
                 seq.update_status(SequenceStatus.COMPLETED)

        # Initiate the final offload (fire-and-forget)
        # _offload_sequences handles freeing GPU pages and updating pending_offloads
        self._offload_sequences(completed_uuids)

        # Note: We don't wait here. If the orchestrator needs to trigger
        # host eviction for completed sequences, it must coordinate
        # by checking pending_offloads or using sequence status.

    # --- Utility Methods ---

    def _process_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """ Extracts the next token ID from logits (e.g., argmax). """
        # Assuming logits shape is [batch_size, vocab_size] for the last token
        if logits.ndim == 3 and logits.shape[1] == 1: # Handle [B, 1, V]
            logits = logits.squeeze(1)
        elif logits.ndim != 2:
             raise ValueError(f"Unexpected logits shape: {logits.shape}")

        # Simple greedy sampling (argmax)
        next_token_ids = torch.argmax(logits, dim=-1) # Shape: [batch_size]
        return next_token_ids

    def _find_completed_sequences(self) -> List[str]:
        """ Checks the active batch for sequences that hit EOS or max length. """
        completed = []
        if not self.active_batch:
            return completed

        for seq in self.active_batch.active_seqs:
            # Check max output length
            if len(seq.output_tokens) >= seq.max_output_length:
                self.logger.debug(f"Sequence {seq.uuid} reached max output length.")
                completed.append(seq.uuid)
                continue # Skip EOS check if max length hit

            # Check EOS token (only if output_tokens is not empty)
            if seq.output_tokens and seq.output_tokens[-1] in self.eos_token_ids:
                self.logger.debug(f"Sequence {seq.uuid} hit EOS token.")
                completed.append(seq.uuid)

        return completed