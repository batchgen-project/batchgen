"""Worker lifecycle behind a worker seam (S9, WorkerLifecycle).

``WorkerLifecycleEngine`` owns the worker rank's lifecycle steps around its
composition root: pool init (``Init``) and batch-config updates, the
manager phase-contract check, the run and decode watchdogs, startup-phase
reporting and deployment preparation, and the teardown path (FP8 weight
unregistration, decode-graph release, deep free of model memory).

The composition root itself stays on the worker: ``__init__``,
``_initialize_core_components`` and ``_initialize_trajectory_book`` build the
worker's collaborators (config, initializer, parallel-strategy manager,
tokenizer, query book). Graph-mode routing (``_warmup_cuda_graphs``) and the
K3 startup trio stay too (W1/W2 material).

The engine is intentionally stateless, the S5/S8/S7 shape: each operation
receives the live worker as its collaborator and rebinds ``self`` to it, so the
moved statements keep their original state access and call spellings. The
worker keeps a same-name delegator for every moved method.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time

import torch

from batchgen.phase_lifecycle import verify_phase_contract
from batchgen.worker import startup_transient as _startup_transient
from batchgen.worker.rank_context import RankContext


class WorkerLifecycleEngine:
    """Stateless worker-lifecycle operations; the worker owns all state."""

    # ---- Pool init and the phase contract ----------------------------------

    def Init(self, worker, max_input_length, max_decoding_length, num_queries):
        """
        Initialize/reconfigure before the persistent generate() loop starts.
        - First call: performs full initialization of core_engine, parallel_manager, etc.
        - Subsequent calls: only updates batch parameters.

        Every caller runs before generate_persistent(), i.e. while global_batch
        is still None; the per-batch state reset the legacy path needed between
        two global batches went away with that path.

        Args:
            max_input_length: Maximum input length hint. If None, will be determined dynamically
                              during tokenization as the longest prompt in the batch.
                              For first initialization, a default of 8192 is used if None.
            max_decoding_length: Decode-budget hint for init-time sizing. In
                                 pool mode it is the budget of the first batch the
                                 server saw, so nothing may cap a sequence with it:
                                 each admission entry carries that sequence's own
                                 budget (``kv_token_budget``).
            num_queries: Number of queries in the global batch.
        """
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        # Update batch-specific parameters
        # max_input_length can be None - will be set during tokenization
        # For first initialization, use a reasonable default if None (needed for scheduler)
        if max_input_length is None or max_input_length <= 0:
            # Default hint for scheduler; actual value determined during tokenization
            self.max_input_length = 8192 if not self._core_initialized else 0
        else:
            self.max_input_length = max_input_length
        self.max_decoding_length = max_decoding_length

        logging.info(f"Initializing batchgen with global rank {self.args.global_rank} and world size {self.args.world_size} with PID: {os.getpid()}")
    
        # One-time initialization (only on first call)
        if not self._core_initialized:
            self._initialize_core_components(num_queries)
            self._core_initialized = True
        else:
            # Just update the num_queries and batch-related config
            self._update_batch_config(num_queries)

        # Freeze this rank's identity. G (`_decode_attn_tp_size`) reads the
        # parallel strategy manager, which `_initialize_core_components`
        # constructs, so this is the earliest point it is knowable. Every
        # field is stable for the process lifetime, so a re-Init rebuilds an
        # equal object.
        self.rank_ctx = RankContext(
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            local_world_size=self.local_world_size,
            decode_group_size=self._decode_attn_tp_size(),
        )

        logging.info(f"Engine on device {self.device} initialized/reconfigured.")

    def _update_batch_config(self, worker, num_queries: int) -> None:
        """
        Update configuration for a new batch without reinitializing heavy components.
        Called on subsequent Init() calls after the first.
        """
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        logging.info(f"Rank {self.rank}: Updating batch config for new batch")
    
        # Update engine config with new batch parameters
        self.engine_config.Basic_Config.max_decoding_length = self.max_decoding_length
        self.engine_config.Basic_Config.set_max_prompt_length(self.max_input_length)
        self.engine_config.Basic_Config.num_queries = num_queries
    
        # Update input_arguments for any components that might reference them
        if hasattr(self, 'input_arguments'):
            self.input_arguments.max_prompt_length = self.max_input_length
            self.input_arguments.padding_length = self.max_input_length
            self.input_arguments.max_decoding_length = self.max_decoding_length
            self.input_arguments.num_queries = num_queries
    
        # Reset per-batch state
        self.query_book = None
        self.model_batch_book = {}
    
        logging.info(f"Rank {self.rank}: Batch config updated (max_input={self.max_input_length}, max_decode={self.max_decoding_length}, num_queries={num_queries})")

    def _verify_manager_phase_contract(self, worker) -> None:
        """Refuse service when the active manager misses a required hook."""
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        errors = verify_phase_contract(self.parallel_manager)
        if errors:
            raise RuntimeError(
                f"phase lifecycle contract failed for "
                f"{type(self.parallel_manager).__name__}: "
                + "; ".join(errors)
            )

    # ---- Watchdogs ---------------------------------------------------------

    def set_watchdog(self, worker, watchdog) -> None:
        """
        Set the watchdog for stuck detection during inference.

        The watchdog will be fed periodically during generation to prevent
        false timeout detection on long-running inference.

        Args:
            watchdog: Watchdog instance with a feed() method, or None to disable
        """
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        self._watchdog = watchdog

    def set_decode_watchdog(self, worker, watchdog) -> None:
        """Set a per-decode-step watchdog. Starts disabled; enabled only during decode."""
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        self._decode_watchdog = watchdog
        # Start disabled — only enable around actual decode iterations
        if hasattr(watchdog, '_active'):
            watchdog._active = False

    def feed_watchdog(self, worker) -> None:
        """Feed the watchdog to prevent timeout during long operations."""
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self._watchdog is not None:
            self._watchdog.feed()

    def feed_decode_watchdog(self, worker) -> None:
        """Feed the decode watchdog at the start of each decode step."""
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self._decode_watchdog is not None:
            self._decode_watchdog.feed()

    def enable_decode_watchdog(self, worker) -> None:
        """Enable decode watchdog monitoring (call before decode loop)."""
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self._decode_watchdog is not None and hasattr(self._decode_watchdog, '_active'):
            self._decode_watchdog._active = True
            self._decode_watchdog.feed()  # Reset timer

    def disable_decode_watchdog(self, worker) -> None:
        """Disable decode watchdog monitoring (call after decode loop)."""
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self._decode_watchdog is not None and hasattr(self._decode_watchdog, '_active'):
            self._decode_watchdog._active = False

    # ---- Startup reporting and deployment ----------------------------------

    def set_startup_phase_reporter(self, worker, reporter) -> None:
        """Install the handle this worker reports its startup phases through.

        Like the cohort-shutdown probe below, the shape behind it -- a
        lock-free shared record the server polls -- stays in the supervisor;
        this class only ever says which phase it has entered.
        """
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        self._startup_phase_reporter = reporter

    def _startup_phase(self, worker, name: str):
        """Report one named startup phase for the duration of the body.

        ``name`` must be one of ``batchgen.worker.startup_phases.STARTUP_PHASES``;
        the reporter refuses anything else.
        """
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return self._startup_phase_reporter.phase(name)

    def prepare_deployment(self, worker) -> None:
        """Resolve and verify L against decode residency before worker readiness.

        The measured provisional budget is retained for every model and mode;
        it reserves the configured fraction's slack for later allocations.
        Both lifecycles probe the startup decode transient at the per-rank cap
        right after the pool is sized (``_probe_startup_transient``); it is
        published, not yet reserved.

        On the default lifecycle a decode instance the strategy manager keeps
        across the switch to prefill has its decode graphs captured here,
        inside that probe and before readiness
        (``_startup_decode_graph_capture``); every later activation replays
        them. An instance the manager rebuilds per phase captures at its
        decode configure.
        """
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        with self._startup_phase("model-build"):
            if not self._core_initialized:
                self.Init(None, 1, 0)
        with self._startup_phase("communicator-init"):
            self._ensure_pynccl_communicator()
        with self._startup_phase("decode-shard-build"):
            if self._persistent_phase_enabled():
                self._build_persistent_phase_instances()
            else:
                prefill_key = self._desired_phase_key("prefill")
                self._prefill_moe_mode_in_effect = prefill_key.prefill_moe_mode
                self._load_decode_model(comm=self.comm)
                self._init_gpu_kv_with_actual_size()
                self._probe_startup_transient(
                    seam=_startup_transient.SEAM_DEPLOYMENT_PREPARE,
                    capture=self._startup_decode_graph_capture(prefill_key),
                )
                self._transition_phase(prefill_key)
                self.set_phase("prefill")
                self._install_prefill_weight_copy_pipeline(False)
        logging.info("Deployment max_model_len verified: %s", self.model_context_length)

    # ---- Teardown ----------------------------------------------------------

    def _unregister_fp8_weights(self, worker):
        # Skip FP8 unregistration for models that don't use FP8 (e.g., GPT-OSS uses MXFP4)
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if not hasattr(self.loaded_model_config, 'first_k_dense_replace'):
            return

        # Models whose MoE layers don't expose DeepSeek-style `.mlp` (e.g.
        # Kimi-Linear uses `.block_sparse_moe` with BF16 experts) have no FP8
        # weights to unregister.
        _fkd = self.loaded_model_config.first_k_dense_replace
        if _fkd < len(self.model.model.layers) and not hasattr(
            self.model.model.layers[_fkd], 'mlp'
        ):
            return

        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            if hasattr(attn_module, '_unregister_fp8_weights'):
                attn_module._unregister_fp8_weights()
            if layer_idx >= self.loaded_model_config.first_k_dense_replace:
                shared_experts = getattr(self.model.model.layers[layer_idx].mlp, 'shared_experts', None)
                if shared_experts is not None and hasattr(shared_experts, '_unregister_fp8_weights'):
                    shared_experts._unregister_fp8_weights()
                for routed_expert_idx in range(self.model_config.num_local_experts):
                    if hasattr(self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx], '_unregister_fp8_weights'):
                        self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx]._unregister_fp8_weights()
                if hasattr(self.model.model.layers[layer_idx].mlp, "cleanup"):
                    self.model.model.layers[layer_idx].mlp.cleanup()

    def _release_decode_graph_state(self, worker) -> None:
        """Drop the decode instance's captured graphs and their bookkeeping."""
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        self._graph.cuda_graph_manager = None
        self._graph.whole_model_segment = None
        self._graph.glm5_whole_model_all_short = False
        self._graph.glm5_whole_model_index_topk = 0
        self._graph.whole_model_bucketing = None
        self._graph.glm5_whole_model_capture_input_ids = None
        self._graph.whole_model_graph = False
        # The K2.5 capture's two descriptors go with the graphs they describe,
        # so a later capture of any mode starts from the field defaults.
        self._graph.k25_whole_model_graph = False
        self._graph.whole_model_segment_name = None
        self._graph.glm5_whole_model_graph = False
        self._graph.glm5_whole_model_graph_failed_buckets = set()
        self._graph.glm5_whole_model_graph_capture_attempted_for_batch = False
        self._graph.glm5_whole_model_graph_state_change_after_capture_logged = False
        self._graph.glm5_whole_model_graph_signature = None
        self._graph.glm5_whole_model_graph_unavailable_reason = None
        # The probe's spin nodes live in the graphs dropped above.
        self._criticality_probe = None
        # Adapter holds Python refs to model/segment/KV manager via _ctx;
        # without release_context() the captured segment's static KV buffers
        # survive empty_cache() and prefill OOMs on the next batch.
        if self._graph.cuda_graph_adapter is not None:
            self._graph.cuda_graph_adapter.release_context()
        # A later decode instance is a new owner and captures its own graphs.
        self._graph.decode_graph_captures = 0

    def deep_free_model_memory(self, worker):
        """Release model memory without CPU transfer overhead.

        Previous implementation moved model to CPU before deletion, causing
        unnecessary PCIe traffic for large models. This minimal approach:
        1. Synchronizes CUDA to ensure pending ops complete
        2. Deletes model reference directly
        3. Releases memory back to CUDA allocator
        """
        # Moved VERBATIM from batchgen_worker.py (S9); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if not hasattr(self, 'model') or self.model is None:
            return

        self._phase_mem_switch_index = (
            getattr(self, "_phase_mem_switch_index", 0) + 1
        )
        self._log_phase_memory_ledger("deep_free_enter")

        # Stage timings on every rank: the first decode configure after the
        # streamed wave-1 prefill spent 34 s here on rank 0 (r30) / ~30 s on
        # node 1 (r29), intermittently, with nothing logged.
        _df_t0 = time.perf_counter()
        # Ensure all GPU operations complete before deletion
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.torch_device)
        _df_t1 = time.perf_counter()

        # Free WGMMA shared buffers if they exist (class-level, survives model deletion).
        # Only when the GLM-5 model module is already loaded: importing it here
        # on a non-GLM-5 server walks the GLM-5 JIT extensions -- MEASURED r33
        # (K3, 2x8 H200, shared TORCH_EXTENSIONS_DIR): del=30.2 s on every rank
        # of one node (0.2 s on the other) at the first decode configure.
        _glm5_model_module = sys.modules.get("batchgen.models.glm.glm5.model")
        if _glm5_model_module is not None:
            Glm5MoE = getattr(_glm5_model_module, "Glm5MoE", None)
            if Glm5MoE is not None and getattr(Glm5MoE, '_wgmma_shared_bufs', None) is not None:
                Glm5MoE._wgmma_shared_bufs.free_buffers()
                Glm5MoE._wgmma_shared_bufs = None
                Glm5MoE._wgmma_next_layer_id = 0

        # Delete model directly without CPU transfer
        del self.model
        self.model = None
        # Decode graph state belongs to the decode instance that captured it.
        # A held decode instance replays those graphs on its next activation
        # (capture-once), so freeing the live prefill model must not drop
        # them; they go when the decode instance itself is retired, whose
        # phase_exit clears the manager's hold before this runs.
        if not self.parallel_manager.holds_instance("decode"):
            self._release_decode_graph_state()

        # Defense-in-depth: free PSM-owned GPU buffers that survive model deletion
        # (INT4 contiguous weight buffers, MoE class-level buffers)
        _df_t2 = time.perf_counter()
        if hasattr(self, 'parallel_manager') and self.parallel_manager is not None:
            pm = self.parallel_manager
            if hasattr(pm, "_release_streamed_sp8_prefill"):
                pm._release_streamed_sp8_prefill()
            # The worker is not the only owner of the model: the parallel strategy
            # manager keeps its own reference from configure_prefill/
            # configure_decoding. Leaving that reference set is what made this
            # function a measured no-op on the decode->prefill switch (allocated
            # bytes unchanged across the call): the decode model was really freed
            # later, by the refcount drop when configure_prefill rebound pm.model,
            # so it stayed resident through the KV destroy and the first part of the
            # prefill configure. Dropping it here is what makes the function free
            # what its name says it frees.
            #
            # Persistent instances own their model tuples; the guard keeps those
            # references intact while a suspend transition is in progress.
            if not getattr(pm, "persistent_phase_instances", False):
                if getattr(pm, "model", None) is not None:
                    pm.model = None
            for attr in ('_int4_packed_gpu_buf', '_int4_scale_gpu_buf'):
                if hasattr(pm, attr):
                    delattr(pm, attr)
        _df_t3 = time.perf_counter()

        # Release memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _df_t4 = time.perf_counter()
        gc.collect()
        logging.info(
            f"[DECODE] Rank {self.rank}: deep_free_model_memory sync={_df_t1 - _df_t0:.1f}s "
            f"del={_df_t2 - _df_t1:.1f}s release_streamed={_df_t3 - _df_t2:.1f}s "
            f"empty_cache={_df_t4 - _df_t3:.1f}s gc={time.perf_counter() - _df_t4:.1f}s"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # After empty_cache: what is still held here is what this switch
        # did NOT give back, which is the number the rebuild path's
        # net-zero invariant is read from.
        self._log_phase_memory_ledger("deep_free_done")
