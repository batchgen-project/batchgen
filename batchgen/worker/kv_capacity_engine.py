"""KV capacity and layout behind a worker seam (S10a, KVCapacity).

``KVCapacityEngine`` owns the worker rank's KV capacity and layout surface:
GPU KV pool sizing (the fixed-address pool, the page-table capacity request,
actual-size init and the post-capture finalize), the GPU KV pool handle,
remap and region asserts, the node memfd identity exchange, the KV stats read
through ``KVCacheManager``, the host-KV health line, the per-rank decode cap
and the per-group host views.

S10b (watermark, migration, host capacity) and S10c (page boundary, GPU
allocation, load, host append, eviction) add to this engine. The frozen S5
straddlers ``_mirror_group_host_kv`` and ``_release_host_kv_pages_for_batch``
stay on the worker.

The engine is intentionally stateless, the S5/S8/S7/S9 shape: each operation
receives the live worker as its collaborator and rebinds ``self`` to it, so the
moved statements keep their original state access and call spellings. The
worker keeps a same-name delegator for every moved method.
"""

from __future__ import annotations

import dataclasses as _dataclasses
import gc
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from batchgen.kv_cache.dual_host_kv_coordinator import DualHostKVCoordinator
from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.kv_cache.host_kv_health import format_host_kv_health
from batchgen.worker.kv_manager import (
    GPU_KV_POOL_HEADROOM_GB,
    KVCacheManager,
    KVStats,
    KVUtilizationRequest,
    PageTableCapacityRequest,
)
from batchgen.worker.kv_plan import provisional_pool_gb
from batchgen.worker.rank_context import node_of_rank


# Max per-rank in-decode batch. Bounds the K2.5 MoE padded buffer (mtp = round_up(world_size *
# this)) so init does not OOM on a large candidate pool, and is the per-rank admission cap so
# the pre-reserved buffer never overflows at runtime. Raise to fill more GPU KV (memory permitting).
_MAX_DECODE_RANK_BSZ = int(os.environ.get("BATCHGEN_MAX_DECODE_RANK_BSZ", "128"))

# GPU KV cache size override (DEPRECATED: use --gpu-memory-frac CLI arg instead)
# If set, overrides the automatic calculation. Otherwise, size is computed as:
# gpu_kv_cache = GPU_mem * gpu_memory_frac - model_instance_size
_GPU_KV_CACHE_SIZE_OVERRIDE = os.environ.get("BATCHGEN_GPU_KV_CACHE_SIZE_GB")


class KVCapacityEngine:
    """Stateless KV capacity and layout operations; the worker owns all state."""

    # ---- GPU KV pool sizing ------------------------------------------------

    def _calculate_gpu_kv_cache_size(self, worker) -> float:
        """
        Calculate GPU KV cache size based on actual GPU memory usage.

        Uses torch.cuda.mem_get_info() to get real memory usage after model is loaded.
        Formula: gpu_kv_cache = total_gpu_mem * gpu_memory_frac - used_mem

        This reserves (1-gpu_memory_frac) of total GPU memory for activations and overhead.

        IMPORTANT: Must be called in _initialize_core_components() right after model loading,
        BEFORE any inference (prefill/decode). If called during/after prefill, activation
        memory will be included in 'used_mem', resulting in incorrect (possibly negative) size.

        Rank 0 calculates and broadcasts to all ranks to ensure consistency.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        # Check for environment variable override first
        if _GPU_KV_CACHE_SIZE_OVERRIDE is not None:
            gpu_kv_cache_gb = float(_GPU_KV_CACHE_SIZE_OVERRIDE)
            if self.rank == 0:
                logging.info(
                    f"[GPU-KV] Size from env override: {gpu_kv_cache_gb:.2f} GB "
                    f"(BATCHGEN_GPU_KV_CACHE_SIZE_GB)"
                )
            return gpu_kv_cache_gb

        # Rank 0 calculates, then broadcasts to all ranks
        if self.rank == 0:
            # Get actual GPU memory usage (after model is loaded)
            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(self.local_rank)
            free_mem_gb = free_mem_bytes / (1024 ** 3)
            total_mem_gb = total_mem_bytes / (1024 ** 3)
            used_mem_gb = total_mem_gb - free_mem_gb

            # Formula: gpu_kv_cache = total * frac - used (single definition:
            # kv_plan.provisional_pool_gb).
            # This reserves (1-frac) of GPU memory for activations and overhead
            calculated_gb = provisional_pool_gb(
                total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=None
            )
            # Ensure positive value: a non-positive remainder warns here and is
            # replaced by the 1 GB floor kv_plan substitutes.
            gpu_kv_cache_gb = provisional_pool_gb(
                total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=1.0
            )
            if calculated_gb <= 0:
                logging.warning(
                    f"[GPU-KV] Calculated size is non-positive ({calculated_gb:.2f} GB). "
                    f"Total: {total_mem_gb:.2f} GB × frac: {self.gpu_memory_frac} - used: {used_mem_gb:.2f} GB. "
                    f"Using minimum 1 GB."
                )

            logging.info(
                f"[GPU-KV] Size calculated: {gpu_kv_cache_gb:.2f} GB "
                f"(total: {total_mem_gb:.2f} GB × frac: {self.gpu_memory_frac} - used: {used_mem_gb:.2f} GB)"
            )
        else:
            gpu_kv_cache_gb = 0.0

        # Broadcast from rank 0 to all ranks
        size_tensor = torch.tensor([gpu_kv_cache_gb], dtype=torch.float32, device=self.torch_device)
        dist.broadcast(size_tensor, src=0)
        gpu_kv_cache_gb = float(size_tensor.item())

        return gpu_kv_cache_gb

    def _kv_fixed_address_pool_available(self, worker) -> bool:
        """Whether the KV pools can be carved from a reserved virtual range.

        CUDA VMM is the only implementation, and it needs an indexed device.
        Every serving path has one; CPU-only unit tests that build a manager
        directly keep the allocator-backed default.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return bool(torch.cuda.is_available() and isinstance(self.local_rank, int))

    def _initialize_gpu_kv_manager_fixed_size(self, worker) -> GPUPagedKVCacheManager:
        """
        Initialize GPU KV manager with pre-determined fixed size.
        Called once at the start of decoding.

        For DSA models, splits the memory budget between primary (MLA) and
        auxiliary (indexer) caches, wrapping both in a DualKVCacheCoordinator.

        The pools are carved from one reserved virtual range per pool
        (``fixed_address_pool``), so the phase-switch lifecycle can hand their
        physical pages back to the driver for the prefill phase and map them
        again afterwards WITHOUT the pool base, the page-pointer tables or the
        page-table storage moving. A captured decode graph bakes those
        pointers, so releasing an allocator-backed pool under it would make the
        next replay read freed memory.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        from batchgen.kv_cache.host_kv_mananger_config import (
            build_gpu_kv_config_fixed_size,
            is_dsa_model,
            _resolve_indexer_profile,
            _torch_dtype_from_string,
        )
        from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig

        # Calculate GPU KV cache size if not already done
        if self.gpu_kv_cache_size_gb is None:
            self.gpu_kv_cache_size_gb = self._calculate_gpu_kv_cache_size()

        if is_dsa_model(self.huggingface_ckpt_name):
            # Split memory budget between primary MLA cache and auxiliary indexer cache.
            # Compute the ratio of bytes-per-page for primary vs auxiliary so both
            # get the same number of pages (they share the same page table).
            from batchgen.kv_cache.host_kv_mananger_config import _resolve_profile
            primary_profile = _resolve_profile(self.huggingface_ckpt_name)
            aux_profile = _resolve_indexer_profile(self.huggingface_ckpt_name)

            primary_bytes_per_page = primary_profile.bytes_per_page() * primary_profile.num_layers
            aux_bytes_per_page = aux_profile.bytes_per_page() * aux_profile.num_layers
            combined_bytes_per_page = primary_bytes_per_page + aux_bytes_per_page

            total_bytes = int(self.gpu_kv_cache_size_gb * (1024 ** 3))
            num_pages = total_bytes // combined_bytes_per_page

            primary_config = GPUPagedKVConfig(
                num_layers=primary_profile.num_layers,
                num_pages=num_pages,
                page_size_tokens=primary_profile.page_size,
                num_k_heads=primary_profile.num_k_heads,
                k_head_dim=primary_profile.k_head_dim,
                num_v_heads=primary_profile.num_v_heads,
                v_head_dim=primary_profile.v_head_dim,
                kv_dtype=_torch_dtype_from_string(primary_profile.kv_dtype),
                # Engine-layer -> cache-slot map, exactly as the dense builder
                # passes it. A fully-DSA model (GLM) has None here (engine ==
                # physical); an interleaved model (Kimi-K3.1: 24 MLA rows among
                # 93 engine layers) addresses BOTH regions by engine layer id,
                # and without the map the primary pool bounds-checks engine ids
                # against its 24 physical slots and refuses layer 27 at the
                # first decode-graph capture.
                logical_to_physical_layer=(
                    list(primary_profile.logical_to_physical_layer)
                    if getattr(primary_profile, "logical_to_physical_layer", None)
                    is not None
                    else None
                ),
            )
            primary_config = self._with_cuda_graph_page_table_capacity(primary_config)
            aux_config = GPUPagedKVConfig(
                num_layers=aux_profile.num_layers,
                num_pages=num_pages,
                page_size_tokens=aux_profile.page_size,
                num_k_heads=aux_profile.num_k_heads,
                k_head_dim=aux_profile.k_head_dim,
                num_v_heads=aux_profile.num_v_heads,
                v_head_dim=aux_profile.v_head_dim,
                kv_dtype=_torch_dtype_from_string(aux_profile.kv_dtype),
                logical_to_physical_layer=(
                    list(aux_profile.logical_to_physical_layer)
                    if getattr(aux_profile, "logical_to_physical_layer", None) is not None
                    else None
                ),
            )
            aux_config = self._with_cuda_graph_page_table_capacity(aux_config)

            fixed_addresses = self._kv_fixed_address_pool_available()
            primary = GPUPagedKVCacheManager(
                config=primary_config,
                device=self.local_rank,
                fixed_address_pool=fixed_addresses,
            )
            primary.initialize()
            auxiliary = GPUPagedKVCacheManager(
                config=aux_config,
                device=self.local_rank,
                fixed_address_pool=fixed_addresses,
            )
            auxiliary.initialize()
            manager = DualKVCacheCoordinator(primary, auxiliary)
            self._bind_gpu_paged_kv_manager(manager)

            if self.rank == 0:
                primary_gb = (primary_bytes_per_page * num_pages) / (1024 ** 3)
                aux_gb = (aux_bytes_per_page * num_pages) / (1024 ** 3)
                logging.info(
                    f"[GPU-KV] DualKVCacheCoordinator initialized: "
                    f"{num_pages} pages, primary={primary_gb:.2f} GB (dim={primary_profile.k_head_dim}), "
                    f"auxiliary={aux_gb:.2f} GB (dim={aux_profile.k_head_dim})"
                )
            return manager
        else:
            config = build_gpu_kv_config_fixed_size(
                model_name=self.huggingface_ckpt_name,
                gpu_kv_cache_size_gb=self.gpu_kv_cache_size_gb,
            )
            config = self._with_cuda_graph_page_table_capacity(config)

            manager = GPUPagedKVCacheManager(
                config=config,
                device=self.local_rank,
                fixed_address_pool=self._kv_fixed_address_pool_available(),
            )
            manager.initialize()
            self._bind_gpu_paged_kv_manager(manager)

            if self.rank == 0:
                logging.info(
                    f"[GPU-KV] Initialized: {self.gpu_kv_cache_size_gb:.2f} GB, {config.num_pages} pages"
                )

            return manager

    def _bind_gpu_paged_kv_manager(self, worker, manager) -> None:
        """Bind GPU KV manager to both worker and core_engine.

        If manager is a DualKVCacheCoordinator, the primary manager is bound
        to existing gpu_paged_kv_manager slots and the auxiliary (indexer) is
        bound to gpu_paged_kv_manager_aux slots.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        self.gpu_paged_kv_cache_manager = manager
        self._gpu_kv_pool_created = True
        if isinstance(manager, DualKVCacheCoordinator):
            if hasattr(self.core_engine, "gpu_paged_kv_manager"):
                self.core_engine.gpu_paged_kv_manager = manager.primary
            if hasattr(self.core_engine, "gpu_paged_kv_manager_aux"):
                self.core_engine.gpu_paged_kv_manager_aux = manager.auxiliary
        else:
            if hasattr(self.core_engine, "gpu_paged_kv_manager"):
                self.core_engine.gpu_paged_kv_manager = manager

    def _deployment_context_bound_tokens(self, worker) -> int:
        """Longest context any sequence can reach for the life of this server.

        Admission clamps every sequence to `model_context_length`
        (`clamp_decode_to_context`, and the `ctx_lens >= model_context_length`
        completion test), so no decode step's `max_seqlen` can ever exceed it.
        That makes it the one DEPLOYMENT bound available to capture-time
        sizing: `max_input_length` / `max_decoding_length` are per-batch — set
        by `Init` and grown at admission — so sizing from them describes
        whichever batch happens to arrive first, not the deployment.

        `_initialize_core_components` sets it from the model config (131072
        default) before any KV pool or graph exists, so it is always available
        to the pool and capture paths. Returns 0 only before core init; the
        caller decides whether that is an error.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return int(getattr(self, "model_context_length", 0) or 0)

    def _make_page_table_capacity_request(self, worker) -> PageTableCapacityRequest:
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        engine_config = getattr(self, "engine_config", None)
        engine_module_global_batch_size: Optional[int] = None
        engine_module_attn_decoding_micro_batch_size: Optional[int] = None
        engine_basic_num_queries: Optional[int] = None
        engine_basic_decode_graph_max_bucket: Optional[int] = None
        if engine_config is not None:
            basic = engine_config.Basic_Config
            module_batching = engine_config.Module_Batching_Config
            engine_module_global_batch_size = module_batching.global_batch_size
            engine_module_attn_decoding_micro_batch_size = module_batching.attn_decoding_micro_batch_size
            engine_basic_num_queries = basic.num_queries
            # A whole-model decode graph addresses one page-table row per
            # sequence of its bucket; the planner's top bucket (K3 H200: the
            # KDA slot count, 160 > the 128 arg default) must fit.
            graph_buckets = getattr(basic, "decode_graph_buckets", None) or ()
            engine_basic_decode_graph_max_bucket = max(graph_buckets) if graph_buckets else None
        model_config = getattr(self, "model_config", None)
        # Deployment bound first, model config second. `model_context_length`
        # is what admission clamps to, and it exists even when the config class
        # does not declare `max_position_embeddings` — a config that lacks the
        # field used to drop the bound from the candidate set entirely, leaving
        # the page table's token span at the first batch's
        # `max_input_length + max_decoding_length`. A graph captured against
        # that span then rejected every later, longer decode step.
        model_max_position_embeddings = self._deployment_context_bound_tokens() or getattr(
            model_config, "max_position_embeddings", None
        )
        args = getattr(self, "args", None)
        args_cuda_graph_max_bucket_size = getattr(args, "cuda_graph_max_bucket_size", None) if args is not None else None
        return PageTableCapacityRequest(
            engine_module_global_batch_size=engine_module_global_batch_size,
            engine_module_attn_decoding_micro_batch_size=engine_module_attn_decoding_micro_batch_size,
            engine_basic_num_queries=engine_basic_num_queries,
            model_max_position_embeddings=model_max_position_embeddings,
            args_cuda_graph_max_bucket_size=args_cuda_graph_max_bucket_size,
            engine_basic_decode_graph_max_bucket=engine_basic_decode_graph_max_bucket,
        )

    def _with_cuda_graph_page_table_capacity(self, worker, config):
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return KVCacheManager.apply_page_table_capacity(
            self._make_page_table_capacity_request(), config
        )

    def _init_gpu_kv_with_actual_size(self, worker) -> None:
        """
        Calculate actual GPU KV size AFTER model loading and initialize the manager.
        This replaces the theoretical estimation - must be called after _load_decode_model().

        Only runs the full calculation and initialization on the first call;
        subsequent calls skip if the manager is already initialized.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        # Skip if GPU KV manager is already initialized (subsequent decode iterations)
        if self.gpu_paged_kv_cache_manager is not None and self.gpu_paged_kv_cache_manager.is_initialized:
            return

        # The pool is sized from what is free once the decode model is in
        # place, so sizing it while the prefill phase is still resident
        # permanently charges that phase's bytes against decode capacity. On
        # the reuse lifecycle the size is then kept for the whole process,
        # which is exactly how an earlier startup-time sizing attempt ended up
        # with roughly half the pages the per-batch path got. Sizing therefore
        # only happens from the decode phase, after the prefill ring and the
        # prefill model have been released.
        if getattr(self, "_phase", None) != "decode":
            raise RuntimeError(
                f"Rank {self.rank}: GPU KV pool sizing attempted in phase "
                f"{self._phase!r}; it must run from the decode phase so the "
                "measurement cannot charge prefill-only residency against the "
                "pool it then keeps"
            )

        # A reserved-range pool is never re-created: its virtual range is still
        # held and its page count was fixed by the sizing below on the first
        # decode configure, so a later decode phase maps the same pages back
        # instead of sizing a second, different pool. The reuse lifecycle has
        # already remapped inside its activation; this covers a rank that
        # rebuilds its decode instance, and keeps every rank's num_total_pages
        # identical, which the cross-rank capacity snapshot requires.
        manager = self.gpu_paged_kv_cache_manager
        if manager is not None and getattr(manager, "uses_fixed_address_pool", False):
            self._remap_reused_kv_pool()
            return

        # Measure-once-cache: once startup decode residency has been
        # measured, every later pool (re)creation reuses the cached size —
        # never a fresh measurement (contract invariant 1: the pool is
        # identical across decode rounds of the same server).
        if self._gpu_kv_cached_pool_gb is not None:
            self.gpu_kv_cache_size_gb = self._gpu_kv_cached_pool_gb
            # The previous pool's segments already returned to CUDA at destroy
            # (dedicated MemPool, whole-segment free). This flushes the
            # GENERAL cache's all-free segments (prefill transients) too, so
            # the full-size cudaMalloc below sees the maximum free memory. The
            # collection first drops tensors still held only by cycles, so
            # empty_cache can return their segments.
            torch.cuda.synchronize(self.torch_device)
            gc.collect()
            torch.cuda.empty_cache()
            # Later residency may consume the fraction's reserved slack, but
            # the verified pool itself must still fit. Name any shortfall
            # instead of silently shrinking the deployment's capacity.
            free_mem_gb = torch.cuda.mem_get_info(self.local_rank)[0] / (1024 ** 3)
            KVCacheManager.check_cached_pool_fits(
                self.gpu_kv_cache_size_gb, free_mem_gb, rank=self.rank
            )
            if self.rank == 0:
                logging.info(
                    f"[GPU-KV] Cached pool size: {self.gpu_kv_cache_size_gb:.2f} GB "
                    f"(startup budget; {free_mem_gb:.2f} GB free)"
                )
            self._initialize_gpu_kv_manager_fixed_size()
            if self.rank == 0:
                stats = self.gpu_paged_kv_cache_manager.get_stats()
                logging.info(
                    f"[GPU-KV] Initialized: {self.gpu_kv_cache_size_gb:.2f} GB, "
                    f"{stats.num_total_pages} pages"
                )
            return

        # First startup decode entry: measure after weights are resident. The
        # remaining fraction is reserved for graph and eager allocations; it
        # is not counted as KV capacity. Every later entry uses the cache.
        torch.cuda.synchronize(self.torch_device)
        torch.cuda.empty_cache()

        free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(self.local_rank)
        free_mem_gb = free_mem_bytes / (1024 ** 3)
        total_mem_gb = total_mem_bytes / (1024 ** 3)
        used_mem_gb = total_mem_gb - free_mem_gb

        local_pool_gb = provisional_pool_gb(
            total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=None
        )
        self._verify_deployment_max_model_len(local_pool_gb)
        # Identical page counts are required by decode admission. The tightest
        # rank supplies the common pool, rather than rank zero's residency.
        size_tensor = torch.tensor([local_pool_gb], dtype=torch.float64, device=self.torch_device)
        dist.all_reduce(size_tensor, op=dist.ReduceOp.MIN)
        self.gpu_kv_cache_size_gb = float(size_tensor.item())
        # Freeze the measured startup pool for every mode before capture. A
        # lazy capture must not change it after L has been published. The
        # fraction's slack covers graph/eager allocations until startup can
        # measure those allocations at the per-rank cap.
        self._gpu_kv_sizing_provisional = False

        # Initialize GPU KV manager with actual size
        self._initialize_gpu_kv_manager_fixed_size()
        self._gpu_kv_cached_pool_gb = self.gpu_kv_cache_size_gb

        if self.rank == 0:
            stats = self.gpu_paged_kv_cache_manager.get_stats()
            logging.info(f"[GPU-KV] Initialized: {self.gpu_kv_cache_size_gb:.2f} GB, {stats.num_total_pages} pages")

    def _finalize_gpu_kv_pool_after_capture(self, worker, gpu_manager) -> None:
        """Measure-once-cache finalization (semantic contract, POIS 2026-10-05).

        Called once, right after the full graph warmup/capture, while the
        PROVISIONAL pool is still allocated. The capture pass has materialized
        every persistent allocation (graph pools, static buffers, NCCL
        per-collective resources, JIT caches), so the steady residency is
        ``used - provisional_pool`` and the final pool is everything the memory
        fraction allows above it. The result is CACHED; every later pool
        (re)creation reuses it, making the pool identical across decode rounds
        (invariant 1) and retiring the fresh-server round-1 protocol.

        The size is the device total minus that steady residency minus the
        explicit ``GPU_KV_POOL_HEADROOM_GB``. ``gpu_memory_frac`` is NOT
        applied again here: its slack is what covers the graph footprint the
        PRE-capture measurement cannot see, and once that footprint is inside
        the measurement, re-applying the fraction charges it twice and parks
        ``(1 - frac) * total`` permanently free with no named consumer. The
        headroom constant is that consumer, measured and auditable.

        The current (first) round keeps the provisional pool: its sequences
        already hold pages, the captured graphs bake its addresses, and this
        codebase deliberately refuses to recapture after a configured capture
        (``_setup_cuda_graphs`` falls back to eager instead, and the reuse
        lifecycle raises), so the live pool cannot be resized under them. With
        the headroom sizing the provisional is the SMALLER pool (it still
        carries the unmeasured-graph slack), so round 1 is a conservative
        startup artifact, never an over-commitment. The cached size applies
        from the NEXT pool creation (the per-round destroy/recreate that
        already exists), from which point the page count is identical for
        every round.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if not self._gpu_kv_sizing_provisional:
            return
        self._gpu_kv_sizing_provisional = False

        torch.cuda.synchronize(self.torch_device)
        torch.cuda.empty_cache()
        free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(self.local_rank)
        total_mem_gb = total_mem_bytes / (1024 ** 3)
        used_mem_gb = total_mem_gb - free_mem_bytes / (1024 ** 3)
        provisional_gb = self.gpu_kv_cache_size_gb
        steady_used_gb = used_mem_gb - provisional_gb

        final_gb = KVCacheManager.compute_final_pool_gb(
            total_mem_gb,
            used_mem_gb,
            provisional_gb,
        )

        # Broadcast rank 0's decision so every rank resizes identically.
        size_tensor = torch.tensor([final_gb], dtype=torch.float32, device=self.torch_device)
        dist.broadcast(size_tensor, src=0)
        final_gb = float(size_tensor.item())

        # Fail-loud floor (contract invariant 3).
        manager = self.gpu_paged_kv_cache_manager
        stats = manager.get_stats()
        bytes_per_page = (
            provisional_gb * (1024 ** 3) / max(stats.num_total_pages, 1)
        )
        final_pages = KVCacheManager.pool_pages_for_gb(final_gb, bytes_per_page)
        bucket_sizes = list(getattr(self._graph.whole_model_bucketing, "bucket_sizes", []) or [1])
        floor_pages = KVCacheManager.pool_floor_pages(max(bucket_sizes))
        if final_gb <= 0 or final_pages < floor_pages:
            raise RuntimeError(
                f"[GPU-KV] Final pool below floor: {final_gb:.2f} GB "
                f"(~{final_pages} pages) < {floor_pages} pages required for the "
                f"largest decode bucket. Steady residency {steady_used_gb:.2f} GB "
                f"of {total_mem_gb:.2f} GB leaves nothing above the "
                f"{GPU_KV_POOL_HEADROOM_GB:.2f} GB pool headroom; fix the "
                "configuration."
            )

        # Cache only: the live round keeps the provisional pool (its sequences
        # hold pages and the captured graphs bake its addresses); every later
        # pool creation uses the cached final size.
        self._gpu_kv_cached_pool_gb = final_gb
        if self.rank == 0:
            logging.info(
                f"[GPU-KV] Finalized pool size: {final_gb:.2f} GB "
                f"(~{final_pages} pages; steady used {steady_used_gb:.2f} GB of "
                f"{total_mem_gb:.2f} GB; headroom {GPU_KV_POOL_HEADROOM_GB:.2f} GB; "
                f"provisional {provisional_gb:.2f} GB stays for this round, delta "
                f"{final_gb - provisional_gb:+.2f} GB; cached for every later pool "
                "creation)"
            )
            logging.info(
                "[GPU-KV] Pool margin is now the explicit %.2f GB headroom; "
                "gpu_memory_frac=%.3f bounds only the pre-capture provisional "
                "measurement (the frac-based final size would have been %.2f GB)",
                GPU_KV_POOL_HEADROOM_GB,
                self.gpu_memory_frac,
                provisional_pool_gb(
                    total_mem_gb, self.gpu_memory_frac, steady_used_gb, floor_gb=None
                ),
            )

    # ---- GPU KV pool handle, remap and region asserts ----------------------

    def _get_cuda_graph_gpu_manager(self, worker):
        """Return the GPU KV manager object to use for CUDA graph setup."""
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if isinstance(manager, DualKVCacheCoordinator):
            return manager
        if manager is not None:
            return manager
        return getattr(self.core_engine, "gpu_paged_kv_manager", None)

    def _release_suspended_decode_kv_pool(self, worker) -> None:
        """Return the suspended decode KV pool's physical pages, keep its addresses.

        The decode graphs baked the pool bases, page-pointer tables and
        page-table storage, so a resumable decode instance must find them at
        the same virtual addresses; only the physical memory goes back to the
        driver for the prefill phase, and ``_remap_reused_kv_pool`` maps it
        again on resume.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        self._assert_kv_pool_addresses_are_fixed("decode->prefill")
        manager = self.gpu_paged_kv_cache_manager
        pages = manager.get_stats().num_total_pages if manager is not None else 0
        start = time.perf_counter()
        self._destroy_gpu_paged_kv_cache()
        if manager is not None:
            logging.info(
                "[GPU-KV] rank=%d released the reserved KV pool's physical "
                "pages in %.3fs (%d pages, %.2f GB)",
                self.rank,
                time.perf_counter() - start,
                pages,
                self.gpu_kv_cache_size_gb,
            )

    def _remap_reused_kv_pool(self, worker) -> None:
        """P->D on the default lifecycle: map the KV pool back where it was.

        ``initialize()`` on a reserved-range pool maps fresh physical memory
        under the SAME virtual addresses, zeroes it the way a freshly built
        pool hands out zeroed pages, and refreshes the page-pointer tables in
        place. Nothing is allocated, nothing is sized: the page count is the
        one the first decode configure measured.

        Called from inside the activation so the pool is live before the
        capacity snapshot, before decode admission allocates a page and before
        a captured graph dereferences a pool pointer.

        A decode instance can be built before readiness (Kimi-K3's startup
        pass), so its first activation precedes the first pool sizing. With no
        pool ever created in this process there is nothing to remap, and
        ``_init_gpu_kv_with_actual_size()``, which the caller runs next from
        the decode phase, sizes it. A pool that existed and is gone is a
        lifecycle bug and still fails here.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if manager is None:
            if not self._gpu_kv_pool_created:
                return
            raise RuntimeError(
                f"Rank {self.rank}: prefill->decode activated a reused decode "
                "instance with no GPU KV manager although this process already "
                "created a GPU KV pool; the reserved pool must survive every "
                "phase switch"
            )
        self._assert_kv_pool_addresses_are_fixed("prefill->decode")
        if manager.pool_is_mapped:
            # decode -> decode returns before the activation, so this is only
            # reached if a future caller activates an already-active instance.
            return
        start = time.perf_counter()
        manager.initialize()
        stats = manager.get_stats()
        logging.info(
            "[GPU-KV] rank=%d remapped the reserved KV pool in %.3fs (%d pages)",
            self.rank,
            time.perf_counter() - start,
            stats.num_total_pages,
        )

    def _assert_kv_pool_addresses_are_fixed(self, worker, context: str) -> None:
        """A reused decode instance requires an address-stable KV pool.

        Its captured graphs dereference the pool bases and page-pointer tables
        recorded at capture time. Releasing an allocator-backed pool under them
        would hand those addresses to the next allocation and the following
        replay would read someone else's memory, so an unsupported pool fails
        the switch instead of silently taking the old destroy/recreate path.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if manager is None:
            return
        if not getattr(manager, "uses_fixed_address_pool", False):
            raise RuntimeError(
                f"Rank {self.rank}: {context} switch would release a GPU KV "
                "pool that is not held at fixed addresses while the decode "
                "instance is reused; its captured graphs bake the pool "
                "pointers, so the pool must be a reserved-range pool"
            )

    def _assert_kv_region_mapped(self, worker, context: str) -> None:
        """Before decode configure and replay the KV pool is mapped.

        Decode admission reads live all-rank capacity through it (#424) and the
        decode graphs bake its pointers, so the activation maps the pool back
        before either can run.

        The one exception is a process that has never created a pool: a decode
        instance built before readiness is activated before the first sizing,
        which ``_init_gpu_kv_with_actual_size()`` performs right after this
        check, before admission or capture can read it.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if manager is None and not self._gpu_kv_pool_created:
            return
        if manager is None or not manager.is_initialized:
            raise RuntimeError(
                f"Rank {self.rank}: {context} switch left the GPU KV region "
                "unmapped while phase instances are reused; decode admission "
                "reads live capacity through it and the decode graphs bake its "
                "pointers"
            )
        if not manager.pool_is_mapped:
            raise RuntimeError(
                f"Rank {self.rank}: {context} switch left the GPU KV pool's "
                "physical pages unmapped at their reserved addresses; a decode "
                "replay or capacity read would touch unmapped device memory"
            )

    def _assert_kv_region_released(self, worker, context: str) -> None:
        """After decode->prefill the KV pool's physical pages are back.

        The addresses stay reserved — that is what lets the activation map the
        same range again — but the pages themselves belong to the prefill phase
        until then. A pool still mapped here is the cleanup bug this lifecycle
        exists to avoid: the prefill phase would run with the whole decode KV
        pool charged against it.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if manager is None:
            # Never created: a lifetime whose decode phase has not sized a pool
            # yet has nothing to release.
            return
        if manager.pool_is_mapped:
            raise RuntimeError(
                f"Rank {self.rank}: {context} switch left the GPU KV pool "
                "mapped; prefill needs its physical pages back, and the "
                "reserved addresses are what make the remap free"
            )

    # ---- Node memfd identity -----------------------------------------------

    def _exchange_node_memfd_identity(self, worker, pid: int, fd: int) -> Tuple[int, int]:
        """Publish this rank's ``input_ids`` memfd identity, take its node's.

        COLLECTIVE, one per pool generation, issued identically by every rank on
        every path. It REPLACES the post-create barrier ``allocate_node_shared_int64``
        used to run: an all_gather cannot deliver on any rank before every rank
        has entered, so the creator's memfd is guaranteed to exist — and to be
        sized — by the time an attacher reads its pid and fd. Non-creators
        contribute ``fd=-1``, which is never selected.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        # Pool allocation can run inside InferenceMode (see the buffer creation
        # in allocate_node_shared_int64); the collective writes into `gathered`.
        with torch.inference_mode(False):
            identity = torch.tensor(
                [pid, fd], dtype=torch.int64, device=self.torch_device
            )
            gathered = [torch.zeros_like(identity) for _ in range(self.world_size)]
            dist.all_gather(gathered, identity)
            # Pre-`rank_ctx`: this collective runs inside core initialization.
            creator_rank = (
                node_of_rank(self.rank, self.local_world_size) * self.local_world_size
            )
            creator = gathered[creator_rank]
            return int(creator[0].item()), int(creator[1].item())

    # ---- KV stats ----------------------------------------------------------

    def _make_kv_cache_manager(self, worker) -> KVCacheManager:
        """Lazy-construct: KV managers may not be bound at __init__ time."""
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self._kv_cache_manager is None:
            worker = self
            class _TorchKVStatsBackend:
                def get_host_stats(self) -> KVStats:
                    s = worker.host_paged_kv_worker_view.get_stats()
                    return KVStats(
                        num_free_pages=s.num_free_pages,
                        num_used_pages=s.num_used_pages,
                        num_total_pages=s.num_total_pages,
                    )
                def get_gpu_stats(self) -> "Optional[KVStats]":
                    m = worker.gpu_paged_kv_cache_manager
                    if m is None:
                        return None
                    s = m.get_stats()
                    return KVStats(
                        num_free_pages=s.num_free_pages,
                        num_used_pages=s.num_used_pages,
                        num_total_pages=s.num_total_pages,
                    )
            self._kv_cache_manager = KVCacheManager(backend=_TorchKVStatsBackend())
        return self._kv_cache_manager

    def _make_kv_utilization_request(self, worker) -> KVUtilizationRequest:
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return KVUtilizationRequest(
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            num_gpus_per_node=self.local_world_size,
            global_batch=self.global_batch,
        )

    def _get_host_kv_free_pages(self, worker) -> int:
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self._kv_stats_compare:
            legacy = self._legacy_get_host_kv_free_pages()
            native = self._make_kv_cache_manager().get_host_free_pages()
            assert legacy == native, f"kv_stats compare mismatch: get_host_kv_free_pages legacy={legacy} native={native}"
            return native if self._kv_stats_native else legacy
        if self._kv_stats_native:
            return self._make_kv_cache_manager().get_host_free_pages()
        return self._legacy_get_host_kv_free_pages()

    def _legacy_get_host_kv_free_pages(self, worker) -> int:
        """Get current free pages from host KV cache."""
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        stats = self.host_paged_kv_worker_view.get_stats()
        return stats.num_free_pages

    def _get_host_kv_utilization(self, worker) -> Dict[str, int]:
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if self._kv_stats_compare:
            legacy = self._legacy_get_host_kv_utilization()
            native_obj = self._make_kv_cache_manager().get_host_utilization(self._make_kv_utilization_request())
            native = _dataclasses.asdict(native_obj)
            assert legacy == native, f"kv_stats compare mismatch: get_host_kv_utilization legacy={legacy} native={native}"
            return native if self._kv_stats_native else legacy
        if self._kv_stats_native:
            native_obj = self._make_kv_cache_manager().get_host_utilization(self._make_kv_utilization_request())
            return _dataclasses.asdict(native_obj)
        return self._legacy_get_host_kv_utilization()

    def _gather_host_kv_stats_by_node(self, worker, worker_view: Optional[object]) -> List[Dict[str, int]]:
        """Gather one host-KV pool stat record per node.

        Host KV is shared by ranks on the same node, not globally. Rank 0 uses
        these per-node stats to plan dynamic host growth against the same pool
        that each owner rank will later allocate from.

        Each record also carries the retained prefix pages the node can reclaim
        right now (resident, no live attachment, no pending load): growth may
        consume them exactly as prefill admission does, and the boundary's
        growth phase releases them on demand before the fresh capacity
        preflight. Zero when the prefix cache is off.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        num_nodes = self.rank_ctx.num_nodes
        report_free = 0
        report_total = 0
        report_reclaimable = -1
        report_node = -1
        if worker_view is not None and self.local_rank == 0:
            try:
                stats = worker_view.get_stats()
                report_node = self.rank_ctx.node_of(self.rank)
                report_free = int(stats.num_free_pages)
                report_total = int(stats.num_total_pages)
                report_reclaimable = (
                    max(0, int(self.prefix_cache_coordinator.count_releasable_pages(0)))
                    if (
                        self.enable_prefix_cache
                        and self.prefix_cache_coordinator is not None
                    )
                    else 0
                )
            except Exception:
                # Carry a rank-local stats failure through the collective so
                # peers do not block while this rank raises before all-gather.
                logging.exception("Rank %s: host-KV stats snapshot failed", self.rank)
                report_node = -2
                report_free = -1
                report_total = -1
                report_reclaimable = -1

        stats_tensor = torch.tensor(
            [report_node, report_free, report_total, report_reclaimable],
            dtype=torch.int64,
            device=self.torch_device,
        )
        gathered = [torch.zeros_like(stats_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, stats_tensor)
        if any(int(item[0].item()) == -2 for item in gathered):
            raise RuntimeError(
                "[SCHED_INVARIANT] host-KV stats snapshot failed on one or more ranks"
            )

        per_node_stats = []
        reports_by_node = {}
        for item in gathered:
            node_id = int(item[0].item())
            if node_id >= 0:
                reports_by_node[node_id] = {
                    'node_id': node_id,
                    'num_free_pages': int(item[1].item()),
                    'num_total_pages': int(item[2].item()),
                    'num_reclaimable_pages': int(item[3].item()),
                }

        for node in range(num_nodes):
            per_node_stats.append(reports_by_node.get(node, {
                'node_id': node,
                'num_free_pages': 0,
                'num_total_pages': 0,
                'num_reclaimable_pages': 0,
            }))

        return per_node_stats

    def _get_host_kv_total_pages(self, worker) -> int:
        """This node's shared Host-KV pool size in pages."""
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return int(self.host_paged_kv_worker_view.get_stats().num_total_pages)

    def _log_host_kv_health(self, worker) -> None:
        """Log the audited health of this node's host-KV region(s).

        One line per region: the used-page identity, the sequence-table
        occupancy, and the region's lifetime counters (see
        ``batchgen.kv_cache.host_kv_health``). ``get_audited_stats()`` is
        O(table + pages) under both region locks; call this only from the
        periodic heartbeat.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        view = getattr(self, "host_paged_kv_worker_view", None)
        if view is None:
            return
        if isinstance(view, DualHostKVCoordinator):
            regions = (
                ("primary", view.primary),
                ("aux", view.require_auxiliary("host-KV health")),
            )
        else:
            regions = (("host", view),)
        node_id = self.rank_ctx.node_id
        for name, region_view in regions:
            logging.info(
                f"[HOST_KV] node={node_id} "
                f"{format_host_kv_health(name, region_view.get_audited_stats())}"
            )

    # ---- Decode cap and host views -----------------------------------------

    def _max_decode_rank_bsz(self, worker) -> int:
        """Per-rank in-decode cap: the generic guard, lifted to the planner's
        per-node cap when a TP-group model (every rank holds the same rows)
        plans more slots than the guard — e.g. K3 H200 with sharded latent
        projections plans 160 rows per node.

        The legacy BATCHGEN_MAX_RANK_BSZ env name sizes the deepseek PSM
        padding separately; silently diverging values corrupt the padded
        buffer contract, so both names must agree when both are set."""
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        explicit = os.environ.get("BATCHGEN_MAX_DECODE_RANK_BSZ")
        legacy = os.environ.get("BATCHGEN_MAX_RANK_BSZ")
        if explicit is not None and legacy is not None and explicit != legacy:
            raise RuntimeError(
                "BATCHGEN_MAX_DECODE_RANK_BSZ and BATCHGEN_MAX_RANK_BSZ "
                "must match when both are set"
            )
        planned = int(getattr(
            self.engine_config.Module_Batching_Config, "MoE_decoding_micro_batch_size", 0,
        ) or 0)
        if self._decode_attn_tp_size() > 1 and planned > _MAX_DECODE_RANK_BSZ:
            return planned
        return _MAX_DECODE_RANK_BSZ

    def _host_views_by_group(self, worker) -> dict:
        """One Host-KV worker view per prefix-cache KV group.

        A DSA model keeps group 0 (MLA latent) in the primary region and group 1
        (indexer keys) in the auxiliary one, so every prefix-cache Host page
        operation has to run on both. Every other model has one region.
        """
        # Moved VERBATIM from batchgen_worker.py (S10a); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if isinstance(self.host_paged_kv_worker_view, DualHostKVCoordinator):
            views = self.host_paged_kv_worker_view.views_by_group()
        else:
            view = getattr(self, "host_paged_kv_worker_view", None)
            if view is None:
                view = self.core_engine.host_paged_kv_worker_view
            views = {0: view}
        runtime = self.prefix_cache_runtime_config
        if runtime is not None and len(runtime.group_specs) != len(views):
            raise RuntimeError(
                "prefix cache groups do not match the Host KV regions: "
                f"groups={len(runtime.group_specs)}, views={len(views)}"
            )
        return views
