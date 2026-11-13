"""KV capacity and layout behind a worker seam (S10a, KVCapacity).

``KVCapacityEngine`` owns the worker rank's KV capacity and layout surface:
GPU KV pool sizing (the fixed-address pool, the page-table capacity request,
actual-size init and the post-capture finalize), the GPU KV pool handle,
remap and region asserts, the node memfd identity exchange, the KV stats read
through ``KVCacheManager``, the host-KV health line, the per-rank decode cap
and the per-group host views.

S10b added the host-KV watermark request and reclaimable-prefix credit, and
the host-capacity due checks and collective transaction helpers. S10c added
GPU KV page allocation and release (with host-identity page dedup), the
decode KV append to host, the host-to-GPU KV load, the decode capacity
snapshot and the page-boundary decision request. Parked on the worker pending
the layer-map ruling: _page_boundary_fast, _finalize_async_load_minimal,
_put_sequences_on_hold, _try_load_new_sequences, _compute_boundary_decisions
(and the S10a/S10b parked methods). The frozen S5
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
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from batchgen.kv_cache.dual_host_kv_coordinator import DualAsyncKVTask, DualHostKVCoordinator
from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator, DualPoolPageKeys
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.kv_cache.host_kv_health import format_host_kv_health
from batchgen.native_loader import core_engine
from batchgen.worker.boundary import BoundaryDecisionRequest, BoundarySeqMeta
from batchgen.worker.decode import DecodeCapacitySnapshot, reduce_decode_capacity_snapshot
from batchgen.worker.decode_engine import BATCHGEN_CB_DEBUG
from batchgen.worker.kv_manager import (
    GPU_KV_POOL_HEADROOM_GB,
    KVCacheManager,
    KVStats,
    KVUtilizationRequest,
    PageTableCapacityRequest,
    WatermarkTriggerRequest,
)
from batchgen.worker.kv_plan import (
    NodeCapacityReport,
    credit_reclaimable_pages,
    host_capacity_due,
    prefill_owner_demand_pages,
    provisional_pool_gb,
    verdict_node_capacity,
)
from batchgen.worker.prefill import prefill_host_reserve_page_deficit
from batchgen.worker.rank_context import node_of_rank


# Max per-rank in-decode batch. Bounds the K2.5 MoE padded buffer (mtp = round_up(world_size *
# this)) so init does not OOM on a large candidate pool, and is the per-rank admission cap so
# the pre-reserved buffer never overflows at runtime. Raise to fill more GPU KV (memory permitting).
_MAX_DECODE_RANK_BSZ = int(os.environ.get("BATCHGEN_MAX_DECODE_RANK_BSZ", "128"))

# GPU KV cache size override (DEPRECATED: use --gpu-memory-frac CLI arg instead)
# If set, overrides the automatic calculation. Otherwise, size is computed as:
# gpu_kv_cache = GPU_mem * gpu_memory_frac - model_instance_size
_GPU_KV_CACHE_SIZE_OVERRIDE = os.environ.get("BATCHGEN_GPU_KV_CACHE_SIZE_GB")

# Optional gate for expensive/critical diagnostics (default off in production)
BATCHGEN_ENABLE_CRITICAL_DIAGS = os.environ.get('BATCHGEN_ENABLE_CRITICAL_DIAGS', '0') == '1'


def _normalize_aux_kv_entry(entry) -> tuple:
    """(layer_idx, k_tensor, v_tensor, scale_tensor) of a deferred aux append.

    A page-split FP8 aux region needs the fp32 per-token scales alongside the
    quantized keys, so a producer may append a 4-tuple; a bf16 AoS producer
    keeps appending a 3-tuple and gets ``scale_tensor=None``.
    """

    if len(entry) == 4:
        return tuple(entry)
    if len(entry) == 3:
        layer_idx, k_tensor, v_tensor = entry
        return layer_idx, k_tensor, v_tensor, None
    raise RuntimeError(
        "deferred aux KV entry must be (layer_idx, k_tensor, v_tensor) or "
        f"(layer_idx, k_tensor, v_tensor, scale_tensor); got {len(entry)} fields"
    )


@dataclass
class _DualKVLoadPointers:
    sequence_tensor: torch.Tensor
    primary_k_ptrs: torch.Tensor
    primary_v_ptrs: Optional[torch.Tensor]
    primary_page_counts: torch.Tensor
    aux_k_ptrs: torch.Tensor
    aux_v_ptrs: Optional[torch.Tensor]
    aux_page_counts: torch.Tensor


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

    # ---- Host-KV watermark -------------------------------------------------

    def _make_watermark_trigger_request(
        self, worker, node_stats: List[dict]
    ) -> WatermarkTriggerRequest:
        """Snapshot the worker state `plan_watermark_trigger` consumes."""
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        return WatermarkTriggerRequest(
            node_stats=tuple(node_stats),
            host_kv_watermark=self.host_kv_watermark,
            has_queued=self.global_batch.has_queueing(),
            has_evicted=self.enable_host_kv_eviction and self.global_batch.has_evicted(),
        )

    def _credit_reclaimable_prefix_pages(
        self, worker, stats: Dict[str, int]
    ) -> Dict[str, int]:
        """Report reclaimable retained prefix pages as free, not as pressure.

        The Host allocator's counters are already PHYSICAL: a prefix several
        sequences attach is one allocation and is charged to ``used`` exactly
        once, so this node's utilisation never double-counts a shared page.
        What it cannot distinguish is a page holding live KV from a page
        holding a RETAINED prefix that nobody is reading -- a copy the prefix
        cache keeps for a future hit and that admission reclaims on demand,
        which is why ``per_node_host_reclaimable`` already counts it as part of
        the admission budget.

        Left as ``used``, those pages hold free% under the watermark forever,
        so a pool filled with retained prefixes never preempts decode for
        prefill -- the one situation the trigger exists for. Crediting them
        here (and only here, so the legacy/native utilisation parity check
        keeps comparing the raw counters) makes the trigger see the capacity
        admission would actually get.

        Read only when work is waiting: the trigger cannot fire otherwise, and
        the count is an O(max_nodes) scan under the coordinator's lock.
        """
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if not self.enable_prefix_cache:
            return stats
        coordinator = getattr(self, "prefix_cache_coordinator", None)
        if coordinator is None:
            return stats
        waiting = self.global_batch.has_queueing() or (
            self.enable_host_kv_eviction and self.global_batch.has_evicted()
        )
        if not waiting:
            return stats
        return credit_reclaimable_pages(
            stats, int(coordinator.count_releasable_pages(0))
        )

    # ---- Host capacity -----------------------------------------------------

    def _host_capacity_due_local(self, worker, batch: List[int]) -> bool:
        """Whether a local decode row needs planning before its next forward."""
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        for local_idx in batch:
            uuid = self._local_to_uuid_map.get(local_idx)
            if uuid is None:
                continue
            seq = self.global_batch.get_sequence(uuid)
            if seq is None or self._is_sequence_completed(seq):
                continue
            if host_capacity_due(
                int(seq.host_token_capacity), int(seq.current_context_length)
            ):
                return True
        return False

    def _host_capacity_due_all_ranks(self, worker, batch: List[int]) -> bool:
        """Collectively detect a re-entry row with no next-token runway.

        Metadata is owner-local during decode, so a local check cannot decide
        whether all ranks must enter the collective boundary planner.  This check
        runs only at decode entry and immediately after a boundary, never once per
        decode token.
        """
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        local_due = self._host_capacity_due_local(batch)
        if self.world_size <= 1:
            return local_due
        flag = torch.tensor(
            [1 if local_due else 0], dtype=torch.int32, device=self.torch_device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(int(flag.item()))

    def _collective_prefill_host_capacity_ok(
        self, worker,
        owner_requests: List[Tuple[str, int]],
        local_error: bool = False,
    ) -> Tuple[bool, Optional[Dict[int, Tuple[int, int]]]]:
        """Check a fresh per-node host-KV snapshot before any mutation.

        Every rank reports its owner demand.  Only the node leader reports the
        shared allocator's free/total pages; summing demands across ranks and
        comparing against that one live snapshot prevents one rank from making a
        decision from stale local state.  The final MIN reduction makes the
        decision explicit even when a rank's stats call fails.

        Returns ``(ok, rows)``, where ``rows`` maps node id to the gathered
        ``(need, free)`` pair the reserve-time reclaim computes its deficit
        from. ``rows`` is ``None`` when the snapshot itself is unusable (a
        rank's stats call failed, a node reported no capacity at all, or two
        ranks of one node disagreed): a deficit derived from it would be
        meaningless, so the caller must not retry on it. Both verdicts are
        reduced, hence rank-uniform -- the retry branch is collective.
        """
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        # Members' demands exclude their shared chains, but the pooled
        # segments themselves allocate one Host copy (mirrored per group, so
        # one count). Leaving them out over-admits the wave and the
        # allocation fails right after the preflight passed. Only the wave's
        # host-KV LEADER allocates segments, so only a rank with owner demand
        # reports them — at G>1 the followers reporting the same segments
        # would multiply the node demand by G and falsely reject the wave.
        local_need = prefill_owner_demand_pages(
            [int(tokens) for _, tokens in owner_requests],
            page_size=self.PAGE_SIZE,
            pool_segment_pages=(
                self._prefill_pool_segment_host_pages() if owner_requests else 0
            ),
        )
        node_id = self.rank_ctx.node_id
        local_free = -1
        local_total = -1
        snapshot_error = 1 if local_error else 0
        host_view = getattr(self, "host_paged_kv_worker_view", None)
        if host_view is None:
            host_view = getattr(
                getattr(self, "core_engine", None),
                "host_paged_kv_worker_view",
                None,
            )
        if self.local_rank == 0:
            try:
                stats = host_view.get_stats()
                local_free = int(stats.num_free_pages)
                local_total = int(stats.num_total_pages)
            except Exception:
                snapshot_error = 1
                logging.exception(
                    "Rank %s: host-KV prefill capacity snapshot failed", self.rank
                )

        report = torch.tensor(
            [node_id, local_need, local_free, local_total, snapshot_error],
            dtype=torch.int64,
            device=self.torch_device,
        )
        if self.world_size == 1:
            if snapshot_error:
                return False, None
            ok = 0 <= local_need <= local_free and local_need <= local_total
            return ok, {node_id: (local_need, local_free)}
        if not dist.is_initialized():
            raise RuntimeError(
                "distributed process group is required for host-KV prefill admission"
            )
        gathered = [torch.zeros_like(report) for _ in range(self.world_size)]
        dist.all_gather(gathered, report)

        verdict = verdict_node_capacity(
            [
                NodeCapacityReport(
                    node_id=int(item[0].item()),
                    need=int(item[1].item()),
                    free=int(item[2].item()),
                    total=int(item[3].item()),
                    snapshot_error=int(item[4].item()),
                )
                for item in gathered
            ],
            want_rows=True,
        )

        decision = torch.tensor(
            [1 if verdict.ok else 0, 1 if verdict.rows_usable else 0],
            dtype=torch.int32, device=self.torch_device,
        )
        dist.all_reduce(decision, op=dist.ReduceOp.MIN)
        if not int(decision[1].item()):
            return bool(int(decision[0].item())), None
        # The reduced rows_usable is the MIN over ranks, so it is 1 only when
        # this rank's local verdict was usable and carries its rows.
        return bool(int(decision[0].item())), verdict.rows

    def _reclaim_host_prefix_pages_collective(
        self, worker,
        page_target: Callable[[], int],
        *,
        reason: str,
    ) -> int:
        """Reclaim retained prefix pages on each node leader; all ranks align.

        The node leader evaluates ``page_target()`` (the Host pages this node is
        short) and releases up to that many cold retained prefix pages -- only
        pages with no live attachment and no pending load, so nothing a running
        row or an attached wave reads is touched. Every other rank of the node
        then retires the trajectory frontiers standing on the released pages.

        Collective discipline: every rank reaches every collective, a
        leader-side failure (including in ``page_target``) becomes a MAX-reduced
        flag, and no rank raises before that flag is reduced. ``reason`` names
        the caller in logs, the ``prefix_reclaim`` metric and errors. Returns
        the largest page count any leader reclaimed.
        """
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        from batchgen.prefix_reuse.eviction import (
            EvictedPageRecorder,
            reclaim_prefix_pages_for_host_admission,
            retire_node_frontiers,
        )
        node_id = self.rank_ctx.node_id
        failure = 0
        local_frontier_hook = self._trajectory_frontier_eviction_hook()
        reclaimed_pages = EvictedPageRecorder(local_frontier_hook)
        if self.local_rank == 0:
            try:
                target = int(page_target())
                if target:
                    released = reclaim_prefix_pages_for_host_admission(
                        core_engine_module=core_engine,
                        coordinator=self.prefix_cache_coordinator,
                        worker_views_by_group=self._host_views_by_group(),
                        group_id=0,
                        page_target=target,
                        max_scan_nodes=self.prefix_cache_runtime_config.max_nodes,
                        frontier_hook=reclaimed_pages,
                    )
                    logging.info(
                        "[PREFIX_CACHE] %s reclaimed %s Host pages on node %s "
                        "(free=%s, target=%s)",
                        reason, released, node_id, self._get_host_kv_free_pages(),
                        target,
                    )
                    if self.prefix_cache_debug_stats:
                        self._emit_prefix_cache_metric(
                            "prefix_reclaim",
                            reason=reason,
                            node=int(node_id),
                            page_target=int(target),
                            released_pages=int(released),
                            **self._prefix_host_resource_fields(),
                        )
            except Exception:
                logging.exception("[PREFIX_CACHE] %s eviction failed", reason)
                failure = 1
        reclaimed_page_count = reclaimed_pages.page_count
        if self.world_size > 1:
            failed = torch.tensor(
                [failure, reclaimed_page_count],
                dtype=torch.int64, device=self.torch_device,
            )
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            failure = int(failed[0].item())
            reclaimed_page_count = int(failed[1].item())
        if failure != 0:
            raise RuntimeError(f"prefix cache {reason} eviction failed on a worker")
        # The leader already retired its own frontiers inside the release. Any
        # other rank of the node may own frontier records standing on the
        # reclaimed pages (G=1, or several serve groups per node), so every rank
        # runs its own hook on its node's reclaimed map.
        if self.world_size > 1 and reclaimed_page_count != 0:
            frontier_failure = 0
            gathered_reclaims = [None] * self.world_size
            dist.all_gather_object(
                gathered_reclaims,
                (
                    node_id,
                    reclaimed_pages.pages_by_group
                    if self.local_rank == 0 else None,
                ),
            )
            if self.local_rank != 0:
                try:
                    retire_node_frontiers(
                        gathered=gathered_reclaims,
                        node_id=node_id,
                        frontier_hook=local_frontier_hook,
                    )
                except Exception:
                    logging.exception(
                        "[PREFIX_CACHE] frontier retirement after %s reclaim "
                        "failed", reason,
                    )
                    frontier_failure = 1
            retired = torch.tensor(
                [frontier_failure], dtype=torch.int64, device=self.torch_device
            )
            dist.all_reduce(retired, op=dist.ReduceOp.MAX)
            if int(retired.item()) != 0:
                raise RuntimeError(
                    f"trajectory frontier retirement after {reason} reclaim "
                    "failed on a worker"
                )
        return reclaimed_page_count

    def _retry_prefill_host_capacity_after_reclaim(
        self, worker,
        owner_requests: List[Tuple[str, int]],
        capacity_rows: Dict[int, Tuple[int, int]],
    ) -> bool:
        """Reclaim this node's host-page deficit once, then re-check capacity.

        Exactly one attempt: a wave sized against free + reclaimable pages can
        be short only by what is retained, and this releases it. A second
        shortfall is a real capacity change (a concurrent decode grew, a peer
        node is the short one and has nothing cold left), which the caller
        rejects exactly as before.

        Safe by construction: the wave's prefix attachments are already held,
        so every prefix this wave HIT is pinned and ``NodeIsProtectedLocked``
        keeps it -- only cold retained pages go, which is precisely the supply
        the admission budget counted.

        Collective discipline is ``_reclaim_host_prefix_pages_collective``'s:
        every rank reaches every collective, and the deficit is computed on the
        leader inside its MAX-reduced failure flag.
        """
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        node_id = node_of_rank(self.rank, self.local_world_size)
        self._reclaim_host_prefix_pages_collective(
            lambda: prefill_host_reserve_page_deficit(capacity_rows, node_id),
            reason="host_reserve",
        )
        capacity_ok, _ = self._collective_prefill_host_capacity_ok(owner_requests)
        return capacity_ok

    def _reclaim_host_prefix_pages_for_growth(
        self, worker,
        growth_entries: List[Tuple[str, int, object]],
        worker_view: Optional[object],
    ) -> int:
        """Release the retained prefix pages this boundary's growth plan counted.

        The rank-0 growth plan covers each node's growth debt with free pages,
        completed/evicted releases and the node's reclaimable retained prefix
        pages. After every rank has released (the caller's barrier), the node
        leader reads the physical free count and reclaims exactly the part of
        this node's growth the free pages do not cover:
        ``max(0, node_growth_pages - free_after_releases)``. A shortfall the
        reclaim cannot close is left to the fresh collective preflight, which
        fails loud. ``growth_entries`` is the broadcast ``(uuid, pages, seq)``
        list, identical on every rank.
        """
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        node_id = node_of_rank(self.rank, self.local_world_size)
        group_size = self._decode_attn_tp_size()

        def node_growth_deficit() -> int:
            from batchgen.decode_dp_group import host_kv_owner_rank

            node_growth_pages = 0
            for uuid, growth_pages, seq in growth_entries:
                if seq is None:
                    raise RuntimeError(f"growth sequence {uuid} is not in global_batch")
                if group_size > 1:
                    if seq.decode_dp_group is None:
                        raise RuntimeError(
                            f"growth sequence {uuid} has no decode_dp_group for "
                            f"attn_tp_size={group_size}"
                        )
                    owner = host_kv_owner_rank(seq.decode_dp_group, group_size)
                else:
                    owner = seq.assigned_rank
                if node_of_rank(int(owner), self.local_world_size) == node_id:
                    node_growth_pages += int(growth_pages)
            if node_growth_pages == 0:
                return 0
            if worker_view is None:
                raise RuntimeError("host-KV worker view is unavailable")
            free_after_releases = int(worker_view.get_stats().num_free_pages)
            return max(0, node_growth_pages - free_after_releases)

        return self._reclaim_host_prefix_pages_collective(
            node_growth_deficit, reason="host_growth",
        )

    def _collective_host_kv_growth_capacity_ok(
        self, worker,
        host_grow_requests: List[Tuple[int, int]],
        worker_view: Optional[object] = None,
    ) -> bool:
        """Preflight one boundary's shared host-KV growth transaction.

        The boundary planner runs from a gathered snapshot taken before completed
        and evicted rows release their pages.  That snapshot is intentionally
        pure, but it cannot prove that the shared allocator still has the pages
        when Phase 4.C executes.  Every rank therefore reports its owner demand;
        only one rank per node reports the node-shared allocator's fresh free/total
        counts.  The all-rank MIN makes the decision identical before any view is
        mutated.

        The second all-rank consensus is performed by the caller after the local
        allocator call.  A post-preflight allocator failure is fatal: the native
        worker-view API has no page-level rollback for a successful peer, and
        continuing would make scalar metadata and shared page tables disagree.
        """
        # Moved VERBATIM from batchgen_worker.py (S10b); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        local_need = sum(int(pages) for _, pages in host_grow_requests)
        node_id = self.rank_ctx.node_id
        local_free = -1
        local_total = -1
        snapshot_error = 0
        if worker_view is None:
            worker_view = getattr(self, "host_paged_kv_worker_view", None)
        if worker_view is None:
            worker_view = getattr(
                getattr(self, "core_engine", None),
                "host_paged_kv_worker_view",
                None,
            )
        # The host region is shared by all ranks on a node.  Use the local
        # representative for the snapshot, matching _gather_host_kv_stats_by_node
        # and _collective_prefill_host_capacity_ok.
        if self.local_rank == 0:
            try:
                if worker_view is None:
                    raise RuntimeError("host-KV worker view is unavailable")
                stats = worker_view.get_stats()
                local_free = int(stats.num_free_pages)
                local_total = int(stats.num_total_pages)
            except Exception:
                snapshot_error = 1
                logging.exception(
                    "Rank %s: host-KV boundary growth snapshot failed", self.rank
                )

        report = torch.tensor(
            [node_id, local_need, local_free, local_total, snapshot_error],
            dtype=torch.int64,
            device=self.torch_device,
        )
        if self.world_size == 1:
            if snapshot_error:
                return False
            return 0 <= local_need <= local_free and local_need <= local_total
        if not dist.is_initialized():
            raise RuntimeError(
                "distributed process group is required for host-KV growth preflight"
            )
        gathered = [torch.zeros_like(report) for _ in range(self.world_size)]
        dist.all_gather(gathered, report)

        verdict = verdict_node_capacity(
            [
                NodeCapacityReport(
                    node_id=int(item[0].item()),
                    need=int(item[1].item()),
                    free=int(item[2].item()),
                    total=int(item[3].item()),
                    snapshot_error=int(item[4].item()),
                )
                for item in gathered
            ],
            want_rows=False,
        )

        decision = torch.tensor(
            [1 if verdict.ok else 0], dtype=torch.int32, device=self.torch_device
        )
        dist.all_reduce(decision, op=dist.ReduceOp.MIN)
        return bool(int(decision.item()))

    # ---- GPU KV page allocation and release --------------------------------

    def _collective_gpu_kv_allocation_preflight(
        self, worker, required_pages: int, free_pages: int
    ) -> bool:
        """Require every rank to approve a GPU-KV allocation before mutation."""
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        local_can_allocate = int(required_pages >= 0 and required_pages <= free_pages)
        if self.world_size == 1:
            return bool(local_can_allocate)
        if not dist.is_initialized():
            raise RuntimeError(
                "distributed process group is required for GPU-KV allocation preflight"
            )
        can_allocate = torch.tensor(
            [local_can_allocate], dtype=torch.int32, device=self.torch_device
        )
        dist.all_reduce(can_allocate, op=dist.ReduceOp.MIN)
        return bool(int(can_allocate.item()))

    def _allocate_gpu_kv_two_page_buffer(
        self, worker, 
        local_sequence_ids: List[int],
        load_from_host: bool = True,
        collective_preflight: bool = False,
    ) -> bool:
        """
        Allocate GPU KV pages using two-page buffer strategy.
    
        Returns:
            True if allocation succeeded, False otherwise.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if manager is None:
            if collective_preflight:
                return self._collective_gpu_kv_allocation_preflight(0, 0)
            return False
        if not manager.is_initialized:
            if collective_preflight:
                return self._collective_gpu_kv_allocation_preflight(0, 0)
            return False
        if not local_sequence_ids:
            if collective_preflight:
                return self._collective_gpu_kv_allocation_preflight(
                    0, manager.get_stats().num_free_pages
                )
            return True
    
        global_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
    
        pages_per_seq = []
        total_pages = 0
    
        # DIAGNOSTIC: Log allocation details for KV corruption investigation (debug-only / opt-in)
        alloc_details = []
        for local_idx in local_sequence_ids:
            uuid = self._local_to_uuid_map[local_idx]
            seq = self.global_batch.get_sequence(uuid)
            pages = seq.get_gpu_pages_for_two_page_buffer()
            pages_per_seq.append(pages * self.PAGE_SIZE)  # tokens for API
            total_pages += pages
        
            # Track details for resuming sequences (decoded_length > 0)
            if seq.decoded_length > 0:
                alloc_details.append({
                    'uuid': uuid[:8],
                    'global_idx': seq.global_idx,
                    'decoded_length': seq.decoded_length,
                    'current_context_length': seq.current_context_length,
                    'pages_allocating': pages,
                    'had_initial_gpu_reservation': seq.had_initial_gpu_reservation,
                })
    
        if alloc_details and BATCHGEN_CB_DEBUG and BATCHGEN_ENABLE_CRITICAL_DIAGS:
            logging.debug(
                f"Rank {self.rank}: _allocate_gpu_kv_two_page_buffer: Allocating GPU KV for {len(alloc_details)} RESUMING sequences. First 5: {alloc_details[:5]}"
            )

        free_pages = manager.get_stats().num_free_pages
        # With prefix sharing the physical demand is lower than the logical
        # page sum: a Host page already mapped (or repeated in this batch)
        # costs nothing. The preview runs the allocator's own validation, so
        # a demand it accepts cannot fail validation at allocation time.
        try:
            page_keys = self._host_identity_page_keys(
                manager, global_ids, pages_per_seq
            )
            demand_pages = (
                total_pages
                if page_keys is None
                else manager.count_new_shared_pages(
                    global_ids, pages_per_seq, page_keys
                )
            )
        except Exception:
            if collective_preflight:
                # The failed rank still joins the collective so peers cannot
                # advance into allocation while this rank raises.
                self._collective_gpu_kv_allocation_preflight(1, 0)
            raise
        if collective_preflight:
            if not self._collective_gpu_kv_allocation_preflight(
                demand_pages, free_pages
            ):
                logging.warning(
                    f"Rank {self.rank}: collective GPU KV allocation preflight failed "
                    f"(need {demand_pages} of {total_pages} logical, "
                    f"local free {free_pages})"
                )
                return False
        elif demand_pages > free_pages:
            logging.error(
                f"Rank {self.rank}: Cannot allocate GPU KV - need {total_pages} pages, "
                f"only {free_pages} free"
            )
            # Don't set gpu_pages_allocated since we're failing
            return False
    
        try:
            self._allocate_gpu_pages_with_host_identity(
                manager, global_ids, pages_per_seq
            )
        except RuntimeError as exc:
            if "Insufficient free pages" not in str(exc):
                raise
            logging.error(
                "Rank %s cannot allocate shared GPU KV pages: %s",
                self.rank,
                exc,
            )
            return False

        # Now safe to update tracking.
        for local_idx in local_sequence_ids:
            uuid = self._local_to_uuid_map[local_idx]
            seq = self.global_batch.get_sequence(uuid)
            pages = seq.get_gpu_pages_for_two_page_buffer()
            seq.gpu_pages_allocated = pages
            # Mark that this sequence has received its initial GPU reservation
            seq.mark_initial_gpu_reservation_done()
    
        manager.rebuild_page_table(global_ids)

        if load_from_host:
            self._load_host_kv_to_gpu(manager, global_ids)
    
        # Track in set
        for local_idx in local_sequence_ids:
            uuid = self._local_to_uuid_map[local_idx]
            self._sequences_with_gpu_kv.add(uuid)
    
        # Rebuild page table with ALL active sequences
        all_active_global_ids = []
        for uuid in self._sequences_with_gpu_kv:
            if uuid in self._uuid_to_local_map:
                seq = self.global_batch.get_sequence(uuid)
                all_active_global_ids.append(seq.global_idx)
        all_active_global_ids.sort()
    
        if all_active_global_ids:
            manager.rebuild_page_table(all_active_global_ids)

        # DIAGNOSTIC: Log page table order after allocation for debugging order mismatch (debug-only / opt-in)
        if manager and manager._gpu_page_table_manager and BATCHGEN_CB_DEBUG and BATCHGEN_ENABLE_CRITICAL_DIAGS:
            final_slot_order = list(manager._gpu_page_table_manager.slot_to_seq_id) if manager._gpu_page_table_manager.slot_to_seq_id else []
            logging.debug(
                f"Rank {self.rank}: _allocate_gpu_kv_two_page_buffer finished. "
                f"input_global_ids={global_ids[:5]}{'...' if len(global_ids) > 5 else ''} (len={len(global_ids)}), "
                f"all_active_sorted={all_active_global_ids[:5]}{'...' if len(all_active_global_ids) > 5 else ''} (len={len(all_active_global_ids)}), "
                f"final_slot_to_seq_id={final_slot_order[:5]}{'...' if len(final_slot_order) > 5 else ''} (len={len(final_slot_order)})"
            )

        logging.debug(
            f"Rank {self.rank}: Allocated GPU KV for {len(global_ids)} sequences"
        )
        return True

    def _extend_gpu_kv_allocation(self, worker, uuids: List[str]) -> bool:
        """
        Extend GPU KV allocation for sequences that need more pages.
    
        Returns:
            True if all extensions succeeded, False if insufficient pages
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if manager is None:
            return False
    
        free_pages = manager.get_stats().num_free_pages
    
        extensions_needed = []
        total_additional = 0
    
        for uuid in uuids:
            if uuid not in self._uuid_to_local_map:
                continue
            seq = self.global_batch.get_sequence(uuid)
            additional = seq.get_additional_gpu_pages_needed()
            if additional > 0:
                extensions_needed.append((uuid, additional))
                total_additional += additional
    
        if total_additional > free_pages:
            logging.warning(
                f"Rank {self.rank}: Insufficient GPU pages for extension: "
                f"need {total_additional}, have {free_pages}"
            )
            return False
    
        # Perform extensions
        for uuid, additional in extensions_needed:
            seq = self.global_batch.get_sequence(uuid)
            local_idx = self._uuid_to_local_map[uuid]
            global_id = seq.global_idx
        
            # Extend allocation
            new_total_pages = seq.gpu_pages_allocated + additional
            new_total_tokens = new_total_pages * self.PAGE_SIZE
        
            manager.extend_pages_for_sequence(global_id, new_total_tokens)
            seq.gpu_pages_allocated = new_total_pages
    
        return True

    def _shared_page_identity_sources(self, worker, manager):
        """The pools whose GPU pages can be deduplicated by Host page id.

        Returns one ``(pool name, GPU manager, Host worker view)`` triple per
        pool, or ``None`` when shared-page identity does not apply at all.
        ``None`` is the single condition behind BOTH the allocator and the
        decode admission credit: plain allocation gives every row its own
        physical copy, so a credit there would charge for a copy that was
        really made.

        A DSA model (GLM-5.x, DeepSeek-V3.2) runs TWO pools -- primary MLA
        latent KV and auxiliary indexer KV -- backed by two separate Host KV
        regions. Each pool therefore gets its own Host worker view, which is
        its own key space; group 0 is the MLA region and group 1 the indexer
        region (``DualHostKVCoordinator.views_by_group``), matching
        primary/auxiliary on the GPU side.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if manager is None or not self.enable_prefix_cache:
            return None
        if isinstance(manager, DualKVCacheCoordinator):
            host_view = getattr(self, "host_paged_kv_worker_view", None)
            if not isinstance(host_view, DualHostKVCoordinator):
                return None
            views = host_view.views_by_group()
            primary_view = views.get(0)
            auxiliary_view = views.get(1)
            if primary_view is None or auxiliary_view is None:
                return None
            return (
                ("primary", manager.primary, primary_view),
                ("auxiliary", manager.auxiliary, auxiliary_view),
            )
        worker_view = getattr(
            self.core_engine, "host_paged_kv_worker_view", None
        )
        if worker_view is None:
            return None
        return (("primary", manager, worker_view),)

    def _host_identity_page_keys(
        self, worker,
        manager,
        global_sequence_ids: Sequence[int],
        sequence_tokens: Sequence[int],
    ):
        """The page-key argument for ``manager``'s keyed allocation, or ``None``.

        One manager: a list of Host page id rows, one row per sequence, in
        logical page order -- equal ids mean one physical GPU page.

        A DSA dual coordinator: ``DualPoolPageKeys``, which carries that same
        row list ONCE PER POOL. The two pools are separate Host KV regions, so
        a Host page id identifies content only inside its own region and the
        two key spaces must never be crossed; each pool deduplicates on its
        own ids.

        ``None`` means shared-page identity does not apply (prefix cache off,
        or the Host views that define the identity are not bound) and plain
        allocation is used.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        sources = self._shared_page_identity_sources(manager)
        if sources is None:
            return None
        rows_by_pool: Dict[str, List[List[int]]] = {}
        for pool_name, pool_manager, worker_view in sources:
            rows_by_pool[pool_name] = self._pool_host_identity_page_keys(
                pool_name,
                pool_manager,
                worker_view,
                global_sequence_ids,
                sequence_tokens,
            )
        if isinstance(manager, DualKVCacheCoordinator):
            return DualPoolPageKeys.from_rows(
                rows_by_pool["primary"], rows_by_pool["auxiliary"]
            )
        return rows_by_pool["primary"]

    def _pool_host_identity_page_keys(
        self, worker,
        pool_name: str,
        pool_manager,
        worker_view,
        global_sequence_ids: Sequence[int],
        sequence_tokens: Sequence[int],
    ) -> List[List[int]]:
        """One pool's Host page ids behind each sequence's GPU reservation."""
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        host_tables = worker_view.build_page_table(global_sequence_ids)
        page_size = int(pool_manager.config.page_size_tokens)
        page_keys: List[List[int]] = []
        for sequence_id, tokens, host_pages in zip(
            global_sequence_ids, sequence_tokens, host_tables
        ):
            required = math.ceil(int(tokens) / page_size)
            if len(host_pages) < required:
                raise RuntimeError(
                    f"Host page table ({pool_name} pool) is shorter than GPU "
                    f"reservation: sequence={sequence_id}, "
                    f"host_pages={len(host_pages)}, required={required}"
                )
            page_keys.append([int(page) for page in host_pages[:required]])
        self._assert_private_writable_page_keys(
            pool_name, pool_manager, global_sequence_ids, page_keys, page_size
        )
        return page_keys

    def _assert_private_writable_page_keys(
        self, worker,
        pool_name: str,
        pool_manager,
        global_sequence_ids: Sequence[int],
        page_keys: Sequence[Sequence[int]],
        page_size: int,
    ) -> None:
        """Fail loud if a page the row will WRITE carries a shared key.

        Sharing a physical page is safe only for a committed prefix, which is
        immutable and page aligned. Everything from the row's write head on is
        its private tail: decode writes the new token at position
        ``current_context_length`` (``update_layer_decode_new_token``), i.e.
        into logical page ``current_context_length // page_size``, and every
        later page of the reservation is future tail. Those pages come from
        the row's OWN Host reservation, so their Host page ids -- the keys --
        are private by construction; this states that invariant as a check
        instead of trusting it, because a shared key here would let two rows
        write one physical page.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        write_heads: List[int] = []
        for sequence_id in global_sequence_ids:
            sequence = self.global_batch.get_sequence_by_global_idx(
                int(sequence_id)
            )
            if sequence is None:
                raise RuntimeError(
                    "shared GPU page identity needs the row's context length: "
                    f"global sequence {sequence_id} is not in the batch"
                )
            write_heads.append(
                int(sequence.current_context_length) // int(page_size)
            )
        rows_per_key: Dict[int, int] = {}
        for row in page_keys:
            for key in row:
                rows_per_key[int(key)] = rows_per_key.get(int(key), 0) + 1
        for sequence_id, row, write_head in zip(
            global_sequence_ids, page_keys, write_heads
        ):
            for index in range(max(0, write_head), len(row)):
                key = int(row[index])
                if (
                    rows_per_key[key] > 1
                    or pool_manager.is_shared_page_key_mapped(key)
                ):
                    raise RuntimeError(
                        f"{pool_name} GPU pool: sequence {sequence_id} would "
                        f"share writable logical page {index} (write head "
                        f"{write_head}); only a committed prefix page may be "
                        "shared"
                    )

    def _allocate_gpu_pages_with_host_identity(
        self, worker,
        manager,
        global_sequence_ids: Sequence[int],
        sequence_tokens: Sequence[int],
    ):
        """Share GPU pages when sequences reference the same Host page ids."""
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker

        page_keys = self._host_identity_page_keys(
            manager, global_sequence_ids, sequence_tokens
        )
        if page_keys is None:
            return manager.allocate_pages_for_sequences(
                global_sequence_ids, sequence_tokens
            )
        allocations = manager.allocate_pages_for_sequences_with_page_keys(
            global_sequence_ids,
            sequence_tokens,
            page_keys,
        )
        if self.prefix_cache_debug_stats:
            sharing = manager.get_prefix_gpu_share_stats()
            new_physical_pages = {
                int(page)
                for row in allocations.values()
                for page in row
            }
            # Both pools request the same logical page count per row, so the
            # primary rows describe the request for either shape.
            primary_rows = (
                page_keys.primary
                if isinstance(page_keys, DualPoolPageKeys)
                else page_keys
            )
            self._emit_prefix_cache_metric(
                "prefix_gpu_share",
                sequences=len(global_sequence_ids),
                requested_logical_pages=sum(len(row) for row in primary_rows),
                new_physical_pages=len(new_physical_pages),
                logical_page_references=sharing.logical_page_references,
                physical_pages=sharing.physical_pages,
                keyed_physical_pages=sharing.keyed_physical_pages,
                multi_referenced_pages=sharing.multi_referenced_pages,
                reused_logical_references=sharing.reused_logical_references,
            )
        return allocations

    def _release_gpu_kv_pages(self, worker, local_sequence_ids: List[int]) -> None:
        """Return GPU KV pages associated with the provided local sequence ids."""
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if manager is None or not local_sequence_ids:
            return
    
        global_sequence_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
    
        if not global_sequence_ids:
            return
    
        # All call sites now intersect `my_completed` with `_sequences_with_gpu_kv`
        # before reaching here, so a KeyError from the manager indicates a real
        # bookkeeping bug (the source-of-truth set drifted from the manager's
        # state). Surface it loudly instead of swallowing.
        manager.free_pages_for_sequences(global_sequence_ids)
        # NOTE: No sync needed - page deallocation is synchronous to the allocator
        logging.debug(
            f"Rank {self.rank} Released GPU KV pages for global_idx: {global_sequence_ids}"
        )

        self._release_kda_state_slots(global_sequence_ids)

        # FIX Bug 2: Remove from tracking set and reset gpu_pages_allocated
        for local_idx in local_sequence_ids:
            uuid = self._local_to_uuid_map.get(local_idx)
            if uuid:
                self._sequences_with_gpu_kv.discard(uuid)
                seq = self.global_batch.get_sequence(uuid)
                if seq is not None:
                    seq.gpu_pages_allocated = 0

    def _release_kda_state_slots(self, worker, global_sequence_ids: List[int]) -> None:
        """Release Kimi-Linear KDA state independently of paged GPU KV."""
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        self._traj.free_kda_sequences(global_sequence_ids)

    # ---- Decode KV append to host ------------------------------------------

    def _flush_deferred_kv_to_host(self, worker) -> None:
        """Flush all deferred KV host offload entries accumulated during forward.

        Batch-launch D2H copies for all layers (primary MLA KV + the DSA
        auxiliary indexer KV if present). The C++ async append APIs order their
        dedicated D2H stream after the producer stream with a CUDA event, so this
        function must not synchronize the CPU with the producer stream first.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        entries = getattr(self, '_deferred_kv_entries', [])
        entries_aux = getattr(self, '_deferred_kv_entries_aux', [])
        if not entries and not entries_aux:
            return

        worker_view = getattr(self, '_deferred_kv_worker_view', None)
        batch_info = getattr(self, '_deferred_kv_batch', None)
        aux_view = getattr(self, '_deferred_kv_worker_view_aux', None)
        if entries_aux and aux_view is None:
            raise RuntimeError(
                "DSA auxiliary host KV worker view is required for deferred aux KV offload"
            )
        if (worker_view is None or batch_info is None) and not aux_view:
            self._deferred_kv_entries = []
            self._deferred_kv_entries_aux = []
            return

        sequence_ids, sequence_lengths = batch_info if batch_info is not None else (None, None)
        if sequence_ids is not None and sequence_lengths is not None:
            self._ensure_host_kv_append_capacity(sequence_ids, sequence_lengths)

        def _assert_deferred_kv_rows(cache_name: str, layer_idx: int, tensor: torch.Tensor) -> None:
            if sequence_ids is None:
                return
            if tensor.shape[0] != len(sequence_ids):
                raise RuntimeError(
                    f"{cache_name} deferred KV row mismatch at layer {layer_idx}: "
                    f"tensor_rows={tensor.shape[0]}, sequence_ids={len(sequence_ids)}, "
                    f"gids={sequence_ids[:8] if sequence_ids is not None else []}, "
                    f"write_pos={sequence_lengths[:8] if sequence_lengths is not None else []}"
                )

        # Fire all D2H copies
        if not hasattr(self, '_pending_kv_append_tensors'):
            self._pending_kv_append_tensors = []

        _use_uva_kernel = os.environ.get(
            "BATCHGEN_KV_OFFLOAD_UVA_KERNEL", "1") == "1"

        if _use_uva_kernel and hasattr(worker_view, "async_append_decode_kv_to_host_batched_kernel"):
            if entries and worker_view is not None and sequence_ids is not None:
                _prepared_entries = []
                for layer_idx, k_tensor, v_tensor in entries:
                    if k_tensor.dim() == 3:
                        k_tensor = k_tensor.unsqueeze(2)
                    if v_tensor is not None and v_tensor.dim() == 3:
                        v_tensor = v_tensor.unsqueeze(2)
                    _assert_deferred_kv_rows("primary", layer_idx, k_tensor)
                    if v_tensor is not None:
                        _assert_deferred_kv_rows("primary_v", layer_idx, v_tensor)
                    _prepared_entries.append((layer_idx, k_tensor, v_tensor))
                    self._pending_kv_append_tensors.append(k_tensor)
                    if v_tensor is not None:
                        self._pending_kv_append_tensors.append(v_tensor)
                task = worker_view.async_append_decode_kv_to_host_batched_kernel(
                    entries=_prepared_entries,
                    sequence_ids=sequence_ids,
                    sequence_lengths=sequence_lengths,
                )
                if task is not None:
                    self._pending_kv_append_tasks.append(task)

            if entries_aux and aux_view is not None and sequence_ids is not None:
                _prepared_aux = []
                for layer_idx, k_tensor, v_tensor, scale_tensor in (
                    _normalize_aux_kv_entry(entry) for entry in entries_aux
                ):
                    if k_tensor.dim() == 3:
                        k_tensor = k_tensor.unsqueeze(2)
                    _assert_deferred_kv_rows("aux", layer_idx, k_tensor)
                    _prepared_aux.append((layer_idx, k_tensor, None, scale_tensor))
                    self._pending_kv_append_tensors.append(k_tensor)
                    if scale_tensor is not None:
                        self._pending_kv_append_tensors.append(scale_tensor)
                task = aux_view.async_append_decode_kv_to_host_batched_kernel(
                    entries=_prepared_aux,
                    sequence_ids=sequence_ids,
                    sequence_lengths=sequence_lengths,
                )
                if task is not None:
                    self._pending_kv_append_tasks.append(task)
        else:
            if entries and worker_view is not None and sequence_ids is not None:
                for layer_idx, k_tensor, v_tensor in entries:
                    if k_tensor.dim() == 3:
                        k_tensor = k_tensor.unsqueeze(2)
                    if v_tensor is not None and v_tensor.dim() == 3:
                        v_tensor = v_tensor.unsqueeze(2)
                    _assert_deferred_kv_rows("primary", layer_idx, k_tensor)
                    if v_tensor is not None:
                        _assert_deferred_kv_rows("primary_v", layer_idx, v_tensor)

                    task = worker_view.async_append_decode_kv_to_host(
                        layer_idx=layer_idx,
                        sequence_ids=sequence_ids,
                        k_tensor=k_tensor,
                        v_tensor=v_tensor,
                        sequence_lengths=sequence_lengths,
                    )

                    self._pending_kv_append_tensors.append(k_tensor)
                    if v_tensor is not None:
                        self._pending_kv_append_tensors.append(v_tensor)
                    if task is not None:
                        self._pending_kv_append_tasks.append(task)

            if entries_aux and aux_view is not None and sequence_ids is not None:
                for layer_idx, k_tensor, v_tensor, scale_tensor in (
                    _normalize_aux_kv_entry(entry) for entry in entries_aux
                ):
                    if k_tensor.dim() == 3:
                        k_tensor = k_tensor.unsqueeze(2)
                    _assert_deferred_kv_rows("aux", layer_idx, k_tensor)

                    task = aux_view.async_append_decode_kv_to_host(
                        layer_idx=layer_idx,
                        sequence_ids=sequence_ids,
                        k_tensor=k_tensor,
                        v_tensor=None,
                        sequence_lengths=sequence_lengths,
                        scale_tensor=scale_tensor,
                    )

                    self._pending_kv_append_tensors.append(k_tensor)
                    if scale_tensor is not None:
                        self._pending_kv_append_tensors.append(scale_tensor)
                    if task is not None:
                        self._pending_kv_append_tasks.append(task)

        # This function only LAUNCHES the D2H append tasks. The decode loop
        # drains them every step right after the sampled-token readback, in
        # a step-split slot of its own; see [KV_APPEND_DRAIN] in
        # ``decoding_continuous`` for why that drain must stay per step.
        self._deferred_kv_entries = []
        self._deferred_kv_entries_aux = []
        self._deferred_kv_batch = None
        self._deferred_kv_worker_view = None
        self._deferred_kv_worker_view_aux = None

    def _ensure_host_kv_append_capacity(
        self, worker,
        sequence_ids: List[int],
        sequence_lengths: List[int],
    ) -> None:
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if len(sequence_ids) != len(sequence_lengths):
            raise RuntimeError(
                f"host KV append metadata mismatch: ids={len(sequence_ids)} lengths={len(sequence_lengths)}"
            )
        if self.global_batch is None:
            return
        # Resolve through SequenceBatch's maintained gid index. This runs on
        # every decode step, so rebuilding a gid -> entry dict here cost one
        # dict comprehension over the whole batch per step.
        grow_requests = []
        for global_idx, write_pos in zip(sequence_ids, sequence_lengths):
            seq = self.global_batch.get_sequence_by_global_idx(int(global_idx))
            if seq is None:
                raise RuntimeError(f"host KV append for unknown global_idx={global_idx}")
            if int(write_pos) < 0:
                raise RuntimeError(
                    f"host KV append negative write position for gid={global_idx}: {write_pos}"
                )
            required_tokens = int(write_pos) + 1
            if int(seq.host_token_capacity) <= 0:
                raise RuntimeError(
                    f"host KV append for unallocated gid={global_idx}: "
                    f"write_pos={write_pos}, host_token_capacity={seq.host_token_capacity}, "
                    f"ctx={seq.current_context_length}, decoded={seq.decoded_length}, "
                    f"status={seq.status.name}"
                )
            if required_tokens > int(seq.kv_token_budget):
                raise RuntimeError(
                    f"host KV append would exceed token budget for gid={global_idx}: "
                    f"required_tokens={required_tokens}, kv_token_budget={seq.kv_token_budget}, "
                    f"ctx={seq.current_context_length}, decoded={seq.decoded_length}, "
                    f"status={seq.status.name}"
                )
            if required_tokens > int(seq.host_token_capacity):
                # The guards above bound required_tokens to the token budget,
                # so the owner's budget cap never engages here.
                growth_pages = seq.get_host_growth_pages(
                    required_tokens - int(seq.host_token_capacity)
                )
                grow_requests.append((int(global_idx), growth_pages))

        if not grow_requests:
            return

        # Host pages are allocated only by the rank-0 boundary planner.  A
        # deferred append is already inside a decode interval whose reservation
        # was validated before forward; reaching this branch means the scheduler
        # violated that invariant.  Growing here would bypass completion release,
        # eviction, mirrored DSA allocation, and G>1 host ownership.
        raise RuntimeError(
            "host KV append reached without a boundary reservation: "
            f"requests={grow_requests[:8]}; decode must enter the page-boundary "
            "planner before issuing the forward"
        )

    def _aux_layers_with_slots(self, worker):
        """Logical layer ids that have a slot in the aux (indexer) KV pools.

        None = no layer mapping (every engine layer has an aux slot). GLM-5.2
        maps only its 21 indexer layers; skip layers' index-K is never read,
        so decode appends for them are dropped at the queueing choke points —
        the mapped host view hard-fails on unmapped layers by design.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        cached = getattr(self, "_aux_layers_with_slots_cache", False)
        if cached is not False:
            return cached
        coord = getattr(self, "gpu_paged_kv_cache_manager", None)
        aux_mgr = getattr(coord, "auxiliary", None)
        if aux_mgr is None:
            return None  # aux not initialized yet; do not cache
        aux_map = getattr(aux_mgr.config, "logical_to_physical_layer", None)
        result = None
        if aux_map is not None:
            result = frozenset(
                l for l, p in enumerate(aux_map) if p is not None and int(p) >= 0
            )
        self._aux_layers_with_slots_cache = result
        return result

    def _append_decode_kv_to_host_aux_async(
        self, worker,
        layer_idx: int,
        batch: List[int],
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor = None,
        scale_tensor: torch.Tensor = None,
    ) -> None:
        """Fire-and-forget auxiliary (indexer) KV append to host.

        Uses the auxiliary host worker view and the shared pending task list,
        so one flush drains primary and auxiliary appends together.

        ``scale_tensor`` carries the fp32 per-token scales that a page-split
        FP8 aux region requires alongside the quantized keys; an AoS region
        rejects it. The host view validates the pairing, so a producer that
        forgets the scales fails loud instead of leaving a page's scale region
        at its previous contents.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
        if aux_view is None or not batch:
            return
        valid_layers = self._aux_layers_with_slots()
        if valid_layers is not None and layer_idx not in valid_layers:
            return

        sequence_ids = []
        sequence_lengths = []
        for local_idx in batch:
            uuid = self._local_to_uuid_map[local_idx]
            seq = self.global_batch.get_sequence(uuid)
            sequence_ids.append(seq.global_idx)
            write_pos = seq.current_context_length - 1
            sequence_lengths.append(write_pos)

        if k_tensor.dim() == 3:
            k_tensor = k_tensor.unsqueeze(2)

        # Launch async D2H — no CPU-side sync needed (same as primary path).
        task = aux_view.async_append_decode_kv_to_host(
            layer_idx=layer_idx,
            sequence_ids=sequence_ids,
            k_tensor=k_tensor,
            v_tensor=None,
            sequence_lengths=sequence_lengths,
            scale_tensor=scale_tensor,
        )

        if not hasattr(self, '_pending_kv_append_tensors'):
            self._pending_kv_append_tensors = []
        self._pending_kv_append_tensors.append(k_tensor)
        if scale_tensor is not None:
            self._pending_kv_append_tensors.append(scale_tensor)

        if task is not None:
            self._pending_kv_append_tasks.append(task)

        MAX_PENDING_KV_TASKS = 256
        if len(self._pending_kv_append_tasks) >= MAX_PENDING_KV_TASKS:
            self._wait_pending_kv_append_tasks()

    def _wait_pending_kv_append_tasks(
        self, worker,
        *,
        sync_distributed_errors: bool = False,
        defer_errors: bool = False,
        device_sync: bool = True,
    ) -> int:
        """
        Wait for all pending KV append tasks at page boundary.
        Returns the number of tasks that were waited for.
    
        CRITICAL: Also syncs CUDA to ensure all D2H DMA operations complete.
        Without this, KV data may not be fully written to host memory when
        sequences are later resumed, causing KV corruption.

        ``device_sync=False`` is only for the deferred-readback decode step,
        where the next forward is already enqueued and a device-wide sync
        would wait for it; see _finish_deferred_decode_step for why each
        task's own copy-stream sync suffices there.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        deferred_errors = getattr(self, "_deferred_kv_append_wait_errors", [])
        if not hasattr(self, '_pending_kv_append_tasks'):
            if sync_distributed_errors:
                error_payload = {
                    "rank": self.rank,
                    "errors": list(deferred_errors),
                } if deferred_errors else None
                all_errors = [None] * self.world_size
                dist.all_gather_object(all_errors, error_payload)
                if hasattr(self, "_deferred_kv_append_wait_errors"):
                    self._deferred_kv_append_wait_errors.clear()
                flat_errors = [e for e in all_errors if e is not None]
                if flat_errors:
                    raise RuntimeError(
                        f"KV append/offload failed on at least one rank: {flat_errors[:8]}"
                    )
            elif deferred_errors and not defer_errors:
                raise RuntimeError(
                    f"Rank {self.rank}: KV append/offload failed: {deferred_errors[:4]}"
                )
            return 0
    
        num_tasks = len(self._pending_kv_append_tasks)
        wait_errors = list(deferred_errors)
        if deferred_errors and hasattr(self, "_deferred_kv_append_wait_errors"):
            self._deferred_kv_append_wait_errors.clear()
        for task in self._pending_kv_append_tasks:
            if task is not None:
                try:
                    task.wait()
                except Exception as e:
                    wait_errors.append(f"{type(e).__name__}: {e}")
    
        # CRITICAL FIX: Sync CUDA after waiting for tasks
        # The async tasks use a separate CUDA stream for D2H copies.
        # Even though each task internally syncs its stream via cudaEventSynchronize,
        # we need a full device sync to ensure ALL pending operations complete
        # before we allow GPU pages to be freed/reused.
        if device_sync and num_tasks > 0 and not wait_errors:
            try:
                torch.cuda.synchronize(self.torch_device)
            except Exception as e:
                wait_errors.append(f"{type(e).__name__}: {e}")
    
        self._pending_kv_append_tasks.clear()
    
        # CRITICAL: Clear tensor references AFTER tasks complete
        # Tensors can now be safely garbage collected / memory reused
        if hasattr(self, '_pending_kv_append_tensors'):
            self._pending_kv_append_tensors.clear()

        if sync_distributed_errors:
            error_payload = {
                "rank": self.rank,
                "errors": wait_errors,
            } if wait_errors else None
            all_errors = [None] * self.world_size
            dist.all_gather_object(all_errors, error_payload)
            flat_errors = [e for e in all_errors if e is not None]
            if flat_errors:
                raise RuntimeError(
                    f"KV append/offload failed on at least one rank: {flat_errors[:8]}"
                )
        elif wait_errors and defer_errors:
            if not hasattr(self, "_deferred_kv_append_wait_errors"):
                self._deferred_kv_append_wait_errors = []
            self._deferred_kv_append_wait_errors.extend(wait_errors)
        elif wait_errors:
            raise RuntimeError(
                f"Rank {self.rank}: KV append/offload failed: {wait_errors[:4]}"
            )
    
        return num_tasks

    # ---- Host-to-GPU KV load -----------------------------------------------

    def _prepare_dual_kv_load_pointers(
        self, worker,
        gpu_manager: DualKVCacheCoordinator,
        new_global_ids: List[int],
        existing_global_ids: Optional[List[int]] = None,
    ) -> _DualKVLoadPointers:
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if not isinstance(gpu_manager, DualKVCacheCoordinator):
            raise RuntimeError("DSA dual KV load requires DualKVCacheCoordinator")
        if not new_global_ids:
            raise ValueError("_prepare_dual_kv_load_pointers requires non-empty sequence ids")

        sequence_tensor = torch.tensor(new_global_ids, dtype=torch.int64, device="cpu")
        try:
            gpu_manager.rebuild_page_table(new_global_ids)

            primary_order = list(gpu_manager.primary._gpu_page_table_manager.slot_to_seq_id)
            aux_order = list(gpu_manager.auxiliary._gpu_page_table_manager.slot_to_seq_id)
            if primary_order != new_global_ids or aux_order != new_global_ids:
                raise RuntimeError(
                    f"DSA dual load page-table order mismatch: requested={new_global_ids[:10]} "
                    f"primary={primary_order[:10]} aux={aux_order[:10]}"
                )

            primary_k, primary_v = gpu_manager.primary.get_padded_3d_page_pointers()
            primary_counts = gpu_manager.primary.export_active_sequence_page_counts()
            aux_k, aux_v = gpu_manager.auxiliary.get_padded_3d_page_pointers()
            aux_counts = gpu_manager.auxiliary.export_active_sequence_page_counts()
            if primary_counts.tolist() != aux_counts.tolist():
                raise RuntimeError(
                    f"DSA dual load page-count mismatch: "
                    f"primary={primary_counts.tolist()} aux={aux_counts.tolist()}"
                )

            return _DualKVLoadPointers(
                sequence_tensor=sequence_tensor,
                primary_k_ptrs=primary_k,
                primary_v_ptrs=primary_v,
                primary_page_counts=primary_counts,
                aux_k_ptrs=aux_k,
                aux_v_ptrs=aux_v,
                aux_page_counts=aux_counts,
            )
        finally:
            if existing_global_ids:
                gpu_manager.rebuild_page_table(existing_global_ids)
            else:
                gpu_manager.clear_page_table()

    def _launch_dual_host_kv_load(self, worker, pointers: _DualKVLoadPointers) -> DualAsyncKVTask:
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        host_view = self.host_paged_kv_worker_view
        if not isinstance(host_view, DualHostKVCoordinator):
            raise RuntimeError("DSA dual KV load requires DualHostKVCoordinator")
        return host_view.async_load_layer_paged_kv_to_device_dual(
            sequence_ids=pointers.sequence_tensor,
            primary_active_page_counts=pointers.primary_page_counts,
            primary_k_device_ptrs=pointers.primary_k_ptrs,
            primary_v_device_ptrs=pointers.primary_v_ptrs,
            aux_active_page_counts=pointers.aux_page_counts,
            aux_k_device_ptrs=pointers.aux_k_ptrs,
            aux_v_device_ptrs=pointers.aux_v_ptrs,
            tensors=pointers,
        )

    def _load_host_kv_to_gpu(
        self, worker,
        manager: GPUPagedKVCacheManager,
        global_sequence_ids: List[int],
    ) -> None:
        """Copy prefetched host KV pages into the GPU cache."""
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if not global_sequence_ids:
            return
        copy_start = time.perf_counter()
        worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
        if worker_view is None:
            raise RuntimeError("Host paged KV worker view is not bound to the core engine")
    
        # DIAGNOSTIC: Check if these are resuming sequences (have decoded tokens)
        resuming_seq_info = []
        for global_idx in global_sequence_ids:
            # Find the sequence by global_idx
            for uuid, local_idx in self._uuid_to_local_map.items():
                seq = self.global_batch.get_sequence(uuid)
                if seq and seq.global_idx == global_idx and seq.decoded_length > 0:
                    resuming_seq_info.append({
                        'global_idx': global_idx,
                        'decoded_length': seq.decoded_length,
                        'current_context_length': seq.current_context_length,
                    })
                    break
    
        if resuming_seq_info and BATCHGEN_CB_DEBUG:
            logging.debug(
                f"Rank {self.rank}: _load_host_kv_to_gpu loading KV for {len(resuming_seq_info)} RESUMING sequences. First 5: {resuming_seq_info[:5]}"
            )
    
        logging.debug(
            f"Rank {self.rank}: _load_host_kv_to_gpu launching async load for "
            f"{len(global_sequence_ids)} sequences..."
        )

        if isinstance(manager, DualKVCacheCoordinator):
            pointers = self._prepare_dual_kv_load_pointers(manager, global_sequence_ids)
            load_task = self._launch_dual_host_kv_load(pointers)
        else:
            sequence_tensor = torch.tensor(global_sequence_ids, dtype=torch.int64, device="cpu")
            k_ptrs, v_ptrs = manager.get_padded_3d_page_pointers()
            active_sequence_page_counts = manager.export_active_sequence_page_counts()
            load_task = worker_view.async_load_layer_paged_kv_to_device(
                sequence_ids=sequence_tensor,
                active_page_counts=active_sequence_page_counts,
                k_device_ptrs=k_ptrs,
                v_device_ptrs=v_ptrs,
            )
    
        # Wait for load to complete (this is synchronous load path used during prefill)
        load_task.wait()
        # CRITICAL: Sync CUDA after async task completes to ensure H2D DMA is done
        torch.cuda.synchronize(self.torch_device)

        load_duration = time.perf_counter() - copy_start
        logging.debug(
            "Rank %s Loaded host KV for %d sequences into GPU cache in %.3fs",
            self.rank, len(global_sequence_ids), load_duration,
        )

    # ---- Decode capacity snapshot ------------------------------------------

    def _snapshot_decode_capacity(self, worker) -> DecodeCapacitySnapshot:
        """Collect the live GPU-KV capacity before decode admission.

        Total pages are immutable after initialization and must match on every
        rank.  Free pages are dynamic; the pure reducer takes the tightest rank
        in each replicated attention group so the admission plan is safe for all
        ranks that will allocate the selected sequence.

        The whole read — local stats and the cross-rank all_gather — runs inside
        the manager's ``capacity_snapshot()`` section (#424), so the phase
        switch cannot release the pool underneath a decision that is already
        being taken against it: a release attempted here fails loud instead of
        producing a plan for pages that no longer exist.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        manager = self.gpu_paged_kv_cache_manager
        if manager is None or not manager.is_initialized:
            raise RuntimeError(
                "GPU KV manager must be initialized before taking a decode capacity snapshot"
            )
        with manager.capacity_snapshot():
            stats = manager.get_stats()
            local = torch.tensor(
                [int(stats.num_total_pages), int(stats.num_free_pages)],
                dtype=torch.int64,
                device=self.torch_device,
            )
            if self.world_size == 1:
                rank_stats = [local]
            else:
                if not dist.is_initialized():
                    raise RuntimeError(
                        "distributed process group is required for a multi-rank decode capacity snapshot"
                    )
                rank_stats = [torch.zeros_like(local) for _ in range(self.world_size)]
                dist.all_gather(rank_stats, local)
        return reduce_decode_capacity_snapshot(
            tuple(int(item[0].item()) for item in rank_stats),
            tuple(int(item[1].item()) for item in rank_stats),
            world_size=self.world_size,
            attn_tp_size=self._decode_attn_tp_size(),
        )

    # ---- Page-boundary decision request ------------------------------------

    def _make_boundary_decision_request(
        self, worker,
        decode_uuids: List[str],
        global_seq_state: Dict[str, Dict],
        global_candidate_info: Dict[str, Dict],
        per_rank_free: List[int],
        per_rank_total: List[int],
        chunk_size: int,
        per_node_host_stats: Optional[List[Dict[str, int]]],
    ) -> BoundaryDecisionRequest:
        """Snapshot the boundary-decision inputs (rank 0 only).

        ``seq_meta`` is sourced from ``global_batch`` (rank 0 holds every
        sequence) for the union of decode + loadable-candidate uuids,
        replacing the handler's direct ``global_batch`` reads. Entries are
        omitted for uuids with no sequence — the handler treats a missing
        entry as ``global_idx = inf`` / ``priority = 0``, matching the
        legacy ``if seq is not None else ...`` fallbacks.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        seq_meta: Dict[str, BoundarySeqMeta] = {}
        for uuid in set(decode_uuids) | set(global_candidate_info.keys()):
            seq = self.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            seq_meta[uuid] = BoundarySeqMeta(
                global_idx=seq.global_idx,
                priority=getattr(seq, 'priority', 0),
                current_context_length=getattr(seq, 'current_context_length', 0),
                host_token_capacity=getattr(seq, 'host_token_capacity', 0),
                host_pages_allocated=getattr(seq, 'host_pages_allocated', 0),
            )
        return BoundaryDecisionRequest(
            decode_uuids=tuple(decode_uuids),
            global_seq_state=global_seq_state,
            global_candidate_info=global_candidate_info,
            per_rank_free=tuple(per_rank_free),
            per_rank_total=tuple(per_rank_total),
            chunk_size=chunk_size,
            per_node_host_stats=tuple(per_node_host_stats) if per_node_host_stats else None,
            seq_meta=seq_meta,
            world_size=self.world_size,
            num_gpus_per_node=self.local_world_size,
            max_rank_bsz=getattr(self, "_decode_padding_bsz", 0) or 0,
            enable_host_kv_eviction=self.enable_host_kv_eviction,
            attn_tp_size=self._decode_attn_tp_size(),
        )

    def _integrate_pending_async_load(
        self, worker,
        pending_async_task: Optional[object],
        pending_load_uuids: List[str],
        pending_load_local_indices: List[int],
        pending_load_global_ids: List[int],
        decode_uuids: List[str],
        batch: List[int],
        gpu_manager,
    ) -> Tuple[List[str], List[int]]:
        """Complete an in-flight boundary GPU-KV load and fold its rows in.

        A launched boundary load has already allocated GPU pages -- and
        registered their shared-page keys -- for rows that are NOT yet in
        ``decode_uuids``; only the finalize step moves them in. Every decode
        exit that drops its pending-load state must therefore run this first:
        an exit that forgets the pending rows strands their GPU pages (no
        later path frees pages for a row outside ``decode_uuids``), and the
        next load of such a row finds its own stale keyed pages still mapped
        and fails the writable-page sharing guard, which the collective
        converts into a cohort-fatal SCHED_INVARIANT.

        All ranks hold identical ``pending_load_uuids`` and reach this from
        broadcast boundary decisions, so the wait/barrier/finalize protocol
        stays rank-aligned.
        """
        # Moved VERBATIM from batchgen_worker.py (S10c); rebinding self to the
        # worker keeps every original self. spelling. The engine is stateless.
        self = worker
        if pending_async_task is not None:
            pending_async_task.wait()
            torch.cuda.synchronize(self.torch_device)
        dist.barrier()
        return self._finalize_async_load_minimal(
            pending_async_task,
            pending_load_uuids,
            pending_load_local_indices,
            pending_load_global_ids,
            decode_uuids,
            batch,
            gpu_manager,
        )
