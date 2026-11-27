"""Load the D6.1-D6.5 definitions without importing the native/CUDA graph."""

import ast
import copy
import dataclasses
import gc
import logging
import os
import time
from pathlib import Path

from batchgen.config.max_model_len import (
    SequenceLengthCapacity,
    verify_max_model_len,
)
from batchgen.lifespan import SeqEvent
from batchgen.query_book import (
    QueryBook,
    QueryBookPoolCapacityError,
    refresh_query_book_entry,
    release_query_book_slot,
)
import batchgen.sequence as _sequence_module
from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.admission.plan import (
    StatusEdge,
    Transition,
    TransitionPlan,
    apply_or_raise,
    update_batch_status,
)
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.indexing import IndexLookupRequest, IndexManager
from batchgen.worker.kv_manager import (
    GPU_KV_POOL_HEADROOM_GB,
    KVCacheManager,
    KVUtilizationRequest,
    MigrationCandidate,
    MigrationPlanRequest,
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
from batchgen.worker.prefill import (
    compute_prefill_host_reservation,
    host_owned_pages_after_sharing,
    prefill_host_reserve_page_deficit,
)
from batchgen.worker.rank_context import node_of_rank


SOURCE = Path(__file__).resolve().parents[1] / "batchgen/worker/kv_capacity_engine.py"
CONTEXT_CLASSES = {
    "DeploymentContext", "PageTableContext",
    # D6.2
    "HostKVStatsContext", "HostKVHealthContext", "DecodeRankBszContext",
    "DeploymentVerifyContext", "DeploymentVerifyOutcome",
    # D6.3
    "HostKVWatermarkContext", "WatermarkTriggerOutcome",
    "MigrationPlanOutcome", "KVMigrationReceiveOutcome",
    # D6.4
    "DeferredKVFlushContext", "DeferredKVFlushOutcome",
    "AuxLayerSlotsContext", "AuxLayerSlotsOutcome",
    # D6.5 (GPUKVDestroyContext loads through the GPUKV prefix rule below)
    "NodeMemfdIdentityContext",
}
# D6.4: the deferred flush normalizes each queued aux entry through this
# module-scope helper, so the loaded body gets the real one.
MODULE_FUNCTIONS = {"_normalize_aux_kv_entry"}
METHODS = {
    "_calculate_gpu_kv_cache_size", "_kv_fixed_address_pool_available",
    "_initialize_gpu_kv_manager_fixed_size", "_bind_gpu_paged_kv_manager",
    "_deployment_context_bound_tokens", "_make_page_table_capacity_request",
    "_with_cuda_graph_page_table_capacity", "_init_gpu_kv_with_actual_size",
    "_finalize_gpu_kv_pool_after_capture", "_get_cuda_graph_gpu_manager",
    "_release_suspended_decode_kv_pool", "_remap_reused_kv_pool",
    "_assert_kv_pool_addresses_are_fixed", "_assert_kv_region_mapped",
    "_assert_kv_region_released",
    # D6.2: KV stats, host views, the decode cap and the deployment length.
    "_make_kv_utilization_request", "_get_host_kv_utilization",
    "_get_host_kv_total_pages", "_log_host_kv_health", "_max_decode_rank_bsz",
    "_legacy_get_host_kv_utilization", "_verify_deployment_max_model_len",
    # D6.3: the host-KV watermark, the host-capacity due checks and the KV
    # migration plan / execution / rebalance.
    "_make_watermark_trigger_request", "_credit_reclaimable_prefix_pages",
    "_host_capacity_due_local", "_host_capacity_due_all_ranks",
    "_check_host_kv_watermark_trigger", "_make_migration_plan_request",
    "_plan_kv_migration", "_execute_kv_migrations_parallel",
    "_group_migrations_for_parallel_execution",
    "_execute_single_kv_migration", "_rebalance_host_kv",
    # D6.4: the decode KV append to host and the host-to-GPU KV load.
    "_flush_deferred_kv_to_host", "_ensure_host_kv_append_capacity",
    "_aux_layers_with_slots", "_append_decode_kv_to_host_aux_async",
    "_load_host_kv_to_gpu", "_integrate_pending_async_load",
    "_allocate_gpu_kv_two_page_buffer",
    # D6.5: the last worker-shaped operations of this engine.
    "_exchange_node_memfd_identity", "_collective_prefill_host_capacity_ok",
    "_retry_prefill_host_capacity_after_reclaim",
    "_destroy_gpu_paged_kv_cache", "_prefill_host_reservation_plan",
    "_put_sequences_on_hold", "_try_load_new_sequences",
}
# The D6.3 names, so a test can assert the guard covers exactly this stage.
D6_3_METHODS = {
    "_make_watermark_trigger_request", "_credit_reclaimable_prefix_pages",
    "_host_capacity_due_local", "_host_capacity_due_all_ranks",
    "_check_host_kv_watermark_trigger", "_make_migration_plan_request",
    "_plan_kv_migration", "_execute_kv_migrations_parallel",
    "_group_migrations_for_parallel_execution",
    "_execute_single_kv_migration", "_rebalance_host_kv",
}
# The D6.4 names, so a test can assert the guard covers exactly this stage.
# ``_wait_pending_kv_append_tasks`` sits in the same file section but is NOT
# here: it arrived context-shaped before Phase D and this stage left it byte
# for byte alone, which is why it is also absent from METHODS.
D6_4_METHODS = {
    "_flush_deferred_kv_to_host", "_ensure_host_kv_append_capacity",
    "_aux_layers_with_slots", "_append_decode_kv_to_host_aux_async",
    "_load_host_kv_to_gpu", "_integrate_pending_async_load",
    "_allocate_gpu_kv_two_page_buffer",
}
# The D6.5 names: the LAST worker-shaped operations of this engine, so after
# this stage ``METHODS`` is every delegated operation the file defines.
D6_5_METHODS = {
    "_exchange_node_memfd_identity", "_collective_prefill_host_capacity_ok",
    "_retry_prefill_host_capacity_after_reclaim",
    "_destroy_gpu_paged_kv_cache", "_prefill_host_reservation_plan",
    "_put_sequences_on_hold", "_try_load_new_sequences",
}


def engine_class_node():
    return next(node for node in ast.parse(SOURCE.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "KVCapacityEngine")


def engine_method_names():
    """Every method name defined on ``KVCapacityEngine``.

    A converted method may still call a SIBLING engine method through ``self``
    (the stats reads call the manager builder, exactly as the free-page reads
    do); the no-worker guard uses this set to tell an engine call apart from a
    worker attribute.
    """
    return {node.name for node in engine_class_node().body
            if isinstance(node, ast.FunctionDef)}


def method_node(name):
    cls = engine_class_node()
    return copy.deepcopy(next(node for node in cls.body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


def load_kv_capacity(namespace=None):
    namespace = dict(namespace or {})
    for name, value in dict(
        __name__=__name__, dataclass=dataclasses.dataclass, gc=gc,
        logging=logging, time=time, os=os, KVCacheManager=KVCacheManager,
        PageTableCapacityRequest=PageTableCapacityRequest,
        KVUtilizationRequest=KVUtilizationRequest,
        GPU_KV_POOL_HEADROOM_GB=GPU_KV_POOL_HEADROOM_GB,
        provisional_pool_gb=provisional_pool_gb, _GPU_KV_CACHE_SIZE_OVERRIDE=None,
        _dataclasses=dataclasses, _MAX_DECODE_RANK_BSZ=128,
        SequenceStatus=SequenceStatus, node_of_rank=node_of_rank, QueryBook=QueryBook,
        SequenceLengthCapacity=SequenceLengthCapacity,
        verify_max_model_len=verify_max_model_len,
        # D6.3 module-scope collaborators of the watermark, host-capacity and
        # migration bodies.
        WatermarkTriggerRequest=WatermarkTriggerRequest,
        MigrationPlanRequest=MigrationPlanRequest,
        MigrationCandidate=MigrationCandidate,
        HostKVRebalancer=HostKVRebalancer,
        credit_reclaimable_pages=credit_reclaimable_pages,
        host_capacity_due=host_capacity_due,
        SequenceEntry=SequenceEntry, SeqEvent=SeqEvent,
        QueryBookPoolCapacityError=QueryBookPoolCapacityError,
        release_query_book_slot=release_query_book_slot,
        refresh_query_book_entry=refresh_query_book_entry,
        BATCHGEN_CB_DEBUG=False,
        # D6.4 module-scope collaborators of the deferred flush, the aux
        # append and the two-page allocation.
        BATCHGEN_ENABLE_CRITICAL_DIAGS=False,
        IndexManager=IndexManager,
        IndexLookupRequest=IndexLookupRequest,
        # D6.5 module-scope collaborators of the identity exchange, the
        # prefill-admission transaction, the pool destroy, the watermark hold
        # and the legacy host-to-GPU load.
        prefill_owner_demand_pages=prefill_owner_demand_pages,
        verdict_node_capacity=verdict_node_capacity,
        NodeCapacityReport=NodeCapacityReport,
        prefill_host_reserve_page_deficit=prefill_host_reserve_page_deficit,
        compute_prefill_host_reservation=compute_prefill_host_reservation,
        host_owned_pages_after_sharing=host_owned_pages_after_sharing,
        _sequence_module=_sequence_module,
        apply_or_raise=apply_or_raise,
        update_batch_status=update_batch_status,
        TransitionPlan=TransitionPlan,
        Transition=Transition,
        StatusEdge=StatusEdge,
    ).items():
        namespace.setdefault(name, value)
    nodes = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "GPU_KV_UNCHANGED"
            for target in node.targets
        ):
            nodes.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name in MODULE_FUNCTIONS
        ):
            # A module-scope helper a converted body calls on the spot; it is
            # compiled from the real source, not re-implemented here.
            nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.name == "KVCapacityEngine":
                node.body = [member for member in node.body
                             if isinstance(member, ast.FunctionDef) and member.name in METHODS]
                nodes.append(node)
            elif node.name.startswith("GPUKV") or node.name in CONTEXT_CLASSES:
                nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace
