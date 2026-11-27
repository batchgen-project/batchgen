"""Load the D6.1/D6.2 definitions without importing the native/CUDA module graph."""

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
from batchgen.query_book import QueryBook
from batchgen.sequence import SequenceStatus
from batchgen.worker.kv_manager import (
    GPU_KV_POOL_HEADROOM_GB,
    KVCacheManager,
    KVUtilizationRequest,
    PageTableCapacityRequest,
)
from batchgen.worker.kv_plan import provisional_pool_gb
from batchgen.worker.rank_context import node_of_rank


SOURCE = Path(__file__).resolve().parents[1] / "batchgen/worker/kv_capacity_engine.py"
CONTEXT_CLASSES = {
    "DeploymentContext", "PageTableContext",
    # D6.2
    "HostKVStatsContext", "HostKVHealthContext", "DecodeRankBszContext",
    "DeploymentVerifyContext", "DeploymentVerifyOutcome",
}
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
        elif isinstance(node, ast.ClassDef):
            if node.name == "KVCapacityEngine":
                node.body = [member for member in node.body
                             if isinstance(member, ast.FunctionDef) and member.name in METHODS]
                nodes.append(node)
            elif node.name.startswith("GPUKV") or node.name in CONTEXT_CLASSES:
                nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace
