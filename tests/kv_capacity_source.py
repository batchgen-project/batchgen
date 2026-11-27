"""Load the D6.1 definitions without importing the native/CUDA module graph."""

import ast
import copy
import dataclasses
import gc
import logging
import time
from pathlib import Path

from batchgen.worker.kv_manager import (
    GPU_KV_POOL_HEADROOM_GB,
    KVCacheManager,
    PageTableCapacityRequest,
)
from batchgen.worker.kv_plan import provisional_pool_gb


SOURCE = Path(__file__).resolve().parents[1] / "batchgen/worker/kv_capacity_engine.py"
METHODS = {
    "_calculate_gpu_kv_cache_size", "_kv_fixed_address_pool_available",
    "_initialize_gpu_kv_manager_fixed_size", "_bind_gpu_paged_kv_manager",
    "_deployment_context_bound_tokens", "_make_page_table_capacity_request",
    "_with_cuda_graph_page_table_capacity", "_init_gpu_kv_with_actual_size",
    "_finalize_gpu_kv_pool_after_capture", "_get_cuda_graph_gpu_manager",
    "_release_suspended_decode_kv_pool", "_remap_reused_kv_pool",
    "_assert_kv_pool_addresses_are_fixed", "_assert_kv_region_mapped",
    "_assert_kv_region_released",
}


def method_node(name):
    cls = next(node for node in ast.parse(SOURCE.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "KVCapacityEngine")
    return copy.deepcopy(next(node for node in cls.body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


def load_kv_capacity(namespace=None):
    namespace = dict(namespace or {})
    for name, value in dict(
        __name__=__name__, dataclass=dataclasses.dataclass, gc=gc,
        logging=logging, time=time, KVCacheManager=KVCacheManager,
        PageTableCapacityRequest=PageTableCapacityRequest,
        GPU_KV_POOL_HEADROOM_GB=GPU_KV_POOL_HEADROOM_GB,
        provisional_pool_gb=provisional_pool_gb, _GPU_KV_CACHE_SIZE_OVERRIDE=None,
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
            elif node.name.startswith("GPUKV") or node.name in {"DeploymentContext", "PageTableContext"}:
                nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace
