"""Load the D7.1 definitions without importing the native/CUDA graph.

``batchgen.worker.prefill_engine`` imports ``batchgen.native_loader``, which
refuses to load without the compiled core engine and UCX, so this module
compiles the pieces under test straight out of the engine's AST: the frozen
context classes, the module-scope probe-key literals (their only reader is
this engine, which is why they live there) and the converted methods. The
sibling of ``tests/kv_capacity_source.py``, and the same contract: the test
fails when the real body changes, and re-implements nothing.
"""

import ast
import copy
import logging
import math
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1] / "batchgen/worker/prefill_engine.py"
CONTEXT_CLASSES = {
    "PrefillPoolParallelContext",
    "PrefillPrefixPoolEnabledContext",
    "PrefillPoolPageBytesContext",
    "PrefillPoolStateSlotsContext",
    "PoolSegmentSidContext",
    "PrefillPoolWaveContext",
    "PrefillModelPassCountContext",
    "PoolProbeContext",
}
# The D7.1 names: the small pure readers and predicates of the prefill engine.
# Two of them (`_prefix_rank_packing_mode`, `_prefix_rank_cost_profile_disabled`)
# read no worker state at all, so they take no context either.
D7_1_METHODS = {
    "_prefill_model_pass_count",
    "_prefill_prefix_pool_enabled",
    "_pool_prefill_group_safe",
    "_prefill_pool_group",
    "_prefill_pool_free_state_slots",
    "_next_pool_segment_sid",
    "_prefix_rank_packing_mode",
    "_prefix_rank_cost_profile_disabled",
    "_prefill_pool_psm_layout",
    "_prefill_pool_page_dims",
    "_prefill_pool_page_bytes",
    "_prefill_pool_segment_host_pages",
    "_pool_probe_debug",
    "_pool_pad_poison_enabled",
}
STATELESS_METHODS = {
    "_prefix_rank_packing_mode", "_prefix_rank_cost_profile_disabled",
}
METHODS = set(D7_1_METHODS)


def engine_class_node():
    return next(node for node in ast.parse(SOURCE.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "PrefillEngine")


def engine_method_names():
    """Every method name defined on ``PrefillEngine``.

    A converted method may still call a SIBLING engine method through ``self``
    (the page dims call the PSM layout with the same context, and the pad-poison
    predicate calls the probe read with it); the no-worker guard uses this set
    to tell an engine call apart from a worker attribute.
    """
    return {node.name for node in engine_class_node().body
            if isinstance(node, ast.FunctionDef)}


def method_node(name):
    cls = engine_class_node()
    return copy.deepcopy(next(node for node in cls.body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


def _probe_key_assigns(tree):
    """The module-scope probe-key literals and the tuple that collects them.

    Compiled from the real source rather than restated here, so the loaded
    ``_pool_probe_debug`` scans exactly the keys production scans.
    """
    out = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out.append(node)
        elif target.id == "_POOL_PROBE_DEBUG_KEYS":
            out.append(node)
    return out


def load_prefill(namespace=None):
    namespace = dict(namespace or {})
    for name, value in dict(
        __name__=__name__, logging=logging, math=math, torch=torch,
    ).items():
        namespace.setdefault(name, value)
    tree = ast.parse(SOURCE.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
            # The contexts are frozen dataclasses; the import is the engine's.
            nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.name == "PrefillEngine":
                node.body = [member for member in node.body
                             if isinstance(member, ast.FunctionDef)
                             and member.name in METHODS]
                nodes.append(node)
            elif node.name in CONTEXT_CLASSES:
                nodes.append(node)
    nodes = nodes[:1] + _probe_key_assigns(tree) + nodes[1:]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace
