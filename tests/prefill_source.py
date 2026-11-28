"""Load the converted prefill definitions without the native/CUDA graph.

``batchgen.worker.prefill_engine`` imports ``batchgen.native_loader``, which
refuses to load without the compiled core engine and UCX, so this module
compiles the pieces under test straight out of the engine's AST: the frozen
context and outcome classes, the module-scope probe-key literals (their only
reader is this engine, which is why they live there), the module-scope
outcome sentinel and the converted methods. The sibling of
``tests/kv_capacity_source.py``, and the same contract: the test fails when
the real body changes, and re-implements nothing.
"""

import ast
import copy
import hashlib
import json
import logging
import math
import os
import time
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
    # D7.2: the prefix-planning contexts.
    "PrefixRankCostProfileContext",
    "MaterializedPrefixHitsContext",
    "PrefixLookupReplacementContext",
    "TrajectoryRestoreSettleContext",
    # D7.3: the pool construction and layout contexts.
    "PrefillPrefixKVBudgetContext",
    "PrefillPoolConstructionContext",
    "PoolPlanIdentityContext",
    "PrefillPoolLayoutContext",
    "PrefillPoolHostPageContext",
    "MaterializedSegmentStagingContext",
    "MaterializedIndexKeyLayersContext",
    "MaterializedSegmentLoadContext",
    # D7.4: wave planning and instrumentation. Three new contexts; the three
    # probe recorders reuse D7.1's `PoolProbeContext` above.
    "PrefillMicroBatchPlanContext",
    "PrefillPoolWavePlanContext",
    "UnshardedEpilogueGatherContext",
    "PrefillPoolWaveFinishContext",
    # D7.5: the execution paths. Four new contexts, no reuse -- no two of the
    # four read sets are equal.
    "PrefillBatchForwardContext",
    "PrefillModelPassContext",
    "PrefillTokenWritebackContext",
    "PrefillPoolWaveRunContext",
    # D7.6: the unpooled prepacked driver's own context. One new context, no
    # reuse: its eighteen fields are no earlier context's read set.
    "PrefillPrepackedRunContext",
}
# The frozen outcomes this engine declares: a converted body hands its worker
# writes back in one of these for the delegator to apply.
OUTCOME_CLASSES = {
    "PrefixRankCostProfileCacheOutcome",
    "PrefillPrefixKVBudgetOutcome",
    "PrefillPoolBaseOutcome",
    "PrefillPoolLayoutOutcome",
    "PrefillPoolWaveFinishOutcome",
}
# Module-scope values the loaded definitions close over, compiled from the
# real source so a test can compare against the engine's own object identity.
MODULE_SENTINELS = {"PREFILL_UNCHANGED"}
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
# The D7.2 names: the prefix-planning operations of the prefill engine.
# `_unmaterializable_frontier` reads no worker state at all, so like the two
# launch-value predicates above it takes no context either.
D7_2_METHODS = {
    "_cached_prefix_rank_cost_profile",
    "_materialized_prefix_hits",
    "_unmaterializable_frontier",
    "_replace_prefix_lookup_hits",
    "_settle_trajectory_restore_plan",
}
# The D7.3 names: pool construction and layout. `_prefix_pool_index_key_carrier`
# reads no worker state at all, so like the three predicates above it takes no
# context either.
D7_3_METHODS = {
    "_measure_prefill_prefix_kv_budget",
    "_construct_prefill_pool",
    "_check_pool_plan_identical",
    "_prefill_pool_layout",
    "_allocate_prefill_pool_host_pages",
    "_materialized_segment_loader",
    "_prefix_pool_index_key_carrier",
    "_materialized_pool_index_key_layers",
    "_load_materialized_segments",
}
# The D7.4 names: wave planning and the pooled instrumentation. Every one of
# the seven reads worker state, so none of them is stateless.
D7_4_METHODS = {
    "_plan_prefill_micro_batches",
    "_plan_prefill_pool_wave",
    "_finish_prefill_pool_wave",
    "_new_pool_epilogue_tracer",
    "_record_unsharded_epilogue_gather",
    "_new_pool_kernel_dump_recorder",
    "_new_pool_offload_check_recorder",
}
# The D7.5 names: the execution paths. All four read worker state, so none of
# them is stateless, and none of them writes a worker attribute, so none has an
# outcome -- the first prefill stage since D7.2 with neither a collective nor
# an outcome.
D7_5_METHODS = {
    "prefill",
    "_prefill_model_pass",
    "_write_prefill_tokens",
    "_prefill_prepacked_pool",
}
# The D7.6 name: the unpooled prepacked driver, alone in its stage. It reads
# worker state, so it is not stateless, and its ONE store is an in-place
# mutation of the model the context carries, so it has no outcome.
D7_6_METHODS = {
    "prefill_prepacked",
}
STATELESS_METHODS = {
    "_prefix_rank_packing_mode", "_prefix_rank_cost_profile_disabled",
    "_unmaterializable_frontier", "_prefix_pool_index_key_carrier",
}
# Every converted operation that hands a worker write back in an outcome.
OUTCOME_METHODS = {
    "_cached_prefix_rank_cost_profile",
    "_measure_prefill_prefix_kv_budget",
    "_construct_prefill_pool",
    "_prefill_pool_layout",
    "_finish_prefill_pool_wave",
}
# The three bodies that issue a process-group collective, and the collective
# each one issues. Pinned here so the position tests and the stage scan read
# one declaration: a rank that issued these in a different order would
# deadlock the serve group. D7.4's wave planner issues the FIRST of the two a
# pooled wave makes -- its budget reduction -- and then calls the plan-digest
# comparison, so the pair's order is planner-reduce then identity-gather.
COLLECTIVE_METHODS = {
    "_construct_prefill_pool": "all_reduce",
    "_check_pool_plan_identical": "all_gather",
    "_plan_prefill_pool_wave": "all_reduce",
}
METHODS = (
    set(D7_1_METHODS) | set(D7_2_METHODS) | set(D7_3_METHODS)
    | set(D7_4_METHODS) | set(D7_5_METHODS) | set(D7_6_METHODS)
)


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
    """The module-scope probe keys, their tuple, and the outcome sentinel.

    Compiled from the real source rather than restated here, so the loaded
    ``_pool_probe_debug`` scans exactly the keys production scans and the
    loaded cost-profile lookup returns the engine's own "nothing added"
    sentinel object.
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
        elif target.id == "_POOL_PROBE_DEBUG_KEYS" or target.id in MODULE_SENTINELS:
            out.append(node)
    return out


def load_prefill(namespace=None):
    namespace = dict(namespace or {})
    for name, value in dict(
        __name__=__name__, logging=logging, math=math, torch=torch,
        # D7.3: the plan digest is hashed with the real blake2b, so a test
        # compares the engine's own digest rather than a restated one.
        # ``dist`` stays caller-supplied: a collective has to be observed.
        hashlib=hashlib,
        # D7.4: the micro-batch planner reads the real L2-balance switch.
        os=os,
        # D7.5: the pooled driver times its wave and serialises its metrics
        # line with the real modules. ``tqdm`` stays caller-supplied, like
        # ``dist``: a progress bar over a fake plan has to be observed.
        time=time,
        json=json,
        # D7.6: the prepacked driver reads its two memory-profile switches off
        # the real environment, like D7.4's L2-balance switch above
        # (``os`` is already supplied for that one).
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
            elif node.name in CONTEXT_CLASSES | OUTCOME_CLASSES:
                nodes.append(node)
    nodes = nodes[:1] + _probe_key_assigns(tree) + nodes[1:]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace
