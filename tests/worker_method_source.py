"""Load individual ``BatchGenWorker`` methods without importing the worker.

``batchgen.batchgen_worker`` pulls in triton, UCX and the CUDA kernel
extensions at import time, so it cannot be imported on a CPU-only machine.
Several worker methods are nonetheless pure host logic that must be unit
tested, and copying them into a test would mean the test keeps passing after
the real method changes.

This loads the method **from the real source file** instead: the module is
parsed with ``ast``, the requested ``FunctionDef`` nodes are lifted out of the
class body, and each is compiled on its own into a namespace the caller
supplies. The result is a plain function whose first parameter is ``self``, so
a test drives it with any stub object that carries the attributes that method
actually touches.

The test therefore fails if the real method's logic changes, which is the whole
point; it does not re-implement anything.
"""

from __future__ import annotations

import ast
import copy
import pathlib
from typing import Any, Dict, Iterable, Optional

WORKER_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
)
WORKER_CLASS = "BatchGenWorker"
PREFILL_ENGINE_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "prefill_engine.py"
)
PREFILL_ENGINE_CLASS = "PrefillEngine"
RANK_SYNC_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "rank_sync_engine.py"
)
RANK_SYNC_CLASS = "RankSyncEngine"
STEP_OBSERVABILITY_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "step_observability_engine.py"
)
STEP_OBSERVABILITY_CLASS = "StepObservabilityEngine"
WORKER_LIFECYCLE_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "worker_lifecycle_engine.py"
)
WORKER_LIFECYCLE_CLASS = "WorkerLifecycleEngine"
KV_CAPACITY_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "kv_capacity_engine.py"
)
KV_CAPACITY_CLASS = "KVCapacityEngine"
GRAPH_RUNTIME_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "graph_runtime.py"
)
GRAPH_RUNTIME_CLASS = "GraphRuntime"
# The worker-seam engines (S8, S7, S9, S10a, repatriation) whose methods the
# worker reaches through same-name canonical delegators. Only an engine method
# in the delegated shape ``(self, worker, ...)`` or ``(self, ctx, ...)`` is a
# delegator; engines that also carry their own state API (GraphRuntime, and
# since D2 the StepObservabilityEngine capture/clock methods) keep those
# methods out of the delegation scan via that shape filter.
DELEGATED_ENGINES = (
    (RANK_SYNC_SOURCE, RANK_SYNC_CLASS),
    (STEP_OBSERVABILITY_SOURCE, STEP_OBSERVABILITY_CLASS),
    (WORKER_LIFECYCLE_SOURCE, WORKER_LIFECYCLE_CLASS),
    (KV_CAPACITY_SOURCE, KV_CAPACITY_CLASS),
    (PREFILL_ENGINE_SOURCE, PREFILL_ENGINE_CLASS),
    (GRAPH_RUNTIME_SOURCE, GRAPH_RUNTIME_CLASS),
)
DECODE_ENTRY_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "admission"
    / "decode_entry.py"
)
PREFILL_SELECTION_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "admission"
    / "prefill_selection.py"
)
# Admission modules whose MODULE-LEVEL helpers hold worker method bodies in
# the frozen-context shape ``(ctx, ...)``. The worker keeps a same-name
# delegator for each, exactly like the class-based engines above.
DELEGATED_CONTEXT_MODULES = (DECODE_ENTRY_SOURCE, PREFILL_SELECTION_SOURCE)


# D7.1: launch values a prefill delegator resolves for an operation that reads
# NO worker state at all, so it takes no context either -- the value IS its
# whole input. Named per method, because the parameter names (`mode`, `value`)
# are ordinary locals in other bodies and must not be rewritten there.
STATELESS_DELEGATED_VALUES = {
    "_prefix_rank_packing_mode": {
        "mode": "self.args.prefix_rank_assignment_mode",
    },
    "_prefix_rank_cost_profile_disabled": {
        "value": "self.args.prefix_rank_cost_profile",
    },
}
# Every converted operation that takes NO seam parameter. D7.2 adds one whose
# delegator resolves nothing for it either -- the frontier predicate reads its
# whole input off its own arguments -- so it has no entry above and its
# worker-shaped view is the body itself under the delegator's signature. D7.3
# adds the index-key carrier, which reads both of its values off the geometry
# it is handed.
STATELESS_CONVERTED = frozenset(STATELESS_DELEGATED_VALUES) | {
    "_unmaterializable_frontier",
    "_prefix_pool_index_key_carrier",
}


def _is_delegator_shape(node: ast.FunctionDef) -> bool:
    args = node.args.args
    if node.name == "_setup_cuda_graphs" and [a.arg for a in args] == ["ctx", "gpu_manager"]:
        return True
    if node.name in STATELESS_CONVERTED:
        return True
    return (
        len(args) >= 2
        and args[0].arg == "self"
        and args[1].arg in {"worker", "ctx"}
    )


def _context_attr_worker_expr(node: ast.AST) -> Optional[ast.AST]:
    """Map a converted engine context path back to the worker spelling."""
    text = ast.unparse(node)
    paths = {
        "ctx.facts.rank": "self.rank",
        "ctx.facts.world_size": "self.world_size",
        "ctx.facts.local_rank": "self.local_rank",
        "ctx.facts.local_world_size": "self.local_world_size",
        "ctx.facts.rank_ctx": "self.rank_ctx",
        "ctx.facts.device": "self.torch_device",
        "ctx.facts.page_size": "self.PAGE_SIZE",
        "ctx.facts.decision_interval": "self.DECISION_INTERVAL",
        "ctx.facts.enable_prefix_cache": "self.enable_prefix_cache",
        "ctx.facts.enable_host_kv_eviction": "self.enable_host_kv_eviction",
        "ctx.facts.host_kv_chunk_size": "self.host_kv_chunk_size",
        "ctx.sched.pool": "self.global_batch",
        "ctx.sched.seq_table": "self.seq_table",
        "ctx.sched.seq_table.uuid_to_local": "self._uuid_to_local_map",
        "ctx.sched.seq_table.local_to_uuid": "self._local_to_uuid_map",
        "ctx.sched.trajectories": "self._traj",
        "ctx.sched.prefix": "self._prefix",
        "ctx.gpu_kv": "self.gpu_paged_kv_cache_manager",
        "ctx.host_view": "getattr(self, 'host_paged_kv_worker_view', None)",
        "ctx.host_view_aux": "getattr(self, 'host_paged_kv_worker_view_aux', None)",
        "ctx.core_engine": "self.core_engine",
        "ctx.gpu_resident": "self._sequences_with_gpu_kv",
        "ctx.prefix_coordinator": "self.prefix_cache_coordinator",
        "ctx.prefix_runtime": "self.prefix_cache_runtime_config",
        "ctx.chunk_sizer": "self.adaptive_chunk_sizer",
        "ctx.facts.prefix_cache_debug_stats": "self.prefix_cache_debug_stats",
        "ctx.facts.kv_stats_native": "self._kv_stats_native",
        "ctx.facts.kv_stats_compare": "self._kv_stats_compare",
        "ctx.facts.shared_prefix_admission": "self.shared_prefix_admission",
        "ctx.parallel_manager": "self.parallel_manager",
        # PrefillPhaseContext (admission.prefill_selection)
        "ctx.prefix": "self._prefix",
        # D6.1/D6.2 KV capacity value contexts: rank identity and the launch
        # objects each context carries by reference.
        "ctx.rank": "self.rank",
        "ctx.local_rank": "self.local_rank",
        "ctx.world_size": "self.world_size",
        "ctx.local_world_size": "self.local_world_size",
        "ctx.engine_config": "self.engine_config",
        "ctx.args": "getattr(self, 'args', None)",
        "ctx.huggingface_ckpt_name": "self.huggingface_ckpt_name",
        "ctx.model_context_length": "self.model_context_length",
        "ctx.traj": "self._traj",
        "ctx._deployment_length_verified": (
            "getattr(self, '_deployment_length_verified', False)"
        ),
        # HostKVHealthContext: the health line needs this node's id only.
        "ctx.node_id": "self.rank_ctx.node_id",
        # D6.3 HostKVWatermarkContext: the watermark pair's launch flags and
        # the per-rank log counter, which the pre-move code read with hasattr.
        "ctx.enable_decode_preemption": "self.enable_decode_preemption",
        "ctx.enable_host_kv_eviction": "self.enable_host_kv_eviction",
        "ctx.host_kv_watermark": "self.host_kv_watermark",
        "ctx.pool": "self.global_batch",
        "ctx._watermark_check_counter": (
            "getattr(self, '_watermark_check_counter', None)"
        ),
        # D6.4 DeferredKVFlushContext: the decode forward's handoff channel,
        # which the pre-move flush read with the same getattr defaults.
        "ctx.deferred_entries": "getattr(self, '_deferred_kv_entries', [])",
        "ctx.deferred_entries_aux": (
            "getattr(self, '_deferred_kv_entries_aux', [])"
        ),
        "ctx.deferred_worker_view": (
            "getattr(self, '_deferred_kv_worker_view', None)"
        ),
        "ctx.deferred_batch": "getattr(self, '_deferred_kv_batch', None)",
        "ctx.deferred_worker_view_aux": (
            "getattr(self, '_deferred_kv_worker_view_aux', None)"
        ),
        # D6.4 AuxLayerSlotsContext: the aux-slot resolution's two reads, both
        # of which the pre-move body took with a getattr default.
        "ctx.aux_coordinator": (
            "getattr(self, 'gpu_paged_kv_cache_manager', None)"
        ),
        "ctx._aux_layers_with_slots_cache": (
            "getattr(self, '_aux_layers_with_slots_cache', False)"
        ),
        # D6.5 NodeMemfdIdentityContext and GPUKVDestroyContext: narrow value
        # contexts whose fields are named after the worker attributes they
        # snapshot (the pool handle shares its name with the D6.1 contexts'
        # field, and resolves to the same worker attribute). The residency set
        # already maps above, under the decode phase context's own field name.
        "ctx.torch_device": "self.torch_device",
        "ctx.gpu_paged_kv_cache_manager": "self.gpu_paged_kv_cache_manager",
        # D7.1 prefill value contexts. `parallel_manager`, `engine_config`,
        # `prefix_runtime`, `pool`, `prefix`, `traj`, `rank` and
        # `huggingface_ckpt_name` already map above, onto the same worker
        # attributes these contexts snapshot.
        "ctx.enable_prefix_cache": "self.enable_prefix_cache",
        "ctx.enable_prepack": "self.enable_prepack",
        "ctx.local_to_uuid": "self._local_to_uuid_map",
        "ctx._prefill_pool_wave": "self._prefill_pool_wave",
        "ctx._phase_prefill_debug": "self._phase_prefill_debug",
        # The pre-move body read this diagnostic latch with a getattr default,
        # at both of its occurrences.
        "ctx._prefix_integrity": "getattr(self, '_prefix_integrity', False)",
        # D7.2 prefix-planning contexts. `rank`, `world_size`, `traj`,
        # `prefix`, `prefix_runtime` and `prefix_coordinator` already map
        # above, onto the same worker attributes these contexts snapshot.
        "ctx.model_name": "self.model_name",
        # D7.3 pool construction and layout contexts. `rank`, `local_rank`,
        # `torch_device`, `engine_config`, `huggingface_ckpt_name`, `prefix`
        # and `_prefill_pool_wave` already map above, onto the same worker
        # attributes these contexts snapshot.
        "ctx.model": "self.model",
        "ctx._prefill_prefix_kv_budget_bytes": (
            "self._prefill_prefix_kv_budget_bytes"
        ),
        "ctx._prefill_pool_page_budget": "self._prefill_pool_page_budget",
        # The pre-move layout walk read its cache with a getattr default: the
        # attribute does not exist until a prefill measured the layout.
        "ctx._prefill_pool_layout_cache": (
            "getattr(self, '_prefill_pool_layout_cache', None)"
        ),
        # D7.5 execution-path contexts. `rank`, `local_rank`, `world_size`,
        # `torch_device`, `engine_config`, `model`, `parallel_manager`, `pool`,
        # `local_to_uuid` and `_prefill_pool_wave` already map above, onto the
        # same worker attributes these contexts snapshot. The three new ones
        # are plain reads: the worker binds all three in ``__init__``, so the
        # pre-move bodies read them without a presence guard.
        "ctx.model_config": "self.model_config",
        "ctx._weight_passes": "self._weight_passes",
        "ctx._prefill_pool_base": "self._prefill_pool_base",
    }
    if text == "ctx.facts.attn_tp_size":
        return ast.parse("self._decode_attn_tp_size()", mode="eval").body
    replacement = paths.get(text)
    if replacement is None:
        return None
    return ast.parse(replacement, mode="eval").body


class _ContextToWorker(ast.NodeTransformer):
    def visit_Attribute(self, node):
        # Longest match first: a full context path (`ctx.sched.seq_table
        # .uuid_to_local`) must map as a whole before its inner attribute
        # (`ctx.sched.seq_table`) is rewritten out from under it.
        replacement = _context_attr_worker_expr(node)
        if replacement is not None:
            return replacement
        node = self.generic_visit(node)
        return _context_attr_worker_expr(node) or node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            replacements = {
                "query_book": "self.query_book",
                "max_rank_bsz": "getattr(self, '_decode_padding_bsz', 0) or 0",
                "pending_tasks": "self._pending_kv_append_tasks",
                "pending_tensors": "getattr(self, '_pending_kv_append_tensors', None)",
                "deferred_errors": "getattr(self, '_deferred_kv_append_wait_errors', [])",
                "decode_local_count_tensor": "self._decode_local_count_tensor",
                "decode_all_rank_counts": "self._decode_all_rank_counts",
                "previous_max": "getattr(self, '_current_decode_max_rank_batch_size', 0) or 0",
                # D2 step-observability value parameters: the worker's
                # delegator resolves these beside the frozen context, so the
                # worker spelling is the delegator's own resolution.
                "decode_model": "getattr(self, 'model', None)",
                "phase_mem_switch_index": "getattr(self, '_phase_mem_switch_index', 0)",
                "graph_state": "self._graph",
                "persistent_phase": "self._persistent_phase_enabled()",
                "profile_enabled": (
                    "self._debug_flag_enabled((self._debug_values() or {})"
                    ".get('glm5_prefill_nsys_profile'))"
                ),
                # D6.2 keyword-only collaborators: the worker spelling of a
                # collaborator callable is the worker's own method of that name.
                "_decode_attn_tp_size": "self._decode_attn_tp_size",
                "_whole_model_decode_graphs_requested": (
                    "self._whole_model_decode_graphs_requested"
                ),
                "_host_views_by_group": "self._host_views_by_group",
                # D6.3 keyword-only collaborators and value parameters: the
                # worker spelling of a collaborator callable is the worker's
                # own method of that name, and of a value parameter the
                # resolution its delegator performs.
                "_is_sequence_completed": "self._is_sequence_completed",
                "_get_host_kv_utilization": "self._get_host_kv_utilization",
                "_credit_reclaimable_prefix_pages": (
                    "self._credit_reclaimable_prefix_pages"
                ),
                "_get_or_create_gloo_group": "self._get_or_create_gloo_group",
                "_execute_single_kv_migration": (
                    "self._execute_single_kv_migration"
                ),
                "_execute_kv_migrations_parallel": (
                    "self._execute_kv_migrations_parallel"
                ),
                "_plan_kv_migration": "self._plan_kv_migration",
                "_sync_sequence_metadata": "self._sync_sequence_metadata",
                "_bump_decode_row_epoch": "self._bump_decode_row_epoch",
                "_trajectory_tokens": "self._trajectory_tokens",
                "_trajectory_slot": "self._trajectory_slot",
                "enable_decode_preemption": "self.enable_decode_preemption",
                # The rebalance delegator creates the staging map before the
                # engine runs; the worker-shaped view keeps the hasattr-safe
                # read, which is what the pre-move statement did and what a
                # shell that never received anything still needs.
                "pending_migrated_query_book": (
                    "getattr(self, '_pending_migrated_query_book', None)"
                ),
                "logged_migration_skip_tp": (
                    "getattr(self, '_logged_migration_skip_tp', False)"
                ),
                # D6.4 keyword-only collaborators and value parameters. The
                # append channel's task list is handed over by reference, and
                # the pre-move bodies named it without a hasattr guard.
                "_ensure_host_kv_append_capacity": (
                    "self._ensure_host_kv_append_capacity"
                ),
                "_aux_layers_with_slots": "self._aux_layers_with_slots",
                "_wait_pending_kv_append_tasks": (
                    "self._wait_pending_kv_append_tasks"
                ),
                "_finalize_async_load_minimal": (
                    "self._finalize_async_load_minimal"
                ),
                # D6.5 keyword-only collaborators: each is the worker's own
                # method of that name, which is what the pre-move body called.
                "_prefill_pool_segment_host_pages": (
                    "self._prefill_pool_segment_host_pages"
                ),
                "_owns_host_kv": "self._owns_host_kv",
                "_assign_decode_dp_groups": "self._assign_decode_dp_groups",
                "_prepare_decode_batch": "self._prepare_decode_batch",
                "_update_batch_status": "self._update_batch_status",
                # D7.1 keyword-only collaborators and the one value a prefill
                # delegator resolves for every converted reader that takes it.
                # Each collaborator is the worker's own method of that name,
                # which is what the pre-move body called.
                "_plan_prefill_micro_batches": (
                    "self._plan_prefill_micro_batches"
                ),
                "_pool_prefill_group_safe": "self._pool_prefill_group_safe",
                "_prefill_pool_layout": "self._prefill_pool_layout",
                "_prefill_pool_page_dims": "self._prefill_pool_page_dims",
                "_debug_flag_enabled": "self._debug_flag_enabled",
                "attn_group_size": "self._prefill_attn_tp_size()",
                # D7.2 keyword-only collaborators, each the worker's own
                # method of that name -- what the pre-move body called.
                "_trajectory_commit_requested": (
                    "self._trajectory_commit_requested"
                ),
                "_unmaterializable_frontier": (
                    "self._unmaterializable_frontier"
                ),
                # D7.3 keyword-only collaborators, each the worker's own
                # method of that name -- what the pre-move body called. The
                # names the outcomes carry back are NOT here: those are write
                # locals, restored by the outcome rule instead.
                "_prefill_prefix_pool_enabled": (
                    "self._prefill_prefix_pool_enabled"
                ),
                "_prefill_pool_page_bytes": "self._prefill_pool_page_bytes",
                "_prefill_pool_group": "self._prefill_pool_group",
                "_new_pool_kernel_dump_recorder": (
                    "self._new_pool_kernel_dump_recorder"
                ),
                "_pool_pad_poison_enabled": "self._pool_pad_poison_enabled",
                "_prefill_host_view_is_dual": (
                    "self._prefill_host_view_is_dual"
                ),
                "_prefill_pool_psm_layout": "self._prefill_pool_psm_layout",
                "_next_pool_segment_sid": "self._next_pool_segment_sid",
                # `_host_views_by_group` and `_prefill_pool_segment_host_pages`
                # already map above, under D6.2 and D6.5.
                "_materialized_pool_layers": "self._materialized_pool_layers",
                "_materialized_pool_index_key_layers": (
                    "self._materialized_pool_index_key_layers"
                ),
                # D7.4 keyword-only collaborators and the second value a
                # prefill delegator resolves. Each collaborator is the
                # worker's own method of that name -- what the pre-move body
                # called. `_trajectory_tokens`, `_prefill_pool_group`,
                # `_prefill_pool_layout`, `_prefill_pool_psm_layout`,
                # `_owns_host_kv`, `_host_views_by_group` and
                # `_debug_flag_enabled` already map above.
                "_prefix_materialization_kv_config": (
                    "self._prefix_materialization_kv_config"
                ),
                "_prefill_pool_free_state_slots": (
                    "self._prefill_pool_free_state_slots"
                ),
                "_materialized_prefix_hits": "self._materialized_prefix_hits",
                "_check_pool_plan_identical": (
                    "self._check_pool_plan_identical"
                ),
                "_replace_prefix_lookup_hits": (
                    "self._replace_prefix_lookup_hits"
                ),
                "_settle_trajectory_restore_plan": (
                    "self._settle_trajectory_restore_plan"
                ),
                # The server mode's ONE prefill cap, which all four prefill
                # delegators that hand it over resolve the same way. Safe in
                # this global map, like `attn_group_size` above: no converted
                # body uses the name as a local, and no worker delegator of a
                # converted method declares a parameter of this name --
                # `prefill_prepacked` and `_config_prefill_for_batch` (D7.6,
                # D7.7) resolve it in their delegators too.
                "token_cap": "self._server_prefill_token_cap()",
                # D7.5 keyword-only collaborators, each the worker's own
                # method of that name -- what the pre-move body called.
                # `_trajectory_tokens` already maps above. `_construct_prefill_pool`,
                # `_new_pool_epilogue_tracer`, `_materialized_segment_loader`
                # and `_load_materialized_segments` are converted operations
                # the pooled driver reaches through the worker's same-name
                # delegator, which is the spelling the pre-move body used.
                # The shared model pass is NOT here: its pre-move spelling in
                # the pooled driver is the direct engine call, restored by
                # ``_restore_pool_model_pass_call`` below.
                "_bind_prefill_host_views": "self._bind_prefill_host_views",
                "_prefill_deepseek_family_name": (
                    "self._prefill_deepseek_family_name"
                ),
                "feed_watchdog": "self.feed_watchdog",
                "_bind_prefill_step_metadata": (
                    "self._bind_prefill_step_metadata"
                ),
                "_bind_prefill_step_batch": "self._bind_prefill_step_batch",
                "_select_tokens": "self._select_tokens",
                "_append_trajectory_batch": "self._append_trajectory_batch",
                "_should_stop_at_eos": "self._should_stop_at_eos",
                "_append_trajectory_token": "self._append_trajectory_token",
                "_record_unsharded_epilogue_gather": (
                    "self._record_unsharded_epilogue_gather"
                ),
                "_construct_prefill_pool": "self._construct_prefill_pool",
                "_new_pool_epilogue_tracer": (
                    "self._new_pool_epilogue_tracer"
                ),
                "_bind_prefill_pool_segments": (
                    "self._bind_prefill_pool_segments"
                ),
                "_bind_prefill_pool_layer_state": (
                    "self._bind_prefill_pool_layer_state"
                ),
                "_materialized_segment_loader": (
                    "self._materialized_segment_loader"
                ),
                "_install_prefill_pool_rows": (
                    "self._install_prefill_pool_rows"
                ),
                "_load_materialized_segments": (
                    "self._load_materialized_segments"
                ),
                "_release_kda_state_slots": "self._release_kda_state_slots",
                "_drop_prefill_pool_offload_recorder": (
                    "self._drop_prefill_pool_offload_recorder"
                ),
                "_unbind_prefill_pool_segments": (
                    "self._unbind_prefill_pool_segments"
                ),
                "_write_prefill_tokens": "self._write_prefill_tokens",
            }
            replacement = replacements.get(node.id)
            if replacement is not None:
                return ast.parse(replacement, mode="eval").body
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "IndexManager"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == "IndexLookupRequest"
        ):
            # The engine builds the lookup request from the context; the worker
            # spelling of the same lookup is its index helper of that name.
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_" + node.func.attr,
                    ctx=ast.Load(),
                ),
                args=[node.args[1]],
                keywords=[],
            )
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id
            in {"StepObservabilityEngine", "KVCapacityEngine"}
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "ctx"
        ):
            # A cross-engine call made with the same context; the worker
            # spelling is the worker's same-name delegator.
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=node.func.attr,
                    ctx=ast.Load(),
                ),
                args=node.args[1:],
                keywords=node.keywords,
            )
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _context_module_function_names()
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "ctx"
        ):
            # A sibling context helper of the same admission module; the
            # worker spelling is the worker's same-name delegator.
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=node.func.id,
                    ctx=ast.Load(),
                ),
                args=node.args[1:],
                keywords=node.keywords,
            )
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "ctx"
        ):
            node.args.pop(0)
            # A sibling called with the same context is also handed the
            # collaborators it was given; the worker's same-name delegator
            # resolves those itself, so they are not part of its call. The
            # same holds for a VALUE the sibling's delegator resolves (D6.5:
            # the capacity vote's per-sequence KDA record pages), which the
            # pre-move body likewise never passed.
            resolved = set(DELEGATOR_EXTRA_VALUE_ARGS.get(node.func.attr, ()))
            node.keywords = [
                kw for kw in node.keywords
                if not (
                    kw.arg
                    and (
                        (
                            kw.arg.startswith("_")
                            and ast.unparse(kw.value) == f"self.{kw.arg}"
                        )
                        or kw.arg in resolved
                    )
                )
            ]
        return node


# D6.5: an engine method may take a VALUE parameter its worker delegator
# resolves and therefore does not declare. The pre-move body read that value
# as a parameter of its own, so the worker-shaped view keeps the parameter
# (with its default) after adopting the delegator's arguments, and a sibling
# call that forwards it drops the keyword the delegator would have supplied.
DELEGATOR_EXTRA_VALUE_ARGS = {
    "_collective_prefill_host_capacity_ok": ("kda_record_pages_per_sequence",),
}


def _extra_value_args(args: ast.arguments, names) -> list:
    """``(arg, default)`` for each named positional parameter of a signature.

    Defaults bind to the LAST ``len(defaults)`` positional parameters, so the
    pairing runs from the tail. A named parameter without a default would
    change the worker-shaped signature's call contract, so it is refused.
    """
    wanted = set(names)
    offset = len(args.args) - len(args.defaults)
    paired = []
    for index, arg in enumerate(args.args):
        if arg.arg not in wanted:
            continue
        if index < offset:
            raise AssertionError(
                f"{arg.arg} is a delegator-resolved value without a default"
            )
        paired.append(
            (copy.deepcopy(arg), copy.deepcopy(args.defaults[index - offset]))
        )
    return paired


# D6.5: the last two converted bodies read a worker attribute through a
# pre-move PRESENCE guard -- the hold's ``hasattr(...) and ...`` pool test and
# the capacity vote's ``getattr(..., None)`` core-engine fallback. A frozen
# context always carries the field, so the engine spells both as a plain read;
# the worker-shaped view puts the original guard back. ``limit`` is how many
# leading occurrences carried the guard: the hold's later
# ``mgr = self.gpu_paged_kv_cache_manager`` read did not, so only the first
# one is restored. ``None`` restores every occurrence.
GUARDED_CONTEXT_READS = {
    "_put_sequences_on_hold": (
        (
            "ctx.gpu_kv",
            "hasattr(self, 'gpu_paged_kv_cache_manager') "
            "and self.gpu_paged_kv_cache_manager",
            1,
        ),
    ),
    "_collective_prefill_host_capacity_ok": (
        ("ctx.core_engine", "getattr(self, 'core_engine', None)", None),
    ),
}


class _GuardedContextReadToWorker(ast.NodeTransformer):
    """Restore the pre-move presence guard around a context read.

    Runs BEFORE :class:`_ContextToWorker`, so the context path is still
    spelled as the engine wrote it. Traversal is depth-first in field order,
    which is source order, so a ``limit`` counts the leading occurrences.
    """

    def __init__(self, guards) -> None:
        self._guards = {
            path: [restored, limit] for path, restored, limit in guards
        }

    def visit_Attribute(self, node):
        guard = self._guards.get(ast.unparse(node))
        if guard is not None and (guard[1] is None or guard[1] > 0):
            if guard[1] is not None:
                guard[1] -= 1
            return ast.parse(guard[0], mode="eval").body
        return self.generic_visit(node)


# D6.3: a converted method returns its worker writes in a frozen outcome the
# delegator applies. For the worker-shaped view the write has to be back where
# the pre-move statement stood, so each outcome names the attributes it binds
# and the lift-back puts the locals and the return back into worker spelling.
OUTCOME_WRITE_FIELDS = {
    "WatermarkTriggerOutcome": (
        "_host_kv_page_stats", "_watermark_check_counter",
    ),
    "MigrationPlanOutcome": ("_logged_migration_skip_tp",),
    "DeferredKVFlushOutcome": (
        "_deferred_kv_entries", "_deferred_kv_entries_aux",
        "_deferred_kv_batch", "_deferred_kv_worker_view",
        "_deferred_kv_worker_view_aux",
    ),
    "AuxLayerSlotsOutcome": ("_aux_layers_with_slots_cache",),
    # D7.3: the prefill engine's three plain-value outcomes.
    "PrefillPrefixKVBudgetOutcome": (
        "_prefill_prefix_kv_budget_bytes", "_prefill_pool_page_budget",
    ),
    "PrefillPoolBaseOutcome": ("_prefill_pool_base",),
    "PrefillPoolLayoutOutcome": ("_prefill_pool_layout_cache",),
    # D7.4: the wave epilogue CLEARS the planned wave, twice, and both
    # statements assign the same value, so a plain-value field expresses them.
    "PrefillPoolWaveFinishOutcome": ("_prefill_pool_wave",),
}
OUTCOME_METHODS = {
    "_check_host_kv_watermark_trigger": "WatermarkTriggerOutcome",
    "_plan_kv_migration": "MigrationPlanOutcome",
    "_flush_deferred_kv_to_host": "DeferredKVFlushOutcome",
    "_aux_layers_with_slots": "AuxLayerSlotsOutcome",
    "_measure_prefill_prefix_kv_budget": "PrefillPrefixKVBudgetOutcome",
    "_construct_prefill_pool": "PrefillPoolBaseOutcome",
    "_prefill_pool_layout": "PrefillPoolLayoutOutcome",
    "_finish_prefill_pool_wave": "PrefillPoolWaveFinishOutcome",
}
# The "this path wrote nothing" sentinel each engine declares. The lift-back
# drops a seed assignment of its outcome's own sentinel; naming it per outcome
# keeps one engine's sentinel from standing for another's.
OUTCOME_SENTINEL = {
    "WatermarkTriggerOutcome": "GPU_KV_UNCHANGED",
    "MigrationPlanOutcome": "GPU_KV_UNCHANGED",
    "DeferredKVFlushOutcome": "GPU_KV_UNCHANGED",
    "AuxLayerSlotsOutcome": "GPU_KV_UNCHANGED",
    "PrefillPrefixKVBudgetOutcome": "PREFILL_UNCHANGED",
    "PrefillPoolBaseOutcome": "PREFILL_UNCHANGED",
    "PrefillPoolLayoutOutcome": "PREFILL_UNCHANGED",
    "PrefillPoolWaveFinishOutcome": "PREFILL_UNCHANGED",
}
# Methods whose pre-move body ended WITHOUT a return statement: the outcome
# return added at the tail is seam plumbing, so the worker-shaped view drops
# it rather than keeping a bare ``return`` the original never had. An outcome
# return anywhere else is a real early exit and stays as ``return``.
OUTCOME_IMPLICIT_TAIL_RETURN = {
    "_flush_deferred_kv_to_host",
    "_measure_prefill_prefix_kv_budget",
    "_construct_prefill_pool",
    # The wave epilogue's pre-move body ended with its ``try``/``finally``.
    "_finish_prefill_pool_wave",
}


# D6.4: the pending append-tensor list is CREATED on first use inside the
# launch and appended to afterwards, so the converted bodies take it by
# reference under a local alias named after the attribute, bound once where the
# pre-move body created it. Its worker spelling is that original create block,
# which no field-by-field outcome rule can express -- the same reason the
# migration receive needs its own restore.
_PENDING_TENSORS_ALIAS = "_pending_kv_append_tensors"
_PENDING_TENSORS_CREATE = """
if not hasattr(self, '_pending_kv_append_tensors'):
    self._pending_kv_append_tensors = []
"""


class _PendingTensorAliasToWorker(ast.NodeTransformer):
    """Restore the append-tensor alias to the worker's create-then-append."""

    def visit_Assign(self, node):
        target = node.targets[0]
        if (
            len(node.targets) == 1
            and isinstance(target, ast.Name)
            and target.id == _PENDING_TENSORS_ALIAS
        ):
            return ast.parse(_PENDING_TENSORS_CREATE).body
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == _PENDING_TENSORS_ALIAS and isinstance(node.ctx, ast.Load):
            return ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=_PENDING_TENSORS_ALIAS,
                ctx=ast.Load(),
            )
        return node


def _is_outcome_call(node: ast.AST, outcome: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == outcome
    )


class _OutcomeToWorkerWrites(ast.NodeTransformer):
    """Restore an outcome's locals and return to the worker's own writes.

    An engine body carries each write as a local named after the worker
    attribute, seeded with the sentinel where a branch does not write, and
    hands them back in the outcome. The worker spelling is the attribute
    assignment at that same statement, with the sentinel seed dropped (the
    worker simply does not write) and the outcome stripped from the return.
    """

    def __init__(self, outcome: str):
        self.outcome = outcome
        self.fields = set(OUTCOME_WRITE_FIELDS[outcome])
        self.sentinel = OUTCOME_SENTINEL[outcome]

    def _bound(self, name: str, ctx) -> ast.Attribute:
        return ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()), attr=name, ctx=ctx,
        )

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in self.fields:
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == self.sentinel
            ):
                return None
            node.targets = [self._bound(target.id, ast.Store())]
        return node

    def visit_AugAssign(self, node):
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.target.id in self.fields:
            node.target = self._bound(node.target.id, ast.Store())
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.fields:
            return self._bound(node.id, ast.Load())
        return node

    def visit_Return(self, node):
        node = self.generic_visit(node)
        value = node.value
        if (
            isinstance(value, ast.Tuple)
            and len(value.elts) == 2
            and _is_outcome_call(value.elts[1], self.outcome)
        ):
            return ast.Return(value=value.elts[0])
        if _is_outcome_call(value, self.outcome):
            return ast.Return(value=None)
        return node


# The migration receive's two writes ADD to containers the pre-move code
# created on first use, so the outcome carries the uuid and the record rather
# than the containers. Its worker spelling is therefore the original
# create-then-add block, which no field-by-field rule can express.
_MIGRATION_RECEIVE_WRITES = """
if not hasattr(self, '_pending_migrated_query_book'):
    self._pending_migrated_query_book = {}
if not hasattr(self, '_migrated_sequences'):
    self._migrated_sequences = set()
self._migrated_sequences.add(MIGRATED_UUID)
self._pending_migrated_query_book[MIGRATED_UUID] = PENDING_RECORD
"""


def _restore_migration_receive_writes(method: ast.FunctionDef) -> ast.FunctionDef:
    """Put ``_execute_single_kv_migration``'s staging writes back on the worker."""

    class _Restore(ast.NodeTransformer):
        def visit_Return(self, node):
            if _is_outcome_call(node.value, "KVMigrationReceiveOutcome"):
                return None
            return node

        def generic_visit(self, node):
            node = super().generic_visit(node)
            for field in ("body", "orelse", "finalbody"):
                body = getattr(node, field, None)
                if not isinstance(body, list):
                    continue
                for index, stmt in enumerate(body):
                    if not (
                        isinstance(stmt, ast.Assign)
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == "_migrated_sequences"
                        and index + 1 < len(body)
                        and isinstance(body[index + 1], ast.Assign)
                        and isinstance(body[index + 1].targets[0], ast.Name)
                        and body[index + 1].targets[0].id
                        == "_pending_migrated_query_book"
                    ):
                        continue
                    restored = ast.parse(_MIGRATION_RECEIVE_WRITES).body
                    fill = _FillMigrationReceive(
                        uuid_value=stmt.value,
                        record_value=body[index + 1].value,
                    )
                    body[index:index + 2] = [
                        fill.visit(stmt) for stmt in restored
                    ]
                    break
            return node

    return _Restore().visit(method)


# D7.2: the cost-profile cache is CREATED on first use and then ADDED to, so
# its outcome carries the one entry the write added rather than the container
# (the same reason the migration receive needs its own restore). The engine
# body works on a local dict named after the attribute, seeded from the
# context; its worker spelling is the original create guard, with every use of
# the local back on the worker and the bookkeeping local that carries the
# entry dropped.
_COST_PROFILE_CACHE = "_prefix_rank_cost_profiles"
_COST_PROFILE_ENTRY = "cache_entry"
_COST_PROFILE_CREATE = """
if not hasattr(self, '_prefix_rank_cost_profiles'):
    self._prefix_rank_cost_profiles = {}
"""


def _restore_cost_profile_cache_writes(
    method: ast.FunctionDef,
) -> ast.FunctionDef:
    """Put ``_cached_prefix_rank_cost_profile``'s cache writes back."""

    class _Restore(ast.NodeTransformer):
        def visit_Assign(self, node):
            # The target is inspected BEFORE the children are visited: the
            # rewrite below turns the local's name into an attribute, which
            # would hide a bare-assignment target from this test.
            target = node.targets[0]
            if len(node.targets) == 1 and isinstance(target, ast.Name):
                if target.id == _COST_PROFILE_CACHE:
                    # The seeded local IS the pre-move create guard.
                    return ast.parse(_COST_PROFILE_CREATE).body
                if target.id == _COST_PROFILE_ENTRY:
                    # Seam bookkeeping: the worker wrote the cache directly.
                    return None
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id == _COST_PROFILE_CACHE:
                return ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=_COST_PROFILE_CACHE,
                    ctx=node.ctx,
                )
            return node

        def visit_Return(self, node):
            node = self.generic_visit(node)
            value = node.value
            if (
                isinstance(value, ast.Tuple)
                and len(value.elts) == 2
                and _is_outcome_call(
                    value.elts[1], "PrefixRankCostProfileCacheOutcome"
                )
            ):
                return ast.Return(value=value.elts[0])
            return node

    return _Restore().visit(method)


# D7.5: the pooled driver's ONE engine-internal call. Step 4 of the three-step
# extraction gave both prefill drivers the direct engine spelling for the
# shared model pass -- ``_engine`` captured before the verbatim self-rebinding,
# with the worker handed over as the pass's first argument -- so the
# worker-shaped view of the pooled driver is that spelling, not a ``self.``
# delegator call. D7.5 converted the pass, so it no longer accepts a worker and
# the converted driver reaches it as a collaborator; this restore puts the two
# statements of the pre-move spelling back (the capture, and the call with the
# worker first) so the lift stays character-exact. It is scoped to the one
# method by name: no other converted body in any engine captures the engine
# instance.
_POOL_MODEL_PASS = "_prefill_model_pass"
_POOL_ENGINE_ALIAS = "_engine"


def _restore_pool_model_pass_call(method: ast.FunctionDef) -> ast.FunctionDef:
    """Put ``_prefill_prepacked_pool``'s direct engine call back."""

    class _Restore(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == _POOL_MODEL_PASS:
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=_POOL_ENGINE_ALIAS, ctx=ast.Load()),
                        attr=_POOL_MODEL_PASS,
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id="self", ctx=ast.Load())] + node.args,
                    keywords=node.keywords,
                )
            return node

    method = _Restore().visit(method)
    calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == _POOL_MODEL_PASS
    ]
    if len(calls) != 1:
        raise AssertionError(
            f"_prefill_prepacked_pool: expected one model-pass call, "
            f"found {len(calls)}"
        )
    # The capture was the pre-move body's FIRST statement, i.e. straight after
    # the docstring, which is where the pristine worker-shaped view has it.
    at = 1 if (
        method.body
        and isinstance(method.body[0], ast.Expr)
        and isinstance(method.body[0].value, ast.Constant)
        and isinstance(method.body[0].value.value, str)
    ) else 0
    method.body.insert(at, ast.parse(f"{_POOL_ENGINE_ALIAS} = self").body[0])
    return method


class _StatelessValueToWorker(ast.NodeTransformer):
    """Resolve a stateless operation's value parameters to worker spelling.

    D7.1: two converted predicates read no worker state, so they take no
    context -- only the launch value their delegator resolves. The
    worker-shaped view of such a body is the delegator's resolution inlined at
    every use, which is exactly the expression the pre-move body evaluated.
    """

    def __init__(self, values) -> None:
        self._values = dict(values)

    def visit_Name(self, node):
        replacement = (
            self._values.get(node.id) if isinstance(node.ctx, ast.Load) else None
        )
        if replacement is None:
            return node
        return ast.parse(replacement, mode="eval").body


class _FillMigrationReceive(ast.NodeTransformer):
    """Put the outcome's two values back into the original write statements.

    Bound in __init__ rather than closed over: a transformer defined inside the
    scanning loop would capture whatever the loop variables hold when it runs,
    which is only safe while the loop breaks immediately after building it.
    """

    def __init__(self, *, uuid_value, record_value) -> None:
        self._uuid_value = uuid_value
        self._record_value = record_value

    def visit_Name(self, name):
        if name.id == "MIGRATED_UUID":
            return self._uuid_value
        if name.id == "PENDING_RECORD":
            return self._record_value
        return name


def load_worker_methods(
    names: Iterable[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile the named ``BatchGenWorker`` methods as standalone functions.

    Args:
        names: method names to lift out of the class body.
        namespace: globals each compiled method sees. Anything the method
            references at module scope (``torch``, ``logging``, module-level
            flags) must be in here, which also makes those dependencies
            explicit at the call site instead of implicit.

    Returns:
        ``{name: function}``. Each function takes ``self`` first.
    """
    wanted = set(names)
    tree = ast.parse(WORKER_SOURCE.read_text(), filename=str(WORKER_SOURCE))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == WORKER_CLASS
    ]
    if len(classes) != 1:
        raise AssertionError(
            f"expected exactly one class {WORKER_CLASS} in {WORKER_SOURCE}, "
            f"found {len(classes)}"
        )

    loaded: Dict[str, Any] = {}
    for node in classes[0].body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in wanted:
            continue
        if node.name in loaded:
            raise AssertionError(
                f"{WORKER_CLASS}.{node.name} is defined more than once; "
                "the test would be pinning an arbitrary one of them"
            )
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        scope: Dict[str, Any] = dict(namespace)
        exec(compile(module, str(WORKER_SOURCE), "exec"), scope)  # noqa: S102
        loaded[node.name] = scope[node.name]

    missing = wanted - set(loaded)
    if missing:
        raise AssertionError(
            f"{WORKER_CLASS} has no method(s) {sorted(missing)} in {WORKER_SOURCE}"
        )
    return loaded


def load_worker_functions(
    names: Iterable[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile the named module-level functions of the worker source.

    The module-level counterpart of :func:`load_worker_methods`, with the same
    ``namespace`` contract.
    """
    wanted = set(names)
    tree = ast.parse(WORKER_SOURCE.read_text(), filename=str(WORKER_SOURCE))
    loaded: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in wanted:
            continue
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        scope: Dict[str, Any] = dict(namespace)
        exec(compile(module, str(WORKER_SOURCE), "exec"), scope)  # noqa: S102
        loaded[node.name] = scope[node.name]

    missing = wanted - set(loaded)
    if missing:
        raise AssertionError(
            f"{WORKER_SOURCE} has no module-level function(s) {sorted(missing)}"
        )
    return loaded


# What the worker's delegators pass to the engine besides their own arguments:
# launch parameters, which the engine takes as values instead of reading the
# worker's args. A fake sets ``worker.args`` the way a server's launch does.
# Still-worker-shaped engine methods whose delegator hands them the prefill
# phase's attention group width. D7.1 converted two that take it
# (`_prefill_prefix_pool_enabled`, `_prefill_pool_group`), D7.2 a third
# (`_cached_prefix_rank_cost_profile`), D7.3 a fourth
# (`_check_pool_plan_identical`) and D7.4 a fifth
# (`_plan_prefill_micro_batches`); those are lifted to the worker's shape
# instead, which inlines the same resolution, so adapting their call no longer
# applies to them. `_config_prefill_for_batch` is D7.7's.
PREFILL_ATTENTION_WIDTH_METHODS = (
    "_config_prefill_for_batch",
)

# D7.1 left this map to the worker-shaped methods only: the two launch-value
# predicates it used to adapt are converted, and their value now travels
# through STATELESS_DELEGATED_VALUES in the lift.
PREFILL_ENGINE_DELEGATOR_KWARGS = {
    name: (lambda worker: {"attn_group_size": worker._prefill_attn_tp_size()})
    for name in PREFILL_ATTENTION_WIDTH_METHODS
}


def load_prefill_engine_methods(
    names: Iterable[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile S5a ``PrefillEngine`` methods from their real source.

    An unconverted engine method has the explicit S4b-style seam
    ``(self, worker, ...)``. This helper adapts it back to the worker-shaped
    call used by CPU fakes: the engine instance is stateless, so ``None`` is
    sufficient for its pre-rebinding ``self`` and the fake is passed as
    ``worker``.

    A method converted by Phase D takes a frozen context (or, where it reads
    no worker state, no seam parameter at all), so there is nothing to pass a
    fake as. It is lifted to the worker's own shape by
    :func:`engine_method_as_worker_method` instead, which is the body that
    actually runs with the delegator's resolutions inlined; the caller's fake
    and namespace reach it exactly as before.
    """
    wanted = set(names)
    tree = ast.parse(
        PREFILL_ENGINE_SOURCE.read_text(), filename=str(PREFILL_ENGINE_SOURCE)
    )
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == PREFILL_ENGINE_CLASS
    ]
    if len(classes) != 1:
        raise AssertionError(
            f"expected exactly one class {PREFILL_ENGINE_CLASS} in "
            f"{PREFILL_ENGINE_SOURCE}, found {len(classes)}"
        )

    loaded: Dict[str, Any] = {}
    for node in classes[0].body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in wanted:
            continue
        converted = node.name in STATELESS_CONVERTED or (
            len(node.args.args) >= 2 and node.args.args[1].arg == "ctx"
        )
        lifted = engine_method_as_worker_method(node) if converted else node
        module = ast.Module(body=[lifted], type_ignores=[])
        ast.fix_missing_locations(module)
        scope: Dict[str, Any] = dict(namespace)
        exec(compile(module, str(PREFILL_ENGINE_SOURCE), "exec"), scope)  # noqa: S102
        engine_method = scope[node.name]
        if converted:
            # Already the worker's own shape: its first parameter is the fake.
            loaded[node.name] = engine_method
            continue

        def worker_method(
            worker, *args, _engine_method=engine_method, _name=node.name, **kwargs
        ):
            extra = PREFILL_ENGINE_DELEGATOR_KWARGS.get(_name)
            if extra is not None:
                kwargs = {**extra(worker), **kwargs}
            return _engine_method(None, worker, *args, **kwargs)

        loaded[node.name] = worker_method

    missing = wanted - set(loaded)
    if missing:
        raise AssertionError(
            f"{PREFILL_ENGINE_CLASS} has no method(s) {sorted(missing)} in "
            f"{PREFILL_ENGINE_SOURCE}"
        )
    return loaded


class _GraphSetupToWorker(ast.NodeTransformer):
    """Restore only the explicit graph setup seam for source-pinned tests.

    Runtime tests exercise the actual delegator separately. These source
    assertions restore explicit helper calls and local outcome assignments
    to their historic spelling; they do not model binding time.
    """

    def visit_Attribute(self, node):
        if (
            isinstance(node.value, ast.Name) and node.value.id == "ctx"
            and node.attr == "disable_decode_host_kv_staging"
        ):
            node.attr = "glm5_disable_decode_host_kv_staging"
        return self.generic_visit(node)

    def visit_Assign(self, node):
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.startswith("setup_"):
            if isinstance(node.value, ast.Name) and node.value.id == "GRAPH_SETUP_UNCHANGED":
                return None
            node.targets = [ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=target.id.removeprefix("setup"), ctx=ast.Store(),
            )]
        return self.generic_visit(node)

    def __init__(self, callable_names):
        self.callable_names = set(callable_names)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.callable_names:
                node.func = ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=node.func.id, ctx=ast.Load(),
                )
        return node

    def visit_Return(self, node):
        assert isinstance(node.value, ast.Call)
        assert node.value.func.id == "GraphSetupOutcome"
        assert not node.value.args
        assert {kw.arg for kw in node.value.keywords} == {
            "_decode_graph_ladder_logged", "_criticality_probe", "_criticality_probe_recapture",
        }
        return ast.Return(value=None)

    def visit_Name(self, node):
        if node.id == "ctx":
            return ast.Name(id="self", ctx=node.ctx)
        return node


def engine_method_as_worker_method(node: ast.FunctionDef) -> ast.FunctionDef:
    """Convert an engine method ``(self, worker/ctx, ...)`` to worker shape.

    Older seam methods take the worker as an explicit collaborator and rebind
    ``self = worker``. AS3.1 methods take a frozen context, so their context
    paths are mapped back to the worker spelling for source-pinned tests.
    """
    method = copy.deepcopy(node)
    args = method.args.args
    if node.name in STATELESS_CONVERTED:
        # No seam parameter at all: the body reads no worker state, so only
        # its delegator-resolved values map back (there may be none at all)
        # and the signature of record is the delegator's.
        stateless = STATELESS_DELEGATED_VALUES.get(node.name, {})
        method = _StatelessValueToWorker(stateless).visit(method)
        worker_method = _class_methods(WORKER_SOURCE, WORKER_CLASS)[node.name]
        method.args = copy.deepcopy(worker_method.args)
        method.returns = copy.deepcopy(worker_method.returns)
        ast.fix_missing_locations(method)
        return method
    if node.name == "_setup_cuda_graphs" and [a.arg for a in args] == ["ctx", "gpu_manager"]:
        method = _GraphSetupToWorker(a.arg for a in node.args.kwonlyargs).visit(method)
        method.args = copy.deepcopy(_class_methods(WORKER_SOURCE, WORKER_CLASS)[node.name].args)
        method.decorator_list = []
        method.returns = None
        ast.fix_missing_locations(method)
        return method
    if len(args) < 2 or args[0].arg != "self" or args[1].arg not in {"worker", "ctx"}:
        raise AssertionError(f"{node.name} is not an engine method (self, worker/ctx, ...)")
    if args[1].arg == "ctx":
        del args[1]
        if node.name == "_make_boundary_decision_request":
            del args[-1]
        guards = GUARDED_CONTEXT_READS.get(node.name)
        if guards is not None:
            method = _GuardedContextReadToWorker(guards).visit(method)
        method.body = _ContextToWorker().visit(method).body
        outcome = OUTCOME_METHODS.get(node.name)
        if outcome is not None:
            method = _OutcomeToWorkerWrites(outcome).visit(method)
        if node.name in OUTCOME_IMPLICIT_TAIL_RETURN:
            tail = method.body[-1]
            if isinstance(tail, ast.Return) and tail.value is None:
                del method.body[-1]
        if any(
            isinstance(n, ast.Name) and n.id == _PENDING_TENSORS_ALIAS
            for n in ast.walk(method)
        ):
            method = _PendingTensorAliasToWorker().visit(method)
        if node.name == "_execute_single_kv_migration":
            method = _restore_migration_receive_writes(method)
        if node.name == "_cached_prefix_rank_cost_profile":
            method = _restore_cost_profile_cache_writes(method)
        if node.name == "_prefill_prepacked_pool":
            method = _restore_pool_model_pass_call(method)
        worker_method = _class_methods(WORKER_SOURCE, WORKER_CLASS).get(node.name)
        if worker_method is not None:
            extras = _extra_value_args(
                method.args, DELEGATOR_EXTRA_VALUE_ARGS.get(node.name, ())
            )
            method.args = copy.deepcopy(worker_method.args)
            # A value parameter the delegator resolves is not in its
            # signature, but the pre-move body read it as its own parameter,
            # so it goes back on the end with its default.
            for arg, default in extras:
                method.args.args.append(arg)
                method.args.defaults.append(default)
            # The delegator is the signature of record, its return annotation
            # included: an engine method that hands its worker writes back in
            # an outcome is annotated with that outcome, which the worker
            # method never returned and a test namespace cannot resolve.
            method.returns = copy.deepcopy(worker_method.returns)
        ast.fix_missing_locations(method)
        return method
    del args[1]
    rebinds = [
        index for index, stmt in enumerate(method.body)
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == "self"
        and isinstance(stmt.value, ast.Name)
        and stmt.value.id == "worker"
    ]
    if not rebinds:
        # The rebinding exists to keep a moved body's ``self.`` spellings. A
        # body that never names ``self`` reads nothing from the worker (it is
        # a function of its arguments), needs no rebinding, and an unused one
        # is a lint finding.
        named = [n for n in ast.walk(method) if isinstance(n, ast.Name) and n.id == "self"]
        if named:
            raise AssertionError(
                f"{node.name}: names `self` without the `self = worker` rebinding"
            )
        return method
    if len(rebinds) != 1:
        raise AssertionError(f"{node.name}: expected one `self = worker`, found {len(rebinds)}")
    del method.body[rebinds[0]]
    return method


def _class_methods(path: pathlib.Path, class_name: str) -> Dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _module_level_functions(path: pathlib.Path) -> Dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _is_context_function_shape(node: ast.FunctionDef) -> bool:
    args = node.args.args
    return len(args) >= 1 and args[0].arg == "ctx"


def _context_module_function_names() -> set:
    names: set = set()
    for source in DELEGATED_CONTEXT_MODULES:
        names |= {
            name for name, node in _module_level_functions(source).items()
            if _is_context_function_shape(node)
        }
    return names


def context_function_as_worker_method(node: ast.FunctionDef) -> ast.FunctionDef:
    """Convert a module-level helper ``(ctx, ...)`` to the worker's shape.

    The admission context helpers take the frozen context first; their context
    paths map back to the worker spelling exactly as the AS3.1 engine methods
    do, and the signature of record is the worker's same-name delegator.
    """
    method = copy.deepcopy(node)
    args = method.args.args
    if not args or args[0].arg != "ctx":
        raise AssertionError(f"{node.name} is not a context function (ctx, ...)")
    method.body = _ContextToWorker().visit(method).body
    worker_method = _class_methods(WORKER_SOURCE, WORKER_CLASS).get(node.name)
    if worker_method is None:
        raise AssertionError(
            f"{WORKER_CLASS} keeps no same-name delegator for {node.name}"
        )
    method.args = copy.deepcopy(worker_method.args)
    if node.name == "_make_prefill_selection_request":
        # The worker delegator resolves the shared-prefix segments and the
        # per-sequence KDA record page constant and passes the VALUES (the
        # context never carries a bound worker method). The worker-shaped
        # body restores those resolutions as its first statements after the
        # docstring, which is exactly where the pre-move method ran them.
        method.body.insert(1, ast.parse(
            "shared_segments = "
            "self._admission_shared_prefix_segments(all_candidates)"
        ).body[0])
        method.body.insert(2, ast.parse(
            "kda_record_pages = self._kda_record_pages_per_sequence()"
        ).body[0])
    ast.fix_missing_locations(method)
    return method


def rank_sync_method_names() -> set:
    """The worker methods whose bodies live in ``RankSyncEngine`` (S8)."""
    return set(_class_methods(RANK_SYNC_SOURCE, RANK_SYNC_CLASS))


def step_observability_method_names() -> set:
    """The worker methods whose bodies live in ``StepObservabilityEngine`` (S7)."""
    return set(_class_methods(STEP_OBSERVABILITY_SOURCE, STEP_OBSERVABILITY_CLASS))


def worker_shaped_method_node(name: str) -> ast.FunctionDef:
    """The body a worker method runs, in the worker's own shape.

    For a method moved into one of the worker seam engines, the worker keeps a
    canonical delegator. The statements that run are the engine's, so this
    returns the engine method converted back by
    :func:`engine_method_as_worker_method`. Every other name resolves to the
    worker class method itself.
    """
    for source, class_name in DELEGATED_ENGINES:
        engine = _class_methods(source, class_name)
        if name in engine and _is_delegator_shape(engine[name]):
            return engine_method_as_worker_method(engine[name])
    for source in DELEGATED_CONTEXT_MODULES:
        functions = _module_level_functions(source)
        if name in functions and _is_context_function_shape(functions[name]):
            return context_function_as_worker_method(functions[name])
    worker = _class_methods(WORKER_SOURCE, WORKER_CLASS)
    if name not in worker:
        raise AssertionError(f"{WORKER_CLASS} has no method {name}")
    return copy.deepcopy(worker[name])


def delegated_method_source(name: str) -> str:
    """The raw source text of a delegated method's engine body.

    Text pins that used to slice the worker file keep their assertions
    byte-identical by reading the engine's own source segment (located by
    AST, so no indentation or delimiter assumptions). The segment is the
    engine method as written, including its seam parameter.
    """
    for source, class_name in DELEGATED_ENGINES:
        engine = _class_methods(source, class_name)
        if name in engine and _is_delegator_shape(engine[name]):
            return ast.get_source_segment(source.read_text(), engine[name])
    for source in DELEGATED_CONTEXT_MODULES:
        functions = _module_level_functions(source)
        if name in functions and _is_context_function_shape(functions[name]):
            return ast.get_source_segment(source.read_text(), functions[name])
    raise KeyError(f"{name} is not a delegated engine method")


def delegated_method_names() -> set:
    """Every worker method whose body lives in a delegated engine (S8/S7/S9/S10a)."""
    names: set = set()
    for source, class_name in DELEGATED_ENGINES:
        names |= {
            name for name, node in _class_methods(source, class_name).items()
            if _is_delegator_shape(node)
        }
    names |= _context_module_function_names()
    return names


def load_worker_shaped_methods(
    names: Iterable[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """:func:`load_worker_methods` for the bodies that actually run.

    A delegated method compiles from its engine body in the worker's shape,
    so the caller's ``namespace`` (stub ``torch``, ``sys``, ...) reaches the
    moved statements exactly as it reached them on the worker; every other
    name compiles from the worker class as before.
    """
    loaded: Dict[str, Any] = {}
    for name in names:
        node = worker_shaped_method_node(name)
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        scope: Dict[str, Any] = dict(namespace)
        exec(compile(module, str(WORKER_SOURCE), "exec"), scope)  # noqa: S102
        loaded[name] = scope[name]
    return loaded


def attention_group_layout(group_size: int, world_size: int = 0):
    """A server layout whose attention runs in groups of ``group_size`` in
    both phases, for fakes of the worker: the worker reads its group widths
    from ``_server_layout``. ``world_size`` defaults to one group."""
    from batchgen.parallel_declaration import (
        ModuleParallelism,
        PhaseLayout,
        ServerLayout,
    )

    size = max(1, int(group_size))
    world = int(world_size) or size
    attention = (
        ModuleParallelism("tp", "resident", group_size=size)
        if size > 1
        else ModuleParallelism("dp", "resident")
    )
    none = ModuleParallelism("none", None)
    return ServerLayout(
        prefill=PhaseLayout("prefill", attention, none, world),
        decode=PhaseLayout("decode", attention, none, world),
    )
