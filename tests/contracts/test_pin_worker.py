"""Interface-stability pins for batchgen_worker.py and batchgen/worker/.

These signatures are load-bearing for downstream callers; changing one is
deliberate -- update this pin in the same PR and say why in the PR
description. Adding or removing a public method, renaming a parameter, and
dropping a return annotation all count.

PINS below is the frozen copy: for a function, its parameter names and
unparsed return annotation; for a class, its base classes, annotated field
names, class-level members, the set of public method names, and each of
those methods' parameters and return annotation.

Nothing here imports batchgen. ``batchgen_worker.py`` is ~21.6k lines and
imports torch, the native engine loader and libucx at module scope. Every
assertion is therefore made against the defining module's own AST, so this
suite passes on a machine with no GPU toolchain and costs milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Frozen by reading the defining sources' AST.
PINS = {
    "batchgen/batchgen_worker.py": {
        "BatchGenWorker": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {
                "PAGE_SIZE": "64",
                "DECISION_INTERVAL": "_sequence_module.DECISION_FREQUENCY_PAGES * 64",
            },
            "public_methods": [
                "Init",
                "cohort_shutdown_observed",
                "decoding_continuous",
                "deep_free_model_memory",
                "disable_decode_watchdog",
                "disable_watchdog",
                "enable_decode_watchdog",
                "feed_decode_watchdog",
                "feed_watchdog",
                "generate",
                "generate_persistent",
                "init_nvshmem",
                "prefill",
                "prefill_prepacked",
                "prepare_kimi_k3_startup",
                "set_admission_queue",
                "set_cohort_shutdown_probe",
                "set_decode_watchdog",
                "set_phase",
                "set_response_queue",
                "set_watchdog",
                "update_new_token",
            ],
            "methods": {
                "Init": {
                    "params": [
                        "self",
                        "max_input_length",
                        "max_decoding_length",
                        "num_queries",
                        "max_context_length",
                    ],
                    "returns": None,
                },
                "set_admission_queue": {
                    "params": ["self", "queue"],
                    "returns": "None",
                },
                "set_response_queue": {
                    "params": ["self", "queue"],
                    "returns": "None",
                },
                "set_watchdog": {
                    "params": ["self", "watchdog"],
                    "returns": "None",
                },
                "set_decode_watchdog": {
                    "params": ["self", "watchdog"],
                    "returns": "None",
                },
                "set_cohort_shutdown_probe": {
                    "params": ["self", "probe"],
                    "returns": "None",
                },
                "cohort_shutdown_observed": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "feed_watchdog": {
                    "params": ["self"],
                    "returns": "None",
                },
                "feed_decode_watchdog": {
                    "params": ["self"],
                    "returns": "None",
                },
                "enable_decode_watchdog": {
                    "params": ["self"],
                    "returns": "None",
                },
                "disable_decode_watchdog": {
                    "params": ["self"],
                    "returns": "None",
                },
                "disable_watchdog": {
                    "params": ["self"],
                    "returns": None,
                },
                "generate_persistent": {
                    "params": ["self"],
                    "returns": None,
                },
                "prepare_kimi_k3_startup": {
                    "params": ["self"],
                    "returns": "None",
                },
                "generate": {
                    "params": ["self"],
                    "returns": None,
                },
                "prefill": {
                    "params": ["self", "batch"],
                    "returns": None,
                },
                "prefill_prepacked": {
                    "params": ["self", "batch"],
                    "returns": None,
                },
                "decoding_continuous": {
                    "params": [
                        "self",
                        "new_tokens",
                        "decode_uuids",
                        "batch",
                        "past_key_states",
                        "past_value_states",
                        "scale_dict",
                    ],
                    "returns": "Tuple[List[str], List[int]]",
                },
                "set_phase": {
                    "params": ["self", "phase"],
                    "returns": None,
                },
                "update_new_token": {
                    "params": ["self", "new_tokens", "query_idx", "new_token_idx"],
                    "returns": None,
                },
                "init_nvshmem": {
                    "params": ["self"],
                    "returns": None,
                },
                "deep_free_model_memory": {
                    "params": ["self"],
                    "returns": None,
                },
            },
        },
        "BatchGenWorkerArgs": {
            "kind": "class",
            "bases": [],
            "fields": [
                "local_rank",
                "global_rank",
                "world_size",
                "local_world_size",
                "nnode_rank",
                "nnodes",
                "dist_init_addr",
                "model_name",
                "hf_cache_dir",
                "cache_dir",
                "converted_ckpt_dir",
                "host_kv_cache_size",
                "global_host_kv_cache_size_gb",
                "shm_name",
                "tensor_meta_shm_name",
                "enable_hugetlbfs",
                "weight_byte_size",
                "skeleton_state_dict_file",
                "device",
                "kv_dtype",
                "gpu_arch",
                "enable_prefix_cache",
                "prefix_cache_debug_stats",
                "watchdog_timeout",
                "watchdog_test_stuck_time",
                "watchdog_heartbeat_interval",
                "decode_step_timeout",
                "enable_prepack",
                "glm5_disable_decode_host_kv_staging",
                "host_kv_watermark",
                "enable_decode_preemption",
                "gpu_memory_frac",
                "initial_gpu_page_buffer",
                "extension_gpu_page_buffer",
                "decision_frequency_pages",
                "enable_ep_with_offloading",
                "ep_offloading_ratio",
                "pre_dequantize_weights",
                "enable_deepep",
                "enable_cuda_graph",
                "disable_cuda_graphs",
                "cuda_graph_max_bucket_size",
                "cuda_graph_num_buckets",
                "persistent_phase_instances",
                "detokenization_include_special_tokens",
                "host_kv_chunk_size",
                "host_kv_eviction_watermark",
                "enable_host_kv_eviction",
                "adaptive_chunk",
                "adaptive_chunk_min",
                "adaptive_chunk_max",
                "adaptive_chunk_ema_alpha",
                "adaptive_chunk_multiplier",
                "fast_init",
                "prefix_memfd_pid",
                "prefix_memfd_fd",
                "kv_memfd_pid",
                "kv_memfd_fd",
                "kv_aux_memfd_fd",
                "weights_memfd_pid",
                "weights_memfd_fd",
                "tensor_meta_memfd_fd",
                "distributed_weight_config",
                "distributed_weight_cohort",
                "max_pool_size",
                "input_ids_pool_size_gb",
                "host_kv_shm_name",
                "host_kv_aux_shm_name",
                "prefix_cache_shm_name",
                "query_book_shm_prefix",
                "runtime_dir",
                "pynccl_port_base",
                "pynccl_port_span",
                "log_dir",
                "log_level",
                "run_id",
                "run_started_at",
                "log_cohort",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
    },
    "batchgen/worker/kv_manager.py": {
        "HostKVUtilization": {
            "kind": "class",
            "bases": [],
            "fields": [
                "rank",
                "node_id",
                "num_free_pages",
                "num_total_pages",
                "num_used_pages",
                "free_percent",
                "num_in_decode",
                "num_onhold",
                "num_prefilled",
                "num_valid_sequences",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "KVStats": {
            "kind": "class",
            "bases": [],
            "fields": ["num_free_pages", "num_used_pages", "num_total_pages"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "KVStatsBackend": {
            "kind": "class",
            "bases": ["Protocol"],
            "fields": [],
            "members": {},
            "public_methods": ["get_gpu_stats", "get_host_stats"],
            "methods": {
                "get_host_stats": {
                    "params": ["self"],
                    "returns": "KVStats",
                },
                "get_gpu_stats": {
                    "params": ["self"],
                    "returns": "Optional[KVStats]",
                },
            },
        },
        "KVUtilizationRequest": {
            "kind": "class",
            "bases": [],
            "fields": [
                "rank",
                "world_size",
                "local_rank",
                "num_gpus_per_node",
                "global_batch",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "MigrationCandidate": {
            "kind": "class",
            "bases": [],
            "fields": [
                "uuid",
                "assigned_rank",
                "global_idx",
                "kv_token_budget",
                "host_pages_allocated",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "MigrationPlanRequest": {
            "kind": "class",
            "bases": [],
            "fields": [
                "node_stats",
                "candidates",
                "num_gpus_per_node",
                "world_size",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "PageTableCapacityRequest": {
            "kind": "class",
            "bases": [],
            "fields": [
                "sequence_tokens",
                "max_input_length",
                "max_decoding_length",
                "engine_max_prompt",
                "engine_max_decode",
                "engine_module_global_batch_size",
                "engine_module_attn_decoding_micro_batch_size",
                "engine_basic_num_queries",
                "model_max_position_embeddings",
                "args_cuda_graph_max_bucket_size",
                "engine_basic_decode_graph_max_bucket",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "TokenBudgetRequest": {
            "kind": "class",
            "bases": [],
            "fields": ["query_book"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "WatermarkGlobalStats": {
            "kind": "class",
            "bases": [],
            "fields": ["used", "total", "free_percent", "num_nodes"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "WatermarkTriggerPlan": {
            "kind": "class",
            "bases": [],
            "fields": ["should_trigger", "max_free_percent", "global_stats"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "WatermarkTriggerRequest": {
            "kind": "class",
            "bases": [],
            "fields": [
                "node_stats",
                "host_kv_watermark",
                "has_queued",
                "has_evicted",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
    },
}

# Named in the module's public surface but deliberately NOT pinned, and why:
SKIPPED = {
    "worker/kv_manager.KVCacheManager": "engine-internal; only the request/plan data types are pinned",
    "worker test-only exports": "reachable for tests only; not part of the stable surface",
}


# ---------------------------------------------------------------------------
# AST harness.  Duplicated verbatim in every tests/contracts/test_pin_*.py so
# that each pin file stands alone: reading one file tells you the whole frozen
# surface of one module, with no shared helper to chase.
# ---------------------------------------------------------------------------

def _params(fn) -> list:
    a = fn.args
    names = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return names


def _returns(fn):
    return ast.unparse(fn.returns) if fn.returns is not None else None


def _class_actual(node: ast.ClassDef) -> dict:
    methods = {
        m.name: {"params": _params(m), "returns": _returns(m)}
        for m in node.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not m.name.startswith("_")
    }
    return {
        "kind": "class",
        "bases": [ast.unparse(b) for b in node.bases],
        "fields": [
            m.target.id
            for m in node.body
            if isinstance(m, ast.AnnAssign)
            and isinstance(m.target, ast.Name)
            and not m.target.id.startswith("_")
        ],
        "members": {
            m.targets[0].id: ast.unparse(m.value)
            for m in node.body
            if isinstance(m, ast.Assign)
            and len(m.targets) == 1
            and isinstance(m.targets[0], ast.Name)
            and not m.targets[0].id.startswith("_")
        },
        "public_methods": sorted(methods),
        "methods": methods,
    }


def _actual(rel: str) -> dict:
    tree = ast.parse((REPO / rel).read_text())
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.setdefault(node.name, _class_actual(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(
                node.name,
                {
                    "kind": "function",
                    "params": _params(node),
                    "returns": _returns(node),
                },
            )
    return out


ACTUAL = {rel: _actual(rel) for rel in PINS}

_RULE = (
    "A pinned signature changed.  If that is intended, update this pin in "
    "the SAME pull request and say why in the PR description."
)


def test_every_pinned_symbol_is_still_defined_in_its_file():
    missing = [
        "%s::%s" % (rel, name)
        for rel, syms in PINS.items()
        for name in syms
        if name not in ACTUAL[rel]
    ]
    assert not missing, "pinned symbols are gone: %r\n%s" % (
        missing,
        _RULE,
    )


def test_pinned_kinds_and_base_classes_are_unchanged():
    bad = []
    for rel, syms in PINS.items():
        for name, pin in syms.items():
            got = ACTUAL[rel].get(name)
            if got is None:
                continue
            if got["kind"] != pin["kind"]:
                bad.append(
                    "%s::%s kind %s -> %s" % (rel, name, pin["kind"], got["kind"])
                )
            elif pin["kind"] == "class" and got["bases"] != pin["bases"]:
                bad.append(
                    "%s::%s bases %r -> %r"
                    % (rel, name, pin["bases"], got["bases"])
                )
    assert not bad, "%r\n%s" % (bad, _RULE)


def test_pinned_class_surfaces_are_unchanged():
    """Public method names, annotated field names, and enum/class members."""
    bad = []
    for rel, syms in PINS.items():
        for name, pin in syms.items():
            if pin["kind"] != "class":
                continue
            got = ACTUAL[rel].get(name)
            if got is None or got["kind"] != "class":
                continue
            for key in ("public_methods", "fields", "members"):
                if got[key] != pin[key]:
                    added = sorted(set(got[key]) - set(pin[key]))
                    removed = sorted(set(pin[key]) - set(got[key]))
                    bad.append(
                        "%s::%s %s added=%r removed=%r"
                        % (rel, name, key, added, removed)
                    )
    assert not bad, "%r\n%s" % (bad, _RULE)


def test_pinned_signatures_are_unchanged():
    """Parameter name tuples and unparsed return annotations."""
    bad = []
    for rel, syms in PINS.items():
        for name, pin in syms.items():
            got = ACTUAL[rel].get(name)
            if got is None or got["kind"] != pin["kind"]:
                continue
            if pin["kind"] == "function":
                pairs = [("", pin, got)]
            else:
                pairs = [
                    (".%s" % m, spec, got["methods"][m])
                    for m, spec in pin["methods"].items()
                    if m in got["methods"]
                ]
            for suffix, want, have in pairs:
                if have["params"] != want["params"]:
                    bad.append(
                        "%s::%s%s params %r -> %r"
                        % (rel, name, suffix, want["params"], have["params"])
                    )
                if have["returns"] != want["returns"]:
                    bad.append(
                        "%s::%s%s returns %r -> %r"
                        % (rel, name, suffix, want["returns"], have["returns"])
                    )
    assert not bad, "%r\n%s" % (bad, _RULE)
