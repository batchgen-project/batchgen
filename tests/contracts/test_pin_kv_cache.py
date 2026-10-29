"""Interface-stability pins for batchgen/kv_cache/.

These signatures are load-bearing for downstream callers; changing one is
deliberate -- update this pin in the same PR and say why in the PR
description. Adding or removing a public method, renaming a parameter, and
dropping a return annotation all count.

PINS below is the frozen copy: for a function, its parameter names and
unparsed return annotation; for a class, its base classes, annotated field
names, class-level members, the set of public method names, and each of
those methods' parameters and return annotation.

Nothing here imports batchgen. ``gpu_paged_kv_manager.py`` imports torch at
module scope. Every assertion is therefore made against the defining
module's own AST, so this suite passes on a machine with no GPU toolchain
and costs milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Frozen by reading the defining sources' AST.
PINS = {
    "batchgen/kv_cache/dual_host_kv_coordinator.py": {
        "DualAsyncKVTask": {
            "kind": "class",
            "bases": [],
            "fields": ["primary_task", "aux_task", "tensors"],
            "members": {},
            "public_methods": ["done", "wait"],
            "methods": {
                "done": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "wait": {
                    "params": ["self"],
                    "returns": "None",
                },
            },
        },
        "DualHostKVCoordinator": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": [
                "allocate_pages_for_sequences",
                "async_load_layer_paged_kv_to_device",
                "async_load_layer_paged_kv_to_device_dual",
                "async_offload_layer_kv_to_host",
                "build_host_configs",
                "create_managers",
                "from_budget",
                "get_stats",
                "grow_pages_for_sequences",
                "initialize",
                "register_sequences",
                "release_all_sequences",
                "release_sequence_pages",
                "require_auxiliary",
                "set_sequence_pages",
                "unregister_sequences",
                "views_by_group",
            ],
            "methods": {
                "build_host_configs": {
                    "params": [
                        "model_name",
                        "host_kv_cache_size",
                        "primary_shm_name",
                        "aux_shm_name",
                    ],
                    "returns": "Optional[Tuple[Any, Any]]",
                },
                "from_budget": {
                    "params": [
                        "cls",
                        "model_name",
                        "host_kv_cache_size",
                        "core_engine_module",
                        "enable_thp",
                        "memfd_creator_pid",
                        "memfd_fd",
                        "aux_memfd_fd",
                        "primary_shm_name",
                        "aux_shm_name",
                    ],
                    "returns": "Optional['DualHostKVCoordinator']",
                },
                "create_managers": {
                    "params": [
                        "cls",
                        "model_name",
                        "host_kv_cache_size",
                        "enable_thp",
                        "primary_shm_name",
                        "aux_shm_name",
                    ],
                    "returns": "Optional[Tuple[Any, Any]]",
                },
                "initialize": {
                    "params": ["self", "**kwargs"],
                    "returns": "None",
                },
                "require_auxiliary": {
                    "params": ["self", "context"],
                    "returns": None,
                },
                "views_by_group": {
                    "params": ["self"],
                    "returns": "dict",
                },
                "register_sequences": {
                    "params": ["self", "sequence_ids"],
                    "returns": "None",
                },
                "set_sequence_pages": {
                    "params": ["self", "sequence_id", "page_ids"],
                    "returns": "None",
                },
                "allocate_pages_for_sequences": {
                    "params": ["self", "seq_token_pairs"],
                    "returns": "None",
                },
                "grow_pages_for_sequences": {
                    "params": ["self", "seq_page_pairs"],
                    "returns": "None",
                },
                "release_sequence_pages": {
                    "params": ["self", "sequence_ids"],
                    "returns": "None",
                },
                "release_all_sequences": {
                    "params": ["self"],
                    "returns": "None",
                },
                "unregister_sequences": {
                    "params": ["self", "sequence_ids"],
                    "returns": "None",
                },
                "get_stats": {
                    "params": ["self"],
                    "returns": None,
                },
                "async_load_layer_paged_kv_to_device": {
                    "params": ["self", "**kwargs"],
                    "returns": None,
                },
                "async_load_layer_paged_kv_to_device_dual": {
                    "params": [
                        "self",
                        "sequence_ids",
                        "primary_active_page_counts",
                        "primary_k_device_ptrs",
                        "primary_v_device_ptrs",
                        "aux_active_page_counts",
                        "aux_k_device_ptrs",
                        "aux_v_device_ptrs",
                        "tensors",
                    ],
                    "returns": "DualAsyncKVTask",
                },
                "async_offload_layer_kv_to_host": {
                    "params": ["self", "**kwargs"],
                    "returns": None,
                },
            },
        },
    },
    "batchgen/kv_cache/dual_kv_cache_coordinator.py": {
        "DualKVCacheCoordinator": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": [
                "allocate_pages",
                "allocate_pages_for_sequences",
                "assert_mirrored_state",
                "assert_pool_mapped",
                "capacity_snapshot",
                "clear_page_table",
                "config",
                "copy_kv_to_tensor",
                "copy_tensor_to_kv",
                "destroy",
                "device",
                "export_active_sequence_page_counts",
                "export_layer_page_pointer_table",
                "extend_pages_for_sequence",
                "free_pages_for_sequences",
                "get_context_kv_page_ptrs",
                "get_kv_tensors",
                "get_layer_kv_with_page_table",
                "get_padded_3d_page_pointers",
                "get_page_table_version",
                "get_sequence_layer_page_pointers",
                "get_stats",
                "grow_pages_for_sequences",
                "grow_sequence_pages",
                "initialize",
                "is_initialized",
                "pool_is_mapped",
                "rebuild_page_table",
                "reset_allocations",
                "update_layer_decode_new_token",
                "uses_fixed_address_pool",
            ],
            "methods": {
                "initialize": {
                    "params": ["self"],
                    "returns": "None",
                },
                "destroy": {
                    "params": ["self", "empty_cuda_cache"],
                    "returns": "None",
                },
                "is_initialized": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "uses_fixed_address_pool": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "pool_is_mapped": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "assert_pool_mapped": {
                    "params": ["self", "context"],
                    "returns": "None",
                },
                "capacity_snapshot": {
                    "params": ["self"],
                    "returns": None,
                },
                "allocate_pages": {
                    "params": ["self", "sequence_id", "num_tokens"],
                    "returns": "List[int]",
                },
                "allocate_pages_for_sequences": {
                    "params": ["self", "sequence_ids", "num_tokens"],
                    "returns": "List[List[int]]",
                },
                "grow_sequence_pages": {
                    "params": ["self", "sequence_id", "additional_tokens"],
                    "returns": "List[int]",
                },
                "grow_pages_for_sequences": {
                    "params": ["self", "sequence_ids", "additional_tokens"],
                    "returns": "List[List[int]]",
                },
                "extend_pages_for_sequence": {
                    "params": ["self", "sequence_id", "new_total_tokens"],
                    "returns": "int",
                },
                "rebuild_page_table": {
                    "params": ["self", "sequence_ids"],
                    "returns": "torch.Tensor",
                },
                "clear_page_table": {
                    "params": ["self"],
                    "returns": "None",
                },
                "reset_allocations": {
                    "params": ["self"],
                    "returns": "None",
                },
                "free_pages_for_sequences": {
                    "params": ["self", "sequence_ids"],
                    "returns": "None",
                },
                "get_stats": {
                    "params": ["self"],
                    "returns": "GPUPagedKVStats",
                },
                "get_kv_tensors": {
                    "params": ["self"],
                    "returns": "Tuple[torch.Tensor, Optional[torch.Tensor]]",
                },
                "get_layer_kv_with_page_table": {
                    "params": ["self", "layer_idx"],
                    "returns": None,
                },
                "update_layer_decode_new_token": {
                    "params": ["self", "*args", "**kwargs"],
                    "returns": None,
                },
                "get_page_table_version": {
                    "params": ["self"],
                    "returns": "int",
                },
                "config": {
                    "params": ["self"],
                    "returns": "GPUPagedKVConfig",
                },
                "device": {
                    "params": ["self"],
                    "returns": None,
                },
                "copy_kv_to_tensor": {
                    "params": ["self", "sequence_id"],
                    "returns": "torch.Tensor",
                },
                "copy_tensor_to_kv": {
                    "params": ["self", "sequence_id", "k_tensor"],
                    "returns": "None",
                },
                "get_context_kv_page_ptrs": {
                    "params": ["self", "*args", "**kwargs"],
                    "returns": None,
                },
                "get_sequence_layer_page_pointers": {
                    "params": ["self", "*args", "**kwargs"],
                    "returns": None,
                },
                "export_layer_page_pointer_table": {
                    "params": ["self", "*args", "**kwargs"],
                    "returns": None,
                },
                "export_active_sequence_page_counts": {
                    "params": ["self"],
                    "returns": "torch.Tensor",
                },
                "get_padded_3d_page_pointers": {
                    "params": ["self", "*args", "**kwargs"],
                    "returns": None,
                },
                "assert_mirrored_state": {
                    "params": ["self", "op_name", "sequence_ids"],
                    "returns": "None",
                },
            },
        },
    },
    "batchgen/kv_cache/gpu_paged_kv_manager.py": {
        "GPUPagedKVCacheManager": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": [
                "allocate_pages",
                "allocate_pages_for_sequences",
                "allocate_pages_for_sequences_with_page_keys",
                "append_layer_prefill_suffix_tokens",
                "assert_pool_mapped",
                "capacity_snapshot",
                "clear_page_table",
                "copy_kv_to_tensor",
                "copy_tensor_to_kv",
                "count_new_shared_pages",
                "destroy",
                "ensure_cuda_graph_page_table",
                "export_active_sequence_page_counts",
                "export_layer_page_pointer_table",
                "extend_pages_for_sequence",
                "free_pages_for_sequences",
                "get_context_kv_page_ptrs",
                "get_cuda_graph_page_table",
                "get_cuda_graph_page_table_state",
                "get_cuda_graph_page_table_storage",
                "get_kv_tensors",
                "get_layer_kv_cache",
                "get_layer_kv_with_page_table",
                "get_padded_3d_page_pointers",
                "get_page_table_version",
                "get_prefix_gpu_share_stats",
                "get_sequence_layer_page_pointers",
                "get_stats",
                "grow_pages_for_sequences",
                "grow_sequence_pages",
                "initialize",
                "is_initialized",
                "pool_is_mapped",
                "prepare_prefill_suffix_append",
                "rebuild_page_table",
                "reset_allocations",
                "resolve_physical_layer",
                "update_layer_decode_new_token",
                "uses_fixed_address_pool",
                "uses_logical_layer_mapping",
            ],
            "methods": {
                "initialize": {
                    "params": ["self"],
                    "returns": "None",
                },
                "uses_fixed_address_pool": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "pool_is_mapped": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "assert_pool_mapped": {
                    "params": ["self", "context"],
                    "returns": "None",
                },
                "capacity_snapshot": {
                    "params": ["self"],
                    "returns": None,
                },
                "destroy": {
                    "params": ["self", "empty_cuda_cache"],
                    "returns": "None",
                },
                "allocate_pages": {
                    "params": ["self", "sequence_id", "num_tokens"],
                    "returns": "List[int]",
                },
                "allocate_pages_for_sequences": {
                    "params": ["self", "sequence_ids", "num_tokens"],
                    "returns": "Dict[int, List[int]]",
                },
                "count_new_shared_pages": {
                    "params": [
                        "self",
                        "sequence_ids",
                        "num_tokens",
                        "page_keys_by_sequence",
                    ],
                    "returns": "int",
                },
                "allocate_pages_for_sequences_with_page_keys": {
                    "params": [
                        "self",
                        "sequence_ids",
                        "num_tokens",
                        "page_keys_by_sequence",
                    ],
                    "returns": "Dict[int, List[int]]",
                },
                "grow_sequence_pages": {
                    "params": ["self", "sequence_id", "num_pages"],
                    "returns": "List[int]",
                },
                "grow_pages_for_sequences": {
                    "params": ["self", "sequence_ids", "num_pages"],
                    "returns": "Dict[int, List[int]]",
                },
                "prepare_prefill_suffix_append": {
                    "params": [
                        "self",
                        "sequence_ids",
                        "prefix_lens",
                        "suffix_lens",
                        "rebuild_page_table",
                    ],
                    "returns": "GPUPagedKVSuffixAppendPlan",
                },
                "append_layer_prefill_suffix_tokens": {
                    "params": [
                        "self",
                        "k_tensor",
                        "v_tensor",
                        "append_plan",
                        "layer_idx",
                    ],
                    "returns": "None",
                },
                "clear_page_table": {
                    "params": ["self"],
                    "returns": "None",
                },
                "reset_allocations": {
                    "params": ["self"],
                    "returns": "None",
                },
                "rebuild_page_table": {
                    "params": ["self", "sequence_ids"],
                    "returns": "torch.Tensor",
                },
                "free_pages_for_sequences": {
                    "params": ["self", "sequence_ids"],
                    "returns": "None",
                },
                "get_context_kv_page_ptrs": {
                    "params": ["self", "sequence_id", "layer_idx", "context_length"],
                    "returns": "Tuple[List[int], Optional[List[int]]]",
                },
                "get_sequence_layer_page_pointers": {
                    "params": ["self", "sequence_id", "layer_idx"],
                    "returns": "Tuple[List[int], Optional[List[int]]]",
                },
                "update_layer_decode_new_token": {
                    "params": [
                        "self",
                        "k_tensor",
                        "v_tensor",
                        "sequence_lengths",
                        "layer_idx",
                        "batch_slice",
                        "slot_indices",
                    ],
                    "returns": "None",
                },
                "get_layer_kv_cache": {
                    "params": ["self", "layer_idx"],
                    "returns": "Tuple[torch.Tensor, Optional[torch.Tensor]]",
                },
                "get_layer_kv_with_page_table": {
                    "params": ["self", "layer_idx"],
                    "returns": "Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]",
                },
                "get_page_table_version": {
                    "params": ["self"],
                    "returns": "int",
                },
                "get_cuda_graph_page_table": {
                    "params": ["self"],
                    "returns": "torch.Tensor",
                },
                "get_cuda_graph_page_table_storage": {
                    "params": ["self"],
                    "returns": "torch.Tensor",
                },
                "ensure_cuda_graph_page_table": {
                    "params": ["self", "sequence_ids"],
                    "returns": "torch.Tensor",
                },
                "get_cuda_graph_page_table_state": {
                    "params": ["self"],
                    "returns": "CUDAGraphPageTableState",
                },
                "get_kv_tensors": {
                    "params": ["self"],
                    "returns": "Tuple[torch.Tensor, Optional[torch.Tensor]]",
                },
                "get_stats": {
                    "params": ["self"],
                    "returns": "GPUPagedKVStats",
                },
                "get_prefix_gpu_share_stats": {
                    "params": ["self"],
                    "returns": "PrefixGPUShareStats",
                },
                "copy_kv_to_tensor": {
                    "params": ["self", "sequence_id"],
                    "returns": "torch.Tensor",
                },
                "copy_tensor_to_kv": {
                    "params": ["self", "sequence_id", "k_tensor"],
                    "returns": "None",
                },
                "is_initialized": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "uses_logical_layer_mapping": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "resolve_physical_layer": {
                    "params": ["self", "logical_layer_id"],
                    "returns": "int",
                },
                "export_layer_page_pointer_table": {
                    "params": ["self"],
                    "returns": "Tuple[torch.Tensor, Optional[torch.Tensor]]",
                },
                "export_active_sequence_page_counts": {
                    "params": ["self"],
                    "returns": "torch.Tensor",
                },
                "get_padded_3d_page_pointers": {
                    "params": ["self"],
                    "returns": "Tuple[torch.Tensor, Optional[torch.Tensor]]",
                },
                "extend_pages_for_sequence": {
                    "params": ["self", "sequence_id", "new_total_tokens"],
                    "returns": "int",
                },
            },
        },
        "GPUPagedKVConfig": {
            "kind": "class",
            "bases": [],
            "fields": [
                "num_layers",
                "num_pages",
                "page_size_tokens",
                "num_k_heads",
                "k_head_dim",
                "num_v_heads",
                "v_head_dim",
                "kv_dtype",
                "cuda_graph_max_pages_per_sequence",
                "cuda_graph_max_slots",
                "logical_to_physical_layer",
            ],
            "members": {},
            "public_methods": ["from_device_config", "from_engine", "has_v_cache"],
            "methods": {
                "from_device_config": {
                    "params": ["cls", "device_config", "model_config", "kv_dtype"],
                    "returns": "'GPUPagedKVConfig'",
                },
                "from_engine": {
                    "params": ["cls", "engine_config", "model_config"],
                    "returns": "'GPUPagedKVConfig'",
                },
                "has_v_cache": {
                    "params": ["self"],
                    "returns": "bool",
                },
            },
        },
        "GPUPagedKVStats": {
            "kind": "class",
            "bases": [],
            "fields": [
                "num_total_pages",
                "num_free_pages",
                "num_used_pages",
                "num_total_pages_allocated",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
    },
    "batchgen/kv_cache/host_kv_mananger_config.py": {
        "apply_page_format": {
            "kind": "function",
            "params": ["config", "page_format"],
            "returns": "None",
        },
        "build_gpu_kv_config": {
            "kind": "function",
            "params": ["model_name", "sequence_tokens"],
            "returns": "GPUPagedKVConfig",
        },
        "build_gpu_kv_config_fixed_size": {
            "kind": "function",
            "params": ["model_name", "gpu_kv_cache_size_gb", "page_size_tokens"],
            "returns": "'GPUPagedKVConfig'",
        },
        "build_host_kv_config": {
            "kind": "function",
            "params": ["model_name", "host_kv_cache_size", "shm_name"],
            "returns": "Any",
        },
        "build_host_kv_worker_view": {
            "kind": "function",
            "params": ["core_engine_module", "config"],
            "returns": "Any",
        },
        "is_dsa_model": {
            "kind": "function",
            "params": ["model_name"],
            "returns": "bool",
        },
    },
}

# Named in the module's public surface but deliberately NOT pinned, and why:
SKIPPED = {
    "gpu_kv_kernels.run_paged_kv_token_update": "re-export; signature owned by batchgen_kernels",
    "gpu_kv_kernels.run_paged_kv_token_update_fused": "re-export; signature owned by batchgen_kernels",
    "KDAStateGPUConfig / KDAStateGPUManager": "slated to relocate",
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
