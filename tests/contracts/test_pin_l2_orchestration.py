"""Interface-stability pins for the batch-planning modules: sequence,
query_book, continuous_batching, batch_order, planner and migration.

These signatures are load-bearing for downstream callers; changing one is
deliberate -- update this pin in the same PR and say why in the PR
description. Adding or removing a public method, renaming a parameter, and
dropping a return annotation all count.

PINS below is the frozen copy: for a function, its parameter names and
unparsed return annotation; for a class, its base classes, annotated field
names, class-level members, the set of public method names, and each of
those methods' parameters and return annotation.

Nothing here imports batchgen. ``sequence.py``, ``query_book.py`` and
``continuous_batching.py`` import torch at module scope. Every assertion is
therefore made against the defining module's own AST, so this suite passes
on a machine with no GPU toolchain and costs milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Frozen by reading the defining sources' AST.
PINS = {
    "batchgen/batch_order.py": {
        "batch_matches_expected_uuid_order": {
            "kind": "function",
            "params": ["local_indices", "expected_uuids", "uuid_to_local_map"],
            "returns": "bool",
        },
        "build_prefill_sequence_spans": {
            "kind": "function",
            "params": [
                "local_indices",
                "seq_lengths",
                "local_to_uuid_map",
                "local_to_global_seq_id_map",
            ],
            "returns": "List[PrefillSequenceSpan]",
        },
        "local_indices_to_uuid_order": {
            "kind": "function",
            "params": ["local_indices", "local_to_uuid_map"],
            "returns": "List[Optional[str]]",
        },
        "prefill_sequence_spans_to_cu_seqlens": {
            "kind": "function",
            "params": ["spans"],
            "returns": "List[int]",
        },
        "prefill_sequence_spans_to_global_seq_ids": {
            "kind": "function",
            "params": ["spans"],
            "returns": "List[int]",
        },
    },
    "batchgen/continuous_batching.py": {
        "AdaptiveChunkSizer": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": ["get_chunk_size", "report_completion"],
            "methods": {
                "report_completion": {
                    "params": ["self", "decoded_length"],
                    "returns": "None",
                },
                "get_chunk_size": {
                    "params": ["self"],
                    "returns": "int",
                },
            },
        },
        "BoundaryDecisions": {
            "kind": "class",
            "bases": [],
            "fields": [
                "completed_uuids",
                "active_uuids",
                "host_growth_uuids",
                "host_growth_pages",
                "growth_feasible",
                "host_evicted_uuids",
                "onhold_uuids",
                "seqs_needing_extension",
                "new_load_uuids",
                "decode_uuids_final",
                "scheduler_error",
                "host_planned_free_pages",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "LoadingStrategy": {
            "kind": "class",
            "bases": ["Enum"],
            "fields": [],
            "members": {
                "LONGEST_FIRST": "'longest_first'",
                "SHORTEST_FIRST": "'shortest_first'",
                "FIFO": "'fifo'",
            },
            "public_methods": [],
            "methods": {},
        },
        "plan_host_kv_growth_evictions": {
            "kind": "function",
            "params": [
                "active_uuids",
                "completed_uuids",
                "host_growth_uuids",
                "host_growth_pages",
                "eviction_candidates",
                "free_pages",
                "total_pages",
                "completed_pages",
                "watermark_percent",
                "safety_margin",
                "strategy",
                "page_key",
            ],
            "returns": "HostKVGrowthEvictionPlan",
        },
        "select_sequences_for_loading": {
            "kind": "function",
            "params": [
                "candidates",
                "per_rank_free_pages",
                "exclude_uuids",
                "strategy",
                "get_global_idx_fn",
                "group_size",
                "per_group_seq_slots",
            ],
            "returns": "Tuple[List[str], Dict[int, int]]",
        },
        "validate_boundary_payload_alignment": {
            "kind": "function",
            "params": ["decode_uuids", "all_payloads", "group_size"],
            "returns": "None",
        },
    },
    "batchgen/migration.py": {
        "HostKVStats": {
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
        "MigrationOp": {
            "kind": "class",
            "bases": [],
            "fields": ["uuid", "from_rank", "to_rank", "pages", "host_pages"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
    },
    "batchgen/planner/base_planner.py": {
        "BasePlanner": {
            "kind": "class",
            "bases": ["ABC"],
            "fields": ["MAGIC_NUM", "DEFAULT_MEM_FRAC", "NUM_EXPERTS"],
            "members": {},
            "public_methods": ["generate_config", "get_module_shapes"],
            "methods": {
                "generate_config": {
                    "params": ["self", "config"],
                    "returns": "EngineConfig",
                },
                "get_module_shapes": {
                    "params": ["self"],
                    "returns": "dict",
                },
            },
        },
    },
    "batchgen/query_book.py": {
        "QueryBook": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {
                "TOKEN_BYTES": "4",
            },
            "public_methods": [
                "active_count",
                "append_prompt",
                "append_token",
                "append_tokens",
                "begin_reentry_turn",
                "bind",
                "bind_batch",
                "can_reserve",
                "can_reserve_batch",
                "capacity_tokens",
                "copy_span_to",
                "copy_to",
                "free_bytes",
                "free_page_count",
                "generated_tokens",
                "has_sequence",
                "has_slot",
                "largest_free_extent_pages",
                "last_generated_token",
                "memory_bytes",
                "metadata",
                "pages_for",
                "release",
                "reservation_capacity",
                "reservation_free_count",
                "reservation_total_capacity",
                "restore",
                "set_prompt_length",
                "slot_for",
                "synchronize_metadata",
                "tokens",
                "turns",
                "turns_from_payload",
                "turns_payload",
                "write_prompt",
                "write_prompts",
            ],
            "methods": {
                "memory_bytes": {
                    "params": ["self"],
                    "returns": "int",
                },
                "capacity_tokens": {
                    "params": ["self"],
                    "returns": "int",
                },
                "free_page_count": {
                    "params": ["self"],
                    "returns": "int",
                },
                "free_bytes": {
                    "params": ["self"],
                    "returns": "int",
                },
                "largest_free_extent_pages": {
                    "params": ["self"],
                    "returns": "int",
                },
                "active_count": {
                    "params": ["self"],
                    "returns": "int",
                },
                "pages_for": {
                    "params": ["self", "max_tokens"],
                    "returns": "int",
                },
                "can_reserve": {
                    "params": ["self", "max_tokens"],
                    "returns": "bool",
                },
                "can_reserve_batch": {
                    "params": ["self", "max_tokens"],
                    "returns": "bool",
                },
                "reservation_total_capacity": {
                    "params": ["self", "max_tokens"],
                    "returns": "int",
                },
                "reservation_free_count": {
                    "params": ["self", "max_tokens"],
                    "returns": "int",
                },
                "reservation_capacity": {
                    "params": ["self", "max_tokens"],
                    "returns": "int",
                },
                "bind_batch": {
                    "params": ["self", "sequence_ids", "max_tokens"],
                    "returns": "list[int]",
                },
                "bind": {
                    "params": ["self", "sequence_id", "max_tokens"],
                    "returns": "int",
                },
                "release": {
                    "params": ["self", "slot"],
                    "returns": "None",
                },
                "slot_for": {
                    "params": ["self", "sequence_id"],
                    "returns": "int",
                },
                "metadata": {
                    "params": ["self", "slot"],
                    "returns": "QueryBookSlot",
                },
                "has_slot": {
                    "params": ["self", "slot"],
                    "returns": "bool",
                },
                "has_sequence": {
                    "params": ["self", "sequence_id"],
                    "returns": "bool",
                },
                "turns": {
                    "params": ["self", "slot"],
                    "returns": "tuple[QueryBookTurn, ...]",
                },
                "write_prompt": {
                    "params": ["self", "slot", "prompt_tokens"],
                    "returns": "None",
                },
                "append_prompt": {
                    "params": ["self", "slot", "prompt_tokens"],
                    "returns": "QueryBookTurn",
                },
                "begin_reentry_turn": {
                    "params": ["self", "slot", "prompt_length"],
                    "returns": "QueryBookTurn",
                },
                "set_prompt_length": {
                    "params": ["self", "slot", "prompt_length"],
                    "returns": "None",
                },
                "synchronize_metadata": {
                    "params": [
                        "self",
                        "slot",
                        "prompt_length",
                        "token_length",
                        "decoded_length",
                    ],
                    "returns": "None",
                },
                "write_prompts": {
                    "params": ["self", "slots", "prompts"],
                    "returns": "None",
                },
                "restore": {
                    "params": [
                        "self",
                        "slot",
                        "tokens",
                        "prompt_length",
                        "decoded_length",
                        "turns",
                    ],
                    "returns": "None",
                },
                "append_token": {
                    "params": ["self", "slot", "token_id"],
                    "returns": "int",
                },
                "append_tokens": {
                    "params": ["self", "slots", "token_ids"],
                    "returns": "None",
                },
                "tokens": {
                    "params": ["self", "slot", "length"],
                    "returns": "torch.Tensor",
                },
                "generated_tokens": {
                    "params": ["self", "slot"],
                    "returns": "torch.Tensor",
                },
                "last_generated_token": {
                    "params": ["self", "slot"],
                    "returns": "torch.Tensor",
                },
                "turns_payload": {
                    "params": ["self", "slot"],
                    "returns": "torch.Tensor",
                },
                "turns_from_payload": {
                    "params": ["payload"],
                    "returns": "list[QueryBookTurn]",
                },
                "copy_span_to": {
                    "params": ["self", "slot", "output", "start", "length"],
                    "returns": "torch.Tensor",
                },
                "copy_to": {
                    "params": ["self", "slot", "device", "dtype", "length", "out"],
                    "returns": "torch.Tensor",
                },
            },
        },
        "bind_local_sequence_to_query_book": {
            "kind": "function",
            "params": [
                "uuid",
                "sequence",
                "query_book",
                "local_to_uuid_map",
                "uuid_to_local_map",
                "free_local_indices",
                "next_local_idx",
                "local_idx",
            ],
            "returns": "Tuple[int, int]",
        },
        "make_query_book_entry": {
            "kind": "function",
            "params": ["sequence"],
            "returns": "QueryBookEntry",
        },
        "release_local_query_slot": {
            "kind": "function",
            "params": [
                "uuid",
                "uuid_to_local_map",
                "local_to_uuid_map",
                "query_book",
                "free_local_indices",
            ],
            "returns": "Optional[int]",
        },
    },
    "batchgen/sequence.py": {
        "SequenceBatch": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": [
                "add_sequence",
                "all_completed",
                "assign_rank",
                "composition_epoch",
                "count_by_status",
                "create_view",
                "get_current_pages_used_for_sequences",
                "get_resumable_sequences",
                "get_resumable_sequences_for_rank",
                "get_sequence",
                "get_sequence_by_global_idx",
                "get_sequences_by_status",
                "get_sequences_for_rank",
                "get_sequences_for_rank_with_status",
                "get_sequences_needing_gpu_extension",
                "get_total_gpu_pages_allocated",
                "get_total_pages_for_sequences",
                "get_two_page_buffer_pages_for_sequences",
                "has_evicted",
                "has_in_decode",
                "has_on_hold",
                "has_prefilled",
                "has_queueing",
                "has_resumable",
                "remove_sequence",
                "reset_gpu_allocations_for_sequences",
                "update_status",
            ],
            "methods": {
                "composition_epoch": {
                    "params": ["self"],
                    "returns": "int",
                },
                "add_sequence": {
                    "params": ["self", "sequence"],
                    "returns": "None",
                },
                "remove_sequence": {
                    "params": ["self", "uuid"],
                    "returns": "Optional[SequenceEntry]",
                },
                "get_sequence": {
                    "params": ["self", "uuid"],
                    "returns": "Optional[SequenceEntry]",
                },
                "get_sequence_by_global_idx": {
                    "params": ["self", "global_idx"],
                    "returns": "Optional[SequenceEntry]",
                },
                "update_status": {
                    "params": ["self", "uuid", "new_status"],
                    "returns": "None",
                },
                "assign_rank": {
                    "params": ["self", "uuid", "rank"],
                    "returns": "None",
                },
                "get_sequences_by_status": {
                    "params": ["self", "status"],
                    "returns": "List[str]",
                },
                "get_sequences_for_rank": {
                    "params": ["self", "rank"],
                    "returns": "List[str]",
                },
                "get_sequences_for_rank_with_status": {
                    "params": ["self", "rank", "status"],
                    "returns": "List[str]",
                },
                "get_resumable_sequences": {
                    "params": ["self"],
                    "returns": "List[str]",
                },
                "get_resumable_sequences_for_rank": {
                    "params": ["self", "rank"],
                    "returns": "List[str]",
                },
                "create_view": {
                    "params": ["self", "uuids"],
                    "returns": "'SequenceBatch'",
                },
                "has_queueing": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "has_prefilled": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "has_in_decode": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "has_on_hold": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "has_resumable": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "has_evicted": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "all_completed": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "count_by_status": {
                    "params": ["self", "status"],
                    "returns": "int",
                },
                "get_total_pages_for_sequences": {
                    "params": ["self", "uuids"],
                    "returns": "int",
                },
                "get_current_pages_used_for_sequences": {
                    "params": ["self", "uuids"],
                    "returns": "int",
                },
                "get_two_page_buffer_pages_for_sequences": {
                    "params": ["self", "uuids"],
                    "returns": "int",
                },
                "get_total_gpu_pages_allocated": {
                    "params": ["self", "uuids"],
                    "returns": "int",
                },
                "get_sequences_needing_gpu_extension": {
                    "params": ["self", "uuids"],
                    "returns": "List[str]",
                },
                "reset_gpu_allocations_for_sequences": {
                    "params": ["self", "uuids"],
                    "returns": "None",
                },
            },
        },
        "SequenceEntry": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {
                "VALID_TRANSITIONS": "{SequenceStatus.QUEUEING: {SequenceStatus.IN_PREFILL}, SequenceStatus.IN_PREFILL: {SequenceStatus.PREFILLED}, SequenceStatus.PREFILLED: {SequenceStatus.IN_DECODE, SequenceStatus.COMPLETED}, SequenceStatus.IN_DECODE: {SequenceStatus.ON_HOLD, SequenceStatus.COMPLETED, SequenceStatus.EVICTED}, SequenceStatus.ON_HOLD: {SequenceStatus.IN_DECODE, SequenceStatus.COMPLETED, SequenceStatus.EVICTED}, SequenceStatus.COMPLETED: set(), SequenceStatus.EVICTED: {SequenceStatus.IN_PREFILL}}",
                "PAGE_SIZE": "64",
            },
            "public_methods": [
                "check_completion",
                "clamp_decode_to_context",
                "get_additional_gpu_pages_needed",
                "get_current_pages_used",
                "get_gpu_page_headroom",
                "get_gpu_pages_for_initial_load",
                "get_gpu_pages_for_two_page_buffer",
                "get_host_growth_pages",
                "get_host_pages_for_initial_chunk",
                "get_pages_required",
                "is_active_in_decode",
                "is_finished",
                "is_resumable",
                "log_event",
                "mark_initial_gpu_reservation_done",
                "needs_gpu_page_extension",
                "needs_host_kv_growth",
                "remaining_decode_tokens",
                "reset_gpu_allocation",
                "should_check_completion",
                "status_transition",
                "tokens_until_next_page_boundary",
                "validate_metadata",
            ],
            "methods": {
                "clamp_decode_to_context": {
                    "params": ["self", "prompt_len", "context_len"],
                    "returns": "int",
                },
                "log_event": {
                    "params": ["self", "event", "rank", "detail"],
                    "returns": "None",
                },
                "validate_metadata": {
                    "params": [
                        "self",
                        "context",
                        "require_owner_tensors",
                        "allow_terminal_capacity_gap",
                    ],
                    "returns": "None",
                },
                "status_transition": {
                    "params": ["self", "new_status"],
                    "returns": "None",
                },
                "get_pages_required": {
                    "params": ["self"],
                    "returns": "int",
                },
                "get_current_pages_used": {
                    "params": ["self"],
                    "returns": "int",
                },
                "get_gpu_pages_for_initial_load": {
                    "params": ["self"],
                    "returns": "int",
                },
                "get_gpu_pages_for_two_page_buffer": {
                    "params": ["self"],
                    "returns": "int",
                },
                "get_additional_gpu_pages_needed": {
                    "params": ["self"],
                    "returns": "int",
                },
                "mark_initial_gpu_reservation_done": {
                    "params": ["self"],
                    "returns": "None",
                },
                "get_gpu_page_headroom": {
                    "params": ["self"],
                    "returns": "int",
                },
                "tokens_until_next_page_boundary": {
                    "params": ["self"],
                    "returns": "int",
                },
                "needs_gpu_page_extension": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "reset_gpu_allocation": {
                    "params": ["self"],
                    "returns": "None",
                },
                "needs_host_kv_growth": {
                    "params": ["self", "chunk_size"],
                    "returns": "bool",
                },
                "get_host_growth_pages": {
                    "params": ["self", "chunk_size"],
                    "returns": "int",
                },
                "get_host_pages_for_initial_chunk": {
                    "params": ["self", "chunk_size"],
                    "returns": "int",
                },
                "is_finished": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "is_active_in_decode": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "is_resumable": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "remaining_decode_tokens": {
                    "params": ["self"],
                    "returns": "int",
                },
                "should_check_completion": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "check_completion": {
                    "params": ["self", "eos_token_id"],
                    "returns": "bool",
                },
            },
        },
        "SequenceStatus": {
            "kind": "class",
            "bases": ["IntEnum"],
            "fields": [],
            "members": {
                "QUEUEING": "0",
                "IN_PREFILL": "1",
                "PREFILLED": "2",
                "IN_DECODE": "3",
                "ON_HOLD": "4",
                "COMPLETED": "5",
                "EVICTED": "6",
            },
            "public_methods": [],
            "methods": {},
        },
        "configure_page_buffers": {
            "kind": "function",
            "params": [
                "initial_gpu_page_buffer",
                "extension_gpu_page_buffer",
                "decision_frequency_pages",
            ],
            "returns": "None",
        },
    },
}

# Named in the module's public surface but deliberately NOT pinned, and why:
SKIPPED = {
    "planner.get_planner": "dead; slated for removal",
    "migration.KVMigrationHelper": "unreachable; slated for removal",
    "page-buffer constants": (
        "module-level ints rebound by configure_page_buffers, not a callable "
        "surface; configure_page_buffers itself is pinned"
    ),
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
