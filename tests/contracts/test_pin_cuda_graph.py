"""Interface-stability pins for batchgen/cuda_graph/.

These signatures are load-bearing for downstream callers; changing one is
deliberate -- update this pin in the same PR and say why in the PR
description. Adding or removing a public method, renaming a parameter, and
dropping a return annotation all count.

PINS below is the frozen copy: for a function, its parameter names and
unparsed return annotation; for a class, its base classes, annotated field
names, class-level members, the set of public method names, and each of
those methods' parameters and return annotation.

Nothing here imports batchgen. ``adapter.py`` and ``graph_manager.py``
import torch at module scope. Every assertion is therefore made against the
defining module's own AST, so this suite passes on a machine with no GPU
toolchain and costs milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Frozen by reading the defining sources' AST.
PINS = {
    "batchgen/cuda_graph/adapter.py": {
        "BatchState": {
            "kind": "class",
            "bases": [],
            "fields": [
                "local_bsz",
                "max_rank_bsz",
                "rank_token_counts",
                "cache_seqlens",
                "position_ids",
                "max_seqlen",
                "cur_batch_sequence_ids",
                "gpu_kv_manager",
                "decode_iter",
                "input_ids",
                "device",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "DebugOpts": {
            "kind": "class",
            "bases": [],
            "fields": [
                "compare_against_eager",
                "fail_on_mismatch",
                "log_path_breadcrumbs",
                "timing",
                "compare_atol",
                "compare_rtol",
                "probe_layers",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "GraphDecision": {
            "kind": "class",
            "bases": [],
            "fields": ["mode", "bucket", "reason", "fail_on_eager"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "GraphMode": {
            "kind": "class",
            "bases": ["str", "Enum"],
            "fields": [],
            "members": {
                "EAGER": "'eager'",
                "SEGMENTED": "'segmented'",
                "LAYER": "'layer'",
                "WHOLE_MODEL": "'whole_model'",
            },
            "public_methods": [],
            "methods": {},
        },
        "HasCudaGraphAdapter": {
            "kind": "class",
            "bases": ["Protocol"],
            "fields": [],
            "members": {},
            "public_methods": ["get_cuda_graph_adapter"],
            "methods": {
                "get_cuda_graph_adapter": {
                    "params": ["self"],
                    "returns": "Optional[ModelCudaGraphAdapter]",
                },
            },
        },
        "ModelCudaGraphAdapter": {
            "kind": "class",
            "bases": ["ABC"],
            "fields": [],
            "members": {},
            "public_methods": [
                "advertised_modes",
                "build_segments",
                "capture_inputs_for",
                "capture_signature",
                "debug_options",
                "eligibility",
                "is_supported",
                "prepare_replay_inputs",
                "release_all",
                "release_context",
                "run_eager_reference",
                "select_buckets",
                "stage_post_graph_kv",
            ],
            "methods": {
                "is_supported": {
                    "params": ["self", "engine_config"],
                    "returns": "bool",
                },
                "select_buckets": {
                    "params": ["self", "engine_config"],
                    "returns": "List[int]",
                },
                "advertised_modes": {
                    "params": ["self"],
                    "returns": "List[GraphMode]",
                },
                "build_segments": {
                    "params": [
                        "self",
                        "model",
                        "bucketing",
                        "gpu_kv_manager",
                        "world_size",
                        "rank",
                        "device",
                        "max_seqlen_cap",
                    ],
                    "returns": "SegmentBundle",
                },
                "capture_signature": {
                    "params": ["self", "bucket", "gpu_kv_manager", "max_seqlen"],
                    "returns": "Hashable",
                },
                "capture_inputs_for": {
                    "params": ["self", "bucket", "segment_name", "batch_state"],
                    "returns": "Dict[str, torch.Tensor]",
                },
                "eligibility": {
                    "params": ["self", "batch_state"],
                    "returns": "GraphDecision",
                },
                "prepare_replay_inputs": {
                    "params": ["self", "decision", "batch_state", "segment_name"],
                    "returns": "Dict[str, torch.Tensor]",
                },
                "stage_post_graph_kv": {
                    "params": ["self", "decision", "batch_state", "graph_outputs"],
                    "returns": "None",
                },
                "run_eager_reference": {
                    "params": [
                        "self",
                        "segment_name",
                        "batch_state",
                        "captured_inputs",
                        "probe_layers",
                    ],
                    "returns": "Dict[str, torch.Tensor]",
                },
                "debug_options": {
                    "params": ["self", "batch_state"],
                    "returns": "DebugOpts",
                },
                "release_all": {
                    "params": ["self", "manager"],
                    "returns": "None",
                },
                "release_context": {
                    "params": ["self"],
                    "returns": "None",
                },
            },
        },
        "SegmentBundle": {
            "kind": "class",
            "bases": [],
            "fields": ["attn", "moe", "layer", "whole_model", "shared_resources"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
    },
    "batchgen/cuda_graph/buckets.py": {
        "generate_bucket_sizes": {
            "kind": "function",
            "params": ["max_bucket", "num_buckets"],
            "returns": "List[int]",
        },
    },
    "batchgen/cuda_graph/compare.py": {
        "compare_decode_outputs": {
            "kind": "function",
            "params": [
                "adapter",
                "decision",
                "batch_state",
                "segment_name",
                "captured_inputs",
                "graph_outputs",
                "probe_layers",
                "atol",
                "rtol",
                "fail_on_mismatch",
            ],
            "returns": "CompareReport",
        },
    },
    "batchgen/cuda_graph/composition.py": {
        "CaptureContext": {
            "kind": "function",
            "params": ["target", "attrs"],
            "returns": None,
        },
        "allocate_kv_staging": {
            "kind": "function",
            "params": [
                "num_layers",
                "max_bucket",
                "kv_staging_dim",
                "dtype",
                "device",
            ],
            "returns": "Dict[str, torch.Tensor]",
        },
        "compose_layer_from_segments": {
            "kind": "function",
            "params": [
                "attn_segment",
                "moe_segment",
                "glue",
                "extra_input_specs",
                "output_specs",
            ],
            "returns": "CapturableSegment",
        },
        "compose_sequential": {
            "kind": "function",
            "params": ["inner", "extra_input_specs", "output_specs", "forward_fn"],
            "returns": "CapturableSegment",
        },
        "compose_whole_model": {
            "kind": "function",
            "params": [
                "layer_segments",
                "embed",
                "final_norm",
                "lm_head",
                "kv_staging_dim",
                "kv_staging_dtype",
                "max_bucket",
                "device",
                "extra_input_specs",
                "output_specs",
                "forward_fn",
            ],
            "returns": "CapturableSegment",
        },
        "resolve_kv_staging_dtypes": {
            "kind": "function",
            "params": ["kv_staging_dim", "dtype"],
            "returns": "Dict[str, torch.dtype]",
        },
    },
    "batchgen/cuda_graph/flags.py": {
        "DecodeGraphFlags": {
            "kind": "class",
            "bases": [],
            "fields": [
                "compare",
                "compare_fail",
                "compare_atol",
                "compare_rtol",
                "timing",
                "probe_layers",
                "path_log",
                "max_seqlen",
                "memory_diag",
            ],
            "members": {},
            "public_methods": ["from_env", "to_debug_opts"],
            "methods": {
                "from_env": {
                    "params": ["cls"],
                    "returns": "'DecodeGraphFlags'",
                },
                "to_debug_opts": {
                    "params": ["self"],
                    "returns": "DebugOpts",
                },
            },
        },
    },
    "batchgen/cuda_graph/graph_manager.py": {
        "BatchSizeBucketing": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {
                "DEFAULT_BUCKET_SIZES": "[1, 2, 4, 8, 16, 32, 64, 128]",
            },
            "public_methods": ["get_padded_size", "get_padding_count"],
            "methods": {
                "get_padded_size": {
                    "params": ["self", "batch_size"],
                    "returns": "int",
                },
                "get_padding_count": {
                    "params": ["self", "batch_size"],
                    "returns": "int",
                },
            },
        },
        "CUDAGraphManager": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {
                "WARMUP_ITERATIONS": "2",
                "WARMUP_ITERATIONS_SUBSEQUENT": "1",
            },
            "public_methods": [
                "drop_bucket",
                "get_capture_stats",
                "has_bucket_for_all_segments",
                "has_graph",
                "is_captured",
                "register_segment",
                "replay",
                "warmup_and_capture_all",
                "warmup_and_capture_buckets",
            ],
            "methods": {
                "is_captured": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "register_segment": {
                    "params": ["self", "name", "segment"],
                    "returns": "None",
                },
                "warmup_and_capture_all": {
                    "params": ["self"],
                    "returns": "None",
                },
                "warmup_and_capture_buckets": {
                    "params": ["self", "bucket_sizes"],
                    "returns": "None",
                },
                "replay": {
                    "params": ["self", "name", "batch_size", "**inputs"],
                    "returns": "Dict[str, torch.Tensor]",
                },
                "get_capture_stats": {
                    "params": ["self"],
                    "returns": "Dict[str, Any]",
                },
                "has_graph": {
                    "params": ["self", "name", "batch_size"],
                    "returns": "bool",
                },
                "has_bucket_for_all_segments": {
                    "params": ["self", "batch_size"],
                    "returns": "bool",
                },
                "drop_bucket": {
                    "params": ["self", "bucket_size"],
                    "returns": "None",
                },
            },
        },
        "CapturableSegment": {
            "kind": "class",
            "bases": ["Protocol"],
            "fields": [],
            "members": {},
            "public_methods": [
                "forward",
                "get_static_input_specs",
                "get_static_output_specs",
            ],
            "methods": {
                "get_static_input_specs": {
                    "params": ["self", "bucket_size"],
                    "returns": "Dict[str, TensorSpec]",
                },
                "get_static_output_specs": {
                    "params": ["self", "bucket_size"],
                    "returns": "Dict[str, TensorSpec]",
                },
                "forward": {
                    "params": ["self", "**inputs"],
                    "returns": "Dict[str, torch.Tensor]",
                },
            },
        },
        "TensorSpec": {
            "kind": "class",
            "bases": [],
            "fields": ["shape", "dtype", "fill_value"],
            "members": {},
            "public_methods": ["resolve_shape"],
            "methods": {
                "resolve_shape": {
                    "params": ["self", "batch_size"],
                    "returns": "Tuple[int, ...]",
                },
            },
        },
    },
    "batchgen/cuda_graph/vmm_arena.py": {
        "VmmArena": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": ["carve", "is_mapped", "map", "unmap"],
            "methods": {
                "is_mapped": {
                    "params": ["self"],
                    "returns": "bool",
                },
                "map": {
                    "params": ["self"],
                    "returns": "None",
                },
                "unmap": {
                    "params": ["self"],
                    "returns": "None",
                },
                "carve": {
                    "params": ["self", "shape", "dtype"],
                    "returns": "torch.Tensor",
                },
            },
        },
    },
}

# Named in the module's public surface but deliberately NOT pinned, and why:
SKIPPED = {
    "timing.py:StepTimingRecorder": "no consumer outside the package; slated to relocate or be removed",
    "CapturedGraph": "exported but test-only; not part of the stable surface",
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
