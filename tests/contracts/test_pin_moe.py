"""Interface-stability pins for batchgen/moe/.

These signatures are load-bearing for downstream callers; changing one is
deliberate -- update this pin in the same PR and say why in the PR
description. Adding or removing a public method, renaming a parameter, and
dropping a return annotation all count.

PINS below is the frozen copy: for a function, its parameter names and
unparsed return annotation; for a class, its base classes, annotated field
names, class-level members, the set of public method names, and each of
those methods' parameters and return annotation.

Nothing here imports batchgen. ``marlin_grouped_moe.py``,
``mxfp4_grouped_gemm.py`` and the resident / streamed layers import torch
and build JIT extensions at module scope. Every assertion is therefore made
against the defining module's own AST, so this suite passes on a machine
with no GPU toolchain and costs milliseconds.

The names pinned here were each resolved to their single top-level
definition inside ``batchgen/moe/``; a name that exists only as a family
pattern is listed in SKIPPED rather than guessed at.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Frozen by reading the defining sources' AST.
PINS = {
    "batchgen/moe/deepep_ll.py": {
        "deepep_available": {
            "kind": "function",
            "params": ["**shape"],
            "returns": "bool",
        },
        "deepep_unavailable_reason": {
            "kind": "function",
            "params": [
                "hidden",
                "topk",
                "num_experts",
                "num_ranks",
                "max_tokens_per_rank",
                "use_fp8",
            ],
            "returns": "Optional[str]",
        },
        "get_low_latency_exchange": {
            "kind": "function",
            "params": [
                "group",
                "max_tokens_per_rank",
                "hidden",
                "num_experts",
                "num_local_experts",
            ],
            "returns": "'DeepEPLowLatencyExchange'",
        },
    },
    "batchgen/moe/dispatch_scatter_3d.py": {
        "dispatch_scatter_3d": {
            "kind": "function",
            "params": [
                "x",
                "topk_indices",
                "act_buffer",
                "expert_start",
                "num_local_experts",
                "max_tokens_padded",
                "expert_counts",
                "expert_counters",
                "topk_pos",
            ],
            "returns": None,
        },
        "dispatch_scatter_ragged": {
            "kind": "function",
            "params": [
                "x",
                "topk_indices",
                "act_buffer",
                "expert_start",
                "num_local_experts",
                "expert_counts",
                "expert_counters",
                "topk_pos",
                "cu_seqlens",
            ],
            "returns": None,
        },
        "reduce_weighted_scatter": {
            "kind": "function",
            "params": [
                "expert_output",
                "topk_pos",
                "topk_weights",
                "N",
                "H",
                "K",
                "output",
            ],
            "returns": "torch.Tensor",
        },
        "reduce_weighted_scatter_fp32": {
            "kind": "function",
            "params": [
                "expert_output",
                "topk_pos",
                "topk_weights",
                "N",
                "H",
                "K",
                "output",
            ],
            "returns": "torch.Tensor",
        },
    },
    "batchgen/moe/fp8_wgmma_pipeline.py": {
        "WGMMAMoEBuffers": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": ["forward", "free_buffers", "register_layer_weights"],
            "methods": {
                "register_layer_weights": {
                    "params": [
                        "self",
                        "layer_id",
                        "gate_weights",
                        "gate_scales",
                        "up_weights",
                        "up_scales",
                        "down_weights",
                        "down_scales",
                    ],
                    "returns": None,
                },
                "free_buffers": {
                    "params": ["self"],
                    "returns": None,
                },
                "forward": {
                    "params": [
                        "self",
                        "layer_id",
                        "global_x",
                        "topk_idx",
                        "topk_weight",
                    ],
                    "returns": None,
                },
            },
        },
        "build_all_modules": {
            "kind": "function",
            "params": [],
            "returns": None,
        },
    },
    "batchgen/moe/fused_moe_bf16_resident.py": {
        "ResidentEPMoELayer": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {
                "num_tokens_per_rank": "None",
            },
            "public_methods": ["forward", "set_num_tokens_per_rank"],
            "methods": {
                "set_num_tokens_per_rank": {
                    "params": ["cls", "num_tokens_per_rank"],
                    "returns": None,
                },
                "forward": {
                    "params": ["self", "x_local", "gate"],
                    "returns": None,
                },
            },
        },
    },
    "batchgen/moe/fused_moe_mxfp4_resident.py": {
        "ResidentEPMXFP4MoELayer": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {
                "num_tokens_per_rank": "None",
                "decode_all_ranks_nonempty": "False",
            },
            "public_methods": [
                "forward",
                "prepare_prefill_output",
                "release_prefill_output",
                "set_latent_tp",
                "set_num_tokens_per_rank",
                "set_rank_token_counts",
            ],
            "methods": {
                "set_latent_tp": {
                    "params": ["self", "tp_size", "tp_rank", "tp_group"],
                    "returns": None,
                },
                "set_num_tokens_per_rank": {
                    "params": ["cls", "num_tokens_per_rank"],
                    "returns": None,
                },
                "set_rank_token_counts": {
                    "params": ["cls", "rank_token_counts"],
                    "returns": None,
                },
                "prepare_prefill_output": {
                    "params": ["cls", "num_global", "hidden_size", "device"],
                    "returns": None,
                },
                "release_prefill_output": {
                    "params": ["cls"],
                    "returns": None,
                },
                "forward": {
                    "params": ["self", "x", "gate", "x_group"],
                    "returns": None,
                },
            },
        },
    },
    "batchgen/moe/int4_single_expert_wgmma.py": {
        "single_expert_int4_forward": {
            "kind": "function",
            "params": [
                "hidden",
                "gate_packed",
                "gate_scale",
                "up_packed",
                "up_scale",
                "down_packed",
                "down_scale",
            ],
            "returns": "torch.Tensor",
        },
    },
    "batchgen/moe/marlin_grouped_moe.py": {
        "single_expert_marlin_decode": {
            "kind": "function",
            "params": [
                "x",
                "gate_qw",
                "gate_scale",
                "up_qw",
                "up_scale",
                "down_qw",
                "down_scale",
                "N",
                "K",
            ],
            "returns": "torch.Tensor",
        },
        "single_expert_marlin_mxfp4_decode": {
            "kind": "function",
            "params": [
                "x",
                "gate_qw",
                "gate_scale",
                "up_qw",
                "up_scale",
                "down_qw",
                "down_scale",
                "N",
                "K",
            ],
            "returns": "torch.Tensor",
        },
    },
    "batchgen/moe/mxfp4_grouped_gemm.py": {
        "mxfp4_linear": {
            "kind": "function",
            "params": ["x", "weight_packed", "weight_scales", "bias", "use_fused"],
            "returns": "torch.Tensor",
        },
    },
    "batchgen/moe/routing/cuda_routing.py": {
        "FusedGateContext": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": ["forward", "router_forward", "warmup"],
            "methods": {
                "warmup": {
                    "params": ["self", "base_buffer"],
                    "returns": None,
                },
                "forward": {
                    "params": [
                        "self",
                        "hidden_states",
                        "logits",
                        "topk_indices",
                        "topk_weights",
                        "num_valid_tokens",
                    ],
                    "returns": None,
                },
                "router_forward": {
                    "params": [
                        "self",
                        "hidden_states",
                        "logits",
                        "num_valid_tokens",
                    ],
                    "returns": None,
                },
            },
        },
        "dispatch_count_gather_cuda": {
            "kind": "function",
            "params": [
                "x",
                "topk_indices",
                "expert_start",
                "num_local_experts",
                "expert_counts",
                "expert_offsets",
                "expert_counters",
                "dispatched_x",
                "topk_pos",
                "num_valid_tokens",
            ],
            "returns": None,
        },
        "gate_sigmoid_topk_cuda": {
            "kind": "function",
            "params": [
                "router_logits",
                "e_score_correction",
                "k",
                "routed_scaling_factor",
                "topk_indices",
                "topk_weights",
                "num_valid_tokens",
                "latent_out",
                "latent_offset",
                "rank_token_counts",
                "rows_per_rank",
            ],
            "returns": None,
        },
        "gate_sigmoid_topk_warp_cuda": {
            "kind": "function",
            "params": [
                "router_logits",
                "e_score_correction",
                "k",
                "routed_scaling_factor",
                "topk_indices",
                "topk_weights",
                "num_valid_tokens",
                "rank_token_counts",
                "rows_per_rank",
            ],
            "returns": None,
        },
        "gate_topk_softmax_cuda": {
            "kind": "function",
            "params": [
                "router_logits",
                "topk_indices",
                "topk_weights",
                "k",
                "num_valid_tokens",
            ],
            "returns": None,
        },
        "glm5_router_gemm_cuda": {
            "kind": "function",
            "params": [
                "hidden_states",
                "router_weight",
                "router_logits",
                "rank_token_counts",
                "bucket_size",
                "world_size",
            ],
            "returns": None,
        },
        "reduce_weighted_scatter_cuda": {
            "kind": "function",
            "params": [
                "expert_output",
                "topk_pos",
                "topk_weights",
                "N",
                "H",
                "K",
                "output",
                "num_valid_tokens",
            ],
            "returns": None,
        },
    },
    "batchgen/moe/streamed_sp8_mxfp4.py": {
        "StreamedSP8LayerBuffer": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": [
                "allow_cross_launch",
                "allow_full_overwrite",
                "begin_prefetch_next",
                "close",
                "load",
                "order_tp_collective_after_cross_launch",
                "participate_empty_prefill_pass",
            ],
            "methods": {
                "begin_prefetch_next": {
                    "params": ["self", "layer_idx"],
                    "returns": None,
                },
                "allow_cross_launch": {
                    "params": ["self"],
                    "returns": None,
                },
                "order_tp_collective_after_cross_launch": {
                    "params": ["self"],
                    "returns": None,
                },
                "allow_full_overwrite": {
                    "params": ["self"],
                    "returns": None,
                },
                "close": {
                    "params": ["self"],
                    "returns": None,
                },
                "load": {
                    "params": ["self", "layer_idx"],
                    "returns": None,
                },
                "participate_empty_prefill_pass": {
                    "params": ["self"],
                    "returns": None,
                },
            },
        },
        "StreamedSP8MXFP4MoELayer": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {
                "prefill_routed_kernel": "'dequant_once'",
            },
            "public_methods": [
                "begin_profile_span",
                "end_profile_span",
                "forward",
                "prefill_dequant_once",
                "prefill_finite_check_enabled",
                "prefill_finite_check_snapshot",
                "prefill_profile_snapshot",
                "prefill_routed_kernel_selected",
                "record_prefill_finite_check",
                "reset_prefill_profile",
            ],
            "methods": {
                "reset_prefill_profile": {
                    "params": ["cls", "enabled"],
                    "returns": "None",
                },
                "prefill_routed_kernel_selected": {
                    "params": ["cls"],
                    "returns": "str",
                },
                "prefill_dequant_once": {
                    "params": ["cls", "shard", "device", "profile"],
                    "returns": None,
                },
                "prefill_finite_check_enabled": {
                    "params": ["cls"],
                    "returns": "bool",
                },
                "record_prefill_finite_check": {
                    "params": ["cls", "layer_idx", "stage", "tensor"],
                    "returns": "None",
                },
                "prefill_finite_check_snapshot": {
                    "params": ["cls"],
                    "returns": "dict",
                },
                "begin_profile_span": {
                    "params": ["cls"],
                    "returns": None,
                },
                "end_profile_span": {
                    "params": ["cls", "name", "start"],
                    "returns": "None",
                },
                "prefill_profile_snapshot": {
                    "params": ["cls"],
                    "returns": "dict",
                },
                "forward": {
                    "params": ["self", "x", "gate", "num_rows"],
                    "returns": "torch.Tensor",
                },
            },
        },
    },
}

# Named in the module's public surface but deliberately NOT pinned, and why:
SKIPPED = {
    "dispatch_scatter_3d_ragged": (
        "no such symbol; the defined name is dispatch_scatter_ragged in the "
        "same module, and that is what is pinned"
    ),
    "fused_mxfp4_{expert,grouped}_*": "pattern, not a single symbol",
    "marlin_* transforms and preps": "pattern, not a single symbol",
    "build_resident_ep_*": "pattern, not a single symbol",
    "grouped_fp8_blockwise_s3/_fused_s1/...": (
        "pattern, not a single symbol; build_all_modules and WGMMAMoEBuffers "
        "from the same group are pinned"
    ),
    "_load_dispatch_reduce_module / _load_module / _get_scale_perms": "private names; not part of the stable surface",
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
