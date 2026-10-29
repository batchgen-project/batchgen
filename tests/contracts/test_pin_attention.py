"""Interface-stability pins for batchgen/attention/.

These signatures are load-bearing for downstream callers; changing one is
deliberate -- update this pin in the same PR and say why in the PR
description. Adding or removing a public method, renaming a parameter, and
dropping a return annotation all count.

PINS below is the frozen copy: for a function, its parameter names and
unparsed return annotation; for a class, its base classes, annotated field
names, class-level members, the set of public method names, and each of
those methods' parameters and return annotation.

Nothing here imports batchgen. ``mla/fa3_backend.py`` imports
flash_attn_interface and deep_gemm at module scope, which no dev machine
has, so the whole mla package is unimportable off a GPU host. Every
assertion is therefore made against the defining module's own AST, so this
suite passes on a machine with no GPU toolchain and costs milliseconds.

The names pinned here were each resolved to their single top-level
definition inside ``batchgen/attention/``; a name that exists only as a
family pattern is listed in SKIPPED rather than guessed at.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Frozen by reading the defining sources' AST.
PINS = {
    "batchgen/attention/dsa/aux_indexer_k_writer.py": {
        "write_aux_indexer_k": {
            "kind": "function",
            "params": [
                "aux_blocked_k",
                "indexer_k",
                "page_table",
                "slot_indices",
                "token_indices",
                "page_size_tokens",
                "k_fp8_out",
                "scales_out",
                "num_valid_tokens",
            ],
            "returns": "Tuple[torch.Tensor, Optional[torch.Tensor]]",
        },
        "write_aux_indexer_k_norm_rope": {
            "kind": "function",
            "params": [
                "aux_blocked_k",
                "k_raw",
                "ln_weight",
                "ln_bias",
                "ln_eps",
                "cos_cache",
                "sin_cache",
                "positions",
                "page_table",
                "slot_indices",
                "token_indices",
                "page_size_tokens",
                "apply_hadamard",
                "k_fp8_out",
                "scales_out",
                "num_valid_tokens",
            ],
            "returns": "Tuple[torch.Tensor, torch.Tensor]",
        },
    },
    "batchgen/attention/dsa/unified_selector.py": {
        "select_mla_kv_for_flashmla_bf16_out": {
            "kind": "function",
            "params": [
                "primary_blocked_k",
                "primary_page_table",
                "cache_seqlens",
                "long_topk_indices",
                "page_size",
                "selected_mla_kv",
                "selected_lengths",
                "selected_indices",
                "row_modes",
                "index_topk",
                "return_indices",
                "primary_slot_indices",
                "num_valid_tokens",
            ],
            "returns": "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]",
        },
    },
    "batchgen/attention/fused_kernels/ops.py": {
        "cuda_add_rmsnorm": {
            "kind": "function",
            "params": [
                "residual",
                "hidden",
                "weight",
                "eps",
                "num_valid_tokens",
                "row_valid_mask",
            ],
            "returns": "tuple",
        },
        "cuda_qkv_split": {
            "kind": "function",
            "params": ["qkv", "q_size", "kv_size", "num_valid_tokens"],
            "returns": "tuple",
        },
        "cuda_qkv_split_inplace": {
            "kind": "function",
            "params": [
                "qkv",
                "q_out",
                "k_out",
                "v_out",
                "q_size",
                "kv_size",
                "num_valid_tokens",
            ],
            "returns": None,
        },
        "cuda_rmsnorm": {
            "kind": "function",
            "params": ["x", "weight", "eps", "num_valid_tokens"],
            "returns": "torch.Tensor",
        },
        "cuda_rope": {
            "kind": "function",
            "params": ["query", "key", "cos", "sin", "num_valid_tokens"],
            "returns": "tuple",
        },
        "preload_fused_attention_kernels": {
            "kind": "function",
            "params": [],
            "returns": None,
        },
    },
    "batchgen/attention/fused_kernels/qkv_wgmma.py": {
        "create_qkv_tma_desc": {
            "kind": "function",
            "params": ["tensor", "block_rows", "block_cols"],
            "returns": "torch.Tensor",
        },
        "cuda_qkv_wgmma": {
            "kind": "function",
            "params": [
                "input",
                "weight",
                "bias",
                "q_size",
                "kv_size",
                "num_valid_tokens",
                "rope_cos",
                "rope_sin",
                "head_dim",
            ],
            "returns": "tuple",
        },
        "cuda_qkv_wgmma_inplace": {
            "kind": "function",
            "params": [
                "Q_out",
                "K_out",
                "V_out",
                "tma_desc_a",
                "tma_desc_b",
                "bias",
                "rope_cos",
                "rope_sin",
                "head_dim",
                "num_valid_tokens",
                "M",
                "N",
                "K",
                "q_size",
                "kv_size",
            ],
            "returns": "None",
        },
        "is_qkv_wgmma_available": {
            "kind": "function",
            "params": [],
            "returns": "bool",
        },
    },
    "batchgen/attention/gqa/batchgen_gqa_decode_bf16.py": {
        "batchgen_gqa_decode_bf16": {
            "kind": "function",
            "params": [
                "q",
                "k_cache",
                "v_cache",
                "cache_seqlens",
                "block_table",
                "sinks",
                "softmax_scale",
                "sliding_window",
            ],
            "returns": "Tuple[torch.Tensor, torch.Tensor]",
        },
    },
    "batchgen/attention/gqa/fa_decode.py": {
        "gqa_decode_fa": {
            "kind": "function",
            "params": [
                "q",
                "k_cache",
                "v_cache",
                "cache_seqlens",
                "block_table",
                "sinks",
                "softmax_scale",
                "sliding_window",
            ],
            "returns": "Tuple[torch.Tensor, torch.Tensor]",
        },
    },
    "batchgen/attention/gqa/fa_extend.py": {
        "gqa_extend_fa": {
            "kind": "function",
            "params": [
                "q",
                "k_cache",
                "v_cache",
                "cache_seqlens",
                "page_table",
                "cu_seqlens_q",
                "max_seqlen_q",
                "sinks",
                "softmax_scale",
                "sliding_window",
            ],
            "returns": "Tuple[torch.Tensor, Optional[torch.Tensor]]",
        },
    },
    "batchgen/attention/gqa/fa_prefill.py": {
        "gqa_prefill_fa": {
            "kind": "function",
            "params": [
                "q",
                "k",
                "v",
                "cu_seqlens_q",
                "cu_seqlens_k",
                "max_seqlen_q",
                "max_seqlen_k",
                "sinks",
                "softmax_scale",
                "sliding_window",
            ],
            "returns": "Tuple[torch.Tensor, torch.Tensor]",
        },
    },
    "batchgen/attention/gqa/gqa_attention.py": {
        "gqa_attention_with_sinks": {
            "kind": "function",
            "params": [
                "query",
                "key",
                "value",
                "sinks",
                "scale",
                "sliding_window",
                "is_decode",
                "cache_seqlens",
            ],
            "returns": "torch.Tensor",
        },
    },
    "batchgen/attention/mla/fa2_backend.py": {
        "mla_prefill_flashattention2": {
            "kind": "function",
            "params": ["self", "hidden_states", "attention_mask", "position_ids"],
            "returns": None,
        },
    },
    "batchgen/attention/mla/fa3_backend.py": {
        "act_quant": {
            "kind": "function",
            "params": [
                "x",
                "block_size",
                "eps",
                "num_valid_tokens",
                "scale_tma_aligned",
            ],
            "returns": "Tuple[torch.Tensor, torch.Tensor]",
        },
        "mla_prefill_flashattention3": {
            "kind": "function",
            "params": ["self", "hidden_states", "attention_mask", "position_ids"],
            "returns": "tuple[torch.Tensor, torch.Tensor]",
        },
        "w8a16_gemm": {
            "kind": "function",
            "params": [
                "weight_data_fp8",
                "weight_scale_inv_fp32",
                "activation_bf16",
                "x_fp8",
            ],
            "returns": "torch.Tensor",
        },
        "w8a16_gemm_act_quant": {
            "kind": "function",
            "params": ["activation_bf16"],
            "returns": "tuple[torch.Tensor, torch.Tensor]",
        },
    },
    "batchgen/attention/mla/flashmla_backend.py": {
        "deepseek_v3_dequantization": {
            "kind": "function",
            "params": ["x", "s", "block_size"],
            "returns": "torch.Tensor",
        },
        "fused_get_query_states_triton": {
            "kind": "function",
            "params": ["self", "q"],
            "returns": None,
        },
        "mla_decoding_flashmla": {
            "kind": "function",
            "params": [
                "self",
                "hidden_states",
                "past_key_states",
                "past_value_states",
                "attention_mask",
                "position_ids",
                "scale",
                "cache_seqlens_override",
                "max_seqlen_override",
            ],
            "returns": None,
        },
    },
    "batchgen/attention/mla/rotary_embedding.py": {
        "rotary_pos_emb_interleaved_native": {
            "kind": "function",
            "params": [
                "t",
                "cos_cached",
                "sin_cached",
                "position_ids",
                "unsqueeze_dim",
            ],
            "returns": None,
        },
    },
    "batchgen/attention/mla/torch_backend.py": {
        "mla_decoding_torch": {
            "kind": "function",
            "params": [
                "self",
                "hidden_states",
                "past_key_states",
                "past_value_states",
                "attention_mask",
                "position_ids",
            ],
            "returns": "tuple[torch.Tensor, torch.Tensor]",
        },
    },
    "batchgen/attention/sink/sink_softmax.py": {
        "softmax_with_sinks": {
            "kind": "function",
            "params": ["logits", "sinks", "dim", "return_lse"],
            "returns": "Tuple[torch.Tensor, Optional[torch.Tensor]]",
        },
    },
}

# Named in the module's public surface but deliberately NOT pinned, and why:
SKIPPED = {
    "mla_prefill_flashattention{2,3}*": "pattern, not a single symbol; the two exact base names are pinned",
    "mla_decoding_flashmla*": (
        "pattern, not a single symbol; the exact base name "
        "mla_decoding_flashmla is pinned"
    ),
    "prepare/run_*sparse_flash_mla_decode*": "pattern, not a single symbol",
    "build_glm5_dsa_*_inputs": "pattern, not a single symbol; slated to relocate",
    "GLM5_SPARSE_FA3_NUM_SPLITS": "a module constant, not a callable surface; slated to relocate",
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
