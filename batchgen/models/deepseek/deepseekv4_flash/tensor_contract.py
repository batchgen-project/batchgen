# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash tensor contract helpers.

This module is the single source of truth for the V4 checkpoint naming
convention used by the parameter server and the parallel strategy manager.  It
mirrors the vendored V4 ``assets/inference/model.py`` checkpoint names and
intentionally does not import DeepSeek-V3 modeling code.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


# Checkpoint-to-BatchGen naming convention
# ---------------------------------------
# V4 checkpoints use the native asset names:
#   embed.weight, head.weight, norm.weight, hc_head_*
#   layers.{L}.attn.*, layers.{L}.ffn.*, layers.{L}.attn_norm.*
#   layers.{L}.ffn_norm.*, layers.{L}.hc_*
#   mtp.{M}.*
#
# BatchGen model instances use the standard worker surface:
#   model.embed_tokens.weight, lm_head.weight, model.norm.weight
#   model.layers.{L}.self_attn.*, model.layers.{L}.mlp.*
#   model.layers.{L}.attn_norm.*, model.layers.{L}.ffn_norm.*
#
# Runtime bundle tensors must be routed to Parameter_Server module keys instead
# of loaded into the skeleton state dict.  Skeleton tensors are loaded by PSM via
# ``model_key_to_checkpoint_key``.  Keep all new checkpoint prefixes in this file
# so PSM does not grow model-specific string remapping logic again.

ROOT_MODEL_TO_CHECKPOINT: Dict[str, str] = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "head.weight",
}

LAYER_MODEL_TO_CHECKPOINT_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("self_attn.", "attn."),
    ("attn.", "attn."),
    ("mlp.gate.", "ffn.gate."),
    ("ffn.gate.", "ffn.gate."),
    ("attn_norm.", "attn_norm."),
    ("ffn_norm.", "ffn_norm."),
)


BASE_ATTN_TENSORS: Tuple[str, ...] = (
    "attn_sink",
    "kv_norm.weight",
    "q_norm.weight",
    "wkv.scale",
    "wkv.weight",
    "wo_a.scale",
    "wo_a.weight",
    "wo_b.scale",
    "wo_b.weight",
    "wq_a.scale",
    "wq_a.weight",
    "wq_b.scale",
    "wq_b.weight",
)

COMPRESSOR_ATTN_TENSORS: Tuple[str, ...] = (
    "compressor.ape",
    "compressor.norm.weight",
    "compressor.wgate.weight",
    "compressor.wkv.weight",
)

INDEXER_ATTN_TENSORS: Tuple[str, ...] = (
    "indexer.compressor.ape",
    "indexer.compressor.norm.weight",
    "indexer.compressor.wgate.weight",
    "indexer.compressor.wkv.weight",
    "indexer.weights_proj.weight",
    "indexer.wq_b.scale",
    "indexer.wq_b.weight",
)

EXPERT_TENSORS: Tuple[str, ...] = (
    "w1.scale",
    "w1.weight",
    "w2.scale",
    "w2.weight",
    "w3.scale",
    "w3.weight",
)

ATTN_TASK_BASE = "attn"
ATTN_TASK_CR4 = "attn_cr4"
ATTN_TASK_CR128 = "attn_cr128"
ATTN_TASK_NAMES: Tuple[str, ...] = (
    ATTN_TASK_BASE,
    ATTN_TASK_CR4,
    ATTN_TASK_CR128,
)

MTP_BASE_TENSORS: Tuple[str, ...] = (
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.scale",
    "attn.wkv.weight",
    "attn.wo_a.scale",
    "attn.wo_a.weight",
    "attn.wo_b.scale",
    "attn.wo_b.weight",
    "attn.wq_a.scale",
    "attn.wq_a.weight",
    "attn.wq_b.scale",
    "attn.wq_b.weight",
    "attn_norm.weight",
    "e_proj.scale",
    "e_proj.weight",
    "enorm.weight",
    "ffn.gate.bias",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.shared_experts.w3.weight",
    "ffn_norm.weight",
    "h_proj.scale",
    "h_proj.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "hnorm.weight",
    "norm.weight",
)


def model_key_to_checkpoint_key(model_key: str) -> str:
    """Map a BatchGen ``named_parameters`` key to the V4 checkpoint key.

    PSM must call this helper instead of carrying its own string rules.  Keys
    that are already checkpoint-native are returned unchanged.
    """

    if model_key in ROOT_MODEL_TO_CHECKPOINT:
        return ROOT_MODEL_TO_CHECKPOINT[model_key]
    if model_key.startswith("model.hc_head_"):
        return model_key.removeprefix("model.")
    if not model_key.startswith("model.layers."):
        return model_key

    parts = model_key.split(".")
    if len(parts) < 4:
        return model_key
    layer_idx = parts[2]
    rest = ".".join(parts[3:])
    for model_prefix, checkpoint_prefix in LAYER_MODEL_TO_CHECKPOINT_PREFIXES:
        if rest.startswith(model_prefix):
            return (
                f"layers.{layer_idx}.{checkpoint_prefix}"
                f"{rest.removeprefix(model_prefix)}"
            )
    if rest.startswith("hc_"):
        return f"layers.{layer_idx}.{rest}"
    return model_key


def iter_attention_tensor_names(compress_ratio: int) -> Iterable[str]:
    yield from BASE_ATTN_TENSORS
    if compress_ratio:
        yield from COMPRESSOR_ATTN_TENSORS
    if compress_ratio == 4:
        yield from INDEXER_ATTN_TENSORS


def _get_config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def attention_task_name(compress_ratio: int) -> str:
    if compress_ratio == 0:
        return ATTN_TASK_BASE
    if compress_ratio == 4:
        return ATTN_TASK_CR4
    if compress_ratio == 128:
        return ATTN_TASK_CR128
    raise ValueError(f"Unsupported DeepSeek-V4 attention compress_ratio={compress_ratio}")


def build_v4_weight_contract(
    config: Any,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    """Build BatchGen tensor routing for V4 checkpoint keys.

    Returns:
        (state_dict_name_map, weight_copy_task)
    """

    num_layers = int(_get_config_value(config, "num_hidden_layers", 43))
    num_experts = int(_get_config_value(config, "n_routed_experts", 256))
    num_mtp_layers = int(_get_config_value(config, "num_nextn_predict_layers", 1))
    compress_ratios = list(_get_config_value(config, "compress_ratios", []))
    if len(compress_ratios) < num_layers:
        compress_ratios.extend([0] * (num_layers - len(compress_ratios)))

    state_dict_name_map: Dict[str, Dict[str, str]] = {}
    weight_copy_task: Dict[str, List[str]] = {
        task_name: []
        for task_name in ATTN_TASK_NAMES
    }
    weight_copy_task.update({
        "routed_expert": [],
        "shared_expert": [],
    })

    for layer_idx in range(num_layers):
        compress_ratio = int(compress_ratios[layer_idx])
        attn_key = f"attn_{layer_idx}"
        for tensor_name in iter_attention_tensor_names(compress_ratio):
            state_dict_name_map[f"layers.{layer_idx}.attn.{tensor_name}"] = {
                "module_key": attn_key,
                "tensor_key": tensor_name,
            }
        weight_copy_task[attention_task_name(compress_ratio)].append(attn_key)

        shared_key = f"shared_expert_{layer_idx}"
        for tensor_name in EXPERT_TENSORS:
            state_dict_name_map[
                f"layers.{layer_idx}.ffn.shared_experts.{tensor_name}"
            ] = {
                "module_key": shared_key,
                "tensor_key": tensor_name,
            }
        weight_copy_task["shared_expert"].append(shared_key)

        for expert_idx in range(num_experts):
            routed_key = f"routed_expert_{layer_idx}_{expert_idx}"
            for tensor_name in EXPERT_TENSORS:
                state_dict_name_map[
                    f"layers.{layer_idx}.ffn.experts.{expert_idx}.{tensor_name}"
                ] = {
                    "module_key": routed_key,
                    "tensor_key": tensor_name,
                }
            weight_copy_task["routed_expert"].append(routed_key)

    for mtp_idx in range(num_mtp_layers):
        module_key = f"unused_mtp_{mtp_idx}"
        for tensor_name in MTP_BASE_TENSORS:
            state_dict_name_map[f"mtp.{mtp_idx}.{tensor_name}"] = {
                "module_key": module_key,
                "tensor_key": tensor_name,
            }
        for expert_idx in range(num_experts):
            for tensor_name in EXPERT_TENSORS:
                state_dict_name_map[
                    f"mtp.{mtp_idx}.ffn.experts.{expert_idx}.{tensor_name}"
                ] = {
                    "module_key": module_key,
                    "tensor_key": f"ffn.experts.{expert_idx}.{tensor_name}",
                }

    return state_dict_name_map, weight_copy_task


def checkpoint_names_from_metadata_rows(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """Extract checkpoint tensor names from extractor JSONL rows."""

    return [str(row["name"]) for row in rows]


def audit_v4_naming_contract(
    checkpoint_names: Iterable[str],
    model_parameter_names: Iterable[str],
    config: Any,
    allowed_unmapped_prefixes: Sequence[str] = (),
) -> Dict[str, List[str]]:
    """Compare metadata-extracted names against the V4 BatchGen contract.

    Args:
        checkpoint_names: tensor names from
            ``batchgen_design/tools/extract_tensor_metadata.py`` JSONL output.
        model_parameter_names: ``name`` values from ``model.named_parameters()``.
        config: V4 config-like object used to build runtime bundle routing.
        allowed_unmapped_prefixes: optional checkpoint prefixes intentionally
            deferred by a caller.

    Returns:
        Dict with deterministic lists:
          - ``missing_skeleton``: BatchGen skeleton params not found in ckpt.
          - ``unmapped_checkpoint``: ckpt tensors neither runtime-routed nor
            skeleton-loaded.
    """

    checkpoint_set = set(checkpoint_names)
    state_dict_name_map, _ = build_v4_weight_contract(config)
    runtime_names = set(state_dict_name_map)
    skeleton_names = {
        model_key_to_checkpoint_key(name)
        for name in model_parameter_names
        if model_key_to_checkpoint_key(name) not in runtime_names
    }
    allowed_prefixes = tuple(allowed_unmapped_prefixes)
    unmapped = sorted(
        name
        for name in checkpoint_set - runtime_names - skeleton_names
        if not name.startswith(allowed_prefixes)
    )
    missing_skeleton = sorted(skeleton_names - checkpoint_set)
    return {
        "missing_skeleton": missing_skeleton,
        "unmapped_checkpoint": unmapped,
    }
