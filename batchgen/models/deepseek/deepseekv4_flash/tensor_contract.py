# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash tensor contract helpers.

This module mirrors the vendored V4 ``assets/inference/model.py`` checkpoint
names. It intentionally does not import DeepSeek-V3 modeling code.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


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


def iter_attention_tensor_names(compress_ratio: int) -> Iterable[str]:
    yield from BASE_ATTN_TENSORS
    if compress_ratio:
        yield from COMPRESSOR_ATTN_TENSORS
    if compress_ratio == 4:
        yield from INDEXER_ATTN_TENSORS


def _get_config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def build_v4_weight_contract(config: Any) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    """Build BatchGen tensor routing for V4 checkpoint keys.

    Returns:
        (state_dict_name_map, weight_copy_task)
    """

    num_layers = int(_get_config_value(config, "num_hidden_layers", 43))
    num_experts = int(_get_config_value(config, "n_routed_experts", 256))
    compress_ratios = list(_get_config_value(config, "compress_ratios", []))
    if len(compress_ratios) < num_layers:
        compress_ratios.extend([0] * (num_layers - len(compress_ratios)))

    state_dict_name_map: Dict[str, Dict[str, str]] = {}
    weight_copy_task: Dict[str, List[str]] = {
        "attn": [],
        "routed_expert": [],
        "shared_expert": [],
    }

    for layer_idx in range(num_layers):
        attn_key = f"attn_{layer_idx}"
        for tensor_name in iter_attention_tensor_names(int(compress_ratios[layer_idx])):
            state_dict_name_map[f"layers.{layer_idx}.attn.{tensor_name}"] = {
                "module_key": attn_key,
                "tensor_key": tensor_name,
            }
        weight_copy_task["attn"].append(attn_key)

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

    return state_dict_name_map, weight_copy_task
