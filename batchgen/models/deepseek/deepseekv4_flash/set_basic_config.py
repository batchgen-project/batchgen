# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash engine basic config.

Keep this file local to the V4 package. Model packages must not import helpers
from other model packages because those helpers often encode architecture-
specific defaults.
"""

from __future__ import annotations

import logging

import torch

from batchgen.config.config import EngineConfig


def _get_arg(input_arguments, name: str, default=None):
    if hasattr(input_arguments, "get"):
        return input_arguments.get(name, default)
    return getattr(input_arguments, name, default)


def set_basic_config(engine_config: EngineConfig, input_arguments):
    engine_config.Basic_Config.log_level = "info"

    engine_config.Basic_Config.weight_dtype = "float8_e4m3fn"
    engine_config.Basic_Config.weight_dtype_torch = torch.float8_e4m3fn

    kv_dtype = _get_arg(input_arguments, "kv_dtype")
    if not kv_dtype:
        logging.info("kv_dtype is not provided, using bfloat16 as default")
        engine_config.Basic_Config.kv_dtype = "bfloat16"
    else:
        logging.info("kv_dtype is set to %s", kv_dtype)
        normalized = kv_dtype.lower()
        if normalized in ["bfloat16", "bf16"]:
            engine_config.Basic_Config.kv_dtype = "bfloat16"
        elif normalized in ["fp8", "float8", "float8_e4m3fn"]:
            engine_config.Basic_Config.kv_dtype = "float8_e4m3fn"
        else:
            raise ValueError(
                f"Unsupported kv_dtype: {kv_dtype}, only support "
                "['bfloat16','float8_e4m3fn']"
            )

    if engine_config.Basic_Config.kv_dtype == "bfloat16":
        engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16
    elif engine_config.Basic_Config.kv_dtype == "float8_e4m3fn":
        engine_config.Basic_Config.kv_dtype_torch = torch.float8_e4m3fn

    attention_dtype = _get_arg(input_arguments, "attention_dtype")
    if not attention_dtype:
        logging.info("attention_dtype is not provided, using bfloat16 as default")
        engine_config.Basic_Config.attention_dtype = "bfloat16"
    else:
        normalized = attention_dtype.lower()
        if normalized in ["bfloat16", "bf16"]:
            engine_config.Basic_Config.attention_dtype = "bfloat16"
        elif normalized in ["fp8", "float8", "float8_e4m3fn"]:
            engine_config.Basic_Config.attention_dtype = "float8_e4m3fn"
        else:
            raise ValueError(
                f"Unsupported attention_dtype: {attention_dtype}, only support "
                "['bfloat16','float8_e4m3fn']"
            )

    engine_config.Basic_Config.activation_dtype = "bfloat16"
    engine_config.Basic_Config.activation_dtype_torch = torch.bfloat16

    device = _get_arg(input_arguments, "device")
    if device is None:
        raise ValueError("Device must be specified")
    engine_config.Basic_Config.device = device
    engine_config.Basic_Config.device_torch = torch.device(f"cuda:{device}")

    attn_mode = _get_arg(input_arguments, "attn_mode", 3)
    if attn_mode not in [1, 2, 3]:
        raise ValueError("Currently attn_mode must be 1, 2, or 3")
    engine_config.Basic_Config.attn_mode = attn_mode

    engine_config.Basic_Config.module_types = [
        "attn",
        "attn_cr4",
        "attn_cr128",
        "routed_expert",
        "shared_expert",
    ]
    engine_config.Basic_Config.num_threads = 0

    padding_length = _get_arg(input_arguments, "padding_length")
    if not padding_length:
        raise ValueError("Padding length must be specified")
    engine_config.Basic_Config.padding_length = padding_length

    max_decoding_length = _get_arg(input_arguments, "max_decoding_length")
    if not max_decoding_length:
        raise ValueError("Max decoding length must be specified")
    engine_config.Basic_Config.max_decoding_length = max_decoding_length

    num_queries = _get_arg(input_arguments, "num_queries")
    if num_queries is None:
        raise ValueError("Num queries must be specified")
    engine_config.Basic_Config.num_queries = num_queries

    rank = _get_arg(input_arguments, "rank")
    if rank is None:
        raise ValueError("Rank must be specified")
    engine_config.Basic_Config.rank = rank

    world_size = _get_arg(input_arguments, "world_size")
    if not world_size:
        raise ValueError("World size must be specified")
    engine_config.Basic_Config.world_size = world_size

    gpu_arch = _get_arg(input_arguments, "gpu_arch")
    if not gpu_arch:
        raise ValueError("GPU architecture must be specified")
    if gpu_arch.lower() not in ["hopper", "ampere"]:
        raise ValueError("Currently gpu_arch must be 'hopper', or 'ampere'")
    engine_config.Basic_Config.gpu_arch = gpu_arch.lower()

    if _get_arg(input_arguments, "enable_ep_with_offloading", False):
        engine_config.EP_Config.enable_offloading = True
        engine_config.EP_Config.offloading_ratio = _get_arg(
            input_arguments,
            "ep_offloading_ratio",
            0.0,
        )
        logging.info(
            "EP offloading config set: enable_offloading=True, offloading_ratio=%s",
            engine_config.EP_Config.offloading_ratio,
        )

    return engine_config
