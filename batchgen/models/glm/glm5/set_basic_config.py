# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 basic config setup. Standalone — no cross-model imports."""

from batchgen.config.config import EngineConfig
import torch
import logging


def set_basic_config(engine_config: EngineConfig, input_arguments):
    """Set basic config for GLM-5.

    Same as DeepSeek: FP8 weights, BF16 KV (default), BF16 activations.
    """
    engine_config.Basic_Config.log_level = "info"

    # Weight dtype: FP8 E4M3
    engine_config.Basic_Config.weight_dtype = "float8_e4m3fn"
    engine_config.Basic_Config.weight_dtype_torch = torch.float8_e4m3fn

    # KV dtype
    if not input_arguments.get("kv_dtype", None):
        logging.info("kv_dtype not provided, using bfloat16")
        engine_config.Basic_Config.kv_dtype = "bfloat16"
    else:
        kv = input_arguments.kv_dtype.lower()
        if kv in ["bfloat16", "bf16"]:
            engine_config.Basic_Config.kv_dtype = "bfloat16"
        elif kv in ["fp8", "float8", "float8_e4m3fn"]:
            engine_config.Basic_Config.kv_dtype = "float8_e4m3fn"
        else:
            raise ValueError(
                f"Unsupported kv_dtype: {input_arguments.kv_dtype}"
            )

    if engine_config.Basic_Config.kv_dtype == "bfloat16":
        engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16
    elif engine_config.Basic_Config.kv_dtype == "float8_e4m3fn":
        engine_config.Basic_Config.kv_dtype_torch = torch.float8_e4m3fn

    # Attention dtype
    if not input_arguments.get("attention_dtype", None):
        engine_config.Basic_Config.attention_dtype = "bfloat16"
    else:
        att = input_arguments.attention_dtype.lower()
        if att in ["bfloat16", "bf16"]:
            engine_config.Basic_Config.attention_dtype = "bfloat16"
        elif att in ["fp8", "float8", "float8_e4m3fn"]:
            engine_config.Basic_Config.attention_dtype = "float8_e4m3fn"
        else:
            raise ValueError(
                f"Unsupported attention_dtype: {input_arguments.attention_dtype}"
            )

    # Activation dtype
    engine_config.Basic_Config.activation_dtype = "bfloat16"
    engine_config.Basic_Config.activation_dtype_torch = torch.bfloat16

    # Device
    if input_arguments.get("device", None) is None:
        raise ValueError("Device must be specified")
    engine_config.Basic_Config.device = input_arguments.device
    engine_config.Basic_Config.device_torch = torch.device(
        f"cuda:{input_arguments.device}"
    )

    # Module types
    engine_config.Basic_Config.module_types = [
        "attn",
        "routed_expert",
        "shared_expert",
    ]

    # Num threads (deprecated)
    engine_config.Basic_Config.num_threads = 0

    # Prompt / decoding lengths
    max_prompt_length = input_arguments.get(
        "max_prompt_length", None
    ) or input_arguments.get("padding_length", None)
    if not max_prompt_length:
        raise ValueError("Max prompt length must be specified")
    engine_config.Basic_Config.set_max_prompt_length(max_prompt_length)

    if not input_arguments.get("max_decoding_length", None):
        raise ValueError("Max decoding length must be specified")
    engine_config.Basic_Config.max_decoding_length = (
        input_arguments.max_decoding_length
    )

    if input_arguments.get("num_queries") is None:
        raise ValueError("Num queries must be specified")
    engine_config.Basic_Config.num_queries = input_arguments.num_queries

    if input_arguments.get("rank", None) is None:
        raise ValueError("Rank must be specified")
    engine_config.Basic_Config.rank = input_arguments.rank

    if not input_arguments.get("world_size", None):
        raise ValueError("World size must be specified")
    engine_config.Basic_Config.world_size = input_arguments.world_size

    if not input_arguments.get("gpu_arch", None):
        raise ValueError("GPU architecture must be specified")
    if input_arguments.gpu_arch.lower() not in [
        "blackwell",
        "hopper",
        "ampere",
    ]:
        raise ValueError(
            "Currently gpu_arch must be 'blackwell', 'hopper' or 'ampere'"
        )
    engine_config.Basic_Config.gpu_arch = input_arguments.gpu_arch.lower()

    # EP offloading
    if input_arguments.get("enable_ep_with_offloading", False):
        engine_config.EP_Config.enable_offloading = True
        engine_config.EP_Config.offloading_ratio = input_arguments.get(
            "ep_offloading_ratio", 0.0
        )
        logging.info(
            f"EP offloading: enable=True, ratio={engine_config.EP_Config.offloading_ratio}"
        )

    return engine_config
