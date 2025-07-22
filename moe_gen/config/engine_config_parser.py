import json
from typing import Any, Dict

import torch

from .config import (
    BasicConfig,
    EngineConfig,
    EPConfig,
    GPUBufferConfig,
    KVStorageConfig,
    ModuleBatchingConfig,
)


def parse_config_from_json(config_path: str) -> EngineConfig:
    """
    Parse a JSON configuration file into an EngineConfig instance.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        An EngineConfig instance with all attributes populated from the JSON

    Raises:
        ValueError: If required keys are missing in the configuration
        FileNotFoundError: If the config file doesn't exist
    """
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in config file: {config_path}")

    # Create a new engine config
    engine_config = EngineConfig()

    # Process Basic_Config
    if "Basic_Config" in config_dict:
        _parse_basic_config(
            engine_config.Basic_Config, config_dict["Basic_Config"]
        )
    # else:
    #     raise ValueError("Missing 'Basic_Config' section in config file")

    # Process Module_Batching_Config
    if "Module_Batching_Config" in config_dict:
        _parse_module_batching_config(
            engine_config.Module_Batching_Config,
            config_dict["Module_Batching_Config"],
        )
    # else:
    #     raise ValueError("Missing 'Module_Batching_Config' section in config file")

    # Process GPU_Buffer_Config
    if "GPU_Buffer_Config" in config_dict:
        _parse_gpu_buffer_config(
            engine_config.GPU_Buffer_Config, config_dict["GPU_Buffer_Config"]
        )
    # else:
    #     raise ValueError("Missing 'GPU_Buffer_Config' section in config file")

    # Process KV_Storage_Config
    if "KV_Storage_Config" in config_dict:
        _parse_kv_storage_config(
            engine_config.KV_Storage_Config, config_dict["KV_Storage_Config"]
        )
    # else:
    #     # This section seems optional in your example JSON
    #     pass

    # Process EP_Config
    if "EP_Config" in config_dict:
        _parse_ep_config(engine_config.EP_Config, config_dict["EP_Config"])
    # else:
    #     raise ValueError("Missing 'EP_Config' section in config file")

    return engine_config


def _parse_basic_config(
    basic_config: BasicConfig, config_dict: Dict[str, Any]
) -> None:
    """
    Parse the Basic_Config section from a dictionary into a BasicConfig instance.

    Args:
        basic_config: The BasicConfig instance to populate
        config_dict: Dictionary containing the configuration values

    Raises:
        ValueError: If an unknown key is found in the configuration
    """
    valid_fields = {f.name for f in basic_config.__dataclass_fields__.values()}

    for key, value in config_dict.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown key in Basic_Config: {key}")

        setattr(basic_config, key, value)

    # Convert string dtypes to torch dtypes if applicable
    if basic_config.weight_dtype:
        basic_config.weight_dtype_torch = BasicConfig._str_to_torch_dtype(
            basic_config.weight_dtype
        )

    if basic_config.kv_dtype:
        basic_config.kv_dtype_torch = BasicConfig._str_to_torch_dtype(
            basic_config.kv_dtype
        )

    if basic_config.activation_dtype:
        basic_config.activation_dtype_torch = BasicConfig._str_to_torch_dtype(
            basic_config.activation_dtype
        )

    # Handle device
    if basic_config.device and basic_config.device.startswith("cuda"):
        basic_config.device_torch = torch.device(basic_config.device)


def _parse_module_batching_config(
    module_config: ModuleBatchingConfig, config_dict: Dict[str, Any]
) -> None:
    """
    Parse the Module_Batching_Config section from a dictionary into a ModuleBatchingConfig instance.

    Args:
        module_config: The ModuleBatchingConfig instance to populate
        config_dict: Dictionary containing the configuration values

    Raises:
        ValueError: If an unknown key is found in the configuration
    """
    valid_fields = {f.name for f in module_config.__dataclass_fields__.values()}

    for key, value in config_dict.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown key in Module_Batching_Config: {key}")

        setattr(module_config, key, value)


def _parse_gpu_buffer_config(
    gpu_config: GPUBufferConfig, config_dict: Dict[str, Any]
) -> None:
    """
    Parse the GPU_Buffer_Config section from a dictionary into a GPUBufferConfig instance.

    Args:
        gpu_config: The GPUBufferConfig instance to populate
        config_dict: Dictionary containing the configuration values

    Raises:
        ValueError: If an unknown key is found in the configuration
    """
    valid_fields = {f.name for f in gpu_config.__dataclass_fields__.values()}

    for key, value in config_dict.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown key in GPU_Buffer_Config: {key}")

        setattr(gpu_config, key, value)


def _parse_kv_storage_config(
    kv_config: KVStorageConfig, config_dict: Dict[str, Any]
) -> None:
    """
    Parse the KV_Storage_Config section from a dictionary into a KVStorageConfig instance.

    Args:
        kv_config: The KVStorageConfig instance to populate
        config_dict: Dictionary containing the configuration values

    Raises:
        ValueError: If an unknown key is found in the configuration
    """
    valid_fields = {f.name for f in kv_config.__dataclass_fields__.values()}

    for key, value in config_dict.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown key in KV_Storage_Config: {key}")

        setattr(kv_config, key, value)


def _parse_ep_config(ep_config: EPConfig, config_dict: Dict[str, Any]) -> None:
    """
    Parse the EP_Config section from a dictionary into an EPConfig instance.

    Args:
        ep_config: The EPConfig instance to populate
        config_dict: Dictionary containing the configuration values

    Raises:
        ValueError: If an unknown key is found in the configuration
    """
    valid_fields = {f.name for f in ep_config.__dataclass_fields__.values()}

    for key, value in config_dict.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown key in EP_Config: {key}")

        setattr(ep_config, key, value)
