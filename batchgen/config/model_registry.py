# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Model configuration registry for BatchGen.

This module provides:
1. A registry for model-specific config classes
2. Auto-detection of model type from config.json
3. A unified load_config() function to replace AutoConfig.from_pretrained()

Usage:
    from batchgen.config.model_registry import load_config

    # Load config from local model directory
    config = load_config("/path/to/model")

    # Access config attributes
    print(config.model_type)
    print(config.num_hidden_layers)
    print(config.is_moe())
"""

from typing import Dict, Type, TYPE_CHECKING
from pathlib import Path
import json
import logging

if TYPE_CHECKING:
    from .model_config import BaseModelConfig

logger = logging.getLogger(__name__)


# Registry mapping model_type -> config class
CONFIG_REGISTRY: Dict[str, Type["BaseModelConfig"]] = {}

# Architecture name patterns for fallback detection
ARCH_PATTERNS: Dict[str, str] = {
    "DeepseekV3": "deepseek_v3",
    "DeepseekV2": "deepseek_v2",
    "Mixtral": "mixtral",
    "GptOss": "gpt_oss",
    "Qwen2Moe": "qwen2_moe",
}


def register_config(model_type: str):
    """Decorator to register a config class for a model type.

    Usage:
        @register_config("gpt_oss")
        @dataclass
        class GptOss120BConfig(BaseModelConfig):
            ...

    Args:
        model_type: The model_type string in config.json

    Returns:
        Decorator function
    """
    def decorator(cls: Type["BaseModelConfig"]) -> Type["BaseModelConfig"]:
        CONFIG_REGISTRY[model_type] = cls
        logger.debug(f"Registered config class {cls.__name__} for model_type={model_type}")
        return cls
    return decorator


def load_config(model_dir: str) -> "BaseModelConfig":
    """Auto-detect model type and load config from local directory.

    This function replaces HuggingFace's AutoConfig.from_pretrained().

    Detection order:
    1. Try model_type field in config.json
    2. Fallback: detect from architectures field
    3. Ultimate fallback: use BaseModelConfig

    Args:
        model_dir: Path to model directory containing config.json

    Returns:
        Appropriate config class instance

    Raises:
        FileNotFoundError: If config.json doesn't exist
        json.JSONDecodeError: If config.json is invalid
    """
    from .model_config import BaseModelConfig

    config_path = Path(model_dir) / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = json.load(f)

    config = None

    # Try model_type first
    model_type = data.get("model_type", "")
    if model_type in CONFIG_REGISTRY:
        logger.debug(f"Loading config for model_type={model_type}")
        config = CONFIG_REGISTRY[model_type].from_dir(model_dir)
    else:
        # Fallback: detect from architectures
        archs = data.get("architectures", [])
        for arch in archs:
            for pattern, config_type in ARCH_PATTERNS.items():
                if pattern in arch and config_type in CONFIG_REGISTRY:
                    logger.debug(
                        f"Detected model_type={config_type} from architecture={arch}"
                    )
                    config = CONFIG_REGISTRY[config_type].from_dir(model_dir)
                    break
            if config is not None:
                break

    if config is None:
        # Ultimate fallback: use base config
        logger.warning(
            f"Unknown model type '{model_type}' with architectures {archs}. "
            f"Using BaseModelConfig. Registered types: {list(CONFIG_REGISTRY.keys())}"
        )
        config = BaseModelConfig.from_dir(model_dir)

    # Set _name_or_path to the model directory for compatibility
    config._name_or_path = model_dir

    return config


def get_registered_models() -> Dict[str, Type["BaseModelConfig"]]:
    """Get all registered model config classes.

    Returns:
        Dict mapping model_type to config class
    """
    return CONFIG_REGISTRY.copy()


# Import model-specific configs to register them
# These imports trigger the @register_config decorators
def _import_model_configs():
    """Import all model-specific config modules to register them."""
    try:
        from batchgen.models.gpt_oss import config as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.deepseek.deepseekv3 import config as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.deepseek.deepseekv2 import config as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.mixtral import config as _  # noqa: F401
    except ImportError:
        pass


# Auto-import on module load
_import_model_configs()
