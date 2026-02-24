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

    # Load config from HuggingFace model identifier
    config = load_config("deepseek-ai/DeepSeek-R1", cache_dir="/path/to/cache")

    # Access config attributes
    print(config.model_type)
    print(config.num_hidden_layers)
    print(config.is_moe())
"""

from typing import Dict, Type, Optional, TYPE_CHECKING
from pathlib import Path
import json
import logging
import os

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
    "MiniMaxM2": "minimax_m25",
    "GlmMoeDsa": "glm_moe_dsa",
}

# Model name/identifier patterns for detection from HuggingFace model IDs
# Maps patterns found in model names to model_type
MODEL_NAME_PATTERNS: Dict[str, str] = {
    "MiniMax-M2.5": "minimax_m25",
    "MiniMaxAI/MiniMax-M2.5": "minimax_m25",
    "moonshotai/Kimi-K2.5": "kimi_k25",
    "DeepSeek-R1": "deepseek_v3",
    "DeepSeek-V3": "deepseek_v3",
    "DeepSeek-V2-Lite": "deepseek_v2",
    "DeepSeek-V2": "deepseek_v2",
    "Mixtral-8x22B": "mixtral",
    "Mixtral-8x7B": "mixtral",
    "gpt-oss": "gpt_oss",
    "GLM-5-FP8": "glm_moe_dsa",
    "GLM-5": "glm_moe_dsa",
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


def _detect_model_type_from_identifier(model_identifier: str) -> Optional[str]:
    """Detect model type from HuggingFace model identifier.

    Args:
        model_identifier: HuggingFace model ID (e.g., "deepseek-ai/DeepSeek-R1")

    Returns:
        model_type string if detected, None otherwise
    """
    # Check each pattern - order matters (more specific patterns first)
    # MODEL_NAME_PATTERNS is ordered with more specific patterns first
    for pattern, model_type in MODEL_NAME_PATTERNS.items():
        if pattern in model_identifier:
            logger.debug(f"Detected model_type={model_type} from identifier={model_identifier}")
            return model_type
    return None


def load_config(model_identifier: str) -> "BaseModelConfig":
    """Auto-detect model type and load appropriate BatchGen config.

    This function replaces HuggingFace's AutoConfig.from_pretrained().

    Detection order:
    1. Try to detect model type from HuggingFace model identifier patterns
    2. If local directory, try model_type field in config.json
    3. Fallback: detect from architectures field in config.json
    4. Ultimate fallback: use BaseModelConfig

    Args:
        model_identifier: Either a HuggingFace model ID (e.g., "deepseek-ai/DeepSeek-R1")
                         or a local path to model directory containing config.json

    Returns:
        Appropriate config class instance with pre-defined BatchGen defaults

    Raises:
        ValueError: If model type cannot be detected and no local config.json exists
    """
    from .model_config import BaseModelConfig

    config = None

    # Step 1: Try to detect model type from identifier patterns
    detected_type = _detect_model_type_from_identifier(model_identifier)
    if detected_type and detected_type in CONFIG_REGISTRY:
        logger.info(f"Using built-in config for model_type={detected_type}")
        config = CONFIG_REGISTRY[detected_type]()
        config._name_or_path = model_identifier
        return config

    # Step 2: Check if it's a local directory with config.json
    config_path = Path(model_identifier) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Try model_type first
        model_type = data.get("model_type", "")
        if model_type in CONFIG_REGISTRY:
            logger.debug(f"Loading config for model_type={model_type}")
            config = CONFIG_REGISTRY[model_type].from_dir(model_identifier)
        else:
            # Fallback: detect from architectures
            archs = data.get("architectures", [])
            for arch in archs:
                for pattern, config_type in ARCH_PATTERNS.items():
                    if pattern in arch and config_type in CONFIG_REGISTRY:
                        logger.debug(
                            f"Detected model_type={config_type} from architecture={arch}"
                        )
                        config = CONFIG_REGISTRY[config_type].from_dir(model_identifier)
                        break
                if config is not None:
                    break

        if config is None:
            # Ultimate fallback: use base config from local file
            logger.warning(
                f"Unknown model type '{model_type}' with architectures {archs}. "
                f"Using BaseModelConfig. Registered types: {list(CONFIG_REGISTRY.keys())}"
            )
            config = BaseModelConfig.from_dir(model_identifier)

        config._name_or_path = model_identifier
        return config

    # Step 3: No local config.json and pattern not recognized
    raise ValueError(
        f"Cannot detect model type for '{model_identifier}'. "
        f"Not a recognized HuggingFace model ID and no local config.json found. "
        f"Known patterns: {list(MODEL_NAME_PATTERNS.keys())}. "
        f"Registered model types: {list(CONFIG_REGISTRY.keys())}"
    )


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
        from batchgen.models.openai.gpt_oss_120b import config as _  # noqa: F401
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

    try:
        from batchgen.models.moonshotai.kimi_k25 import config as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.minimax.minimax_m25 import config as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.glm.glm5 import config as _  # noqa: F401
    except ImportError:
        pass


# Auto-import on module load
_import_model_configs()
