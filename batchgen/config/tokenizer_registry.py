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

"""Tokenizer registry for BatchGen.

This module provides:
1. A registry for model-specific tokenizer classes
2. Auto-detection of tokenizer type from model identifier
3. A unified load_tokenizer() function to replace AutoTokenizer.from_pretrained()

Key Design Principle:
    Tokenizer files are bundled with BatchGen, NOT loaded from user's cache directory.
    Each tokenizer class loads its tokenizer.json from its own package directory.

Usage:
    from batchgen.config.tokenizer_registry import load_tokenizer

    # Load tokenizer using model identifier for pattern matching
    tokenizer = load_tokenizer("deepseek-ai/DeepSeek-R1")

    # Encode/decode
    tokens = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(tokens)

    # Batch tokenization (HuggingFace-compatible API)
    batch = tokenizer(["Hello", "World"], return_tensors="pt", padding=True)
"""

from typing import Dict, Type, Optional, TYPE_CHECKING
import logging

from .model_name_utils import KIMI_K25_BACKEND_MODEL_IDS

if TYPE_CHECKING:
    from .base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


# Registry mapping tokenizer_type -> tokenizer class
TOKENIZER_REGISTRY: Dict[str, Type["BaseTokenizer"]] = {}

# Model name/identifier patterns for tokenizer detection
# Maps patterns found in model names to tokenizer_type
TOKENIZER_NAME_PATTERNS: Dict[str, str] = {
    "DeepSeek-R1": "deepseek_v3",
    "DeepSeek-V3": "deepseek_v3",
    "DeepSeek-V2-Lite": "deepseek_v2",
    "DeepSeek-V2": "deepseek_v2",
    "Mixtral-8x22B": "mixtral",
    "Mixtral-8x7B": "mixtral",
    "gpt-oss": "gpt_oss",
    # GLM-5 and GLM-5.1 share tokenizer.json (vocab_size=154880, identical
    # EOS/pad), but GLM-5.1 ships a richer chat template (tool_to_json macro,
    # thinking_indices tracking, tool_reference responses). We route them to
    # separate tokenizer types so each loads its own Jinja template.
    # More-specific patterns first so `GLM-5.1-FP8` doesn't get swallowed by
    # `GLM-5`.
    "GLM-5.1-FP8": "glm_moe_dsa_5_1",
    "GLM-5.1": "glm_moe_dsa_5_1",
    "GLM-5-FP8": "glm_moe_dsa",
    "GLM-5": "glm_moe_dsa",
    "MiniMax-M2.5": "minimax_m25",
    "MiniMaxAI/MiniMax-M2.5": "minimax_m25",
}

for model_id in KIMI_K25_BACKEND_MODEL_IDS:
    TOKENIZER_NAME_PATTERNS[model_id] = "kimi_k25"


def register_tokenizer(tokenizer_type: str):
    """Decorator to register a tokenizer class for a model type.

    Usage:
        @register_tokenizer("deepseek_v3")
        class DeepSeekV3Tokenizer(FastTokenizer):
            ...

    Args:
        tokenizer_type: The tokenizer type identifier

    Returns:
        Decorator function
    """
    def decorator(cls: Type["BaseTokenizer"]) -> Type["BaseTokenizer"]:
        TOKENIZER_REGISTRY[tokenizer_type] = cls
        logger.debug(f"Registered tokenizer class {cls.__name__} for type={tokenizer_type}")
        return cls
    return decorator


def load_tokenizer(model_identifier: str) -> "BaseTokenizer":
    """Load appropriate tokenizer for model.

    This function replaces HuggingFace's AutoTokenizer.from_pretrained().

    The model_identifier is used ONLY for pattern matching to determine which
    tokenizer type to use. Tokenizer files are loaded from the package directory
    by each tokenizer class (not from user-provided paths).

    Args:
        model_identifier: Model name or path for pattern matching
                         (e.g., "deepseek-ai/DeepSeek-R1", "DeepSeek-R1")

    Returns:
        Appropriate tokenizer instance (loads files from package directory)

    Raises:
        ValueError: If no registered tokenizer matches the model identifier
    """
    # Pattern match to find registered tokenizer
    for pattern, tokenizer_type in TOKENIZER_NAME_PATTERNS.items():
        if pattern in model_identifier:
            if tokenizer_type in TOKENIZER_REGISTRY:
                logger.info(f"Using registered tokenizer for type={tokenizer_type}")
                # Tokenizer loads from its own package directory (no path argument)
                return TOKENIZER_REGISTRY[tokenizer_type]()

    raise ValueError(
        f"No tokenizer registered for model: {model_identifier}. "
        f"Known patterns: {list(TOKENIZER_NAME_PATTERNS.keys())}. "
        f"Registered types: {list(TOKENIZER_REGISTRY.keys())}"
    )


def get_registered_tokenizers() -> Dict[str, Type["BaseTokenizer"]]:
    """Get all registered tokenizer classes.

    Returns:
        Dict mapping tokenizer_type to tokenizer class
    """
    return TOKENIZER_REGISTRY.copy()


# Import model-specific tokenizers to register them
# These imports trigger the @register_tokenizer decorators
def _import_tokenizers():
    """Import all model-specific tokenizer modules to register them."""
    try:
        from batchgen.models.deepseek.deepseekv3 import tokenizer as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.deepseek.deepseekv2 import tokenizer as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.openai.gpt_oss_120b import tokenizer as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.mixtral import tokenizer as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.moonshotai.kimi_k25 import tokenizer as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.glm.glm5 import tokenizer as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.minimax.minimax_m25 import tokenizer as _  # noqa: F401
    except ImportError:
        pass

    try:
        from batchgen.models.glm.glm5 import tokenizer as _  # noqa: F401
    except ImportError:
        pass


# Auto-import on module load
_import_tokenizers()
