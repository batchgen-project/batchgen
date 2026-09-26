# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
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
import importlib
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
    "DeepSeek-V4-Flash": "deepseek_v4",
    "DeepSeek-V4-Pro": "deepseek_v4",
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
    # GLM-5.2 and GLM-5.3 have their own tokenizer identities, now backed by
    # dedicated classes and templates. They share vocab bytes but not chat
    # semantics.
    # GLM-5.2 has its own tokenizer identity (glm_moe_dsa_5_2), now backed by a
    # registered GLM52Tokenizer. It shares GLM-5's vocab (tokenizer.json is
    # byte-identical) and stop tokens, but ships a DIFFERENT chat template that
    # prepends "<|system|>Reasoning Effort: Max" under thinking. The old
    # fall-through to "glm_moe_dsa" silently dropped that directive, so it is
    # no longer permitted — the class must stay registered.
    # More-specific patterns first so `GLM-5.2-FP8` / `GLM-5.1-FP8` don't get
    # swallowed by `GLM-5`.
    "GLM-5.3-FP8": "glm_moe_dsa_5_3",
    "GLM-5.3": "glm_moe_dsa_5_3",
    "GLM-5.2-FP8": "glm_moe_dsa_5_2",
    "GLM-5.2": "glm_moe_dsa_5_2",
    "GLM-5.1-FP8": "glm_moe_dsa_5_1",
    "GLM-5.1": "glm_moe_dsa_5_1",
    "GLM-5-FP8": "glm_moe_dsa",
    "GLM-5": "glm_moe_dsa",
    "MiniMax-M2.5": "minimax_m25",
    "MiniMaxAI/MiniMax-M2.5": "minimax_m25",
    "Kimi-Linear": "kimi_linear",
    "kimi-linear": "kimi_linear",
    # Kimi-K3 shares the Kimi-Linear ARCHITECTURE but NOT its tokenizer. The two
    # ship different added_tokens_decoder tables over a byte-identical BPE merge
    # file (163586 is "<|end_of_msg|>" in K3, "<|im_end|>" in the 48B) and K3 has
    # no Jinja chat template at all -- its XTML format is Python. Cross-loading
    # renders a 12-token K3 fragment as 32 marker-free tokens, silently
    # (bug_log.md 2026-07-31). Neither string is a substring of the other, so
    # ordering against "Kimi-Linear" does not matter.
    "Kimi-K3": "kimi_k3",
    "kimi-k3": "kimi_k3",
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
            module_path = _TOKENIZER_MODULES.get(tokenizer_type)
            if module_path is None:
                raise RuntimeError(
                    f"model {model_identifier!r} selected tokenizer type "
                    f"{tokenizer_type!r}, but no tokenizer module is declared"
                )
            try:
                importlib.import_module(module_path)
            except Exception as exc:  # noqa: BLE001 - preserve selected failure
                raise RuntimeError(
                    f"tokenizer module {module_path!r} for model "
                    f"{model_identifier!r} failed to import: {exc}"
                ) from exc
            tokenizer_cls = TOKENIZER_REGISTRY.get(tokenizer_type)
            if tokenizer_cls is None:
                raise RuntimeError(
                    f"tokenizer module {module_path!r} did not register "
                    f"tokenizer type {tokenizer_type!r}"
                )
            logger.info("Using registered tokenizer for type=%s", tokenizer_type)
            return tokenizer_cls()

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


_TOKENIZER_MODULES = {
    "deepseek_v4": "batchgen.models.deepseek.deepseekv4_flash.tokenizer",
    "deepseek_v3": "batchgen.models.deepseek.deepseekv3.tokenizer",
    "deepseek_v2": "batchgen.models.deepseek.deepseekv2.tokenizer",
    "gpt_oss": "batchgen.models.openai.gpt_oss_120b.tokenizer",
    "mixtral": "batchgen.models.mixtral.tokenizer",
    "kimi_k25": "batchgen.models.moonshotai.kimi_k25.tokenizer",
    "kimi_linear": "batchgen.models.moonshotai.kimi_linear.tokenizer",
    "kimi_k3": "batchgen.models.moonshotai.kimi_k3.tokenizer",
    "glm_moe_dsa": "batchgen.models.glm.glm5.tokenizer",
    "glm_moe_dsa_5_1": "batchgen.models.glm.glm5.tokenizer",
    "glm_moe_dsa_5_2": "batchgen.models.glm.glm5.tokenizer",
    "glm_moe_dsa_5_3": "batchgen.models.glm.glm5.tokenizer",
    "minimax_m25": "batchgen.models.minimax.minimax_m25.tokenizer",
}


# Preserve the historical registry population for callers that inspect
# TOKENIZER_REGISTRY directly, but isolate failures to the affected model.
# `load_tokenizer()` still imports the selected module on demand and never
# falls through to a different model's tokenizer.
def _import_tokenizers():
    for module_path in dict.fromkeys(_TOKENIZER_MODULES.values()):
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - report unrelated module failure
            logger.error(
                "Tokenizer module %s unavailable during registry warm-up: %s",
                module_path,
                exc,
            )


_import_tokenizers()
