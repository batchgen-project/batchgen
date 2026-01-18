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

"""BatchGen configuration module.

This module provides configuration utilities for BatchGen, including:
- Model configuration classes (BaseModelConfig and model-specific configs)
- Tokenizer abstraction layer (BaseTokenizer and implementations)
- Registry systems for auto-detecting model types and tokenizers

Usage:
    # Load model config
    from batchgen.config import load_config
    config = load_config("/path/to/model")

    # Load tokenizer
    from batchgen.config import load_tokenizer
    tokenizer = load_tokenizer("/path/to/model")
"""

# Model configuration
from .model_config import BaseModelConfig
from .model_registry import load_config, register_config, CONFIG_REGISTRY

# Tokenizer abstraction
from .base_tokenizer import BaseTokenizer
from .fast_tokenizer import FastTokenizer
from .tokenizer_registry import load_tokenizer, register_tokenizer, TOKENIZER_REGISTRY

__all__ = [
    # Model config
    "BaseModelConfig",
    "load_config",
    "register_config",
    "CONFIG_REGISTRY",
    # Tokenizer
    "BaseTokenizer",
    "FastTokenizer",
    "load_tokenizer",
    "register_tokenizer",
    "TOKENIZER_REGISTRY",
]
