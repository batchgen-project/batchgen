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

"""Base module wrappers for BatchGen execution.

This package provides base classes for attention and expert module wrappers.
Wrappers handle:
- Weight loading from core engine
- Dequantization hooks (overridable for model-specific logic)
- Micro-batching to avoid OOM
- Safe distributed operations for world_size == 1

Model-specific wrappers should be defined in their respective model directories:
- GPT-OSS: batchgen/models/openai/gpt_oss_120b/wrappers.py
- DeepSeek: batchgen/models/deepseek/deepseekv3/wrappers.py

Usage:
    from batchgen.models.wrappers import (
        BaseModuleWrapper,
        ExpertWrapperBase,
        AttnWrapperBase,
    )
"""

from .base import BaseModuleWrapper
from .expert import ExpertWrapperBase
from .attention import AttnWrapperBase

__all__ = [
    "BaseModuleWrapper",
    "ExpertWrapperBase",
    "AttnWrapperBase",
]
