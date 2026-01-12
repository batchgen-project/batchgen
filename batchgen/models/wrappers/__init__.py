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

"""Module wrappers for BatchGen execution.

This package provides base classes and model-specific wrappers for attention
and expert modules. Wrappers handle:
- Weight loading from core engine
- Dequantization (FP8, MXFP4, etc.)
- Micro-batching to avoid OOM
- Safe distributed operations for world_size == 1

Usage:
    from batchgen.models.wrappers import (
        BaseModuleWrapper,
        ExpertWrapperBase,
        AttnWrapperBase,
        DeepSeekExpertWrapper,
        DeepSeekAttnWrapper,
        GptOssExpertWrapper,
        GptOssAttnWrapper,
    )
"""

from .base import BaseModuleWrapper
from .expert import ExpertWrapperBase
from .attention import AttnWrapperBase

# Model-specific wrappers
from .deepseek_wrappers import DeepSeekExpertWrapper, DeepSeekAttnWrapper
from .gpt_oss_wrappers import GptOssExpertWrapper, GptOssAttnWrapper

__all__ = [
    # Base classes
    "BaseModuleWrapper",
    "ExpertWrapperBase",
    "AttnWrapperBase",
    # DeepSeek wrappers
    "DeepSeekExpertWrapper",
    "DeepSeekAttnWrapper",
    # GPT-OSS wrappers
    "GptOssExpertWrapper",
    "GptOssAttnWrapper",
]
