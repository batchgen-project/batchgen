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

"""Planner module for model-specific configuration planning.

Planners decide config values, workers execute based on those configs.
"""

from batchgen.planner.base_planner import BasePlanner
from typing import Type


def get_planner(model_name: str) -> Type[BasePlanner]:
    """Return the appropriate planner class for a model.

    Args:
        model_name: HuggingFace model name (e.g., "deepseek-ai/DeepSeek-R1")

    Returns:
        Planner class (not instance) for the model

    Raises:
        ValueError: If no planner is available for the model
    """
    model_lower = model_name.lower()

    if model_lower in ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"]:
        from batchgen.models.deepseek.deepseekv3.planner import DeepSeekV3Planner
        return DeepSeekV3Planner

    raise ValueError(f"No planner available for model: {model_name}")


__all__ = ["BasePlanner", "get_planner"]
