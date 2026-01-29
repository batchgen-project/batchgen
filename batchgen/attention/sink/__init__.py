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

"""Attention sink module for GPT-OSS-style models.

Provides utilities for integrating learned sink tokens into attention computation.
Sink tokens act as "attention sinks" that absorb attention weight without
contributing to the output values.

Reference: OpenAI GPT-OSS-120B architecture
"""

from .sink_softmax import softmax_with_sinks
from .sink_correction import apply_sink_correction

__all__ = [
    "softmax_with_sinks",
    "apply_sink_correction",
]
